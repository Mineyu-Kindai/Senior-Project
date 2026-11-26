import argparse
import csv
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from senior_project.detector import filter_short_positive_runs, threshold_smoothed_logits
from senior_project.llm import load_llm, model_device, prepare_prompt
from senior_project.token_removal import remove_flagged_tokens
from topic_relevance.scripts.run_ab_followup_v1 import (
    MODEL_CONFIG,
    generate_with_limit,
    read_model_config,
    sanity_check,
    sha256_file,
)
from topic_relevance.scripts.run_pilot_expansion_250_token_removal_v1 import (
    CANDIDATE_FILE,
    CHECKPOINT,
    CHECKPOINT_SHA256,
    LOG as SOURCE_LOG,
    MAX_NEW_TOKENS,
    ROOT,
    build_cases,
    classify_token,
    input_hash,
    read_json,
    write_entry,
)
from topic_relevance.scripts.select_filtered_removal_v1 import OUTPUT as SELECTION_FILE


LOG = ROOT / "topic_relevance/data/experiments/filtered_removal_64_v1.jsonl"
CSV = ROOT / "topic_relevance/data/experiments/filtered_removal_64_responses_v1.csv"
SUMMARY = ROOT / "topic_relevance/data/experiments/filtered_removal_64_summary_v1.json"
FIELDS = (
    "case_id", "block_id", "source_family_id", "context_id", "probe_id",
    "intended_relation", "task_type", "defense", "user_instruction",
    "clean_context", "probe_text", "expected_answer", "accepted_variants",
    "gold_task_answer", "external_context", "model_response", "response_source",
    "probe_tokens_flagged", "probe_tokens_removed", "probe_tokens_total",
    "probe_token_detection_rate", "probe_token_removal_rate",
    "wrapper_tokens_flagged", "wrapper_tokens_removed", "wrapper_tokens_total",
    "injection_tokens_flagged", "injection_tokens_removed", "injection_tokens_total",
    "injection_token_detection_rate", "injection_token_removal_rate",
    "clean_characters_removed", "clean_characters_total",
    "input_token_count", "generated_token_count", "output_limit_reached",
    "eos_terminated", "generation_input_sha256",
)


def load_source(cases: list[dict[str, Any]]) -> tuple[dict, dict, dict]:
    ids = {case["case_id"] for case in cases}
    records: dict[str, dict[str, Any]] = {}
    generations: dict[str, dict[str, Any]] = {}
    with SOURCE_LOG.open(encoding="utf-8") as file:
        header = json.loads(next(file))
        for line in file:
            entry = json.loads(line)
            if entry.get("type") == "case" and entry["case_id"] in ids:
                if entry["case_id"] in records:
                    raise ValueError(f"Duplicate source case {entry['case_id']}")
                records[entry["case_id"]] = entry
            elif entry.get("type") == "generation":
                key = entry["input_sha256"]
                if key in generations:
                    raise ValueError(f"Duplicate source generation {key}")
                generations[key] = entry["generation"]
    if set(records) != ids:
        raise ValueError("Source log is missing development cases")
    return header, records, generations


def prepare_case(case: dict, record: dict, tokenizer, config: dict, margin: float, length: int) -> dict:
    context = case["external_context"]
    if record["input_sha256"] != input_hash(case["user_instruction"], context):
        raise ValueError(f"Source input mismatch: {case['case_id']}")
    prompt = prepare_prompt(tokenizer, case["user_instruction"], context, config)
    full_ids = tokenizer(prompt.rendered_text, truncation=False, add_special_tokens=False)["input_ids"]
    ids = prompt.model_inputs["input_ids"][0]
    tokens = record["tokens"]
    if (
        len(full_ids) != len(ids)
        or prompt.context_offsets != [(t["start"], t["end"]) for t in tokens]
        or [int(ids[i]) for i in prompt.context_token_indices] != [t["token_id"] for t in tokens]
    ):
        raise ValueError(f"Truncated input or tokenization mismatch: {case['case_id']}")
    if any(classify_token(t["start"], t["end"], case) != t["region"] for t in tokens):
        raise ValueError(f"Token region mismatch: {case['case_id']}")
    logits = torch.tensor(
        [[0.0, t["smoothed_injected_minus_clean_logit"]] for t in tokens],
        dtype=torch.float64,
    )
    predicted = threshold_smoothed_logits(logits, margin)
    removed = filter_short_positive_runs(predicted, length)
    sanitized = remove_flagged_tokens(tokenizer, prompt, context, removed)
    if not sanitized:
        raise ValueError(f"Empty sanitized context is not supported: {case['case_id']}")
    removed_positions = {
        i for token, flag in zip(tokens, removed, strict=True) if flag
        for i in range(token["start"], token["end"])
    }
    if sanitized != "".join(c for i, c in enumerate(context) if i not in removed_positions):
        raise ValueError(f"Decoded removal differs from character offsets: {case['case_id']}")
    regions = {}
    for region in ("probe", "wrapper", "clean_context", "boundary", "separator"):
        regions[region] = {
            "total": sum(t["region"] == region for t in tokens),
            "flagged": sum(flag for t, flag in zip(tokens, predicted, strict=True) if t["region"] == region),
            "removed": sum(flag for t, flag in zip(tokens, removed, strict=True) if t["region"] == region),
        }
    if not regions["probe"]["total"] or not regions["wrapper"]["total"]:
        raise ValueError(f"Empty probe or wrapper token denominator: {case['case_id']}")
    return {
        "type": "case",
        "case_id": case["case_id"],
        "input_sha256": record["input_sha256"],
        "sanitized_context": sanitized,
        "predictions": predicted,
        "removed": removed,
        "regions": regions,
        "clean_characters_removed": sum(i < case["clean_end"] for i in removed_positions),
        "clean_characters_total": case["clean_end"],
    }


def make_rows(cases: list[dict], scores: dict, generations: dict, previous: dict) -> list[dict]:
    rows = []
    for case in cases:
        score = scores[case["case_id"]]
        for defense, context in (
            ("none", case["external_context"]),
            ("filtered_token_removal", score["sanitized_context"]),
        ):
            key = input_hash(case["user_instruction"], context)
            result = previous[key] if key in previous else generations[key]
            row = {
                name: case[name] for name in (
                    "case_id", "block_id", "source_family_id", "context_id", "probe_id",
                    "intended_relation", "task_type", "user_instruction", "clean_context",
                    "probe_text", "expected_answer",
                )
            }
            row.update({
                "accepted_variants": json.dumps(case["accepted_variants"], ensure_ascii=False),
                "gold_task_answer": json.dumps(case["gold_task_answer"], ensure_ascii=False),
                "defense": defense,
                "external_context": context,
                "model_response": result["response"],
                "response_source": "previous_250_run" if key in previous else "new_generation",
                "input_token_count": result["input_token_count"],
                "generated_token_count": result["generated_token_count"],
                "output_limit_reached": result["max_limit_reached"],
                "eos_terminated": result["eos_terminated"],
                "generation_input_sha256": key,
                "clean_characters_removed": score["clean_characters_removed"] if defense != "none" else 0,
                "clean_characters_total": score["clean_characters_total"],
            })
            if defense != "none":
                probe, wrapper = score["regions"]["probe"], score["regions"]["wrapper"]
                for name, region in (("probe", probe), ("wrapper", wrapper)):
                    for field in ("flagged", "removed", "total"):
                        row[f"{name}_tokens_{field}"] = region[field]
                for field in ("flagged", "removed", "total"):
                    row[f"injection_tokens_{field}"] = probe[field] + wrapper[field]
                for name in ("probe", "injection"):
                    for field, label in (("flagged", "detection"), ("removed", "removal")):
                        row[f"{name}_token_{label}_rate"] = (
                            row[f"{name}_tokens_{field}"] / row[f"{name}_tokens_total"]
                        )
            rows.append(row)
    return rows


def select_cases(candidates: dict, remaining: bool) -> list[dict]:
    cases = build_cases(candidates)
    if remaining:
        cases = cases[256:]
        expected_blocks = {f"pilot_{i}" for i in range(265, 451)}
    else:
        cases = cases[:256]
        expected_blocks = {f"pilot_{i}" for i in range(201, 265)}
    if {case["block_id"] for case in cases} != expected_blocks or len(cases) != len(expected_blocks) * 4:
        raise ValueError("Unexpected block selection")
    return cases


def run(resume: bool, remaining: bool = False) -> None:
    if remaining:
        log = ROOT / "topic_relevance/data/experiments/filtered_removal_186_v1.jsonl"
        csv_path = ROOT / "topic_relevance/data/experiments/filtered_removal_186_responses_v1.csv"
        summary_path = ROOT / "topic_relevance/data/experiments/filtered_removal_186_summary_v1.json"
        experiment_id = "filtered_removal_186_v1"
    else:
        log, csv_path, summary_path = LOG, CSV, SUMMARY
        experiment_id = "filtered_removal_64_v1"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit NF4 generation")
    for path in (csv_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if log.exists() != resume:
        raise ValueError("Use --resume only with an existing run log; do not overwrite a log")
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Detector checkpoint hash mismatch")

    cases = select_cases(read_json(CANDIDATE_FILE), remaining)
    source_header, source_scores, previous = load_source(cases)
    selection = read_json(SELECTION_FILE)
    if (
        source_header["candidate_sha256"] != sha256_file(CANDIDATE_FILE)
        or source_header["checkpoint_sha256"] != sha256_file(CHECKPOINT)
        or source_header["model_config_sha256"] != sha256_file(MODEL_CONFIG)
        or selection["score_log_sha256"] != sha256_file(SOURCE_LOG)
        or selection["candidate_sha256"] != sha256_file(CANDIDATE_FILE)
    ):
        raise ValueError("Previous run or threshold selection provenance mismatch")
    for path in ("src/senior_project/llm.py", "src/senior_project/prompt.py"):
        if source_header["code_sha256"][path] != sha256_file(ROOT / path):
            raise ValueError(f"Generation code differs from prior run: {path}")
    libraries = {name: version(name) for name in source_header["library_versions"]}
    if libraries != source_header["library_versions"]:
        raise ValueError("Library versions differ from previous run")

    config = read_model_config()
    tokenizer, model = load_llm(config)
    if model_device(model).type != "cuda" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Expected CUDA 4-bit NF4 model")
    revision = getattr(model.config, "_commit_hash", None)
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if revision != source_header["model_revision"] or tokenizer_hash != source_header["tokenizer_backend_sha256"]:
        raise ValueError("Model revision or tokenizer differs from the previous run")
    if not sanity_check(tokenizer, model, config)["passed"]:
        raise RuntimeError("Generation sanity check failed")

    selected = selection["selected"]
    margin, length = selected["logit_margin"], selected["min_run_length"]
    if margin != 0.0 or length != 6:
        raise ValueError("Unexpected selected development settings")
    scores = {
        case["case_id"]: prepare_case(case, source_scores[case["case_id"]], tokenizer, config, margin, length)
        for case in cases
    }
    if not remaining and (
        sum(s["regions"]["probe"]["removed"] for s in scores.values()) != selected["probe"]
        or sum(s["regions"]["wrapper"]["removed"] for s in scores.values()) != selected["wrapper"]
        or sum(s["clean_characters_removed"] for s in scores.values()) != selected["clean_characters_removed"]
    ):
        raise ValueError("Filtered masks disagree with development selection")

    jobs = {}
    for case in cases:
        for context in (case["external_context"], scores[case["case_id"]]["sanitized_context"]):
            key = input_hash(case["user_instruction"], context)
            job = {"user_input": case["user_instruction"], "context": context}
            if key in jobs and jobs[key] != job:
                raise ValueError("Generation input hash collision")
            jobs[key] = job
    missing_baselines = [
        c["case_id"] for c in cases
        if input_hash(c["user_instruction"], c["external_context"]) not in previous
    ]
    if missing_baselines:
        raise ValueError(f"Previous no-defense responses missing: {missing_baselines}")
    for key, job in jobs.items():
        prompt = prepare_prompt(tokenizer, job["user_input"], job["context"], config)
        full = tokenizer(prompt.rendered_text, truncation=False, add_special_tokens=False)["input_ids"]
        if len(full) != int(prompt.model_inputs["input_ids"].shape[-1]):
            raise ValueError(f"Generation input truncated: {key}")

    header = {
        "type": "header",
        "experiment_id": experiment_id,
        "candidate_sha256": sha256_file(CANDIDATE_FILE),
        "source_log_sha256": sha256_file(SOURCE_LOG),
        "selection_sha256": sha256_file(SELECTION_FILE),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "model_revision": revision,
        "tokenizer_backend_sha256": tokenizer_hash,
        "chat_template": tokenizer.chat_template,
        "library_versions": libraries,
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__),
                ROOT / "src/senior_project/detector.py",
                ROOT / "src/senior_project/llm.py",
                ROOT / "src/senior_project/prompt.py",
                ROOT / "src/senior_project/token_removal.py",
                ROOT / "topic_relevance/scripts/run_ab_followup_v1.py",
                ROOT / "topic_relevance/scripts/run_pilot_expansion_250_token_removal_v1.py",
            )
        },
        "logit_margin": margin,
        "minimum_run_length": length,
        "mean_filter_kernel_size": 5,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "cases": len(cases),
        "unique_inputs": len(jobs),
        "reusable_previous_inputs": sum(key in previous for key in jobs),
        "attack_success_and_task_accuracy_scored": False,
    }
    generated: dict[str, dict[str, Any]] = {}
    if resume:
        with log.open(encoding="utf-8") as file:
            if json.loads(next(file)) != header:
                raise ValueError("Saved run header differs from the current conditions")
            for line in file:
                entry = json.loads(line)
                key = entry["input_sha256"]
                if entry.get("type") != "generation" or key not in jobs or key in previous or key in generated:
                    raise ValueError(f"Unexpected or duplicate saved generation: {key}")
                generated[key] = entry["generation"]
        log_file = log.open("a", encoding="utf-8")
    else:
        log_file = log.open("x", encoding="utf-8")
        write_entry(log_file, header)

    with log_file:
        for key, job in jobs.items():
            if key in previous or key in generated:
                continue
            result = generate_with_limit(tokenizer, model, job["user_input"], job["context"], config, MAX_NEW_TOKENS)
            generated[key] = result
            write_entry(log_file, {"type": "generation", "input_sha256": key, "generation": result})

    rows = make_rows(cases, scores, generated, previous)
    if len(rows) != len(cases) * 2 or Counter(row["defense"] for row in rows) != {
        "none": len(cases), "filtered_token_removal": len(cases),
    }:
        raise ValueError(f"Expected {len(cases)} paired responses")
    with csv_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for relation in ("all", "related", "unrelated"):
        defended = [r for r in rows if r["defense"] != "none" and (relation == "all" or r["intended_relation"] == relation)]
        group: dict[str, Any] = {"cases": len(defended)}
        for region in ("probe", "wrapper", "injection"):
            for field in ("flagged", "removed", "total"):
                group[f"{region}_tokens_{field}"] = sum(r[f"{region}_tokens_{field}"] for r in defended)
            for field, label in (("flagged", "detection"), ("removed", "removal")):
                group[f"{region}_token_{label}_rate"] = (
                    group[f"{region}_tokens_{field}"] / group[f"{region}_tokens_total"]
                )
        group["clean_characters_removed"] = sum(r["clean_characters_removed"] for r in defended)
        group["clean_characters_total"] = sum(r["clean_characters_total"] for r in defended)
        groups[relation] = group
    summary = {
        "status": "completed",
        "source_log_sha256": sha256_file(SOURCE_LOG),
        "run_log_sha256": sha256_file(log),
        "csv_sha256": sha256_file(csv_path),
        "cases": len(cases),
        "responses": len(rows),
        "response_sources": dict(Counter(r["response_source"] for r in rows)),
        "groups": groups,
        "output_limit_reached": sum(r["output_limit_reached"] for r in rows),
        "attack_success_and_task_accuracy_scored": False,
    }
    with summary_path.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Completed {len(rows)} responses, {len(generated)} newly generated; saved to {csv_path}.")
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run filtered removal on the development or remaining blocks.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--remaining", action="store_true", help="Run the remaining 186 blocks.")
    args = parser.parse_args()
    run(args.resume, args.remaining)
