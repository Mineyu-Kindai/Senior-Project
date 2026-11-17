import argparse
import csv
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from senior_project.llm import load_llm, model_device, prepare_prompt
from senior_project.token_removal import remove_flagged_tokens
from topic_relevance.scripts.capture_threshold_scores_v1 import CHECKPOINT, input_hash
from topic_relevance.scripts.run_ab_followup_v1 import (
    ABLATION_REPORT,
    CALIBRATION_REPORT,
    CANDIDATE_FILE,
    INTERVENTION_REPORT,
    MODEL_CONFIG,
    ROOT,
    generate_with_limit,
    read_model_config,
    sanity_check,
    sha256_file,
)
from topic_relevance.scripts.run_rennervate_pilot_v1 import (
    OUTPUT as PREVIOUS_RUN,
    SCORE_REPORT,
    token_decisions,
)


LOG = ROOT / "topic_relevance/data/experiments/token_removal_pilot_201_264_v1.jsonl"
CSV = ROOT / "topic_relevance/data/experiments/token_removal_pilot_201_264_pairs_v1.csv"
REFERENCE_GENERATION = ROOT / (
    "topic_relevance/data/experiments/"
    "threshold_generations_12blocks_development_20261112_v1.jsonl"
)
FIELDS = (
    "example_id", "block_id", "source_family_id", "context_id", "probe_id",
    "intended_relation", "defense", "user_instruction", "external_context",
    "model_response", "probe_tokens_flagged", "probe_tokens_total",
    "probe_token_detection_rate", "clean_characters_removed",
    "clean_characters_total", "clean_character_removal_rate",
    "input_token_count", "generated_token_count", "output_limit_reached",
    "eos_terminated",
)


def load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def build_cases(
    tokenizer, scores: dict[str, Any], sources: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]], candidates: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    cases = []
    jobs = {}
    for record in scores["records"]:
        if record["input_kind"] != "injected":
            continue
        case_id = record["id"]
        source = sources[case_id]
        block = candidates[source["block_id"]]
        original = record["external_context"]
        if (
            record["user_input"] != source["user_input"]
            or original != source["external_context"]
            or source["clean_context"] != block["contexts"][source["context_id"]]["text"]
            or block["probes"][source["probe_id"]]["text"] not in original
            or record["input_sha256"] != input_hash(record["user_input"], original)
        ):
            raise ValueError(f"Pilot input mismatch: {case_id}")
        old = baseline[case_id]
        if old["external_context_sha256"] != hashlib.sha256(original.encode()).hexdigest():
            raise ValueError(f"Baseline context hash mismatch: {case_id}")
        if record["truncated"]:
            raise ValueError(f"Saved detector input was truncated: {case_id}")
        prompt = prepare_prompt(tokenizer, record["user_input"], original, config)
        tokens = record["tokens"]
        ids = prompt.model_inputs["input_ids"][0]
        if (
            [int(ids[index]) for index in prompt.context_token_indices]
            != [token["token_id"] for token in tokens]
            or prompt.context_offsets != [(token["start"], token["end"]) for token in tokens]
        ):
            raise ValueError(f"Current tokenization differs from saved scores: {case_id}")
        flags = token_decisions(record)
        sanitized = remove_flagged_tokens(tokenizer, prompt, original, flags)
        if not sanitized:
            raise ValueError(f"Empty sanitized context is not supported in this pilot: {case_id}")

        removed = set()
        probe_total = probe_detected = 0
        for token, flag in zip(tokens, flags, strict=True):
            if flag:
                removed.update(range(token["start"], token["end"]))
            if token["evaluation_region"] == "probe":
                probe_total += 1
                probe_detected += flag
        char_removed_text = "".join(
            character for index, character in enumerate(original) if index not in removed
        )
        if char_removed_text != sanitized:
            raise ValueError(
                f"Token decoding differs from character removal for {case_id}; "
                "character removal rate cannot be attributed safely"
            )
        clean = next(region for region in record["regions"] if region["name"] == "clean_context")
        clean_total = clean["end"] - clean["start"]
        clean_removed = sum(index in removed for index in range(clean["start"], clean["end"]))
        if not clean_total or not probe_total:
            raise ValueError(f"Empty metric denominator: {case_id}")
        key = input_hash(source["user_input"], sanitized)
        job = {"user_input": source["user_input"], "context": sanitized}
        if key in jobs and jobs[key] != job:
            raise ValueError("Generation input hash collision")
        jobs[key] = job
        cases.append({
            "example_id": case_id,
            "block_id": source["block_id"],
            "source_family_id": source["source_family_id"],
            "context_id": source["context_id"],
            "probe_id": source["probe_id"],
            "intended_relation": source["intended_relation"],
            "user_instruction": source["user_input"],
            "original_context": original,
            "sanitized_context": sanitized,
            "generation_input_sha256": key,
            "probe_tokens_flagged": probe_detected,
            "probe_tokens_total": probe_total,
            "clean_characters_removed": clean_removed,
            "clean_characters_total": clean_total,
        })
    if len(cases) != 256 or len({case["example_id"] for case in cases}) != 256:
        raise ValueError("Expected 256 distinct attack inputs")
    if Counter(case["block_id"] for case in cases) != {block: 4 for block in candidates}:
        raise ValueError("Expected four attack inputs for each of the 64 blocks")
    return cases, jobs


def write_entry(file, value: dict[str, Any]) -> None:
    file.write(json.dumps(value, ensure_ascii=False) + "\n")
    file.flush()


def run(resume: bool) -> None:
    if LOG.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {LOG}")
    if resume and not LOG.is_file():
        raise FileNotFoundError(f"Cannot resume missing log {LOG}")
    if CSV.exists():
        raise FileExistsError(f"Refusing to overwrite {CSV}")
    scores = load_json(SCORE_REPORT)
    source_report = load_json(ABLATION_REPORT)
    historical = load_json(INTERVENTION_REPORT)
    candidates_file = load_json(CANDIDATE_FILE)
    with PREVIOUS_RUN.open(encoding="utf-8") as file:
        previous_header = json.loads(file.readline())
    with REFERENCE_GENERATION.open(encoding="utf-8") as file:
        reference = json.loads(file.readline())
    for key, path in (
        ("score_report_sha256", SCORE_REPORT), ("checkpoint_sha256", CHECKPOINT),
        ("source_report_sha256", ABLATION_REPORT),
        ("historical_control_sha256", INTERVENTION_REPORT),
        ("candidate_file_sha256", CANDIDATE_FILE),
        ("calibration_report_sha256", CALIBRATION_REPORT),
        ("model_config_sha256", MODEL_CONFIG),
    ):
        if previous_header[key] != sha256_file(path):
            raise ValueError(f"Previous run fingerprint mismatch for {path}")
    if (
        scores["status"] != "baseline_audit_passed"
        or historical["status"] != "completed"
        or historical["max_new_tokens"] != 384
        or historical["source_report_sha256"] != sha256_file(ABLATION_REPORT)
        or historical["candidate_sha256"] != sha256_file(CANDIDATE_FILE)
    ):
        raise ValueError("Stored scores or baseline provenance do not match")
    libraries = {name: version(name) for name in reference["library_versions"]}
    if libraries != reference["library_versions"]:
        raise ValueError("Library versions differ from previous generation")
    for name in ("src/senior_project/llm.py", "src/senior_project/prompt.py"):
        if sha256_file(ROOT / name) != reference["code_sha256"][name]:
            raise ValueError(f"Generation code differs from previous generation: {name}")
    sources = {item["id"]: item for item in source_report["examples"]}
    baseline = {
        item["id"]: item for item in historical["examples"]
        if item["condition"] == "unprocessed_attack"
    }
    candidates = {block["block_id"]: block for block in candidates_file["blocks"]}
    if len(sources) != 256 or len(baseline) != 256 or len(candidates) != 64:
        raise ValueError("Pilot sources, controls, or candidate blocks are incomplete")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit NF4 generation")

    config = read_model_config()
    tokenizer, model = load_llm(config)
    if model_device(model).type != "cuda" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Expected a CUDA 4-bit model")
    revision = getattr(model.config, "_commit_hash", None)
    if revision != previous_header["model_revision"] or revision != reference["model_revision"]:
        raise ValueError("Model revision differs from previous experiment")
    tokenizer_hash = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")
    ).hexdigest()
    if tokenizer_hash != reference["tokenizer_backend_sha256"]:
        raise ValueError("Tokenizer differs from previous generation")
    if tokenizer.chat_template != reference["chat_template"]:
        raise ValueError("Chat template differs from previous generation")
    check = sanity_check(tokenizer, model, config)
    if not check["passed"]:
        raise RuntimeError("Generation sanity check failed")

    cases, jobs = build_cases(tokenizer, scores, sources, baseline, candidates, config)
    header = {
        "type": "header",
        "previous_run_sha256": sha256_file(PREVIOUS_RUN),
        "score_report_sha256": sha256_file(SCORE_REPORT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "candidate_sha256": sha256_file(CANDIDATE_FILE),
        "baseline_sha256": sha256_file(INTERVENTION_REPORT),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "model_revision": revision,
        "tokenizer_backend_sha256": tokenizer_hash,
        "library_versions": libraries,
        "script_sha256": sha256_file(Path(__file__)),
        "token_removal_sha256": sha256_file(ROOT / "src/senior_project/token_removal.py"),
        "max_new_tokens": 384,
        "do_sample": False,
        "cases": len(cases),
        "unique_generation_inputs": len(jobs),
        "generation_sanity_check": check,
    }
    saved = {}
    if resume:
        with LOG.open(encoding="utf-8") as file:
            if json.loads(file.readline()) != header:
                raise ValueError("Saved run header differs from the current conditions")
            for line in file:
                entry = json.loads(line)
                key = entry.get("input_sha256")
                if entry.get("type") != "generation" or key not in jobs or key in saved:
                    raise ValueError("Unexpected or duplicate entry in generation log")
                saved[key] = entry["generation"]
        log_file = LOG.open("a", encoding="utf-8")
    else:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        log_file = LOG.open("x", encoding="utf-8")
        write_entry(log_file, header)

    with log_file:
        for key, job in jobs.items():
            if key in saved:
                continue
            prompt = prepare_prompt(tokenizer, job["user_input"], job["context"], config)
            full_count = len(tokenizer(
                prompt.rendered_text, truncation=False, add_special_tokens=False
            )["input_ids"])
            if full_count > int(prompt.model_inputs["input_ids"].shape[-1]):
                raise ValueError(f"Sanitized generation input was truncated: {key}")
            generated = generate_with_limit(
                tokenizer, model, job["user_input"], job["context"], config, 384
            )
            saved[key] = generated
            write_entry(log_file, {
                "type": "generation", "input_sha256": key, "generation": generated
            })

    with CSV.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            old = baseline[case["example_id"]]
            generated = saved[case["generation_input_sha256"]]
            common = {key: case[key] for key in FIELDS if key in case}
            for mode, context, result in (
                ("none", case["original_context"], old),
                ("token_removal", case["sanitized_context"], generated),
            ):
                enabled = mode == "token_removal"
                writer.writerow({
                    **common,
                    "defense": mode,
                    "external_context": context,
                    "model_response": result["response"],
                    "probe_tokens_flagged": case["probe_tokens_flagged"] if enabled else "",
                    "probe_tokens_total": case["probe_tokens_total"] if enabled else "",
                    "probe_token_detection_rate": (
                        case["probe_tokens_flagged"] / case["probe_tokens_total"] if enabled else ""
                    ),
                    "clean_characters_removed": case["clean_characters_removed"] if enabled else 0,
                    "clean_character_removal_rate": (
                        case["clean_characters_removed"] / case["clean_characters_total"] if enabled else 0
                    ),
                    "input_token_count": result["input_token_count"],
                    "generated_token_count": result["generated_token_count"],
                    "output_limit_reached": result["max_limit_reached"],
                    "eos_terminated": result["eos_terminated"],
                })
    print(f"Generated {len(saved)} unique responses; exported {len(cases) * 2} rows to {CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run token removal on the pilot.")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args().resume)
