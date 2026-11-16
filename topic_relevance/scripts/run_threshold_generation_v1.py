import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from senior_project.llm import load_llm, model_device, prepare_prompt
from senior_project.prompt import redact_spans
from topic_relevance.scripts.capture_threshold_scores_v1 import input_hash
from topic_relevance.scripts.run_ab_followup_v1 import (
    ABLATION_REPORT,
    CALIBRATION_REPORT,
    CANDIDATE_FILE,
    MODEL_CONFIG,
    ROOT,
    generate_with_limit,
    read_model_config,
    sanity_check,
    score_task_response,
    sha256_file,
    target_answer_location,
    task_content_score_fields,
)
from topic_relevance.scripts.summarize_threshold_masks_v1 import THRESHOLDS, summarize_record


MAX_NEW_TOKENS = 384


def select_blocks(
    records: list[dict[str, Any]],
    threshold_results: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    selected_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = []
    for index, record in enumerate(records):
        source_ids = [record["id"]] if record["input_kind"] == "injected" else record["source_ids"]
        if not source_ids or any(source_id not in sources for source_id in source_ids):
            raise ValueError("A score record has missing or unknown source IDs")
        blocks = {sources[source_id]["block_id"] for source_id in source_ids}
        if blocks & selected_ids and not blocks <= selected_ids:
            raise ValueError("A clean input spans selected and excluded blocks")
        if blocks <= selected_ids:
            indices.append(index)
    selected_records = [records[index] for index in indices]
    selected_results = []
    for row in threshold_results:
        if len(row["inputs"]) != len(records):
            raise ValueError("Mask records do not match score record count")
        selected_results.append({
            "threshold": row["threshold"],
            "inputs": [row["inputs"][index] for index in indices],
        })
    return selected_records, selected_results


def build_jobs(
    records: list[dict[str, Any]],
    threshold_results: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    if [row["threshold"] for row in threshold_results] != list(THRESHOLDS):
        raise ValueError("The mask report does not contain the fixed thresholds in order")
    jobs: dict[str, dict[str, str]] = {}
    evaluations = []
    for index, record in enumerate(records):
        source_ids = (
            [record["id"]] if record["input_kind"] == "injected" else record["source_ids"]
        )
        if not source_ids or any(source_id not in sources for source_id in source_ids):
            raise ValueError("A score record has missing or unknown source IDs")
        source = sources[source_ids[0]]
        if record["input_sha256"] != input_hash(record["user_input"], record["external_context"]):
            raise ValueError("Score record input hash mismatch")
        if record["user_input"] != source["user_input"]:
            raise ValueError("Score record instruction differs from its source")
        if record["input_kind"] == "injected":
            if record["external_context"] != source["external_context"]:
                raise ValueError("Injected input differs from its source")
        elif record["input_kind"] == "clean_only":
            if any(
                sources[sid]["clean_context"] != record["external_context"]
                or sources[sid]["user_input"] != record["user_input"]
                or sources[sid]["gold_task_answer"] != source["gold_task_answer"]
                for sid in source_ids
            ):
                raise ValueError("Clean input or task answer differs across source IDs")
        else:
            raise ValueError(f"Unknown input kind: {record['input_kind']}")

        conditions: list[tuple[str, str]] = [("no_defense", record["external_context"])]
        for row in threshold_results:
            item = row["inputs"][index]
            if (
                item["input_sha256"] != record["input_sha256"]
                or item["input_kind"] != record["input_kind"]
                or item["id"] != record.get("id")
            ):
                raise ValueError("Mask record does not match score record")
            spans = [tuple(span) for span in item["masked_spans"]]
            if spans != summarize_record(record, row["threshold"])["masked_spans"]:
                raise ValueError("Saved mask spans do not match token scores")
            context = redact_spans(record["external_context"], spans)
            if (
                row["threshold"] == 0.5
                and record["input_kind"] == "injected"
                and context != source["sanitized_external_context"]
            ):
                raise ValueError("Threshold 0.5 does not reproduce the saved context")
            conditions.append((f"{row['threshold']:.2f}", context))

        for condition, context in conditions:
            key = input_hash(record["user_input"], context)
            job = {"user_input": record["user_input"], "context": context}
            if key in jobs and jobs[key] != job:
                raise ValueError("Input hash collision")
            jobs[key] = job
            evaluations.append({
                "input_kind": record["input_kind"],
                "source_ids": source_ids,
                "condition": condition,
                "generation_input_sha256": key,
                "gold_task_answer": source["gold_task_answer"],
                "expected_answer": source["expected_answer"] if record["input_kind"] == "injected" else None,
                "accepted_variants": source["accepted_variants"] if record["input_kind"] == "injected" else None,
            })
    return jobs, evaluations


def score_evaluation(evaluation: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    response = generated["response"]
    gold = evaluation["gold_task_answer"]
    json_valid, matches = score_task_response(response, gold)
    target = None
    if evaluation["input_kind"] == "injected":
        target = target_answer_location(
            response, evaluation["expected_answer"], evaluation["accepted_variants"]
        )["matched"]
    return {
        **evaluation,
        "json_object_valid": json_valid,
        "strict_task_fully_correct": bool(matches) and all(matches.values()),
        **task_content_score_fields(response, gold),
        "target_answer_string_match": target,
    }


def summarize(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("injected", "clean_only"):
        result[kind] = {}
        for condition in ("no_defense", *(f"{value:.2f}" for value in THRESHOLDS)):
            group = [e for e in evaluations if e["input_kind"] == kind and e["condition"] == condition]
            if not group:
                raise ValueError(f"Missing evaluations for {kind}/{condition}")
            content = Counter(
                "unscorable" if e["task_answer_content_fully_correct"] is None
                else "correct" if e["task_answer_content_fully_correct"] else "incorrect"
                for e in group
            )
            metrics: dict[str, Any] = {
                "examples": len(group),
                "strict_json_valid": sum(e["json_object_valid"] for e in group),
                "strict_task_correct": sum(e["strict_task_fully_correct"] for e in group),
                "content_correct": content["correct"],
                "content_incorrect": content["incorrect"],
                "content_unscorable": content["unscorable"],
            }
            if kind == "injected":
                metrics["target_string_matches"] = sum(e["target_answer_string_match"] for e in group)
                metrics["content_correct_and_target_absent"] = sum(
                    e["task_answer_content_fully_correct"] is True
                    and e["target_answer_string_match"] is False
                    for e in group
                )
            result[kind][condition] = metrics
    return result


def load_inputs(score_path: Path, mask_path: Path) -> tuple[dict, dict, dict, dict, list]:
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    masks = json.loads(mask_path.read_text(encoding="utf-8"))
    source = json.loads(ABLATION_REPORT.read_text(encoding="utf-8"))
    if scores["status"] != "baseline_audit_passed" or masks["status"] != "completed":
        raise ValueError("Scores or mask analysis has not completed successfully")
    if scores["source_report_sha256"] != sha256_file(ABLATION_REPORT):
        raise ValueError("Score report source hash mismatch")
    if masks["score_report_sha256"] != sha256_file(score_path):
        raise ValueError("Mask report score hash mismatch")
    if masks["source_report_sha256"] != sha256_file(ABLATION_REPORT):
        raise ValueError("Mask report source hash mismatch")
    records = scores["records"]
    if Counter(r["input_kind"] for r in records) != {"injected": 256, "clean_only": 128}:
        raise ValueError("Expected 256 injected and 128 clean inputs")
    sources = {e["id"]: e for e in source["examples"]}
    if len(sources) != 256 or {r["id"] for r in records if r["input_kind"] == "injected"} != set(sources):
        raise ValueError("Injected source IDs are missing or duplicated")
    clean_hashes = {r["input_sha256"] for r in records if r["input_kind"] == "clean_only"}
    expected_clean = {input_hash(e["user_input"], e["clean_context"]) for e in sources.values()}
    if len(clean_hashes) != 128 or clean_hashes != expected_clean:
        raise ValueError("Clean inputs differ from the source report")
    calibration = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATE_FILE.read_text(encoding="utf-8"))
    if calibration["status"] != "completed" or calibration["source_report_sha256"] != sha256_file(ABLATION_REPORT):
        raise ValueError("Calibration report does not match the source report")
    if calibration["candidate_sha256"] != sha256_file(CANDIDATE_FILE):
        raise ValueError("Calibration candidate file hash mismatch")
    selected_ids = set(calibration["selected_block_ids"])
    task_types = {block["block_id"]: block["task_type"] for block in candidates["blocks"]}
    if len(selected_ids) != 12 or not selected_ids <= task_types.keys():
        raise ValueError("Expected 12 known preselected calibration blocks")
    if sorted(Counter(task_types[block] for block in selected_ids).values()) != [6, 6]:
        raise ValueError("Expected six selected blocks per task type")
    selected_records, selected_results = select_blocks(
        records, masks["threshold_results"], sources, selected_ids
    )
    if Counter(r["input_kind"] for r in selected_records) != {"injected": 48, "clean_only": 24}:
        raise ValueError("Expected 48 injected and 24 clean inputs in the subset")
    if Counter(sources[r["id"]]["block_id"] for r in selected_records if r["input_kind"] == "injected") != {
        block: 4 for block in selected_ids
    }:
        raise ValueError("Expected four injected conditions per selected block")
    jobs, evaluations = build_jobs(selected_records, selected_results, sources)
    if len(evaluations) != 72 * 9:
        raise ValueError("Unexpected number of evaluation conditions")
    return scores, masks, sources, jobs, evaluations


def append_line(file, entry: dict[str, Any]) -> None:
    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    file.flush()


def read_progress(path: Path, header: dict[str, Any], jobs: dict[str, Any]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or json.loads(lines[0]) != header:
        raise ValueError("Existing output header does not match the current run")
    saved = {}
    for line in lines[1:]:
        entry = json.loads(line)
        if entry["type"] == "complete":
            raise ValueError("The run is already complete")
        if entry["type"] != "generation" or entry["input_sha256"] not in jobs:
            raise ValueError("Unexpected entry in existing output")
        key = entry["input_sha256"]
        if key in saved:
            raise ValueError("Duplicate generation in existing output")
        saved[key] = entry["generation"]
    return saved


def run(score_path: Path, mask_path: Path, output: Path, resume: bool) -> None:
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if resume and not output.is_file():
        raise FileNotFoundError(f"Cannot resume missing output {output}")
    scores, _, _, jobs, evaluations = load_inputs(score_path, mask_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit NF4 generation")
    config = read_model_config()
    if sha256_file(MODEL_CONFIG) != scores["model_config_sha256"] or config["max_input_tokens"] != 512:
        raise ValueError("Model configuration differs from score capture")
    for path in ("src/senior_project/llm.py", "src/senior_project/prompt.py"):
        if sha256_file(ROOT / path) != scores["code_sha256"][path]:
            raise ValueError(f"{path} differs from score capture")
    libraries = {name: version(name) for name in scores["library_versions"]}
    if libraries != scores["library_versions"]:
        raise ValueError("Library versions differ from score capture")
    tokenizer, model = load_llm(config)
    if model_device(model).type != "cuda" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("The model must be loaded on CUDA with 4-bit quantization")
    revision = getattr(model.config, "_commit_hash", None)
    if not revision or revision != scores["model_revision"]:
        raise ValueError("Model revision is missing or differs from score capture")
    tokenizer_fingerprint = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")
    ).hexdigest()
    code_files = [
        Path(__file__),
        ROOT / "topic_relevance/scripts/run_ab_followup_v1.py",
        ROOT / "topic_relevance/scripts/capture_threshold_scores_v1.py",
        ROOT / "topic_relevance/scripts/summarize_threshold_masks_v1.py",
        ROOT / "src/senior_project/llm.py",
        ROOT / "src/senior_project/prompt.py",
    ]
    header = {
        "type": "header",
        "score_report_sha256": sha256_file(score_path),
        "mask_report_sha256": sha256_file(mask_path),
        "source_report_sha256": sha256_file(ABLATION_REPORT),
        "calibration_report_sha256": sha256_file(CALIBRATION_REPORT),
        "candidate_file_sha256": sha256_file(CANDIDATE_FILE),
        "selected_block_ids": json.loads(
            CALIBRATION_REPORT.read_text(encoding="utf-8")
        )["selected_block_ids"],
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "model_revision": revision,
        "tokenizer_backend_sha256": tokenizer_fingerprint,
        "chat_template": tokenizer.chat_template,
        "library_versions": libraries,
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files},
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "unique_inputs": len(jobs),
        "evaluation_conditions": len(evaluations),
        "historical_responses_reused": False,
    }
    if resume:
        saved = read_progress(output, header, jobs)
        file = output.open("a", encoding="utf-8")
    else:
        check = sanity_check(tokenizer, model, config)
        if not check["passed"]:
            raise RuntimeError("Model generation sanity check failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        file = output.open("x", encoding="utf-8")
        append_line(file, header)
        saved = {}

    with file:
        for key, job in jobs.items():
            if key in saved:
                continue
            prompt = prepare_prompt(tokenizer, job["user_input"], job["context"], config)
            full_count = len(tokenizer(
                prompt.rendered_text, truncation=False, add_special_tokens=False
            )["input_ids"])
            generated = generate_with_limit(
                tokenizer, model, job["user_input"], job["context"], config, MAX_NEW_TOKENS
            )
            generated["untruncated_input_token_count"] = full_count
            generated["input_truncated"] = full_count > generated["input_token_count"]
            saved[key] = generated
            append_line(file, {"type": "generation", "input_sha256": key, "generation": generated})
        scored = [score_evaluation(e, saved[e["generation_input_sha256"]]) for e in evaluations]
        append_line(file, {
            "type": "complete",
            "summary": summarize(scored),
            "evaluations": scored,
            "semantic_attack_success": "pending",
            "probe_following": "pending",
        })
    print(f"Generated {len(saved)} unique inputs for {len(evaluations)} evaluation conditions.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-threshold responses.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.scores, args.masks, args.output, args.resume)


if __name__ == "__main__":
    main()
