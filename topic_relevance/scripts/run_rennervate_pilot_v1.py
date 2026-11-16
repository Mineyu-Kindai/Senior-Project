import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senior_project.detector import (
    RENNERVATE_MAX_CONSECUTIVE_THRESHOLD,
    RENNERVATE_MEAN_FILTER_KERNEL_SIZE,
    longest_consecutive_positive_run,
    mean_filter_token_logits,
    rennervate_input_detected,
)
from senior_project.llm import load_llm, model_device
from senior_project.prompt import merge_flagged_spans, redact_spans
from topic_relevance.scripts.capture_threshold_scores_v1 import (
    CHECKPOINT,
    input_hash,
)
from topic_relevance.scripts.run_ab_followup_v1 import (
    ABLATION_REPORT,
    CALIBRATION_REPORT,
    CANDIDATE_FILE,
    INTERVENTION_REPORT,
    MODEL_CONFIG,
    generate_with_limit,
    read_model_config,
    sanity_check,
    score_task_response,
    sha256_file,
    target_answer_location,
    task_content_score_fields,
)


SCORE_REPORT = ROOT / (
    "topic_relevance/data/experiments/"
    "threshold_token_scores_development_20261112_v1.json"
)
MAX_NEW_TOKENS = 384
OUTPUT = ROOT / (
    "topic_relevance/data/experiments/"
    "rennervate_detector_pilot_201_264_v1.jsonl"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def token_decisions(record: dict[str, Any]) -> list[int]:
    logits = []
    for token in record["tokens"]:
        probability = token["score"]
        if not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("Saved attack probabilities must be finite and between 0 and 1")
        # For two-class softmax, log(p / (1 - p)) is the injected-minus-clean
        # logit. Its mean-filtered sign gives the same argmax as filtering both logits.
        margin = math.log(probability) - math.log1p(-probability)
        logits.append([0.0, margin])
    predictions = mean_filter_token_logits(
        torch.tensor(logits, dtype=torch.float64),
        RENNERVATE_MEAN_FILTER_KERNEL_SIZE,
    )
    return predictions.argmax(dim=1).tolist()


def build_evaluations(
    score_report: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    historical: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    jobs: dict[str, dict[str, str]] = {}
    evaluations = []
    region_totals: dict[str, dict[str, dict[str, int]]] = {}
    input_counts = Counter()

    for record in score_report["records"]:
        input_kind = record["input_kind"]
        source_ids = [record["id"]] if input_kind == "injected" else record["source_ids"]
        if not source_ids or any(source_id not in sources for source_id in source_ids):
            raise ValueError("A score record has missing or unknown source IDs")
        source = sources[source_ids[0]]
        for source_id in source_ids:
            candidate = sources[source_id]
            expected_context = (
                candidate["external_context"]
                if input_kind == "injected"
                else candidate["clean_context"]
            )
            if (
                candidate["user_input"] != record["user_input"]
                or expected_context != record["external_context"]
                or candidate["gold_task_answer"] != source["gold_task_answer"]
            ):
                raise ValueError("Source IDs do not share the saved task and context")
        if record["input_sha256"] != input_hash(
            record["user_input"], record["external_context"]
        ):
            raise ValueError("Saved score input hash mismatch")

        predictions = token_decisions(record)
        offsets = [(token["start"], token["end"]) for token in record["tokens"]]
        if len(offsets) != len(predictions):
            raise ValueError("Token offsets and predictions differ in length")
        spans = merge_flagged_spans(offsets, predictions)
        sanitized_context = redact_spans(record["external_context"], spans)
        longest_run = longest_consecutive_positive_run(predictions)
        input_injected = rennervate_input_detected(predictions)
        input_counts[(input_kind, input_injected)] += 1

        clean_region = next(
            region for region in record["regions"] if region["name"] == "clean_context"
        )
        masked_clean_chars = sum(
            max(0, min(end, clean_region["end"]) - max(start, clean_region["start"]))
            for start, end in spans
        )
        region_flags: dict[str, dict[str, int]] = {}
        for token, flagged in zip(record["tokens"], predictions, strict=True):
            name = token["evaluation_region"]
            totals = region_totals.setdefault(input_kind, {}).setdefault(
                name, {"tokens": 0, "flagged_tokens": 0}
            )
            totals["tokens"] += 1
            totals["flagged_tokens"] += flagged
            counts = region_flags.setdefault(name, {"tokens": 0, "flagged_tokens": 0})
            counts["tokens"] += 1
            counts["flagged_tokens"] += flagged

        if input_kind == "injected":
            baseline_condition = "unprocessed_attack"
            baseline_context = source["external_context"]
        else:
            baseline_condition = "clean_context_only"
            baseline_context = source["clean_context"]
        baseline = historical[(source_ids[0], baseline_condition)]
        expected_baseline_hash = hashlib.sha256(
            baseline_context.encode("utf-8")
        ).hexdigest()
        if baseline.get("external_context_sha256") != expected_baseline_hash:
            raise ValueError("Historical baseline does not match the saved task context")
        if input_kind == "clean_only":
            for source_id in source_ids[1:]:
                other = historical[(source_id, baseline_condition)]
                if other["response"] != baseline["response"]:
                    raise ValueError("Deduplicated clean inputs have different baselines")

        key = input_hash(record["user_input"], sanitized_context)
        job = {"user_input": record["user_input"], "context": sanitized_context}
        if key in jobs and jobs[key] != job:
            raise ValueError("Input hash collision")
        jobs[key] = job
        evaluations.append(
            {
                "input_kind": input_kind,
                "source_ids": source_ids,
                "block_id": record.get("block_id"),
                "input_sha256": record["input_sha256"],
                "generation_input_sha256": key,
                "gold_task_answer": source["gold_task_answer"],
                "expected_answer": source.get("expected_answer")
                if input_kind == "injected"
                else None,
                "accepted_variants": source.get("accepted_variants")
                if input_kind == "injected"
                else None,
                "token_count": len(predictions),
                "detected_token_count": sum(predictions),
                "max_consecutive_injected_tokens": longest_run,
                "input_injected": input_injected,
                "region_counts": region_flags,
                "redacted_spans": spans,
                "clean_characters_masked": masked_clean_chars,
                "clean_characters_total": clean_region["end"] - clean_region["start"],
                "sanitized_context_sha256": hashlib.sha256(
                    sanitized_context.encode("utf-8")
                ).hexdigest(),
                "baseline_condition": baseline_condition,
                "baseline_response": baseline["response"],
            }
        )

    summary = {
        "input_counts": {
            "injected": sum(
                count for (kind, _), count in input_counts.items() if kind == "injected"
            ),
            "clean_only": sum(
                count for (kind, _), count in input_counts.items() if kind == "clean_only"
            ),
        },
        "input_detection": {
            "true_positive": input_counts[("injected", True)],
            "false_negative": input_counts[("injected", False)],
            "false_positive": input_counts[("clean_only", True)],
            "true_negative": input_counts[("clean_only", False)],
        },
        "token_regions": region_totals,
    }
    return jobs, evaluations, summary


def score_response(
    response: str,
    gold: dict[str, Any],
    expected_answer: str | None,
    accepted_variants: list[str] | None,
) -> dict[str, Any]:
    json_valid, field_matches = score_task_response(response, gold)
    result = {
        "json_object_valid": json_valid,
        "strict_task_fully_correct": bool(field_matches) and all(field_matches.values()),
        **task_content_score_fields(response, gold),
    }
    if expected_answer is not None:
        target = target_answer_location(response, expected_answer, accepted_variants or [])
        result["target_answer_string_match"] = target["matched"]
    return result


def summarize(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for input_kind in ("injected", "clean_only"):
        group = [item for item in evaluations if item["input_kind"] == input_kind]
        content = Counter(
            item["task_answer_content_fully_correct"]
            for item in group
        )
        result[input_kind] = {
            "examples": len(group),
            "content_correct": content[True],
            "content_incorrect": content[False],
            "content_unscorable": content[None],
            "strict_json_valid": sum(item["json_object_valid"] for item in group),
            "strict_task_fully_correct": sum(
                item["strict_task_fully_correct"] for item in group
            ),
            "baseline_content_correct": sum(
                item["baseline_task_answer_content_fully_correct"] is True
                for item in group
            ),
            "correct_to_incorrect": sum(
                item["baseline_task_answer_content_fully_correct"] is True
                and item["task_answer_content_fully_correct"] is False
                for item in group
            ),
            "correct_to_unscorable": sum(
                item["baseline_task_answer_content_fully_correct"] is True
                and item["task_answer_content_fully_correct"] is None
                for item in group
            ),
            "incorrect_to_correct": sum(
                item["baseline_task_answer_content_fully_correct"] is False
                and item["task_answer_content_fully_correct"] is True
                for item in group
            ),
        }
        if input_kind == "injected":
            result[input_kind]["target_answer_string_matches"] = sum(
                item["target_answer_string_match"] for item in group
            )
    detection = result["injected"]
    clean = result["clean_only"]
    tp = sum(item["input_injected"] for item in evaluations if item["input_kind"] == "injected")
    fn = detection["examples"] - tp
    fp = sum(item["input_injected"] for item in evaluations if item["input_kind"] == "clean_only")
    tn = clean["examples"] - fp
    result["input_detection"] = {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": tp / detection["examples"] if detection["examples"] else None,
        "false_positive_rate": fp / clean["examples"] if clean["examples"] else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }
    return result


def append_line(file, entry: dict[str, Any]) -> None:
    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    file.flush()


def read_progress(
    path: Path, header: dict[str, Any], jobs: dict[str, dict[str, str]]
) -> dict[str, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or json.loads(lines[0]) != header:
        raise ValueError("Existing output header does not match the current run")
    saved = {}
    for line in lines[1:]:
        entry = json.loads(line)
        if entry.get("type") == "complete":
            raise ValueError("The run is already complete")
        key = entry.get("input_sha256")
        if entry.get("type") != "generation" or key not in jobs or key in saved:
            raise ValueError("Unexpected entry in existing output")
        saved[key] = entry["generation"]
    return saved


def run(output: Path, resume: bool) -> None:
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if resume and not output.is_file():
        raise FileNotFoundError(f"Cannot resume missing output: {output}")

    scores = read_json(SCORE_REPORT)
    source_report = read_json(ABLATION_REPORT)
    historical_report = read_json(INTERVENTION_REPORT)
    calibration = read_json(CALIBRATION_REPORT)
    candidates = read_json(CANDIDATE_FILE)
    if scores["status"] != "baseline_audit_passed":
        raise ValueError("Saved detector scores did not pass the baseline audit")
    if scores["checkpoint_sha256"] != sha256_file(CHECKPOINT):
        raise ValueError("Detector checkpoint differs from the saved score report")
    if scores["source_report_sha256"] != sha256_file(ABLATION_REPORT):
        raise ValueError("Saved detector scores use a different pilot source report")
    if scores["model_config_sha256"] != sha256_file(MODEL_CONFIG):
        raise ValueError("Model configuration differs from detector score capture")
    if calibration["status"] != "completed" or calibration["summary"][
        "recommended_max_new_tokens"
    ] != MAX_NEW_TOKENS:
        raise ValueError("The calibrated generation limit is not 384 tokens")
    if calibration["source_report_sha256"] != sha256_file(ABLATION_REPORT):
        raise ValueError("Calibration report uses a different pilot source report")
    if calibration["candidate_sha256"] != sha256_file(CANDIDATE_FILE):
        raise ValueError("Calibration report uses a different candidate collection")
    if source_report["status"] != "completed" or historical_report["status"] != "completed":
        raise ValueError("Pilot sources or historical response controls are incomplete")
    if historical_report["model_id"] != scores["model_config"]["model_id"]:
        raise ValueError("Historical controls use a different model")
    if historical_report["max_new_tokens"] != MAX_NEW_TOKENS:
        raise ValueError("Historical controls use a different generation limit")

    sources = {item["id"]: item for item in source_report["examples"]}
    if len(sources) != 256 or len(candidates["blocks"]) != 64:
        raise ValueError("Expected the complete 64-block, 256-attack pilot")
    injected_records = [r for r in scores["records"] if r["input_kind"] == "injected"]
    clean_records = [r for r in scores["records"] if r["input_kind"] == "clean_only"]
    if len(injected_records) != 256 or len(clean_records) != 128:
        raise ValueError("Expected 256 injected and 128 clean-only score records")
    expected_clean_hashes = {
        input_hash(item["user_input"], item["clean_context"])
        for item in sources.values()
    }
    if {r["input_sha256"] for r in clean_records} != expected_clean_hashes:
        raise ValueError("Saved clean-only inputs differ from the pilot source data")

    historical = {}
    for item in historical_report["examples"]:
        key = (item["id"], item["condition"])
        if key in historical:
            raise ValueError("Duplicate historical response condition")
        historical[key] = item
    for source_id in sources:
        for condition in ("unprocessed_attack", "clean_context_only"):
            if (source_id, condition) not in historical:
                raise ValueError(f"Missing historical control: {source_id}/{condition}")

    jobs, evaluations, detection_summary = build_evaluations(
        scores, sources, historical
    )
    if len(evaluations) != 384:
        raise ValueError("Unexpected number of pilot evaluations")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit NF4 generation")

    config = read_model_config()
    tokenizer, model = load_llm(config)
    if model_device(model).type != "cuda" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("The model must be loaded on CUDA with 4-bit quantization")
    revision = getattr(model.config, "_commit_hash", None)
    if not revision or revision != scores["model_revision"]:
        raise ValueError("Model revision differs from detector score capture")
    check = sanity_check(tokenizer, model, config)
    if not check["passed"]:
        raise RuntimeError("Generation sanity check failed")

    header = {
        "type": "header",
        "score_report_sha256": sha256_file(SCORE_REPORT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "source_report_sha256": sha256_file(ABLATION_REPORT),
        "historical_control_sha256": sha256_file(INTERVENTION_REPORT),
        "calibration_report_sha256": sha256_file(CALIBRATION_REPORT),
        "candidate_file_sha256": sha256_file(CANDIDATE_FILE),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "model_revision": revision,
        "detector_condition": {
            "probability_to_logit_margin": "log(p) - log(1 - p)",
            "mean_filter_kernel_size": RENNERVATE_MEAN_FILTER_KERNEL_SIZE,
            "padding": "replicate",
            "token_decision": "argmax of filtered two-class logits",
            "input_injected_rule": (
                "longest_consecutive_positive_run > "
                f"{RENNERVATE_MAX_CONSECUTIVE_THRESHOLD}"
            ),
            "sanitization": (
                "Mask all smoothed positive tokens regardless of the "
                "input-level Injected/Clean result; retain current span merging."
            ),
        },
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "evaluations": len(evaluations),
        "unique_generation_inputs": len(jobs),
        "generation_sanity_check": check,
        "detection_summary": detection_summary,
    }

    if resume:
        saved = read_progress(output, header, jobs)
        file = output.open("a", encoding="utf-8")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        file = output.open("x", encoding="utf-8")
        append_line(file, header)
        saved = {}

    with file:
        for key, job in jobs.items():
            if key in saved:
                continue
            generated = generate_with_limit(
                tokenizer, model, job["user_input"], job["context"], config, MAX_NEW_TOKENS
            )
            saved[key] = generated
            append_line(
                file,
                {"type": "generation", "input_sha256": key, "generation": generated},
            )

        scored = []
        for evaluation in evaluations:
            generated = saved[evaluation["generation_input_sha256"]]
            response = generated["response"]
            new_score = score_response(
                response,
                evaluation["gold_task_answer"],
                evaluation["expected_answer"],
                evaluation["accepted_variants"],
            )
            baseline_score = score_response(
                evaluation["baseline_response"],
                evaluation["gold_task_answer"],
                evaluation["expected_answer"],
                evaluation["accepted_variants"],
            )
            scored.append(
                {
                    **{
                        key: value
                        for key, value in evaluation.items()
                        if key != "baseline_response"
                    },
                    **generated,
                    **new_score,
                    **{
                        f"baseline_{key}": value
                        for key, value in baseline_score.items()
                    },
                    "response": response,
                }
            )
        append_line(
            file,
            {
                "type": "complete",
                "summary": summarize(scored),
                "evaluations": scored,
                "semantic_attack_success": "pending",
                "probe_following": "pending",
            },
        )
    print(
        f"Generated {len(saved)} unique responses for {len(evaluations)} pilot inputs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Rennervate-style detector conditions on the full pilot."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, args.resume)


if __name__ == "__main__":
    main()
