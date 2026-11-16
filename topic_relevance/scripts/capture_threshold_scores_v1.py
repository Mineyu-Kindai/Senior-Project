import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from senior_project.detector import load_detector, predict_tokens
from senior_project.llm import capture_attention_features, load_llm, model_device, prepare_prompt
from senior_project.prompt import merge_flagged_spans, redact_spans
from topic_relevance.scripts.run_ab_followup_v1 import (
    ABLATION_REPORT,
    MODEL_CONFIG,
    ROOT,
    load_experiment_inputs,
    read_model_config,
    sha256_file,
)


CHECKPOINT = ROOT / "models/attention_detector_fixed_20261015.pt"
CHECKPOINT_SHA256 = "093810b3e65490f0059b507a2b814f5b47309d2053b18b6dac81ac0ec51248db"
THRESHOLD = 0.5


def input_hash(user_input: str, context: str) -> str:
    encoded = json.dumps([user_input, context], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_region(start: int, end: int, regions: list[dict[str, Any]]) -> str:
    overlapping = [
        region["name"]
        for region in regions
        if start < region["end"] and end > region["start"]
    ]
    if len(overlapping) != 1:
        return "boundary" if overlapping else "unclassified"
    region = next(region for region in regions if region["name"] == overlapping[0])
    if start < region["start"] or end > region["end"]:
        return "boundary"
    return overlapping[0]


def audit_baseline(
    source: dict[str, Any], tokens: list[dict[str, Any]]
) -> dict[str, Any]:
    offsets = [(token["start"], token["end"]) for token in tokens]
    predictions = [int(token["score"] >= THRESHOLD) for token in tokens]
    ranges = merge_flagged_spans(offsets, predictions)
    original_ranges = [(span["start"], span["end"]) for span in source["detected_spans"]]
    original_tokens = [
        token for span in source["detected_spans"] for token in span["token_scores"]
    ]
    original_keys = {
        (token["token_index"], token["start"], token["end"]) for token in original_tokens
    }
    new_keys = {
        (token["context_token_index"], token["start"], token["end"])
        for token, flag in zip(tokens, predictions, strict=True)
        if flag
    }
    original_scores = {
        (token["token_index"], token["start"], token["end"]): token["score"]
        for token in original_tokens
    }
    differences = [
        abs(token["score"] - original_scores[key])
        for token in tokens
        if (key := (token["context_token_index"], token["start"], token["end"]))
        in original_scores
    ]
    checks = {
        "detected_token_count": sum(predictions) == source["detected_token_count"],
        "detected_tokens": new_keys == original_keys
        and len(original_keys) == len(original_tokens),
        "redacted_spans": ranges == original_ranges,
        "sanitized_context": redact_spans(source["external_context"], ranges)
        == source["sanitized_external_context"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "previous_only_tokens": sorted(original_keys - new_keys),
        "recomputed_only_tokens": sorted(new_keys - original_keys),
        "max_saved_score_difference": max(differences) if differences else None,
        "total_context_token_count_previously_recorded": False,
    }


def unique_clean_inputs(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for example in examples:
        key = (example["user_input"], example["clean_context"])
        if key not in grouped:
            grouped[key] = {
                "user_input": key[0],
                "external_context": key[1],
                "source_ids": [],
                "source_block_ids": [],
            }
        grouped[key]["source_ids"].append(example["id"])
        if example["block_id"] not in grouped[key]["source_block_ids"]:
            grouped[key]["source_block_ids"].append(example["block_id"])
    return list(grouped.values())


def capture_one(
    user_input: str,
    context: str,
    regions: list[dict[str, Any]],
    tokenizer,
    model,
    config: dict[str, Any],
    detector,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, Any]:
    prompt = prepare_prompt(tokenizer, user_input, context, config)
    features = capture_attention_features(prompt, model)
    _, scores = predict_tokens(detector, features, mean, std, THRESHOLD)
    if len(scores) != len(prompt.context_offsets):
        raise ValueError("Detector scores and context offsets have different lengths")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Detector returned an invalid score")
    ids = prompt.model_inputs["input_ids"][0]
    tokens = [
        {
            "context_token_index": index,
            "prompt_token_index": prompt_index,
            "token_id": int(ids[prompt_index]),
            "start": start,
            "end": end,
            "score": score,
            "evaluation_region": classify_region(start, end, regions),
        }
        for index, (prompt_index, (start, end), score) in enumerate(
            zip(prompt.context_token_indices, prompt.context_offsets, scores, strict=True)
        )
    ]
    full_ids = tokenizer(
        prompt.rendered_text, truncation=False, add_special_tokens=False
    )["input_ids"]
    return {
        "input_sha256": input_hash(user_input, context),
        "user_input": user_input,
        "external_context": context,
        "input_token_count": int(ids.shape[0]),
        "untruncated_input_token_count": len(full_ids),
        "truncated": len(full_ids) > int(ids.shape[0]),
        "tokens": tokens,
    }


def run(output: Path) -> bool:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved 4-bit NF4 setup")
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Detector checkpoint hash differs from the recorded checkpoint")

    _, examples = load_experiment_inputs()
    if len({example["id"] for example in examples}) != 256:
        raise ValueError("Expected 256 distinct attack input IDs")
    clean_inputs = unique_clean_inputs(examples)
    if len(clean_inputs) != 128:
        raise ValueError("Expected 128 distinct clean instruction/context pairs")
    config = read_model_config()
    if config["max_input_tokens"] != 512:
        raise ValueError("Expected the B experiment input limit of 512 tokens")

    tokenizer, model = load_llm(config)
    if model_device(model).type != "cuda" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("The model must be loaded on CUDA with 4-bit quantization")
    detector, mean, std, threshold = load_detector(
        CHECKPOINT, config["model_id"], model_device(model)
    )
    if threshold != THRESHOLD:
        raise ValueError(f"Expected detector threshold {THRESHOLD}, got {threshold}")

    records = []
    for example in examples:
        captured = capture_one(
            example["user_input"], example["external_context"],
            example["detection_regions"], tokenizer, model, config, detector, mean, std,
        )
        audit = audit_baseline(example, captured["tokens"])
        records.append({
            "input_kind": "injected",
            "id": example["id"],
            "block_id": example["block_id"],
            "source_family_id": example["source_family_id"],
            "regions": example["detection_regions"],
            **captured,
            "baseline_audit": audit,
        })

    for clean in clean_inputs:
        context = clean["external_context"]
        regions = [{"name": "clean_context", "start": 0, "end": len(context)}]
        records.append({
            "input_kind": "clean_only",
            "source_ids": clean["source_ids"],
            "source_block_ids": clean["source_block_ids"],
            "regions": regions,
            **capture_one(
                clean["user_input"], context, regions, tokenizer, model, config,
                detector, mean, std,
            ),
        })

    failures = [
        record["id"] for record in records
        if record["input_kind"] == "injected" and not record["baseline_audit"]["passed"]
    ]
    code_files = [
        Path(__file__),
        ROOT / "topic_relevance/scripts/run_ab_followup_v1.py",
        ROOT / "src/senior_project/detector.py",
        ROOT / "src/senior_project/llm.py",
        ROOT / "src/senior_project/attention.py",
        ROOT / "src/senior_project/prompt.py",
    ]
    report = {
        "status": "baseline_audit_failed" if failures else "baseline_audit_passed",
        "threshold": THRESHOLD,
        "source_report_sha256": sha256_file(ABLATION_REPORT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "model_config": config,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "library_versions": {
            name: version(name)
            for name in ("torch", "transformers", "accelerate", "bitsandbytes")
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in code_files
        },
        "baseline_audit_failed_ids": failures,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote {len(records)} score records to {output}")
    print(f"0.5 baseline mismatches: {len(failures)}/256")
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture token scores and audit the saved 0.5 baseline."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not run(args.output):
        raise SystemExit("Baseline mismatch; do not proceed to threshold comparisons.")


if __name__ == "__main__":
    main()
