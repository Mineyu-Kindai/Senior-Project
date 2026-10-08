import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import require_task_fields
from .detector import detector_metrics, load_detector, predict_tokens
from .llm import (
    capture_attention_features,
    generate_answer,
    model_device,
    prepare_prompt,
)
from .prompt import labels_for_span, merge_flagged_spans, redact_spans


def collect_features(
    records: list[dict[str, Any]],
    tokenizer,
    model,
    model_config: dict[str, Any],
    output_path: Path,
) -> dict[str, int]:
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    example_count = 0

    for index, record in enumerate(records):
        user_input, context, injection = require_task_fields(record, index)
        prompt = prepare_prompt(tokenizer, user_input, context, model_config)
        features = capture_attention_features(prompt, model)
        labels = [0] * len(prompt.context_offsets)
        if injection is not None:
            injection_start = context.index(injection)
            labels = labels_for_span(
                prompt.context_offsets,
                (injection_start, injection_start + len(injection)),
            )
        if injection is not None and not any(labels):
            raise ValueError(
                f"Dataset entry {index} has no injected tokens after truncation"
            )
        if not any(label == 0 for label in labels):
            raise ValueError(
                f"Dataset entry {index} must contain visible clean context tokens"
            )
        all_features.append(features.numpy().astype(np.float32))
        all_labels.append(np.asarray(labels, dtype=np.int64))
        example_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.concatenate(all_features, axis=0),
        labels=np.concatenate(all_labels, axis=0),
    )
    token_count = sum(len(labels) for labels in all_labels)
    return {"example_count": example_count, "token_count": token_count}


def evaluate_defense(
    records: list[dict[str, Any]],
    tokenizer,
    model,
    model_config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    detector, mean, std, threshold = load_detector(
        checkpoint_path, model_config["model_id"], model_device(model)
    )
    examples = []
    token_labels: list[int] = []
    token_predictions: list[int] = []
    baseline_successes = 0
    defended_successes = 0
    successes_with_indicator = 0

    for index, record in enumerate(records):
        user_input, context, injection = require_task_fields(record, index)
        prompt = prepare_prompt(tokenizer, user_input, context, model_config)
        features = capture_attention_features(prompt, model)
        predictions, _ = predict_tokens(
            detector, features, mean, std, threshold
        )
        labels = [0] * len(prompt.context_offsets)
        if injection is not None:
            injection_start = context.index(injection)
            labels = labels_for_span(
                prompt.context_offsets,
                (injection_start, injection_start + len(injection)),
            )
        spans = merge_flagged_spans(prompt.context_offsets, predictions)
        sanitized_context = redact_spans(context, spans)

        baseline = generate_answer(tokenizer, model, user_input, context, model_config)
        defended = generate_answer(
            tokenizer, model, user_input, sanitized_context, model_config
        )
        indicator = record.get("attack_success_indicator")
        if isinstance(indicator, str) and indicator:
            successes_with_indicator += 1
            baseline_success = indicator in baseline
            defended_success = indicator in defended
            baseline_successes += int(baseline_success)
            defended_successes += int(defended_success)
        else:
            baseline_success = None
            defended_success = None

        token_labels.extend(labels)
        token_predictions.extend(predictions)
        examples.append(
            {
                "id": record.get("id", record.get("data_id", str(index))),
                "detected_token_count": sum(predictions),
                "injected_token_count": sum(labels),
                "redacted_spans": len(spans),
                "baseline_attack_success": baseline_success,
                "defended_attack_success": defended_success,
                "baseline_response": baseline,
                "defended_response": defended,
            }
        )

    result: dict[str, Any] = {
        "model_id": model_config["model_id"],
        "examples": examples,
        "token_metrics": detector_metrics(token_labels, token_predictions),
    }
    if successes_with_indicator:
        baseline_rate = baseline_successes / successes_with_indicator
        defended_rate = defended_successes / successes_with_indicator
        result["attack_success"] = {
            "evaluated_examples": successes_with_indicator,
            "baseline_rate": baseline_rate,
            "defended_rate": defended_rate,
            "absolute_reduction": baseline_rate - defended_rate,
            "defense_reduction": (
                1.0 - defended_rate / baseline_rate if baseline_rate else None
            ),
        }
    return result


def save_report(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_baseline(
    records: list[dict[str, Any]],
    tokenizer,
    model,
    model_config: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for index, record in enumerate(records):
        user_input, context, _ = require_task_fields(record, index)
        response = generate_answer(tokenizer, model, user_input, context, model_config)
        indicator = record.get("attack_success_indicator")
        results.append(
            {
                "id": record.get("id", record.get("data_id", str(index))),
                "response": response,
                "attack_success": (
                    indicator in response if isinstance(indicator, str) and indicator else None
                ),
            }
        )
    return results
