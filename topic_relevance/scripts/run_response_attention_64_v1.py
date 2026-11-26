import argparse
import csv
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from senior_project.detector import longest_consecutive_positive_run, rennervate_input_detected
from senior_project.llm import (
    capture_response_attention_with_tokens,
    load_llm,
    model_device,
    prepare_prompt,
)
from senior_project.prompt import SYSTEM_PROMPT
from senior_project.rennervate import (
    load_response_detector,
    reference_token_predictions,
    response_token_logits,
)
from senior_project.token_removal import remove_flagged_tokens
from topic_relevance.scripts.run_ab_followup_v1 import (
    CANDIDATE_FILE,
    MODEL_CONFIG,
    read_json,
    read_model_config,
    sanity_check,
    sha256_file,
)
from topic_relevance.scripts.run_pilot_expansion_250_token_removal_v1 import (
    FAKE_COMPLETION_WRAPPER,
    task_instruction_without_constraints,
    write_entry,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models/response_attention_detector.pt"
LOG = ROOT / "topic_relevance/data/experiments/response_attention_64_rawprobe_v1.jsonl"
CSV = ROOT / "topic_relevance/data/experiments/response_attention_64_rawprobe_responses_v1.csv"
SUMMARY = ROOT / "topic_relevance/data/experiments/response_attention_64_rawprobe_summary_v1.json"
WRAPPER_LOG = ROOT / "topic_relevance/data/experiments/response_attention_64_wrapper_v1.jsonl"
WRAPPER_CSV = ROOT / "topic_relevance/data/experiments/response_attention_64_wrapper_responses_v1.csv"
WRAPPER_SUMMARY = ROOT / "topic_relevance/data/experiments/response_attention_64_wrapper_summary_v1.json"
OUTPUT_LIMIT = 384
FIELDS = (
    "case_id", "block_id", "source_family_id", "context_id", "probe_id",
    "intended_relation", "task_type", "defense", "user_instruction",
    "clean_context", "probe_text", "probe_wrapper", "expected_answer", "accepted_variants",
    "gold_task_answer", "external_context", "model_response",
    "probe_tokens_flagged", "probe_tokens_total", "probe_token_detection_rate",
    "wrapper_tokens_flagged", "wrapper_tokens_total", "wrapper_token_detection_rate",
    "injection_tokens_flagged", "injection_tokens_total", "injection_token_detection_rate",
    "clean_tokens_flagged", "clean_tokens_total", "boundary_tokens_flagged",
    "boundary_tokens_total", "clean_characters_removed", "clean_characters_total",
    "input_injected", "max_consecutive_injected_tokens",
    "input_token_count", "generated_token_count", "output_limit_reached",
    "eos_terminated", "response_source",
)


def build_cases(
    candidates: dict[str, Any], probe_wrapper: str | None = None
) -> list[dict[str, Any]]:
    blocks = candidates.get("blocks")
    if candidates.get("candidate_count") != 64 or not isinstance(blocks, list):
        raise ValueError("Expected 64 candidate blocks")
    if [block.get("block_id") for block in blocks] != [f"pilot_{i}" for i in range(201, 265)]:
        raise ValueError("Expected pilot_201 through pilot_264 in order")
    cases = []
    for block in blocks:
        instruction = task_instruction_without_constraints(block["user_instruction"])
        for context_id in ("A", "B"):
            context = block["contexts"][context_id]
            clean = context["text"]
            for probe_id in ("A", "B"):
                probe = block["probes"][probe_id]
                relation = block["intended_crossing"][context_id + probe_id]
                if relation not in ("related", "unrelated"):
                    raise ValueError(f"Unknown intended relation in {block['block_id']}")
                wrapper_start = len(clean) + 2
                probe_start = wrapper_start + len(probe_wrapper) if probe_wrapper is not None else wrapper_start
                external_context = clean + "\n\n" + (probe_wrapper or "") + probe["text"]
                cases.append({
                    "case_id": f"{block['block_id']}__context_{context_id}_probe_{probe_id}",
                    "block_id": block["block_id"],
                    "source_family_id": block["source_family_id"],
                    "context_id": context_id,
                    "probe_id": probe_id,
                    "intended_relation": relation,
                    "task_type": block["task_type"],
                    "user_instruction": instruction,
                    "clean_context": clean,
                    "probe_text": probe["text"],
                    "probe_wrapper": probe_wrapper,
                    "expected_answer": probe["expected_answer"],
                    "accepted_variants": probe["accepted_variants"],
                    "gold_task_answer": context["gold_answer"],
                    "external_context": external_context,
                    "wrapper_start": wrapper_start if probe_wrapper is not None else None,
                    "probe_start": probe_start,
                })
    if len(cases) != 256 or len({case["case_id"] for case in cases}) != 256:
        raise ValueError("Expected 256 unique crossed cases")
    if Counter(case["intended_relation"] for case in cases) != {"related": 128, "unrelated": 128}:
        raise ValueError("Expected 128 cases for each intended relation")
    return cases


def region_for_token(
    start: int,
    end: int,
    clean_end: int,
    probe_start: int,
    wrapper_start: int | None = None,
) -> str:
    if end <= clean_end:
        return "clean"
    if start >= probe_start:
        return "probe"
    if wrapper_start is not None:
        if start >= wrapper_start and end <= probe_start:
            return "wrapper"
        if start >= clean_end and end <= wrapper_start:
            return "separator"
        return "boundary"
    if start >= clean_end and end <= probe_start:
        return "separator"
    return "boundary"


def checked_prompt(tokenizer, user_input: str, context: str, config: dict[str, Any]):
    prompt = prepare_prompt(tokenizer, user_input, context, config)
    full_ids = tokenizer(prompt.rendered_text, truncation=False, add_special_tokens=False)["input_ids"]
    if full_ids != prompt.model_inputs["input_ids"][0].tolist():
        raise ValueError("Generation or detection prompt was truncated")
    if prompt.context_offsets[-1][1] != len(context):
        raise ValueError("External context was truncated")
    return prompt


def empty_context_inputs(tokenizer, user_input: str, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_input}\n\n<external_context>\n\n</external_context>"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer(rendered, truncation=False, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        rendered, return_tensors="pt", truncation=True,
        max_length=int(config["max_input_tokens"]), add_special_tokens=False,
    )
    if full != encoded["input_ids"][0].tolist():
        raise ValueError("Empty-context generation prompt was truncated")
    return dict(encoded)


def generate(tokenizer, model, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    inputs = {key: value.to(model_device(model)) for key, value in inputs.items()}
    with torch.inference_mode():
        sequence = model.generate(
            **inputs, max_new_tokens=OUTPUT_LIMIT, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_count = int(inputs["input_ids"].shape[-1])
    ids = sequence[0, input_count:].tolist()
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
    stopped = bool(ids and ids[-1] in eos_ids)
    return {
        "response": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip(),
        "token_ids": ids,
        "input_token_count": input_count,
        "generated_token_count": len(ids),
        "eos_terminated": stopped,
        "output_limit_reached": len(ids) >= OUTPUT_LIMIT and not stopped,
    }


def run_case(case: dict[str, Any], tokenizer, model, config: dict[str, Any], detector) -> dict[str, Any]:
    context = case["external_context"]
    prompt = checked_prompt(tokenizer, case["user_instruction"], context, config)
    features, detection_response = capture_response_attention_with_tokens(prompt, model, tokenizer, config)
    logits = response_token_logits(detector, features)
    predictions = reference_token_predictions(logits)
    if len(predictions) != len(prompt.context_offsets):
        raise ValueError(f"Attention and context token counts differ: {case['case_id']}")
    baseline = generate(tokenizer, model, prompt.model_inputs)
    if baseline["token_ids"][:len(detection_response)] != detection_response:
        raise ValueError(f"96- and 384-token generation prefixes differ: {case['case_id']}")

    sanitized = remove_flagged_tokens(tokenizer, prompt, context, predictions)
    removed_positions = {
        i for (start, end), flag in zip(prompt.context_offsets, predictions, strict=True) if flag
        for i in range(start, end)
    }
    if sanitized != "".join(char for i, char in enumerate(context) if i not in removed_positions):
        raise ValueError(f"Decoded token removal differs from character offsets: {case['case_id']}")
    if sanitized == context:
        defended, response_source = baseline, "same_input_as_baseline"
    else:
        if sanitized:
            inputs = checked_prompt(tokenizer, case["user_instruction"], sanitized, config).model_inputs
        else:
            inputs = empty_context_inputs(tokenizer, case["user_instruction"], config)
        defended, response_source = generate(tokenizer, model, inputs), "new_generation"

    counts = {
        name: {"flagged": 0, "total": 0}
        for name in ("probe", "wrapper", "clean", "boundary", "separator")
    }
    tokens = []
    ids = prompt.model_inputs["input_ids"][0]
    for index, ((start, end), flag, scores) in enumerate(zip(
        prompt.context_offsets, predictions, logits.tolist(), strict=True
    )):
        region = region_for_token(
            start, end, len(case["clean_context"]), case["probe_start"], case["wrapper_start"]
        )
        counts[region]["total"] += 1
        counts[region]["flagged"] += flag
        tokens.append({
            "token_id": int(ids[prompt.context_token_indices[index]]),
            "start": start, "end": end, "region": region,
            "logits": scores, "flagged": flag,
        })
    if not counts["probe"]["total"]:
        raise ValueError(f"No probe tokens: {case['case_id']}")
    return {
        "type": "case", "case_id": case["case_id"], "sanitized_context": sanitized,
        "regions": counts, "tokens": tokens,
        "clean_characters_removed": sum(i < len(case["clean_context"]) for i in removed_positions),
        "input_injected": rennervate_input_detected(predictions),
        "max_consecutive_injected_tokens": longest_consecutive_positive_run(predictions),
        "detection_response_token_count": len(detection_response),
        "baseline": baseline, "defended": defended, "response_source": response_source,
    }


def export(cases: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for case in cases:
        result = results[case["case_id"]]
        for defense, context, generation in (
            ("none", case["external_context"], result["baseline"]),
            ("response_attention_token_removal", result["sanitized_context"], result["defended"]),
        ):
            row = {name: case[name] for name in (
                "case_id", "block_id", "source_family_id", "context_id", "probe_id",
                "intended_relation", "task_type", "user_instruction", "clean_context",
                "probe_text", "expected_answer",
            )}
            row["probe_wrapper"] = case["probe_wrapper"]
            row.update({
                "accepted_variants": json.dumps(case["accepted_variants"], ensure_ascii=False),
                "gold_task_answer": json.dumps(case["gold_task_answer"], ensure_ascii=False),
                "defense": defense, "external_context": context, "model_response": generation["response"],
                "clean_characters_total": len(case["clean_context"]),
                "input_token_count": generation["input_token_count"],
                "generated_token_count": generation["generated_token_count"],
                "output_limit_reached": generation["output_limit_reached"],
                "eos_terminated": generation["eos_terminated"],
                "response_source": "new_generation" if defense == "none" else result["response_source"],
            })
            if defense != "none":
                for name in ("probe", "wrapper", "clean", "boundary"):
                    for field in ("flagged", "total"):
                        row[f"{name}_tokens_{field}"] = result["regions"].get(
                            name, {"flagged": 0, "total": 0}
                        )[field]
                row["probe_token_detection_rate"] = (
                    row["probe_tokens_flagged"] / row["probe_tokens_total"]
                )
                row["wrapper_token_detection_rate"] = (
                    row["wrapper_tokens_flagged"] / row["wrapper_tokens_total"]
                    if row["wrapper_tokens_total"] else None
                )
                row["injection_tokens_flagged"] = (
                    row["probe_tokens_flagged"] + row["wrapper_tokens_flagged"]
                )
                row["injection_tokens_total"] = (
                    row["probe_tokens_total"] + row["wrapper_tokens_total"]
                )
                row["injection_token_detection_rate"] = (
                    row["injection_tokens_flagged"] / row["injection_tokens_total"]
                )
                row["clean_characters_removed"] = result["clean_characters_removed"]
                row["input_injected"] = result["input_injected"]
                row["max_consecutive_injected_tokens"] = result["max_consecutive_injected_tokens"]
            rows.append(row)
    groups = {}
    for relation in ("all", "related", "unrelated"):
        selected = [
            row for row in rows
            if row["defense"] != "none" and (relation == "all" or row["intended_relation"] == relation)
        ]
        flagged = sum(row["probe_tokens_flagged"] for row in selected)
        total = sum(row["probe_tokens_total"] for row in selected)
        wrapper_flagged = sum(row["wrapper_tokens_flagged"] for row in selected)
        wrapper_total = sum(row["wrapper_tokens_total"] for row in selected)
        injection_flagged = sum(row["injection_tokens_flagged"] for row in selected)
        injection_total = sum(row["injection_tokens_total"] for row in selected)
        groups[relation] = {
            "cases": len(selected), "probe_tokens_flagged": flagged, "probe_tokens_total": total,
            "probe_token_detection_rate": flagged / total,
            "wrapper_tokens_flagged": wrapper_flagged, "wrapper_tokens_total": wrapper_total,
            "wrapper_token_detection_rate": wrapper_flagged / wrapper_total if wrapper_total else None,
            "injection_tokens_flagged": injection_flagged, "injection_tokens_total": injection_total,
            "injection_token_detection_rate": injection_flagged / injection_total,
            "clean_tokens_flagged": sum(row["clean_tokens_flagged"] for row in selected),
            "clean_tokens_total": sum(row["clean_tokens_total"] for row in selected),
            "boundary_tokens_flagged": sum(row["boundary_tokens_flagged"] for row in selected),
            "boundary_tokens_total": sum(row["boundary_tokens_total"] for row in selected),
            "clean_characters_removed": sum(row["clean_characters_removed"] for row in selected),
            "clean_characters_total": sum(row["clean_characters_total"] for row in selected),
            "inputs_flagged": sum(row["input_injected"] for row in selected),
        }
    if len(rows) != 2 * len(cases):
        raise ValueError("Expected two paired response rows per case")
    return rows, {
        "status": "completed", "cases": len(cases), "responses": len(rows),
        "groups": groups,
        "output_limit_reached": sum(row["output_limit_reached"] for row in rows),
        "attack_success_and_task_accuracy_scored": False,
    }


def run(resume: bool, wrapped: bool = False) -> None:
    log, csv_path, summary_path = (
        (WRAPPER_LOG, WRAPPER_CSV, WRAPPER_SUMMARY)
        if wrapped else (LOG, CSV, SUMMARY)
    )
    probe_wrapper = FAKE_COMPLETION_WRAPPER if wrapped else None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit NF4 generation")
    for path in (csv_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if log.exists() != resume:
        raise ValueError("Use --resume only with an existing run log")
    cases = build_cases(read_json(CANDIDATE_FILE), probe_wrapper)
    config = read_model_config()
    if int(config["max_new_tokens"]) != 96:
        raise ValueError("Expected detector generation limit of 96")
    tokenizer, model = load_llm(config)
    if model_device(model).type != "cuda" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Expected CUDA 4-bit NF4 model")
    detector = load_response_detector(CHECKPOINT, config, model_device(model))
    if not sanity_check(tokenizer, model, config)["passed"]:
        raise RuntimeError("Generation sanity check failed")
    for case in cases:
        checked_prompt(tokenizer, case["user_instruction"], case["external_context"], config)
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("Model revision is missing")
    header = {
        "type": "header",
        "experiment_id": (
            "response_attention_64_wrapper_v1" if wrapped else "response_attention_64_rawprobe_v1"
        ),
        "candidate_sha256": sha256_file(CANDIDATE_FILE),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_config_sha256": sha256_file(MODEL_CONFIG), "model_config": config,
        "model_revision": revision,
        "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "chat_template": tokenizer.chat_template,
        "library_versions": {name: version(name) for name in ("torch", "transformers", "accelerate", "bitsandbytes")},
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__), ROOT / "src/senior_project/llm.py",
                ROOT / "src/senior_project/rennervate.py", ROOT / "src/senior_project/detector.py",
                ROOT / "src/senior_project/prompt.py", ROOT / "src/senior_project/token_removal.py",
                ROOT / "topic_relevance/scripts/run_pilot_expansion_250_token_removal_v1.py",
            )
        },
        "case_count": len(cases), "probe_wrapper": probe_wrapper,
        "task_constraint_ablation": True, "detection_max_new_tokens": 96,
        "response_max_new_tokens": OUTPUT_LIMIT, "do_sample": False,
        "detector_condition": "class-0 score, reference width-5 mean, <=0 positive; remove all positive tokens",
        "probe_rate_excludes_boundary_tokens": True,
    }
    results: dict[str, dict[str, Any]] = {}
    if resume:
        with log.open(encoding="utf-8") as file:
            if json.loads(next(file)) != header:
                raise ValueError("Saved experiment header differs from the current conditions")
            for line in file:
                entry = json.loads(line)
                key = entry["case_id"]
                if entry.get("type") != "case" or key in results:
                    raise ValueError(f"Unexpected or duplicate saved case: {key}")
                results[key] = entry
    if not set(results) <= {case["case_id"] for case in cases}:
        raise ValueError("Saved log contains cases absent from the source")
    with log.open("a" if resume else "x", encoding="utf-8") as file:
        if not resume:
            write_entry(file, header)
        for case in cases:
            if case["case_id"] in results:
                continue
            result = run_case(case, tokenizer, model, config, detector)
            results[case["case_id"]] = result
            write_entry(file, result)
            print(f"Completed {len(results)}/{len(cases)}: {case['case_id']}", flush=True)

    rows, summary = export(cases, results)
    with csv_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary.update({
        "probe_wrapper": probe_wrapper,
        "candidate_sha256": header["candidate_sha256"],
        "checkpoint_sha256": header["checkpoint_sha256"],
        "log_sha256": sha256_file(log),
        "csv_sha256": sha256_file(csv_path),
    })
    with summary_path.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the response-attention detector on 64 blocks.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wrapper", action="store_true", help="Insert the standard fake-completion wrapper.")
    arguments = parser.parse_args()
    run(arguments.resume, arguments.wrapper)
