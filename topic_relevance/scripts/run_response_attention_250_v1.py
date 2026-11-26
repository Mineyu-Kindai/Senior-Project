import argparse
import csv
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from senior_project.llm import load_llm, model_device
from senior_project.rennervate import load_response_detector
from topic_relevance.scripts.run_ab_followup_v1 import (
    CANDIDATE_FILE as PREVIOUS_CANDIDATES,
    MODEL_CONFIG,
    read_json,
    read_model_config,
    sanity_check,
    sha256_file,
)
from topic_relevance.scripts.run_pilot_expansion_250_token_removal_v1 import (
    CANDIDATE_FILE,
    FAKE_COMPLETION_WRAPPER,
    ROOT,
    build_cases as build_expansion_cases,
    write_entry,
)
from topic_relevance.scripts.run_response_attention_64_v1 import (
    CHECKPOINT,
    FIELDS,
    OUTPUT_LIMIT,
    WRAPPER_LOG as PREVIOUS_LOG,
    build_cases as build_previous_cases,
    checked_prompt,
    export,
    run_case,
)


LOG = ROOT / "topic_relevance/data/experiments/response_attention_250_wrapper_v1.jsonl"
CSV = ROOT / "topic_relevance/data/experiments/response_attention_250_wrapper_responses_v1.csv"
SUMMARY = ROOT / "topic_relevance/data/experiments/response_attention_250_wrapper_summary_v1.json"
PREVIOUS_RUNNER_SHA256 = "54283fb0dfbacfe6ff1cfbf297d5dd6d2dc0dcf6248667e323facaff3db8ba32"
MATCH_FIELDS = (
    "case_id", "block_id", "source_family_id", "context_id", "probe_id",
    "intended_relation", "task_type", "user_instruction", "clean_context",
    "probe_text", "expected_answer", "accepted_variants", "gold_task_answer",
    "external_context", "wrapper_start", "probe_start",
)


def build_cases() -> list[dict[str, Any]]:
    cases = build_expansion_cases(read_json(CANDIDATE_FILE))
    for case in cases:
        case["probe_wrapper"] = FAKE_COMPLETION_WRAPPER
    return cases


def load_previous(cases: list[dict], header: dict) -> dict[str, dict]:
    previous_cases = {
        case["case_id"]: case
        for case in build_previous_cases(read_json(PREVIOUS_CANDIDATES), FAKE_COMPLETION_WRAPPER)
    }
    if header["candidate_sha256"] != sha256_file(PREVIOUS_CANDIDATES):
        raise ValueError("Previous candidate provenance mismatch")
    if header["code_sha256"]["topic_relevance/scripts/run_response_attention_64_v1.py"] != PREVIOUS_RUNNER_SHA256:
        raise ValueError("Unexpected previous runner version")
    for path, digest in header["code_sha256"].items():
        if path != "topic_relevance/scripts/run_response_attention_64_v1.py" and sha256_file(ROOT / path) != digest:
            raise ValueError(f"Code differs from previous run: {path}")
    if header["checkpoint_sha256"] != sha256_file(CHECKPOINT):
        raise ValueError("Detector checkpoint differs from previous run")
    if header["model_config_sha256"] != sha256_file(MODEL_CONFIG):
        raise ValueError("Model configuration differs from previous run")
    if header["library_versions"] != {name: version(name) for name in header["library_versions"]}:
        raise ValueError("Library versions differ from previous run")
    if (
        header["probe_wrapper"] != FAKE_COMPLETION_WRAPPER
        or header["detection_max_new_tokens"] != 96
        or header["response_max_new_tokens"] != OUTPUT_LIMIT
        or header["do_sample"] is not False
    ):
        raise ValueError("Previous experiment conditions differ")

    reusable = {
        case["case_id"] for case in cases
        if case["case_id"] in previous_cases
        and all(case[field] == previous_cases[case["case_id"]][field] for field in MATCH_FIELDS)
    }
    records = {}
    with PREVIOUS_LOG.open(encoding="utf-8") as file:
        if json.loads(next(file)) != header:
            raise ValueError("Previous header changed")
        for line in file:
            record = json.loads(line)
            case_id = record["case_id"]
            if record.get("type") != "case" or case_id in records or case_id not in previous_cases:
                raise ValueError(f"Unexpected or duplicate previous case: {case_id}")
            records[case_id] = record
    if set(records) != set(previous_cases):
        raise ValueError("Previous log does not contain all 256 cases")
    return {case_id: records[case_id] for case_id in reusable}


def run(resume: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit NF4 generation")
    for path in (CSV, SUMMARY):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if LOG.exists() != resume:
        raise ValueError("Use --resume only with an existing run log")
    cases = build_cases()
    with PREVIOUS_LOG.open(encoding="utf-8") as file:
        previous_header = json.loads(next(file))
    reusable = load_previous(cases, previous_header)
    config = read_model_config()
    tokenizer, model = load_llm(config)
    if model_device(model).type != "cuda" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Expected CUDA 4-bit NF4 model")
    revision = getattr(model.config, "_commit_hash", None)
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if (
        revision != previous_header["model_revision"]
        or tokenizer_hash != previous_header["tokenizer_backend_sha256"]
        or tokenizer.chat_template != previous_header["chat_template"]
    ):
        raise ValueError("Model revision, tokenizer, or chat template differs from previous run")
    detector = load_response_detector(CHECKPOINT, config, model_device(model))
    if not sanity_check(tokenizer, model, config)["passed"]:
        raise RuntimeError("Generation sanity check failed")
    for case in cases:
        checked_prompt(tokenizer, case["user_instruction"], case["external_context"], config)

    header = {
        "type": "header",
        "experiment_id": "response_attention_250_wrapper_v1",
        "candidate_sha256": sha256_file(CANDIDATE_FILE),
        "previous_log_sha256": sha256_file(PREVIOUS_LOG),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "model_config": config,
        "model_revision": revision,
        "tokenizer_backend_sha256": tokenizer_hash,
        "chat_template": tokenizer.chat_template,
        "library_versions": previous_header["library_versions"],
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__),
                ROOT / "topic_relevance/scripts/run_response_attention_64_v1.py",
                ROOT / "topic_relevance/scripts/run_pilot_expansion_250_token_removal_v1.py",
                ROOT / "src/senior_project/llm.py",
                ROOT / "src/senior_project/rennervate.py",
                ROOT / "src/senior_project/detector.py",
                ROOT / "src/senior_project/prompt.py",
                ROOT / "src/senior_project/token_removal.py",
            )
        },
        "case_count": len(cases),
        "reusable_case_count": len(reusable),
        "probe_wrapper": FAKE_COMPLETION_WRAPPER,
        "task_constraint_ablation": True,
        "detection_max_new_tokens": 96,
        "response_attention_limit": 32,
        "response_max_new_tokens": OUTPUT_LIMIT,
        "do_sample": False,
        "detector_condition": previous_header["detector_condition"],
        "probe_rate_excludes_boundary_tokens": True,
    }
    results: dict[str, dict] = {}
    if resume:
        with LOG.open(encoding="utf-8") as file:
            if json.loads(next(file)) != header:
                raise ValueError("Saved experiment conditions differ")
            for line in file:
                entry = json.loads(line)
                case_id = entry["case_id"]
                if entry.get("type") != "case" or case_id in results:
                    raise ValueError(f"Unexpected or duplicate saved case: {case_id}")
                if entry["case_source"] not in ("previous_64_wrapper", "current_run"):
                    raise ValueError(f"Unexpected case source: {case_id}")
                results[case_id] = entry
    if not set(results) <= {case["case_id"] for case in cases}:
        raise ValueError("Saved log contains unknown cases")
    with LOG.open("a" if resume else "x", encoding="utf-8") as file:
        if not resume:
            write_entry(file, header)
        for case in cases:
            case_id = case["case_id"]
            if case_id in results:
                continue
            if case_id in reusable:
                result = {**reusable[case_id], "case_source": "previous_64_wrapper"}
            else:
                result = {**run_case(case, tokenizer, model, config, detector), "case_source": "current_run"}
            results[case_id] = result
            write_entry(file, result)
            print(f"Completed {len(results)}/{len(cases)}: {case_id} ({result['case_source']})", flush=True)

    rows, summary = export(cases, results)
    for row in rows:
        source = results[row["case_id"]]["case_source"]
        row["case_source"] = source
        if source == "previous_64_wrapper":
            row["response_source"] = source
    with CSV.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(*FIELDS, "case_source"))
        writer.writeheader()
        writer.writerows(rows)
    summary.update({
        "case_sources": dict(Counter(result["case_source"] for result in results.values())),
        "response_sources": dict(Counter(row["response_source"] for row in rows)),
        "probe_wrapper": FAKE_COMPLETION_WRAPPER,
        "candidate_sha256": header["candidate_sha256"],
        "previous_log_sha256": header["previous_log_sha256"],
        "checkpoint_sha256": header["checkpoint_sha256"],
        "log_sha256": sha256_file(LOG),
        "csv_sha256": sha256_file(CSV),
    })
    with SUMMARY.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate response attention on 250 wrapped blocks.")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args().resume)
