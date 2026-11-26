import hashlib
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from senior_project.detector import filter_short_positive_runs
from senior_project.llm import prepare_prompt
from senior_project.token_removal import remove_flagged_tokens
from topic_relevance.scripts.run_ab_followup_v1 import read_model_config, sha256_file
from topic_relevance.scripts.run_pilot_expansion_250_token_removal_v1 import (
    CANDIDATE_FILE,
    LOG,
    build_cases,
    input_hash,
    read_json,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "topic_relevance/data/experiments/filtered_removal_development_64_v1.json"
MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
RUN_LENGTHS = (1, 2, 3, 4, 5, 6)


def metrics(case: dict, tokens: list[dict], decisions: list[int]) -> dict[str, int]:
    result = {"probe": 0, "wrapper": 0, "clean_characters_removed": 0}
    clean_indices: set[int] = set()
    for token, flag in zip(tokens, decisions, strict=True):
        if not flag:
            continue
        if token["region"] in ("probe", "wrapper"):
            result[token["region"]] += 1
        clean_indices.update(range(max(0, token["start"]), min(case["clean_end"], token["end"])))
    result["clean_characters_removed"] = len(clean_indices)
    return result


def run() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    cases = build_cases(read_json(CANDIDATE_FILE))[:256]
    if {case["block_id"] for case in cases} != {
        f"pilot_{number}" for number in range(201, 265)
    }:
        raise ValueError("Expected the first 64 exploratory blocks")
    by_id = {case["case_id"]: case for case in cases}
    records = {}
    with LOG.open(encoding="utf-8") as file:
        header = json.loads(next(file))
        if header["candidate_sha256"] != sha256_file(CANDIDATE_FILE):
            raise ValueError("The saved score log refers to a different candidate file")
        for line in file:
            item = json.loads(line)
            if item.get("type") == "case" and item["case_id"] in by_id:
                if item["case_id"] in records:
                    raise ValueError(f"Duplicate record: {item['case_id']}")
                records[item["case_id"]] = item
                if len(records) == len(cases):
                    break
    if len(records) != 256:
        raise ValueError("Expected 256 saved development-case records")

    baseline = {"probe": 0, "wrapper": 0, "clean_characters_removed": 0}
    for case in cases:
        record = records[case["case_id"]]
        if record["input_sha256"] != input_hash(case["user_instruction"], case["external_context"]):
            raise ValueError(f"Input mismatch: {case['case_id']}")
        tokens = record["tokens"]
        if any(not math.isfinite(t["smoothed_injected_minus_clean_logit"]) for t in tokens):
            raise ValueError(f"Nonfinite smoothed logit: {case['case_id']}")
        decisions = [int(token["flagged"]) for token in tokens]
        if decisions != [int(token["smoothed_injected_minus_clean_logit"] > 0) for token in tokens]:
            raise ValueError(f"Saved flags do not match smoothed margins: {case['case_id']}")
        for key, value in metrics(case, tokens, decisions).items():
            baseline[key] += value
    if not baseline["probe"] or not baseline["wrapper"]:
        raise ValueError("Baseline detection counts must be nonzero")

    results = []
    for margin in MARGINS:
        for min_run_length in RUN_LENGTHS:
            totals = {"probe": 0, "wrapper": 0, "clean_characters_removed": 0}
            for case in cases:
                tokens = records[case["case_id"]]["tokens"]
                raw = [int(t["smoothed_injected_minus_clean_logit"] > margin) for t in tokens]
                decisions = filter_short_positive_runs(raw, min_run_length)
                for key, value in metrics(case, tokens, decisions).items():
                    totals[key] += value
            passes = (
                10 * totals["probe"] >= 9 * baseline["probe"]
                and 10 * totals["wrapper"] >= 9 * baseline["wrapper"]
            )
            results.append({
                "logit_margin": margin,
                "min_run_length": min_run_length,
                **totals,
                "probe_retention": totals["probe"] / baseline["probe"],
                "wrapper_retention": totals["wrapper"] / baseline["wrapper"],
                "passes_retention_constraint": passes,
            })
    eligible = [row for row in results if row["passes_retention_constraint"]]
    if not eligible:
        raise ValueError("No candidate preserves at least 90% of both baseline regions")
    selected = min(eligible, key=lambda row: (
        row["clean_characters_removed"], -row["probe"], -row["wrapper"],
        row["logit_margin"], row["min_run_length"],
    ))

    config = read_model_config()
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True, local_files_only=True)
    tokenizer_sha256 = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")
    ).hexdigest()
    if tokenizer_sha256 != header["tokenizer_backend_sha256"]:
        raise ValueError("Current tokenizer does not match the saved token scores")
    for case in cases:
        record = records[case["case_id"]]
        tokens = record["tokens"]
        prompt = prepare_prompt(tokenizer, case["user_instruction"], case["external_context"], config)
        ids = prompt.model_inputs["input_ids"][0]
        if (
            prompt.context_offsets != [(t["start"], t["end"]) for t in tokens]
            or [int(ids[i]) for i in prompt.context_token_indices] != [t["token_id"] for t in tokens]
        ):
            raise ValueError(f"Tokenization mismatch: {case['case_id']}")
        flags = filter_short_positive_runs(
            [int(t["smoothed_injected_minus_clean_logit"] > selected["logit_margin"]) for t in tokens],
            selected["min_run_length"],
        )
        sanitized = remove_flagged_tokens(tokenizer, prompt, case["external_context"], flags)
        removed = {
            index
            for token, flag in zip(tokens, flags, strict=True) if flag
            for index in range(token["start"], token["end"])
        }
        by_char = "".join(
            char for index, char in enumerate(case["external_context"]) if index not in removed
        )
        if sanitized != by_char:
            raise ValueError(f"Token decoding differs from character removal: {case['case_id']}")

    result = {
        "status": "development_selection_only",
        "candidate_sha256": header["candidate_sha256"],
        "score_log_sha256": sha256_file(LOG),
        "block_ids": sorted({case["block_id"] for case in cases}, key=lambda x: int(x.split("_")[1])),
        "case_count": len(cases),
        "candidate_margins": MARGINS,
        "candidate_min_run_lengths": RUN_LENGTHS,
        "selection_rule": (
            "Retain at least 90% of baseline flagged probe tokens and 90% of baseline "
            "flagged wrapper tokens after removal filtering; minimize removed clean "
            "characters. Ties: maximize probe, then wrapper retained counts; then "
            "prefer smaller margin and smaller minimum run length."
        ),
        "boundary_policy": "Excluded from probe and wrapper counts; counted in clean character removal.",
        "baseline": baseline,
        "selected": selected,
        "candidates": results,
        "verification": "Selected token removal equals character-offset deletion for all 256 cases.",
        "limitations": "Exploratory development data; no generated responses or attack/task outcomes evaluated.",
    }
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({"baseline": baseline, "selected": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
