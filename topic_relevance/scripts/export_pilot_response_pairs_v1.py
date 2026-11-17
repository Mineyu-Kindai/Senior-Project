import csv
import hashlib
import json
from pathlib import Path

from senior_project.prompt import redact_spans
from topic_relevance.scripts.run_rennervate_pilot_v1 import (
    ABLATION_REPORT,
    CALIBRATION_REPORT,
    CANDIDATE_FILE,
    CHECKPOINT,
    INTERVENTION_REPORT,
    MODEL_CONFIG,
    OUTPUT,
    SCORE_REPORT,
    build_evaluations,
    read_json,
    sha256_file,
)


EXPORT = (
    Path(__file__).resolve().parents[1]
    / "data/experiments/rennervate_pilot_201_264_response_pairs_v1.csv"
)

FIELDS = (
    "example_id",
    "block_id",
    "source_family_id",
    "context_id",
    "probe_id",
    "intended_relation",
    "defense",
    "user_instruction",
    "external_context",
    "model_response",
    "probe_tokens_flagged",
    "probe_tokens_total",
    "probe_token_detection_rate",
    "clean_characters_masked",
    "clean_characters_total",
    "clean_character_mask_rate",
    "input_token_count",
    "generated_token_count",
    "output_limit_reached",
    "eos_terminated",
)


def export(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    scores = read_json(SCORE_REPORT)
    source_report = read_json(ABLATION_REPORT)
    historical_report = read_json(INTERVENTION_REPORT)
    candidates = read_json(CANDIDATE_FILE)
    with OUTPUT.open(encoding="utf-8") as file:
        entries = [json.loads(line) for line in file]
    if not entries or entries[0].get("type") != "header" or entries[-1].get("type") != "complete":
        raise ValueError("The stored pilot run is incomplete")
    header = entries[0]
    for key, path in (
        ("score_report_sha256", SCORE_REPORT),
        ("checkpoint_sha256", CHECKPOINT),
        ("source_report_sha256", ABLATION_REPORT),
        ("historical_control_sha256", INTERVENTION_REPORT),
        ("candidate_file_sha256", CANDIDATE_FILE),
        ("calibration_report_sha256", CALIBRATION_REPORT),
        ("model_config_sha256", MODEL_CONFIG),
    ):
        if header[key] != sha256_file(path):
            raise ValueError(f"Stored pilot fingerprint does not match {path}")
    if historical_report["source_report_sha256"] != sha256_file(ABLATION_REPORT):
        raise ValueError("Historical baseline uses a different source report")
    if historical_report["candidate_sha256"] != sha256_file(CANDIDATE_FILE):
        raise ValueError("Historical baseline uses different candidates")
    if historical_report["status"] != "completed" or scores["status"] != "baseline_audit_passed":
        raise ValueError("The source reports are not complete")

    sources = {item["id"]: item for item in source_report["examples"]}
    historical = {
        (item["id"], item["condition"]): item
        for item in historical_report["examples"]
    }
    if len(sources) != 256 or len(historical) != len(historical_report["examples"]):
        raise ValueError("Source examples are missing or historical conditions are duplicated")
    jobs, expected, _ = build_evaluations(scores, sources, historical)
    expected_by_id = {
        item["source_ids"][0]: item
        for item in expected
        if item["input_kind"] == "injected"
    }
    stored = [item for item in entries[-1]["evaluations"] if item["input_kind"] == "injected"]
    if len(stored) != 256 or {item["source_ids"][0] for item in stored} != sources.keys():
        raise ValueError("Saved detector results do not cover the 256 attack inputs")
    generations = {
        item["input_sha256"]: item["generation"]
        for item in entries[1:-1]
        if item.get("type") == "generation"
    }
    if len(generations) != len(entries) - 2 or set(generations) != set(jobs):
        raise ValueError("Saved generation records are incomplete or duplicated")
    blocks = {item["block_id"]: item for item in candidates["blocks"]}
    if len(blocks) != 64 or {s["block_id"] for s in sources.values()} != blocks.keys():
        raise ValueError("The 64 candidate blocks do not match the stored pilot")

    rows = []
    for saved in stored:
        source_id = saved["source_ids"][0]
        source = sources[source_id]
        reconstructed = expected_by_id[source_id]
        block = blocks[source["block_id"]]
        if (
            source["clean_context"] != block["contexts"][source["context_id"]]["text"]
            or block["probes"][source["probe_id"]]["text"] not in source["external_context"]
        ):
            raise ValueError(f"Input does not match candidate block {source_id}")
        for key in (
            "input_sha256",
            "generation_input_sha256",
            "region_counts",
            "clean_characters_masked",
            "clean_characters_total",
            "sanitized_context_sha256",
        ):
            if saved[key] != reconstructed[key]:
                raise ValueError(f"Stored detector result differs for {source_id}: {key}")
        if saved["redacted_spans"] != [list(span) for span in reconstructed["redacted_spans"]]:
            raise ValueError(f"Stored mask spans differ for {source_id}")
        sanitized = redact_spans(source["external_context"], reconstructed["redacted_spans"])
        if hashlib.sha256(sanitized.encode("utf-8")).hexdigest() != saved["sanitized_context_sha256"]:
            raise ValueError(f"Sanitized context hash differs for {source_id}")
        baseline = historical[(source_id, "unprocessed_attack")]
        generation = generations[saved["generation_input_sha256"]]
        if saved["response"] != generation["response"]:
            raise ValueError(f"Stored response differs from generation for {source_id}")
        if baseline["response"] != reconstructed["baseline_response"]:
            raise ValueError(f"Stored baseline response differs for {source_id}")
        probe = saved["region_counts"]["probe"]
        if probe["tokens"] <= 0 or saved["clean_characters_total"] <= 0:
            raise ValueError(f"Empty metric denominator for {source_id}")
        common = {
            "example_id": source_id,
            "block_id": source["block_id"],
            "source_family_id": source["source_family_id"],
            "context_id": source["context_id"],
            "probe_id": source["probe_id"],
            "intended_relation": source["intended_relation"],
            "user_instruction": source["user_input"],
            "clean_characters_total": saved["clean_characters_total"],
        }
        rows.append(
            {
                **common,
                "defense": "none",
                "external_context": source["external_context"],
                "model_response": baseline["response"],
                "probe_tokens_flagged": "",
                "probe_tokens_total": "",
                "probe_token_detection_rate": "",
                "clean_characters_masked": 0,
                "clean_character_mask_rate": 0,
                "input_token_count": baseline["input_token_count"],
                "generated_token_count": baseline["generated_token_count"],
                "output_limit_reached": baseline["max_limit_reached"],
                "eos_terminated": baseline["eos_terminated"],
            }
        )
        rows.append(
            {
                **common,
                "defense": "rennervate",
                "external_context": sanitized,
                "model_response": generation["response"],
                "probe_tokens_flagged": probe["flagged_tokens"],
                "probe_tokens_total": probe["tokens"],
                "probe_token_detection_rate": probe["flagged_tokens"] / probe["tokens"],
                "clean_characters_masked": saved["clean_characters_masked"],
                "clean_character_mask_rate": (
                    saved["clean_characters_masked"] / saved["clean_characters_total"]
                ),
                "input_token_count": generation["input_token_count"],
                "generated_token_count": generation["generated_token_count"],
                "output_limit_reached": generation["max_limit_reached"],
                "eos_terminated": generation["eos_terminated"],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows for {len(stored)} paired attack inputs to {output}")


if __name__ == "__main__":
    export(EXPORT)
