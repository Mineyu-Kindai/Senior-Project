import argparse
import json
import math
from pathlib import Path
from typing import Any

from senior_project.prompt import merge_flagged_spans
from topic_relevance.scripts.capture_threshold_scores_v1 import input_hash
from topic_relevance.scripts.run_ab_followup_v1 import ABLATION_REPORT, ROOT, sha256_file


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
INJECTED_REGIONS = ("clean_context", "fake_completion_wrapper", "probe", "boundary")


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Cannot calculate a rate with an empty denominator")
    return numerator / denominator


def summarize_record(record: dict[str, Any], threshold: float) -> dict[str, Any]:
    tokens = record["tokens"]
    allowed = set(INJECTED_REGIONS) if record["input_kind"] == "injected" else {"clean_context"}
    counts = {name: {"flagged": 0, "total": 0} for name in allowed}
    offsets = []
    predictions = []
    context = record["external_context"]
    for token in tokens:
        start, end, score, region = (
            token["start"], token["end"], token["score"], token["evaluation_region"]
        )
        if not (0 <= start < end <= len(context)):
            raise ValueError("Invalid token offset")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid token score")
        if region not in allowed:
            raise ValueError(f"Unexpected token region: {region}")
        flagged = score >= threshold
        counts[region]["total"] += 1
        counts[region]["flagged"] += int(flagged)
        offsets.append((start, end))
        predictions.append(int(flagged))

    spans = merge_flagged_spans(offsets, predictions)
    clean_region = next(r for r in record["regions"] if r["name"] == "clean_context")
    clean_start, clean_end = clean_region["start"], clean_region["end"]
    masked_chars = sum(
        max(0, min(end, clean_end) - max(start, clean_start))
        for start, end in spans
    )
    return {
        "input_kind": record["input_kind"],
        "id": record.get("id"),
        "input_sha256": record["input_sha256"],
        "region_token_counts": counts,
        "clean_characters_masked": masked_chars,
        "clean_characters_total": clean_end - clean_start,
        "masked_spans": spans,
    }


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for threshold in THRESHOLDS:
        details = [summarize_record(record, threshold) for record in records]
        groups: dict[str, dict[str, Any]] = {}
        for kind in ("injected", "clean_only"):
            group = [item for item in details if item["input_kind"] == kind]
            if not group:
                raise ValueError(f"Missing input kind: {kind}")
            regions = INJECTED_REGIONS if kind == "injected" else ("clean_context",)
            region_summary = {}
            for name in regions:
                flagged = sum(item["region_token_counts"][name]["flagged"] for item in group)
                total = sum(item["region_token_counts"][name]["total"] for item in group)
                region_summary[name] = {
                    "flagged": flagged,
                    "total": total,
                    "rate": rate(flagged, total),
                }
            masked = sum(item["clean_characters_masked"] for item in group)
            total_chars = sum(item["clean_characters_total"] for item in group)
            groups[kind] = {
                "input_count": len(group),
                "regions": region_summary,
                "clean_characters_masked": masked,
                "clean_characters_total": total_chars,
                "clean_character_mask_rate": rate(masked, total_chars),
            }
        results.append({"threshold": threshold, "groups": groups, "inputs": details})

    for earlier, later in zip(results, results[1:]):
        for kind in ("injected", "clean_only"):
            before, after = earlier["groups"][kind], later["groups"][kind]
            if after["clean_characters_masked"] > before["clean_characters_masked"]:
                raise ValueError("Clean character masking increased with threshold")
            for name in before["regions"]:
                if after["regions"][name]["flagged"] > before["regions"][name]["flagged"]:
                    raise ValueError("Flagged token count increased with threshold")
    return results


def run(score_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    source = json.loads(ABLATION_REPORT.read_text(encoding="utf-8"))
    if scores["status"] != "baseline_audit_passed":
        raise ValueError("The 0.5 baseline audit did not pass")
    if scores["source_report_sha256"] != sha256_file(ABLATION_REPORT):
        raise ValueError("The source report differs from the score capture")
    records = scores["records"]
    injected = [r for r in records if r["input_kind"] == "injected"]
    clean = [r for r in records if r["input_kind"] == "clean_only"]
    if len(records) != 384 or len(injected) != 256 or len(clean) != 128:
        raise ValueError("Expected 256 injected and 128 clean input records")
    sources = {example["id"]: example for example in source["examples"]}
    if len(sources) != 256 or {r["id"] for r in injected} != set(sources):
        raise ValueError("Injected input IDs differ from the source report")
    expected_clean = {
        input_hash(example["user_input"], example["clean_context"])
        for example in sources.values()
    }
    if {r["input_sha256"] for r in clean} != expected_clean or len(expected_clean) != 128:
        raise ValueError("Clean inputs differ from the source report")
    for record in records:
        if record["input_sha256"] != input_hash(record["user_input"], record["external_context"]):
            raise ValueError("Stored input hash does not match the stored input")
        if record["input_kind"] == "injected":
            example = sources[record["id"]]
            if record["input_sha256"] != input_hash(example["user_input"], example["external_context"]):
                raise ValueError("Injected input differs from the source report")
            if record["regions"] != example["detection_regions"]:
                raise ValueError("Injected evaluation regions differ from the source report")

    report = {
        "status": "completed",
        "score_report_sha256": sha256_file(score_path),
        "source_report_sha256": sha256_file(ABLATION_REPORT),
        "script_sha256": sha256_file(Path(__file__)),
        "scope": "Token and mask metrics only; no model generation or threshold selection.",
        "boundary_policy": "Boundary tokens are reported separately and excluded from region rates.",
        "character_policy": "Merged mask spans intersected with clean character ranges.",
        "threshold_results": summarize(records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote threshold mask metrics to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-score threshold masks.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.scores, args.output)


if __name__ == "__main__":
    main()
