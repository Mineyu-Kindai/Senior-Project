import copy
import csv
import hashlib
import json
from pathlib import Path

from topic_relevance.scorer.topic_relatedness import TopicRelatednessScorer


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "topic_relevance/data/candidates/pilot_201_264_candidates_v1.json"
OUTPUT = ROOT / "topic_relevance/data/candidates/pilot_expansion_original_64_draft_v1.json"
SCORES = ROOT / "topic_relevance/data/experiments/pilot_expansion_original_64_scores_v1.csv"
REVISION = "4328cf26390c98c5e3c738b4460a05b95f4911f5"
PAIRINGS = ("AA", "AB", "BA", "BB")


def main() -> None:
    for path in (OUTPUT, SCORES):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    raw = SOURCE.read_bytes()
    original = json.loads(raw)["blocks"]
    expected_ids = {f"pilot_{i}" for i in range(201, 265)}
    if len(original) != 64 or {b["block_id"] for b in original} != expected_ids:
        raise ValueError("Expected the 64 distinct pilot 201–264 blocks")

    revised = copy.deepcopy(original)
    target = next(b for b in revised if b["block_id"] == "pilot_257")
    target["contexts"]["B"]["topic"] = "大気中メタン濃度の調査"
    target["contexts"]["B"]["text"] = (
        "The Greenhouse team monitored atmospheric methane concentration at "
        "Greenhouse Test Bay 3 on November 19, 2026. A review at Greenhouse "
        "Operations Office was scheduled for November 27, 2026."
    )
    target["probes"]["B"]["text"] = (
        "As a separate question, what instrument measures methane concentration in air?"
    )
    target["probes"]["B"]["expected_answer"] = "methane analyzer"
    target["probes"]["B"]["accepted_variants"] = ["methane analyser"]
    target["generator_notes"]["crossing_rationale"] = {
        "AA": "Probe A asks about water temperature, the subject of context A.",
        "AB": "Probe B asks about methane rather than water temperature.",
        "BA": "Probe A asks about water temperature rather than methane.",
        "BB": "Probe B asks about methane concentration, the subject of context B.",
    }
    target["generator_notes"]["possible_confounds"].append(
        "Context B and probe B were revised after the original failed the cosine "
        "ordering criterion; their answer and relevance require fresh review."
    )
    target["generator_notes"]["unresolved_issues"].append(
        "Recheck the revised context B and probe B independently; previous pilot "
        "results for the original block do not apply to the revised block."
    )

    scorer = TopicRelatednessScorer(device="cpu", revision=REVISION)
    rows = []
    for version, blocks in (("original", original), ("draft", revised)):
        pairs = [
            (b["contexts"][pair[0]]["text"], b["probes"][pair[1]]["text"])
            for b in blocks
            for pair in PAIRINGS
        ]
        values = scorer.score_pairs(pairs)
        for index, block in enumerate(blocks):
            scores = values[index * 4:index * 4 + 4]
            margin = min(scores[0], scores[3]) - max(scores[1], scores[2])
            rows.append({
                "version": version,
                "block_id": block["block_id"],
                "s_A_A": scores[0],
                "s_A_B": scores[1],
                "s_B_A": scores[2],
                "s_B_B": scores[3],
                "margin": margin,
                "passes": margin > 0,
            })
    if not all(row["passes"] for row in rows if row["version"] == "draft"):
        raise ValueError("A revised original block still fails the cosine criterion")

    result = {
        "collection_id": "pilot_expansion_original_64_draft_v1",
        "status": "incomplete_expansion_draft_not_250",
        "candidate_count": len(revised),
        "target_count": 250,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "revised_block_ids": ["pilot_257"],
        "blocks": revised,
    }
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with SCORES.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(revised)} draft blocks and {len(rows)} score rows.")
    print(f"Original failures: {[r['block_id'] for r in rows if r['version'] == 'original' and not r['passes']]}")
    print(f"Source SHA-256: {result['source_sha256']}")


if __name__ == "__main__":
    main()
