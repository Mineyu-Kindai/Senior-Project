import csv
import hashlib
import json
from pathlib import Path

from topic_relevance.scorer.topic_relatedness import TopicRelatednessScorer


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "topic_relevance/data/candidates/pilot_201_264_candidates_v1.json"
OUTPUT = ROOT / "topic_relevance/data/experiments/relatedness_pilot_201_210_v1.csv"
REVISION = "4328cf26390c98c5e3c738b4460a05b95f4911f5"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

    source_bytes = SOURCE.read_bytes()
    data = json.loads(source_bytes)
    wanted = {f"pilot_{i}" for i in range(201, 211)}
    blocks = [block for block in data["blocks"] if block["block_id"] in wanted]
    if len(blocks) != 10 or {block["block_id"] for block in blocks} != wanted:
        raise ValueError("Expected exactly pilot_201 through pilot_210")

    scorer = TopicRelatednessScorer(device="cpu", revision=REVISION)
    rows = []
    pairs = []
    for block in sorted(blocks, key=lambda value: value["block_id"]):
        for context_id in ("A", "B"):
            for probe_id in ("A", "B"):
                context = block["contexts"][context_id]["text"]
                probe = block["probes"][probe_id]["text"]
                pairing = context_id + probe_id
                rows.append({
                    "block_id": block["block_id"],
                    "task_type": block["task_type"],
                    "pairing": pairing,
                    "intended_relation": block["intended_crossing"][pairing],
                    "context": context,
                    "probe": probe,
                    "context_tokens": len(scorer.model.tokenizer(
                        context, add_special_tokens=True, truncation=False
                    )["input_ids"]),
                    "probe_tokens": len(scorer.model.tokenizer(
                        probe, add_special_tokens=True, truncation=False
                    )["input_ids"]),
                })
                pairs.append((context, probe))

    scores = scorer.score_pairs(pairs)
    for row, score in zip(rows, scores, strict=True):
        row["cosine_similarity"] = score

    with OUTPUT.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Source SHA-256: {hashlib.sha256(source_bytes).hexdigest()}")
    print(f"Model: {scorer.model_name} @ {REVISION}")
    print(f"Maximum input tokens: {scorer.model.max_seq_length}")
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
