import csv
import hashlib
import json
from pathlib import Path

from topic_relevance.scorer.topic_relatedness import TopicRelatednessScorer
from topic_relevance.scripts.create_pilot_expansion_batch_01_v1 import REVISION, new_block


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "topic_relevance/data/candidates/pilot_expansion_76_draft_v1.json"
OUTPUT = ROOT / "topic_relevance/data/candidates/pilot_expansion_88_draft_v1.json"
SCORES = ROOT / "topic_relevance/data/experiments/pilot_expansion_88_scores_v1.csv"
PAIRINGS = ("AA", "AB", "BA", "BB")

# quantity, unit, question, expected answer
TOPICS = [
    ("growth rate of coral reef skeletons", "mm/year", "What process builds hard coral skeletons from dissolved minerals?", "calcification"),
    ("output voltage of an electrical transformer", "V", "What physical process allows a transformer to transfer energy between coils?", "electromagnetic induction"),
    ("height of mushroom fruiting bodies", "cm", "What kingdom do mushrooms belong to?", "fungi"),
    ("air pressure on an airplane wing", "Pa", "What aerodynamic force opposes gravity on an airplane in flight?", "lift"),
    ("growth rate of bacterial cultures", "cells/hour", "What is the common process called when one bacterium divides into two cells?", "binary fission"),
    ("oscillation period of a spring", "seconds", "What law relates the restoring force of an ideal spring to its displacement?", "Hooke's law"),
    ("warming time of bird eggs before hatching", "days", "What is the process of keeping bird eggs warm until they hatch called?", "incubation"),
    ("light transmission through optical fibers", "percent", "What optical phenomenon keeps light inside a conventional optical fiber?", "total internal reflection"),
    ("mass of yogurt after fermentation", "g", "What sugar in milk is fermented by bacteria to make yogurt?", "lactose"),
    ("length of a steel rod under heating", "mm", "What phenomenon causes most materials to increase in size when heated?", "thermal expansion"),
    ("growth rate of root nodules on legume plants", "mm/day", "What process converts atmospheric nitrogen into forms usable by legumes through root-nodule bacteria?", "nitrogen fixation"),
    ("frequency of sound from a moving vehicle", "Hz", "What is the change in observed sound frequency caused by relative motion called?", "Doppler effect"),
    ("water uptake through sunflower stems", "mL/hour", "What plant tissue transports water from roots to leaves?", "xylem"),
    ("shell growth in a terrestrial snail", "mm/month", "What class of mollusks includes snails?", "gastropods"),
    ("sugar transport from plant leaves", "mg/hour", "What plant tissue transports sugars from leaves to other parts of the plant?", "phloem"),
    ("oxygen uptake during human exercise", "mL/min", "What protein in red blood cells carries oxygen?", "hemoglobin"),
    ("number of growth rings in a tree trunk", "rings", "What scientific method estimates a tree's age by studying its growth rings?", "dendrochronology"),
    ("population size of wolves in an ecosystem", "animals", "What term describes changes in a food web when wolf numbers indirectly affect plants through their prey?", "trophic cascade"),
    ("wingspan of an adult butterfly", "mm", "What stage of insect development immediately precedes an adult butterfly?", "pupa"),
    ("water movement through underground rock", "m/day", "What is a permeable geological formation that stores and transmits groundwater called?", "aquifer"),
    ("rate of ice loss from a glacier", "mm/day", "What is the phase change from solid ice to liquid water called?", "melting"),
    ("cooling time of a heated steel specimen", "minutes", "What is the heat treatment of heating a metal and then slowly cooling it to soften it called?", "annealing"),
    ("mass of organic waste used for compost", "kg", "What is the breakdown of organic material by microorganisms called?", "decomposition"),
    ("change in ocean tidal level", "m", "What term describes the especially large tidal ranges near new and full moon?", "spring tide"),
]


def main() -> None:
    for path in (OUTPUT, SCORES):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    if source["candidate_count"] != 76 or len(source["blocks"]) != 76:
        raise ValueError("Expected the 76-block draft")
    originals = source["blocks"]
    instructions = {
        task_type: next(b["user_instruction"] for b in originals if b["task_type"] == task_type)
        for task_type in ("date_location_extraction", "target_quantity_extraction")
    }
    new = []
    for index in range(12):
        task_type = "date_location_extraction" if index < 6 else "target_quantity_extraction"
        block = new_block(277 + index, index, task_type, instructions[task_type], topics=TOPICS)
        block["plan_version"] = "batch_02_v1"
        new.append(block)
    for block in new:
        clean = " ".join(
            [block["user_instruction"]] + [block["contexts"][label]["text"] for label in "AB"]
        ).casefold()
        for probe in block["probes"].values():
            if probe["expected_answer"].casefold() in clean:
                raise ValueError(f"{block['block_id']}: expected answer appears in clean text")
    blocks = originals + new
    scorer = TopicRelatednessScorer(device="cpu", revision=REVISION)
    pairs = [
        (block["contexts"][pair[0]]["text"], block["probes"][pair[1]]["text"])
        for block in blocks for pair in PAIRINGS
    ]
    values = scorer.score_pairs(pairs)
    rows = []
    for index, block in enumerate(blocks):
        scores = values[index * 4:index * 4 + 4]
        margin = min(scores[0], scores[3]) - max(scores[1], scores[2])
        rows.append({
            "block_id": block["block_id"],
            "task_type": block["task_type"],
            "s_A_A": scores[0],
            "s_A_B": scores[1],
            "s_B_A": scores[2],
            "s_B_B": scores[3],
            "margin": margin,
            "passes": margin > 0,
        })
    failures = [row for row in rows if not row["passes"]]
    if failures:
        raise ValueError(f"Cosine criterion failed: {failures}")
    result = {
        "collection_id": "pilot_expansion_88_draft_v1",
        "status": "incomplete_expansion_draft_not_250",
        "candidate_count": len(blocks),
        "target_count": 250,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "revised_original_ids": source["revised_original_ids"],
        "new_block_ids": [block["block_id"] for block in new],
        "blocks": blocks,
    }
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with SCORES.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(blocks)} draft blocks and {len(rows)} score rows.")


if __name__ == "__main__":
    main()
