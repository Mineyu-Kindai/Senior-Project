import copy
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from topic_relevance.scorer.topic_relatedness import TopicRelatednessScorer


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "topic_relevance/data/candidates/pilot_expansion_original_64_draft_v1.json"
OUTPUT = ROOT / "topic_relevance/data/candidates/pilot_expansion_76_draft_v1.json"
SCORES = ROOT / "topic_relevance/data/experiments/pilot_expansion_76_scores_v1.csv"
REVISION = "4328cf26390c98c5e3c738b4460a05b95f4911f5"
PAIRINGS = ("AA", "AB", "BA", "BB")

# quantity, unit, question, expected answer
TOPICS = [
    ("photosynthesis rate in aquatic plants", "mg/hour", "What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("lava flow speed on a volcanic slope", "m/hour", "What is molten rock called while it is still beneath a volcano?", "magma"),
    ("flower visits by bees", "visits/hour", "What substance do bees transfer between flowers during pollination?", "pollen"),
    ("bread dough volume during fermentation", "cm3", "What microorganism is commonly used to make bread dough rise?", "yeast"),
    ("development time from tadpole to adult frog", "days", "What is the process called when a tadpole develops into an adult frog?", "metamorphosis"),
    ("water loss rate from a shallow pond", "mm/day", "What is the process called when liquid water changes into water vapor?", "evaporation"),
    ("power output of a solar cell", "W", "What effect allows a solar cell to convert light into electricity?", "photovoltaic effect"),
    ("duration of successive phases of the Moon", "days", "What body does the Moon orbit as its phases change?", "Earth"),
    ("number of books borrowed from a library", "books", "What classification system commonly organizes library books by numbers from 000 to 999?", "Dewey Decimal Classification"),
    ("number of internet domain lookups", "lookups", "What system translates internet domain names into IP addresses?", "Domain Name System"),
    ("crust thickness at a geological survey site", "km", "What layer of Earth lies directly beneath the crust?", "mantle"),
    ("ocean trench depth in a bathymetric survey", "m", "What is the deepest known ocean trench called?", "Mariana Trench"),
    ("duration of a whale dive", "minutes", "What organs do whales use to breathe air?", "lungs"),
    ("development time of a butterfly", "days", "What is the immature stage of a butterfly commonly called?", "caterpillar"),
    ("electric current in a light-emitting diode", "mA", "What class of material is used to make a light-emitting diode?", "semiconductor"),
    ("length of a computer digit sequence", "digits", "What number system uses only the digits zero and one?", "binary"),
    ("width of a circle on a drawing", "mm", "What is a straight line segment through a circle's center connecting two points on the circle called?", "diameter"),
    ("size of an angle in a triangle", "degrees", "What is a triangle with three equal sides called?", "equilateral triangle"),
    ("duration of blood circulation through the body", "seconds", "What organ pumps blood through the human body?", "heart"),
    ("volume of air inhaled during breathing", "L", "What muscle beneath the lungs contracts during normal inhalation?", "diaphragm"),
    ("time for iron corrosion in damp air", "days", "What common reddish-brown corrosion product forms on iron?", "rust"),
    ("mass of water consumed in a chemical reaction", "g", "Which two chemical elements make up water?", "hydrogen and oxygen"),
    ("distance traveled by migrating salmon", "km", "What kind of water do salmon typically enter to spawn after living in the ocean?", "freshwater"),
    ("duration of a honeybee's foraging trip", "minutes", "What dance do honeybees use to communicate the direction of a food source?", "waggle dance"),
]


def new_block(
    number: int,
    index: int,
    task_type: str,
    instruction: str,
    topics: list[tuple[str, str, str, str]] = TOPICS,
) -> dict:
    contexts = {}
    probes = {}
    for offset, label in enumerate(("A", "B")):
        quantity, unit, question, answer = topics[index * 2 + offset]
        if task_type == "date_location_extraction":
            day = date(2027, 1, 5) + timedelta(days=index * 4 + offset)
            review_day = day + timedelta(days=7)
            location = f"Research Station {number}-{label}"
            review_location = f"Review Office {number}-{label}"
            day_text = f"{day.strftime('%B')} {day.day}, {day.year}"
            review_text = f"{review_day.strftime('%B')} {review_day.day}, {review_day.year}"
            text = (
                f"The research team investigated {quantity} at {location} on {day_text}. "
                f"A review at {review_location} was scheduled for {review_text}."
            )
            gold = {"event_date": day.isoformat(), "event_location": location}
            evidence = {"event_date": [day_text], "event_location": [location]}
        else:
            preliminary = str(10 + index * 2 + offset)
            final = str(13 + index * 2 + offset)
            text = (
                f"The research team compared {quantity} across trials. "
                f"Preliminary measurement: {quantity} = {preliminary} {unit}. "
                f"Final measurement: {quantity} = {final} {unit}."
            )
            gold = {"measured_quantity": quantity, "value": final, "unit": unit}
            evidence = {"measured_quantity": [quantity], "value": [final], "unit": [unit]}
        contexts[label] = {
            "topic": quantity,
            "text": text,
            "synthetic": True,
            "gold_answer": gold,
            "evidence_quotes": evidence,
        }
        probes[label] = {
            "text": f"As a separate question, {question}",
            "expected_answer": answer,
            "accepted_variants": [],
            "answer_type": "short general-knowledge answer",
            "verification_status": "unverified",
            "verification_basis": [],
            "verification_needed": [
                "Verify the answer and accepted variants against a reliable source.",
                "Independently rate all four context-probe pairings for topic relevance.",
                "Check answer leakage and comparability of both probes.",
            ],
        }
    return {
        "status": "candidate",
        "block_id": f"pilot_{number}",
        "source_family_id": f"expansion_family_{number}",
        "plan_id": "pilot_expansion_250",
        "plan_version": "batch_01_v1",
        "generation_prompt_version": "manual_topic_pairs_v1",
        "language": "en",
        "task_type": task_type,
        "user_instruction": instruction,
        "contexts": contexts,
        "probes": probes,
        "intended_crossing": {"AA": "related", "AB": "unrelated", "BA": "unrelated", "BB": "related"},
        "generator_notes": {
            "crossing_rationale": {
                pair: f"Intended {'related' if pair in ('AA', 'BB') else 'unrelated'}; human rating pending."
                for pair in PAIRINGS
            },
            "possible_confounds": [
                "Synthetic template and topic wording may affect similarity and downstream responses.",
                "Answer difficulty and format comparability require human review.",
            ],
            "unresolved_issues": [
                "Verify probe answers and accepted variants, task gold answers, leakage, difficulty, and all four relevance ratings.",
                "Check measurement values and units for plausibility and verify the downstream model's input length.",
            ],
            "possible_source_family_dependency": "New synthetic source; conceptual overlap requires review.",
        },
    }


def main() -> None:
    for path in (OUTPUT, SCORES):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    if source["candidate_count"] != 64 or len(source["blocks"]) != 64:
        raise ValueError("Expected the audited 64-block draft")
    originals = copy.deepcopy(source["blocks"])
    instructions = {
        t: next(b["user_instruction"] for b in originals if b["task_type"] == t)
        for t in ("date_location_extraction", "target_quantity_extraction")
    }
    new = [
        new_block(
            265 + index,
            index,
            "date_location_extraction" if index < 6 else "target_quantity_extraction",
            instructions["date_location_extraction" if index < 6 else "target_quantity_extraction"],
        )
        for index in range(12)
    ]
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
        "collection_id": "pilot_expansion_76_draft_v1",
        "status": "incomplete_expansion_draft_not_250",
        "candidate_count": len(blocks),
        "target_count": 250,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "revised_original_ids": source["revised_block_ids"],
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
