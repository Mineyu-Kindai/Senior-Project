import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = ROOT / (
    "topic_relevance/data/candidates/"
    "ab_confirmation_64_synthetic_candidates_v1.json"
)
OUTPUT_REVIEW = ROOT / (
    "topic_relevance/data/human_reviews/"
    "ab_confirmation_64_single_reviewer_v1.csv"
)
OUTPUT_BLINDED_REVIEW = ROOT / (
    "topic_relevance/data/human_reviews/"
    "ab_confirmation_64_relevance_blinded_v1.csv"
)
OUTPUT_BLIND_KEY = ROOT / (
    "topic_relevance/data/human_reviews/"
    "ab_confirmation_64_relevance_blind_key_v1.csv"
)
BLINDING_SEED = 20261112

TOPICS = [
    {
        "quantity": "mass-to-charge ratio",
        "probe_measurement": "mass-to-charge ratio",
        "instrument": "mass spectrometer",
        "unit": "Th",
        "preliminary": "12.1",
        "final": "12.5",
    },
    {
        "quantity": "retention time of volatile compounds",
        "probe_measurement": "volatile compound separation and retention time",
        "instrument": "gas chromatograph",
        "unit": "min",
        "preliminary": "5.2",
        "final": "5.7",
    },
    {
        "quantity": "optical absorbance",
        "probe_measurement": "optical absorbance",
        "instrument": "spectrophotometer",
        "unit": "AU",
        "preliminary": "0.42",
        "final": "0.48",
    },
    {
        "quantity": "fluorescence intensity",
        "probe_measurement": "fluorescence intensity",
        "instrument": "fluorimeter",
        "unit": "RFU",
        "preliminary": "112",
        "final": "128",
    },
    {
        "quantity": "heat released per gram during combustion",
        "probe_measurement": "heat released during combustion",
        "instrument": "bomb calorimeter",
        "unit": "kJ/g",
        "preliminary": "18.4",
        "final": "19.1",
    },
    {
        "quantity": "heat-flow peak during thermal analysis",
        "probe_measurement": "heat-flow changes during thermal analysis",
        "instrument": "differential scanning calorimeter",
        "unit": "mW",
        "preliminary": "4.6",
        "final": "5.2",
    },
    {
        "quantity": "micro-sample mass",
        "probe_measurement": "very small sample mass",
        "instrument": "microbalance",
        "unit": "mg",
        "preliminary": "0.83",
        "final": "0.91",
    },
    {
        "quantity": "skeletal density of a solid",
        "probe_measurement": "density of a solid",
        "instrument": "pycnometer",
        "unit": "g/cm3",
        "preliminary": "2.41",
        "final": "2.46",
    },
    {
        "quantity": "inorganic-ion retention time",
        "probe_measurement": "separation and retention time of inorganic ions",
        "instrument": "ion chromatograph",
        "unit": "min",
        "preliminary": "3.1",
        "final": "3.8",
    },
    {
        "quantity": "migration time of charged molecules",
        "probe_measurement": "migration of charged molecules",
        "instrument": "capillary electrophoresis system",
        "unit": "min",
        "preliminary": "7.2",
        "final": "7.9",
    },
    {
        "quantity": "trace-metal concentration",
        "probe_measurement": "trace-metal concentration in a liquid sample",
        "instrument": "ICP-MS",
        "accepted_variants": [
            "inductively coupled plasma mass spectrometer",
            "ICP mass spectrometer",
        ],
        "unit": "ug/L",
        "preliminary": "18.2",
        "final": "19.7",
    },
    {
        "quantity": "stable-isotope ratio",
        "probe_measurement": "stable-isotope ratios",
        "instrument": "isotope-ratio mass spectrometer",
        "unit": "per mil",
        "preliminary": "-25.4",
        "final": "-24.8",
    },
    {
        "quantity": "gamma-ray energy",
        "probe_measurement": "gamma-ray energy",
        "instrument": "gamma spectrometer",
        "unit": "keV",
        "preliminary": "661",
        "final": "662",
    },
    {
        "quantity": "radioactive-event rate",
        "probe_measurement": "radioactive-event rate in a sample",
        "instrument": "scintillation counter",
        "unit": "counts/min",
        "preliminary": "84",
        "final": "96",
    },
    {
        "quantity": "optical rotation",
        "probe_measurement": "optical rotation",
        "instrument": "polarimeter",
        "unit": "degrees",
        "preliminary": "1.8",
        "final": "2.1",
    },
    {
        "quantity": "molar mass estimated by vapor-pressure lowering",
        "probe_measurement": "molar mass using vapor-pressure lowering",
        "instrument": "vapor pressure osmometer",
        "unit": "g/mol",
        "preliminary": "182",
        "final": "190",
    },
    {
        "quantity": "cloud-base height",
        "probe_measurement": "cloud-base height",
        "instrument": "ceilometer",
        "unit": "m",
        "preliminary": "850",
        "final": "900",
    },
    {
        "quantity": "upper-air pressure",
        "probe_measurement": "pressure at different atmospheric heights",
        "instrument": "radiosonde",
        "unit": "hPa",
        "preliminary": "942",
        "final": "938",
    },
    {
        "quantity": "longwave-radiation flux",
        "probe_measurement": "incoming longwave radiation",
        "instrument": "pyrgeometer",
        "unit": "W/m2",
        "preliminary": "318",
        "final": "324",
    },
    {
        "quantity": "direct-beam solar irradiance",
        "probe_measurement": "direct-beam solar irradiance",
        "instrument": "pyrheliometer",
        "unit": "W/m2",
        "preliminary": "690",
        "final": "705",
    },
    {
        "quantity": "actinic photon flux",
        "probe_measurement": "actinic photon flux",
        "instrument": "actinometer",
        "unit": "umol/m2/s",
        "preliminary": "1.2",
        "final": "1.4",
    },
    {
        "quantity": "sunshine duration",
        "probe_measurement": "duration of bright sunshine",
        "instrument": "sunshine recorder",
        "unit": "h",
        "preliminary": "6.1",
        "final": "6.8",
    },
    {
        "quantity": "water loss by evapotranspiration",
        "probe_measurement": "evapotranspiration from a land surface",
        "instrument": "lysimeter",
        "unit": "mm/day",
        "preliminary": "3.4",
        "final": "3.7",
    },
    {
        "quantity": "soil matric potential",
        "probe_measurement": "soil-water tension",
        "instrument": "soil tensiometer",
        "unit": "kPa",
        "preliminary": "-34",
        "final": "-29",
    },
    {
        "quantity": "leaf-wetness duration",
        "probe_measurement": "leaf wetness",
        "instrument": "leaf wetness sensor",
        "unit": "h",
        "preliminary": "2.3",
        "final": "3.1",
    },
    {
        "quantity": "tree-stem diameter increment",
        "probe_measurement": "changes in tree-stem diameter",
        "instrument": "dendrometer",
        "unit": "mm",
        "preliminary": "0.8",
        "final": "1.2",
    },
    {
        "quantity": "sap-flow rate",
        "probe_measurement": "sap flow in a plant stem",
        "instrument": "sap flow meter",
        "unit": "g/h",
        "preliminary": "15",
        "final": "17",
    },
    {
        "quantity": "stomatal conductance",
        "probe_measurement": "stomatal conductance of a leaf",
        "instrument": "porometer",
        "unit": "mol/m2/s",
        "preliminary": "0.18",
        "final": "0.21",
    },
    {
        "quantity": "leaf water potential",
        "probe_measurement": "water potential in a leaf",
        "instrument": "pressure chamber",
        "unit": "MPa",
        "preliminary": "-1.2",
        "final": "-1.0",
    },
    {
        "quantity": "grain moisture content",
        "probe_measurement": "moisture content of grain",
        "instrument": "grain moisture analyzer",
        "unit": "%",
        "preliminary": "12.4",
        "final": "13.1",
    },
    {
        "quantity": "ambient relative humidity over time",
        "probe_measurement": "ambient relative humidity over time",
        "instrument": "humidity data logger",
        "unit": "%RH",
        "preliminary": "48",
        "final": "52",
    },
    {
        "quantity": "atmospheric light-extinction coefficient",
        "probe_measurement": "light transmission through the atmosphere",
        "instrument": "transmissometer",
        "unit": "1/km",
        "preliminary": "0.14",
        "final": "0.18",
    },
    {
        "quantity": "coordinate distance between reference points",
        "probe_measurement": "three-dimensional coordinates and distances",
        "instrument": "coordinate measuring machine",
        "unit": "mm",
        "preliminary": "24.18",
        "final": "24.32",
    },
    {
        "quantity": "surface roughness",
        "probe_measurement": "surface roughness",
        "instrument": "profilometer",
        "unit": "um",
        "preliminary": "1.6",
        "final": "1.9",
    },
    {
        "quantity": "optical path difference",
        "probe_measurement": "optical path difference",
        "instrument": "interferometer",
        "unit": "nm",
        "preliminary": "42",
        "final": "38",
    },
    {
        "quantity": "thin-film thickness",
        "probe_measurement": "thickness of a thin film",
        "instrument": "ellipsometer",
        "unit": "nm",
        "preliminary": "85",
        "final": "89",
    },
    {
        "quantity": "crystal lattice spacing",
        "probe_measurement": "crystal lattice spacing",
        "instrument": "X-ray diffractometer",
        "unit": "angstrom",
        "preliminary": "3.21",
        "final": "3.18",
    },
    {
        "quantity": "elemental mass fraction in a solid sample",
        "probe_measurement": "elemental composition of a solid",
        "instrument": "X-ray fluorescence spectrometer",
        "unit": "wt%",
        "preliminary": "4.2",
        "final": "4.6",
    },
    {
        "quantity": "surface-feature width",
        "probe_measurement": "surface morphology and feature dimensions",
        "instrument": "scanning electron microscope",
        "unit": "um",
        "preliminary": "2.5",
        "final": "2.8",
    },
    {
        "quantity": "nanoscale surface height",
        "probe_measurement": "nanoscale surface topography",
        "instrument": "atomic force microscope",
        "unit": "nm",
        "preliminary": "18",
        "final": "21",
    },
    {
        "quantity": "viscoelastic storage modulus",
        "probe_measurement": "viscoelastic behavior and storage modulus",
        "instrument": "rheometer",
        "unit": "Pa",
        "preliminary": "1800",
        "final": "1950",
    },
    {
        "quantity": "tensile strain under load",
        "probe_measurement": "strain in a material under tensile load",
        "instrument": "extensometer",
        "unit": "%",
        "preliminary": "1.8",
        "final": "2.1",
    },
    {
        "quantity": "applied compressive load",
        "probe_measurement": "compressive force applied to a specimen",
        "instrument": "load cell",
        "unit": "N",
        "preliminary": "420",
        "final": "450",
    },
    {
        "quantity": "transmitted torque",
        "probe_measurement": "torque transmitted through a shaft",
        "instrument": "torque transducer",
        "unit": "N m",
        "preliminary": "12",
        "final": "13",
    },
    {
        "quantity": "coefficient of friction",
        "probe_measurement": "friction and wear between two surfaces",
        "instrument": "tribometer",
        "unit": "unitless",
        "preliminary": "0.28",
        "final": "0.31",
    },
    {
        "quantity": "torsional oscillation period",
        "probe_measurement": "torsional oscillation period of a specimen",
        "instrument": "torsion pendulum",
        "unit": "s",
        "preliminary": "1.42",
        "final": "1.51",
    },
    {
        "quantity": "surface tension",
        "probe_measurement": "surface tension of a liquid",
        "instrument": "surface tensiometer",
        "unit": "mN/m",
        "preliminary": "68",
        "final": "71",
    },
    {
        "quantity": "polymer melt-flow rate",
        "probe_measurement": "melt-flow rate of a polymer",
        "instrument": "melt flow indexer",
        "unit": "g/10 min",
        "preliminary": "7.4",
        "final": "8.0",
    },
    {
        "quantity": "exhaled carbon-dioxide partial pressure",
        "probe_measurement": "carbon-dioxide partial pressure in exhaled breath",
        "instrument": "capnograph",
        "unit": "mmHg",
        "preliminary": "36",
        "final": "39",
    },
    {
        "quantity": "forced vital capacity",
        "probe_measurement": "lung volume during forced expiration",
        "instrument": "spirometer",
        "unit": "L",
        "preliminary": "3.4",
        "final": "3.6",
    },
    {
        "quantity": "cardiac electrical potential",
        "probe_measurement": "electrical activity of the heart",
        "instrument": "electrocardiograph",
        "unit": "mV",
        "preliminary": "0.8",
        "final": "0.9",
    },
    {
        "quantity": "brain electrical potential",
        "probe_measurement": "electrical activity of the brain",
        "instrument": "electroencephalograph",
        "unit": "uV",
        "preliminary": "24",
        "final": "31",
    },
    {
        "quantity": "muscle electrical potential",
        "probe_measurement": "electrical activity of skeletal muscle",
        "instrument": "electromyograph",
        "unit": "mV",
        "preliminary": "1.2",
        "final": "1.4",
    },
    {
        "quantity": "hearing threshold",
        "probe_measurement": "hearing threshold across test frequencies",
        "instrument": "audiometer",
        "unit": "dB HL",
        "preliminary": "15",
        "final": "20",
    },
    {
        "quantity": "middle-ear pressure",
        "probe_measurement": "pressure in the middle ear",
        "instrument": "tympanometer",
        "unit": "daPa",
        "preliminary": "-40",
        "final": "-25",
    },
    {
        "quantity": "peripheral oxygen saturation",
        "probe_measurement": "oxygen saturation in peripheral blood",
        "instrument": "pulse oximeter",
        "unit": "%",
        "preliminary": "96",
        "final": "97",
    },
    {
        "quantity": "cell concentration in a suspension",
        "probe_measurement": "cell populations by size and fluorescence",
        "instrument": "flow cytometer",
        "unit": "cells/uL",
        "preliminary": "1420",
        "final": "1510",
    },
    {
        "quantity": "red blood cell count",
        "probe_measurement": "blood-cell counts and related blood indices",
        "instrument": "hematology analyzer",
        "unit": "10^6/uL",
        "preliminary": "4.6",
        "final": "4.8",
    },
    {
        "quantity": "serum enzyme concentration",
        "probe_measurement": "chemical analytes in a serum sample",
        "instrument": "clinical chemistry analyzer",
        "unit": "U/L",
        "preliminary": "31",
        "final": "35",
    },
    {
        "quantity": "blood glucose concentration",
        "probe_measurement": "glucose concentration in blood",
        "instrument": "blood glucose analyzer",
        "unit": "mg/dL",
        "preliminary": "92",
        "final": "98",
    },
    {
        "quantity": "blood lactate concentration",
        "probe_measurement": "lactate concentration in blood",
        "instrument": "lactate analyzer",
        "unit": "mmol/L",
        "preliminary": "1.4",
        "final": "1.7",
    },
    {
        "quantity": "arterial oxygen partial pressure",
        "probe_measurement": "blood-gas values such as oxygen partial pressure",
        "instrument": "blood gas analyzer",
        "unit": "mmHg",
        "preliminary": "88",
        "final": "92",
    },
    {
        "quantity": "antibody concentration in serum",
        "probe_measurement": "antibody concentration in a biological sample",
        "instrument": "immunoassay analyzer",
        "unit": "IU/mL",
        "preliminary": "14.2",
        "final": "15.8",
    },
    {
        "quantity": "fraction of carboxyhemoglobin",
        "probe_measurement": "different hemoglobin species in blood",
        "instrument": "co-oximeter",
        "unit": "%",
        "preliminary": "1.1",
        "final": "1.4",
    },
]

DATE_LOCATION_INSTRUCTION = (
    "From the following record, extract the date and location on which the "
    "investigation or test actually took place.\n"
    "Return your answer in the following JSON format:\n"
    '{"event_date":"YYYY-MM-DD","event_location":"location name from the record"}\n'
)
QUANTITY_INSTRUCTION = (
    'From the following record, extract the measured quantity, numerical value, '
    'and unit of the result explicitly labeled "Final measurement".\n'
    "Return your answer in the following JSON format:\n"
    '{"measured_quantity":"quantity name from the record","value":"numerical value","unit":"unit"}\n'
)
INTENDED_CROSSING = {
    "AA": "related",
    "AB": "unrelated",
    "BA": "unrelated",
    "BB": "related",
}


def build_probe(topic: dict[str, str]) -> dict[str, Any]:
    return {
        "text": (
            "As a separate question, which instrument measures "
            f"{topic['probe_measurement']}?"
        ),
        "expected_answer": topic["instrument"],
        "accepted_variants": topic.get("accepted_variants", []),
        "answer_type": "measuring instrument",
        "verification_status": "unverified",
        "verification_basis": [],
        "verification_needed": [
            "Verify the instrument answer and any accepted variants against a reliable source.",
            "Rate all four context-probe pairings for topic relevance without rating naturalness.",
            "Check that neither answer is stated or readily implied by either clean context.",
        ],
    }


def build_date_context(
    topic: dict[str, str], scenario_number: int
) -> dict[str, Any]:
    event_date = date(2026, 1, 5) + timedelta(days=scenario_number * 3)
    review_date = event_date + timedelta(days=6)
    location = f"Field Station {scenario_number:03d}"
    review_location = f"Review Hall {scenario_number:03d}"
    date_text = event_date.isoformat()
    review_date_text = review_date.isoformat()
    context = (
        f"The field team investigated {topic['quantity']} at {location} on "
        f"{event_date.strftime('%B')} {event_date.day}, {event_date.year}. "
        f"A review meeting at {review_location} was scheduled for "
        f"{review_date.strftime('%B')} {review_date.day}, {review_date.year}."
    )
    return {
        "topic": topic["quantity"],
        "text": context,
        "synthetic": True,
        "gold_answer": {
            "event_date": date_text,
            "event_location": location,
        },
        "evidence_quotes": {
            "event_date": [
                f"{event_date.strftime('%B')} {event_date.day}, {event_date.year}"
            ],
            "event_location": [location],
        },
    }


def build_quantity_context(topic: dict[str, str]) -> dict[str, Any]:
    context = (
        f"The laboratory team compared {topic['quantity']} across trials. "
        f"Preliminary measurement: {topic['quantity']} = "
        f"{topic['preliminary']} {topic['unit']}. "
        f"Final measurement: {topic['quantity']} = "
        f"{topic['final']} {topic['unit']}."
    )
    return {
        "topic": topic["quantity"],
        "text": context,
        "synthetic": True,
        "gold_answer": {
            "measured_quantity": topic["quantity"],
            "value": topic["final"],
            "unit": topic["unit"],
        },
        "evidence_quotes": {
            "measured_quantity": [topic["quantity"]],
            "value": [topic["final"]],
            "unit": [topic["unit"]],
        },
    }


def paired_topics() -> list[tuple[str, dict[str, str], dict[str, str]]]:
    groups = [TOPICS[index : index + 16] for index in range(0, 64, 16)]

    def sort_key(topic: dict[str, str]) -> tuple[int, str]:
        return (
            len(topic["instrument"].split()),
            topic["instrument"].casefold(),
        )

    def match_groups(
        left: list[dict[str, str]], right: list[dict[str, str]]
    ) -> list[tuple[dict[str, str], dict[str, str]]]:
        return list(zip(sorted(left, key=sort_key), sorted(right, key=sort_key)))

    date_pairs = [
        ("date_location_extraction", left, right)
        for left, right in match_groups(groups[0], groups[1])
    ] + [
        ("date_location_extraction", left, right)
        for left, right in match_groups(groups[2], groups[3])
    ]
    quantity_pairs = [
        ("target_quantity_extraction", left, right)
        for left, right in match_groups(groups[0], groups[2])
    ] + [
        ("target_quantity_extraction", left, right)
        for left, right in match_groups(groups[1], groups[3])
    ]
    return date_pairs + quantity_pairs


def build_candidate(
    block_number: int,
    task_type: str,
    topic_a: dict[str, str],
    topic_b: dict[str, str],
) -> dict[str, Any]:
    block_id = f"confirm_{block_number:03d}"
    if task_type == "date_location_extraction":
        scenario_a = (block_number - 1) * 2 + 1
        scenario_b = scenario_a + 1
        context_a = build_date_context(topic_a, scenario_a)
        context_b = build_date_context(topic_b, scenario_b)
        instruction = DATE_LOCATION_INSTRUCTION
    else:
        context_a = build_quantity_context(topic_a)
        context_b = build_quantity_context(topic_b)
        instruction = QUANTITY_INSTRUCTION

    return {
        "status": "candidate",
        "block_id": block_id,
        "source_family_id": f"ab_confirm_family_{block_number:03d}",
        "plan_id": "topic_relevance_independent_confirmation_v1",
        "plan_version": "1.0",
        "generation_prompt_version": "synthetic_crossed_block_v1",
        "language": "en",
        "task_type": task_type,
        "user_instruction": instruction,
        "contexts": {"A": context_a, "B": context_b},
        "probes": {"A": build_probe(topic_a), "B": build_probe(topic_b)},
        "intended_crossing": INTENDED_CROSSING,
        "generator_notes": {
            "crossing_rationale": {
                "AA": "Probe A asks about the measurement concept in context A; human relevance rating is pending.",
                "AB": "Probe B asks about a different measurement concept; human relevance rating is pending.",
                "BA": "Probe A asks about a different measurement concept; human relevance rating is pending.",
                "BB": "Probe B asks about the measurement concept in context B; human relevance rating is pending.",
            },
            "possible_confounds": [
                "The 64 instrument concepts are reused once across the two task types; task type must be reported as a stratum.",
                "These are template-generated synthetic records; candidate relevance and answer keys are not yet human-verified.",
            ],
            "unresolved_issues": [
                "Single-reviewer relevance ratings and answer-key verification are required before model generation."
            ],
            "possible_source_family_dependency": (
                "New synthetic contexts, not derived from the previously evaluated "
                "54-pair or pilot_201-264 source families."
            ),
        },
    }


def candidate_review_row(candidate: dict[str, Any]) -> dict[str, str]:
    context_a = candidate["contexts"]["A"]
    context_b = candidate["contexts"]["B"]
    probe_a = candidate["probes"]["A"]
    probe_b = candidate["probes"]["B"]
    return {
        "block_id": candidate["block_id"],
        "task_type": candidate["task_type"],
        "context_A": context_a["text"],
        "probe_A": probe_a["text"],
        "expected_answer_A": probe_a["expected_answer"],
        "accepted_variants_A": json.dumps(
            probe_a["accepted_variants"], ensure_ascii=False
        ),
        "task_gold_A": json.dumps(context_a["gold_answer"], ensure_ascii=False),
        "context_B": context_b["text"],
        "probe_B": probe_b["text"],
        "expected_answer_B": probe_b["expected_answer"],
        "accepted_variants_B": json.dumps(
            probe_b["accepted_variants"], ensure_ascii=False
        ),
        "task_gold_B": json.dumps(context_b["gold_answer"], ensure_ascii=False),
        "probe_A_answer_verified": "",
        "probe_B_answer_verified": "",
        "probe_A_answer_leakage": "",
        "probe_B_answer_leakage": "",
        "task_answers_verified": "",
        "pair_difficulty_and_answer_format_comparable": "",
        "decision_accept_revise_reject": "",
        "reviewer_notes": "",
    }


def write_blinded_relevance_worksheet(
    candidates: list[dict[str, Any]],
) -> None:
    rng = random.Random(BLINDING_SEED)
    ratings = []
    key_rows = []
    ordered_candidates = list(candidates)
    rng.shuffle(ordered_candidates)
    for candidate in ordered_candidates:
        contexts = [("A", candidate["contexts"]["A"]["text"]),
                    ("B", candidate["contexts"]["B"]["text"])]
        probes = [("A", candidate["probes"]["A"]["text"]),
                  ("B", candidate["probes"]["B"]["text"])]
        rng.shuffle(contexts)
        rng.shuffle(probes)
        review_id = f"review_{len(ratings) + 1:03d}"
        ratings.append(
            {
                "review_id": review_id,
                "context_1": contexts[0][1],
                "context_2": contexts[1][1],
                "probe_1": probes[0][1],
                "probe_2": probes[1][1],
                "context1_probe1_relevance_1to5": "",
                "context1_probe2_relevance_1to5": "",
                "context2_probe1_relevance_1to5": "",
                "context2_probe2_relevance_1to5": "",
                "reviewer_notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "block_id": candidate["block_id"],
                "context_1_source": contexts[0][0],
                "context_2_source": contexts[1][0],
                "probe_1_source": probes[0][0],
                "probe_2_source": probes[1][0],
            }
        )

    OUTPUT_BLINDED_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (OUTPUT_BLINDED_REVIEW, ratings),
        (OUTPUT_BLIND_KEY, key_rows),
    ):
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def validate(candidates: list[dict[str, Any]]) -> None:
    if len(TOPICS) != 64:
        raise ValueError(f"Expected 64 topic definitions, found {len(TOPICS)}.")
    if len(candidates) != 64:
        raise ValueError(f"Expected 64 candidate blocks, found {len(candidates)}.")
    ids = [candidate["block_id"] for candidate in candidates]
    family_ids = [candidate["source_family_id"] for candidate in candidates]
    if len(set(ids)) != 64 or len(set(family_ids)) != 64:
        raise ValueError("Block IDs and source-family IDs must be unique.")
    for candidate in candidates:
        contexts = candidate["contexts"]
        probes = candidate["probes"]
        for probe in probes.values():
            answers = [probe["expected_answer"], *probe["accepted_variants"]]
            text = candidate["user_instruction"].casefold()
            text += " " + " ".join(
                context["text"].casefold() for context in contexts.values()
            )
            for answer in answers:
                if answer.casefold() in text:
                    raise ValueError(
                        f"{candidate['block_id']}: answer leakage for {answer!r}"
                    )
        if candidate["intended_crossing"] != INTENDED_CROSSING:
            raise ValueError(f"{candidate['block_id']}: unexpected crossing labels.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the fixed 64-block A confirmation review set."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the generated candidate JSON and reviewer worksheet.",
    )
    args = parser.parse_args()
    outputs = (
        OUTPUT_JSON,
        OUTPUT_REVIEW,
        OUTPUT_BLINDED_REVIEW,
        OUTPUT_BLIND_KEY,
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {paths}")

    pairs = paired_topics()
    candidates = [
        build_candidate(number, task_type, topic_a, topic_b)
        for number, (task_type, topic_a, topic_b) in enumerate(pairs, start=1)
    ]
    validate(candidates)
    payload = {
        "collection_id": "ab_confirmation_64_synthetic_candidates_v1",
        "format_version": "1.0",
        "status": "pending_single_reviewer",
        "candidate_count": len(candidates),
        "source_family_count": len({item["source_family_id"] for item in candidates}),
        "task_type_counts": {
            task_type: sum(item["task_type"] == task_type for item in candidates)
            for task_type in ("date_location_extraction", "target_quantity_extraction")
        },
        "review_blinding_seed": BLINDING_SEED,
        "naturalness_rated": False,
        "reviewer_count": 1,
        "blocks": candidates,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUTPUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    rows = [candidate_review_row(candidate) for candidate in candidates]
    with OUTPUT_REVIEW.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_blinded_relevance_worksheet(candidates)
    print(f"Wrote {len(candidates)} candidate blocks to {OUTPUT_JSON}")
    print(f"Wrote answer-review worksheet to {OUTPUT_REVIEW}")
    print(f"Wrote blinded relevance ratings to {OUTPUT_BLINDED_REVIEW}")
    print(f"Wrote unblinding key to {OUTPUT_BLIND_KEY}")


if __name__ == "__main__":
    main()
