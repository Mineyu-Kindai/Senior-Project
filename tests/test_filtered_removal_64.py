import json
import unittest

from topic_relevance.scripts.run_filtered_removal_64_v1 import make_rows, select_cases
from topic_relevance.scripts.run_pilot_expansion_250_token_removal_v1 import (
    CANDIDATE_FILE,
    input_hash,
    read_json,
)


class FilteredRemoval64Tests(unittest.TestCase):
    def test_remaining_blocks_are_disjoint_and_complete(self):
        candidates = read_json(CANDIDATE_FILE)
        development = select_cases(candidates, remaining=False)
        remaining = select_cases(candidates, remaining=True)
        self.assertEqual(len(development), 256)
        self.assertEqual(len(remaining), 744)
        self.assertEqual(len({case["block_id"] for case in remaining}), 186)
        self.assertFalse(
            {case["block_id"] for case in development}
            & {case["block_id"] for case in remaining}
        )

    def test_paired_rows_keep_detection_and_removal_distinct(self):
        case = {
            "case_id": "pilot_201__context_A_probe_A",
            "block_id": "pilot_201",
            "source_family_id": "family_001",
            "context_id": "A",
            "probe_id": "A",
            "intended_relation": "related",
            "task_type": "date_location_extraction",
            "user_instruction": "Task",
            "clean_context": "clean",
            "probe_text": "probe",
            "expected_answer": "answer",
            "accepted_variants": ["alternative"],
            "gold_task_answer": {"event_date": "2025-01-01"},
            "external_context": "clean wrapper probe",
        }
        scores = {
            case["case_id"]: {
                "sanitized_context": "clean probe",
                "clean_characters_removed": 0,
                "clean_characters_total": 5,
                "regions": {
                    "probe": {"flagged": 2, "removed": 1, "total": 4},
                    "wrapper": {"flagged": 3, "removed": 3, "total": 3},
                },
            }
        }
        result = {
            "response": "response",
            "input_token_count": 10,
            "generated_token_count": 4,
            "max_limit_reached": False,
            "eos_terminated": True,
        }
        old = {input_hash("Task", case["external_context"]): result}
        new = {input_hash("Task", "clean probe"): result}
        baseline, defended = make_rows([case], scores, new, old)
        self.assertEqual(baseline["response_source"], "previous_250_run")
        self.assertNotIn("probe_token_detection_rate", baseline)
        self.assertEqual(defended["response_source"], "new_generation")
        self.assertEqual(defended["probe_token_detection_rate"], 0.5)
        self.assertEqual(defended["probe_token_removal_rate"], 0.25)
        self.assertEqual(defended["injection_token_detection_rate"], 5 / 7)
        self.assertEqual(defended["injection_token_removal_rate"], 4 / 7)
        self.assertEqual(json.loads(defended["accepted_variants"]), ["alternative"])
        self.assertEqual(json.loads(defended["gold_task_answer"]), case["gold_task_answer"])


if __name__ == "__main__":
    unittest.main()
