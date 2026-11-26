import json
import unittest

from topic_relevance.scripts.run_response_attention_250_v1 import (
    PREVIOUS_LOG,
    build_cases,
    load_previous,
)


class ResponseAttention250Tests(unittest.TestCase):
    def test_expansion_and_safe_reuse(self):
        cases = build_cases()
        self.assertEqual(len(cases), 1000)
        self.assertEqual(len({case["block_id"] for case in cases}), 250)
        with PREVIOUS_LOG.open(encoding="utf-8") as file:
            header = json.loads(next(file))
        reusable = load_previous(cases, header)
        self.assertEqual(len(reusable), 253)
        for suffix in ("context_A_probe_B", "context_B_probe_A", "context_B_probe_B"):
            self.assertNotIn(f"pilot_257__{suffix}", reusable)
        self.assertIn("pilot_257__context_A_probe_A", reusable)


if __name__ == "__main__":
    unittest.main()
