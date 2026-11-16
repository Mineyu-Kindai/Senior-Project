import unittest

from topic_relevance.scripts.capture_threshold_scores_v1 import (
    audit_baseline,
    classify_region,
    unique_clean_inputs,
)


class ThresholdScoreCaptureTests(unittest.TestCase):
    def test_baseline_requires_matching_tokens_spans_and_sanitized_text(self):
        source = {
            "external_context": "abc def",
            "detected_token_count": 1,
            "detected_spans": [{
                "start": 0, "end": 3,
                "token_scores": [{"token_index": 0, "start": 0, "end": 3, "score": 0.5}],
            }],
            "sanitized_external_context": "[REDACTED UNTRUSTED CONTENT] def",
        }
        tokens = [
            {"context_token_index": 0, "start": 0, "end": 3, "score": 0.5},
            {"context_token_index": 1, "start": 4, "end": 7, "score": 0.2},
        ]
        self.assertTrue(audit_baseline(source, tokens)["passed"])
        tokens[0]["score"] = 0.49
        audit = audit_baseline(source, tokens)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["detected_tokens"])
        self.assertFalse(audit["checks"]["sanitized_context"])

    def test_boundary_token_is_not_assigned_to_a_single_region(self):
        regions = [
            {"name": "clean_context", "start": 0, "end": 4},
            {"name": "probe", "start": 4, "end": 8},
        ]
        self.assertEqual(classify_region(0, 4, regions), "clean_context")
        self.assertEqual(classify_region(3, 5, regions), "boundary")

    def test_clean_deduplication_includes_instruction(self):
        examples = [
            {"id": "a", "block_id": "x", "user_input": "one", "clean_context": "body"},
            {"id": "b", "block_id": "x", "user_input": "one", "clean_context": "body"},
            {"id": "c", "block_id": "y", "user_input": "two", "clean_context": "body"},
        ]
        result = unique_clean_inputs(examples)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["source_ids"], ["a", "b"])
        self.assertEqual(result[1]["source_ids"], ["c"])


if __name__ == "__main__":
    unittest.main()
