import unittest

from topic_relevance.scripts.summarize_threshold_masks_v1 import summarize, summarize_record


class ThresholdMaskSummaryTests(unittest.TestCase):
    def test_masking_counts_merged_span_and_excludes_boundary_from_clean_rate(self):
        record = {
            "input_kind": "injected",
            "id": "example",
            "input_sha256": "hash",
            "external_context": "abc def ghi",
            "regions": [{"name": "clean_context", "start": 0, "end": 7}],
            "tokens": [
                {"start": 0, "end": 3, "score": 0.9, "evaluation_region": "clean_context"},
                {"start": 4, "end": 7, "score": 0.8, "evaluation_region": "clean_context"},
                {"start": 7, "end": 9, "score": 0.9, "evaluation_region": "boundary"},
                {"start": 9, "end": 10, "score": 0.1, "evaluation_region": "probe"},
                {"start": 10, "end": 11, "score": 0.1, "evaluation_region": "fake_completion_wrapper"},
            ],
        }
        result = summarize_record(record, 0.8)
        self.assertEqual(result["region_token_counts"]["clean_context"], {"flagged": 2, "total": 2})
        self.assertEqual(result["region_token_counts"]["boundary"], {"flagged": 1, "total": 1})
        self.assertEqual(result["clean_characters_masked"], 7)
        self.assertEqual(result["masked_spans"], [(0, 9)])

    def test_thresholds_use_same_scores_and_regions(self):
        injected = {
            "input_kind": "injected", "id": "a", "input_sha256": "a",
            "external_context": "abcd",
            "regions": [{"name": "clean_context", "start": 0, "end": 1}],
            "tokens": [
                {"start": 0, "end": 1, "score": 0.5, "evaluation_region": "clean_context"},
                {"start": 1, "end": 2, "score": 0.6, "evaluation_region": "boundary"},
                {"start": 2, "end": 3, "score": 0.7, "evaluation_region": "fake_completion_wrapper"},
                {"start": 3, "end": 4, "score": 0.8, "evaluation_region": "probe"},
            ],
        }
        clean = {
            "input_kind": "clean_only", "input_sha256": "b", "external_context": "x",
            "regions": [{"name": "clean_context", "start": 0, "end": 1}],
            "tokens": [{"start": 0, "end": 1, "score": 0.5, "evaluation_region": "clean_context"}],
        }
        results = summarize([injected, clean])
        self.assertEqual(results[0]["groups"]["injected"]["regions"]["clean_context"]["flagged"], 1)
        self.assertEqual(results[1]["groups"]["injected"]["regions"]["clean_context"]["flagged"], 0)
        self.assertEqual(results[1]["groups"]["injected"]["regions"]["boundary"]["flagged"], 1)
        self.assertEqual(results[1]["groups"]["clean_only"]["clean_characters_masked"], 0)


if __name__ == "__main__":
    unittest.main()
