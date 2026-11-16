import unittest

from topic_relevance.scripts.run_ab_followup_v1 import (
    score_task_answer_content,
    score_task_response,
)


class TaskAnswerContentScoringTests(unittest.TestCase):
    def test_embedded_single_object_scores_content_without_passing_strict_json(self):
        response = 'Here is the result: {"value": 2}'
        gold = {"value": "2"}

        strict_json_valid, strict_matches = score_task_response(response, gold)
        status, content_matches = score_task_answer_content(response, gold)

        self.assertFalse(strict_json_valid)
        self.assertFalse(strict_matches["value"])
        self.assertEqual(status, "single_json_object")
        self.assertTrue(content_matches["value"])

    def test_multiple_objects_are_unscorable(self):
        status, matches = score_task_answer_content(
            '{"value": 2} Also: {"value": 3}', {"value": "2"}
        )

        self.assertEqual(status, "multiple_json_objects")
        self.assertIsNone(matches)

    def test_duplicate_keys_are_unscorable(self):
        status, matches = score_task_answer_content(
            '{"value": 2, "value": 3}', {"value": "2"}
        )

        self.assertEqual(status, "duplicate_json_keys")
        self.assertIsNone(matches)

    def test_json_array_is_not_a_task_answer_object(self):
        status, matches = score_task_answer_content(
            '[{"value": 2}]', {"value": "2"}
        )

        self.assertEqual(status, "non_object_json")
        self.assertIsNone(matches)


if __name__ == "__main__":
    unittest.main()
