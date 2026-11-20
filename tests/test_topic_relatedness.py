import unittest
from unittest.mock import patch

import numpy as np

from topic_relevance.scorer.topic_relatedness import TopicRelatednessScorer


class FakeSentenceTransformer:
    max_seq_length = 6

    def __init__(self):
        self.encoded = []
        self.output = None

    def tokenizer(self, text, *, add_special_tokens, truncation):
        return {"input_ids": list(range(len(text.split()) + 2))}

    def encode(self, texts, **kwargs):
        self.encoded.append((texts, kwargs))
        if self.output is not None:
            return self.output
        vectors = {
            "context": [1.0, 0.0],
            "related": [1.0, 0.0],
            "other": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts])


class TopicRelatednessTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeSentenceTransformer()
        with patch(
            "topic_relevance.scorer.topic_relatedness.SentenceTransformer",
            return_value=self.model,
        ):
            self.scorer = TopicRelatednessScorer(device="cpu")

    def test_batch_scores_are_ordered_and_repeated_texts_are_encoded_once(self):
        scores = self.scorer.score_pairs(
            [("context", "related"), ("context", "other"), ("related", "context")],
            batch_size=2,
        )
        self.assertEqual(scores, [1.0, 0.0, 1.0])
        self.assertEqual(self.model.encoded[0][0], ["context", "related", "other"])
        self.assertEqual(self.model.encoded[0][1]["batch_size"], 2)
        self.assertEqual(self.model.encoded[0][1]["prompt"], "")
        self.assertEqual(
            self.model.encoded[0][1]["processing_kwargs"],
            {"text": {"truncation": False}},
        )
        self.assertEqual(self.scorer.score("context", "other"), 0.0)

    def test_empty_batch_and_invalid_inputs(self):
        self.assertEqual(self.scorer.score_pairs([]), [])
        self.assertEqual(self.model.encoded, [])
        for pairs, error in (
            ([(" ", "other")], ValueError),
            ([("context", [])], TypeError),
            (["context"], ValueError),
            ([("one two three four five", "other")], ValueError),
        ):
            with self.subTest(pairs=pairs), self.assertRaises(error):
                self.scorer.score_pairs(pairs)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.scorer.score_pairs([], batch_size=size)
        self.assertEqual(self.model.encoded, [])

    def test_nonfinite_zero_or_incomplete_embeddings_fail_explicitly(self):
        for output in (
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[np.nan, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0]]),
        ):
            self.model.output = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.scorer.score("context", "other")

    def test_unknown_model_token_limit_fails_explicitly(self):
        self.model.max_seq_length = None
        with self.assertRaisesRegex(ValueError, "最大入力長"):
            self.scorer.score("context", "other")


if __name__ == "__main__":
    unittest.main()
