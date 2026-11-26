import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from senior_project.llm import capture_response_attention, capture_response_attention_with_tokens
from senior_project.pipeline import collect_response_features, evaluate_defense
from senior_project.prompt import EncodedPrompt
from senior_project.rennervate import (
    ResponseAttentionDetector,
    load_response_detector,
    predict_response_tokens,
    reference_token_predictions,
    train_response_detector,
)


class CachedModel(nn.Module):
    def __init__(self, immediate_eos=False):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.calls = 0
        self.immediate_eos = immediate_eos
        self.consumed = []

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, use_cache, pad_token_id):
        response = [2] if self.immediate_eos else [3, 1, 2]
        return torch.cat((input_ids, torch.tensor([response], device=input_ids.device)), dim=1)

    def forward(self, input_ids, attention_mask, output_attentions, use_cache, past_key_values=None):
        self.calls += 1
        if output_attentions:
            self.consumed.append(int(input_ids.item()))
        logits = torch.zeros((1, input_ids.shape[1], 4))
        token = 2 if self.immediate_eos or self.calls >= 3 else 1
        logits[:, -1, token] = 1
        if output_attentions:
            length = attention_mask.shape[-1]
            values = torch.arange(length, dtype=torch.float32).reshape(1, 1, 1, length)
            attentions = (values.repeat(1, 2, 1, 1), values.repeat(1, 2, 1, 1) + 10)
        else:
            attentions = None
        return SimpleNamespace(logits=logits, attentions=attentions, past_key_values=self.calls)


class ResponseAttentionTests(unittest.TestCase):
    def test_cached_attention_matches_full_forward_for_generated_tokens(self):
        torch.manual_seed(7)
        config = Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            eos_token_id=None, pad_token_id=31,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config).eval()
        model.generation_config.eos_token_id = None
        prompt = EncodedPrompt(
            model_inputs={"input_ids": torch.tensor([[1, 4, 5, 6, 7]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)},
            context_token_indices=[1, 3],
            context_offsets=[(0, 1), (1, 2)],
            rendered_text="ab",
        )
        tokenizer = SimpleNamespace(eos_token_id=31)
        features = capture_response_attention(prompt, model, tokenizer, {"max_new_tokens": 5})
        with torch.inference_mode():
            sequences = model.generate(
                **prompt.model_inputs, max_new_tokens=5, do_sample=False,
                use_cache=True, pad_token_id=31,
            )
            outputs = model(
                input_ids=sequences, attention_mask=torch.ones_like(sequences),
                output_attentions=True, use_cache=False,
            )
        expected = torch.stack([
            layer[0, :, 5:9, :].index_select(-1, torch.tensor([1, 3]))
            for layer in outputs.attentions
        ]).permute(3, 0, 1, 2)
        torch.testing.assert_close(features.float(), expected, rtol=1e-3, atol=1e-4)

    def test_extracts_generated_query_attention_for_context_only(self):
        prompt = EncodedPrompt(
            model_inputs={"input_ids": torch.tensor([[0, 0, 0, 0]]), "attention_mask": torch.ones(1, 4)},
            context_token_indices=[1, 3],
            context_offsets=[(0, 1), (1, 2)],
            rendered_text="ab",
        )
        config = {"max_new_tokens": 8}
        model = CachedModel()
        features, tokens = capture_response_attention_with_tokens(prompt, model, SimpleNamespace(eos_token_id=2), config)
        self.assertEqual(tuple(features.shape), (2, 2, 2, 2))
        self.assertEqual(features[0, 0, 0].tolist(), [1.0, 1.0])
        self.assertEqual(features[1, 1, 1].tolist(), [13.0, 13.0])
        self.assertEqual(model.calls, 3)
        self.assertEqual(model.consumed, [3, 1])
        self.assertEqual(tokens, [3, 1, 2])
        with self.assertRaisesRegex(RuntimeError, "no response tokens"):
            capture_response_attention(prompt, CachedModel(True), SimpleNamespace(eos_token_id=2), config)
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            capture_response_attention(prompt, CachedModel(), SimpleNamespace(eos_token_id=2), {"max_new_tokens": 1})

    def test_collect_train_load_and_evaluate_separate_detector(self):
        torch.manual_seed(1)
        prompt = EncodedPrompt(
            model_inputs={"input_ids": torch.tensor([[97, 98, 99, 100]])},
            context_token_indices=[0, 1, 2, 3],
            context_offsets=[(0, 1), (1, 2), (2, 3), (3, 4)],
            rendered_text="abcd",
        )
        records = [
            {"id": "attack", "user_input": "task", "external_context": "abcd", "injected_instruction": "bc"},
            {"id": "clean", "user_input": "task", "external_context": "abcd"},
        ]
        config = {"model_id": "test", "max_input_tokens": 20, "max_new_tokens": 8}
        features = torch.rand(4, 2, 2, 3, dtype=torch.float16)

        class Tokenizer:
            def decode(self, tokens, skip_special_tokens, clean_up_tokenization_spaces):
                return "".join(chr(token) for token in tokens)

        with tempfile.TemporaryDirectory() as temp_dir:
            data = Path(temp_dir) / "features.npz"
            checkpoint = Path(temp_dir) / "detector.pt"
            with (
                patch("senior_project.pipeline.prepare_prompt", return_value=prompt),
                patch("senior_project.pipeline.capture_response_attention", return_value=features),
            ):
                collected = collect_response_features(records, Tokenizer(), None, config, data)
                self.assertEqual(collected, {"example_count": 2, "token_count": 8})
                with np.load(data, allow_pickle=False) as saved:
                    self.assertEqual(saved["labels_0"].tolist(), [0, 1, 1, 0])
                    self.assertEqual(saved["features_0"].shape, (4, 2, 2, 3))
                with patch("torch.cuda.is_available", return_value=False):
                    train_response_detector(data, checkpoint, config, epochs=1, seed=3)
                detector = load_response_detector(checkpoint, config, torch.device("cpu"))
                self.assertEqual(len(predict_response_tokens(detector, features, batch_size=2)), 4)
                with self.assertRaisesRegex(ValueError, "model ID"):
                    load_response_detector(checkpoint, {**config, "model_id": "other"}, torch.device("cpu"))
                with (
                    patch("senior_project.pipeline.model_device", return_value=torch.device("cpu")),
                    patch("senior_project.pipeline.predict_response_tokens", return_value=[0, 1, 0, 0]),
                    patch("senior_project.pipeline.generate_answer", return_value="answer"),
                ):
                    report = evaluate_defense(
                        records[:1], Tokenizer(), None, config, checkpoint,
                        detector_mode="rennervate-pooling",
                    )
                self.assertEqual(report["examples"][0]["sanitized_context"], "acd")
                self.assertFalse(report["examples"][0]["input_injected"])
                self.assertEqual(report["detector_condition"]["classifier"], "two-stage attentive statistics pooling")

    def test_detector_accepts_variable_response_lengths_and_rejects_wrong_dimensions(self):
        detector = ResponseAttentionDetector(2, 3, hidden_size=8).eval()
        with torch.inference_mode():
            self.assertEqual(tuple(detector(torch.rand(5, 2, 3, 1)).shape), (5, 2))
            self.assertEqual(tuple(detector(torch.rand(5, 2, 3, 32)).shape), (5, 2))
            with self.assertRaisesRegex(ValueError, "Expected attention features"):
                detector(torch.rand(5, 3, 2, 4))
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_response_tokens(detector, torch.full((1, 2, 3, 1), float("nan")))

    def test_reference_score_sign_and_short_window_edges(self):
        self.assertEqual(
            reference_token_predictions(torch.tensor([[2.0, -100.0], [-1.0, -100.0], [-1.0, -100.0]])),
            [1, 1, 1],
        )
        self.assertEqual(
            reference_token_predictions(torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])),
            [0, 0, 0, 1],
        )
        self.assertEqual(reference_token_predictions(torch.tensor([[1.0, 0.0], [-1.0, 0.0]])), [1, 1])

    def test_reference_mean_uses_double_precision_at_zero_boundary(self):
        for scores in ([1.0, 1e-8, -1.0], [1.0, 1e-8, -1.0, 0.0, 0.0]):
            logits = torch.tensor([[score, 0.0] for score in scores])
            self.assertEqual(reference_token_predictions(logits)[0], 0)
        self.assertEqual(reference_token_predictions(torch.tensor([[0.0, 1.0]])), [1])

    def test_collection_rejects_truncated_context(self):
        prompt = EncodedPrompt(
            model_inputs={"input_ids": torch.tensor([[1]])},
            context_token_indices=[0],
            context_offsets=[(0, 1)],
            rendered_text="a",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("senior_project.pipeline.prepare_prompt", return_value=prompt):
                with self.assertRaisesRegex(ValueError, "truncated external context"):
                    collect_response_features(
                        [{"user_input": "task", "external_context": "abc"}],
                        None, None, {}, Path(temp_dir) / "features.npz",
                    )


if __name__ == "__main__":
    unittest.main()
