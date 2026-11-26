from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RESPONSE_LIMIT = 32
TOKENS_PER_EXAMPLE = 20


class AttentiveStatistics(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Conv1d(channels * 3, channels * 2, 1)
        self.norm = nn.BatchNorm1d(channels * 2)
        self.score = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-12)).sqrt()
        context = torch.cat((x, mean.expand_as(x), std.expand_as(x)), dim=1)
        weights = self.score(torch.tanh(self.norm(F.relu(self.project(context)))))
        weights = weights.softmax(dim=-1)
        mean = (weights * x).sum(dim=-1)
        variance = (weights * (x - mean.unsqueeze(-1)).square()).sum(dim=-1)
        return torch.cat((mean, variance.clamp_min(1e-12).sqrt()), dim=-1)


class ResidualStack(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, blocks: int) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_size) for _ in range(blocks))
        self.layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(blocks))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(self.input_norm(x))
        for norm, layer in zip(self.norms, self.layers, strict=True):
            x = x + layer(F.relu(norm(x)))
        return self.dropout(x)


class ResponseAttentionDetector(nn.Module):
    def __init__(self, layers: int, heads: int, hidden_size: int = 512) -> None:
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.response_pool = AttentiveStatistics(heads)
        self.head_residual = ResidualStack(heads * 2, heads * 2, 2)
        self.head_pool = AttentiveStatistics(layers)
        self.layer_residual = ResidualStack(layers * 2, hidden_size, 2)
        self.class_weights = nn.Parameter(torch.empty(2, hidden_size))
        nn.init.xavier_uniform_(self.class_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1:3] != (self.layers, self.heads):
            raise ValueError("Expected attention features [tokens, layers, heads, response]")
        if features.shape[0] == 0 or features.shape[-1] == 0:
            raise ValueError("Attention features must contain tokens and response steps")
        tokens, layers, heads, steps = features.shape
        x = features.reshape(tokens * layers, heads, steps)
        x = self.head_residual(self.response_pool(x))
        x = x.reshape(tokens, layers, heads * 2)
        x = self.layer_residual(self.head_pool(x))
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.class_weights, dim=-1))


def _validate_model_config(metadata: Any, model_config: dict[str, Any], source: str) -> None:
    if str(metadata["model_id"]) != model_config["model_id"]:
        raise ValueError(f"{source} model ID does not match the configured model")
    for key in ("max_input_tokens", "max_new_tokens"):
        if int(metadata[key]) != int(model_config[key]):
            raise ValueError(f"{source} {key} does not match the configured model")


def train_response_detector(
    feature_path: Path,
    output_path: Path,
    model_config: dict[str, Any],
    epochs: int = 40,
    seed: int = 42,
) -> dict[str, float | int]:
    if epochs < 1:
        raise ValueError("epochs must be greater than zero")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    with np.load(feature_path, allow_pickle=False) as data:
        if str(data["format"]) != "response_attention_v1":
            raise ValueError("Expected response-attention features")
        _validate_model_config(data, model_config, "Feature")
        sample_ids = data["sample_ids"].astype(str).tolist()
        if not sample_ids:
            raise ValueError("No training samples")
        features = [data[f"features_{i}"] for i in range(len(sample_ids))]
        labels = [data[f"labels_{i}"] for i in range(len(sample_ids))]

    if features[0].ndim != 4:
        raise ValueError("Expected attention features [tokens, layers, heads, response]")
    layers, heads = features[0].shape[1:3]
    counts = np.zeros(2, dtype=np.int64)
    for x, y in zip(features, labels, strict=True):
        if (
            x.ndim != 4 or x.shape[1:3] != (layers, heads)
            or not 1 <= x.shape[-1] <= RESPONSE_LIMIT
            or y.ndim != 1 or x.shape[0] != len(y) or len(y) < 2
            or not np.isin(y, [0, 1]).all()
            or not np.isfinite(x).all()
        ):
            raise ValueError("Invalid response-attention training sample")
        counts += np.bincount(y.astype(np.int64), minlength=2)
    if (counts == 0).any():
        raise ValueError("Training data must contain both clean and injected tokens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = ResponseAttentionDetector(layers, heads).to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)
    detector.train()
    loss_value = 0.0
    for _ in range(epochs):
        for i in rng.permutation(len(features)):
            x, y = features[i], labels[i]
            selected = rng.choice(len(y), size=min(TOKENS_PER_EXAMPLE, len(y)), replace=False)
            batch = torch.from_numpy(x[selected].astype(np.float32)).to(device)
            batch = F.pad(batch, (0, RESPONSE_LIMIT - batch.shape[-1]))
            target = torch.from_numpy(y[selected].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(detector(batch), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite detector training loss")
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "response_attention_detector_v1",
        "model_id": model_config["model_id"],
        "max_input_tokens": int(model_config["max_input_tokens"]),
        "max_new_tokens": int(model_config["max_new_tokens"]),
        "layers": layers,
        "heads": heads,
        "response_limit": RESPONSE_LIMIT,
        "training_sample_ids": sorted(set(sample_ids)),
        "training_class_counts": {"clean": int(counts[0]), "injected": int(counts[1])},
        "training_epochs": epochs,
        "seed": seed,
        "state_dict": detector.cpu().state_dict(),
    }, output_path)
    return {"example_count": len(sample_ids), "token_count": int(counts.sum()), "last_batch_loss": loss_value}


def load_response_detector(
    path: Path, model_config: dict[str, Any], device: torch.device
) -> ResponseAttentionDetector:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "response_attention_detector_v1":
        raise ValueError("Expected a response-attention detector checkpoint")
    _validate_model_config(checkpoint, model_config, "Detector")
    if checkpoint["response_limit"] != RESPONSE_LIMIT:
        raise ValueError("Unsupported detector response limit")
    detector = ResponseAttentionDetector(checkpoint["layers"], checkpoint["heads"])
    detector.load_state_dict(checkpoint["state_dict"])
    return detector.to(device).eval()


def response_token_logits(
    detector: ResponseAttentionDetector, features: torch.Tensor, batch_size: int = 32
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if features.ndim != 4 or features.shape[0] == 0:
        raise ValueError("Expected non-empty response-attention features")
    if not torch.isfinite(features).all():
        raise ValueError("Response-attention features must be finite")
    device = next(detector.parameters()).device
    with torch.inference_mode():
        logits = torch.cat([
            detector(chunk.to(device=device, dtype=torch.float32)).cpu()
            for chunk in features.split(batch_size)
        ])
    if not torch.isfinite(logits).all():
        raise RuntimeError("Detector produced non-finite logits")
    return logits


def predict_response_tokens(
    detector: ResponseAttentionDetector, features: torch.Tensor, batch_size: int = 32
) -> list[int]:
    return reference_token_predictions(response_token_logits(detector, features, batch_size))


def reference_token_predictions(logits: torch.Tensor) -> list[int]:
    if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] == 0:
        raise ValueError("Logits must have shape [non-empty tokens, 2]")
    if not torch.isfinite(logits).all():
        raise ValueError("Detector logits must be finite")
    # Match the reference's Python-float summation and short-sequence window edges.
    scores = logits[:, 0].tolist()
    length = len(scores)
    initial_width = min(3, length)
    return [
        int(sum(scores[max(0, i - 2, i + initial_width - length):min(length, i + 3)]) <= 0)
        for i in range(length)
    ]
