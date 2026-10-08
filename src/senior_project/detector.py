from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class AttentionTokenClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def train_detector(
    feature_path: Path,
    output_path: Path,
    model_id: str,
    epochs: int = 40,
    learning_rate: float = 1e-3,
) -> dict[str, float]:
    with np.load(feature_path, allow_pickle=False) as dataset:
        features = torch.from_numpy(dataset["features"].astype(np.float32))
        labels = torch.from_numpy(dataset["labels"].astype(np.int64))

    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Feature file must contain [tokens, features] and [tokens] labels")
    class_counts = torch.bincount(labels, minlength=2)
    if torch.any(class_counts == 0):
        raise ValueError(
            "Training data must contain both clean and injected tokens. "
            "Add labeled examples before training."
        )
    if epochs < 1:
        raise ValueError("epochs must be greater than zero")

    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    model = AttentionTokenClassifier(features.shape[1])
    class_weights = len(labels) / (2 * class_counts.float())
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(normalized), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        predictions = model(normalized).argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": model_id,
            "input_size": features.shape[1],
            "hidden_size": 128,
            "mean": mean,
            "std": std,
            "state_dict": model.state_dict(),
            "threshold": 0.5,
        },
        output_path,
    )
    return {"training_accuracy": accuracy, "token_count": float(len(labels))}


def load_detector(
    checkpoint_path: Path, expected_model_id: str, device: torch.device
) -> tuple[AttentionTokenClassifier, torch.Tensor, torch.Tensor, float]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {checkpoint_path}. Run the train command first."
        )
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint["model_id"] != expected_model_id:
        raise ValueError(
            "Detector checkpoint was trained for "
            f"{checkpoint['model_id']!r}, not {expected_model_id!r}"
        )
    model = AttentionTokenClassifier(
        int(checkpoint["input_size"]), int(checkpoint["hidden_size"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return (
        model,
        checkpoint["mean"].to(device),
        checkpoint["std"].to(device),
        float(checkpoint["threshold"]),
    )


def predict_tokens(
    model: AttentionTokenClassifier,
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    threshold: float,
) -> tuple[list[int], list[float]]:
    with torch.inference_mode():
        logits = model((features.to(mean.device) - mean) / std)
        probabilities = logits.softmax(dim=1)[:, 1]
    scores = probabilities.cpu().tolist()
    return [int(score >= threshold) for score in scores], scores


def detector_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metric inputs must be non-empty and have equal lengths")
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    return {
        "token_accuracy": (tp + tn) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
