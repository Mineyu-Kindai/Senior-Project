import argparse
import json
from pathlib import Path

import yaml

from .data import load_records, select_split
from .detector import train_detector
from .llm import load_llm
from .pipeline import (
    collect_features,
    evaluate_baseline,
    evaluate_defense,
    save_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Expected a top-level 'model' mapping in {path}")
    return config["model"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="senior-project",
        description="Collect attention features, train a token detector, and evaluate redaction.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="Extract token-labeled attention features from a dataset"
    )
    collect_parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "task_samples.json"
    )
    collect_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "attention_features.npz"
    )
    collect_parser.add_argument("--split", choices=("train", "test"))

    train_parser = subparsers.add_parser(
        "train", help="Train the lightweight attention token classifier"
    )
    train_parser.add_argument(
        "--features", type=Path, default=PROJECT_ROOT / "data" / "attention_features.npz"
    )
    train_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "attention_detector.pt"
    )
    train_parser.add_argument("--epochs", type=int, default=40)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compare unfiltered and attention-redacted model responses"
    )
    evaluate_parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "task_samples.json"
    )
    evaluate_parser.add_argument(
        "--detector",
        type=Path,
        default=PROJECT_ROOT / "models" / "attention_detector.pt",
    )
    evaluate_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "evaluation.json"
    )
    evaluate_parser.add_argument("--split", choices=("train", "test"))

    baseline_parser = subparsers.add_parser(
        "baseline", help="Run the unfiltered model for a baseline attack measurement"
    )
    baseline_parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "task_samples.json"
    )
    baseline_parser.add_argument("--split", choices=("train", "test"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    model_config = read_config(args.config)

    if args.command == "train":
        result = train_detector(
            args.features,
            args.output,
            model_config["model_id"],
            epochs=args.epochs,
        )
        print(json.dumps({"checkpoint": str(args.output), **result}, ensure_ascii=False))
        return

    records = select_split(load_records(args.input), args.split)
    tokenizer, model = load_llm(model_config)
    if args.command == "baseline":
        result = evaluate_baseline(records, tokenizer, model, model_config)
        print(json.dumps({"model_id": model_config["model_id"], "examples": result}, ensure_ascii=False))
        return
    if args.command == "collect":
        result = collect_features(
            records, tokenizer, model, model_config, args.output
        )
        print(json.dumps({"features": str(args.output), **result}, ensure_ascii=False))
        return

    result = evaluate_defense(
        records, tokenizer, model, model_config, args.detector
    )
    save_report(result, args.output)
    print(json.dumps({"report": str(args.output), **result}, ensure_ascii=False))
