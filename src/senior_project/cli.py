import argparse
import json
import math
from pathlib import Path

import yaml

from .data import load_records, select_split
from .detector import (
    FILTERED_REMOVAL_DEFAULT_LOGIT_MARGIN,
    FILTERED_REMOVAL_DEFAULT_MIN_RUN_LENGTH,
    train_detector,
)
from .llm import load_llm
from .pipeline import (
    collect_features,
    collect_response_features,
    evaluate_baseline,
    evaluate_defense,
    save_report,
)
from .rennervate import train_response_detector


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
        "--output", type=Path
    )
    collect_parser.add_argument("--split", choices=("train", "test"))
    collect_parser.add_argument(
        "--architecture", choices=("prototype", "rennervate-pooling"), default="prototype"
    )

    train_parser = subparsers.add_parser(
        "train", help="Train the lightweight attention token classifier"
    )
    train_parser.add_argument(
        "--features", type=Path
    )
    train_parser.add_argument(
        "--output", type=Path
    )
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--architecture", choices=("prototype", "rennervate-pooling"), default="prototype"
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compare unfiltered and attention-redacted model responses"
    )
    evaluate_parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "task_samples.json"
    )
    evaluate_parser.add_argument(
        "--detector",
        type=Path,
    )
    evaluate_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "evaluation.json"
    )
    evaluate_parser.add_argument("--split", choices=("train", "test"))
    evaluate_parser.add_argument(
        "--detector-mode",
        choices=("threshold", "rennervate", "rennervate-remove", "rennervate-filtered-remove", "rennervate-pooling"),
        default="threshold",
        help="Choose the detection and sanitization mode",
    )
    evaluate_parser.add_argument(
        "--logit-margin", type=float, default=None,
        help=(
            "Experimental smoothed logit-difference threshold "
            f"(filtered-removal mode only; default {FILTERED_REMOVAL_DEFAULT_LOGIT_MARGIN})"
        ),
    )
    evaluate_parser.add_argument(
        "--min-run-length", type=int, default=None,
        help=(
            "Minimum consecutive positive tokens to remove "
            f"(filtered-removal mode only; default {FILTERED_REMOVAL_DEFAULT_MIN_RUN_LENGTH})"
        ),
    )

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
    if args.command == "evaluate" and args.detector_mode != "rennervate-filtered-remove":
        if args.logit_margin is not None or args.min_run_length is not None:
            raise ValueError("--logit-margin and --min-run-length require rennervate-filtered-remove")
    if args.command == "evaluate":
        if args.logit_margin is not None and (
            not math.isfinite(args.logit_margin) or args.logit_margin < 0
        ):
            raise ValueError("--logit-margin must be finite and non-negative")
        if args.min_run_length is not None and args.min_run_length < 1:
            raise ValueError("--min-run-length must be a positive integer")
    model_config = read_config(args.config)

    if args.command == "train":
        response_mode = args.architecture == "rennervate-pooling"
        features = args.features or PROJECT_ROOT / "data" / (
            "response_attention_features.npz" if response_mode else "attention_features.npz"
        )
        output = args.output or PROJECT_ROOT / "models" / (
            "response_attention_detector.pt" if response_mode else "attention_detector.pt"
        )
        if response_mode:
            result = train_response_detector(
                features, output, model_config, epochs=args.epochs, seed=args.seed
            )
        else:
            result = train_detector(
                features, output, model_config["model_id"], epochs=args.epochs, seed=args.seed
            )
        print(json.dumps({"checkpoint": str(output), **result}, ensure_ascii=False))
        return

    records = select_split(load_records(args.input), args.split)
    tokenizer, model = load_llm(model_config)
    if args.command == "baseline":
        result = evaluate_baseline(records, tokenizer, model, model_config)
        print(json.dumps({"model_id": model_config["model_id"], "examples": result}, ensure_ascii=False))
        return
    if args.command == "collect":
        response_mode = args.architecture == "rennervate-pooling"
        output = args.output or PROJECT_ROOT / "data" / (
            "response_attention_features.npz" if response_mode else "attention_features.npz"
        )
        collector = collect_response_features if response_mode else collect_features
        result = collector(records, tokenizer, model, model_config, output)
        print(json.dumps({"features": str(output), **result}, ensure_ascii=False))
        return

    checkpoint = args.detector or PROJECT_ROOT / "models" / (
        "response_attention_detector.pt"
        if args.detector_mode == "rennervate-pooling" else "attention_detector.pt"
    )
    result = evaluate_defense(
        records,
        tokenizer,
        model,
        model_config,
        checkpoint,
        detector_mode=args.detector_mode,
        **({"logit_margin": args.logit_margin} if args.logit_margin is not None else {}),
        **({"min_run_length": args.min_run_length} if args.min_run_length is not None else {}),
    )
    save_report(result, args.output)
    print(json.dumps({"report": str(args.output), **result}, ensure_ascii=False))
