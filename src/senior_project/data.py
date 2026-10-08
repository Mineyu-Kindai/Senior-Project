import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in file if line.strip()]
        else:
            records = json.load(file)

    if isinstance(records, dict):
        if {"user_input", "external_context"} <= records.keys():
            records = [records]
        else:
            if not records or any(not isinstance(record, dict) for record in records.values()):
                raise ValueError(
                    "Expected a JSON object mapping task sample names to task objects: "
                    f"{path}"
                )
            records = [
                {"id": sample_name, **record}
                for sample_name, record in records.items()
            ]
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"Expected a non-empty task-sample object, JSON array, or JSONL file: {path}"
        )
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Every dataset entry must be a JSON object: {path}")
    return records


def select_split(
    records: list[dict[str, Any]], split: str | None
) -> list[dict[str, Any]]:
    if split is None:
        return records
    selected = [record for record in records if record.get("split") == split]
    if not selected:
        raise ValueError(f"No task samples found for split {split!r}")
    return selected


def require_task_fields(
    record: dict[str, Any], index: int
) -> tuple[str, str, str | None]:
    required = ("user_input", "external_context")
    missing = [key for key in required if not isinstance(record.get(key), str)]
    if missing:
        raise ValueError(f"Dataset entry {index} is missing string field(s): {', '.join(missing)}")

    instruction = record.get("injected_instruction")
    context = record["external_context"]
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError(
            f"Dataset entry {index} field 'injected_instruction' must be a string when provided"
        )
    if instruction == "":
        instruction = None
    if instruction is not None and instruction not in context:
        raise ValueError(
            f"Dataset entry {index} injected_instruction must appear inside external_context"
        )
    return record["user_input"], context, instruction
