from dataclasses import dataclass
from uuid import uuid4

import torch


SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's request."


@dataclass(frozen=True)
class EncodedPrompt:
    model_inputs: dict[str, torch.Tensor]
    context_token_indices: list[int]
    context_offsets: list[tuple[int, int]]
    rendered_text: str


def encode_prompt(tokenizer, user_input: str, context: str, max_length: int) -> EncodedPrompt:
    placeholder = f"EXTERNAL_CONTEXT_{uuid4().hex}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_input}\n\n<external_context>\n{placeholder}\n</external_context>"
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    context_start = rendered.find(placeholder)
    if context_start < 0:
        raise ValueError("The tokenizer chat template did not preserve the external context")
    rendered = (
        rendered[:context_start]
        + context
        + rendered[context_start + len(placeholder) :]
    )
    context_end = context_start + len(context)

    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    context_token_indices: list[int] = []
    context_offsets: list[tuple[int, int]] = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        overlap_start = max(token_start, context_start)
        overlap_end = min(token_end, context_end)
        if overlap_start < overlap_end:
            context_token_indices.append(token_index)
            context_offsets.append(
                (overlap_start - context_start, overlap_end - context_start)
            )

    if not context_token_indices:
        raise ValueError(
            "No external-context tokens remain after tokenization. "
            "Increase max_input_tokens or provide a non-empty context."
        )
    return EncodedPrompt(
        model_inputs=dict(encoded),
        context_token_indices=context_token_indices,
        context_offsets=context_offsets,
        rendered_text=rendered,
    )


def labels_for_span(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    span_start, span_end = span
    return [
        int(token_start < span_end and token_end > span_start)
        for token_start, token_end in offsets
    ]


def merge_flagged_spans(
    offsets: list[tuple[int, int]], predictions: list[int]
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for (start, end), flagged in zip(offsets, predictions, strict=True):
        if not flagged:
            continue
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def redact_spans(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in reversed(spans):
        text = text[:start] + "[REDACTED UNTRUSTED CONTENT]" + text[end:]
    return text
