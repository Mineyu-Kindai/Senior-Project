from typing import Any

import torch

from .llm import model_device
from .prompt import SYSTEM_PROMPT, EncodedPrompt


def remove_flagged_tokens(
    tokenizer, prompt: EncodedPrompt, context: str, predictions: list[int]
) -> str:
    if len(predictions) != len(prompt.context_token_indices):
        raise ValueError("Token predictions and context indices must have equal lengths")
    if any(flag not in (0, 1) for flag in predictions):
        raise ValueError("Token predictions must contain only binary labels")
    if not prompt.context_offsets or prompt.context_offsets[-1][1] != len(context):
        raise ValueError("Cannot remove tokens from a truncated external context")

    input_ids = prompt.model_inputs["input_ids"][0]
    token_ids = [int(input_ids[index]) for index in prompt.context_token_indices]
    full_text = tokenizer.decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    context_start = full_text.find(context)
    if context_start < 0 or full_text.find(context, context_start + 1) >= 0:
        raise ValueError("Decoded context tokens do not uniquely contain the original context")
    prefix = full_text[:context_start]
    suffix = full_text[context_start + len(context):]
    kept = [token for token, flag in zip(token_ids, predictions, strict=True) if not flag]
    result = tokenizer.decode(
        kept, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if prefix and not predictions[0]:
        if not result.startswith(prefix):
            raise ValueError("Decoded token prefix does not match the context boundary")
        result = result[len(prefix):]
    if suffix and not predictions[-1]:
        if not result.endswith(suffix):
            raise ValueError("Decoded token suffix does not match the context boundary")
        result = result[:-len(suffix)]
    return result


def generate_with_empty_context(
    tokenizer, model, user_input: str, config: dict[str, Any]
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_input}\n\n<external_context>\n\n</external_context>"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        rendered, return_tensors="pt", truncation=True,
        max_length=int(config["max_input_tokens"]), add_special_tokens=False,
    )
    inputs = {key: value.to(model_device(model)) for key, value in encoded.items()}
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            max_new_tokens=int(config["max_new_tokens"]),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        sequences[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
