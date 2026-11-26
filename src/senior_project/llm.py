from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .attention import extract_context_features
from .prompt import EncodedPrompt, encode_prompt
from .rennervate import RESPONSE_LIMIT


def load_llm(config: dict[str, Any]):
    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    load_options: dict[str, Any] = {
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_options["device_map"] = "auto"
        load_options["torch_dtype"] = torch.float16
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_options["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_options)
    model.eval()
    return tokenizer, model


def model_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def prepare_prompt(tokenizer, user_input: str, context: str, config: dict[str, Any]):
    return encode_prompt(
        tokenizer,
        user_input,
        context,
        max_length=int(config["max_input_tokens"]),
    )


def _generate_sequences(tokenizer, model, inputs: dict[str, torch.Tensor], config: dict[str, Any]):
    return model.generate(
        **inputs,
        max_new_tokens=int(config["max_new_tokens"]),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )


def capture_attention_features(
    prompt: EncodedPrompt, model
) -> torch.Tensor:
    device = model_device(model)
    model_inputs = {key: value.to(device) for key, value in prompt.model_inputs.items()}
    with torch.inference_mode():
        outputs = model(**model_inputs, output_attentions=True, use_cache=False)
    if outputs.attentions is None:
        raise RuntimeError(
            "The selected model did not return attention weights. "
            "Confirm eager attention is enabled for this Transformers version."
        )
    sequence_length = int(model_inputs["input_ids"].shape[-1])
    features = extract_context_features(
        outputs.attentions, prompt.context_token_indices, sequence_length
    )
    del outputs
    return features


def capture_response_attention(
    prompt: EncodedPrompt, model, tokenizer, config: dict[str, Any]
) -> torch.Tensor:
    features, _ = capture_response_attention_with_tokens(prompt, model, tokenizer, config)
    return features


def capture_response_attention_with_tokens(
    prompt: EncodedPrompt, model, tokenizer, config: dict[str, Any]
) -> tuple[torch.Tensor, list[int]]:
    max_new_tokens = int(config["max_new_tokens"])
    if max_new_tokens < 2:
        raise ValueError("Response-attention detection requires max_new_tokens >= 2")
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in prompt.model_inputs.items()}
    if "attention_mask" not in inputs:
        raise ValueError("Response-attention detection requires an attention mask")
    context_indices = torch.tensor(prompt.context_token_indices, device=device)
    attention_mask = inputs["attention_mask"]
    steps = []
    with torch.inference_mode():
        sequences = _generate_sequences(tokenizer, model, inputs, config)
        response = sequences[:, inputs["input_ids"].shape[-1]:]
        response_queries = min(RESPONSE_LIMIT, response.shape[-1] - 1)
        if response_queries <= 0:
            raise RuntimeError("The model generated no response tokens usable for attention detection")
        outputs = model(**inputs, output_attentions=False, use_cache=True)
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("The model did not return a generation cache")
        for index in range(response_queries):
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((1, 1))), dim=1
            )
            outputs = model(
                input_ids=response[:, index:index + 1],
                attention_mask=attention_mask,
                past_key_values=cache,
                output_attentions=True,
                use_cache=True,
            )
            if outputs.attentions is None or outputs.past_key_values is None:
                raise RuntimeError("The model did not return cached response attention")
            layer_attention = []
            for attention in outputs.attentions:
                if attention is None or attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[2] != 1:
                    raise RuntimeError("Expected one cached response attention query per layer")
                layer_attention.append(
                    attention[0, :, 0, :].index_select(-1, context_indices).to(
                        device="cpu", dtype=torch.float16
                    )
                )
            steps.append(torch.stack(layer_attention))
            cache = outputs.past_key_values
    features = torch.stack(steps, dim=-1).permute(2, 0, 1, 3).contiguous()
    if not torch.isfinite(features).all():
        raise RuntimeError("The model returned non-finite response attention")
    return features, response[0].tolist()


def generate_answer(
    tokenizer, model, user_input: str, context: str, config: dict[str, Any]
) -> str:
    prompt = prepare_prompt(tokenizer, user_input, context, config)
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in prompt.model_inputs.items()}
    with torch.inference_mode():
        sequences = _generate_sequences(tokenizer, model, inputs, config)
    input_length = int(inputs["input_ids"].shape[-1])
    return tokenizer.decode(
        sequences[0, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
