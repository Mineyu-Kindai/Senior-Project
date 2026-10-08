from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .attention import extract_context_features
from .prompt import EncodedPrompt, encode_prompt


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


def generate_answer(
    tokenizer, model, user_input: str, context: str, config: dict[str, Any]
) -> str:
    prompt = prepare_prompt(tokenizer, user_input, context, config)
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in prompt.model_inputs.items()}
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            max_new_tokens=int(config["max_new_tokens"]),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_length = int(inputs["input_ids"].shape[-1])
    return tokenizer.decode(
        sequences[0, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
