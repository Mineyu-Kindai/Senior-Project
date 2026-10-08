import torch


def extract_context_features(
    attentions: tuple[torch.Tensor, ...] | list[torch.Tensor],
    context_token_indices: list[int],
    sequence_length: int,
) -> torch.Tensor:
    """Pool attention received by each context token across context query tokens."""
    if not attentions:
        raise RuntimeError(
            "The model returned no attention weights; use eager attention and "
            "enable output_attentions."
        )
    if not context_token_indices:
        raise ValueError("At least one context token is required")

    query_indices = torch.tensor(context_token_indices, dtype=torch.long)
    layer_features = []
    for layer_index, attention in enumerate(attentions):
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(
                f"Expected one-example attention shaped [1, heads, query, key], "
                f"got {tuple(attention.shape)} at layer {layer_index}"
            )
        if attention.shape[-2] != sequence_length or attention.shape[-1] != sequence_length:
            raise ValueError(
                "Expected full prompt attention. The model may have returned "
                "cached decoding attention instead."
            )
        query_indices = query_indices.to(attention.device)
        context_attention = attention[0].index_select(1, query_indices)
        context_attention = context_attention.index_select(2, query_indices)
        # [heads, context queries, context keys] -> [context keys, heads]
        layer_features.append(context_attention.float().mean(dim=1).transpose(0, 1))

    features = torch.cat(layer_features, dim=1)
    positions = torch.linspace(
        0.0, 1.0, steps=len(context_token_indices), device=features.device
    ).unsqueeze(1)
    return torch.cat((features, positions), dim=1).cpu().contiguous()
