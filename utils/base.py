import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME_MAP = {
    "chatglm3-6b": "./LLMs/chatglm3-6b",
    "Llama-2-7b": "./LLMs/Llama-2-7b",
}