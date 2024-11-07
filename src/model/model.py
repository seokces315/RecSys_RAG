import torch
from transformers import pipeline

HF_TOKEN = "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"

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)

print()
print("[ Answer ]")
print(outputs[0]["generated_text"][-1]["content"])
print()
