# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "What is the derivative of 5x^2?"},
]
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Math-1.5B")
print(pipe(messages))

