from src.model_loader.four_bit_quantized import *
import numpy as np 

def answer(response):
    print("test")
    lower = response.lower()
    classification = lower[lower.index("question type"):]
    answer = classification[classification.index("important values"):]
    question_type = classification[:classification.index('.')]
    important_values = answer[:answer.index('.')]

    return question_type, important_values

while True:
    llama, llama_tokenizer = load_quantized_model(r"c:\users\allan\nvim\tinymath\tinymathllm\models\extract_weights\quantized\checkpoint-1000", "tinyllama/tinyllama-1.1b-chat-v1.0")
    response = get_quantized_response(llama, llama_tokenizer, max_new_tokens=120, modify = "what kind of question is this, and what are the important values?", should_print = True)
    print(answer(response))
