from src.model_loader.four_bit_quantized import *
import numpy as np 

while True:
    llama, llama_tokenizer = load_quantized_model(r"c:\users\allan\nvim\tinymath\tinymathllm\models\answer_weights\quantized\checkpoint-1000", "tinyllama/tinyllama-1.1b-chat-v1.0")
    response = get_quantized_response(llama, llama_tokenizer, max_new_tokens=20, should_print = True)


