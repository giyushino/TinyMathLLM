from src.model_loader.base_pipeline import load_model_pipeline
from src.model_loader.four_bit_quantized import load_quantized_model, get_quantized_response
import json
import os
from dotenv import load_dotenv
load_dotenv()

#*******************************
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#*******************************





#pipeline = load_model_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0", role = "You are a mathematics professor who helps students with their math problems. Provide them with the answer")


llama, llama_tokenizer = load_quantized_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
get_quantized_response(llama, llama_tokenizer, role = "You are a mathematics professor who helps students with their math problems. Provide them with the answer")
 



