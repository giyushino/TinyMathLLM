from src.model_loader.four_bit_quantized import load_quantized_model, get_quantized_response
from datasets import load_dataset
import json 

DATASET_PATH = r"C:\Users\allan\nvim\TinyMath\TinyMathLLM\datasets\testset\answer.json"
SAVE_PATH =  r"C:\Users\allan\nvim\TinyMath\TinyMathLLM\src\eval\inference_results\finetuned_answer_testset.json"

def model_inference(model_type = 'base', dataset_path = DATASET_PATH, save_path = SAVE_PATH):
    """
    Computes all model outputs on dataset, saves to json file 

    Args:
        model_type (str): What model you want to use, default = 'base'
        dataset_path (str): Path to dataset you want to run model inference on, default is dataset_answer
    Returns:
        None
    """
    if model_type == "base":
        llama, llama_tokenizer = load_quantized_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    elif model_type == "answer":
        llama, llama_tokenizer = load_quantized_model(r"C:\Users\allan\nvim\TinyMath\TinyMathLLM\models\answer_weights\checkpoint-1000", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    elif model_type == "evaluate":
        llama, llama_tokenizer = load_quantized_model(r"C:\Users\allan\nvim\TinyMath\TinyMathLLM\models\evaluate_weights\checkpoint-900", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    #temp = get_quantized_response(llama, llama_tokenizer, role = "You are a mathematics professor who helps students with their math problems. Provide them with the answer")

    dataset = load_dataset("json", data_files = dataset_path)
    with open(save_path, "w") as file:
        for i in range(1000):
            temp = {}
            
            role = dataset["train"][i]["messages"][0]["content"]
            prompt = dataset["train"][i]["messages"][1]["content"]
            model_output = get_quantized_response(llama, llama_tokenizer, role = role, prompt = prompt)  
            expected_output = dataset["train"][i]["messages"][2]
            
            temp["prompt"] = prompt 
            temp["model_output"] = model_output.split("<|assistant|>")[1]    
            temp["correct_output"] = expected_output["content"]
            file.write(json.dumps(temp) + "\n")
            
            print(f"Line {i} created")
        
model_inference("answer")
