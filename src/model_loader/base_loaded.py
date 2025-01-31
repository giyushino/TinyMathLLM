from transformers import LlamaForCausalLM, LlamaTokenizer 
import torch

def load_model(model_path, tokenizer_path = None):
    """
    Load base TinyLlama model or finetuned version using transformers
    
    Args:
        model_path (str): Model name or path to model if finetuned 
        tokenizer_path (str, optional): Tokenizer name or path to custom tokenizer 

    Returns:
        model, tokenizer
    """

    if tokenizer_path is None:
        tokenizer_path = model_path

    llama = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    return llama, tokenizer


def get_response(model, tokenizer, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), prompt = None, role = None, modify = None):
    """
    Returns response of TinyLlama based on prompt or user input
    
    Args:
        model (LlamaForCausalLM): Model used
        tokenizer (LlamaTokenizer): Tokenized used
        prompt (str, optional): Message fed into tokenizer and model. If left blank, will prompt user for input
        modify (str, optional): Modify prompt before feeding into model
    
    Returns:
        (str): User's input prompt + role explanation, model's output
    """
    
    llama = model
    llama.to(device)
    llama_tokenizer = tokenizer
   
    if prompt is None:
        prompt = input("Enter: ")
    
    if modify is not None:
        prompt += modify 

    messages = [
        {
            "role": "system",
            "content": "{}".format(role),
        },
        {"role": "user", "content": "{}".format(prompt)},
    ]

    enter = llama_tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors ="pt").to(device)
    outputs = llama.generate(enter, max_new_tokens = 128)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])
    

llama, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0") 

get_response(llama, tokenizer, role = "You are a mathematics professor who helps students with their math problems. Provide them with the answer")

