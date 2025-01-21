from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig 
import torch

def load_quantized_model(model_path, tokenizer_path = None):
    """
    Load quantized TinyLlama model or finetuned version using transformers
    
    Args:
        model_path (str): Model name or path to model if finetuned 
        tokenizer_path (str, optional): Tokenizer name or path to custom tokenizer 

    Returns:
        model, tokenizer
    """

    if tokenizer_path is None:
        tokenizer_path = model_path

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  
    )

    model = LlamaForCausalLM.from_pretrained(model_path, quantization_config = quantization_config)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer

def get_quantized_response(model, tokenizer, prompt = None, role = None, modify = None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Returns response of quantized TinyLlama based on prompt or user input

    Args:
        model (LlamaForCausalLM): Model used
        tokenizer (LlamaTokenizer): Tokenized used
        role (str, optional): Instructions on how model should respond
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


llama, llama_tokenizer = load_quantized_model(r"C:\Users\allan\nvim\TinyMath\TinyMathLLM\models\extract_weights\checkpoint-900", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
get_quantized_response(llama, llama_tokenizer, modify = "What kind of question is this, and what are the important values?")
