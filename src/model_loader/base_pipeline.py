import torch 
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer 

def load_model_pipeline(custom, model_path, prompt = None, role = None, modify = None):
    """
    Load LLM using transformers pipeline

    Args:
        model (str): Model name

    Returns:
        (str): Model generated text
    """

    if custom:
        model = LlamaForCausalLM.from_pretrained(model_path)
    else:
        model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    pipe = pipeline("text-generation", model=model, tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

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

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    return outputs[0]["generated_text"]

load_model_pipeline(custom=True, model_path = r"C:\users\allan\nvim\tinymath\trained_models\answer_no_mult")
