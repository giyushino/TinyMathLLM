from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig 
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = LlamaTokenizer.from_pretrained(model_id)


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

for name, param in model.named_parameters():
    print(name)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


data = load_dataset("json", data_files=r"C:\Users\allan\nvim\tinyMath\TinyMathLLM\datasets\dataset_extract.json")

def format_messages(dataset):
    """
    Applies .apply_chat_template to dataset
    """
    formatted = [tokenizer.apply_chat_template(messages, tokenize = False) for messages in dataset["messages"]]
    return {"formatted": formatted}

formatted_data = data.map(format_messages, batched = True)

def tokenize_function(examples):
    """
    Tokenizes formatted message
    """

    tokenized = tokenizer(examples["formatted"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  
    return tokenized

tokenized_dataset = formatted_data.map(tokenize_function, batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs = 3,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=r"C:\Users\allan\nvim\tinyMath\TinyMathLLM\models\exract_weights\quantized",
        optim="paged_adamw_8bit", 
        save_steps=300,
        save_total_limit=3
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
