import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import os

def tokenize_function(examples):
    return tokenizer(examples["Posts"], truncation=True, padding="max_length", max_length=512)

# Load dataset
DATA_FILE = "post_training_data.jsonl"
dataset = load_dataset("json", data_files=DATA_FILE, split="train").shuffle(seed=42)

# Load model and tokenizer
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

dataset = dataset.map(tokenize_function, batched=True, num_proc=8) 

# Load Llama3 with 4-bit quantization
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=16,  # Rank of LoRA matrices (lower = faster training, but less accuracy)
    lora_alpha=32,  
    lora_dropout=0.05,  # Dropout (prevents overfitting)
    target_modules=["q_proj", "v_proj"]  
)

model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="./llama3_finetuned",  # checkpoint
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=8,  # Simulates batch size of 32
    num_train_epochs=3,  
    fp16=True,  # Mixed precision for faster training

    learning_rate=2e-5,  

    # Save checkpoints every 500 steps
    save_steps=500,  
    
    # Resume training from the latest checkpoint automatically
    save_total_limit=2,  # keep the last 2 checkpoints
    logging_steps=100,  

    save_strategy="steps", 
    save_safetensors=True, 
)

# set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)
 
trainer.train() # training start

if trainer.state.global_step >= trainer.state.max_steps:
    model.save_pretrained("llama3_finetuned")
    tokenizer.save_pretrained("llama3_finetuned")
    print("Training complete! Fine-tuned model saved in 'llama3_finetuned'.")
else:
    print("Training was interrupted before completion. Checkpoints saved, but model not saved yet.")