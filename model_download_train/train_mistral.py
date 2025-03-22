import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# Move dataset to fastest storage
DATA_FILE = "trainingdata3.jsonl"
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Load tokenizer
MODEL_PATH = "/home/johnkimm/InstaAIModel/Mistral_7B-Instruct-v0.3"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=False)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# Load Mistral-7B-Instruct-v0.3 with 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Use bf16 instead of fp16
    device_map="auto",  # Keep everything on GPU
    attn_implementation="flash_attention_2"  # Enable Flash Attention
)

# Enable training mode
model.train()

# LoRA Configuration (optimized for Mistral)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=8,  
    lora_alpha=32,  
    lora_dropout=0.1,  
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Adjusted for Mistral
)

model = get_peft_model(model, lora_config)

# Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="/home/johnkimm/InstaAIModel/mistral_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    fp16=True,  # Enable fp16 instead of bf16
    bf16=False,  # Disable bf16

    learning_rate=2e-5,
    save_steps=25,
    save_total_limit=2,
    logging_steps=25,
    save_strategy="steps",
    save_safetensors=True,
)

# Check for the latest checkpoint
checkpoint_dir = "/home/johnkimm/InstaAIModel/mistral_finetuned"
last_checkpoint = None
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    last_checkpoint = max(
        [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")],
        key=os.path.getmtime,
    )
    print(f"Resuming from checkpoint: {last_checkpoint}")
else: 
    print("No previous checkpoint found. Starting fresh training.")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start Training (Resume if a checkpoint exists)
trainer.train(resume_from_checkpoint=last_checkpoint)

# Save model after training
model.save_pretrained("/home/johnkimm/InstaAIModel/mistral_finetuned")
tokenizer.save_pretrained("/home/johnkimm/InstaAIModel/mistral_finetuned")
print("Training complete! Model saved in '/home/johnkimm/InstaAIModel/mistral_finetuned'.")
