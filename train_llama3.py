import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import os

# Move dataset to fastest storage
DATA_FILE = "trainingdata1.jsonl"
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Load tokenizer
MODEL_PATH = "/tmp_data/models/Llama-3.1-8B-Instruct"
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

# Load Llama3 with 4-bit quantization (GPU only)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Use bf16 instead of fp16
    device_map="auto",  # **Keep everything on GPU**
    attn_implementation="flash_attention_2"  # Enable Flash Attention
)

# Enable training mode
model.train()

# LoRA Configuration (optimized for speed)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=8,  
    lora_alpha=32,  
    lora_dropout=0.1,  
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="/tmp_data/llama3_finetuned",
    per_device_train_batch_size=2,  # Reduce batch size
    gradient_accumulation_steps=16,  # Increase accumulation to keep effective batch size
    num_train_epochs=3,
    bf16=True,  # Use bfloat16 (if supported)
    fp16=False,

    learning_rate=2e-5,
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_steps=100,  # Log progress every 100 steps

    save_strategy="steps",
    save_safetensors=True,
)

# Check for the latest checkpoint
checkpoint_dir = "/tmp_data/llama3_finetuned"
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
model.save_pretrained("/tmp_data/llama3_finetuned")
tokenizer.save_pretrained("/tmp_data/llama3_finetuned")
print("Training complete! Model saved in '/tmp_data/llama3_finetuned'.")
