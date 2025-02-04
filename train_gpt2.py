import torch
import random
import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ✅ Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Training on: {device}")

# ✅ Set save directory
SAVE_DIR = "./fine_tuned_gpt2"

# ✅ Load dataset (assumes JSONL format)
dataset = load_dataset("json", data_files="gpt2_finetune.jsonl", split="train")

# ✅ Shuffle the dataset before splitting
dataset = dataset.shuffle(seed=42)

# ✅ Split the dataset (Train on first half, save second half for later training)
total_size = len(dataset)
half_size = total_size // 100
train_dataset = dataset.select(range(half_size))  # Use first half
remaining_dataset = dataset.select(range(half_size, total_size))  # Save for later

# ✅ Save remaining dataset for future training
remaining_dataset.to_json("gpt2_finetune_remaining.jsonl")
print(f"✅ Training on {half_size} samples. Remaining {total_size - half_size} saved for future training.")

# ✅ Load tokenizer & model (Resume from checkpoint if exists)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a padding token

try:
    model = GPT2LMHeadModel.from_pretrained(SAVE_DIR)  # Resume from checkpoint
    print("✅ Resuming training from last checkpoint!")
except:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("⚡ Starting training from scratch!")

# ✅ Move model to GPU
model.to(device)

# ✅ Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# ✅ Data collator for batching
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Training arguments (automatically saves after each epoch)
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # Adjust based on GPU memory
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # Train for 3 epochs on half the dataset
    weight_decay=0.01,
    save_strategy="epoch",  # Save checkpoint after each epoch
    save_total_limit=2,  # Keep latest 2 checkpoints
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # Mixed precision training (faster on A40, A100, V100)
)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ✅ Train model
trainer.train()

# ✅ Save model checkpoint
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Training completed! Model saved to {SAVE_DIR}")
print(f"🚀 You can resume training later using 'gpt2_finetune_remaining.jsonl'.")

