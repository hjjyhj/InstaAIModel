import torch
import os
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from flash_attn.layers import FlashAttention  # Explicitly importing Flash Attention

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define dataset processing
def tokenize_function(examples):
    """Tokenize text and ensure labels are included"""
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",  # Ensures all tensors are same length
        max_length=512,
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Copy input_ids to labels
    return tokenized_inputs

# Load dataset
DATA_FILE = "post_training_data.jsonl"
dataset = load_dataset("json", data_files=DATA_FILE, split="train").shuffle(seed=42)

# Define model path
MODEL_PATH = "/tmp/huggingface_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=False)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
dataset = dataset.map(tokenize_function, batched=True, num_proc=8)

# Load Llama3 with updated 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto"
)

# Enable Flash Attention
print("✅ Enabling Flash Attention...")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.MultiheadAttention):
        module.forward = FlashAttention.apply  # Replace standard attention with FlashAttention

# Enable memory optimizations
model.gradient_checkpointing_enable()  # Reduces memory usage
model = torch.compile(model)  # Speeds up execution

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=16,  
    lora_alpha=32,  
    lora_dropout=0.05,  
    target_modules=["q_proj", "v_proj"]  
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Trainer with correct loss computation
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()

        # Shift labels and logits for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# Adjust batch size dynamically based on GPU memory
def get_batch_size():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory >= 48 * 10**9:  # 48GB+ GPU
        return 16
    elif total_memory >= 40 * 10**9:  # 40GB+ GPU
        return 8
    elif total_memory >= 24 * 10**9:  # 24GB GPU
        return 4
    else:  # 16GB or lower GPU
        return 2

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama3_finetuned",  
    per_device_train_batch_size=get_batch_size(), 
    gradient_accumulation_steps=4,  # Helps if memory is tight
    num_train_epochs=3,  
    fp16=True,  # Mixed precision for speed and memory optimization
    optim="adamw_torch",  # Faster optimizer for training

    learning_rate=1e-5,  # Lower learning rate for stable training
    warmup_steps=500,  # Helps with convergence

    save_steps=500,  
    save_total_limit=2,  
    logging_steps=100,  

    save_strategy="steps", 
    save_safetensors=True, 
)

# Initialize trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer  # Use tokenizer correctly
)

# Start training
trainer.train()

# Save the fine-tuned model
if trainer.state.global_step >= trainer.state.max_steps:
    model.save_pretrained("llama3_finetuned")
    tokenizer.save_pretrained("llama3_finetuned")
    print("Training complete! Fine-tuned model saved in 'llama3_finetuned'.")
else:
    print("Training was interrupted before completion. Checkpoints saved, but model not fully saved yet.")
