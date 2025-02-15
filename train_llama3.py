import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os

# Load dataset
DATA_FILE = "post_training_data.jsonl"
dataset = load_dataset("json", data_files=DATA_FILE, split="train").shuffle(seed=42)

# Load tokenizer
MODEL_PATH = "/tmp_data/models/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=False)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Tokenization function (fixes padding/truncation issue)
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",  # Ensure all inputs are of fixed length
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()  # Ensure labels are of the same length
    return tokenized

# Apply tokenization with consistent padding
dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])  

# Load Llama3 with 4-bit quantization using BitsAndBytesConfig
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,  # Use float32 for stability
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Enable training mode
model.train()

# LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=16,  
    lora_alpha=32,  
    lora_dropout=0.05,  
    target_modules=["q_proj", "v_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./llama3_finetuned",  
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=8,  
    num_train_epochs=3,  
    fp16=True,  

    learning_rate=2e-5,  
    save_steps=500,  
    save_total_limit=2,  
    logging_steps=100,  

    save_strategy="steps", 
    save_safetensors=True, 
)

# Custom Trainer to explicitly compute loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = None

        return (loss, outputs) if return_outputs else loss

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start Training
trainer.train()

# Save model if training is complete
if trainer.state.global_step >= trainer.state.max_steps:
    model.save_pretrained("llama3_finetuned")
    tokenizer.save_pretrained("llama3_finetuned")
    print("Training complete! Fine-tuned model saved in 'llama3_finetuned'.")
else:
    print("Training not done yet. Checkpoints saved.")
