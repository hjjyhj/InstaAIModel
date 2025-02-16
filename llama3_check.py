from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./Llama-3.1-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")
