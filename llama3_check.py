from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/tmp/huggingface_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")
