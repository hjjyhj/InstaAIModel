from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Model Name (Ensure you have access)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ✅ Download the tokenizer
print("🚀 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Download the model (this may take time)
print("🚀 Downloading model weights...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("✅ Model download complete!")
