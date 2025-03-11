from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Define a Persistent Save Path
SAVE_PATH = "./Qwen2.5-1.5B"  # Change to a relevant directory

# ✅ Model Name from Hugging Face Hub
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Qwen model name

# ✅ Download Tokenizer
print("🚀 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ✅ Download and Save the Model to a Persistent Path
print("🚀 Downloading model weights... This may take time.")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH, trust_remote_code=True)

# ✅ Save the tokenizer and model
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"✅ Model saved to: {SAVE_PATH}")
