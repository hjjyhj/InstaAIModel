from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Define a Persistent Save Path
SAVE_PATH = "/tmp_data/models/Llama-3.1-8B-Instruct"  # Fix the incorrect assignment

# ✅ Model Name from Hugging Face Hub
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ✅ Download Tokenizer
print("🚀 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Download and Save the Model to a Persistent Path
print("🚀 Downloading model weights... This may take time.")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)

# ✅ Save the tokenizer and model
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"✅ Model saved to: {SAVE_PATH}")
