from transformers import pipeline

model_path = "/tmp/huggingface_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Load pipeline
pipe = pipeline("text-generation", model=model_path)

# Test model output
output = pipe("Hello, how are you?", max_length=50)
print(output)