from transformers import pipeline

model_path = "./Llama-3.1-8B-Instruct"

# Load pipeline
pipe = pipeline("text-generation", model=model_path)

# Test model output
output = pipe("Hello, how are you?", max_length=50)
print(output)