from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
import os
from openai import OpenAI
import gc
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_PATH = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/"

model_paths = [
    # DEFAULT_PATH + "llama-fine-tuned",
    # DEFAULT_PATH + "qwen_finetuned",
    # DEFAULT_PATH + "mistral_finetuned",
    DEFAULT_PATH + "Llama-3.1-8B-Instruct",
    DEFAULT_PATH + "models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
]

# Load products from input.json
with open("input_all.json", "r") as f:
    products = json.load(f)

# Helper function to create prompt
def create_prompt(product):
    prompt = (
        f"Category: {product['Category']}\n"
        f"Name: {product['Name']}\n"
        f"Product Description:\n"
    )
    for i, desc in enumerate(product["Product Description"], 1):
        prompt += f"{i}. {desc}\n"
    prompt += "\nInclude all the product details and try to mimic the actual Instagram posts. Try to be descriptive.\nInstagram Post:\n"
    return prompt

# Get clean model name
def get_model_name(path):
    return os.path.basename(path.strip("/"))

# Final output dictionary
final_results = {}

for idx, product in enumerate(products):
    prompt = create_prompt(product)
    output_entry = {}

    print(f"\n📦 Generating outputs for: {product['Name']}")

    for model_path in model_paths:
        model_name = get_model_name(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=512)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_entry[model_name] = generated_text
        print(f"✅ {model_name} done.")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    final_results[f"output_{idx + 1}"] = output_entry

# Save results
with open("untuned_ouptut.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("\n✅ All outputs saved to test.json")



# combined_outputs = "\n\n".join(
#     f"Output from model {i+1}:\n{result}" for i, result in enumerate(results.values())
# )
chat_prompt = (
    "You're a professional social media content creator.\n"
    "You are given several Instagram post drafts written by different AI models based on the same product.\n"
    "Your task is to analyze them, pick the best parts, and write one perfect, polished Instagram post "
    "that is influencer-style, engaging, and uses natural, fluent English. It must include the name of the product "
    "and all important details. Feel free to add more to make the post better, but try to use as much parts as possible from the input.\n\n"
    "Here are the drafts:\n"
    f"{combined_outputs}\n\n"
    "Now write the final Instagram post:"
)

# client = OpenAI(api_key="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")

# # Send the combined outputs to GPT-4o
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a skilled Instagram influencer and content writer."},
#         {"role": "user", "content": chat_prompt}
#     ],
#     temperature=0.7,
#     max_tokens=300
# )

# # Print the final post
# final_post = response["choices"][0]["message"]["content"]
# print("\n💡 Final Instagram Post by ChatGPT:")
# print(final_post)