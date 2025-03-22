import os
os.environ['HF_HOME'] = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_list = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "stabilityai/stablelm-zephyr-3b",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-Math-7B"
]

for model in model_list:
    print(f"Downloading {model} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        print(f"Error {e}. Retrying for tokenizer ...")
        continue

    try:
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        print(f"Error {e}. Retrying for model ...")
        continue
