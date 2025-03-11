from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/checkpoint-42108"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "Category: Beauty & Cosmetics, Name: Thayers facial toner, Product Description: 1. Alcohol-free 2.  Formulated to soothe, tone, hydrate, and balance the pH level of skin 3. reduce the look of pores, balance oily skin, and maintain the skin's moisture barrier"
prompt += "Write me an instagram post that includes those product description and the name of the product. Try to mimic the way the Instagram posts from real influencers that I used for training."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=512)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)