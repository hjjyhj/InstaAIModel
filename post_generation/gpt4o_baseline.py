import json
from openai import OpenAI

with open('all_input_product_desc.json', 'r') as f:
    products = json.load(f)
 
"""
Convert each product into a formatted string
"""
formatted_products = []
for product in products:
    category = product["Category"]
    name = product["Name"].lower()  
    description = product["Product Description"]
    
    formatted_desc = " ".join([f"{i+1}. {desc[0].upper() + desc[1:]}" for i, desc in enumerate(description)])
    
    formatted_str = f"Category: {category}, Name: {name}, Product Description: {formatted_desc}"
    formatted_products.append(formatted_str)


api_key = ""
client = OpenAI(api_key=api_key)

"""
Prompt GPT-4o
"""
with open("output_4o_baseline.txt", 'a', encoding='utf-8') as f:
    for item in formatted_products:
        full_prompt = item
        full_prompt += "\nWrite me an instagram post that includes those product description and the name of the product. Try to mimic the way the Instagram posts from real influencers that I used for training."
        response = client.responses.create(
            model="gpt-4o-2024-05-13",
            input = full_prompt
        )
        generated_text = response.output[0].content[0].text

        f.write(generated_text)
        f.write("\n\n\n")
