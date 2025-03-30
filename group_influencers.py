# group_influencers.py

import json
import sys
from collections import defaultdict

def group_usernames_by_category(input_file, output_file="influencers_by_category.json"):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        category_map = defaultdict(list)

        for influencer in data:
            category = influencer.get("Category")
            username = influencer.get("Username")
            if category and username:
                category_map[category].append(username)

        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump(category_map, out, indent=2, ensure_ascii=False)

        print(f"Grouped influencer usernames saved to '{output_file}'")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python group_influencers.py input_file.json")
    else:
        group_usernames_by_category(sys.argv[1])
