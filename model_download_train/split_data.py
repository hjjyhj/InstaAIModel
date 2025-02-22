import json
import random

# ✅ Load dataset from JSONL file
input_file = "post_training_data.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ✅ Shuffle data randomly (for randomness across chunks)
random.seed(42)  # Ensures reproducibility
random.shuffle(data)

# ✅ Split dataset into 3 equal parts
split_size = len(data) // 3
chunks = [data[:split_size], data[split_size:2*split_size], data[2*split_size:]]

# ✅ Save each chunk as a new file
for i, chunk in enumerate(chunks, start=1):
    output_file = f"trainingdata{i}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in chunk:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Saved {output_file} with {len(chunk)} samples.")

print("🚀 Dataset successfully split into 3 files!")
