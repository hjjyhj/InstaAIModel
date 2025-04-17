from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L12-v2')

# File paths
input_file = "keywords_bigbird.jsonl"
checkpoint_file = "influencer_embeddings_checkpoint.npz"

# Load checkpoint if available
if os.path.exists(checkpoint_file):
    print("Loading checkpoint...")
    data = np.load(checkpoint_file, allow_pickle=True)
    usernames = list(data["usernames"])
    embeddings = list(data["embeddings"])
    print(f"Resuming from {len(usernames)} influencers...")
else:
    usernames = []
    embeddings = []

# Track already processed usernames to avoid duplicates
processed_set = set(usernames)

# Read JSONL file and process new influencers
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        username = data["Username"]

        # Skip if already processed
        if username in processed_set:
            continue

        important_tokens = data["ImportantTokens"]

        # Convert to embedding
        embedding = model.encode(important_tokens)

        # Store results
        usernames.append(username)
        embeddings.append(embedding)
        processed_set.add(username)

        # Save checkpoint every 100 influencers
        if len(usernames) % 100 == 0:
            np.savez(checkpoint_file, usernames=usernames, embeddings=embeddings)
            print(f"Checkpoint saved at {len(usernames)} influencers")

# Final save
np.savez(checkpoint_file, usernames=usernames, embeddings=embeddings)
print(f"Final checkpoint saved: {len(usernames)} influencers processed.")
