import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("faiss_index_flatIP.idx")

# Load influencer usernames (needed because FAISS only stores embeddings)
data = np.load("influencer_embeddings_checkpoint.npz", allow_pickle=True)
usernames = data["usernames"]

# Load SBERT model (use the same model you used for indexing)
model = SentenceTransformer('all-MiniLM-L12-v2')

# Example advertisement (query)
query_text = "Check out this amazing new skincare product that hydrates your skin all day!"
query_embedding = model.encode(query_text).astype(np.float32).reshape(1, -1)

# Search for the top 5 closest influencers
k = 3
distances, indices = index.search(query_embedding, k)

# Retrieve influencer names
best_influencers = [usernames[i] for i in indices[0]]

# Print results
print("Top matching influencers:")
for i, influencer in enumerate(best_influencers):
    print(f"{i+1}. {influencer} (Distance: {distances[0][i]})")
