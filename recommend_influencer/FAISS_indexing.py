import faiss
import numpy as np

# Load influencer embeddings and usernames
data = np.load("influencer_embeddings_checkpoint.npz", allow_pickle=True)
usernames = data["usernames"]
embeddings = data["embeddings"]

# Convert embeddings to float32 (FAISS requirement)
embeddings = np.array(embeddings, dtype=np.float32)

# Create a FAISS index (using L2 distance)
# index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (Euclidean)
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (Cosine similarity alternative)

# Add embeddings to FAISS
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, "faiss_index_flatIP.idx")
print(f"FAISS index saved with {index.ntotal} embeddings")
