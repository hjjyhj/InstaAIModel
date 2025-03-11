import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_data(json_path):
    """Load JSON data from a file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_checkpoint(data, checkpoint_path):
    """
    Save checkpoint data to disk atomically.
    This writes to a temporary file first and then renames it to ensure the checkpoint file remains valid.
    """
    temp_file = checkpoint_path + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint data from disk, if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded checkpoint with {len(data)} items")
        return data
    return []

def process_batch(data, start_index, batch_size, model):
    """Process a batch of data, embedding the 'Info' field."""
    batch_results = []
    end_index = min(start_index + batch_size, len(data))
    for i in range(start_index, end_index):
        entry = data[i]
        username = entry.get("Username", "").strip()
        category = entry.get("Category", "").strip()
        info_text = entry.get("Info", "")
        embedding = model.encode(info_text)
        batch_results.append({
            "Username": username,
            "Category": category,
            "embedding": embedding.tolist()
        })
    return batch_results, end_index

#def initialize_faiss_index(embedding_dim):
#     """Initialize a FAISS index to store embeddings."""
#     index = faiss.IndexFlatL2(embedding_dim)
#     return index

# def save_faiss_index(index, index_file):
#     """Save the FAISS index to disk."""
#     faiss.write_index(index, index_file)
#     print(f"FAISS index saved to {index_file}")

# def load_faiss_index(index_file):
#     """Load an existing FAISS index."""
#     if os.path.exists(index_file):
#         index = faiss.read_index(index_file)
#         print(f"Loaded FAISS index from {index_file}")
#         return index
#     return None

def main():
    # File paths
    base_dir = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/influencer_recommendation"
    input_json = os.path.join(base_dir, "combined.json")
    checkpoint_file = os.path.join(base_dir, "embedding_checkpoint.json")
    output_file = os.path.join(base_dir, "info_embeddings.json")
    #faiss_index_file = os.path.join(base_dir, "faiss_index.index")
    
    # Load full data
    data = load_data(input_json)
    total = len(data)
    print(f"Total items to process: {total}")

    # Initialize SBERT model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Load existing checkpoint if available
    processed = load_checkpoint(checkpoint_file)  # List of processed entries
    processed_count = len(processed)
    
    batch_size = 100  # Adjust based on available GPU time/memory
    current_index = processed_count  # Start where we left off

    # Initialize FAISS index
    # embedding_dim = 768  # For "all-mpnet-base-v2", embedding dimension 768
    # faiss_index = load_faiss_index(faiss_index_file)
    # if faiss_index is None:
    #     faiss_index = initialize_faiss_index(embedding_dim)

    while current_index < total:
        print(f"Processing items {current_index} to {min(current_index + batch_size, total)}")
        batch_results, current_index = process_batch(data, current_index, batch_size, model)
        # Extract embeddings and add them to FAISS index
        # embeddings = [entry["embedding"] for entry in batch_results]
        # embeddings = np.array(embeddings, dtype='float32')
        # faiss_index.add(embeddings)
        # Append new batch results to processed list
        processed.extend(batch_results)
        # Save checkpoint after each batch atomically
        save_checkpoint(processed, checkpoint_file)
        # (Optional) Save final output incrementally
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
    # Save the FAISS index
    # save_faiss_index(faiss_index, faiss_index_file)
    print(f"All embeddings processed and saved to {output_file}")

if __name__ == "__main__":
    main()
