import json
import os
from transformers import pipeline
from keybert import KeyBERT

def truncate_text(text, max_tokens=512):
    """Truncate text to max_tokens words (approximation)."""
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text

def load_checkpoint(checkpoint_path):
    """
    Load checkpoint data from disk (a list of processed JSON objects)
    and return that list.
    """
    processed = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed.append(json.loads(line))
                    except Exception as e:
                        print(f"Error parsing checkpoint line: {e}")
        print(f"Loaded checkpoint with {len(processed)} items")
    return processed

def save_checkpoint(processed, checkpoint_path):
    """
    Save checkpoint data to disk atomically.
    Writes the processed items as JSON Lines.
    """
    temp_file = checkpoint_path + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temp_file, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Initialize the Hugging Face pipeline with BigBird.
# "google/bigbird-roberta-base" is designed for long inputs (up to ~4096 tokens).
hf_model = pipeline(
    "feature-extraction", 
    model="google/bigbird-roberta-base", 
    tokenizer="google/bigbird-roberta-base",
    device=0  # Use GPU; set to -1 for CPU if needed.
)
global_kw_model = KeyBERT(model=hf_model)

def extract_important_tokens(text, max_tokens=512, top_n=200):
    """
    Extract key phrases from the input text using the pre-initialized KeyBERT model,
    split those phrases into tokens, and return a string of the top (up to) max_tokens tokens.
    """
    try:
        keywords = global_kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=top_n
        )
    except Exception as e:
        print(f"KeyBERT extraction error: {e}")
        raise e

    tokens = []
    for phrase, score in keywords:
        tokens.extend(phrase.split())
    tokens = tokens[:max_tokens]
    return " ".join(tokens)

def process_influencers(input_file, output_file, checkpoint_file, batch_size=100, max_tokens=512, checkpoint_interval=100):
    """
    Process each influencer entry by extracting important tokens from the combined Bio and Posts.
    Uses checkpointing so that if the process is interrupted, it resumes from the last processed entry.
    """
    processed = load_checkpoint(checkpoint_file)
    processed_count = len(processed)
    
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    
    total = len(lines)
    print(f"Total items to process: {total}")
    
    out_f = open(output_file, "a", encoding="utf-8")
    
    for i in range(processed_count, total):
        line = lines[i].strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception as e:
            print(f"Error parsing line {i}: {e}")
            continue
        
        username = data.get("Username", "").strip()
        bio = data.get("Bio", "")
        posts = data.get("Posts", "")
        combined_text = bio + "\n" + posts
        
        try:
            tokens_str = extract_important_tokens(combined_text, max_tokens=max_tokens, top_n=200)
            print(f"Keyword extraction for {username} successful")
        except Exception as e:
            print(f"Keyword extraction failed for {username}: {e}")
            tokens_str = truncate_text(combined_text, max_tokens)
        
        tokens_str = truncate_text(tokens_str, max_tokens)
        output_data = {"Username": username, "ImportantTokens": tokens_str}
        
        out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        processed.append(output_data)
        
        if (i + 1) % 25 == 0:
            save_checkpoint(processed, checkpoint_file)
            print(f"Processed {i+1}/{total} items (checkpoint saved).")

        
    out_f.close()
    save_checkpoint(processed, checkpoint_file)
    print(f"All influencer token extraction processed and saved to {output_file}")

if __name__ == "__main__":
    base_dir = "/home/johnkimm/InstaAIModel/SBERT_Indexing"
    input_file = os.path.join(base_dir, "influencer_recommendation_data.jsonl")
    output_file = os.path.join(base_dir, "influencer_keywords_bigbird.jsonl")
    checkpoint_file = os.path.join(base_dir, "keywords_bigbird_checkpoint.jsonl")
    
    process_influencers(
        input_file, 
        output_file, 
        checkpoint_file, 
        batch_size=100, 
        max_tokens=512, 
        checkpoint_interval=100
    )
