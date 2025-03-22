from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path("/home/johnkimm/InstaAIModel/Mistral_7B-Instruct-v0.3")
mistral_models_path.mkdir(parents=True, exist_ok=True)

# Download all necessary model files, including config.json
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir=mistral_models_path,
    allow_patterns=["config.json", "params.json", "consolidated.safetensors", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
)
