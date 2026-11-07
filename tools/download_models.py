"""Explicit network step; runtime CLI itself is local/offline by default."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "qwen": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "Qwen3-VL-2B-Instruct"),
    "qwen8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "Qwen3-VL-8B-Instruct"),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=MODELS)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    repo, revision, name = MODELS[args.model]
    dest = snapshot_download(repo, revision=revision, local_dir=root/"models"/name,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    (root/"artifacts").mkdir(exist_ok=True)
    (root/"artifacts"/f"{args.model}_model_revision.json").write_text(json.dumps(dict(repo=repo, revision=revision), indent=2))
    print(dest)
