"""Upload a trained model to HuggingFace Hub.

Usage:
    python scripts/push_to_hub.py --checkpoint models/checkpoints/best_model.pt --repo_id your-username/conformer-stt-medium
    python scripts/push_to_hub.py --checkpoint models/checkpoints/best_model.pt --repo_id your-username/conformer-stt-large --private
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.tokenizer import CharTokenizer
from src.model.model import STTModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("push_to_hub")


def main():
    parser = argparse.ArgumentParser(description="Push trained model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (e.g. username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    # Load config
    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = CharTokenizer(lowercase=True)
    config.setdefault("model", {}).setdefault("decoder", {})["vocab_size"] = len(tokenizer)

    # Load model to verify it works
    model = STTModel.from_config(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Model loaded: {model.count_parameters():,} parameters")

    # Prepare upload directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Save model weights (just state_dict, smaller)
        weights_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), weights_path)
        logger.info(f"Saved weights: {weights_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Save config
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save tokenizer vocab
        vocab_path = tmp_path / "vocab.json"
        vocab = {char: idx for idx, char in enumerate(tokenizer.get_vocab_list())}
        with open(vocab_path, "w") as f:
            json.dump(vocab, f, indent=2)

        # Save model card
        enc = config["model"]["encoder"]
        model_card = f"""---
language: en
license: mit
library_name: pytorch
tags:
  - speech
  - stt
  - asr
  - conformer
  - ctc
---

# Conformer-CTC Speech-to-Text

A from-scratch Conformer-CTC model for English speech recognition.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Conformer-CTC |
| Parameters | {model.count_parameters():,} |
| Encoder Layers | {enc['num_layers']} |
| d_model | {enc['d_model']} |
| Attention Heads | {enc['num_heads']} |
| Vocab Size | {len(tokenizer)} (character-level) |
| Audio | 16kHz, 80-dim log-mel spectrogram |

## Usage

```python
from conformer_stt import ConformerSTT

model = ConformerSTT.from_pretrained("{args.repo_id}")
text = model.transcribe("audio.wav")
print(text)
```

## Quick Start (manual)

```python
import torch
import yaml
from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.tokenizer import CharTokenizer
from src.model.model import STTModel
from src.decoding.greedy import GreedyDecoder

# Load
config = yaml.safe_load(open("config.yaml"))
model = STTModel.from_config(config)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# Transcribe
loader = AudioLoader(target_sample_rate=16000)
extractor = FeatureExtractor.from_config(config)
tokenizer = CharTokenizer()
decoder = GreedyDecoder(tokenizer)

waveform, sr = loader.load("audio.wav")
features = extractor.extract(waveform).unsqueeze(0)
lengths = torch.tensor([features.shape[2]])

with torch.no_grad():
    log_probs, out_lengths = model(features, lengths)
    text = decoder.decode(log_probs, out_lengths)[0]
print(text)
```
"""
        readme_path = tmp_path / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)

        # Create repo and upload
        api = HfApi()
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        logger.info(f"Uploading to {args.repo_id}...")

        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=args.repo_id,
            commit_message=f"Upload Conformer-CTC model ({model.count_parameters():,} params)",
        )

        logger.info(f"Done! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
