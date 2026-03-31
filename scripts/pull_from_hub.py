"""Download a model from HuggingFace Hub and transcribe audio.

Usage:
    python scripts/pull_from_hub.py --repo_id your-username/conformer-stt-medium --audio path/to/audio.wav
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.tokenizer import CharTokenizer
from src.model.model import STTModel
from src.decoding.greedy import GreedyDecoder
from src.postprocessing.normalization import TextNormalizer
from src.postprocessing.capitalization import TrueCase
from src.postprocessing.punctuation import PunctuationRestorer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pull_from_hub")


def download_model(repo_id: str, cache_dir: str = "models/hub") -> Path:
    """Download model files from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    local_dir = Path(cache_dir) / repo_id.replace("/", "--")
    if local_dir.exists() and (local_dir / "model.pt").exists():
        logger.info(f"Using cached model from {local_dir}")
        return local_dir

    logger.info(f"Downloading {repo_id} from HuggingFace...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    logger.info(f"Downloaded to {local_dir}")
    return local_dir


def load_model(model_dir: Path, device: str = "cpu"):
    """Load model from a downloaded directory."""
    config_path = model_dir / "config.yaml"
    weights_path = model_dir / "model.pt"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = CharTokenizer(lowercase=True)
    config.setdefault("model", {}).setdefault("decoder", {})["vocab_size"] = len(tokenizer)

    model = STTModel.from_config(config)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, config, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Download model and transcribe")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo (e.g. username/conformer-stt)")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache_dir", type=str, default="models/hub")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Download
    model_dir = download_model(args.repo_id, args.cache_dir)

    # Load
    model, config, tokenizer = load_model(model_dir, device)
    logger.info(f"Model loaded: {model.count_parameters():,} params on {device}")

    # Transcribe
    audio_loader = AudioLoader(target_sample_rate=16000)
    feature_extractor = FeatureExtractor.from_config(config)
    decoder = GreedyDecoder(tokenizer)
    normalizer = TextNormalizer()
    capitalizer = TrueCase()
    punctuator = PunctuationRestorer()

    import time
    start = time.time()

    waveform, sr = audio_loader.load(args.audio)
    duration = audio_loader.get_duration(waveform)
    features = feature_extractor.extract(waveform).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[2]], dtype=torch.long).to(device)

    with torch.no_grad():
        log_probs, out_lengths = model(features, lengths)
    raw_text = decoder.decode(log_probs, out_lengths)[0]

    text = normalizer.normalize(raw_text)
    text = punctuator.restore(text)
    text = capitalizer.apply(text)

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Model:     {args.repo_id}")
    print(f"Audio:     {args.audio}")
    print(f"Duration:  {duration:.2f}s")
    print(f"Time:      {elapsed:.3f}s (RTF: {elapsed / duration:.2f})")
    print(f"Text:      {text}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
