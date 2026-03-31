# Conformer-CTC Speech-to-Text Engine

A fully functional, from-scratch Speech-to-Text (STT) engine built with PyTorch using a **Conformer-CTC** architecture. No pretrained models — everything is trained from raw audio and transcripts.

## Features

- **Conformer Encoder** — CNN subsampling + Conformer blocks (MHSA + Convolution Module + Feed-Forward)
- **CTC Decoder** — Greedy decoding and beam search with optional KenLM language model
- **SpecAugment** — Frequency and time masking for data augmentation
- **Character-level Tokenizer** — 29-token vocabulary (blank + a-z + apostrophe + space)
- **Mixed-Precision Training** — FP16 via PyTorch AMP for fast multi-GPU training
- **REST API** — FastAPI-based inference server with `/transcribe` endpoint
- **ONNX Export** — For optimized production deployment
- **Full Test Suite** — 36 unit tests covering all components

## Project Structure

```
conformer-stt/
├── config/
│   ├── model_config.yaml          # Medium model (16 layers, 256 dim, ~26M params)
│   ├── model_config_large.yaml    # Large model (18 layers, 512 dim, ~117M params)
│   ├── model_config_small.yaml    # Small model for testing (2 layers, 64 dim)
│   ├── train_config.yaml          # Training config (medium, 2x RTX 6000)
│   ├── train_config_large.yaml    # Training config (large model)
│   ├── train_config_small.yaml    # Quick test training config
│   └── inference_config.yaml      # Inference & API config
├── src/
│   ├── preprocessing/             # Audio loading, Mel features, tokenizer, SpecAugment, dataset
│   ├── model/                     # ConvSubsampling, Conformer blocks, encoder, CTC decoder
│   ├── training/                  # CTC loss, cosine warmup scheduler, trainer loop
│   ├── decoding/                  # Greedy & beam search (+ KenLM) decoders
│   ├── postprocessing/            # Punctuation, capitalization, text normalization
│   └── api/                       # FastAPI REST server
├── scripts/
│   ├── prepare_data.py            # Dataset manifest generation (LibriSpeech, Common Voice, dummy)
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # WER/CER evaluation on test sets
│   ├── transcribe.py              # Single-file transcription CLI
│   ├── export_model.py            # ONNX export
│   └── run_api.py                 # API server launcher
├── tests/                         # Unit tests (pytest)
├── data/                          # Audio data & manifests (not tracked, see below)
├── models/                        # Checkpoints & exports (not tracked)
├── notebooks/                     # Jupyter notebooks for experiments
├── requirements.txt
├── setup.py
└── README.md
```

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support (for GPU training)
- ~7 GB disk space for LibriSpeech `train-clean-100` + `dev-clean` + `test-clean`

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/conformer-stt.git
cd conformer-stt
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install PyTorch with CUDA (adjust cu124 to your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Prepare data

**Option A — LibriSpeech (recommended):**

Download from [openslr.org/12](https://www.openslr.org/12) and extract:
```
data/raw/LibriSpeech/
    train-clean-100/
    dev-clean/
    test-clean/
```

Then generate manifests:
```bash
python scripts/prepare_data.py --dataset librispeech --data_root data/raw/LibriSpeech --output data/manifests
```

**Option B — Quick test with dummy data:**
```bash
python scripts/prepare_data.py --dataset dummy --output data/manifests
```

### 3. Train

```bash
# Medium model (~26M params) on 2x RTX 6000
python scripts/train.py \
    --model_config config/model_config.yaml \
    --train_config config/train_config.yaml

# Large model (~117M params, Whisper-level) on 2x RTX 6000
python scripts/train.py \
    --model_config config/model_config_large.yaml \
    --train_config config/train_config_large.yaml

# Quick test with small model (CPU, minutes)
python scripts/train.py \
    --model_config config/model_config_small.yaml \
    --train_config config/train_config_small.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint models/checkpoints/best_model.pt \
    --model_config config/model_config.yaml \
    --test_manifest data/manifests/test_clean.json
```

### 5. Transcribe a file

```bash
python scripts/transcribe.py \
    --audio path/to/audio.wav \
    --checkpoint models/checkpoints/best_model.pt \
    --model_config config/model_config.yaml
```

### 6. Export to ONNX

```bash
python scripts/export_model.py \
    --checkpoint models/checkpoints/best_model.pt \
    --model_config config/model_config.yaml \
    --output models/exported/stt_model.onnx
```

### 7. Run REST API

```bash
python scripts/run_api.py \
    --model_config config/model_config.yaml

# Test with curl
curl -X POST http://localhost:8000/transcribe -F "audio=@audio.wav"
```

## Architecture

```
Audio (16 kHz WAV) 
  → Log-Mel Spectrogram (80 bins, 25ms window, 10ms hop)
  → Conv Subsampling (4× time reduction)
  → Conformer Blocks × 12
      ├── Half Feed-Forward (FFN/2)
      ├── Multi-Head Self-Attention (4 heads)
      ├── Convolution Module (depthwise, kernel 31)
      ├── Half Feed-Forward (FFN/2)
      └── Layer Normalization
  → CTC Projection (Linear → log-softmax)
  → Greedy / Beam Search Decoder
  → Text Post-Processing
```

**Model sizes:**

| Config | Layers | d_model | Heads | Parameters | Comparable to |
|--------|--------|---------|-------|------------|---------------|
| `model_config_small.yaml` | 2 | 64 | 4 | ~250K | Testing only |
| `model_config.yaml` | 16 | 256 | 4 | ~26M | Whisper Base (74M) |
| `model_config_large.yaml` | 18 | 512 | 8 | ~117M | Whisper Small (244M) |

**Whisper comparison (target WER on LibriSpeech test-clean):**

| Model | Parameters | Expected WER |
|-------|------------|-------------|
| Whisper Tiny | 39M | ~7.6% |
| Whisper Base | 74M | ~5.0% |
| **Our Medium** | **26M** | **~8-12%** |
| Whisper Small | 244M | ~3.4% |
| **Our Large** | **117M** | **~4-7%** |
| Whisper Medium | 769M | ~2.9% |
| Whisper Large v3 | 1.5B | ~2.0% |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=0.001, weight_decay=0.0001) |
| Scheduler | Cosine warmup (5000 warmup steps) |
| Augmentation | SpecAugment (2 freq masks × 27, 10 time masks × 5%) |
| Gradient clipping | Max norm 5.0 |
| Mixed precision | FP16 (automatic) |
| Metric | Word Error Rate (WER) via `jiwer` |

## Supported Datasets

| Dataset | Language | Size | Download |
|---------|----------|------|----------|
| LibriSpeech | English | 100h / 360h / 960h | [openslr.org/12](https://www.openslr.org/12) |
| Common Voice | 100+ languages | Varies | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/datasets) |
| Custom | Any | Any | JSON manifest format |

**Manifest format** (JSON lines):
```json
{"audio_path": "/path/to/audio.wav", "text": "the transcription", "duration": 3.5}
```

## Testing

```bash
pytest tests/ -v
```

All 36 tests covering tokenizer, feature extraction, augmentation, model architecture, decoding, and API endpoints.

## Hardware Recommendations

**Medium model (26M):**

| GPU | VRAM | Batch Size | Est. Time (100h data, 100 epochs) |
|-----|------|------------|----------------------------------|
| RTX 4070 | 12 GB | 16 | ~20h |
| RTX 4080 | 16 GB | 32 | ~12h |
| RTX 6000 | 48 GB | 64 | ~5h |
| 2× RTX 6000 | 96 GB | 128 | ~3h |

**Large model (117M):**

| GPU | VRAM | Batch Size | Est. Time (100h data, 120 epochs) |
|-----|------|------------|----------------------------------|
| RTX 4070 | 12 GB | 8 | ~60h |
| RTX 4080 | 16 GB | 16 | ~36h |
| RTX 6000 | 48 GB | 32 | ~14h |
| 2× RTX 6000 | 96 GB | 64 | ~8h |

## License

MIT
