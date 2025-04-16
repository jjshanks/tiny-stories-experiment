# TinyStories Experiment

A simple script for training a small GPT-2 style transformer model on a subset of the TinyStories dataset using HuggingFace Transformers and Datasets.

## Features
- Trains a small GPT-2 style transformer model on TinyStories
- Supports dataset filtering, caching, and checkpointing
- Easily configurable via command-line arguments
- Saves checkpoints and generated outputs to the `results/` directory

## Requirements
- Python 3.8+
- torch >= 2.0.0
- transformers >= 4.35.0
- datasets >= 2.16.0
- pyyaml >= 6.0

Install dependencies:
```bash
python -m venv tiny-story-experiments
source tiny-story-experiments/bin/activate
pip install -r requirements.txt
```

## Usage
Run training with default settings:
```bash
python train_tiny_stories.py
```

### Common Options
- `--num_examples` — Number of training samples (default: 10000)
- `--epochs` — Number of training epochs (default: 1)
- `--filter_word` — Only use stories containing this word
- `--prompt_word` — Use this word in the generation prompt

All available options can be viewed with:
```bash
python train_tiny_stories.py --help
```

Checkpoints and generated outputs are saved in the `results/` directory.

For more details and advanced usage, see comments in `train_tiny_stories.py`. The script is self-documented for further reference.
