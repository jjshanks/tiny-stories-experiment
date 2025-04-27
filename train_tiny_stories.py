"""
TinyStories LLM Training Script

Requirements:
- Python 3.8+
- torch >= 2.0.0
- transformers >= 4.35.0
- datasets >= 2.16.0
- pyyaml >= 6.0

Caching & Checkpointing:
=======================
- This script uses disk caching to speed up repeated runs and avoid recomputation of dataset filtering, tokenization, and model training.
- Caching is controlled by the --no_cache flag (disable all caching) and --skip_training (skip training if a cached model exists).
- Model checkpoints are saved at the end of each epoch (default), and can be controlled with --save_each_epoch/--no_save_each_epoch.
- Training can resume from the latest checkpoint with --continue_training (default) or start fresh with --no_continue_training.
- All cache and checkpoint files are stored in ./cache and ./results/epoch_checkpoints_*/ by default, but can be changed with --cache_dir.
- To force recomputation, use --no_cache or delete the cache/checkpoint directories.

A proof-of-concept implementation for training a small language model (LLM) on a laptop
using a subset of the TinyStories dataset.

How to experiment
==================
- Change --num_examples to use more or fewer training samples (e.g. 1000, 5000, 20000).
- Adjust --dimensions and --layers to make the model smaller or larger (e.g. 128/256/512 dims, 2/4/6/8 layers).
- Use --filter_word to train on stories containing a specific word (e.g. --filter_word dragon).
- Try --prompt_word to generate stories about a specific topic.
- Use --epochs to train for more epochs (but beware of overfitting on tiny datasets).
- Use --no_cache to force recomputation if you change code or want to see effects of parameter changes.
- Check ./results/ for checkpoints and generated outputs.
- See inline comments for ML concepts and tips!

Tips:
- Start with small models and datasets for fast iteration.
- Use GPU if available (automatically detected).
- Inspect generated stories for coherence and creativity.
- Read the code comments for explanations of transformers, attention, and embeddings.
- For more, see: Vaswani et al. "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
- HuggingFace Transformers docs: https://huggingface.co/docs/transformers/index
- TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Example transformer block (ASCII):

    Input Embedding
         |
    +----v-----+
    | Multi-   |
    | Head     |  <--- Attention: Each token attends to all others
    | Attention|
    +----v-----+
         |
    +----v-----+
    | Feed     |
    | Forward  |
    +----v-----+
         |
    Output Embedding

"""

import os
import torch
import argparse
import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
import yaml
from typing import Any, Dict, Optional, Tuple
import sys
import time

# Import the new CacheManager
from cache_utils import CacheManager
from dataset_utils import load_tinystories_dataset, create_dataset_subsets
from tokenization_utils import setup_tokenizer, tokenize_datasets
from model_utils import setup_model_and_config

# =====================
# Prompt Selection Logic
# =====================

def determine_generation_prompt(args: argparse.Namespace) -> str:
    """
    Determines the text generation prompt based on command-line arguments.

    Prioritizes arguments in the order: --prompt_text, --prompt_word, --filter_word.
    Falls back to a default prompt if none are provided.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The selected prompt string.
    """
    if args.prompt_text:
        print(
            f"Using full prompt text: '{args.prompt_text}' (overriding prompt/filter word)"
        )
        return args.prompt_text
    elif args.prompt_word:
        print(
            f"Using '{args.prompt_word}' for generation prompt (overriding filter word)"
        )
        return f"Once upon a time, there was a {args.prompt_word}"
    elif args.filter_word:
        print(f"Using filter word '{args.filter_word}' for generation prompt")
        return f"Once upon a time, there was a {args.filter_word}"
    else:
        print("Using default generation prompt")
        return "Once upon a time, there was a little"

# =====================
# Model utilities
# =====================


# =====================
# Main script
# =====================

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a small LLM on TinyStories dataset")
parser.add_argument(
    "--num_examples",
    type=int,
    default=10000,
    help="Number of examples to use for training (default: 10000)",
)
parser.add_argument(
    "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
)
parser.add_argument(
    "--save_each_epoch",
    action="store_true",
    default=True,
    help="Save model checkpoint at each epoch (default: True)",
)
parser.add_argument(
    "--no_save_each_epoch",
    action="store_false",
    dest="save_each_epoch",
    help="Disable saving model checkpoint at each epoch",
)
parser.add_argument(
    "--continue_training",
    action="store_true",
    default=True,
    help="Continue training from the latest available epoch checkpoint (default: True)",
)
parser.add_argument(
    "--no_continue_training",
    action="store_false",
    dest="continue_training",
    help="Disable continuing from previous checkpoints",
)
parser.add_argument(
    "--dimensions",
    type=int,
    default=256,
    help="Embedding dimension size (default: 256)",
)
parser.add_argument(
    "--layers", type=int, default=6, help="Number of transformer layers (default: 6)"
)
parser.add_argument(
    "--heads", type=int, default=8, help="Number of attention heads (default: 8)"
)
parser.add_argument(
    "--filter_word",
    type=str,
    default=None,
    help="Filter dataset to include only examples containing this word",
)
parser.add_argument(
    "--prompt_word",
    type=str,
    default=None,
    help="Word to use in generation prompt (e.g., 'dragon'). Overridden by --prompt_text.",
)
parser.add_argument(
    "--prompt_text",
    type=str,
    default=None,
    help="Full text prompt to use for generation (e.g., 'Once upon a time in a land far away'). Overrides --prompt_word.",
)
parser.add_argument(
    "--no_cache",
    action="store_true",
    help="Disable caching and force recomputation of all steps",
)
parser.add_argument(
    "--skip_training",
    action="store_true",
    help="Skip training and use cached model if available (implies caching is enabled)",
)
# New hyperparameters
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Learning rate for optimizer (default: 5e-5)",
)
parser.add_argument(
    "--batch_size", type=int, default=4, help="Batch size per device (default: 4)"
)
parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    help="Max sequence length for tokenization and model (default: 512)",
)
parser.add_argument(
    "--gen_max_length",
    type=int,
    default=128,
    help="Max sequence length for text generation (default: 128)",
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    choices=["cpu", "cuda", "mps", None],
    help="Device to use for training/generation (cpu, cuda, mps, or auto)",
)
parser.add_argument(
    "--fp16",
    dest="fp16",
    action="store_true",
    default=None,
    help="Use mixed precision (fp16) if available (default: auto)",
)
parser.add_argument(
    "--no_fp16",
    dest="fp16",
    action="store_false",
    help="Disable mixed precision (fp16)",
)
parser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default="linear",
    help="Learning rate scheduler type (default: linear)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for generation (default: 0.7)",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.9,
    help="Top-p (nucleus) sampling for generation (default: 0.9)",
)
parser.add_argument(
    "--do_sample",
    dest="do_sample",
    action="store_true",
    default=True,
    help="Use sampling for generation (default: True)",
)
parser.add_argument(
    "--no_sample",
    dest="do_sample",
    action="store_false",
    help="Disable sampling for generation (use greedy decoding)",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default="./cache",
    help="Directory to use for caching intermediate results (default: ./cache)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (default: None, random behavior)",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=0.0,
    help="Ratio of total training steps used for linear warmup from 0 to learning_rate (default: 0.0)",
)

args = parser.parse_args()

# --- New Validation Step ---
# Validate that dimensions is divisible by heads
if args.dimensions % args.heads != 0:
    print(
        f"\\nERROR: Embedding dimensions ({args.dimensions}) must be divisible by the number of attention heads ({args.heads})."
    )
    print("Tip: Adjust --dimensions or --heads.")
    sys.exit(1)
# --- End Validation Step ---


# --- New Validation Step for warmup_ratio ---
if not (0.0 <= args.warmup_ratio <= 1.0):
    print(
        f"\\nERROR: warmup_ratio ({args.warmup_ratio}) must be between 0.0 and 1.0 (inclusive)."
    )
    print("Tip: Adjust --warmup_ratio to a value in this range.")
    sys.exit(1)
# --- End Validation Step ---


# Device and precision selection
if args.device:
    device = args.device
else:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

if device == "cpu":
    print(
        "WARNING: You are running on CPU. Training will be much slower.\nTip: If you have a GPU, install CUDA and PyTorch with GPU support. See: https://pytorch.org/get-started/locally/"
    )

if args.device == "cuda" and not torch.cuda.is_available():
    print(
        "WARNING: --device cuda specified but CUDA is not available. Falling back to CPU.\nTip: Check your CUDA installation and PyTorch build."
    )
    device = "cpu"

print(f"Using device: {device}")
if args.fp16 is None:
    fp16 = torch.cuda.is_available() and device == "cuda"
else:
    fp16 = args.fp16
print(f"Mixed precision (fp16): {fp16}")

# Print all key parameters for reproducibility
print("\n===== Training Configuration =====")
for k, v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print(f"device: {device}")
print(f"fp16: {fp16}")
print("=================================\n")

# Set up cache directory from argument
CACHE_DIR = Path(args.cache_dir)
# Instantiate CacheManager
cache_manager = CacheManager(CACHE_DIR)

# Step 1: Load the TinyStories dataset
try:
    tinystories = load_tinystories_dataset()
except Exception as e:
    print("\nERROR: Failed to load the TinyStories dataset.")
    print(
        "This may be due to network issues, missing dependencies, or HuggingFace Datasets not being installed."
    )
    print("Tip: Try running 'pip install datasets' and check your internet connection.")
    print(f"Exception: {e}")
    sys.exit(1)

# Step 2: Create dataset subsets (with caching)
try:
    train_subset, val_subset, dataset_cache_key = create_dataset_subsets(
        tinystories, args.num_examples, args.filter_word, args.no_cache, cache_manager # Pass cache_manager
    )
except Exception as e:
    print("\nERROR: Failed to create dataset subsets.")
    print(
        "This may be due to dataset format changes, filtering errors, or insufficient examples."
    )
    print("Tip: Try reducing --num_examples or removing --filter_word.")
    print(f"Exception: {e}")
    sys.exit(1)

# Step 3: Initialize the tokenizer
try:
    tokenizer = setup_tokenizer()
except Exception as e:
    print("\nERROR: Failed to load or configure the GPT2 tokenizer.")
    print("This may be due to missing model files or internet issues.")
    print(
        "Tip: Try running 'pip install transformers' and check your internet connection."
    )
    print(f"Exception: {e}")
    sys.exit(1)

# Step 4: Apply tokenization to the dataset (with caching)
try:
    tokenized_train, tokenized_val, tokenization_cache_key = tokenize_datasets(
        train_subset,
        val_subset,
        tokenizer,
        dataset_cache_key,
        args.no_cache,
        args.max_length, # Use args.max_length here
        cache_manager, # Pass cache_manager
    )
except Exception as e:
    print("\nERROR: Failed to tokenize the dataset.")
    print("This may be due to tokenizer configuration or dataset format issues.")
    print(
        "Tip: Check that the dataset has a 'text' field and the tokenizer is loaded correctly."
    )
    print(f"Exception: {e}")
    sys.exit(1)

# Step 5: Set up a small transformer model configuration
try:
    config, model_params = setup_model_and_config(
        tokenizer, args.dimensions, args.layers, args.heads, args.max_length
    )
except Exception as e:
    print("\\nERROR: Failed to set up the model configuration.")
    print("This may be due to invalid model parameters or tokenizer issues.")
    print("Tip: Try reducing --dimensions or --layers, or check the tokenizer setup.")
    print(f"Exception: {e}")
    sys.exit(1)

# Step 6: Initialize the model with our configuration
# Set up the path for epoch checkpoints
epoch_checkpoint_params = {
    "dataset_cache_key": dataset_cache_key,
    "tokenization_cache_key": tokenization_cache_key,
    "model_config": {
        "n_embd": args.dimensions,
        "n_layer": args.layers,
        "n_head": args.heads,
    },
    "filter_word": args.filter_word,
}
# Use cache_manager method
epoch_checkpoint_key = cache_manager.get_cache_key(epoch_checkpoint_params)
epoch_save_path = f"./results/epoch_checkpoints_{epoch_checkpoint_key}"
os.makedirs(epoch_save_path, exist_ok=True)

# Check if we should continue training from a previous checkpoint
start_epoch = 0
skip_training_due_to_checkpoint = False  # NEW FLAG
if args.continue_training:
    # Check for existing epoch checkpoints
    existing_checkpoints = []
    for dir_name in os.listdir(epoch_save_path):
        if dir_name.startswith("epoch-"):
            try:
                epoch_num = int(dir_name.split("-")[1])
                existing_checkpoints.append(epoch_num)
            except (IndexError, ValueError):
                continue

    if existing_checkpoints:
        # Sort checkpoints and find the highest one
        existing_checkpoints.sort()
        latest_epoch = existing_checkpoints[-1]

        # Always check if the exact target epoch checkpoint exists first
        target_epoch = args.epochs
        if target_epoch in existing_checkpoints:
            # If we have already trained up to the requested epoch, just load that checkpoint
            checkpoint_path = os.path.join(epoch_save_path, f"epoch-{target_epoch}")
            print(
                f"Found checkpoint for exact target epoch {target_epoch}, loading it directly"
            )
            model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
            start_epoch = target_epoch
            print(f"Starting from epoch {start_epoch}, effective epochs to train: 0")
            skip_training_due_to_checkpoint = True  # SET FLAG
            print(f"Model loaded from checkpoint: {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")
        # If requested epochs is greater than what we have, load the latest checkpoint
        elif args.epochs > latest_epoch:
            checkpoint_path = os.path.join(epoch_save_path, f"epoch-{latest_epoch}")
            print(f"Continuing training from epoch {latest_epoch} checkpoint")
            model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
            start_epoch = latest_epoch
        # If requested epochs < what we have, find the nearest lower epoch checkpoint
        elif args.epochs < latest_epoch:
            # Find the highest epoch checkpoint that is less than or equal to the target
            available_checkpoints = [
                ep for ep in existing_checkpoints if ep <= args.epochs
            ]
            if available_checkpoints:
                target_epoch = max(available_checkpoints)
                checkpoint_path = os.path.join(epoch_save_path, f"epoch-{target_epoch}")
                print(
                    f"Loading epoch {target_epoch} checkpoint (to train for {args.epochs - target_epoch} more epochs)"
                )
                model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
                start_epoch = target_epoch
            else:
                print(
                    f"Warning: No suitable checkpoint found for epoch <= {args.epochs}, starting from scratch"
                )
                model = GPT2LMHeadModel(config)
                print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")
        else:
            model = GPT2LMHeadModel(config)
            print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")
    else:
        print("No existing checkpoints found, initializing new model")
        model = GPT2LMHeadModel(config)
        print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")
else:
    # Initialize a new model
    model = GPT2LMHeadModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")

# Step 7: Configure training parameters
print("Setting up training configuration...")
effective_epochs = args.epochs - start_epoch

if skip_training_due_to_checkpoint:
    effective_epochs = 0

training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints
    do_eval=True,  # Enable evaluation during training
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save at the end of each epoch
    save_total_limit=args.epochs,  # Keep one checkpoint per epoch
    learning_rate=args.learning_rate,  # Learning rate for optimizer
    weight_decay=0.01,  # Regularization technique to prevent overfitting by penalizing large weights.
    per_device_train_batch_size=args.batch_size,  # Batch size per device
    per_device_eval_batch_size=args.batch_size,  # Evaluation batch size
    num_train_epochs=effective_epochs,  # Train for the remaining epochs
    logging_steps=100,  # Log metrics every 100 steps
    fp16=fp16,  # Use mixed precision if specified or GPU available
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps before updating model weights. Increases effective batch size without using more memory (effective batch size = batch_size * accumulation_steps).
    report_to="none",  # Don't report to external services
    load_best_model_at_end=True,  # Load the best model (lowest eval loss) found during training at the end.
    metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
    greater_is_better=False,  # Lower loss is better
    lr_scheduler_type=args.lr_scheduler_type,  # How the learning rate changes over time (e.g., 'linear' decay).
    warmup_ratio=args.warmup_ratio, # Ratio of total steps for linear warmup
)

# Step 8: Create data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling (next token prediction), not masked language modeling
)


# Create a custom callback to save model at the end of each epoch
class SaveEpochCallback(TrainerCallback):
    """
    Custom callback to save model checkpoints at the end of each epoch.

    This allows us to track the model's progress over time and potentially
    resume training from any epoch checkpoint.
    """

    def __init__(self, save_path, start_epoch=0):
        self.save_path = save_path
        self.start_epoch = start_epoch
        self.saved_epochs = set()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        current_epoch = int(state.epoch)
        actual_epoch = self.start_epoch + current_epoch
        if actual_epoch in self.saved_epochs:
            return control
        epoch_dir = os.path.join(self.save_path, f"epoch-{actual_epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        print(f"Model saved at epoch {actual_epoch} to {epoch_dir}")
        self.saved_epochs.add(actual_epoch)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if state.epoch is not None:
            try:
                total_epochs = int(
                    getattr(args, "num_train_epochs", None)
                    or getattr(args, "epochs", None)
                    or 0
                )
            except Exception:
                total_epochs = None
            if total_epochs:
                final_epoch = self.start_epoch + total_epochs
            else:
                import math

                final_epoch = self.start_epoch + math.ceil(state.epoch)
            if final_epoch not in self.saved_epochs:
                epoch_dir = os.path.join(self.save_path, f"epoch-{final_epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                model.save_pretrained(epoch_dir)
                print(f"Model saved at final epoch {final_epoch} to {epoch_dir}")
                self.saved_epochs.add(final_epoch)
        return control


# Step 9: Set up the trainer
print("Initializing the Trainer...")
callbacks = []
if args.save_each_epoch:
    callbacks.append(
        SaveEpochCallback(save_path=epoch_save_path, start_epoch=start_epoch)
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    callbacks=callbacks,
)

# Step 10: Start training (with caching)
should_use_cache = not args.no_cache and not args.continue_training

training_params = {
    "dataset_cache_key": dataset_cache_key,
    "tokenization_cache_key": tokenization_cache_key,
    "model_config": {
        "n_embd": args.dimensions,
        "n_layer": args.layers,
        "n_head": args.heads,
    },
    "training_args": {
        "learning_rate": args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio, # Add warmup ratio here
    },
}
# Use cache_manager method
training_cache_key = cache_manager.get_cache_key(training_params)
# Use the cache_manager's cache_dir attribute
model_cache_dir = cache_manager.cache_dir / f"model_{training_cache_key}"

if (should_use_cache or args.skip_training) and model_cache_dir.exists():
    print(f"Loading trained model from cache: {model_cache_dir}")
    model = GPT2LMHeadModel.from_pretrained(model_cache_dir)
    print("Cached model loaded successfully!")
    if args.skip_training and not model_cache_dir.exists():
        print(
            "Warning: --skip_training specified but no cached model found. Will train a new model."
        )
else:
    print("Starting model training...")
    if effective_epochs > 0:
        trainer.train()
        print("Training complete!")
    else:
        print(
            f"No additional training needed (requested epochs {args.epochs} <= already trained epochs {start_epoch})"
        )
    if not args.no_cache and not args.continue_training:
        print(f"Saving trained model to cache: {model_cache_dir}")
        trainer.save_model(model_cache_dir)
        print("Model cached successfully!")

# Step 11: Save the final model
final_model_path = "./results/final-model"
if not (not args.no_cache and model_cache_dir.exists()):
    trainer.save_model(final_model_path)
else:
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)

tokenizer.save_pretrained(final_model_path)
print(f"Model and tokenizer saved to {final_model_path}")

# Step 12: Test the model with text generation
print("\\nTesting the model with text generation:")

# Determine the prompt using the new function
prompt = determine_generation_prompt(args) # Use the extracted function

print(f"Prompt: '{prompt}'")
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs = inputs.to(device)
model = model.to(device)

if args.seed is not None:
    torch.manual_seed(args.seed)
else:
    # Generate a random seed if none is provided for slightly different outputs each run
    torch.manual_seed(int.from_bytes(os.urandom(8), byteorder="big"))

"""--- Text Generation Parameters ---
These parameters control how the model generates new text:
* temperature: Controls the randomness of the output.
    - Lower values (~0.2) make the output more focused and deterministic.
    - Higher values (~1.0+) make the output more random and creative.
* do_sample: If True, use sampling strategies (like temperature and top_p).
    If False, use greedy decoding (always pick the most likely next token).
* top_p (nucleus sampling): Considers only the most probable tokens whose
    cumulative probability mass exceeds p. Narrows down the choices, often
    improving quality over pure temperature sampling.
* no_repeat_ngram_size: Prevents the model from repeating the exact same
    sequence of N words (e.g., N=2 prevents repeating bigrams).
"""
output = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.pad_token_id,
    max_length=args.gen_max_length,
    num_return_sequences=1,
    temperature=args.temperature,
    do_sample=args.do_sample,
    top_p=args.top_p,
    no_repeat_ngram_size=2,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated story:")
print(generated_text)
print("\nDone!")

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
generation_output_path = f"./results/output_{now}.yaml"
yaml_output = {"params": vars(args), "prompt": prompt, "output": generated_text}
with open(generation_output_path, "w") as f:
    yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
print(f"Generation output saved to {generation_output_path}")

if __name__ == "__main__":
    print(
        f"Training with {len(train_subset)} examples, total requested epochs: {args.epochs}"
    )
    print(f"Model configuration: {args.dimensions} dimensions, {args.layers} layers, {args.heads} heads")
    print(f"Learning rate: {args.learning_rate}, Warmup ratio: {args.warmup_ratio}, LR Scheduler: {args.lr_scheduler_type}") # Added warmup_ratio here
    if args.filter_word:
        print(
            f"Dataset filtered to include only examples with the word: '{args.filter_word}'"
        )
    if args.continue_training:
        if start_epoch > 0:
            print(
                f"Continuing training from epoch {start_epoch} to epoch {args.epochs}"
            )
            print(f"Training for {effective_epochs} additional epochs")
        else:
            print(
                "Continue training enabled but no usable checkpoints found, starting from scratch"
            )
    if args.save_each_epoch:
        print(f"Saving model checkpoint at each epoch in ./results/epoch_checkpoints/")
    if args.skip_training:
        print("Skip training mode: Will use cached model if available")
    elif not args.no_cache:
        print("Caching enabled: Steps with unchanged inputs will use cached results")
    else:
        print("Caching disabled: All computation steps will be performed")
