"""
tokenization_utils.py

Utility functions for tokenizer setup and dataset tokenization for TinyStories experiments.
"""

from typing import Any, Dict, Tuple
from transformers import GPT2Tokenizer
from cache_utils import CacheManager


def setup_tokenizer() -> GPT2Tokenizer:
    """
    Load and configure the GPT2 tokenizer, adding a pad token if needed.
    Returns:
        The configured tokenizer.
    """
    print("Setting up the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Added padding token (same as EOS token)")
    return tokenizer


def tokenize_datasets(
    train_subset: Any,
    val_subset: Any,
    tokenizer: GPT2Tokenizer,
    dataset_cache_key: str,
    no_cache: bool,
    max_length: int,
    cache_manager: CacheManager,
) -> Tuple[Any, Any, str]:
    """
    Tokenize the training and validation datasets, with caching.

    Args:
        train_subset: Training dataset subset.
        val_subset: Validation dataset subset.
        tokenizer: The tokenizer to use.
        dataset_cache_key: Cache key for the dataset subset.
        no_cache: If True, disables caching.
        max_length: Maximum sequence length for tokenization.
        cache_manager: Instance of CacheManager for handling caching.
    Returns:
        (tokenized_train, tokenized_val, tokenization_cache_key)
    """
    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenization_params = {
        "dataset_cache_key": dataset_cache_key,
        "max_length": max_length,
        "tokenizer_name": "gpt2",
    }

    def compute_tokenization():
        print("Tokenizing the dataset (this might take a while)...")
        tokenized_train = train_subset.map(
            tokenize_function, batched=True, desc="Tokenizing training data"
        )
        tokenized_val = val_subset.map(
            tokenize_function, batched=True, desc="Tokenizing validation data"
        )
        return {"tokenized_train": tokenized_train, "tokenized_val": tokenized_val}

    cached_data, tokenization_cache_key = cache_manager.get_or_compute(
        tokenization_params, "tokenized_datasets", compute_tokenization, no_cache
    )

    return cached_data["tokenized_train"], cached_data["tokenized_val"], tokenization_cache_key
