"""
dataset_utils.py

Utility functions for dataset loading, filtering, and splitting for TinyStories experiments.
"""

from typing import Any, Optional, Tuple
from cache_utils import CacheManager

def load_tinystories_dataset() -> Any:
    """
    Load the TinyStories dataset from HuggingFace Datasets.
    Returns:
        The loaded dataset object.
    """
    from datasets import load_dataset
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    print(f"Full TinyStories dataset size: {len(dataset['train'])} training examples")
    return dataset

def create_dataset_subsets(
    dataset: Any,
    num_examples: int,
    filter_word: Optional[str],
    no_cache: bool,
    cache_manager: CacheManager,
) -> Tuple[Any, Any, str]:
    """
    Create (and cache) training and validation subsets from the TinyStories dataset.
    Returns:
        (train_subset, val_subset, dataset_cache_key)
    """
    dataset_params = {"num_examples": num_examples, "filter_word": filter_word}
    def compute_subsets():
        print(f"Creating dataset subsets with {num_examples} examples...")
        subset_size = num_examples
        if filter_word:
            print(f"Filtering dataset to include only examples with the word: '{filter_word}'")
            def filter_function(example):
                return filter_word.lower() in example["text"].lower()
            filtered_dataset = dataset["train"].filter(
                filter_function, desc=f"Filtering for '{filter_word}'"
            )
            print(f"Found {len(filtered_dataset)} examples containing '{filter_word}'")
            if len(filtered_dataset) > subset_size:
                train_subset = filtered_dataset.select(range(subset_size))
            else:
                train_subset = filtered_dataset
                print(
                    f"Warning: Only {len(train_subset)} examples contain '{filter_word}', less than requested {subset_size}"
                )
        else:
            train_subset = dataset["train"].select(range(subset_size))
        print(f"Using subset size: {len(train_subset)} training examples")
        val_size = int(len(train_subset) * 0.1)
        val_start_idx = subset_size if not filter_word else 0
        val_subset = dataset["train"].select(range(val_start_idx, val_start_idx + val_size))
        print(f"Created validation set with {len(val_subset)} examples")
        return {"train_subset": train_subset, "val_subset": val_subset}
    cached_data, dataset_cache_key = cache_manager.get_or_compute(
        dataset_params, "dataset_subsets", compute_subsets, no_cache
    )
    return cached_data["train_subset"], cached_data["val_subset"], dataset_cache_key
