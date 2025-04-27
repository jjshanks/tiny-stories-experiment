"""
model_utils.py

Utility functions for model configuration and setup for TinyStories experiments.
"""

from typing import Any, Dict, Tuple
from transformers import GPT2Config, GPT2Tokenizer


def setup_model_and_config(
    tokenizer: GPT2Tokenizer,
    dimensions: int,
    layers: int,
    heads: int,
    max_length: int,
    dropout: float = 0.1,  # Add dropout parameter with default
) -> Tuple[GPT2Config, Dict[str, Any]]:
    """
    Create a GPT2Config and model parameter dictionary for a small LLM.

    This defines the architecture of our transformer model.
    - n_embd (dimensions): The size of the vector representing each token (embedding dimension).
    - n_layer (layers): The number of transformer blocks stacked on top of each other. More layers = deeper model.
    - n_head (heads): The number of attention heads in the multi-head attention mechanism. Allows the model to focus on different parts of the input simultaneously. Must divide n_embd evenly.
    - n_positions (max_length): The maximum sequence length the model can process.
    - resid_pdrop, embd_pdrop, attn_pdrop (dropout): Dropout probability for various layers.

    Args:
        tokenizer: The tokenizer to use.
        dimensions: Embedding dimension size.
        layers: Number of transformer layers.
        heads: Number of attention heads.
        max_length: Maximum sequence length for the model.
        dropout: Dropout probability for residual, embedding, and attention layers.
    Returns:
        (config, model_params)
    """
    model_params = {
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "n_positions": max_length,
        "n_embd": dimensions,
        "n_layer": layers,
        "n_head": heads,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "dropout": dropout,  # Include dropout in params
    }
    print("Creating a small GPT-2 model configuration...")
    config = GPT2Config(
        vocab_size=model_params["tokenizer_vocab_size"],
        n_positions=model_params["n_positions"],
        n_embd=model_params["n_embd"],
        n_layer=model_params["n_layer"],
        n_head=model_params["n_head"],
        pad_token_id=model_params["pad_token_id"],
        eos_token_id=model_params["eos_token_id"],
        resid_pdrop=dropout,  # Apply dropout
        embd_pdrop=dropout,   # Apply dropout
        attn_pdrop=dropout,   # Apply dropout
    )
    print(
        f"Model parameters: layers={config.n_layer}, heads={config.n_head}, embd_dim={config.n_embd}, dropout={dropout}"
    )
    return config, model_params
