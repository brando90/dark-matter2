#!/usr/bin/env python3
"""
Extract loss values from Pythia HuggingFace model configs.

This script loads Pythia model checkpoints from HuggingFace and extracts
training loss values from the model configs or training logs.
"""

import json
import requests
import pandas as pd

# Pythia model sizes and their non-embedding parameter counts
MODEL_SIZES = {
    "70m": 18915328,
    "160m": 85056000,
    "410m": 302311424,
    "1b": 805736448,
    "1.4b": 1208602624,
    "2.8b": 2517652480,
    "6.9b": 6444163072,
    "12b": 11327027200,
}

# Training batch size in tokens
BATCH_SIZE = 2097152  # ~2M tokens

# Checkpoint steps to sample
CHECKPOINT_STEPS = [
    143,     # ~300M tokens
    1430,    # ~3B tokens  
    7150,    # ~15B tokens
    14300,   # ~30B tokens
    28600,   # ~60B tokens
    57200,   # ~120B tokens
    85800,   # ~180B tokens
    114400,  # ~240B tokens
    143000,  # ~300B tokens (final)
]


def get_model_card_info(model_name: str, revision: str = "main") -> dict:
    """Fetch model card information from HuggingFace."""
    url = f"https://huggingface.co/EleutherAI/pythia-{model_name}/raw/{revision}/config.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def get_training_args(model_name: str, revision: str = "main") -> dict:
    """Try to fetch training_args.bin info."""
    url = f"https://huggingface.co/api/models/EleutherAI/pythia-{model_name}/tree/{revision}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def main():
    print("Checking Pythia model configs on HuggingFace...")
    print("=" * 60)
    
    results = []
    
    for model_name, n_params in MODEL_SIZES.items():
        print(f"\n{model_name} ({n_params:,} params):")
        
        # Check the main config
        config = get_model_card_info(model_name)
        if config:
            print(f"  Config keys: {list(config.keys())[:10]}...")
            
            # Check for loss-related keys
            for key in ['training_loss', 'train_loss', 'final_loss', 'eval_loss']:
                if key in config:
                    print(f"  Found {key}: {config[key]}")
        
        # Check a specific checkpoint
        for step in [143000, 1000]:
            revision = f"step{step}"
            config = get_model_card_info(model_name, revision)
            if config:
                print(f"  Step {step} config available")
                for key in config:
                    if 'loss' in key.lower():
                        print(f"    {key}: {config[key]}")
            
            # Check tree for training info
            tree = get_training_args(model_name, revision)
            if tree:
                filenames = [f.get('path', '') for f in tree if isinstance(f, dict)]
                training_files = [f for f in filenames if 'train' in f.lower() or 'log' in f.lower()]
                if training_files:
                    print(f"  Training files at step {step}: {training_files}")


if __name__ == "__main__":
    main()

