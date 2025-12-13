#!/usr/bin/env python3
"""
Convert Chinchilla verification data to match Ludwig's scaling laws format.

This script takes the original chinchilla data with columns [N_params, tokens_D, loss, flops_C, family]
and converts it to match the Ludwig format with 29 columns including hyperparameters and metadata.
"""

import pandas as pd
from pathlib import Path
import numpy as np


def convert_chinchilla_to_ludwig_format(chinchilla_csv: Path, ludwig_csv: Path) -> pd.DataFrame:
    """
    Convert chinchilla data to Ludwig format by adding missing columns with defaults.

    Args:
        chinchilla_csv: Path to original chinchilla data
        ludwig_csv: Path to Ludwig format CSV (to get column structure)

    Returns:
        DataFrame in Ludwig format
    """
    # Load chinchilla data
    chinchilla_df = pd.read_csv(chinchilla_csv)
    print(f"Loaded {len(chinchilla_df)} chinchilla data points")

    # Load Ludwig format to get column structure
    ludwig_df = pd.read_csv(ludwig_csv, nrows=1)  # Just need header
    ludwig_columns = ludwig_df.columns.tolist()

    # Create new dataframe with Ludwig columns
    converted_df = pd.DataFrame(columns=ludwig_columns)

    # Map chinchilla columns to Ludwig columns
    # N_params -> width * depth (we'll set width=depth for simplicity)
    # tokens_D -> derived from max_step * seq_len * bs
    # loss -> final_val_loss
    # flops_C -> derived FLOP calculation

    for idx, row in chinchilla_df.iterrows():
        new_row = {}

        # Core scaling law parameters
        n_params = row['N_params']
        tokens = row['tokens_D']
        loss = row['loss']
        flops = row['flops_C']
        family = row['family']

        # Set model architecture - assume square models for simplicity
        # width * depth = n_params, so width = depth = sqrt(n_params)
        model_size = np.sqrt(n_params)
        new_row['width'] = model_size
        new_row['depth'] = model_size

        # Dataset info
        new_row['dataset'] = 'synthetic'  # Since it's synthetic chinchilla data

        # Hyperparameters (using chinchilla defaults)
        new_row['hparams'] = 'chinchilla'
        new_row['warmup'] = 'chinchilla'
        new_row['decay'] = 'chinchilla'

        # Training configuration
        new_row['warmup_tokens'] = 0  # Chinchilla uses different warmup
        new_row['max_step'] = 1  # Placeholder - we'll derive from tokens
        new_row['bs'] = 1  # Placeholder
        new_row['seq_len'] = 1  # Placeholder

        # Derive training steps from tokens (rough approximation)
        # tokens = max_step * seq_len * bs, so max_step = tokens / (seq_len * bs)
        # We'll set reasonable defaults
        new_row['seq_len'] = 2048  # Standard sequence length
        new_row['bs'] = max(1, int(tokens / (2048 * 1000)))  # Rough estimate
        new_row['max_step'] = max(1, int(tokens / (new_row['seq_len'] * new_row['bs'])))

        # Optimizer settings (chinchilla uses Adam)
        new_row['lr'] = 0.001  # Chinchilla learning rate
        new_row['optimizer'] = 'adam'
        new_row['beta1'] = 0.9
        new_row['beta2'] = 0.999
        new_row['grad_clip_norm'] = True
        new_row['independent_wd'] = 0.0

        # Model architecture
        new_row['vocab_size'] = 32000  # Chinchilla vocab size
        new_row['qk_norm'] = False
        new_row['z_loss_coefficient'] = 0.0

        # Training setup
        new_row['world_size'] = 1  # Single GPU
        new_row['precision'] = 'fp32'
        new_row['seed'] = 42

        # Performance metrics
        new_row['final_val_loss'] = loss
        new_row['min_val_loss'] = loss  # Assume this is the minimum
        new_row['initial_val_loss'] = loss + 2.0  # Rough estimate of initial loss
        new_row['final_train_loss'] = loss - 0.1  # Train loss usually slightly better
        new_row['min_train_loss'] = loss - 0.1
        new_row['total_training_steps'] = 100  # Placeholder

        # Metadata
        new_row['ckpt_path'] = f'chinchilla/{family}/params_{int(n_params)}_tokens_{int(tokens)}'

        # Add to dataframe
        for col in ludwig_columns:
            if col not in new_row:
                if col in ['width', 'depth', 'warmup_tokens', 'beta1', 'beta2', 'world_size', 'lr', 'seq_len', 'vocab_size', 'bs', 'independent_wd', 'max_step', 'final_val_loss', 'min_val_loss', 'initial_val_loss', 'final_train_loss', 'min_train_loss', 'total_training_steps']:
                    new_row[col] = 0  # Numeric defaults
                elif col in ['grad_clip_norm', 'qk_norm']:
                    new_row[col] = False  # Boolean defaults
                else:
                    new_row[col] = 'unknown'  # String defaults

        converted_df = pd.concat([converted_df, pd.DataFrame([new_row])], ignore_index=True)

    return converted_df


def main():
    """Convert chinchilla data to Ludwig format and save."""
    print("🔄 Converting Chinchilla data to Ludwig format...")

    # File paths
    chinchilla_csv = Path.home() / "dark-matter2" / "data" / "chinchilla_verification_data.csv"
    ludwig_csv = Path.home() / "dark-matter2" / "data" / "data_resolving_discrepancies" / "experiment_results_summary.csv"
    output_csv = Path.home() / "dark-matter2" / "data" / "chinchilla_verification_data_ludwig_format.csv"

    # Convert data
    converted_df = convert_chinchilla_to_ludwig_format(chinchilla_csv, ludwig_csv)

    # Save converted data
    converted_df.to_csv(output_csv, index=False)
    print(f"✅ Saved converted data to: {output_csv}")
    print(f"   Rows: {len(converted_df)}")
    print(f"   Columns: {len(converted_df.columns)}")

    # Show sample
    print("\n📊 Sample converted row:")
    sample = converted_df.iloc[0]
    key_cols = ['dataset', 'width', 'depth', 'max_step', 'final_val_loss', 'ckpt_path']
    for col in key_cols:
        print(f"  {col}: {sample[col]}")


if __name__ == "__main__":
    main()
