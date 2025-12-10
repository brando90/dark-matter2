#!/usr/bin/env python3
"""
Extract loss values from Pythia W&B logs.

This script extracts the training loss values for Pythia models at various checkpoints
from the official EleutherAI W&B project.

References:
- W&B Project: https://wandb.ai/eleutherai/pythia
- Paper: https://arxiv.org/pdf/2304.01373
"""

import wandb
import pandas as pd
import os

# Pythia model sizes and their non-embedding parameter counts
# From Table 1 of the Pythia paper (arxiv.org/pdf/2304.01373)
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

# Training info: batch size = 2M tokens (2,097,152), 143000 steps total
# Total tokens = 2,097,152 * 143000 ≈ 300B tokens
BATCH_SIZE = 2097152
TOTAL_STEPS = 143000

# Checkpoints we want to sample (these correspond to specific token counts)
# Tokens = step * batch_size
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

def extract_losses_from_wandb():
    """Extract loss values from the EleutherAI Pythia W&B project."""
    
    # Initialize W&B API
    api = wandb.Api()
    
    results = []
    
    # Map of run groups from the GitHub README
    # https://github.com/EleutherAI/pythia#models
    run_groups = {
        "70m": "Pythia 70M",
        "160m": "Pythia 125M",  # Note: old naming was 125M, now 160M
        "410m": "Pythia 350M",  # Note: old naming was 350M, now 410M
        "1b": "Pythia 800M",    # Note: old naming was 800M, now 1B
        "1.4b": "Pythia 1.3B",
        "2.8b": "Pythia 2.7B",  # Note: old naming was 2.7B, now 2.8B
        "6.9b": "Pythia 6.7B",
        "12b": "Pythia 12B",
    }
    
    project_path = "eleutherai/pythia"
    
    for model_name, n_params in MODEL_SIZES.items():
        print(f"\nProcessing {model_name} ({n_params:,} params)...")
        
        try:
            # Get runs for this model
            runs = api.runs(
                project_path,
                filters={"group": {"$regex": run_groups.get(model_name, model_name)}}
            )
            
            for run in runs:
                print(f"  Found run: {run.name}")
                
                # Get history with loss values
                history = run.history(keys=["global_step", "train/loss", "lm_loss"])
                
                if len(history) == 0:
                    continue
                    
                for step in CHECKPOINT_STEPS:
                    # Find the closest step in the history
                    closest_idx = (history["global_step"] - step).abs().idxmin()
                    row = history.iloc[closest_idx]
                    
                    loss_col = "train/loss" if "train/loss" in row else "lm_loss"
                    loss = row.get(loss_col)
                    
                    if pd.notna(loss):
                        tokens = step * BATCH_SIZE
                        flops = 6 * n_params * tokens
                        
                        results.append({
                            "N_params": n_params,
                            "tokens_D": tokens,
                            "loss": loss,
                            "flops_C": flops,
                            "family": "pythia",
                            "model_name": model_name,
                            "step": int(row["global_step"]),
                        })
                        print(f"    Step {step}: loss={loss:.4f}, tokens={tokens/1e9:.1f}B")
                
                # Only use first matching run
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return pd.DataFrame(results)


def main():
    print("Extracting Pythia loss values from W&B...")
    print("=" * 60)
    
    # Check if wandb is logged in
    try:
        api = wandb.Api()
        print("W&B API initialized successfully")
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease run 'wandb login' first to authenticate")
        return
    
    # Extract losses
    df = extract_losses_from_wandb()
    
    if len(df) == 0:
        print("\nNo data extracted. The W&B API might require authentication.")
        print("Alternative: manually download from https://wandb.ai/eleutherai/pythia")
        return
    
    # Save to CSV
    output_path = "/Users/brandomiranda/dark-matter2/data/pythia_data.csv"
    df_output = df[["N_params", "tokens_D", "loss", "flops_C", "family"]]
    df_output.to_csv(output_path, index=False)
    
    print(f"\n\nSaved {len(df)} data points to {output_path}")
    print(df_output.head(20))


if __name__ == "__main__":
    main()

