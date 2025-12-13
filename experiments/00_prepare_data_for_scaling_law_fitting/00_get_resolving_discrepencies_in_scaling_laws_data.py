"""
Script to download and process the resolving-scaling-law-discrepancies dataset.

This script:
1. Downloads the experiment results from the GitHub repository
2. Processes the compressed pickle file containing scaling laws data
3. Exports both a human-readable CSV summary and complete JSON data
4. Creates documentation explaining the dataset structure

Usage: python 00_get_resolving_discrepencies_in_scaling_laws_data.py
For a summary see: ~/dark-matter2/data/data_resolving_discrepancies/EXPERIMENTS_RESULTS_SUMMARY.md
"""

import os
import json
import requests
from pathlib import Path
import pandas as pd
from urllib.request import urlretrieve


def download_experiment_data(data_dir: Path, url: str, filename: str) -> Path:
    """
    Download the experiment results pickle file from GitHub.

    Args:
        data_dir: Directory to save the downloaded file
        url: URL to download from
        filename: Name of the file to save

    Returns:
        Path to the downloaded file
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / filename

    if file_path.exists():
        print(f"✓ File already exists: {file_path}")
        return file_path

    print(f"⬇️ Downloading {filename}...")
    try:
        # Use requests for better error handling and progress
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(".1f", end='', flush=True)

        print(f"\n✓ Downloaded to: {file_path}")
        return file_path

    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def load_dataframe(pickle_path: Path) -> pd.DataFrame:
    """
    Load the compressed pickle file containing experiment results.

    Args:
        pickle_path: Path to the .pickle.xz file

    Returns:
        Pandas DataFrame with experiment data
    """
    print(f"📖 Loading dataframe from: {pickle_path}")
    df = pd.read_pickle(pickle_path)
    print(f"✓ Loaded {len(df)} experiments with {len(df.columns)} columns")
    return df


def inspect_dataframe(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataframe structure.

    Args:
        df: The experiment results dataframe
    """
    print(f"\n📊 DataFrame columns ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\n🔍 First experiment sample:")
    first_row = df.iloc[0]
    print(f"  Model: {first_row['width']}×{first_row['depth']} = {first_row['width']*first_row['depth']:,} parameters")
    print(f"  Dataset: {first_row['dataset']}")
    print(f"  Training steps: {first_row['max_step']:,}")
    print(f"  Final validation loss: {first_row.get('val/loss', pd.Series()).iloc[-1]:.3f}")


def export_json_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Export complete dataframe to JSON format.

    Converts pandas Series objects to dictionaries for JSON serialization.
    Note: This creates a very large file (~72MB) with all time-series data.

    Args:
        df: DataFrame to export
        output_path: Path for the JSON output file
    """
    print(f"📝 Exporting complete data to JSON: {output_path}")

    # Convert Series objects to dicts for JSON serialization
    df_json = df.copy()
    series_converted = 0

    for col in df_json.columns:
        if df_json[col].dtype == 'object':  # Check if column contains Series
            df_json[col] = df_json[col].apply(lambda x: x.to_dict() if hasattr(x, 'to_dict') else x)
            series_converted += 1

    print(f"  Converted {series_converted} Series columns to dictionaries")

    # Export with nice formatting
    df_json.to_json(output_path, orient='records', indent=2)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(".1f")
    print("  ⚠️  Warning: This JSON file is very large and not recommended for manual inspection")


def create_summary_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a human-readable CSV summary with key metrics extracted from time-series data.

    This flattens the complex nested data into a spreadsheet-friendly format.

    Args:
        df: Original dataframe with time-series data
        output_path: Path for the CSV output file
    """
    print(f"📊 Creating summary CSV: {output_path}")

    summary_data = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing experiment {idx+1}/{len(df)}...")

        summary_row = {}

        # Copy scalar columns directly (hyperparameters and metadata)
        scalar_cols = ['dataset', 'hparams', 'warmup', 'decay', 'width', 'depth', 'warmup_tokens',
                       'beta1', 'beta2', 'world_size', 'lr', 'seed', 'seq_len', 'vocab_size',
                       'grad_clip_norm', 'optimizer', 'precision', 'qk_norm', 'z_loss_coefficient',
                       'bs', 'independent_wd', 'max_step', 'ckpt_path']

        for col in scalar_cols:
            summary_row[col] = row[col]

        # Extract key performance metrics from time-series data
        # Validation loss metrics
        if hasattr(row['val/loss'], 'iloc') and len(row['val/loss']) > 0:
            summary_row['final_val_loss'] = row['val/loss'].iloc[-1]
            summary_row['min_val_loss'] = row['val/loss'].min()
            summary_row['initial_val_loss'] = row['val/loss'].iloc[0]
        else:
            summary_row['final_val_loss'] = row['val/loss']
            summary_row['min_val_loss'] = row['val/loss']
            summary_row['initial_val_loss'] = row['val/loss']

        # Training loss metrics
        if hasattr(row['train/loss'], 'iloc') and len(row['train/loss']) > 0:
            summary_row['final_train_loss'] = row['train/loss'].iloc[-1]
            summary_row['min_train_loss'] = row['train/loss'].min()
            summary_row['total_training_steps'] = len(row['train/loss'])
        else:
            summary_row['final_train_loss'] = row['train/loss']
            summary_row['min_train_loss'] = row['train/loss']
            summary_row['total_training_steps'] = 1

        summary_data.append(summary_row)

    # Create and save summary dataframe
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)

    file_size_kb = output_path.stat().st_size / 1024
    print(".1f")
    print(f"  ✓ Contains {len(summary_df)} experiments with {len(summary_df.columns)} columns each")


def create_documentation(data_dir: Path) -> None:
    """
    Create a markdown documentation file explaining the dataset.

    Args:
        data_dir: Directory containing the data files
    """
    doc_path = data_dir / "EXPERIMENTS_RESULTS_SUMMARY.md"

    if doc_path.exists():
        print(f"✓ Documentation already exists: {doc_path}")
        return

    print(f"📝 Creating documentation: {doc_path}")

    # Copy the documentation content from before
    doc_content = """# EXPERIMENTS RESULTS SUMMARY

## Overview
This dataset contains the results of systematic scaling laws experiments from the paper "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" by Tomer Porian et al. Each row represents one complete training run of a transformer language model.

## Data Structure
- **Total rows**: 977 (one per training experiment)
- **Total columns**: 29
- **Source**: https://github.com/formll/resolving-scaling-law-discrepancies

## What Each Row Represents
Each row is one experimental measurement in a scaling laws study. It captures the complete training trajectory of a language model with specific:
- Model architecture (width × depth)
- Training hyperparameters
- Dataset and training budget
- Final performance metrics

## Key Columns Explained

### Model Architecture
- `width`: Hidden dimension size (e.g., 1312.0)
- `depth`: Number of transformer layers (e.g., 26.0)
- `vocab_size`: Vocabulary size (e.g., 50432.0)
- `seq_len`: Maximum sequence length (e.g., 2048.0)

### Training Configuration
- `dataset`: Training data ("rw" = RefinedWeb, "owt2" = OpenWebText2)
- `max_step`: Total training steps (e.g., 13581)
- `warmup_tokens`: Number of tokens for learning rate warmup (e.g., 611958784)
- `bs`: Batch size (e.g., 256.0)
- `world_size`: Number of GPUs used (e.g., 32)

### Optimization Hyperparameters
- `lr`: Learning rate (e.g., 0.003)
- `optimizer`: Optimization algorithm ("adamw")
- `beta1`, `beta2`: Adam optimizer parameters (0.9, 0.95)
- `decay`: Learning rate decay schedule ("kaplan", "constant", "cosine")
- `warmup`: Warmup strategy ("short", "long")
- `grad_clip_norm`: Whether gradient clipping is enabled (True)
- `qk_norm`, `z_loss_coefficient`, `independent_wd`: Additional regularization parameters

### Experimental Conditions
- `hparams`: Hyperparameter tuning level ("base" = default, "tuned" = optimized)
- `precision`: Mixed precision training ("amp_bfloat16")
- `seed`: Random seed (0.0)

### Performance Metrics
- `final_val_loss`: Validation loss at end of training (lower = better)
- `min_val_loss`: Best validation loss achieved during training
- `initial_val_loss`: Validation loss at start of training (random initialization)
- `final_train_loss`: Training loss at end of training
- `min_train_loss`: Best training loss achieved during training
- `total_training_steps`: Number of checkpoints logged during training

### Metadata
- `ckpt_path`: Path to saved model checkpoints on Hugging Face

## Scaling Laws Context
This data enables fitting compute-optimal scaling laws of the form:
```
Loss = A × (parameters)^(-α) × (tokens)^(-β)
```

Where:
- **Parameters**: Model size (width × depth × other factors)
- **Tokens**: Training data seen (max_step × seq_len × bs)
- **Loss**: Final validation loss

## Usage Examples
1. **Model size vs performance**: Plot `final_val_loss` vs `width*depth`
2. **Training compute vs performance**: Plot `final_val_loss` vs `max_step`
3. **Hyperparameter effects**: Compare losses across different `decay` or `warmup` settings
4. **Dataset effects**: Compare performance on "rw" vs "owt2" datasets

## Data Quality Notes
- All experiments use consistent evaluation (same validation set, metrics)
- Multiple runs with different random seeds for statistical reliability
- Comprehensive hyperparameter sweeps across model sizes and training budgets
- Both tuned and untuned hyperparameter configurations included

## File Format
This summary describes the CSV file: `experiment_results_summary.csv`
- **Size**: ~241KB
- **Format**: Comma-separated values, human-readable
- **Compatible with**: Excel, Google Sheets, pandas, R, etc.

For complete time-series data (all training checkpoints), see: `experiment_results.json` (72MB, not recommended for manual inspection).
"""

    with open(doc_path, 'w') as f:
        f.write(doc_content)

    print(f"✓ Documentation created: {doc_path}")


def main():
    """Main function to download and process the scaling laws dataset."""
    print("🚀 Starting scaling laws data processing...")

    # Configuration
    data_dir = Path.home() / "dark-matter2" / "data" / "data_resolving_discrepancies"
    download_url = "https://github.com/formll/resolving-scaling-law-discrepancies/raw/main/data/experiment_results.pickle.xz"
    pickle_filename = "experiment_results.pickle.xz"

    json_filename = "experiment_results.json"
    csv_filename = "experiment_results_summary.csv"

    try:
        # Step 1: Download the data
        pickle_path = download_experiment_data(data_dir, download_url, pickle_filename)

        # Step 2: Load and inspect the dataframe
        df = load_dataframe(pickle_path)
        inspect_dataframe(df)

        # Step 3: Export complete data to JSON (for completeness)
        json_path = data_dir / json_filename
        export_json_data(df, json_path)

        # Step 4: Create human-readable CSV summary (recommended for inspection)
        csv_path = data_dir / csv_filename
        create_summary_csv(df, csv_path)

        # Step 5: Create documentation
        create_documentation(data_dir)

        print("\n🎉 All done! Files created:")
        print(f"  📄 Summary CSV: {csv_path}")
        print(f"  📄 Full JSON: {json_path}")
        print(f"  📖 Documentation: {data_dir}/EXPERIMENTS_RESULTS_SUMMARY.md")
        print("\n💡 For manual inspection, use the CSV file - it's spreadsheet-friendly!")
        print("   Open it in Excel, Google Sheets, or any CSV viewer.")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
