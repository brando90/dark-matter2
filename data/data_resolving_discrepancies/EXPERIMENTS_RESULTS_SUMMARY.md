# EXPERIMENTS RESULTS SUMMARY

## Overview
This dataset contains the results of systematic scaling laws experiments from the paper "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" by Tomer Porian et al. Each row represents one complete training run of a transformer language model.

## Data Structure
- **Total rows**: 977 (one per training experiment, 1 header/columns, 1 empty new line)
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

L(N, D) = E + A * N^{-α} + B * D^{-β}

Additive Chinchilla-style loss with irreducible floor E and separate model/data exponents. Compute scaling follows C ≈ 6 N D.

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
