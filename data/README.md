# Resolving Discrepancies in Compute-Optimal Scaling Laws - Data

**Source**: https://github.com/formll/resolving-scaling-law-discrepancies  
**Paper**: "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" by Tomer Porian, Mitchell Wortsman, et al.

## Files

1. **`experiment_results.pickle.xz`** - Original pickle file (compressed) with pandas DataFrames containing time-series loss data per training run. Requires pandas compatibility fix for newer pandas versions.

2. **`scaling_law_data.csv`** - Flattened CSV with 975 training runs, ready for scaling law fitting.

3. **`data_hitchhickers_guide/*.csv`** - 65 CSVs from the “Hitchhiker’s Guide” aggregation (e.g., Pythia, LLaMA, Striped Hyena, etc.). These contain ~572 columns per file, including many benchmark metrics and a `loss` column for training trajectories.

4. **Unified, loss-only filtered CSVs**
   - `data_resolving_discrepancies/scaling_law_data_unified.csv`
   - `data_hitchhickers_guide/hitchhickers_guide_unified.csv`
   - `scaling_law_unified.csv` (combined)
   - All three keep only rows with finite numeric `loss` and trim to a common schema (below). Original CSVs remain untouched; no NaNs are imputed or merged—rows without a numeric `loss` are simply dropped.

### Unified Schema (core columns)
| Column | Description |
|--------|-------------|
| `num_params` | Model parameters (N) |
| `tokens` | Tokens seen (D) |
| `flops` | Approximate FLOPs (C) |
| `loss` | Validation loss (finite only; non-numeric/NaN dropped) |
| `dataset` | Training dataset identifier |
| `model_name` | Model or checkpoint identifier |
| `source` | Data source tag (`resolving_discrepancies` or `hitchhickers_guide`) |
| `family` | Model family |
| `arch` | Architecture type |

### Unified File Details
- `data_resolving_discrepancies/scaling_law_data_unified.csv`  
  - Source: `scaling_law_data.csv`  
  - Transformations: `final_val_loss → loss`; `ckpt_path → model_name`; add `source=family='resolving_discrepancies'`, `arch='dec'`; keep only finite `loss`.  
  - Rows: 975.
- `data_hitchhickers_guide/hitchhickers_guide_unified.csv`  
  - Source: all `data_hitchhickers_guide/*.csv`.  
  - Transformations: `tokens_seen → tokens`; `data → dataset`; add `source='hitchhickers_guide'`; keep only finite numeric `loss`; other columns are untouched/trimmed away.  
  - Rows: 1,639,961 (mostly dense training trajectories such as LLaMA and Striped Hyena runs).
- `scaling_law_unified.csv`  
  - Combined concat of the two unified files above.  
  - Rows: 1,640,936.

## CSV Columns (scaling_law_data.csv)

### Experiment Metadata
| Column | Description | Example Values |
|--------|-------------|----------------|
| `dataset` | Training dataset | `RefinedWeb`, `OpenWebText2` |
| `hparams` | Hyperparameter setting | `base`, `tuned`, `sweep`, `seed` |
| `warmup` | LR warmup duration | `short`, `long` |
| `decay` | LR decay schedule | `kaplan`, `chinchilla`, `const` |

### Model Architecture
| Column | Description | Range |
|--------|-------------|-------|
| `width` | Hidden dimension (d_model) | 96 - 1504 |
| `depth` | Number of transformer layers | 3 - 30 |
| `seq_len` | Sequence length | 2048 |
| `vocab_size` | Vocabulary size | 50432 |
| `num_params` | Number of parameters (computed) | ~5M - 890M |

### Training Configuration
| Column | Description |
|--------|-------------|
| `batch_size` | Batch size (in sequences) |
| `lr` | Learning rate |
| `max_step` | Maximum training steps |
| `tokens` | Total tokens seen (max_step × batch_size × seq_len) |
| `flops` | Approximate FLOPs (6 × num_params × tokens) |
| `optimizer` | Optimizer used (AdamW) |
| `precision` | Training precision (amp_bfloat16) |
| `seed` | Random seed |

### Results
| Column | Description |
|--------|-------------|
| `final_val_loss` | Final validation loss |
| `final_train_loss` | Final training loss |
| `final_step` | Step at which final loss was recorded |
| `ckpt_path` | HuggingFace checkpoint path |

## Data Statistics

- **Total runs**: 975
- **Model sizes**: ~5M to ~890M parameters
- **Datasets**: RefinedWeb, OpenWebText2
- **Final validation loss range**: 2.65 - 7.97

## Usage for Scaling Law Fitting

Key columns for Chinchilla-style scaling laws (L = f(N, D)):
- `N` (parameters): `num_params`
- `D` (tokens): `tokens`  
- `L` (loss): `final_val_loss`
- `C` (compute): `flops`

Filter by `hparams == 'tuned'` for best hyperparameters, or explore different decay schedules (`kaplan`, `chinchilla`, `const`).

## Loading the Original Pickle (pandas ≥ 2.0)

```python
import sys
import pandas as pd
import types

# Compatibility fix for newer pandas
fake_module = types.ModuleType('pandas.core.indexes.numeric')
fake_module.Int64Index = pd.Index
fake_module.Float64Index = pd.Index
fake_module.NumericIndex = pd.Index
sys.modules['pandas.core.indexes.numeric'] = fake_module

import lzma
import pickle

with lzma.open('experiment_results.pickle.xz', 'rb') as f:
    df = pickle.load(f)
```

Note: The original pickle contains time-series data in `val/loss` and `train/loss` columns (pd.Series indexed by training step).
