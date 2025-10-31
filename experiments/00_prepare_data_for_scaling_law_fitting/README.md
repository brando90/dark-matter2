# Scaling Laws Fit Implementation

This directory contains the implementation of the scaling laws fitting methodology for the Dark Matter Project, based on the Chinchilla replication approach.

## Overview

The implementation tests whether adding a **sub-leading correction term** to standard single-slope power-law models improves prediction accuracy for neural network scaling laws.

## What's Implemented

### 1. Joint Power Law Fit: `L(N,D) = E + A/N^α + B/D^β`

- Uses **Huber loss** objective (robust to outliers)
- Implements **log-sum-exp trick** from Epoch AI Chinchilla replication (Approach 3)
- Fits parameters: `E` (asymptotic loss), `A`, `α` (param scaling), `B`, `β` (token scaling)

### 2. Two-Term N-Only Fit: `L(N) = E + A/N^α + B/N^β` with `β < α`

- Models sub-leading corrections in the parameter-only scaling dimension
- Enforces constraint `β < α` via reparameterization
- Tests hypothesis that finite-regime corrections matter

### 3. Statistical Testing

- **Likelihood-ratio test** to determine if the second term is warranted
- Compares 1-term vs 2-term models using Huber-as-likelihood
- Reports test statistic, degrees of freedom, and p-value

### 4. Compute-Optimal Analysis

- Calculates optimal compute split: `a = β/(α+β)` for params, `b = 1-a` for tokens
- Reports tokens-per-parameter ratio at compute optimum

## Files

- **`scaling_laws_fit.py`**: Main implementation with all fitting functions
- **`codex_prompt.md`**: Original specification from ChatGPT GPT-5 Pro

## Usage

### Quick Test with Synthetic Data

```bash
cd /Users/brandomiranda/dark-matter2
uv run experiments/00_prepare_data_for_scaling_law_fitting/scaling_laws_fit.py
```

This will:
1. Generate synthetic scaling ladder data
2. Fit the joint law and recover ground truth parameters
3. Fit the two-term N-only model
4. Run likelihood-ratio test
5. Report compute-optimal split

### Using Real Data

Create a CSV file with the following schema:

| field           | type  | notes                                       |
| --------------- | ----- | ------------------------------------------- |
| `run_id`        | str   | unique per training run                     |
| `family`        | str   | model family (e.g., Pythia-*, OLMO-*)       |
| `N_params`      | float | **total** params (include embeddings)       |
| `tokens_D`      | float | tokens seen at this point                   |
| `flops_C`       | float | optional; use `C≈6ND` if not logged         |
| `loss`          | float | final or checkpoint loss (bits/word or NLL) |
| `split`         | str   | e.g., `train_eval` or `valid`               |
| `is_checkpoint` | bool  | `true` if intermediate checkpoint           |
| `notes`         | str   | anything special (warmup, schedule)         |

Then in Python:

```python
from scaling_laws_fit import load_ladder_csv, fit_joint_powerlaw, compute_optimal_from_joint

# Load data
df = load_ladder_csv("ladder.csv")

# Fit joint law
jp = fit_joint_powerlaw(df["N_params"].values, df["tokens_D"].values, df["loss"].values)
print(jp, compute_optimal_from_joint(jp))

# For N-only fit, use a slice at near-constant tokens
mid = df["tokens_D"].median()
band = (df["tokens_D"] > 0.9*mid) & (df["tokens_D"] < 1.1*mid)
tp = fit_two_term_N_only(df.loc[band, "N_params"].values, df.loc[band, "loss"].values)
print(tp)
```

## Key Results from Synthetic Test

The implementation successfully recovers ground truth parameters:

**Ground Truth:**
- E = 1.8, A = 500.0, α = 0.35, B = 2000.0, β = 0.36

**Fitted Values:**
- E = 1.8066, A = 537.89, α = 0.3552, B = 2194.82, β = 0.3639

**Compute-Optimal Split:**
- a = β/(α+β) = 0.5061 (~50.6% compute on params, ~49.4% on tokens)
- This matches the Chinchilla finding of balanced scaling!

## Dependencies

All dependencies are handled by the project's `pyproject.toml`:
- `numpy`
- `pandas` 
- `scipy` (for optimization and special functions)

## References

- **Epoch AI Chinchilla Replication**: Huber + log-sum-exp joint fit and LR-test methodology
- **Chinchilla Paper (Hoffmann et al.)**: Original compute-optimal scaling laws
- **CS336 Lecture Notes**: Method-1/2/3 playbook and IsoFLOPs
- **Reconciliation Papers (Pearce, Porian et al.)**: Measurement corrections for consistent comparisons

## Next Steps

1. **Gather real scaling ladders** from public sources:
   - Hitchhiker's Guide compendium (485 models, 1.9M loss steps)
   - Pythia checkpoints
   - OLMo training runs

2. **Run back-tests**: Fit on 70% of data, predict the rest

3. **Generate plots**: 
   - Ladder with fitted curves
   - IsoFLOPs contours with predicted `(N*, D*)`
   - Residual analysis

4. **Compare with Joshua's analysis** to determine if higher-order terms reduce extrapolation error

