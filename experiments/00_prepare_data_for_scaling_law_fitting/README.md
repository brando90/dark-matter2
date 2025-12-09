# Chinchilla Verification

Verify scaling law implementation fits L(N,D) = E + A/N^α + B/D^β correctly.

## Install

```bash
# Create uv environment at standard location
uv venv ~/uv_envs/dark-matter2

# Activate and sync dependencies to active environment
source ~/uv_envs/dark-matter2/bin/activate
cd ~/dark-matter2
uv sync --active
```

## Run

- Activate env:
  ```bash
  source "$HOME/uv_envs/dark-matter2/bin/activate"
  cd ~/dark-matter2
  ```

- Synthetic run (default paths/plots):
  ```bash
  python experiments/00_prepare_data_for_scaling_law_fitting/main_chinchilla_verification.py \
    --mode dryrun \
    --n_points 400 \
    --seed 42
  ```

- Run on a specific CSV (must have `N_params`, `tokens_D`, `loss`; `flops_C` auto-computed if missing) and suffix output:
  ```bash
  python experiments/00_prepare_data_for_scaling_law_fitting/main_chinchilla_verification.py \
    --mode dryrun \
    --data_csv data/pythia_style_data.csv \
    --plots_suffix pythia_style
  ```
  Plots and `verification.log` land in `experiments/00_prepare_data_for_scaling_law_fitting/plots/plots_out_<suffix>/`.

- Online W&B logging: add `--mode online` (optionally `--wandb_project`, `--wandb_entity`).

## Expected Output

**Besiroglu Targets:**
- E ≈ 1.82, A ≈ 482, α ≈ 0.34, B ≈ 2085, β ≈ 0.36
- D/N ≈ 20 (Chinchilla-optimal)

**Files:**
- Data: `~/dark-matter2/data/chinchilla_verification_data.csv`
- Plots: `plots/chinchilla_verification_*.png`
