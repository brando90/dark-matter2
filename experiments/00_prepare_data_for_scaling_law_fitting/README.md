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

```bash
bash ~/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/main_chinchilla_verification.bash
```

## Expected Output

**Besiroglu Targets:**
- E ≈ 1.82, A ≈ 482, α ≈ 0.34, B ≈ 2085, β ≈ 0.36
- D/N ≈ 20 (Chinchilla-optimal)

**Files:**
- Data: `~/dark-matter2/data/chinchilla_verification_data.csv`
- Plots: `plots/chinchilla_verification_*.png`
