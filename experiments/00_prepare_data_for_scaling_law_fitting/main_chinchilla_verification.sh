# Launch Chinchilla verification: fits L(N,D)=E+A/N^α+B/D^β, verifies against Chinchilla replication targets

# Activate uv environment
source "$HOME/uv_envs/dark-matter2/bin/activate"

# Run verification
python "$HOME/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/main_chinchilla_verification.py" \
    --mode=online \
    --n_points=400 \
    --seed=42
