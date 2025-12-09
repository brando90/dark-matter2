# Dark Matter 2

The Dark Matter Project investigates the "missing terms" in neural scaling laws—the higher-order components that standard single-slope power-law models fail to capture. Its central aim is to build more accurate predictive laws that can extrapolate large-model performance from smaller, cheaper runs with lower error.

## Goal

Explore extensions of the canonical Chinchilla-style equation:

```
L(N) = E + A / N^α + B / N^β   with β < α
```

This introduces a sub-leading correction term to capture curvature in scaling-law fits. The formulation arose in discussions with Joshua as a way to test whether adding a second power-law term improves back-test and extrapolation accuracy over the standard single-term model.

**Note:** This is treated as a conjecture rather than a commitment—the broader goal is to identify whatever functional form most effectively explains residual "dark matter" structure in existing scaling ladders (e.g., Chinchilla, Kaplan, Pythia) and thereby enable more reliable forecasts of model behavior across compute regimes.

## Structure

```
dark-matter2/
├── py_src/                    # Reusable Python modules
├── experiments/               # Individual experiments
│   └── 00_prepare_data_.../   # Chinchilla verification
├── data/                      # Scaling data (CSV format)
└── README.md
```

## Environment

```bash
uv venv ~/uv_envs/dark-matter2
source ~/uv_envs/dark-matter2/bin/activate
cd ~/dark-matter2
uv sync --active
```

## References

- Hoffmann et al. (2022): "Training Compute-Optimal Large Language Models" (Chinchilla)
- Kaplan et al. (2020): "Scaling Laws for Neural Language Models"
- Besiroglu et al.: Chinchilla replication with corrected coefficients
