# Dark Matter 2 - Chinchilla Scaling Law Verification

Verify Chinchilla scaling law implementation by fitting L(N,D) = E + A/N^α + B/D^β to synthetic data.

## Quick Start

```bash
bash ~/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/main_chinchilla_verification.bash
```

## Expected Results

**Besiroglu Targets:**
- E ≈ 1.82, A ≈ 482, α ≈ 0.34, B ≈ 2085, β ≈ 0.36
- D/N ≈ 20 (Chinchilla-optimal)

## Structure

```
dark-matter2/
├── py_src/
│   └── scaling_laws_fit.py              # Reusable fitting code
├── experiments/
│   └── 00_prepare_data_for_scaling_law_fitting/
│       ├── main_chinchilla_verification.bash  # Launcher
│       ├── main_chinchilla_verification.py    # Main script (fire CLI)
│       └── plots/                             # Generated plots
├── data/
│   └── chinchilla_verification_data.csv       # Generated data
└── README.md                                   # This file
```

## Environment

```bash
source "$HOME/uv_envs/dark-matter2/bin/activate"
```

## References

- Hoffmann et al. (2022): Chinchilla paper
- Besiroglu et al.: Corrected replication
