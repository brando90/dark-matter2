# CS336 Scaling Laws Implementation

Clean, minimal implementation for CS336 Assignment 3 and Dark Matter research.

## Files

### Core Implementations
- **`isoflops_scaling.py`** - CS336 IsoFLOPs approach (Chinchilla Approach 1)
- **`scaling_laws_fit.py`** - Joint fit with Huber loss (Chinchilla Approach 3)

### Testing
- **`test_pipeline.py`** - Single test that validates everything

### Reference
- **`codex_prompt.md`** - Original specification
- **`cs336_spring2025_assignment3_scaling.pdf`** - CS336 assignment
- **`Reconciling Kaplan and Chinchilla Scaling Laws.pdf`** - Reference paper

## Quick Test

```bash
cd /Users/brandomiranda/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting
uv run python test_pipeline.py
```

**This validates:**
- CS336 IsoFLOPs method works correctly
- Joint fit method works correctly
- Both find balanced scaling (a ≈ 0.5)
- Plots saved to `plots/`
- Data saved to `../../data/`

## Usage for CS336

```python
from isoflops_scaling import *

# Load your CS336 data
df = load_isoflops_data("data/isoflops_curves.json")

# Extract and fit
points = extract_isoflops_points(df)
law = fit_isoflops_scaling_law(points)

# Get predictions for 10^23 and 10^24 FLOPs
N_23, D_23 = law.predict_both(1e23)
N_24, D_24 = law.predict_both(1e24)

print(f"For 10²³ FLOPs: N = {N_23:.3e}, D = {D_23:.3e}")
print(f"For 10²⁴ FLOPs: N = {N_24:.3e}, D = {D_24:.3e}")
```

## Directory Structure

```
dark-matter2/
├── data/
│   └── cs336_test_data.json         # Test data
└── experiments/00_prepare_data_for_scaling_law_fitting/
    ├── isoflops_scaling.py          # CS336 implementation
    ├── scaling_laws_fit.py          # Joint fit implementation
    ├── test_pipeline.py             # Test script
    ├── plots/
    │   ├── cs336_scaling_laws.png   # CS336 validation
    │   └── joint_fit_validation.png # Joint fit validation
    └── README.md                    # This file
```

Simple and correct! ✓
