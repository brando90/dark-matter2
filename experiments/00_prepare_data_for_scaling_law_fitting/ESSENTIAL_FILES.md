# Essential Files for CS336 Scaling Laws

## 🎯 What You Actually Need

### Core Files (KEEP):
1. **`isoflops_scaling.py`** - Main CS336 implementation
2. **`test_cs336_pipeline.py`** - Simple end-to-end test
3. **`cs336_scaling_law_validation.png`** - Validation plot
4. **`test_data.json`** - Example CS336 data format

### Optional (for advanced use):
5. **`scaling_laws_fit.py`** - Chinchilla Approach 3 (joint fit)

### Documentation (KEEP ONE):
6. **This file** - Quick reference

---

## 🚀 Quick Start

### Test the Pipeline:
```bash
cd /Users/brandomiranda/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting
uv run python test_cs336_pipeline.py
```

**This does everything:**
- Creates test data in CS336 format
- Runs the IsoFLOPs pipeline
- Generates required plots
- Makes predictions for 10²³ and 10²⁴ FLOPs
- Validates results

### Use with Your Data:
```python
from isoflops_scaling import *

# Load your CS336 data
df = load_isoflops_data("your_data.json")

# Extract optimal points
points = extract_isoflops_points(df)

# Fit scaling law
law = fit_isoflops_scaling_law(points)

# Make predictions
N_opt, D_opt = law.predict_both(1e23)

# Generate plots
plot_scaling_law(points, law, extrapolate_to=1e24, save_path="results.png")
```

---

## 📊 CS336 Data Format

Your JSON should look like:
```json
[
  {
    "parameters": 49999999,
    "compute_budget": 6e18,
    "final_loss": 7.192
  },
  ...
]
```

---

## 📁 Files to Delete (Optional Cleanup)

**Test/Validation Scripts (redundant):**
- `test_with_csv.py`
- `test_both_approaches.py`
- `validate_with_plots.py`
- `statistical_validation.py`

**Old Validation Plots:**
- `joint_fit_validation.png`
- `isoflops_validation.png`
- `isoflops_scaling_law.png`
- `statistical_validation.png`

**Documentation (too much detail):**
- `COMPARISON_CHINCHILLA_APPROACHES.md`
- `IMPLEMENTATION_SUMMARY.md`
- `FINAL_SUMMARY.md`
- `README.md` (unless you want the detailed version)

**Example Data:**
- `example_ladder.csv` (replaced by test_data.json)

---

## ✅ What test_cs336_pipeline.py Validates

1. **Data loading works** - Reads CS336 JSON format ✓
2. **IsoFLOPs extraction works** - Finds optimal N for each C ✓
3. **Power law fitting works** - Fits N ∝ C^a and D ∝ C^b ✓
4. **Constraint satisfied** - a + b = 1.0 ✓
5. **Balanced scaling** - a ≈ 0.5 ✓
6. **Predictions work** - Extrapolates to 10²³ and 10²⁴ FLOPs ✓
7. **Plots generate** - Creates CS336-required visualizations ✓

---

## 🎓 For CS336 Assignment Submission

**You need:**
1. Your predictions (from `law.predict_both()`)
2. The plot (`cs336_scaling_law_validation.png`)
3. Your writeup explaining the methodology

**The plot shows:**
- Left: Model size scaling with fitted power law
- Right: Dataset size scaling with fitted power law
- Extrapolation to 10²³ and 10²⁴ FLOPs marked
- Data points from your runs

---

## 🧹 Cleanup Command (Optional)

To remove redundant files:
```bash
cd /Users/brandomiranda/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting

# Remove redundant test scripts
rm -f test_with_csv.py test_both_approaches.py validate_with_plots.py statistical_validation.py

# Remove old plots
rm -f joint_fit_validation.png isoflops_validation.png statistical_validation.png

# Remove excessive docs
rm -f COMPARISON_CHINCHILLA_APPROACHES.md IMPLEMENTATION_SUMMARY.md FINAL_SUMMARY.md

# Remove old example
rm -f example_ladder.csv
```

**After cleanup, you'll have:**
```
00_prepare_data_for_scaling_law_fitting/
├── isoflops_scaling.py              # Main implementation
├── test_cs336_pipeline.py            # Simple test
├── scaling_laws_fit.py               # Optional: advanced method
├── test_data.json                    # Example CS336 format
├── cs336_scaling_law_validation.png  # Validation plot
└── ESSENTIAL_FILES.md               # This file
```

Clean, minimal, and everything works! ✅

