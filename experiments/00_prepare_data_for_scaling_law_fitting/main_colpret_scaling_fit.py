#!/usr/bin/env python3
"""
Run scaling-law fit on ColPret (Hitchhiker) data converted to resolving_discrepancies schema.

Steps:
1) Ensure ColPret data is converted via convert_colpret_to_resolving_format.py (output CSV path below).
2) Load the converted CSV, derive N_params/tokens_D/loss.
3) Run the Chinchilla-style fit and write plots + README into plots_out_colpret/.
"""

import argparse
import sys
import logging
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_src.chinchilla_verification import run_chinchilla_verification


def write_plots_readme(plots_dir: Path, results, data_label: str):
    plots_dir = Path(plots_dir)
    readme = plots_dir / "README_plots.md"
    fitted = results['fitted']
    content = f"""# Scaling-Law Plots Guide ({data_label})

Fitted params: E={fitted.E:.3f}, A={fitted.A:.2f}, α={fitted.alpha:.3f}, B={fitted.B:.2f}, β={fitted.beta:.3f}
Fit quality: R²={results['r_squared']:.4f}, RMSE={results['rmse']:.4f}, D/N≈{results['avg_ratio']:.1f}

If this is the synthetic Chinchilla replication, expected targets are E≈1.82, A≈482, α≈0.34, B≈2085, β≈0.36, D/N≈20.
For Hitchhiker/ColPret or other datasets, these targets do not apply—use plot shapes and R² to judge fit quality.

- `fitted_vs_true_scaling_law.png`: Points should hug y=x on log-log axes; big curvature or spread ⇒ bad fit.
- `chinchilla_verification_residuals.png`: Residuals centered at 0 with symmetric cloud and bell-ish histogram; funnels/skew ⇒ bias.
- `kaplan_chinchilla_compute_loss.png`: Solid (offset) curve should sit on points; its RMSE must be ≪ dashed (no-offset) curve.
- `isoflops_model_size_vs_compute.png`: Log-log line slope should be plausible (≈β/(α+β) if joint law holds); should be straight, not curved.
- `isoflops_dataset_size_vs_compute.png`: Complementary to model-size plot; slopes should sum to ≈1 if C∝N·D holds.
- `verification.log` / `scaling_laws_fit.log`: Check fitted params and R². For non-Chinchilla data, ignore ✓/✗ against Chinchilla targets.
- `fit_summary.txt` (if present): One-line sanity check of fitted params and metrics.

If D/N is far from ~20 or any slope/plots look off, treat the fit as suspect.
"""
    readme.write_text(content)


def load_converted_colpret(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"width", "depth", "max_step", "seq_len", "bs", "final_val_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Converted ColPret CSV missing columns: {missing}")

    df["N_params"] = df["width"] * df["depth"]
    df["tokens_D"] = df["max_step"] * df["seq_len"] * df["bs"]
    df["loss"] = df["final_val_loss"]
    out = df[["N_params", "tokens_D", "loss"]].copy()
    return out


def main():
    parser = argparse.ArgumentParser(description="Scaling-law fit on ColPret (converted) data")
    parser.add_argument(
        "--converted-csv",
        type=str,
        default=str(REPO_ROOT / "data" / "data_hitchhickers" / "colpret_resolving_format.csv"),
        help="Path to ColPret data converted to resolving_discrepancies schema",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "00_prepare_data_for_scaling_law_fitting" / "plots_out_colpret"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    csv_path = Path(args.converted_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Converted ColPret CSV not found at {csv_path}. Run convert_colpret_to_resolving_format.py first.")

    df = load_converted_colpret(csv_path)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Logging to file
    log_path = plots_dir / "scaling_laws_fit.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    print(f"Running scaling-law fit on ColPret converted data ({len(df)} rows)...")
    results = run_chinchilla_verification(df, plots_dir, n_points=0, seed=42, load_only=True)
    write_plots_readme(plots_dir, results, data_label="colpret")

    summary_path = plots_dir / "fit_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Dataset: colpret (hitchhikers)\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"E: {results['fitted'].E:.6f}\n")
        f.write(f"A: {results['fitted'].A:.6f}\n")
        f.write(f"alpha: {results['fitted'].alpha:.6f}\n")
        f.write(f"B: {results['fitted'].B:.6f}\n")
        f.write(f"beta: {results['fitted'].beta:.6f}\n")
        f.write(f"R2: {results['r_squared']:.6f}\n")
        f.write(f"RMSE: {results['rmse']:.6f}\n")
        f.write(f"D_over_N: {results['avg_ratio']:.6f}\n")

    logger.info(f"Saved fit summary to {summary_path}")
    print(f"✅ Done. Plots in: {plots_dir}")


if __name__ == "__main__":
    main()
