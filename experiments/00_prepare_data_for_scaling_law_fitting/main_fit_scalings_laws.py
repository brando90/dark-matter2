"""
Scaling Law Fit Launcher (generic datasets, no Chinchilla target checks unless enabled)

All fitting logic is in py_src/chinchilla_verification.py
"""
import sys
import logging
from pathlib import Path

import logging

# Ensure repo root is on sys.path for `py_src` imports when run from subdir
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_src.chinchilla_verification import run_chinchilla_verification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def write_plots_readme(plots_dir: Path, results, data_label: str):
    """Write a concise guide explaining each plot and the correctness checks."""
    plots_dir = Path(plots_dir)
    readme = plots_dir / "README_plots.md"
    fitted = results['fitted']
    content = f"""# Scaling-Law Plots Guide ({data_label})

Fitted params: E={fitted.E:.3f}, A={fitted.A:.2f}, α={fitted.alpha:.3f}, B={fitted.B:.2f}, β={fitted.beta:.3f}
Fit quality: R²={results['r_squared']:.4f}, RMSE={results['rmse']:.4f}, D/N≈{results['avg_ratio']:.1f}

Checks (no ground-truth targets assumed):
- `residuals.png`: Roughly symmetric around 0; strong skew/funnel implies bias or heteroscedasticity.
- `compute_loss_offset_vs_nooffset.png`: Solid (offset) curve should clearly beat dashed (no-offset); otherwise the irreducible-loss term may be misfit.
- `isoflops_model_size_vs_compute.png`: Log–log line should be straight-ish; slope ~β/(α+β) if the joint law holds. Curvature or extreme slope ⇒ unreliable optimum sizing.
- `isoflops_dataset_size_vs_compute.png`: Complements model frontier; slopes should roughly sum to 1 (C∝N·D). Curvature or mismatch to model plot ⇒ unstable minima.
- `verification.log` / `scaling_laws_fit.log`: Inspect fitted params and R²; use domain intuition to judge plausibility.
- `fit_summary.txt` (if present): One-line recap of fitted params and metrics.

Flag as suspect if residuals funnel/skew, isoFLOPs curves bend or have implausible slopes, or offset vs no-offset curves are indistinguishable.
"""
    readme.write_text(content)
    logger.info(f"Wrote plots README to {readme}")


def _main(
    data_csv: str,
    plots_suffix: str = None,
):
    """
    Run scaling-law fit on a CSV (resolving discrepancies or Hitchhiker/ColPret).
    """
    data_path = Path(data_csv)
    exp_dir = Path(__file__).parent
    plots_dir = exp_dir / f"plots_out_{plots_suffix}" if plots_suffix else exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # File logging to plots dir
    file_handler = logging.FileHandler(plots_dir / "verification.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=== SCALING LAW FIT ===")
    logger.info(f"Data: {data_path}")
    
    results = run_chinchilla_verification(
        data_path,
        plots_dir,
        n_points=0,
        seed=42,
        load_only=True,
    )
    
    logger.info("\n" + "="*60)
    logger.info("=== FIT SUMMARY ===")
    logger.info("="*60)
    logger.info(f"E: {results['fitted'].E:.3f}")
    logger.info(f"A: {results['fitted'].A:.3f}")
    logger.info(f"alpha: {results['fitted'].alpha:.3f}")
    logger.info(f"B: {results['fitted'].B:.3f}")
    logger.info(f"beta: {results['fitted'].beta:.3f}")
    logger.info(f"D/N ratio: {results['avg_ratio']:.1f}")
    logger.info(f"R²: {results['r_squared']:.4f}")
    logger.info(f"RMSE: {results['rmse']:.4f}")
    logger.info("="*60)

    # Write per-run README for quick inspection
    data_label = data_path.name
    write_plots_readme(plots_dir, results, data_label)
    
    # Remove file handler to avoid duplicate logs on repeated runs
    logger.removeHandler(file_handler)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scaling law fit")
    parser.add_argument('--data_csv', required=True, help='Path to CSV to load')
    parser.add_argument('--plots_suffix', type=str, default=None, help='Optional suffix for plots/logs dir')
    args = parser.parse_args()
    _main(
        data_csv=args.data_csv,
        plots_suffix=args.plots_suffix,
    )
