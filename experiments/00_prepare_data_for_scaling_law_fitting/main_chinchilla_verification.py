"""
Chinchilla Scaling Law Verification - Launcher

Thin launcher for Chinchilla replication verification.
All fitting/verification logic is in py_src/chinchilla_verification.py
Expected: E≈1.82, A≈482, α≈0.34, B≈2085, β≈0.36, D/N≈20
"""
import sys
import logging
from pathlib import Path

import fire
import wandb

# Ensure repo root is on sys.path for `py_src` imports when run from subdir
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_src.chinchilla_verification import (
    run_chinchilla_verification,
    generate_synthetic_data_chinchilla_replication,
    CHINCHILLA_REPLICATION_PARAMS
)

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

If you ran the synthetic Chinchilla replication, expected targets are E≈1.82, A≈482, α≈0.34, B≈2085, β≈0.36, D/N≈20.
For other datasets (resolving discrepancies, Hitchhiker/ColPret, etc.), these targets do not apply—focus on plot shapes and R².

- `fitted_vs_true_scaling_law.png`: Points should hug y=x on log-log axes; large curvature/spread ⇒ poor fit.
- `chinchilla_verification_residuals.png`: Residuals centered at 0 with symmetric cloud and bell-ish histogram; funnels/skew ⇒ bias/misspecification.
- `kaplan_chinchilla_compute_loss.png`: Offset (solid) curve should outperform no-offset (dashed); RMSE(solid) ≪ RMSE(dashed).
- `isoflops_model_size_vs_compute.png`: Straight line in log-log; slope should be plausible (≈β/(α+β) if model-data joint law holds).
- `isoflops_dataset_size_vs_compute.png`: Complements model-size plot; slopes should sum to ≈1 if C∝N·D holds.
- `verification.log` / `scaling_laws_fit.log`: Inspect fitted params and R². For non-Chinchilla data, ignore ✓/✗ against Chinchilla targets.
- `fit_summary.txt` (if present): One-line sanity check of fitted params and metrics.

Treat the fit as suspect if plots show curvature/funnels, R² is low, or offset vs no-offset is not clearly separated.
"""
    readme.write_text(content)
    logger.info(f"Wrote plots README to {readme}")


def _main(
    mode: str = "online",
    n_points: int = 400,
    seed: int = 42,
    data_csv: str = None,
    plots_suffix: str = None,
    wandb_project: str = "dark-matter2-chinchilla-verification",
    wandb_entity: str = "brando-su"
):
    """
    Run Chinchilla replication verification.
    
    Args:
        mode: 'online' to log to wandb, 'dryrun' to skip wandb
        n_points: Number of synthetic training runs
        seed: Random seed
        data_csv: Optional path to CSV with columns [N_params, tokens_D, loss, flops_C]; if provided, load instead of generating
        plots_suffix: Optional suffix for plots/logs subdir (e.g., model family); if None, uses default plots dir
        wandb_project: W&B project name
        wandb_entity: W&B entity (brando's wandb)
    """
    # Paths
    default_data = Path.home() / "dark-matter2" / "data" / "chinchilla_verification_data.csv"
    data_path = Path(data_csv) if data_csv else default_data
    exp_dir = Path(__file__).parent
    plots_dir = exp_dir / f"plots_out_{plots_suffix}" if plots_suffix else exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # File logging to plots dir
    file_handler = logging.FileHandler(plots_dir / "verification.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Initialize wandb
    if mode == "online":
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "n_points": n_points,
                "seed": seed,
                **{f"target_{k}": v for k, v in CHINCHILLA_REPLICATION_PARAMS.items()}
            }
        )
        logger.info(f"W&B run URL: {wandb.run.url}")
    
    logger.info("=== Chinchilla Replication Verification ===")
    
    # Run verification
    load_only = data_csv is not None
    results = run_chinchilla_verification(data_path, plots_dir, n_points, seed, load_only=load_only)
    
    # Log to wandb
    if mode == "online":
        wandb.log({
            "fitted_E": results['fitted'].E,
            "fitted_A": results['fitted'].A,
            "fitted_alpha": results['fitted'].alpha,
            "fitted_B": results['fitted'].B,
            "fitted_beta": results['fitted'].beta,
            "D_N_ratio": results['avg_ratio'],
            "compute_split_a": results['opt']['a'],
            "r_squared": results['r_squared'],
            "rmse": results['rmse'],
            "all_params_passed": results['all_passed'],
            "ratio_ok": results['ratio_ok']
        })
        wandb.log({"residuals_plot": wandb.Image(str(plots_dir / "chinchilla_verification_residuals.png"))})
        wandb.finish()
    
    # Final summary (compact - details already in DELIVERABLES above)
    logger.info("\n" + "="*60)
    logger.info("=== VERIFICATION RESULT ===")
    logger.info("="*60)
    
    # Key metrics
    logger.info(f"D/N ratio: {results['avg_ratio']:.1f} (target ~20) {'✓' if results['ratio_ok'] else '✗'}")
    logger.info(f"R²: {results['r_squared']:.4f} {'✓' if results['r_squared'] > 0.95 else '✗'}")
    n_passed = sum(1 for r in results['verification'].values() if r['passed'])
    logger.info(f"Params matched: {n_passed}/5")
    
    if results['ratio_ok'] and results['r_squared'] > 0.95:
        logger.info("✅ VERIFICATION SUCCESSFUL")
    else:
        logger.info("⚠️  VERIFICATION INCOMPLETE - review plots above")
    logger.info("="*60)

    # Write per-run README for quick inspection
    data_label = data_csv if data_csv else "synthetic"
    write_plots_readme(plots_dir, results, data_label)
    
    # Remove file handler to avoid duplicate logs on repeated runs
    logger.removeHandler(file_handler)


if __name__ == "__main__":
    fire.Fire(_main)
