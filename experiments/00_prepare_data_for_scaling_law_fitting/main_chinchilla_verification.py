"""
Chinchilla Scaling Law Verification - Launcher

Thin launcher for Chinchilla replication verification.
All fitting/verification logic is in py_src/chinchilla_verification.py
Expected: E≈1.82, A≈482, α≈0.34, B≈2085, β≈0.36, D/N≈20
"""
import sys
import logging
from pathlib import Path

# Add py_src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fire
import wandb

from py_src.chinchilla_verification import (
    run_chinchilla_verification,
    generate_synthetic_data_chinchilla_replication,
    CHINCHILLA_REPLICATION_PARAMS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _main(
    mode: str = "online",
    n_points: int = 400,
    seed: int = 42,
    wandb_project: str = "dark-matter2-chinchilla-verification",
    wandb_entity: str = "brando-su"
):
    """
    Run Chinchilla replication verification.
    
    Args:
        mode: 'online' to log to wandb, 'dryrun' to skip wandb
        n_points: Number of synthetic training runs
        seed: Random seed
        wandb_project: W&B project name
        wandb_entity: W&B entity (brando's wandb)
    """
    # Paths
    data_path = Path.home() / "dark-matter2" / "data" / "chinchilla_verification_data.csv"
    plots_dir = Path(__file__).parent / "plots"
    
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
    results = run_chinchilla_verification(data_path, plots_dir, n_points, seed)
    
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


if __name__ == "__main__":
    fire.Fire(_main)
