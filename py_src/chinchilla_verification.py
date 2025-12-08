"""
Chinchilla replication verification utilities.
Contains data generation and verification functions for the Chinchilla scaling law.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scaling_laws_fit import fit_joint_powerlaw, predict_joint, compute_optimal_from_joint

logger = logging.getLogger(__name__)

# Chinchilla replication paper (Besiroglu et al.) corrected coefficients
CHINCHILLA_REPLICATION_PARAMS = {'E': 1.82, 'A': 482.0, 'alpha': 0.34, 'B': 2085.0, 'beta': 0.36}


def generate_synthetic_data_chinchilla_replication(
    n_points: int = 400,
    seed: int = 42,
    save_path: Path = None
) -> pd.DataFrame:
    """
    Generate synthetic Chinchilla-scale training runs using ground truth from replication paper.
    
    Args:
        n_points: Number of synthetic training runs
        seed: Random seed for reproducibility
        save_path: Optional path to save CSV
    
    Returns:
        DataFrame with columns: N_params, tokens_D, loss, flops_C
    """
    rng = np.random.default_rng(seed)
    
    # Chinchilla scale: N∈[70M,16B], D∈[5B,500B]
    log_N = rng.uniform(np.log(70e6), np.log(16e9), n_points)
    log_D = rng.uniform(np.log(5e9), np.log(500e9), n_points)
    N, D = np.exp(log_N), np.exp(log_D)
    
    # Ground truth loss with multiplicative noise
    E, A, alpha, B, beta = CHINCHILLA_REPLICATION_PARAMS.values()
    L = (E + A / (N ** alpha) + B / (D ** beta)) * np.exp(rng.normal(0.0, 0.02, n_points))
    
    df = pd.DataFrame({'N_params': N, 'tokens_D': D, 'loss': L, 'flops_C': 6.0 * N * D})
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved synthetic data to {save_path}")
    
    return df


def verify_chinchilla_coefficients(fitted, tolerance: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Compare fitted parameters vs Chinchilla replication targets.
    
    Args:
        fitted: JointParams from fit_joint_powerlaw
        tolerance: Dict of relative error tolerances per param
    
    Returns:
        Dict with fitted/target/error/passed for each parameter
    """
    if tolerance is None:
        tolerance = {'E': 0.10, 'A': 0.20, 'alpha': 0.10, 'B': 0.25, 'beta': 0.15}
    
    results = {}
    for param in ['E', 'A', 'alpha', 'B', 'beta']:
        fitted_val = getattr(fitted, param)
        target_val = CHINCHILLA_REPLICATION_PARAMS[param]
        rel_error = abs(fitted_val - target_val) / target_val
        results[param] = {
            'fitted': fitted_val,
            'target': target_val,
            'rel_error': rel_error,
            'passed': rel_error <= tolerance[param]
        }
    return results


def plot_chinchilla_residuals(df: pd.DataFrame, fitted, save_path: Path):
    """Plot residuals (L_actual - L_pred) vs L_pred"""
    L_pred = predict_joint(fitted, df['N_params'].values, df['tokens_D'].values)
    residuals = df['loss'].values - L_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(L_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Loss')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals (should be symmetric around 0)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved residuals plot to {save_path}")


def run_chinchilla_verification(
    data_path: Path,
    plots_dir: Path,
    n_points: int = 400,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run full Chinchilla verification pipeline.
    
    1. Generate/load synthetic data
    2. Fit L(N,D) = E + A/N^α + B/D^β
    3. Verify coefficients match Chinchilla replication targets
    4. Generate diagnostic plots
    
    Args:
        data_path: Path to save/load CSV data
        plots_dir: Directory for plots
        n_points: Number of synthetic runs
        seed: Random seed
    
    Returns:
        Dict with verification results and metrics
    """
    data_path = Path(data_path)
    plots_dir = Path(plots_dir)
    
    # Generate or load data
    if data_path.exists():
        logger.info(f"Loading existing data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info(f"Generating {n_points} synthetic runs...")
        df = generate_synthetic_data_chinchilla_replication(n_points, seed, data_path)
    
    # Fit scaling law
    logger.info("Fitting L(N,D) = E + A/N^α + B/D^β...")
    fitted = fit_joint_powerlaw(df['N_params'].values, df['tokens_D'].values, df['loss'].values)
    
    # Verify coefficients
    verification = verify_chinchilla_coefficients(fitted)
    
    # Log results
    logger.info("\n=== Fitted Parameters ===")
    for param, result in verification.items():
        status = "✓" if result['passed'] else "✗"
        logger.info(f"{param:6s}: {result['fitted']:8.2f} (target: {result['target']:8.2f}, "
                   f"error: {result['rel_error']*100:5.1f}%) {status}")
    
    # Compute D/N ratio
    opt = compute_optimal_from_joint(fitted)
    test_sizes = [100e6, 1e9, 10e9]
    ratios = [((fitted.beta * fitted.B) / (fitted.alpha * fitted.A)) ** (1.0 / fitted.beta) * 
              (N ** ((fitted.alpha - fitted.beta) / fitted.beta)) for N in test_sizes]
    avg_ratio = np.mean(ratios)
    
    logger.info(f"\nD/N ratio: {avg_ratio:.1f} (target: ~20)")
    logger.info(f"Compute split: a={opt['a']:.4f}, b={opt['b']:.4f}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    plot_chinchilla_residuals(df, fitted, plots_dir / "chinchilla_verification_residuals.png")
    
    # Compute metrics
    L_pred = predict_joint(fitted, df['N_params'].values, df['tokens_D'].values)
    residuals = df['loss'].values - L_pred
    r_squared = 1 - (np.sum(residuals ** 2) / np.sum((df['loss'].values - np.mean(df['loss'].values)) ** 2))
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    logger.info(f"\nFit quality: R²={r_squared:.6f}, RMSE={rmse:.6f}")
    
    # Print file inspection instructions
    logger.info("\n" + "="*60)
    logger.info("FILES TO INSPECT")
    logger.info("="*60)
    
    logger.info(f"\nData CSV at path: {data_path}")
    logger.info(f"Check: Open and verify it has columns [N_params, tokens_D, loss, flops_C] with {len(df)} rows")
    
    residuals_path = plots_dir / 'chinchilla_verification_residuals.png'
    logger.info(f"\nResiduals plot at path: {residuals_path}")
    logger.info(f"Check: Open it and it should have LEFT panel with scatter around y=0 (symmetric cloud, no funnel) and RIGHT panel with bell-shaped histogram centered at 0")
    
    return {
        'fitted': fitted,
        'verification': verification,
        'avg_ratio': avg_ratio,
        'opt': opt,
        'r_squared': r_squared,
        'rmse': rmse,
        'all_passed': all(r['passed'] for r in verification.values()),
        'ratio_ok': 15 <= avg_ratio <= 30
    }

