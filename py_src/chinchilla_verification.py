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

from .scaling_laws_fit import fit_joint_powerlaw, predict_joint, compute_optimal_from_joint
from .scaling_laws_fit import JointParams

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


def compute_optimal_allocation_for_C(p: JointParams, C_value: float, k: float = 6.0):
    """
    Compute optimal N and D for a given compute budget C ≈ k N D using the closed-form ratio.
    """
    const = ((p.beta * p.B) / (p.alpha * p.A)) ** (1.0 / p.beta)
    exponent = p.beta / (p.alpha + p.beta)
    N_star = ((C_value / k) / const) ** exponent
    D_star = const * (N_star ** (p.alpha / p.beta))
    return N_star, D_star


def plot_kaplan_vs_chinchilla_compute_loss(fitted: JointParams, plots_dir: Path, df: pd.DataFrame):
    """
    Compare compute-loss fits with and without offset (Chinchilla vs Kaplan form) on synthetic optimal data.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    C_grid = np.geomspace(df['flops_C'].min(), df['flops_C'].max(), 80)
    
    losses = []
    for C in C_grid:
        N_star, D_star = compute_optimal_allocation_for_C(fitted, C)
        losses.append(predict_joint(fitted, np.array([N_star]), np.array([D_star]))[0])
    losses = np.array(losses)
    
    # Fit compute-loss power laws with and without offset
    eps = 1e-12
    # With offset: L = E + a*C^{-gamma}
    y_chi = np.log(np.clip(losses - fitted.E, eps, None))
    gamma_chi, log_a_chi = np.polyfit(np.log(C_grid), y_chi, 1)
    gamma_chi = -gamma_chi
    a_chi = np.exp(log_a_chi)
    pred_chi = fitted.E + a_chi * (C_grid ** (-gamma_chi))
    
    # Without offset: L = a*C^{-gamma}
    y_kap = np.log(losses)
    gamma_kap, log_a_kap = np.polyfit(np.log(C_grid), y_kap, 1)
    gamma_kap = -gamma_kap
    a_kap = np.exp(log_a_kap)
    pred_kap = a_kap * (C_grid ** (-gamma_kap))
    
    rmse_chi = np.sqrt(np.mean((pred_chi - losses) ** 2))
    rmse_kap = np.sqrt(np.mean((pred_kap - losses) ** 2))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(C_grid, losses, 'o', alpha=0.5, label='Synthetic optimal (from fitted)')
    ax.loglog(
        C_grid, pred_chi, '-',
        label=f"Chinchilla: L = {fitted.E:.2f} + {a_chi:.2e}·C^(-{gamma_chi:.3f}) (RMSE={rmse_chi:.4f})"
    )
    ax.loglog(
        C_grid, pred_kap, '--',
        label=f"Kaplan: L = {a_kap:.2e}·C^(-{gamma_kap:.3f}) (RMSE={rmse_kap:.4f})"
    )
    ax.set_xlabel('Compute C (FLOPs) [log scale]')
    ax.set_ylabel('Loss [log scale]')
    ax.set_title('Compute–Loss Frontier: Chinchilla (offset) vs Kaplan (no offset)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    out_path = plots_dir / "kaplan_chinchilla_compute_loss.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved compute-loss comparison plot to {out_path}")
    return dict(path=out_path, rmse_chi=rmse_chi, rmse_kap=rmse_kap)


def plot_fitted_vs_true_scaling_law(df: pd.DataFrame, fitted: JointParams, plots_dir: Path):
    """
    Plot fitted scaling law predictions vs true (synthetic) loss values.
    Since we know the exact functional form, these should be nearly identical.
    Uses log-log scale and shows both functional forms in legend.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    L_true = df['loss'].values
    L_pred = predict_joint(fitted, df['N_params'].values, df['tokens_D'].values)
    
    # Target params for legend
    t = CHINCHILLA_REPLICATION_PARAMS
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter with label showing fitted params
    fitted_label = (f'Fitted: L = {fitted.E:.2f} + {fitted.A:.0f}/N^{fitted.alpha:.2f} '
                   f'+ {fitted.B:.0f}/D^{fitted.beta:.2f}')
    ax.scatter(L_true, L_pred, alpha=0.5, s=20, label=fitted_label)
    
    # Perfect fit line with label showing target params
    lims = [min(L_true.min(), L_pred.min()), max(L_true.max(), L_pred.max())]
    target_label = (f'Target: L = {t["E"]:.2f} + {t["A"]:.0f}/N^{t["alpha"]:.2f} '
                   f'+ {t["B"]:.0f}/D^{t["beta"]:.2f}')
    ax.plot(lims, lims, 'r--', linewidth=2, label=target_label)
    
    # Log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('True Loss (synthetic data from target params)')
    ax.set_ylabel('Predicted Loss (fitted model)')
    ax.set_title('Fitted vs True Scaling Law (log-log)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Compute deviations (absolute and relative)
    max_dev = np.max(np.abs(L_pred - L_true))
    mean_dev = np.mean(np.abs(L_pred - L_true))
    rel_err = np.abs(L_pred - L_true) / np.clip(L_true, 1e-12, None)
    rmse_rel = np.sqrt(np.mean(rel_err ** 2))
    max_rel_err = np.max(rel_err)
    
    out_path = plots_dir / "fitted_vs_true_scaling_law.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved fitted vs true plot to {out_path}")
    return dict(path=out_path, max_dev=max_dev, mean_dev=mean_dev, rmse_rel=rmse_rel, max_rel_err=max_rel_err)


def plot_isoflops_frontiers(df: pd.DataFrame, plots_dir: Path):
    """
    Plot IsoFLOPs frontiers for model size and dataset size vs compute budget.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    budgets = np.geomspace(df['flops_C'].min(), df['flops_C'].max(), 12)
    records = []
    for C in budgets:
        mask = (df['flops_C'] >= 0.8 * C) & (df['flops_C'] <= 1.2 * C)
        if mask.any():
            subset = df[mask].sort_values('loss')
            row = subset.iloc[0]
            records.append((C, row['N_params'], row['tokens_D'], row['loss']))
    if len(records) < 3:
        logger.warning("Not enough points to plot IsoFLOPs frontiers (need >=3). Skipping.")
        return None, None
    
    C_vals, N_vals, D_vals, L_vals = map(np.array, zip(*records))
    
    # Fit simple power laws: N = a*C^b, D = a*C^b
    b_N, log_a_N = np.polyfit(np.log(C_vals), np.log(N_vals), 1)
    b_D, log_a_D = np.polyfit(np.log(C_vals), np.log(D_vals), 1)
    a_N, a_D = np.exp(log_a_N), np.exp(log_a_D)
    C_fit = np.geomspace(C_vals.min(), C_vals.max() * 5, 100)
    N_fit = a_N * (C_fit ** b_N)
    D_fit = a_D * (C_fit ** b_D)
    
    # Model-size frontier
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.loglog(C_vals, N_vals, 'o', label='Min-loss runs per budget')
    ax1.loglog(C_fit, N_fit, '-', label=f'Fit: N = {a_N:.2e} * C^{b_N:.3f}')
    ax1.set_xlabel('Compute C (FLOPs)')
    ax1.set_ylabel('Model size N (params)')
    ax1.set_title('IsoFLOPs Frontier: Model Size vs Compute')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    iso_model_path = plots_dir / "isoflops_model_size_vs_compute.png"
    plt.tight_layout()
    plt.savefig(iso_model_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved IsoFLOPs model-size frontier to {iso_model_path}")
    
    # Dataset-size frontier
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.loglog(C_vals, D_vals, 'o', label='Min-loss runs per budget')
    ax2.loglog(C_fit, D_fit, '-', label=f'Fit: D = {a_D:.2e} * C^{b_D:.3f}')
    ax2.set_xlabel('Compute C (FLOPs)')
    ax2.set_ylabel('Dataset size D (tokens)')
    ax2.set_title('IsoFLOPs Frontier: Dataset Size vs Compute')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()
    iso_data_path = plots_dir / "isoflops_dataset_size_vs_compute.png"
    plt.tight_layout()
    plt.savefig(iso_data_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved IsoFLOPs dataset-size frontier to {iso_data_path}")
    
    return dict(path=iso_model_path, exponent=b_N), dict(path=iso_data_path, exponent=b_D)


def run_chinchilla_verification(
    data_path,
    plots_dir: Path,
    n_points: int = 400,
    seed: int = 42,
    load_only: bool = False
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
    plots_dir = Path(plots_dir)
    df = None

    # Accept DataFrame or path
    if isinstance(data_path, pd.DataFrame):
        df = data_path.copy()
        data_source = "dataframe"
    else:
        data_path = Path(data_path)
        if data_path.exists():
            logger.info(f"Loading existing data from {data_path}")
            df = pd.read_csv(data_path)
        elif not load_only:
            logger.info(f"Generating {n_points} synthetic runs...")
            df = generate_synthetic_data_chinchilla_replication(n_points, seed, data_path)
        else:
            raise FileNotFoundError(f"Data CSV not found and load_only=True: {data_path}")
        data_source = str(data_path)

    logger.info(f"Using data source: {data_source}")
    
    # Ensure required columns
    required = {'N_params', 'tokens_D', 'loss'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {data_path}")
    if 'flops_C' not in df.columns:
        df['flops_C'] = 6.0 * df['N_params'] * df['tokens_D']
    
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
    compute_loss_info = plot_kaplan_vs_chinchilla_compute_loss(fitted, plots_dir, df)
    iso_model_info, iso_data_info = plot_isoflops_frontiers(df, plots_dir)
    fitted_vs_true_info = plot_fitted_vs_true_scaling_law(df, fitted, plots_dir)
    
    # Compute metrics
    L_pred = predict_joint(fitted, df['N_params'].values, df['tokens_D'].values)
    residuals = df['loss'].values - L_pred
    r_squared = 1 - (np.sum(residuals ** 2) / np.sum((df['loss'].values - np.mean(df['loss'].values)) ** 2))
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    logger.info(f"\nFit quality: R²={r_squared:.6f}, RMSE={rmse:.6f}")
    
    # Build deliverables with detailed self-contained reasons
    residuals_path = plots_dir / 'chinchilla_verification_residuals.png'
    t = CHINCHILLA_REPLICATION_PARAMS  # target params for reference
    
    # Compute relative errors for plot1
    rel_errors = np.abs(L_pred - df['loss'].values) / df['loss'].values
    rmse_rel = np.sqrt(np.mean(rel_errors**2))
    max_rel_error = np.max(rel_errors)
    
    # Get exponents for cross-check
    a_N = iso_model_info['exponent'] if iso_model_info else 0.5
    a_D = iso_data_info['exponent'] if iso_data_info else 0.5
    
    reason1 = (
        f"Synthetic L uses the same form L=E+A/N^α+B/D^β with 2% log-normal noise, so rel errors should be tiny (RMSE_rel≈0.02, max_rel<0.10). "
        f"If points stray far from y=x (large rel errors/curvature), the fit or functional form is wrong."
    )
    
    reason2 = (
        f"2% multiplicative noise ⇒ residuals ~ Normal(0, 0.02·loss); expect symmetric cloud around 0 and a bell-shaped histogram. "
        f"Slope or funnel = bias/heteroscedasticity; skew/bimodality = model misspecification or outliers."
    )
    
    reason3 = (
        f"True loss floors at E≈{t['E']}; Chinchilla fits L→E as C→∞, Kaplan forces L→0, so Kaplan underfits the high-compute tail. "
        f"We expect Chinchilla RMSE ≪ Kaplan; if similar, the offset isn’t working or there’s a bug."
    )
    
    reason4 = (
        f"IsoFLOPs theory: N_opt ∝ C^a with a=α/(α+β)≈0.486; straight line slope ≈0.5 in log-log and a_N+a_D≈1 (C∝N·D). "
        f"Curvature or slope far from 0.5/ a_N+a_D≠1 means bad minima or fit."
    )
    
    reason5 = (
        f"D_opt exponent should be b=β/(α+β)≈0.514 and complement N (1−a_N); expect straight log-log line, a_N+a_D≈1. "
        f"Large deviations/curvature mean the IsoFLOPs minima or fit are off; consistency with plot4 validates D/N≈20 guidance."
    )
    
    deliverables = [
        {
            'name': 'fitted_vs_true',
            'path': fitted_vs_true_info['path'],
            'check': f"Points tight on y=x; RMSE_rel={rmse_rel:.4f} (expect ≈0.02), max_rel={max_rel_error:.4f} (expect <0.10)",
            'reason': reason1
        },
        {
            'name': 'residuals',
            'path': residuals_path,
            'check': "Residuals vs fitted loss: symmetric around y=0, no funnel. Histogram: bell-shaped at 0.",
            'reason': reason2
        },
        {
            'name': 'kaplan_vs_chinchilla',
            'path': compute_loss_info['path'],
            'check': f"Chinchilla RMSE ({compute_loss_info['rmse_chi']:.6f}) ≪ Kaplan RMSE ({compute_loss_info['rmse_kap']:.4f})",
            'reason': reason3
        },
    ]
    if iso_model_info and iso_data_info:
        deliverables.extend([
            {
                'name': 'isoflops_model',
                'path': iso_model_info['path'],
                'check': f"N_opt ∝ C^{a_N:.3f} (expect ≈0.5), straight line in log-log, a_N+a_D={a_N+a_D:.3f} (expect ≈1)",
                'reason': reason4
            },
            {
                'name': 'isoflops_data',
                'path': iso_data_info['path'],
                'check': f"D_opt ∝ C^{a_D:.3f} (expect ≈0.5), straight line in log-log, consistent with plot4",
                'reason': reason5
            },
        ])
    
    # Print as numbered list with wrapped reasons
    import textwrap
    logger.info("\n" + "="*80)
    logger.info("DELIVERABLES TO INSPECT")
    logger.info("="*80)
    for i, d in enumerate(deliverables, 1):
        logger.info(f"\nplot{i}: {d['name']}")
        logger.info(f"  Path:  {d['path']}")
        logger.info(f"  Check: {d['check']}")
        logger.info("  Reason:")
        for line in textwrap.wrap(d['reason'], width=76):
            logger.info(f"    {line}")

    return {
        'fitted': fitted,
        'verification': verification,
        'avg_ratio': avg_ratio,
        'opt': opt,
        'r_squared': r_squared,
        'rmse': rmse,
        'all_passed': all(r['passed'] for r in verification.values()),
        'ratio_ok': 15 <= avg_ratio <= 30,
        'compute_loss_info': compute_loss_info,
        'iso_model_info': iso_model_info,
        'iso_data_info': iso_data_info,
        'fitted_vs_true_info': fitted_vs_true_info
    }
