#!/usr/bin/env python3
"""
Unified Scaling Laws Fitting for Multiple Datasets

This script can fit scaling laws on both:
1. Chinchilla verification data (original format: N_params, tokens_D, loss, flops_C)
2. Ludwig scaling laws data (29-column format from resolving-scaling-law-discrepancies)

Usage: python unified_scaling_laws_fit.py --dataset chinchilla|chinchilla_ludwig|ludwig
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# Add py_src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from py_src directory
import importlib.util
py_src_path = Path.home() / "dark-matter2" / "py_src"

spec = importlib.util.spec_from_file_location("chinchilla_verification", py_src_path / "chinchilla_verification.py")
chinchilla_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chinchilla_module)

spec = importlib.util.spec_from_file_location("scaling_laws_fit", py_src_path / "scaling_laws_fit.py")
scaling_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling_module)

run_chinchilla_verification = chinchilla_module.run_chinchilla_verification
CHINCHILLA_REPLICATION_PARAMS = chinchilla_module.CHINCHILLA_REPLICATION_PARAMS
JointParams = scaling_module.JointParams


def load_and_prepare_data(dataset: str) -> pd.DataFrame:
    """
    Load and prepare data for scaling laws fitting based on dataset type.

    Args:
        dataset: One of 'chinchilla', 'chinchilla_ludwig', 'ludwig'

    Returns:
        DataFrame with columns [N_params, tokens_D, loss, flops_C]
    """
    data_dir = Path.home() / "dark-matter2" / "data"

    if dataset == 'chinchilla':
        # Original chinchilla format
        csv_path = data_dir / "chinchilla_verification_data.csv"
        print(f"Loading original chinchilla data: {csv_path}")
        df = pd.read_csv(csv_path)
        required_cols = ['N_params', 'tokens_D', 'loss', 'flops_C']
        assert all(col in df.columns for col in required_cols), f"Missing required columns: {required_cols}"

    elif dataset == 'chinchilla_ludwig':
        # Chinchilla data converted to Ludwig format
        csv_path = data_dir / "chinchilla_verification_data_ludwig_format.csv"
        print(f"Loading chinchilla data in Ludwig format: {csv_path}")
        df = pd.read_csv(csv_path)

        # Extract scaling law parameters from Ludwig format
        # num_params = width * depth, tokens = max_step * seq_len * bs (approximate)
        df['N_params'] = df['width'] * df['depth']
        df['tokens_D'] = df['max_step'] * df['seq_len'] * df['bs']
        df['loss'] = df['final_val_loss']
        df['flops_C'] = 6.0 * df['N_params'] * df['tokens_D']  # Chinchilla FLOP estimate

        # Keep only required columns
        df = df[['N_params', 'tokens_D', 'loss', 'flops_C']].copy()

    elif dataset == 'ludwig':
        # Ludwig scaling laws data
        csv_path = data_dir / "data_resolving_discrepancies" / "experiment_results_summary.csv"
        print(f"Loading Ludwig scaling laws data: {csv_path}")
        df = pd.read_csv(csv_path)

        # Extract scaling law parameters from Ludwig format
        df['N_params'] = df['width'] * df['depth']
        df['tokens_D'] = df['max_step'] * df['seq_len'] * df['bs']
        df['loss'] = df['final_val_loss']
        df['flops_C'] = 6.0 * df['N_params'] * df['tokens_D']  # Chinchilla FLOP estimate

        # Keep only required columns
        df = df[['N_params', 'tokens_D', 'loss', 'flops_C']].copy()

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Clean data
    df = df.dropna(subset=['N_params', 'tokens_D', 'loss'])
    df = df[df['N_params'] > 0]
    df = df[df['tokens_D'] > 0]
    df = df[df['loss'] > 0]

    print(f"Prepared {len(df)} data points for scaling laws fitting")
    print(f"  N_params range: {df['N_params'].min():.1e} - {df['N_params'].max():.1e}")
    print(f"  tokens_D range: {df['tokens_D'].min():.1e} - {df['tokens_D'].max():.1e}")
    print(f"  loss range: {df['loss'].min():.3f} - {df['loss'].max():.3f}")

    return df


def fit_scaling_laws(df: pd.DataFrame, plots_dir: Path) -> dict:
    """
    Fit scaling laws on the prepared data.

    Args:
        df: DataFrame with columns [N_params, tokens_D, loss, flops_C]
        plots_dir: Directory to save plots

    Returns:
        Dictionary with fitting results
    """
    print("\n🔬 Fitting scaling laws...")
    print(f"  Data points: {len(df)}")
    print(".1e")
    print(".1e")
    print(".3f")

    # Run chinchilla verification (includes fitting)
    results = run_chinchilla_verification(df, plots_dir, n_points=0, seed=42, load_only=True)

    return results


def main():
    """Main function to run unified scaling laws fitting."""
    parser = argparse.ArgumentParser(description="Unified Scaling Laws Fitting")
    parser.add_argument('--dataset', choices=['chinchilla', 'chinchilla_ludwig', 'ludwig'],
                       default='chinchilla', help='Dataset to use for fitting')
    parser.add_argument('--plots-dir', type=str, default=None,
                       help='Directory to save plots (default: plots_out_{dataset})')

    args = parser.parse_args()

    print(f"🚀 Unified Scaling Laws Fitting - Dataset: {args.dataset}")

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Setup directories
    exp_dir = Path(__file__).parent
    plots_dir_name = args.plots_dir or f"plots_out_{args.dataset}"
    plots_dir = exp_dir / plots_dir_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    file_handler = logging.FileHandler(plots_dir / "scaling_laws_fit.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    try:
        # Load and prepare data
        df = load_and_prepare_data(args.dataset)

        # Fit scaling laws
        results = fit_scaling_laws(df, plots_dir)

        # Log results
        logger.info("\n" + "="*60)
        logger.info(f"=== SCALING LAWS FIT RESULTS ({args.dataset.upper()}) ===")
        logger.info("="*60)
        logger.info(f"Fitted parameters:")
        logger.info(f"  E (irreducible loss): {results['fitted'].E:.3f}")
        logger.info(f"  A (model size coeff): {results['fitted'].A:.1f}")
        logger.info(f"  α (model size exponent): {results['fitted'].alpha:.3f}")
        logger.info(f"  B (data size coeff): {results['fitted'].B:.1f}")
        logger.info(f"  β (data size exponent): {results['fitted'].beta:.3f}")
        logger.info(f"Data/Model ratio: {results['avg_ratio']:.1f}")
        logger.info(f"R²: {results['r_squared']:.4f}")
        logger.info(f"RMSE: {results['rmse']:.4f}")

        # Expected chinchilla values for comparison
        expected = CHINCHILLA_REPLICATION_PARAMS
        logger.info(f"\nExpected Chinchilla values:")
        logger.info(f"  E: {expected['E']:.3f}, A: {expected['A']:.1f}, α: {expected['alpha']:.3f}")
        logger.info(f"  B: {expected['B']:.1f}, β: {expected['beta']:.3f}")

        # Check if parameters are close to expected
        fitted = results['fitted']
        e_ok = abs(fitted.E - expected['E']) < 0.1
        a_ok = abs(fitted.A - expected['A']) / expected['A'] < 0.1
        alpha_ok = abs(fitted.alpha - expected['alpha']) < 0.05
        b_ok = abs(fitted.B - expected['B']) / expected['B'] < 0.1
        beta_ok = abs(fitted.beta - expected['beta']) < 0.05

        all_params_ok = all([e_ok, a_ok, alpha_ok, b_ok, beta_ok])
        logger.info(f"\nParameter check: {'✅ PASS' if all_params_ok else '❌ FAIL'}")

        # Save results summary
        summary_path = plots_dir / "fit_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Data points: {len(df)}\n")
            f.write(f"Fitted E: {fitted.E:.4f}\n")
            f.write(f"Fitted A: {fitted.A:.2f}\n")
            f.write(f"Fitted α: {fitted.alpha:.4f}\n")
            f.write(f"Fitted B: {fitted.B:.2f}\n")
            f.write(f"Fitted β: {fitted.beta:.4f}\n")
            f.write(f"R²: {results['r_squared']:.4f}\n")
            f.write(f"RMSE: {results['rmse']:.4f}\n")
            f.write(f"All params OK: {all_params_ok}\n")

        print("\n✅ Scaling laws fitting completed!")
        print(f"📊 Results saved to: {plots_dir}")
        print(f"📄 Summary: {summary_path}")
        print(f"📈 Plots: {plots_dir}/")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
