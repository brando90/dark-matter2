#!/usr/bin/env python3
"""
Test script demonstrating how to use scaling_laws_fit with a CSV file.
"""

from scaling_laws_fit import (
    load_ladder_csv,
    fit_joint_powerlaw,
    fit_two_term_N_only,
    compute_optimal_from_joint,
    residuals_joint,
    huber_loglike
)
import numpy as np

def main():
    print("=" * 80)
    print("SCALING LAWS FIT - CSV DATA TEST")
    print("=" * 80)
    print()
    
    # Load the example CSV
    csv_path = "example_ladder.csv"
    df = load_ladder_csv(csv_path)
    
    print(f"Loaded {len(df)} data points from {csv_path}")
    print(f"Families: {df['family'].unique()}")
    print(f"N range: [{df['N_params'].min():.2e}, {df['N_params'].max():.2e}]")
    print(f"D range: [{df['tokens_D'].min():.2e}, {df['tokens_D'].max():.2e}]")
    print(f"Loss range: [{df['loss'].min():.4f}, {df['loss'].max():.4f}]")
    print()
    print("-" * 80)
    
    # Extract arrays
    N = df["N_params"].values
    D = df["tokens_D"].values
    L = df["loss"].values
    
    # Fit joint law
    print("(1) FITTING JOINT LAW: L(N,D) = E + A/N^α + B/D^β")
    print()
    jp = fit_joint_powerlaw(N, D, L)
    print(f"  E     = {jp.E:.4f}")
    print(f"  A     = {jp.A:.2f}")
    print(f"  α     = {jp.alpha:.4f}")
    print(f"  B     = {jp.B:.2f}")
    print(f"  β     = {jp.beta:.4f}")
    print()
    
    # Compute residuals and fit quality
    resids = residuals_joint(jp, N, D, L)
    rmse = np.sqrt(np.mean(resids**2))
    print(f"  RMSE (in log-space): {rmse:.4f}")
    print()
    
    # Compute-optimal exponents
    opt = compute_optimal_from_joint(jp)
    print(f"  Compute-optimal split: a = β/(α+β) = {opt['a']:.4f}")
    print(f"  This means ~{opt['a']*100:.1f}% compute on params, ~{opt['b']*100:.1f}% on tokens")
    print()
    print("-" * 80)
    
    # Fit two-term N-only using a slice at near-constant tokens
    print("(2) FITTING TWO-TERM N-ONLY LAW at near-constant D")
    print()
    
    # Take data near median tokens
    mid = df["tokens_D"].median()
    band = (df["tokens_D"] > 0.8*mid) & (df["tokens_D"] < 1.2*mid)
    
    print(f"  Median D = {mid:.2e}")
    print(f"  Selected {band.sum()} points in band [0.8*median, 1.2*median]")
    
    if band.sum() >= 5:
        N_band = df.loc[band, "N_params"].values
        L_band = df.loc[band, "loss"].values
        
        tp = fit_two_term_N_only(N_band, L_band)
        print(f"  E     = {tp.E:.4f}")
        print(f"  A     = {tp.A:.2f}")
        print(f"  α     = {tp.alpha:.4f}")
        print(f"  B     = {tp.B:.2f}")
        print(f"  β     = {tp.beta:.4f}")
        print(f"  Constraint satisfied: β < α? {tp.beta < tp.alpha}")
    else:
        print("  (Not enough points for N-only fit in this band)")
    
    print()
    print("=" * 80)
    print("CSV TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

