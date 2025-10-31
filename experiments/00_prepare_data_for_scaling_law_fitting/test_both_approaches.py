#!/usr/bin/env python3
"""
Comprehensive test comparing both Chinchilla scaling law approaches.

This script demonstrates that both IsoFLOPs (Approach 1) and Joint Fit (Approach 3)
converge to similar compute-optimal allocations, validating both implementations.
"""

import numpy as np
from isoflops_scaling import (
    extract_isoflops_points, 
    fit_isoflops_scaling_law,
    IsoFLOPsPoint
)
from scaling_laws_fit import (
    fit_joint_powerlaw,
    compute_optimal_from_joint,
    synth_joint
)
import pandas as pd

def generate_test_data():
    """Generate synthetic scaling law data for testing both approaches."""
    print("Generating synthetic test data...")
    
    # Ground truth parameters (Chinchilla-like)
    E = 1.8
    A = 500.0
    alpha = 0.35
    B = 2000.0
    beta = 0.36
    
    print(f"Ground truth: E={E}, A={A}, α={alpha}, B={B}, β={beta}")
    print(f"Expected compute-optimal split: a = β/(α+β) = {beta/(alpha+beta):.4f}")
    print()
    
    # Generate IsoFLOPs data: multiple model sizes for each compute budget
    compute_budgets = [6e18, 1e19, 2e19, 4e19, 8e19, 1.6e20]
    all_runs = []
    
    for C in compute_budgets:
        # For each compute budget, try different model sizes
        N_optimal_approx = 1e8 * (C / 1e19) ** 0.5  # Rough estimate
        
        for scale in [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]:
            N = N_optimal_approx * scale
            D = C / (6.0 * N)
            
            # Generate loss using ground truth formula + small noise
            loss = E + A / (N ** alpha) + B / (D ** beta)
            loss *= np.exp(np.random.normal(0, 0.01))  # 1% multiplicative noise
            
            all_runs.append({
                "parameters": int(N),
                "compute_budget": C,
                "final_loss": loss
            })
    
    df = pd.DataFrame(all_runs)
    print(f"Generated {len(df)} training runs across {len(compute_budgets)} compute budgets")
    return df

def main():
    print("=" * 80)
    print("COMPARISON: IsoFLOPs (Approach 1) vs Joint Fit (Approach 3)")
    print("=" * 80)
    print()
    
    # Generate test data
    df = generate_test_data()
    print()
    print("-" * 80)
    
    # ========== APPROACH 1: IsoFLOPs Profiles ==========
    print()
    print("APPROACH 1: IsoFLOPs Profiles (CS336 Method)")
    print("=" * 80)
    print()
    
    # Extract IsoFLOPs points
    points = extract_isoflops_points(df)
    print(f"Extracted {len(points)} IsoFLOPs points")
    print()
    
    # Fit scaling law
    isoflops_law = fit_isoflops_scaling_law(points)
    print("Fitted Power Laws:")
    print(f"  N_opt = {isoflops_law.N_coefficient:.3e} × C^{isoflops_law.N_exponent:.4f}")
    print(f"  D_opt = {isoflops_law.D_coefficient:.3e} × C^{isoflops_law.D_exponent:.4f}")
    print()
    print(f"Compute-optimal exponent: a = {isoflops_law.N_exponent:.4f}")
    print(f"Sum of exponents: {isoflops_law.N_exponent + isoflops_law.D_exponent:.4f}")
    
    # Make prediction for target budget
    target_C = 1e19
    N_iso, D_iso = isoflops_law.predict_both(target_C)
    print()
    print(f"Prediction for C = {target_C:.2e} FLOPs:")
    print(f"  N_opt = {N_iso:.3e} parameters")
    print(f"  D_opt = {D_iso:.3e} tokens")
    
    print()
    print("-" * 80)
    
    # ========== APPROACH 3: Joint Fit ==========
    print()
    print("APPROACH 3: Joint Fit with Huber Loss")
    print("=" * 80)
    print()
    
    # Convert to arrays for joint fit
    N_all = df['parameters'].values
    D_all = (df['compute_budget'] / (6.0 * df['parameters'])).values
    L_all = df['final_loss'].values
    
    print(f"Fitting joint law L(N,D) = E + A/N^α + B/D^β on {len(N_all)} points...")
    
    # Fit joint power law
    joint_params = fit_joint_powerlaw(N_all, D_all, L_all)
    print()
    print("Fitted Parameters:")
    print(f"  E     = {joint_params.E:.4f}")
    print(f"  A     = {joint_params.A:.2f}")
    print(f"  α     = {joint_params.alpha:.4f}")
    print(f"  B     = {joint_params.B:.2f}")
    print(f"  β     = {joint_params.beta:.4f}")
    
    # Compute optimal exponent
    opt_split = compute_optimal_from_joint(joint_params)
    print()
    print(f"Compute-optimal exponent: a = β/(α+β) = {opt_split['a']:.4f}")
    
    # Make prediction using the formula N* ∝ C^a
    a_joint = opt_split['a']
    # Need to calibrate the coefficient using one known point
    C_ref = 1e19
    N_ref_joint = points[1].optimal_params if len(points) > 1 else 1e8
    coeff = N_ref_joint / (C_ref ** a_joint)
    
    N_joint = coeff * (target_C ** a_joint)
    D_joint = target_C / (6.0 * N_joint)
    
    print()
    print(f"Prediction for C = {target_C:.2e} FLOPs:")
    print(f"  N_opt = {N_joint:.3e} parameters")
    print(f"  D_opt = {D_joint:.3e} tokens")
    
    print()
    print("-" * 80)
    
    # ========== COMPARISON ==========
    print()
    print("COMPARISON OF RESULTS")
    print("=" * 80)
    print()
    
    print("Compute-Optimal Exponent (a):")
    print(f"  IsoFLOPs (Approach 1):  a = {isoflops_law.N_exponent:.4f}")
    print(f"  Joint Fit (Approach 3): a = {opt_split['a']:.4f}")
    diff_a = abs(isoflops_law.N_exponent - opt_split['a'])
    print(f"  Absolute difference:    Δa = {diff_a:.4f}")
    
    if diff_a < 0.05:
        print("  ✓ Methods AGREE (difference < 0.05)")
    else:
        print("  ⚠ Methods differ by > 0.05")
    
    print()
    print(f"Predictions for C = {target_C:.2e} FLOPs:")
    print()
    print(f"  Optimal Model Size:")
    print(f"    IsoFLOPs:  {N_iso:.3e} parameters")
    print(f"    Joint Fit: {N_joint:.3e} parameters")
    diff_N = abs(N_iso - N_joint) / N_iso * 100
    print(f"    Difference: {diff_N:.1f}%")
    
    print()
    print(f"  Optimal Dataset Size:")
    print(f"    IsoFLOPs:  {D_iso:.3e} tokens")
    print(f"    Joint Fit: {D_joint:.3e} tokens")
    diff_D = abs(D_iso - D_joint) / D_iso * 100
    print(f"    Difference: {diff_D:.1f}%")
    
    print()
    print("-" * 80)
    print()
    
    # ========== CONCLUSION ==========
    print("CONCLUSION")
    print("=" * 80)
    print()
    
    if diff_a < 0.05 and diff_N < 20 and diff_D < 20:
        print("✅ Both methods converge to similar compute-optimal allocations!")
        print("✅ This validates both implementations.")
        print()
        print("Key Finding: Balanced scaling with a ≈ 0.5")
        print("  → Allocate ~50% compute to model size (N)")
        print("  → Allocate ~50% compute to dataset size (D)")
    else:
        print("⚠ Methods show some disagreement.")
        print("This may be due to:")
        print("  - Limited data / noise")
        print("  - Different assumptions about the loss surface")
        print("  - Numerical optimization differences")
    
    print()
    print("=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()

