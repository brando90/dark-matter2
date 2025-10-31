#!/usr/bin/env python3
"""
Simple end-to-end test for CS336 scaling laws pipeline.

This script:
1. Creates synthetic data in CS336 format
2. Runs the IsoFLOPs pipeline
3. Generates required plots
4. Makes predictions for 10^23 and 10^24 FLOPs

This is ALL you need to validate the pipeline works.
"""

import json
import numpy as np
from isoflops_scaling import (
    load_isoflops_data,
    extract_isoflops_points,
    fit_isoflops_scaling_law,
    plot_scaling_law
)

def create_test_data_cs336_format(filepath="test_data.json"):
    """
    Create synthetic test data in CS336 format.
    
    Format: [{"parameters": N, "compute_budget": C, "final_loss": L}, ...]
    """
    print("Creating synthetic test data in CS336 format...")
    
    # Ground truth: balanced scaling (a ≈ 0.5)
    E, A, alpha, B, beta = 1.8, 500.0, 0.35, 2000.0, 0.36
    a_true = beta / (alpha + beta)
    
    print(f"Ground truth: a = {a_true:.4f} (balanced scaling)")
    
    data = []
    compute_budgets = [6e18, 1e19, 2e19, 4e19, 8e19, 1.6e20]
    
    for C in compute_budgets:
        # Optimal N for this budget (from ground truth)
        N_optimal = 1e8 * (C / 1e19) ** a_true
        
        # Generate several models around optimal
        for scale in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.4, 1.8, 2.5]:
            N = int(N_optimal * scale)
            D = C / (6.0 * N)
            
            # Loss from ground truth
            loss = E + A / (N ** alpha) + B / (D ** beta)
            loss *= np.exp(np.random.normal(0, 0.01))  # 1% noise
            
            data.append({
                "parameters": N,
                "compute_budget": C,
                "final_loss": float(loss)
            })
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Created {len(data)} runs across {len(compute_budgets)} compute budgets")
    print(f"✓ Saved to {filepath}")
    return filepath

def test_pipeline(data_file):
    """Run the complete CS336 pipeline."""
    print("\n" + "="*80)
    print("RUNNING CS336 PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    df = load_isoflops_data(data_file)
    print(f"✓ Loaded {len(df)} runs")
    
    # Step 2: Extract IsoFLOPs points
    print("\nStep 2: Extracting IsoFLOPs points...")
    points = extract_isoflops_points(df)
    print(f"✓ Found {len(points)} optimal points")
    
    print("\nOptimal points:")
    print(f"{'Compute (C)':>15} | {'Opt N':>12} | {'Opt D':>12} | {'Loss':>8}")
    print("-" * 60)
    for p in points:
        print(f"{p.compute_budget:>15.2e} | {p.optimal_params:>12.2e} | "
              f"{p.optimal_tokens:>12.2e} | {p.min_loss:>8.4f}")
    
    # Step 3: Fit scaling laws
    print("\nStep 3: Fitting power laws...")
    law = fit_isoflops_scaling_law(points)
    print(f"✓ N_opt = {law.N_coefficient:.3e} × C^{law.N_exponent:.4f}")
    print(f"✓ D_opt = {law.D_coefficient:.3e} × C^{law.D_exponent:.4f}")
    print(f"✓ Sum: a + b = {law.N_exponent + law.D_exponent:.4f} (should be 1.0)")
    
    # Step 4: Make predictions (CS336 requirement)
    print("\nStep 4: Making predictions for target budgets...")
    
    targets = [1e23, 1e24]
    for C_target in targets:
        N_pred, D_pred = law.predict_both(C_target)
        print(f"\nC = {C_target:.2e} FLOPs:")
        print(f"  N_opt = {N_pred:.3e} parameters")
        print(f"  D_opt = {D_pred:.3e} tokens")
        print(f"  Check: 6×N×D = {6*N_pred*D_pred:.3e}")
    
    # Step 5: Generate plots (CS336 requirement)
    print("\nStep 5: Generating CS336-required plots...")
    plot_scaling_law(points, law, extrapolate_to=1e24, 
                    save_path='cs336_scaling_law_validation.png')
    print("✓ Saved plots to cs336_scaling_law_validation.png")
    
    return law

def validate_results(law):
    """Check if results make sense."""
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80 + "\n")
    
    issues = []
    
    # Check constraint
    sum_exp = law.N_exponent + law.D_exponent
    if abs(sum_exp - 1.0) > 0.05:
        issues.append(f"Exponents don't sum to 1: {sum_exp:.4f}")
    else:
        print(f"✓ Constraint satisfied: a + b = {sum_exp:.4f}")
    
    # Check balanced scaling
    if abs(law.N_exponent - 0.5) > 0.2:
        issues.append(f"Exponent far from balanced: a = {law.N_exponent:.4f}")
    else:
        print(f"✓ Balanced scaling: a = {law.N_exponent:.4f}")
    
    # Check reasonable ranges
    if law.N_exponent < 0.3 or law.N_exponent > 0.7:
        issues.append(f"Exponent outside typical range [0.3, 0.7]: {law.N_exponent:.4f}")
    else:
        print(f"✓ Exponent in reasonable range")
    
    print()
    if not issues:
        print("✅ ALL VALIDATION CHECKS PASSED!")
        print("\nThe pipeline works correctly and produces valid results.")
    else:
        print("⚠️  VALIDATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    
    return len(issues) == 0

def main():
    print("="*80)
    print("CS336 SCALING LAWS PIPELINE TEST")
    print("="*80 + "\n")
    
    # Create test data
    data_file = create_test_data_cs336_format()
    
    # Run pipeline
    law = test_pipeline(data_file)
    
    # Validate
    success = validate_results(law)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    if success:
        print("✅ Pipeline test PASSED")
        print("\nYou can now use this pipeline for real CS336 data:")
        print("  1. Put your data in JSON format (see test_data.json)")
        print("  2. Run: isoflops_scaling.py or import the functions")
        print("  3. Plots will be generated automatically")
    else:
        print("⚠️  Pipeline test found issues (see above)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

