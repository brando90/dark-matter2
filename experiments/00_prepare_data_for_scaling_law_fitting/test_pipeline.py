#!/usr/bin/env python3
"""
Simple CS336 Scaling Laws Pipeline Test

This single script validates everything:
- Creates synthetic test data in correct format
- Tests IsoFLOPs approach (CS336 method)
- Tests Joint Fit approach (from codex_prompt)
- Generates validation plots
- Saves everything to correct directories

Directories:
- Plots → experiments/00_prepare_data_for_scaling_law_fitting/plots/
- Data → data/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the implementations
from isoflops_scaling import (
    load_isoflops_data,
    extract_isoflops_points,
    fit_isoflops_scaling_law,
)
from scaling_laws_fit import (
    fit_joint_powerlaw,
    predict_joint,
    compute_optimal_from_joint,
    synth_joint
)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = Path(__file__).parent / "plots"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def create_test_data_cs336_format():
    """Create synthetic test data in CS336 format."""
    print("Creating synthetic test data...")
    
    # Ground truth: balanced scaling (a ≈ 0.5)
    E, A, alpha, B, beta = 1.8, 500.0, 0.35, 2000.0, 0.36
    a_true = beta / (alpha + beta)
    
    print(f"  Ground truth: a = {a_true:.4f} (balanced scaling)")
    
    data = []
    compute_budgets = [6e18, 1e19, 2e19, 4e19, 8e19, 1.6e20]
    
    for C in compute_budgets:
        N_optimal = 1e8 * (C / 1e19) ** a_true
        for scale in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.4, 1.8, 2.5]:
            N = int(N_optimal * scale)
            D = C / (6.0 * N)
            loss = E + A / (N ** alpha) + B / (D ** beta)
            loss *= np.exp(np.random.normal(0, 0.01))
            
            data.append({
                "parameters": N,
                "compute_budget": C,
                "final_loss": float(loss)
            })
    
    # Save to data directory
    filepath = DATA_DIR / "cs336_test_data.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ Created {len(data)} runs")
    print(f"  ✓ Saved to {filepath}")
    return filepath

def plot_cs336_validation(points, law):
    """Generate CS336-required plots."""
    print("\nGenerating CS336 validation plots...")
    
    C_data = np.array([p.compute_budget for p in points])
    N_data = np.array([p.optimal_params for p in points])
    D_data = np.array([p.optimal_tokens for p in points])
    
    C_range = np.logspace(np.log10(C_data.min()), np.log10(1e24), 100)
    N_pred = law.predict_optimal_params(C_range)
    D_pred = law.predict_optimal_tokens(C_range)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Model size scaling
    ax1.loglog(C_data, N_data, 'o', markersize=10, label='IsoFLOPs optima',
               color='blue', markeredgecolor='black', linewidth=1.5, alpha=0.7)
    ax1.loglog(C_range, N_pred, '-', linewidth=2.5,
               label=f'N ∝ C^{law.N_exponent:.3f}', color='red', alpha=0.8)
    ax1.axvline(1e23, color='red', linestyle='--', alpha=0.5, label='10²³ FLOPs')
    ax1.axvline(1e24, color='orange', linestyle='--', alpha=0.5, label='10²⁴ FLOPs')
    ax1.set_xlabel('Compute Budget (FLOPs)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Optimal Model Size (parameters)', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling Law: Model Size vs Compute', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Dataset size scaling
    ax2.loglog(C_data, D_data, 's', markersize=10, label='IsoFLOPs optima',
               color='purple', markeredgecolor='black', linewidth=1.5, alpha=0.7)
    ax2.loglog(C_range, D_pred, '-', linewidth=2.5,
               label=f'D ∝ C^{law.D_exponent:.3f}', color='red', alpha=0.8)
    ax2.axvline(1e23, color='red', linestyle='--', alpha=0.5, label='10²³ FLOPs')
    ax2.axvline(1e24, color='orange', linestyle='--', alpha=0.5, label='10²⁴ FLOPs')
    ax2.set_xlabel('Compute Budget (FLOPs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Optimal Dataset Size (tokens)', fontsize=12, fontweight='bold')
    ax2.set_title('Scaling Law: Dataset Size vs Compute', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    save_path = PLOTS_DIR / "cs336_scaling_laws.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    plt.close()

def plot_joint_fit_validation(NN, DD, L, ground_truth, fitted_params):
    """Generate joint fit validation plot."""
    print("\nGenerating joint fit validation plot...")
    
    L_pred = predict_joint(fitted_params, NN, DD)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # True vs Predicted
    ax1 = axes[0, 0]
    scatter = ax1.scatter(L, L_pred, c=np.log10(NN), cmap='viridis', alpha=0.6, s=30)
    L_range = [L.min(), L.max()]
    ax1.plot(L_range, L_range, 'r--', linewidth=2, label='Perfect Fit')
    ax1.set_xlabel('True Loss', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Joint Fit: True vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='log₁₀(N)')
    
    # R²
    ss_res = np.sum((L - L_pred)**2)
    ss_tot = np.sum((L - L.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    ax1.text(0.05, 0.95, f'R² = {r2:.5f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Parameter recovery
    ax2 = axes[0, 1]
    E_t, A_t, alpha_t, B_t, beta_t = ground_truth
    params_true = [E_t, A_t, alpha_t, B_t, beta_t]
    params_fit = [fitted_params.E, fitted_params.A, fitted_params.alpha,
                  fitted_params.B, fitted_params.beta]
    param_names = ['E', 'A', 'α', 'B', 'β']
    rel_errors = [abs(f - t) / t * 100 for f, t in zip(params_fit, params_true)]
    
    x = np.arange(len(param_names))
    ax2.bar(x, rel_errors, color=['green' if e < 10 else 'red' for e in rel_errors],
            alpha=0.7, edgecolor='black')
    ax2.axhline(10, color='red', linestyle='--', label='10% threshold')
    ax2.set_xlabel('Parameter', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Parameter Recovery', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Residuals
    ax3 = axes[1, 0]
    residuals = np.log(L_pred) - np.log(L)
    ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Log Residuals', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Compute-optimal
    ax4 = axes[1, 1]
    a_true = beta_t / (alpha_t + beta_t)
    opt = compute_optimal_from_joint(fitted_params)
    a_fit = opt['a']
    
    bars = ax4.bar(['True', 'Fitted'], [a_true, a_fit], 
                   color=['green', 'orange'], alpha=0.7, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, [a_true, a_fit]):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax4.set_ylabel('Exponent (a)', fontsize=11, fontweight='bold')
    ax4.set_title('Compute-Optimal Split\na = β/(α+β)', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Joint Fit Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = PLOTS_DIR / "joint_fit_validation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    plt.close()

def main():
    print("="*80)
    print("CS336 SCALING LAWS PIPELINE TEST")
    print("="*80)
    
    # Create test data
    data_file = create_test_data_cs336_format()
    
    print("\n" + "="*80)
    print("TESTING ISOFLOPS APPROACH (CS336)")
    print("="*80)
    
    # Load and process
    df = load_isoflops_data(str(data_file))
    print(f"\n✓ Loaded {len(df)} runs")
    
    points = extract_isoflops_points(df)
    print(f"✓ Extracted {len(points)} IsoFLOPs points")
    
    law = fit_isoflops_scaling_law(points)
    print(f"\n✓ Fitted scaling laws:")
    print(f"  N_opt = {law.N_coefficient:.3e} × C^{law.N_exponent:.4f}")
    print(f"  D_opt = {law.D_coefficient:.3e} × C^{law.D_exponent:.4f}")
    print(f"  Constraint: a + b = {law.N_exponent + law.D_exponent:.4f}")
    
    # Predictions
    print(f"\n✓ Predictions:")
    for C in [1e23, 1e24]:
        N, D = law.predict_both(C)
        print(f"  C = {C:.2e}: N = {N:.3e}, D = {D:.3e}")
    
    # Generate CS336 plot
    plot_cs336_validation(points, law)
    
    print("\n" + "="*80)
    print("TESTING JOINT FIT APPROACH (FROM CODEX)")
    print("="*80)
    
    # Generate data for joint fit
    print("\nGenerating synthetic data for joint fit...")
    E, A, alpha, B, beta = 1.8, 500.0, 0.35, 2000.0, 0.36
    ground_truth = (E, A, alpha, B, beta)
    
    Ns = np.geomspace(1e7, 1e11, 20)
    Ds = np.geomspace(1e9, 1e12, 20)
    NN, DD = np.meshgrid(Ns, Ds)
    NN = NN.ravel()
    DD = DD.ravel()
    L = synth_joint(NN, DD, E=E, A=A, alpha=alpha, B=B, beta=beta, noise=0.01, seed=42)
    
    print(f"✓ Generated {len(L)} data points")
    
    # Fit
    print("\nFitting joint power law...")
    fitted_params = fit_joint_powerlaw(NN, DD, L)
    print(f"\n✓ Fitted parameters:")
    print(f"  E = {fitted_params.E:.4f} (true: {E})")
    print(f"  A = {fitted_params.A:.2f} (true: {A})")
    print(f"  α = {fitted_params.alpha:.4f} (true: {alpha})")
    print(f"  B = {fitted_params.B:.2f} (true: {B})")
    print(f"  β = {fitted_params.beta:.4f} (true: {beta})")
    
    opt = compute_optimal_from_joint(fitted_params)
    print(f"\n✓ Compute-optimal: a = {opt['a']:.4f}")
    
    # Generate joint fit plot
    plot_joint_fit_validation(NN, DD, L, ground_truth, fitted_params)
    
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Check results
    issues = []
    
    # Check IsoFLOPs constraint
    sum_exp = law.N_exponent + law.D_exponent
    if abs(sum_exp - 1.0) > 0.05:
        issues.append(f"IsoFLOPs: a + b = {sum_exp:.4f} (should be 1.0)")
    else:
        print(f"\n✓ IsoFLOPs constraint satisfied: a + b = {sum_exp:.4f}")
    
    # Check balanced scaling
    if abs(law.N_exponent - 0.5) > 0.2:
        issues.append(f"IsoFLOPs: a = {law.N_exponent:.4f} (far from 0.5)")
    else:
        print(f"✓ Balanced scaling: a = {law.N_exponent:.4f}")
    
    # Check joint fit
    if opt['a'] < 0.4 or opt['a'] > 0.6:
        issues.append(f"Joint fit: a = {opt['a']:.4f} (outside [0.4, 0.6])")
    else:
        print(f"✓ Joint fit balanced: a = {opt['a']:.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not issues:
        print("\n✅ ALL TESTS PASSED!")
        print(f"\nGenerated files:")
        print(f"  Data: {data_file}")
        print(f"  Plots: {PLOTS_DIR}/")
        print(f"    - cs336_scaling_laws.png")
        print(f"    - joint_fit_validation.png")
    else:
        print("\n⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

