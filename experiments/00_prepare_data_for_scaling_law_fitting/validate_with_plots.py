#!/usr/bin/env python3
"""
Visual validation of scaling law fits.

This script generates comprehensive plots showing that both approaches
successfully recover the ground truth scaling laws from synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from scaling_laws_fit import (
    fit_joint_powerlaw,
    predict_joint,
    compute_optimal_from_joint,
    synth_joint
)
from isoflops_scaling import (
    extract_isoflops_points,
    fit_isoflops_scaling_law
)

def generate_clean_synthetic_data():
    """Generate synthetic data with known ground truth."""
    print("Generating synthetic data with known ground truth...")
    
    # Ground truth parameters (Chinchilla-like)
    E = 1.8
    A = 500.0
    alpha = 0.35
    B = 2000.0
    beta = 0.36
    
    print(f"Ground Truth Parameters:")
    print(f"  E = {E}")
    print(f"  A = {A}")
    print(f"  α = {alpha}")
    print(f"  B = {B}")
    print(f"  β = {beta}")
    print(f"  Expected a = β/(α+β) = {beta/(alpha+beta):.4f}")
    print()
    
    # Generate a grid of (N, D) points
    Ns = np.geomspace(1e7, 1e11, 20)
    Ds = np.geomspace(1e9, 1e12, 20)
    NN, DD = np.meshgrid(Ns, Ds)
    NN = NN.ravel()
    DD = DD.ravel()
    
    # Generate losses using ground truth with small noise
    L = synth_joint(NN, DD, E=E, A=A, alpha=alpha, B=B, beta=beta, noise=0.01, seed=42)
    
    return NN, DD, L, (E, A, alpha, B, beta)

def generate_isoflops_data(ground_truth):
    """Generate IsoFLOPs-style data."""
    E, A, alpha, B, beta = ground_truth
    
    compute_budgets = [6e18, 1e19, 2e19, 4e19, 8e19, 1.6e20]
    all_runs = []
    
    for C in compute_budgets:
        # Optimal N for this compute budget (from ground truth)
        # For L = E + A/N^α + B/D^β with D = C/(6N), minimize over N
        # Optimal: N ∝ C^(β/(α+β))
        a_true = beta / (alpha + beta)
        N_optimal_true = 1e8 * (C / 1e19) ** a_true
        
        # Generate models around the optimal
        for scale in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.4, 1.8, 2.5]:
            N = N_optimal_true * scale
            D = C / (6.0 * N)
            
            # Generate loss using ground truth + noise
            loss = E + A / (N ** alpha) + B / (D ** beta)
            loss *= np.exp(np.random.normal(0, 0.01))
            
            all_runs.append({
                "parameters": int(N),
                "compute_budget": C,
                "final_loss": loss
            })
    
    return pd.DataFrame(all_runs)

def plot_joint_fit_validation(NN, DD, L, ground_truth, fitted_params, save_path=None):
    """Plot validation of joint fit approach."""
    E_true, A_true, alpha_true, B_true, beta_true = ground_truth
    
    # Predict using fitted parameters
    L_pred = predict_joint(fitted_params, NN, DD)
    
    # Compute residuals
    residuals = np.log(L_pred) - np.log(L)
    rel_error = np.abs(L_pred - L) / L * 100
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. True vs Predicted Loss (main plot)
    ax1 = fig.add_subplot(gs[0, :2])
    scatter = ax1.scatter(L, L_pred, c=np.log10(NN), cmap='viridis', 
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    L_range = [L.min(), L.max()]
    ax1.plot(L_range, L_range, 'r--', linewidth=2, label='Perfect Fit', alpha=0.8)
    
    # ±10% error bands
    ax1.fill_between(L_range, 
                     [l * 0.9 for l in L_range], 
                     [l * 1.1 for l in L_range],
                     alpha=0.2, color='red', label='±10% error')
    
    ax1.set_xlabel('True Loss', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Joint Fit: True vs Predicted Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('log₁₀(N params)', fontsize=10)
    
    # Add R² score
    ss_res = np.sum((L - L_pred)**2)
    ss_tot = np.sum((L - L.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    ax1.text(0.05, 0.95, f'R² = {r2:.5f}', transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Residuals histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel('Log Residuals', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add mean and std
    ax2.text(0.05, 0.95, f'μ = {residuals.mean():.4f}\nσ = {residuals.std():.4f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Parameter comparison
    ax3 = fig.add_subplot(gs[1, 0])
    params_true = [E_true, A_true, alpha_true, B_true, beta_true]
    params_fit = [fitted_params.E, fitted_params.A, fitted_params.alpha, 
                  fitted_params.B, fitted_params.beta]
    param_names = ['E', 'A', 'α', 'B', 'β']
    
    x = np.arange(len(param_names))
    width = 0.35
    
    ax3.bar(x - width/2, params_true, width, label='Ground Truth', 
            alpha=0.7, color='green', edgecolor='black')
    ax3.bar(x + width/2, params_fit, width, label='Fitted', 
            alpha=0.7, color='orange', edgecolor='black')
    
    ax3.set_xlabel('Parameter', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax3.set_title('Parameter Recovery', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_names, fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Relative errors by parameter magnitude
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(np.log10(NN), rel_error, alpha=0.5, s=20)
    ax4.axhline(10, color='red', linestyle='--', alpha=0.5, label='10% error')
    ax4.set_xlabel('log₁₀(N params)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Error vs Model Size', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add median error
    median_err = np.median(rel_error)
    ax4.text(0.05, 0.95, f'Median: {median_err:.2f}%',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 5. Compute-optimal comparison
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Ground truth
    a_true = beta_true / (alpha_true + beta_true)
    
    # Fitted
    opt = compute_optimal_from_joint(fitted_params)
    a_fit = opt['a']
    
    categories = ['Ground Truth', 'Fitted']
    a_values = [a_true, a_fit]
    colors = ['green', 'orange']
    
    bars = ax5.bar(categories, a_values, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, a_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax5.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax5.set_ylabel('Exponent (a)', fontsize=11, fontweight='bold')
    ax5.set_title('Compute-Optimal Split\na = β/(α+β)', fontsize=12, fontweight='bold')
    ax5.set_ylim([0, 1])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Joint Fit (Approach 3) Validation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved joint fit validation plot to {save_path}")
    
    return fig

def plot_isoflops_validation(df, ground_truth, save_path=None):
    """Plot validation of IsoFLOPs approach."""
    E_true, A_true, alpha_true, B_true, beta_true = ground_truth
    a_true = beta_true / (alpha_true + beta_true)
    
    # Extract IsoFLOPs points
    points = extract_isoflops_points(df)
    
    # Fit scaling law
    law = fit_isoflops_scaling_law(points)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Extract data
    C_data = np.array([p.compute_budget for p in points])
    N_data = np.array([p.optimal_params for p in points])
    D_data = np.array([p.optimal_tokens for p in points])
    
    # Generate prediction range
    C_range = np.logspace(np.log10(C_data.min()), 
                          np.log10(C_data.max() * 10), 100)
    N_pred = law.predict_optimal_params(C_range)
    D_pred = law.predict_optimal_tokens(C_range)
    
    # Ground truth curves
    coeff_N_true = N_data[len(N_data)//2] / (C_data[len(C_data)//2] ** a_true)
    N_true = coeff_N_true * (C_range ** a_true)
    D_true = C_range / (6.0 * N_true)
    
    # 1. N vs C
    ax1 = axes[0, 0]
    ax1.loglog(C_data, N_data, 'o', markersize=10, label='IsoFLOPs optima',
               color='blue', alpha=0.7, markeredgecolor='black', linewidth=1.5)
    ax1.loglog(C_range, N_true, '--', linewidth=2.5, 
               label=f'Ground Truth (a={a_true:.3f})', color='green', alpha=0.8)
    ax1.loglog(C_range, N_pred, '-', linewidth=2.5,
               label=f'Fitted (a={law.N_exponent:.3f})', color='red', alpha=0.8)
    
    ax1.set_xlabel('Compute Budget (FLOPs)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Optimal Model Size (params)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Size Scaling: N ∝ C^a', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # 2. D vs C
    ax2 = axes[0, 1]
    ax2.loglog(C_data, D_data, 's', markersize=10, label='IsoFLOPs optima',
               color='purple', alpha=0.7, markeredgecolor='black', linewidth=1.5)
    ax2.loglog(C_range, D_true, '--', linewidth=2.5,
               label=f'Ground Truth (b={1-a_true:.3f})', color='green', alpha=0.8)
    ax2.loglog(C_range, D_pred, '-', linewidth=2.5,
               label=f'Fitted (b={law.D_exponent:.3f})', color='red', alpha=0.8)
    
    ax2.set_xlabel('Compute Budget (FLOPs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Optimal Dataset Size (tokens)', fontsize=12, fontweight='bold')
    ax2.set_title('Dataset Size Scaling: D ∝ C^b', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Loss curves for each compute budget
    ax3 = axes[1, 0]
    
    # Plot a few IsoFLOPs curves
    budgets_to_plot = sorted(df['compute_budget'].unique())[:4]
    colors_iso = plt.cm.viridis(np.linspace(0, 1, len(budgets_to_plot)))
    
    for C, color in zip(budgets_to_plot, colors_iso):
        subset = df[df['compute_budget'] == C]
        Ns = subset['parameters'].values
        losses = subset['final_loss'].values
        
        # Sort by N
        sort_idx = np.argsort(Ns)
        Ns = Ns[sort_idx]
        losses = losses[sort_idx]
        
        ax3.semilogx(Ns, losses, 'o-', color=color, 
                     label=f'C={C:.1e}', alpha=0.7, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Model Size N (params)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax3.set_title('IsoFLOPs Curves', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Exponent comparison
    ax4 = axes[1, 1]
    
    exponents_true = [a_true, 1-a_true, a_true + (1-a_true)]
    exponents_fit = [law.N_exponent, law.D_exponent, 
                     law.N_exponent + law.D_exponent]
    labels = ['a (N exp)', 'b (D exp)', 'a + b']
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, exponents_true, width, label='Ground Truth',
                    alpha=0.7, color='green', edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, exponents_fit, width, label='Fitted',
                    alpha=0.7, color='red', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    ax4.axhline(1.0, color='blue', linestyle='--', alpha=0.5, 
                label='Expected sum = 1', linewidth=2)
    ax4.set_ylabel('Exponent Value', fontsize=12, fontweight='bold')
    ax4.set_title('Exponent Recovery', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.2])
    
    plt.suptitle('IsoFLOPs (Approach 1) Validation', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved IsoFLOPs validation plot to {save_path}")
    
    return fig, law

def main():
    print("=" * 80)
    print("VISUAL VALIDATION OF SCALING LAW FITS")
    print("=" * 80)
    print()
    
    # Generate synthetic data
    NN, DD, L, ground_truth = generate_clean_synthetic_data()
    print(f"Generated {len(L)} data points")
    print()
    print("-" * 80)
    
    # ========== APPROACH 3: Joint Fit Validation ==========
    print()
    print("VALIDATING APPROACH 3: Joint Fit")
    print("=" * 80)
    print()
    
    # Fit the model
    print("Fitting joint power law...")
    fitted_params = fit_joint_powerlaw(NN, DD, L)
    
    print("Fitted Parameters:")
    print(f"  E = {fitted_params.E:.4f} (true: {ground_truth[0]})")
    print(f"  A = {fitted_params.A:.2f} (true: {ground_truth[1]})")
    print(f"  α = {fitted_params.alpha:.4f} (true: {ground_truth[2]})")
    print(f"  B = {fitted_params.B:.2f} (true: {ground_truth[3]})")
    print(f"  β = {fitted_params.beta:.4f} (true: {ground_truth[4]})")
    print()
    
    # Generate validation plot
    print("Generating validation plots...")
    plot_joint_fit_validation(NN, DD, L, ground_truth, fitted_params,
                             save_path='joint_fit_validation.png')
    print()
    print("-" * 80)
    
    # ========== APPROACH 1: IsoFLOPs Validation ==========
    print()
    print("VALIDATING APPROACH 1: IsoFLOPs")
    print("=" * 80)
    print()
    
    # Generate IsoFLOPs-style data
    print("Generating IsoFLOPs data...")
    df = generate_isoflops_data(ground_truth)
    print(f"Generated {len(df)} runs across {df['compute_budget'].nunique()} budgets")
    print()
    
    # Generate validation plot
    print("Generating validation plots...")
    fig, law = plot_isoflops_validation(df, ground_truth,
                                       save_path='isoflops_validation.png')
    print()
    print("-" * 80)
    
    # ========== SUMMARY ==========
    print()
    print("SUMMARY")
    print("=" * 80)
    print()
    
    # Joint fit accuracy
    E_err = abs(fitted_params.E - ground_truth[0]) / ground_truth[0] * 100
    A_err = abs(fitted_params.A - ground_truth[1]) / ground_truth[1] * 100
    alpha_err = abs(fitted_params.alpha - ground_truth[2]) / ground_truth[2] * 100
    B_err = abs(fitted_params.B - ground_truth[3]) / ground_truth[3] * 100
    beta_err = abs(fitted_params.beta - ground_truth[4]) / ground_truth[4] * 100
    
    print("Joint Fit Parameter Recovery Errors:")
    print(f"  E:     {E_err:5.2f}%")
    print(f"  A:     {A_err:5.2f}%")
    print(f"  α:     {alpha_err:5.2f}%")
    print(f"  B:     {B_err:5.2f}%")
    print(f"  β:     {beta_err:5.2f}%")
    print(f"  Average: {np.mean([E_err, A_err, alpha_err, B_err, beta_err]):.2f}%")
    print()
    
    # IsoFLOPs accuracy
    a_true = ground_truth[4] / (ground_truth[2] + ground_truth[4])
    a_err = abs(law.N_exponent - a_true) / a_true * 100
    sum_err = abs((law.N_exponent + law.D_exponent) - 1.0)
    
    print("IsoFLOPs Exponent Recovery:")
    print(f"  True a:    {a_true:.4f}")
    print(f"  Fitted a:  {law.N_exponent:.4f}")
    print(f"  Error:     {a_err:.2f}%")
    print(f"  a + b:     {law.N_exponent + law.D_exponent:.4f} (should be 1.0)")
    print(f"  |a+b-1|:   {sum_err:.4f}")
    print()
    
    print("✅ Validation plots generated:")
    print("  - joint_fit_validation.png")
    print("  - isoflops_validation.png")
    print()
    print("=" * 80)
    print("VISUAL VALIDATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()

