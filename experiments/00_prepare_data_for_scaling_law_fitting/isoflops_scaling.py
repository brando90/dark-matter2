#!/usr/bin/env python3
"""
IsoFLOPs Scaling Laws (Chinchilla Approach 1)

This implements the IsoFLOPs profile method from CS336 Assignment 3 and 
Hoffmann et al. (2022), which differs from the joint fit approach in scaling_laws_fit.py.

The IsoFLOPs method:
1. For each compute budget C, find the model size N that minimizes training loss
2. Fit power laws: N_opt ∝ C^a and D_opt ∝ C^b
3. Extrapolate to predict optimal model/dataset sizes for target compute budgets

This complements the joint fit approach (Approach 3) already implemented.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.optimize import curve_fit
import json

@dataclass
class IsoFLOPsPoint:
    """A single point in the IsoFLOPs scaling law."""
    compute_budget: float
    optimal_params: float
    optimal_tokens: float
    min_loss: float

def load_isoflops_data(path: str) -> pd.DataFrame:
    """
    Load training run data from JSON file.
    
    Expected format:
    [
        {
            "parameters": 49999999,
            "compute_budget": 6e18,
            "final_loss": 7.192784500319437
        },
        ...
    ]
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def extract_isoflops_points(df: pd.DataFrame) -> List[IsoFLOPsPoint]:
    """
    For each compute budget C, find the model size N that achieves minimum loss.
    
    Returns a list of IsoFLOPsPoint objects, one per unique compute budget.
    """
    points = []
    
    # Group by compute budget
    for C, group in df.groupby('compute_budget'):
        # Find the run with minimum loss for this budget
        min_idx = group['final_loss'].idxmin()
        min_row = group.loc[min_idx]
        
        N_opt = min_row['parameters']
        L_min = min_row['final_loss']
        
        # Compute tokens: D = C / (6N)
        D_opt = C / (6.0 * N_opt)
        
        points.append(IsoFLOPsPoint(
            compute_budget=C,
            optimal_params=N_opt,
            optimal_tokens=D_opt,
            min_loss=L_min
        ))
    
    # Sort by compute budget
    points.sort(key=lambda p: p.compute_budget)
    return points

def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law function: y = b * x^a"""
    return b * np.power(x, a)

def fit_power_law(C_values: np.ndarray, Y_values: np.ndarray) -> Tuple[float, float]:
    """
    Fit a power law Y = b * C^a to the data.
    
    Returns:
        (a, b): the fitted exponent and coefficient
    """
    # Use log-log linear regression for robustness
    log_C = np.log(C_values)
    log_Y = np.log(Y_values)
    
    # Fit log(Y) = log(b) + a * log(C)
    coeffs = np.polyfit(log_C, log_Y, deg=1)
    a = coeffs[0]  # exponent
    log_b = coeffs[1]
    b = np.exp(log_b)  # coefficient
    
    return a, b

@dataclass
class ScalingLaw:
    """Fitted scaling law parameters."""
    N_exponent: float  # a in N_opt ∝ C^a
    N_coefficient: float
    D_exponent: float  # b in D_opt ∝ C^b  
    D_coefficient: float
    
    def predict_optimal_params(self, compute_budget: float) -> float:
        """Predict optimal number of parameters for given compute budget."""
        return self.N_coefficient * (compute_budget ** self.N_exponent)
    
    def predict_optimal_tokens(self, compute_budget: float) -> float:
        """Predict optimal number of tokens for given compute budget."""
        return self.D_coefficient * (compute_budget ** self.D_exponent)
    
    def predict_both(self, compute_budget: float) -> Tuple[float, float]:
        """Predict both optimal parameters and tokens."""
        N = self.predict_optimal_params(compute_budget)
        D = self.predict_optimal_tokens(compute_budget)
        return N, D

def fit_isoflops_scaling_law(points: List[IsoFLOPsPoint]) -> ScalingLaw:
    """
    Fit power laws to the IsoFLOPs data points.
    
    Fits:
        N_opt ∝ C^a
        D_opt ∝ C^b
    
    Returns ScalingLaw with fitted parameters.
    """
    C_values = np.array([p.compute_budget for p in points])
    N_values = np.array([p.optimal_params for p in points])
    D_values = np.array([p.optimal_tokens for p in points])
    
    # Fit N vs C
    a_N, b_N = fit_power_law(C_values, N_values)
    
    # Fit D vs C  
    a_D, b_D = fit_power_law(C_values, D_values)
    
    return ScalingLaw(
        N_exponent=a_N,
        N_coefficient=b_N,
        D_exponent=a_D,
        D_coefficient=b_D
    )

def verify_exponents_sum_to_one(law: ScalingLaw, tol: float = 0.1) -> bool:
    """
    Verify that a + b ≈ 1, as expected from C = 6ND.
    
    If N ∝ C^a and D ∝ C^b, then C = 6ND implies a + b = 1.
    """
    sum_exp = law.N_exponent + law.D_exponent
    return abs(sum_exp - 1.0) < tol

# ---------------------------
# Visualization helpers
# ---------------------------

def plot_scaling_law(points: List[IsoFLOPsPoint], 
                     law: ScalingLaw,
                     extrapolate_to: float = 1e24,
                     save_path: str = None):
    """
    Plot the fitted scaling laws with data points.
    
    Args:
        points: List of IsoFLOPs data points
        law: Fitted ScalingLaw
        extrapolate_to: Maximum compute budget to extrapolate to
        save_path: If provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    # Extract data
    C_data = np.array([p.compute_budget for p in points])
    N_data = np.array([p.optimal_params for p in points])
    D_data = np.array([p.optimal_tokens for p in points])
    
    # Generate extrapolation range
    C_min = C_data.min()
    C_max = max(C_data.max(), extrapolate_to)
    C_range = np.logspace(np.log10(C_min), np.log10(C_max), 100)
    
    # Predict from scaling law
    N_pred = law.predict_optimal_params(C_range)
    D_pred = law.predict_optimal_tokens(C_range)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot N vs C
    ax1.loglog(C_data, N_data, 'o', markersize=8, label='Data points', alpha=0.7)
    ax1.loglog(C_range, N_pred, '-', linewidth=2, 
               label=f'N ∝ C^{law.N_exponent:.3f}')
    ax1.axvline(1e23, color='red', linestyle='--', alpha=0.5, label='10²³ FLOPs')
    ax1.axvline(1e24, color='orange', linestyle='--', alpha=0.5, label='10²⁴ FLOPs')
    ax1.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax1.set_ylabel('Optimal Model Size (parameters)', fontsize=12)
    ax1.set_title('Scaling Law: Model Size vs Compute', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot D vs C
    ax2.loglog(C_data, D_data, 's', markersize=8, label='Data points', alpha=0.7)
    ax2.loglog(C_range, D_pred, '-', linewidth=2,
               label=f'D ∝ C^{law.D_exponent:.3f}')
    ax2.axvline(1e23, color='red', linestyle='--', alpha=0.5, label='10²³ FLOPs')
    ax2.axvline(1e24, color='orange', linestyle='--', alpha=0.5, label='10²⁴ FLOPs')
    ax2.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax2.set_ylabel('Optimal Dataset Size (tokens)', fontsize=12)
    ax2.set_title('Scaling Law: Dataset Size vs Compute', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

# ---------------------------
# Example usage / Test
# ---------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("ISOFLOPS SCALING LAWS (CS336 / Chinchilla Approach 1)")
    print("=" * 80)
    print()
    
    # Check if the CS336 data file exists
    import os
    cs336_data_path = "data/isoflops_curves.json"
    
    if os.path.exists(cs336_data_path):
        print(f"Loading data from {cs336_data_path}...")
        df = load_isoflops_data(cs336_data_path)
    else:
        # Generate synthetic data for testing
        print("CS336 data not found, generating synthetic data for testing...")
        synthetic_data = []
        
        # Generate runs at different compute budgets
        compute_budgets = [6e18, 1e19, 2e19, 4e19, 8e19]
        
        for C in compute_budgets:
            # Generate several model sizes for this budget
            # Roughly following N_opt ∝ C^0.5 (balanced scaling)
            N_optimal = 1e8 * (C / 1e19) ** 0.5
            
            # Generate models around the optimal
            for scale in [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]:
                N = N_optimal * scale
                D = C / (6.0 * N)
                
                # Simulate loss with a minimum around N_optimal
                # Loss is higher when N is too small or too large
                deviation = np.log(scale)
                base_loss = 2.0 + 5.0 / (C / 1e19) ** 0.1
                loss = base_loss + 0.5 * deviation**2
                
                synthetic_data.append({
                    "parameters": int(N),
                    "compute_budget": C,
                    "final_loss": loss
                })
        
        df = pd.DataFrame(synthetic_data)
        print(f"Generated {len(df)} synthetic data points")
    
    print(f"Data shape: {df.shape}")
    print(f"Compute budgets: {sorted(df['compute_budget'].unique())}")
    print()
    print("-" * 80)
    
    # Extract IsoFLOPs points
    print("Extracting IsoFLOPs points (finding optimal N for each C)...")
    points = extract_isoflops_points(df)
    print(f"Found {len(points)} IsoFLOPs points")
    print()
    
    print("IsoFLOPs Points:")
    print(f"{'Compute (C)':>15} | {'Opt Params (N)':>15} | {'Opt Tokens (D)':>15} | {'Min Loss':>10}")
    print("-" * 70)
    for p in points:
        print(f"{p.compute_budget:>15.2e} | {p.optimal_params:>15.2e} | "
              f"{p.optimal_tokens:>15.2e} | {p.min_loss:>10.4f}")
    print()
    print("-" * 80)
    
    # Fit scaling law
    print("Fitting power law scaling laws...")
    law = fit_isoflops_scaling_law(points)
    print()
    print("Fitted Scaling Laws:")
    print(f"  N_opt = {law.N_coefficient:.3e} × C^{law.N_exponent:.4f}")
    print(f"  D_opt = {law.D_coefficient:.3e} × C^{law.D_exponent:.4f}")
    print()
    print(f"  Sum of exponents: {law.N_exponent + law.D_exponent:.4f} (should be ≈ 1.0)")
    
    if verify_exponents_sum_to_one(law):
        print("  ✓ Exponents sum to 1 (as expected from C = 6ND)")
    else:
        print("  ⚠ Exponents don't sum to 1 (may indicate fitting issues)")
    print()
    print("-" * 80)
    
    # Make predictions for target budgets
    target_budgets = [1e23, 1e24]
    print("Predictions for target compute budgets:")
    print()
    for C_target in target_budgets:
        N_pred, D_pred = law.predict_both(C_target)
        print(f"  Compute Budget: {C_target:.2e} FLOPs")
        print(f"    Optimal Model Size:   {N_pred:.3e} parameters")
        print(f"    Optimal Dataset Size: {D_pred:.3e} tokens")
        print(f"    Verification: 6×N×D = {6*N_pred*D_pred:.3e} (should ≈ {C_target:.2e})")
        print()
    
    print("=" * 80)
    print("ISOFLOPS SCALING LAWS TEST COMPLETED!")
    print("=" * 80)
    
    # Generate plot if matplotlib is available
    try:
        plot_scaling_law(points, law, extrapolate_to=1e24, 
                        save_path="isoflops_scaling_law.png")
    except Exception as e:
        print(f"\nNote: Could not generate plot: {e}")

