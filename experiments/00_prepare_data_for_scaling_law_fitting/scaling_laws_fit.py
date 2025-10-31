# file: scaling_laws_fit.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import chi2

# ---------------------------
# Utilities
# ---------------------------
def huber(z, delta=1e-3):
    """Elementwise Huber loss (same delta used in the Chinchilla replication)."""
    az = np.abs(z)
    quad = az <= delta
    out = np.empty_like(z)
    out[quad] = 0.5 * (z[quad] ** 2)
    out[~quad] = delta * (az[~quad] - 0.5 * delta)
    return out

def huber_logpdf(z, delta=1e-3):
    """
    Density whose negative log-likelihood equals Huber loss (up to a constant).
    We only need log-likelihood differences for LR tests, so constants cancel.
    See Epoch AI replication Appendix A.2.
    """
    # -log p ∝ huber(z); return log p up to additive constant
    return -huber(z, delta=delta)

def softplus(x):  # stable softplus
    return np.where(x > 20, x, np.log1p(np.exp(x)))

# ---------------------------
# Data access
# ---------------------------
def load_ladder_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimal cleaning
    assert {'N_params', 'tokens_D', 'loss'}.issubset(df.columns)
    df = df.dropna(subset=['N_params', 'tokens_D', 'loss'])
    return df

# ---------------------------
# (A) Joint fit: L(N, D) = E + A/N^alpha + B/D^beta
#     Robust Chinchilla-style Huber objective with log-sum-exp
# ---------------------------
@dataclass
class JointParams:
    E: float; A: float; alpha: float; B: float; beta: float

def _pack_joint(theta):
    # Unconstrained params -> positive scales/exponents
    # theta = [e_raw, a_raw, alpha_raw, b_raw, beta_raw]
    E = np.exp(theta[0])     # keep positive
    A = np.exp(theta[1])
    alpha = softplus(theta[2]) + 1e-6
    B = np.exp(theta[3])
    beta = softplus(theta[4]) + 1e-6
    return E, A, alpha, B, beta

def fit_joint_powerlaw(N, D, L, delta=1e-3) -> JointParams:
    """
    Fit using the Huber objective used in Hoffmann et al. (Approach 3)
    as reconstructed in the Epoch AI replication: minimize
      sum Huber( LSE(a - alpha log N, b - beta log D, e) - log L ),
    where a=log A, b=log B, e=log E.
    """
    logN = np.log(N); logD = np.log(D); logL = np.log(L)

    def obj(theta):
        E, A, alpha, B, beta = _pack_joint(theta)
        a, b, e = np.log(A), np.log(B), np.log(E)
        # r_i = LSE(a - alpha log N_i, b - beta log D_i, e) - log L_i
        X = np.vstack([a - alpha * logN,
                       b - beta * logD,
                       np.full_like(logN, e)])
        r = logsumexp(X, axis=0) - logL
        return np.sum(huber(r, delta=delta))

    # Sensible init from rough scalings
    theta0 = np.array([np.log(np.median(L)),
                       np.log(np.median(L)*N.mean()**0.3),
                       np.log(0.3),  # alpha_raw ~ log(0.3) -> softplus ~ 0.3
                       np.log(np.median(L)*D.mean()**0.3),
                       np.log(0.3)])
    res = minimize(obj, theta0, method='L-BFGS-B')
    E, A, alpha, B, beta = _pack_joint(res.x)
    return JointParams(E, A, alpha, B, beta)

def predict_joint(params: JointParams, N, D):
    return params.E + params.A / (N ** params.alpha) + params.B / (D ** params.beta)

# ---------------------------
# (B) Two-term N-only fit: L(N) = E + A/N^alpha + B/N^beta, with beta < alpha
# ---------------------------
@dataclass
class TwoTermParams:
    E: float; A: float; alpha: float; B: float; beta: float

def _pack_two_term(theta):
    """
    Enforce beta < alpha by construction:
      alpha = softplus(u) + eps
      beta  = alpha * sigmoid(v)
    Keep E, A, B positive via exp.
    """
    u_alpha, v_frac, e_raw, a_raw, b_raw = theta
    alpha = softplus(u_alpha) + 1e-6
    frac = expit(v_frac)  # in (0,1)
    beta = np.maximum(1e-6, alpha * frac)
    E = np.exp(e_raw); A = np.exp(a_raw); B = np.exp(b_raw)
    return E, A, alpha, B, beta

def fit_two_term_N_only(N, L, delta=1e-3) -> TwoTermParams:
    logN = np.log(N); logL = np.log(L)
    def obj(theta):
        E, A, alpha, B, beta = _pack_two_term(theta)
        # Construct log(E + A N^{-alpha} + B N^{-beta}) via log-sum-exp in log-space
        t1 = np.log(A) - alpha * logN
        t2 = np.log(B) - beta * logN
        t3 = np.log(E) + 0.0 * logN
        # r_i = log(pred_i) - log(L_i)
        r = logsumexp(np.vstack([t1, t2, t3]), axis=0) - logL
        return np.sum(huber(r, delta=delta))
    theta0 = np.array([np.log(0.3),  # u_alpha
                       0.0,           # v_frac -> beta ~ 0.5*alpha
                       np.log(np.median(L)), 
                       np.log(np.median(L) * N.mean()**0.3),
                       np.log(np.median(L) * N.mean()**0.1)])
    res = minimize(obj, theta0, method='L-BFGS-B')
    E, A, alpha, B, beta = _pack_two_term(res.x)
    return TwoTermParams(E, A, alpha, B, beta)

def predict_two_term(p: TwoTermParams, N):
    return p.E + p.A / (N ** p.alpha) + p.B / (N ** p.beta)

# ---------------------------
# Likelihood-ratio test (Huber-as-likelihood)
# ---------------------------
def huber_loglike(residuals, delta=1e-3):
    return np.sum(huber_logpdf(residuals, delta=delta))

def lr_test(loglike_small, k_small, loglike_big, k_big):
    """Return LR statistic and p-value comparing nested models."""
    stat = 2.0 * (loglike_big - loglike_small)
    dof = k_big - k_small
    p = 1.0 - chi2.cdf(stat, dof)
    return stat, dof, p

def residuals_joint(params: JointParams, N, D, L):
    Lhat = predict_joint(params, N, D)
    return np.log(Lhat) - np.log(L)

def residuals_two_term(params: TwoTermParams, N, L):
    Lhat = predict_two_term(params, N)
    return np.log(Lhat) - np.log(L)

# ---------------------------
# Compute-optimal policy under C ≈ k N D
# For L = E + A/N^alpha + B/D^beta, the optimal exponents are:
#   N* ∝ C^{β/(α+β)},  D* ∝ C^{α/(α+β)},  and a = β/(α+β)
# The tokens-per-parameter ratio at optimum:
#   D/N = ((β B)/(α A))^{1/β} * N^{α/β - 1}
# ---------------------------
def compute_optimal_from_joint(p: JointParams, k=6.0):
    a = p.beta / (p.alpha + p.beta)
    b = 1.0 - a
    return dict(a=a, b=b)

def tokens_per_param_opt(p: JointParams, N_value):
    const = ((p.beta * p.B) / (p.alpha * p.A)) ** (1.0 / p.beta)
    return const * (N_value ** (p.alpha / p.beta - 1.0))

# ---------------------------
# Synthetic data generator (for dry runs)
# ---------------------------
def synth_joint(Ns, Ds, E=1.8, A=500.0, alpha=0.35, B=2000.0, beta=0.36, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    L = E + A / (Ns ** alpha) + B / (Ds ** beta)
    L *= np.exp(rng.normal(0.0, noise, size=L.shape))  # multiplicative noise in log-loss
    return L

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("SCALING LAWS FIT - SYNTHETIC DATA TEST")
    print("=" * 80)
    print()
    
    # Ground truth parameters
    print("Ground Truth Parameters:")
    print("  E = 1.8, A = 500.0, α = 0.35, B = 2000.0, β = 0.36")
    print()
    
    # 1) Try synthetic to verify the pipeline
    Ns = np.geomspace(1e7, 1e11, 20)
    Ds = np.geomspace(1e9, 1e12, 20)
    # make a small grid
    NN, DD = np.meshgrid(Ns, Ds); NN = NN.ravel(); DD = DD.ravel()
    L = synth_joint(NN, DD)
    
    print(f"Generated {len(L)} synthetic data points")
    print(f"N range: [{NN.min():.2e}, {NN.max():.2e}]")
    print(f"D range: [{DD.min():.2e}, {DD.max():.2e}]")
    print(f"Loss range: [{L.min():.4f}, {L.max():.4f}]")
    print()
    print("-" * 80)

    # Fit joint law
    print("(1) FITTING JOINT LAW: L(N,D) = E + A/N^α + B/D^β")
    print()
    jp = fit_joint_powerlaw(NN, DD, L)
    print(f"  E     = {jp.E:.4f}  (true: 1.8)")
    print(f"  A     = {jp.A:.2f}  (true: 500.0)")
    print(f"  α     = {jp.alpha:.4f}  (true: 0.35)")
    print(f"  B     = {jp.B:.2f}  (true: 2000.0)")
    print(f"  β     = {jp.beta:.4f}  (true: 0.36)")
    print()
    
    # Compute-optimal exponents
    opt = compute_optimal_from_joint(jp)
    print(f"  Compute-optimal split: a = β/(α+β) = {opt['a']:.4f}")
    print(f"  This means ~{opt['a']*100:.1f}% compute on params, ~{opt['b']*100:.1f}% on tokens")
    print()
    print("-" * 80)
    
    # Fit two-term N-only using a slice at fixed D (mimic an N-ladder)
    print("(2) FITTING TWO-TERM N-ONLY LAW: L(N) = E + A/N^α + B/N^β with β < α")
    print()
    # Take a narrow band around the median D value
    med_D = np.median(DD)
    mask = (DD > med_D * 0.9) & (DD < med_D * 1.1)
    print(f"  Selected {mask.sum()} points for N-ladder (D around {med_D:.2e})")
    
    if mask.sum() >= 10:  # Need enough points for fitting
        tp = fit_two_term_N_only(NN[mask], L[mask])
    else:
        # Use entire dataset for N-only fit if not enough points in band
        print("  (Using all data since not enough points in narrow D-band)")
        tp = fit_two_term_N_only(NN, L)
    
    print(f"  E     = {tp.E:.4f}")
    print(f"  A     = {tp.A:.2f}")
    print(f"  α     = {tp.alpha:.4f}")
    print(f"  B     = {tp.B:.2f}")
    print(f"  β     = {tp.beta:.4f}")
    print(f"  Constraint satisfied: β < α? {tp.beta < tp.alpha}")
    print()
    print("-" * 80)

    # LR test between 1-term (A-only) and 2-term (A+B) on the N-ladder
    # Fit a 1-term baseline by zeroing B (quick-and-dirty via optimization with B tiny)
    def fit_one_term_N_only(N, L):
        def obj(theta):
            # theta = [u_alpha, e_raw, a_raw]
            u_alpha, e_raw, a_raw = theta
            alpha = softplus(u_alpha) + 1e-6
            E = np.exp(e_raw); A = np.exp(a_raw); B = 1e-30; beta = 0.5*alpha
            t1 = np.log(A) - alpha * np.log(N)
            t2 = np.log(B) - beta * np.log(N)
            t3 = np.log(E) + 0.0 * np.log(N)
            r = logsumexp(np.vstack([t1, t2, t3]), axis=0) - np.log(L)
            return np.sum(huber(r))
        res = minimize(obj, np.array([np.log(0.3), np.log(np.median(L)), np.log(np.median(L))]), method="L-BFGS-B")
        u_alpha, e_raw, a_raw = res.x
        alpha = softplus(u_alpha) + 1e-6
        return TwoTermParams(np.exp(e_raw), np.exp(a_raw), alpha, 0.0, 1e-6)

    # Use the same data for LR test as we used for two-term fit
    print("(3) LIKELIHOOD-RATIO TEST: 1-term vs 2-term N-only model")
    print()
    test_N = NN[mask] if mask.sum() >= 10 else NN
    test_L = L[mask] if mask.sum() >= 10 else L
    
    one = fit_one_term_N_only(test_N, test_L)
    r1 = residuals_two_term(one, test_N, test_L)
    r2 = residuals_two_term(tp, test_N, test_L)
    ll1 = huber_loglike(r1); ll2 = huber_loglike(r2)
    stat, dof, p = lr_test(ll1, k_small=3, loglike_big=ll2, k_big=5)
    
    print(f"  Test statistic: {stat:.2f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p:.3e}")
    
    if p < 0.05:
        print(f"  ✓ Second term is statistically significant (p < 0.05)")
    else:
        print(f"  × Second term is NOT statistically significant (p >= 0.05)")
    print()
    print("=" * 80)
    print("SCALING LAWS FIT TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

