import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2


def huber(z: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """Elementwise Huber loss (delta follows the Chinchilla replication)."""
    abs_z = np.abs(z)
    quadratic = abs_z <= delta
    out = np.empty_like(z)
    out[quadratic] = 0.5 * (z[quadratic] ** 2)
    out[~quadratic] = delta * (abs_z[~quadratic] - 0.5 * delta)
    return out


def huber_logpdf(z: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """
    Log-density whose negative equals the Huber loss up to an additive constant.
    Constants cancel in likelihood-ratio tests, so only differences matter.
    """
    return -huber(z, delta=delta)


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def load_ladder_csv(path: str) -> pd.DataFrame:
    """Load a scaling ladder CSV and drop rows missing the core columns."""
    df = pd.read_csv(path)
    required = {"N_params", "tokens_D", "loss"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")
    return df.dropna(subset=list(required))


@dataclass
class JointParams:
    E: float
    A: float
    alpha: float
    B: float
    beta: float


def _pack_joint(theta: np.ndarray) -> JointParams:
    """Map unconstrained optimizer variables to the joint model parameters."""
    e_raw, a_raw, alpha_raw, b_raw, beta_raw = theta
    E = float(np.exp(e_raw))
    A = float(np.exp(a_raw))
    # alpha_raw and beta_raw pass through softplus to stay positive
    alpha = float(softplus(alpha_raw) + 1e-6)
    B = float(np.exp(b_raw))
    beta = float(softplus(beta_raw) + 1e-6)
    return JointParams(E=E, A=A, alpha=alpha, B=B, beta=beta)


def fit_joint_powerlaw(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, delta: float = 1e-3
) -> JointParams:
    """
    Robust variant of the Chinchilla joint law fit:
        L(N, D) = E + A / N**alpha + B / D**beta
    Parameters are estimated via an L-BFGS minimization of the summed Huber loss
    in log space using the log-sum-exp trick from the Epoch AI replication.
    """
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    logN = np.log(N)
    logD = np.log(D)
    logL = np.log(L)

    def objective(theta: np.ndarray) -> float:
        params = _pack_joint(theta)
        logA = np.log(params.A)
        logB = np.log(params.B)
        logE = np.log(params.E)
        stacked = np.vstack(
            [
                logA - params.alpha * logN,
                logB - params.beta * logD,
                np.full_like(logN, logE),
            ]
        )
        residuals = logsumexp(stacked, axis=0) - logL
        return float(np.sum(huber(residuals, delta=delta)))

    theta0 = np.array(
        [
            np.log(np.median(L)),
            np.log(np.median(L) * np.mean(N) ** 0.3),
            np.log(0.3),
            np.log(np.median(L) * np.mean(D) ** 0.3),
            np.log(0.3),
        ]
    )

    result = minimize(objective, theta0, method="L-BFGS-B")
    return _pack_joint(result.x)


def predict_joint(params: JointParams, N: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Evaluate the fitted joint law."""
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    return params.E + params.A / (N ** params.alpha) + params.B / (D ** params.beta)


@dataclass
class TwoTermParams:
    E: float
    A: float
    alpha: float
    B: float
    beta: float


def _pack_two_term(theta: np.ndarray) -> TwoTermParams:
    """
    Map unconstrained optimizer parameters to the two-term model while enforcing
    beta < alpha through a sigmoid fraction.
    """
    u_alpha, v_frac, e_raw, a_raw, b_raw = theta
    alpha = float(softplus(u_alpha) + 1e-6)
    beta = float(max(1e-6, alpha * expit(v_frac)))
    E = float(np.exp(e_raw))
    A = float(np.exp(a_raw))
    B = float(np.exp(b_raw))
    return TwoTermParams(E=E, A=A, alpha=alpha, B=B, beta=beta)


def fit_two_term_n_only(N: np.ndarray, L: np.ndarray, delta: float = 1e-3) -> TwoTermParams:
    """
    Fit the N-only curve:
        L(N) = E + A / N**alpha + B / N**beta    with beta < alpha.
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)
    logN = np.log(N)
    logL = np.log(L)

    def objective(theta: np.ndarray) -> float:
        params = _pack_two_term(theta)
        t1 = np.log(params.A) - params.alpha * logN
        t2 = np.log(params.B) - params.beta * logN
        t3 = np.log(params.E) + 0.0 * logN
        residuals = logsumexp(np.vstack([t1, t2, t3]), axis=0) - logL
        return float(np.sum(huber(residuals, delta=delta)))

    theta0 = np.array(
        [
            np.log(0.3),
            0.0,
            np.log(np.median(L)),
            np.log(np.median(L) * np.mean(N) ** 0.3),
            np.log(np.median(L) * np.mean(N) ** 0.1),
        ]
    )

    result = minimize(objective, theta0, method="L-BFGS-B")
    return _pack_two_term(result.x)


def predict_two_term(params: TwoTermParams, N: np.ndarray) -> np.ndarray:
    """Evaluate the two-term N-only law."""
    N = np.asarray(N, dtype=float)
    return params.E + params.A / (N ** params.alpha) + params.B / (N ** params.beta)


def huber_loglike(residuals: np.ndarray, delta: float = 1e-3) -> float:
    """Aggregate Huber log-likelihood."""
    return float(np.sum(huber_logpdf(residuals, delta=delta)))


def lr_test(
    loglike_small: float, k_small: int, loglike_big: float, k_big: int
) -> Dict[str, float]:
    """Likelihood-ratio test between nested models."""
    stat = 2.0 * (loglike_big - loglike_small)
    dof = k_big - k_small
    p_value = 1.0 - chi2.cdf(stat, dof)
    return {"statistic": float(stat), "dof": float(dof), "p_value": float(p_value)}


def residuals_joint(params: JointParams, N: np.ndarray, D: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Log-space residuals for the joint model."""
    L_hat = predict_joint(params, N, D)
    return np.log(L_hat) - np.log(L)


def residuals_two_term(params: TwoTermParams, N: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Log-space residuals for the two-term N-only model."""
    L_hat = predict_two_term(params, N)
    return np.log(L_hat) - np.log(L)


def compute_optimal_split(params: JointParams) -> Dict[str, float]:
    """
    Compute the exponents a and b for the compute-optimal allocation with
    the assumption C ≈ k * N * D. Returns {\"a\": params-exponent, \"b\": tokens-exponent}.
    """
    a = params.beta / (params.alpha + params.beta)
    b = 1.0 - a
    return {"a": float(a), "b": float(b)}


def tokens_per_param_opt(params: JointParams, N_value: float) -> float:
    """
    Compute the tokens-per-parameter ratio at a specific N under the compute-optimal
    policy derived from the joint model.
    """
    const = ((params.beta * params.B) / (params.alpha * params.A)) ** (1.0 / params.beta)
    return float(const * (N_value ** (params.alpha / params.beta - 1.0)))


def synth_joint(
    N: np.ndarray,
    D: np.ndarray,
    *,
    E: float = 1.8,
    A: float = 500.0,
    alpha: float = 0.35,
    B: float = 2000.0,
    beta: float = 0.36,
    noise: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Generate synthetic ladder data with multiplicative log-normal noise."""
    rng = np.random.default_rng(seed)
    L = E + A / (N ** alpha) + B / (D ** beta)
    return L * np.exp(rng.normal(0.0, noise, size=L.shape))


def run_synthetic_demo(seed: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Convenience function used by tests: generate synthetic data, fit both models,
    and return a summary dict with likelihood-ratio statistics.
    """
    Ns = np.geomspace(1e7, 1e11, 20)
    Ds = np.geomspace(1e9, 1e12, 20)
    NN, DD = np.meshgrid(Ns, Ds)
    flat_N = NN.ravel()
    flat_D = DD.ravel()
    L = synth_joint(flat_N, flat_D, seed=seed)

    joint = fit_joint_powerlaw(flat_N, flat_D, L)
    unique_D = np.unique(flat_D)
    log_median_D = np.log(np.median(flat_D))
    target_idx = np.argmin(np.abs(np.log(unique_D) - log_median_D))
    target_D = unique_D[target_idx]
    mask = flat_D == target_D
    if np.sum(mask) < 3:
        raise RuntimeError("Expected at least three points in the N-ladder slice.")
    two_term = fit_two_term_n_only(flat_N[mask], L[mask])

    def fit_one_term_n_only(N: np.ndarray, L_vals: np.ndarray) -> TwoTermParams:
        """Baseline single-term fit used for LR comparison."""
        def objective(theta: np.ndarray) -> float:
            u_alpha, e_raw, a_raw = theta
            alpha = float(softplus(u_alpha) + 1e-6)
            E = float(np.exp(e_raw))
            A = float(np.exp(a_raw))
            # Emulate a one-term curve by fixing B near zero.
            B = 1e-30
            beta = 0.5 * alpha
            logN = np.log(N)
            logL_vals = np.log(L_vals)
            t1 = np.log(A) - alpha * logN
            t2 = np.log(B) - beta * logN
            t3 = np.log(E) + 0.0 * logN
            residuals = logsumexp(np.vstack([t1, t2, t3]), axis=0) - logL_vals
            return float(np.sum(huber(residuals)))

        theta0 = np.array(
            [
                np.log(0.3),
                np.log(np.median(L_vals)),
                np.log(np.median(L_vals)),
            ]
        )
        result = minimize(objective, theta0, method="L-BFGS-B")
        alpha = float(softplus(result.x[0]) + 1e-6)
        return TwoTermParams(E=float(np.exp(result.x[1])), A=float(np.exp(result.x[2])), alpha=alpha, B=0.0, beta=1e-6)

    one_term = fit_one_term_n_only(flat_N[mask], L[mask])

    residual_one = residuals_two_term(one_term, flat_N[mask], L[mask])
    residual_two = residuals_two_term(two_term, flat_N[mask], L[mask])
    ll_one = huber_loglike(residual_one)
    ll_two = huber_loglike(residual_two)
    lr_stats = lr_test(ll_one, k_small=3, loglike_big=ll_two, k_big=5)

    return {
        "joint": joint.__dict__,
        "two_term": two_term.__dict__,
        "lr_test": lr_stats,
        "compute_optimal": compute_optimal_split(joint),
    }


if __name__ == "__main__":
    summary = run_synthetic_demo()
    print("Joint parameters:", summary["joint"])
    print("Two-term parameters:", summary["two_term"])
    print("Likelihood-ratio test:", summary["lr_test"])
    print("Compute-optimal split:", summary["compute_optimal"])
