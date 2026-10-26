"""Fits for DM-2: DM-1's N-only variable-projection fits (copied verbatim from dm1.py at DM-1 commit 63560e5, from
`def law` to `def bic`) plus the joint (N, D) fit with an optional second N term, and a residual bootstrap.

N-only:  L = E + a n^-alpha [+ b n^-beta],                  n = N / 1e6, beta = alpha + delta > alpha
Joint:   L = E + a n^-alpha [+ b n^-beta] + g d^-gamma,     d = D / 1e9 tokens
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

# ---- verbatim from dm1.py (63560e5) ----

def law(n, E, a, alpha, b=0.0, beta=1.0):
    return E + a * n ** -alpha + b * n ** -beta


ALPHA_GRID = np.linspace(0.01, 2.0, 400)
A2_GRID, D2_GRID = np.linspace(0.02, 1.5, 60), np.linspace(0.02, 3.0, 60)


def _solve(X, y):
    """Batched linear least squares: X (G, n, k), y (n,) -> coefficients (G, k), RSS (G,)."""
    XtX = np.einsum("gni,gnj->gij", X, X)
    Xty = np.einsum("gni,n->gi", X, y)
    coef = np.linalg.solve(XtX + 1e-12 * np.eye(X.shape[2]), Xty[..., None])[..., 0]
    r = np.einsum("gnk,gk->gn", X, coef) - y
    return coef, (r ** 2).sum(1)


def _profile1(n, y, alphas):
    alphas = np.atleast_1d(alphas)
    X = np.stack([np.ones((len(alphas), len(n))), n[None, :] ** -alphas[:, None]], axis=2)
    coef, rss = _solve(X, y)
    bad = (coef[:, 0] < 0) | (coef[:, 1] < 0)
    return coef, np.where(bad, np.inf, rss)


def _profile2(n, y, alphas, betas):
    X = np.stack([np.ones((len(alphas), len(n))), n[None, :] ** -alphas[:, None], n[None, :] ** -betas[:, None]], axis=2)
    coef, rss = _solve(X, y)
    bad = (coef < 0).any(1)
    return coef, np.where(bad, np.inf, rss)


def fit1(n, y):
    """1-term fit by variable projection: grid over alpha, linear solve for (E, a), then bounded refinement."""
    coef, rss = _profile1(n, y, ALPHA_GRID)
    i = int(np.argmin(rss))
    lo, hi = ALPHA_GRID[max(i - 1, 0)], ALPHA_GRID[min(i + 1, len(ALPHA_GRID) - 1)]
    r = minimize_scalar(lambda al: _profile1(n, y, al)[1][0], bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-7})
    al = r.x if r.fun <= rss[i] else ALPHA_GRID[i]
    c, s = _profile1(n, y, al)
    return dict(E=c[0, 0], a=c[0, 1], alpha=float(al), rss=float(s[0]))


def fit2(n, y, f1, rng=None):
    """2-term fit by variable projection over (alpha, delta = beta - alpha) with E, a, b >= 0; b = 0 is allowed
    (the 1-term fit), so RSS_2 <= RSS_1 always."""
    ln = np.log(n)
    AA, DD = np.meshgrid(A2_GRID, D2_GRID, indexing="ij")
    aa, dd = AA.ravel(), DD.ravel()
    coef, rss = _profile2(n, y, aa, aa + dd)
    i = int(np.argmin(rss))
    best = dict(E=f1["E"], a=f1["a"], alpha=f1["alpha"], b=0.0, beta=f1["alpha"] + 1.0, rss=f1["rss"])
    if np.isfinite(rss[i]) and rss[i] < best["rss"]:
        def obj(p):
            if not (0.005 < p[0] < 2.5 and 0.005 < p[1] < 4.0):
                return 1e9
            v = _profile2(n, y, np.array([p[0]]), np.array([p[0] + p[1]]))[1][0]
            return v if np.isfinite(v) else 1e9
        r = minimize(obj, [aa[i], dd[i]], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 2000})
        p = r.x if r.fun < rss[i] else np.array([aa[i], dd[i]])
        c, s = _profile2(n, y, np.array([p[0]]), np.array([p[0] + p[1]]))
        if np.isfinite(s[0]) and s[0] < best["rss"]:
            best = dict(E=c[0, 0], a=c[0, 1], alpha=float(p[0]), b=c[0, 2], beta=float(p[0] + p[1]), rss=float(s[0]))
    E, a, al, b, beta, rss2 = (best[k] for k in ("E", "a", "alpha", "b", "beta", "rss"))
    # Wald test for B in natural parameters (E, a, alpha, b, beta)
    J = np.column_stack([np.ones_like(n), n ** -al, -a * n ** -al * ln, n ** -beta, -b * n ** -beta * ln])
    dof = len(n) - 5
    cov = max(rss2, 1e-300) / dof * np.linalg.pinv(J.T @ J)
    se_b = np.sqrt(cov[3, 3]) if cov[3, 3] > 0 else np.nan
    z = b / se_b if b > 0 and np.isfinite(se_b) and se_b > 0 else 0.0
    norms = np.linalg.norm(J, axis=0)
    Js = J / np.where(norms > 0, norms, 1.0)
    cond = np.linalg.cond(Js.T @ Js)
    return dict(E=E, a=a, alpha=al, b=b, beta=beta, rss=rss2, z=z, cond=cond)


def bic(rss, npts, k):
    return npts * np.log(max(rss, 1e-300) / npts) + k * np.log(npts)

# ---- end of the verbatim block ----


def nparams_ok(npts):
    return npts > 6  # DM-1's rule: 5 parameters need >= 2 residual dof


def _profileJ(n, d, y, al, be, ga, with_b):
    """Batched joint design; al, be, ga are 1-D arrays of the same length G."""
    cols = [np.ones((len(al), len(n))), n[None, :] ** -al[:, None]]
    if with_b:
        cols.append(n[None, :] ** -be[:, None])
    cols.append(d[None, :] ** -ga[:, None])
    X = np.stack(cols, axis=2)
    coef, rss = _solve(X, y)
    return coef, np.where((coef < 0).any(1), np.inf, rss)


def _chunked(n, d, y, al, be, ga, with_b, chunk=4000):
    out_c, out_r = [], []
    for i in range(0, len(al), chunk):
        c, r = _profileJ(n, d, y, al[i:i + chunk], be[i:i + chunk], ga[i:i + chunk], with_b)
        out_c.append(c)
        out_r.append(r)
    return np.concatenate(out_c), np.concatenate(out_r)


def fitJ1(n, d, y, k=50, x0=None):
    """Joint fit without the second N term: grid over (alpha, gamma), linear solve, Nelder-Mead refinement."""
    g = np.linspace(0.01, 2.0, k)
    AA, GG = np.meshgrid(g, g, indexing="ij")
    al, ga = AA.ravel(), GG.ravel()
    coef, rss = _chunked(n, d, y, al, al, ga, False)
    starts = [np.array([al[np.argmin(rss)], ga[np.argmin(rss)]])]
    if x0 is not None:
        starts.append(np.asarray(x0, float))

    def obj(p):
        if not (0.005 < p[0] < 3.0 and 0.005 < p[1] < 3.0):
            return 1e9
        v = _chunked(n, d, y, p[:1], p[:1], p[1:], False)[1][0]
        return v if np.isfinite(v) else 1e9
    best = None
    for s in starts:
        r = minimize(obj, s, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 2000})
        for p, f in ((r.x, r.fun), (s, obj(s))):
            if best is None or f < best[1]:
                best = (p, f)
    p = best[0]
    c, s = _chunked(n, d, y, p[:1], p[:1], p[1:], False)
    return dict(E=c[0, 0], a=c[0, 1], alpha=float(p[0]), b=0.0, beta=float(p[0]) + 1.0, g=c[0, 2],
                gamma=float(p[1]), rss=float(s[0]))


def fitJ2(n, d, y, f1, k=(24, 24, 24), x0=None):
    """Joint fit with the second N term (b >= 0, beta = alpha + delta); b = 0 is allowed, so RSS_2 <= RSS_1."""
    ln, ld = np.log(n), np.log(d)
    A_, D_, G_ = np.linspace(0.02, 1.5, k[0]), np.linspace(0.02, 3.0, k[1]), np.linspace(0.02, 1.5, k[2])
    AA, DD, GG = np.meshgrid(A_, D_, G_, indexing="ij")
    al, dl, ga = AA.ravel(), DD.ravel(), GG.ravel()
    coef, rss = _chunked(n, d, y, al, al + dl, ga, True)
    best = dict(E=f1["E"], a=f1["a"], alpha=f1["alpha"], b=0.0, beta=f1["alpha"] + 1.0, g=f1["g"], gamma=f1["gamma"],
                rss=f1["rss"])

    def obj(p):
        if not (0.005 < p[0] < 3.0 and 0.005 < p[1] < 5.0 and 0.005 < p[2] < 3.0):
            return 1e9
        v = _chunked(n, d, y, p[:1], p[:1] + p[1:2], p[2:], True)[1][0]
        return v if np.isfinite(v) else 1e9
    starts = []
    if np.isfinite(rss).any():
        i = int(np.argmin(rss))
        starts.append(np.array([al[i], dl[i], ga[i]]))
    if x0 is not None:
        starts.append(np.asarray(x0, float))
    for s in starts:
        r = minimize(obj, s, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 3000})
        for p, f in ((r.x, r.fun), (s, obj(s))):
            if f < best["rss"]:
                c, ss = _chunked(n, d, y, p[:1], p[:1] + p[1:2], p[2:], True)
                if np.isfinite(ss[0]):
                    best = dict(E=c[0, 0], a=c[0, 1], alpha=float(p[0]), b=c[0, 2], beta=float(p[0] + p[1]),
                                g=c[0, 3], gamma=float(p[2]), rss=float(ss[0]))
    E, a, al_, b, be, g, ga_ = (best[q] for q in ("E", "a", "alpha", "b", "beta", "g", "gamma"))
    J = np.column_stack([np.ones_like(n), n ** -al_, -a * n ** -al_ * ln, n ** -be, -b * n ** -be * ln,
                         d ** -ga_, -g * d ** -ga_ * ld])
    se_b, cond = _scaled_se(J, best["rss"], len(n) - 7, 3)
    z = b / se_b if b > 0 and np.isfinite(se_b) and se_b > 0 else 0.0
    best.update(z=z, se_b=se_b, cond=cond)
    return best


def _scaled_se(J, rss, dof, k):
    """SE of parameter k from sigma^2 (J^T J)^-1, inverted with unit-norm columns. DM-1's fit2 inverts the unscaled
    J^T J with pinv; when every n >> 1 and beta is large, the n^-beta column is ~1e-6 and pinv's cutoff drops that
    direction, so SE(B) collapses to ~0 and z explodes (found in DM-2, Oct 19). Same formula, stable inversion."""
    norms = np.linalg.norm(J, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    Js = J / norms
    M = Js.T @ Js
    cov = max(rss, 1e-300) / dof * np.linalg.pinv(M, rcond=1e-13) / np.outer(norms, norms)
    se = np.sqrt(cov[k, k]) if cov[k, k] > 0 else np.nan
    return se, np.linalg.cond(M)


def waldN(n, f2):
    """Wald z and SE(B) for an N-only 2-term fit, with the stable inversion (replaces fit2's z)."""
    E, a, al, b, be = (f2[q] for q in ("E", "a", "alpha", "b", "beta"))
    ln = np.log(n)
    J = np.column_stack([np.ones_like(n), n ** -al, -a * n ** -al * ln, n ** -be, -b * n ** -be * ln])
    se, cond = _scaled_se(J, f2["rss"], len(n) - 5, 3)
    z = b / se if b > 0 and np.isfinite(se) and se > 0 else 0.0
    return z, se, cond


def predN(f, n):
    return law(n, f["E"], f["a"], f["alpha"], f.get("b", 0.0), f.get("beta", 1.0))


def predJ(f, n, d):
    return predN(f, n) + f["g"] * d ** -f["gamma"]

