"""DM-1: power analysis for a second power-law term in L(N) = E + A N^-alpha + B N^-beta (see PREREG.md).

Plants the 2-term law on synthetic ladders, adds Gaussian loss noise, fits 1-term and 2-term laws, and records the
Wald z for B, dBIC, the conditioning of the 2-term fit and the extrapolation error at 100 x N_max.

    python dm1.py --out results --workers 4                       # whole sweep on one machine
    python dm1.py --out results --shard $SLURM_ARRAY_TASK_ID --nshards 20 --workers 1   # one Slurm array task
"""
import argparse
import csv
import glob
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize, minimize_scalar

E0, A0, ALPHA0 = 1.69, 406.4, 0.34
N_REF = 1e6
A_SCALED = A0 * N_REF ** -ALPHA0  # main-term size at N_REF; the fit works in n = N / N_REF
S_GRID = [0.0, 0.03, 0.1, 0.3, 1.0]
BETA_GRID = [0.68, 1.0]
WINDOWS = [(1e5, 1e7), (1e7, 1e9), (1e6, 1e9), (1e5, 1e9)]
PPD_GRID = [2, 4, 8]
SIGMA_GRID = [0.005, 0.01, 0.02]
REPS_ALT, REPS_NULL = 100, 400
PRED_FACTOR = 100.0

FIELDS = ["cell", "rep", "nmin", "nmax", "ppd", "npts", "sigma", "s", "beta", "testable",
          "E1", "a1", "alpha1", "rss1", "E2", "a2", "alpha2", "b2", "beta2", "rss2",
          "z_wald", "dbic", "cond", "detect_wald", "detect_bic", "detect_primary",
          "L_true_pred", "err1_pred", "errsel_pred"]


def cells():
    out = []
    for (nmin, nmax), ppd, sigma, s, beta in itertools.product(WINDOWS, PPD_GRID, SIGMA_GRID, S_GRID, BETA_GRID):
        if s == 0.0 and beta != BETA_GRID[0]:
            continue  # the null does not depend on beta
        out.append(dict(nmin=nmin, nmax=nmax, ppd=ppd, sigma=sigma, s=s, beta=beta))
    for i, c in enumerate(out):
        c["cell"] = i
        c["reps"] = REPS_NULL if c["s"] == 0.0 else REPS_ALT
    return out


def ladder(nmin, nmax, ppd):
    k = int(round(np.log10(nmax / nmin) * ppd)) + 1
    return np.logspace(np.log10(nmin), np.log10(nmax), k)


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


def run_one(c, rep):
    rng = np.random.default_rng([int(c["cell"]), int(rep), 20261019])
    N = ladder(c["nmin"], c["nmax"], c["ppd"])
    n = N / N_REF
    b_true = c["s"] * A_SCALED  # share s of the main term at N_REF (n = 1)
    y = law(n, E0, A_SCALED, ALPHA0, b_true, c["beta"]) + rng.normal(0, c["sigma"], len(n))
    npred = PRED_FACTOR * c["nmax"] / N_REF
    L_true = law(npred, E0, A_SCALED, ALPHA0, b_true, c["beta"])
    row = dict(cell=c["cell"], rep=rep, nmin=c["nmin"], nmax=c["nmax"], ppd=c["ppd"], npts=len(n),
               sigma=c["sigma"], s=c["s"], beta=c["beta"], L_true_pred=L_true)
    f1 = fit1(n, y)
    p1 = law(npred, f1["E"], f1["a"], f1["alpha"])
    row.update(E1=f1["E"], a1=f1["a"], alpha1=f1["alpha"], rss1=f1["rss"], err1_pred=p1 - L_true)
    if len(n) <= 6:  # 5 parameters need >= 2 residual dof for a usable variance estimate
        row.update(testable=0, detect_wald=0, detect_bic=0, detect_primary=0, errsel_pred=p1 - L_true)
        return row
    f2 = fit2(n, y, f1, rng)
    dbic = bic(f1["rss"], len(n), 3) - bic(f2["rss"], len(n), 5)
    dw, db = int(f2["z"] > 2), int(dbic > 2)
    prim = dw * db
    p2 = law(npred, f2["E"], f2["a"], f2["alpha"], f2["b"], f2["beta"])
    row.update(testable=1, E2=f2["E"], a2=f2["a"], alpha2=f2["alpha"], b2=f2["b"], beta2=f2["beta"],
               rss2=f2["rss"], z_wald=f2["z"], dbic=dbic, cond=f2["cond"], detect_wald=dw, detect_bic=db,
               detect_primary=prim, errsel_pred=(p2 if prim else p1) - L_true)
    return row


def run_job(job):
    c, reps = job
    return [run_one(c, r) for r in reps]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--reps-scale", type=float, default=1.0, help="multiply replicate counts (smoke tests)")
    args = ap.parse_args()
    done = set()
    for p in glob.glob(os.path.join(args.out, "raw_shard*.csv")):  # resume: skip replicates already written
        with open(p) as fh:
            done |= {(int(r["cell"]), int(r["rep"])) for r in csv.DictReader(fh)}
    jobs = []
    for c in cells():
        reps = [r for r in range(max(1, int(c["reps"] * args.reps_scale))) if (c["cell"], r) not in done]
        for i in range(0, len(reps), 25):
            jobs.append((c, reps[i:i + 25]))
    jobs = jobs[args.shard::args.nshards]
    print(f"{len(done)} replicates already done, {sum(len(j[1]) for j in jobs)} to run", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"raw_shard{args.shard:03d}_of{args.nshards:03d}.csv")
    k = 0
    while os.path.exists(path):  # never overwrite a partial run
        k += 1
        path = os.path.join(args.out, f"raw_shard{args.shard:03d}_of{args.nshards:03d}_part{k}.csv")
    t0 = time.time()
    with open(path, "w", newline="") as f, Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k, rows in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            w.writerows(rows)
            if k % 50 == 0:
                print(f"{k + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
