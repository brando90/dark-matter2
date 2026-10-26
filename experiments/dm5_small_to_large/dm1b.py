"""DM-1b: parametric-bootstrap likelihood-ratio test for a second power-law term, on DM-1's replicates (PREREG.md).

Regenerates each DM-1 replicate's data from its seed, fits the 1-term and 2-term laws with a vectorized
variable-projection fitter, and calibrates T = n ln(RSS_1 / RSS_2) by B refits of data simulated under the fitted
1-term null. Writes one row per replicate.

    python dm1b.py --out results_dm1b --workers 4                         # whole sweep on one machine
    python dm1b.py --out results_dm1b --shard $SLURM_ARRAY_TASK_ID --nshards 20 --workers 1   # one Slurm array task
"""
import argparse
import csv
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

import dm1

B_BOOT = 200
ALPHA_LEVEL = 0.05
CHUNK = 1000  # series fit at once
B_STAGE1 = 50
K_MAX = int(np.floor(ALPHA_LEVEL * (B_BOOT + 1))) - 1  # detect iff (1 + k) / (B + 1) <= alpha, i.e. k <= 9
DELTA_BANDS = [(0.0, 0.8), (0.8, 2.0), (2.0, 4.01)]

FIELDS = ["cell", "rep", "nmin", "nmax", "ppd", "npts", "sigma", "s", "beta", "testable",
          "E1", "a1", "alpha1", "rss1", "E2", "a2", "alpha2", "b2", "beta2", "rss2",
          "T", "k_exceed", "b_used", "p_lrt", "detect_lrt", "sigma_hat",
          "L_true_pred", "err1_pred", "errsel_lrt_pred"]


def _proj(X, Yc, ym):
    """Nonnegative-checked least squares with an intercept for many series at once, by stable orthogonalization
    (classical Gram-Schmidt applied twice) of the centred power columns, so RSS = |y_c|^2 - |Q^T y_c|^2 is accurate even
    when n^-alpha and n^-beta are nearly collinear. X: (..., k, n) raw columns (k = 1 or 2), shared ((G, k, n) ->
    results (G, R)) or one set per series ((R, C, k, n) -> results (R, C)). Returns (E, slopes (..., k), rss), rss =
    inf where E or a slope is < 0."""
    xm = X.mean(-1)
    Xc = X - xm[..., None]
    shared = X.ndim == 3
    dot = (lambda Q: np.einsum("gn,rn->gr", Q, Yc)) if shared else (lambda Q: np.einsum("rcn,rn->rc", Q, Yc))
    yy = (Yc ** 2).sum(1)[None, :] if shared else (Yc ** 2).sum(1)[:, None]
    x1 = Xc[..., 0, :]
    r11 = np.linalg.norm(x1, axis=-1)
    q1 = x1 / r11[..., None]
    c1 = dot(q1)
    if X.shape[-2] == 1:
        a = c1 / (r11[:, None] if shared else r11)
        E = (ym[None, :] - a * xm[:, None, 0]) if shared else (ym[:, None] - a * xm[..., 0])
        rss = np.maximum(yy - c1 ** 2, 0.0)
        return E, a[..., None], np.where((E < 0) | (a < 0), np.inf, rss)
    x2 = Xc[..., 1, :]
    r12 = (q1 * x2).sum(-1)
    v = x2 - r12[..., None] * q1
    t = (q1 * v).sum(-1)
    v = v - t[..., None] * q1
    r12 = r12 + t
    r22 = np.linalg.norm(v, axis=-1)
    r22 = np.where(r22 > 0, r22, 1e-300)
    q2 = v / r22[..., None]
    c2 = dot(q2)
    if shared:
        b = c2 / r22[:, None]
        a = (c1 - r12[:, None] * b) / r11[:, None]
        E = ym[None, :] - a * xm[:, None, 0] - b * xm[:, None, 1]
    else:
        b = c2 / r22
        a = (c1 - r12 * b) / r11
        E = ym[:, None] - a * xm[..., 0] - b * xm[..., 1]
    rss = np.maximum(yy - c1 ** 2 - c2 ** 2, 0.0)
    return E, np.stack([a, b], -1), np.where((E < 0) | (a < 0) | (b < 0), np.inf, rss)


def _pow(n, exps):
    """Power columns n^-e for exponents exps (..., k) -> (..., k, n)."""
    return np.exp(-exps[..., None] * np.log(n))


def fit_batch(n, Y, max_rounds=150):
    """1-term and 2-term fits of every row of Y (R, n). Grid search on DM-1's grids, then a compass search per series
    (move to the best neighbour and double the step, up to 4 grid steps, or halve it when the centre is best) until
    the step is < 1e-7 (1-term) or 1e-3 (2-term, then polished by Levenberg-Marquardt).
    Returns dicts of arrays; rss2 = min(rss1, best 2-term rss), so the 1-term fit is the 2-term fit with b = 0."""
    ym = Y.mean(1)
    Yc = Y - ym[:, None]
    R = len(Y)
    ar = np.arange(R)
    # 1-term
    _, _, rss = _proj(_pow(n, dm1.ALPHA_GRID[:, None]), Yc, ym)
    al = dm1.ALPHA_GRID[np.argmin(rss, 0)].copy()
    best = rss.min(0)
    step = np.full(R, dm1.ALPHA_GRID[1] - dm1.ALPHA_GRID[0])
    off = np.array([-1.0, 1.0])
    for _ in range(max_rounds):
        act = np.flatnonzero(step >= 1e-7)  # only series still searching
        if not len(act):
            break
        cand = np.clip(al[act, None] + step[act, None] * off[None, :], 0.005, 2.5)
        _, _, rss = _proj(_pow(n, cand[..., None]), Yc[act], ym[act])
        j = np.argmin(rss, 1)
        v = rss[np.arange(len(act)), j]
        better = v < best[act]
        al[act] = np.where(better, cand[np.arange(len(act)), j], al[act])
        best[act] = np.where(better, v, best[act])
        step[act] = np.where(better, step[act], step[act] * 0.5)
    E1, c1, rss1 = _proj(_pow(n, al[:, None, None]), Yc, ym)
    f1 = dict(E=E1[:, 0], a=c1[:, 0, 0], alpha=al, rss=rss1[:, 0])
    # 2-term: DM-1's grid, extended to the delta bound (4.0); the RSS surface is multimodal in delta (a fast term
    # fitting only the smallest models sits at the bound), so a compass search starts from the best grid point in
    # each of three delta bands and from two boundary points (below), and the best end point is kept
    D2 = np.concatenate([dm1.D2_GRID, [3.25, 3.5, 3.75, 4.0]])
    AA, DD = np.meshgrid(dm1.A2_GRID, D2, indexing="ij")
    aa, dd = AA.ravel(), DD.ravel()
    _, _, rss = _proj(_pow(n, np.stack([aa, aa + dd], 1)), Yc, ym)
    starts = []
    for lo, hi in DELTA_BANDS:
        m = (dd >= lo) & (dd < hi)
        i = np.argmin(np.where(m[:, None], rss, np.inf), 0)
        starts.append((aa[i], dd[i], rss[i, ar]))
    # two boundary starts DM-1's Nelder-Mead reached from the grid: a "spike" (delta at its bound, the fast term fitting
    # only the smallest models) and a near-constant slow term (alpha at its bound, E -> 0)
    for sa0, sd0 in ((f1["alpha"], np.full(R, 3.99)), (np.full(R, 0.01), np.maximum(f1["alpha"] - 0.01, 0.005))):
        _, _, r0 = _proj(_pow(n, np.stack([sa0, sa0 + sd0], -1)[:, None, :]), Yc, ym)
        starts.append((sa0, sd0, r0[:, 0]))
    pa, pd, best = (np.concatenate(x) for x in zip(*starts))
    K = len(starts)
    Yk, ymk, Rk = np.tile(Yc, (K, 1)), np.tile(ym, K), K * R
    sa = np.full(Rk, dm1.A2_GRID[1] - dm1.A2_GRID[0])
    sd = np.full(Rk, dm1.D2_GRID[1] - dm1.D2_GRID[0])
    oa = np.array([-1.0, 1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0])
    od = np.array([0.0, 0.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    SA_MAX, SD_MAX = 4 * sa[0], 4 * sd[0]
    for _ in range(max_rounds):
        act = np.flatnonzero(np.maximum(sa, sd) >= 1e-3)
        if not len(act):
            break
        ra = np.arange(len(act))
        ca = np.clip(pa[act, None] + sa[act, None] * oa[None, :], 0.005, 2.5)
        cd = np.clip(pd[act, None] + sd[act, None] * od[None, :], 0.005, 4.0)
        _, _, rss = _proj(_pow(n, np.stack([ca, ca + cd], -1)), Yk[act], ymk[act])
        j = np.argmin(rss, 1)
        v = rss[ra, j]
        better = v < best[act]
        pa[act], pd[act] = np.where(better, ca[ra, j], pa[act]), np.where(better, cd[ra, j], pd[act])
        best[act] = np.where(better, v, best[act])
        sa[act] = np.where(better, np.minimum(sa[act] * 2.0, SA_MAX), sa[act] * 0.5)
        sd[act] = np.where(better, np.minimum(sd[act] * 2.0, SD_MAX), sd[act] * 0.5)
    # polish every end point with projected Levenberg-Marquardt in all five parameters (narrow curved valleys and the
    # E = 0 bound stall a compass search), then keep the best of the K
    E2, c2, _ = _proj(_pow(n, np.stack([pa, pa + pd], -1)[:, None, :]), Yk, ymk)
    ok = np.isfinite(best)
    P = np.stack([np.where(ok, E2[:, 0], 0.0), np.where(ok, c2[:, 0, 0], 0.0), pa,
                  np.where(ok, c2[:, 0, 1], 0.0), pd], 1)
    P, rssk = _lm(n, np.tile(Y, (K, 1)), P, ok)
    k = np.argmin(rssk.reshape(K, R), 0)
    P, r2 = P.reshape(K, R, 5)[k, ar], rssk.reshape(K, R)[k, ar]
    use2 = np.isfinite(r2) & (r2 < f1["rss"])
    f2 = dict(E=np.where(use2, P[:, 0], f1["E"]), a=np.where(use2, P[:, 1], f1["a"]),
              alpha=np.where(use2, P[:, 2], f1["alpha"]), b=np.where(use2, P[:, 3], 0.0),
              beta=np.where(use2, P[:, 2] + P[:, 4], f1["alpha"] + 1.0), rss=np.where(use2, r2, f1["rss"]))
    return f1, f2


LO = np.array([0.0, 0.0, 0.005, 0.0, 0.005])
HI = np.array([np.inf, np.inf, 2.5, np.inf, 4.0])


def _lm(n, Y, P, ok, iters=40):
    """Projected Levenberg-Marquardt for E + a n^-alpha + b n^-(alpha + delta), P = (E, a, alpha, b, delta) per row,
    bounds LO/HI. Rows with ok = False keep rss = inf. Returns (P, rss) with rss computed from the residuals."""
    ln = np.log(n)

    def resid(P):
        ua = np.exp(-P[:, 2:3] * ln)
        ub = np.exp(-(P[:, 2:3] + P[:, 4:5]) * ln)
        return Y - (P[:, 0:1] + P[:, 1:2] * ua + P[:, 3:4] * ub), ua, ub

    r, ua, ub = resid(P)
    rss = np.where(ok, (r ** 2).sum(1), np.inf)
    lam = np.full(len(P), 1e-3)
    for _ in range(iters):
        act = np.flatnonzero(ok & (lam < 1e8))
        if not len(act):
            break
        Pa, ra, uaa, uba = P[act], r[act], ua[act], ub[act]
        db = -Pa[:, 3:4] * ln * uba
        J = np.stack([np.ones_like(uaa), uaa, -Pa[:, 1:2] * ln * uaa + db, uba, db], 2)  # (A, n, 5)
        D = np.sqrt((J ** 2).sum(1)) + 1e-300
        Js = J / D[:, None, :]
        M = np.einsum("anj,ank->ajk", Js, Js) + lam[act, None, None] * np.eye(5)
        g = np.einsum("anj,an->aj", Js, ra)
        step = np.linalg.solve(M, g[..., None])[..., 0] / D
        Pn = np.clip(Pa + step, LO, HI)
        rn, uan, ubn = resid_rows(Y[act], Pn, ln)
        sn = (rn ** 2).sum(1)
        better = sn < rss[act]
        P[act] = np.where(better[:, None], Pn, Pa)
        r[act] = np.where(better[:, None], rn, ra)
        ua[act] = np.where(better[:, None], uan, uaa)
        ub[act] = np.where(better[:, None], ubn, uba)
        rss[act] = np.where(better, sn, rss[act])
        lam[act] = np.where(better, lam[act] / 3.0, lam[act] * 4.0)
    return P, rss


def resid_rows(Y, P, ln):
    ua = np.exp(-P[:, 2:3] * ln)
    ub = np.exp(-(P[:, 2:3] + P[:, 4:5]) * ln)
    return Y - (P[:, 0:1] + P[:, 1:2] * ua + P[:, 3:4] * ub), ua, ub


def lrt_stat(npts, rss1, rss2):
    return npts * np.log(np.maximum(rss1, 1e-300) / np.maximum(rss2, 1e-300))


def fit_chunked(n, Y):
    parts = [fit_batch(n, Y[i:i + CHUNK]) for i in range(0, len(Y), CHUNK)]
    return [{k: np.concatenate([p[j][k] for p in parts]) for k in parts[0][j]} for j in (0, 1)]


def run_cell(job):
    """All requested replicates of one cell: observed fits, then B bootstrap refits per testable replicate."""
    c, reps = job
    N = dm1.ladder(c["nmin"], c["nmax"], c["ppd"])
    n = N / dm1.N_REF
    npts = len(n)
    b_true = c["s"] * dm1.A_SCALED
    Y = np.array([dm1.law(n, dm1.E0, dm1.A_SCALED, dm1.ALPHA0, b_true, c["beta"]) +
                  np.random.default_rng([int(c["cell"]), int(r), 20261019]).normal(0, c["sigma"], npts) for r in reps])
    npred = dm1.PRED_FACTOR * c["nmax"] / dm1.N_REF
    L_true = dm1.law(npred, dm1.E0, dm1.A_SCALED, dm1.ALPHA0, b_true, c["beta"])
    f1, f2 = fit_chunked(n, Y)
    testable = npts > 6
    rows = []
    if testable:
        T = lrt_stat(npts, f1["rss"], f2["rss"])
        sig = np.sqrt(f1["rss"] / (npts - 3))
        mu = f1["E"][:, None] + f1["a"][:, None] * n[None, :] ** -f1["alpha"][:, None]
        eps = np.stack([np.random.default_rng([int(c["cell"]), int(r), 20261019, 1]).normal(0, 1, (B_BOOT, npts))
                        for r in reps])  # (reps, B, n); sample b is the same whatever stage it is used in
        # Exact shortcuts for the decision (detect iff at most K_MAX of all B bootstrap T* reach T): T = 0 means every
        # T* >= T (p = 1, no refits); after the first B_STAGE1 samples, k >= K_MAX + 1 already means no detection.
        kex = np.where(T > 0, 0, B_BOOT)
        bused = np.where(T > 0, 0, B_BOOT)
        for lo, hi in ((0, B_STAGE1), (B_STAGE1, B_BOOT)):
            idx = np.flatnonzero((T > 0) & (kex <= K_MAX))
            if not len(idx):
                break
            Ys = (mu[idx, None, :] + sig[idx, None, None] * eps[idx, lo:hi]).reshape(-1, npts)
            g1, g2 = fit_chunked(n, Ys)
            Ts = lrt_stat(npts, g1["rss"], g2["rss"]).reshape(len(idx), hi - lo)
            kex[idx] += (Ts >= T[idx, None] - 1e-9 * np.maximum(T[idx, None], 1.0)).sum(1)
            bused[idx] = hi
        p = (1 + kex) / (bused + 1)
        detect = (bused == B_BOOT) & (kex <= K_MAX)
    for k, r in enumerate(reps):
        p1 = dm1.law(npred, f1["E"][k], f1["a"][k], f1["alpha"][k])
        row = dict(cell=c["cell"], rep=r, nmin=c["nmin"], nmax=c["nmax"], ppd=c["ppd"], npts=npts, sigma=c["sigma"],
                   s=c["s"], beta=c["beta"], testable=int(testable), E1=f1["E"][k], a1=f1["a"][k],
                   alpha1=f1["alpha"][k], rss1=f1["rss"][k], L_true_pred=L_true, err1_pred=p1 - L_true)
        if testable:
            det = int(detect[k])
            p2 = dm1.law(npred, f2["E"][k], f2["a"][k], f2["alpha"][k], f2["b"][k], f2["beta"][k])
            row.update(E2=f2["E"][k], a2=f2["a"][k], alpha2=f2["alpha"][k], b2=f2["b"][k], beta2=f2["beta"][k],
                       rss2=f2["rss"][k], T=T[k], k_exceed=int(kex[k]), b_used=int(bused[k]), p_lrt=p[k], detect_lrt=det,
                       sigma_hat=sig[k], errsel_lrt_pred=(p2 if det else p1) - L_true)
        else:
            row.update(detect_lrt=0, errsel_lrt_pred=p1 - L_true)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results_dm1b")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--cells", default="", help="comma-separated cell ids to run (timing / smoke tests)")
    ap.add_argument("--reps-scale", type=float, default=1.0)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    done = set()
    for p in glob.glob(os.path.join(args.out, "raw_dm1b*.csv")):  # resume
        with open(p) as fh:
            done |= {(int(r["cell"]), int(r["rep"])) for r in csv.DictReader(fh)}
    only = {int(x) for x in args.cells.split(",") if x}
    jobs = []
    for c in dm1.cells():
        if only and c["cell"] not in only:
            continue
        reps = [r for r in range(max(1, int(c["reps"] * args.reps_scale))) if (c["cell"], r) not in done]
        for i in range(0, len(reps), 25):
            jobs.append((c, reps[i:i + 25]))
    jobs = jobs[args.shard::args.nshards]
    print(f"{len(done)} replicates already done, {sum(len(j[1]) for j in jobs)} to run", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"raw_dm1b_shard{args.shard:03d}_of{args.nshards:03d}.csv")
    k = 0
    while os.path.exists(path):
        k += 1
        path = os.path.join(args.out, f"raw_dm1b_shard{args.shard:03d}_of{args.nshards:03d}_part{k}.csv")
    t0 = time.time()
    with open(path, "w", newline="") as f, Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k, rows in enumerate(pool.imap_unordered(run_cell, jobs, chunksize=1)):
            w.writerows(rows)
            f.flush()
            if k % 20 == 0:
                print(f"{k + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
