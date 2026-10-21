"""DM-5: rolling-origin small-to-large extrapolation, three rules (PREREG_dm5.md).

For every ladder and every k = 4..n-1: fit the smallest k sizes, predict every larger size with
  one   1-term,
  lrt   DM-1b bootstrap-LRT-selected 2-term (B = 200; 1-term when k <= 6, the DM-1 testable rule),
  drop1 / drop2   1-term after dropping the smallest 1 / 2 fit sizes (needs >= 4 sizes left).

    python dm5.py --out results/folds.csv --workers 3            # the grid on results/ladders.csv
    python dm5.py --out results/sample.csv --sample 0.01          # timing sample (random 1% of the jobs)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dm1  # noqa: E402
import dm1b  # noqa: E402

B_BOOT, K_MAX, B_STAGE1 = dm1b.B_BOOT, dm1b.K_MAX, dm1b.B_STAGE1
MIN_FIT = 4
SEED = 20261021


def law(n, f):
    return dm1.law(n, f["E"], f["a"], f["alpha"], f.get("b", 0.0), f.get("beta", 1.0))


def one(n, y):
    f1, f2 = dm1b.fit_chunked(n, y[None, :])
    return {k: float(v[0]) for k, v in f1.items()}, {k: float(v[0]) for k, v in f2.items()}


def lrt_rule(n, y, rng):
    """DM-1b's detection rule on one series: returns (f1, f2, detect, p, T, tested)."""
    f1, f2 = one(n, y)
    npts = len(n)
    if npts <= 6:
        return f1, f2, 0, np.nan, np.nan, 0, 0
    T = dm1b.lrt_stat(npts, f1["rss"], f2["rss"])
    if not T > 0:
        return f1, f2, 0, 1.0, float(T), 1, 0
    sig = np.sqrt(f1["rss"] / (npts - 3))
    mu = f1["E"] + f1["a"] * n ** -f1["alpha"]
    eps = rng.normal(0, 1, (B_BOOT, npts))
    kex, used, nbad = 0, 0, 0
    for lo, hi in ((0, B_STAGE1), (B_STAGE1, B_BOOT)):
        g1, g2 = dm1b.fit_chunked(n, mu[None, :] + sig * eps[lo:hi])
        Ts = dm1b.lrt_stat(npts, g1["rss"], g2["rss"])
        bad = ~np.isfinite(Ts)  # a refit with no feasible solution counts as an exceedance (conservative)
        nbad += int(bad.sum())
        kex += int(((Ts >= T - 1e-9 * max(T, 1.0)) | bad).sum())
        used = hi
        if kex > K_MAX:
            break
    detect = int(used == B_BOOT and kex <= K_MAX)
    return f1, f2, detect, (1 + kex) / (used + 1), float(T), 1, nbad


def run_job(job):
    """One fold of one ladder: all rules, one row per (rule, target)."""
    ladder, conv, primary, N, y, k, jid = job
    n = N / dm1.N_REF
    nf, yf = n[:k], y[:k]
    rng = np.random.default_rng([SEED, jid])
    rows = []
    tg = np.arange(k, len(n))

    def emit(rule, f, defined, extra):
        for j in tg:
            pred = float(law(n[j], f)) if defined else np.nan
            rows.append(dict(ladder=ladder, convention=conv, primary=primary, k=k, rule=rule, defined=int(defined),
                             N_fit_max=float(N[k - 1]), N_pred=float(N[j]), ratio=float(N[j] / N[k - 1]),
                             y=float(y[j]), pred=pred, err=pred - float(y[j]), abs_err=abs(pred - float(y[j])),
                             **extra))

    f1, f2, det, p, T, tested, nbad = lrt_rule(nf, yf, rng)
    emit("one", f1, True, dict(alpha=f1["alpha"], used2=0, p_lrt=np.nan, lrt_tested=0, boot_nonfinite=0))
    emit("lrt", f2 if det else f1, True, dict(alpha=f2["alpha"] if det else f1["alpha"], used2=det, p_lrt=p,
                                              lrt_tested=tested, boot_nonfinite=nbad))
    for d in (1, 2):
        ok = k - d >= MIN_FIT
        fd = one(nf[d:], yf[d:])[0] if ok else None
        emit(f"drop{d}", fd if ok else f1, ok, dict(alpha=fd["alpha"] if ok else np.nan, used2=0, p_lrt=np.nan,
                                                    lrt_tested=0))
    return rows


def jobs_from(ladders, only_primary=False):
    jobs, jid = [], 0
    for (lad, conv, prim), g in ladders.groupby(["ladder", "convention", "primary"], sort=True):
        if only_primary and not prim:
            continue
        g = g.sort_values("N")
        N, y = g.N.values.astype(float), g.y.values.astype(float)
        for k in range(MIN_FIT, len(N)):
            jobs.append((lad, conv, int(prim), N, y, k, jid))
            jid += 1
    return jobs


def run(jobs, workers):
    with Pool(workers) as pool:
        out = pool.map(run_job, jobs, chunksize=1)
    df = pd.DataFrame([r for o in out for r in o])
    df["boot_nonfinite"] = df.boot_nonfinite.fillna(0).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ladders", default=os.path.join(HERE, "results", "ladders.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "folds.csv"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of jobs, random, for timing")
    a = ap.parse_args()
    jobs = jobs_from(pd.read_csv(a.ladders))
    if a.sample < 1.0:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), max(1, int(a.sample * len(jobs))), replace=False))]
    t0 = time.time()
    df = run(jobs, a.workers)
    df.to_csv(a.out, index=False)
    print(f"{len(jobs)} folds, {len(df)} rows, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
