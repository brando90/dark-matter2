"""DM-2: 1-term vs 2-term fits on real ladders (analysis A, N-only per curve) and joint (N, D) fits (analysis B).

    python data.py                     # builds results/curves.csv, results/joint_points.csv, results/seed_sigma.csv
    python dm2.py --workers 2          # fits, bootstraps (1,000 refits each), writes results/table*.csv
    python verify.py                   # PASS/FAIL against PREREG.md
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import fitlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_REF, D_REF = 1e6, 1e9
NBOOT = 1000
COUNTS = {"porian": ["N_total", "N_nonemb"], "chinchilla": ["N_total", "N_nonemb", "N_reported"],
          "llama": ["N_total"]}


def fit_pair(n, y):
    f1 = F.fit1(n, y)
    f2 = F.fit2(n, y, f1) if F.nparams_ok(len(n)) else None
    return f1, f2


def holdout(n, y):
    o = np.argsort(n)
    n, y = n[o], y[o]
    k = math.ceil(2 * len(n) / 3)
    f1, f2 = fit_pair(n[:k], y[:k])
    e1 = F.predN(f1, n[k:]) - y[k:]
    out = dict(ho_fit=k, ho_pred=len(n) - k, ho_rmse1=float(np.sqrt(np.mean(e1 ** 2))))
    if f2 is not None:
        e2 = F.predN(f2, n[k:]) - y[k:]
        out["ho_rmse2"] = float(np.sqrt(np.mean(e2 ** 2)))
    else:
        out["ho_rmse2"] = np.nan
    return out


def reverse(n, y, cut=1e8 / N_REF):
    big = n >= cut
    if big.sum() < 4 or (~big).sum() == 0:
        return dict(rev_fit=int(big.sum()), rev_mean=np.nan, rev_pos=np.nan, rev_n=int((~big).sum())), []
    f1 = F.fit1(n[big], y[big])
    r = y[~big] - F.predN(f1, n[~big])
    rows = [dict(N=float(a * N_REF), resid=float(b)) for a, b in zip(n[~big], r)]
    return dict(rev_fit=int(big.sum()), rev_mean=float(r.mean()), rev_pos=int((r > 0).sum()), rev_n=int(len(r)),
                rev_smallest=float(r[np.argmin(n[~big])])), rows


def boot_N(n, y, f2, seed):
    rng = np.random.default_rng(seed)
    yhat = F.predN(f2, n)
    res = y - yhat
    res = res - res.mean()
    bs = np.empty(NBOOT)
    for i in range(NBOOT):
        ys = yhat + rng.choice(res, len(res), replace=True)
        f1s = F.fit1(n, ys)
        bs[i] = F.fit2(n, ys, f1s)["b"]
    return bs


def dm1_match(cells, nmin, nmax, npts, sigma):
    span = np.log10(nmax / nmin)
    ppd = (npts - 1) / span
    wins = cells.groupby(["nmin", "nmax"]).size().index
    full = [w for w in wins if (cells[(cells.nmin == w[0]) & (cells.nmax == w[1])].power_primary.notna().sum() > 0)]
    w = min(full, key=lambda q: (abs(np.log10(q[1] / q[0]) - span), abs(np.log10(q[0]) - np.log10(nmin))))
    sub = cells[(cells.nmin == w[0]) & (cells.nmax == w[1])]
    p = min(sub.ppd.unique(), key=lambda q: abs(np.log(q / ppd)))
    sg = min(sub.sigma.unique(), key=lambda q: abs(np.log(q / max(sigma, 1e-6))))
    sub = sub[(sub.ppd == p) & (sub.sigma == sg)]
    out = dict(dm1_window=f"[{w[0]:.0e},{w[1]:.0e}]", dm1_ppd=p, dm1_sigma=sg, ppd=ppd, span_dec=span)
    for _, r in sub.iterrows():
        key = "dm1_fpr" if r.s == 0 else f"dm1_pow_s{r.s:g}_b{r.beta:g}"
        out[key] = r.power_primary
    return out


def job_A(args):
    curve, slice_, count, df, cells, seed = args
    d = df.dropna(subset=["y", count]).sort_values(count)
    n, y = d[count].values / N_REF, d["y"].values
    row = dict(curve=curve, slice=slice_, count=count, how=d["how"].iloc[0], npts=len(n),
               Nmin=float(n.min() * N_REF), Nmax=float(n.max() * N_REF), tpp_min=float(d.tpp.min()),
               tpp_max=float(d.tpp.max()))
    f1, f2 = fit_pair(n, y)
    row.update(E1=f1["E"], A1=f1["a"], alpha1=f1["alpha"], rss1=f1["rss"])
    rev_rows = []
    if f2 is None:
        row.update(testable=0)
        return row, rev_rows
    z_dm1 = f2["z"]  # fit2's own z (unscaled pinv), kept for comparison
    f2["z"], se_w, _ = F.waldN(n, f2)
    dbic = F.bic(f1["rss"], len(n), 3) - F.bic(f2["rss"], len(n), 5)
    sig = math.sqrt(f2["rss"] / (len(n) - 5))
    bs = boot_N(n, y, f2, seed)
    se_boot = float(np.std(bs, ddof=1))
    row.update(testable=1, E2=f2["E"], A2=f2["a"], alpha2=f2["alpha"], B2=f2["b"], beta2=f2["beta"], rss2=f2["rss"],
               z_wald=f2["z"], z_wald_dm1code=z_dm1, se_wald=se_w, dbic=dbic, cond=f2["cond"], sigma_resid=sig,
               se_boot=se_boot, z_boot=(f2["b"] / se_boot if se_boot > 0 else 0.0),
               boot_frac_pos=float((bs > 1e-12).mean()), nboot=len(bs),
               detect_primary=int(f2["z"] > 2 and dbic > 2),
               shows_term=int(f2["z"] > 2 and dbic > 2 and se_boot > 0 and f2["b"] / se_boot > 2))
    # B in natural units at 1e6 params is B2 (n = 1); share of the fast term at the smallest size of the ladder
    nm = n.min()
    row["fast_share_at_Nmin"] = float(f2["b"] * nm ** -f2["beta"] / max(f2["a"] * nm ** -f2["alpha"], 1e-12))
    row.update(holdout(n, y))
    rv, rev_rows = reverse(n, y)
    row.update(rv)
    rev_rows = [dict(curve=curve, slice=slice_, count=count, **r) for r in rev_rows]
    row.update(dm1_match(cells, row["Nmin"], row["Nmax"], len(n), sig))
    return row, rev_rows


def boot_J(n, d, y, f2, seed):
    rng = np.random.default_rng(seed)
    yhat = F.predJ(f2, n, d)
    res = y - yhat
    res = res - res.mean()
    bs = np.empty(NBOOT)
    x1 = [f2["alpha"], f2["gamma"]]
    x2 = [f2["alpha"], max(f2["beta"] - f2["alpha"], 0.03), f2["gamma"]]
    for i in range(NBOOT):
        ys = yhat + rng.choice(res, len(res), replace=True)
        f1s = F.fitJ1(n, d, ys, k=25, x0=x1)
        bs[i] = F.fitJ2(n, d, ys, f1s, k=(10, 10, 10), x0=x2)["b"]
    return bs


def job_B(args):
    ladder, count, df, seed = args
    p = df.dropna(subset=[count, "L", "D"])
    n, d, y = p[count].values / N_REF, p["D"].values / D_REF, p["L"].values
    row = dict(ladder=ladder, count=count, npts=len(n), sizes=int(p[count].nunique()), Nmin=float(n.min() * N_REF),
               Nmax=float(n.max() * N_REF), Dmin=float(d.min() * D_REF), Dmax=float(d.max() * D_REF))
    f1 = F.fitJ1(n, d, y)
    f2 = F.fitJ2(n, d, y, f1, x0=[f1["alpha"], 0.5, f1["gamma"]])
    dbic = F.bic(f1["rss"], len(n), 5) - F.bic(f2["rss"], len(n), 7)
    bs = boot_J(n, d, y, f2, seed)
    se_boot = float(np.std(bs, ddof=1))
    row.update(E1=f1["E"], A1=f1["a"], alpha1=f1["alpha"], G1=f1["g"], gamma1=f1["gamma"], rss1=f1["rss"],
               E2=f2["E"], A2=f2["a"], alpha2=f2["alpha"], B2=f2["b"], beta2=f2["beta"], G2=f2["g"],
               gamma2=f2["gamma"], rss2=f2["rss"], z_wald=f2["z"], se_wald=f2["se_b"], dbic=dbic, cond=f2["cond"],
               sigma_resid=math.sqrt(f2["rss"] / (len(n) - 7)), se_boot=se_boot,
               z_boot=(f2["b"] / se_boot if se_boot > 0 else 0.0), boot_frac_pos=float((bs > 1e-12).mean()),
               nboot=len(bs), detect_primary=int(f2["z"] > 2 and dbic > 2),
               shows_term=int(f2["z"] > 2 and dbic > 2 and se_boot > 0 and f2["b"] / se_boot > 2))
    # hold-out by size: fit the smaller 2/3 of sizes (all their D), predict the larger third
    sizes = np.sort(np.unique(n))
    cut = sizes[math.ceil(2 * len(sizes) / 3) - 1]
    tr = n <= cut
    g1 = F.fitJ1(n[tr], d[tr], y[tr])
    g2 = F.fitJ2(n[tr], d[tr], y[tr], g1, x0=[g1["alpha"], 0.5, g1["gamma"]])
    for k, g in (("ho_rmse1", g1), ("ho_rmse2", g2)):
        row[k] = float(np.sqrt(np.mean((F.predJ(g, n[~tr], d[~tr]) - y[~tr]) ** 2)))
    row.update(ho_fit_sizes=int(len(np.unique(n[tr]))), ho_pred_sizes=int(len(np.unique(n[~tr]))))
    # reverse ladder: fit the no-B joint law on N >= 100M, residuals of the smaller sizes (mean per size)
    big = n >= 1e8 / N_REF
    rev_rows = []
    if len(np.unique(n[big])) >= 4 and (~big).any():
        h = F.fitJ1(n[big], d[big], y[big])
        r = y[~big] - F.predJ(h, n[~big], d[~big])
        per = pd.DataFrame(dict(N=n[~big] * N_REF, r=r)).groupby("N").r.mean()
        row.update(rev_fit_sizes=int(len(np.unique(n[big]))), rev_mean=float(per.mean()), rev_pos=int((per > 0).sum()),
                   rev_n=int(len(per)), rev_smallest=float(per.iloc[0]))
        rev_rows = [dict(curve=ladder, slice="joint", count=count, N=float(a), resid=float(b)) for a, b in per.items()]
    else:
        row.update(rev_fit_sizes=int(len(np.unique(n[big]))), rev_mean=np.nan, rev_pos=np.nan, rev_n=0)
    return row, rev_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", choices=["A", "B", "AB", "explore"], default="AB")
    args = ap.parse_args()
    cur = pd.read_csv(os.path.join(RES, "curves.csv"))
    joint = pd.read_csv(os.path.join(RES, "joint_points.csv"))
    cells = pd.read_csv(os.path.join(HERE, "data", "dm1_cells_partial.csv"))
    jobsA, jobsB = [], []
    for k, ((curve, slice_), g) in enumerate(cur.groupby(["curve", "slice"])):
        for j, count in enumerate(COUNTS[g.family.iloc[0]]):
            jobsA.append((curve, slice_, count, g, cells, [20261019, k, j]))
    for k, (ladder, g) in enumerate(joint.groupby("ladder")):
        fam = ladder.split("/")[0]
        for j, count in enumerate(COUNTS[fam]):
            jobsB.append((ladder, count, g, [20261020, k, j]))
    t0 = time.time()
    if args.only == "explore":  # post hoc joint fits on the tuned constant-LR curves (PREREG changes)
        ex = pd.read_csv(os.path.join(RES, "joint_points_explore.csv"))
        jobs = [(lad, count, g, [20261021, k, j]) for k, (lad, g) in enumerate(ex.groupby("ladder"))
                for j, count in enumerate(COUNTS["porian"])]
        with Pool(args.workers) as pool:
            out = pool.map(job_B, jobs, chunksize=1)
        pd.DataFrame([o[0] for o in out]).to_csv(os.path.join(RES, "tableB_explore.csv"), index=False)
        pd.DataFrame([r for o in out for r in o[1]]).to_csv(os.path.join(RES, "reverse_residuals_explore.csv"),
                                                            index=False)
        print(f"explore done: {len(out)} fits, {time.time() - t0:.0f}s")
        return
    with Pool(args.workers) as pool:
        if "A" in args.only:
            out = pool.map(job_A, jobsA, chunksize=1)
            pd.DataFrame([o[0] for o in out]).to_csv(os.path.join(RES, "tableA.csv"), index=False)
            revA = [r for o in out for r in o[1]]
            print(f"A done: {len(out)} fits, {time.time() - t0:.0f}s", flush=True)
        else:
            revA = []
        if "B" in args.only:
            out = pool.map(job_B, jobsB, chunksize=1)
            pd.DataFrame([o[0] for o in out]).to_csv(os.path.join(RES, "tableB.csv"), index=False)
            revB = [r for o in out for r in o[1]]
            print(f"B done: {len(out)} fits, {time.time() - t0:.0f}s", flush=True)
        else:
            revB = []
    if revA or revB:
        pd.DataFrame(revA + revB).to_csv(os.path.join(RES, f"reverse_residuals_{args.only}.csv"), index=False)
    json.dump(dict(nboot=NBOOT, seconds=time.time() - t0, jobsA=len(jobsA), jobsB=len(jobsB)),
              open(os.path.join(RES, f"run_{args.only}.json"), "w"))


if __name__ == "__main__":
    main()
