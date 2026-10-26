"""DM-3: is the fast term on Porian's tuned constant-LR curves a real second power law, or embedding accounting?
Counts N_c = N_nonemb + c * N_emb with c profiled (A3, N-only at D = 20N) and a Kaplan-style joint law (B3).
See PREREG.md, "DM-3 addendum".

    python dm3.py --workers 2              # writes results/dm3_*.csv, results/dm3_summary.md, results/dm3_run.json
    python dm3.py --nboot 20 --out smoke   # quick run into results_smoke/
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

import fitlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
N_REF, D_REF = 1e6, 1e9
CURVES = {"rw": "porian/rw/tuned-short-const", "owt2": "porian/owt2/tuned-short-const"}
C_GRID = np.round(np.arange(0, 2.0001, 0.005), 3)
C_GRID_BOOT = np.linspace(0, 2, 21)
C_GRID_K = np.round(np.arange(0, 1.5001, 0.05), 3)
CHI2_95 = 3.841
NBOOT = 1000


def n_of(c, nt, ne):
    """N_c / 1e6 from N_total and N_nonemb (both in params)."""
    return (ne + c * (nt - ne)) / N_REF


# ---------------- A3: N-only, c profiled ----------------

def fitA(c, nt, ne, y, two=True):
    n = n_of(c, nt, ne)
    f1 = F.fit1(n, y)
    if not two:
        return f1, None
    f2 = F.fit2(n, y, f1)
    f2["z"], f2["se_b"], f2["cond"] = F.waldN(n, f2)
    return f1, f2


def refine(fun, grid, vals, lo=None, hi=None):
    """Bounded scalar refinement around the grid minimum of fun; returns (c, value)."""
    i = int(np.nanargmin(vals))
    a = grid[max(i - 1, 0)] if lo is None else lo
    b = grid[min(i + 1, len(grid) - 1)] if hi is None else hi
    if b <= a:
        return float(grid[i]), float(vals[i])
    r = minimize_scalar(fun, bounds=(a, b), method="bounded", options={"xatol": 1e-5})
    return (float(r.x), float(r.fun)) if r.fun <= vals[i] else (float(grid[i]), float(vals[i]))


def profile_ci(grid, rss, rss_min, npts):
    """95% profile-likelihood set {c : n ln(RSS(c)/RSS_min) <= 3.84} on the grid, edges linearly interpolated.
    Returns (outer lo, outer hi, intervals as text, 1 excluded?). An edge at the grid boundary stays the boundary
    value and is marked with ']' / '[' open in the text ("2.0+" means it runs into the grid's upper end)."""
    stat = npts * np.log(np.maximum(rss, 1e-300) / rss_min)
    ok = stat <= CHI2_95
    if not ok.any():
        return np.nan, np.nan, "", True

    def edge(i, j):
        s0, s1 = stat[i], stat[j]
        return grid[i] + (CHI2_95 - s0) * (grid[j] - grid[i]) / (s1 - s0) if np.isfinite(s1) and s1 != s0 else grid[i]
    ivs, i = [], 0
    while i < len(grid):
        if ok[i]:
            j = i
            while j + 1 < len(grid) and ok[j + 1]:
                j += 1
            lo = edge(i, i - 1) if i > 0 else grid[0]
            hi = edge(j, j + 1) if j < len(grid) - 1 else grid[-1]
            ivs.append((float(lo), float(hi), j == len(grid) - 1))
            i = j + 1
        else:
            i += 1
    txt = " U ".join(f"[{a:.3f}, {b:.3f}{'+' if open_ else ''}]" for a, b, open_ in ivs)
    excl1 = not any(a <= 1.0 <= b for a, b, _ in ivs)
    return ivs[0][0], ivs[-1][1], txt, excl1


C_POSTHOC = [3.0, 5.0, 10.0, 30.0, 100.0, 1e4]


def best_c1(nt, ne, y, grid):
    r = np.array([F.fit1(n_of(c, nt, ne), y)["rss"] for c in grid])
    c, v = refine(lambda c: F.fit1(n_of(c, nt, ne), y)["rss"], grid, r)
    return c, v


def best_c2(nt, ne, y, grid):
    def rss2(c):
        n = n_of(c, nt, ne)
        return F.fit2(n, y, F.fit1(n, y))["rss"]
    r = np.array([rss2(c) for c in grid])
    return refine(rss2, grid, r)


def row_at(tag, c, nt, ne, y):
    f1, f2 = fitA(c, nt, ne, y)
    npts = len(y)
    n = n_of(c, nt, ne)
    return dict(at=tag, c=c, npts=npts, E1=f1["E"], A1=f1["a"], alpha1=f1["alpha"], rss1=f1["rss"],
                sigma1=math.sqrt(f1["rss"] / (npts - 3)), E2=f2["E"], A2=f2["a"], alpha2=f2["alpha"], B2=f2["b"],
                beta2=f2["beta"], rss2=f2["rss"], sigma2=math.sqrt(f2["rss"] / (npts - 5)), z_wald=f2["z"],
                cond=f2["cond"], dbic=F.bic(f1["rss"], npts, 3) - F.bic(f2["rss"], npts, 5),
                fast_share_at_Nmin=float(f2["b"] * n.min() ** -f2["beta"] / max(f2["a"] * n.min() ** -f2["alpha"],
                                                                                    1e-12)))


def job_A(args):
    ds, d, nboot, seed = args
    nt, ne, y = d.N_total.values, d.N_nonemb.values, d.y.values
    npts = len(y)
    t0 = time.time()
    prof = []
    for c in C_GRID:
        f1, f2 = fitA(c, nt, ne, y)
        prof.append(dict(dataset=ds, c=c, rss1=f1["rss"], alpha1=f1["alpha"], E1=f1["E"], rss2=f2["rss"],
                         B2=f2["b"], beta2=f2["beta"], alpha2=f2["alpha"], z_wald=f2["z"]))
    P = pd.DataFrame(prof)
    c1, rss1 = refine(lambda c: F.fit1(n_of(c, nt, ne), y)["rss"], C_GRID, P.rss1.values)

    def rss2f(c):
        n = n_of(c, nt, ne)
        return F.fit2(n, y, F.fit1(n, y))["rss"]
    c2, rss2 = refine(rss2f, C_GRID, P.rss2.values)
    lo1, hi1, iv1, ex1 = profile_ci(C_GRID, P.rss1.values, rss1, npts)
    lo2, hi2, iv2, ex2 = profile_ci(C_GRID, P.rss2.values, rss2, npts)
    rows = [row_at("c=0 (N_nonemb)", 0.0, nt, ne, y), row_at("c=1 (N_total)", 1.0, nt, ne, y),
            row_at("c1 (1-term best)", c1, nt, ne, y), row_at("c2 (2-term best)", c2, nt, ne, y)]
    f1c1, _ = fitA(c1, nt, ne, y, two=False)
    _, f2c2 = fitA(c2, nt, ne, y)
    dbic_free = F.bic(rss1, npts, 4) - F.bic(rss2, npts, 6)

    # bootstrap 1: c1 from the 1-term-with-c model
    rng = np.random.default_rng(seed + [1])
    yhat1 = F.predN(f1c1, n_of(c1, nt, ne))
    res1 = y - yhat1
    res1 -= res1.mean()
    coarse = np.linspace(0, 2, 41)
    c1s = np.empty(nboot)
    for i in range(nboot):
        ys = yhat1 + rng.choice(res1, npts, replace=True)
        c1s[i] = best_c1(nt, ne, ys, coarse)[0]

    # bootstrap 2: B at the c-free 2-term fit, c re-profiled per refit
    rng = np.random.default_rng(seed + [2])
    n2 = n_of(c2, nt, ne)
    yhat2 = F.predN(f2c2, n2)
    res2 = y - yhat2
    res2 -= res2.mean()
    bs, c2s = np.empty(nboot), np.empty(nboot)
    for i in range(nboot):
        ys = yhat2 + rng.choice(res2, npts, replace=True)
        cb, _ = best_c2(nt, ne, ys, C_GRID_BOOT)
        nb = n_of(cb, nt, ne)
        bs[i] = F.fit2(nb, ys, F.fit1(nb, ys))["b"]
        c2s[i] = cb
    se_boot = float(np.std(bs, ddof=1))
    test = dict(dataset=ds, npts=npts, c1=c1, c1_ci_lo=lo1, c1_ci_hi=hi1, c1_ci_set=iv1, c1_ci_excludes_1=ex1,
                c1_boot_lo=float(np.percentile(c1s, 2.5)), c1_boot_hi=float(np.percentile(c1s, 97.5)),
                c1_boot_median=float(np.median(c1s)), c2=c2, c2_ci_lo=lo2, c2_ci_hi=hi2, c2_ci_set=iv2,
                c2_ci_excludes_1=ex2, rss1_free=rss1, rss2_free=rss2, sigma1_free=math.sqrt(rss1 / (npts - 4)),
                sigma2_free=math.sqrt(rss2 / (npts - 6)), dbic_free=dbic_free, B_c2=f2c2["b"],
                beta_c2=f2c2["beta"], alpha_c2=f2c2["alpha"], z_wald_c2=f2c2["z"], se_boot=se_boot,
                z_boot=(f2c2["b"] / se_boot if se_boot > 0 else 0.0), boot_frac_pos=float((bs > 1e-12).mean()),
                c2_boot_median=float(np.median(c2s)), nboot=nboot,
                rss1_at_c1_vs_c1=float(rss1 / P.loc[P.c == 1.0, "rss1"].iloc[0]))
    # post hoc (not in the addendum): does the 1-term RSS keep falling past the grid's c = 2?
    for c in C_POSTHOC:
        test[f"posthoc_rss1_c{c:g}"] = F.fit1(n_of(c, nt, ne), y)["rss"]
    test["posthoc_rss1_width_only"] = F.fit1((nt - ne) / N_REF, y)["rss"]
    test["detect_primary"] = int(test["z_wald_c2"] > 2 and dbic_free > 2)
    test["shows_term"] = int(test["detect_primary"] and test["z_boot"] > 2)

    # hold-out: smaller 10 of 14 by N_total, predict the largest 4
    o = np.argsort(nt)
    k = math.ceil(2 * npts / 3)
    tr, te = o[:k], o[k:]
    ho = dict(dataset=ds, ho_fit=k, ho_pred=npts - k)
    for mode in ("c1_all_data", "strict"):
        if mode == "c1_all_data":
            ca = cb = c1
        else:
            ca = best_c1(nt[tr], ne[tr], y[tr], C_GRID[::4])[0]
            cb = best_c2(nt[tr], ne[tr], y[tr], C_GRID[::8])[0]
        g1 = F.fit1(n_of(ca, nt[tr], ne[tr]), y[tr])
        nb = n_of(cb, nt[tr], ne[tr])
        g2 = F.fit2(nb, y[tr], F.fit1(nb, y[tr]))
        e1 = F.predN(g1, n_of(ca, nt[te], ne[te])) - y[te]
        e2 = F.predN(g2, n_of(cb, nt[te], ne[te])) - y[te]
        ho.update({f"{mode}_c_1term": ca, f"{mode}_c_2term": cb,
                   f"{mode}_rmse1": float(np.sqrt(np.mean(e1 ** 2))), f"{mode}_rmse2": float(np.sqrt(np.mean(e2 ** 2)))})
    # the DM-2 reference (c = 1) for the same split
    g1 = F.fit1(nt[tr] / N_REF, y[tr])
    g2 = F.fit2(nt[tr] / N_REF, y[tr], g1)
    ho["c=1_rmse1"] = float(np.sqrt(np.mean((F.predN(g1, nt[te] / N_REF) - y[te]) ** 2)))
    ho["c=1_rmse2"] = float(np.sqrt(np.mean((F.predN(g2, nt[te] / N_REF) - y[te]) ** 2)))

    # reverse ladder: 1-term on N_total >= 100M at c1 (and strict, and c = 1), per-size residuals of the rest;
    # the 2-term model's full-fit residuals at c2 next to them
    big = nt >= 1e8
    cs = best_c1(nt[big], ne[big], y[big], C_GRID[::4])[0]
    rev = []
    fits = {"1term_c1": (c1, F.fit1(n_of(c1, nt[big], ne[big]), y[big])),
            "1term_strict": (cs, F.fit1(n_of(cs, nt[big], ne[big]), y[big])),
            "1term_c=1": (1.0, F.fit1(nt[big] / N_REF, y[big]))}
    f2full = F.predN(f2c2, n2)
    f1full = F.predN(f1c1, n_of(c1, nt, ne))
    for j in np.argsort(nt):
        r = dict(dataset=ds, N_total=nt[j], N_nonemb=ne[j], emb_share=1 - ne[j] / nt[j], big=bool(big[j]), y=y[j])
        for key, (c, f) in fits.items():
            r[f"rev_{key}"] = float(y[j] - F.predN(f, n_of(c, nt[j], ne[j])))
            r[f"c_{key}"] = c
        r["full_2term_c2"] = float(y[j] - f2full[j])
        r["full_1term_c1"] = float(y[j] - f1full[j])
        rev.append(r)
    R = pd.DataFrame(rev)
    sm = R[~R.big]
    for key in fits:
        test[f"rev_mean_{key}"] = float(sm[f"rev_{key}"].mean())
        test[f"rev_pos_{key}"] = int((sm[f"rev_{key}"] > 0).sum())
        test[f"rev_smallest_{key}"] = float(sm[f"rev_{key}"].iloc[0])
    test["rev_c_strict"] = cs
    test["rev_n"] = int(len(sm))
    test["seconds"] = time.time() - t0
    rows = [dict(dataset=ds, **r) for r in rows]
    return dict(kind="A", ds=ds, prof=P, rows=rows, test=test, ho=ho, rev=R, c1s=c1s, bs=bs)


# ---------------- B3: Kaplan-style joint law ----------------

def kap(th, n, d, model):
    """th (natural): E, A, p, G, q, B, delta; beta = p q + delta."""
    E, A, p, G, q, B, dl = th
    out = (A * n ** -p + G / d) ** q
    if model != "K0":
        out = out + E
    if model == "K2":
        out = out + B * n ** -(p * q + dl)
    return out


NAMES = ["E", "A", "p", "G", "q", "B", "delta"]
ACTIVE = {"K0": [1, 2, 3, 4], "K1": [0, 1, 2, 3, 4], "K2": [0, 1, 2, 3, 4, 5, 6]}
LB = np.log([1e-6, 1e-6, 0.02, 1e-8, 0.01, 1e-10, 0.01])
UB = np.log([10.0, 1e6, 10.0, 1e6, 5.0, 1e4, 5.0])


def _full(x, model, fixed):
    th = np.array(fixed, float)
    th[ACTIVE[model]] = np.exp(x)
    return th


def fitK(n, d, y, model, nstart=40, rng=None, x0s=()):
    rng = rng if rng is not None else np.random.default_rng(0)
    act = ACTIVE[model]
    fixed = [0.0, 1, 1, 1, 1, 0.0, 1.0]
    lb, ub = LB[act], UB[act]

    def res(x):
        v = kap(_full(x, model, fixed), n, d, model) - y
        return np.where(np.isfinite(v), v, 1e3)
    starts = [np.clip(np.log(np.asarray(s, float)[act]), lb + 1e-9, ub - 1e-9) for s in x0s]
    for _ in range(nstart):
        s = np.array([rng.uniform(0.3, 2.5), 10 ** rng.uniform(-1, 3), rng.uniform(0.2, 3.0),
                      10 ** rng.uniform(-3, 1.5), rng.uniform(0.03, 1.0), 10 ** rng.uniform(-2, 1),
                      rng.uniform(0.1, 2.0)])
        starts.append(np.clip(np.log(s[act]), lb + 1e-9, ub - 1e-9))
    best = None
    for s in starts:
        try:
            r = least_squares(res, s, bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=600,
                              ftol=1e-12, xtol=1e-12, gtol=1e-12)
        except Exception:
            continue
        rss = float((r.fun ** 2).sum())
        if best is None or rss < best[1]:
            best = (r.x, rss)
    th = _full(best[0], model, fixed)
    if model != "K2":
        th[5] = 0.0
    out = dict(zip(NAMES, th))
    out.update(rss=best[1], model=model, beta=th[2] * th[4] + th[6])
    return out


def theta(f):
    return np.array([f[k] for k in NAMES], float)


def waldK(n, d, f):
    """Wald z of B in natural parameters (c held fixed), numerical Jacobian, stable inversion."""
    th = theta(f)
    act = ACTIVE["K2"]
    J = []
    for k in act:
        h = 1e-6 * max(abs(th[k]), 1e-8)
        tp, tm = th.copy(), th.copy()
        tp[k] += h
        tm[k] -= h
        J.append((kap(tp, n, d, "K2") - kap(tm, n, d, "K2")) / (2 * h))
    J = np.column_stack(J)
    se, cond = F._scaled_se(J, f["rss"], len(n) - len(act), 5)
    z = f["B"] / se if f["B"] > 0 and np.isfinite(se) and se > 0 else 0.0
    return z, se, cond


def fitK12(n, d, y, rng, nstart=40):
    k1 = fitK(n, d, y, "K1", nstart, rng)
    s = theta(k1)
    s[5], s[6] = 1e-6, 0.5
    k2 = fitK(n, d, y, "K2", nstart, rng, x0s=[s])
    if k2["rss"] > k1["rss"]:  # B -> 0 is K1; keep nesting exact
        k2 = dict(k1, model="K2", B=0.0, delta=0.5, beta=k1["p"] * k1["q"] + 0.5)
    return k1, k2


def job_K(args):
    ds, p, nboot, seed = args
    nt, ne, D, y = p.N_total.values, p.N_nonemb.values, p.D.values, p.L.values
    d = D / D_REF
    npts = len(y)
    t0 = time.time()
    rng = np.random.default_rng(seed + [3])
    prof = []
    for c in C_GRID_K:
        n = n_of(c, nt, ne)
        k0 = fitK(n, d, y, "K0", 20, rng)
        k1, k2 = fitK12(n, d, y, rng)
        cj = F.fitJ1(n, d, y)
        prof.append(dict(dataset=ds, c=c, rss_K0=k0["rss"], rss_K1=k1["rss"], rss_K2=k2["rss"], rss_chin=cj["rss"],
                         B_K2=k2["B"], p_K1=k1["p"], q_K1=k1["q"], E_K1=k1["E"]))
    P = pd.DataFrame(prof)

    def best(col, fun):
        i = int(np.argmin(P[col].values))
        lo, hi = C_GRID_K[max(i - 1, 0)], C_GRID_K[min(i + 1, len(C_GRID_K) - 1)]
        fine = np.linspace(lo, hi, 11)
        v = np.array([fun(c) for c in fine])
        j = int(np.argmin(v))
        return (float(fine[j]), float(v[j])) if v[j] <= P[col].values[i] else (float(C_GRID_K[i]),
                                                                                float(P[col].values[i]))
    cK1, rK1 = best("rss_K1", lambda c: fitK(n_of(c, nt, ne), d, y, "K1", 20, rng)["rss"])
    cK2, rK2 = best("rss_K2", lambda c: fitK12(n_of(c, nt, ne), d, y, rng, 20)[1]["rss"])
    cK0, rK0 = best("rss_K0", lambda c: fitK(n_of(c, nt, ne), d, y, "K0", 20, rng)["rss"])
    cCh, rCh = best("rss_chin", lambda c: F.fitJ1(n_of(c, nt, ne), d, y)["rss"])
    rK1, rK2 = min(rK1, P.rss_K1.min()), min(rK2, P.rss_K2.min())
    nK1, nK2 = n_of(cK1, nt, ne), n_of(cK2, nt, ne)
    k1 = fitK(nK1, d, y, "K1", 40, rng)
    _, k2 = fitK12(nK2, d, y, rng)
    z, se_w, cond = waldK(nK2, d, k2)
    k1_at_c2, _ = fitK12(nK2, d, y, rng)
    ci = {}
    for col, r in (("rss_K1", k1["rss"]), ("rss_K2", k2["rss"]), ("rss_chin", rCh)):
        ci[col] = profile_ci(C_GRID_K, P[col].values, min(r, P[col].min()), npts)
    # bootstrap B at the K2 best c (c fixed: conditional, labelled so)
    yhat = kap(theta(k2), nK2, d, "K2")
    res = y - yhat
    res -= res.mean()
    bs = np.empty(nboot)
    s0 = theta(k2)
    for i in range(nboot):
        ys = yhat + rng.choice(res, npts, replace=True)
        b1 = fitK(nK2, d, ys, "K1", 4, rng, x0s=[theta(k1_at_c2)])
        s = theta(b1)
        s[5], s[6] = 1e-6, 0.5
        b2 = fitK(nK2, d, ys, "K2", 4, rng, x0s=[s0, s])
        bs[i] = b2["B"] if b2["rss"] <= b1["rss"] else 0.0
    se_boot = float(np.std(bs, ddof=1))
    dbic = F.bic(k1["rss"], npts, 6) - F.bic(k2["rss"], npts, 8)
    t = dict(dataset=ds, npts=npts, sizes=int(len(np.unique(nt))), c_K0=cK0, sigma_K0=math.sqrt(rK0 / (npts - 5)),
             c_K1=cK1, c_K1_ci_lo=ci["rss_K1"][0], c_K1_ci_hi=ci["rss_K1"][1],
             c_K1_ci_set=ci["rss_K1"][2], c_K1_ci_excludes_1=ci["rss_K1"][3], sigma_K1=math.sqrt(k1["rss"] / (npts - 6)),
             E_K1=k1["E"], A_K1=k1["A"], p_K1=k1["p"], G_K1=k1["G"], q_K1=k1["q"], alphaN_K1=k1["p"] * k1["q"],
             c_K2=cK2, c_K2_ci_lo=ci["rss_K2"][0], c_K2_ci_hi=ci["rss_K2"][1],
             c_K2_ci_set=ci["rss_K2"][2], c_K2_ci_excludes_1=ci["rss_K2"][3], sigma_K2=math.sqrt(k2["rss"] / (npts - 8)),
             E_K2=k2["E"], p_K2=k2["p"], q_K2=k2["q"], B_K2=k2["B"], beta_K2=k2["beta"], z_wald_K2=z, se_wald_K2=se_w,
             cond_K2=cond, dbic_K1_K2=dbic, se_boot=se_boot, z_boot=(k2["B"] / se_boot if se_boot > 0 else 0.0),
             boot_frac_pos=float((bs > 1e-12).mean()), nboot=nboot,
             c_chin=cCh, c_chin_ci_lo=ci["rss_chin"][0], c_chin_ci_hi=ci["rss_chin"][1], c_chin_ci_set=ci["rss_chin"][2],
             sigma_chin=math.sqrt(rCh / (npts - 5)),
             sigma_chin_c1=math.sqrt(P.loc[P.c == 1.0, "rss_chin"].iloc[0] / (npts - 5)),
             sigma_K1_c1=math.sqrt(P.loc[P.c == 1.0, "rss_K1"].iloc[0] / (npts - 6)),
             sigma_K1_c0=math.sqrt(P.loc[P.c == 0.0, "rss_K1"].iloc[0] / (npts - 6)))
    t["detect_primary"] = int(z > 2 and dbic > 2)
    t["shows_term"] = int(t["detect_primary"] and t["z_boot"] > 2)
    # hold-out by size and reverse ladder, K1 at c_K1 and K2 at c_K2
    sizes = np.sort(np.unique(nt))
    cut = sizes[math.ceil(2 * len(sizes) / 3) - 1]
    tr = nt <= cut
    big = nt >= 1e8
    rev = []
    for key, c in (("K1", cK1), ("K2", cK2)):
        n = n_of(c, nt, ne)
        g = fitK12(n[tr], d[tr], y[tr], rng)[0 if key == "K1" else 1]
        t[f"ho_rmse_{key}"] = float(np.sqrt(np.mean((kap(theta(g), n[~tr], d[~tr], key) - y[~tr]) ** 2)))
        h = fitK12(n[big], d[big], y[big], rng)[0 if key == "K1" else 1]
        r = y[~big] - kap(theta(h), n[~big], d[~big], key)
        per = pd.DataFrame(dict(N_total=nt[~big], r=r)).groupby("N_total").r.mean()
        t[f"rev_mean_{key}"] = float(per.mean())
        t[f"rev_pos_{key}"] = int((per > 0).sum())
        t[f"rev_smallest_{key}"] = float(per.iloc[0])
        rev += [dict(dataset=ds, model=key, c=c, N_total=a, resid=b) for a, b in per.items()]
    t.update(ho_fit_sizes=int(len(np.unique(nt[tr]))), ho_pred_sizes=int(len(np.unique(nt[~tr]))),
             rev_fit_sizes=int(len(np.unique(nt[big]))), rev_n=int(len(np.unique(nt[~big]))),
             seconds=time.time() - t0)
    return dict(kind="K", ds=ds, prof=P, test=t, rev=pd.DataFrame(rev), bs=bs)


def load():
    cur = pd.read_csv(os.path.join(HERE, "results", "curves.csv"))
    ex = pd.read_csv(os.path.join(HERE, "results", "joint_points_explore.csv"))
    A = {ds: cur[(cur.curve == c) & (cur.slice == "D=20N")].dropna(subset=["y"]).sort_values("N_total")
         for ds, c in CURVES.items()}
    K = {ds: ex[ex.ladder == f"{c}-curve"].dropna(subset=["L"]) for ds, c in CURVES.items()}
    return A, K


def run(job):
    return job_A(job[1:]) if job[0] == "A" else job_K(job[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = os.path.join(HERE, args.out if args.out == "results" else f"results_{args.out}")
    os.makedirs(out, exist_ok=True)
    A, K = load()
    jobs = [("A", ds, A[ds], args.nboot, [20261019, 3, i]) for i, ds in enumerate(CURVES)] + \
           [("K", ds, K[ds], args.nboot, [20261019, 4, i]) for i, ds in enumerate(CURVES)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    a = [r for r in res if r["kind"] == "A"]
    k = [r for r in res if r["kind"] == "K"]
    pd.concat([r["prof"] for r in a]).to_csv(os.path.join(out, "dm3_profileA.csv"), index=False)
    pd.DataFrame([x for r in a for x in r["rows"]]).to_csv(os.path.join(out, "dm3_tableA_at_c.csv"), index=False)
    pd.DataFrame([r["test"] for r in a]).to_csv(os.path.join(out, "dm3_testA.csv"), index=False)
    pd.DataFrame([r["ho"] for r in a]).to_csv(os.path.join(out, "dm3_holdoutA.csv"), index=False)
    pd.concat([r["rev"] for r in a]).to_csv(os.path.join(out, "dm3_residualsA.csv"), index=False)
    pd.DataFrame({f"{r['ds']}_{q}": r[q] for r in a for q in ("c1s", "bs")}).to_csv(
        os.path.join(out, "dm3_bootA.csv"), index=False)
    pd.concat([r["prof"] for r in k]).to_csv(os.path.join(out, "dm3_profileK.csv"), index=False)
    pd.DataFrame([r["test"] for r in k]).to_csv(os.path.join(out, "dm3_testK.csv"), index=False)
    pd.concat([r["rev"] for r in k]).to_csv(os.path.join(out, "dm3_reverseK.csv"), index=False)
    pd.DataFrame({f"{r['ds']}_bs": r["bs"] for r in k}).to_csv(os.path.join(out, "dm3_bootK.csv"), index=False)
    json.dump(dict(nboot=args.nboot, seconds=time.time() - t0, c_grid=[float(C_GRID[0]), float(C_GRID[-1]), 0.005],
                   c_grid_K=[0.0, 1.5, 0.05]), open(os.path.join(out, "dm3_run.json"), "w"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
