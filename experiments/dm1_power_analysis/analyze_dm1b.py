"""Aggregate DM-1b replicates into one row per cell (results_dm1b/cells_dm1b.csv), next to DM-1's primary rule and BIC
alone on the same replicates (results/cells.csv), and print the readout tables.

    python analyze_dm1b.py --results results_dm1b --dm1-results results
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

KEY = ("nmin", "nmax", "ppd", "sigma", "s", "beta")


def load(results):
    """Rows of every raw_dm1b*.csv, one per (cell, rep), the first file's copy kept."""
    seen, rows = set(), []
    for p in sorted(glob.glob(os.path.join(results, "raw_dm1b*.csv"))):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                k = (r["cell"], r["rep"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
    return rows


def f(x):
    return float(x) if x not in ("", None) else np.nan


def aggregate(rows, dm1cells):
    by = defaultdict(list)
    for r in rows:
        by[int(r["cell"])].append(r)
    out = []
    for cell, rs in sorted(by.items()):
        r0 = rs[0]
        d = {k: f(r0[k]) for k in KEY}
        d.update(cell=cell, reps=len(rs), npts=int(r0["npts"]), testable=int(r0["testable"]))
        det = np.array([int(r["detect_lrt"]) for r in rs]).astype(bool)
        d["power_lrt"] = det.mean()
        b2 = np.array([f(r["beta2"]) for r in rs])
        d["beta_relerr_med_detected_lrt"] = (np.median(np.abs(b2[det] - d["beta"]) / d["beta"])
                                             if d["s"] > 0 and det.any() else np.nan)
        d["alpha2_med_detected_lrt"] = np.median([f(r["alpha2"]) for r in rs if r["detect_lrt"] == "1"]) \
            if det.any() else np.nan
        d["abs_errsel_lrt_pred_med"] = np.median(np.abs([f(r["errsel_lrt_pred"]) for r in rs]))
        d["abs_err1_pred_med"] = np.median(np.abs([f(r["err1_pred"]) for r in rs]))
        d["boot_samples_mean"] = np.mean([f(r["b_used"]) for r in rs]) if d["testable"] else 0.0
        c1 = dm1cells[cell]
        for t in ("primary", "bic", "wald"):
            d[f"power_{t}_dm1"] = float(c1[f"power_{t}"])
        d["abs_errsel_primary_pred_med_dm1"] = float(c1["abs_errsel_pred_med"])
        out.append(d)
    return out


def fitter_agreement(rows, dm1_results):
    """dm1b's vectorized fitter vs DM-1's stored fits on the same observed data (first DM-1 file's copy kept)."""
    stored = {}
    for p in sorted(glob.glob(os.path.join(dm1_results, "raw_shard*.csv"))):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                stored.setdefault((r["cell"], r["rep"]), (f(r["rss1"]), f(r["rss2"])))
    t = [r for r in rows if r["testable"] == "1" and (r["cell"], r["rep"]) in stored]
    r1 = np.array([f(r["rss1"]) / stored[(r["cell"], r["rep"])][0] - 1 for r in t])
    r2 = np.array([f(r["rss2"]) / stored[(r["cell"], r["rep"])][1] - 1 for r in t])
    return dict(n=len(t), rss1_within_1e6=np.mean(np.abs(r1) <= 1e-6), rss2_within_1e3=np.mean(np.abs(r2) <= 1e-3),
                rss2_worse_1pct=np.mean(r2 > 0.01), rss2_better_1e3=np.mean(r2 < -1e-3), rss2_rel_max=r2.max())


def win(c):
    return f"[{c['nmin']:.0e},{c['nmax']:.0e}]"


def min_s(cells, nmin, nmax, beta, col, ppd=4, sigma=0.01):
    sel = [c for c in cells if c["nmin"] == nmin and c["nmax"] == nmax and c["ppd"] == ppd and c["sigma"] == sigma
           and c["beta"] == beta and c["s"] > 0]
    ok = [c["s"] for c in sel if c[col] >= 0.8]
    return min(ok) if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results_dm1b")
    ap.add_argument("--dm1-results", default="results")
    args = ap.parse_args()
    rows = load(args.results)
    with open(os.path.join(args.dm1_results, "cells.csv")) as fh:
        dm1cells = {int(r["cell"]): r for r in csv.DictReader(fh)}
    cells = aggregate(rows, dm1cells)
    path = os.path.join(args.results, "cells_dm1b.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    print(f"{len(rows)} unique (cell, rep); wrote {path}: {len(cells)} cells")
    a = fitter_agreement(rows, args.dm1_results)
    print(f"Fitter vs DM-1's stored fits ({a['n']} testable replicates): RSS_1 within 1e-6 rel {a['rss1_within_1e6']:.4f}; "
          f"RSS_2 within 1e-3 rel {a['rss2_within_1e3']:.4f}, >1% worse {a['rss2_worse_1pct']:.4f}, "
          f">0.1% better {a['rss2_better_1e3']:.4f}, max rel worse {a['rss2_rel_max']:.3f}")
    boot = [c["boot_samples_mean"] * c["reps"] for c in cells if c["testable"]]
    print(f"Bootstrap refits actually run (exact shortcuts): {sum(boot):,.0f} of "
          f"{200 * sum(c['reps'] for c in cells if c['testable']):,}")

    nulls = [c for c in cells if c["s"] == 0 and c["testable"]]
    print("\nNull rejection rate per testable null cell: LRT | BIC alone | primary (DM-1)")
    for c in nulls:
        print(f"{win(c)} ppd={c['ppd']:.0f} sd={c['sigma']:.3f} n={c['npts']:>2}: {c['power_lrt']:.3f} | "
              f"{c['power_bic_dm1']:.3f} | {c['power_primary_dm1']:.3f}")
    for col, name in (("power_lrt", "LRT"), ("power_bic_dm1", "BIC"), ("power_primary_dm1", "primary")):
        wts = [c["reps"] for c in nulls]
        print(f"{name}: pooled {np.average([c[col] for c in nulls], weights=wts):.4f}, "
              f"worst cell {max(c[col] for c in nulls):.3f}")

    print("\nSmallest s with power >= 0.8 (sigma=0.01, 4/decade): LRT | primary (DM-1) | BIC alone (DM-1)")
    for nmin, nmax in [(1e5, 1e7), (1e7, 1e9), (1e6, 1e9), (1e5, 1e9)]:
        for beta in (0.68, 1.0):
            vals = [min_s(cells, nmin, nmax, beta, col) for col in ("power_lrt", "power_primary_dm1", "power_bic_dm1")]
            print(f"[{nmin:.0e},{nmax:.0e}] beta={beta}: " + " | ".join("none <= 1" if v is None else f"{v}" for v in vals))

    print("\nPower at sigma=0.01, 4/decade, s = 0.03 / 0.1 / 0.3 / 1: LRT || primary || BIC; "
          "median |100x extrapolation error| at s=0.1, LRT-selected vs primary-selected vs 1-term")
    for nmin, nmax in [(1e5, 1e7), (1e7, 1e9), (1e6, 1e9), (1e5, 1e9)]:
        for beta in (0.68, 1.0):
            sel = sorted([c for c in cells if c["nmin"] == nmin and c["nmax"] == nmax and c["ppd"] == 4 and
                          c["sigma"] == 0.01 and c["beta"] == beta and c["s"] > 0], key=lambda c: c["s"])
            g = lambda col: " / ".join(f"{c[col]:.2f}" for c in sel)
            c01 = [c for c in sel if c["s"] == 0.1][0]
            print(f"[{nmin:.0e},{nmax:.0e}] beta={beta}: {g('power_lrt')} || {g('power_primary_dm1')} || "
                  f"{g('power_bic_dm1')}; err {c01['abs_errsel_lrt_pred_med']:.4f} vs "
                  f"{c01['abs_errsel_primary_pred_med_dm1']:.4f} vs {c01['abs_err1_pred_med']:.4f}")

    print("\nLRT power, full grid: columns s=0 (null rate), then s=0.03..1 for beta=0.68 | beta=1.0")
    for c0 in [c for c in cells if c["s"] == 0]:
        line = [f"{win(c0)} ppd={c0['ppd']:.0f} sd={c0['sigma']:.3f} n={c0['npts']:>2}  null={c0['power_lrt']:.3f} |"]
        for beta in (0.68, 1.0):
            ps = [c["power_lrt"] for c in sorted(cells, key=lambda c: c["s"]) if c["s"] > 0 and c["beta"] == beta and
                  all(c[k] == c0[k] for k in ("nmin", "nmax", "ppd", "sigma"))]
            line.append(" ".join(f"{p:.2f}" for p in ps) + " |")
        print(" ".join(line))


if __name__ == "__main__":
    main()
