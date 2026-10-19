"""Aggregate DM-1 replicates into one row per cell (results/cells.csv) and print the readout tables.

    python analyze.py --results results
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

KEY = ("nmin", "nmax", "ppd", "sigma", "s", "beta")


def load(results):
    rows = []
    for p in sorted(glob.glob(os.path.join(results, "raw_shard*.csv"))):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def f(x):
    return float(x) if x not in ("", None) else np.nan


def aggregate(rows):
    by = defaultdict(list)
    for r in rows:
        by[int(r["cell"])].append(r)
    out = []
    for cell, rs in sorted(by.items()):
        r0 = rs[0]
        d = {k: f(r0[k]) for k in KEY}
        d.update(cell=cell, reps=len(rs), npts=int(r0["npts"]), testable=int(r0["testable"]))
        for t in ("primary", "wald", "bic"):
            d[f"power_{t}"] = np.mean([int(r[f"detect_{t}"]) for r in rs])
        s, beta = d["s"], d["beta"]
        b2 = np.array([f(r["beta2"]) for r in rs])
        det = np.array([int(r["detect_primary"]) for r in rs]).astype(bool)
        d["beta_relerr_med_detected"] = (np.median(np.abs(b2[det] - beta) / beta)
                                         if s > 0 and det.any() else np.nan)
        d["cond_med"] = np.nanmedian([f(r["cond"]) for r in rs]) if d["testable"] else np.nan
        d["alpha1_med"] = np.median([f(r["alpha1"]) for r in rs])
        e1 = np.array([f(r["err1_pred"]) for r in rs])
        es = np.array([f(r["errsel_pred"]) for r in rs])
        d["abs_err1_pred_med"] = np.median(np.abs(e1))
        d["abs_errsel_pred_med"] = np.median(np.abs(es))
        d["err1_pred_med"] = np.median(e1)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    cells = aggregate(load(args.results))
    path = os.path.join(args.results, "cells.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    print(f"wrote {path}: {len(cells)} cells")

    def win(c):
        return f"[{c['nmin']:.0e},{c['nmax']:.0e}]"

    print("\nPower of the primary rule (z>2 and dBIC>2) by window, points/decade and sigma; columns s=0 (FPR), "
          "then s=0.03..1 for beta=0.68 | beta=1.0")
    for c0 in [c for c in cells if c["s"] == 0]:
        line = [f"{win(c0)} ppd={c0['ppd']:.0f} sd={c0['sigma']:.3f} n={c0['npts']:>2}  FPR={c0['power_primary']:.3f} |"]
        for beta in (0.68, 1.0):
            ps = [c["power_primary"] for c in cells if c["s"] > 0 and c["beta"] == beta and
                  all(c[k] == c0[k] for k in ("nmin", "nmax", "ppd", "sigma"))]
            line.append(" ".join(f"{p:.2f}" for p in ps) + " |")
        print(" ".join(line))

    print("\nSmallest s with power >= 0.8 (sigma=0.01, 4/decade); median |extrapolation error| at 100x N_max, "
          "1-term vs selected model, at that s=0.1")
    for nmin, nmax in sorted({(c["nmin"], c["nmax"]) for c in cells}):
        for beta in (0.68, 1.0):
            sel = [c for c in cells if c["nmin"] == nmin and c["nmax"] == nmax and c["ppd"] == 4 and
                   c["sigma"] == 0.01 and c["beta"] == beta and c["s"] > 0]
            ok = [c["s"] for c in sel if c["power_primary"] >= 0.8]
            c01s = [c for c in sel if c["s"] == 0.1]
            if not c01s:
                continue
            c01 = c01s[0]
            print(f"[{nmin:.0e},{nmax:.0e}] beta={beta}: min s = {min(ok) if ok else 'none <= 1.0'}; "
                  f"s=0.1 power {c01['power_primary']:.2f}, |err| 1-term {c01['abs_err1_pred_med']:.4f}, "
                  f"selected {c01['abs_errsel_pred_med']:.4f}, alpha1 {c01['alpha1_med']:.3f}")


if __name__ == "__main__":
    main()
