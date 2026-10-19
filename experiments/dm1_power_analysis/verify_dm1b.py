"""Check DM-1b results against the pre-registered criteria in PREREG.md (DM-1b addendum); prints PASS/FAIL per check.

    python verify_dm1b.py --results results_dm1b      # after analyze_dm1b.py has written cells_dm1b.csv
"""
import argparse
import csv
import glob
import hashlib
import os
import sys

import numpy as np

import dm1
import dm1b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results_dm1b")
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")

    raw = []
    for p in sorted(glob.glob(os.path.join(args.results, "raw_dm1b*.csv"))):
        with open(p) as fh:
            raw += list(csv.DictReader(fh))
    with open(os.path.join(args.results, "cells_dm1b.csv")) as fh:
        cells = [{k: float(v) if v not in ("", "nan") else np.nan for k, v in r.items()} for r in csv.DictReader(fh)]

    # 0. overlapping shards must agree, then keep the first copy
    first, dup, disagree = {}, 0, 0
    for r in raw:
        k = (int(r["cell"]), int(r["rep"]))
        if k in first:
            dup += 1
            disagree += first[k]["detect_lrt"] != r["detect_lrt"]
        else:
            first[k] = r
    check("shard overlap agrees", disagree == 0, f"{dup} overlapping replicates, {disagree} disagree")
    raw = list(first.values())

    # 1. completeness: every planned (cell, rep) of DM-1 exactly once
    planned = {(c["cell"], r) for c in dm1.cells() for r in range(c["reps"])}
    got = {(int(r["cell"]), int(r["rep"])) for r in raw}
    check("completeness", got == planned and len(raw) == len(planned),
          f"{len(raw)} rows, {len(planned)} planned, {len(got)} unique")

    # 2. key fields finite, p in (0, 1], decision consistent with k and B
    test = [r for r in raw if r["testable"] == "1"]
    bad = sum(1 for r in test if not all(np.isfinite(float(r[k])) for k in ("T", "p_lrt", "rss1", "rss2", "sigma_hat")))
    incons = sum(1 for r in test if int(r["detect_lrt"]) != int(int(r["b_used"]) == dm1b.B_BOOT and
                                                             int(r["k_exceed"]) <= dm1b.K_MAX))
    check("finite fields and consistent decisions", bad == 0 and incons == 0,
          f"{bad} testable rows with a non-finite field, {incons} decisions inconsistent with (k, B)")

    # 3. nesting
    worse = sum(1 for r in test if float(r["rss2"]) > float(r["rss1"]) * (1 + 1e-9) or float(r["T"]) < 0)
    check("nesting RSS2 <= RSS1, T >= 0", worse == 0, f"{worse} rows violate")

    # 4. determinism: recompute 5 stored replicates (observed fits and the full bootstrap decision)
    idx = np.random.default_rng(1).choice(len(test), 5, replace=False)
    cmap = {c["cell"]: c for c in dm1.cells()}
    diffs, flips = [], 0
    for i in idx:
        r = test[i]
        again = dm1b.run_cell((cmap[int(r["cell"])], [int(r["rep"])]))[0]
        diffs += [abs(float(again["T"]) - float(r["T"])), abs(float(again["rss1"]) - float(r["rss1"]))]
        flips += int(again["detect_lrt"]) != int(r["detect_lrt"]) or int(again["k_exceed"]) != int(r["k_exceed"])
    check("determinism", max(diffs) < 1e-6 and flips == 0,
          f"max abs diff {max(diffs):.2e}, {flips} decisions or exceedance counts differ over 5 recomputed replicates")

    # 5. pre-registered positive control
    pc = [c for c in cells if c["nmin"] == 1e5 and c["nmax"] == 1e9 and c["ppd"] == 8 and c["sigma"] == 0.005
          and c["s"] == 1.0 and c["beta"] == 1.0][0]
    check("positive control", pc["power_lrt"] >= 0.95 and pc["beta_relerr_med_detected_lrt"] <= 0.10,
          f"power {pc['power_lrt']:.3f} (>= 0.95), median |beta_hat-beta|/beta {pc['beta_relerr_med_detected_lrt']:.4f}"
          f" (<= 0.10)")

    # 6. pre-registered false-positive criteria, applied literally
    nulls = [c for c in cells if c["s"] == 0 and c["testable"] == 1]
    worst = max(nulls, key=lambda c: c["power_lrt"])
    pooled = np.average([c["power_lrt"] for c in nulls], weights=[c["reps"] for c in nulls])
    nnull = sum(c["reps"] for c in nulls)
    se = np.sqrt(0.05 * 0.95 / nnull)
    check("false positives", worst["power_lrt"] <= 0.10 and pooled <= 0.05,
          f"worst null cell {worst['power_lrt']:.3f} (<= 0.10) at [{worst['nmin']:.0e},{worst['nmax']:.0e}] "
          f"ppd={worst['ppd']:.0f} sd={worst['sigma']}; pooled {pooled:.4f} (<= 0.05; {nnull} null replicates, "
          f"Monte Carlo SE at 0.05 = {se:.4f})")
    if 0.05 < pooled <= 0.054:
        print(f"INFO  pooled null rate {pooled:.4f} is within Monte Carlo error of the nominal 0.05 (pre-registered note)")

    # pre-registered predictions (findings, reported either way)
    print(f"INFO  prediction 1: LRT null rate per testable cell ranges {min(c['power_lrt'] for c in nulls):.3f}-"
          f"{worst['power_lrt']:.3f} (BIC alone worst {max(c['power_bic_dm1'] for c in nulls):.3f})")
    p3 = [c for c in cells if c["nmin"] == 1e7 and c["nmax"] == 1e9 and c["ppd"] == 4 and c["sigma"] == 0.01
          and 0 < c["s"] <= 0.1]
    held = all(c["power_lrt"] < 0.5 for c in p3)
    print(f"INFO  prediction 3 ({'held' if held else 'refuted'}): [1e7,1e9] 4/decade sd=0.01, LRT power at s<=0.1 = "
          + ", ".join(f"s={c['s']} beta={c['beta']}: {c['power_lrt']:.2f}" for c in p3))

    with open(os.path.join(args.results, "SHA256SUMS"), "w") as fh:
        for p in sorted(glob.glob(os.path.join(args.results, "*.csv"))):
            fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.basename(p)}\n")
    print("wrote", os.path.join(args.results, "SHA256SUMS"))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
