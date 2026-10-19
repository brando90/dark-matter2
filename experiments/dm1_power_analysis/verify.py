"""Check DM-1 results against the pre-registered criteria in PREREG.md; prints PASS/FAIL per check.

    python verify.py --results results          # after analyze.py has written results/cells.csv
"""
import argparse
import csv
import glob
import hashlib
import os
import sys

import numpy as np

import dm1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")

    raw = []
    for p in sorted(glob.glob(os.path.join(args.results, "raw_shard*.csv"))):
        with open(p) as fh:
            raw += list(csv.DictReader(fh))
    with open(os.path.join(args.results, "cells.csv")) as fh:
        cells = [{k: float(v) if v not in ("", "nan") else np.nan for k, v in r.items()} for r in csv.DictReader(fh)]

    # 1. completeness: every planned (cell, rep) exactly once
    planned = {(c["cell"], r) for c in dm1.cells() for r in range(c["reps"])}
    got = [(int(r["cell"]), int(r["rep"])) for r in raw]
    check("completeness", set(got) == planned and len(got) == len(planned),
          f"{len(got)} rows, {len(planned)} planned, {len(set(got))} unique")

    # 2. key fields finite on testable rows
    bad = sum(1 for r in raw if r["testable"] == "1" and not all(np.isfinite(float(r[k]))
              for k in ("z_wald", "dbic", "rss1", "rss2", "err1_pred", "errsel_pred")))
    check("finite fields", bad == 0, f"{bad} testable rows with a non-finite field")

    # 3. nesting: the 2-term fit never fits worse than the 1-term fit
    worse = sum(1 for r in raw if r["testable"] == "1" and float(r["rss2"]) > float(r["rss1"]) * (1 + 1e-9))
    check("nesting RSS2 <= RSS1", worse == 0, f"{worse} rows violate")

    # 4. determinism: recompute 5 stored replicates
    idx = np.random.default_rng(0).choice(len(raw), 5, replace=False)
    cmap = {c["cell"]: c for c in dm1.cells()}
    diffs = []
    for i in idx:
        r = raw[i]
        again = dm1.run_one(cmap[int(r["cell"])], int(r["rep"]))
        diffs.append(abs(float(again["err1_pred"]) - float(r["err1_pred"])))
        if r["testable"] == "1":
            diffs.append(abs(float(again["dbic"]) - float(r["dbic"])))
    check("determinism", max(diffs) < 1e-6, f"max abs diff {max(diffs):.2e} over 5 recomputed replicates")

    # 5. pre-registered positive control
    pc = [c for c in cells if c["nmin"] == 1e5 and c["nmax"] == 1e9 and c["ppd"] == 8 and c["sigma"] == 0.005
          and c["s"] == 1.0 and c["beta"] == 1.0][0]
    check("positive control", pc["power_primary"] >= 0.95 and pc["beta_relerr_med_detected"] <= 0.10,
          f"power {pc['power_primary']:.3f} (>= 0.95), median |beta_hat-beta|/beta {pc['beta_relerr_med_detected']:.4f} (<= 0.10)")

    # 6. pre-registered false-positive criteria
    nulls = [c for c in cells if c["s"] == 0 and c["testable"] == 1]
    worst = max(nulls, key=lambda c: c["power_primary"])
    pooled = np.average([c["power_primary"] for c in nulls], weights=[c["reps"] for c in nulls])
    check("false positives", worst["power_primary"] <= 0.10 and pooled <= 0.05,
          f"worst null cell {worst['power_primary']:.3f} (<= 0.10) at [{worst['nmin']:.0e},{worst['nmax']:.0e}] "
          f"ppd={worst['ppd']:.0f} sd={worst['sigma']}; pooled {pooled:.4f} (<= 0.05)")

    # pre-registered prediction 4 (a finding, reported either way)
    p4 = [c for c in cells if c["nmin"] == 1e7 and c["nmax"] == 1e9 and c["ppd"] == 4 and c["sigma"] == 0.01
          and 0 < c["s"] <= 0.1]
    held = all(c["power_primary"] < 0.5 for c in p4)
    print(f"INFO  prediction 4 ({'held' if held else 'refuted'}): [1e7,1e9] 4/decade sd=0.01, power at s<=0.1 = "
          + ", ".join(f"s={c['s']} beta={c['beta']}: {c['power_primary']:.2f}" for c in p4))

    # checksums of the raw files
    with open(os.path.join(args.results, "SHA256SUMS"), "w") as fh:
        for p in sorted(glob.glob(os.path.join(args.results, "*.csv"))):
            fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.basename(p)}\n")
    print("wrote", os.path.join(args.results, "SHA256SUMS"))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
