"""Check DM-2 against the criteria in PREREG.md; prints PASS/FAIL per check and exits 1 on any FAIL.

    python verify.py            # after data.py, dm2.py and summarize.py
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

import dm2
import fitlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")

    cur = pd.read_csv(os.path.join(RES, "curves.csv"))
    A = pd.read_csv(os.path.join(RES, "tableA.csv"))
    B = pd.read_csv(os.path.join(RES, "tableB.csv"))

    # 1. positive control on the Porian sizes (50 synthetic replicates, sigma = 0.005)
    n = np.sort(cur[cur.family == "porian"].N_total.unique()) / dm2.N_REF
    a0 = 406.4 * 1e6 ** -0.34
    rng = np.random.default_rng(20261019)
    res = {}
    for s in (1.0, 0.0):
        det, betas = [], []
        for _ in range(50):
            y = F.law(n, 1.69, a0, 0.34, s * a0, 1.0) + rng.normal(0, 0.005, len(n))
            f1 = F.fit1(n, y)
            f2 = F.fit2(n, y, f1)
            z, _, _ = F.waldN(n, f2)
            det.append(z > 2 and F.bic(f1["rss"], len(n), 3) - F.bic(f2["rss"], len(n), 5) > 2)
            betas.append(f2["beta"])
        res[s] = (np.mean(det), np.median(betas))
    check("positive control (s=1, beta=1)", res[1.0][0] >= 0.9 and abs(res[1.0][1] - 1) <= 0.2,
          f"detected {res[1.0][0]:.2f} of 50, median beta {res[1.0][1]:.3f}")
    check("negative control (s=0)", res[0.0][0] <= 0.1, f"false detections {res[0.0][0]:.2f} of 50")

    # 2. Porian parameter counts vs checkpoint names
    p = cur[(cur.family == "porian") & cur.N_name.notna()].drop_duplicates("width")
    same = (np.floor(p.N_total / 1e6) * 1e6 == p.N_name)
    check("Porian N_total vs names", len(p) == 16 and same.all(),
          f"{int(same.sum())} of {len(p)} sizes: floor(N_total / 1M) equals the checkpoint name exactly "
          f"(the repo's width x depth is ~2e4 x too small)")

    # 3. completeness
    expect = sum(len(dm2.COUNTS[g.family.iloc[0]]) for _, g in cur.groupby(["curve", "slice"]))
    got_pts = all(r.npts == cur[(cur.curve == r.curve) & (cur.slice == r.slice)].dropna(subset=["y", r["count"]]).shape[0]
                  for _, r in A.iterrows())
    check("analysis A complete", len(A) == expect and got_pts,
          f"{len(A)} rows of {expect}; every row uses all sizes with a value (drops = no value at D=20N)")
    t = A[A.testable == 1]
    check("A bootstraps", (t.nboot == dm2.NBOOT).all(), f"{len(t)} testable rows, nboot {sorted(t.nboot.unique())}")
    check("analysis B complete", len(B) == 8 and (B.nboot == dm2.NBOOT).all(),
          f"{len(B)} joint fits, nboot {sorted(B.nboot.unique())}")

    ex = os.path.join(RES, "tableB_explore.csv")
    if os.path.exists(ex):
        X = pd.read_csv(ex)
        check("post hoc joint fits", len(X) == 4 and (X.nboot == dm2.NBOOT).all(), f"{len(X)} fits")
        B = pd.concat([B, X])

    # 4. nested fits
    bad = (t.rss2 > t.rss1 * (1 + 1e-9)).sum() + (B.rss2 > B.rss1 * (1 + 1e-9)).sum()
    check("RSS_2 <= RSS_1", bad == 0, f"{bad} violations")

    # 5. manifest
    man = os.path.join(RES, "manifest.txt")
    if os.path.exists(man):
        rows = [l.split() for l in open(man) if l.strip()]
        same = all(hashlib.sha256(open(os.path.join(RES, f), "rb").read()).hexdigest() == h for h, f in rows)
        check("manifest", same and len(rows) > 0, f"{len(rows)} files hashed")
    else:
        check("manifest", False, "results/manifest.txt missing (run summarize.py)")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
