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

    if os.path.exists(os.path.join(RES, "dm3_testA.csv")):
        dm3_checks(check, A)

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


def dm3_checks(check, tableA):
    """DM-3 addendum checks (PREREG.md, "verify.py additions (DM-3)")."""
    import dm3
    A3, K3 = dm3.load()
    d = A3["rw"]
    nt, ne, y = d.N_total.values, d.N_nonemb.values, d.y.values
    # 1-2. c recovery and the null control with c free, on synthetic curves shaped like rw at c = 0.3
    f = F.fit1(dm3.n_of(0.3, nt, ne), y)
    truth = F.law(dm3.n_of(0.3, nt, ne), f["E"], f["a"], f["alpha"])
    rng = np.random.default_rng(20261019)
    grid1, grid2 = dm3.C_GRID[::4], dm3.C_GRID[::8]
    cs, cover, fire = [], [], []
    for _ in range(50):
        ys = truth + rng.normal(0, 0.005, len(y))
        r1 = np.array([F.fit1(dm3.n_of(c, nt, ne), ys)["rss"] for c in grid1])
        c1, rss1 = dm3.refine(lambda c: F.fit1(dm3.n_of(c, nt, ne), ys)["rss"], grid1, r1)
        lo, hi, iv, _ = dm3.profile_ci(grid1, r1, min(rss1, r1.min()), len(ys))
        cs.append(c1)
        cover.append(any(a <= 0.3 <= b for a, b in
                         [tuple(map(float, q.strip("[]+ ").split(","))) for q in iv.split(" U ")]))
        c2, rss2 = dm3.best_c2(nt, ne, ys, grid2)
        n2 = dm3.n_of(c2, nt, ne)
        f2 = F.fit2(n2, ys, F.fit1(n2, ys))
        z, _, _ = F.waldN(n2, f2)
        fire.append(z > 2 and F.bic(rss1, len(ys), 4) - F.bic(rss2, len(ys), 6) > 2)
    check("DM-3 c recovery (c = 0.3, sigma 0.005, 50 curves)",
          abs(np.median(cs) - 0.3) <= 0.1 and np.mean(cover) >= 0.8,
          f"median c1 {np.median(cs):.3f}, profile set covers 0.3 in {np.mean(cover):.2f}")
    check("DM-3 null control, c free", np.mean(fire) <= 0.1,
          f"c-free second-term test fires in {np.mean(fire):.2f} of 50 (c grid step 0.04 for the 2-term profile)")
    # 3. Kaplan fitter on a noiseless K1 surface (rw B3 design, c = 0.5)
    p = K3["rw"]
    n, dd = dm3.n_of(0.5, p.N_total.values, p.N_nonemb.values), p.D.values / dm3.D_REF
    th = np.array([1.8, 100.0, 0.9, 2.0, 0.28, 0.0, 1.0])
    yk = dm3.kap(th, n, dd, "K1")
    k = dm3.fitK(n, dd, yk, "K1", 40, np.random.default_rng(1))
    check("DM-3 Kaplan fitter recovery", abs(k["p"] / 0.9 - 1) <= 0.05 and abs(k["q"] / 0.28 - 1) <= 0.05 and
          k["rss"] < 1e-8, f"p {k['p']:.4f} (0.9), q {k['q']:.4f} (0.28), RSS {k['rss']:.1e}")
    # 4. reproduction of DM-2 at c = 0 and c = 1
    T = pd.read_csv(os.path.join(RES, "dm3_tableA_at_c.csv"))
    ok, det = True, []
    for ds, curve in dm3.CURVES.items():
        for tag, count in (("c=0 (N_nonemb)", "N_nonemb"), ("c=1 (N_total)", "N_total")):
            a = T[(T.dataset == ds) & (T["at"] == tag)].iloc[0]
            b = tableA[(tableA.curve == curve) & (tableA.slice == "D=20N") & (tableA["count"] == count)].iloc[0]
            e = max(abs(a.alpha1 / b.alpha1 - 1), abs(a.rss1 / b.rss1 - 1))
            ok &= e <= 1e-6
            det.append(f"{ds} {count} {e:.1e}")
    check("DM-3 reproduces DM-2 at c = 0, 1", ok, "max rel. diff (alpha1, rss1): " + ", ".join(det))
    # 5. bootstraps and nesting
    tA, tK = pd.read_csv(os.path.join(RES, "dm3_testA.csv")), pd.read_csv(os.path.join(RES, "dm3_testK.csv"))
    bA, bK = pd.read_csv(os.path.join(RES, "dm3_bootA.csv")), pd.read_csv(os.path.join(RES, "dm3_bootK.csv"))
    check("DM-3 bootstraps", (tA.nboot == dm2.NBOOT).all() and (tK.nboot == dm2.NBOOT).all() and
          len(bA) == dm2.NBOOT and len(bK) == dm2.NBOOT and not bA.isna().any().any() and not bK.isna().any().any(),
          f"A3 {sorted(tA.nboot.unique())}, B3 {sorted(tK.nboot.unique())}, {len(bA)} and {len(bK)} rows")
    PA, PK = pd.read_csv(os.path.join(RES, "dm3_profileA.csv")), pd.read_csv(os.path.join(RES, "dm3_profileK.csv"))
    bad = int((PA.rss2 > PA.rss1 * (1 + 1e-9)).sum() + (PK.rss_K2 > PK.rss_K1 * (1 + 1e-9)).sum())
    check("DM-3 RSS_2 <= RSS_1", bad == 0, f"{bad} violations over {len(PA)} A3 and {len(PK)} B3 grid points")


if __name__ == "__main__":
    main()
