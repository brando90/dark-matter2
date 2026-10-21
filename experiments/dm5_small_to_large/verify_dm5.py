"""DM-5 checks (PREREG_dm5.md): fold counts, no NaNs, positive control (planted 2-term ladder), null control.

    python verify_dm5.py --workers 3 | tee results/verify_dm5.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dm5  # noqa: E402

RES = os.path.join(HERE, "results")
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def planted(second, rep, npts=14, sigma=0.005):
    N = np.logspace(5, 9, npts)
    n = N / 1e6
    y = 1.8 + 3 * n ** -0.3 + (12 * n ** -1.2 if second else 0) + np.random.default_rng([99, rep, int(second)]).normal(0, sigma, npts)
    return N, y


def control(second, reps, workers):
    jobs, j = [], 0
    for r in range(reps):
        N, y = planted(second, r)
        for k in range(dm5.MIN_FIT, len(N)):
            jobs.append((f"planted{int(second)}-{r}", "x", 1, N, y, k, 10_000 + j))
            j += 1
    return dm5.run(jobs, workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(RES, "folds.csv"))
    lad = pd.read_csv(os.path.join(RES, "ladders.csv"))

    # (a) fold counts: every (ladder, convention) has n-4 folds, sum_k (n-k) targets per rule
    ok, bad = True, []
    for (l, c), g in lad.groupby(["ladder", "convention"]):
        n = len(g)
        exp_folds = n - dm5.MIN_FIT
        exp_rows = sum(n - k for k in range(dm5.MIN_FIT, n))
        s = d[(d.ladder == l) & (d.convention == c)]
        if s.k.nunique() != exp_folds or any(len(s[s.rule == r]) != exp_rows for r in ("one", "lrt", "drop1", "drop2")):
            ok = False
            bad.append((l, c))
    check("fold counts (n-4 folds, sum_k (n-k) rows per rule, every ladder x convention)", ok, str(bad))
    check("all 6 primary ladders present", set(d[d.primary == 1].ladder) == {"chinchilla", "gpt3", "hyena", "llama", "porian-owt2", "porian-rw"})
    dd = d[d.defined == 1]
    check("no NaN/inf in any defined prediction or error", np.isfinite(dd[["pred", "err", "abs_err", "ratio"]].values).all(),
          f"{len(dd)} defined rows of {len(d)}")
    und = d[d.defined == 0]
    check("undefined rows are only drop rules with k - d < 4", ((und.rule.isin(["drop1", "drop2"])) &
          (und.k - und.rule.str[-1].astype(int) < dm5.MIN_FIT)).all(), f"{len(und)} rows")
    t = d[(d.rule == "lrt") & (d.k <= 6)]
    check("LRT rule equals 1-term for k <= 6 (untestable)", (t.lrt_tested == 0).all() and (t.used2 == 0).all())
    check("extrapolation ratio >= 1 for every row", (d.ratio > 1).all())

    # (c) positive control
    pc = control(True, a.reps, a.workers)
    L = pc[pc.rule == "lrt"]
    tested = L[L.lrt_tested == 1].drop_duplicates(["ladder", "k"])
    rate = tested.used2.mean()
    check("positive control: LRT detects the planted 2nd term in a majority of k > 6 folds", rate > 0.5, f"rate {rate:.2f} over {len(tested)} folds")
    hi = pc[pc.ratio >= 10].pivot_table(index=["ladder", "k", "N_pred"], columns="rule", values="abs_err").dropna(subset=["one", "lrt"])
    hi = hi[hi.index.get_level_values("k") > 6]
    check("positive control: LRT beats 1-term at r >= 10 (median |err|, k > 6)", hi.lrt.median() < hi.one.median(),
          f"lrt {hi.lrt.median():.4f} vs one {hi.one.median():.4f}, {len(hi)} pairs")
    # (d) null control
    nc = control(False, a.reps, a.workers)
    Ln = nc[nc.rule == "lrt"]
    tn = Ln[Ln.lrt_tested == 1].drop_duplicates(["ladder", "k"])
    fr = tn.used2.mean()
    check("null control: LRT false-detection rate <= 0.15 on a planted pure 1-term ladder", fr <= 0.15, f"rate {fr:.3f} over {len(tn)} folds")
    for df_, nm in ((pc, "positive"), (nc, "null")):
        x = df_[df_.defined == 1]
        check(f"{nm} control: no NaN in predictions", np.isfinite(x[["pred", "err"]].values).all())
    print("ALL PASS" if all(CHECKS) else "SOME FAILED")
    sys.exit(0 if all(CHECKS) else 1)


if __name__ == "__main__":
    main()
