"""DM-5 summary: median |error| (nats) vs extrapolation ratio per rule and ladder, the paired gains at r >= 10 and the
H2 verdict by PREREG_dm5.md's fixed rule, and the plot.

    python summarize_dm5.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BINS = [(1, 3), (3, 10), (10, 30), (30, np.inf)]
BIN_NAMES = ["1-3x", "3-10x", "10-30x", ">=30x"]
RULES = ["one", "lrt", "drop1", "drop2"]
MIN_GAIN = 0.005
KEY = ["ladder", "convention", "k", "N_pred"]


def bin_of(r):
    for (lo, hi), nm in zip(BINS, BIN_NAMES):
        if lo <= r < hi:
            return nm
    return None


def tables(d):
    d = d[d.defined == 1].copy()
    d["bin"] = d.ratio.map(bin_of)
    s = d.groupby(["ladder", "convention", "primary", "rule", "bin"]).agg(
        n=("abs_err", "size"), median_abs_err=("abs_err", "median"), median_err=("err", "median")).reset_index()
    wide = d.pivot_table(index=KEY + ["primary", "ratio"], columns="rule", values="abs_err").reset_index()
    return d, s, wide


def gains(wide, rmin=10.0):
    rows = []
    hi = wide[wide.ratio >= rmin]
    for (lad, conv, prim), g in hi.groupby(["ladder", "convention", "primary"]):
        for rule in ("lrt", "drop1", "drop2"):
            gg = g.dropna(subset=["one", rule])
            if not len(gg):
                continue
            gain = gg["one"] - gg[rule]
            rows.append(dict(ladder=lad, convention=conv, primary=prim, rule=rule, n_pairs=len(gg),
                             n_folds=gg.k.nunique(), median_abs_err_one=gg["one"].median(),
                             median_abs_err_rule=gg[rule].median(), gain=gain.median(),
                             gain_pct=100 * gain.median() / gg["one"].median(), frac_rule_better=float((gain > 0).mean()), frac_rule_worse=float((gain < 0).mean())))
    return pd.DataFrame(rows)


def verdicts(g):
    out = []
    p = g[g.primary == 1]
    n_lad = p.ladder.nunique()
    for rule in ("lrt", "drop1", "drop2"):
        q = p[p.rule == rule]
        avail = len(q)
        thr = 5 if avail >= 6 else max(avail - 1, 1)
        pos, neg = int((q.gain > 0).sum()), int((q.gain < 0).sum())
        med = float(q.gain.median()) if avail else np.nan
        if pos >= thr and med >= MIN_GAIN:
            v = "SUPPORTED"
        elif neg >= thr:
            v = "WORSE"
        else:
            v = "NULL"
        out.append(dict(rule=rule, ladders_with_pairs=avail, ladders_better=pos, ladders_worse=neg, threshold=thr,
                        median_gain_nats=med, median_gain_pct=float(q.gain_pct.median()) if avail else np.nan,
                        verdict=v))
    return pd.DataFrame(out)


def plot(s, path):
    p = s[s.primary == 1]
    lads = sorted(p.ladder.unique())
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    col = dict(one="#444444", lrt="#d1495b", drop1="#2e86ab", drop2="#66a182")
    for ax, lad in zip(axs.ravel(), lads):
        q = p[p.ladder == lad]
        for rule in RULES:
            r = q[q.rule == rule].set_index("bin").reindex(BIN_NAMES)
            ax.plot(range(len(BIN_NAMES)), r.median_abs_err.values, "o-", color=col[rule], label=rule)
        ax.set_xticks(range(len(BIN_NAMES)))
        ax.set_xticklabels(BIN_NAMES)
        ax.set_title(lad)
        ax.set_ylabel("median |error| (nats)")
        ax.set_xlabel("extrapolation ratio N_pred / N_max_fit")
        ax.set_yscale("log")
    axs.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    d = pd.read_csv(os.path.join(RES, "folds.csv"))
    d, s, wide = tables(d)
    g = gains(wide)
    v = verdicts(g)
    g3 = gains(wide, 3.0)  # post hoc, labelled: ratio >= 3 so that llama and hyena (<10x ranges) enter
    v3 = verdicts(g3)
    g3.to_csv(os.path.join(RES, "posthoc_gains_at_3x.csv"), index=False)
    v3.to_csv(os.path.join(RES, "posthoc_verdict_at_3x.csv"), index=False)
    s.to_csv(os.path.join(RES, "summary_by_ratio.csv"), index=False)
    g.to_csv(os.path.join(RES, "gains_at_10x.csv"), index=False)
    v.to_csv(os.path.join(RES, "verdict_h2.csv"), index=False)
    plot(s, os.path.join(RES, "dm5_error_vs_ratio.png"))
    pd.set_option("display.width", 250)
    print(s[s.primary == 1].pivot_table(index=["ladder", "rule"], columns="bin", values="median_abs_err")[BIN_NAMES]
          .round(4).to_string())
    print(g[g.primary == 1].round(4).to_string())
    print(v.round(4).to_string())
    print('POST HOC r>=3'); print(g3[g3.primary == 1].round(4).to_string()); print(v3.round(4).to_string())


if __name__ == "__main__":
    main()
