"""DM-4 power check (CPU, no training): can the DM-4 design pin the embedding weight c, and does its second-term
test stay quiet under a 1-term law and fire under DM-3's 2-term law?

    python power.py [--sims 200]      # writes power.csv and power.md next to this file

Planted laws are DM-3's fits to Porian rw tuned-short-const at D = 20N (nats per NeoX token):
  acc-c0.1 / acc-c0.5 / acc-c1.0: 1-term law in N_c = N_nonemb + c N_emb, refitted at that c (fitlib.fit1);
  dm-c1.07: DM-3's 2-term best fit (c2 = 1.066, E 1.294, A 4.560, alpha 0.151, B 3.220, beta 1.462).
Each simulated dataset is the DM-4 design (runs.csv, constant-LR runs at D/N = 20, seed means) plus, for the full
design, Porian's own tied-50k runs at w384..w1120 (the sizes DM-3 used). Noise per point: Porian's seed sd at that
width (the larger of rw and owt2, log-interpolated) divided by sqrt(seeds). Tokenizer groups get their own
intercept E_g in the fit (the planted offsets are 0).
The fit is the one PREREG DM-4 fixes: variable projection over grids of c, alpha (and beta - alpha), linear
non-negative solve for (E_g, A[, B]); 95% profile set for c from n ln(RSS / RSS_min) <= 3.84.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIGMA = HERE.parent / "dm2_real_ladders" / "results" / "seed_sigma.csv"
PORIAN_LARGE = [(384, 10), (480, 12), (576, 14), (640, 15), (704, 18), (832, 21), (1024, 23), (1120, 26)]
LAWS = {
    "acc-c0.1": dict(c=0.1, E=0.4129, A=4.5759, alpha=0.0877, B=0.0, beta=1.0),
    "acc-c0.5": dict(c=0.5, E=2.0230, A=3.7090, alpha=0.2001, B=0.0, beta=1.0),
    "acc-c1.0": dict(c=1.0, E=2.2139, A=4.3154, alpha=0.2543, B=0.0, beta=1.0),
    "dm-c1.07": dict(c=1.066, E=1.2941, A=4.5598, alpha=0.1508, B=3.2203, beta=1.4622),
}
C_GRID = np.round(np.arange(0.0, 2.0001, 0.05), 3)
AL_GRID = np.linspace(0.03, 1.0, 98)
A2, D2 = np.linspace(0.03, 0.8, 40), np.linspace(0.1, 2.5, 40)


def hidden(w):
    return 256 * int(np.ceil(int(8 * w / 3) / 256))


def design(full):
    r = pd.read_csv(HERE / "runs.csv")
    r = r[(r.run == "const") & (r.phase <= (2 if full else 1))]
    r = r[~((r.tokenizer == "own-bpe") & (r.vocab == 50432))]          # the family control is not in the fit
    pts = r.groupby(["width", "vocab", "tie", "tokenizer"]).agg(N_nonemb=("N_nonemb", "first"),
                                                                 N_emb=("N_emb", "first"),
                                                                 seeds=("seed", "nunique")).reset_index()
    if full:
        extra = []
        for w, d in PORIAN_LARGE:
            if ((pts.width == w) & (pts.vocab == 50432) & (pts.tie == "tied")).any():
                continue
            extra.append(dict(width=w, vocab=50432, tie="tied", tokenizer="neox",
                              N_nonemb=d * (4 * w * w + 3 * w * hidden(w)), N_emb=50432 * w, seeds=1))
        pts = pd.concat([pts, pd.DataFrame(extra)], ignore_index=True)
    s = pd.read_csv(SIGMA).groupby("width").sd.max()
    sd = np.exp(np.interp(np.log(pts.width), np.log(s.index.values), np.log(s.values)))
    pts["sd"] = sd / np.sqrt(pts.seeds)
    pts["group"] = pts.vocab.map({8192: 0, 32768: 1, 50432: 2})
    return pts


def truth(p, law):
    n = (p.N_nonemb.values + law["c"] * p.N_emb.values) / 1e6
    return law["E"] + law["A"] * n ** -law["alpha"] + law["B"] * n ** -law["beta"]


def _solve(X, y):
    XtX = np.einsum("gni,gnj->gij", X, X)
    coef = np.linalg.solve(XtX + 1e-12 * np.eye(X.shape[2]), np.einsum("gni,n->gi", X, y)[..., None])[..., 0]
    rss = ((np.einsum("gnk,gk->gn", X, coef) - y) ** 2).sum(1)
    return coef, np.where((coef < 0).any(1), np.inf, rss)


def fit(p, y, two):
    G = np.eye(3)[p.group.values]                                    # tokenizer intercepts
    best = []
    for c in C_GRID:
        n = (p.N_nonemb.values + c * p.N_emb.values) / 1e6
        if two:
            aa, dd = [g.ravel() for g in np.meshgrid(A2, D2, indexing="ij")]
            cols = [np.repeat(G[None], len(aa), 0), (n[None, :] ** -aa[:, None])[..., None],
                    (n[None, :] ** -(aa + dd)[:, None])[..., None]]
        else:
            aa, dd = AL_GRID, None
            cols = [np.repeat(G[None], len(aa), 0), (n[None, :] ** -aa[:, None])[..., None]]
        X = np.concatenate(cols, axis=2)
        X = X[:, :, np.concatenate([np.unique(p.group.values), [3, 4][: 1 + two]])]
        coef, rss = _solve(X, y)
        i = int(np.argmin(rss))
        best.append((rss[i], c, aa[i], None if dd is None else dd[i], coef[i]))
    return best


def analyse(p, y):
    f1, f2 = fit(p, y, False), fit(p, y, True)
    # b = 0 is allowed (the 1-term fit at the same c), so RSS_2 <= RSS_1 at every c, as in DM-1..3
    f2 = [b2 if b2[0] <= b1[0] else (b1[0], b1[1], b1[2], 1.0, np.append(b1[4], 0.0)) for b1, b2 in zip(f1, f2)]
    r1, r2 = np.array([b[0] for b in f1]), np.array([b[0] for b in f2])
    npts, ng = len(y), len(np.unique(p.group.values))
    i1, i2 = int(np.argmin(r1)), int(np.argmin(r2))
    cset = C_GRID[npts * np.log(r1 / r1[i1]) <= 3.84]
    k1, k2 = ng + 3, ng + 5
    dbic = (npts * np.log(r1[i1] / npts) + k1 * np.log(npts)) - (npts * np.log(r2[i2] / npts) + k2 * np.log(npts))
    # Wald z of B at the 2-term optimum (c held), natural parameters
    rss, c, al, dd, coef = f2[i2]
    n = (p.N_nonemb.values + c * p.N_emb.values) / 1e6
    ln, be, A, B = np.log(n), al + dd, coef[-2], coef[-1]
    G = np.eye(3)[p.group.values][:, np.unique(p.group.values)]
    J = np.column_stack([G, n ** -al, -A * n ** -al * ln, n ** -be, -B * n ** -be * ln])
    cov = max(rss, 1e-300) / max(npts - J.shape[1], 1) * np.linalg.pinv(J.T @ J)
    z = B / np.sqrt(cov[-2, -2]) if B > 0 and cov[-2, -2] > 0 else 0.0
    c2set = C_GRID[npts * np.log(r2 / r2[i2]) <= 3.84]
    return dict(c1=C_GRID[i1], c1_lo=cset.min(), c1_hi=cset.max(), c1_split=bool(len(cset) and
                np.any(np.diff(np.searchsorted(C_GRID, cset)) > 1)), c2=C_GRID[i2], c2_lo=c2set.min(),
                c2_hi=c2set.max(), dbic=dbic, z=z, fires=bool(dbic > 2 and z > 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    a = ap.parse_args()
    rows = []
    for full in (False, True):
        p = design(full)
        for name, law in LAWS.items():
            mu = truth(p, law)
            rng = np.random.default_rng(20261019 + 7 * full + sum(map(ord, name)))
            for k in range(a.sims):
                y = mu + rng.normal(0, p.sd.values)
                rows.append(dict(design="phase1+2+Porian" if full else "phase1", law=name, c_true=law["c"],
                                 npts=len(p), sim=k, **analyse(p, y)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "power.csv", index=False)
    df["cover1"] = (df.c1_lo <= df.c_true + 1e-9) & (df.c_true <= df.c1_hi + 1e-9)
    df["cover2"] = (df.c2_lo <= df.c_true + 1e-9) & (df.c_true <= df.c2_hi + 1e-9)
    df["w1"], df["w2"] = df.c1_hi - df.c1_lo, df.c2_hi - df.c2_lo
    s = df.groupby(["design", "law"]).agg(npts=("npts", "first"), c1_median=("c1", "median"),
                                          c1_set_width=("w1", "median"), c1_cover=("cover1", "mean"),
                                          c1_split=("c1_split", "mean"), c2_median=("c2", "median"),
                                          c2_set_width=("w2", "median"), c2_cover=("cover2", "mean"),
                                          fires=("fires", "mean"), dbic_median=("dbic", "median"))
    txt = ("# DM-4 power check\n\n" + f"{a.sims} simulated datasets per cell; c grid step 0.05 over [0, 2].\n\n"
           + s.to_string(float_format=lambda v: f"{v:.3f}") + "\n")
    (HERE / "power.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
