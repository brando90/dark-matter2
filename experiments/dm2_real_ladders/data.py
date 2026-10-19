"""Build DM-2's ladders from the data in this repo (see PREREG.md): per-size curves for analysis A and annealed
(N, D, L) points for analysis B.

    python data.py            # writes results/curves.csv and results/joint_points.csv, prints a summary
"""
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PORIAN = os.path.join(ROOT, "data", "data_resolving_discrepancies", "experiment_results.pickle.xz")
LLAMA = os.path.join(ROOT, "data", "data_hitchhikers_guide", "colpret_LLaMA.csv.gz")
CHIN_ARCH = os.path.join(HERE, "data", "chinchilla_model_parameters.csv")
CHIN_PTS = os.path.join(HERE, "data", "epoch_research_svg_extracted_data.csv")
RATIO = 20.0
PORIAN_VOCAB = 50432
PORIAN_CURVES = [("base", "short", "chinchilla"), ("base", "short", "kaplan"), ("base", "long", "kaplan"),
                 ("tuned", "short", "const"), ("tuned", "long", "const")]


def porian_counts(w, depth):
    """OpenLM SwiGLU: hidden = 256 * ceil(int(8w/3) / 256); tied embedding counted once."""
    h = 256 * ((int(2 * 4 * w / 3) + 255) // 256)
    nonemb = depth * (4 * w * w + 3 * w * h)
    return dict(N_total=nonemb + PORIAN_VOCAB * w, N_nonemb=nonemb)


def interp_log(x, xs, ys):
    """Linear interpolation of ys in log(xs) at x; nan when x is outside [min, max]."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if len(xs) == 0 or x < xs[0] or x > xs[-1]:
        return np.nan
    return float(np.interp(np.log(x), np.log(xs), ys))


def load_porian():
    df = pd.read_pickle(PORIAN)
    rows = []
    for i, r in df.iterrows():
        c = porian_counts(r.width, r.depth)
        vl = r["val/loss"].dropna()
        tok_per_step = r.bs * r.seq_len
        name = r.ckpt_path if isinstance(r.ckpt_path, str) else ""
        m = re.search(r"/params=(\d+)M", name)
        pname = float(m.group(1)) * 1e6 if m else np.nan
        rows.append(dict(run=i, dataset=r.dataset, hparams=r.hparams, warmup=r.warmup, decay=r.decay,
                         width=r.width, depth=r.depth, seed=r.seed, N_name=pname, **c,
                         steps=vl.index.values.astype(float), losses=vl.values.astype(float),
                         tok_per_step=tok_per_step, max_step=r.max_step,
                         final_loss=float(vl.values[-1]), final_tokens=float(vl.index.values[-1]) * tok_per_step))
    return pd.DataFrame(rows)


def porian_curve_value(run, ratio=RATIO):
    """Validation loss at tokens = ratio * N_total, interpolated in log step (nan if the run ends before)."""
    target_step = ratio * run.N_total / run.tok_per_step
    return interp_log(target_step, run.steps, run.losses)


def chinchilla_points():
    arch = pd.read_csv(CHIN_ARCH)
    per_layer = arch["KQV Params Per Layer"] + arch["Output Proj Params Per Layer"] + arch["FFN Params Per Layer"]
    arch["N_nonemb"] = arch["n_layers"] * per_layer
    arch["N_total"] = arch["N_nonemb"] + arch["Embedding Parameters"]
    arch["N_reported"] = arch["Reported Parameters (M)"] * 1e6
    pts = pd.read_csv(CHIN_PTS)
    # match Epoch's model sizes (Hoffmann's reported N, unrounded) to the table's reported N (rounded to 1M)
    rep = arch["N_reported"].values
    j = np.abs(pts["Model Size"].values[:, None] - rep[None, :]).argmin(1)
    rel = np.abs(pts["Model Size"].values - rep[j]) / rep[j]
    pts["arch_row"] = j
    pts["match_relerr"] = rel
    for k in ("N_total", "N_nonemb", "N_reported"):
        pts[k] = arch[k].values[j]
    pts["N_source"] = pts["Model Size"]
    pts["D"] = pts["Training FLOP"] / (6 * pts["Model Size"])  # Hoffmann's C = 6 N D with the reported N
    pts["L"] = pts["loss"]
    return pts, arch


def llama_runs():
    d = pd.read_csv(LLAMA, usecols=["model_name", "num_params", "tokens_seen", "loss"]).dropna()
    runs = []
    for name, g in d.groupby("model_name"):
        g = g.sort_values("tokens_seen")
        N = float(g.num_params.iloc[0])
        t, l = g.tokens_seen.values.astype(float), g.loss.values.astype(float)
        # final: median of the last 1% of logged points (training-loss noise), at the run's last token count
        k = max(1, len(l) // 100)
        runs.append(dict(name=name, N_total=N, tokens=t, losses=l, final_loss=float(np.median(l[-k:])),
                         final_tokens=float(t[-1])))
    return runs


def llama_curve_value(run, ratio=RATIO):
    """Median training loss within +-5% (in tokens) of tokens = ratio * N (nan if the run ends before)."""
    tgt = ratio * run["N_total"]
    if run["tokens"][-1] < tgt / 1.05:
        return np.nan
    m = (run["tokens"] >= tgt / 1.05) & (run["tokens"] <= tgt * 1.05)
    if m.sum() == 0:
        return interp_log(tgt, run["tokens"], run["losses"])
    return float(np.median(run["losses"][m]))


def build():
    """Returns (curves, joint): curves = one row per (curve, size) for analysis A; joint = annealed points for B."""
    cur, joint = [], []
    P = load_porian()
    for ds in ("rw", "owt2"):
        for hp, wu, dec in PORIAN_CURVES:
            sub = P[(P.dataset == ds) & (P.hparams == hp) & (P.warmup == wu) & (P.decay == dec)]
            cname = f"porian/{ds}/{hp}-{wu}-{dec}"
            for w, g in sub.groupby("width"):
                r0 = g.iloc[0]
                base = dict(family="porian", curve=cname, width=w, depth=r0.depth, N_total=r0.N_total,
                            N_nonemb=r0.N_nonemb, N_name=r0.N_name, n_runs=len(g), seeds=g.seed.nunique())
                tpp_final = g.final_tokens / g.N_total
                longest = g.loc[g.final_tokens.idxmax()]
                if dec == "chinchilla":  # several annealed runs: interpolate the finals at the ratio
                    y20 = interp_log(RATIO, tpp_final, g.final_loss)
                    how = "interp-finals"
                else:  # one run per size: read its validation curve at the ratio
                    y20 = porian_curve_value(longest)
                    how = "curve-const-lr" if dec == "const" else "curve-schedule-confounded"
                cur.append(dict(base, slice="D=20N", y=y20, how=how, tpp=RATIO))
                cur.append(dict(base, slice="final", y=longest.final_loss, how="last-checkpoint",
                                tpp=longest.final_tokens / longest.N_total))
                if dec == "chinchilla":
                    for _, r in g.iterrows():
                        joint.append(dict(ladder=f"porian/{ds}/base-short-chinchilla", N_total=r.N_total,
                                          N_nonemb=r.N_nonemb, N_reported=np.nan, D=r.final_tokens, L=r.final_loss))
    pts, _ = chinchilla_points()
    for row, g in pts.groupby("arch_row"):
        r0 = g.iloc[0]
        tpp = g.D / g.N_source
        base = dict(family="chinchilla", curve="chinchilla/hoffmann-epoch", width=np.nan, depth=np.nan,
                    N_total=r0.N_total, N_nonemb=r0.N_nonemb, N_reported=r0.N_reported, N_name=r0.N_source,
                    n_runs=len(g), seeds=1)
        cur.append(dict(base, slice="D=20N", y=interp_log(RATIO, tpp, g.L), how="interp-finals", tpp=RATIO))
        lg = g.loc[g.D.idxmax()]
        cur.append(dict(base, slice="final", y=lg.L, how="last-checkpoint", tpp=lg.D / lg.N_source))
    for _, r in pts.iterrows():
        joint.append(dict(ladder="chinchilla/hoffmann-epoch", N_total=r.N_total, N_nonemb=r.N_nonemb,
                          N_reported=r.N_reported, D=r.D, L=r.L))
    runs = llama_runs()
    by_n = {}
    for r in runs:
        by_n.setdefault(r["N_total"], []).append(r)
        joint.append(dict(ladder="llama/mad", N_total=r["N_total"], N_nonemb=np.nan, N_reported=np.nan,
                          D=r["final_tokens"], L=r["final_loss"]))
    for N, rs in sorted(by_n.items()):
        base = dict(family="llama", curve="llama/mad", width=np.nan, depth=np.nan, N_total=N, N_nonemb=np.nan,
                    N_name=N, n_runs=len(rs), seeds=1)
        ok = [r for r in rs if r["final_tokens"] >= RATIO * N / 1.05]
        if ok:  # the run that ends nearest after 20N is the least far from its annealed end
            r = min(ok, key=lambda q: q["final_tokens"])
            y20 = llama_curve_value(r)
        else:
            y20 = np.nan
        cur.append(dict(base, slice="D=20N", y=y20, how="curve-schedule-confounded", tpp=RATIO))
        lg = max(rs, key=lambda q: q["final_tokens"])
        cur.append(dict(base, slice="final", y=lg["final_loss"], how="last-checkpoint", tpp=lg["final_tokens"] / N))
    return pd.DataFrame(cur), pd.DataFrame(joint), P, pts


def seed_sigma(P):
    s = P[P.hparams == "seed"]
    out = s.groupby(["dataset", "width"]).agg(N_total=("N_total", "first"), seeds=("seed", "nunique"),
                                             sd=("final_loss", "std"), mean=("final_loss", "mean"),
                                             tpp=("final_tokens", "median"))
    out["tpp"] = out["tpp"] / out["N_total"]
    return out.reset_index()


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    cur, joint, P, pts = build()
    cur.to_csv(os.path.join(HERE, "results", "curves.csv"), index=False)
    joint.to_csv(os.path.join(HERE, "results", "joint_points.csv"), index=False)
    seed_sigma(P).to_csv(os.path.join(HERE, "results", "seed_sigma.csv"), index=False)
    print(cur.groupby(["curve", "slice"]).agg(sizes=("y", "count"), planned=("y", "size"),
                                              Nmin=("N_total", "min"), Nmax=("N_total", "max")).to_string())
    print(joint.groupby("ladder").agg(points=("L", "size"), sizes=("N_total", "nunique")).to_string())
    print("chinchilla match max rel err", pts.match_relerr.max())
