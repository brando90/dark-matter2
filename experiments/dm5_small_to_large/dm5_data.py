"""DM-5 ladders (PREREG_dm5.md): one (N, y) point per size for every qualifying ladder; writes results/ladders.csv.

    python dm5_data.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DM2 = os.path.join(ROOT, "experiments", "dm2_real_ladders")
sys.path.insert(0, DM2)
import data as D2  # noqa: E402  DM-2's loaders (llama_curve_value etc.)

HG = os.path.join(ROOT, "data", "data_hitchhikers_guide")
PORIAN_V = 50432
OUT = os.path.join(HERE, "results", "ladders.csv")


def from_curves():
    cur = pd.read_csv(os.path.join(DM2, "results", "curves.csv"))
    cur = cur[cur["slice"] == "D=20N"]
    rows = []

    def add(ladder, conv, primary, g, N, how):
        for i, (n, y, nm) in enumerate(sorted(zip(N, g.y, g.get("N_name", [np.nan] * len(g))))):
            rows.append(dict(ladder=ladder, convention=conv, primary=int(primary), N=float(n), y=float(y), how=how))

    g = cur[cur.curve == "chinchilla/hoffmann-epoch"]
    add("chinchilla", "N_total", True, g, g.N_total.values, "Hoffmann loss at D=20N")
    add("chinchilla", "N_nonemb", False, g, g.N_nonemb.values, "Hoffmann loss at D=20N")
    add("chinchilla", "N_reported", False, g, g.N_reported.values, "Hoffmann loss at D=20N")
    g = cur[cur.curve == "llama/mad"]
    add("llama", "num_params", True, g, g.N_total.values, g.how.iloc[0])
    for ds in ("rw", "owt2"):
        g = cur[cur.curve == f"porian/{ds}/tuned-short-const"]
        w = g.width.values.astype(float)
        add(f"porian-{ds}", "untied", True, g, g.N_total.values + PORIAN_V * w, g.how.iloc[0])
        add(f"porian-{ds}", "N_nonemb", False, g, g.N_nonemb.values, g.how.iloc[0])
        add(f"porian-{ds}", "tied", False, g, g.N_total.values, g.how.iloc[0])
    return rows


def hyena():
    d = pd.read_csv(os.path.join(HG, "hitchhickers_guide_unified.csv.gz"),
                    usecols=["num_params", "tokens", "loss", "family", "model_name"])
    d = d[(d.family == "Striped Hyena 1-11") & (d.num_params > 0)].dropna(subset=["loss"])
    rows = []
    for name, g in d.groupby("model_name"):
        g = g.sort_values("tokens")
        run = dict(N_total=float(g.num_params.iloc[0]), tokens=g.tokens.values.astype(float),
                   losses=g.loss.values.astype(float))
        y = D2.llama_curve_value(run)
        if np.isfinite(y):
            rows.append(dict(ladder="hyena", convention="num_params", primary=1, N=run["N_total"], y=float(y),
                             how="median train loss within +-5% tokens of 20N", name=name))
    out = pd.DataFrame(rows)
    # models of one size (variants with the same num_params) -> mean y at that N
    out = out.groupby(["ladder", "convention", "primary", "N", "how"], as_index=False).agg(
        y=("y", "mean"), name=("name", lambda s: "+".join(sorted(s))))
    print("hyena models per size:", out.name.str.count(r"\+").add(1).value_counts().to_dict())
    return out.to_dict("records")


def gpt3():
    d = pd.read_csv(os.path.join(HG, "colpret_gpt3.csv.gz"), usecols=["model_name", "num_params", "tokens_seen", "val_loss"],
                    low_memory=False).dropna(subset=["val_loss"])
    rows = []
    for name, g in d.groupby("model_name"):
        g = g.sort_values("tokens_seen")
        rows.append(dict(ladder="gpt3", convention="num_params", primary=1, N=float(g.num_params.iloc[0]),
                         y=float(g.val_loss.iloc[-1]), how="last val_loss, fixed D ~300B tokens", name=name))
    return rows


def main():
    df = pd.DataFrame(from_curves() + hyena() + gpt3())
    print("dropped (no y at D=20N):", int(df.y.isna().sum()), "points")
    df = df.dropna(subset=["y"])
    df = df.sort_values(["ladder", "convention", "N"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.groupby(["ladder", "convention", "primary"]).agg(n=("N", "size"), Nmin=("N", "min"), Nmax=("N", "max"),
                                                               ymin=("y", "min"), ymax=("y", "max")).to_string())
    bad = df.groupby(["ladder", "convention"]).apply(lambda g: bool(g.N.duplicated().any()))
    print("dups/NaN:", bad[bad].to_dict())


if __name__ == "__main__":
    main()
