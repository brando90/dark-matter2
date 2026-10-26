"""DM-4 design: the run manifest and a FLOP-based GPU-hour estimate for a 1M-100M body ladder that varies embedding
size at fixed body size (vocab 8,192 / 32,768 / 50,432, tied vs untied). Design only: nothing here trains.

    python design.py            # writes runs.csv, budget.csv, budget.md next to this file

Shapes and tuned hyperparameters are Porian et al.'s (tuned, short warmup, constant LR; read from
data/data_resolving_discrepancies). Assumptions that phase 0 replaces with measurements are marked ASSUME; when
phase 0's measurements exist (data_phase0/bpt.json: bytes per token; data_phase0/probe.csv: tokens/s and eval time
per body x vocab on one GPU), runs.csv uses the measured bytes per token and budget.md adds measured GPU-hours.

Amendment 1 (phase 0, before any DM-4 training run): Porian's checkpoints are UNTIED (see PREREG), so the anchor
arm is untied-50k (NeoX), phase 2 runs untied-50k (Porian's config; its w480 run is G2) and tied-8k, and the LR
check uses tied-8k and tied-50k. N_ref = N_nonemb + 50,432 w (Porian's N: body + head) sets every body's data.
"""
import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PORIAN = HERE.parents[1] / "data" / "data_resolving_discrepancies" / "experiment_results.pickle.xz"

WIDTHS = [128, 160, 224, 288, 320, 384, 480, 576, 640, 704]   # N_nonemb 1.05M .. 113.5M
SMALL = WIDTHS[:5]                                              # 3 seeds here
VOCABS = [8192, 32768, 50432]
TIES = ["tied", "untied"]
SEEDS_SMALL, SEEDS_LARGE = [0, 1, 2], [0]
REF_V, TPP = 50432, 20            # data per body: 20 x N_ref = 20 x (N_nonemb + 50,432 w) reference tokens
EVAL_TPP = [2.5, 5, 10, 20]       # read off the constant-LR run (no extra cost)
COOL_TPP = [5, 10, 20]            # WSD cooldown branches: from the const-LR checkpoint at 0.8 D, linear to 0 over 0.2 D
COOL_FRAC = 0.2
CTX = 2048
# ASSUME: UTF-8 bytes per token on RefinedWeb for byte-level BPE trained on the same slice (measured in phase 0)
BPT = {8192: 3.4, 32768: 4.15, 50432: 4.35}
P0 = HERE / "data_phase0"
if (P0 / "bpt.json").exists():                     # measured on the first training shard (as train.py)
    _m = json.loads((P0 / "bpt.json").read_text())
    BPT = {8192: _m["bpe8k"]["train_first"]["bytes_per_token"], 32768: _m["bpe32k"]["train_first"]["bytes_per_token"],
           50432: _m["neox"]["train_first"]["bytes_per_token"]}
# ASSUME: L40S dense bf16 peak and model-FLOP utilisation by width (pessimistic, central, optimistic)
PEAK = 362e12
def mfu(w):
    return (0.06, 0.10, 0.15) if w <= 224 else (0.12, 0.18, 0.25) if w <= 384 else (0.20, 0.28, 0.35)


def hidden(w):
    return 256 * math.ceil(int(8 * w / 3) / 256)        # OpenLM SwiGLU


def n_nonemb(w, depth):
    return depth * (4 * w * w + 3 * w * hidden(w))


def porian_hparams():
    df = pd.read_pickle(PORIAN)
    s = df[(df.hparams == "tuned") & (df.warmup == "short") & (df.decay == "const") & (df.dataset == "rw")]
    s = s.drop_duplicates(subset=["width"]).set_index("width")
    return {int(w): dict(depth=int(r.depth), lr=float(r.lr), bs=int(r.bs), beta2=float(r.beta2),
                         warmup_tokens=int(r.warmup_tokens)) for w, r in s.iterrows()}


def flops_per_token(w, depth, V):
    # 6 N_nonemb + unembedding matmul (same for tied and untied; the input lookup is ~free) + causal attention
    return 6 * n_nonemb(w, depth) + 6 * V * w + 6 * depth * CTX * w


def arms_for(w):
    """(vocab, tie, tokenizer, seeds, extra runs) per body. 50,432 is the GPT-NeoX tokenizer Porian used, so the
    untied-50k arm is Porian's own configuration (re-run at every body; Porian's runs above w320 add to it).
    8,192 and 32,768 are byte-level BPE trained on a RefinedWeb slice outside the training/validation shards."""
    cool_all = [f"cool@{k}" for k in COOL_TPP]
    if w in SMALL:
        a = [(V, t, "neox" if V == REF_V else "own-bpe", SEEDS_SMALL, cool_all) for V in VOCABS for t in TIES]
        if w in (128, 224):
            a.append((REF_V, "tied", "own-bpe", [0], []))                   # tokenizer-family control
        return a
    # untied-50k = Porian's config (its w480 run is the G2 pipeline check against Porian mid-ladder) and tied-8k
    return [(REF_V, "untied", "neox", SEEDS_LARGE, ["cool@20"]), (8192, "tied", "own-bpe", SEEDS_LARGE, ["cool@20"])]


LR_CHECK = {(128, 8192, "tied"), (128, REF_V, "tied"), (224, 8192, "tied"), (224, REF_V, "tied")}


def main():
    hp = porian_hparams()
    rows = []
    for w in WIDTHS:
        h = hp[w]
        nb = n_nonemb(w, h["depth"])
        d_ref = TPP * (nb + REF_V * w)                        # reference (NeoX) tokens at D/N = 20
        d_bytes = d_ref * BPT[REF_V]
        phase = 1 if w in SMALL else 2
        for V, t, tok, seeds, extra in arms_for(w):
            n_emb = V * w * (1 if t == "tied" else 2)
            d_tok = d_bytes / BPT[V]
            f = flops_per_token(w, h["depth"], V)
            for s in seeds:
                kinds = [("const", 1.0)]
                if s == 0:
                    kinds += [(k, COOL_FRAC * float(k.split("@")[1]) / TPP) for k in extra]
                    if (w, V, t) in LR_CHECK and not (V == REF_V and tok == "own-bpe"):
                        kinds += [("lr x0.5", 1.0), ("lr x2", 1.0)]
                for kind, frac in kinds:
                    tokens = d_tok * frac
                    C = f * tokens
                    r = dict(phase=phase, width=w, depth=h["depth"], vocab=V, tie=t, tokenizer=tok, seed=s,
                             run=kind, N_nonemb=nb, N_emb=n_emb, N_total=nb + n_emb, D_bytes=d_bytes * frac,
                             tokens=tokens, steps=math.ceil(tokens / (h["bs"] * CTX)),
                             lr=h["lr"] * (0.5 if kind == "lr x0.5" else 2.0 if kind == "lr x2" else 1.0),
                             bs=h["bs"], beta2=h["beta2"], warmup_tokens=h["warmup_tokens"], flops=C)
                    for name, m in zip(("gpuh_pess", "gpuh_central", "gpuh_opt"), mfu(w)):
                        r[name] = C / (PEAK * m) / 3600
                    rows.append(r)
    runs = pd.DataFrame(rows)
    runs.insert(0, "array_index", range(len(runs)))
    runs.to_csv(HERE / "runs.csv", index=False)

    meas = P0 / "probe.csv"
    if meas.exists():                                  # measured: tokens / (tokens/s) + one full eval per eval point
        pr = pd.read_csv(meas).set_index(["width", "vocab"])
        n_eval = runs.run.map(lambda k: len(EVAL_TPP) if k in ("const", "lr x0.5", "lr x2") else 1)

        def probe(w, V, col):
            # a body the probe has not reached yet: the widest measured body at this vocab, scaled by FLOPs per token
            if (w, V) in pr.index:
                return pr.loc[(w, V), col], False
            wm = max(x for x, v in pr.index if v == V and x < w)
            ratio = flops_per_token(w, hp[w]["depth"], V) / flops_per_token(wm, hp[wm]["depth"], V)
            return (pr.loc[(wm, V), col] / ratio if col == "tokens_per_sec" else pr.loc[(wm, V), col] * ratio), True

        tps = [probe(w, V, "tokens_per_sec")[0] for w, V in zip(runs.width, runs.vocab)]
        ev = [probe(w, V, "eval_sec_full")[0] for w, V in zip(runs.width, runs.vocab)]
        runs["gpuh_measured"] = (runs.tokens / tps + n_eval * ev) / 3600
        runs["gpuh_extrapolated"] = [probe(w, V, "tokens_per_sec")[1] for w, V in zip(runs.width, runs.vocab)]
        runs.to_csv(HERE / "runs.csv", index=False)
    g = runs.groupby(["phase", "width"]).agg(N_nonemb=("N_nonemb", "first"), runs=("run", "size"),
                                              tokens_B=("tokens", lambda x: x.sum() / 1e9),
                                              gpuh_pess=("gpuh_pess", "sum"), gpuh_central=("gpuh_central", "sum"),
                                              gpuh_opt=("gpuh_opt", "sum"),
                                              **({"gpuh_measured": ("gpuh_measured", "sum")} if meas.exists() else {})
                                              ).reset_index()
    g.to_csv(HERE / "budget.csv", index=False)
    cols = ["gpuh_pess", "gpuh_central", "gpuh_opt"] + (["gpuh_measured"] if meas.exists() else [])
    ph = runs.groupby("phase")[cols].sum()
    lines = ["# DM-4 GPU-hour estimate (L40S, FLOP-based; phase 0 replaces the ASSUME lines with measured throughput)",
             "", g.to_string(index=False, float_format=lambda v: f"{v:.2f}"), "", ph.to_string(float_format=lambda v: f"{v:.1f}"), "",
             f"Total runs: {len(runs)} (phase 1: {(runs.phase == 1).sum()}, phase 2: {(runs.phase == 2).sum()})",
             f"Total GPU-h: pessimistic {runs.gpuh_pess.sum():.1f}, central {runs.gpuh_central.sum():.1f}, "
             f"optimistic {runs.gpuh_opt.sum():.1f}",
             f"Largest single run (central): {runs.gpuh_central.max():.2f} GPU-h "
             f"({runs.loc[runs.gpuh_central.idxmax(), ['width', 'vocab', 'tie', 'run']].to_dict()})"]
    if meas.exists():
        dev = pd.read_csv(meas).device.iloc[0]
        lines += ["", f"MEASURED (phase 0 probe on one {dev}, bytes per token measured): total "
                  f"{runs.gpuh_measured.sum():.1f} GPU-h; phase 1 {runs[runs.phase == 1].gpuh_measured.sum():.1f}, "
                  f"phase 2 {runs[runs.phase == 2].gpuh_measured.sum():.1f}; largest single run "
                  f"{runs.gpuh_measured.max():.2f} GPU-h "
                  f"({runs.loc[runs.gpuh_measured.idxmax(), ['width', 'vocab', 'tie', 'run']].to_dict()})"]
        if runs.gpuh_extrapolated.any():
            lines += [f"Not yet probed, scaled from the widest probed body by FLOPs per token: widths "
                      f"{sorted(runs[runs.gpuh_extrapolated].width.unique().tolist())} "
                      f"({runs[runs.gpuh_extrapolated].gpuh_measured.sum():.1f} GPU-h of the total)"]
    (HERE / "budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
