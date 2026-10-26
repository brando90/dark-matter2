"""G1 trainer-gap diagnostic (AT-669). Resume from Porian et al.'s w128 untied-50k checkpoint at step 1211 and train to
step 2421 under OUR trainer in two arms, then compare with Porian's own step-2421 checkpoint on the FULL validation slice:

    arm A  our stream: step s reads bs consecutive 2,048-token windows (train.py default)
    arm B  --shuffle-windows: the same windows (steps 1211..2421), drawn from a seeded global permutation
    P1211, P2421  Porian's checkpoints evaluated with train.evaluate (same slice, same tokenizer, same metric)

Reading: A - P2421 = what is left of the gap after trainer + our data order (weights and step count equal);
B - A = the effect of batch composition alone. Fresh AdamW (Porian's optimizer state is not released), constant LR
0.0115 (past the warmup), beta2 0.99, bs 28, as Porian's tuned w128 run.

    python diag_g1.py --root $DM4 --out $DM4/g1diag --fetch-only          # login node: stage the two checkpoints (~60 MB)
    python diag_g1.py --root $DM4 --out $DM4/g1diag                      # the real run (about 25 min on one L40S)
    python diag_g1.py --root $DM4 --out o --bs 4 --warmup-tokens 8192 --start-step 3 --end-step 6 --limit-windows 8 \
        --ckpt-1211 a.pt --ckpt-2421 b.pt                               # CPU smoke with local checkpoints
Writes <out>/diag_g1.json (and <out>/armA, <out>/armB: train.py's result.json and log).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

from eval_porian import BASE, list_files
from model import Transformer
from train import Split, evaluate

HERE = Path(__file__).resolve().parent
PREFIX = "dataset=rw/hparams=tuned_warmup=short_decay=const/params=7M_"


def fetch(files, step, cache):
    f = next(f for f in files if f.startswith(PREFIX) and f.endswith(f"_step_{step}.pt"))
    local = cache / f.replace("/", "__")
    if not local.exists():
        urllib.request.urlretrieve(BASE + f.replace("+", "%2B"), local)
    return f, local


def load(path, dev):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    m = Transformer(128, 4, 50432, tied=False)
    m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    return m.to(dev)


def trace(log):
    return [dict(step=int(m[1]), ce=float(m[2])) for m in re.finditer(r"const step (\d+)/\d+ lr \S+ ce (\S+)", log)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("DM4", ".")))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-step", type=int, default=1211)
    ap.add_argument("--end-step", type=int, default=2421)
    ap.add_argument("--bs", type=int, default=28)
    ap.add_argument("--lr", type=float, default=0.0115)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--warmup-tokens", type=int, default=7503872)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-windows", type=int, default=None, help="score only the first windows of the slice (smoke)")
    ap.add_argument("--ckpt-1211"); ap.add_argument("--ckpt-2421")
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--fetch-only", action="store_true", help="download the two checkpoints into <root>/porian_ckpts and exit")
    a = ap.parse_args()
    root, out = a.root.resolve(), a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = {}
    if a.ckpt_1211 and a.ckpt_2421:
        ck = {a.start_step: (a.ckpt_1211, Path(a.ckpt_1211)), a.end_step: (a.ckpt_2421, Path(a.ckpt_2421))}
    else:
        cache = root / "porian_ckpts"
        cache.mkdir(parents=True, exist_ok=True)
        files = list_files()
        ck = {s: fetch(files, s, cache) for s in (a.start_step, a.end_step)}
    print("checkpoints:", {s: v[0] for s, v in ck.items()}, flush=True)
    if a.fetch_only:
        return

    val = Split(root, "neox", "val")
    stream = np.concatenate([val.doc(g) for g in range(len(val.lens))])
    n_ref, nbytes = int(val.lens.sum()), int(val.bytes.sum())
    ev = lambda m: evaluate(m, stream, n_ref, nbytes, dev, micro=8, limit_windows=a.limit_windows)
    res = dict(args={k: str(v) for k, v in vars(a).items()}, val_tokens=n_ref, val_bytes=nbytes, evals={})
    for arm in a.arms:
        d = out / f"arm{arm}"
        cmd = [sys.executable, str(HERE / "train.py"), "--root", str(root), "--out", str(d), "--width", "128", "--depth", "4",
               "--vocab-name", "neox", "--tie", "untied", "--seed", str(a.seed), "--lr", str(a.lr), "--bs", str(a.bs),
               "--beta2", str(a.beta2), "--warmup-tokens", str(a.warmup_tokens), "--init-from", str(ck[a.start_step][1]),
               "--start-step", str(a.start_step), "--end-step", str(a.end_step), "--log-every", str(a.log_every),
               "--ckpt-minutes", "1e9"] + (["--eval-limit-windows", str(a.limit_windows)] if a.limit_windows else []) \
              + (["--shuffle-windows"] if arm == "B" else [])
        d.mkdir(parents=True, exist_ok=True)
        print("arm", arm, " ".join(cmd), flush=True)
        log = subprocess.run(cmd, capture_output=True, text=True)
        (d / "train.log").write_text(log.stdout + log.stderr)
        if log.returncode:
            raise SystemExit(f"arm {arm} failed:\n{(log.stdout + log.stderr)[-3000:]}")
        r = json.loads((d / "result.json").read_text())
        final = [e for e in r["evals"] if e["step"] == a.end_step][-1]
        res["evals"][f"arm{arm}"] = dict(final, train_trace=trace(log.stdout), gpu_seconds=r["gpu_seconds"])
        print("arm", arm, "y =", final["y"], flush=True)

    for s, tag in ((a.start_step, "porian_start"), (a.end_step, "porian_end")):
        m = load(ck[s][1], dev)
        e = ev(m)
        e.update(checkpoint=ck[s][0], step=s)
        res["evals"][tag] = e
        print(tag, json.dumps(e), flush=True)
        del m

    y = {k: v["y"] for k, v in res["evals"].items()}
    res["summary"] = dict(y=y)
    if "armA" in y:
        res["summary"]["A_minus_porian_end"] = y["armA"] - y["porian_end"]
        res["summary"]["A_gain_over_start"] = y["porian_start"] - y["armA"]
        res["summary"]["porian_gain_over_start"] = y["porian_start"] - y["porian_end"]
    if "armA" in y and "armB" in y:
        res["summary"]["B_minus_A"] = y["armB"] - y["armA"]
        res["summary"]["B_minus_porian_end"] = y["armB"] - y["porian_end"]
    (out / "diag_g1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
