"""Check the DM-4 model and eval against Porian et al.'s own released checkpoints (formll/resolving-scaling-law-
discrepancies on Hugging Face): load a checkpoint into model.Transformer (strict: every tensor name and shape must
match) and evaluate it on our held-out RefinedWeb validation slice with train.evaluate.

    python eval_porian.py --root $DM4 --out porian_check.json                 # w128 and w160 near D = 20 N
    python eval_porian.py --root $DM4 --out x.json --limit-windows 400        # quick CPU check on a subset

Porian's reported value at the checkpoint's step (val/loss in experiment_results) is on their own validation shards;
the difference to ours is the validation-set offset, which G1 can be read against (reported, not a gate).
With --limit-windows only part of the slice is scored: read loss_own (nats per target token), not y.
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

from model import Transformer
from train import Split, evaluate

BASE = "https://huggingface.co/formll/resolving-scaling-law-discrepancies/resolve/main/"


def list_files():
    with urllib.request.urlopen("https://huggingface.co/api/models/formll/resolving-scaling-law-discrepancies") as r:
        return [s["rfilename"] for s in json.load(r)["siblings"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("DM4", ".")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 160])
    a = ap.parse_args()
    root = a.root.resolve()
    cache = a.cache or root / "porian_ckpts"
    cache.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = Split(root, "neox", "val")
    stream = np.concatenate([val.doc(g) for g in range(len(val.lens))])
    n_ref, nbytes = int(val.lens.sum()), int(val.bytes.sum())
    files = list_files()
    shapes = {128: (4, "7M"), 160: (5, "9M"), 224: (6, "15M"), 288: (8, "22M"), 320: (9, "28M")}
    out = []
    for w in a.widths:
        depth, tag = shapes[w]
        cands = [f for f in files if f.startswith(f"dataset=rw/hparams=tuned_warmup=short_decay=const/params={tag}_")]
        bs = {128: 28, 160: 32, 224: 44, 288: 56, 320: 64}[w]
        n_porian = depth * (4 * w * w + 3 * w * (256 * -(-int(8 * w / 3) // 256))) + 50432 * w
        target_step = 20 * n_porian / (bs * 2048)
        f = min(cands, key=lambda f: abs(int(f.split("_step_")[1][:-3]) - target_step))
        step = int(f.split("_step_")[1][:-3])
        local = cache / f.replace("/", "__")
        if not local.exists():
            urllib.request.urlretrieve(BASE + f.replace("+", "%2B"), local)
        sd = torch.load(local, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        m = Transformer(w, depth, 50432, tied=False)
        m.load_state_dict(sd, strict=True)
        m.to(dev)
        e = evaluate(m, stream, n_ref, nbytes, dev, micro=8, limit_windows=a.limit_windows)
        e.update(width=w, depth=depth, checkpoint=f, step=step, tokens=step * bs * 2048,
                 tpp=step * bs * 2048 / n_porian, limit_windows=a.limit_windows)
        out.append(e)
        print(json.dumps(e), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
