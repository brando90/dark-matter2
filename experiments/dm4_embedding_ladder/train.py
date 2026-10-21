"""DM-4 trainer: one run (its constant-LR leg plus its WSD cooldown branches) or a throughput probe, on one GPU.

    python train.py --root $DM4 --index 7 --out runs/7                 # run row 7 of runs.csv (const + cooldowns)
    python train.py --root $DM4 --index 7 --out probe/7 --probe-steps 60
    python train.py --root $DM4 --width 128 --depth 4 --vocab-name neox --tie untied --seed 0 --lr 0.0115 --bs 28 \
        --beta2 0.99 --warmup-tokens 7503872 --out runs/x                # the same without runs.csv

Recipe as Porian et al. (OpenLM): AdamW (beta1 0.9, beta2 per width, eps 1e-8), independent weight decay 1e-4 (the
decay per step is 1e-4 at peak LR: AdamW weight_decay = 1e-4 / lr), z-loss 1e-4, grad clip 1.0, bf16 autocast,
context 2048, linear warmup lr (step+1)/W with W = warmup_tokens // (bs 2048), then constant LR. Global batch bs
sequences, split into micro-batches that fit memory (gradient accumulation; the update is the same).

Data: the training pool's documents shard by shard, shuffled within each shard by the seed (the same order for every
tokenizer, so arms at a body are paired by seed and see the same bytes; adding shards never changes a run's data), concatenated with EOS and cut into 2,048-token windows (each window
also reads the next token as its last target). The run trains until its byte budget, D_bytes = tpp x N_ref x the
measured NeoX bytes per token of the first training shard (fixed whatever the pool size), N_ref = N_nonemb + 50,432 w (Porian's N: body + head), so the NeoX arm
sees ~tpp x N_ref tokens, as Porian's D = 20 N.
Eval: the held-out validation slice in the run's own tokenizer, 2,048-token windows; y = summed NLL (nats) / number
of NeoX tokens in the same documents (nats per reference token; for NeoX arms the usual loss).
G1 trainer-gap diagnostic (diag_g1.py): --init-from <checkpoint> loads weights only (Porian's {"state_dict": module.*}
or our own {"model": ...}), a fresh AdamW; --start-step N trains steps N..--end-step M on the stream positions
step N would have read (the LR is the constant LR, which needs N >= warmup steps); --shuffle-windows draws each
batch from a seeded global permutation of the 2,048-token windows of that step range instead of bs consecutive windows
(same windows, same bytes, different batch composition). Only the final point is evaluated (tpp field = the token/N_ref ratio).
Cooldown branch cool@k: from the checkpoint at 0.8 D_k, LR linear to 0 over the last 0.2 D_k, evaluated at D_k.
Checkpoints every --ckpt-minutes; a requeued job resumes from the last one.
"""
import argparse
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import Transformer

HERE = Path(__file__).resolve().parent
CTX, REF_V = 2048, 50432
EVAL_TPP, COOL_TPP, COOL_FRAC = [2.5, 5, 10, 20], [5, 10, 20], 0.2
VOCAB = {"neox": 50432, "bpe8k": 8192, "bpe32k": 32768, "bpe50k": 50432}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------- data
class Split:
    """All documents of one split in one tokenizer: memmapped token shards + per-doc token and byte counts."""

    def __init__(self, root, name, split):
        man = json.loads((root / "data" / "manifest.json").read_text())
        self.toks, lens, nbytes = [], [], []
        for i in man["roles"][split]:
            stem = f"{split}-{i:05d}"
            if not (root / "data" / "tok" / name / f"{stem}.off.npy").exists():
                break            # shards are used in order; a run needing more bytes than are tokenized stops loudly
            off = np.load(root / "data" / "tok" / name / f"{stem}.off.npy")
            self.toks.append((np.memmap(root / "data" / "tok" / name / f"{stem}.u16", dtype=np.uint16, mode="r"), off))
            lens.append(np.diff(off))
            nbytes.append(np.load(root / "data" / "bytes" / f"{stem}.npy"))
        self.lens, self.bytes = np.concatenate(lens), np.concatenate(nbytes)
        self.n_shards = len(lens)
        self.shard_of = np.concatenate([np.full(len(l), k) for k, l in enumerate(lens)])
        self.local = np.concatenate([np.arange(len(l)) for l in lens])
        assert len(self.lens) == len(self.bytes)

    def doc(self, g):
        mm, off = self.toks[self.shard_of[g]]
        j = self.local[g]
        return mm[off[j]:off[j + 1]]


def doc_order(split, seed):
    """Shard by shard in index order, each shard's documents shuffled by (seed, shard): the first D bytes of a run
    come from the same shards whatever the pool size, and the same seed gives the same byte order in every arm."""
    out, k0 = [], 0
    for k in range(split.n_shards):
        n = int((split.shard_of == k).sum())
        out.append(k0 + np.random.default_rng([seed, k]).permutation(n))
        k0 += n
    return np.concatenate(out)


class Stream:
    """The token stream of the documents in `order`; window g covers positions [g CTX, g CTX + CTX + 1)."""

    def __init__(self, split, order):
        self.s, self.order = split, order
        self.cumT = np.cumsum(split.lens[order])
        self.cumB = np.cumsum(split.bytes[order])

    def pos_for_bytes(self, b):
        j = int(np.searchsorted(self.cumB, b))
        if j >= len(self.order):
            raise SystemExit(f"training pool too small: need {b:.4g} bytes, have {self.cumB[-1]:.4g}")
        t0, b0 = (self.cumT[j - 1], self.cumB[j - 1]) if j else (0, 0)
        return t0 + self.s.lens[self.order[j]] * (b - b0) / self.s.bytes[self.order[j]]

    def bytes_at(self, p):
        j = int(np.searchsorted(self.cumT, p, side="right"))
        t0, b0 = (self.cumT[j - 1], self.cumB[j - 1]) if j else (0, 0)
        if j >= len(self.order):
            return float(self.cumB[-1])
        return float(b0 + self.s.bytes[self.order[j]] * (p - t0) / self.s.lens[self.order[j]])

    def slice(self, a, b):
        j = int(np.searchsorted(self.cumT, a, side="right"))
        out, pos = [], a
        while pos < b:
            t0 = self.cumT[j - 1] if j else 0
            d = self.s.doc(self.order[j])
            take = d[pos - t0: min(len(d), b - t0)]
            out.append(take)
            pos += len(take)
            j += 1
        return np.concatenate(out)

    def batch(self, step, bs):
        a = step * bs * CTX
        x = self.slice(a, a + bs * CTX + 1).astype(np.int64)
        w = np.lib.stride_tricks.sliding_window_view(x, CTX + 1)[::CTX][:bs]
        return torch.from_numpy(np.ascontiguousarray(w))

    def window(self, g):
        return self.slice(g * CTX, g * CTX + CTX + 1).astype(np.int64)


class ShuffledWindows:
    """Batch s = bs windows from a seeded global permutation of the windows of steps [start, end): the same windows (and
    bytes) as Stream.batch over that range, but every batch mixes windows from all over the range (consecutive windows
    of one long document no longer share a batch)."""

    def __init__(self, stream, bs, start, end, seed):
        self.stream, self.bs, self.start = stream, bs, start
        self.perm = start * bs + np.random.default_rng([seed, 4242]).permutation((end - start) * bs)

    def batch(self, step, bs):
        assert bs == self.bs
        k = (step - self.start) * bs
        return torch.from_numpy(np.stack([self.stream.window(int(g)) for g in self.perm[k:k + bs]]))


class ChunkShuffled:
    """Amendment 2: batch s = bs windows of a seeded permutation of the windows of s's chunk of `chunk` steps. The same
    windows (and bytes) as Stream.batch, every batch drawn from `chunk` x bs windows spread over the chunk instead of
    bs consecutive windows (consecutive windows of one long document no longer share a batch). The permutation depends
    only on (seed, chunk index), never on the run's length, so the const leg, the cooldown legs and a resumed run see
    the same batch at the same step, and the data seen up to a chunk boundary equals the stream's prefix."""

    def __init__(self, stream, bs, chunk, seed):
        self.stream, self.bs, self.chunk, self.seed = stream, bs, chunk, seed
        self.n_windows = int(stream.cumT[-1] - 1) // CTX
        self._c, self._perm = None, None

    def _perm_of(self, c):
        if c != self._c:
            lo, hi = c * self.chunk * self.bs, min((c + 1) * self.chunk * self.bs, self.n_windows)
            assert hi > lo, "training pool too small for the run"
            self._c, self._perm = c, lo + np.random.default_rng([self.seed, 4243, c]).permutation(hi - lo)
        return self._perm

    def batch(self, step, bs):
        assert bs == self.bs
        perm = self._perm_of(step // self.chunk)
        k = (step % self.chunk) * bs
        return torch.from_numpy(np.stack([self.stream.window(int(g)) for g in perm[k:k + bs]]))


def prefetch(stream, bs, first, last):
    q = queue.Queue(maxsize=6)

    def work():
        for s in range(first, last):
            q.put((s, stream.batch(s, bs).pin_memory() if torch.cuda.is_available() else stream.batch(s, bs)))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


# ----------------------------------------------------------------------------------------------------- training
def loss_fn(logits, y, zl=1e-4):
    lse = torch.logsumexp(logits, dim=-1)
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
    return ce + zl * (lse ** 2).mean(), ce


def pick_micro(bs, V, cap_tokens=None):
    cap = cap_tokens or min(16 * CTX, int(12e9 / (V * 4 * 3)))
    return max(d for d in range(1, bs + 1) if bs % d == 0 and d * CTX <= max(cap, CTX))


@torch.no_grad()
def evaluate(model, val, n_ref_tokens, val_bytes, dev, micro, limit_windows=None):
    model.eval()
    T = len(val)
    starts = list(range(0, T - 1, CTX))[:limit_windows]
    nll, n = 0.0, 0
    for k in range(0, len(starts), micro):
        chunk = [torch.from_numpy(val[s: s + CTX + 1].astype(np.int64)) for s in starts[k:k + micro]]
        L = min(len(c) for c in chunk)
        full = [c for c in chunk if len(c) == L]
        for grp in ([full] + [[c] for c in chunk if len(c) != L]):
            x = torch.stack(grp).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(x[:, :-1])
            nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum").item()
            n += x[:, 1:].numel()
    model.train()
    return dict(nll_sum=nll, targets=n, y=nll / n_ref_tokens, nats_per_byte=nll / val_bytes, loss_own=nll / n)


def run(a):
    root = a.root.resolve()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    V = VOCAB[a.vocab_name]
    tied = a.tie == "tied"
    bpt = json.loads((root / "data" / "bpt.json").read_text())
    n_ref = a.n_nonemb_ref + REF_V * a.width
    bpt_ref = bpt["neox"]["train_first"]["bytes_per_token"]   # first training shard: fixed whatever the pool size
    D_bytes = a.tpp * n_ref * bpt_ref

    train = Split(root, a.vocab_name, "train")
    order = doc_order(train, a.seed)
    stream = Stream(train, order)
    tok_per_step = a.bs * CTX
    W = a.warmup_tokens // tok_per_step
    step_for = lambda b: math.ceil(stream.pos_for_bytes(b) / tok_per_step)
    if a.probe_steps:                    # a probe needs only probe_steps of data, whatever the body's budget
        total, evals, branches = a.probe_steps, {}, {}
    elif a.end_step:                     # a diagnostic leg: stop (and evaluate) at --end-step, no cooldown branches
        total, evals, branches = a.end_step, {a.end_step: round(a.end_step * tok_per_step / (n_ref * bpt_ref), 3)}, {}
        assert a.start_step < total
    else:
        total = step_for(D_bytes)
        evals = {step_for(k * n_ref * bpt_ref): k for k in a.eval_tpp if k <= a.tpp}
        branches = {} if a.no_cooldown else {k: (step_for((1 - COOL_FRAC) * k * n_ref * bpt_ref),
                                                 step_for(k * n_ref * bpt_ref)) for k in a.cool_tpp if k <= a.tpp}

    torch.manual_seed(a.seed)
    model = Transformer(a.width, a.depth, V, tied).to(dev)
    counts = model.counts()
    assert counts["N_nonemb"] == a.n_nonemb_ref, (counts, a.n_nonemb_ref)
    params = list(model.parameters())          # a tied weight appears once
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, a.beta2), eps=1e-8, weight_decay=a.indep_wd / a.lr,
                            fused=dev.type == "cuda")
    fwd = model
    if a.compile and dev.type == "cuda":
        fwd = torch.compile(model)

    meta = dict(args={k: str(v) for k, v in vars(a).items()}, V=V, tied=tied, **counts, N_ref=n_ref,
                bpt_ref=bpt_ref, D_bytes=D_bytes, total_steps=total, warmup_steps=W, tokens_per_step=tok_per_step,
                eval_steps={str(s): k for s, k in evals.items()}, branches={str(k): v for k, v in branches.items()},
                device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", torch=torch.__version__,
                pool_docs=len(order), pool_bytes=float(stream.cumB[-1]))
    res_p = out / "result.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else dict(meta=meta, evals=[], gpu_seconds=0.0, done=[])
    if "const" in res["done"] and all(f"cool@{k}" in res["done"] for k in branches):
        log("already complete:", res_p)
        return res

    val = Split(root, a.vocab_name, "val")
    val_stream = np.concatenate([val.doc(g) for g in range(len(val.lens))])
    ref_val = Split(root, "neox", "val")
    n_ref_val, val_bytes = int(ref_val.lens.sum()), int(val.bytes.sum())
    assert int(ref_val.bytes.sum()) == val_bytes

    micro = a.micro_bs or pick_micro(a.bs, V)
    t_start = time.time()
    last_ckpt = time.time()
    ck = out / "ckpt.pt"

    def save(path, step, leg):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, leg=leg, micro=micro), path)

    def train_leg(leg, start, end, lr_at):
        nonlocal micro, last_ckpt
        step, timed = start, []
        it = prefetch(batches, a.bs, start, end)
        for s, xb in it:
            for g in opt.param_groups:
                g["lr"] = lr_at(s)
            t0 = time.time()
            while True:
                try:
                    opt.zero_grad(set_to_none=True)
                    n_micro, ce_sum = a.bs // micro, 0.0
                    for m in range(n_micro):
                        x = xb[m * micro:(m + 1) * micro].to(dev, non_blocking=True)
                        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                            logits = fwd(x[:, :-1])
                        loss, ce = loss_fn(logits, x[:, 1:])
                        (loss / n_micro).backward()
                        ce_sum += ce.detach()
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro == 1:
                        raise
                    micro = max(d for d in range(1, micro) if a.bs % d == 0)
                    log("OOM: micro-batch ->", micro)
                    torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if dev.type == "cuda" and (a.probe_steps or s % 50 == 0):
                torch.cuda.synchronize()
            dt = time.time() - t0
            timed.append(dt)
            step = s + 1
            if s % a.log_every == 0 or step == end:
                log(f"{leg} step {step}/{end} lr {lr_at(s):.3g} ce {float(ce_sum) / n_micro:.4f} {dt:.3f}s")
            if not math.isfinite(float(ce_sum)):
                raise SystemExit(f"non-finite loss at step {step}")
            if leg == "const" and step in evals:
                k = evals[step]
                e = evaluate(model, val_stream, n_ref_val, val_bytes, dev, micro, limit_windows=a.eval_limit_windows)
                e.update(kind="const", tpp=k, step=step, tokens=step * tok_per_step, bytes=stream.bytes_at(step * tok_per_step))
                res["evals"].append(e)
                log("eval", leg, k, f"y={e['y']:.4f}")
            if leg == "const":
                for k, (b0, _) in branches.items():
                    if step == b0 and not (out / f"branch{k}.pt").exists():
                        save(out / f"branch{k}.pt", step, leg)
            if not a.probe_steps and time.time() - last_ckpt > a.ckpt_minutes * 60:
                save(ck, step, leg)
                res["gpu_seconds"] += time.time() - last_ckpt
                last_ckpt = time.time()
                res_p.write_text(json.dumps(res, indent=1))
        return timed

    const_lr = lambda s: a.lr * (s + 1) / W if s < W else a.lr
    start, batches = a.start_step, stream
    if a.shuffle_chunk_steps and not a.end_step:      # diagnostic legs (--end-step) keep their own order (arm A / --shuffle-windows)
        batches = ChunkShuffled(stream, a.bs, a.shuffle_chunk_steps, a.seed)
    if a.init_from:
        assert a.start_step >= W, f"--start-step {a.start_step} is inside the warmup ({W} steps)"
        sd = torch.load(a.init_from, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd.get("model", sd))
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
        log("initialised from", a.init_from, "(weights only, fresh AdamW) at step", start)
    if a.shuffle_windows:
        assert a.end_step, "--shuffle-windows needs --end-step (the permuted range is [start-step, end-step))"
        batches = ShuffledWindows(stream, a.bs, a.start_step, a.end_step, a.seed)
    if ck.exists() and not a.probe_steps:
        st = torch.load(ck, map_location=dev)
        if st["leg"] == "const":
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start, micro = st["step"], st["micro"]
            log("resumed const at step", start)
    if a.probe_steps:
        timed = train_leg("probe", 0, total, const_lr)
        k0 = min(a.probe_skip, len(timed) - 5)
        sps = float(np.mean(timed[k0:]))
        n_eval = 64
        te = time.time()
        evaluate(model, val_stream, n_ref_val, val_bytes, dev, micro, limit_windows=n_eval)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        eval_s_per_window = (time.time() - te) / n_eval
        probe = dict(meta=meta, micro_bs=micro, sec_per_step=sps, tokens_per_sec=tok_per_step / sps,
                     eval_sec_full=eval_s_per_window * math.ceil(len(val_stream) / CTX),
                     peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
                     compiled=bool(a.compile and dev.type == "cuda"), wall_s=time.time() - t_start)
        (out / "probe.json").write_text(json.dumps(probe, indent=1))
        log("probe", f"{probe['tokens_per_sec']:.0f} tok/s, micro {micro}, eval {probe['eval_sec_full']:.0f}s")
        return probe
    if "const" not in res["done"]:
        train_leg("const", start, total, const_lr)
        res["done"].append("const")
        res["gpu_seconds"] += time.time() - last_ckpt
        last_ckpt = time.time()
        res_p.write_text(json.dumps(res, indent=1))
    for k, (b0, b1) in branches.items():
        if f"cool@{k}" in res["done"]:
            continue
        st = torch.load(out / f"branch{k}.pt", map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        cool = lambda s, b0=b0, b1=b1: a.lr * (1 - (s - b0) / (b1 - b0))
        train_leg(f"cool@{k}", b0, b1, cool)
        e = evaluate(model, val_stream, n_ref_val, val_bytes, dev, micro)
        e.update(kind=f"cool@{k}", tpp=k, step=b1, tokens=b1 * tok_per_step, bytes=stream.bytes_at(b1 * tok_per_step))
        res["evals"].append(e)
        res["done"].append(f"cool@{k}")
        res["gpu_seconds"] += time.time() - last_ckpt
        last_ckpt = time.time()
        res_p.write_text(json.dumps(res, indent=1))
        log("eval", f"cool@{k}", f"y={e['y']:.4f}")
    for p in [ck] + [out / f"branch{k}.pt" for k in branches]:
        if p.exists() and not a.keep_ckpt:
            p.unlink()
    log("done", res_p)
    return res


def from_runs_csv(a):
    import csv
    rows = list(csv.DictReader(open(a.runs_csv)))
    r = rows[a.index]
    assert r["run"] == "const", f"row {a.index} is a {r['run']} branch; run its const row"
    a.width, a.depth, a.seed = int(r["width"]), int(r["depth"]), int(r["seed"])
    if a.vocab_name is None:     # --vocab-name / --tie given with --index override the row (the probe uses this)
        a.vocab_name = "neox" if r["tokenizer"] == "neox" else {8192: "bpe8k", 32768: "bpe32k", 50432: "bpe50k"}[int(r["vocab"])]
    a.tie = a.tie or r["tie"]
    a.lr, a.bs, a.beta2 = float(r["lr"]), int(r["bs"]), float(r["beta2"])
    a.warmup_tokens, a.n_nonemb_ref = int(r["warmup_tokens"]), int(r["N_nonemb"])
    lr_rows = [x for x in rows if x["width"] == r["width"] and x["vocab"] == r["vocab"] and x["tie"] == r["tie"]
               and x["tokenizer"] == r["tokenizer"] and x["seed"] == r["seed"] and x["run"].startswith("cool@")]
    a.tpp = 20
    a.no_cooldown = a.no_cooldown or not lr_rows
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("DM4", ".")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int)
    ap.add_argument("--runs-csv", default=str(HERE / "runs.csv"))
    ap.add_argument("--width", type=int); ap.add_argument("--depth", type=int)
    ap.add_argument("--vocab-name", choices=list(VOCAB)); ap.add_argument("--tie", choices=["tied", "untied"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float); ap.add_argument("--bs", type=int)
    ap.add_argument("--beta2", type=float, default=0.99); ap.add_argument("--warmup-tokens", type=int)
    ap.add_argument("--n-nonemb-ref", type=int, help="N_nonemb check (from runs.csv)")
    ap.add_argument("--tpp", type=float, default=20)
    ap.add_argument("--eval-tpp", type=float, nargs="+", default=EVAL_TPP)
    ap.add_argument("--cool-tpp", type=float, nargs="+", default=COOL_TPP)
    ap.add_argument("--indep-wd", type=float, default=1e-4)
    ap.add_argument("--micro-bs", type=int)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-cooldown", action="store_true")
    ap.add_argument("--init-from", help="checkpoint to load weights from (fresh AdamW); needs --start-step >= warmup steps")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--end-step", type=int, help="stop and evaluate here (diagnostic leg; no cooldown branches)")
    ap.add_argument("--shuffle-windows", action="store_true")
    ap.add_argument("--shuffle-chunk-steps", type=int, default=1024,
                    help="Amendment 2: shuffle the windows within each chunk of this many steps (0 = the old consecutive windows)")
    ap.add_argument("--eval-limit-windows", type=int, help="score only the first windows of the validation slice (CPU smoke)")
    ap.add_argument("--probe-steps", type=int, default=0)
    ap.add_argument("--probe-skip", type=int, default=20)
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--keep-ckpt", action="store_true")
    a = ap.parse_args()
    if a.index is not None:
        a = from_runs_csv(a)
    if a.n_nonemb_ref is None:
        from model import ffn_hidden
        a.n_nonemb_ref = a.depth * (4 * a.width ** 2 + 3 * a.width * ffn_hidden(a.width))
    run(a)


if __name__ == "__main__":
    main()
