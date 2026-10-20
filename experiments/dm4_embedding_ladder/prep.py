"""DM-4 data prep (CPU): RefinedWeb shards -> held-out validation slice, BPE training text, byte-level BPEs and uint16
token memmaps per tokenizer. Every step is idempotent (skips finished outputs), so a killed job just re-runs.

    python prep.py --root $DM4 fetch     --train-shards 8     # download + sha256-check shards (HF LFS oids)
    python prep.py --root $DM4 bpe                            # own byte-level BPE at 50,432; 8,192/32,768 = prefixes
    python prep.py --root $DM4 tokenize  --tokenizers neox bpe8k bpe32k bpe50k
    python prep.py --root $DM4 stats                          # bytes per token -> data/bpt.json
    python prep.py --root $DM4 all       --train-shards 8

Shard roles (fixed by index in the sorted HF file list, written to data/manifest.json):
  0 = validation (first 25,000,000 UTF-8 bytes of whole documents), 1-3 = BPE training (first 1e9 bytes of whole
  documents), 4.. = the training pool (phase 0/1: 8 shards = ~3.8 GB; phase 2 adds shards with --train-shards N).
Layout: data/tok/<tokenizer>/<split>-<shard>.u16 (each document's tokens then EOS id 0) and .off.npy (int64 offsets,
n_docs + 1); data/bytes/<split>-<shard>.npy (int64 UTF-8 bytes per document, the same for every tokenizer).
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = "tiiuae/falcon-refinedweb"
NEOX_URL = "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json"
VAL_BYTES = int(os.environ.get("DM4_VAL_BYTES", 25_000_000))
BPE_BYTES = int(os.environ.get("DM4_BPE_BYTES", 1_000_000_000))
TRAIN_LIMIT = int(os.environ.get("DM4_TRAIN_LIMIT", 0)) or None   # smoke tests only: cap bytes per train shard
VAL_SHARDS, BPE_SHARDS, TRAIN_START = [0], [1, 2, 3], 4
OWN_VOCABS = {"bpe8k": 8192, "bpe32k": 32768, "bpe50k": 50432}
EOS = "<|endoftext|>"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, dest, sha=None, tries=4):
    dest = Path(dest)
    if dest.exists() and (sha is None or sha256(dest) == sha):
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            if sha is None or sha256(tmp) == sha:
                tmp.rename(dest)
                return
            log("sha256 mismatch", dest.name)
        except Exception as e:  # network hiccup: wait and retry a few times, then fail loudly
            log("retry", i + 1, dest.name, repr(e))
        time.sleep(10 * (i + 1))
    raise SystemExit(f"could not fetch {url}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(a, root):
    data = root / "data"
    (data / "shards").mkdir(parents=True, exist_ok=True)
    man_p = data / "manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
    else:
        # the tree API returns the first 1,000 files (sorted); DM-4 uses only the first few dozen shards
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/data") as r:
            files = sorted((f for f in json.load(r) if f["path"].endswith(".parquet")), key=lambda f: f["path"])
        man = dict(repo=REPO, shards=[dict(index=i, path=f["path"], size=f["size"], sha256=f["lfs"]["oid"])
                                      for i, f in enumerate(files)])
        for s in man["shards"]:   # refuse a listing whose order would mis-index shards
            if not s["path"].startswith(f"data/train-{s['index']:05d}-of-"):
                raise SystemExit(f"unexpected shard order: {s}")
    need = VAL_SHARDS + BPE_SHARDS + list(range(TRAIN_START, TRAIN_START + a.train_shards))
    man["roles"] = dict(val=VAL_SHARDS, bpe=BPE_SHARDS, train=list(range(TRAIN_START, TRAIN_START + a.train_shards)))
    for i in need:
        s = man["shards"][i]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{s['path']}"
        get(url, data / "shards" / f"{i:05d}.parquet", s["sha256"])
        s["verified"] = True
        log("shard", i, "ok", s["path"])
    get(NEOX_URL, data / "neox_tokenizer.json")
    man["neox_tokenizer_sha256"] = sha256(data / "neox_tokenizer.json")
    man_p.write_text(json.dumps(man, indent=1))


def docs(root, i, limit_bytes=None):
    import pyarrow.parquet as pq
    col = pq.read_table(root / "data" / "shards" / f"{i:05d}.parquet", columns=["content"]).column("content")
    out, n = [], 0
    for s in col.to_pylist():
        b = len(s.encode("utf-8"))
        if limit_bytes is not None and n + b > limit_bytes:
            break
        out.append(s)
        n += b
    return out


def split_docs(root, split, i):
    if split == "val":
        return docs(root, i, VAL_BYTES)
    return docs(root, i, TRAIN_LIMIT)


def cmd_bpe(a, root):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tdir = root / "data" / "tokenizers"
    tdir.mkdir(parents=True, exist_ok=True)
    full = tdir / "bpe50k.json"
    if not full.exists():
        neox = Tokenizer.from_file(str(root / "data" / "neox_tokenizer.json"))
        tok = Tokenizer(models.BPE())
        tok.normalizer, tok.pre_tokenizer, tok.decoder = neox.normalizer, neox.pre_tokenizer, neox.decoder
        tr = trainers.BpeTrainer(vocab_size=OWN_VOCABS["bpe50k"], special_tokens=[EOS], show_progress=False,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2)

        def it():
            left = BPE_BYTES
            for i in BPE_SHARDS:
                d = docs(root, i, left)
                left -= sum(len(s.encode("utf-8")) for s in d)
                log("bpe text shard", i, len(d), "docs; left", left)
                for k in range(0, len(d), 1000):
                    yield d[k:k + 1000]
                if left <= 0:
                    break
        t = time.time()
        tok.train_from_iterator(it(), trainer=tr)
        tok.save(str(full))
        log("bpe50k trained in", round(time.time() - t), "s, vocab", tok.get_vocab_size())
    # 8,192 and 32,768: the first V - 257 merges of the same run (BPE is greedy, so a smaller run = this prefix)
    j = json.loads(full.read_text())
    n_base = sum(1 for v in j["model"]["vocab"].values() if v < 257)
    assert n_base == 257, n_base
    for name, V in OWN_VOCABS.items():
        p = tdir / f"{name}.json"
        if p.exists():
            continue
        k = json.loads(full.read_text())
        k["model"]["vocab"] = {t: i for t, i in k["model"]["vocab"].items() if i < V}
        k["model"]["merges"] = k["model"]["merges"][:V - 257]
        p.write_text(json.dumps(k))
        log(name, "vocab", len(k["model"]["vocab"]))


def load_tok(root, name):
    from tokenizers import Tokenizer
    if name == "neox":
        return Tokenizer.from_file(str(root / "data" / "neox_tokenizer.json"))
    return Tokenizer.from_file(str(root / "data" / "tokenizers" / f"{name}.json"))


def shards_for(root, split):
    man = json.loads((root / "data" / "manifest.json").read_text())
    return man["roles"][split]


def cmd_tokenize(a, root):
    data = root / "data"
    (data / "bytes").mkdir(parents=True, exist_ok=True)
    for name in a.tokenizers:
        tok = load_tok(root, name)
        eos = tok.token_to_id(EOS)
        assert eos == 0, (name, eos)
        out = data / "tok" / name
        out.mkdir(parents=True, exist_ok=True)
        for split in ("val", "train"):
            todo = shards_for(root, split)
            if split == "train" and a.tok_shards:
                todo = todo[:a.tok_shards]
            for i in todo:
                stem = f"{split}-{i:05d}"
                if (out / f"{stem}.off.npy").exists():
                    continue
                t = time.time()
                d = split_docs(root, split, i)
                bp = data / "bytes" / f"{stem}.npy"
                if not bp.exists():
                    np.save(bp.with_suffix(".tmp.npy"), np.array([len(s.encode("utf-8")) for s in d], dtype=np.int64))
                    os.replace(bp.with_suffix(".tmp.npy"), bp)
                enc = getattr(tok, "encode_batch_fast", tok.encode_batch)   # _fast skips offsets (tokenizers >= 0.20)
                lens, chunks = [], []
                for k in range(0, len(d), 8192):
                    ids = [e.ids + [eos] for e in enc(d[k:k + 8192], add_special_tokens=False)]
                    lens += [len(x) for x in ids]
                    chunks.append(np.fromiter((t for x in ids for t in x), dtype=np.uint16))
                arr = np.concatenate(chunks)
                assert arr.max() < 65536
                arr.tofile(out / f"{stem}.u16.part")
                os.replace(out / f"{stem}.u16.part", out / f"{stem}.u16")
                np.save(out / f"{stem}.off.tmp.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
                os.replace(out / f"{stem}.off.tmp.npy", out / f"{stem}.off.npy")   # readers key on .off.npy
                log(name, stem, len(d), "docs", len(arr), "tokens", round(time.time() - t), "s")


def cmd_stats(a, root):
    data = root / "data"
    res = {}
    for name in sorted(p.name for p in (data / "tok").iterdir()):
        r = {}
        for split in ("val", "train"):
            nb = nt = nd = 0
            n_sh = 0
            for i in shards_for(root, split):
                stem = f"{split}-{i:05d}"
                if not (data / "tok" / name / f"{stem}.off.npy").exists():
                    break
                n_sh += 1
                off = np.load(data / "tok" / name / f"{stem}.off.npy")
                nb += int(np.load(data / "bytes" / f"{stem}.npy").sum())
                nt += int(off[-1])
                nd += len(off) - 1
            r[split] = dict(shards=n_sh, bytes=nb, tokens_incl_eos=nt, docs=nd, bytes_per_token=nb / nt)
        # the first training shard: fixed whatever the pool size; train.py sets D_bytes with NeoX's value here
        i0 = shards_for(root, "train")[0]
        off = np.load(data / "tok" / name / f"train-{i0:05d}.off.npy")
        b0 = int(np.load(data / "bytes" / f"train-{i0:05d}.npy").sum())
        r["train_first"] = dict(shard=i0, bytes=b0, tokens_incl_eos=int(off[-1]), bytes_per_token=b0 / int(off[-1]))
        res[name] = r
    (data / "bpt.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("DM4", ".")))
    ap.add_argument("cmd", choices=["fetch", "bpe", "tokenize", "stats", "all"])
    ap.add_argument("--train-shards", type=int, default=8)
    ap.add_argument("--tokenizers", nargs="+", default=["neox", "bpe8k", "bpe32k", "bpe50k"])
    ap.add_argument("--tok-shards", type=int, default=0, help="tokenize only the first N train shards (0 = all)")
    a = ap.parse_args()
    root = a.root.resolve()
    steps = ["fetch", "bpe", "tokenize", "stats"] if a.cmd == "all" else [a.cmd]
    for s in steps:
        log("==", s)
        globals()[f"cmd_{s}"](a, root)


if __name__ == "__main__":
    main()
