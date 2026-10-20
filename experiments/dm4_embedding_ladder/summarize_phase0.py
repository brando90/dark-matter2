"""Gather DM-4 phase 0 outputs on Sherlock into small files that come back with the results tgz.

    python summarize_phase0.py --root $DM4 probe   # probe/*/probe.json -> probe/probe.csv
    python summarize_phase0.py --root $DM4 all     # + G1 (g1/*/result.json) and Porian check -> phase0_summary.json

Locally: copy probe/probe.csv and data/bpt.json to data_phase0/, re-run design.py (measured budget.md).
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

PORIAN_G1 = {128: 4.796, 160: 4.598}      # Porian rw tuned-short-const at D = 20N (DM-2 values; PREREG G1)
TOL = 0.05
VOCAB = {"neox": 50432, "bpe32k": 32768, "bpe8k": 8192, "bpe50k": 50432}


def probe(root):
    rows = []
    for p in sorted((root / "probe").glob("*/probe.json")):
        j = json.loads(p.read_text())
        m = j["meta"]
        rows.append(dict(name=p.parent.name, width=int(m["args"]["width"]), vocab=VOCAB[m["args"]["vocab_name"]],
                         tie=m["args"]["tie"], compiled=j["compiled"], micro_bs=j["micro_bs"],
                         tokens_per_sec=j["tokens_per_sec"], sec_per_step=j["sec_per_step"],
                         eval_sec_full=j["eval_sec_full"], peak_mem_gb=j["peak_mem_gb"], wall_s=j["wall_s"],
                         device=m["device"]))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no probe results")
        return df
    df[~df.compiled].drop(columns=["name"]).to_csv(root / "probe" / "probe.csv", index=False)
    df.to_csv(root / "probe" / "probe_all.csv", index=False)
    print(df.to_string(index=False))
    print(f"probe wall time (all probes): {df.wall_s.sum() / 3600:.3f} GPU-h")
    return df


def g1(root):
    rows = []
    for p in sorted((root / "g1").glob("*/result.json")):
        j = json.loads(p.read_text())
        a = j["meta"]["args"]
        for e in j["evals"]:
            rows.append(dict(width=int(a["width"]), seed=int(a["seed"]), tpp=e["tpp"], y=e["y"], step=e["step"],
                             tokens=e["tokens"], bytes=e["bytes"], done="const" in j["done"],
                             gpu_h=j["gpu_seconds"] / 3600))
    df = pd.DataFrame(rows)
    out = dict(rows=rows, gate={})
    if df.empty:
        return out
    for w, ref in PORIAN_G1.items():
        s = df[(df.width == w) & (df.tpp == 20)]
        if s.empty:
            out["gate"][w] = dict(status="missing")
            continue
        m = float(s.y.mean())
        out["gate"][w] = dict(seeds=int(len(s)), y=list(map(float, s.y)), mean=m,
                              sd=float(s.y.std(ddof=1)) if len(s) > 1 else None, porian=ref, diff=m - ref,
                              passes=bool(abs(m - ref) <= TOL and len(s) == 3))
    out["G1"] = "PASS" if all(v.get("passes") for v in out["gate"].values()) else "FAIL_OR_INCOMPLETE"
    out["g1_gpu_h"] = float(df.groupby(["width", "seed"]).gpu_h.first().sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("DM4", ".")))
    ap.add_argument("what", choices=["probe", "all"])
    a = ap.parse_args()
    root = a.root.resolve()
    df = probe(root)
    if a.what == "all":
        s = g1(root)
        pc = root / "probe" / "porian_check.json"
        s["porian_check"] = json.loads(pc.read_text()) if pc.exists() else None
        s["probe_gpu_h"] = float(df.wall_s.sum() / 3600) if not df.empty else None
        bpt = root / "data" / "bpt.json"
        s["bpt"] = json.loads(bpt.read_text()) if bpt.exists() else None
        (root / "phase0_summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
