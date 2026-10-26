"""Compare a fresh data/bpt.json with the reference (Sherlock's data_phase0/bpt.json): bytes/token must match to 3
decimals for every tokenizer and split; prints token counts side by side, writes --out, exits 3 on any mismatch.

    python farmshare/compare_bpt.py $DM4/data/bpt.json data_phase0/bpt.json --out $DM4/data/bpt_compare.json
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("new")
ap.add_argument("ref")
ap.add_argument("--out")
a = ap.parse_args()
new, ref = json.load(open(a.new)), json.load(open(a.ref))
rows, ok = [], True
for tok in sorted(ref):
    for split in ("val", "train", "train_first"):
        r, n = ref[tok][split], new.get(tok, {}).get(split)
        same = n is not None and round(n["bytes_per_token"], 3) == round(r["bytes_per_token"], 3)
        ok &= same
        rows.append(dict(tokenizer=tok, split=split, ref_bpt=r["bytes_per_token"], new_bpt=n and n["bytes_per_token"],
                         ref_tokens=r["tokens_incl_eos"], new_tokens=n and n["tokens_incl_eos"],
                         ref_bytes=r["bytes"], new_bytes=n and n["bytes"], match_3dp=same))
        print(f"{tok:7s} {split:11s} bpt {r['bytes_per_token']:.4f} vs {n['bytes_per_token'] if n else float('nan'):.4f}"
              f"  tokens {r['tokens_incl_eos']:>11,} vs {n['tokens_incl_eos'] if n else 0:>11,}  {'OK' if same else 'DIFF'}")
if a.out:
    json.dump(dict(all_match_3dp=ok, rows=rows), open(a.out, "w"), indent=1)
print("BPT_MATCH" if ok else "BPT_MISMATCH")
raise SystemExit(0 if ok else 3)
