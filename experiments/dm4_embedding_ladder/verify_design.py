"""DM-4 design checks (PREREG DM-4, verify.py additions 1-2): run after design.py and power.py.

    python verify_design.py      # prints PASS/FAIL per check, exit code 1 on any FAIL
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dm2_real_ladders"))
from data import porian_counts  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


before = sha(HERE / "runs.csv")
subprocess.run([sys.executable, str(HERE / "design.py")], check=True, capture_output=True)
check("1a runs.csv regenerates byte-identical from design.py", sha(HERE / "runs.csv") == before, before[:12])

r = pd.read_csv(HERE / "runs.csv")
bad = 0
for (w, d), g in r.groupby(["width", "depth"]):
    c = porian_counts(w, d)
    bad += int((g.N_nonemb != c["N_nonemb"]).any())
    t = g[(g.vocab == 50432) & (g.tie == "tied")]
    bad += int((t.N_total != c["N_total"]).any())
    u = g[g.tie == "untied"]
    bad += int((u.N_emb != 2 * u.vocab * w).any())
check("1b N_nonemb, tied-50k N_total (DM-2 porian_counts) and untied N_emb = 2Vw", bad == 0, f"{bad} mismatches")
check("1c every phase-1 main arm has 3 seeds, phase 2 has 1",
      bool((r[(r.run == "const") & (r.phase == 1) & ~((r.tokenizer == "own-bpe") & (r.vocab == 50432))]
            .groupby(["width", "vocab", "tie"]).seed.nunique() == 3).all()
           and (r[(r.run == "const") & (r.phase == 2)].groupby(["width", "vocab", "tie"]).seed.nunique() == 1).all()))

p = pd.read_csv(HERE / "power.csv")
p["cover"] = (p.c2_lo <= p.c_true + 1e-9) & (p.c_true <= p.c2_hi + 1e-9)
for (dz, law), g in p.groupby(["design", "law"]):
    if law.startswith("acc"):
        check(f"2 {dz} {law}: median c1 within 0.05, M1 set coverage >= 0.8, false fires <= 0.10",
              abs(g.c1.median() - g.c_true.iloc[0]) <= 0.05 + 1e-9
              and ((g.c1_lo <= g.c_true + 1e-9) & (g.c_true <= g.c1_hi + 1e-9)).mean() >= 0.8
              and g.fires.mean() <= 0.10,
              f"median c1 {g.c1.median():.2f}, cover {((g.c1_lo <= g.c_true + 1e-9) & (g.c_true <= g.c1_hi + 1e-9)).mean():.2f},"
              f" fires {g.fires.mean():.2f}")
    elif dz == "phase1+2+Porian":
        check(f"2 {dz} {law}: second term fires >= 0.8", g.fires.mean() >= 0.8, f"fires {g.fires.mean():.2f}")

print("ALL PASS" if all(results) else "SOME FAIL")
sys.exit(0 if all(results) else 1)
