import json,glob,csv,os,statistics as st
root="/lfs/mercury1/0/brando9/at1211/dm4data"
rows=list(csv.DictReader(open("/lfs/mercury1/0/brando9/at1211/dm/experiments/dm4_embedding_ladder/runs.csv")))
def y(path):
    try: r=json.load(open(path))
    except Exception: return None
    if not r.get("done"): return None
    e=[x for x in r["evals"] if x["kind"]=="const" and abs(x["tpp"]-20)<1e-9]
    return e[-1]["nats_per_byte"] if e else None
vals={}
for r in rows:
    if r["phase"]=="1" and r["width"]=="128" and r["run"]=="const" and r["depth"]=="4":
        v=y(root+"/phase1/run"+r["array_index"]+"/result.json")
        if v is not None: vals[(r["tokenizer"],r["vocab"],r["tie"],int(r["seed"]))]=v
print("frozen w128 const rows found:",len(vals), sorted(set(k[:3] for k in vals)))
names={"bpe8k":"8192","bpe32k":"32768","neox":"50432"}
for f in glob.glob(root+"/ext_seeds/*/result.json"):
    n=os.path.basename(os.path.dirname(f)); vn,tie,s=n.split("_"); s=int(s[1:])
    v=y(f)
    if v is not None: vals[("ext",vn,tie,s)]=v
import collections
D=collections.defaultdict(dict)
for k,v in vals.items():
    tok,voc,tie,s=k
    voc={"bpe8k":"8192","bpe32k":"32768","neox":"50432"}.get(voc,voc)
    D[(voc,tie)][s]=v
for voc in sorted(set(k[0] for k in D),key=int):
    t=D[(voc,"tied")];u=D[(voc,"untied")]
    seeds=sorted(set(t)&set(u))
    diffs=[u[s]-t[s] for s in seeds]
    if not diffs: print(voc,"no pairs",sorted(t),sorted(u)); continue
    m=st.mean(diffs); sd=st.stdev(diffs) if len(diffs)>1 else float("nan")
    print(f"vocab {voc}: n={len(seeds)} seeds {seeds} tied mean {st.mean(t[s] for s in seeds):.4f} untied mean {st.mean(u[s] for s in seeds):.4f} untied-tied mean {m:+.4f} sd {sd:.4f} se {sd/len(diffs)**.5:.4f} untied better (diff<0) in {sum(d<0 for d in diffs)}/{len(diffs)}")
    print("   diffs",[round(d,4) for d in diffs])
