"""M2-M4,M6: TEN/train manifest timescale stats, cluster counts, val max rows."""
import json, gzip
from pathlib import Path
import numpy as np
REPO = Path("/Users/eddiegulay/Documents/Work/Misc/vad-models")

def load_jsonl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]

def run_stats(recs):
    total_dur = 0.0; speech_dur = 0.0; transitions = 0
    interior_gaps = []   # non-speech runs strictly between speech (interior)
    all_nonspeech = []   # every non-speech run incl. edges
    for r in recs:
        iv = r["label_intervals"]
        total_dur += iv[-1][1]
        labs = [x[2] for x in iv]
        for a,b,l in iv:
            if l==1: speech_dur += b-a
        for i in range(1, len(labs)):
            if labs[i] != labs[i-1]: transitions += 1
        for i,(a,b,l) in enumerate(iv):
            if l==0:
                all_nonspeech.append(b-a)
                if 0 < i < len(iv)-1:
                    interior_gaps.append(b-a)
    g = np.array(interior_gaps); ns = np.array(all_nonspeech)
    out = {
        "n_examples": len(recs),
        "total_dur_s": round(total_dur,1),
        "occupancy": round(speech_dur/total_dur,4),
        "transitions_per_min": round(transitions/(total_dur/60),2),
        "n_interior_gaps": len(g),
    }
    if len(g):
        out["interior_gap_ms_p10_p50_p90"] = [round(float(np.percentile(g,q))*1000) for q in (10,50,90)]
        out["interior_gap_max_s"] = round(float(g.max()),2)
    if len(ns):
        mass = ns.sum()
        out["nonspeech_run_median_s"] = round(float(np.median(ns)),3)
        out["nonspeech_mass_lt_0.5s"] = round(float(ns[ns<0.5].sum()/mass),3)
        out["nonspeech_mass_gt_2s"] = round(float(ns[ns>2.0].sum()/mass),3)
    return out

ten = load_jsonl(REPO/"data_cache/manifests/test_ten.jsonl")
print("TEN test:", json.dumps(run_stats(ten)))

train = load_jsonl(REPO/"runs/v1/manifests/train.jsonl.gz")
print("v1 train pooled:", json.dumps(run_stats(train)))
for kind in sorted({r["kind"] for r in train}):
    sub = [r for r in train if r["kind"]==kind]
    print(f"v1 train {kind}:", json.dumps(run_stats(sub)))

val = load_jsonl(REPO/"runs/v1/manifests/val.jsonl.gz")
print("v1 val pooled:", json.dumps(run_stats(val)))
durs = [r["label_intervals"][-1][1] for r in val]
mx = max(durs)
chunks = int(mx*16000)//512
print(f"val max duration {mx:.1f}s -> {chunks} chunks; x64 batch = {chunks*64} rows")

# cluster counts for Phase 1.2
ls = json.load(open(REPO/"data_cache/index/librispeech_index.json"))
rec0 = ls[0] if isinstance(ls, list) else None
print("librispeech index type/keys:", type(ls).__name__, (list(rec0.keys()) if rec0 else list(ls.keys())[:10]))
