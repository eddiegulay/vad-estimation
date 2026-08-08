"""M4: cluster counts backing Phase 1.2 (48 LS speakers, 200 FLEURS, ESC-50 fold5, 9 AMI series)."""
import json, gzip, re
from pathlib import Path
from collections import Counter
REPO = Path("/Users/eddiegulay/Documents/Work/Misc/vad-models")

def load_jsonl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]

ls = json.load(open(REPO/"data_cache/index/librispeech_index.json"))
def spk(p): return Path(p).name.split("-")[0]
by_split = {}
for r in ls:
    by_split.setdefault(r["split"], set()).add(spk(r["cached_path"]))
for s, ss in sorted(by_split.items()):
    print(f"LS split {s}: {len(ss)} speakers, {sum(1 for r in ls if r['split']==s)} utts, "
          f"{sum(r['duration_s'] for r in ls if r['split']==s)/3600:.2f} h")

# which speakers did v1 train/val use?
used = set()
for mf in ["train.jsonl.gz","val.jsonl.gz"]:
    for r in load_jsonl(REPO/"runs/v1/manifests"/mf):
        for srcs in r.get("sources", []):
            p = srcs.get("path","")
            if "librispeech" in p.lower():
                used.add(spk(p))
allspk = set().union(*by_split.values())
print(f"v1 train+val used {len(used)} LS speakers; held-out speakers: {len(allspk-used)}")
for s, ss in sorted(by_split.items()):
    print(f"  fully held-out split? {s}: {len(ss - used)}/{len(ss)} speakers unused")

fl = load_jsonl(REPO/"data_cache/manifests/sanity_fleurs.jsonl")
srcs = set()
for r in fl:
    for srec in r["sources"]: srcs.add(srec["path"])
print(f"FLEURS sanity manifest: {len(fl)} records, {len(srcs)} unique source files")
fdir = None
for r in fl[:1]:
    fdir = Path(r["sources"][0]["path"]).parent
if fdir and fdir.exists():
    n = len(list(fdir.glob("*.wav")))
    print(f"FLEURS dir {fdir}: {n} wav files on disk")

esc = json.load(open(REPO/"data_cache/index/esc50_index.json"))
folds = Counter(r.get("fold") for r in esc)
print("ESC-50 folds:", dict(folds))

ami = json.load(open(REPO/"data_cache/index/ami_index.json"))
print("AMI index entries:", len(ami))
ids = [r.get("meeting_id") or r.get("id") or "" for r in ami] if isinstance(ami,list) else list(ami.keys())
series = Counter(re.match(r"([A-Z]+\d+)", i).group(1) if re.match(r"([A-Z]+\d+)", i) else i for i in ids)
print("AMI series:", dict(series), "-> n series:", len(series))
