"""M5: recurrent cold-start transient length. Compare probs from cold start at offset k
vs full-context probs, as a function of chunks-since-cold-start. CRNN + TCN last.pt, CPU."""
import sys, json
from pathlib import Path
REPO = Path("/Users/eddiegulay/Documents/Work/Misc/vad-models")
sys.path.insert(0, str(REPO/"src"))
import numpy as np, torch
from vad.config import load_config, load_yaml
from vad.data.dataset import VADDataset
from vad.engine.checkpoint import load_checkpoint
from vad.models import build_model

data_cfg = load_config(REPO/"configs/data/paths.yaml", REPO/"configs/data/default.yaml")
cache_root = REPO/data_cfg["cache_root"]; sr = data_cfg["sample_rate"]

def transient(model_name, ckpt):
    mcfg = load_yaml(REPO/f"configs/model/{model_name}.yaml")
    chunk = mcfg["frontend"]["chunk_samples"]; hop_s = chunk/sr
    ds = VADDataset(REPO/"data_cache/manifests/test_ten.jsonl", cache_root, sr, hop_s)
    model = build_model(mcfg["architecture"], mcfg); load_checkpoint(ckpt, model); model.eval()
    # |dp| indexed by chunks-since-cold-start
    buckets = {}
    offsets = [16, 32, 64, 96, 128]
    with torch.no_grad():
        for i in range(len(ds)):
            wav = ds[i]["waveform"].unsqueeze(0)
            usable = (wav.shape[1]//chunk)*chunk
            wav = wav[:, :usable]
            full = model.forward_full(wav).numpy()[0]
            nch = usable//chunk
            for k in offsets:
                if k+40 >= nch: continue
                sub = model.forward_full(wav[:, k*chunk:]).numpy()[0]
                m = min(len(sub), len(full)-k, 60)
                dp = np.abs(sub[:m] - full[k:k+m])
                for j in range(m):
                    buckets.setdefault(j, []).append(dp[j])
    res = []
    for j in sorted(buckets):
        v = np.array(buckets[j])
        res.append((j, float(np.median(v)), float(np.percentile(v,95)), float(v.max())))
    print(f"== {model_name}: |Δprob| by chunks since cold start (median / p95 / max), n={len(buckets[0])} probes/chunk")
    for j, med, p95, mx in res[:20]:
        print(f" chunk {j:2d}: {med:.4f} / {p95:.4f} / {mx:.4f}")
    # first chunk where p95 < 0.05 and < 0.01
    for thr in (0.05, 0.01):
        first = next((j for j,_,p95,_ in res if p95 < thr), None)
        print(f" first chunk with p95 |Δp| < {thr}: {first}")

transient("crnn_v1", str(REPO/"runs/v1/crnn_v1_full/last.pt"))
transient("tcn_v1", str(REPO/"runs/v1/tcn_v1_full/last.pt"))
