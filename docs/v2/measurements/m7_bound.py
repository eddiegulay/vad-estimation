"""M7: is the pipeline dataloader-bound? Measure per-item dataset time vs MPS train-step time."""
import sys, time, json
from pathlib import Path
REPO = Path("/Users/eddiegulay/Documents/Work/Misc/vad-models")
sys.path.insert(0, str(REPO/"src"))
import numpy as np, torch
from vad.config import load_config, load_yaml
from vad.data.dataset import VADDataset
from vad.data.manifest import load_index
from vad.models import build_model

data_cfg = load_config(REPO/"configs/data/paths.yaml", REPO/"configs/data/default.yaml")
cache_root = REPO/data_cfg["cache_root"]; sr = data_cfg["sample_rate"]
mcfg = load_yaml(REPO/"configs/model/crnn_v1.yaml")
chunk = mcfg["frontend"]["chunk_samples"]; hop_s = chunk/sr
esc = load_index(cache_root/"index/esc50_index.json")
rir = load_index(cache_root/"index/rir_index.json")
ds = VADDataset(REPO/"data_cache/manifests/train.jsonl", cache_root, sr, hop_s, esc, rir, run_seed=1)
ds.set_epoch(1)

# per-item fetch+augment time over 24 random items
rng = np.random.default_rng(0)
idx = rng.choice(len(ds), 24, replace=False)
_ = ds[int(idx[0])]  # warm imports/caches
t0 = time.perf_counter(); durs = []
tt = []
for i in idx:
    s = time.perf_counter(); item = ds[int(i)]; tt.append(time.perf_counter()-s)
    durs.append(item["waveform"].shape[0]/sr)
tt = np.array(tt)
print(f"per-item __getitem__: mean {tt.mean()*1000:.0f} ms  median {np.median(tt)*1000:.0f} ms  p95 {np.percentile(tt,95)*1000:.0f} ms  (mean audio dur {np.mean(durs):.1f}s)")
per_batch_data_4workers = 64*tt.mean()/4
print(f"implied data time per batch of 64 with 4 workers: {per_batch_data_4workers:.2f} s")

# MPS train step: CRNN and TCN forward+backward on [64, 625*512] (v1 regime) and [64, 250*512] (v2 8s regime)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def step_time(name, nchunks, reps=5):
    cfg = load_yaml(REPO/f"configs/model/{name}.yaml")
    m = build_model(cfg["architecture"], cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    wav = torch.randn(64, nchunks*chunk, device=dev)*0.05
    lab = torch.randint(0,2,(64,nchunks),device=dev).float()
    for r in range(reps+2):
        if r==2:
            if dev.type=="mps": torch.mps.synchronize()
            t0=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        p = m.forward_full(wav)
        loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6), lab)
        loss.backward(); opt.step()
    if dev.type=="mps": torch.mps.synchronize()
    dt=(time.perf_counter()-t0)/reps
    print(f"{name} {nchunks}-chunk batch64 train step on {dev.type}: {dt:.2f} s")
    return dt

t_crnn_625 = step_time("crnn_v1", 625)
t_crnn_250 = step_time("crnn_v1", 250)
t_tcn_625  = step_time("tcn_v1", 625)
t_tcn_250  = step_time("tcn_v1", 250)
print(f"\nv1 regime: data {per_batch_data_4workers:.2f}s vs crnn step {t_crnn_625:.2f}s vs crnn+tcn {t_crnn_625+t_tcn_625:.2f}s")
print(f"v2 8s regime: data {per_batch_data_4workers:.2f}s vs crnn step {t_crnn_250:.2f}s vs crnn+tcn {t_crnn_250+t_tcn_250:.2f}s")
print("(v2 data time may drop if decode is proportional to crop; v1 decodes full sources regardless)")
