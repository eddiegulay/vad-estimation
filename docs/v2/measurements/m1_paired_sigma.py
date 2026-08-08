"""M1: per-file TEN AUROC/F1 for CRNN & TCN last.pt on CPU -> paired vs absolute sigma, MDE arithmetic."""
import sys, json
from pathlib import Path
REPO = Path("/Users/eddiegulay/Documents/Work/Misc/vad-models")
sys.path.insert(0, str(REPO / "src"))
import numpy as np, torch
from vad.config import load_config, load_yaml
from vad.data.dataset import VADDataset
from vad.engine.checkpoint import load_checkpoint
from vad.models import build_model

torch.manual_seed(0)
data_cfg = load_config(REPO/"configs/data/paths.yaml", REPO/"configs/data/default.yaml")
cache_root = REPO / data_cfg["cache_root"]
sr = data_cfg["sample_rate"]

def per_file_metrics(model_cfg_name, ckpt):
    mcfg = load_yaml(REPO/f"configs/model/{model_cfg_name}.yaml")
    chunk = mcfg["frontend"]["chunk_samples"]; hop_s = chunk/sr
    ds = VADDataset(REPO/"data_cache/manifests/test_ten.jsonl", cache_root, sr, hop_s)
    model = build_model(mcfg["architecture"], mcfg)
    load_checkpoint(ckpt, model, map_location="cpu")
    model.eval()
    out = []
    for i in range(len(ds)):
        item = ds[i]
        wav = item["waveform"].unsqueeze(0)
        labels = item["labels"].numpy()
        usable = (wav.shape[1]//chunk)*chunk
        with torch.no_grad():
            probs = model.forward_full(wav[:, :usable]).numpy()[0]
        n = min(len(probs), len(labels))
        p, y = probs[:n], labels[:n]
        # AUROC (rank-based)
        pos, neg = p[y==1], p[y==0]
        if len(pos)==0 or len(neg)==0:
            auroc = np.nan
        else:
            order = np.argsort(np.concatenate([pos,neg]), kind="mergesort")
            ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order)+1)
            # handle ties via average ranks
            allp = np.concatenate([pos,neg])
            sidx = np.argsort(allp, kind="mergesort"); sorted_p = allp[sidx]
            avr = np.empty(len(allp)); i0=0
            while i0 < len(sorted_p):
                j = i0
                while j+1 < len(sorted_p) and sorted_p[j+1]==sorted_p[i0]: j+=1
                avr[sidx[i0:j+1]] = (i0+j)/2 + 1
                i0 = j+1
            auroc = (avr[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
        pred = (p > 0.5).astype(int)
        tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
        f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else np.nan
        out.append({"id": ds.records[i]["id"], "auroc": float(auroc), "f1": float(f1), "n": n})
    return out

crnn = per_file_metrics("crnn_v1", str(REPO/"runs/v1/crnn_v1_full/last.pt"))
tcn  = per_file_metrics("tcn_v1",  str(REPO/"runs/v1/tcn_v1_full/last.pt"))

a1 = np.array([r["auroc"] for r in crnn]); a2 = np.array([r["auroc"] for r in tcn])
f1_1 = np.array([r["f1"] for r in crnn]); f1_2 = np.array([r["f1"] for r in tcn])
d = a1 - a2; df = f1_1 - f1_2
def s(x): return float(np.std(x, ddof=1))
print(json.dumps({
  "n_files": len(crnn),
  "auroc_mean": {"crnn": float(np.mean(a1)), "tcn": float(np.mean(a2))},
  "sigma_abs_auroc": {"crnn": s(a1), "tcn": s(a2)},
  "sigma_paired_auroc": s(d),
  "ratio_abs_over_paired_auroc": s(a1)/s(d),
  "sigma_abs_f1": {"crnn": s(f1_1), "tcn": s(f1_2)},
  "sigma_paired_f1": s(df),
  "ratio_abs_over_paired_f1": s(f1_1)/s(df),
  "mde80_paired_auroc_n30": 2.8*s(d)/np.sqrt(30),
  "mde80_paired_auroc_n278_if_sigma_transfers": 2.8*s(d)/np.sqrt(278),
  "n_needed_mde_0.0082": (2.8*s(d)/0.0082)**2,
  "n_needed_mde_0.010": (2.8*s(d)/0.010)**2,
}, indent=1))
np.savez(str(Path(__file__).parent/"m1_per_file.npz"),
         ids=[r["id"] for r in crnn], crnn_auroc=a1, tcn_auroc=a2, crnn_f1=f1_1, tcn_f1=f1_2)
