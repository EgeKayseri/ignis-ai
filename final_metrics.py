#!/usr/bin/env python3
"""Full metric set for the final model, from ONE test evaluation.

No new tuning. The threshold is the one frozen on validation (2023) inside
final_run.py; this script only reports more numbers from the same predictions.

Accuracy is printed next to the "predict nothing" accuracy on purpose. At about
1 % prevalence, an accuracy figure on its own says nothing about skill, and the
project has a standing rule against quoting one.

Doğruluk yüzdesi, "hiçbir şey tahmin etme" doğruluğunun yanında basılır: %1
pozitif oranında tek başına doğruluk hiçbir beceri ölçmez.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

import experiments as X  # noqa: E402
from dataset import SpreadDataset  # noqa: E402
from device import get_device  # noqa: E402
from features import compute_norm_stats  # noqa: E402
from model import UNet  # noqa: E402
from tfrecord_to_npy import load_cache  # noqa: E402

TRAIN_YEARS = (2019, 2020, 2021, 2022, 2024)
TEST_YEARS = (2025, 2026)


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    f = 2 * p * r / max(p + r, 1e-9)
    return p, r, f, tp / max(tp + fp + fn, 1.0)


def main():
    ck = torch.load(BASE / "models" / "spread_unet_2024train.pt",
                    map_location="cpu", weights_only=False)
    thr, cfg = ck["thr"], ck["cfg"]

    array, meta, man = load_cache("v5")
    bands = man["bands"]
    years = np.asarray(meta["date"]) // 10000
    tr = np.nonzero(np.isin(years, TRAIN_YEARS))[0]
    te = np.nonzero(np.isin(years, TEST_YEARS))[0]

    device = get_device()
    ns = compute_norm_stats(array, bands, tr, valid_band_idx=bands.index("valid"))
    ds = SpreadDataset("test", version="v5", stats=ns, augment=False, indices=te)
    model = UNet(in_channels=ds.n_channels, dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    prob, tgt, val = X.infer(model, dl, device)

    keep = val > 0.5
    t = (tgt > 0.5)[keep]
    p = (prob >= thr)[keep]
    tp = float((p & t).sum()); fp = float((p & ~t).sum())
    fn = float((~p & t).sum()); tn = float((~p & ~t).sum())
    n = tp + tn + fp + fn
    prec, rec, f1, iou = prf(tp, fp, fn)
    acc = (tp + tn) / max(n, 1.0)
    prev = (tp + fn) / max(n, 1.0)

    i_f = bands.index("fire")
    today = (np.asarray(array[np.sort(te)][:, i_f, 16:48, 16:48]) > 0.5)[keep]
    pp, pr, pf, pi = prf(float((today & t).sum()), float((today & ~t).sum()),
                         float((~today & t).sum()))

    print("=" * 66)
    print("FINAL MODEL — TEST (2025+2026), threshold frozen on validation")
    print("=" * 66)
    print(f"  pixels scored         : {int(n):,}")
    print(f"  positive prevalence   : {100 * prev:.4f} %")
    print()
    print(f"  {'':<22}{'MODEL':>10}{'persistence':>13}")
    print(f"  {'precision':<22}{prec:>10.4f}{pp:>13.4f}")
    print(f"  {'recall':<22}{rec:>10.4f}{pr:>13.4f}")
    print(f"  {'F1':<22}{f1:>10.4f}{pf:>13.4f}")
    print(f"  {'IoU':<22}{iou:>10.4f}{pi:>13.4f}")
    print()
    print(f"  pixel accuracy        : {100 * acc:.4f} %")
    print(f"  'predict nothing'     : {100 * (1 - prev):.4f} %  <- needs no model")
    print("=" * 66)

    (BASE / "outputs" / "experiments" / "final_metrics.json").write_text(json.dumps({
        "prevalence": prev, "accuracy": acc, "null_accuracy": 1 - prev,
        "model": {"precision": prec, "recall": rec, "f1": f1, "iou": iou},
        "persistence": {"precision": pp, "recall": pr, "f1": pf, "iou": pi},
    }, indent=2))


if __name__ == "__main__":
    main()
