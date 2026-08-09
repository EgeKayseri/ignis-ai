#!/usr/bin/env python3
"""Diagnostic: model vs persistence, per test year, at the VALIDATION threshold.

This is diagnosis, not tuning. The threshold stays exactly where validation put
it; the test split is only being split by year to see where the loss comes from.
Ayar değil teşhis: eşik doğrulamada bulunduğu yerde kalıyor.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from tfrecord_to_npy import load_cache
from dataset import SpreadDataset
from features import load_norm_stats
from device import get_device
import evaluate as E

THR = 0.2333   # calibrated on validation, held fixed


def scores(pred, true):
    inter = (pred & true).sum()
    union = (pred | true).sum()
    f1 = 2 * inter / max(pred.sum() + true.sum(), 1)
    return inter / max(union, 1), f1


def main():
    arr, meta, man = load_cache("v5")
    years = np.asarray(meta["date"]) // 10000
    stats = load_norm_stats("models/norm_stats.json", version="v5")
    dev = get_device()
    model, _ = E._load_model(dev, "models/spread_unet.pt")

    i_fire = man["bands"].index("fire")
    sl = slice(16, 48)

    print(f"\n{'year':<7}{'n':>7}{'model IoU':>11}{'persist IoU':>13}"
          f"{'model F1':>10}{'persist F1':>12}   verdict")
    print("-" * 74)
    for y in (2025, 2026):
        idx = np.nonzero(years == y)[0]
        ds = SpreadDataset("test", version="v5", stats=stats,
                           augment=False, indices=idx)
        prob, tgt, val = E.predict_split(model, ds, dev)
        keep = val > 0.5
        true = (tgt > 0.5) & keep
        pred = (prob >= THR) & keep
        today = (np.asarray(arr[np.sort(idx)][:, i_fire, sl, sl]) > 0.5) & keep

        mi, mf = scores(pred, true)
        pi, pf = scores(today, true)
        verdict = "MODEL WINS" if mi > pi else "persistence wins"
        print(f"{y:<7}{len(idx):>7}{mi:>11.4f}{pi:>13.4f}{mf:>10.4f}"
              f"{pf:>12.4f}   {verdict}")


if __name__ == "__main__":
    main()
