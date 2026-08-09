#!/usr/bin/env python3
"""How big is run-to-run noise? VALIDATION ONLY.

The final run beat persistence on test by 0.0004 in IoU. That is only a result
if it is larger than the spread produced by changing nothing but the random
seed. Measured on validation so the test split is not touched again.

Test bölmesindeki 0.0004'lük fark, yalnızca tohum değiştirince oluşan
dalgalanmadan büyükse anlamlıdır. Doğrulamada ölçülür.
"""
import sys, json
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
import experiments as X
from tfrecord_to_npy import load_cache
from device import get_device

TRAIN_YEARS = (2019, 2020, 2021, 2022, 2024)
VAL_YEARS = (2023,)

def main():
    array, meta, man = load_cache("v5")
    bands = man["bands"]
    years = np.asarray(meta["date"]) // 10000
    dev = get_device()
    tr = np.nonzero(np.isin(years, TRAIN_YEARS))[0]
    va = np.nonzero(np.isin(years, VAL_YEARS))[0]

    rows = []
    for seed in (42, 7, 1234):
        cfg = {**X.BASELINE, "wd": 1e-3, "dropout": 0.3, "seed": seed}
        print(f"\n=== seed {seed} ===", flush=True)
        r = X.run(cfg, array, meta, bands, tr, va, dev, quiet=True)
        rows.append({"seed": seed, "iou": r["iou"], "f1": r["f1"],
                     "ap": r["ap"], "thr": r["thr"], "epoch": r["epoch"]})
        print(f"  val IoU {r['iou']:.4f}  F1 {r['f1']:.4f}  AP {r['ap']:.4f} "
              f"@ ep {r['epoch']}  thr {r['thr']:.4f}", flush=True)

    ious = np.array([r["iou"] for r in rows])
    spread = ious.max() - ious.min()
    print(f"\nval IoU: {ious.round(4).tolist()}")
    print(f"ortalama {ious.mean():.4f}  std {ious.std():.4f}  aralık {spread:.4f}")
    print(f"\ntestteki fark 0.0004 idi.")
    print("SONUC: " + ("fark GURULTUNUN ICINDE - beraberlik, galibiyet degil"
                       if spread > 0.0004 else
                       "fark gurultuden buyuk - anlamli olabilir"))
    (BASE/"outputs"/"experiments"/"seed_noise.json").write_text(
        json.dumps({"runs": rows, "spread": float(spread)}, indent=2))

if __name__ == "__main__":
    main()
