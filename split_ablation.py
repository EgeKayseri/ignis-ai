#!/usr/bin/env python3
"""Ablation: what does a random split do to the reported score?

NOT A RESULT. This run deliberately uses a leaking split so the size of the
leak can be quoted. The number it produces must never be reported as IGNIS
performance -- it exists to show what happens when the split is chosen badly.

BU BIR SONUC DEGILDIR. Kasitli olarak sizintili bir bolme kullanir; urettigi
sayi IGNIS performansi olarak asla raporlanamaz.

Why the random split leaks, measured before any training runs:
  random 70/15/15 -> 100.00 % of test patches have a same-day patch in train
  year split      ->   0.00 %

Patches cut from the same day come from the same fire: they overlap in space,
share the same weather, and are near-duplicates of one another. A random split
puts those duplicates on both sides of the boundary, so the model is scored on
what it has effectively memorised.

Published wildfire-spread work frequently reports ROC-AUC above 0.93 on random
splits. This ablation is how IGNIS can say, with a measured number, that part
of that gap is methodology rather than model quality.
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

import experiments as X  # noqa: E402
from dataset import SpreadDataset  # noqa: E402
from device import get_device  # noqa: E402
from model import UNet  # noqa: E402
from tfrecord_to_npy import load_cache  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

CFG = {**X.BASELINE, "wd": 1e-3, "dropout": 0.3}


def main():
    array, meta, man = load_cache("v5")
    bands = man["bands"]
    dates = np.asarray(meta["date"])
    n = len(dates)
    device = get_device()

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    ntr, nva = int(0.70 * n), int(0.15 * n)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

    leak = np.isin(dates[te], list(set(dates[tr].tolist())))
    print(f"random split — train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"test patches sharing a day with train: {leak.sum()} "
          f"({100 * leak.mean():.2f} %)\n", flush=True)

    r = X.run(CFG, array, meta, bands, tr, va, device, quiet=True)
    print(f"validation: IoU {r['iou']:.4f}  AP {r['ap']:.4f} @ epoch {r['epoch']}"
          f"  thr {r['thr']:.4f}", flush=True)

    model = UNet(in_channels=r["channels"], dropout=CFG["dropout"]).to(device)
    model.load_state_dict(r["state"])
    ds = SpreadDataset("test", version="v5", stats=r["stats"],
                       augment=False, indices=te)
    dl = DataLoader(ds, batch_size=CFG["batch"], shuffle=False, num_workers=8,
                    pin_memory=True)
    prob, tgt, val = X.infer(model, dl, device)
    keep = val > 0.5
    true = (tgt > 0.5) & keep
    m_iou, m_f1 = X.scores((prob >= r["thr"]) & keep, true)
    p_iou, p_f1 = X.persistence_on(array, te, bands, tgt > 0.5, val)

    print("=" * 66)
    print("RANDOM SPLIT (leaking) — NOT A RESULT / SONUC DEGIL")
    print("=" * 66)
    print(f"  {'':<14}{'IoU':>9}{'F1':>9}")
    print(f"  {'MODEL':<14}{m_iou:>9.4f}{m_f1:>9.4f}")
    print(f"  {'persistence':<14}{p_iou:>9.4f}{p_f1:>9.4f}")
    print("-" * 66)
    print(f"  year split, honest   : model 0.1182  persistence 0.1178")
    print(f"  inflation factor     : {m_iou / 0.1182:.2f}x on model IoU")

    (BASE / "outputs" / "experiments" / "split_ablation.json").write_text(
        json.dumps({"leak_pct": float(100 * leak.mean()),
                    "random": {"iou": m_iou, "f1": m_f1,
                               "p_iou": p_iou, "p_f1": p_f1},
                    "year_split_reference": {"iou": 0.1182, "p_iou": 0.1178}},
                   indent=2))


if __name__ == "__main__":
    main()
