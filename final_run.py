#!/usr/bin/env python3
"""Decisive run: does training on the most recent year fix the 2025 loss?

THE QUESTION
------------
Leave-one-year-out scored 2025 at IoU 0.1927 and beat persistence, while the
real test protocol scored the same year at 0.1270 and lost. Two things differ,
and they have to be separated:

  (a) LOYO trained on 2019-2024, the real split trains on 2019-2023 only,
      because the configured split reserves 2024 for validation. The model
      never sees the year immediately before the test period.
  (b) LOYO picked both the stopping epoch and the threshold ON the held-out
      year, which is optimistic and cannot be reproduced operationally.

This run isolates (a) by removing (b):

  train  2019, 2020, 2021, 2022, 2024      <- 2024 moved INTO training
  val    2023                              <- epoch and threshold come from here
  test   2025, 2026                        <- opened exactly once, at the end

The test split is unchanged from the original protocol, so the headline number
stays comparable. The threshold never touches test.

HONEST CAVEAT
-------------
The idea of moving 2024 into training was prompted by a LOYO fold that included
2025. That is a mild dependence on the test period, and it is recorded here
rather than hidden. The change is independently defensible -- using all
non-test data for training is standard -- but the reader should know.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

import experiments as X  # noqa: E402
from dataset import SpreadDataset  # noqa: E402
from device import get_device  # noqa: E402
from tfrecord_to_npy import load_cache  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from model import UNet  # noqa: E402

TRAIN_YEARS = (2019, 2020, 2021, 2022, 2024)
VAL_YEARS = (2023,)
TEST_YEARS = (2025, 2026)

# Sweep winner, selected on validation only.
CFG = {**X.BASELINE, "wd": 1e-3, "dropout": 0.3}


def main():
    array, meta, man = load_cache("v5")
    bands = man["bands"]
    years = np.asarray(meta["date"]) // 10000
    device = get_device()

    idx = {n: np.nonzero(np.isin(years, ys))[0]
           for n, ys in (("train", TRAIN_YEARS), ("val", VAL_YEARS),
                         ("test", TEST_YEARS))}
    print(f"train {len(idx['train'])}  val {len(idx['val'])}  test {len(idx['test'])}")
    print(f"config: {CFG}\n", flush=True)

    t0 = time.time()
    r = X.run(CFG, array, meta, bands, idx["train"], idx["val"], device)
    print(f"\nvalidation (2023): IoU {r['iou']:.4f}  persistence {r['p_iou']:.4f}  "
          f"AP {r['ap']:.4f} @ epoch {r['epoch']}  thr {r['thr']:.4f}")
    print(f"training took {(time.time() - t0) / 60:.1f} min\n", flush=True)

    # ---- test, opened once, threshold frozen from validation ----------
    model = UNet(in_channels=r["channels"], dropout=CFG["dropout"]).to(device)
    model.load_state_dict(r["state"])
    ds = SpreadDataset("test", version="v5", stats=r["stats"],
                       augment=False, indices=idx["test"])
    dl = DataLoader(ds, batch_size=CFG["batch"], shuffle=False, num_workers=8,
                    pin_memory=True)
    prob, tgt, val = X.infer(model, dl, device)

    keep = val > 0.5
    true = (tgt > 0.5) & keep
    pred = (prob >= r["thr"]) & keep          # threshold from 2023, NOT from test
    m_iou, m_f1 = X.scores(pred, true)
    p_iou, p_f1 = X.persistence_on(array, idx["test"], bands, tgt > 0.5, val)

    print("=" * 62)
    print(f"TEST (2025+2026), threshold {r['thr']:.4f} fixed on validation")
    print("=" * 62)
    print(f"  {'':<14}{'IoU':>9}{'F1':>9}")
    print(f"  {'MODEL':<14}{m_iou:>9.4f}{m_f1:>9.4f}")
    print(f"  {'persistence':<14}{p_iou:>9.4f}{p_f1:>9.4f}")
    print("-" * 62)
    print("  " + ("MODEL BEATS PERSISTENCE" if m_iou > p_iou
                  else "model loses to persistence"))
    print(f"  reference: previous protocol scored {0.0974:.4f} vs {0.1178:.4f}")

    # per-year, same frozen threshold
    print("\n  per year, same frozen threshold:")
    off = 0
    for y in TEST_YEARS:
        n = int((years[np.sort(idx["test"])] == y).sum())
        sl = slice(off, off + n)
        off += n
        k = keep[sl]
        yi, yf = X.scores(pred[sl] & k, true[sl] & k)
        pi, pf = X.persistence_on(array, np.nonzero(years == y)[0], bands,
                                  tgt[sl] > 0.5, val[sl])
        print(f"    {y}  model {yi:.4f}  persistence {pi:.4f}  "
              f"{'model' if yi > pi else 'persistence'}")

    out = BASE / "outputs" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_run.json").write_text(json.dumps({
        "train_years": list(TRAIN_YEARS), "val_years": list(VAL_YEARS),
        "test_years": list(TEST_YEARS), "cfg": {k: v for k, v in CFG.items()},
        "val": {k: r[k] for k in ("iou", "f1", "ap", "epoch", "thr", "p_iou")},
        "test": {"iou": m_iou, "f1": m_f1, "p_iou": p_iou, "p_f1": p_f1,
                 "beats": bool(m_iou > p_iou)},
    }, indent=2))
    torch.save({"model": r["state"], "cfg": CFG, "thr": r["thr"]},
               BASE / "models" / "spread_unet_2024train.pt")
    print(f"\nsaved -> {out / 'final_run.json'}")


if __name__ == "__main__":
    main()
