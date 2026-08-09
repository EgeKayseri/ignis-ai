#!/usr/bin/env python3
"""Hyperparameter experiments — VALIDATION ONLY.

Her şey doğrulama bölmesinde (2024) ölçülür. Test bölmesi (2025-2026) bu
dosyada hiç açılmaz; en iyi ayar seçildikten SONRA, tek seferlik olarak
src/evaluate.py ile ölçülür. Aksi hâlde test bölmesi bir ayar setine karşı
seçilmiş olur ve raporlanan sayı artık dürüst olmaz.

Everything is scored on validation. The test split is never opened here.

Diagnosis this is built to answer:
  1. best val AUC-PR arrived at epoch 2 -> is this over-fitting, and does
     dropout / weight decay / a lower LR move it later?
  2. recall 0.56 vs precision 0.11 -> is SPREAD_POS_WEIGHT = 12.0 too high?
  3. is 2025 an outlier year or is the model simply fragile? (--loyo)

Usage:
    python experiments.py --sweep            # ayar taraması
    python experiments.py --loyo             # yıl-dışarıda-bırak çapraz doğrulama
    python experiments.py --time             # tek epoch süresini ölç
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from config import SPREAD_CACHE_DIR  # noqa: E402
from dataset import SpreadDataset, year_split  # noqa: E402
from device import autocast, get_device, set_seed  # noqa: E402
from features import compute_norm_stats  # noqa: E402
from losses import build_loss  # noqa: E402
from model import UNet  # noqa: E402
from tfrecord_to_npy import load_cache  # noqa: E402
from train import average_precision  # noqa: E402

VERSION = "v5"
OUT = BASE / "outputs" / "experiments"
CROP = slice(16, 48)


# ---------------------------------------------------------------- helpers
def norm_stats_for(train_idx, bands, array):
    """Statistics from THESE training indices only. Sızıntıyı önler."""
    vi = bands.index("valid") if "valid" in bands else None
    return compute_norm_stats(array, bands, train_idx, valid_band_idx=vi)


def scores(pred, true):
    inter = float((pred & true).sum())
    union = float((pred | true).sum())
    f1 = 2 * inter / max(float(pred.sum() + true.sum()), 1.0)
    return inter / max(union, 1.0), f1


def best_threshold(prob, true, valid, steps=60):
    """F1-optimal threshold on the split given. Only ever called on validation."""
    keep = valid > 0.5
    p, t = prob[keep], true[keep] > 0.5
    lo, hi = float(np.percentile(p, 50)), float(np.percentile(p, 99.9))
    best = (0.5, -1.0)
    for thr in np.linspace(lo, hi, steps):
        _, f1 = scores(p >= thr, t)
        if f1 > best[1]:
            best = (float(thr), f1)
    return best


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    P, T, V = [], [], []
    for x, y, v in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device):
            logits = model(x)
        P.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
        T.append(y.numpy()[:, 0])
        V.append(v.numpy()[:, 0])
    return np.concatenate(P), np.concatenate(T), np.concatenate(V)


def persistence_on(array, idx, bands, prob_shape_true, valid):
    """"Tomorrow = today" scored on exactly the same pixels as the model.

    idx MUST be read in the order SpreadDataset iterates it, which is the order
    it was given -- not sorted. Sorting here was harmless while every caller
    passed np.nonzero() output (already ascending), and silently misaligned the
    baseline against the target the moment a shuffled index array arrived.

    idx, SpreadDataset'in gezdiği sırayla okunmalıdır. Sıralamak, karışık indeks
    gelen ilk anda temel çizgiyi hedefle yanlış eşleştirir.
    """
    i_fire = bands.index("fire")
    today = np.asarray(array[idx][:, i_fire, CROP, CROP]) > 0.5
    keep = valid > 0.5
    return scores(today & keep, prob_shape_true & keep)


# ---------------------------------------------------------------- one run
def run(cfg, array, meta, bands, train_idx, val_idx, device, quiet=False):
    set_seed(cfg.get("seed", 42))
    stats = norm_stats_for(train_idx, bands, array)

    tr = SpreadDataset("train", version=VERSION, stats=stats,
                       augment=cfg["augment"], indices=train_idx)
    va = SpreadDataset("val", version=VERSION, stats=stats,
                       augment=False, indices=val_idx)

    tl = DataLoader(tr, batch_size=cfg["batch"], shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=cfg["batch"], shuffle=False, num_workers=8,
                    pin_memory=True, persistent_workers=True)

    model = UNet(in_channels=tr.n_channels, dropout=cfg["dropout"]).to(device)
    # FocalTversky weights false negatives through beta, not through a
    # pos_weight argument, so passing one is a TypeError rather than a no-op.
    # FocalTversky pozitifi beta ile ağırlıklar; pos_weight kabul etmez.
    loss_kw = {} if cfg["loss"].startswith("focal") else {"pos_weight": cfg["pos_weight"]}
    crit = build_loss(cfg["loss"], **loss_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.02)

    best = {"ap": -1.0, "epoch": -1, "state": None}
    since = 0
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        for x, y, v in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(x)
            loss = crit(logits.float(), y, v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        prob, tgt, val = infer(model, vl, device)
        keep = val > 0.5
        ap = average_precision(tgt[keep] > 0.5, prob[keep])
        mark = ""
        if ap > best["ap"]:
            best = {"ap": float(ap), "epoch": ep,
                    "state": {k: t.detach().cpu().clone()
                              for k, t in model.state_dict().items()}}
            since = 0
            mark = "  <- best"
        else:
            since += 1
        if not quiet:
            print(f"    epoch {ep + 1:>2}/{cfg['epochs']}  val AP {ap:.4f}  "
                  f"{time.time() - t0:5.1f}s{mark}", flush=True)
        if since >= cfg["patience"]:
            break

    model.load_state_dict(best["state"])
    prob, tgt, val = infer(model, vl, device)
    thr, f1 = best_threshold(prob, tgt, val)
    keep = val > 0.5
    iou, f1 = scores((prob >= thr) & keep, (tgt > 0.5) & keep)
    p_iou, p_f1 = persistence_on(array, val_idx, bands, tgt > 0.5, val)
    return {"ap": best["ap"], "epoch": best["epoch"] + 1, "thr": thr,
            "iou": iou, "f1": f1, "p_iou": p_iou, "p_f1": p_f1,
            "beats": bool(iou > p_iou), "state": best["state"],
            "stats": stats, "channels": tr.n_channels}


# ---------------------------------------------------------------- modes
BASELINE = dict(pos_weight=12.0, dropout=0.2, lr=1e-3, wd=1e-4,
                loss="bce_dice", augment=True, batch=32, epochs=24, patience=6)


def sweep(array, meta, bands, device, start=0):
    splits = year_split(np.asarray(meta["date"]))
    tr, va = splits["train"], splits["val"]

    trials = [
        ("mevcut ayar (referans)", {}),
        ("pos_weight 4", dict(pos_weight=4.0)),
        ("pos_weight 1", dict(pos_weight=1.0)),
        ("dropout 0.4", dict(dropout=0.4)),
        ("lr 3e-4", dict(lr=3e-4)),
        ("wd 1e-3 + dropout 0.3", dict(wd=1e-3, dropout=0.3)),
        ("focal_tversky", dict(loss="focal_tversky")),
        ("augment kapalı (kontrol)", dict(augment=False)),
    ]
    rows = []
    for name, over in trials[start:]:
        cfg = {**BASELINE, **over}
        print(f"\n=== {name} ===", flush=True)
        r = run(cfg, array, meta, bands, tr, va, device)
        r["name"], r["cfg"] = name, over
        rows.append(r)
        print(f"  -> val IoU {r['iou']:.4f} (kalıcılık {r['p_iou']:.4f}) "
              f"AP {r['ap']:.4f} @ epoch {r['epoch']}  "
              f"{'GEÇTI' if r['beats'] else 'geçemedi'}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sweep_{start}.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'ayar':<26}{'val IoU':>9}{'val F1':>9}{'AP':>8}{'en iyi ep':>11}  sonuç")
    print("-" * 76)
    for r in sorted(rows, key=lambda x: -x["iou"]):
        print(f"{r['name']:<26}{r['iou']:>9.4f}{r['f1']:>9.4f}{r['ap']:>8.4f}"
              f"{r['epoch']:>11}  {'GEÇTI' if r['beats'] else '-'}")
    print(f"\nkalıcılık (2024): IoU {rows[0]['p_iou']:.4f}  F1 {rows[0]['p_f1']:.4f}")


def loyo(array, meta, bands, device, cfg_over=None):
    """Leave-one-year-out. Is 2025 an outlier, or is the model fragile?

    CAVEAT / UYARI: these numbers are OPTIMISTIC. Both the stopping epoch and
    the decision threshold are chosen on the held-out year itself, so this
    measures "how well could the model do on year X if we could tune for
    year X" -- an upper bound, not clean generalisation. Use it to compare
    years against each other, never as a headline score.

    Bu sayılar İYİMSERDİR: hem durma epoch'u hem de eşik, dışarıda bırakılan
    yılın kendisinde seçiliyor. Yıllar arası karşılaştırma için kullanılır,
    başlık skor olarak asla.
    """
    years = np.asarray(meta["date"]) // 10000
    cfg = {**BASELINE, **(cfg_over or {})}
    rows = []
    for held in (2021, 2022, 2023, 2024, 2025):
        # Train on every full season except the held-out one. 2026 is excluded
        # because it stops on 25 July and would bias whatever fold it joined.
        # NOTE: a fold may train on years LATER than the one it is scored on.
        # That is deliberate — the question here is year-to-year variance, not
        # operational forecasting, and this is never reported as a test result.
        tr = np.nonzero((years != held) & (years != 2026) & (years <= 2025))[0]
        va = np.nonzero(years == held)[0]
        if len(va) == 0 or len(tr) == 0:
            continue
        print(f"\n=== dışarıda: {held}  (train {len(tr)}, val {len(va)}) ===", flush=True)
        r = run(cfg, array, meta, bands, tr, va, device, quiet=True)
        r["held"] = int(held)
        rows.append(r)
        print(f"  model IoU {r['iou']:.4f}   kalıcılık {r['p_iou']:.4f}   "
              f"{'MODEL KAZANDI' if r['beats'] else 'kalıcılık kazandı'}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "loyo.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{'dışarıdaki yıl':<16}{'model IoU':>11}{'kalıcılık':>12}  sonuç")
    print("-" * 52)
    for r in rows:
        print(f"{r['held']:<16}{r['iou']:>11.4f}{r['p_iou']:>12.4f}  "
              f"{'model' if r['beats'] else 'kalıcılık'}")
    ious = [r["iou"] for r in rows]
    print(f"\nmodel IoU ortalama {np.mean(ious):.4f} ± {np.std(ious):.4f} "
          f"(min {min(ious):.4f}, max {max(ious):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--from-trial", type=int, default=0,
                    help="skip the first N trials (resume a crashed sweep)")
    ap.add_argument("--loyo", action="store_true")
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()

    array, meta, man = load_cache(VERSION)
    bands = man["bands"]
    device = get_device()

    if a.time:
        splits = year_split(np.asarray(meta["date"]))
        cfg = {**BASELINE, "epochs": 2, "patience": 99}
        t0 = time.time()
        run(cfg, array, meta, bands, splits["train"], splits["val"], device)
        print(f"\n2 epoch toplam {time.time() - t0:.1f}s")
    elif a.sweep:
        sweep(array, meta, bands, device, start=a.from_trial)
    elif a.loyo:
        loyo(array, meta, bands, device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
