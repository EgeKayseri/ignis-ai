"""
Training — U-Net on the memmap cache
Eğitim — memmap önbelleği üzerinde U-Net
========================================

AdamW + CosineAnnealingWarmRestarts, bfloat16 autocast, gradient clipping,
checkpointing on best VALIDATION AUC-PR.

Why AUC-PR is the checkpoint criterion / Neden AUC-PR
-----------------------------------------------------
Validation loss is not comparable across a weighted loss, and accuracy is
meaningless at 0.2686 % prevalence — predicting "no fire" everywhere scores
99.73 %. AUC-PR is threshold-free and its random baseline equals the prevalence
itself, so it measures ranking skill on the rare class, which is the only thing
this model is for.
Doğruluk %0.27 pozitif oranında anlamsızdır; AUC-PR eşikten bağımsızdır ve
rastgele tabanı pozitif oranın kendisidir.

Usage / kullanım
----------------
    python src/train.py
    python src/train.py --version v3 --epochs 60 --loss focal_tversky
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    MODELS_DIR,
    REPORTS_DIR,
    SPREAD_BATCH_SIZE,
    SPREAD_COSINE_T0,
    SPREAD_COSINE_TMULT,
    SPREAD_EARLY_STOP_PATIENCE,
    SPREAD_EPOCHS,
    SPREAD_GRAD_CLIP,
    SPREAD_LEARNING_RATE,
    SPREAD_LOSS,
    SPREAD_MIN_LR,
    SPREAD_MODEL_FILE,
    SPREAD_NORM_STATS_FILE,
    SPREAD_VERSION,
    SPREAD_WEIGHT_DECAY,
    ensure_directories_exist,
)
from dataset import SpreadDataset, year_split  # noqa: E402
from device import autocast, describe, get_device, set_seed  # noqa: E402
from features import compute_norm_stats, save_norm_stats  # noqa: E402
from losses import build_loss  # noqa: E402
from model import UNet, count_parameters  # noqa: E402
from tfrecord_to_npy import load_cache  # noqa: E402


def average_precision(y_true, y_score):
    """AUC-PR without sklearn, so training never hard-depends on it.
    sklearn olmadan AUC-PR."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n_pos = float((y_true > 0.5).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = (y_true[order] > 0.5).astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    # AP = sum over thresholds of (R_k - R_{k-1}) * P_k
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def _ensure_norm_stats(version, force=False):
    """Compute and persist normalisation stats from the TRAINING split only.
    İstatistikleri YALNIZCA eğitim bölmesinden hesapla ve kaydet."""
    path = Path(SPREAD_NORM_STATS_FILE)
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") == version:
            print(f"  norm stats      : reusing {path.name}")
            return payload["stats"]

    array, meta, man = load_cache(version)
    splits = year_split(meta["date"])
    train_idx = splits["train"]
    if len(train_idx) == 0:
        raise RuntimeError("Training split is empty. / Eğitim bölmesi boş.")
    vi = man["bands"].index("valid") if "valid" in man["bands"] else None
    print(f"  norm stats      : computing from {len(train_idx)} training patches "
          f"(training split ONLY)")
    stats = compute_norm_stats(array, man["bands"], train_idx, valid_band_idx=vi)
    save_norm_stats(stats, path, extra={
        "version": version,
        "n_train_patches": int(len(train_idx)),
        "note": "Computed from the training split only. Taking these from "
                "validation or test would be leakage.",
    })
    print(f"  norm stats      : written to {path}")
    return stats


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, max_batches=None):
    """Loss and AUC-PR over a loader. / Bir yükleyici üzerinde kayıp ve AUC-PR."""
    model.eval()
    losses, ys, ps = [], [], []
    for bi, (x, y, v) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y, v = x.to(device), y.to(device), v.to(device)
        with autocast(device):
            logits = model(x)
            loss = criterion(logits.float(), y, v)
        losses.append(float(loss))
        prob = torch.sigmoid(logits.float())
        keep = v > 0.5
        ys.append(y[keep].detach().cpu().numpy())
        ps.append(prob[keep].detach().cpu().numpy())
    if not ys:
        return float("nan"), float("nan")
    return float(np.mean(losses)), average_precision(
        np.concatenate(ys), np.concatenate(ps))


def train(version=SPREAD_VERSION, epochs=SPREAD_EPOCHS, batch_size=SPREAD_BATCH_SIZE,
          lr=SPREAD_LEARNING_RATE, loss_name=SPREAD_LOSS, workers=4,
          recompute_stats=False, cpu=False, val_batches=None):
    ensure_directories_exist()
    set_seed(42)

    print("IGNIS — training / eğitim")
    print("=" * 62)
    device = get_device(prefer_gpu=not cpu)
    stats = _ensure_norm_stats(version, force=recompute_stats)

    train_ds = SpreadDataset("train", version=version, stats=stats, augment=True)
    val_ds = SpreadDataset("val", version=version, stats=stats, augment=False)

    print(f"  version         : {version}")
    print(f"  train patches   : {len(train_ds)}")
    print(f"  val patches     : {len(val_ds)}")
    print(f"  channels        : {train_ds.n_channels}")
    print(f"  crop            : {train_ds.crop}x{train_ds.crop}")
    print(f"  target mode     : {train_ds.target_mode}")
    print(f"  loss            : {loss_name}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin, drop_last=True,
                              persistent_workers=bool(workers))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin,
                            persistent_workers=bool(workers))

    model = UNet(in_channels=train_ds.n_channels).to(device)
    trainable, total = count_parameters(model)
    print(f"  parameters      : {total:,}")
    print("=" * 62)

    criterion = build_loss(loss_name)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=SPREAD_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=SPREAD_COSINE_T0, T_mult=SPREAD_COSINE_TMULT,
        eta_min=SPREAD_MIN_LR)

    history = {"train_loss": [], "val_loss": [], "val_ap": [], "lr": []}
    best_ap, best_epoch, since_improved = -1.0, -1, 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running, nb = 0.0, 0
        for x, y, v in train_loader:
            x, y, v = x.to(device, non_blocking=True), y.to(device, non_blocking=True), \
                v.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(x)
                # Loss in float32: bfloat16 has only 7 mantissa bits and the
                # reductions here sum over ~33k pixels.
                loss = criterion(logits.float(), y, v)
            loss.backward()
            # No GradScaler: bfloat16 has float32's exponent range and does not
            # underflow, so scaling would add complexity for no benefit.
            torch.nn.utils.clip_grad_norm_(model.parameters(), SPREAD_GRAD_CLIP)
            optimiser.step()
            running += float(loss)
            nb += 1
        scheduler.step()

        train_loss = running / max(nb, 1)
        val_loss, val_ap = evaluate_split(model, val_loader, criterion, device,
                                          max_batches=val_batches)
        current_lr = optimiser.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ap"].append(val_ap)
        history["lr"].append(current_lr)

        marker = ""
        if np.isfinite(val_ap) and val_ap > best_ap:
            best_ap, best_epoch, since_improved = val_ap, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "in_channels": train_ds.n_channels,
                "channel_names": train_ds.channel_names,
                "version": version,
                "epoch": epoch,
                "val_ap": val_ap,
                "loss": loss_name,
                "target_mode": train_ds.target_mode,
                "crop": train_ds.crop,
            }, SPREAD_MODEL_FILE)
            marker = "  <- best, saved"
        else:
            since_improved += 1

        print(f"  epoch {epoch + 1:3d}/{epochs}  "
              f"train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val AUC-PR {val_ap:.4f}  lr {current_lr:.2e}{marker}")

        if since_improved >= SPREAD_EARLY_STOP_PATIENCE:
            print(f"  early stop: no improvement for {since_improved} epochs")
            break

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (Path(REPORTS_DIR) / "spread_history.json").write_text(
        json.dumps({**history, "best_val_ap": best_ap, "best_epoch": best_epoch,
                    "environment": describe()}, indent=2), encoding="utf-8")

    print("=" * 62)
    print(f"  best val AUC-PR : {best_ap:.4f} (epoch {best_epoch + 1})")
    print(f"  checkpoint      : {SPREAD_MODEL_FILE}")
    print(f"  elapsed         : {(time.time() - t0) / 60:.1f} min")
    print()
    print("  Validation AUC-PR is a model-selection number, NOT a result.")
    print("  Report only src/evaluate.py output on the test split.")
    print("  Doğrulama AUC-PR'ı sonuç DEĞİLDİR; yalnızca model seçimi içindir.")
    return best_ap


def main():
    ap = argparse.ArgumentParser(description="Train the fire-spread U-Net")
    ap.add_argument("--version", default=SPREAD_VERSION, choices=["v1", "v2", "v3"])
    ap.add_argument("--epochs", type=int, default=SPREAD_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=SPREAD_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=SPREAD_LEARNING_RATE)
    ap.add_argument("--loss", default=SPREAD_LOSS,
                    choices=["bce_dice", "focal_tversky", "bce", "dice"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--recompute-stats", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(version=args.version, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, loss_name=args.loss, workers=args.workers,
          recompute_stats=args.recompute_stats, cpu=args.cpu)


if __name__ == "__main__":
    main()
