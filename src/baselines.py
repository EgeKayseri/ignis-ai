"""
Baselines — the bar the model has to clear
Temel çizgiler — modelin aşması gereken eşik
============================================

Three deliberately trivial predictors. If the U-Net cannot beat these, it has
contributed nothing, however sophisticated it is.

Üç kasıtlı olarak basit tahminci. U-Net bunları geçemiyorsa, ne kadar karmaşık
olursa olsun hiçbir katkı sunmamıştır.

  persistence          tomorrow = today. One line of code, no parameters, no
                       training. On the v1 archive it scored IoU 0.0306 and
                       F1 0.0595 — and BEAT the trained network (0.0165 / 0.0324).
                       yarın = bugün.

  dilated persistence  today's mask grown by one pixel in every direction. A
                       fire usually spreads a little, so this is persistence
                       plus the crudest possible growth model.
                       bugünün maskesi her yöne bir piksel büyütülmüş.

  wind-directed        today's mask shifted one pixel downwind and unioned with
  growth               itself. The cheapest way of using the wind channels at
                       all. If the U-Net cannot beat THIS, it has not learned to
                       use wind.
                       bugünün maskesi rüzgâr yönünde bir piksel kaydırılmış.

Every baseline is scored on exactly the same pixels as the model — same split,
same crop, same `valid` mask — otherwise the comparison is meaningless.
Her temel çizgi, modelle AYNI piksellerde ölçülür.

Usage / kullanım
----------------
    python src/baselines.py                 # test split
    python src/baselines.py --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import REPORTS_DIR, SPREAD_VERSION, ensure_directories_exist  # noqa: E402
from dataset import SpreadDataset  # noqa: E402


# ----------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------
def persistence(today, **_):
    """Tomorrow = today. / Yarın = bugün."""
    return (today > 0.5).astype(np.float32)


def _dilate(mask, iterations=1):
    """Binary dilation with a 3x3 structuring element.
    3x3 yapı elemanıyla ikili genişletme.

    Uses scipy when available; the numpy fallback keeps this runnable without it.
    """
    out = (mask > 0.5).astype(np.float32)
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(out > 0.5, iterations=iterations).astype(np.float32)
    except ImportError:
        for _ in range(iterations):
            padded = np.pad(out, 1, mode="constant")
            acc = np.zeros_like(out)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    acc = np.maximum(
                        acc, padded[1 + dy:1 + dy + out.shape[0],
                                    1 + dx:1 + dx + out.shape[1]])
            out = acc
        return out


def dilated_persistence(today, iterations=1, **_):
    """Persistence grown by one pixel. / Bir piksel büyütülmüş kalıcılık."""
    return _dilate(today, iterations)


def wind_directed_growth(today, wind=(0.0, 0.0), **_):
    """Persistence unioned with a one-pixel downwind shift.
    Kalıcılık ile rüzgâr yönünde bir piksel kayma birleşimi.

    u is eastward (+x, i.e. +column), v is northward (+y). In array coordinates
    row 0 is the NORTH edge, so a northward wind moves fire towards SMALLER row
    indices — hence the minus sign on dy.
    u doğuya (+sütun), v kuzeye bakar. Dizide 0. satır KUZEY kenarıdır; bu yüzden
    dy'de eksi işareti vardır.
    """
    base = (today > 0.5).astype(np.float32)
    u, v = wind
    if not np.isfinite(u) or not np.isfinite(v) or (u == 0.0 and v == 0.0):
        return base
    mag = float(np.hypot(u, v))
    if mag < 1e-6:
        return base
    dx = int(np.sign(u)) if abs(u) / mag > 0.383 else 0     # 22.5 deg sectors
    dy = -int(np.sign(v)) if abs(v) / mag > 0.383 else 0
    if dx == 0 and dy == 0:
        return base
    shifted = np.zeros_like(base)
    h, w = base.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    yd0, yd1 = max(0, -dy), min(h, h - dy)
    xd0, xd1 = max(0, -dx), min(w, w - dx)
    shifted[ys0:ys1, xs0:xs1] = base[yd0:yd1, xd0:xd1]
    return np.maximum(base, shifted)


BASELINES = {
    "persistence": persistence,
    "dilated_persistence": dilated_persistence,
    "wind_directed_growth": wind_directed_growth,
}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def binary_scores(tp, fp, fn):
    """Precision / recall / F1 / IoU from counts. / Sayımlardan metrikler."""
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
          else float("nan"))
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def run_baselines(split="test", version=SPREAD_VERSION, names=None, limit=0):
    """Score every baseline on one split. / Bir bölmede tüm temel çizgiler."""
    ds = SpreadDataset(split, version=version, stats={}, augment=False)
    names = names or list(BASELINES)
    n = len(ds) if not limit else min(limit, len(ds))

    counts = {k: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for k in names}
    n_pos = n_pix = 0.0

    for i in range(n):
        today = ds.today_mask(i)
        target = (ds.target_mask(i) > 0.5)
        valid = ds.valid_mask(i) > 0.5
        wind = ds.wind_uv(i)

        n_pos += float((target & valid).sum())
        n_pix += float(valid.sum())

        for k in names:
            pred = BASELINES[k](today, wind=wind) > 0.5
            p, t = pred & valid, target & valid
            counts[k]["tp"] += float((p & t).sum())
            counts[k]["fp"] += float((p & ~t).sum())
            counts[k]["fn"] += float((~p & t).sum())

    results = {}
    for k in names:
        c = counts[k]
        results[k] = {**binary_scores(c["tp"], c["fp"], c["fn"]),
                      "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]}

    results["_meta"] = {
        "split": split, "version": version, "patches": n,
        "prevalence": n_pos / n_pix if n_pix else float("nan"),
        "crop": ds.crop, "target_mode": ds.target_mode,
    }
    return results


def print_results(results):
    meta = results["_meta"]
    print(f"IGNIS — baselines / temel çizgiler  ({meta['split']} split, "
          f"{meta['version']})")
    print("=" * 74)
    print(f"  patches         : {meta['patches']}")
    print(f"  crop            : {meta['crop']}x{meta['crop']}")
    print(f"  target mode     : {meta['target_mode']}")
    print(f"  positive rate   : {meta['prevalence'] * 100:.4f}%")
    print("=" * 74)
    print(f"  {'baseline':<24}{'precision':>11}{'recall':>10}{'F1':>10}{'IoU':>10}")
    print("  " + "-" * 65)
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"  {k:<24}{v['precision']:>11.4f}{v['recall']:>10.4f}"
              f"{v['f1']:>10.4f}{v['iou']:>10.4f}")
    print("=" * 74)
    print("  The model must beat the best of these to have contributed anything.")
    print("  Model, bunların en iyisini geçmelidir.")


def main():
    ap = argparse.ArgumentParser(description="Trivial baselines for fire spread")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--version", default=SPREAD_VERSION, choices=["v1", "v2", "v3"])
    ap.add_argument("--limit", type=int, default=0, help="score only N patches")
    args = ap.parse_args()

    ensure_directories_exist()
    results = run_baselines(split=args.split, version=args.version, limit=args.limit)
    print_results(results)

    out = Path(REPORTS_DIR) / f"baselines_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
