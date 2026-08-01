"""
Feature engineering — raw bands -> network input
Öznitelik mühendisliği — ham bantlar -> ağ girdisi
=================================================

Turns the raw channels (14 in v2, 19 in v3, 21 in v4) into the tensor the U-Net
actually consumes. Four transformations, each fixing a defect in the raw data.

Ham kanalları U-Net'in gerçekten tükettiği tensöre çevirir.

1) z-score normalisation, statistics from the TRAINING SPLIT ONLY
   In the v1 archive `elevation` had std 515.44 and max 4978, sitting next to
   `soil_moisture` with std 0.07 and `ndvi` with std 0.20. A convolution sums
   weight*input across channels, so the elevation term dominated the sum by
   three orders of magnitude and the first layer effectively saw only elevation
   and aspect. Normalising puts every channel on comparable footing.
   Statistics MUST come from the training split; taking them from validation or
   test leaks information about the evaluation data into training.
   İstatistikler YALNIZCA eğitim bölmesinden gelir; aksi hâlde sızıntı olur.

2) aspect -> sin, cos
   Aspect is a compass bearing in [0, 360). 359 deg and 1 deg are two degrees
   apart on the ground but 358 apart numerically. Feeding the raw angle asks the
   network to learn that the largest value is adjacent to the smallest. Encoding
   as (sin, cos) puts the circle on a circle.
   Bakı dairesel bir değişkendir; 359 ile 1 derece yan yanadır.

3) landcover -> 6-group fuel one-hot
   IGBP classes 0-17 are LABELS, not magnitudes. Feeding the integer tells the
   network that class 12 (cropland) is "twice" class 6 (shrubland) and that
   class 17 is the largest of all, which is meaningless. The 18 classes are
   collapsed to 6 fuel groups because that is the distinction that matters for
   fire behaviour, and 18 one-hot channels on 22k patches would be sparse.
   IGBP sınıfları etikettir, büyüklük değil; tek-sıcak kodlanmalıdır.

4) precip -> log1p
   Daily rainfall is strongly right-skewed and mostly exactly zero: a few
   millimetres on most wet days, occasionally 80 mm. log1p compresses the tail
   without touching the zeros (log1p(0) = 0).
   Günlük yağış çok çarpıktır; log1p kuyruğu sıkıştırır, sıfırları bozmaz.

The `valid` band is passed through as an input channel as well, so the network
is told which pixels were genuinely observed rather than having to infer it.
`valid` bandı da girdi kanalı olarak geçirilir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import SPREAD_NORM_STATS_FILE  # noqa: E402

# ----------------------------------------------------------------------
# Channel groups / kanal grupları
# ----------------------------------------------------------------------
# Bands that are z-scored directly.
CONTINUOUS = (
    "ndvi", "lst", "air_temp", "humidity", "vpd",
    "wind_speed", "wind_u", "wind_v",
    "soil_moisture", "elevation", "slope",
    # v4: recency channels, already in days and bounded by their caps, but
    # z-scored like any other continuous band so no channel dominates the
    # first convolution. / v4: gün cinsinden geçmiş kanalları.
    "days_since_rain", "burn_age",
)
# Bands that get log1p before z-scoring (right-skewed, mostly zero).
LOG_SCALED = ("precip", "precip_7d", "precip_30d")
# Bands already in [0, 1] that are passed through untouched.
BINARY = ("fire", "fire_prev1", "fire_prev2", "valid",
          "valid_next", "valid_next2")
# Special encodings.
CIRCULAR = ("aspect",)
CATEGORICAL = ("landcover",)

# ----------------------------------------------------------------------
# IGBP LC_Type1 -> 6 fuel groups
# ----------------------------------------------------------------------
# 0 water · 1-5 forest · 6-7 shrub · 8-9 savanna · 10 grass · 11 wetland
# 12 cropland · 13 urban · 14 crop/natural mosaic · 15 snow · 16 barren · 17 water
FUEL_GROUPS = {
    0: "non_fuel",     # water, urban, snow, barren — cannot carry fire
    1: "forest",       # closed canopy; crown fire, highest intensity
    2: "shrubland",    # maquis / garrigue — the Mediterranean fire carrier
    3: "savanna",      # woody savanna, open tree cover
    4: "grassland",    # fast, low-intensity, wind-driven
    5: "cropland",     # cropland, mosaic and wetland
}
_IGBP_TO_FUEL = {
    0: 0, 13: 0, 15: 0, 16: 0, 17: 0,          # non-fuel
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,              # forest
    6: 2, 7: 2,                                # shrubland
    8: 3, 9: 3,                                # savanna
    10: 4,                                     # grassland
    11: 5, 12: 5, 14: 5,                       # cropland / wetland
}
N_FUEL_GROUPS = len(FUEL_GROUPS)

_LUT = np.zeros(256, dtype=np.int64)
for _k, _v in _IGBP_TO_FUEL.items():
    _LUT[_k] = _v


def output_channel_names(input_bands) -> list:
    """Names of the channels the network receives, in order.
    Ağın aldığı kanalların sırayla adları."""
    names = []
    for b in input_bands:
        if b in CIRCULAR:
            names += [f"{b}_sin", f"{b}_cos"]
        elif b in CATEGORICAL:
            names += [f"fuel_{FUEL_GROUPS[i]}" for i in range(N_FUEL_GROUPS)]
        else:
            names.append(b)
    return names


def n_output_channels(input_bands) -> int:
    """Channel count after feature engineering. / Dönüşüm sonrası kanal sayısı."""
    return len(output_channel_names(input_bands))


# ----------------------------------------------------------------------
# Normalisation statistics
# ----------------------------------------------------------------------
def compute_norm_stats(array, band_names, train_idx, valid_band_idx=None,
                       max_patches=20000, seed=42):
    """Mean/std per normalised band, from the TRAINING SPLIT ONLY.
    Yalnızca EĞİTİM bölmesinden bant başına ortalama/std.

    Only pixels where `valid` == 1 contribute, so unobserved pixels do not drag
    the mean towards zero.
    Yalnızca valid==1 pikselleri katkı verir.
    """
    rng = np.random.default_rng(seed)
    idx = np.asarray(train_idx)
    if len(idx) > max_patches:
        idx = rng.choice(idx, size=max_patches, replace=False)
    idx = np.sort(idx)

    stats = {}
    for b in band_names:
        if b not in CONTINUOUS and b not in LOG_SCALED:
            continue
        bi = band_names.index(b)
        vals = []
        # Chunked so we never materialise the whole archive.
        for start in range(0, len(idx), 512):
            chunk = idx[start:start + 512]
            block = np.asarray(array[chunk, bi, :, :], dtype=np.float64)
            if valid_band_idx is not None:
                mask = np.asarray(array[chunk, valid_band_idx, :, :]) > 0.5
                block = block[mask]
            else:
                block = block.ravel()
            block = block[np.isfinite(block)]
            if b in LOG_SCALED:
                block = np.log1p(np.clip(block, 0, None))
            vals.append(block)
        allv = np.concatenate(vals) if vals else np.zeros(1)
        mean = float(allv.mean()) if allv.size else 0.0
        std = float(allv.std()) if allv.size else 1.0
        # A constant band would divide by zero.
        stats[b] = {"mean": mean, "std": std if std > 1e-6 else 1.0}
    return stats


def save_norm_stats(stats, path=None, extra=None):
    """Persist normalisation statistics. / İstatistikleri kaydet."""
    path = Path(path or SPREAD_NORM_STATS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stats": stats}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_norm_stats(path=None):
    """Load normalisation statistics. / İstatistikleri yükle."""
    path = Path(path or SPREAD_NORM_STATS_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"No normalisation statistics at {path}\n"
            f"They are written by src/train.py from the training split.\n"
            f"İstatistikler eğitim sırasında yazılır; önce train.py çalıştırın.")
    return json.loads(path.read_text(encoding="utf-8"))["stats"]


# ----------------------------------------------------------------------
# Transform
# ----------------------------------------------------------------------
def build_input(raw, band_names, stats):
    """Raw band stack -> network input. / Ham bant yığını -> ağ girdisi.

    raw : (B, H, W) or (N, B, H, W) float32, raw band values in band_names order
    out : (C, H, W) or (N, C, H, W) float32
    """
    single = raw.ndim == 3
    if single:
        raw = raw[None, ...]
    n, _, h, w = raw.shape
    out = np.empty((n, n_output_channels(band_names), h, w), dtype=np.float32)

    c = 0
    for bi, b in enumerate(band_names):
        chan = np.asarray(raw[:, bi], dtype=np.float32)
        chan = np.nan_to_num(chan, nan=0.0, posinf=0.0, neginf=0.0)

        if b in CIRCULAR:
            rad = np.deg2rad(chan)
            out[:, c] = np.sin(rad)
            out[:, c + 1] = np.cos(rad)
            c += 2

        elif b in CATEGORICAL:
            codes = np.clip(chan.astype(np.int64), 0, 255)
            fuel = _LUT[codes]
            for g in range(N_FUEL_GROUPS):
                out[:, c + g] = (fuel == g).astype(np.float32)
            c += N_FUEL_GROUPS

        elif b in LOG_SCALED:
            v = np.log1p(np.clip(chan, 0, None))
            s = stats.get(b, {"mean": 0.0, "std": 1.0})
            out[:, c] = (v - s["mean"]) / s["std"]
            c += 1

        elif b in CONTINUOUS:
            s = stats.get(b, {"mean": 0.0, "std": 1.0})
            out[:, c] = (chan - s["mean"]) / s["std"]
            c += 1

        elif b in BINARY:
            out[:, c] = np.clip(chan, 0.0, 1.0)
            c += 1

        else:
            # Unknown band: pass through unchanged rather than dropping it,
            # so a band-contract change fails loudly downstream, not silently.
            out[:, c] = chan
            c += 1

    return out[0] if single else out


if __name__ == "__main__":
    from config import spread_bands

    for v in ("v2", "v3"):
        inp, aux, _ = spread_bands(v)
        names = output_channel_names(inp)
        print(f"{v}: {len(inp)} raw input bands -> {len(names)} network channels")
        print(f"     {names}")
        print()
