"""
Dataset — memmap patches -> training tensors
Veri kümesi — memmap yamalar -> eğitim tensörleri
=================================================

Reads the memory-mapped cache written by `tfrecord_to_npy.py` and produces
(input, target, valid) triples for the U-Net.

Four decisions are implemented here, each documented at its point of use:
  * year-based splitting, never random patch splitting
  * centre crop 65x65 -> 32x32
  * target = max(fire_next, fire_next2)
  * direction-aware augmentation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    MODEL_PATCH,
    PATCH_SIZE,
    SPREAD_SPLIT_YEARS,
    SPREAD_TARGET_MODE,
    SPREAD_VALID_BAND,
)
from features import build_input, n_output_channels, output_channel_names  # noqa: E402
from tfrecord_to_npy import load_cache  # noqa: E402


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def year_split(dates, split_years=None):
    """Indices per split, by YEAR. / Yıla göre bölme indeksleri.

    Splitting by year rather than at random is not a stylistic choice. Patches
    from the same fire day are cut from the same fire, overlap spatially, and
    share the same meteorology. A random split puts near-duplicates of one patch
    on both sides of the boundary, so the model is scored on data it has
    effectively memorised and every reported number is inflated.

    A year split also matches the operational question: the model is asked about
    a fire season it has never seen.

    Rastgele bölme SIZINTIDIR: aynı günün kareleri örtüşür ve neredeyse aynıdır.
    """
    split_years = split_years or SPREAD_SPLIT_YEARS
    years = np.asarray(dates, dtype=np.int64) // 10000
    out = {}
    for name, yrs in split_years.items():
        mask = np.isin(years, np.asarray(yrs, dtype=np.int64))
        out[name] = np.nonzero(mask)[0]
    unassigned = np.setdiff1d(
        np.arange(len(years)), np.concatenate(list(out.values())) if out else [])
    out["_unassigned"] = unassigned
    return out


# ----------------------------------------------------------------------
# Cropping
# ----------------------------------------------------------------------
def centre_slice(src=PATCH_SIZE, dst=MODEL_PATCH):
    """Slice that takes the central dst x dst from a src x src patch.
    src x src yamadan merkezî dst x dst dilimi.

    65 -> 32 keeps [16:48]. The fire-centred pixel (index 32) lands at index 16
    of the crop, i.e. still central.
    A 65x65 patch is 4225 pixels of which around 12 burn. Cropping to the centre
    raises positive prevalence without discarding the fire, and 32 is divisible
    by 2^3 so the three pooling stages are exact.
    """
    if dst > src:
        raise ValueError(f"crop {dst} larger than patch {src}")
    start = (src - dst) // 2
    return slice(start, start + dst)


# ----------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------
def _flip_channel_signs(x, names, axis):
    """Negate the vector components a flip reverses.
    Çevirmenin ters çevirdiği vektör bileşenlerini işaretle.

    This is the whole point of direction-aware augmentation. Flipping the image
    without flipping the wind vector teaches the network that fire spreads
    upwind, which is worse than not augmenting at all.

      horizontal flip (left-right, W axis) -> east becomes west
          wind_u    *= -1     (eastward component)
          aspect_sin*= -1     (sin of bearing carries the east-west part)

      vertical flip (up-down, H axis) -> north becomes south
          wind_v    *= -1     (northward component)
          aspect_cos*= -1     (cos of bearing carries the north-south part)

    Görüntüyü çevirip rüzgâr vektörünü çevirmemek, ağa yangının rüzgâra karşı
    yayıldığını öğretir.
    """
    if axis == "h":
        targets = ("wind_u", "aspect_sin")
    else:
        targets = ("wind_v", "aspect_cos")
    for t in targets:
        if t in names:
            x[names.index(t)] *= -1.0
    return x


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class SpreadDataset(Dataset):
    """Fire-spread patches from the memmap cache.
    Memmap önbelleğinden yangın yayılım yamaları."""

    def __init__(self, split="train", version=None, stats=None, augment=None,
                 target_mode=None, crop=MODEL_PATCH, indices=None):
        self.array, self.meta, self.manifest = load_cache(version)
        self.bands = self.manifest["bands"]
        self.input_bands = self.manifest["input_bands"]
        self.split = split
        self.crop = crop
        self.sl = centre_slice(self.manifest["patch_size"], crop)
        self.target_mode = target_mode or SPREAD_TARGET_MODE
        self.augment = (split == "train") if augment is None else augment
        self.stats = stats or {}

        # `valid` is fed to the network as an input channel too, so the model is
        # told which pixels were observed instead of having to infer it.
        # `valid` bandı da girdi kanalı olarak verilir.
        self.feature_bands = list(self.input_bands)
        if SPREAD_VALID_BAND in self.bands and SPREAD_VALID_BAND not in self.feature_bands:
            self.feature_bands.append(SPREAD_VALID_BAND)
        self.channel_names = output_channel_names(self.feature_bands)
        self.n_channels = len(self.channel_names)

        self._band_idx = [self.bands.index(b) for b in self.feature_bands]
        self._i_next = self.bands.index("fire_next")
        self._i_next2 = self.bands.index("fire_next2") if "fire_next2" in self.bands else None
        self._i_valid = self.bands.index(SPREAD_VALID_BAND) if SPREAD_VALID_BAND in self.bands else None
        self._i_fire = self.bands.index("fire")

        if indices is not None:
            self.indices = np.asarray(indices)
        else:
            splits = year_split(self.meta["date"])
            if split not in splits:
                raise KeyError(f"unknown split {split!r}; have {sorted(splits)}")
            self.indices = splits[split]

        if len(self.indices) == 0:
            raise RuntimeError(
                f"Split {split!r} is empty. / {split!r} bölmesi boş.\n"
                f"Years present in the archive: "
                f"{sorted(set(np.asarray(self.meta['date']) // 10000))}\n"
                f"Configured: {SPREAD_SPLIT_YEARS}")

    def __len__(self):
        return len(self.indices)

    def raw_patch(self, i):
        """Full uncropped band stack for one patch. / Kırpılmamış tam yığın."""
        return np.asarray(self.array[self.indices[i]], dtype=np.float32)

    def __getitem__(self, i):
        j = self.indices[i]
        block = np.asarray(self.array[j], dtype=np.float32)

        raw = block[self._band_idx][:, self.sl, self.sl]
        x = build_input(raw, self.feature_bands, self.stats)

        # target = max(fire_next, fire_next2): fire activity within 24-48 h.
        # The strict t+1 mask largely encodes satellite luck — 58.9 % of v1
        # patches have zero fire on t+1 while 12.3 pixels burn on average on t.
        # Katı t+1 maskesi büyük ölçüde uydu şansını kodlar.
        y = block[self._i_next][self.sl, self.sl]
        if self.target_mode == "pm1day" and self._i_next2 is not None:
            y = np.maximum(y, block[self._i_next2][self.sl, self.sl])
        y = np.clip(np.nan_to_num(y), 0.0, 1.0)[None, ...]

        if self._i_valid is not None:
            v = (block[self._i_valid][self.sl, self.sl] > 0.5).astype(np.float32)[None, ...]
        else:
            v = np.ones_like(y)

        if self.augment:
            if np.random.rand() < 0.5:                      # horizontal
                x = x[:, :, ::-1].copy()
                y = y[:, :, ::-1].copy()
                v = v[:, :, ::-1].copy()
                x = _flip_channel_signs(x, self.channel_names, "h")
            if np.random.rand() < 0.5:                      # vertical
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1, :].copy()
                v = v[:, ::-1, :].copy()
                x = _flip_channel_signs(x, self.channel_names, "v")

        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(v)))

    # -- helpers used by baselines / evaluation ------------------------
    def today_mask(self, i):
        """Today's fire mask, cropped. / Bugünün yangın maskesi (kırpılmış)."""
        j = self.indices[i]
        return np.asarray(self.array[j, self._i_fire, self.sl, self.sl], dtype=np.float32)

    def target_mask(self, i):
        """Target mask, cropped. / Hedef maske (kırpılmış)."""
        j = self.indices[i]
        y = np.asarray(self.array[j, self._i_next, self.sl, self.sl], dtype=np.float32)
        if self.target_mode == "pm1day" and self._i_next2 is not None:
            y = np.maximum(y, np.asarray(
                self.array[j, self._i_next2, self.sl, self.sl], dtype=np.float32))
        return np.clip(np.nan_to_num(y), 0.0, 1.0)

    def valid_mask(self, i):
        """Observed-pixel mask, cropped. / Gözlenen piksel maskesi."""
        j = self.indices[i]
        if self._i_valid is None:
            return np.ones((self.crop, self.crop), dtype=np.float32)
        return (np.asarray(self.array[j, self._i_valid, self.sl, self.sl]) > 0.5
                ).astype(np.float32)

    def wind_uv(self, i):
        """Mean (u, v) wind for a patch, raw units. / Yama ortalaması rüzgâr."""
        j = self.indices[i]
        if "wind_u" not in self.bands or "wind_v" not in self.bands:
            return 0.0, 0.0
        u = float(np.nanmean(self.array[j, self.bands.index("wind_u"), self.sl, self.sl]))
        v = float(np.nanmean(self.array[j, self.bands.index("wind_v"), self.sl, self.sl]))
        return u, v

    def coords(self, i):
        j = self.indices[i]
        return float(self.meta["lon"][j]), float(self.meta["lat"][j])

    def date(self, i):
        return int(self.meta["date"][self.indices[i]])


def describe_splits(version=None):
    """Print patch counts and positive prevalence per split.
    Bölme başına yama sayısı ve pozitif oran."""
    array, meta, man = load_cache(version)
    splits = year_split(meta["date"])
    bands = man["bands"]
    i_next = bands.index("fire_next")
    i_next2 = bands.index("fire_next2") if "fire_next2" in bands else None
    sl = centre_slice(man["patch_size"], MODEL_PATCH)

    print(f"{'split':<14}{'patches':>9}{'years':>28}{'positive %':>12}")
    print("-" * 63)
    for name, idx in splits.items():
        if len(idx) == 0:
            print(f"{name:<14}{0:>9}{'—':>28}{'—':>12}")
            continue
        yrs = sorted(set(np.asarray(meta["date"])[idx] // 10000))
        sub = idx[:: max(1, len(idx) // 2000)]
        y = np.asarray(array[sub][:, i_next, sl, sl], dtype=np.float32)
        if i_next2 is not None:
            y = np.maximum(y, np.asarray(array[sub][:, i_next2, sl, sl], dtype=np.float32))
        pos = 100.0 * float((y > 0.5).mean())
        print(f"{name:<14}{len(idx):>9}{str(yrs):>28}{pos:>11.4f}%")
    return splits


if __name__ == "__main__":
    describe_splits()
