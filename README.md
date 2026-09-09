# IGNIS — Next-Day Wildfire Spread Prediction over Türkiye

**Predicting where an already-burning fire will be burning tomorrow, from satellite
observations.**

IGNIS is an Earth Observation system that takes a fire which is actively burning on day
*t* and predicts which 1 km pixels will be burning within the next 24–48 hours. A U-Net
consumes a fire-centred patch of environmental driver channels and emits a per-pixel
probability map.

> **Türkçe okur için.** Bu belge, uluslararası okur ve IAC değerlendiricileri için
> İngilizce yazılmıştır; bölüm başlıkları iki dillidir. Konunun sıfırdan anlatıldığı tam
> Türkçe eğitim rehberi için **[`docs/REHBER_TR.md`](docs/REHBER_TR.md)** dosyasına
> bakınız (İngilizce muadili: [`docs/GUIDE_EN.md`](docs/GUIDE_EN.md)).

---

## Scientific status / Bilimsel durum

**The model is statistically indistinguishable from its own baseline, and does not beat
it.** On the held-out 2025–2026 test split a trivial "tomorrow = today" persistence rule
scores IoU 0.1178. The trained network scores IoU 0.0974 under the configured split, and
0.1182 when the 2024 season is moved from validation into training. That best margin,
**0.0004**, is a quarter of the run-to-run noise (0.0023 across random seeds), so it is a
tie rather than a win. Section 9 reports all of it, including the earlier figures it
supersedes.

The defensible contribution today is a reproducible pipeline and an honestly reported
baseline — not an operational prediction system.

- **Paper:** `IAC-26,B1,IP,107,x110901` — IAF Earth Observation Symposium (B1),
  Interactive Presentations. 77th International Astronautical Congress,
  Antalya, 5–9 October 2026. Final manuscript due **14 September 2026**.
- **Authors:** six students at Antalya Yusuf Ziya Öner Science High School.
- **Repository:** `github.com/WhiteFoxYT/ignis-ai`

---

## Contents / İçindekiler

1. [The problem](#1-the-problem--problem)
2. [What this is not](#2-what-this-is-not--bu-ne-değildir)
3. [Method](#3-method--yöntem)
4. [Dataset and band contract](#4-dataset-and-band-contract--veri-seti-ve-bant-sözleşmesi)
5. [Target definition](#5-target-definition--hedef-tanımı)
6. [Splits](#6-splits--bölmeler)
7. [Known data exclusions](#7-known-data-exclusions--bilinen-veri-dışlamaları)
8. [Model architecture](#8-model-architecture--model-mimarisi)
9. [Results](#9-results--sonuçlar)
10. [Installation](#10-installation--kurulum)
11. [Running the pipeline](#11-running-the-pipeline--hattı-çalıştırma)
12. [Repository layout](#12-repository-layout--depo-yapısı)
13. [Limitations and future work](#13-limitations-and-future-work--kısıtlar-ve-gelecek-çalışma)
14. [References](#14-references--kaynakça)

---

## 1. The problem / Problem

Türkiye lies in the Mediterranean fire regime: hot dry summers, *Pinus brutia* forest that
carries crown fire, and a fire season running roughly June–October. Fire management has
four phases — mitigation, preparedness, response, recovery. Suppression decisions in the
**response** phase are made under time pressure with incomplete information: where to
place crews, which villages to evacuate, where the head of the fire will be tomorrow.

IGNIS targets exactly that question. Given the observed fire mask on day *t* plus the
environmental state, predict the fire mask for day *t*+1.

## 2. What this is not / Bu ne değildir

This is **not** fire susceptibility mapping — "where might a fire start". That is a static
problem, and published work reports ROC-AUC above 0.93 on it. Susceptibility maps answer a
question about long-run landscape properties and can be validated against decades of
ignition records.

Spread prediction is the **temporal** problem, and it is considerably harder: the target
changes every 24 hours, the positive class is 0.2686 % of pixels, and the label itself is
a satellite observation with its own failure modes.

Conflating the two would make our numbers look far better than they are. The distinction
is load-bearing throughout the manuscript.

Eight static-risk modules from an earlier, abandoned susceptibility model (`preprocess`,
`train`, `predict`, `test_accuracy`, `map_visualization`, `main`, `examples`,
`gee_data_processor`) and their weights have been deleted from this repository to prevent
that confusion.

## 3. Method / Yöntem

```
Google Earth Engine                     Local workstation (Arch Linux + ROCm)
─────────────────────                   ──────────────────────────────────────
8 source products                       data/spread_v5/*.tfrecord.gz
   │  harmonise: EPSG:32635, 1 km             │
   │  daily composite                         │  src/tfrecord_to_npy.py
   ▼                                          ▼
stratifiedSample on burning pixels      ~/ignis-cache/*.npy   (memmap)
   │                                          │  src/features.py  (normalise, encode)
   │  neighborhoodToArray(radius 32)          ▼
   ▼                                    src/dataset.py  → 32×32 patches
65×65 patch per burning pixel                 │
   │                                          │  src/train.py
   ▼                                          ▼
Export.table.toDrive (TFRecord)         models/spread_unet.pt
                                              │  src/evaluate.py  (test split only)
                                              ▼
                                        outputs/reports/*.html
```

**Study area and grid.** Türkiye (`USDOS/LSIB_SIMPLE/2017`), reprojected to **EPSG:32635**
(UTM zone 35N) at **1 km** resolution. Every band is co-registered onto that grid so pixel
*(i, j)* refers to the same ground in all channels.

**Source products.** MODIS `MOD14A1`/`MYD14A1` (active fire, Terra + Aqua), `MOD13Q1`
(NDVI), `MOD11A1` (land surface temperature), `MCD12Q1` (IGBP land cover), ERA5-Land
`DAILY_AGGR` (air temperature, dewpoint, wind *u*/*v*, soil moisture), CHIRPS `DAILY`
(precipitation), SRTM `USGS/SRTMGL1_003` (elevation, slope, aspect).

**Fire mask.** MODIS FireMask classes 7, 8, 9 (low / nominal / high confidence) are all
treated as fire. Threshold 7 maximises recall; accepting only class 9 would discard the
majority of genuine detections.

**Patch extraction.** For each fire day, up to 150 burning pixels are drawn with
`stratifiedSample`, and `neighborhoodToArray` with a 32-pixel-radius square kernel cuts a
**65×65** patch around each. Patches are exported as gzip TFRecord shards, one per day.

**Feature engineering** (`src/features.py`, applied at load time, not in GEE):

| Transformation | Reason |
|---|---|
| z-score normalisation, statistics from the **training split only** | `elevation` has std 515.44 and max 4978 next to `soil_moisture` std 0.07; without this the first convolution effectively sees only elevation and aspect. Statistics taken from validation or test would be leakage. |
| `aspect` → `sin`, `cos` | Aspect is circular: 359° and 1° are adjacent on the ground but maximally distant numerically. |
| `landcover` → 6-way fuel-group one-hot | IGBP classes 0–17 are labels, not magnitudes. Feeding the integer implies class 12 is "twice" class 6. |
| `precip` → `log1p` | Daily precipitation is strongly right-skewed and mostly zero. |
| Direction-aware augmentation | A horizontal flip must negate `wind_u` and `aspect_sin`; a vertical flip must negate `wind_v` and `aspect_cos`. A naive flip teaches the network that fire spreads upwind. |

Normalisation statistics are written to `models/norm_stats.json` and reused verbatim at
evaluation time.

## 4. Dataset and band contract / Veri seti ve bant sözleşmesi

| Location | Input bands | Period | Shards | Valid patches |
|---|---|---|---|---|
| `data/spread_v5/` | 21 | 2019 – 2026 | 1096 | 67,056 |

Band order is **contractual** across three files — `noteboks/colab_notebook_v5.ipynb`,
`src/gee_config.py` and `src/config.py`. Change it in all three or in none. The loader
reconstructs the channel axis from band order alone, so a record written against a
different band order is silently misinterpreted rather than rejected.
`tfrecord_to_npy.py` therefore detects the schema from each record's feature *names* and
aborts rather than guessing.

**21 input bands + 5 target/auxiliary:**

```
ndvi  lst  air_temp  humidity  vpd
wind_speed  wind_u  wind_v
precip  precip_7d  precip_30d  days_since_rain
soil_moisture
elevation  slope  aspect  landcover
burn_age
fire_prev2  fire_prev1  fire
fire_next  fire_next2  valid  valid_next  valid_next2
```

After feature engineering the network sees **28 channels**: `aspect` becomes sin + cos,
`landcover` becomes six fuel groups, and `valid` is fed as an input.

Why each of the less obvious bands is there. `vpd` (vapour pressure deficit) is the
single best atmospheric predictor of fuel dryness. `precip_7d` and `precip_30d` carry
antecedent drying, which a 24-hour rainfall figure cannot express. `days_since_rain`
tracks fine-fuel dryness, which a 24-hour total also cannot — that band is exactly zero
on 91.4 % of pixels. `fire_prev1` and `fire_prev2` give the network two days of fire
history, and so an observed spread *direction* rather than a single frame. `burn_age` is
days since the pixel last burned: fire does not spread into ground whose fuel it has
already consumed, and without it the network cannot tell the burnt interior of a fire
from the unburnt land ahead of its front.

The important pair is `valid_next` / `valid_next2`, described below.

### The `valid` band

An early implementation used `clip(REGION)` followed by `unmask(0)`, which wrote a
literal zero wherever a source product had no observation. Roughly **15 % of every patch was a fabricated zero**, with an
identical zero rate across all environmental bands — "relative humidity = 0 %" was
numerically indistinguishable from "not observed".

The archive exports a `valid` band that is 1 only where every input was genuinely
observed, and where MODIS observed the pixel *today* rather than merely leaving the
environmental bands unmasked. The loss is masked with it, so out-of-region and
unobserved pixels contribute no gradient.

### `valid_next` — the fix for the biggest diagnosed problem

Diagnosed cause 3 said the target "largely encodes satellite luck": 58.9 % of patches
had zero fire on *t*+1 while 12.3 pixels burn on average on *t*.

That was not purely a limitation of MODIS. It was partly **manufactured by our own
code.** The `FireMask` band already encodes observation quality:

| Value | Meaning | Observed? |
|---|---|---|
| 0, 1, 2 | not processed (no input data) | **no** |
| 3 | water | yes |
| 4 | **cloud** | **no** |
| 5 | non-fire land | yes |
| 6 | unknown | **no** |
| 7, 8, 9 | fire (low / nominal / high confidence) | yes |

An early implementation used `fm.gte(7).unmask(0)`, which collapses "observed, not burning",
"hidden by cloud" and "never processed" into the single value 0. A pixel behind a
cloud was labelled *no fire*, and the network was trained to reproduce that.

The archive exports `valid_next` and `valid_next2`, so the loss can mask target pixels the
satellite never actually looked at. `dataset.py` applies an asymmetric rule,
because the evidence is asymmetric: a **detection** is trustworthy on its own,
but an **absence** is only evidence if we looked. With
`target = max(fire_next, fire_next2)`, a zero counts only when both days were
observed.

## 5. Target definition / Hedef tanımı

```python
target = max(fire_next, fire_next2)      # fire activity within the next 24–48 h
```

The strict *t*+1 mask is still exported, so both definitions remain available and directly
comparable.

The reason for widening the window is that the strict target largely encodes satellite
luck rather than fire behaviour: **58.9 % of patches have zero fire pixels on *t*+1**
while 12.3 pixels burn on average on *t*. A fire that is plainly still burning disappears
from the label because Terra and Aqua happened to overpass through cloud, or the thermal
anomaly fell below the detection threshold at that moment. Training a network to reproduce
that is training it to predict orbital geometry.

The paper framing therefore becomes "next 24–48 h fire activity". The title and author
list are fixed by IAF rules and do not change.

**Patch growth class**, applied identically to observed and predicted masks:

```
r = N(t+1) / max(N(t), 1)

r > 1.25          →  growing        (büyüyen)
0.75 ≤ r ≤ 1.25   →  stable         (durağan)
r < 0.75          →  extinguishing  (sönen)
```

The stable band was widened from an earlier 1.15 / 0.85: the narrow band made "stable" so
rare that the class was effectively unlearnable.

## 6. Splits / Bölmeler

Splitting is **by year**, never by random patch:

| Split | Years |
|---|---|
| Train | 2019 – 2023 |
| Validation | 2024 |
| Test | 2025 – 2026 |

Patches from the same fire day are spatially overlapping and meteorologically near
identical. A random patch split puts near-duplicates of the same fire on both sides of the
boundary, which is leakage and inflates every reported score. A year split also matches
the operational question — the model is asked about a fire season it has never seen.

Normalisation statistics come from the **training split only**.

## 7. Known data exclusions / Bilinen veri dışlamaları

Terra MODIS acquired **no data from 10–19 October 2022** (Constellation Exit Manoeuvres;
instruments recovering through 21 October). `MOD11A1`'s three-day compositing window falls
entirely inside the outage for several of those days, producing a band-less image that
fails at `.rename()`.

Those days are **excluded** via `KNOWN_OUTAGES` rather than gap-filled, because filling
them would fabricate two of the input channels. This affects approximately 5 days out of
~1136, or **0.4 %** of the archive.

## 8. Model architecture / Model mimarisi

A U-Net (Ronneberger et al., 2015) for binary semantic segmentation. The architecture is
**unchanged from the TensorFlow implementation** — the PyTorch port reproduces it layer
for layer so that Section 2.6 of the manuscript remains valid.

| Property | Value |
|---|---|
| Input | 32 × 32 × *C* (centre crop from the 65 × 65 patch) |
| Depth | 3 |
| Encoder filters | 32 → 64 → 128 |
| Bottleneck | 4 × 4 × 256 |
| Decoder | transposed convolutions with skip connections |
| Output | 32 × 32 × 1, sigmoid |
| Parameters | ~1.9 M |

**Why crop 65 × 65 down to 32 × 32.** Fires are small. A 65 × 65 patch is 4225 pixels of
which around 12 burn. Cropping to the central 32 × 32 raises the positive prevalence
without discarding the fire, and 32 is divisible by 2³ so the three pooling stages are
exact.

**Skip connections** carry the high-resolution fire front from encoder to decoder; without
them the 4 × 4 bottleneck cannot restore a sharp perimeter.

**Loss.** Default `0.5 × BCE(pos_weight) + 0.5 × SoftDice`; `FocalTversky(α=0.3, β=0.7)`
available as an alternative. The loss is masked with the `valid` band so out-of-bounds
pixels never enter the gradient.

## 9. Results / Sonuçlar

### 9.1 The earlier implementation, measured

Measured on **45 shards / 1054 patches** sampled from the earlier archive with a
pure-Python TFRecord reader. **These are superseded by the results in § 9.3** and are
retained only as the starting point the corrections below were measured against.

| Quantity | Model | Persistence baseline |
|---|---|---|
| AUC-PR | **0.0210** | — |
| ROC-AUC | **0.8468** | — |
| Precision | 0.0601 | — |
| Recall | 0.0222 | — |
| **F1** | **0.0324** | **0.0595** |
| **IoU** | **0.0165** | **0.0306** |

| Context | Value |
|---|---|
| Positive pixel prevalence | 0.2686 % |
| Patch-level accuracy | 0.7714 |
| Majority-class ("extinguishing") share | 0.7774 |

**Read this honestly:**

- The model **loses to persistence** on both F1 and IoU. A rule that copies today's fire
  mask to tomorrow outperforms the trained network.
- Patch-level accuracy 0.7714 is **below** the majority-class share 0.7774. Always
  answering "extinguishing" would score higher.
- Any "77 % accuracy" figure is an artefact of class prevalence, not evidence of skill.
- ROC-AUC 0.8468 alongside AUC-PR 0.0210 is not a contradiction. ROC-AUC is dominated by
  true negatives, of which there are 372 for every positive. AUC-PR is the honest metric
  here, and its random baseline is the prevalence itself, 0.00269.
- These numbers are also **optimistic**: that evaluation globbed all shards including
  training days, so it was partly in-sample.

### 9.2 Diagnosed causes

1. **No input normalisation.** `elevation` std 515.44 (max 4978), `aspect` 0–359,
   `landcover` 0–17 integer classes, against `soil_moisture` std 0.07 and `ndvi` std 0.20.
2. **~15 % of every patch was a fabricated zero** from `clip` + `unmask(0)`.
3. **The target largely encodes satellite luck** — 58.9 % of patches have zero fire on
   *t*+1.
4. **Patch far too large** — 4225 pixels for a fire of ~12.
5. **Evaluation was in-sample.**

Causes 1 and 2 are absent from the manuscript's own diagnosis and were found by direct
inspection of the archive.

### 9.3 Rebuilt pipeline, measured 9 August 2026

Held-out test split: **2025 + 2026, 13,792 patches**, never seen in training or model
selection. The decision threshold (0.2333) was calibrated on the **validation** split to
maximise F1 and then held fixed. It was never retuned on test.

| | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| **MODEL** | 0.1054 | 0.5641 | 0.1776 | **0.0974** |
| **Persistence** | 0.1721 | 0.2720 | 0.2108 | **0.1178** |
| Dilated persistence | 0.1209 | 0.4208 | 0.1878 | 0.1036 |
| Wind-directed growth | 0.1460 | 0.3299 | 0.2024 | 0.1126 |

| Context | Value |
|---|---|
| AUC-PR | 0.1789 |
| ROC-AUC | 0.8932 |
| Positive pixel prevalence (of observed pixels) | 1.0816 % |
| Patch-level accuracy | 0.6212 |
| Majority-class share | 0.7683 |

**The acceptance criterion is not met. The model loses to persistence on the test split
(IoU 0.0974 against 0.1178).** Patch accuracy 0.6212 remains below the majority-class
share 0.7683. This is stated first because it is the result.

Every quantity improved substantially over the earlier implementation — IoU
0.0165 → 0.0974, AUC-PR 0.0210 → 0.1789 — but **the baseline improved too**, and it is
the comparison that decides, not the absolute number. The two sets of figures are not
directly comparable in any case: the rebuilt pipeline scores only pixels that were
actually observed, which is a different and more honest denominator.

#### Where the loss comes from

Diagnosis at the **same fixed validation threshold**, splitting the held-out data by year:

| Split | Patches | Model IoU | Persistence IoU | |
|---|---|---|---|---|
| Validation 2024 | 13,049 | **0.0442** | 0.0332 | model wins, +33 % |
| Test 2025 | 8,906 | 0.1270 | **0.1493** | persistence wins |
| Test 2026 | 4,886 | **0.0658** | 0.0428 | model wins, +54 % |

The model beats persistence in **two of the three held-out years**, and loses in 2025 —
which is 65 % of the test split. 2025 was an anomalous season: mean burning pixels per
patch **19.03**, against ~11 in both 2024 and 2026, with persistence IoU 0.1474 against
~0.035. Large, long-lived fires are precisely the regime in which "tomorrow = today" is
hardest to beat, because such a fire is still burning tomorrow almost by definition.

All three years cover the same June–October window, so the comparison is seasonally fair.
**2026 ends on 25 July** (55 days against ~151), so it is a partial season and its result
carries correspondingly less weight.

The error profile is systematic **over-prediction**: recall 0.5641 against persistence's
0.2720, but precision 0.1054 against 0.1721. The network finds roughly twice as much of
the real fire as the baseline and pays for it in false positives.

#### What this does not yet establish

The result is a measurement of one training protocol, not a ceiling on the approach.
Best validation AUC-PR occurred at **epoch 2**; the following 18 epochs never improved on
it while training loss fell 0.60 → 0.50 and validation loss rose 0.63 → 0.70. That is
memorisation of the training years, and dropout, augmentation and the early-stopping rule
have not been revisited since.

Three hypotheses have since been tested and **failed**. They are recorded so that nobody
spends a day re-testing them:

- `SPREAD_POS_WEIGHT` does **not** explain the precision collapse — 12 / 4 / 1 give
  validation IoU 0.0463 / 0.0445 / 0.0463, with no trend.
- Direction-aware augmentation has no measurable effect — 0.0461 off against 0.0463 on,
  inside seed noise. The flip logic is correct; it simply does not help here.
- `focal_tversky` is much worse at 0.0228, below persistence. Eliminated.

Reproduce the per-year table with `python peryear_diag.py`, the seed spread with
`python seed_noise.py`, and the split comparison with `python split_ablation.py`.

### 9.4 The largest single lever was data, not tuning

An eight-configuration hyperparameter sweep moved validation IoU between 0.0445 and
0.0489 — best `wd 1e-3 + dropout 0.3`, an improvement of 5.6 %. Moving **one season**
from validation into training moved the test result four times as far.

The configured split reserves 2024 for validation, so the model never trains on the year
immediately before the test period — 13,049 patches, 24 % of the usable data, spent on
model selection alone. Training on 2019–2022 + 2024 and validating on 2023 keeps the test
split byte-for-byte identical:

| | IoU | F1 |
|---|---|---|
| Model, configured split | 0.0974 | 0.1776 |
| **Model, 2024 moved into training** | **0.1182** | **0.2114** |
| Persistence | 0.1178 | 0.2108 |

**Read this as a tie, not a win.** The margin is 0.0004. Changing only the random seed
moves validation IoU by 0.0023 (0.0463 / 0.0457 / 0.0480 across three seeds), and two runs
at the *same* seed differ by 0.0015 because ROCm backward kernels are not deterministic.
The margin is a quarter of same-seed noise. It is a real improvement on the earlier clear
loss, and it is not a win.

Leave-one-year-out gives mean model IoU 0.1209 ± 0.0871. **These figures are optimistic**:
`experiments.loyo()` selects both the stopping epoch and the threshold on the held-out
year itself, so it measures an upper bound, not clean generalisation. Use it to compare
years, never as a headline. What it does establish is that **persistence itself varies
six-fold across years** — 0.0323 in 2023 against 0.2003 in 2021 — so any single
train/val/test split was always going to give a high-variance verdict.

**Never use a random split on this data.** Measured, not assumed: under a random 70/15/15
split **100.00 %** of test patches share a fire day with a training patch, because all
1096 days appear in training; under the year split, **0.00 %**. Patches cut from the same
day come from the same fire, overlap in space and share the same weather.
`split_ablation.py` quantifies the inflation this causes, and its output is an ablation —
it must never be quoted as IGNIS performance.

## 10. Installation / Kurulum

### Hardware and platform

Developed and run on:

- **Arch Linux**, kernel 7.1.5-zen
- **AMD Radeon RX 9070 XT** (Navi 48, **gfx1201**, RDNA 4)
- Ryzen 7 7800X3D (8 cores), 30 GB RAM

### Dependencies

```bash
sudo pacman -S rocm-hip-sdk rocminfo python-pytorch-rocm \
               python-scikit-learn python-matplotlib python-scipy
```

`extra/python-pytorch-rocm` is built against system Python 3.14.6 and depends on
`rocm-hip-sdk 7.2.4`.

### Verification

```bash
rocminfo | grep gfx                      # expect: gfx1201
python -c "import torch; print(torch.cuda.get_device_name(0))"
# expected: Radeon RX 9070 XT
```

### Two settings that matter

- **Do not set `HSA_OVERRIDE_GFX_VERSION`.** gfx1201 is natively supported from ROCm 7.2.
  The override is advice for older RDNA cards and will select the wrong kernel library
  here.
- **Use `bfloat16` autocast, never `float16`.** bfloat16 keeps float32's 8 exponent bits,
  so it has the same dynamic range and needs no loss scaling. float16 underflows on the
  small gradients this heavily imbalanced loss produces.

### Why not TensorFlow

The project was ported away from TensorFlow. There is no TensorFlow wheel for Python 3.14,
and AMD's TensorFlow-ROCm path is Docker-only. Earlier revisions of this README described
a WSL2 + DirectML setup; that was for a Windows machine and no longer describes this
project.

### A note on storage

The working copy is `~/Projects/ignis` on local ext4. An earlier location on an NTFS
`fuseblk` mount proved unusable: it is forced read-only whenever Windows has been
hibernated, and decompressing gzip TFRecords every epoch across FUSE dominated training
time. `src/tfrecord_to_npy.py` converts the archive **once** into a memory-mapped array
under `~/ignis-cache/`; training reads only from there.

Because the cache is a `np.memmap`, it must never be pickled into a `DataLoader` worker.
Python 3.14 defaults to the `forkserver` start method, which pickles the dataset object,
and a memmap pickles as a fully materialised array — every worker would allocate the
entire 27 GB cache. `SpreadDataset` defines `__getstate__`/`__setstate__` to drop the
handle and reopen it inside the worker.

## 11. Running the pipeline / Hattı çalıştırma

### The short version

Once `data/spread_v5/` holds the shards, the whole project runs from one command:

```bash
python start.py
```

That checks the environment and the GPU, builds the memmap cache if it is missing or
stale, trains the U-Net, evaluates it on the **test** split against recomputed
baselines, and prints where the outputs went. Useful variants:

```bash
python start.py --epochs 40        # shorter training run
python start.py --only eval        # one stage: check|data|cache|train|eval
python start.py --skip-train       # evaluate the existing checkpoint
python start.py --force-retrain    # overwrite an existing checkpoint
```

It refuses to launch a second training while another process holds `/dev/kfd`, so two
runs cannot quietly compete for the GPU. The remaining steps below describe what each
stage does, and how to run them individually.

### Step 1 — Generate the dataset (Google Colab + GEE)

Open `noteboks/colab_notebook_v5.ipynb` in Colab and run all cells. It is **resumable**:
re-running skips days already written to Drive, already queued in Earth Engine, or
recorded in the Drive-side submission ledger. `SUBMIT_LIMIT` caps tasks per run.
A preflight and a one-day smoke test run before any bulk submission, because three
earlier notebook revisions each reported success while exporting nothing.

Task descriptions are namespaced (`firespread_v5_YYYYMMDD`) so that an earlier run and a
current one cannot be mistaken for one another by the resume scan — a namespacing bug is
precisely what let one of those three revisions report success while exporting nothing.
Downloaded shard names stay `firespread_YYYYMMDD.tfrecord.gz`.

Download the resulting Drive folder into `data/spread_v5/` and unzip it there, so the
directory holds `*.tfrecord.gz` shards directly.

### Step 2 — Convert to the local memmap cache

```bash
python src/tfrecord_to_npy.py --verify
```

Writes `~/ignis-cache/` and prints an integrity report: per-band zero rates, band count
per record, and any short or malformed records. Uses a pure-Python protobuf decoder —
**no TensorFlow dependency**.

### Step 3 — Train

```bash
python src/train.py
```

AdamW, `CosineAnnealingWarmRestarts`, bfloat16 autocast, gradient clipping. Checkpoints on
best validation AUC-PR to `models/spread_unet.pt`.

### Step 4 — Baselines and evaluation

```bash
python src/baselines.py     # persistence, dilated persistence, wind-directed growth
python src/evaluate.py      # test split only, calibrated threshold
```

`evaluate.py` reads **only the test split**, calibrates the decision threshold to maximise
F1 **on validation**, and writes an HTML report, scorecard and folium map to
`outputs/reports/`.

## 12. Repository layout / Depo yapısı

```
noteboks/
  colab_notebook_v5.ipynb     GEE export, 21 input bands

src/
  config.py                   all constants; the SPREAD_* section is the live one
  gee_config.py               GEE collections and the band contract
  device.py                   ROCm device selection, bfloat16 autocast
  tfrecord_to_npy.py          TFRecord → ~/ignis-cache memmap .npy (no TensorFlow)
  features.py                 raw channels → network input; normalisation, encodings
  dataset.py                  memmap Dataset, centre crop, year split, augmentation
  model.py                    U-Net, depth 3, 32→64→128, bottleneck 256, ~1.9 M params
  losses.py                   masked BCE + SoftDice, FocalTversky
  train.py                    AdamW, cosine warm restarts, bf16, best-AUC-PR checkpoint
  baselines.py                persistence, dilated persistence, wind-directed growth
  evaluate.py                 test split only, threshold calibration, HTML reporting

data/
  spread_v5/                  the archive, 1096 shards (git-ignored)

models/
  spread_unet.pt              trained weights
  norm_stats.json             normalisation statistics, training split only

docs/
  GUIDE_EN.md                 36 000-word educational guide, English
  REHBER_TR.md                Turkish counterpart
  TANITIM.md                  outreach strategy

outputs/reports/              generated HTML reports, scorecards, maps
```

`paper/` is deliberately **not** tracked. This repository is public and that directory
holds parental consent forms, CVs of six minors, and sponsorship correspondence.

## 13. Limitations and future work / Kısıtlar ve gelecek çalışma

**The label is the ceiling.** The target is a MODIS thermal anomaly mask at 1 km on a
twice-daily overpass. It misses fires under cloud, fires smaller than the detection
threshold, and any spread that happens between overpasses. No architecture can predict
better than its labels can describe. Obtaining OGM fire-perimeter records — which are
mapped, not inferred — would raise that ceiling more than any modelling change.

**Resolution.** 1 km pixels cannot resolve a fire front. Sentinel-2 at 20 m would, at the
cost of a 5-day revisit instead of daily.

**No fuel moisture.** Live and dead fuel moisture are the strongest physical predictors of
spread rate and are not among the current inputs. `vpd`, `days_since_rain` and
antecedent precipitation are proxies for them, not measurements.

**Class imbalance remains extreme** at 1.0816 % of observed pixels on the test split
(0.2686 % of all pixels under the earlier accounting).

**Planned:** channel ablations to test whether wind actually contributes; comparison
against a physics-based spread model; evaluation against OGM perimeters if obtained.

## 14. References / Kaynakça

- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for
  Biomedical Image Segmentation. *MICCAI 2015*.
- Giglio, L., Schroeder, W., Justice, C. O. (2016). The Collection 6 MODIS active fire
  detection algorithm and fire products. *Remote Sensing of Environment*, 178, 31–41.
- Dozier, J. (1981). A method for satellite identification of surface temperature fields of
  subpixel resolution. *Remote Sensing of Environment*, 11, 221–229.
- Gorelick, N. et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for
  everyone. *Remote Sensing of Environment*, 202, 18–27.
- Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset
  for land applications. *Earth System Science Data*, 13, 4349–4383.
- Funk, C. et al. (2015). The climate hazards infrared precipitation with stations.
  *Scientific Data*, 2, 150066.
- Farr, T. G. et al. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics*,
  45, RG2004.
- Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.

A fuller bibliography, with the reasoning behind each choice, is in
[`docs/GUIDE_EN.md`](docs/GUIDE_EN.md) § 12.
