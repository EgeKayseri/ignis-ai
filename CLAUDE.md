# IGNIS — working notes for Claude Code

## What this is

An Earth Observation system that predicts **next-day wildfire spread** over Türkiye:
given a fire that is already burning on day *t*, predict which 1 km pixels will be
burning on day *t*+1. A U-Net consumes a fire-centred patch of environmental driver
channels and emits a per-pixel probability map.

This is **not** fire susceptibility mapping ("where might a fire start"). That is a
static problem with published ROC-AUC above 0.93. This is the temporal problem, and it
is much harder. Never conflate the two — the distinction is load-bearing in the paper.

- **Paper:** `IAC-26,B1,IP,107,x110901`, IAF Earth Observation Symposium (B1),
  Interactive Presentations. 77th International Astronautical Congress,
  Antalya, 5–9 October 2026.
- **Final manuscript deadline: 14 September 2026.** Everything is scheduled against this.
- **Authors:** six students at Antalya Yusuf Ziya Öner Science High School. Explain
  reasoning; do not assume ML or remote-sensing background.
- **Repo:** `github.com/WhiteFoxYT/ignis-ai`
- **Manuscript:** `paper/IGNIS_IAC2026_manuscript.docx` (read it before changing any
  scientific claim — extract `word/document.xml` with `zipfile`).

## Current status — read this first

**Snapshot: 9 August 2026.** Update this section when the situation changes; a new
session should be able to start work from it alone.

### Where the project lives

**`~/Projects/ignis` on ext4.** This is now the working copy — not `/mnt/windows/PROJE/ignis`.
The NTFS mount went read-only and then unmounted entirely mid-session because Windows
was left **hibernated** (`ntfs-3g`: *"Windows is hibernated, refused to mount"*). Until
Fast Startup is disabled and Windows is shut down fully, that mount is read-only at best.
`~/Projects/ignis` has no such problem and is where everything runs.

### Where the project actually is

| Thing | State |
|---|---|
| Pipeline code | **Ported to PyTorch, complete, runs end to end.** |
| GPU | **Working.** `AMD Radeon RX 9070 XT`, ROCm/HIP 7.2.53211, torch 2.13.0, bf16. |
| U-Net | **Verified 1,935,617 params** at 28 input channels (v5), bottleneck 45.8 %. Matches the manuscript's ~1.9 M. |
| v5 archive | **Landed and verified.** 1096 shards, 2019–2026, in `data/spread_v5/`. |
| Cache | `~/ignis-cache/v5` — 67,056 patches × 26 bands, 27.4 GB. 0 short, 0 missing, 0 NaN. |
| Splits | train 40,215 / val 13,049 / test 13,792. |
| Training | **First real run in progress.** No test result yet. |
| Reported results | Still the v1 numbers below. **Nothing new has been measured.** |

### What v5 actually fixed, measured

| Check | v2 | v5 |
|---|---|---|
| Training patches | 10,595 | **40,215** (2019–2020 now present) |
| Patches with an empty target | 58.9 % | **47.5 %** |
| Positive prevalence | 0.27 % of all pixels | **1.23 % of observed pixels** |
| Environmental zero-rate spread | — | **18.79 pp** — no uniform-zero signature |

GEE produced **no shards at all** between 2022-10-10 and 2022-10-22, independently
confirming the Terra outage; `KNOWN_OUTAGES` therefore had nothing to skip.

### The immediate next step

```bash
python start.py                 # environment -> data -> cache -> train -> evaluate
```

**Acceptance criterion, agreed with the team:** the model must beat the persistence
baseline. `evaluate.py` recomputes the baselines on the **same pixels** as the model,
and that same-pixel number is the real bar — the v1 figures (IoU 0.0306, F1 0.0595)
were measured on different data and are only a reference point. If the model does not
clear it, stop and diagnose. **Do not tune on the test split and do not report a number
that was not measured on it.**

### Open issues

- Training's validation AUC-PR peaked early (epoch 2) and drifted down while train loss
  kept falling — mild overfitting. Worth revisiting dropout / augmentation / early stop
  before treating any checkpoint as final.
- `describe_splits()` prints years as `np.int32(2022)` — cosmetic only.

### Recent history worth knowing

Three notebook bugs were found and fixed in sequence; the pattern matters more than
the details, because all three were the same shape — *something reported success while
doing nothing*.

1. **v3 finished in two minutes and exported nothing.** `DRIVE_FOLDER` was
   version-scoped but the Earth Engine task *description* was not, so the resume scan
   saw v2's ~1131 SUCCEEDED tasks and marked the whole range done.
2. **v4 exported nothing at all.** `_days_since` called `ee.Number.subtract(ee.Image)`
   (type error), and `ee.ImageCollection([]).max()` returns a band-less image whose
   `.rename()` fails. CHIRPS lags real time by weeks, so the empty case was the norm for
   2025–2026, not an edge case. **v4 was withdrawn and its notebook deleted.**
3. **The progress monitor reported the wrong version.** It matched `firespread_\d{8}$`,
   which does not match `firespread_v4_...`, so it silently counted old v2 tasks.

v5 therefore has a **preflight** (forces server-side evaluation of the full stack on
four probe dates, including the most recent — where v4 died) and a **one-day smoke
test**, both before any bulk submission. Submission is parallel (8 workers) and keeps a
durable ledger in Drive so a Colab disconnect cannot cause duplicate or lost days.

## Ground truth about current performance

These were **measured**, not estimated — 45 shards / 1054 patches sampled from the v1
archive with a pure-Python TFRecord reader. Do not soften or round them away.

| Quantity | Value |
|---|---|
| Model AUC-PR | 0.0210 |
| Model ROC-AUC | 0.8468 |
| Model Precision / Recall / F1 | 0.0601 / 0.0222 / 0.0324 |
| Model IoU | 0.0165 |
| **Persistence baseline ("tomorrow = today") IoU** | **0.0306** |
| **Persistence baseline F1** | **0.0595** |
| Positive pixel prevalence | 0.2686 % |
| Patch-level accuracy | 0.7714 |
| Majority-class ("extinguishing") share | 0.7774 |

**The model loses to persistence, and its patch accuracy is below the majority-class
baseline.** The reported "77 % accuracy" is an artefact of class prevalence. The
manuscript already states this honestly in Sections 4–5; keep it that way. Never quote
an improved number that has not been measured on the held-out split.

### Root causes found (the first two are absent from the manuscript's own diagnosis)

1. **No input normalisation.** `elevation` std 515.44 (max 4978), `aspect` 0–359,
   `landcover` 0–17 integer classes, next to `soil_moisture` std 0.07 and `ndvi` std
   0.20. The first convolution effectively sees only elevation and aspect.
2. **~15 % of every patch is a fabricated zero** — identical zero rate across all
   environmental bands, from `clip(REGION)` + `unmask(0)`. "Humidity = 0 %" was
   indistinguishable from "not observed". Fixed by the `valid` band (v2+).
3. **The target largely encodes satellite luck.** 58.9 % of patches have zero fire
   pixels on *t*+1 while 12.3 burn on average on *t*. Addressed by the ±1 day target.
4. **Patch far too large** — fires are ≤65 px, patches are 4225 px.
5. Evaluation was in-sample (`evaluate_spread.py` globbed all shards including
   training days), so even the bad numbers above are optimistic.

## Dataset versions

**Only v5 is in the project.** The older archives were moved out to `~/ignis-archive/`
so nothing stale can be loaded or quoted by accident. They are still on disk and can be
deleted whenever — they cost hours of GEE export to regenerate, which is the only
reason they were kept.

`tfrecord_to_npy.py` still supports every schema, and the historical contracts remain
documented below, because the v1 numbers in this file were measured on v1 data.

| Version | Location | Input bands | Notes |
|---|---|---|---|
| **v5** | **`data/spread_v5/`** | **21** | **Live.** 1096 shards, 2019–2026. Adds `days_since_rain`, `burn_age`, `valid_next`/`valid_next2`. |
| v3 | archived | 19 | v2 plus temporal context and fire weather. Never used for a result. |
| v2 | archived | 14 | 2019–2026 nominally, but **no 2019 or 2020 on disk**. |
| v1 | archived | 14 | Original archive, 2019 – 26 Jul 2021. The measured numbers below are from this. |

**Never mix schemas in one directory** — the loader reconstructs the channel axis from
band order alone, so a mixed directory is silently wrong rather than an error.
`tfrecord_to_npy.py` now detects the schema from feature *names* and refuses.

v5 is generated by `noteboks/colab_notebook_v5.ipynb`, which is resumable: re-running
skips days already in Drive, already queued, or recorded in the Drive-side ledger.

### Band contract (v5 — the live one)

Order is contractual across the notebook, `src/gee_config.py` and `src/config.py`.
Change it in all three or not at all. Verified equal across all three for every schema.

```
ndvi lst air_temp humidity vpd
wind_speed wind_u wind_v
precip precip_7d precip_30d days_since_rain
soil_moisture
elevation slope aspect landcover
burn_age
fire_prev2 fire_prev1 fire                       <- 21 input bands
fire_next fire_next2                             <- target
valid valid_next valid_next2                     <- observation validity
```

v3 omits `days_since_rain`, `burn_age`, `valid_next`, `valid_next2` (19 input bands).
v2 also omits `vpd`, `precip_7d`, `precip_30d`, `fire_prev1`, `fire_prev2` (14 input).

After feature engineering the network sees 21 channels on v2 and **28 on v5**
(`aspect` becomes sin+cos, `landcover` becomes 6 fuel groups, `valid` is fed as input).

### v5: `valid_next` and the observation mask

MODIS `FireMask` encodes observation quality (0/1/2 not processed, 4 cloud,
6 unknown, 3/5 observed non-fire, 7/8/9 fire). v2 and v3 did `fm.gte(7).unmask(0)`,
which labels a **clouded** pixel as *no fire*. That is a significant part of root
cause 3 and it was our own code, not a MODIS limitation.

**v4 was withdrawn.** It introduced the right idea but crashed server-side and
exported nothing: `_days_since` called `ee.Number.subtract(ee.Image)` (a type
error) and `ee.ImageCollection([]).max()` returns a band-less image whose
`.rename()` fails — the same failure class as the Terra October 2022 outage. CHIRPS
lags real time by weeks, so the empty-collection case was the norm for 2025-2026
dates, not an edge case. v5 fixes that, version-scopes the progress monitor (v4
still matched `firespread_\d{8}$`, so it reported OLD v2 tasks and every run looked
successful), and adds a preflight that forces server-side evaluation plus a
one-day smoke test before any bulk submission.

v5 exports `valid_next`/`valid_next2`. `dataset.py` masks the loss with an
asymmetric rule: a detection is trusted on its own, an absence only when the day
was actually observed. `tfrecord_to_npy.py` now also detects the schema from the
record's feature NAMES and refuses a mixed directory, so the positional band
contract can no longer be violated silently.

### Known data exclusion

Terra MODIS acquired **no data 10–19 October 2022** (Constellation Exit Manoeuvres;
instruments recovering through 21 Oct). `MOD11A1`'s 3-day compositing window falls
entirely inside the outage for several days, producing a band-less image that fails at
`.rename()`. Those days are excluded via `KNOWN_OUTAGES` rather than filled, because
filling would fabricate two of the input channels. This is ~5 days of ~1136 (0.4 %) and
should be stated in the paper.

## Target definition

```python
target = max(fire_next, fire_next2)      # fire activity within the next 24–48 h
```

Chosen deliberately over the strict *t*+1 mask because of root cause 3. The strict mask
is still exported so both definitions remain available and comparable. The paper framing
becomes "next 24–48 h fire activity"; the title and author list are fixed by IAF rules
and do not change.

Patch growth class, applied identically to observed and predicted masks:

```
r = N(t+1) / max(N(t), 1)
r > 1.25 -> growing ; 0.75 <= r <= 1.25 -> stable ; r < 0.75 -> extinguishing
```

## Stack and environment

- **Arch Linux**, kernel 7.1.5-zen, Ryzen 7 7800X3D (8 cores), 30 GB RAM.
- **GPU: AMD RX 9070 XT (Navi 48, gfx1201, RDNA4).** `/dev/kfd` is already mode 666 —
  no group changes needed.
- **PyTorch + ROCm, not TensorFlow.** `extra/python-pytorch-rocm` is built for system
  Python 3.14.6 and depends on `rocm-hip-sdk 7.2.4`. There is no TensorFlow wheel for
  Python 3.14 and AMD's TF path is Docker-only. Install:
  ```bash
  sudo pacman -S rocm-hip-sdk rocminfo python-pytorch-rocm \
                 python-scikit-learn python-matplotlib python-scipy
  ```
  gfx1201 is natively supported in ROCm 7.2 — do **not** set `HSA_OVERRIDE_GFX_VERSION`.
  Use `bfloat16` autocast, not `float16`.
- **The working copy is `~/Projects/ignis` on ext4.** The old location
  `/mnt/windows/PROJE/ignis` is an NTFS fuseblk mount that is read-only while Windows is
  hibernated, and it unmounted itself under memory pressure. Do not work there.
- **Python is 3.14**, which changed the default multiprocessing start method from `fork`
  to **`forkserver`**. Anything a `DataLoader` worker receives is now *pickled*, and
  `np.memmap` pickles as a fully materialised array. `SpreadDataset` therefore defines
  `__getstate__`/`__setstate__` to drop and reopen the handle; without them every worker
  loads all 27 GB and the machine OOMs. Keep any new dataset attribute picklable.
- TFRecords are converted once into a memory-mapped cache under `~/ignis-cache/`
  and training reads from there.
- **`rm` AND `cp` are aliased to sudo-requiring safe wrappers** in this user's zsh.
  Use `/usr/bin/rm` and `/usr/bin/cp` for anything scripted. A bare `cp` fails on
  the sudo prompt *after* partially running, and leaves a stray directory named
  after its destination argument in the cwd — the `0/` directory once found at the
  repo root came from exactly that.

## Conventions

- Comments and docs are **bilingual, English first then Turkish**, matching the
  notebooks. The team presents in English but works in Turkish.
- Prose is formal and scientific — this material feeds an IAC manuscript.
- Cite concrete measured numbers, never vague qualifiers. "0.27 %, about 11 of 4225
  pixels", not "very sparse".
- Never invent a metric. Unmeasured results are marked
  "to be filled in once training is complete".
- Normalisation statistics come from the **training split only**. Taking them from
  validation or test is leakage.

## Repository map

```
start.py                           single entry point: check -> cache -> train -> evaluate
noteboks/colab_notebook_v5.ipynb   GEE export, v5 schema (21 input bands) — the only one kept
src/config.py                      all constants; SPREAD_* section is the live one
src/gee_config.py                  GEE collections and band contract
src/device.py                      ROCm device selection, bfloat16 autocast
src/tfrecord_to_npy.py             TFRecord -> ~/ignis-cache memmap (pure-Python
                                   protobuf reader, no TensorFlow)
src/features.py                    raw bands -> network input; z-score (train split
                                   only), aspect sin/cos, landcover one-hot, log1p
src/dataset.py                     memmap Dataset, centre crop, year split,
                                   direction-aware augmentation
src/model.py                       U-Net, architecture preserved from TF layer-for-layer
src/losses.py                      masked BCE+SoftDice (default), FocalTversky
src/train.py                       AdamW, cosine warm restarts, bf16, best val AUC-PR
src/baselines.py                   persistence, dilated, wind-directed growth
src/evaluate.py                    TEST SPLIT ONLY, threshold calibrated on val,
                                   HTML/scorecard/folium ported from evaluate_spread.py
docs/GUIDE_EN.md                   36 k-word educational guide, English
docs/REHBER_TR.md                  same guide, Turkish (3058 lines)
docs/TANITIM.md                    outreach strategy: validate first, publicise second
docs/sunum.html                    lay-audience presentation (also published as an Artifact)
paper/                             manuscript, IAC guidelines, admin documents
```

The eight static-risk modules (`preprocess`, `train`, `predict`, `test_accuracy`,
`map_visualization`, `main`, `examples`, `gee_data_processor`) and their model weights
were deleted — they belonged to an abandoned susceptibility model.

Moved out to `~/ignis-archive/` on 9 August 2026 so nothing stale can be quoted as a
current result: the v1/v2/v3 archives, `data/raw` and `data/processed` (CSVs of the
abandoned tabular model), every file that was in `outputs/`, `models/spread_unet.keras`
(the old TensorFlow model), the v2 and v3 notebooks, the stale `graphify-out/`, and
`src/utils.py` (imported by nothing). The broken 3.8 GB `venv/` was deleted outright.
**`outputs/` is now empty by design** — anything in it was produced by the current code.

The four legacy TensorFlow modules (`spread_dataset.py`, `spread_model.py`,
`train_spread.py`, `evaluate_spread.py`) were deleted once the PyTorch port landed.
The reporting half of `evaluate_spread.py` was ported into `evaluate.py`, not rewritten.

## Commands

```bash
python start.py                          # EVERYTHING: check -> cache -> train -> evaluate
python start.py --epochs 40              # shorter training run
python start.py --only eval              # one stage: check|data|cache|train|eval
python start.py --skip-train             # evaluate the existing checkpoint
python start.py --force-retrain          # overwrite an existing checkpoint

python src/device.py                     # confirm the GPU is visible to PyTorch
python src/tfrecord_to_npy.py --verify   # TFRecord -> memmap cache + integrity report
python src/dataset.py                    # patch counts and prevalence per split
python src/model.py                      # architecture + parameter breakdown
python src/baselines.py                  # persistence / dilated / wind-directed
```

`start.py` refuses to start a second training while another process holds `/dev/kfd`,
so two runs cannot quietly compete for the GPU.

`SPREAD_VERSION` in `src/config.py` selects the schema (**v5** currently); every
script also takes `--version`.

## graphify

`graphify` is installed (`pipx install graphifyy`) and its skill is registered at
`~/.claude/skills/graphify/`. Run `/graphify .` to build a queryable knowledge graph of
the repo instead of grepping through files; then `graphify query`, `graphify path A B`
and `graphify explain X` against `graphify-out/graph.json`. Code parsing is local and
LLM-free. Prefer it over broad file sweeps on this repo.

## How to work with Claude on this project

Written for the humans as much as for the model. These are the habits that have
actually produced good results here.

### Start every session by saying what changed

The single most useful opening is a one-liner of current state: *"v5 data is now in
`data/spread_v5/`, 1120 shards"* or *"training finished, here is the output"*. This file
covers everything durable; only the volatile part needs restating.

### Ask for the diagnosis, not the fix

The three most valuable things produced here all came from *"why is this doing X?"*
rather than *"make it do Y"*: the v3 resume-scan bug, the v4 band-less-image crash, and
the discovery that `FireMask` already encodes cloud so a third of the empty targets were
self-inflicted. Describe the **symptom** precisely — "everything says succeeded but
Drive is empty", "it finishes in two minutes" — and let the cause be found.

### Insist on the baseline, every time

Any performance number is meaningless without the bar next to it. If a report ever comes
back with a model score and no baseline score, that is a bug in the report. `evaluate.py`
prints both by construction; keep it that way.

### Never accept an unmeasured number

If a number is not in `outputs/reports/spread_metrics.json` or measured in-session, it
does not exist. "Probably around 0.05" is worse than "not measured". This rule is why
the README's section 9.3 is still blank.

### Say when a guide or doc is internal

Long-form docs here are for the team, not for publication. Ask for compact unless the
document is genuinely going outside — a full-length treatment costs a great deal for
something only six people will read.

### Things that will bite

- `rm` and `cp` are sudo-wrapper aliases. Scripted use must be `/usr/bin/rm`,
  `/usr/bin/cp`. A bare `cp` leaves a stray directory named after its destination.
- `paper/` must never be committed. The repo is public and that directory holds consent
  forms and CVs of minors. Check `git diff --cached --name-only | grep ^paper/` before
  every commit.
- Dataset schemas must never share a directory. `tfrecord_to_npy.py` now enforces this
  by detecting the schema from feature names, but put v5 in `data/spread_v5/`.
- Colab disconnecting does **not** kill Earth Engine exports — they run on Google's
  servers. Only submission stops.
- **Never hand a `np.memmap` to a `DataLoader` worker.** Under Python 3.14's `forkserver`
  default it is pickled as a real array and each worker allocates the whole cache. This
  already caused three OOM kills at ~27 GB resident each, and took the NTFS mount down
  with it. `num_workers=0` masks the bug — it looks merely slow.
- **A memory alarm during training is not automatically "just page cache".** Check
  `dmesg | grep -i "killed process"` before saying so; here it was genuine RSS.

### Good openers

```
"Read CLAUDE.md, then <task>."
"v5 data has landed — <n> shards. Convert, train, evaluate, and report against
 the persistence baseline. Stop if it loses."
"<paste error>. What is actually failing here?"
"Update the manuscript section on <x> using only measured numbers."
```

## Outreach position

Decided with the team: **validate first, publicise second.** Do not present this to OGM,
AFAD or the press as an operational prediction system while it loses to persistence. The
defensible framing today is "a reproducible pipeline and an honestly reported baseline".

When the model does beat its baselines, approach OGM as a **data request** — asking for
their fire-perimeter records for validation — rather than as a solution offer. Their
perimeter data would replace the MODIS thermal-anomaly target, which is the single
biggest ceiling on accuracy.

The genuinely publishable story right now, requiring no accuracy claim at all: a high
school team from Antalya had a paper accepted to IAC 2026, which is being held in Antalya.
