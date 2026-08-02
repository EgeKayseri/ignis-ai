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

**Snapshot: 2 August 2026.** Update this section when the situation changes; a new
session should be able to start work from it alone.

### Where the project actually is

| Thing | State |
|---|---|
| Pipeline code | **Ported to PyTorch, complete, runs.** All modules execute on the GPU. |
| GPU | **Working.** `AMD Radeon RX 9070 XT`, ROCm/HIP 7.2.53211, torch 2.13.0, bf16 supported. |
| U-Net | **Verified 1,935,617 params**, bottleneck 45.8 % — matches the manuscript's ~1.9 M and the guide's "46 %". |
| v2 archive | 598 shards on disk, converted to `~/ignis-cache/v2` (10 GB, 0 short, 0 missing). **Missing 2019 and 2020.** |
| v5 archive | **Being generated.** Exports submitted from `colab_notebook_v5.ipynb`; `data/spread_v5/` still empty. |
| Training | **Not yet run for real.** Only a smoke test on v2. No result exists for the rebuilt pipeline. |
| Reported results | Still the v1 numbers below. Nothing new has been measured. |

### The immediate next step

When `data/spread_v5/` is populated:

```bash
python src/tfrecord_to_npy.py --version v5 --verify
python src/dataset.py
python src/baselines.py --version v5 --split test
python src/train.py --version v5
python src/evaluate.py --version v5
```

**Acceptance criterion, agreed with the team:** the model must beat the persistence
baseline (IoU 0.0306, F1 0.0595). If it does not, stop and diagnose. **Do not tune on
the test split and do not report a number that was not measured on it.**

### Open issues

- **v2 has no 2019 or 2020 data** (years present: 2021×1, 2022, 2023, 2024, 2025, 2026).
  The configured split is train 2019–2023 / val 2024 / test 2025–2026, so on v2 the
  training set is effectively 2021–2023. v5 is being regenerated over the full range;
  check the year coverage again once it lands rather than assuming it is complete.
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

Four schemas exist. **Never mix them in one directory** — the loader reconstructs the
channel axis from band order alone and short records are silently wrong.

| Version | Location | Input bands | Notes |
|---|---|---|---|
| v1 | `data/spread_v1_legacy/` | 14 | Original archive, 2019 – 26 Jul 2021, 360 shards. Superseded. |
| v2 | `data/spread/` | 14 | 2019–2026, ~1131 shards. Adds `fire_next2`, `valid`. |
| v3 | `data/spread_v3/` | 19 | v2 plus temporal context and fire weather. |
| v5 | `data/spread_v5/` | 21 | v3 plus `days_since_rain`, `burn_age`, `valid_next`/`valid_next2`. Current. |

Generated by `noteboks/colab_notebook.ipynb` (v2), `colab_notebook_v3.ipynb` (v3)
and `colab_notebook_v5.ipynb` (v5).
All are resumable: re-running skips days already in Drive, already queued, or
recorded in the Drive-side submission ledger.

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
- **The repo lives on `/mnt/windows`, an NTFS fuseblk mount.** Per-epoch I/O against it
  is slow. Convert TFRecords once into a memory-mapped cache under `~/ignis-cache/`
  (local ext4) and train from there.
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
noteboks/colab_notebook.ipynb      GEE export, v2 schema (14 input bands)
noteboks/colab_notebook_v3.ipynb   GEE export, v3 schema (19 input bands)
noteboks/colab_notebook_v5.ipynb   GEE export, v5 schema (21 input bands) — CURRENT
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

The four legacy TensorFlow modules (`spread_dataset.py`, `spread_model.py`,
`train_spread.py`, `evaluate_spread.py`) were deleted once the PyTorch port landed.
The reporting half of `evaluate_spread.py` was ported into `evaluate.py`, not rewritten.

## Commands

```bash
python src/device.py                     # confirm the GPU is visible to PyTorch
python src/tfrecord_to_npy.py --verify   # TFRecord -> memmap cache + integrity report
python src/dataset.py                    # patch counts and prevalence per split
python src/model.py                      # architecture + parameter breakdown
python src/train.py                      # train the U-Net
python src/baselines.py                  # persistence / dilated / wind-directed
python src/evaluate.py                   # TEST split only, calibrated threshold
python start.py                          # cache -> train -> evaluate, end to end
```

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
