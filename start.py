#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IGNIS — Next-Day Fire Spread — single entry point / tek giriş noktası

Run `python start.py` and the whole project runs end to end:

  0. Environment      torch + ROCm GPU + numpy + matplotlib
  1. Data             data/spread_v5/*.tfrecord.gz
  2. Cache            TFRecord -> ~/ignis-cache/v5 memmap (skipped if current)
  3. Training         U-Net, best checkpoint by validation AUC-PR
  4. Evaluation       TEST split only, against recomputed baselines
  5. Summary          where the outputs are, and whether the model won

Nothing here reports success without checking for it. Every step verifies its
own result before moving on, because the failures this project has actually hit
were all the same shape: something claimed to succeed while doing nothing.

Tek komut: `python start.py`. Her adım kendi sonucunu doğrular; bu projede
yaşanan hataların hepsi "başarılı dedi ama hiçbir şey yapmadı" biçimindeydi.

Usage / Kullanım
----------------
    python start.py                    # everything / hepsi
    python start.py --epochs 40        # shorter training run
    python start.py --only eval        # one stage: check|data|cache|train|eval
    python start.py --skip-train       # evaluate the existing checkpoint
    python start.py --cpu              # no GPU (very slow, for debugging only)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
VERSION = "v5"
SPREAD_DIR = BASE_DIR / "data" / f"spread_{VERSION}"
MODEL_FILE = BASE_DIR / "models" / "spread_unet.pt"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

sys.path.insert(0, str(SRC_DIR))


class C:
    BLUE = "\033[94m"; CYAN = "\033[96m"; GREEN = "\033[92m"
    YELLOW = "\033[93m"; RED = "\033[91m"; ENDC = "\033[0m"; BOLD = "\033[1m"


def header():
    print(f"{C.CYAN}{C.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        IGNIS — YANGIN BÜYÜME TAHMİN SİSTEMİ              ║")
    print("║        Next-Day Fire Spread — U-Net segmentation         ║")
    print(f"║        schema {VERSION}                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(C.ENDC)


def section(title, n=None):
    p = f"[{n}]" if n is not None else "→"
    print(f"\n{C.BOLD}{C.BLUE}{p} {title}{C.ENDC}")
    print(f"{C.BLUE}{'─' * 60}{C.ENDC}")


def ok(m):   print(f"{C.GREEN}✅ {m}{C.ENDC}")
def warn(m): print(f"{C.YELLOW}⚠️  {m}{C.ENDC}")
def err(m):  print(f"{C.RED}❌ {m}{C.ENDC}")
def info(m): print(f"{C.CYAN}ℹ️  {m}{C.ENDC}")


# ---------------------------------------------------------------- 0. environment
def check_environment(want_gpu=True):
    section("ENVIRONMENT / ORTAM", 0)
    missing = []
    for pkg, name in (("torch", "PyTorch (ROCm)"), ("numpy", "NumPy"),
                      ("matplotlib", "Matplotlib")):
        try:
            __import__(pkg)
            ok(f"{name}")
        except ImportError:
            err(f"{name} MISSING / YOK")
            missing.append(pkg)
    if missing:
        info("sudo pacman -S rocm-hip-sdk rocminfo python-pytorch-rocm \\")
        info("               python-scikit-learn python-matplotlib python-scipy")
        sys.exit(1)

    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        ver = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", "?")
        ok(f"GPU: {name}  (ROCm/HIP {ver}, torch {torch.__version__})")
        # gfx1201 is natively supported in ROCm 7.2; the override breaks it.
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            warn("HSA_OVERRIDE_GFX_VERSION is set — unset it on RDNA 4 / RX 9070 XT")
    elif want_gpu:
        err("No GPU visible to PyTorch / PyTorch GPU göremiyor")
        info("Check with: python src/device.py")
        info("Or run on CPU (very slow): python start.py --cpu")
        sys.exit(1)
    else:
        warn("Running on CPU — this is for debugging only / sadece hata ayıklama")


# ---------------------------------------------------------------- 1. data
def check_data():
    section(f"DATA / VERİ  (data/spread_{VERSION}/)", 1)
    SPREAD_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(SPREAD_DIR.glob("*.tfrecord.gz"))
    if not shards:
        err(f"data/spread_{VERSION}/ is empty / boş")
        print()
        info("To generate the archive / arşivi üretmek için:")
        print(f"   {C.YELLOW}1){C.ENDC} open noteboks/colab_notebook_{VERSION}.ipynb in Google Colab")
        print(f"   {C.YELLOW}2){C.ENDC} run all cells (GEE project id + authentication)")
        print(f"   {C.YELLOW}3){C.ENDC} download Drive > GEE_FireSpread_{VERSION}/*.tfrecord.gz")
        print(f"   {C.YELLOW}4){C.ENDC} put them in data/spread_{VERSION}/ and rerun: "
              f"{C.BOLD}python start.py{C.ENDC}")
        return False

    gb = sum(s.stat().st_size for s in shards) / 1024 ** 3
    years = sorted({s.name.split("_")[1][:4] for s in shards if "_" in s.name})
    ok(f"{len(shards)} shards, {gb:.1f} GB")
    ok(f"years / yıllar: {', '.join(years)}")
    # A schema mix is silently wrong: the loader rebuilds the channel axis from
    # band ORDER alone. Şema karışımı sessizce yanlış sonuç verir.
    for other in BASE_DIR.glob("data/spread*"):
        if other.is_dir() and other != SPREAD_DIR and any(other.glob("*.tfrecord.gz")):
            warn(f"another archive present: {other.name} — never mix schemas")
    return True


# ---------------------------------------------------------------- 2. cache
def run_cache(verify=False):
    section("CACHE / ÖNBELLEK  (TFRecord -> memmap)", 2)
    import tfrecord_to_npy
    t0 = time.time()
    out = tfrecord_to_npy.convert(version=VERSION, verify=verify)
    arr, meta, man = tfrecord_to_npy.load_cache(VERSION)
    ok(f"{man['n_patches']} patches, {len(man['bands'])} bands  ({time.time() - t0:.1f}s)")
    info(f"cache: {out}")
    return man


# ---------------------------------------------------------------- 3. training
def _training_already_running():
    """Is another process already holding the GPU?
    Başka bir süreç GPU'yu tutuyor mu?

    Matching on script names is unreliable — a run launched through a wrapper
    is still a run. Ask the real question instead: who else has /dev/kfd open,
    which is the AMD compute device every ROCm process must hold.
    """
    me = os.getpid()
    busy = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == me:
            continue
        try:
            if not any(fd.resolve(strict=False).name == "kfd"
                       for fd in (proc / "fd").iterdir()):
                continue
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        except (PermissionError, FileNotFoundError, OSError):
            continue
        if cmd:
            busy.append(f"{proc.name}  {cmd[:90]}")
    return busy


def run_training(epochs=None, cpu=False, workers=8, force=False):
    section("TRAINING / EĞİTİM", 3)

    if MODEL_FILE.exists() and not force:
        warn(f"A checkpoint already exists: {MODEL_FILE}")
        info("Retrain with --force-retrain, or evaluate it with --skip-train")
        return False

    import train
    kw = dict(version=VERSION, cpu=cpu, workers=workers)
    if epochs:
        kw["epochs"] = epochs
    t0 = time.time()
    train.train(**kw)
    if not MODEL_FILE.exists():
        # No checkpoint means no epoch ever improved on the initial score.
        err("Training finished but wrote no checkpoint / kontrol noktası yazılmadı")
        return False
    ok(f"training complete ({(time.time() - t0) / 60:.1f} min)")
    return True


# ---------------------------------------------------------------- 4. evaluation
def run_evaluation():
    section("EVALUATION / DEĞERLENDİRME  — TEST split only", 4)
    if not MODEL_FILE.exists():
        err(f"No checkpoint at {MODEL_FILE} — train first / önce eğitin")
        return None
    import evaluate
    metrics, baselines = evaluate.evaluate(version=VERSION, split="test",
                                           model_path=MODEL_FILE,
                                           report_dir=REPORT_DIR)
    print()
    ok(f"model   IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  "
       f"AUC-PR={metrics['ap']:.4f}  ROC-AUC={metrics.get('roc', float('nan')):.4f}")

    # A score means nothing without the bar it has to clear.
    # Bir skor, aşması gereken çizgi olmadan anlamsızdır.
    best_name, best = max(baselines.items(), key=lambda kv: kv[1]["iou"])
    info(f"best baseline / en iyi temel çizgi: {best_name}  "
         f"IoU={best['iou']:.4f}  F1={best['f1']:.4f}")
    if metrics["beats_baseline"]:
        ok("MODEL BEATS ITS BASELINE / MODEL TEMEL ÇİZGİYİ GEÇİYOR")
    else:
        err("MODEL LOSES TO ITS BASELINE / MODEL TEMEL ÇİZGİYİ GEÇEMİYOR")
        info("Do not report this as a working predictor. Diagnose first.")
        info("Bunu çalışan bir tahminci olarak sunmayın; önce nedenini bulun.")
    return metrics


# ---------------------------------------------------------------- 5. summary
def summary():
    section("SUMMARY / ÖZET")
    for name, p in (("U-Net checkpoint", MODEL_FILE),
                    ("metrics (json)", REPORT_DIR / "spread_metrics.json"),
                    ("report (HTML)", REPORT_DIR / "spread_report.html"),
                    ("scorecard (PNG)", REPORT_DIR / "spread_scorecard.png"),
                    ("figures (PNG)", REPORT_DIR / "spread_figures.png"),
                    ("map (HTML)", REPORT_DIR / "spread_map.html")):
        (ok if Path(p).exists() else warn)(f"{name}: {p}")


def main():
    ap = argparse.ArgumentParser(description="IGNIS end-to-end runner")
    ap.add_argument("--only", choices=["check", "data", "cache", "train", "eval"],
                    help="run a single stage / tek aşama")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="CRC-check every record while caching (slow)")
    a = ap.parse_args()

    header()
    try:
        if a.only == "check":
            return check_environment(want_gpu=not a.cpu)
        if a.only == "data":
            return sys.exit(0 if check_data() else 1)
        if a.only == "cache":
            check_environment(want_gpu=not a.cpu); run_cache(a.verify); return
        if a.only == "train":
            check_environment(want_gpu=not a.cpu)
            run_training(a.epochs, a.cpu, a.workers, a.force_retrain); return
        if a.only == "eval":
            check_environment(want_gpu=not a.cpu); run_evaluation(); summary(); return

        check_environment(want_gpu=not a.cpu)
        if not check_data():
            sys.exit(0)          # told the user what to do; not an error
        run_cache(a.verify)

        busy = _training_already_running()
        if busy and not a.skip_train:
            warn("Another training process is already running:")
            for ln in busy:
                print(f"    {ln}")
            info("Two runs would compete for the GPU. Wait for it, or use --skip-train.")
            sys.exit(1)

        if not a.skip_train:
            run_training(a.epochs, a.cpu, a.workers, a.force_retrain)
        run_evaluation()
        summary()
        print(f"\n{C.GREEN}{C.BOLD}Done / Hazır{C.ENDC}\n")

    except KeyboardInterrupt:
        err("\nInterrupted by user / kullanıcı durdurdu")
        sys.exit(130)
    except Exception as e:
        err(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
