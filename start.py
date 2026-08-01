#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🔥 IGNIS — Yangın Büyüme (Next-Day Spread) Tahmin Sistemi — Ana Başlatıcı

`python start.py` çalıştırıldığında pipeline'ı DOĞRUDAN başlatır:
  0. Ortam kontrolü (gerekli paketler)
  1. Veri kontrolü (data/spread/*.tfrecord.gz)
  2. TFRecord -> memmap önbelleği (src/tfrecord_to_npy.py)
  3. U-Net eğitimi  (src/train.py)
  4. Değerlendirme — YALNIZCA test bölmesi (src/evaluate.py)
  5. Özet

Veri henüz yoksa (Colab'dan indirilmediyse) ne yapılacağını net şekilde söyler
ve hata vermeden durur. Tüm proje detayı için: README.md
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
SPREAD_DIR = BASE_DIR / "data" / "spread"
MODEL_FILE = BASE_DIR / "models" / "spread_unet.pt"


class C:
    HEADER = "\033[95m"; BLUE = "\033[94m"; CYAN = "\033[96m"
    GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
    ENDC = "\033[0m"; BOLD = "\033[1m"


def header():
    print(f"{C.CYAN}{C.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║     🔥 IGNIS — YANGIN BÜYÜME TAHMİN SİSTEMİ 🔥          ║")
    print("║        Next-Day Fire Spread (U-Net segmentasyon)         ║")
    print("║                                                          ║")
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


def check_environment():
    section("ORTAM KONTROLÜ", 0)
    required = {
        "torch": "PyTorch (ROCm)",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
    }
    missing = []
    for pkg, name in required.items():
        try:
            __import__(pkg)
            ok(f"{name} kuruldu")
        except ImportError:
            err(f"{name} YOK")
            missing.append(pkg)
    if missing:
        warn(f"Eksik paketler: {', '.join(missing)}")
        info("Kurulum:  sudo pacman -S rocm-hip-sdk rocminfo python-pytorch-rocm \\")
        info("                        python-scikit-learn python-matplotlib python-scipy")
        sys.exit(1)
    ok("Tüm paketler hazır!")


def check_data():
    """data/spread/ içinde TFRecord var mı?  Yoksa yönerge verip nazikçe dur."""
    section("VERİ KONTROLÜ (data/spread/)", 1)
    SPREAD_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(SPREAD_DIR.glob("*.tfrecord*"))
    if shards:
        total_mb = sum(s.stat().st_size for s in shards) / (1024 * 1024)
        ok(f"{len(shards)} TFRecord bulundu ({total_mb:.1f} MB)")
        return True

    warn("data/spread/ boş — henüz eğitim verisi yok.")
    print()
    info("Yangın büyüme verisini üretmek için:")
    print(f"   {C.YELLOW}1){C.ENDC} noteboks/colab_notebook.ipynb dosyasını Google Colab'da aç")
    print(f"   {C.YELLOW}2){C.ENDC} Hücreleri çalıştır (GEE proje ID + kimlik doğrulama)")
    print(f"   {C.YELLOW}3){C.ENDC} Drive > GEE_FireSpread_v2/ klasöründeki *.tfrecord.gz dosyalarını indir")
    print(f"   {C.YELLOW}4){C.ENDC} Bunları data/spread/ içine koy ve tekrar: {C.BOLD}python start.py{C.ENDC}")
    print()
    info("Ayrıntılı anlatım: README.md → 'Veri Üretimi' bölümü")
    return False


def run_cache():
    section("TFRECORD -> MEMMAP ÖNBELLEĞİ", 2)
    sys.path.insert(0, str(SRC_DIR))
    info("src/tfrecord_to_npy.py çalıştırılıyor...")
    t0 = time.time()
    import tfrecord_to_npy
    tfrecord_to_npy.convert(verify=True)
    ok(f"Önbellek hazır ({time.time() - t0:.1f}s)")


def run_training():
    section("U-NET EĞİTİMİ", 3)
    sys.path.insert(0, str(SRC_DIR))
    info("src/train.py çalıştırılıyor...")
    t0 = time.time()
    import train
    train.train()
    ok(f"Eğitim tamamlandı ({time.time() - t0:.1f}s)")


def run_evaluation():
    section("DEĞERLENDİRME (SONUÇLAR)", 4)
    sys.path.insert(0, str(SRC_DIR))
    import evaluate
    info("Model AYRILMIŞ TEST bölmesiyle değerlendiriliyor...")
    metrics, baselines = evaluate.evaluate()
    ok(f"IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  "
       f"AUC-PR={metrics['ap']:.4f}")
    # Bir skor, ancak asmasi gereken temel cizgiyle birlikte anlamlidir.
    # A score is only meaningful next to the bar it has to clear.
    best = max(baselines.items(), key=lambda kv: kv[1]["iou"])
    if metrics["beats_baseline"]:
        ok(f"Model en iyi temel çizgiyi GEÇİYOR "
           f"({best[0]}, IoU={best[1]['iou']:.4f})")
    else:
        err(f"Model temel çizgiyi GEÇEMİYOR: "
            f"{best[0]} IoU={best[1]['iou']:.4f} > model {metrics['iou']:.4f}")


def summary():
    section("ÖZET")
    rep = BASE_DIR / "outputs" / "reports"
    items = [
        ("U-Net modeli", MODEL_FILE),
        ("📄 Makale raporu (HTML)", rep / "spread_report.html"),
        ("🎯 Basit skor kartı", rep / "spread_scorecard.png"),
        ("🖼️  Detaylı figürler", rep / "spread_figures.png"),
        ("🗺️  Harita", rep / "spread_map.html"),
    ]
    for name, p in items:
        (ok if Path(p).exists() else warn)(f"{name}: {p}")
    print()
    info("Sunum için 'spread_report.html'i tarayıcıda aç; slaytlara 'spread_scorecard.png' ve "
         "'spread_figures.png' koy.")


def main():
    try:
        header()
        check_environment()
        if not check_data():
            # Veri yok: kullanıcıyı yönlendirdik, hata değil.
            sys.exit(0)
        run_cache()
        run_training()
        run_evaluation()
        summary()
        print(f"\n{C.GREEN}{C.BOLD}🔥 Hazır!{C.ENDC}\n")
    except KeyboardInterrupt:
        err("\nKullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as e:
        err(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        info("Sorun giderme için README.md → 'Hata Ayıklama' bölümüne bakın.")
        sys.exit(1)


if __name__ == "__main__":
    main()
