"""
Yangın Büyüme Modeli — Eğitim
=============================
data/spread/ altındaki TFRecord yamalarıyla U-Net'i eğitir ve modeli
models/spread_unet.keras olarak kaydeder.

Kullanım:
    python src/train_spread.py
"""

from __future__ import annotations

import json
from config import (
    SPREAD_TFRECORD_GLOB, SPREAD_BATCH_SIZE, SPREAD_EPOCHS,
    SPREAD_MODEL_FILE, SPREAD_LOSS, ensure_directories_exist,
    SPREAD_EARLY_STOP_PATIENCE, SPREAD_REDUCE_LR_PATIENCE, REPORTS_DIR,
)
from spread_dataset import train_val_split
from spread_model import compile_model


def setup_device():
    """GPU varsa bildir + mixed precision aç; yoksa CPU (bu model için yeterli)."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        print(f"✅ GPU bulundu: {[g.name for g in gpus]}")
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("⚡ mixed_float16 etkin (daha hızlı + daha az VRAM)")
        except Exception as e:
            print(f"mixed precision atlandı: {e}")
    else:
        print("ℹ️  GPU görünmüyor — CPU'da eğitiliyor. Bu küçük U-Net için CPU yeterlidir.")
        print("   AMD GPU (RX 9070 XT) için: README → 'AMD GPU ile Çalıştırma' bölümü.")


def main():
    ensure_directories_exist()
    setup_device()

    print("TFRecord'lar yükleniyor...")
    # train_val_split .repeat() eder + adım sayılarını döndürür (BOŞ VERİ KORUMASI da
    # içeride: 0 geçerli kare varsa net hata verir).
    train_ds, val_ds, train_steps, val_steps = train_val_split(
        SPREAD_TFRECORD_GLOB, batch_size=SPREAD_BATCH_SIZE, val_frac=0.2)
    print(f"   train adım/epoch: {train_steps} | val adım: {val_steps}")

    print("U-Net kuruluyor...")
    model = compile_model(loss=SPREAD_LOSS)
    model.summary()

    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=SPREAD_EARLY_STOP_PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=SPREAD_REDUCE_LR_PATIENCE),
        tf.keras.callbacks.ModelCheckpoint(str(SPREAD_MODEL_FILE),
                                           monitor="val_auc", mode="max",
                                           save_best_only=True),
    ]

    print("Eğitim başlıyor...")
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=SPREAD_EPOCHS,
                     steps_per_epoch=train_steps, validation_steps=val_steps,
                     callbacks=callbacks)

    model.save(str(SPREAD_MODEL_FILE))
    print(f"✅ Model kaydedildi: {SPREAD_MODEL_FILE}")

    # Eğitim geçmişini kaydet (öğrenme eğrileri için)
    hist_path = REPORTS_DIR / "spread_history.json"
    hist_path.write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in hist.history.items()}),
        encoding="utf-8")
    print(f"📈 Eğitim geçmişi kaydedildi: {hist_path}")

    # En iyi doğrulama metriklerini yazdır (doğruluk burada görünür)
    h = hist.history
    if h.get("val_auc"):
        i = int(max(range(len(h["val_auc"])), key=lambda k: h["val_auc"][k]))
        print("\n📊 En iyi epoch doğrulama metrikleri:")
        print(f"   val AUC-PR : {h['val_auc'][i]:.4f}")
        if h.get("val_precision"): print(f"   val Precision: {h['val_precision'][i]:.4f}")
        if h.get("val_recall"):    print(f"   val Recall   : {h['val_recall'][i]:.4f}")
    print("   Ayrıntılı rapor için: python src/evaluate_spread.py")


if __name__ == "__main__":
    main()
