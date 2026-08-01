"""
Yangın Büyüme Veri Seti — TFRecord Yükleyici
============================================
Google Earth Engine'in (noteboks/colab_notebook.ipynb) Drive'a yazdığı TFRecord
yamalarını okur ve TensorFlow eğitim hattına dönüştürür.

Her TFRecord kaydı (her aktif-yangın karesi) şu tensörleri içerir:
  - INPUT_BANDS'teki 14 kanal (her biri PATCH_SIZE x PATCH_SIZE float)
  - TARGET_BAND ('fire_next') : ertesi günün yangın maskesi (PATCH_SIZE x PATCH_SIZE)

Bu modül:
  1. Kayıtları (X, y) çiftlerine çözer;  X: (H, W, 14),  y: (H, W, 1)
  2. Yamayı MODEL_PATCH x MODEL_PATCH (64) boyutuna kırpar
  3. Her yama için grow/stable/extinguish sınıfını türetir
  4. tf.data.Dataset döndürür (eğitim/doğrulama)

SAĞLAMLIK / BUGFIX:
  GEE, maskeli (bulut/no-data) veya kenar piksellerin bulunduğu yamalarda ARRAY'i
  KISALTABİLİR. Bu durumda sabit uzunluklu (FixedLenFeature) çözümleme şu hatayı verir:
      "Can't parse serialized Example ... Some float_list slice was not parsed. Key: fire"
  Çözüm: bantları VarLenFeature ile oku, uzunluğu PATCH_SIZE^2'e eşit OLMAYAN
  (bozuk/kısa) kayıtları filtreyle at. Böylece tek bir bozuk kare eğitimi düşürmez.
  (Ayrıca notebook artık yığını unmask(0, False) ile dışa aktardığı için yeni
   çekilen verilerde kısa yama oluşmaz.)

Not: TensorFlow olmadan da içe aktarılabilsin diye tf importu fonksiyon içinde yapılır.
"""

from __future__ import annotations
import glob

from config import (
    SPREAD_INPUT_BANDS, SPREAD_TARGET_BAND, PATCH_SIZE, MODEL_PATCH,
    GROWTH_GROW_RATIO, GROWTH_STABLE_LOW, SPREAD_TFRECORD_GLOB,
    SPREAD_SHUFFLE_BUFFER, SPREAD_MAX_FILES,
)


def _list_files(glob_pattern):
    """TFRecord dosyalarını sırala; SPREAD_MAX_FILES>0 ise ilk N tanesini al (hızlı deneme)."""
    files = sorted(glob.glob(glob_pattern))
    if SPREAD_MAX_FILES and SPREAD_MAX_FILES > 0:
        files = files[:SPREAD_MAX_FILES]
    return files

ALL_BANDS = SPREAD_INPUT_BANDS + [SPREAD_TARGET_BAND]
_N = PATCH_SIZE * PATCH_SIZE


# ------------------------------------------------------------------
# TFRecord çözümleme (değişken uzunluğa dayanıklı)
# ------------------------------------------------------------------
def _feature_description():
    import tensorflow as tf
    # VarLenFeature: hem tam hem kısa listeleri okur (FixedLen'in tersine hata vermez)
    return {b: tf.io.VarLenFeature(tf.float32) for b in ALL_BANDS}


def _parse_and_validate(proto):
    """Kaydı çöz, her bandı yoğunlaştır, uzunlukların doğruluğunu işaretle."""
    import tensorflow as tf
    parsed = tf.io.parse_single_example(proto, _feature_description())
    dense = {b: tf.sparse.to_dense(parsed[b]) for b in ALL_BANDS}
    valid = tf.reduce_all(
        tf.stack([tf.equal(tf.size(dense[b]), _N) for b in ALL_BANDS]))
    return dense, valid


def _to_xy(dense):
    import tensorflow as tf
    chans = [tf.reshape(dense[b], (PATCH_SIZE, PATCH_SIZE)) for b in SPREAD_INPUT_BANDS]
    x = tf.stack(chans, axis=-1)                              # (H, W, 14)
    y = tf.reshape(dense[SPREAD_TARGET_BAND], (PATCH_SIZE, PATCH_SIZE))
    y = tf.expand_dims(y, axis=-1)                            # (H, W, 1)

    # NaN/eksik -> 0
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    y = tf.where(tf.math.is_finite(y), y, tf.zeros_like(y))
    y = tf.clip_by_value(y, 0.0, 1.0)

    # 65x65 -> 64x64 (sol-üst köşe kırpma)
    x = x[:MODEL_PATCH, :MODEL_PATCH, :]
    y = y[:MODEL_PATCH, :MODEL_PATCH, :]
    return x, y


def growth_class(x, y):
    """Yamadaki bugünkü ('fire' kanalı) ve yarınki (y) yangın piksel sayısından
    grow(2)/stable(1)/extinguish(0) sınıfını üretir."""
    import tensorflow as tf
    fire_idx = SPREAD_INPUT_BANDS.index("fire")
    n_today = tf.reduce_sum(x[..., fire_idx])
    n_next = tf.reduce_sum(y[..., 0])
    ratio = n_next / tf.maximum(n_today, 1.0)
    cls = tf.where(ratio > GROWTH_GROW_RATIO, 2,
                   tf.where(ratio >= GROWTH_STABLE_LOW, 1, 0))
    return tf.cast(cls, tf.int32)


# ------------------------------------------------------------------
# tf.data hattı
# ------------------------------------------------------------------
def _base_pipeline(files, shuffle):
    """Dosya listesinden çöz + bozuk kayıtları at + (X, y) üret."""
    import tensorflow as tf
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_and_validate, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda dense, valid: valid)               # kısa/bozuk yamaları at
    ds = ds.map(lambda dense, valid: _to_xy(dense),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(SPREAD_SHUFFLE_BUFFER, seed=42)
    return ds


def make_dataset(glob_pattern: str = SPREAD_TFRECORD_GLOB,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 with_growth_label: bool = False):
    """
    with_growth_label=False -> (X, y_mask)        segmentasyon (ertesi gün maskesi)
    with_growth_label=True  -> (X, (y_mask, cls)) segmentasyon + büyüme sınıfı
    """
    import tensorflow as tf
    files = _list_files(glob_pattern)
    if not files:
        raise FileNotFoundError(
            f"TFRecord bulunamadı: {glob_pattern}\n"
            f"Önce noteboks/colab_notebook.ipynb ile veri çekip "
            f"Drive'daki dosyaları data/spread/ altına indirin.")
    ds = _base_pipeline(files, shuffle)
    if with_growth_label:
        ds = ds.map(lambda x, y: (x, (y, growth_class(x, y))),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, files


def count_valid(files):
    """Bir dosya listesindeki GEÇERLİ (tam 65x65) kare sayısı."""
    import tensorflow as tf
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_and_validate, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda dense, valid: valid)
    return int(ds.reduce(0, lambda acc, _: acc + 1).numpy())


def train_val_split(glob_pattern: str = SPREAD_TFRECORD_GLOB,
                    batch_size: int = 32, val_frac: float = 0.2, repeat: bool = True):
    """
    Dosya bazında train/val ayrımı (aynı günün kareleri aynı bölmede kalır).

    Döndürür: (train_ds, val_ds, train_steps, val_steps)
    train/val veri setleri .repeat() edilir; böylece .filter() kaynaklı 'bilinmeyen
    kardinalite' nedeniyle Keras'ın "input ran out of data" ile erken durmasını önleriz.
    Bunun yerine steps_per_epoch/validation_steps açıkça verilir.
    """
    import math
    import tensorflow as tf
    files = _list_files(glob_pattern)
    if not files:
        raise FileNotFoundError(f"TFRecord bulunamadı: {glob_pattern}")
    n_val = max(1, int(len(files) * val_frac))
    val_files, train_files = files[:n_val], files[n_val:] or files[:1]

    n_train = count_valid(train_files)
    n_valid = count_valid(val_files)
    if n_train == 0:
        raise RuntimeError(
            "\n❌ GEÇERLİ eğitim karesi yok (0). data/spread/ içindeki TFRecord'lar\n"
            "   eski/bozuk olabilir. Güncel notebook ile veriyi yeniden çekin.")

    def build(fs, shuffle):
        ds = _base_pipeline(fs, shuffle)
        if repeat:
            ds = ds.repeat()
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = build(train_files, True)
    val_ds = build(val_files, False)
    train_steps = max(1, math.ceil(n_train / batch_size))
    val_steps = max(1, math.ceil(n_valid / batch_size))
    return train_ds, val_ds, train_steps, val_steps


# ------------------------------------------------------------------
# Hızlı kontrol
# ------------------------------------------------------------------
def inspect(glob_pattern: str = SPREAD_TFRECORD_GLOB, n: int = 200):
    """Veri setini özetle: kaç kare, sınıf dağılımı."""
    ds, files = make_dataset(glob_pattern, batch_size=1, shuffle=False,
                             with_growth_label=True)
    counts = {0: 0, 1: 0, 2: 0}
    seen = 0
    for x, (y, cls) in ds.take(n):
        counts[int(cls.numpy()[0])] += 1
        seen += 1
    print(f"Dosya sayısı        : {len(files)}")
    print(f"Geçerli kare (ilk {n}): {seen}")
    print(f"  extinguish (0)     : {counts[0]}")
    print(f"  stable     (1)     : {counts[1]}")
    print(f"  grow       (2)     : {counts[2]}")
    print(f"Girdi kanalı         : {len(SPREAD_INPUT_BANDS)} -> {SPREAD_INPUT_BANDS}")
    print(f"Yama                 : {MODEL_PATCH}x{MODEL_PATCH}")
    return counts


if __name__ == "__main__":
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else SPREAD_TFRECORD_GLOB
    try:
        inspect(pattern)
    except FileNotFoundError as e:
        print(e)
