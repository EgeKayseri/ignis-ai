"""
Proje Konfigürasyonu
======================
Tüm sabitler ve konfigürasyon ayarları bu dosyada tanımlanmıştır.

Genel Bilgiler:
- Proje: Orman Yangını Tahmin Sistemi
- Amaç: Uydu ve meteorolojik verileri kullanarak yangın riskini tahmin et
- Version: 1.0.0
"""

# ============================================================
# TENSORFLOW UYARILARI - GİZLE (Normal başlangıç mesajları)
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sadece ERROR ve WARNING göster
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU memory dinamik büyüme
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0'ı kullan (varsa)

from pathlib import Path

# ============================================================
# PROJE DİZİNLERİ
# ============================================================

# Base proje dizini
BASE_DIR = Path(__file__).parent.parent

# Veri dizinleri
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SPREAD_DIR = DATA_DIR / "spread"  # YENİ: yangın büyüme TFRecord yamaları

# Model dizini
MODELS_DIR = BASE_DIR / "models"

# Çıktı dizinleri
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MAPS_DIR = OUTPUTS_DIR / "maps"

# Notebook dizini
NOTEBOOKS_DIR = BASE_DIR / "noteboks"

# ============================================================
# VERİ DOSYALARI
# ============================================================

# Giriş verileri
RAW_DATA_FILES = {
    "model_data": DATA_RAW_DIR / "yangin_model_verisi.csv",
    "random_data": DATA_RAW_DIR / "yangin_rastgele_verisi.csv",
}

# İşlenmiş veriler
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_data.csv"

# Model dosyaları
MODEL_FILE = MODELS_DIR / "yangin_model.keras"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

# Çıktı dosyaları
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"
CLASSIFICATION_REPORT = REPORTS_DIR / "classification_report.txt"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
TRAINING_HISTORY_FILE = REPORTS_DIR / "training_history.png"
FIRE_RISK_MAP = MAPS_DIR / "fire_risk_map.html"
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality_report.txt"

# ============================================================
# VERİ ÖZELLİKLERİ
# ============================================================

# Giriş özellikleri (features)
FEATURES = [
    "NDVI",           # Bitki Örtüsü İndeksi
    "bagil_nem",      # Bağıl Nem (%)
    "sicaklik",       # Sıcaklık (K veya °C)
    "yagis",          # Yağış (mm)
    "latitude",       # Enlem
    "longitude",      # Boylam
]

# Hedef değişken
TARGET = "yangin_var"

# Coğrafik bilgiler
GEO_COLUMN = ".geo"
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"

# Yangın etiketi
FIRE_LABEL_COLUMN = "etiket"

# ============================================================
# MODEL PARAMETRELERI
# ============================================================

# Random state (tekrarlanabilirlik için)
RANDOM_STATE = 42

# Train-Test split oranı
TEST_SIZE = 0.3  # %70 eğitim, %30 test
VALIDATION_SIZE = 0.1

# Veri ön işleme
SCALER_METHOD = "StandardScaler"  # StandardScaler veya MinMaxScaler

# ============================================================
# TensorFlow MODEL MİMARİSİ
# ============================================================

# Model katmanları
MODEL_CONFIG = {
    "input_shape": len(FEATURES),
    "layers": [
        {"type": "Dense", "units": 64, "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "Dropout", "rate": 0.3},
        {"type": "Dense", "units": 32, "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "Dropout", "rate": 0.2},
        {"type": "Dense", "units": 16, "activation": "relu"},
        {"type": "Dense", "units": 1, "activation": "sigmoid"},
    ]
}

# Optimizer ve loss
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
LOSS = "binary_crossentropy"
METRICS = ["accuracy", "Precision", "Recall", "AUC"]

# ============================================================
# EĞİTİM PARAMETRELERI
# ============================================================

# Epoch ve batch size
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ============================================================
# RİSK SEVİYESİ KATEGORİLERİ
# ============================================================

# Risk eşikleri
RISK_THRESHOLDS = {
    "low": (0.0, 0.30),           # Düşük Risk
    "medium": (0.30, 0.70),       # Orta Risk
    "high": (0.70, 1.0),          # Yüksek Risk
}

# Risk seviyeleri için renkler (Folium haritası)
RISK_COLORS = {
    "low": "green",
    "medium": "orange",
    "high": "red",
}

# ============================================================
# HARITA AYARLARI
# ============================================================

# Türkiye merkez koordinatları
MAP_CENTER = [39.0, 35.0]
MAP_ZOOM_START = 6

# CircleMarker ayarları
CIRCLE_RADIUS = 5
CIRCLE_OPACITY = 0.7
CIRCLE_WEIGHT = 2

# Harita Optimizasyonu
MAX_MARKERS_ON_MAP = 5000          # Max nokta sayısı (performans için)
USE_MARKER_CLUSTERING = True        # Marker cluster'ı kullan
SIMPLIFIED_POPUPS = True            # Basit popup'lar (daha hızlı render)
HEATMAP_ONLY_MODE = False           # True = sadece heatmap (çok hızlı, nokta yok)

# ============================================================
# LOGGING AYARLARI
# ============================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================
# GOOGLE EARTH ENGINE AYARLARI
# ============================================================

GEE_PROJECT_ID = "ignisai-496207"
GEE_REGION_COLLECTION = "USDOS/LSIB_SIMPLE/2017"
GEE_REGION_NAME = "Turkey"

# ============================================================
# VERİ TOPLAMA PARAMETRELERI (GEE)
# ============================================================

GEE_SAMPLING_CONFIG = {
    "num_points": 1000,
    "scale": 10000,  # 10 km
    "seed": 42,
}

# Tarih aralıkları
DATE_RANGE_FULL = {
    "start": "2019-06-01",
    "end": "2025-01-12",
}

# ============================================================
# DİĞER AYARLAR
# ============================================================

# CSV encoding
CSV_ENCODING = "utf-8"

# Hata toleransı
MISSING_VALUE_THRESHOLD = 0.5  # %50'den fazla eksik veri varsa sütunu kaldır
DUPLICATE_THRESHOLD = 0.95      # %95'ten fazla benzerlik varsa duplicate say

# ============================================================
# KONTROL FONKSİYONU
# ============================================================

def ensure_directories_exist() -> None:
    """
    Tüm gerekli dizinlerin var olmasını sağla.
    Yoksa oluştur.
    """
    directories = [
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_SPREAD_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR,
        MAPS_DIR,
        NOTEBOOKS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# ============================================================
# YANGIN BÜYÜME (RASTER / NEXT-DAY SPREAD) PIPELINE
# ============================================================
# NOT: Yukarıdaki FEATURES/TARGET/MODEL_CONFIG eski STATİK RİSK modeline aittir
# (Dense NN, yangin_var 0/1). Aşağıdaki sabitler yeni RASTER YANGIN BÜYÜME
# pipeline'ı içindir. Eski risk modülleri (preprocess/train/predict) artık
# LEGACY'dir; yerlerini spread_dataset.py + spread_model.py + train_spread.py alır.
# ============================================================

# ============================================================
# BANT SÖZLEŞMESİ / BAND CONTRACT
# ============================================================
# Bant SIRASI sözleşmedir. Yükleyici kanal eksenini yalnızca bu sıradan yeniden
# kurar; kısa bir kayıt sessizce yanlış yorumlanır, hata vermez. Bu sıra üç yerde
# birebir aynı olmalıdır: noteboks/colab_notebook*.ipynb, src/gee_config.py,
# src/config.py. Üçünde birden değiştirin ya da hiçbirinde değiştirmeyin.
#
# Band ORDER is contractual. The loader reconstructs the channel axis from this
# order alone, so a short record is silently misread rather than rejected. It must
# match in three places: the notebooks, gee_config.py and here.
# ============================================================

# ---- v2 şeması: 14 girdi bandı ----
SPREAD_INPUT_BANDS_V2 = [
    "ndvi",          # bitki örtüsü indeksi          vegetation index
    "lst",           # arazi yüzey sıcaklığı (°C)    land surface temperature
    "air_temp",      # 2m hava sıcaklığı (°C)        2 m air temperature
    "humidity",      # bağıl nem (%)                 relative humidity
    "wind_speed",    # rüzgâr hızı (m/s)             wind magnitude
    "wind_u",        # rüzgâr doğu bileşeni          eastward wind
    "wind_v",        # rüzgâr kuzey bileşeni         northward wind
    "precip",        # yağış, 24 saat (mm)           precipitation, 24 h
    "soil_moisture", # toprak nemi (m³/m³)           soil water
    "elevation",     # yükseklik (m)                 elevation
    "slope",         # eğim (derece)                 slope
    "aspect",        # bakı (derece)                 aspect
    "landcover",     # arazi örtüsü / yakıt sınıfı   IGBP fuel class
    "fire",          # bugünün yangın maskesi (0/1)  fire mask, day t
]

# ---- v3 şeması: 19 girdi bandı (v2 + zamansal bağlam + yangın hava durumu) ----
SPREAD_INPUT_BANDS_V3 = [
    "ndvi", "lst", "air_temp", "humidity",
    "vpd",                                  # buhar basıncı açığı (kPa)  vapour pressure deficit
    "wind_speed", "wind_u", "wind_v",
    "precip",
    "precip_7d",                            # birikimli yağış, 7 gün     antecedent precip, 7 d
    "precip_30d",                           # birikimli yağış, 30 gün    antecedent precip, 30 d
    "soil_moisture",
    "elevation", "slope", "aspect", "landcover",
    "fire_prev2",                           # t-2 yangın maskesi         fire mask, day t-2
    "fire_prev1",                           # t-1 yangın maskesi         fire mask, day t-1
    "fire",
]

# ---- v4/v5 şeması: 21 girdi bandı (v3 + yakıt/kuruluk geçmişi) ----
# v5, v4 ile AYNI bant şemasını kullanır. v4 notebook'u sunucu tarafında
# çöktüğü için hiç veri üretmedi; v5 onun düzeltilmiş hâlidir.
# v5 uses the SAME band schema as v4. The v4 notebook never produced any data
# because it crashed server-side; v5 is its corrected replacement.
SPREAD_INPUT_BANDS_V4 = [
    "ndvi", "lst", "air_temp", "humidity", "vpd",
    "wind_speed", "wind_u", "wind_v",
    "precip", "precip_7d", "precip_30d",
    "days_since_rain",   # son >1 mm yağıştan bu yana gün   days since measurable rain
    "soil_moisture",
    "elevation", "slope", "aspect", "landcover",
    "burn_age",          # son yanıştan bu yana gün (0-14)  days since last burn
    "fire_prev2", "fire_prev1", "fire",
]

# ---- Hedef ve yardımcı bantlar (v2 ve v3'te aynı) ----
SPREAD_TARGET_BAND = "fire_next"    # t+1 yangın maskesi   fire mask, day t+1
SPREAD_TARGET2_BAND = "fire_next2"  # t+2 yangın maskesi   fire mask, day t+2
SPREAD_VALID_BAND = "valid"         # 1 = tüm girdiler gözlendi   all inputs observed
SPREAD_AUX_BANDS = [SPREAD_TARGET_BAND, SPREAD_TARGET2_BAND, SPREAD_VALID_BAND]

# v1 arşivinde fire_next2 ve valid YOKTUR (yalnızca fire_next).
SPREAD_AUX_BANDS_V1 = [SPREAD_TARGET_BAND]

# ---- v4: hedefin GERÇEKTEN gözlenip gözlenmediğini söyleyen bantlar ----
# MODIS FireMask gözlem kalitesini zaten kodluyor (0,1,2 işlenmedi; 4 bulut;
# 6 bilinmiyor; 3,5 gözlenmiş kara/su; 7,8,9 yangın). v2/v3 `fm.gte(7).unmask(0)`
# yaptığı için BULUTLU bir piksel "yangın yok" olarak etiketleniyordu — yamaların
# %58.9'unda t+1 hedefinin boş çıkmasının sebebi buydu (3. kök neden).
# v4 bu bilgiyi taşır; eğitimde gözlenmemiş hedef pikselleri kayba girmez.
SPREAD_VALID_NEXT_BAND = "valid_next"
SPREAD_VALID_NEXT2_BAND = "valid_next2"
SPREAD_AUX_BANDS_V4 = SPREAD_AUX_BANDS + [SPREAD_VALID_NEXT_BAND,
                                          SPREAD_VALID_NEXT2_BAND]

# ---- Aktif şema ----
# 'v2' -> data/spread/ (14 girdi) ; 'v3' -> data/spread_v3/ (19 girdi)
# 'v5' -> data/spread_v5/ (21 girdi) — güncel şema
SPREAD_VERSION = "v5"

_BANDS_BY_VERSION = {
    "v1": (SPREAD_INPUT_BANDS_V2, SPREAD_AUX_BANDS_V1),
    "v2": (SPREAD_INPUT_BANDS_V2, SPREAD_AUX_BANDS),
    "v3": (SPREAD_INPUT_BANDS_V3, SPREAD_AUX_BANDS),
    "v4": (SPREAD_INPUT_BANDS_V4, SPREAD_AUX_BANDS_V4),
    "v5": (SPREAD_INPUT_BANDS_V4, SPREAD_AUX_BANDS_V4),   # same schema as v4
}

DATA_SPREAD_V3_DIR = DATA_DIR / "spread_v3"
DATA_SPREAD_V4_DIR = DATA_DIR / "spread_v4"
DATA_SPREAD_V5_DIR = DATA_DIR / "spread_v5"
DATA_SPREAD_V1_DIR = DATA_DIR / "spread_v1_legacy"
_DIR_BY_VERSION = {
    "v1": DATA_SPREAD_V1_DIR,
    "v2": DATA_SPREAD_DIR,
    "v3": DATA_SPREAD_V3_DIR,
    "v4": DATA_SPREAD_V4_DIR,
    "v5": DATA_SPREAD_V5_DIR,
}


def spread_bands(version: str = None):
    """(input_bands, aux_bands, all_bands) for a schema version.
    Bir şema sürümü için bant listeleri."""
    version = version or SPREAD_VERSION
    if version not in _BANDS_BY_VERSION:
        raise ValueError(f"Bilinmeyen sürüm / unknown version: {version!r}. "
                         f"Beklenen / expected: {sorted(_BANDS_BY_VERSION)}")
    inp, aux = _BANDS_BY_VERSION[version]
    return list(inp), list(aux), list(inp) + list(aux)


def spread_data_dir(version: str = None):
    """Archive directory for a schema version. Sürüme ait arşiv dizini."""
    version = version or SPREAD_VERSION
    if version not in _DIR_BY_VERSION:
        raise ValueError(f"Bilinmeyen sürüm / unknown version: {version!r}")
    return _DIR_BY_VERSION[version]


# Geriye dönük uyumluluk: eski modüller SPREAD_INPUT_BANDS bekliyor.
# Backwards compatibility for modules that expect the flat name.
SPREAD_INPUT_BANDS = SPREAD_INPUT_BANDS_V2
SPREAD_N_CHANNELS = len(SPREAD_INPUT_BANDS)

# ---- Yama boyutu / patch size ----
PATCH_RADIUS = 32
PATCH_SIZE = 2 * PATCH_RADIUS + 1  # 65
MODEL_PATCH = 32                   # merkez kırpma / centre crop
# 65x65 = 4225 piksel; bunların yalnızca ~12'si yanıyor. Merkezi 32x32'ye kırpmak
# pozitif oranı yükseltir ve 32, 2^3'e tam bölündüğü için üç havuzlama katmanı
# tam sayı boyut verir (32 -> 16 -> 8 -> 4).
# Cropping to the central 32x32 raises positive prevalence without discarding the
# fire, and 32 is divisible by 2^3 so the three pooling stages are exact.
MODEL_PATCH_LEGACY_TF = 64         # eski TF hattının kullandığı boyut (referans)

# ---- Büyüme sınıfları (grow / stable / extinguish) ----
# NOT: "stable" bandı genişletildi (0.75-1.25). Eski dar bant (0.85-1.15) stable'ı çok
# nadir ve bulanık yapıyordu; bu yüzden model stable'ı zor öğreniyordu. Geniş bant daha
# dengeli 3-sınıf problemi verir.
GROWTH_CLASSES = {0: "extinguish", 1: "stable", 2: "grow"}
GROWTH_GROW_RATIO = 1.25           # ratio > 1.25 -> grow
GROWTH_STABLE_LOW = 0.75           # 0.75 <= ratio <= 1.25 -> stable, aksi -> extinguish

# ---- U-Net segmentasyon modeli ----
# Mimari TensorFlow sürümünden BİREBİR korunmuştur; makale Bölüm 2.6 geçerli kalsın.
# Architecture preserved layer-for-layer from the TensorFlow implementation so that
# Section 2.6 of the manuscript remains valid.
SPREAD_MODEL_FILE = MODELS_DIR / "spread_unet.pt"
SPREAD_MODEL_FILE_LEGACY_TF = MODELS_DIR / "spread_unet.keras"
SPREAD_NORM_STATS_FILE = MODELS_DIR / "norm_stats.json"
SPREAD_MODEL_CONFIG = {
    "base_filters": 32,            # 32 -> 64 -> 128, darboğaz 256 / bottleneck 256
    "depth": 3,                    # encoder/decoder derinliği
    "dropout": 0.2,
    "final_activation": "sigmoid", # piksel başına yayılım olasılığı
}

# ---- Eğitim / training ----
SPREAD_OPTIMIZER = "adamw"
SPREAD_LEARNING_RATE = 1e-3
SPREAD_WEIGHT_DECAY = 1e-4
SPREAD_EPOCHS = 120
SPREAD_BATCH_SIZE = 32
SPREAD_GRAD_CLIP = 1.0             # gradyan normu kırpma / gradient norm clipping
SPREAD_EARLY_STOP_PATIENCE = 18
# CosineAnnealingWarmRestarts
SPREAD_COSINE_T0 = 10              # ilk yeniden başlatma periyodu / first restart period
SPREAD_COSINE_TMULT = 2            # her yeniden başlatmada periyot çarpanı
SPREAD_MIN_LR = 1e-6

# ---- Kayıp / loss ----
# Varsayılan: 0.5*BCE(pos_weight) + 0.5*SoftDice.  Alternatif: focal_tversky.
SPREAD_LOSS = "bce_dice"
SPREAD_BCE_DICE_WEIGHT = 0.5       # BCE payı; Dice payı = 1 - bu
SPREAD_POS_WEIGHT = 12.0           # yangın pikselleri seyrek -> pozitif sınıf ağırlığı
SPREAD_DICE_SMOOTH = 1.0
SPREAD_TVERSKY_ALPHA = 0.3         # yanlış pozitif cezası / false-positive penalty
SPREAD_TVERSKY_BETA = 0.7          # yanlış negatif cezası / false-negative penalty
SPREAD_TVERSKY_GAMMA = 0.75        # focal üssü / focal exponent
# Eski TF focal loss parametreleri (referans için tutuldu)
SPREAD_FOCAL_GAMMA = 2.0
SPREAD_FOCAL_ALPHA = 0.80

# ---- Hedef tanımı / target definition ----
# target = max(fire_next, fire_next2) : önümüzdeki 24-48 saatteki yangın etkinliği.
# Katı t+1 maskesi büyük ölçüde uydu şansını kodluyor (yamaların %58.9'unda t+1'de
# hiç yangın pikseli yok, oysa t'de ortalama 12.3 piksel yanıyor).
# The strict t+1 mask largely encodes satellite luck; see README section 5.
SPREAD_TARGET_MODE = "pm1day"      # "pm1day" -> max(fire_next, fire_next2) ; "strict" -> fire_next

# ---- Yıl bazlı bölme / year-based split ----
# Rastgele yama bölmesi SIZINTIDIR: aynı yangın gününün kareleri uzamsal olarak
# örtüşür ve meteorolojik olarak neredeyse aynıdır.
# A random patch split is leakage; patches from one fire day overlap spatially.
SPREAD_SPLIT_YEARS = {
    "train": (2019, 2020, 2021, 2022, 2023),
    "val":   (2024,),
    "test":  (2025, 2026),
}

# ---- Yerel memmap önbelleği / local memmap cache ----
# Depo NTFS fuseblk üzerinde; her epoch gzip açmak eğitim süresini domine eder.
# Arşiv BİR KEZ yerel ext4'e memmap .npy olarak çevrilir.
# The repo lives on an NTFS fuseblk mount; convert once to local ext4.
SPREAD_CACHE_DIR = Path.home() / "ignis-cache"

# ---- TFRecord konumu / archive location ----
SPREAD_TFRECORD_GLOB = str(DATA_SPREAD_DIR / "*.tfrecord.gz")
SPREAD_MAX_FILES = 0               # 0 = hepsi / all; >0 = hızlı deneme için ilk N dosya

# ---- Bilinen veri kesintileri / known data outages ----
# Terra MODIS 10-19 Ekim 2022'de veri almadı (Constellation Exit Manoeuvres;
# cihazlar 21 Ekim'e kadar toparlandı). MOD11A1'in 3 günlük kompozit penceresi
# bazı günler için tamamen kesintinin içine düşüyor ve bantsız bir görüntü
# üretip .rename() aşamasında hata veriyor. Bu günler DOLDURULMAZ, DIŞLANIR —
# doldurmak iki girdi kanalını uydurmak olurdu. ~1136 günün ~5'i (%0.4).
KNOWN_OUTAGES = [
    ("2022-10-10", "2022-10-21"),
]


if __name__ == "__main__":
    # Konfigürasyonu test et
    ensure_directories_exist()
    print("✅ Tüm dizinler oluşturuldu!")
    print(f"📁 BASE_DIR: {BASE_DIR}")
    print("--- ESKİ (risk) ---")
    print(f"📊 Özellikler: {FEATURES}")
    print(f"🎯 Hedef: {TARGET}")
    print("--- YENİ (yangın büyüme / raster) ---")
    print(f"🧊 Girdi kanalları ({SPREAD_N_CHANNELS}): {SPREAD_INPUT_BANDS}")
    print(f"🎯 Hedef bant: {SPREAD_TARGET_BAND}")
    print(f"🗺️  Yama: {PATCH_SIZE}x{PATCH_SIZE} (model {MODEL_PATCH})")
    print(f"🏷️  Büyüme sınıfları: {GROWTH_CLASSES}")
