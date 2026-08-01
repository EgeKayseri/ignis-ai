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

# ---- Girdi kanalları (yamanın kanal sırası; GEE export ile aynı) ----
SPREAD_INPUT_BANDS = [
    "ndvi",          # bitki örtüsü indeksi
    "lst",           # arazi yüzey sıcaklığı (°C)
    "air_temp",      # 2m hava sıcaklığı (°C)
    "humidity",      # bağıl nem (%)
    "wind_speed",    # rüzgâr hızı (m/s)
    "wind_u",        # rüzgâr doğu-batı bileşeni
    "wind_v",        # rüzgâr kuzey-güney bileşeni
    "precip",        # yağış (mm)
    "soil_moisture", # toprak nemi (m³/m³)
    "elevation",     # yükseklik (m)
    "slope",         # eğim (derece)
    "aspect",        # bakı (derece)
    "landcover",     # arazi örtüsü / yakıt sınıfı
    "fire",          # bugünün aktif yangın maskesi (0/1)
]
SPREAD_TARGET_BAND = "fire_next"   # ertesi günün yangın maskesi (0/1) — segmentasyon hedefi
SPREAD_N_CHANNELS = len(SPREAD_INPUT_BANDS)

# ---- Yama boyutu ----
PATCH_RADIUS = 32
PATCH_SIZE = 2 * PATCH_RADIUS + 1  # 65
MODEL_PATCH = 64                   # eğitimde 64x64'e kırpılır

# ---- Büyüme sınıfları (grow / stable / extinguish) ----
# NOT: "stable" bandı genişletildi (0.75-1.25). Eski dar bant (0.85-1.15) stable'ı çok
# nadir ve bulanık yapıyordu; bu yüzden model stable'ı zor öğreniyordu. Geniş bant daha
# dengeli 3-sınıf problemi verir.
GROWTH_CLASSES = {0: "extinguish", 1: "stable", 2: "grow"}
GROWTH_GROW_RATIO = 1.25           # ratio > 1.25 -> grow
GROWTH_STABLE_LOW = 0.75           # 0.75 <= ratio <= 1.25 -> stable, aksi -> extinguish

# ---- U-Net segmentasyon modeli ----
SPREAD_MODEL_FILE = MODELS_DIR / "spread_unet.keras"
SPREAD_MODEL_CONFIG = {
    "input_shape": (MODEL_PATCH, MODEL_PATCH, SPREAD_N_CHANNELS),
    # Tam model (en yüksek doğruluk). CPU'da yavaştır; gece açık bırakarak eğitilir.
    # Çok yavaşsa geçici olarak base_filters=16, depth=2 yapılabilir.
    "base_filters": 32,            # U-Net taban filtre sayısı
    "depth": 3,                    # encoder/decoder derinliği
    "dropout": 0.2,
    "final_activation": "sigmoid", # piksel başına yayılım olasılığı
}
SPREAD_OPTIMIZER = "adam"
SPREAD_LEARNING_RATE = 1e-3
SPREAD_LOSS = "focal"              # dengesiz maske için focal / weighted BCE
SPREAD_METRICS = ["AUC", "Precision", "Recall"]
SPREAD_EPOCHS = 120               # daha uzun eğitim (erken durdurma zaten koruyor)
SPREAD_BATCH_SIZE = 32
SPREAD_SHUFFLE_BUFFER = 512        # düşük RAM'de OOM'u önlemek için (16GB+ ise 2048 yapılabilir)
SPREAD_POS_WEIGHT = 12.0           # yangın pikselleri seyrek -> pozitif ağırlık (arttırıldı)
SPREAD_EARLY_STOP_PATIENCE = 18    # daha uzun sabır -> daha iyi yakınsama
SPREAD_REDUCE_LR_PATIENCE = 7
# Focal loss parametreleri (ayarlanabilir)
SPREAD_FOCAL_GAMMA = 2.0           # zor örneklere odak
SPREAD_FOCAL_ALPHA = 0.80          # pozitif (yangın) sınıfa ağırlık (0.75 -> 0.80)

# ---- TFRecord konumu ----
SPREAD_TFRECORD_GLOB = str(DATA_SPREAD_DIR / "*.tfrecord.gz")
# CPU'da hızlı deneme: sadece ilk N dosyayı kullan (0 = hepsi).
# Tam veri seti çok yavaşsa 10-20 gibi bir değer verip hızlı baseline al.
SPREAD_MAX_FILES = 0


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
