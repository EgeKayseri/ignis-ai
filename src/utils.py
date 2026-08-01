"""
Yardımcı Fonksiyonlar
=====================
Proje genelinde kullanılan yardımcı fonksiyonlar ve utiliteler.

İçerir:
- Logging ayarları
- Path yönetimi
- Genel kullanım fonksiyonları
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json

from config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


# ============================================================
# LOGGING
# ============================================================

def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Logger'ı ayarla ve döndür.
    
    Args:
        name: Logger adı (genelde __name__ kullanılır)
        log_file: İsteğe bağlı, log dosyası yolu
    
    Returns:
        Ayarlanmış logger nesnesi
    
    Örnek:
        logger = setup_logger(__name__)
        logger.info("Proje başladı!")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (isteğe bağlı)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Proje logger'ı
logger = setup_logger("fire_ai")


# ============================================================
# PATH YÖNETİMİ
# ============================================================

def ensure_path_exists(path: Path, is_file: bool = False) -> Path:
    """
    Dosya veya klasörün var olmasını sağla.
    
    Args:
        path: Dosya/klasör yolu
        is_file: True ise dosya, False ise klasör olarak işle
    
    Returns:
        Doğrulanmış path
    """
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    
    return path


def get_file_size(path: Path) -> str:
    """
    Dosya boyutunu okunaklı formatta döndür.
    
    Args:
        path: Dosya yolu
    
    Returns:
        Boyut (KB, MB vb.)
    """
    if not path.exists():
        return "Dosya bulunamadı"
    
    size_bytes = path.stat().st_size
    
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.2f} TB"


# ============================================================
# JSON DOSYA İŞLERİ
# ============================================================

def save_json(data: dict, path: Path) -> None:
    """
    Dict'i JSON dosyasına kaydet.
    
    Args:
        data: Kaydedilecek veri
        path: Dosya yolu
    """
    ensure_path_exists(path, is_file=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"   JSON kaydedildi: {path}")


def load_json(path: Path) -> dict:
    """
    JSON dosyasını yükle.
    
    Args:
        path: Dosya yolu
    
    Returns:
        Yüklenen veri
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"   JSON yüklendi: {path}")
    return data


# ============================================================
# MEYETRİK HESAPLAMALARI
# ============================================================

def print_metrics_summary(metrics: dict) -> None:
    """
    Model metriklerini güzel formatta yazdır.
    
    Args:
        metrics: Metrik adı -> değer
    """
    logger.info("Model Metrikleri:")
    logger.info("-" * 50)
    
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name:.<40} {value:.4f}")
        else:
            logger.info(f"  {name:.<40} {value}")
    
    logger.info("-" * 50)


def format_percentage(value: float) -> str:
    """
    Yüzde formatına dönüştür.
    
    Args:
        value: 0-1 arasında değer
    
    Returns:
        Yüzde formatı
    """
    return f"{value * 100:.2f}%"


# ============================================================
# GEO KOORDİNAT İŞLEMLERİ
# ============================================================

def parse_geojson(geojson_str: str) -> tuple[float, float]:
    """
    GeoJSON string'indne lat/lon çıkar.
    
    Args:
        geojson_str: GeoJSON formatında koordinat
                    {"type":"Point","coordinates":[lon, lat]}
    
    Returns:
        (latitude, longitude) tuple'ı
    
    Örnek:
        lat, lon = parse_geojson('{"type":"Point","coordinates":[35.2, 39.1]}')
    """
    try:
        geojson = json.loads(geojson_str)
        coords = geojson.get("coordinates", [])
        if len(coords) >= 2:
            longitude, latitude = coords[0], coords[1]
            return latitude, longitude
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.warning(f"GeoJSON parse hatası: {e}")
    
    return None, None


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Koordinatların geçerli olup olmadığını kontrol et.
    
    Args:
        latitude: Enlem (-90 - 90)
        longitude: Boylam (-180 - 180)
    
    Returns:
        Geçerli ise True
    """
    if latitude is None or longitude is None:
        return False
    
    try:
        lat = float(latitude)
        lon = float(longitude)
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except (ValueError, TypeError):
        return False


# ============================================================
# RİSK SEVİYESİ İŞLEMLERİ
# ============================================================

def get_risk_level(probability: float) -> str:
    """
    Olasılıktan risk seviyesi belirle.
    
    Args:
        probability: 0-1 arasında olasılık değeri
    
    Returns:
        Risk seviyesi (low, medium, high)
    """
    from config import RISK_THRESHOLDS
    
    for level, (min_val, max_val) in RISK_THRESHOLDS.items():
        if min_val <= probability <= max_val:
            return level
    
    return "low"  # Default


def get_risk_color(probability: float) -> str:
    """
    Olasılıktan harita rengini belirle.
    
    Args:
        probability: 0-1 arasında olasılık değeri
    
    Returns:
        Folium rengi (green, orange, red)
    """
    from config import RISK_COLORS
    
    risk_level = get_risk_level(probability)
    return RISK_COLORS.get(risk_level, "gray")


# ============================================================
# İSTATİSTİK FONKSİYONLARI
# ============================================================

def print_data_summary(df) -> None:
    """
    DataFrame istatistiklerini yazdır.
    
    Args:
        df: pandas DataFrame
    """
    logger.info("Veri Özeti:")
    logger.info(f"  Satır: {len(df):,}")
    logger.info(f"  Sütun: {len(df.columns)}")
    logger.info(f"  Eksik Değer: {df.isnull().sum().sum()}")
    logger.info(f"  Veri Tipi: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def print_class_distribution(y) -> None:
    """
    Sınıf dağılımını yazdır.
    
    Args:
        y: Target değişkeni (pandas Series)
    """
    from config import TARGET
    
    logger.info(f"{TARGET} Dağılımı:")
    
    value_counts = y.value_counts()
    for value, count in value_counts.items():
        percentage = (count / len(y)) * 100
        logger.info(f"  {value}: {count:,} ({percentage:.1f}%)")


# ============================================================
# TEST FONKSİYONU
# ============================================================

if __name__ == "__main__":
    logger.info("🧪 Utils test başlıyor...")
    
    # Logger test
    logger.info("Logger çalışıyor!")
    
    # JSON test
    test_data = {"name": "Fire AI", "version": "1.0"}
    from config import OUTPUTS_DIR
    save_json(test_data, OUTPUTS_DIR / "test.json")
    loaded = load_json(OUTPUTS_DIR / "test.json")
    logger.info(f"JSON test tamamlandı: {loaded}")
    
    # GeoJSON test
    geojson = '{"type":"Point","coordinates":[35.2, 39.1]}'
    lat, lon = parse_geojson(geojson)
    logger.info(f"GeoJSON test: lat={lat}, lon={lon}")
    
    # Risk level test
    for prob in [0.2, 0.5, 0.8]:
        risk = get_risk_level(prob)
        color = get_risk_color(prob)
        logger.info(f"Prob={prob}: Risk={risk}, Color={color}")
    
    logger.info("Tüm testler tamamlandı!")
