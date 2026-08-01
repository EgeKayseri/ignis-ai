"""
Google Earth Engine Konfigürasyonu — YANGIN BÜYÜME / YAYILIM (Next-Day Spread)
=============================================================================
Bu dosya artık STATİK YANGIN RİSKİ değil, RASTER "ertesi gün yayılım" veri seti
üretimi için ayarları tutar. `noteboks/colab_notebook.ipynb` bu ayarları kullanır.

Eski (risk) yaklaşımı: tüm 2019-2025 aralığı tek bir görüntüde toplanıp rastgele
noktalardan "yangın var/yok" örnekleniyordu. Yeni yaklaşımda her örnek, aktif bir
yangının etrafından kesilmiş 65x65 piksellik bir HARİTA KARESİDİR ve hedef, ertesi
günün yangın maskesidir.
"""


class GEEConfig:
    """Yangın büyüme veri seti üretimi için GEE konfigürasyonu."""

    # Proje
    PROJECT_ID = 'ignisai-496207'

    # Bölge
    REGION_NAME = 'Turkey'
    REGION_COLLECTION = 'USDOS/LSIB_SIMPLE/2017'

    # Analiz gridi (Türkiye için UTM 35N, 1 km)
    SCALE = 1000                      # metre / piksel
    CRS = 'EPSG:32635'

    # Patch (yama) — merkezi aktif yangın pikseli
    PATCH_RADIUS = 32                 # -> 65x65
    PATCH_SIZE = 2 * PATCH_RADIUS + 1
    MODEL_PATCH = 64                  # eğitimde 64x64'e kırpılır (2^n bölünebilirlik)

    # Yangın sezonu ve yıllar
    FIRE_SEASON_MONTHS = (6, 10)      # Haziran - Ekim
    YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

    # Aktif yangın (MODIS FireMask: 7=düşük, 8=orta, 9=yüksek güven)
    FIRE_CONFIDENCE = 7
    MIN_FIRE_PIXELS = 5               # bir günü işlemek için gerekli min aktif yangın pikseli
    MAX_POINTS_PER_DAY = 150          # günde en fazla kaç yangın karesi örneklensin
    SEED = 42

    # ---- VERİ KAYNAKLARI (yeni) ----
    DATA_SOURCES = {
        # Aktif yangın (GÜNLÜK) — yayılım sinyali. Terra + Aqua birleşik.
        'fire_terra': {'collection': 'MODIS/061/MOD14A1', 'band': 'FireMask',
                       'description': 'Günlük aktif yangın (Terra)'},
        'fire_aqua':  {'collection': 'MODIS/061/MYD14A1', 'band': 'FireMask',
                       'description': 'Günlük aktif yangın (Aqua)'},
        'ndvi':       {'collection': 'MODIS/061/MOD13Q1', 'band': 'NDVI',
                       'scale_factor': 0.0001, 'description': 'Bitki örtüsü indeksi'},
        'lst':        {'collection': 'MODIS/061/MOD11A1', 'band': 'LST_Day_1km',
                       'scale_factor': 0.02, 'description': 'Arazi yüzey sıcaklığı (K)'},
        # DİKKAT: eski 'ECMWF/ERA5_LAND/DAILY' koleksiyonu KALDIRILDI.
        # Doğru koleksiyon: ECMWF/ERA5_LAND/DAILY_AGGR
        'era5_land':  {'collection': 'ECMWF/ERA5_LAND/DAILY_AGGR',
                       'bands': ['temperature_2m', 'dewpoint_temperature_2m',
                                 'u_component_of_wind_10m', 'v_component_of_wind_10m',
                                 'volumetric_soil_water_layer_1'],
                       'description': 'Hava sıcaklığı, çiğ, rüzgâr u/v, toprak nemi'},
        'precip':     {'collection': 'UCSB-CHG/CHIRPS/DAILY', 'band': 'precipitation',
                       'description': 'Günlük yağış'},
        'terrain':    {'image': 'USGS/SRTMGL1_003',
                       'description': 'Yükseklik + eğim + bakı (ee.Terrain)'},
        'landcover':  {'collection': 'MODIS/061/MCD12Q1', 'band': 'LC_Type1',
                       'description': 'Arazi örtüsü / yakıt tipi (yıllık)'},
    }

    # ---- KANAL SIRASI / BAND CONTRACT ----
    # Bu sıra SÖZLEŞMEDİR ve üç yerde birebir aynı olmalıdır:
    #   noteboks/colab_notebook*.ipynb, src/gee_config.py, src/config.py
    # Yükleyici kanal eksenini yalnızca bu sıradan kurar; kısa bir kayıt
    # hata vermez, SESSİZCE yanlış yorumlanır.
    # This order is contractual across the notebooks, this file and config.py.
    # A short record is silently misread rather than rejected.

    # v2 — 14 girdi bandı / 14 input bands
    INPUT_BANDS_V2 = [
        'ndvi', 'lst', 'air_temp', 'humidity',
        'wind_speed', 'wind_u', 'wind_v',
        'precip', 'soil_moisture',
        'elevation', 'slope', 'aspect',
        'landcover', 'fire',
    ]

    # v3 — 19 girdi bandı / 19 input bands
    # v2 + vpd, precip_7d, precip_30d, fire_prev1, fire_prev2
    INPUT_BANDS_V3 = [
        'ndvi', 'lst', 'air_temp', 'humidity',
        'vpd',
        'wind_speed', 'wind_u', 'wind_v',
        'precip', 'precip_7d', 'precip_30d',
        'soil_moisture',
        'elevation', 'slope', 'aspect', 'landcover',
        'fire_prev2', 'fire_prev1', 'fire',
    ]

    # Hedef + yardımcı bantlar (v2 ve v3'te ortak) / target + auxiliary
    TARGET_BAND = 'fire_next'      # t+1 yangın maskesi
    TARGET2_BAND = 'fire_next2'    # t+2 yangın maskesi (+-1 gün hedefi için)
    VALID_BAND = 'valid'           # 1 = tüm girdiler gerçekten gözlendi
    AUX_BANDS = [TARGET_BAND, TARGET2_BAND, VALID_BAND]

    # v1 arşivinde fire_next2 ve valid YOKTUR.
    AUX_BANDS_V1 = [TARGET_BAND]

    # Aktif şema / active schema
    VERSION = 'v2'
    INPUT_BANDS = INPUT_BANDS_V2          # geriye dönük uyumluluk
    ALL_BANDS = INPUT_BANDS_V2 + AUX_BANDS

    @classmethod
    def bands(cls, version='v2'):
        """(input, aux, all) for a schema version. / Sürüme göre bant listeleri."""
        inp = {'v1': cls.INPUT_BANDS_V2,
               'v2': cls.INPUT_BANDS_V2,
               'v3': cls.INPUT_BANDS_V3}[version]
        aux = cls.AUX_BANDS_V1 if version == 'v1' else cls.AUX_BANDS
        return list(inp), list(aux), list(inp) + list(aux)

    # ---- BÜYÜME SINIFI EŞİKLERİ (grow / stable / extinguish) ----
    # ratio = (yarınki yangın pikseli) / (bugünkü yangın pikseli)
    GROWTH_CLASSES = {0: 'extinguish', 1: 'stable', 2: 'grow'}
    GROW_RATIO = 1.25                 # ratio > 1.25  -> grow  (stable bandı genişletildi)
    STABLE_LOW = 0.75                 # 0.75 <= ratio <= 1.25 -> stable
    #                                   ratio < 0.75 (0 dahil) -> extinguish

    # Çıktı
    # Her şema sürümü AYRI bir Drive klasörüne yazar; devam taraması
    # aksi hâlde v2 parçalarını v3 sanır.
    # Each schema version writes to its OWN Drive folder.
    DRIVE_FOLDER = 'GEE_FireSpread_v2'
    DRIVE_FOLDER_V3 = 'GEE_FireSpread_v3'
    PATHS = {'data_dir': '../data/spread', 'outputs_dir': '../outputs', 'models_dir': '../models'}

    @classmethod
    def get_config(cls):
        return {
            'project_id': cls.PROJECT_ID, 'region_name': cls.REGION_NAME,
            'scale': cls.SCALE, 'crs': cls.CRS, 'patch_size': cls.PATCH_SIZE,
            'input_bands': cls.INPUT_BANDS, 'target_band': cls.TARGET_BAND,
            'data_sources': cls.DATA_SOURCES, 'drive_folder': cls.DRIVE_FOLDER,
        }

    @classmethod
    def print_config(cls):
        print('GEE YANGIN BUYUME KONFIGURASYONU')
        print(f'  Proje ID : {cls.PROJECT_ID}')
        print(f'  Bolge    : {cls.REGION_NAME} ({cls.CRS} @ {cls.SCALE} m)')
        print(f'  Patch    : {cls.PATCH_SIZE}x{cls.PATCH_SIZE}')
        print(f'  Surum    : {cls.VERSION}')
        print(f'  Kanallar : v2 {len(cls.INPUT_BANDS_V2)} girdi | '
              f'v3 {len(cls.INPUT_BANDS_V3)} girdi')
        print(f'  Yardimci : {cls.AUX_BANDS}')
        print(f'  Siniflar : {cls.GROWTH_CLASSES}')


if __name__ == '__main__':
    GEEConfig.print_config()
