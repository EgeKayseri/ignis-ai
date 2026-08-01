# IGNIS — Uydu Verisiyle Yangın Büyüme (Yayılım) Tahmini

**IAC 2026 projesi.** Bu depo, geçmiş yangın olaylarına ait uydu ve meteorolojik
Dünya Gözlem (Earth Observation) verilerini kullanarak bir orman yangınının **ertesi
gün nasıl yayılacağını** tahmin eden bir makine öğrenmesi sistemidir. Model, her aktif
yangın için yayılım eğilimini **büyüyor (grow) / sabit (stable) / sönüyor (extinguish)**
olarak olasılıksal bir çerçevede değerlendirir.

> Bu README, projenin **tek ve eksiksiz kaynağıdır**: özet, motivasyon, ilgili çalışmalar,
> yöntem, veri seti şeması, model mimarisi, eğitim kurulumu, değerlendirme, çalıştırma
> talimatları, dosya yapısı ve kaynakça. Makalenin tüm bölümleri bu belge üzerinden
> yazılabilecek şekilde hazırlanmıştır.

---

## İçindekiler

1. [Özet (Abstract)](#1-özet-abstract)
2. [Problem ve Motivasyon](#2-problem-ve-motivasyon)
3. [Katkı / Yenilik](#3-katkı--yenilik)
4. [İlgili Çalışmalar](#4-i̇lgili-çalışmalar)
5. [Yöntem](#5-yöntem)
6. [Veri Seti](#6-veri-seti)
7. [Model Mimarisi](#7-model-mimarisi)
8. [Eğitim Kurulumu](#8-eğitim-kurulumu)
9. [Değerlendirme](#9-değerlendirme)
10. [Sonuçlar (doldurulacak)](#10-sonuçlar-doldurulacak)
11. [Kurulum ve Çalıştırma](#11-kurulum-ve-çalıştırma)
12. [Proje Yapısı](#12-proje-yapısı)
13. [Konfigürasyon Referansı](#13-konfigürasyon-referansı)
14. [Kısıtlar ve Gelecek Çalışma](#14-kısıtlar-ve-gelecek-çalışma)
15. [Sözlük ve Kısaltmalar](#15-sözlük-ve-kısaltmalar)
16. [Kaynakça](#16-kaynakça)

---

## 1. Özet (Abstract)

İklim değişikliği, uzun süreli kuraklık dönemleri ve artan küresel sıcaklıklar nedeniyle
orman yangınları son yıllarda giderek daha sık ve şiddetli hale gelmiştir. Yangınların
nasıl gelişeceğini önceden kestirmek; çevresel zararı azaltmak, ekonomik kayıpları en aza
indirmek ve insan hayatına yönelik riskleri hafifletmek için kritik öneme sahiptir. Bu
proje, uydu tabanlı Dünya Gözlem verilerini kullanarak yangın büyüme davranışını tahmin
eden bir makine öğrenmesi modeli sunar.

Model, geçmiş yangın olaylarından elde edilen **uzamsal ve zamansal** verilerle eğitilir.
Bu veriler; Normalize Edilmiş Fark Bitki Örtüsü İndeksi (NDVI), Arazi Yüzey Sıcaklığı
(LST), termal anomali (aktif yangın) tespitleri, meteorolojik değişkenler (rüzgâr, nem,
sıcaklık, yağış, toprak nemi) ve topografik değişkenleri (yükseklik, eğim, bakı) içerir.
Ön işleme; radyometrik düzeltme, bulut maskeleme, yeniden örnekleme (resampling),
eş-kayıt (co-registration), zamansal pencereleme ve öznitelik çıkarımını kapsar. Elde
edilen çok-kanallı öznitelik yığınları üzerinde bir **U-Net segmentasyon modeli**
eğitilerek, belirli çevresel koşullar altında yangının bir sonraki gün genişleme
olasılığı tahmin edilir. Model çıktısı, yangının büyüme, sabit kalma veya sönme eğilimini
olasılıksal bir çerçevede değerlendirir.

Mevcut izleme yöntemleri çoğunlukla yer gözlemlerine ve meteorolojik tahminlere dayanır ve
sınırlı bir arama alanı sunar. Buna karşın uydu tabanlı Dünya Gözlem sistemleri, büyük
ölçekli ve sürdürülebilir veri toplama olanağı sağlayarak hem izleme hem de tahmin
performansını belirgin şekilde artırır. Modelin, Dünya Gözlem sistemleri aracılığıyla
erken risk değerlendirmesi yaparak afet yönetimi öngörüsünü güçlendirmesi ve veri temelli
karar alma süreçlerini mümkün kılarak çevre sorunlarıyla mücadelede sürdürülebilir
sistemlerin gelişimine katkı sağlaması beklenmektedir.

---

## 2. Problem ve Motivasyon

Yangın izleme ve modellemede iki farklı problem vardır ve bunların karıştırılmaması
gerekir:

- **Yangın riski / duyarlılığı (susceptibility):** "Bu konumda yangın çıkma olasılığı
  nedir?" Zamansal bir bileşen içermez; statik bir risk haritasıdır.
- **Yangın büyümesi / yayılımı (spread):** "Halihazırda yanan bir yangın bir sonraki gün
  nereye ve ne kadar yayılacaktır?" Doğası gereği **zamansaldır** (gün *t* → gün *t+1*).

Bu proje **ikinci** problemi çözer. Afet yönetimi açısından kritik olan, yangının
çıkıp çıkmayacağı kadar, çıkmış bir yangının **nasıl ilerleyeceğidir**; çünkü tahliye,
müdahale ve kaynak tahsisi kararları yayılım yönü ve hızına bağlıdır.

Yer tabanlı gözlemler dar bir alanı kapsarken, uydu tabanlı sistemler tüm ülke ölçeğinde
günlük ve tutarlı veri sağlar. Bu proje, MODIS aktif yangın ürünleriyle günlük yangın
cephesini takip eder ve çevresel sürücüleri (rüzgâr, topografya, yakıt, nem) kullanarak
ertesi günün yayılımını öğrenir.

---

## 3. Katkı / Yenilik

- **Türkiye'ye özgü, uydu-türevli next-day spread veri seti.** MODIS Terra+Aqua günlük
  aktif yangın maskeleri, ERA5-Land meteorolojisi, SRTM topografyası ve MODIS arazi örtüsü
  Google Earth Engine üzerinde ortak bir 1 km gride hizalanır.
- **Uçtan uca yeniden üretilebilir hat.** Veri çekme (Colab/GEE) → TFRecord → U-Net eğitimi
  tek depoda; tüm sabitler `src/config.py`'de.
- **İki düzeyli çıktı.** Piksel bazında ertesi-gün yayılım olasılık haritası *ve* yama
  düzeyinde grow/stable/extinguish sınıfı — abstract'taki üçlü hedefle birebir.

---

## 4. İlgili Çalışmalar

Tasarım, literatürdeki uydu-tabanlı "next-day fire spread" kurgusunu izler:

- **Next Day Wildfire Spread / WildfireSpreadTS tarzı veri setleri** — çevresel sürücü
  yığınından ertesi-gün yangın maskesini kestiren segmentasyon kurulumunun temelini
  oluşturur (bkz. MDPI *Remote Sensing* 16(8):1467; *Fire* 7(12):482; *Applied Sciences*
  13(14):8275; Nature *Scientific Reports* 2024, s41598-024-52821-x).
- **Fiziksel yayılım modelleri** — Rothermel (1972) yayılım denklemleri ve physics-informed
  sinir ağları (arXiv:2406.14591), rüzgâr ve eğimin yayılım hızındaki baskın rolünü
  vurgular; bu değişkenler öznitelik setimizin merkezindedir.
- **ML tabanlı yangın modelleme derlemeleri** — (Environmental Reviews, er-2020-0019;
  *Symmetry* 12(6):1022) veri kaynakları ve değerlendirme metrikleri için referanstır.

> Makalenin "Related Work" bölümü için: her makalenin kullandığı girdi değişkenlerini,
> model ailesini (CNN/U-Net/LSTM/physics-informed) ve raporladığı metriği (AUC-PR, IoU,
> F1) tablo haline getirin; bizim yaklaşımımızı bu tablonun son satırı olarak konumlandırın.

---

## 5. Yöntem

### 5.1 Genel akış

```
[GEE / Colab]                          [Yerel eğitim]
 MODIS aktif yangın (Terra+Aqua)         data/spread/*.tfrecord.gz
 NDVI, LST, ERA5-Land, CHIRPS,     ──►    │
 SRTM, MODIS arazi örtüsü                 ▼
        │  günlük hizalama (1km, UTM35N)  spread_dataset.py  (X:64×64×14, y:64×64×1)
        ▼                                 ▼
 14 kanal + fire_next hedefi        ►     spread_model.py  (U-Net)
        │  65×65 yama (yangın merkezli)   ▼
        ▼                                 train_spread.py → models/spread_unet.keras
   TFRecord (Drive)                       ▼
                                          Ertesi-gün yayılım olasılık haritası
                                          + grow/stable/extinguish sınıfı
```

### 5.2 Çalışma alanı ve dönem

- **Bölge:** Türkiye (`USDOS/LSIB_SIMPLE/2017`, `country_na = 'Turkey'`).
- **Grid:** EPSG:32635 (UTM 35N), 1 km çözünürlük — kareler metrik ve kare olur.
- **Dönem:** Yangın sezonu ayları Haziran–Ekim; yıllar 2019–2024 (MODIS aktif yangın
  arşivi). `TEST_MODE` ile hızlı deneme için 2021 Temmuz sonu (Manavgat/Marmaris) dönemi.

### 5.3 Veri kaynakları

| Değişken | GEE Koleksiyonu | Doğal çözünürlük / sıklık | Kullanım |
|---|---|---|---|
| Aktif yangın (Terra) | `MODIS/061/MOD14A1` | 1 km / günlük | Yangın maskesi (bugün & yarın) |
| Aktif yangın (Aqua) | `MODIS/061/MYD14A1` | 1 km / günlük | Terra ile birleştirilir |
| NDVI | `MODIS/061/MOD13Q1` | 250 m / 16 gün | Bitki örtüsü / yakıt |
| LST | `MODIS/061/MOD11A1` | 1 km / günlük | Yüzey sıcaklığı |
| Meteoroloji | `ECMWF/ERA5_LAND/DAILY_AGGR` | ~9 km / günlük | Sıcaklık, çiğ, rüzgâr u/v, toprak nemi |
| Yağış | `UCSB-CHG/CHIRPS/DAILY` | ~5 km / günlük | Yağış |
| Topografya | `USGS/SRTMGL1_003` | 30 m / statik | Yükseklik, eğim, bakı |
| Arazi örtüsü | `MODIS/061/MCD12Q1` | 500 m / yıllık | Yakıt tipi |

> **Not:** Eski kod `ECMWF/ERA5_LAND/DAILY` koleksiyonunu çağırıyordu; bu koleksiyon GEE'den
> kaldırıldığı için `ImageCollection asset ... not found` hatası veriyordu. Doğru koleksiyon
> **`ECMWF/ERA5_LAND/DAILY_AGGR`**'dır ve kod güncellenmiştir.

### 5.4 Ön işleme

Google Earth Engine, MODIS/ERA5/CHIRPS ürünlerini kalibre (radyometrik düzeltme uygulanmış)
ve bulut-maskeli olarak sunar. Hat şu adımları uygular:

1. **Yeniden örnekleme + eş-kayıt (co-registration):** tüm bantlar `reproject` ile ortak
   projeksiyon (UTM 35N) ve 1 km ölçeğe getirilir, böylece pikseller uzamsal olarak hizalı
   olur.
2. **Bulut/geçersiz maske:** MODIS FireMask sınıfları (su/bulut/işlenmemiş) yangın dışı
   kabul edilir; eksik/NaN değerler eğitim tarafında 0'a çekilir.
3. **Zamansal pencereleme:**
   - NDVI için son 32 gündeki en yeni 16-günlük kompozit alınır (16-günlük ürünün
     boşlukları için).
   - LST için son 3 günün ortalaması (günlük boşlukları doldurmak için).
   - Meteoroloji ve yangın için ilgili günün değeri.
4. **Birim dönüşümleri:** LST ve hava sıcaklığı Kelvin→Celsius; ERA5 yağışı m→mm; NDVI
   ölçek faktörü 0.0001.

### 5.5 Örnekleme ve yama çıkarımı

Her yangın günü için, aktif yangın piksellerinden en fazla `MAX_POINTS_PER_DAY` (varsayılan
150) nokta seçilir (`stratifiedSample`). Her noktanın etrafında `ee.Kernel.square(radius=32)`
ile **65×65 piksellik** yama, `neighborhoodToArray` kullanılarak çıkarılır. Eğitimde yamalar
2ⁿ bölünebilirlik için **64×64**'e kırpılır. Yamalar `Export.table.toDrive(..., 'TFRecord')`
ile Drive'a yazılır.

### 5.6 Etiketleme

**Piksel hedefi (`fire_next`):** ertesi günün (t+1) aktif yangın maskesi — U-Net'in
segmentasyon hedefi.

**Yama sınıfı (grow / stable / extinguish):** her yama için bugünkü ve yarınki yangın
piksel sayısından oran hesaplanır:

```
ratio = fire_next_pikselleri / max(fire_bugün_pikselleri, 1)
  ratio > 1.15          → grow        (2)  büyüyor
  0.85 ≤ ratio ≤ 1.15   → stable      (1)  sabit
  ratio < 0.85 (0 dahil)→ extinguish  (0)  sönüyor
```

Eşikler `config.py` (`GROWTH_GROW_RATIO`, `GROWTH_STABLE_LOW`) üzerinden ayarlanabilir.

### 5.7 Öznitelik mühendisliği

- **Rüzgâr:** ERA5 u/v bileşenlerinden hız (`hypot(u, v)`) hesaplanır; u ve v ayrıca yön
  bilgisi olarak korunur.
- **Bağıl nem:** ERA5 hava sıcaklığı ve çiğlenme noktasından **Magnus formülü** ile:
  `RH = 100 · exp(A − B)`, `A = 17.625·Td/(243.04+Td)`, `B = 17.625·T/(243.04+T)`, [0,100]
  aralığına kırpılır.
- **Topografya:** SRTM DEM'den `ee.Terrain.products` ile yükseklik, eğim, bakı.

---

## 6. Veri Seti

Her örnek = aktif yangın merkezli 64×64 yama, 14 girdi kanalı + 1 hedef.

| # | Kanal | Birim | Açıklama |
|---|---|---|---|
| 1 | `ndvi` | [-1,1] | Bitki örtüsü indeksi (yakıt yoğunluğu) |
| 2 | `lst` | °C | Arazi yüzey sıcaklığı |
| 3 | `air_temp` | °C | 2 m hava sıcaklığı |
| 4 | `humidity` | % | Bağıl nem (yakıt kuruluğu) |
| 5 | `wind_speed` | m/s | Rüzgâr hızı (yayılımın baskın sürücüsü) |
| 6 | `wind_u` | m/s | Rüzgâr doğu-batı bileşeni |
| 7 | `wind_v` | m/s | Rüzgâr kuzey-güney bileşeni |
| 8 | `precip` | mm | Günlük yağış |
| 9 | `soil_moisture` | m³/m³ | Toprak nemi (kuraklık göstergesi) |
| 10 | `elevation` | m | Yükseklik |
| 11 | `slope` | ° | Eğim (yokuş yukarı hızlanma) |
| 12 | `aspect` | ° | Bakı |
| 13 | `landcover` | sınıf | Arazi örtüsü / yakıt tipi (MODIS IGBP) |
| 14 | `fire` | 0/1 | **Bugünün** aktif yangın maskesi |
| Hedef | `fire_next` | 0/1 | **Yarının** aktif yangın maskesi |

Kanal sırası `src/config.py → SPREAD_INPUT_BANDS`, `src/gee_config.py → INPUT_BANDS` ve
Colab notebook'undaki `INPUT_BANDS` arasında **birebir aynıdır** ve değiştirilecekse üçü
birlikte değişmelidir.

---

## 7. Model Mimarisi

**U-Net segmentasyon** (`src/spread_model.py`):

- **Girdi:** 64×64×14 öznitelik yığını.
- **Encoder:** `depth=3` seviye; her seviyede iki 3×3 Conv + BatchNorm + Dropout, ardından
  2×2 MaxPooling. Taban filtre `base_filters=32`, seviyeyle iki katına çıkar (32→64→128).
- **Bottleneck:** 256 filtreli conv bloğu.
- **Decoder:** her seviyede 2×2 transpose-conv ile büyütme + encoder skip bağlantısıyla
  birleştirme + conv bloğu.
- **Çıktı:** 1×1 Conv + sigmoid → 64×64×1, **piksel başına ertesi-gün yangın olasılığı**.
- **Parametre sayısı:** ~1.9 milyon.

**Kayıp fonksiyonu:** Yangın pikselleri seyrek olduğundan sınıf dengesizliğine karşı
**focal loss** (γ=2, α=0.75) varsayılandır; alternatif olarak pozitif-ağırlıklı BCE
(`SPREAD_POS_WEIGHT=10`) sağlanır.

**Metrikler:** AUC-PR (precision-recall eğrisi altındaki alan), Precision, Recall.

---

## 8. Eğitim Kurulumu

| Parametre | Değer | Sabit (`config.py`) |
|---|---|---|
| Girdi boyutu | 64×64×14 | `SPREAD_MODEL_CONFIG.input_shape` |
| Optimizasyon | Adam, lr=1e-3 | `SPREAD_OPTIMIZER`, `SPREAD_LEARNING_RATE` |
| Kayıp | focal | `SPREAD_LOSS` |
| Epoch | 60 | `SPREAD_EPOCHS` |
| Batch | 32 | `SPREAD_BATCH_SIZE` |
| Train/Val ayrımı | %80 / %20 (dosya bazında) | `train_val_split(val_frac=0.2)` |
| Erken durdurma | val_auc, patience=10 | `train_spread.py` |
| LR azaltma | val_loss, factor=0.5, patience=5 | `train_spread.py` |

Train/val ayrımı **dosya (gün) bazında** yapılır; aynı yangın gününün yamaları aynı bölmede
kalır, böylece uzamsal/zamansal sızıntı (leakage) engellenir.

---

## 9. Değerlendirme

- **Piksel düzeyi (yayılım maskesi):** AUC-PR, Precision, Recall ve IoU / F1. Seyrek
  pozitif sınıf nedeniyle AUC-PR, ham accuracy'den daha bilgilendiricidir.
- **Yama düzeyi (grow/stable/extinguish):** 3×3 karışıklık matrisi ve makro-F1. Tahmin
  edilen maskeden türetilen sınıf, gerçek sınıfa karşı değerlendirilir.
- **Baz çizgileri (öneri):** (a) kalıcılık (persistence) — yarın = bugün maskesi;
  (b) sadece rüzgâr yönünde büyüme. Modelin bu basit baz çizgilerini geçtiği gösterilmelidir.

---

## 10. Sonuçlar (doldurulacak)

Eğitim tamamlandıktan sonra doldurun:

| Metrik | Değer |
|---|---|
| AUC-PR (val) | — |
| Precision / Recall (val) | — / — |
| IoU (yayılım maskesi) | — |
| Makro-F1 (grow/stable/extinguish) | — |
| Kalıcılık baz çizgisine göre kazanç | — |

Görseller (öneri): (1) örnek yamada bugün/yarın/tahmin maskeleri; (2) PR eğrisi;
(3) 3-sınıf karışıklık matrisi; (4) öznitelik önem/ablation (kanal çıkarımı) analizi.

---

## 11. Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Adım 1 — Veriyi üret (Google Colab / GEE)

1. `noteboks/colab_notebook.ipynb` dosyasını Google Colab'da açın (File → Upload notebook).
2. İlk hücrede kendi GEE proje ID'nizi girin ve kimlik doğrulaması yapın.
3. Konfigürasyon hücresinde hızlı deneme için `TEST_MODE = True`, tam veri için `False`.
4. Hücreleri çalıştırın; veriler Drive → `GEE_FireSpread/` klasörüne `*.tfrecord.gz` olarak
   yazılır.
5. Bu dosyaları indirip **`data/spread/`** içine koyun.

### Adım 2 — Eğit

```bash
python start.py            # ortam + veri kontrolü, ardından U-Net eğitimi
# veya doğrudan:
python src/train_spread.py
```

`start.py` çalıştırıldığında pipeline **doğrudan başlar**: paketleri ve `data/spread/`'i
kontrol eder; veri varsa U-Net'i eğitir, yoksa veriyi nasıl üreteceğinizi söyler.

### Adım 3 — Veri setini incele (opsiyonel)

```bash
python src/spread_dataset.py    # kare sayısı, grow/stable/extinguish dağılımı
```

### Adım 4 — Sonuçlar / doğruluk

`start.py` eğitimden sonra değerlendirmeyi **otomatik** çalıştırır. Ayrıca elle:

```bash
python src/evaluate_spread.py
```

Bu, eğitilmiş modeli doğrulama verisiyle test edip **doğruluk raporunu** kaydeder:

- `outputs/reports/spread_evaluation.txt` — AUC-PR, Precision, Recall, F1, IoU, piksel
  doğruluğu ve grow/stable/extinguish karışıklık matrisi.
- `outputs/reports/spread_evaluation.png` — PR eğrisi, karışıklık matrisi ve örnek
  tahmin görselleri.

> **"Doğruluk oranı nerede?"** — İşte burada. Eğitim sırasında her epoch'ta ekrana
> `loss / auc / precision / recall` basılır; eğitim bitince en iyi epoch metrikleri
> yazdırılır; ve yukarıdaki rapor dosyası kalıcı olarak kaydedilir. (Eğitim `0/step`
> ilerliyorsa veri boştur — bkz. Hata Ayıklama.)

### AMD GPU (RX 9070 XT) ile Çalıştırma

Önce net olalım: bu U-Net küçüktür (~1.9M parametre, 64×64 yama). **CPU'da eğitim bu proje
için tamamen yeterlidir** (epoch başına dakikalar). GPU şart değildir. Yine de hızlandırmak
isterseniz:

**Neden GPU görünmüyor?** Native Windows'ta TensorFlow ≥ 2.11 GPU'yu **hiç desteklemez**
(NVIDIA dâhil). Aldığınız uyarı tam olarak bunu söyler. AMD için native Windows'ta bir
TF-GPU yolu yoktur; aşağıdaki iki yoldan biri gerekir.

**Önerilen yol — WSL2 + ROCm 7.2 + `tensorflow-rocm`.** ROCm 7.2 (Mart 2026) RX 9070 XT'yi
(gfx1201) **resmî olarak** destekler.

1. Windows'a WSL2 + Ubuntu 24.04 (veya 22.04) kurun.
2. Windows tarafında WSL2 destekli AMD Adrenalin sürücüsünü (26.x) kurun.
3. WSL içinde ROCm 7.2 ve ardından `tensorflow-rocm`'u AMD'nin "Install TensorFlow for
   ROCm on WSL" kılavuzuna göre kurun.
4. Projeyi WSL içinde çalıştırın: `python src/train_spread.py`. `setup_device()` GPU'yu
   otomatik bulur ve `mixed_float16` (daha hızlı, daha az VRAM) açar.

**Alternatif — DirectML (native Windows, herhangi bir DX12 GPU).** Kurulumu kolaydır ama
**TensorFlow 2.10 / Python 3.10** ile sınırlıdır (eski sürüm):

```bash
py -3.10 -m pip install "tensorflow-cpu==2.10" tensorflow-directml-plugin
```

Öneri: denemelere CPU ile başlayın; tam veri setinde hız gerekiyorsa WSL2 + ROCm'e geçin.

### Hata Ayıklama

- **`Can't parse serialized Example ... Key: fire`** veya eğitimde **0 geçerli kare:**
  Eski notebook, örnekleme noktalarına skaler bir `fire` özelliği ekliyor; bu, 65×65 `fire`
  bandıyla çakışıp onu 1×1'e indirerek bozuyordu. **Çözüm:** güncel `colab_notebook.ipynb`
  ile veriyi **yeniden çekin** (nokta özellikleri artık atılıyor, `unmask(0, False)` ile
  yamalar tam boyutlu geliyor). Yükleyici ayrıca bozuk/kısa kareleri otomatik filtreler.
- **`TensorFlow GPU support is not available on native Windows`:** Zararsız bir uyarıdır;
  eğitim CPU'da çalışır. GPU için yukarıdaki "AMD GPU ile Çalıştırma" bölümüne bakın.

---

## 12. Proje Yapısı

```
ignis/
├── README.md                    ← BU DOSYA (projenin tek kaynağı)
├── start.py                     ← tek komutla eğitim başlatıcı
├── requirements.txt
├── noteboks/
│   └── colab_notebook.ipynb     ← GEE next-day spread veri çekme (raster)
├── src/
│   ├── config.py                ← tüm sabitler (SPREAD_* bölümü aktif)
│   ├── gee_config.py            ← GEE koleksiyon/kanal/patch ayarları
│   ├── spread_dataset.py        ← TFRecord → tf.data (X:64×64×14, y:64×64×1)
│   ├── spread_model.py          ← U-Net + focal/weighted-BCE kayıp
│   ├── train_spread.py          ← eğitim döngüsü → models/spread_unet.keras
│   ├── evaluate_spread.py       ← sonuç/doğruluk raporu → outputs/reports/
│   ├── utils.py                 ← yardımcılar
│   └── (LEGACY) preprocess.py, train.py, predict.py, test_accuracy.py,
│        map_visualization.py, main.py, examples.py, gee_data_processor.py
├── data/
│   └── spread/                  ← buraya *.tfrecord.gz koyun
├── models/                      ← spread_unet.keras (eğitim çıktısı)
└── outputs/                     ← raporlar, haritalar, tahminler
```

**LEGACY dosyalar:** `preprocess.py`, `train.py`, `predict.py`, `test_accuracy.py`,
`map_visualization.py`, eski `data/raw/*.csv` ve `models/yangin_model.keras` **eski statik
risk modeline** aittir. Yangın büyüme pipeline'ında kullanılmazlar; referans için
tutulmaktadır. İstenirse silinebilirler.

---

## 13. Konfigürasyon Referansı

`src/config.py → SPREAD_*` ve `src/gee_config.py → GEEConfig` başlıca ayarlar:

| Sabit | Varsayılan | Anlamı |
|---|---|---|
| `SCALE` | 1000 m | Analiz gridi çözünürlüğü |
| `CRS` | EPSG:32635 | Projeksiyon (UTM 35N) |
| `PATCH_RADIUS` / `PATCH_SIZE` | 32 / 65 | Yama yarıçapı / boyutu |
| `MODEL_PATCH` | 64 | Eğitimde kırpılan boyut |
| `FIRE_CONFIDENCE` | 7 | MODIS FireMask yangın eşiği (7=düşük güven) |
| `MIN_FIRE_PIXELS` | 5 | Bir günü işlemek için min aktif yangın |
| `MAX_POINTS_PER_DAY` | 150 | Gün başına yama sayısı |
| `FIRE_SEASON_MONTHS` | (6, 10) | Haziran–Ekim |
| `YEARS` | 2019–2024 | Veri yılları |
| `GROWTH_GROW_RATIO` | 1.15 | grow eşiği |
| `GROWTH_STABLE_LOW` | 0.85 | stable alt sınırı |
| `SPREAD_POS_WEIGHT` | 10 | Pozitif sınıf ağırlığı |

---

## 14. Kısıtlar ve Gelecek Çalışma

- **Çözünürlük:** MODIS aktif yangın 1 km'dir; küçük/erken yangınlar gözden kaçabilir.
  VIIRS (`NOAA/VIIRS/001/VNP14A1`, 375–750 m) daha ince cephe verir — gelecek çalışma.
- **Zaman adımı:** Günlük MODIS geçişleri bulut ve yörünge nedeniyle boşluklu olabilir;
  Terra+Aqua birleştirilse de sub-günlük dinamik yakalanmaz.
- **Arazi örtüsü statik:** MCD12Q1 yıllıktır; yangın sonrası değişimi anlık yansıtmaz.
- **Sınıf dengesizliği:** Yangın pikselleri seyrektir; focal loss ve pozitif ağırlık
  kullanılsa da eşik kalibrasyonu (PR eğrisi) gereklidir.
- **Öneriler:** ConvLSTM ile çok-günlük diziler; rüzgârı vektörel yönde açık modelleme;
  yakıt nem endeksleri (dead/live fuel moisture); baz çizgisi karşılaştırmaları.

---

## 15. Sözlük ve Kısaltmalar

- **NDVI** — Normalize Edilmiş Fark Bitki Örtüsü İndeksi.
- **LST** — Land Surface Temperature (Arazi Yüzey Sıcaklığı).
- **EO** — Earth Observation (Dünya Gözlem).
- **GEE** — Google Earth Engine.
- **ERA5-Land** — ECMWF yeniden analiz meteoroloji veri seti.
- **CHIRPS** — Climate Hazards Group InfraRed Precipitation with Station data.
- **SRTM** — Shuttle Radar Topography Mission (topografya).
- **U-Net** — encoder-decoder + skip bağlantılı segmentasyon ağı.
- **AUC-PR** — Precision-Recall eğrisi altındaki alan.
- **IoU** — Intersection over Union (maske örtüşme metriği).
- **TFRecord** — TensorFlow'un ikili veri formatı.

---

## 16. Kaynakça

**İncelenen makaleler**

1. *Environmental Reviews*, er-2020-0019 — ML tabanlı yangın modelleme derlemesi.
2. *Symmetry* 12(6):1022 (MDPI) — yangın tahmininde ML.
3. *Remote Sensing* 16(8):1467 (MDPI) — uydu tabanlı yangın yayılımı.
4. arXiv:2406.14591 — physics-informed sinir ağlarıyla yangın yayılımı.
5. Nature *Scientific Reports* (2024), s41598-024-52821-x — yangın modelleme.
6. arXiv:2505.17556 — derin öğrenmeyle yangın.
7. *Fire* 7(12):482 (MDPI) — yangın yayılım tahmini.
8. *Applied Sciences* 13(14):8275 (MDPI) — yangın yayılım modelleme.

**Veri kaynakları**

- MODIS MOD14A1/MYD14A1 (Thermal Anomalies/Fire Daily), MOD13Q1 (NDVI), MOD11A1 (LST),
  MCD12Q1 (Land Cover) — NASA LP DAAC.
- ERA5-Land Daily Aggregated — ECMWF / Copernicus.
- CHIRPS Daily — UCSB Climate Hazards Group.
- SRTM GL1 — NASA/USGS.

**Klasik referans**

- Rothermel, R.C. (1972). *A mathematical model for predicting fire spread in wildland
  fuels.* USDA Forest Service RMRS.
