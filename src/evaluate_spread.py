"""
Yangın Büyüme Modeli — Değerlendirme & Raporlama
=================================================
Eğitilmiş U-Net'i doğrulama verisiyle test eder ve ÜÇ tür çıktı üretir:

  1) DETAYLI (makale için) : outputs/reports/spread_report.html
       - PR & ROC eğrileri, karışıklık matrisi, metrik barları, öğrenme eğrileri,
         örnek tahminler; hepsi tek bir makale görünümlü HTML'de + yorum metni.
       - Ayrıca yüksek çözünürlüklü figür: outputs/reports/spread_figures.png
  2) BASİT (sunum için)     : outputs/reports/spread_scorecard.png
       - Konuyu bilmeyen birinin anlayacağı sade "skor kartı" + net doğruluk.
  3) HARİTA (havalı)        : outputs/reports/spread_map.html
       - Yangın karelerini Türkiye haritasında tahmin sınıfına göre renklendirir.
         (Yalnızca veri lon/lat içeriyorsa; güncel notebook bunu ekliyor.)

  + Ham metrikler           : outputs/reports/spread_evaluation.txt

Kullanım:
    python src/evaluate_spread.py
"""

from __future__ import annotations
import base64
import numpy as np

from config import (
    SPREAD_MODEL_FILE, SPREAD_TFRECORD_GLOB, SPREAD_BATCH_SIZE,
    SPREAD_INPUT_BANDS, SPREAD_TARGET_BAND, GROWTH_CLASSES,
    GROWTH_GROW_RATIO, GROWTH_STABLE_LOW, PATCH_SIZE, MODEL_PATCH,
    REPORTS_DIR, ensure_directories_exist,
)

ALL_BANDS = SPREAD_INPUT_BANDS + [SPREAD_TARGET_BAND]
_FIRE_IDX = SPREAD_INPUT_BANDS.index("fire")
_N = PATCH_SIZE * PATCH_SIZE
CLS_TR = {0: "Sönüyor", 1: "Sabit", 2: "Büyüyor"}      # Türkçe etiketler
CLS_COLOR = {0: "#2e9e4f", 1: "#e8a13a", 2: "#d43d2f"}  # yeşil / turuncu / kırmızı


# ------------------------------------------------------------------
# Tek geçişli, DETERMİNİSTİK okuyucu (metrik + koordinat hizalı)
# ------------------------------------------------------------------
def _eval_dataset(files, batch_size):
    import tensorflow as tf
    desc = {b: tf.io.VarLenFeature(tf.float32) for b in ALL_BANDS}
    desc["lon"] = tf.io.VarLenFeature(tf.float32)
    desc["lat"] = tf.io.VarLenFeature(tf.float32)

    def parse(proto):
        p = tf.io.parse_single_example(proto, desc)
        dense = {b: tf.sparse.to_dense(p[b]) for b in ALL_BANDS}
        valid = tf.reduce_all(tf.stack([tf.equal(tf.size(dense[b]), _N) for b in ALL_BANDS]))
        lon = tf.sparse.to_dense(p["lon"]); lat = tf.sparse.to_dense(p["lat"])
        lon = tf.cond(tf.size(lon) > 0, lambda: lon[0], lambda: tf.constant(float("nan")))
        lat = tf.cond(tf.size(lat) > 0, lambda: lat[0], lambda: tf.constant(float("nan")))
        return dense, valid, lon, lat

    def to_xy(dense, valid, lon, lat):
        chans = [tf.reshape(dense[b], (PATCH_SIZE, PATCH_SIZE)) for b in SPREAD_INPUT_BANDS]
        x = tf.stack(chans, axis=-1)
        y = tf.expand_dims(tf.reshape(dense[SPREAD_TARGET_BAND], (PATCH_SIZE, PATCH_SIZE)), -1)
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        y = tf.clip_by_value(tf.where(tf.math.is_finite(y), y, tf.zeros_like(y)), 0.0, 1.0)
        x = x[:MODEL_PATCH, :MODEL_PATCH, :]; y = y[:MODEL_PATCH, :MODEL_PATCH, :]
        return x, y, lon, lat

    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")   # sıralı = deterministik
    ds = ds.map(parse).filter(lambda d, v, lo, la: v).map(to_xy)
    return ds.batch(batch_size)


def _growth(n_today, n_next):
    r = n_next / max(n_today, 1.0)
    return 2 if r > GROWTH_GROW_RATIO else (1 if r >= GROWTH_STABLE_LOW else 0)


def _b64(fig):
    import io
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0); return base64.b64encode(buf.read()).decode("ascii")


# ------------------------------------------------------------------
# Ana değerlendirme
# ------------------------------------------------------------------
def evaluate(glob_pattern: str = SPREAD_TFRECORD_GLOB,
             model_path=SPREAD_MODEL_FILE,
             threshold: float = 0.5,
             report_dir=REPORTS_DIR):
    import glob
    from pathlib import Path
    import tensorflow as tf
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 confusion_matrix, precision_recall_fscore_support,
                                 precision_recall_curve, roc_curve)

    ensure_directories_exist()
    report_dir = Path(report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"TFRecord yok: {glob_pattern}")

    model = tf.keras.models.load_model(str(model_path), compile=False)

    yt, yp, gc_true, gc_pred = [], [], [], []
    lons, lats = [], []
    samples, n = [], 0
    for xb, yb, lob, lab in _eval_dataset(files, SPREAD_BATCH_SIZE):
        pb = model.predict(xb, verbose=0)
        x = xb.numpy(); y = yb.numpy()
        yt.append(y.ravel()); yp.append(pb.ravel())
        lo = lob.numpy(); la = lab.numpy()
        for b in range(x.shape[0]):
            n += 1
            nt = float(x[b, ..., _FIRE_IDX].sum()); nn = float((pb[b, ..., 0] > threshold).sum())
            gc_pred.append(_growth(nt, nn)); gc_true.append(_growth(nt, float(y[b, ..., 0].sum())))
            lons.append(float(lo[b])); lats.append(float(la[b]))
            if len(samples) < 3:
                samples.append((x[b, ..., _FIRE_IDX], y[b, ..., 0], pb[b, ..., 0]))
    if n == 0:
        raise RuntimeError("Değerlendirilecek geçerli kare yok. Veriyi güncel notebook ile çekip eğitin.")

    yt = np.concatenate(yt); yp = np.concatenate(yp)
    ytb = (yt > 0.5).astype(int); ypb = (yp > threshold).astype(int)
    gc_true = np.array(gc_true); gc_pred = np.array(gc_pred)

    # --- metrikler ---
    ap = average_precision_score(ytb, yp) if ytb.max() else float("nan")
    roc = roc_auc_score(ytb, yp) if 0 < ytb.mean() < 1 else float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(ytb, ypb, average="binary", zero_division=0)
    inter = np.logical_and(ytb == 1, ypb == 1).sum(); union = np.logical_or(ytb == 1, ypb == 1).sum()
    iou = inter / union if union else float("nan")
    cm = confusion_matrix(gc_true, gc_pred, labels=[0, 1, 2])
    growth_acc = np.trace(cm) / cm.sum() if cm.sum() else float("nan")
    _, _, gmac_f1, _ = precision_recall_fscore_support(gc_true, gc_pred, labels=[0, 1, 2],
                                                       average="macro", zero_division=0)
    has_coords = np.isfinite(lons).any() and np.isfinite(lats).any()

    metrics = dict(ap=ap, roc=roc, precision=prec, recall=rec, f1=f1, iou=iou,
                   growth_accuracy=growth_acc, growth_macro_f1=gmac_f1,
                   frames=n, pos_rate=float(ytb.mean()))

    # --- ham metin ---
    _write_txt(report_dir, metrics, cm, len(files), threshold)
    # --- figürler ---
    fig_b64 = _make_figures(report_dir, ytb, yp, cm, samples, metrics, threshold)
    card_b64 = _make_scorecard(report_dir, metrics, cm)
    # --- HTML rapor ---
    _write_html(report_dir, metrics, cm, fig_b64, card_b64, has_coords)
    # --- harita ---
    map_msg = _make_map(report_dir, lons, lats, gc_true, gc_pred) if has_coords else \
        "Harita atlandı: veride konum (lon/lat) yok. Güncel notebook ile yeniden çekince gelir."

    print("\n" + "=" * 60)
    print(f"NET DOĞRULUK (büyüme sınıfı): %{growth_acc*100:.1f}   "
          f"| AUC-PR: {ap:.3f} | IoU: {iou:.3f}")
    print("=" * 60)
    print(f"📄 Makale raporu : {report_dir/'spread_report.html'}")
    print(f"🖼️  Figürler      : {report_dir/'spread_figures.png'}")
    print(f"🎯 Skor kartı    : {report_dir/'spread_scorecard.png'}")
    print(f"🗺️  Harita        : {map_msg}")
    return metrics


# ------------------------------------------------------------------
# Ham metin
# ------------------------------------------------------------------
def _write_txt(report_dir, m, cm, nfiles, thr):
    L = ["=" * 60, "YANGIN BÜYÜME MODELİ — DEĞERLENDİRME", "=" * 60,
         f"Dosya: {nfiles} | Kare: {m['frames']} | Eşik: {thr}", "",
         "PİKSEL DÜZEYİ (ertesi gün yayılım maskesi):",
         f"  AUC-PR : {m['ap']:.4f}   ROC-AUC: {m['roc']:.4f}",
         f"  Precision/Recall/F1 : {m['precision']:.4f} / {m['recall']:.4f} / {m['f1']:.4f}",
         f"  IoU : {m['iou']:.4f}   Pozitif piksel oranı: {m['pos_rate']:.4%}", "",
         "YAMA DÜZEYİ (Sönüyor / Sabit / Büyüyor):",
         f"  NET DOĞRULUK : {m['growth_accuracy']:.4f}   Makro-F1: {m['growth_macro_f1']:.4f}",
         "  Karışıklık matrisi (satır=gerçek, sütun=tahmin):",
         "              Sönüyor     Sabit    Büyüyor"]
    for i, name in CLS_TR.items():
        L.append(f"  {name:>8} " + "  ".join(f"{cm[i,j]:>8d}" for j in range(3)))
    L.append("=" * 60)
    (report_dir / "spread_evaluation.txt").write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------
# Detaylı figürler (makale)
# ------------------------------------------------------------------
def _make_figures(report_dir, ytb, yp, cm, samples, m, thr):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Yangın Büyüme Modeli — Performans Panosu", fontsize=16, weight="bold")

    # 1 PR
    ax = fig.add_subplot(2, 3, 1)
    if ytb.max():
        p_, r_, _ = precision_recall_curve(ytb, yp); ax.plot(r_, p_, color="#d43d2f")
        ax.axhline(m["pos_rate"], ls="--", c="gray", lw=1, label="rastgele taban")
        ax.legend(fontsize=8)
    ax.set_title(f"Precision–Recall (AP={m['ap']:.3f})"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.grid(alpha=.3)

    # 2 ROC
    ax = fig.add_subplot(2, 3, 2)
    if 0 < ytb.mean() < 1:
        fpr, tpr, _ = roc_curve(ytb, yp); ax.plot(fpr, tpr, color="#2e6fd4")
        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_title(f"ROC (AUC={m['roc']:.3f})"); ax.set_xlabel("Yanlış pozitif"); ax.set_ylabel("Doğru pozitif")
    ax.grid(alpha=.3)

    # 3 confusion
    ax = fig.add_subplot(2, 3, 3)
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLS_TR.values(), rotation=30, ha="right"); ax.set_yticklabels(CLS_TR.values())
    ax.set_title(f"Karışıklık Matrisi (Doğruluk %{m['growth_accuracy']*100:.0f})")
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", weight="bold")

    # 4 metric bars
    ax = fig.add_subplot(2, 3, 4)
    names = ["AUC-PR", "ROC", "Precision", "Recall", "F1", "IoU"]
    vals = [m["ap"], m["roc"], m["precision"], m["recall"], m["f1"], m["iou"]]
    ax.bar(names, vals, color="#2e6fd4"); ax.set_ylim(0, 1); ax.set_title("Piksel metrikleri")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, (v if np.isfinite(v) else 0) + .02, f"{v:.2f}", ha="center", fontsize=8)

    # 5 learning curves (varsa)
    ax = fig.add_subplot(2, 3, 5)
    hp = report_dir / "spread_history.json"
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        if h.get("loss"): ax.plot(h["loss"], label="eğitim loss")
        if h.get("val_loss"): ax.plot(h["val_loss"], label="doğrulama loss")
        if h.get("val_auc"): ax.plot(h["val_auc"], label="val AUC-PR")
        ax.legend(fontsize=8); ax.set_xlabel("Epoch"); ax.set_title("Öğrenme eğrileri")
    else:
        ax.text(.5, .5, "Öğrenme eğrisi için\neğitim geçmişi yok", ha="center", va="center")
        ax.axis("off")
    ax.grid(alpha=.3)

    # 6 sample prediction
    ax = fig.add_subplot(2, 3, 6)
    if samples:
        today, ytrue, ppred = samples[0]
        ax.imshow(today, cmap="Greys", alpha=.5)
        ax.contour(ytrue, levels=[.5], colors="#2e9e4f", linewidths=1.5)
        ax.imshow(ppred, cmap="Reds", alpha=.5)
        ax.set_title("Örnek: bugün(gri) • gerçek yarın(yeşil) • tahmin(kırmızı)", fontsize=9)
    ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    b64 = _b64(fig)
    fig.savefig(report_dir / "spread_figures.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return b64


# ------------------------------------------------------------------
# Basit skor kartı (sunum)
# ------------------------------------------------------------------
def _make_scorecard(report_dir, m, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = m["growth_accuracy"] * 100
    fig = plt.figure(figsize=(10, 6)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(.5, .92, "Yangın Büyüme Tahmini", ha="center", fontsize=22, weight="bold")
    ax.text(.5, .84, "Yanan bir orman yangını yarın ne yapacak?", ha="center", fontsize=13, color="#555")

    # üç sınıf kutusu
    for i, x0 in zip([0, 1, 2], [0.10, 0.40, 0.70]):
        ax.add_patch(plt.Rectangle((x0, 0.55), 0.20, 0.16, color=CLS_COLOR[i], alpha=.85))
        ax.text(x0 + .10, 0.63, CLS_TR[i], ha="center", va="center", color="white",
                fontsize=14, weight="bold")

    # büyük doğruluk
    ax.text(.5, .40, f"%{acc:.0f}", ha="center", fontsize=64, weight="bold", color="#d43d2f")
    ax.text(.5, .29, "doğrulukla doğru sınıfı tahmin ediyor", ha="center", fontsize=14)
    ax.text(.5, .19, f"(test edilen {m['frames']} yangın karesi üzerinde)", ha="center",
            fontsize=11, color="#777")
    ax.text(.5, .08, "Uydu verisiyle (bitki örtüsü, sıcaklık, rüzgâr, nem, arazi) çalışır.",
            ha="center", fontsize=11, color="#555")
    fig.savefig(report_dir / "spread_scorecard.png", dpi=130, facecolor="white", bbox_inches="tight")
    b64 = _b64(fig); plt.close(fig)
    return b64


# ------------------------------------------------------------------
# Makale görünümlü HTML rapor
# ------------------------------------------------------------------
def _write_html(report_dir, m, cm, fig_b64, card_b64, has_coords):
    def pct(v): return "—" if not np.isfinite(v) else f"{v*100:.1f}%"
    def f3(v): return "—" if not np.isfinite(v) else f"{v:.3f}"
    rows = "".join(
        f"<tr><td>{CLS_TR[i]}</td>" + "".join(
            f"<td class='{'hit' if i==j else ''}'>{cm[i,j]}</td>" for j in range(3)) + "</tr>"
        for i in range(3))
    map_note = ("<p>Harita: <a href='spread_map.html'>spread_map.html</a></p>" if has_coords
                else "<p><em>Harita, veri konum (lon/lat) içerince otomatik üretilir.</em></p>")
    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>Yangın Büyüme Modeli — Rapor</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:900px;margin:32px auto;
      padding:0 20px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #d43d2f;padding-bottom:8px}}
 h2{{margin-top:34px;color:#b5311f}}
 .kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
 .kpi div{{flex:1;min-width:150px;background:#faf5f4;border:1px solid #eee;border-radius:12px;
          padding:16px;text-align:center}}
 .kpi b{{display:block;font-size:30px;color:#d43d2f}}
 table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #ddd;padding:8px 14px;text-align:center}}
 th{{background:#f4f4f4}} td.hit{{background:#d6f0d8;font-weight:bold}}
 img{{max-width:100%;border:1px solid #eee;border-radius:8px;margin:10px 0}}
 .muted{{color:#777;font-size:14px}}
</style></head><body>
<h1>🔥 Yangın Büyüme (Yayılım) Tahmini — Değerlendirme Raporu</h1>
<p class="muted">Uydu tabanlı Dünya Gözlem verisiyle eğitilmiş U-Net segmentasyon modeli.
Değerlendirilen kare sayısı: <b>{m['frames']}</b>.</p>

<div class="kpi">
 <div><b>%{m['growth_accuracy']*100:.0f}</b>Net doğruluk<br><span class="muted">Sönüyor/Sabit/Büyüyor</span></div>
 <div><b>{f3(m['ap'])}</b>AUC-PR<br><span class="muted">yayılım maskesi</span></div>
 <div><b>{f3(m['iou'])}</b>IoU<br><span class="muted">alan örtüşmesi</span></div>
 <div><b>{f3(m['roc'])}</b>ROC-AUC</div>
</div>

<h2>1. Basit özet (sunum)</h2>
<p>Model, halihazırda yanan bir orman yangınının <b>ertesi gün</b> büyüyeceğini, aynı kalacağını
mı yoksa söneceğini mi tahmin eder. Test verisinde büyüme sınıfını <b>%{m['growth_accuracy']*100:.0f}</b>
doğrulukla bilmektedir.</p>
<img src="data:image/png;base64,{card_b64}" alt="skor karti">

<h2>2. Detaylı performans (makale)</h2>
<img src="data:image/png;base64,{fig_b64}" alt="figurler">
<p>Piksel düzeyinde model, ertesi günün yangın maskesini tahmin eder. Yangın pikselleri çok
seyrek olduğundan (pozitif oran %{m['pos_rate']*100:.1f}) düz doğruluk yanıltıcıdır; bu yüzden
<b>AUC-PR</b>, <b>IoU</b> ve <b>F1</b> raporlanır.</p>
<table>
 <tr><th>Metrik</th><th>Değer</th></tr>
 <tr><td>AUC-PR (Average Precision)</td><td>{f3(m['ap'])}</td></tr>
 <tr><td>ROC-AUC</td><td>{f3(m['roc'])}</td></tr>
 <tr><td>Precision / Recall / F1</td><td>{f3(m['precision'])} / {f3(m['recall'])} / {f3(m['f1'])}</td></tr>
 <tr><td>IoU (maske örtüşmesi)</td><td>{f3(m['iou'])}</td></tr>
</table>

<h2>3. Büyüme sınıfı ayrıntısı</h2>
<p>Net doğruluk <b>%{m['growth_accuracy']*100:.0f}</b>, makro-F1 <b>{f3(m['growth_macro_f1'])}</b>.</p>
<table>
 <tr><th>gerçek ↓ / tahmin →</th><th>Sönüyor</th><th>Sabit</th><th>Büyüyor</th></tr>
 {rows}
</table>
{map_note}
<p class="muted">Otomatik üretildi — IGNIS yangın büyüme pipeline'ı.</p>
</body></html>"""
    (report_dir / "spread_report.html").write_text(html, encoding="utf-8")


# ------------------------------------------------------------------
# Harita (Folium)
# ------------------------------------------------------------------
def _make_map(report_dir, lons, lats, gc_true, gc_pred):
    try:
        import folium
    except Exception as e:
        return f"folium yok ({e}) — pip install folium"
    lons = np.array(lons); lats = np.array(lats)
    fmap = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="CartoDB positron")
    cap = min(len(lons), 800)
    for i in range(cap):
        if not (np.isfinite(lons[i]) and np.isfinite(lats[i])):
            continue
        p = int(gc_pred[i]); t = int(gc_true[i])
        folium.CircleMarker(
            location=[lats[i], lons[i]], radius=4, color=CLS_COLOR[p],
            fill=True, fill_color=CLS_COLOR[p], fill_opacity=0.8,
            popup=f"Tahmin: {CLS_TR[p]} | Gerçek: {CLS_TR[t]}",
        ).add_to(fmap)
    # basit lejant
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;"
              "padding:10px 14px;border:1px solid #ccc;border-radius:8px;font:13px sans-serif'>"
              "<b>Tahmin</b><br>"
              "<span style='color:#d43d2f'>●</span> Büyüyor<br>"
              "<span style='color:#e8a13a'>●</span> Sabit<br>"
              "<span style='color:#2e9e4f'>●</span> Sönüyor</div>")
    fmap.get_root().html.add_child(folium.Element(legend))
    out = report_dir / "spread_map.html"
    fmap.save(str(out))
    return str(out)


def main():
    evaluate()


if __name__ == "__main__":
    main()
