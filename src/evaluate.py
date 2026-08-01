"""
Evaluation — held-out test split only
Değerlendirme — yalnızca ayrılmış test bölmesi
==============================================

Replaces `evaluate_spread.py`. Two things changed beyond the framework port, and
both change the NUMBERS, not just the code:

1) TEST SPLIT ONLY.
   The old script globbed every shard in `data/spread/`, including the days the
   model trained on. Its scores were therefore partly in-sample and optimistic.
   This one reads the test years and nothing else.
   Eski betik TÜM parçaları okuyordu — eğitim günleri dâhil. Skorları örneklem
   içiydi ve iyimserdi. Bu betik yalnızca test yıllarını okur.

2) THRESHOLD CALIBRATED ON VALIDATION.
   The old script hard-coded 0.5. With 0.2686 % positives and a pos_weight'd
   loss the network is deliberately mis-calibrated, so 0.5 is arbitrary. The
   threshold that maximises F1 is chosen on the VALIDATION split and then
   applied unchanged to test. Choosing it on test would be fitting to the test
   set — the exact thing a held-out split exists to prevent.
   Eşik, DOĞRULAMA bölmesinde F1'i maksimize edecek şekilde seçilir ve test'e
   değiştirilmeden uygulanır. Test'te seçmek, test setine uydurmak olurdu.

The reporting half (HTML, scorecard, folium map) is ported from
`evaluate_spread.py` rather than rewritten, with baseline rows added so a
headline number can never appear without the bar it has to clear.
Raporlama kısmı yeniden yazılmadı, porte edildi; temel çizgi satırları eklendi.

Usage / kullanım
----------------
    python src/evaluate.py
    python src/evaluate.py --split test --version v3
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    GROWTH_GROW_RATIO,
    GROWTH_STABLE_LOW,
    REPORTS_DIR,
    SPREAD_BATCH_SIZE,
    SPREAD_MODEL_FILE,
    SPREAD_VERSION,
    ensure_directories_exist,
)
from baselines import BASELINES, binary_scores  # noqa: E402
from dataset import SpreadDataset  # noqa: E402
from device import autocast, get_device  # noqa: E402
from features import load_norm_stats  # noqa: E402
from model import UNet  # noqa: E402
from train import average_precision  # noqa: E402

CLS_TR = {0: "Sönüyor", 1: "Sabit", 2: "Büyüyor"}
CLS_EN = {0: "Extinguishing", 1: "Stable", 2: "Growing"}
CLS_COLOR = {0: "#2e9e4f", 1: "#e8a13a", 2: "#d43d2f"}

# Measured on the v1 archive (45 shards / 1054 patches). Shown for context only.
V1_REFERENCE = {
    "model": {"ap": 0.0210, "roc": 0.8468, "precision": 0.0601,
              "recall": 0.0222, "f1": 0.0324, "iou": 0.0165},
    "persistence": {"f1": 0.0595, "iou": 0.0306},
}


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def _load_model(device, path=SPREAD_MODEL_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}\nRun: python src/train.py")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNet(in_channels=ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_split(model, ds, device, batch_size=SPREAD_BATCH_SIZE):
    """Per-patch probability maps. / Yama başına olasılık haritaları."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    probs, targets, valids = [], [], []
    for x, y, v in loader:
        x = x.to(device)
        with autocast(device):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(y.numpy())
        valids.append(v.numpy())
    return (np.concatenate(probs)[:, 0], np.concatenate(targets)[:, 0],
            np.concatenate(valids)[:, 0])


# ----------------------------------------------------------------------
# Threshold calibration
# ----------------------------------------------------------------------
def calibrate_threshold(prob, target, valid, n_steps=200):
    """Threshold maximising F1. MUST be run on validation, never on test.
    F1'i maksimize eden eşik. Test'te DEĞİL, doğrulamada çalıştırılmalıdır."""
    m = valid > 0.5
    p, t = prob[m], target[m] > 0.5
    if t.sum() == 0:
        return 0.5, float("nan")
    lo, hi = float(np.percentile(p, 50)), float(p.max())
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(lo, hi, n_steps):
        pred = p >= thr
        tp = float((pred & t).sum())
        if tp == 0:
            continue
        fp = float((pred & ~t).sum())
        fn = float((~pred & t).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    return best_thr, best_f1


def roc_auc(y_true, y_score):
    """ROC-AUC via rank statistic. / Sıra istatistiğiyle ROC-AUC."""
    y = np.asarray(y_true).ravel() > 0.5
    s = np.asarray(y_score).ravel()
    n_pos, n_neg = float(y.sum()), float((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within ties.
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def growth_class(n_today, n_next):
    r = n_next / max(n_today, 1.0)
    return 2 if r > GROWTH_GROW_RATIO else (1 if r >= GROWTH_STABLE_LOW else 0)


# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------
def evaluate(version=SPREAD_VERSION, split="test", model_path=SPREAD_MODEL_FILE,
             report_dir=REPORTS_DIR, threshold=None):
    ensure_directories_exist()
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("IGNIS — evaluation / değerlendirme")
    print("=" * 66)
    device = get_device()
    stats = load_norm_stats()
    model, ckpt = _load_model(device, model_path)

    # --- threshold from VALIDATION -----------------------------------
    if threshold is None:
        val_ds = SpreadDataset("val", version=version, stats=stats, augment=False)
        vp, vt, vv = predict_split(model, val_ds, device)
        threshold, val_f1 = calibrate_threshold(vp, vt, vv)
        print(f"  threshold       : {threshold:.4f} "
              f"(maximises F1={val_f1:.4f} on VALIDATION, {len(val_ds)} patches)")
        del vp, vt, vv
    else:
        val_f1 = float("nan")
        print(f"  threshold       : {threshold:.4f} (supplied)")

    # --- test split ---------------------------------------------------
    ds = SpreadDataset(split, version=version, stats=stats, augment=False)
    print(f"  split           : {split}  ({len(ds)} patches)")
    print(f"  version         : {version}")
    prob, target, valid = predict_split(model, ds, device)

    m = valid > 0.5
    p_flat, t_flat = prob[m], (target[m] > 0.5)
    pred_flat = p_flat >= threshold

    tp = float((pred_flat & t_flat).sum())
    fp = float((pred_flat & ~t_flat).sum())
    fn = float((~pred_flat & t_flat).sum())
    model_scores = binary_scores(tp, fp, fn)
    ap = average_precision(t_flat, p_flat)
    roc = roc_auc(t_flat, p_flat)
    prevalence = float(t_flat.mean())

    # --- baselines on the SAME pixels ---------------------------------
    base_counts = {k: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for k in BASELINES}
    gc_true, gc_pred, lons, lats, samples = [], [], [], [], []
    for i in range(len(ds)):
        today = ds.today_mask(i)
        tgt = ds.target_mask(i) > 0.5
        vmask = ds.valid_mask(i) > 0.5
        wind = ds.wind_uv(i)
        for k, fn_ in BASELINES.items():
            pr = fn_(today, wind=wind) > 0.5
            a, b = pr & vmask, tgt & vmask
            base_counts[k]["tp"] += float((a & b).sum())
            base_counts[k]["fp"] += float((a & ~b).sum())
            base_counts[k]["fn"] += float((~a & b).sum())

        n_today = float(today.sum())
        gc_true.append(growth_class(n_today, float(tgt.sum())))
        gc_pred.append(growth_class(n_today, float(((prob[i] >= threshold) & vmask).sum())))
        lon, lat = ds.coords(i)
        lons.append(lon)
        lats.append(lat)
        if len(samples) < 3:
            samples.append((today, tgt.astype(float), prob[i]))

    baselines = {k: binary_scores(c["tp"], c["fp"], c["fn"])
                 for k, c in base_counts.items()}

    gc_true, gc_pred = np.array(gc_true), np.array(gc_pred)
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(gc_true, gc_pred):
        cm[a, b] += 1
    growth_acc = float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")
    majority = float(np.bincount(gc_true, minlength=3).max() / len(gc_true)) \
        if len(gc_true) else float("nan")

    metrics = {
        "ap": ap, "roc": roc, **model_scores,
        "threshold": threshold, "val_f1": val_f1,
        "growth_accuracy": growth_acc, "majority_class_share": majority,
        "prevalence": prevalence, "patches": len(ds), "split": split,
        "version": version, "checkpoint_epoch": ckpt.get("epoch"),
        "target_mode": ds.target_mode,
    }

    best_base = max(baselines.items(),
                    key=lambda kv: kv[1]["iou"] if np.isfinite(kv[1]["iou"]) else -1)
    beats = (np.isfinite(model_scores["iou"])
             and model_scores["iou"] > best_base[1]["iou"])
    metrics["beats_baseline"] = bool(beats)
    metrics["best_baseline"] = best_base[0]

    # --- outputs ------------------------------------------------------
    _write_txt(report_dir, metrics, baselines, cm)
    fig_b64 = _make_figures(report_dir, t_flat, p_flat, cm, samples, metrics,
                            baselines)
    card_b64 = _make_scorecard(report_dir, metrics, baselines)
    _write_html(report_dir, metrics, baselines, cm, fig_b64, card_b64)
    map_msg = _make_map(report_dir, lons, lats, gc_true, gc_pred)

    (report_dir / "spread_metrics.json").write_text(
        json.dumps({"model": metrics, "baselines": baselines}, indent=2),
        encoding="utf-8")

    # --- console ------------------------------------------------------
    print("=" * 66)
    print(f"  {'':<24}{'precision':>11}{'recall':>10}{'F1':>10}{'IoU':>10}")
    print("  " + "-" * 63)
    print(f"  {'MODEL':<24}{model_scores['precision']:>11.4f}"
          f"{model_scores['recall']:>10.4f}{model_scores['f1']:>10.4f}"
          f"{model_scores['iou']:>10.4f}")
    for k, v in baselines.items():
        print(f"  {k:<24}{v['precision']:>11.4f}{v['recall']:>10.4f}"
              f"{v['f1']:>10.4f}{v['iou']:>10.4f}")
    print("  " + "-" * 63)
    print(f"  AUC-PR {ap:.4f}   ROC-AUC {roc:.4f}   "
          f"positive rate {prevalence * 100:.4f}%")
    print(f"  patch accuracy {growth_acc:.4f}   "
          f"majority-class share {majority:.4f}")
    print("=" * 66)
    if beats:
        print(f"  MODEL BEATS the best baseline ({best_base[0]}, "
              f"IoU {best_base[1]['iou']:.4f}).")
    else:
        print(f"  MODEL LOSES to {best_base[0]} "
              f"(IoU {best_base[1]['iou']:.4f} vs {model_scores['iou']:.4f}).")
        print(f"  Report this as measured. Do not tune on the test split.")
        print(f"  Bunu ölçüldüğü gibi raporlayın; test bölmesinde ayar yapmayın.")
    if np.isfinite(growth_acc) and np.isfinite(majority) and growth_acc < majority:
        print(f"  NOTE: patch accuracy {growth_acc:.4f} is BELOW the "
              f"majority-class share {majority:.4f}.")
    print("=" * 66)
    print(f"  report   : {report_dir / 'spread_report.html'}")
    print(f"  figures  : {report_dir / 'spread_figures.png'}")
    print(f"  scorecard: {report_dir / 'spread_scorecard.png'}")
    print(f"  map      : {map_msg}")
    return metrics, baselines


# ----------------------------------------------------------------------
# Reporting (ported from evaluate_spread.py)
# ----------------------------------------------------------------------
def _b64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _write_txt(report_dir, m, baselines, cm):
    L = ["=" * 66,
         "YANGIN BÜYÜME MODELİ — DEĞERLENDİRME / EVALUATION",
         "=" * 66,
         f"Sürüm/version: {m['version']}   Bölme/split: {m['split']}   "
         f"Kare/patches: {m['patches']}",
         f"Eşik/threshold: {m['threshold']:.4f} (doğrulamada kalibre edildi / "
         f"calibrated on validation)",
         f"Hedef/target: {m['target_mode']}",
         "",
         "PİKSEL DÜZEYİ / PIXEL LEVEL:",
         f"  AUC-PR  : {m['ap']:.4f}    ROC-AUC : {m['roc']:.4f}",
         f"  P/R/F1  : {m['precision']:.4f} / {m['recall']:.4f} / {m['f1']:.4f}",
         f"  IoU     : {m['iou']:.4f}",
         f"  Pozitif piksel oranı / prevalence: {m['prevalence']:.4%}",
         "",
         "TEMEL ÇİZGİLER / BASELINES (aynı piksellerde / same pixels):"]
    for k, v in baselines.items():
        L.append(f"  {k:<22} F1 {v['f1']:.4f}   IoU {v['iou']:.4f}")
    L += ["",
          f"Model en iyi temel çizgiyi geçiyor mu? / beats baseline: "
          f"{'EVET / YES' if m['beats_baseline'] else 'HAYIR / NO'}",
          "",
          "YAMA DÜZEYİ / PATCH LEVEL (Sönüyor / Sabit / Büyüyor):",
          f"  Doğruluk / accuracy      : {m['growth_accuracy']:.4f}",
          f"  Çoğunluk sınıfı / majority: {m['majority_class_share']:.4f}",
          "  Karışıklık matrisi (satır=gerçek, sütun=tahmin):",
          "              Sönüyor     Sabit    Büyüyor"]
    for i, name in CLS_TR.items():
        L.append(f"  {name:>8} " + "  ".join(f"{cm[i, j]:>8d}" for j in range(3)))
    L.append("=" * 66)
    (report_dir / "spread_evaluation.txt").write_text("\n".join(L), encoding="utf-8")


def _make_figures(report_dir, ytb, yp, cm, samples, m, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Yangın Büyüme Modeli — Performans Panosu / Performance Dashboard",
                 fontsize=15, weight="bold")

    # 1 PR curve
    ax = fig.add_subplot(2, 3, 1)
    if ytb.sum():
        order = np.argsort(-yp)
        y = ytb[order].astype(float)
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        prec = tp / np.maximum(tp + fp, 1e-12)
        rec = tp / ytb.sum()
        ax.plot(rec, prec, color="#d43d2f")
        ax.axhline(m["prevalence"], ls="--", c="gray", lw=1,
                   label=f"random ({m['prevalence']:.4f})")
        ax.legend(fontsize=8)
    ax.set_title(f"Precision–Recall (AP={m['ap']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=.3)

    # 2 model vs baselines — the comparison that matters
    ax = fig.add_subplot(2, 3, 2)
    names = ["MODEL"] + list(baselines)
    ious = [m["iou"]] + [v["iou"] for v in baselines.values()]
    colors = ["#2e6fd4"] + ["#999999"] * len(baselines)
    ax.barh(range(len(names)), ious, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("IoU: model vs baselines")
    ax.grid(alpha=.3, axis="x")
    for i, v in enumerate(ious):
        if np.isfinite(v):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)

    # 3 confusion matrix
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLS_TR.values(), rotation=30, ha="right")
    ax.set_yticklabels(CLS_TR.values())
    ax.set_title(f"Karışıklık Matrisi (acc {m['growth_accuracy']:.3f} / "
                 f"majority {m['majority_class_share']:.3f})", fontsize=9)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", weight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    # 4 pixel metric bars
    ax = fig.add_subplot(2, 3, 4)
    labels = ["AUC-PR", "ROC", "Prec", "Recall", "F1", "IoU"]
    vals = [m["ap"], m["roc"], m["precision"], m["recall"], m["f1"], m["iou"]]
    ax.bar(labels, vals, color="#2e6fd4")
    ax.set_ylim(0, 1)
    ax.set_title("Piksel metrikleri / pixel metrics")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, (v if np.isfinite(v) else 0) + .02, f"{v:.3f}",
                ha="center", fontsize=8)

    # 5 learning curves
    ax = fig.add_subplot(2, 3, 5)
    hp = report_dir / "spread_history.json"
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        if h.get("train_loss"):
            ax.plot(h["train_loss"], label="train loss")
        if h.get("val_loss"):
            ax.plot(h["val_loss"], label="val loss")
        if h.get("val_ap"):
            ax.plot(h["val_ap"], label="val AUC-PR")
        ax.legend(fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_title("Öğrenme eğrileri / learning curves")
    else:
        ax.text(.5, .5, "eğitim geçmişi yok\nno training history",
                ha="center", va="center")
        ax.axis("off")
    ax.grid(alpha=.3)

    # 6 sample prediction
    ax = fig.add_subplot(2, 3, 6)
    if samples:
        today, ytrue, ppred = samples[0]
        ax.imshow(today, cmap="Greys", alpha=.5)
        ax.contour(ytrue, levels=[.5], colors="#2e9e4f", linewidths=1.5)
        ax.imshow(ppred, cmap="Reds", alpha=.5)
        ax.set_title("bugün(gri) • gerçek(yeşil) • tahmin(kırmızı)", fontsize=9)
    ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    b64 = _b64(fig)
    fig.savefig(report_dir / "spread_figures.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return b64


def _make_scorecard(report_dir, m, baselines):
    """Plain-language scorecard.

    The original showed one big accuracy number. That number is an artefact of
    class prevalence, so the model's IoU is shown against the best baseline's
    instead — a headline figure should never appear without the bar it must clear.
    Özgün skor kartı tek bir büyük doğruluk sayısı gösteriyordu; o sayı sınıf
    oranının bir yan ürünüdür. Yerine model IoU'su temel çizgiyle karşılaştırılır.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = max(baselines.items(),
               key=lambda kv: kv[1]["iou"] if np.isfinite(kv[1]["iou"]) else -1)
    wins = m["beats_baseline"]

    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(.5, .93, "Yangın Büyüme Tahmini", ha="center", fontsize=22, weight="bold")
    ax.text(.5, .86, "Yanan bir orman yangını yarın ne yapacak?",
            ha="center", fontsize=13, color="#555")

    for i, x0 in zip([0, 1, 2], [0.10, 0.40, 0.70]):
        ax.add_patch(plt.Rectangle((x0, 0.68), 0.20, 0.11, color=CLS_COLOR[i], alpha=.85))
        ax.text(x0 + .10, 0.735, CLS_TR[i], ha="center", va="center",
                color="white", fontsize=13, weight="bold")

    ax.text(.28, .52, "MODEL", ha="center", fontsize=13, color="#555")
    ax.text(.28, .38, f"{m['iou']:.4f}", ha="center", fontsize=42, weight="bold",
            color="#2e6fd4")
    ax.text(.72, .52, f"EN İYİ TEMEL ÇİZGİ\n({best[0].replace('_', ' ')})",
            ha="center", fontsize=11, color="#555")
    ax.text(.72, .38, f"{best[1]['iou']:.4f}", ha="center", fontsize=42,
            weight="bold", color="#888")
    ax.text(.5, .29, "IoU — alan örtüşmesi (yüksek olan iyi)",
            ha="center", fontsize=11, color="#777")

    verdict = ("Model temel çizgiyi GEÇİYOR" if wins
               else "Model temel çizgiyi GEÇEMİYOR")
    ax.text(.5, .17, verdict, ha="center", fontsize=16, weight="bold",
            color="#2e9e4f" if wins else "#d43d2f")
    ax.text(.5, .09, f"{m['patches']} yangın karesi · {m['split']} bölmesi "
                     f"· eşik {m['threshold']:.3f}",
            ha="center", fontsize=10, color="#777")
    ax.text(.5, .03, "Uydu verisiyle (bitki örtüsü, sıcaklık, rüzgâr, nem, arazi).",
            ha="center", fontsize=10, color="#555")

    fig.savefig(report_dir / "spread_scorecard.png", dpi=130, facecolor="white",
                bbox_inches="tight")
    b64 = _b64(fig)
    plt.close(fig)
    return b64


def _write_html(report_dir, m, baselines, cm, fig_b64, card_b64):
    def f4(v):
        return "—" if not np.isfinite(v) else f"{v:.4f}"

    rows = "".join(
        f"<tr><td>{CLS_TR[i]}</td>" + "".join(
            f"<td class='{'hit' if i == j else ''}'>{cm[i, j]}</td>"
            for j in range(3)) + "</tr>" for i in range(3))

    brows = "".join(
        f"<tr><td>{k.replace('_', ' ')}</td><td>{f4(v['precision'])}</td>"
        f"<td>{f4(v['recall'])}</td><td>{f4(v['f1'])}</td><td>{f4(v['iou'])}</td></tr>"
        for k, v in baselines.items())

    wins = m["beats_baseline"]
    verdict_cls = "good" if wins else "bad"
    verdict = ("Model, en iyi temel çizgiyi geçiyor."
               if wins else
               f"Model, <b>{m['best_baseline'].replace('_', ' ')}</b> temel "
               f"çizgisini geçemiyor. Bu, ölçüldüğü gibi raporlanır.")

    acc_note = ""
    if np.isfinite(m["growth_accuracy"]) and np.isfinite(m["majority_class_share"]) \
            and m["growth_accuracy"] < m["majority_class_share"]:
        acc_note = (f"<p class='bad'><b>Dikkat:</b> yama doğruluğu "
                    f"{f4(m['growth_accuracy'])}, çoğunluk sınıfı payının "
                    f"({f4(m['majority_class_share'])}) <b>altındadır</b>. "
                    f"Her zaman &quot;Sönüyor&quot; demek daha yüksek skor verirdi.</p>")

    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>Yangın Büyüme Modeli — Rapor</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:940px;
      margin:32px auto;padding:0 20px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #d43d2f;padding-bottom:8px}}
 h2{{margin-top:34px;color:#b5311f}}
 .kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
 .kpi div{{flex:1;min-width:150px;background:#faf5f4;border:1px solid #eee;
          border-radius:12px;padding:16px;text-align:center}}
 .kpi b{{display:block;font-size:28px;color:#d43d2f}}
 table{{border-collapse:collapse;margin:12px 0;width:100%}}
 td,th{{border:1px solid #ddd;padding:8px 14px;text-align:center}}
 th{{background:#f4f4f4}} td.hit{{background:#d6f0d8;font-weight:bold}}
 img{{max-width:100%;border:1px solid #eee;border-radius:8px;margin:10px 0}}
 .muted{{color:#777;font-size:14px}}
 .good{{color:#2e7d32}} .bad{{color:#c62828}}
 .banner{{padding:14px 18px;border-radius:10px;margin:18px 0;font-size:16px}}
 .banner.good{{background:#e8f5e9;border:1px solid #a5d6a7}}
 .banner.bad{{background:#ffebee;border:1px solid #ef9a9a}}
</style></head><body>
<h1>🔥 Yangın Büyüme (Yayılım) Tahmini — Değerlendirme Raporu</h1>
<p class="muted">Sürüm <b>{m['version']}</b> · bölme <b>{m['split']}</b> ·
kare sayısı <b>{m['patches']}</b> · eşik <b>{m['threshold']:.4f}</b>
(doğrulama bölmesinde kalibre edildi).<br>
Bu rapor <b>yalnızca ayrılmış test bölmesini</b> okur; eğitim günleri dâhil değildir.</p>

<div class="banner {verdict_cls}">{verdict}</div>

<div class="kpi">
 <div><b>{f4(m['ap'])}</b>AUC-PR<br><span class="muted">rastgele taban {m['prevalence']:.5f}</span></div>
 <div><b>{f4(m['iou'])}</b>IoU<br><span class="muted">alan örtüşmesi</span></div>
 <div><b>{f4(m['f1'])}</b>F1</div>
 <div><b>{f4(m['roc'])}</b>ROC-AUC</div>
</div>

<h2>1. Özet</h2>
<img src="data:image/png;base64,{card_b64}" alt="skor kartı">

<h2>2. Model ve temel çizgiler</h2>
<p>Bir modelin skoru, ancak aşması gereken basit yöntemlerle birlikte anlamlıdır.
Aşağıdaki temel çizgiler <b>tam olarak aynı piksellerde</b> ölçülmüştür.</p>
<table>
 <tr><th>Yöntem</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr>
 <tr><td><b>MODEL</b></td><td><b>{f4(m['precision'])}</b></td>
     <td><b>{f4(m['recall'])}</b></td><td><b>{f4(m['f1'])}</b></td>
     <td><b>{f4(m['iou'])}</b></td></tr>
 {brows}
</table>

<h2>3. Detaylı performans</h2>
<img src="data:image/png;base64,{fig_b64}" alt="figürler">
<p>Yangın pikselleri çok seyrektir (pozitif oran
<b>{m['prevalence']:.4%}</b>), bu yüzden düz doğruluk yanıltıcıdır ve
<b>AUC-PR</b>, <b>IoU</b>, <b>F1</b> raporlanır. ROC-AUC'nin yüksek çıkması
doğru negatiflerin ezici çokluğundandır ve tek başına yanıltıcıdır.</p>

<h2>4. Yama düzeyi sınıflandırma</h2>
<p>Doğruluk <b>{f4(m['growth_accuracy'])}</b>, çoğunluk sınıfı payı
<b>{f4(m['majority_class_share'])}</b>.</p>
{acc_note}
<table>
 <tr><th>gerçek ↓ / tahmin →</th><th>Sönüyor</th><th>Sabit</th><th>Büyüyor</th></tr>
 {rows}
</table>

<h2>5. Referans — v1 arşivi</h2>
<p class="muted">Karşılaştırma için, eski v1 arşivinde ölçülen değerler:
model IoU {V1_REFERENCE['model']['iou']}, F1 {V1_REFERENCE['model']['f1']};
kalıcılık temel çizgisi IoU {V1_REFERENCE['persistence']['iou']},
F1 {V1_REFERENCE['persistence']['f1']}. O değerlendirme örneklem içiydi
(tüm parçalar okunuyordu), dolayısıyla iyimserdi.</p>

<p>Harita: <a href="spread_map.html">spread_map.html</a></p>
<p class="muted">Otomatik üretildi — IGNIS yangın büyüme pipeline'ı.</p>
</body></html>"""
    (report_dir / "spread_report.html").write_text(html, encoding="utf-8")


def _make_map(report_dir, lons, lats, gc_true, gc_pred):
    try:
        import folium
    except ImportError as e:
        return f"folium yok ({e}) — pip install folium"
    lons, lats = np.asarray(lons), np.asarray(lats)
    if not (np.isfinite(lons).any() and np.isfinite(lats).any()):
        return "atlandı: veride konum (lon/lat) yok"

    fmap = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="CartoDB positron")
    for i in range(min(len(lons), 800)):
        if not (np.isfinite(lons[i]) and np.isfinite(lats[i])):
            continue
        p, t = int(gc_pred[i]), int(gc_true[i])
        folium.CircleMarker(
            location=[lats[i], lons[i]], radius=4, color=CLS_COLOR[p],
            fill=True, fill_color=CLS_COLOR[p], fill_opacity=0.8,
            popup=f"Tahmin: {CLS_TR[p]} | Gerçek: {CLS_TR[t]}",
        ).add_to(fmap)

    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;"
              "background:white;padding:10px 14px;border:1px solid #ccc;"
              "border-radius:8px;font:13px sans-serif'><b>Tahmin</b><br>"
              "<span style='color:#d43d2f'>●</span> Büyüyor<br>"
              "<span style='color:#e8a13a'>●</span> Sabit<br>"
              "<span style='color:#2e9e4f'>●</span> Sönüyor</div>")
    fmap.get_root().html.add_child(folium.Element(legend))
    out = report_dir / "spread_map.html"
    fmap.save(str(out))
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Evaluate on the held-out split")
    ap.add_argument("--version", default=SPREAD_VERSION, choices=["v1", "v2", "v3"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="override; by default calibrated on validation")
    args = ap.parse_args()
    evaluate(version=args.version, split=args.split, threshold=args.threshold)


if __name__ == "__main__":
    main()
