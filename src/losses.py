"""
Losses — masked, imbalance-aware
Kayıp fonksiyonları — maskeli, dengesizliğe duyarlı
==================================================

Every loss here is MASKED with the `valid` band. Pixels that were never observed
(outside Türkiye, under cloud, outside a source product's swath) must contribute
no gradient at all. Letting them in trains the network to reproduce the shape of
the study-area boundary, which it can do perfectly and which is worthless.

Buradaki her kayıp `valid` bandıyla MASKELENİR. Hiç gözlenmemiş pikseller
gradyana girmemelidir; girerlerse ağ, çalışma alanının sınır şeklini ezberler.

Two families / iki aile
-----------------------
`bce_dice` (default) — 0.5 * BCE(pos_weight) + 0.5 * SoftDice
    BCE is per-pixel and gives a dense, well-behaved gradient everywhere, but
    with 0.2686 % positives it is dominated by the negative class, so the
    positive term is up-weighted by `pos_weight`.
    Dice is a REGION overlap measure: it ignores true negatives entirely, so
    prevalence cannot drown it. On its own, though, it is unstable early in
    training when predictions are near zero and the ratio is 0/0.
    The two failure modes are complementary, which is why they are summed.

    BCE piksel başınadır ve her yerde iyi huylu gradyan verir ama negatif sınıf
    onu boğar. Dice ALAN örtüşmesidir; doğru negatifleri hiç saymaz, dolayısıyla
    sınıf oranından etkilenmez, ama eğitim başında kararsızdır. Toplanırlar.

`focal_tversky` (alternative) — Tversky with alpha=0.3, beta=0.7, focal gamma
    Tversky generalises Dice by weighting false positives (alpha) and false
    negatives (beta) separately. beta > alpha punishes MISSED fire harder than
    false alarms, which matches the operational asymmetry: failing to warn about
    a pixel that burns is worse than warning about one that does not.
    Tversky, yanlış pozitif ve yanlış negatifi ayrı ağırlıklandırır. beta > alpha,
    KAÇIRILAN yangını yanlış alarmdan daha ağır cezalandırır.

All losses take LOGITS, not probabilities.
Tüm kayıplar LOGIT alır, olasılık değil.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    SPREAD_BCE_DICE_WEIGHT,
    SPREAD_DICE_SMOOTH,
    SPREAD_POS_WEIGHT,
    SPREAD_TVERSKY_ALPHA,
    SPREAD_TVERSKY_BETA,
    SPREAD_TVERSKY_GAMMA,
)


def _prepare(valid, target):
    """Broadcast the valid mask to the target's shape. / Maskeyi yayınla."""
    if valid is None:
        return torch.ones_like(target)
    if valid.shape != target.shape:
        valid = valid.expand_as(target)
    return valid


class MaskedBCE(nn.Module):
    """Positively weighted BCE, masked by `valid`.
    `valid` ile maskelenmiş, pozitif ağırlıklı BCE."""

    def __init__(self, pos_weight=SPREAD_POS_WEIGHT):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits, target, valid=None):
        valid = _prepare(valid, target)
        pw = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        per_pixel = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pw, reduction="none")
        denom = valid.sum().clamp_min(1.0)
        return (per_pixel * valid).sum() / denom


class SoftDice(nn.Module):
    """Soft Dice loss, masked by `valid`. / `valid` ile maskelenmiş Soft Dice.

    Computed over the whole batch rather than per image. Per-image Dice is
    undefined for a patch whose target is entirely empty — and with the strict
    t+1 target 58.9 % of patches were exactly that. Batch-level pooling keeps
    the denominator away from zero.
    Toplu (batch) hesaplanır: hedefi tamamen boş bir yamada görüntü başına Dice
    tanımsızdır ve yamaların büyük kısmı böyleydi.
    """

    def __init__(self, smooth=SPREAD_DICE_SMOOTH):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, target, valid=None):
        valid = _prepare(valid, target)
        probs = torch.sigmoid(logits) * valid
        tgt = target * valid
        inter = (probs * tgt).sum()
        total = probs.sum() + tgt.sum()
        dice = (2.0 * inter + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class BCEDice(nn.Module):
    """w * BCE(pos_weight) + (1-w) * SoftDice. The default.
    Varsayılan bileşik kayıp."""

    def __init__(self, pos_weight=SPREAD_POS_WEIGHT, weight=SPREAD_BCE_DICE_WEIGHT,
                 smooth=SPREAD_DICE_SMOOTH):
        super().__init__()
        self.bce = MaskedBCE(pos_weight)
        self.dice = SoftDice(smooth)
        self.w = float(weight)

    def forward(self, logits, target, valid=None):
        return self.w * self.bce(logits, target, valid) \
            + (1.0 - self.w) * self.dice(logits, target, valid)


class FocalTversky(nn.Module):
    """Focal Tversky loss, masked by `valid`.
    `valid` ile maskelenmiş Focal Tversky kaybı.

        TI = TP / (TP + alpha*FP + beta*FN)
        loss = (1 - TI) ** gamma

    alpha=0.3, beta=0.7 -> a missed fire pixel costs 2.33x a false alarm.
    gamma < 1 flattens the curve so easy patches stop dominating late training.
    alpha=0.3, beta=0.7 -> kaçırılan yangın pikseli, yanlış alarmın 2.33 katı.
    """

    def __init__(self, alpha=SPREAD_TVERSKY_ALPHA, beta=SPREAD_TVERSKY_BETA,
                 gamma=SPREAD_TVERSKY_GAMMA, smooth=SPREAD_DICE_SMOOTH):
        super().__init__()
        self.alpha, self.beta = float(alpha), float(beta)
        self.gamma, self.smooth = float(gamma), float(smooth)

    def forward(self, logits, target, valid=None):
        valid = _prepare(valid, target)
        probs = torch.sigmoid(logits) * valid
        tgt = target * valid
        tp = (probs * tgt).sum()
        fp = (probs * (1.0 - tgt)).sum()
        fn = ((1.0 - probs) * tgt).sum()
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - ti) ** self.gamma


def build_loss(name="bce_dice", **kwargs):
    """Loss factory. / Kayıp fabrikası."""
    name = (name or "bce_dice").lower()
    if name in ("bce_dice", "bcedice", "default"):
        return BCEDice(**kwargs)
    if name in ("focal_tversky", "tversky", "focaltversky"):
        return FocalTversky(**kwargs)
    if name in ("bce", "weighted_bce"):
        return MaskedBCE(**kwargs)
    if name in ("dice", "soft_dice"):
        return SoftDice(**kwargs)
    raise ValueError(f"unknown loss {name!r} / bilinmeyen kayıp")


if __name__ == "__main__":
    torch.manual_seed(0)
    b, h, w = 4, 32, 32
    logits = torch.randn(b, 1, h, w)
    target = (torch.rand(b, 1, h, w) < 0.0027).float()      # ~0.27 % positive
    valid = (torch.rand(b, 1, h, w) > 0.05).float()

    print(f"positives: {target.sum():.0f} / {target.numel()} "
          f"({100 * target.mean():.3f}%)")
    print(f"valid    : {100 * valid.mean():.1f}%")
    print("-" * 44)
    for nm in ("bce", "dice", "bce_dice", "focal_tversky"):
        fn = build_loss(nm)
        print(f"  {nm:<16}{fn(logits, target, valid).item():.6f}")

    # Masked-out pixels must not move the loss.
    # Maskelenen pikseller kaybı değiştirmemeli.
    fn = build_loss("bce_dice")
    a = fn(logits, target, valid)
    poisoned = logits.clone()
    poisoned[valid < 0.5] = 1e3
    b_ = fn(poisoned, target, valid)
    print("-" * 44)
    print(f"  mask check: {a.item():.6f} vs {b_.item():.6f} "
          f"-> {'OK' if torch.allclose(a, b_, atol=1e-5) else 'LEAK'}")
