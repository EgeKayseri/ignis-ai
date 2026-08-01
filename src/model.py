"""
U-Net — next-day fire spread segmentation
U-Net — ertesi gün yangın yayılımı bölütlemesi
==============================================

PyTorch port of the TensorFlow model. The architecture is reproduced LAYER FOR
LAYER so that Section 2.6 of the IAC manuscript remains valid without amendment.
Do not "improve" it here — the numbers in the paper describe this network.

TensorFlow modelinin PyTorch portu. Mimari BİREBİR korunmuştur; makalenin
Bölüm 2.6'sı değişmeden geçerli kalsın diye. Burada "iyileştirme" yapmayın.

Correspondence with the Keras implementation / Keras karşılığı
--------------------------------------------------------------
    _conv_block(x, f)  ->  Conv2d(f,3,pad=1) + ReLU + BatchNorm2d
                           Conv2d(f,3,pad=1) + ReLU + BatchNorm2d
                           Dropout2d
    encoder            ->  conv_block(32), pool -> conv_block(64), pool
                           -> conv_block(128), pool
    bottleneck         ->  conv_block(256)
    decoder            ->  ConvTranspose2d(f,2,stride=2), concat skip,
                           conv_block(f)   for f in 128, 64, 32
    head               ->  Conv2d(1, kernel 1)

Note on ordering: Keras used Conv -> ReLU (via activation=) -> BatchNorm, which
is unusual (BN normally precedes the activation) but is what the trained model
did. The port keeps that order deliberately.
Keras'ta sıra Conv -> ReLU -> BatchNorm idi; alışılmadık olsa da eğitilen model
buydu, port bu sırayı bilerek koruyor.

The head emits LOGITS, not probabilities. The sigmoid lives in the loss
(`BCEWithLogits`) for numerical stability, and `predict_proba()` applies it for
inference. This differs from the Keras version only in where the sigmoid sits,
not in what the network computes.
Baş katman LOGIT üretir; sigmoid kayıp fonksiyonunun içindedir (sayısal
kararlılık). Ağın hesapladığı şey değişmez.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from config import SPREAD_MODEL_CONFIG  # noqa: E402


class ConvBlock(nn.Module):
    """Conv-ReLU-BN twice, then dropout. / İki kez Conv-ReLU-BN, sonra dropout."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        return self.drop(x)


class UNet(nn.Module):
    """Parametric U-Net. Defaults come from config.SPREAD_MODEL_CONFIG.
    Parametrik U-Net; varsayılanlar config'ten gelir.

    With the defaults (in_channels=21, base=32, depth=3) on a 32x32 input:

        input      32x32x21
        enc0       32x32x32   -> skip     pool -> 16x16
        enc1       16x16x64   -> skip     pool -> 8x8
        enc2        8x8x128   -> skip     pool -> 4x4
        bottleneck  4x4x256                        <- 46 % of parameters
        dec2        8x8x128   (up + skip)
        dec1       16x16x64   (up + skip)
        dec0       32x32x32   (up + skip)
        head       32x32x1    logits

    Skip connections carry the high-resolution fire front from encoder to
    decoder. Without them the 4x4 bottleneck cannot restore a sharp perimeter:
    the spatial detail simply is not present at that depth.
    Atlama bağlantıları yüksek çözünürlüklü yangın cephesini taşır; onlarsız
    4x4 darboğaz keskin bir sınır geri getiremez.
    """

    def __init__(self, in_channels, base_filters=None, depth=None, dropout=None,
                 out_channels=1):
        super().__init__()
        cfg = SPREAD_MODEL_CONFIG
        base = base_filters or cfg["base_filters"]
        depth = depth or cfg["depth"]
        dropout = cfg["dropout"] if dropout is None else dropout

        self.in_channels = in_channels
        self.base_filters = base
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for d in range(depth):
            f = base * (2 ** d)
            self.encoders.append(ConvBlock(ch, f, dropout))
            ch = f
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        bf = base * (2 ** depth)
        self.bottleneck = ConvBlock(ch, bf, dropout)

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = bf
        for d in reversed(range(depth)):
            f = base * (2 ** d)
            self.ups.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f, dropout))   # f from up + f from skip
            ch = f

        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)          # logits / logit

    @torch.no_grad()
    def predict_proba(self, x):
        """Sigmoid-activated probabilities. / Sigmoid uygulanmış olasılıklar."""
        return torch.sigmoid(self(x))


def build_unet(in_channels, **kwargs):
    """Factory mirroring the old build_unet(). / Eski build_unet() karşılığı."""
    return UNet(in_channels, **kwargs)


def count_parameters(model):
    """(trainable, total) parameter counts. / Parametre sayıları."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def parameter_report(model, input_size=None):
    """Per-stage parameter breakdown. / Aşama başına parametre dökümü.

    Printed because the manuscript quotes both the total and the share held by
    the bottleneck, and those figures should be reproducible from the code.
    """
    groups = {}
    for name, p in model.named_parameters():
        stage = name.split(".")[0]
        if stage in ("encoders", "decoders", "ups"):
            stage = f"{stage}[{name.split('.')[1]}]"
        groups[stage] = groups.get(stage, 0) + p.numel()
    total = sum(groups.values())
    lines = [f"{'stage':<16}{'params':>12}{'share':>9}", "-" * 37]
    for k, v in groups.items():
        lines.append(f"{k:<16}{v:>12,}{100.0 * v / total:>8.1f}%")
    lines.append("-" * 37)
    lines.append(f"{'TOTAL':<16}{total:>12,}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys as _sys

    from config import spread_bands
    from features import n_output_channels

    version = _sys.argv[1] if len(_sys.argv) > 1 else "v2"
    inp, _, _ = spread_bands(version)
    # dataset.py also feeds `valid` as an input channel
    ch = n_output_channels(list(inp) + ["valid"])

    model = UNet(in_channels=ch)
    trainable, total = count_parameters(model)

    print(f"IGNIS U-Net — {version}")
    print("=" * 40)
    print(f"  input channels : {ch}")
    print(f"  base filters   : {model.base_filters}")
    print(f"  depth          : {model.depth}")
    print(f"  parameters     : {total:,}")
    print("=" * 40)
    print(parameter_report(model))

    x = torch.zeros(2, ch, 32, 32)
    with torch.no_grad():
        y = model(x)
    print(f"\n  forward: {tuple(x.shape)} -> {tuple(y.shape)}")
