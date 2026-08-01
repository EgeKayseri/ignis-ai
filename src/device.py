"""
Device selection and mixed precision — ROCm / RDNA 4
Cihaz seçimi ve karışık hassasiyet
====================================================

Selects the AMD GPU through PyTorch's HIP backend and configures bfloat16
autocast. Two decisions here are load-bearing and are documented rather than
buried, because both are easy to get wrong on this hardware.

AMD GPU'yu PyTorch'un HIP arka ucu üzerinden seçer ve bfloat16 autocast ayarlar.
Buradaki iki karar kritiktir ve gömülü bırakılmak yerine belgelenmiştir.

1) DO NOT set HSA_OVERRIDE_GFX_VERSION.
   gfx1201 (Navi 48, RX 9070 XT, RDNA 4) is NATIVELY supported from ROCm 7.2.
   The override exists so older or unsupported cards can borrow another
   architecture's kernel library. Setting it here would make ROCm load kernels
   compiled for a different ISA — which either crashes or, worse, silently
   produces wrong numerics.
   gfx1201 ROCm 7.2'de YEREL olarak desteklenir. Bu değişkeni ayarlamak, ROCm'in
   başka bir mimari için derlenmiş çekirdekleri yüklemesine yol açar.

2) Use bfloat16, NOT float16.
   bfloat16 keeps float32's 8 exponent bits and gives up mantissa precision
   instead. Same dynamic range as float32, so it needs no loss scaling. float16
   has only 5 exponent bits and underflows to zero on the very small gradients a
   0.27 %-positive segmentation loss produces — the gradient silently vanishes
   and the network stops learning the rare class, which is the only class we
   care about.
   bfloat16, float32'nin 8 üs bitini korur; dinamik aralık aynıdır, loss scaling
   gerekmez. float16'nın 5 üs biti vardır ve bu kadar dengesiz bir kayıpta
   üretilen çok küçük gradyanlarda sıfıra düşer.
"""

from __future__ import annotations

import os

import torch


# gfx1201 is natively supported; the override must not be inherited from the
# environment either. Warn loudly rather than silently honouring it.
_GFX_OVERRIDE = "HSA_OVERRIDE_GFX_VERSION"


def _check_no_gfx_override(verbose: bool = True) -> None:
    """Warn if HSA_OVERRIDE_GFX_VERSION is set. / Ayarlıysa uyar."""
    val = os.environ.get(_GFX_OVERRIDE)
    if val and verbose:
        print(f"  !! {_GFX_OVERRIDE}={val} is set.")
        print(f"     gfx1201 is natively supported from ROCm 7.2; this override")
        print(f"     makes ROCm load kernels for a different ISA. Unset it:")
        print(f"       unset {_GFX_OVERRIDE}")


def get_device(prefer_gpu: bool = True, verbose: bool = True) -> torch.device:
    """Return the training device. / Eğitim cihazını döndür.

    Falls back to CPU with an explicit message rather than failing, so the
    pipeline stays runnable on a machine without ROCm.
    ROCm olmayan makinede de çalışabilsin diye sessizce değil, açıkça CPU'ya düşer.
    """
    _check_no_gfx_override(verbose)

    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        if verbose:
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            hip = getattr(torch.version, "hip", None)
            print(f"  Device / cihaz : {name}")
            print(f"  Backend        : {'ROCm/HIP ' + hip if hip else 'CUDA'}")
            print(f"  Compute units  : {props.multi_processor_count}")
            print(f"  VRAM           : {props.total_memory / 1024**3:.1f} GB")
            print(f"  torch          : {torch.__version__}")
        return device

    if verbose:
        if prefer_gpu:
            print("  !! No GPU visible to PyTorch — falling back to CPU.")
            print("     GPU görünmüyor, CPU'ya düşülüyor. Kontrol / check:")
            print("       rocminfo | grep gfx        (expect gfx1201)")
            print("       ls -l /dev/kfd             (expect mode 666)")
            print("       pacman -Q python-pytorch-rocm rocm-hip-sdk")
        else:
            print("  Device / cihaz : CPU (requested)")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on GPU, float32 on CPU. / GPU'da bfloat16, CPU'da float32.

    Never float16 — see the module docstring.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def autocast(device: torch.device, enabled: bool = True):
    """Context manager for mixed-precision forward passes.
    Karışık hassasiyetli ileri geçiş için bağlam yöneticisi.

    Usage / kullanım:
        with autocast(device):
            logits = model(x)
            loss = criterion(logits, y, valid)

    No GradScaler is needed. GradScaler exists to stop float16 gradients from
    underflowing; bfloat16 has float32's exponent range and does not underflow,
    so scaling would add complexity for no benefit.
    GradScaler gerekmez: float16 gradyanlarının sıfıra düşmesini engellemek için
    vardır; bfloat16'da bu sorun yoktur.
    """
    dtype = autocast_dtype(device)
    use = enabled and device.type == "cuda" and dtype is torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=use)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every RNG the pipeline touches. / Tüm rastgelelik kaynaklarını sabitle."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Slower, and some ROCm kernels have no deterministic implementation.
        # Daha yavaş; bazı ROCm çekirdeklerinin deterministik karşılığı yok.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def describe() -> dict:
    """Environment summary for the run log. / Çalışma kaydı için ortam özeti."""
    info = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "gpu_available": torch.cuda.is_available(),
        "gfx_override": os.environ.get(_GFX_OVERRIDE),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


if __name__ == "__main__":
    print("IGNIS — device check / cihaz kontrolü")
    print("=" * 52)
    dev = get_device()
    print(f"  autocast dtype : {autocast_dtype(dev)}")
    print("=" * 52)
    for k, v in describe().items():
        print(f"  {k:16s}: {v}")
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(0)
        expected = "Radeon RX 9070 XT"
        print()
        if expected in name:
            print(f"  OK — {expected} detected.")
        else:
            print(f"  NOTE: expected {expected!r}, got {name!r}.")
