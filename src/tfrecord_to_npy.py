"""
TFRecord -> memory-mapped .npy cache
TFRecord -> bellek eşlemeli .npy önbelleği
==========================================

Converts the gzip TFRecord archive ONCE into memory-mapped float32 arrays under
`~/ignis-cache/` (local ext4). Training then reads only from there.

Arşivi BİR KEZ `~/ignis-cache/` altındaki bellek eşlemeli float32 dizilere çevirir.

Why this exists / Neden var
---------------------------
The repository lives on `/mnt/windows`, an NTFS `fuseblk` mount. Decompressing
every gzip shard on every epoch, across FUSE, dominates training time — the GPU
sits idle while the CPU inflates the same bytes again. Converting once removes
that: afterwards an epoch is a sequence of page-cache reads from a flat array.

Depo NTFS fuseblk üzerinde. Her epoch'ta her gzip parçasını yeniden açmak eğitim
süresini domine eder; GPU, CPU aynı baytları tekrar açarken boş bekler.

No TensorFlow dependency / TensorFlow bağımlılığı yok
-----------------------------------------------------
There is no TensorFlow wheel for Python 3.14, so the TFRecord container and the
`tf.train.Example` protobuf inside it are decoded here in pure Python. Both
formats are small and stable:

  TFRecord framing:
      uint64  length              (little endian)
      uint32  masked_crc32c(length bytes)
      bytes   data[length]
      uint32  masked_crc32c(data)

  Example protobuf:
      Example   { Features features = 1; }
      Features  { map<string, Feature> feature = 1; }
      Feature   { BytesList bytes_list = 1 | FloatList float_list = 2
                                           | Int64List int64_list = 3; }
      FloatList { repeated float value = 1 [packed]; }

Usage / kullanım
----------------
    python src/tfrecord_to_npy.py                 # convert the active version
    python src/tfrecord_to_npy.py --verify        # convert + integrity report
    python src/tfrecord_to_npy.py --version v3
    python src/tfrecord_to_npy.py --force         # rebuild even if cache exists
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import struct
import sys
import time
from datetime import date as _date
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    KNOWN_OUTAGES,
    PATCH_SIZE,
    SPREAD_CACHE_DIR,
    SPREAD_VERSION,
    spread_bands,
    spread_data_dir,
)

_N_PIX = PATCH_SIZE * PATCH_SIZE


# ======================================================================
# Schema detection
# ======================================================================
def detect_schema(feature_keys):
    """Which schema version a record belongs to, from its FEATURE NAMES.

    Bir kaydın hangi şema sürümüne ait olduğunu ALAN ADLARINDAN belirle.

    The band contract is positional — the loader rebuilds the channel axis from
    band ORDER alone — so a record from the wrong schema is silently misread
    rather than rejected. But the TFRecord itself stores each band under its own
    NAME, and those names are unambiguous. Checking them turns "never mix
    versions in one directory" from a documented rule into an enforced one.

    Bant sözleşmesi konumsaldır; yanlış şemadan bir kayıt reddedilmez, SESSİZCE
    yanlış yorumlanır. Ama TFRecord her bandı kendi ADIYLA saklar. Adları
    kontrol etmek, "sürümleri karıştırma" kuralını belgeden zorunluluğa çevirir.
    """
    keys = set(feature_keys)
    best = None
    for ver in ("v5", "v4", "v3", "v2", "v1"):
        _, _, bands = spread_bands(ver)
        if set(bands).issubset(keys):
            # Prefer the most specific schema whose bands are all present.
            if best is None or len(bands) > best[1]:
                best = (ver, len(bands))
    return best[0] if best else None


# ======================================================================
# CRC32C (Castagnoli) — TFRecord integrity
# ======================================================================
_CRC_POLY = 0x82F63B78
_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ (_CRC_POLY if _c & 1 else 0)
    _CRC_TABLE.append(_c)


def crc32c(data: bytes, crc: int = 0) -> int:
    """Castagnoli CRC-32C — the checksum TFRecord uses.
    TFRecord'un kullandığı sağlama toplamı."""
    crc ^= 0xFFFFFFFF
    table = _CRC_TABLE
    for b in data:
        crc = (crc >> 8) ^ table[(crc ^ b) & 0xFF]
    return crc ^ 0xFFFFFFFF


def unmask_crc(masked: int) -> int:
    """Undo the TFRecord CRC mask. / TFRecord CRC maskesini geri al."""
    rot = (masked - 0xA282EAD8) & 0xFFFFFFFF
    return ((rot >> 17) | (rot << 15)) & 0xFFFFFFFF


# ======================================================================
# Minimal protobuf reader
# ======================================================================
def _read_varint(buf: memoryview, pos: int):
    """Read a base-128 varint. / Varint oku."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long / varint çok uzun")


def _iter_fields(buf: memoryview):
    """Yield (field_number, wire_type, payload) for one protobuf message.
    Bir protobuf mesajının alanlarını üret."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 0x07
        if wire == 0:                      # varint
            val, pos = _read_varint(buf, pos)
            yield field, wire, val
        elif wire == 1:                    # 64-bit
            yield field, wire, buf[pos:pos + 8]
            pos += 8
        elif wire == 2:                    # length-delimited
            ln, pos = _read_varint(buf, pos)
            yield field, wire, buf[pos:pos + ln]
            pos += ln
        elif wire == 5:                    # 32-bit
            yield field, wire, buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire} / desteklenmeyen tip")


def _parse_feature(buf: memoryview):
    """Decode one tf.train.Feature into a numpy array or list of bytes.
    Bir Feature'ı numpy dizisine ya da bayt listesine çöz."""
    for field, wire, payload in _iter_fields(buf):
        if field == 2:                     # float_list
            for f2, w2, p2 in _iter_fields(payload):
                if f2 == 1 and w2 == 2:    # packed
                    return np.frombuffer(bytes(p2), dtype="<f4")
                if f2 == 1 and w2 == 5:    # unpacked (rare)
                    return np.array([struct.unpack("<f", bytes(p2))[0]], dtype="<f4")
            return np.zeros(0, dtype="<f4")
        if field == 3:                     # int64_list
            vals = []
            for f2, w2, p2 in _iter_fields(payload):
                if f2 == 1 and w2 == 2:
                    mv, q = memoryview(p2), 0
                    while q < len(mv):
                        v, q = _read_varint(mv, q)
                        vals.append(v)
                elif f2 == 1 and w2 == 0:
                    vals.append(p2)
            return np.array(vals, dtype=np.int64)
        if field == 1:                     # bytes_list
            out = []
            for f2, w2, p2 in _iter_fields(payload):
                if f2 == 1 and w2 == 2:
                    out.append(bytes(p2))
            return out
    return np.zeros(0, dtype="<f4")


def parse_example(record: bytes) -> dict:
    """Decode a serialised tf.train.Example into {name: value}.
    Serileştirilmiş Example'ı sözlüğe çöz."""
    out = {}
    for field, wire, payload in _iter_fields(memoryview(record)):
        if field != 1 or wire != 2:
            continue
        for f2, w2, entry in _iter_fields(payload):      # Features.feature map
            if f2 != 1 or w2 != 2:
                continue
            key, val = None, None
            for f3, w3, p3 in _iter_fields(entry):       # MapEntry
                if f3 == 1 and w3 == 2:
                    key = bytes(p3).decode("utf-8", "replace")
                elif f3 == 2 and w3 == 2:
                    val = _parse_feature(p3)
            if key is not None:
                out[key] = val
    return out


def iter_tfrecords(path: Path, check_crc: bool = False):
    """Yield raw record bytes from a gzip TFRecord file.
    Gzip TFRecord dosyasından ham kayıtları üret."""
    with gzip.open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                return
            length = struct.unpack("<Q", header)[0]
            len_crc = struct.unpack("<I", fh.read(4))[0]
            if check_crc and crc32c(header) != unmask_crc(len_crc):
                raise ValueError(f"length CRC mismatch in {path.name}")
            data = fh.read(length)
            if len(data) < length:
                raise ValueError(f"truncated record in {path.name} / kayıt kesik")
            data_crc = struct.unpack("<I", fh.read(4))[0]
            if check_crc and crc32c(data) != unmask_crc(data_crc):
                raise ValueError(f"data CRC mismatch in {path.name}")
            yield data


# ======================================================================
# Outage filtering
# ======================================================================
def _parse_outages(spec):
    """[('YYYY-MM-DD','YYYY-MM-DD'), ...] -> set of excluded YYYYMMDD ints."""
    excluded = set()
    for start, end in spec:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
        for o in range(d0.toordinal(), d1.toordinal() + 1):
            excluded.add(int(_date.fromordinal(o).strftime("%Y%m%d")))
    return excluded


def date_from_filename(path: Path):
    """firespread_20220717.tfrecord.gz -> 20220717"""
    for part in path.name.replace(".", "_").split("_"):
        if len(part) == 8 and part.isdigit():
            return int(part)
    return None


# ======================================================================
# Conversion
# ======================================================================
def convert(version: str = None, force: bool = False, verify: bool = False,
            max_files: int = 0) -> Path:
    """Convert one archive version into the memmap cache.
    Bir arşiv sürümünü memmap önbelleğine çevir."""
    version = version or SPREAD_VERSION
    input_bands, aux_bands, all_bands = spread_bands(version)
    src_dir = spread_data_dir(version)
    out_dir = Path(SPREAD_CACHE_DIR) / version
    bin_path = out_dir / "patches.f32"
    manifest_path = out_dir / "manifest.json"

    files = sorted(glob.glob(str(Path(src_dir) / "*.tfrecord.gz")))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No TFRecord shards in {src_dir}\n"
            f"{src_dir} içinde TFRecord yok. Önce notebook ile veri üretip indirin.")

    if manifest_path.exists() and not force:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        if man.get("n_shards") == len(files) and man.get("bands") == all_bands:
            print(f"Cache already current / önbellek güncel: {out_dir}")
            print(f"  {man['n_patches']} patches, {len(all_bands)} bands")
            print("  Use --force to rebuild. / Yeniden kurmak için --force.")
            return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = _parse_outages(KNOWN_OUTAGES)
    n_bands = len(all_bands)

    print(f"IGNIS — TFRecord -> memmap  ({version})")
    print("=" * 64)
    print(f"  source / kaynak : {src_dir}")
    print(f"  cache  / hedef  : {out_dir}")
    print(f"  shards          : {len(files)}")
    print(f"  bands           : {n_bands}  ({', '.join(all_bands)})")
    print(f"  patch           : {PATCH_SIZE}x{PATCH_SIZE}")
    print("=" * 64)

    dates, lons, lats = [], [], []
    n_written = n_short = n_missing = n_skipped_outage = 0
    band_zero = np.zeros(n_bands, dtype=np.int64)
    band_nan = np.zeros(n_bands, dtype=np.int64)
    band_total = 0
    t0 = time.time()

    with open(bin_path, "wb") as out:
        for fi, fname in enumerate(files):
            fpath = Path(fname)
            fdate = date_from_filename(fpath)
            if fdate is not None and fdate in excluded:
                n_skipped_outage += 1
                continue
            try:
                records = list(iter_tfrecords(fpath, check_crc=verify))
            except (ValueError, OSError) as exc:
                print(f"  !! {fpath.name}: {exc}")
                continue

            # Verify this shard really is the schema we were asked for. A mixed
            # directory is the one failure mode the positional band contract
            # cannot survive, so refuse rather than produce corrupt training data.
            # Karışık bir dizin, konumsal bant sözleşmesinin atlatamayacağı tek
            # hata modudur; bozuk eğitim verisi üretmek yerine reddet.
            if records:
                found = detect_schema(parse_example(records[0]).keys())
                if found is None:
                    raise RuntimeError(
                        f"{fpath.name}: no known schema matches this record.\n"
                        f"Bu kayıt bilinen hiçbir şemayla eşleşmiyor.")
                if found != version:
                    raise RuntimeError(
                        f"SCHEMA MISMATCH / ŞEMA UYUŞMAZLIĞI\n"
                        f"  {fpath.name} is a {found} record, but --version {version} "
                        f"was requested.\n"
                        f"  {src_dir} contains mixed schemas, or the wrong "
                        f"--version was given.\n"
                        f"  Each schema needs its OWN directory: v2 -> data/spread/, "
                        f"v3 -> data/spread_v3/, v5 -> data/spread_v5/.\n"
                        f"  Her şema KENDİ dizininde olmalıdır.")

            for raw in records:
                feat = parse_example(raw)
                if any(b not in feat for b in all_bands):
                    n_missing += 1
                    continue
                if any(len(feat[b]) != _N_PIX for b in all_bands):
                    # GEE can emit a short array for masked/edge patches.
                    # GEE, maskeli/kenar yamalarda diziyi kısaltabilir.
                    n_short += 1
                    continue

                stack = np.empty((n_bands, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                for bi, b in enumerate(all_bands):
                    stack[bi] = np.asarray(feat[b], dtype=np.float32).reshape(
                        PATCH_SIZE, PATCH_SIZE)

                if verify:
                    band_zero += (stack == 0).sum(axis=(1, 2))
                    band_nan += (~np.isfinite(stack)).sum(axis=(1, 2))
                    band_total += _N_PIX

                out.write(stack.tobytes())
                n_written += 1

                dates.append(fdate if fdate is not None else 0)
                lon, lat = feat.get("lon"), feat.get("lat")
                lons.append(float(lon[0]) if lon is not None and len(lon) else np.nan)
                lats.append(float(lat[0]) if lat is not None and len(lat) else np.nan)

            if (fi + 1) % 50 == 0 or fi + 1 == len(files):
                print(f"  {fi + 1:4d}/{len(files)} shards  "
                      f"{n_written:7d} patches  {time.time() - t0:6.1f}s")

    if n_written == 0:
        raise RuntimeError(
            "No valid patches converted. / Geçerli yama çevrilemedi.\n"
            "A band-contract mismatch is the usual cause — check that the archive "
            "version matches SPREAD_VERSION in config.py.\n"
            "Genellikle bant sözleşmesi uyuşmazlığıdır.")

    np.savez(out_dir / "meta.npz",
             date=np.array(dates, dtype=np.int32),
             lon=np.array(lons, dtype=np.float64),
             lat=np.array(lats, dtype=np.float64))

    manifest = {
        "version": version,
        "bands": all_bands,
        "input_bands": input_bands,
        "aux_bands": aux_bands,
        "n_patches": n_written,
        "patch_size": PATCH_SIZE,
        "dtype": "float32",
        "shape": [n_written, n_bands, PATCH_SIZE, PATCH_SIZE],
        "n_shards": len(files),
        "created": datetime.now().isoformat(timespec="seconds"),
        "skipped": {"short": n_short, "missing_bands": n_missing,
                    "outage_shards": n_skipped_outage},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"  patches written / yazılan : {n_written}")
    print(f"  short records   / kısa    : {n_short}")
    print(f"  missing bands   / eksik   : {n_missing}")
    print(f"  outage shards   / kesinti : {n_skipped_outage}")
    print(f"  cache size                : {bin_path.stat().st_size / 1024**3:.2f} GB")
    print(f"  elapsed                   : {time.time() - t0:.1f}s")

    if verify:
        _report_integrity(all_bands, band_zero, band_nan, band_total, out_dir, n_written)
    return out_dir


def _report_integrity(bands, band_zero, band_nan, band_total, out_dir, n):
    """Per-band zero and NaN rates. / Bant başına sıfır ve NaN oranları.

    Root cause 2 in the v1 diagnosis was ~15 % fabricated zeros with an IDENTICAL
    zero rate across every environmental band, produced by clip + unmask(0).
    An identical non-trivial zero rate across unrelated bands is the signature of
    that bug, so it is checked for explicitly rather than left to be noticed.
    v1 teşhisindeki 2. kök neden, clip + unmask(0) kaynaklı ~%15 uydurma sıfırdı.
    """
    print("=" * 64)
    print("INTEGRITY REPORT / BÜTÜNLÜK RAPORU")
    print("=" * 64)
    print(f"  {'band':<16} {'zero %':>9} {'NaN %':>9}")
    print("  " + "-" * 36)
    rates = {}
    for i, b in enumerate(bands):
        z = 100.0 * band_zero[i] / band_total if band_total else float("nan")
        nn = 100.0 * band_nan[i] / band_total if band_total else float("nan")
        rates[b] = z
        print(f"  {b:<16} {z:>8.2f}% {nn:>8.2f}%")

    # Bands that are legitimately mostly-zero (masks, rainfall, class codes) are
    # excluded from the uniform-zero test.
    skip = {"fire", "fire_next", "fire_next2", "fire_prev1", "fire_prev2",
            "valid", "landcover", "precip", "precip_7d", "precip_30d"}
    env = [b for b in bands if b not in skip]
    if len(env) >= 2:
        vals = [rates[b] for b in env]
        spread = max(vals) - min(vals)
        print("  " + "-" * 36)
        if spread < 0.5 and min(vals) > 1.0:
            print(f"  !! Environmental bands share a {min(vals):.2f}% zero rate")
            print(f"     (spread {spread:.3f} pp). That is the clip+unmask(0)")
            print(f"     signature — those zeros may be fabricated, not observed.")
            print(f"     Çevresel bantların sıfır oranı aynı: uydurma sıfır işareti.")
        else:
            print(f"  Environmental zero rates differ by {spread:.2f} pp — no")
            print(f"  uniform-zero signature. / Tekdüze sıfır imzası yok.")

    if "valid" in bands:
        vi = bands.index("valid")
        obs = 100.0 * (1.0 - band_zero[vi] / band_total) if band_total else float("nan")
        print("  " + "-" * 36)
        print(f"  valid coverage / gözlenen piksel : {obs:.2f}%")

    (out_dir / "integrity.json").write_text(
        json.dumps({"n_patches": int(n),
                    "zero_pct": {b: float(rates[b]) for b in bands}}, indent=2),
        encoding="utf-8")


def load_cache(version: str = None):
    """Open the cache read-only. / Önbelleği salt-okunur aç.

    Returns (memmap[N, B, 65, 65], meta dict, manifest dict).
    """
    version = version or SPREAD_VERSION
    out_dir = Path(SPREAD_CACHE_DIR) / version
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No cache at {out_dir}\n"
            f"Run: python src/tfrecord_to_npy.py --version {version}")
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    arr = np.memmap(out_dir / "patches.f32", dtype=np.float32, mode="r",
                    shape=tuple(man["shape"]))
    meta = dict(np.load(out_dir / "meta.npz"))
    return arr, meta, man


def main():
    ap = argparse.ArgumentParser(description="TFRecord -> memmap .npy cache")
    ap.add_argument("--version", default=SPREAD_VERSION, choices=["v1", "v2", "v3", "v4", "v5"],
                    help="dataset schema version / veri seti şema sürümü")
    ap.add_argument("--verify", action="store_true",
                    help="check CRCs and print an integrity report / bütünlük raporu")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the cache looks current")
    ap.add_argument("--max-files", type=int, default=0,
                    help="convert only the first N shards (quick test)")
    args = ap.parse_args()
    convert(version=args.version, force=args.force, verify=args.verify,
            max_files=args.max_files)


if __name__ == "__main__":
    main()
