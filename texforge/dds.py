"""Lecture et écriture DDS (formats DX9 : DXT1/3/5 et non compressés) avec mipmaps.

Lecture : en-tête analysé ici, décodage des formats compressés par Pillow,
décodage des formats non compressés (masques RGB/luminance/alpha) fait ici.
Écriture : en-tête écrit ici ; chaque niveau de mipmap est encodé séparément
(Pillow pour DXT1/3/5, empaquetage par masques pour le non compressé).
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .i18n import tr

DDS_MAGIC = b"DDS "

# Drapeaux d'en-tête
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSD_DEPTH = 0x800000

DDPF_ALPHAPIXELS = 0x1
DDPF_ALPHA = 0x2
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40
DDPF_LUMINANCE = 0x20000

DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000
DDSCAPS2_CUBEMAP = 0x200
DDSCAPS2_VOLUME = 0x200000

# Formats DXGI (en-tête DX10) que l'on sait au moins lire
DXGI_NAMES = {
    28: "DX10_RGBA8", 29: "RGBA8_SRGB", 87: "DX10_BGRA8", 91: "BGRA8_SRGB", 88: "DX10_BGRX8",
    71: "BC1", 72: "BC1_SRGB", 74: "BC2", 75: "BC2_SRGB", 77: "BC3", 78: "BC3_SRGB",
    80: "BC4", 83: "BC5", 98: "BC7", 99: "BC7_SRGB",
}

# Écriture avec en-tête DX10 (garde le format DXGI exact, sRGB compris, de l'original)
# nom -> (code DXGI écrit, encodage des données)
DX10_LAYOUT = {
    "BC1": (71, "DXT1"), "BC1_SRGB": (72, "DXT1"), "BC2": (74, "DXT3"), "BC2_SRGB": (75, "DXT3"),
    "BC3": (77, "DXT5"), "BC3_SRGB": (78, "DXT5"),
    "DX10_RGBA8": (28, "RGBA8"), "RGBA8_SRGB": (29, "RGBA8"),
    "DX10_BGRA8": (87, "BGRA8"), "BGRA8_SRGB": (91, "BGRA8"), "DX10_BGRX8": (88, "BGRX8"),
    # BC7 : pas d'encodeur disponible -> non compressé, même espace colorimétrique
    "BC7": (28, "RGBA8"), "BC7_SRGB": (29, "RGBA8"),
}

# Masques des formats non compressés que l'on sait écrire
UNCOMPRESSED_LAYOUTS = {
    "BGRA8": (32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000, DDPF_RGB | DDPF_ALPHAPIXELS),
    "BGRX8": (32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000, DDPF_RGB),
    "RGBA8": (32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000, DDPF_RGB | DDPF_ALPHAPIXELS),
    "BGR8": (24, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000, DDPF_RGB),
    "R5G6B5": (16, 0xF800, 0x07E0, 0x001F, 0x0000, DDPF_RGB),
    "A1R5G5B5": (16, 0x7C00, 0x03E0, 0x001F, 0x8000, DDPF_RGB | DDPF_ALPHAPIXELS),
    "X1R5G5B5": (16, 0x7C00, 0x03E0, 0x001F, 0x0000, DDPF_RGB),
    "A4R4G4B4": (16, 0x0F00, 0x00F0, 0x000F, 0xF000, DDPF_RGB | DDPF_ALPHAPIXELS),
    "L8": (8, 0xFF, 0, 0, 0, DDPF_LUMINANCE),
    "A8L8": (16, 0x00FF, 0, 0, 0xFF00, DDPF_LUMINANCE | DDPF_ALPHAPIXELS),
    "A8": (8, 0, 0, 0, 0xFF, DDPF_ALPHA),
}

COMPRESSED_WRITABLE = {"DXT1", "DXT3", "DXT5"}
WRITABLE_FORMATS = COMPRESSED_WRITABLE | set(UNCOMPRESSED_LAYOUTS) | set(DX10_LAYOUT)


class DDSError(Exception):
    pass


@dataclass
class DDSInfo:
    width: int
    height: int
    mipmaps: int
    format: str              # ex. "DXT1", "BGRA8", "R5G6B5", "BC7", "UNKNOWN"
    cubemap: bool = False
    volume: bool = False
    dx10: bool = False
    masks: tuple = field(default_factory=tuple)

    @property
    def writable(self) -> bool:
        return self.format in WRITABLE_FORMATS

    @property
    def has_alpha_channel(self) -> bool:
        if self.format in ("DXT1", "DXT3", "DXT5", "BC1", "BC1_SRGB", "BC2", "BC2_SRGB",
                           "BC3", "BC3_SRGB", "BC7", "BC7_SRGB", "DX10_RGBA8", "RGBA8_SRGB",
                           "DX10_BGRA8", "BGRA8_SRGB"):
            return True
        layout = UNCOMPRESSED_LAYOUTS.get(self.format)
        return bool(layout and layout[4])


def _identify_uncompressed(flags: int, bits: int, r: int, g: int, b: int, a: int) -> str:
    """a doit être le masque alpha effectif (0 si l'alpha n'est pas déclaré)."""
    if flags & DDPF_LUMINANCE:
        if bits == 8 and r == 0xFF and not a:
            return "L8"
        if bits == 16 and r == 0x00FF and a == 0xFF00:
            return "A8L8"
        return "UNKNOWN"
    if flags & DDPF_ALPHA and not flags & DDPF_RGB:
        return "A8" if bits == 8 and a == 0xFF else "UNKNOWN"
    if flags & DDPF_RGB:
        for name, (lb, lr, lg, lbm, la, _f) in UNCOMPRESSED_LAYOUTS.items():
            if name in ("L8", "A8L8", "A8"):
                continue
            if bits == lb and (r, g, b) == (lr, lg, lbm) and a == la:
                return name
    return "UNKNOWN"


def read_info(data: bytes) -> DDSInfo:
    if len(data) < 128 or data[:4] != DDS_MAGIC:
        raise DDSError(tr("fichier DDS invalide (signature absente)"))
    (size, flags, height, width, _pitch, _depth, mips) = struct.unpack_from("<7I", data, 4)
    if size != 124:
        raise DDSError(tr("taille d'en-tête inattendue ({size})", size=size))
    pf_size, pf_flags, fourcc, bits, rm, gm, bm, am = struct.unpack_from("<2I4s5I", data, 76)
    caps, caps2 = struct.unpack_from("<2I", data, 108)
    # Masque alpha effectif : ignoré si le fichier ne déclare pas d'alpha (cas X8R8G8B8)
    am_eff = am if pf_flags & (DDPF_ALPHAPIXELS | DDPF_ALPHA) else 0
    # Certains outils omettent DDSD_MIPMAPCOUNT tout en remplissant le champ
    mip_levels = mips if (flags & DDSD_MIPMAPCOUNT or caps & DDSCAPS_MIPMAP) else 1
    info = DDSInfo(width=width, height=height, mipmaps=max(1, mip_levels),
                   format="UNKNOWN", cubemap=bool(caps2 & DDSCAPS2_CUBEMAP),
                   volume=bool(caps2 & DDSCAPS2_VOLUME), masks=(bits, rm, gm, bm, am_eff))
    if pf_flags & DDPF_FOURCC:
        cc = fourcc.decode("ascii", "replace")
        if cc == "DX10":
            info.dx10 = True
            if len(data) < 148:
                raise DDSError(tr("en-tête DX10 tronqué"))
            dxgi = struct.unpack_from("<I", data, 128)[0]
            info.format = DXGI_NAMES.get(dxgi, f"DXGI_{dxgi}")
        elif cc in ("DXT1", "DXT2", "DXT3", "DXT4", "DXT5"):
            info.format = {"DXT2": "DXT3", "DXT4": "DXT5"}.get(cc, cc)
        elif cc in ("ATI1", "BC4U"):
            info.format = "BC4"
        elif cc in ("ATI2", "BC5U"):
            info.format = "BC5"
        else:
            # Codes numériques D3DFMT dans le champ FourCC (ex. 113 = A16B16G16R16F)
            info.format = f"FOURCC_{struct.unpack('<I', fourcc)[0]}"
    else:
        info.format = _identify_uncompressed(pf_flags, bits, rm, gm, bm, am_eff)
    return info


def _mask_shift_bits(mask: int) -> tuple[int, int]:
    if mask == 0:
        return 0, 0
    shift = (mask & -mask).bit_length() - 1
    bits = bin(mask).count("1")
    return shift, bits


def _decode_uncompressed(data: bytes, info: DDSInfo) -> Image.Image:
    bits, rm, gm, bm, am = info.masks
    bpp = bits // 8
    if bpp not in (1, 2, 3, 4):
        raise DDSError(tr("profondeur non gérée ({bits} bits)", bits=bits))
    offset = 128
    count = info.width * info.height
    raw = np.frombuffer(data, dtype=np.uint8, count=count * bpp, offset=offset)
    raw = raw.reshape(-1, bpp).astype(np.uint32)
    px = np.zeros(count, dtype=np.uint32)
    for i in range(bpp):
        px |= raw[:, i] << (8 * i)

    def channel(mask: int) -> np.ndarray | None:
        if not mask:
            return None
        shift, nbits = _mask_shift_bits(mask)
        v = (px & mask) >> shift
        return np.round(v.astype(np.float32) * (255.0 / ((1 << nbits) - 1))).astype(np.uint8)

    out = np.zeros((count, 4), dtype=np.uint8)
    if info.format in ("L8", "A8L8"):
        lum = channel(rm)
        out[:, 0] = out[:, 1] = out[:, 2] = lum
        a = channel(am)
        out[:, 3] = 255 if a is None else a
    elif info.format == "A8":
        out[:, 0:3] = 255
        out[:, 3] = channel(am)
    else:
        for i, m in enumerate((rm, gm, bm)):
            c = channel(m)
            out[:, i] = 0 if c is None else c
        a = channel(am)
        out[:, 3] = 255 if a is None else a
    return Image.fromarray(out.reshape(info.height, info.width, 4), "RGBA")


# Variante sRGB -> variante linéaire de même encodage (les données sont identiques ;
# seul l'interprétation par le GPU change). Sert uniquement au décodage par Pillow.
_DX10_DECODE_ALIAS = {72: 71, 75: 74, 78: 77, 29: 28, 99: 98}


def load_dds(data: bytes) -> tuple[Image.Image, DDSInfo]:
    """Renvoie le niveau 0 en RGBA (valeurs brutes, sans conversion de gamma) et
    les informations de format."""
    info = read_info(data)
    if info.cubemap or info.volume:
        raise DDSError(tr("cubemap ou texture volumique : non traitée"))
    if info.format in UNCOMPRESSED_LAYOUTS and not info.dx10:
        return _decode_uncompressed(data, info), info
    if info.dx10:
        dxgi = struct.unpack_from("<I", data, 128)[0]
        if dxgi in (87, 88, 91):          # BGRA8 / BGRX8 : décodage direct
            n = info.width * info.height * 4
            raw = np.frombuffer(data, dtype=np.uint8, count=n, offset=148).reshape(info.height, info.width, 4)
            rgba = raw[..., [2, 1, 0, 3]].copy()
            if dxgi == 88:
                rgba[..., 3] = 255
            return Image.fromarray(rgba, "RGBA"), info
        if dxgi in _DX10_DECODE_ALIAS:
            patched = bytearray(data)
            struct.pack_into("<I", patched, 128, _DX10_DECODE_ALIAS[dxgi])
            data = bytes(patched)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # Pillow ne connaît pas ce format
        raise DDSError(tr("décodage impossible du format {format} : {exc}", format=info.format, exc=exc)) from exc
    return img.convert("RGBA"), info


def mip_count_full(width: int, height: int) -> int:
    return max(width, height).bit_length()


def _alpha_coverage(alpha: np.ndarray, ref: float = 0.5) -> float:
    return float((alpha >= ref * 255).mean())


def build_mip_chain(img: Image.Image, levels: int, preserve_coverage: bool) -> list[Image.Image]:
    """Mipmaps par moyenne 2x2. Pour les découpes (feuillage, grilles), l'alpha de
    chaque niveau est rééchelonné pour garder la même couverture que le niveau 0,
    sinon les objets « fondent » au loin."""
    chain = [img]
    target_cov = _alpha_coverage(np.asarray(img)[..., 3]) if preserve_coverage else None
    cur = img
    for _ in range(1, levels):
        w = max(1, cur.width // 2)
        h = max(1, cur.height // 2)
        nxt = cur.resize((w, h), Image.Resampling.BOX)
        if target_cov is not None and 0.0 < target_cov < 1.0:
            arr = np.asarray(nxt).copy()
            a = arr[..., 3].astype(np.float32)
            lo, hi = 0.25, 4.0
            for _ in range(12):
                mid = (lo + hi) / 2
                cov = float((np.clip(a * mid, 0, 255) >= 127.5).mean())
                if cov < target_cov:
                    lo = mid
                else:
                    hi = mid
            arr[..., 3] = np.clip(a * ((lo + hi) / 2), 0, 255).astype(np.uint8)
            nxt = Image.fromarray(arr, "RGBA")
        chain.append(nxt)
        cur = nxt
    return chain


def _pillow_blocks(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="DDS", pixel_format=fmt)
    return buf.getvalue()[128:]


def _pad4(arr: np.ndarray) -> np.ndarray:
    """Complète à un multiple de 4 en répétant le bord : contenu des blocs BC maîtrisé,
    quelle que soit la façon dont l'encodeur traiterait un bloc partiel."""
    h, w = arr.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return arr


def _dxt3_alpha(alpha: np.ndarray) -> bytes:
    """Bloc alpha explicite DXT3 : 16 valeurs sur 4 bits, texel 0 dans les bits de poids faible.
    (L'encodeur DXT3 de Pillow 12 duplique la valeur d'un texel sur deux : on l'écrit ici.)"""
    h, w = alpha.shape
    q = np.round(alpha.astype(np.float32) * (15.0 / 255.0)).astype(np.uint8)
    blocks = q.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)
    packed = (blocks[:, 0::2] | (blocks[:, 1::2] << 4)).astype(np.uint8)       # 8 octets par bloc
    return packed.tobytes()


def encode_blocks(rgba: np.ndarray, codec: str) -> bytes:
    """Encode un niveau RGBA (H, W, 4) uint8 en blocs DXT1/DXT3/DXT5 (encodeur intégré)."""
    arr = _pad4(np.ascontiguousarray(rgba, dtype=np.uint8))
    img = Image.fromarray(arr, "RGBA")
    if codec == "DXT3":
        color = np.frombuffer(_pillow_blocks(img, "DXT5"), np.uint8).reshape(-1, 16)[:, 8:]
        alpha = np.frombuffer(_dxt3_alpha(arr[..., 3]), np.uint8).reshape(-1, 8)
        return np.concatenate([alpha, color], axis=1).tobytes()
    if codec not in ("DXT1", "DXT5"):
        raise DDSError(tr("encodeur intégré : format {codec} non géré", codec=codec))
    return _pillow_blocks(img, codec)


def _encode_level_compressed(img: Image.Image, fmt: str) -> bytes:
    return encode_blocks(np.asarray(img.convert("RGBA")), fmt)


def _encode_level_uncompressed(img: Image.Image, fmt: str) -> bytes:
    bits, rm, gm, bm, am, _flags = UNCOMPRESSED_LAYOUTS[fmt]
    arr = np.asarray(img.convert("RGBA")).astype(np.float32)
    h, w = arr.shape[:2]
    px = np.zeros((h, w), dtype=np.uint32)

    def put(values: np.ndarray, mask: int) -> None:
        nonlocal px
        if not mask:
            return
        shift, nbits = _mask_shift_bits(mask)
        q = np.round(values * (((1 << nbits) - 1) / 255.0)).astype(np.uint32)
        px |= (q << shift) & mask

    if fmt in ("L8", "A8L8"):
        lum = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        put(lum, rm)
        put(arr[..., 3], am)
    elif fmt == "A8":
        put(arr[..., 3], am)
    else:
        put(arr[..., 0], rm)
        put(arr[..., 1], gm)
        put(arr[..., 2], bm)
        put(arr[..., 3], am)
    bpp = bits // 8
    out = np.zeros((h, w, bpp), dtype=np.uint8)
    for i in range(bpp):
        out[..., i] = (px >> (8 * i)) & 0xFF
    return out.tobytes()


# ------------------------------------------------------------------ conteneur DDS
# Codes DXGI des formats écrits avec en-tête DX10, et encodage des données de chaque niveau
DX10_WRITE = {name: code for name, (code, _c) in DX10_LAYOUT.items()}
DX10_WRITE.update({"BC7": 98, "BC7_SRGB": 99})
BLOCK_BYTES = {"DXT1": 8, "DXT3": 16, "DXT5": 16, "BC7": 16}
LEGACY_OF = {"BC1": "DXT1", "BC1_SRGB": "DXT1", "BC2": "DXT3", "BC2_SRGB": "DXT3", "BC3": "DXT5",
             "BC3_SRGB": "DXT5", "BC7": "DXT5", "BC7_SRGB": "DXT5", "DX10_RGBA8": "RGBA8",
             "RGBA8_SRGB": "RGBA8", "DX10_BGRA8": "BGRA8", "BGRA8_SRGB": "BGRA8", "DX10_BGRX8": "BGRX8"}


def codec_of(fmt: str) -> str:
    """Encodage des données d'un format : DXT1/DXT3/DXT5/BC7 (blocs) ou une disposition non compressée."""
    if fmt in ("BC7", "BC7_SRGB"):
        return "BC7"
    if fmt in DX10_LAYOUT:
        return DX10_LAYOUT[fmt][1]
    return fmt


def needs_dx10(fmt: str) -> bool:
    return fmt in DX10_WRITE


def level_dims(w: int, h: int, i: int) -> tuple[int, int]:
    return max(1, w >> i), max(1, h >> i)


def level_bytes(w: int, h: int, codec: str) -> int:
    if codec in BLOCK_BYTES:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * BLOCK_BYTES[codec]
    bits = UNCOMPRESSED_LAYOUTS[codec][0]
    return h * ((w * bits + 7) // 8)


def make_header(fmt: str, w: int, h: int, levels: int) -> bytes:
    """En-tête DDS (signature comprise). Formats legacy : en-tête lisible par D3D9 ;
    formats DX10 : extension DX10 avec le code DXGI exact (sRGB compris)."""
    codec = codec_of(fmt)
    if fmt not in WRITABLE_FORMATS and fmt not in DX10_WRITE:
        raise DDSError(tr("format d'écriture non géré : {fmt}", fmt=fmt))
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    caps = DDSCAPS_TEXTURE
    if levels > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    if codec in BLOCK_BYTES:
        flags |= DDSD_LINEARSIZE
        pitch = level_bytes(w, h, codec)
        cc = codec.encode("ascii") if codec != "BC7" else b"DX10"
        pixfmt = struct.pack("<2I4s5I", 32, DDPF_FOURCC, cc, 0, 0, 0, 0, 0)
    else:
        bits, rm, gm, bm, am, pflags = UNCOMPRESSED_LAYOUTS[codec]
        flags |= DDSD_PITCH
        pitch = (w * bits + 7) // 8
        pixfmt = struct.pack("<2I4s5I", 32, pflags, b"\0\0\0\0", bits, rm, gm, bm, am)
    extra = b""
    if needs_dx10(fmt):
        pixfmt = struct.pack("<2I4s5I", 32, DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0)
        # dxgiFormat, dimension (3 = TEXTURE2D), miscFlag, arraySize, miscFlags2
        extra = struct.pack("<5I", DX10_WRITE[fmt], 3, 0, 1, 0)
    header = struct.pack("<7I", 124, flags, h, w, pitch, 0, levels if levels > 1 else 0)
    header += b"\0" * 44 + pixfmt + struct.pack("<5I", caps, 0, 0, 0, 0)
    assert len(header) == 124
    return DDS_MAGIC + header + extra


def write_dds(fmt: str, w: int, h: int, payloads: list[bytes]) -> bytes:
    """Assemble un DDS à partir des données déjà encodées de chaque niveau (vérifie les tailles)."""
    codec = codec_of(fmt)
    for i, p in enumerate(payloads):
        lw, lh = level_dims(w, h, i)
        if len(p) != level_bytes(lw, lh, codec):
            raise DDSError(tr("niveau {i} ({lw}x{lh}) : {n} octets au lieu de {expected}",
                              i=i, lw=lw, lh=lh, n=len(p), expected=level_bytes(lw, lh, codec)))
    return make_header(fmt, w, h, len(payloads)) + b"".join(payloads)


def split_levels(data: bytes) -> tuple[DDSInfo, str, list[tuple[int, int, bytes]]]:
    """Découpe un DDS 2D en niveaux (largeur, hauteur, données)."""
    info = read_info(data)
    if info.cubemap or info.volume:
        raise DDSError(tr("cubemap ou texture volumique : non gérée"))
    codec = codec_of(info.format) if info.dx10 else info.format
    if codec not in BLOCK_BYTES and codec not in UNCOMPRESSED_LAYOUTS:
        raise DDSError(tr("format {format} : découpage non géré", format=info.format))
    off = 148 if info.dx10 else 128
    out = []
    for i in range(info.mipmaps):
        lw, lh = level_dims(info.width, info.height, i)
        n = level_bytes(lw, lh, codec)
        if off + n > len(data):
            raise DDSError(tr("fichier tronqué au niveau {i}", i=i))
        out.append((lw, lh, data[off:off + n]))
        off += n
    return info, codec, out


def decode_level(codec: str, w: int, h: int, payload: bytes) -> np.ndarray:
    """Décode un niveau isolé en RGBA uint8 (H, W, 4)."""
    if codec in UNCOMPRESSED_LAYOUTS:
        bits, rm, gm, bm, am, _f = UNCOMPRESSED_LAYOUTS[codec]
        info = DDSInfo(width=w, height=h, mipmaps=1, format=codec, masks=(bits, rm, gm, bm, am))
        return np.asarray(_decode_uncompressed(b"\0" * 128 + payload, info))
    pw, ph = max(4, (w + 3) // 4 * 4), max(4, (h + 3) // 4 * 4)
    fmt = "BC7" if codec == "BC7" else codec
    raw = make_header(fmt, pw, ph, 1) + payload
    img = Image.open(io.BytesIO(raw))
    img.load()
    return np.asarray(img.convert("RGBA"))[:h, :w]


def encode_dds(img: Image.Image, fmt: str, mipmaps: int, preserve_coverage: bool = False) -> bytes:
    """Encode une image RGBA en DDS. Formats legacy (DXT1/3/5, masques) : en-tête
    compatible DX9. Formats DX10_LAYOUT : en-tête DX10 avec le code DXGI d'origine."""
    if fmt not in WRITABLE_FORMATS:
        raise DDSError(tr("format d'écriture non géré : {fmt}", fmt=fmt))
    dxgi = None
    if fmt in DX10_LAYOUT:
        dxgi, fmt = DX10_LAYOUT[fmt]
    img = img.convert("RGBA")
    w, h = img.size
    levels = max(1, min(mipmaps, mip_count_full(w, h)))
    chain = build_mip_chain(img, levels, preserve_coverage)
    payload = bytearray()
    for lvl in chain:
        if fmt in COMPRESSED_WRITABLE:
            payload += _encode_level_compressed(lvl, fmt)
        else:
            payload += _encode_level_uncompressed(lvl, fmt)

    name = next((n for n, (code, _c) in DX10_LAYOUT.items() if code == dxgi), None) if dxgi is not None else fmt
    return make_header(name or fmt, w, h, levels) + bytes(payload)


def output_format_for(info_format: str, policy: str, has_alpha: bool) -> str:
    """Choisit le format d'écriture selon la politique de la configuration."""
    policy = (policy or "match").lower()
    if policy == "dxt5":
        return "DXT5"
    if policy == "dxt1":
        return "DXT1" if not has_alpha else "DXT5"
    if policy == "bgra8":
        return "BGRA8"
    # "match" : même format que l'original (en-tête DX10 compris) quand on sait l'écrire
    if info_format in WRITABLE_FORMATS:
        return info_format
    return "BGRA8"
