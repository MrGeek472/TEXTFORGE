"""Outils d'image : chargement, analyse, préparation pour l'IA, post-traitement,
agrandissement local. Tout est fait en numpy + Pillow (aucune dépendance GPU)."""

from __future__ import annotations

import io
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import dds

IMAGE_EXTS = {".dds", ".png", ".tga", ".bmp", ".jpg", ".jpeg"}
LANCZOS = Image.Resampling.LANCZOS


# ---------------------------------------------------------------- chargement

@dataclass
class SourceTexture:
    image: Image.Image          # RGBA, niveau 0
    fmt: str                    # format d'origine ("DXT1", "BGRA8", "PNG"...)
    mipmaps: int
    is_dds: bool


def load_texture(path: Path) -> SourceTexture:
    data = path.read_bytes()
    if path.suffix.lower() == ".dds":
        img, info = dds.load_dds(data)
        return SourceTexture(img, info.format, info.mipmaps, True)
    img = Image.open(io.BytesIO(data))
    img.load()
    return SourceTexture(img.convert("RGBA"), (img.format or path.suffix[1:]).upper(), 1, False)


# ---------------------------------------------------------------- outils numpy

def to_f32(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0


def from_f32(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGBA")


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def box_blur(a: np.ndarray, passes: int = 2) -> np.ndarray:
    out = a.astype(np.float32)
    for _ in range(passes):
        p = np.pad(out, ((1, 1), (1, 1)) + ((0, 0),) * (out.ndim - 2), mode="edge")
        out = sum(p[dy:dy + out.shape[0], dx:dx + out.shape[1]] for dy in range(3) for dx in range(3)) / 9.0
    return out


def resize_rgba(img: Image.Image, size: tuple[int, int], resample=LANCZOS) -> Image.Image:
    """Redimensionne en évitant que les pixels transparents ne bavent dans les
    couleurs (prémultiplication)."""
    arr = to_f32(img)
    a = arr[..., 3:4]
    if a.min() >= 0.999:
        return img.resize(size, resample)
    pre = np.concatenate([arr[..., :3] * a, a], axis=-1)
    pre_img = from_f32(pre)
    big = to_f32(pre_img.resize(size, resample))
    alpha = np.clip(big[..., 3:4], 0, 1)
    rgb = np.where(alpha > 1e-4, big[..., :3] / np.maximum(alpha, 1e-4), 0)
    return from_f32(np.concatenate([np.clip(rgb, 0, 1), alpha], axis=-1))


def bleed_rgb(img: Image.Image, iterations: int = 24) -> Image.Image:
    """Étend les couleurs opaques dans les zones transparentes : le modèle IA voit
    une surface continue au lieu de couleurs parasites cachées par l'alpha."""
    arr = to_f32(img)
    alpha = arr[..., 3]
    known = alpha > 0.5
    if known.all() or not known.any():
        out = arr.copy()
        out[..., 3] = 1.0
        return from_f32(out)
    # « Push-pull » : moyenne des couleurs connues sur une pyramide de résolutions,
    # puis remplissage des trous depuis le niveau grossier. Donne un fond doux, sans
    # motif parasite que le modèle pourrait prendre pour du contenu.
    weight = known.astype(np.float32)
    color = arr[..., :3] * weight[..., None]
    levels = [(color, weight)]
    while max(levels[-1][1].shape) > 1:
        c, wgt = levels[-1]
        h, w = wgt.shape
        ph, pw = h + (h % 2), w + (w % 2)
        c = np.pad(c, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
        wgt = np.pad(wgt, ((0, ph - h), (0, pw - w)), mode="edge")
        c2 = c.reshape(ph // 2, 2, pw // 2, 2, 3).sum(axis=(1, 3))
        w2 = wgt.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
        levels.append((c2, w2))
    filled_rgb = levels[-1][0] / np.maximum(levels[-1][1], 1e-6)[..., None]
    for c, wgt in reversed(levels[:-1]):
        h, w = wgt.shape
        up = np.repeat(np.repeat(filled_rgb, 2, axis=0), 2, axis=1)[:h, :w]
        up = box_blur(up, 1)
        own = c / np.maximum(wgt, 1e-6)[..., None]
        t = np.clip(wgt, 0, 1)[..., None]
        filled_rgb = own * t + up * (1 - t)
    rgb = np.where(known[..., None], arr[..., :3], filled_rgb)
    out = np.concatenate([rgb, np.ones_like(alpha)[..., None]], axis=-1)
    return from_f32(out)


# ---------------------------------------------------------------- analyse

@dataclass
class Analysis:
    width: int
    height: int
    alpha_class: str        # "none" | "binary" | "gradient"
    tile_x: bool
    tile_y: bool
    lum_std: float
    chroma_std: float
    normal_map: bool
    mode: str               # "ai" | "upscale" | "keep"
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def seam_stats(arr: np.ndarray, axis: int) -> tuple[float, float, float]:
    """Écart moyen au raccord (dernière -> première colonne, ou ligne), comparé aux
    écarts entre colonnes voisines à l'intérieur : (raccord, 95e centile, maximum)."""
    rgb = arr[..., :3]
    if axis == 0:
        rgb = rgb.transpose(1, 0, 2)
    pairs = np.abs(np.diff(rgb, axis=1)).mean(axis=(0, 2))       # un écart par paire de colonnes
    seam = float(np.abs(rgb[:, 0] - rgb[:, -1]).mean())
    if pairs.size == 0:
        return seam, 1e-4, 1e-4
    return seam, float(np.percentile(pairs, 95)), float(pairs.max())


def wrap_seam_ratio(arr: np.ndarray, axis: int) -> float:
    """~1 : le raccord ressemble à n'importe quelle transition interne (invisible)."""
    seam, p95, _ = seam_stats(arr, axis)
    return seam / max(p95, 1e-4)


def _box_down(arr: np.ndarray, f: int) -> np.ndarray:
    h, w = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
    return arr[:h, :w].reshape(h // f, f, w // f, f, arr.shape[2]).mean(axis=(1, 3))


def is_tileable(arr: np.ndarray, axis: int, threshold: float) -> bool:
    """Répétable si le raccord passe à pleine résolution ET après moyenne 4x4 : sur une
    texture à grain fin, le bruit masque une cassure de bord que la moyenne fait ressortir."""
    if not _tileable_at(arr, axis, threshold):
        return False
    if min(arr.shape[:2]) >= 32:
        return _tileable_at(_box_down(arr, 4), axis, threshold)
    return True


def _tileable_at(arr: np.ndarray, axis: int, threshold: float) -> bool:
    """Le raccord n'est pas plus marqué que les transitions internes courantes, ou il
    reproduit, ligne par ligne, une transition qui existe déjà dans la texture (cas d'un
    joint de carrelage placé pile au bord)."""
    seam, p95, _mx = seam_stats(arr, axis)
    if seam <= threshold * max(p95, 1e-4):
        return True
    rgb = arr[..., :3] if axis == 1 else arr[..., :3].transpose(1, 0, 2)
    s = np.abs(rgb[:, -1] - rgb[:, 0])                       # (lignes, 3)
    d = np.abs(np.diff(rgb, axis=1))                         # (lignes, paires, 3)
    norm = max(float(np.sqrt((s * s).sum())), 1e-6)
    dist = np.sqrt(((d - s[:, None, :]) ** 2).sum(axis=(0, 2))) / norm
    return bool(dist.min() < 0.35)


def classify_alpha(alpha: np.ndarray) -> str:
    if alpha.min() >= 0.98:
        return "none"
    mid = ((alpha > 0.06) & (alpha < 0.94)).mean()
    return "binary" if mid < 0.08 else "gradient"


def analyze(img: Image.Image, cfg: dict) -> Analysis:
    ac = cfg["analysis"]
    arr = to_f32(img)
    h, w = arr.shape[:2]
    alpha_class = classify_alpha(arr[..., 3])
    opaque = arr[..., 3] > 0.5
    rgb = arr[..., :3][opaque] if opaque.any() else arr[..., :3].reshape(-1, 3)
    lum = luminance(rgb)
    lum_std = float(lum.std())
    chroma_std = float((rgb - lum[:, None]).std())
    mean = rgb.mean(axis=0)
    normal_map = bool(mean[2] > 0.7 and abs(mean[0] - 0.5) < 0.12 and abs(mean[1] - 0.5) < 0.12
                      and alpha_class == "none")
    # Détail local = énergie du laplacien, mesurée à 64 px max pour ne pas dépendre
    # de la résolution. Un dégradé (ciel, vitre) a un laplacien quasi nul.
    small = img if max(w, h) <= 64 else img.resize(
        (max(1, round(w * 64 / max(w, h))), max(1, round(h * 64 / max(w, h)))), LANCZOS)
    sa = to_f32(small)
    sl = luminance(sa[..., :3])
    sm = sa[..., 3] > 0.5
    p = np.pad(sl, 1, mode="edge")
    lap = np.abs(4 * sl - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:])
    detail = float(lap[sm].mean()) if sm.any() else 0.0
    smooth = detail < float(ac.get("smooth_threshold", 0.006))
    thr = float(ac.get("tile_threshold", 1.5))
    tile_x = w >= 8 and is_tileable(arr, 1, thr)
    tile_y = h >= 8 and is_tileable(arr, 0, thr)
    # Bords presque entièrement transparents (icône, feuillage découpé) : rien à raccorder
    a = arr[..., 3]
    if tile_x and ((a[:, 0] < 0.5).mean() > 0.9 and (a[:, -1] < 0.5).mean() > 0.9):
        tile_x = False
    if tile_y and ((a[0, :] < 0.5).mean() > 0.9 and (a[-1, :] < 0.5).mean() > 0.9):
        tile_y = False

    min_size = int(ac.get("min_ai_size", 32))
    copy_below = int(ac.get("copy_below", 8))
    aspect = max(w, h) / max(1, min(w, h))
    if max(w, h) < copy_below:
        mode, reason = "keep", "minuscule (table de couleurs probable)"
    elif max(w, h) < min_size:
        mode, reason = "upscale", f"plus petite que {min_size}px"
    elif lum_std < 0.015 and chroma_std < 0.015:
        mode, reason = "upscale", "couleur quasi unie"
    elif smooth:
        mode, reason = "upscale", "dégradé lisse, sans détail à réinterpréter"
    elif normal_map:
        mode, reason = "upscale", "carte de relief probable (normal map)"
    elif aspect > float(ac.get("max_ai_aspect", 3.0)):
        mode, reason = "upscale", f"format très allongé ({aspect:.1f}:1)"
    else:
        mode, reason = "ai", "texture de couleur"
    return Analysis(w, h, alpha_class, bool(tile_x), bool(tile_y), lum_std, chroma_std,
                    normal_map, mode, reason)


# ---------------------------------------------------------------- taille de sortie

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def output_scale(w: int, h: int, scale: int, max_side: int) -> int:
    """Facteur d'agrandissement réel. Une texture aux côtés en puissance de 2 reste
    en puissance de 2 (beaucoup de jeux DX9 l'exigent)."""
    k = max(1, int(scale))
    pow2 = _is_pow2(w) and _is_pow2(h)
    if pow2 and not _is_pow2(k):
        k = 1 << (k.bit_length() - 1)
    while k > 1 and max(w, h) * k > max_side:
        k = k // 2 if pow2 else k - 1
    return max(1, k)


# ---------------------------------------------------------------- préparation IA

def parse_ratio(r: str) -> float:
    a, b = r.split(":")
    return float(a) / float(b)


@dataclass
class PrepInfo:
    src_w: int
    src_h: int
    pad_x: int
    pad_y: int
    padded_w: int
    padded_h: int
    aspect_ratio: str
    fit: str                 # "stretch" | "pad"
    rect: tuple              # zone utile normalisée (x0, y0, x1, y1) dans l'image envoyée
    input_w: int
    input_h: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rect"] = list(self.rect)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PrepInfo":
        d = dict(d)
        d["rect"] = tuple(d["rect"])
        return cls(**d)


def prepare_for_model(img: Image.Image, an: Analysis, cfg: dict, ratios: list[str]) -> tuple[Image.Image, PrepInfo]:
    """Construit l'image envoyée au modèle et mémorise comment l'inverser."""
    w, h = img.size
    base = bleed_rgb(img) if an.alpha_class != "none" else img.convert("RGBA")
    arr = np.asarray(base)[..., :3]
    frac = float(cfg["analysis"].get("wrap_fraction", 0.125))
    px = max(1, round(w * frac)) if an.tile_x else 0
    py = max(1, round(h * frac)) if an.tile_y else 0
    if px or py:
        arr = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="wrap")
    ph, pw = arr.shape[:2]
    padded = Image.fromarray(arr, "RGB")

    r = pw / ph
    best = min(ratios, key=lambda s: abs(math.log(r / parse_ratio(s))))
    t = parse_ratio(best)
    long_side = int(cfg["provider"].get("input_long_side", 1024))
    cw, ch = (long_side, max(1, round(long_side / t))) if t >= 1 else (max(1, round(long_side * t)), long_side)
    tol = float(cfg["analysis"].get("stretch_tolerance", 1.35))

    if abs(math.log(r / t)) <= math.log(tol):
        model_in = padded.resize((cw, ch), LANCZOS)
        return model_in, PrepInfo(w, h, px, py, pw, ph, best, "stretch", (0.0, 0.0, 1.0, 1.0), cw, ch)

    s = min(cw / pw, ch / ph)
    iw, ih = max(1, round(pw * s)), max(1, round(ph * s))
    inner = np.asarray(padded.resize((iw, ih), LANCZOS))
    x0, y0 = (cw - iw) // 2, (ch - ih) // 2
    canvas = np.pad(inner, ((y0, ch - ih - y0), (x0, cw - iw - x0), (0, 0)), mode="reflect" if min(iw, ih) > 1 else "edge")
    rect = (x0 / cw, y0 / ch, (x0 + iw) / cw, (y0 + ih) / ch)
    return Image.fromarray(canvas, "RGB"), PrepInfo(w, h, px, py, pw, ph, best, "pad", rect, cw, ch)


def prepare_native(img: Image.Image, an: Analysis) -> tuple[Image.Image, PrepInfo]:
    """Entrée d'un moteur qui gère lui-même le raccord et la résolution (workflow ComfyUI) :
    texture à sa taille d'origine, transparence remplie, aucune marge."""
    w, h = img.size
    base = bleed_rgb(img) if an.alpha_class != "none" else img.convert("RGBA")
    return base.convert("RGB"), PrepInfo(w, h, 0, 0, w, h, "1:1", "stretch", (0.0, 0.0, 1.0, 1.0), w, h)


# ---------------------------------------------------------------- post-traitement

def _feather_wrap(arr: np.ndarray, pad: int, size: int, axis: int) -> np.ndarray:
    """Recadre la zone centrale et fond le bord droit (ou bas) avec la marge gauche
    (ou haute) générée : la texture se raccorde sans joint en répétition."""
    if pad <= 0:
        return arr
    if axis == 0:
        return _feather_wrap(arr.transpose(1, 0, 2), pad, size, 1).transpose(1, 0, 2)
    center = arr[:, pad:pad + size].copy()
    left_margin = arr[:, :pad]                       # correspond aux colonnes [size-pad, size)
    t = ((np.arange(pad, dtype=np.float32) + 0.5) / pad)[None, :, None]
    band = center[:, size - pad:size]
    center[:, size - pad:size] = band * (1.0 - t) + left_margin * t
    return center


def color_match(gen: np.ndarray, ref: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Recale moyenne et contraste (espace luminance/chrominance) sur la référence."""
    if strength <= 0 or mask.sum() < 4:
        return gen

    def to_ycc(x):
        y = luminance(x)
        return np.stack([y, x[..., 2] - y, x[..., 0] - y], axis=-1)

    def from_ycc(c):
        y, cb, cr = c[..., 0], c[..., 1], c[..., 2]
        r = cr + y
        b = cb + y
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        return np.stack([r, g, b], axis=-1)

    g, rf = to_ycc(gen), to_ycc(ref)
    out = g.copy()
    for c in range(3):
        gm, gs = g[..., c][mask].mean(), g[..., c][mask].std()
        rm, rs = rf[..., c][mask].mean(), rf[..., c][mask].std()
        scale = rs / gs if gs > 1e-4 else 1.0
        scale = float(np.clip(scale, 0.5, 2.0))
        matched = (g[..., c] - gm) * scale + rm
        out[..., c] = g[..., c] * (1 - strength) + matched * strength
    return np.clip(from_ycc(out), 0, 1)


def finalize_ai(output: Image.Image, src: Image.Image, an: Analysis, prep: PrepInfo, k: int, cfg: dict) -> tuple[Image.Image, dict]:
    """Ramène la sortie du modèle à la géométrie exacte de la texture d'origine."""
    ow, oh = output.size
    x0, y0, x1, y1 = prep.rect
    box = (round(x0 * ow), round(y0 * oh), max(round(x0 * ow) + 1, round(x1 * ow)), max(round(y0 * oh) + 1, round(y1 * oh)))
    crop = output.convert("RGB").crop(box)
    fw, fh = prep.padded_w * k, prep.padded_h * k
    if prep.pad_x == 0 and prep.pad_y == 0 and (an.tile_x or an.tile_y):
        # sortie déjà raccordable (padding circulaire du workflow) : redimensionner en boucle
        arr = _wrap_resize(np.asarray(crop), (fw, fh), an.tile_x, an.tile_y).astype(np.float32) / 255.0
    else:
        arr = np.asarray(crop.resize((fw, fh), LANCZOS), dtype=np.float32) / 255.0
    arr = _feather_wrap(arr, prep.pad_x * k, prep.src_w * k, axis=1)
    arr = _feather_wrap(arr, prep.pad_y * k, prep.src_h * k, axis=0)
    W, H = prep.src_w * k, prep.src_h * k
    arr = arr[:H, :W]

    ref_rgba = local_upscale(src, k, an, cfg, use_esrgan=False)
    ref = to_f32(ref_rgba)
    src_alpha = ref[..., 3]
    mask = src_alpha > 0.5 if an.alpha_class != "none" else np.ones(src_alpha.shape, bool)
    drift = float(np.abs(arr[mask].mean(axis=0) - ref[..., :3][mask].mean(axis=0)).mean() * 255) if mask.any() else 0.0
    strength = float(cfg["quality"].get("color_match", 0.6))
    rgb = color_match(arr, ref[..., :3], mask, strength)
    alpha = final_alpha(src, an, (W, H))
    out = from_f32(np.concatenate([rgb, alpha[..., None]], axis=-1))
    return out, {"color_drift": round(drift, 2)}


def final_alpha(src: Image.Image, an: Analysis, size: tuple[int, int]) -> np.ndarray:
    if an.alpha_class == "none":
        return np.ones((size[1], size[0]), np.float32)
    a = src.getchannel("A")
    if an.tile_x or an.tile_y:
        a = Image.fromarray(_wrap_resize(np.asarray(a)[..., None], size, an.tile_x, an.tile_y)[..., 0], "L")
    else:
        a = a.resize(size, LANCZOS)
    arr = np.asarray(a, dtype=np.float32) / 255.0
    if an.alpha_class == "binary":
        arr = (arr >= 0.5).astype(np.float32)
    return arr


def _wrap_resize(arr: np.ndarray, size: tuple[int, int], tx: bool, ty: bool) -> np.ndarray:
    """Redimensionne une texture répétable sans bords flous : marge en boucle,
    agrandissement, puis recadrage."""
    h, w = arr.shape[:2]
    m = 4
    px, py = (m if tx else 0), (m if ty else 0)
    padded = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="wrap" if (tx or ty) else "edge")
    kx, ky = size[0] / w, size[1] / h
    big_size = (round((w + 2 * px) * kx), round((h + 2 * py) * ky))
    mode = "RGBA" if arr.shape[2] == 4 else ("RGB" if arr.shape[2] == 3 else "L")
    img = Image.fromarray(padded[..., 0] if mode == "L" else padded, mode)
    big = np.asarray(img.resize(big_size, LANCZOS))
    if big.ndim == 2:
        big = big[..., None]
    ox, oy = round(px * kx), round(py * ky)
    return big[oy:oy + size[1], ox:ox + size[0]]


# ---------------------------------------------------------------- agrandissement local

def _esrgan(img: Image.Image, exe: str, model: str) -> Image.Image | None:
    if not exe or not Path(exe).exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        i, o = Path(td) / "in.png", Path(td) / "out.png"
        img.convert("RGB").save(i)
        try:
            subprocess.run([exe, "-i", str(i), "-o", str(o), "-n", model, "-s", "4"],
                           check=True, capture_output=True, timeout=300)
            out = Image.open(o)
            out.load()
            return out.convert("RGB")
        except Exception:
            return None


def local_upscale(src: Image.Image, k: int, an: Analysis, cfg: dict, use_esrgan: bool = True) -> Image.Image:
    """Agrandissement sans IA générative : Real-ESRGAN si configuré, sinon Lanczos.
    Sert de repli et de référence pour la validation."""
    W, H = src.width * k, src.height * k
    base = bleed_rgb(src) if an.alpha_class != "none" else src.convert("RGBA")
    rgb = None
    if use_esrgan and k > 1:
        lc = cfg.get("local", {})
        up = _esrgan(base, lc.get("esrgan_exe", ""), lc.get("esrgan_model", "realesrgan-x4plus"))
        if up is not None:
            rgb = np.asarray(up.resize((W, H), LANCZOS))
    if rgb is None:
        arr = np.asarray(base)[..., :3]
        rgb = _wrap_resize(arr, (W, H), an.tile_x, an.tile_y)
    alpha = (final_alpha(src, an, (W, H)) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(np.concatenate([rgb[..., :3], alpha[..., None]], axis=-1), "RGBA")


def save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)          # remplace aussi un fichier existant sous Windows


def thumbnail(img: Image.Image, max_side: int = 384) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s >= 1:
        return img.copy()
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), LANCZOS)
