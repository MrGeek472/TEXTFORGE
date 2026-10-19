"""Mesures d'image pour le contrôle qualité : couleur (CIELAB, ΔE2000), SSIM multi-échelle,
distance de palette, énergie de détail, raccord. numpy + Pillow uniquement.

Principe : la direction artistique d'une texture tient dans ses basses fréquences (ce que
l'original contient à sa propre résolution) ; le micro-détail photoréaliste vit au-dessus.
Les mesures de fidélité se font donc sur l'image générée RAMENÉE à la résolution d'origine,
et les mesures de détail sur l'image pleine résolution.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# ------------------------------------------------------------------ espaces de couleur

_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] (…, 3) -> CIELAB D65 (L 0-100, a, b)."""
    xyz = srgb_to_linear(rgb) @ _RGB2XYZ.T / _WHITE_D65
    e = (6 / 29) ** 3
    f = np.where(xyz > e, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 (Sharma, Wu, Dalal 2005), vectorisé. 1 ≈ seuil de perception en vision
    attentive, 2-3 ≈ écart remarqué au premier coup d'œil."""
    L1, a1, b1 = (lab1[..., i].astype(np.float64) for i in range(3))
    L2, a2, b2 = (lab2[..., i].astype(np.float64) for i in range(3))
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where(C1p == 0, 0.0, np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = np.where(C2p == 0, 0.0, np.degrees(np.arctan2(b2, a2p)) % 360)
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0.0,
                   np.where(np.abs(dh) <= 180, dh, np.where(dh > 180, dh - 360, dh + 360)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                            np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Cbp7 = Cbp ** 7
    RC = 2 * np.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                   + RT * (dCp / SC) * (dHp / SH)).astype(np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


# ------------------------------------------------------------------ filtres

def _pad(a: np.ndarray, r: int, axis: int, wrap: bool) -> np.ndarray:
    pw = [(0, 0)] * a.ndim
    pw[axis] = (r, r)
    n = a.shape[axis]
    mode = "wrap" if wrap else ("reflect" if n > r else "edge")
    return np.pad(a, pw, mode=mode)


def conv1d(a: np.ndarray, kernel: np.ndarray, axis: int, wrap: bool = False) -> np.ndarray:
    r = len(kernel) // 2
    p = _pad(a, r, axis, wrap)
    out = np.zeros_like(a, dtype=np.float32)
    n = a.shape[axis]
    for i, k in enumerate(kernel):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + n)
        out += k * p[tuple(sl)]
    return out


def gauss(a: np.ndarray, sigma: float, wrap_x: bool = False, wrap_y: bool = False) -> np.ndarray:
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    return conv1d(conv1d(a.astype(np.float32), k, 0, wrap_y), k, 1, wrap_x)


def box(a: np.ndarray, radius: int = 1, passes: int = 2, wrap_x: bool = False, wrap_y: bool = False) -> np.ndarray:
    k = np.full(2 * radius + 1, 1.0 / (2 * radius + 1), np.float32)
    out = a.astype(np.float32)
    for _ in range(passes):
        out = conv1d(conv1d(out, k, 0, wrap_y), k, 1, wrap_x)
    return out


def masked_blur(a: np.ndarray, mask: np.ndarray, radius: int, passes: int,
                wrap_x: bool = False, wrap_y: bool = False) -> np.ndarray:
    """Flou normalisé : les zones hors masque (transparentes) ne contaminent pas les valeurs."""
    m = mask.astype(np.float32)
    mm = m[..., None] if a.ndim == 3 else m
    num = box(a * mm, radius, passes, wrap_x, wrap_y)
    den = box(m, radius, passes, wrap_x, wrap_y)
    den = den[..., None] if a.ndim == 3 else den
    return num / np.maximum(den, 1e-4)


def resize_f32(arr: np.ndarray, size: tuple[int, int], resample=Image.Resampling.LANCZOS,
               tile_x: bool = False, tile_y: bool = False) -> np.ndarray:
    """Redimensionne un tableau float (H, W[, C]) canal par canal (mode « F » de Pillow,
    sans quantification). Axes répétables : marge en boucle pour des bords sans cassure."""
    squeeze = arr.ndim == 2
    a = arr[..., None] if squeeze else arr
    h, w = a.shape[:2]
    W, H = size
    if (W, H) == (w, h):
        return arr.astype(np.float32).copy()
    m = 4
    px, py = (m if tile_x and w > 1 else 0), (m if tile_y and h > 1 else 0)
    if px or py:
        a = np.pad(a, ((py, py), (px, px), (0, 0)), mode="wrap")
    sx, sy = W / w, H / h
    big = (max(1, round((w + 2 * px) * sx)), max(1, round((h + 2 * py) * sy)))
    ox, oy = round(px * sx), round(py * sy)
    chans = []
    for c in range(a.shape[2]):
        im = Image.fromarray(np.ascontiguousarray(a[..., c], dtype=np.float32), "F").resize(big, resample)
        chans.append(np.asarray(im, dtype=np.float32)[oy:oy + H, ox:ox + W])
    out = np.stack(chans, axis=-1)
    return out[..., 0] if squeeze else out


# ------------------------------------------------------------------ SSIM

def ssim(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None,
         wrap_x: bool = False, wrap_y: bool = False) -> float:
    """SSIM (Wang et al. 2004) sur deux images en niveaux de gris [0,1], fenêtre gaussienne σ=1.5."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    sigma = 1.5 if min(x.shape) >= 16 else 1.0
    mx, my = gauss(x, sigma, wrap_x, wrap_y), gauss(y, sigma, wrap_x, wrap_y)
    sxx = gauss(x * x, sigma, wrap_x, wrap_y) - mx * mx
    syy = gauss(y * y, sigma, wrap_x, wrap_y) - my * my
    sxy = gauss(x * y, sigma, wrap_x, wrap_y) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    if mask is not None and mask.sum() >= 4:
        return float(m[mask].mean())
    return float(m.mean())


def _down2(a: np.ndarray) -> np.ndarray:
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w]
    return a.reshape(h // 2, 2, w // 2, 2, *a.shape[2:]).mean(axis=(1, 3))


def ms_ssim(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None, *, min_side: int = 12,
            max_scales: int = 3, wrap_x: bool = False, wrap_y: bool = False) -> dict:
    """SSIM à plusieurs échelles (1, 1/2, 1/4…). À l'échelle 1 (pixel d'origine), une matière
    stochastique (pierre, herbe) réinventée par l'IA ne coïncide jamais pixel à pixel : le score
    de décision pondère donc davantage les échelles grossières, qui portent la composition."""
    scores = []
    m = mask
    for _ in range(max_scales):
        if min(x.shape) < min_side:
            break
        scores.append(ssim(x, y, m, wrap_x, wrap_y))
        x, y = _down2(x), _down2(y)
        if m is not None:
            m = _down2(m.astype(np.float32)) > 0.5
    if not scores:
        scores = [ssim(x, y, None)]
    weights = [0.2, 0.35, 0.45][:len(scores)]
    combined = float(np.average(scores, weights=weights))
    return {"scales": [round(s, 3) for s in scores], "combined": round(combined, 3)}


# ------------------------------------------------------------------ couleur et palette

def color_stats(gen_rgb: np.ndarray, ref_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Écarts de couleur entre deux images alignées (même taille), mesurés sur des versions
    légèrement floutées : on compare des zones de couleur, pas le grain."""
    g = masked_blur(gen_rgb, mask, 1, 2)
    r = masked_blur(ref_rgb, mask, 1, 2)
    lg, lr = rgb_to_lab(g), rgb_to_lab(r)
    m = mask if mask.sum() >= 4 else np.ones(mask.shape, bool)
    de = delta_e2000(lg[m], lr[m])
    Lg, Lr = lg[..., 0][m], lr[..., 0][m]
    Cg = np.hypot(lg[..., 1][m], lg[..., 2][m])
    Cr = np.hypot(lr[..., 1][m], lr[..., 2][m])
    out = {
        "delta_e": round(float(de.mean()), 2),
        "delta_e_p95": round(float(np.percentile(de, 95)), 2),
        "lightness_shift": round(float(Lg.mean() - Lr.mean()), 2),
        "contrast_ratio": round(float((Lg.std() + 0.5) / (Lr.std() + 0.5)), 3),
        "chroma_ratio": None,
        "hue_shift": None,
    }
    if Cr.mean() >= 4.0:                       # sinon texture quasi grise : rapport sans objet
        out["chroma_ratio"] = round(float((Cg.mean() + 0.5) / (Cr.mean() + 0.5)), 3)
    both = (Cg > 6) & (Cr > 6)
    if both.sum() >= max(8, int(0.05 * m.sum())):
        hg = np.arctan2(lg[..., 2][m][both], lg[..., 1][m][both])
        hr = np.arctan2(lr[..., 2][m][both], lr[..., 1][m][both])
        w = np.minimum(Cg[both], Cr[both])
        d = hg - hr
        out["hue_shift"] = round(float(np.degrees(np.arctan2((w * np.sin(d)).sum(), (w * np.cos(d)).sum()))), 2)
    return out


def palette_distance(gen_rgb: np.ndarray, ref_rgb: np.ndarray, mask: np.ndarray, bins: int = 6) -> float:
    """Distance de Hellinger entre les histogrammes Lab 3D (0 = même palette, 1 = disjointes).
    Insensible à l'emplacement des couleurs : repère une palette changée même si la
    composition est décalée."""
    m = mask if mask.sum() >= 4 else np.ones(mask.shape, bool)

    def hist(rgb: np.ndarray) -> np.ndarray:
        lab = rgb_to_lab(rgb[m])
        iL = np.clip((lab[:, 0] / 100.0 * bins).astype(int), 0, bins - 1)
        ia = np.clip(((lab[:, 1] + 60) / 120.0 * bins).astype(int), 0, bins - 1)
        ib = np.clip(((lab[:, 2] + 60) / 120.0 * bins).astype(int), 0, bins - 1)
        h = np.bincount((iL * bins + ia) * bins + ib, minlength=bins ** 3).astype(np.float64)
        h = h.reshape(bins, bins, bins)
        for ax in range(3):                       # lissage : un pas de bin ne compte pas comme un saut
            h = (np.roll(h, 1, ax) * 0.25 + h * 0.5 + np.roll(h, -1, ax) * 0.25)
        return h / max(h.sum(), 1e-9)

    p, q = hist(gen_rgb), hist(ref_rgb)
    bc = float(np.sqrt(p * q).sum())
    return round(float(np.sqrt(max(0.0, 1.0 - bc))), 3)


# ------------------------------------------------------------------ détail et raccord

def detail_energy(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Énergie haute fréquence : écart moyen entre la luminance et sa version floutée."""
    y = luma(rgb)
    hf = np.abs(y - box(y, 1, 2))
    if mask is not None and mask.sum() >= 16:
        return float(hf[mask].mean())
    return float(hf.mean())


def seam_ratios(rgb: np.ndarray, axis: int) -> tuple[float, float]:
    """Raccord en répétition (dernière -> première colonne/ligne), rapporté aux transitions
    internes courantes, à pleine résolution et après moyenne 4x4 (cassure basse fréquence
    masquée par le grain). ~1 = invisible."""
    from .imaging import _box_down, wrap_seam_ratio
    full = wrap_seam_ratio(rgb, axis)
    low = full
    if min(rgb.shape[:2]) >= 32:
        low = wrap_seam_ratio(_box_down(rgb, 4), axis)
    return float(full), float(low)


def seam_local(rgb: np.ndarray, axis: int, m: int) -> float:
    """Raccord comparé aux transitions VOISINES du bord (m colonnes de chaque côté) : un joint
    de mortier ou une veine qui ne se rejoint pas ressort, même si la texture contient
    ailleurs des transitions plus fortes."""
    a = rgb if axis == 1 else rgb.transpose(1, 0, 2)
    m = max(1, min(m, a.shape[1] // 4))
    s = float(np.abs(a[:, -1] - a[:, 0]).mean())
    left = float(np.abs(np.diff(a[:, :m + 1], axis=1)).mean())
    right = float(np.abs(np.diff(a[:, -(m + 1):], axis=1)).mean())
    return s / max((left + right) / 2, 1e-4)


def seam_step(gen_small: np.ndarray, ref: np.ndarray, axis: int, mask: np.ndarray | None = None) -> float:
    """Dérive de couleur au raccord, à la résolution d'origine. On forme le champ d'erreur
    basse fréquence E = Lab(flou(généré)) − Lab(flou(original)), flou SANS bouclage pour garder
    les deux bords séparés ; l'original étant raccordable, le raccord du rendu ne casse que si E
    diffère d'un bord à l'autre. Repère une dérive lente de luminosité ou de teinte : invisible
    au pixel près, elle dessine une grille quand la texture se répète. Résultat en ΔE (Lab)."""
    g = gen_small if axis == 1 else gen_small.transpose(1, 0, 2)
    r = ref if axis == 1 else ref.transpose(1, 0, 2)
    mk = None if mask is None else (mask if axis == 1 else mask.T)
    h, w = g.shape[:2]
    E = rgb_to_lab(box(g, 2, 2)) - rgb_to_lab(box(r, 2, 2))
    n = max(2, w // 8)                       # bandes larges : une arête nette au bord ne compte pas
    if mk is not None and mk[:, :n].sum() >= 4 and mk[:, -n:].sum() >= 4:
        last = E[:, -n:][mk[:, -n:]].mean(axis=0)
        first = E[:, :n][mk[:, :n]].mean(axis=0)
    else:
        last, first = E[:, -n:].reshape(-1, 3).mean(axis=0), E[:, :n].reshape(-1, 3).mean(axis=0)
    d = last - first
    return float(np.sqrt((d * d).sum()))


def seam_alignment(rgb: np.ndarray, axis: int, max_shift: int) -> tuple[int, float]:
    """Les motifs se rejoignent-ils au raccord ? On cherche le décalage le long du bord qui
    aligne le mieux la dernière colonne sur la première (détail seul, tendance retirée).
    Raccord continu : meilleur décalage 0. Joint de mortier ou veine qui « saute » : décalage
    franc. Renvoie (décalage en pixels, gain de corrélation par rapport au décalage nul)."""
    a = rgb if axis == 1 else rgb.transpose(1, 0, 2)
    y = luma(a)
    last, first = y[:, -2:].mean(axis=1), y[:, :2].mean(axis=1)
    rad = max(2, len(last) // 32)
    k = np.full(2 * rad + 1, 1.0 / (2 * rad + 1), np.float32)
    hp = lambda v: v - conv1d(v[:, None], k, 0, wrap=True)[:, 0]          # noqa: E731
    L, F = hp(last), hp(first)

    def corr(s: int) -> float:
        b = np.roll(F, s)
        d = np.sqrt((L * L).sum() * (b * b).sum())
        return float((L * b).sum() / d) if d > 1e-9 else 0.0

    c0 = corr(0)
    best_s, best_c = 0, c0
    if c0 > 0.6:                                  # bords déjà bien alignés : inutile de chercher
        return 0, 0.0
    for s in range(-max_shift, max_shift + 1):
        c = corr(s)
        if c > best_c + 1e-6:
            best_s, best_c = s, c
    return best_s, best_c - c0
