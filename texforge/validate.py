"""Validation automatique d'une texture générée par rapport à l'originale.

Contrôles (tous à basse résolution, sur la luminance floutée) :
- structure : corrélation globale de la luminance et des contours
- décalage : translation qui maximise la corrélation (recherche exhaustive ±12 %)
- local : pire bloc d'une grille 6x6 (corrélation locale et écart de couleur moyen),
  pour repérer un objet ajouté ou effacé que le score global laisserait passer
- raccord : pour une texture répétable, le bord droit rejoint-il le bord gauche ?

Seuils réglables dans [quality]. Ils ont été calibrés sur des variantes
synthétiques (grain, flou, miroir, décalage, objet ajouté) : à recalibrer
d'après les résultats réels affichés dans la galerie de revue.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .i18n import tr
from .imaging import (LANCZOS, Analysis, _wrap_resize, bleed_rgb, box_blur, luminance, to_f32,
                      wrap_seam_ratio)


def _compare_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    s = long_side / max(w, h)
    return max(8, round(w * s)), max(8, round(h * s))


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    x = a - a.mean()
    y = b - b.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-9 else 0.0


def _grad(a: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.hypot(gx, gy)


def _aligned(a: np.ndarray, b: np.ndarray, dx: int, dy: int, tile_x: bool, tile_y: bool):
    """Vues de a et b décalées de (dx, dy) : en boucle sur un axe répétable,
    sinon seulement la zone de recouvrement."""
    h, w = a.shape
    if tile_x:
        b = np.roll(b, dx, axis=1)
    elif dx > 0:
        a, b = a[:, dx:], b[:, :w - dx]
    elif dx < 0:
        a, b = a[:, :w + dx], b[:, -dx:]
    if tile_y:
        b = np.roll(b, dy, axis=0)
    elif dy > 0:
        a, b = a[dy:, :], b[:h - dy, :]
    elif dy < 0:
        a, b = a[:h + dy, :], b[-dy:, :]
    return a, b


def _best_shift(a: np.ndarray, b: np.ndarray, tile_x: bool, tile_y: bool, frac: float = 0.12) -> tuple[float, float]:
    """Translation (dx, dy) en fraction de la taille qui aligne le mieux b sur a."""
    h, w = a.shape
    sx, sy = max(1, int(round(w * frac))), max(1, int(round(h * frac)))
    best, best_d = -2.0, (0, 0)
    for dy in range(-sy, sy + 1):
        for dx in range(-sx, sx + 1):
            aa, bb = _aligned(a, b, dx, dy, tile_x, tile_y)
            c = _ncc(aa, bb)
            # légère préférence pour le décalage nul en cas d'égalité (motifs périodiques)
            c -= 1e-4 * (abs(dx) + abs(dy))
            if c > best:
                best, best_d = c, (dx, dy)
    return best_d[0] / w, best_d[1] / h


def _local_worst(la: np.ndarray, lb: np.ndarray, ref_rgb: np.ndarray, gen_rgb: np.ndarray,
                 mask: np.ndarray, grid: int = 6) -> tuple[float, float]:
    h, w = la.shape
    worst_ncc, worst_col = 1.0, 0.0
    for gy in range(grid):
        for gx in range(grid):
            ys = slice(gy * h // grid, (gy + 1) * h // grid)
            xs = slice(gx * w // grid, (gx + 1) * w // grid)
            m = mask[ys, xs]
            if m.sum() < max(4, m.size // 3):
                continue
            col = float(np.abs(ref_rgb[ys, xs][m].mean(axis=0) - gen_rgb[ys, xs][m].mean(axis=0)).mean())
            worst_col = max(worst_col, col)
            a, b = la[ys, xs], lb[ys, xs]
            if a[m].std() > 0.03:
                worst_ncc = min(worst_ncc, _ncc(a, b, m))
    return worst_ncc, worst_col


def _comparable(img: Image.Image, an: Analysis) -> Image.Image:
    """Image comparable après réduction : fond étendu sous une transparence réelle ; sinon couleurs
    seules (l'alpha d'une normal map DXT5nm est une donnée, que Pillow prémultiplierait)."""
    return bleed_rgb(img) if an.cutout else img.convert("RGB").convert("RGBA")


def evaluate(src: Image.Image, result: Image.Image, an: Analysis, cfg: dict) -> dict:
    q = cfg["quality"]
    base = _comparable(src, an)
    # Même traitement des zones transparentes des deux côtés : sinon Pillow réduit le résultat
    # RGBA en alpha prémultiplié (fond noir) face à un original au fond étendu, et une découpe
    # peu couvrante (feuillage, grillage) paraît décalée ou réinventée.
    res_base = _comparable(result, an)

    def prepare(long_side: int):
        cw, ch = _compare_size(src.width, src.height, long_side)
        ref = to_f32(base.resize((cw, ch), LANCZOS))
        gen = to_f32(res_base.resize((cw, ch), LANCZOS))
        if an.cutout:
            mask = to_f32(src.resize((cw, ch), LANCZOS))[..., 3] > 0.5
        else:
            mask = np.ones((ch, cw), bool)
        if mask.sum() < 16:
            mask = np.ones((ch, cw), bool)
        return ref, gen, mask

    ref, gen, mask = prepare(int(np.clip(max(src.width, src.height), 64, 192)))
    la = box_blur(luminance(ref[..., :3]))
    lb = box_blur(luminance(gen[..., :3]))
    low_contrast = float(la[mask].std()) < 0.02

    def struct_at(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        nl = _ncc(a, b, mask)
        ga = _grad(a)
        if float(ga[mask].mean()) < 0.004:          # original lisse : les contours ne disent rien
            return nl, nl, 0.0
        ne = _ncc(box_blur(ga, 1), box_blur(_grad(b), 1), mask)
        return 0.6 * nl + 0.4 * ne, nl, ne

    # Deux échelles : le détail fin ajouté par l'IA ne doit pas masquer une disposition conservée
    fine = struct_at(la, lb)
    coarse = struct_at(box_blur(la, 3), box_blur(lb, 3))
    best_scale = fine if fine[0] >= coarse[0] else coarse
    structure, ncc_lum, ncc_edge = best_scale
    if low_contrast:
        structure = 1.0

    # Décalage : recherche exhaustive à basse résolution (robuste au grain ajouté par l'IA)
    r2, g2, m2 = prepare(64)
    sa = box_blur(luminance(r2[..., :3]), 2)
    sb = box_blur(luminance(g2[..., :3]), 2)
    dx, dy = _best_shift(sa, sb, an.tile_x, an.tile_y)
    shift = float(max(abs(dx), abs(dy)))

    worst_ncc, worst_col = _local_worst(la, lb, box_blur(ref[..., :3], 1), box_blur(gen[..., :3], 1), mask)

    # Raccord : jugé par rapport à l'original agrandi en boucle à la même taille
    res_arr = to_f32(res_base)
    seam_max, seam_limit = None, None
    if an.tile_x or an.tile_y:
        ref_up = _wrap_resize(np.asarray(base)[..., :3], result.size, an.tile_x, an.tile_y).astype(np.float32) / 255.0
        for axis, on in ((1, an.tile_x), (0, an.tile_y)):
            if not on:
                continue
            r_res = wrap_seam_ratio(res_arr, axis)
            limit = max(float(q.get("max_seam_ratio", 1.8)), wrap_seam_ratio(ref_up, axis) * 1.25)
            if seam_max is None or r_res - limit > seam_max - seam_limit:
                seam_max, seam_limit = r_res, limit

    reasons = []
    min_struct = float(q.get("min_structure", 0.65))
    if structure < min_struct:
        reasons.append(tr("structure {structure:.2f} < {lim}", structure=structure, lim=min_struct))
    if not low_contrast and shift > float(q.get("max_shift", 0.03)):
        reasons.append(tr("décalage {shift:.3f} > {lim}", shift=shift, lim=q.get("max_shift", 0.03)))
    if not low_contrast and worst_ncc < float(q.get("min_local", 0.3)):
        reasons.append(tr("zone modifiée (corrélation locale {v:.2f})", v=worst_ncc))
    if worst_col > float(q.get("max_local_color", 0.12)):
        reasons.append(tr("couleur locale changée ({v:.0f}/255)", v=worst_col * 255))
    if seam_max is not None and seam_max > seam_limit:
        reasons.append(tr("raccord {v:.2f} > {lim:.2f}", v=seam_max, lim=seam_limit))

    score = (structure - 4.0 * shift + 0.3 * min(worst_ncc, 1.0) - 1.0 * worst_col
             - 0.1 * max(0.0, (seam_max or 0) - (seam_limit or 0)))
    return {
        "passed": not reasons,
        "score": round(score, 4),
        "structure": round(structure, 3),
        "ncc_lum": round(ncc_lum, 3),
        "ncc_edge": round(ncc_edge, 3),
        "shift": round(shift, 3),
        "local": round(worst_ncc, 3),
        "local_color": round(worst_col * 255, 1),
        "seam": None if seam_max is None else round(seam_max, 3),
        "low_contrast": low_contrast,
        "reasons": reasons,
    }
