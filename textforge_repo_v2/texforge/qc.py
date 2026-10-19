"""Contrôle qualité final d'une texture remasterisée, avant écriture du DDS.

Pour chaque texture, dans l'ordre :
1. géométrie : taille cible exacte (facteur k de l'original, puissance de 2 conservée) ;
2. transparence : le masque de l'original est réinjecté (découpe : seuil net ; dégradé :
   agrandissement doux), les couleurs sous les zones transparentes sont étendues ;
3. mesures, sur l'image ramenée à la résolution d'origine (fidélité à la DA) :
   structure (corrélation, décalage, pire bloc, SSIM multi-échelle), couleur (ΔE2000 moyen
   et 95e centile, dérive de teinte, saturation, luminosité, distance de palette) ;
   puis à pleine résolution : raccord des textures répétables, gain de détail ;
4. correctifs automatiques : verrou colorimétrique basses fréquences (la couleur et la
   composition de l'original à sa résolution, le micro-détail de l'IA au-dessus), fondu des
   raccords ; une structure trahie (objet inventé, composition déplacée) n'est pas corrigée ;
5. décision : validée | corrigée | rejetée (version agrandie localement conservée).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from .imaging import LANCZOS, Analysis, _wrap_resize, bleed_rgb, final_alpha, local_upscale, to_f32
from .metrics import (box, color_stats, detail_energy, masked_blur, ms_ssim, palette_distance, resize_f32,
                      seam_alignment, seam_local, seam_ratios, seam_step)
from .validate import evaluate

BOX = Image.Resampling.BOX
QC_VERSION = 3          # à incrémenter quand les mesures changent : les résultats en cache sont recontrôlés

DECISIONS = ("validated", "corrected", "rejected", "local", "measured", "error")
LABELS = {"validated": "validée", "corrected": "corrigée", "rejected": "rejetée", "local": "locale",
          "measured": "mesurée (choix manuel)", "error": "erreur"}


@dataclass
class QCItem:
    tid: str
    name: str
    original: Image.Image                  # RGBA niveau 0 de la texture du jeu
    an: Analysis
    candidate: Image.Image | None          # résultat à contrôler (ComfyUI finalisé, UI, local)
    source: str                            # ai | ui | local | fallback
    k: int                                 # facteur d'agrandissement cible
    fallback: Callable[[], Image.Image] | None = None
    reviewed: bool = False                 # choix manuel en galerie : mesurer sans modifier
    data: bool = False                     # normal map, masque, carte de données : pas de mesure couleur


@dataclass
class QCResult:
    image: Image.Image
    decision: str
    source: str
    reasons: list[str] = field(default_factory=list)       # échecs constatés (avant correctifs)
    fixes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return LABELS.get(self.decision, self.decision)


# ------------------------------------------------------------------ préparation

def to_target(img: Image.Image, original: Image.Image, an: Analysis, k: int) -> tuple[Image.Image, list[str]]:
    """Ramène une image à la taille cible exacte (w·k, h·k) et réinjecte l'alpha d'origine."""
    notes = []
    W, H = original.width * k, original.height * k
    rgb = img.convert("RGB")
    if rgb.size != (W, H):
        ar_c, ar_o = rgb.width / rgb.height, original.width / original.height
        if abs(np.log(ar_c / ar_o)) > 0.02:
            notes.append(f"proportions {rgb.width}x{rgb.height} ≠ original {original.width}x{original.height}")
        if rgb.width < W or rgb.height < H:
            notes.append(f"résolution {rgb.width}x{rgb.height} < cible {W}x{H} : agrandie")
        if an.tile_x or an.tile_y:
            arr = _wrap_resize(np.asarray(rgb), (W, H), an.tile_x, an.tile_y)
        else:
            arr = np.asarray(rgb.resize((W, H), LANCZOS))
    else:
        arr = np.asarray(rgb)
    alpha = (final_alpha(original, an, (W, H)) * 255 + 0.5).astype(np.uint8)
    out = Image.fromarray(np.concatenate([arr[..., :3], alpha[..., None]], axis=-1), "RGBA")
    if an.alpha_class != "none":
        # couleurs sous la transparence : étendues depuis les zones visibles (pas de liseré)
        bled = np.asarray(bleed_rgb(out))
        out = Image.fromarray(np.concatenate([bled[..., :3], alpha[..., None]], axis=-1), "RGBA")
    return out, notes


def _baseline(item: QCItem) -> np.ndarray:
    """Agrandissement Lanczos de l'original : référence « sans détail ajouté »."""
    W, H = item.original.width * item.k, item.original.height * item.k
    base = bleed_rgb(item.original) if item.an.alpha_class != "none" else item.original.convert("RGBA")
    return _wrap_resize(np.asarray(base)[..., :3], (W, H), item.an.tile_x, item.an.tile_y).astype(np.float32) / 255.0


# ------------------------------------------------------------------ mesures

def measure(item: QCItem, img: Image.Image, cfg: dict, baseline: np.ndarray | None = None,
            cache: dict | None = None) -> dict:
    """Toutes les mesures d'une image déjà à la taille cible (alpha réinjecté). `cache` garde
    les mesures de la référence (agrandissement simple) d'un appel à l'autre."""
    cache = {} if cache is None else cache
    c = cfg["quality"]
    an, o = item.an, item.original
    w, h = o.size
    ref = to_f32(bleed_rgb(o) if an.alpha_class != "none" else o.convert("RGBA"))
    mask = to_f32(o)[..., 3] > 0.5 if an.alpha_class != "none" else np.ones((h, w), bool)
    if mask.sum() < 16:
        mask = np.ones((h, w), bool)
    full = to_f32(img)
    down = resize_f32(full[..., :3], (w, h), BOX)

    m: dict = {"size": [img.width, img.height]}
    ev = evaluate(o, img, an, cfg)                      # mêmes contrôles que pendant la génération
    m["structure"] = ev["structure"]
    m["shift"] = ev["shift"]
    m["local"] = ev["local"]
    m["local_color"] = ev["local_color"]
    m["low_contrast"] = ev["low_contrast"]
    ms = ms_ssim(_luma(down), _luma(ref[..., :3]), mask, wrap_x=an.tile_x, wrap_y=an.tile_y)
    m["ssim"] = ms["combined"]
    m["ssim_scales"] = ms["scales"]
    if not item.data:
        m.update(color_stats(down, ref[..., :3], mask))
        m["palette"] = palette_distance(down, ref[..., :3], mask)

    # transparence : le masque final reproduit-il celui de l'original ?
    if an.alpha_class != "none":
        a_down = resize_f32(full[..., 3], (w, h), BOX)
        a_ref = to_f32(o)[..., 3]
        if an.alpha_class == "binary":
            # Comparaison hors du liseré d'un pixel autour des contours : sur une petite icône,
            # l'agrandissement lissé puis seuillé déplace légitimement les bords d'un demi-pixel ;
            # un écart à l'intérieur des formes, lui, trahirait une découpe perdue.
            A, B = a_down >= 0.5, a_ref >= 0.5
            edge = np.zeros_like(B)
            edge[:, 1:] |= B[:, 1:] != B[:, :-1]
            edge[:, :-1] |= B[:, 1:] != B[:, :-1]
            edge[1:, :] |= B[1:, :] != B[:-1, :]
            edge[:-1, :] |= B[1:, :] != B[:-1, :]
            keep = ~edge
            A, B = A & keep, B & keep
            m["alpha_iou"] = round(float((A & B).sum() / max(1, (A | B).sum())), 4)
        m["alpha_error"] = round(float(np.abs(a_down - a_ref).mean()), 4)

    # raccord : trois regards complémentaires par axe répétable
    #   full/low : cassure par rapport aux transitions courantes de la texture (pleine résolution, 4x4)
    #   local    : cassure par rapport aux colonnes voisines du bord (joint qui ne se rejoint pas)
    #   step     : dérive de couleur d'un bord à l'autre, comparée à l'original (ΔE)
    seams = {}
    base = baseline if baseline is not None else _baseline(item)
    if an.tile_x or an.tile_y:
        lim = float(c.get("max_seam_ratio", 1.8))
        lim_local = float(cfg["qc"].get("max_seam_local", 1.6))
        for axis, on, nm in ((1, an.tile_x, "x"), (0, an.tile_y, "y")):
            if not on:
                continue
            f_full, f_low = seam_ratios(full[..., :3], axis)
            f_loc = seam_local(full[..., :3], axis, 2 * item.k)
            if ("seam", axis) not in cache:
                cache[("seam", axis)] = (*seam_ratios(base, axis), seam_local(base, axis, 2 * item.k))
            r_full, r_low, r_loc = cache[("seam", axis)]
            shift, gain = seam_alignment(full[..., :3], axis, 4 * item.k)
            seams[nm] = {"full": round(f_full, 3), "low": round(f_low, 3), "local": round(f_loc, 3),
                         "step": round(seam_step(down, ref[..., :3], axis, mask), 2),
                         "align_shift": shift, "align_gain": round(gain, 3),
                         "limit_full": round(max(lim, r_full * 1.25), 3),
                         "limit_low": round(max(lim, r_low * 1.25), 3),
                         "limit_local": round(max(lim_local, r_loc * 1.3), 3)}
    m["seam"] = seams

    # détail : gain par rapport à un simple agrandissement, et niveau absolu de grain
    big_mask = full[..., 3] > 0.5 if an.alpha_class != "none" else None
    if "detail" not in cache:
        cache["detail"] = detail_energy(base, big_mask)
    e_gen, e_base = detail_energy(full[..., :3], big_mask), cache["detail"]
    m["detail_gain"] = round(e_gen / e_base, 2) if e_base > 1e-3 else None     # aplat : rapport sans objet
    m["grain"] = round(e_gen, 4)
    return m


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


def judge(m: dict, item: QCItem, cfg: dict) -> tuple[dict[str, list[str]], list[str]]:
    """Échecs par catégorie (structure, color, seam, alpha) et avertissements."""
    q, c = cfg["qc"], cfg["quality"]
    fails: dict[str, list[str]] = {"structure": [], "color": [], "seam_step": [], "seam": [], "seam_align": [],
                                   "alpha": []}
    warns: list[str] = []
    if not m.get("low_contrast"):
        if m["structure"] < float(c.get("min_structure", 0.65)):
            fails["structure"].append(f"structure {m['structure']:.2f} < {c.get('min_structure', 0.65)}")
        if m["shift"] > float(c.get("max_shift", 0.03)):
            fails["structure"].append(f"composition décalée ({m['shift']:.3f})")
        if m["local"] < float(c.get("min_local", 0.3)):
            fails["structure"].append(f"zone réinventée (corrélation locale {m['local']:.2f})")
        if m["ssim"] < float(q.get("min_ssim", 0.5)):
            fails["structure"].append(f"SSIM {m['ssim']:.2f} < {q.get('min_ssim', 0.5)}")
    if not item.data:
        if m["delta_e"] > float(q.get("max_delta_e", 6.0)):
            fails["color"].append(f"ΔE moyen {m['delta_e']:.1f} > {q.get('max_delta_e', 6.0)}")
        if m["delta_e_p95"] > float(q.get("max_delta_e_p95", 14.0)):
            fails["color"].append(f"zones recolorées (ΔE p95 {m['delta_e_p95']:.1f})")
        if m.get("hue_shift") is not None and abs(m["hue_shift"]) > float(q.get("max_hue_shift", 8.0)):
            fails["color"].append(f"teinte {m['hue_shift']:+.1f}°")
        cr = m.get("chroma_ratio")
        if cr is not None and not (float(q.get("min_chroma_ratio", 0.7)) <= cr <= float(q.get("max_chroma_ratio", 1.4))):
            fails["color"].append(f"saturation ×{cr:.2f}")
        if abs(m["lightness_shift"]) > float(q.get("max_lightness_shift", 6.0)):
            fails["color"].append(f"luminosité {m['lightness_shift']:+.1f} L*")
        if m["palette"] > float(q.get("max_palette_distance", 0.35)):
            fails["color"].append(f"palette changée ({m['palette']:.2f})")
        if m["local_color"] > float(c.get("max_local_color", 0.12)) * 255:
            fails["color"].append(f"couleur locale changée ({m['local_color']:.0f}/255)")
        if not 0.67 <= m["contrast_ratio"] <= 1.5:
            warns.append(f"contraste ×{m['contrast_ratio']:.2f}")
    max_step = float(q.get("max_seam_step", 4.0))
    for nm, s in m.get("seam", {}).items():
        if s["full"] > s["limit_full"] or s["low"] > s["limit_low"]:
            fails["seam"].append(f"raccord {nm} {max(s['full'], s['low']):.2f} > {s['limit_full']:.2f}")
        elif s["local"] > s["limit_local"]:
            fails["seam"].append(f"raccord {nm} : bords qui ne se rejoignent pas ({s['local']:.2f} > {s['limit_local']:.2f})")
        if abs(s.get("align_shift", 0)) >= max(2, item.k) and s.get("align_gain", 0) > float(q.get("max_seam_align_gain", 0.3)):
            fails["seam_align"].append(f"raccord {nm} : motifs décalés de {abs(s['align_shift'])} px au bord")
        if s["step"] > max_step:
            fails["seam_step"].append(f"raccord {nm} : dérive de couleur d'un bord à l'autre (ΔE {s['step']:.1f})")
    if "alpha_iou" in m and m["alpha_iou"] < float(q.get("min_alpha_iou", 0.97)):
        fails["alpha"].append(f"découpe alpha IoU {m['alpha_iou']:.3f}")
    elif "alpha_error" in m and item.an.alpha_class == "gradient" and m["alpha_error"] > float(q.get("max_alpha_error", 0.03)):
        fails["alpha"].append(f"alpha écart {m['alpha_error']:.3f}")
    if item.source == "ai":
        if m["detail_gain"] is None:
            pass
        elif m["detail_gain"] < float(q.get("min_detail_gain", 1.1)):
            warns.append(f"peu de détail ajouté (×{m['detail_gain']:.2f} vs agrandissement simple)")
        elif m["grain"] > float(q.get("max_grain", 0.11)) and m["detail_gain"] > 3:
            warns.append(f"grain très fort ({m['grain']:.3f}) : bruit possible")
    return {k: v for k, v in fails.items() if v}, warns


# ------------------------------------------------------------------ correctifs

def color_lock(img: Image.Image, item: QCItem, strength: float) -> Image.Image:
    """Verrou basses fréquences : on ajoute à l'image l'écart (original − image réduite),
    flouté puis agrandi. L'image réduite retrouve les couleurs et la répartition des valeurs
    de l'original ; le détail au-dessus de la résolution d'origine reste celui de l'IA.
    Le flou de l'écart (≈ 1,5 pixel d'origine) évite de réimprimer les blocs DXT de l'original."""
    an, o = item.an, item.original
    w, h = o.size
    full = to_f32(img)
    ref = to_f32(bleed_rgb(o) if an.alpha_class != "none" else o.convert("RGBA"))[..., :3]
    down = resize_f32(full[..., :3], (w, h), BOX)
    resid = ref - down
    if an.alpha_class != "none":
        m = to_f32(o)[..., 3] > 0.5
        resid = masked_blur(np.where(m[..., None], resid, 0), m, 1, 2)
    else:
        resid = box(resid, 1, 2)
    # sans bouclage : chaque bord n'est corrigé qu'avec ses propres valeurs, ce qui supprime
    # aussi une dérive d'un bord à l'autre (l'original, lui, se raccorde)
    up = resize_f32(resid, img.size, Image.Resampling.BICUBIC)
    out = full.copy()
    out[..., :3] = np.clip(full[..., :3] + strength * up, 0, 1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def fix_seam(img: Image.Image, item: QCItem, band: float, feather: float, baseline: np.ndarray) -> Image.Image:
    """Fondu du raccord sur chaque axe répétable, appliqué au seul DÉTAIL de l'IA.

    L'image est décomposée en G = B + R, B étant l'original agrandi en boucle (raccord parfait,
    joints et arêtes de l'original à leur place) et R le détail ajouté. Seul R est rendu
    continu au bord, ce qui laisse intact un joint de carrelage ou une arête qui tombe
    pile sur le raccord :
    - basses fréquences : l'écart de R entre les deux bords est réparti en rampe douce
      (bande `band` du côté) ;
    - hautes fréquences : fondu croisé miroir de R sur `feather` du côté."""
    arr = to_f32(img)
    resid = arr[..., :3] - baseline
    for axis, on in ((1, item.an.tile_x), (0, item.an.tile_y)):
        if on:
            r = resid if axis == 1 else resid.transpose(1, 0, 2)
            r = _fix_axis(r, band, feather, item.k)
            resid = r if axis == 1 else r.transpose(1, 0, 2)
    arr[..., :3] = np.clip(baseline + resid, 0, 1)
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGBA")


def _fix_axis(rgb: np.ndarray, band: float, feather: float, k: int) -> np.ndarray:
    H, W, _ = rgb.shape
    out = rgb.copy()
    m = max(2, k)                                   # ≈ un pixel d'origine de chaque côté
    d = out[:, :m].mean(axis=1) - out[:, -m:].mean(axis=1)            # (H, 3)
    r = max(1, H // 64)
    d = box(d[:, None, :], r, 2)[:, 0, :]          # seulement l'écart basse fréquence le long du bord
    b = max(4, round(W * band))
    ramp = 1.0 - _smoothstep((np.arange(b, dtype=np.float32) + 0.5) / b)  # 1 au bord -> 0
    out[:, :b] -= (d / 2)[:, None, :] * ramp[None, :, None]
    out[:, W - b:] += (d / 2)[:, None, :] * ramp[::-1][None, :, None]
    c = max(2, round(W * feather))
    left, right = out[:, :c].copy(), out[:, W - c:][:, ::-1].copy()   # right[:, i] = colonne W-1-i
    wgt = (0.5 * (1.0 - (np.arange(c, dtype=np.float32) + 0.5) / c))[None, :, None]
    out[:, :c] = left * (1 - wgt) + right * wgt
    out[:, W - c:] = (right * (1 - wgt) + left * wgt)[:, ::-1]
    return out


# ------------------------------------------------------------------ décision

def _flat(fails: dict[str, list[str]]) -> list[str]:
    return [r for v in fails.values() for r in v]


def run_qc(item: QCItem, cfg: dict) -> QCResult:
    q = cfg["qc"]
    auto_fix = bool(q.get("auto_fix", True))

    def fallback_image() -> tuple[Image.Image, list[str]]:
        if item.fallback is not None:
            try:
                return to_target(item.fallback(), item.original, item.an, item.k)
            except Exception as exc:                  # repli amélioré indisponible : Lanczos
                img, notes = to_target(local_upscale(item.original, item.k, item.an, cfg, use_esrgan=False),
                                       item.original, item.an, item.k)
                return img, notes + [f"repli : {exc}"]
        return to_target(local_upscale(item.original, item.k, item.an, cfg, use_esrgan=False),
                         item.original, item.an, item.k)

    baseline = _baseline(item)
    cache: dict = {}
    if item.candidate is None:
        img, notes = fallback_image()
        m = measure(item, img, cfg, baseline, cache)
        return QCResult(img, "local", "local", warnings=notes, before=m, after=m)

    img, notes = to_target(item.candidate, item.original, item.an, item.k)
    before = measure(item, img, cfg, baseline, cache)
    fails, warns = judge(before, item, cfg)
    reasons = _flat(fails)
    warns = notes + warns
    res = QCResult(img, "validated", item.source, reasons=reasons, warnings=warns, before=before, after=before)

    if item.reviewed:
        # choix fait à la main dans la galerie : mesuré, jamais modifié
        res.decision = "measured"
        if fails:
            res.warnings.append("le QC aurait refusé : " + "; ".join(reasons))
        return res
    if item.source in ("local", "fallback"):
        # déjà une version locale : rien de mieux à proposer, on signale seulement
        res.decision = "local"
        res.warnings += [f"contrôle : {r}" for r in reasons]
        return res

    # structure trahie, motifs qui ne se rejoignent pas au raccord : pas de correctif honnête
    if not fails.get("structure") and not fails.get("alpha") and not fails.get("seam_align"):
        cur, cur_fails = img, fails
        strength_always = float(q.get("color_lock_always", 0.0))
        need_lock = bool(cur_fails.get("color") or cur_fails.get("seam_step"))
        if auto_fix and (need_lock or strength_always > 0):
            s = float(q.get("color_lock", 1.0)) if need_lock else strength_always
            cur = color_lock(cur, item, s)
            res.fixes.append(f"verrou couleur ×{s:.2f}" if need_lock else f"verrou couleur léger ×{s:.2f}")
        if auto_fix and cur_fails.get("seam"):
            cur = fix_seam(cur, item, float(q.get("seam_band", 0.06)), float(q.get("seam_feather", 0.015)), baseline)
            res.fixes.append("fondu du raccord")
        if res.fixes:
            after = measure(item, cur, cfg, baseline, cache)
            after_fails, after_warns = judge(after, item, cfg)
            res.after = after
            if not after_fails:
                res.image, res.decision = cur, ("corrected" if fails else "validated")
                res.warnings = notes + after_warns
                return res
            fails = after_fails
            res.reasons = reasons + [f"après correctif : {r}" for r in _flat(after_fails)]
        elif not fails:
            return res

    # rejet : version agrandie localement
    fb, fb_notes = fallback_image()
    res.image, res.decision = fb, "rejected"
    res.after = measure(item, fb, cfg, baseline, cache)
    res.warnings += fb_notes
    return res


def summarize(m: dict) -> str:
    """Résumé court des mesures pour la console."""
    parts = [f"SSIM {m.get('ssim', 0):.2f}"]
    if "delta_e" in m:
        parts.append(f"ΔE {m['delta_e']:.1f}")
    seams = m.get("seam") or {}
    if seams:
        parts.append("raccord " + "/".join(f"{max(v['full'], v['local']):.2f}" for v in seams.values()))
    if m.get("detail_gain") is not None:
        parts.append(f"détail ×{m['detail_gain']:.2f}")
    return ", ".join(parts)


__all__ = ["QCItem", "QCResult", "run_qc", "measure", "judge", "color_lock", "fix_seam", "to_target",
           "summarize", "DECISIONS", "LABELS"]
