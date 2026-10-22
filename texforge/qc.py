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

from .i18n import N_, tr
from .imaging import (JOINT_MATCH, LANCZOS, Analysis, _wrap_resize, bleed_rgb, final_alpha, local_upscale,
                      opaque_rgba, seam_joint, to_f32)
from .metrics import (box, color_stats, detail_energy, masked_blur, ms_ssim, palette_distance, resize_f32,
                      seam_alignment, seam_local, seam_ratios, seam_step)
from .validate import evaluate

BOX = Image.Resampling.BOX
QC_VERSION = 4          # à incrémenter quand les mesures changent : les résultats en cache sont recontrôlés

DECISIONS = ("validated", "corrected", "rejected", "local", "measured", "error")
LABELS = {"validated": N_("validée"), "corrected": N_("corrigée"), "rejected": N_("rejetée"), "local": N_("locale"),
          "measured": N_("mesurée (choix manuel)"), "error": N_("erreur")}          # tr(LABELS[…]) à l'affichage


def label_of(decision: str) -> str:
    return tr(LABELS[decision]) if decision in LABELS else decision


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
        return label_of(self.decision)


# ------------------------------------------------------------------ préparation

def to_target(img: Image.Image, original: Image.Image, an: Analysis, k: int) -> tuple[Image.Image, list[str]]:
    """Ramène une image à la taille cible exacte (w·k, h·k) et réinjecte l'alpha d'origine."""
    notes = []
    W, H = original.width * k, original.height * k
    rgb = img.convert("RGB")
    if rgb.size != (W, H):
        ar_c, ar_o = rgb.width / rgb.height, original.width / original.height
        if abs(np.log(ar_c / ar_o)) > 0.02:
            notes.append(tr("proportions {w}x{h} ≠ original {ow}x{oh}", w=rgb.width, h=rgb.height,
                            ow=original.width, oh=original.height))
        if rgb.width < W or rgb.height < H:
            notes.append(tr("résolution {w}x{h} < cible {tw}x{th} : agrandie", w=rgb.width, h=rgb.height, tw=W, th=H))
        if an.tile_x or an.tile_y:
            arr = _wrap_resize(np.asarray(rgb), (W, H), an.tile_x, an.tile_y)
        else:
            arr = np.asarray(rgb.resize((W, H), LANCZOS))
    else:
        arr = np.asarray(rgb)
    alpha = (final_alpha(original, an, (W, H)) * 255 + 0.5).astype(np.uint8)
    out = Image.fromarray(np.concatenate([arr[..., :3], alpha[..., None]], axis=-1), "RGBA")
    if an.cutout:
        # couleurs sous la transparence : étendues depuis les zones visibles (pas de liseré)
        bled = np.asarray(bleed_rgb(out))
        out = Image.fromarray(np.concatenate([bled[..., :3], alpha[..., None]], axis=-1), "RGBA")
    return out, notes


def _baseline(item: QCItem) -> np.ndarray:
    """Agrandissement Lanczos de l'original : référence « sans détail ajouté »."""
    W, H = item.original.width * item.k, item.original.height * item.k
    base = opaque_rgba(item.original, item.an)
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
    ref = to_f32(opaque_rgba(o, an))
    mask = to_f32(o)[..., 3] > 0.5 if an.cutout else np.ones((h, w), bool)
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
    # plus « joint » : à la résolution d'origine, le saut au bord reproduit-il une transition
    # interne du rendu, l'original ayant lui aussi un joint au bord ? (carrelage posé pile au bord)
    seams = {}
    base = baseline if baseline is not None else _baseline(item)
    if an.tile_x or an.tile_y:
        lim = float(c.get("max_seam_ratio", 1.8))
        lim_local = float(cfg["qc"].get("max_seam_local", 1.6))
        q_joint = float(cfg["qc"].get("seam_joint_match", JOINT_MATCH))
        for axis, on, nm in ((1, an.tile_x, "x"), (0, an.tile_y, "y")):
            if not on:
                continue
            f_full, f_low = seam_ratios(full[..., :3], axis)
            f_loc = seam_local(full[..., :3], axis, 2 * item.k)
            if ("seam", axis) not in cache:
                cache[("seam", axis)] = (*seam_ratios(base, axis), seam_local(base, axis, 2 * item.k))
            r_full, r_low, r_loc = cache[("seam", axis)]
            shift, gain = seam_alignment(full[..., :3], axis, 4 * item.k)
            if ("joint", axis) not in cache:
                cache[("joint", axis)] = seam_joint(ref[..., :3], axis)
            seams[nm] = {"full": round(f_full, 3), "low": round(f_low, 3), "local": round(f_loc, 3),
                         "joint": round(seam_joint(down, axis), 3), "joint_ref": round(cache[("joint", axis)], 3),
                         # le rendu doit reproduire le joint au moins aussi bien que l'original (marge 25 %)
                         "limit_joint": round(min(float(q_joint), max(0.12, 1.25 * cache[("joint", axis)])), 3),
                         "step": round(seam_step(down, ref[..., :3], axis, mask), 2),
                         "align_shift": shift, "align_gain": round(gain, 3),
                         "limit_full": round(max(lim, r_full * 1.25), 3),
                         "limit_low": round(max(lim, r_low * 1.25), 3),
                         "limit_local": round(max(lim_local, r_loc * 1.3), 3)}
    m["seam"] = seams

    # détail : gain par rapport à un simple agrandissement, et niveau absolu de grain
    big_mask = full[..., 3] > 0.5 if an.cutout else None
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
            fails["structure"].append(tr("structure {v:.2f} < {lim}", v=m["structure"], lim=c.get("min_structure", 0.65)))
        if m["shift"] > float(c.get("max_shift", 0.03)):
            fails["structure"].append(tr("composition décalée ({v:.3f})", v=m["shift"]))
        if m["local"] < float(c.get("min_local", 0.3)):
            fails["structure"].append(tr("zone réinventée (corrélation locale {v:.2f})", v=m["local"]))
        if m["ssim"] < float(q.get("min_ssim", 0.5)):
            fails["structure"].append(tr("SSIM {v:.2f} < {lim}", v=m["ssim"], lim=q.get("min_ssim", 0.5)))
    if not item.data:
        if m["delta_e"] > float(q.get("max_delta_e", 6.0)):
            fails["color"].append(tr("ΔE moyen {v:.1f} > {lim}", v=m["delta_e"], lim=q.get("max_delta_e", 6.0)))
        if m["delta_e_p95"] > float(q.get("max_delta_e_p95", 14.0)):
            fails["color"].append(tr("zones recolorées (ΔE p95 {v:.1f})", v=m["delta_e_p95"]))
        if m.get("hue_shift") is not None and abs(m["hue_shift"]) > float(q.get("max_hue_shift", 8.0)):
            fails["color"].append(tr("teinte {v:+.1f}°", v=m["hue_shift"]))
        cr = m.get("chroma_ratio")
        if cr is not None and not (float(q.get("min_chroma_ratio", 0.7)) <= cr <= float(q.get("max_chroma_ratio", 1.4))):
            fails["color"].append(tr("saturation ×{v:.2f}", v=cr))
        if abs(m["lightness_shift"]) > float(q.get("max_lightness_shift", 6.0)):
            fails["color"].append(tr("luminosité {v:+.1f} L*", v=m["lightness_shift"]))
        if m["palette"] > float(q.get("max_palette_distance", 0.35)):
            fails["color"].append(tr("palette changée ({v:.2f})", v=m["palette"]))
        if m["local_color"] > float(c.get("max_local_color", 0.12)) * 255:
            fails["color"].append(tr("couleur locale changée ({v:.0f}/255)", v=m["local_color"]))
        if not 0.67 <= m["contrast_ratio"] <= 1.5:
            warns.append(tr("contraste ×{v:.2f}", v=m["contrast_ratio"]))
    max_step = float(q.get("max_seam_step", 4.0))
    joint_match = float(q.get("seam_joint_match", JOINT_MATCH))
    for nm, s in m.get("seam", {}).items():
        # Joint au bord, dans l'original comme dans le rendu : l'arête est voulue. Un rendu plus
        # net ou moins bruité que l'agrandissement de référence la fait ressortir davantage
        # par rapport aux transitions internes, sans qu'il y ait de cassure.
        joint = (s.get("joint_ref", 1.0) < joint_match
                 and s.get("joint", 1.0) <= s.get("limit_joint", 0.0))
        if joint:
            pass
        elif s["full"] > s["limit_full"]:
            fails["seam"].append(tr("raccord {axis} {v:.2f} > {lim:.2f}", axis=nm, v=s["full"], lim=s["limit_full"]))
        elif s["low"] > s["limit_low"]:
            fails["seam"].append(tr("raccord {axis} {v:.2f} > {lim:.2f}", axis=nm, v=s["low"], lim=s["limit_low"]))
        elif s["local"] > s["limit_local"]:
            fails["seam"].append(tr("raccord {axis} : bords qui ne se rejoignent pas ({v:.2f} > {lim:.2f})", axis=nm,
                                    v=s["local"], lim=s["limit_local"]))
        if abs(s.get("align_shift", 0)) >= max(2, item.k) and s.get("align_gain", 0) > float(q.get("max_seam_align_gain", 0.3)):
            fails["seam_align"].append(tr("raccord {axis} : motifs décalés de {px} px au bord", axis=nm,
                                          px=abs(s["align_shift"])))
        if s["step"] > max_step:
            fails["seam_step"].append(tr("raccord {axis} : dérive de couleur d'un bord à l'autre (ΔE {v:.1f})",
                                         axis=nm, v=s["step"]))
    if "alpha_iou" in m and m["alpha_iou"] < float(q.get("min_alpha_iou", 0.97)):
        fails["alpha"].append(tr("découpe alpha IoU {v:.3f}", v=m["alpha_iou"]))
    elif "alpha_error" in m and item.an.alpha_class == "gradient" and m["alpha_error"] > float(q.get("max_alpha_error", 0.03)):
        fails["alpha"].append(tr("alpha écart {v:.3f}", v=m["alpha_error"]))
    if item.source == "ai":
        if m["detail_gain"] is None:
            pass
        elif m["detail_gain"] < float(q.get("min_detail_gain", 1.1)):
            warns.append(tr("peu de détail ajouté (×{v:.2f} vs agrandissement simple)", v=m["detail_gain"]))
        elif m["grain"] > float(q.get("max_grain", 0.11)) and m["detail_gain"] > 3:
            warns.append(tr("grain très fort ({v:.3f}) : bruit possible", v=m["grain"]))
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
    ref = to_f32(opaque_rgba(o, an))[..., :3]
    down = resize_f32(full[..., :3], (w, h), BOX)
    resid = ref - down
    if an.cutout:
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
                return img, notes + [tr("repli : {err}", err=exc)]
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
            res.warnings.append(tr("le QC aurait refusé : {reasons}", reasons="; ".join(reasons)))
        return res
    if item.source in ("local", "fallback"):
        # déjà une version locale : rien de mieux à proposer, on signale seulement
        res.decision = "local"
        res.warnings += [tr("contrôle : {reason}", reason=r) for r in reasons]
        return res

    # structure trahie, motifs qui ne se rejoignent pas au raccord : pas de correctif honnête
    if not fails.get("structure") and not fails.get("alpha") and not fails.get("seam_align"):
        cur, cur_fails = img, fails
        strength_always = float(q.get("color_lock_always", 0.0))
        need_lock = bool(cur_fails.get("color") or cur_fails.get("seam_step"))
        if auto_fix and (need_lock or strength_always > 0):
            s = float(q.get("color_lock", 1.0)) if need_lock else strength_always
            cur = color_lock(cur, item, s)
            res.fixes.append(tr("verrou couleur ×{v:.2f}", v=s) if need_lock else tr("verrou couleur léger ×{v:.2f}", v=s))
        if auto_fix and cur_fails.get("seam"):
            cur = fix_seam(cur, item, float(q.get("seam_band", 0.06)), float(q.get("seam_feather", 0.015)), baseline)
            res.fixes.append(tr("fondu du raccord"))
        if res.fixes:
            after = measure(item, cur, cfg, baseline, cache)
            after_fails, after_warns = judge(after, item, cfg)
            res.after = after
            if not after_fails:
                res.image, res.decision = cur, ("corrected" if fails else "validated")
                res.warnings = notes + after_warns
                return res
            fails = after_fails
            res.reasons = reasons + [tr("après correctif : {reason}", reason=r) for r in _flat(after_fails)]
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
        parts.append(tr("raccord {values}", values="/".join(f"{max(v['full'], v['local']):.2f}" for v in seams.values())))
    if m.get("detail_gain") is not None:
        parts.append(tr("détail ×{v:.2f}", v=m["detail_gain"]))
    return ", ".join(parts)


__all__ = ["QCItem", "QCResult", "run_qc", "measure", "judge", "color_lock", "fix_seam", "to_target",
           "summarize", "DECISIONS", "LABELS", "label_of"]
