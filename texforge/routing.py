"""Routage des textures vers le bon pipeline.

    ui       graphismes plats : HUD, polices, icônes, cartes, panneaux à texte
             -> agrandissement local rapide (Real-ESRGAN anime / Waifu2x), sans IA générative
    comfyui  matériaux complexes : pierre, métal, bois, tissu, végétation…
             -> file d'attente ComfyUI (img2img + ControlNet Tile)
    local    données techniques : normal maps, cartes de rugosité/spéculaire, dégradés
             -> agrandissement Lanczos
    keep     minuscules (tables de couleurs) -> gardées telles quelles

Signaux, du plus fiable au moins fiable :
  1. règles [[rules]] de la configuration (toujours prioritaires) ;
  2. mots du nom de fichier (inutiles sur les dumps Special K, nommés par empreinte) ;
  3. mesures sur l'image : nombre de couleurs, part d'aplats, netteté des bords,
     transparence, feuille de glyphes, raccord, énergie de texture, teintes.
Les seuils sont réglables dans [routing] ; `texforge route --report` permet de les vérifier.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .imaging import Analysis, luminance, to_f32

# --------------------------------------------------------------------------- vocabulaire des noms

UI_STRONG = {"hud", "gui", "ui", "menu", "button", "btn", "icon", "icons", "font", "fonts", "glyph", "glyphs",
             "cursor", "crosshair", "reticle", "minimap", "radar", "inventory", "hotbar", "logo", "healthbar",
             "compass", "subtitle", "subtitles"}
UI_WEAK = {"map", "text", "title", "splash", "loading", "loadscreen", "panel", "frame", "bar", "sign", "label"}
DATA_TOKENS = {"normal", "normals", "nrm", "norm", "bump", "height", "disp", "displacement", "spec", "specular",
               "gloss", "rough", "roughness", "metallic", "metalness", "ao", "occlusion", "mask", "lightmap", "lm",
               "shadow", "cube", "cubemap", "env", "reflection", "lut", "noise", "flow"}
MATERIAL_TOKENS = {
    "stone": {"stone", "rock", "rocks", "cliff", "concrete", "plaster", "marble", "gravel", "cave", "ground",
              "dirt", "sand", "mud", "terrain", "asphalt"},
    "masonry": {"brick", "bricks", "tile", "tiles", "cobble", "cobblestone", "pavement", "wall", "walls", "floor"},
    "metal": {"metal", "steel", "iron", "rust", "rusty", "chrome", "copper", "brass", "pipe", "grate", "panel"},
    "wood": {"wood", "wooden", "plank", "planks", "bark", "tree", "trunk", "log", "crate", "board"},
    "fabric": {"cloth", "fabric", "leather", "carpet", "rope", "canvas", "curtain", "flag", "wool"},
    "vegetation": {"leaf", "leaves", "grass", "foliage", "plant", "plants", "bush", "moss", "ivy", "fern", "vine"},
    "skin": {"skin", "face", "head", "body", "hand", "hands", "arm"},
}


def name_tokens(name: str) -> list[str]:
    """« HUD_Health01 » -> ['hud', 'health', '01'] ; découpe aussi le camelCase."""
    parts = re.split(r"[^A-Za-z0-9]+", name)
    out: list[str] = []
    for part in parts:
        out += [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", part)]
    return [t for t in out if t]


@dataclass
class NameHints:
    ui: float = 0.0
    data: bool = False
    material: str = ""
    words: list[str] = field(default_factory=list)


def name_hints(name: str) -> NameHints:
    tokens = name_tokens(name)
    hints = NameHints()
    for t in tokens:
        if t in UI_STRONG:
            hints.ui = max(hints.ui, 0.6)
            hints.words.append(t)
        elif t in UI_WEAK:
            hints.ui = max(hints.ui, 0.3)
            hints.words.append(t)
        if t in DATA_TOKENS:
            hints.data = True
            hints.words.append(t)
        for mat, words in MATERIAL_TOKENS.items():
            if t in words and not hints.material:
                hints.material = mat
                hints.words.append(t)
    return hints


# --------------------------------------------------------------------------- mesures sur l'image

def _small(img: Image.Image, max_side: int, resample: Image.Resampling) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), resample)


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    delta = mx - mn
    sat = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.zeros_like(mx)
    d = np.maximum(delta, 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6, hue)
    hue = np.where(mx == g, (b - r) / d + 2, hue)
    hue = np.where(mx == b, (r - g) / d + 4, hue)
    hue = np.where(delta < 1e-6, 0.0, hue) * 60.0
    return hue, sat, mx


def _components(mask: np.ndarray) -> list[int]:
    """Tailles des composantes connexes (4-voisinage) d'un masque binaire."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue = deque([(y0, x0)])
        n = 0
        while queue:
            y, x = queue.popleft()
            n += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        sizes.append(n)
    return sizes


@dataclass
class Features:
    colors: int               # couleurs distinctes (5 bits/canal) sur les pixels opaques
    flat: float               # part de pixels sans variation (aplats)
    hard_edges: float         # part des bords francs parmi les bords
    energy: float             # énergie de texture (écart local moyen de luminance)
    transparent: float        # part de pixels transparents
    sat: float                # saturation moyenne
    green: float              # part de pixels verts saturés
    brown: float              # part de pixels bruns
    skin: float               # part de pixels couleur peau
    axis_edges: float         # part des bords alignés sur les axes (briques, panneaux)
    coherence: float          # orientation dominante (fibres du bois, écorce)
    periodic: float           # pic de périodicité (trame de tissu)
    glyphs: int               # composantes détachées sur fond transparent
    glyph_spread: float       # dispersion des tailles de ces composantes
    mono: float               # pixels opaques quasi monochromes

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def measure(img: Image.Image) -> Features:
    rgba = img.convert("RGBA")
    # couleurs et transparence : sans mélange (NEAREST) ; bords et énergie : moyenne (BOX)
    exact = to_f32(_small(rgba, 128, Image.Resampling.NEAREST))
    soft = to_f32(_small(rgba, 128, Image.Resampling.BOX))
    alpha = exact[..., 3]
    opaque = alpha > 0.5
    n_op = int(opaque.sum())
    transparent = float(1.0 - opaque.mean())
    rgb = exact[..., :3]
    if n_op:
        q = np.round(rgb[opaque] * 31).astype(np.int32)
        colors = int(np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]).size)
    else:
        colors = 0

    lum = luminance(soft[..., :3])
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, :-1] = lum[:, 1:] - lum[:, :-1]
    gy[:-1, :] = lum[1:, :] - lum[:-1, :]
    mag = np.hypot(gx, gy)
    m_soft = soft[..., 3] > 0.5
    if not m_soft.any():
        m_soft = np.ones_like(lum, bool)
    g = mag[m_soft]
    flat = float((g < 0.012).mean())        # tolère le bruit de compression DXT (~3/255)
    edges = g > 0.03
    hard_edges = float((g > 0.25).sum() / max(1, edges.sum()))
    # alignement des bords : gradient presque horizontal ou vertical
    gxm, gym = np.abs(gx)[m_soft], np.abs(gy)[m_soft]
    sel = edges
    axis_edges = float(((gxm[sel] > 3 * gym[sel]) | (gym[sel] > 3 * gxm[sel])).mean()) if sel.any() else 0.0
    # cohérence d'orientation (tenseur de structure global)
    jxx, jyy, jxy = float((gx * gx)[m_soft].mean()), float((gy * gy)[m_soft].mean()), float((gx * gy)[m_soft].mean())
    tr = jxx + jyy
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tr) if tr > 1e-9 else 0.0
    # énergie : écart-type local en blocs de 4x4
    h4, w4 = (lum.shape[0] // 4) * 4, (lum.shape[1] // 4) * 4
    if h4 and w4:
        blocks = lum[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).std(axis=(1, 3))
        energy = float(blocks.mean())
    else:
        energy = float(lum.std())
    # périodicité : pic hors composante continue du spectre
    periodic = 0.0
    if min(lum.shape) >= 16:
        spec = np.abs(np.fft.fft2(lum - lum.mean()))
        spec[0, 0] = 0
        flat_spec = spec.ravel()
        if flat_spec.mean() > 1e-9:
            periodic = float(flat_spec.max() / flat_spec.mean())
    hue, sat_a, val = _hsv(rgb)
    if n_op:
        hh, ss, vv = hue[opaque], sat_a[opaque], val[opaque]
        sat = float(ss.mean())
        green = float(((hh >= 65) & (hh <= 170) & (ss > 0.2) & (vv > 0.12)).mean())
        brown = float(((hh >= 15) & (hh <= 45) & (ss > 0.2) & (ss < 0.75) & (vv > 0.12) & (vv < 0.75)).mean())
        skin = float(((hh >= 5) & (hh <= 40) & (ss > 0.15) & (ss < 0.6) & (vv > 0.35)).mean())
        mono = float((ss < 0.08).mean())
    else:
        sat = green = brown = skin = mono = 0.0

    glyphs, glyph_spread = 0, 0.0
    if transparent > 0.3 and n_op:
        mask = to_f32(_small(rgba, 256, Image.Resampling.NEAREST))[..., 3] > 0.5
        sizes = [s for s in _components(mask) if s >= 3]
        glyphs = len(sizes)
        if glyphs >= 4:
            arr = np.array(sorted(sizes), dtype=np.float64)
            core = arr[int(len(arr) * 0.1): max(int(len(arr) * 0.9), int(len(arr) * 0.1) + 1)]
            glyph_spread = float(core.std() / max(core.mean(), 1e-9))
    return Features(colors, flat, hard_edges, energy, transparent, sat, green, brown, skin, axis_edges,
                    coherence, periodic, glyphs, glyph_spread, mono)


# --------------------------------------------------------------------------- décision

@dataclass
class RouteDecision:
    route: str            # ui | comfyui | local | keep
    category: str         # ui | material | data | tiny
    ui_kind: str = ""     # font | icon | hud | map
    material: str = ""    # stone | masonry | metal | wood | fabric | vegetation | skin | generic
    ui_score: float = 0.0
    reason: str = ""
    features: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return {"ui": "ui", "comfyui": "ai", "local": "upscale", "keep": "keep"}[self.route]


def _ui_score(f: Features, an: Analysis, hints: NameHints, rc: dict) -> tuple[float, list[str], str]:
    score, why = 0.0, []
    if hints.ui:
        score += hints.ui
        why.append(f"nom « {'/'.join(w for w in hints.words if w in UI_STRONG | UI_WEAK)} »")
    if hints.material:
        score -= 0.5
    # Sans aucune arête franche (dégradé de ciel, peau lisse), aplats et palette réduite ne
    # prouvent rien : un graphisme a toujours des formes
    shapes = f.hard_edges >= 0.05 or f.glyphs >= 4
    if not shapes:
        pass
    elif f.flat > float(rc["flat_strong"]):
        score += 0.3
        why.append(f"{f.flat:.0%} d'aplats")
    elif f.flat > float(rc["flat_weak"]):
        score += 0.15
        why.append(f"{f.flat:.0%} d'aplats")
    if not shapes:
        score -= 0.2
    elif f.colors <= int(rc["few_colors"]):
        score += 0.3
        why.append(f"{f.colors} couleurs")
    elif f.colors <= int(rc["some_colors"]):
        score += 0.15
        why.append(f"{f.colors} couleurs")
    elif f.colors > int(rc["many_colors"]):
        score -= 0.2
    if f.hard_edges > 0.35 and f.flat > 0.3:
        score += 0.15
        why.append("bords francs")
    font = (f.glyphs >= int(rc["font_min_glyphs"]) and f.mono > 0.6 and f.transparent > 0.4
            and f.glyph_spread < float(rc["font_max_spread"]))
    if font:
        score += 0.5
        why.append(f"feuille de {f.glyphs} glyphes")
    # Un bord transparent ou uni « se raccorde » trivialement : le raccord ne pèse que sur
    # une image opaque et texturée
    textured = f.transparent < 0.2 and f.flat < float(rc["flat_strong"])
    if textured and an.tile_x and an.tile_y:
        score -= 0.35
    elif textured and (an.tile_x or an.tile_y):
        score -= 0.1
    # Les arêtes franches d'un graphisme gonflent aussi l'énergie : pénalité seulement sans aplats
    if f.energy > float(rc["energy_material"]) and f.flat < float(rc["flat_weak"]):
        score -= 0.25
    if f.green > 0.35 and an.alpha_class == "binary":
        score -= 0.3          # feuillage découpé, pas une icône
    kind = ""
    if font:
        kind = "font"
    elif "minimap" in hints.words or "map" in hints.words or "radar" in hints.words:
        kind = "map"
    elif f.transparent > 0.25 and max(an.width, an.height) <= 256:
        kind = "icon"
    else:
        kind = "hud"
    return score, why, kind


def _material(f: Features, an: Analysis, hints: NameHints) -> tuple[str, str]:
    if hints.material:
        word = next((w for w in hints.words if w in MATERIAL_TOKENS[hints.material]), hints.material)
        return hints.material, f"nom « {word} »"
    if f.green > 0.35:
        return "vegetation", f"{f.green:.0%} de vert"
    if f.skin > 0.5 and f.energy < 0.03:
        return "skin", f"{f.skin:.0%} de teintes peau, surface lisse"
    if f.axis_edges > 0.45 and (an.tile_x or an.tile_y) and f.hard_edges > 0.1:
        return "masonry", f"{f.axis_edges:.0%} de bords alignés"
    if f.coherence > 0.45 and f.brown > 0.25:
        return "wood", f"fibres orientées ({f.coherence:.2f}), {f.brown:.0%} de brun"
    if f.periodic > 60 and f.energy > 0.02:
        return "fabric", f"trame périodique ({f.periodic:.0f})"
    if f.sat < 0.2:
        return "stone", f"saturation faible ({f.sat:.2f})"
    return "generic", "aucun indice dominant"


def decide(img: Image.Image, an: Analysis, name: str, cfg: dict) -> RouteDecision:
    rc = cfg["routing"]
    hints = name_hints(name)
    if an.mode == "keep":
        return RouteDecision("keep", "tiny", reason=an.reason)
    f = measure(img)
    fd = f.to_dict()
    if an.normal_map or hints.data:
        why = "carte de relief probable (normal map)" if an.normal_map else f"donnée technique (nom « {'/'.join(w for w in hints.words if w in DATA_TOKENS)} »)"
        return RouteDecision("local", "data", reason=why, features=fd)
    score, why, kind = _ui_score(f, an, hints, rc)
    # Un nom explicite (hud, font, icon…) l'emporte : les archives extraites nomment leurs fichiers
    if hints.ui >= 0.6 and not hints.material:
        score = max(score, float(rc["ui_threshold"]))
    if score >= float(rc["ui_threshold"]):
        return RouteDecision("ui", "ui", ui_kind=kind, ui_score=round(score, 3),
                             reason=f"graphisme plat ({kind}) : " + ", ".join(why), features=fd)
    if an.mode == "upscale":
        return RouteDecision("local", "data", ui_score=round(score, 3), reason=an.reason, features=fd)
    material, mwhy = _material(f, an, hints)
    if material == "skin" and str(rc.get("skin_route", "local")) == "local":
        return RouteDecision("local", "material", material="skin", ui_score=round(score, 3),
                             reason=f"peau ou visage ({mwhy}) : agrandissement sans IA générative", features=fd)
    return RouteDecision("comfyui", "material", material=material, ui_score=round(score, 3),
                         reason=f"matériau {material} ({mwhy})", features=fd)
