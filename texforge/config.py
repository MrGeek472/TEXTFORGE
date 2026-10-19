"""Configuration : valeurs par défaut, lecture du fichier TOML, empreinte des
réglages qui influencent le résultat (pour savoir quand un résultat est périmé)."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore

DEFAULT_PROMPT = (
    "Image 1 is a texture map taken from an old video game. Re-render it as a high-resolution, "
    "{style} texture. STRICT RULES: keep exactly the same layout. Every shape, edge, line, region "
    "and color area must stay at exactly the same position, size and orientation. Do not move, add, "
    "remove, rotate or mirror anything. Keep the image flat and orthographic, filling the whole frame: "
    "no perspective, no camera, no cast shadows, no vignette, no border, no frame. Keep the same "
    "overall colors and brightness. Only improve surface detail and material realism{hint}.{refs}"
)

DEFAULT_NEGATIVE = (
    "perspective, 3d object render, frame, border, watermark, text, logo, vignette, cast shadows, "
    "blur, cartoon, new objects, shifted layout, rotated, mirrored"
)

DEFAULTS: dict = {
    "paths": {"inputs": [], "work": "./texforge_work"},
    "output": {"scale": 4, "max_side": 2048, "dds_format": "match", "mipmaps": "always"},
    "analysis": {
        "min_ai_size": 32, "copy_below": 8, "max_ai_aspect": 3.0, "wrap_fraction": 0.125,
        "tile_threshold": 1.5, "stretch_tolerance": 1.35, "smooth_threshold": 0.006,
    },
    "provider": {
        "name": "comfyui", "workers": 3, "max_attempts": 3, "max_requests": 50,
        "price_per_request_usd": 0.0, "base_seed": 1234, "poll_interval": 3.0,
        "timeout": 900, "input_long_side": 1024,
    },
    "higgsfield": {
        "model": "alibaba/qwen-image-3/edit",
        "image_field": "image_urls",
        "max_images": 3,
        "aspect_ratios": ["1:1", "2:3", "3:2", "3:4", "4:3", "7:9", "9:7", "9:16", "16:9", "21:9"],
        "supports_seed": True,
        "supports_negative": True,
        "style_references": [],
        "api_key": "",
        "extra_args": {"resolution": "1k", "prompt_extend": False, "enable_thinking": False},
    },
    "mock": {"shift": 0.0},
    "prompt": {"style": "photorealistic, highly detailed physically based", "template": DEFAULT_PROMPT,
               "negative": DEFAULT_NEGATIVE},
    "quality": {"min_structure": 0.65, "max_shift": 0.03, "min_local": 0.3, "max_local_color": 0.12,
                "max_seam_ratio": 1.8, "color_match": 0.6},
    "local": {"esrgan_exe": "", "esrgan_model": "realesrgan-x4plus"},
    "qc": {
        "enabled": True, "auto_fix": True, "workers": 4, "chunk": 32,
        # fidélité à la DA, mesurée à la résolution d'origine
        "min_ssim": 0.5, "max_delta_e": 6.0, "max_delta_e_p95": 14.0, "max_hue_shift": 8.0,
        "min_chroma_ratio": 0.7, "max_chroma_ratio": 1.4, "max_lightness_shift": 6.0,
        "max_palette_distance": 0.35,
        # détail (avertissements seulement)
        "min_detail_gain": 1.1, "max_grain": 0.11,
        # raccord (en plus de [quality] max_seam_ratio)
        "max_seam_local": 1.6, "max_seam_step": 4.0, "max_seam_align_gain": 0.3,
        # transparence
        "min_alpha_iou": 0.97, "max_alpha_error": 0.03, "alpha_ref": 0.5,
        # correctifs
        "color_lock": 1.0, "color_lock_always": 0.0, "seam_band": 0.06, "seam_feather": 0.015,
        # DDS
        "encoder": "auto", "texconv_exe": "", "texconv_args": [], "texconv_timeout": 600,
        "target_api": "d3d9", "allow_bc7": False, "dxt1_alpha": False, "mip_filter": "lanczos",
        "mip_gamma": "srgb", "min_psnr": 28.0, "verify": True, "memory_budget_mb": 0,
    },
    "comfyui": {
        "server": "http://127.0.0.1:8188",
        "workflow": "workflows/texforge_sdxl_tile_api.json",
        "workflow_edges": "workflows/texforge_sdxl_tile_canny_api.json",
        "checkpoint": "", "controlnet_model": "", "lora": "", "lora_strength": 0.8,
        "max_in_flight": 2, "timeout": 600,
        "batch_size": 12, "free_vram": "auto", "vram_threshold_gb": 1.5,
        "free_between_groups": True, "free_every": 100,
        "max_transient_retries": 2, "max_consecutive_failures": 8,
        "on_server_down": "wait", "server_wait": 120,
        "prompt_prefix": "high resolution game texture, ",
        "prompt_prefix_tiled": "seamless high resolution game texture, ",
        "prompt_suffix": ", fine surface micro detail, flat orthographic top-down view, even diffuse lighting",
        "negative": ("perspective, 3d render, object, frame, border, vignette, text, watermark, logo, cast shadows, "
                     "strong highlights, blurry, lowres, cartoon, new objects, faces"),
        "fallback_engines": ["realesrgan", "waifu2x", "lanczos"],
        "fallback_model": "realesrgan-x4plus",
    },
    "routing": {
        "enabled": True, "ui_threshold": 0.5, "flat_strong": 0.55, "flat_weak": 0.35,
        "few_colors": 24, "some_colors": 64, "many_colors": 200, "energy_material": 0.06,
        "font_min_glyphs": 20, "font_max_spread": 1.2, "skin_route": "local",
    },
    "ui_upscale": {
        "engines": ["realesrgan", "waifu2x", "lanczos"],
        "realesrgan_exe": "", "realesrgan_model": "realesr-animevideov3",
        "waifu2x_exe": "", "waifu2x_noise": 1, "waifu2x_model": "",
        "tile": 256, "gpu": "", "timeout": 120, "batch_size": 64,
    },
    "materials": {
        "stone": {"prompt": "weathered natural stone, rough rock surface, small pores and chips",
                  "denoise": 0.5, "cn_strength": 0.7, "edges": False},
        "masonry": {"prompt": "brick and stone masonry, worn mortar joints, chipped edges",
                    "denoise": 0.4, "cn_strength": 0.8, "edges": True},
        "metal": {"prompt": "worn painted metal, fine scratches, light rust",
                  "denoise": 0.4, "cn_strength": 0.8, "edges": False},
        "wood": {"prompt": "natural wood grain and bark, fibers and small cracks",
                 "denoise": 0.5, "cn_strength": 0.7, "edges": False},
        "fabric": {"prompt": "woven fabric and leather, visible threads, fine weave",
                   "denoise": 0.4, "cn_strength": 0.75, "edges": False},
        "vegetation": {"prompt": "leaves and foliage, leaf veins, natural plant detail",
                       "denoise": 0.35, "cn_strength": 0.85, "edges": False},
        "skin": {"prompt": "realistic skin, fine pores, natural tones",
                 "denoise": 0.25, "cn_strength": 0.9, "edges": False},
        "generic": {"prompt": "detailed physically based surface",
                    "denoise": 0.42, "cn_strength": 0.75, "edges": False},
    },
    "rules": [],
}


class ConfigError(Exception):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"configuration introuvable : {path} (lance d'abord : texforge init)")
    try:
        user = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"erreur de syntaxe dans {path} : {exc}") from exc
    cfg = _merge(DEFAULTS, user)
    base = path.parent.resolve()
    cfg["_base"] = str(base)
    work = Path(cfg["paths"]["work"])
    cfg["_work"] = str(work if work.is_absolute() else (base / work).resolve())
    inputs = []
    for p in cfg["paths"]["inputs"]:
        pp = Path(p)
        inputs.append(str(pp if pp.is_absolute() else (base / pp).resolve()))
    cfg["_inputs"] = inputs
    return cfg


def result_fingerprint(cfg: dict) -> str:
    """Réglages qui changent le résultat d'une génération IA."""
    relevant = {
        "scale": cfg["output"]["scale"], "max_side": cfg["output"]["max_side"],
        "provider": cfg["provider"]["name"],
        "model": cfg["higgsfield"]["model"], "extra": cfg["higgsfield"]["extra_args"],
        "refs": cfg["higgsfield"]["style_references"], "prompt": cfg["prompt"],
        "long_side": cfg["provider"]["input_long_side"], "wrap": cfg["analysis"]["wrap_fraction"],
        "mock": cfg.get("mock", {}),
        "comfyui": {k: cfg["comfyui"].get(k) for k in ("workflow", "workflow_edges", "prompt_prefix",
                                                     "prompt_prefix_tiled", "prompt_suffix", "negative",
                                                     "checkpoint", "controlnet_model", "lora", "lora_strength")},
        "materials": cfg.get("materials", {}),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


EXAMPLE_TOML = r"""# =====================================================================
# TEXFORGE — configuration
# Chemins Windows : utilise des apostrophes simples 'C:\...' (pas besoin de doubler les \)
# =====================================================================

[paths]
# Dossiers contenant les textures extraites (DDS, PNG, TGA, BMP, JPG). Sous-dossiers inclus.
# Exemple Special K :  'C:\Jeux\MonJeu\SK_Res\dump\textures'
inputs = ['./entree']
# Dossier de travail : cache, résultats, galerie de revue
work = './texforge_work'

[output]
scale = 4            # facteur d'agrandissement (puissance de 2 conservée si la texture en est une)
max_side = 2048      # côté maximal. Jeux 32 bits : 2048 ou moins, sinon risque de plantage mémoire
# "match" = même format que l'original (le jeu ou l'injecteur l'attend ; original PNG -> BGRA8 sans perte)
# "auto"  = DXT1 sans alpha, DXT5 avec alpha | "dxt1" | "dxt5" | "bgra8"
# "bc7"   = Direct3D 11 seulement (dgVoodoo2 + Special K) : exige [qc] target_api = "d3d11" et allow_bc7
dds_format = "match"
mipmaps = "always"   # "always" (chaîne complète, recommandé) | "match" (seulement si l'original en avait) | "never"

[analysis]
min_ai_size = 32     # en dessous : agrandissement local seulement (pas d'appel payant)
copy_below = 8       # en dessous : texture gardée telle quelle (tables de couleurs, dégradés)
max_ai_aspect = 3.0  # format plus allongé que 3:1 -> agrandissement local
tile_threshold = 1.5 # détection des textures répétables (plus haut = plus permissif)

[provider]
name = "comfyui"     # "comfyui" (local), "mock" (test gratuit) ou "higgsfield" (cloud, optionnel)
max_attempts = 3     # essais par texture (graines différentes) avant repli
base_seed = 1234

[comfyui]
server = "http://127.0.0.1:8188"
# Chemins relatifs : cherchés à côté de ce fichier, puis dans le dossier workflows/ de TEXFORGE
workflow = "workflows/texforge_sdxl_tile_api.json"
workflow_edges = "workflows/texforge_sdxl_tile_canny_api.json"   # matériaux à joints (briques, carrelage)
checkpoint = ''            # vide : celui du workflow ; sinon nom exact côté ComfyUI (ex. 'juggernautXL_ragnarok.safetensors')
controlnet_model = ''      # vide : celui du workflow
lora = ''                  # LoRA insérée après le checkpoint (vide : aucune)
lora_strength = 0.8
max_in_flight = 2         # 2 = le GPU enchaîne pendant l'envoi de la texture suivante
timeout = 600             # secondes max par texture
batch_size = 12           # textures entre deux contrôles de VRAM
free_vram = "auto"        # "auto" (sous le seuil) | "always" | "never"
vram_threshold_gb = 1.5
free_between_groups = true  # libère la VRAM quand on change de workflow (Tile -> Tile+Canny)
free_every = 100          # libération forcée toutes les N textures (gros volumes)
max_transient_retries = 2 # OOM, délai, coupure : nouveaux essais après libération de la VRAM
max_consecutive_failures = 8
on_server_down = "wait"   # "wait" puis repli | "fallback" immédiat | "abort"
server_wait = 120         # secondes d'attente du redémarrage de ComfyUI
fallback_engines = ["realesrgan", "waifu2x", "lanczos"]   # repli d'une texture ratée
fallback_model = "realesrgan-x4plus"

[routing]
enabled = true
ui_threshold = 0.5   # score à partir duquel une texture part en pipeline UI (plus bas = plus d'UI)
skin_route = "local" # peau et visages : "local" (sans IA générative) ou "comfyui"

[ui_upscale]
# Moteurs essayés dans l'ordre ; lanczos (Python) sert toujours de dernier recours
engines = ["realesrgan", "waifu2x", "lanczos"]
realesrgan_exe = ''           # ex. 'C:\Outils\realesrgan-ncnn-vulkan\realesrgan-ncnn-vulkan.exe'
realesrgan_model = "realesr-animevideov3"   # léger et net pour l'UI ; x4plus pour les photos
waifu2x_exe = ''
waifu2x_noise = 1
tile = 256                    # taille de tuile ncnn : plus petit = moins de VRAM
batch_size = 64

# Réglages par matériau : prompt, denoise, force du ControlNet Tile, variante à contours
[materials.stone]
denoise = 0.5
cn_strength = 0.7
[materials.masonry]
denoise = 0.4
cn_strength = 0.8
edges = true

[prompt]
style = "photorealistic, highly detailed physically based"

[quality]
min_structure = 0.65 # fidélité minimale à la disposition d'origine (0 à 1) — à recalibrer après le 1er lot réel
max_shift = 0.03     # décalage global maximal (fraction de la taille)
min_local = 0.3      # corrélation minimale du pire bloc (repère un objet ajouté ou effacé)
max_local_color = 0.12 # écart de couleur maximal du pire bloc (0 à 1)
max_seam_ratio = 1.8 # raccord maximal pour les textures répétables
color_match = 0.6    # recalage des couleurs sur l'original (0 = aucun, 1 = total)

[local]
# Optionnel : chemin de realesrgan-ncnn-vulkan.exe pour un meilleur repli local
esrgan_exe = ''
esrgan_model = "realesrgan-x4plus"

[qc]
# Contrôle qualité final avant DDS (texforge qc / texforge export). Mesures de fidélité faites sur
# l'image ramenée à la résolution d'origine : la DA est jugée là, le micro-détail au-dessus.
auto_fix = true             # correctifs automatiques (verrou couleur, fondu du raccord) avant de rejeter
min_ssim = 0.5              # SSIM multi-échelle minimal (structure)
max_delta_e = 6.0           # écart de couleur moyen ΔE2000 (2,3 ≈ juste perceptible)
max_delta_e_p95 = 14.0      # zones localement recolorées
max_hue_shift = 8.0         # dérive de teinte (degrés)
min_chroma_ratio = 0.7      # saturation : plage autorisée
max_chroma_ratio = 1.4
max_lightness_shift = 6.0   # luminosité (L*)
max_palette_distance = 0.35 # histogrammes Lab (0 = même palette)
max_seam_local = 1.6        # raccord : cassure par rapport aux colonnes voisines
max_seam_step = 4.0         # raccord : dérive de couleur d'un bord à l'autre (ΔE)
color_lock = 1.0            # force du verrou couleur quand il est déclenché (0 à 1)
color_lock_always = 0.0     # > 0 : verrou léger appliqué à toutes les textures IA
alpha_ref = 0.5             # seuil du test alpha du jeu (0x80 = 0.5) : couverture des mipmaps conservée
# DDS
encoder = "auto"            # "auto" (texconv s'il est trouvé) | "texconv" | "builtin"
texconv_exe = ''            # ex. 'C:\Outils\texconv.exe' ; sinon cherché dans le PATH et dans tools\
target_api = "d3d9"         # "d3d9" (en-têtes legacy) | "d3d10" | "d3d11" (dgVoodoo2 + Special K, BC7 possible)
allow_bc7 = false
dxt1_alpha = false          # true : découpes en DXT1 1 bit (plus léger, bords moins nets) au lieu de DXT5
mip_filter = "lanczos"      # "lanczos" (mips nettes) | "box"
min_psnr = 28.0             # avertissement si la compression dégrade trop (dégradés, texte)
memory_budget_mb = 0        # > 0 : alerte si l'export dépasse ce total (jeux 32 bits : ~1024)
workers = 4                 # textures contrôlées en parallèle

# Règles par nom de fichier (motifs avec * et ?), appliquées dans l'ordre, prioritaires sur le routage.
# mode = "ai" (ComfyUI) | "ui" (pipeline UI) | "upscale" (Lanczos) | "keep" (original)
# dds_format = format DDS imposé pour ces textures (ex. "bgra8" pour un texte ou un dégradé)
# hint = précision ajoutée au prompt ; material = stone, masonry, metal, wood, fabric, vegetation, skin, generic
# engine = moteur imposé au pipeline UI : realesrgan, waifu2x, nearest (pixel art), lanczos
# tile = true/false force la détection « texture répétable »
# [[rules]]
# match = "*Compressed_1A2B3C4D*"
# mode = "ai"
# material = "masonry"
# hint = "mossy cobblestone"
# tile = true
"""
