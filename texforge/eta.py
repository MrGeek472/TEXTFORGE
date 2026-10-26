"""Estimation de la durée totale d'un lancement.

Le temps de génération ComfyUI se compte en « passages » : une génération à la résolution native du
modèle (environ 1 mégapixel). Une texture en une passe en coûte un ; par morceaux, un par morceau,
plus la passe globale (rendu photoréaliste) ou l'essai du denoise automatique. Les agrandissements
locaux, Real-ESRGAN et l'export (contrôle qualité + DDS) se comptent au mégapixel produit.

Chaque durée est apprise lancement après lancement sur ce PC (timings.json, moyenne glissante) ;
avant le premier lancement, des valeurs indicatives servent. Pendant le traitement, ProgressBus
recale l'estimation sur le temps réellement mis par les textures déjà terminées.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from .i18n import tr

# secondes par passage d'environ 1 mégapixel avec ControlNet Tile, frais du lot compris (agrandissement de
# base, fondu, contrôle, PNG) ; indicatif pour une carte graphique de milieu de gamme, remplacé par la mesure
DEFAULT_PASS_S = {"sdxl": 12.0, "sd15": 3.5, "zimage": 16.0, "custom": 12.0}
DEFAULT_RATES = {"local_s_per_mp": 0.15, "esrgan_s_per_mp": 1.5, "export_s_per_mp": 0.8}
LEARN = 0.5                                       # poids de la dernière mesure dans la moyenne glissante
GROUPS = ("ui", "local", "ai")                    # ordre des sous-étapes de la génération


def timings_path() -> Path:
    if os.environ.get("TEXFORGE_TIMINGS"):
        return Path(os.environ["TEXFORGE_TIMINGS"])
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "TEXFORGE"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "texforge"
    return base / "timings.json"


def load_timings() -> dict:
    try:
        d = json.loads(timings_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    p = timings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass                                      # une estimation moins juste ne doit rien bloquer


def family_of(cfg: dict) -> str:
    name = Path(str(cfg.get("comfyui", {}).get("workflow") or "")).name.lower()
    return next((f for f in ("sdxl", "sd15", "zimage") if f in name), "custom")


def _k(w: int, h: int, cfg: dict) -> int:
    from .imaging import output_scale
    out = cfg["output"]
    return output_scale(w, h, int(out.get("scale", 4)), int(out.get("max_side", 4096)))


def output_mp(row: dict, cfg: dict) -> float:
    """Mégapixels de la texture produite."""
    w, h = int(row.get("width") or 0), int(row.get("height") or 0)
    if not (w and h):
        return 0.0
    k = _k(w, h, cfg)
    return w * k * h * k / 1e6


def passes_for(row: dict, cfg: dict) -> int:
    """Passages ComfyUI d'une texture, comme les fera le moteur au premier essai (un refus du contrôle
    qualité en ajoute : le recalage pendant le lot les prend en compte)."""
    from .imaging import tile_plan, tile_starts
    w, h = int(row.get("width") or 0), int(row.get("height") or 0)
    if not (w and h):
        return 1
    k = _k(w, h, cfg)
    plan = tile_plan(SimpleNamespace(width=w, height=h), k, cfg)
    if plan is None:
        return 1
    size, overlap = plan
    an = row.get("analysis") or {}
    p = max(1, math.ceil(overlap / k))            # marge en boucle, comme imaging.prepare_tiled
    W = w * k + (2 * p * k if an.get("tile_x") else 0)
    H = h * k + (2 * p * k if an.get("tile_y") else 0)
    n = len(tile_starts(W, size, overlap)) * len(tile_starts(H, size, overlap))
    c = cfg.get("comfyui", {})
    if c.get("style") == "photoreal":
        n += 1                                    # passe globale à 1 mégapixel
    elif len(c.get("denoise_ladder") or []) > 1:
        n += 1                                    # essai du denoise automatique à 1 mégapixel
    return n


def rates(cfg: dict) -> dict:
    t = load_timings()
    fam = family_of(cfg)
    learned = (t.get("pass_s") or {}).get(fam)
    out = {"family": fam, "pass_s": float(learned or DEFAULT_PASS_S.get(fam, DEFAULT_PASS_S["custom"])),
           "pass_learned": bool(learned), "tail_s": float((t.get("tail_s") or {}).get(fam) or 0.0)}
    for key, val in DEFAULT_RATES.items():
        out[key] = float(t.get(key) or val)
        out[key.replace("_s_per_mp", "_learned")] = bool(t.get(key))
    return out


def plan(cfg: dict, provider: str, groups: dict[str, list[dict]], export_rows: list[dict]) -> tuple[dict, dict]:
    """Durée prévue de chaque texture à générer et résumé du lancement.

    groups : textures par sous-étape ("ui", "local", "ai"). Renvoie (par texture : id -> (sous-étape,
    secondes, passages, mégapixels), résumé en secondes : ui_s, local_s, ai_s, generate_s, export_s, total_s,
    passes, pass_s, pass_learned…)."""
    r = rates(cfg)
    per: dict[str, tuple[str, float, int, float]] = {}
    sums = {g: 0.0 for g in GROUPS}
    passes = 0
    for g in GROUPS:
        for row in groups.get(g, []):
            mp = output_mp(row, cfg)
            n = 0
            if g == "ai" and provider == "comfyui":
                n = passes_for(row, cfg)
                s = n * r["pass_s"]
                passes += n
            elif g == "ai" and provider == "esrgan":
                s = mp * r["esrgan_s_per_mp"]
            else:                                 # agrandissement local, élément d'interface, simulation
                s = mp * r["local_s_per_mp"]
            per[str(row["id"])] = (g, s, n, mp)
            sums[g] += s
    if passes:
        sums["ai"] += r["tail_s"]                 # fin du dernier lot, que plus aucun passage ne recouvre
    exp_mp = sum(output_mp(row, cfg) for row in export_rows)
    export_s = exp_mp * r["export_s_per_mp"]
    gen = sum(sums.values())
    summary = {"ui_s": round(sums["ui"]), "local_s": round(sums["local"]), "ai_s": round(sums["ai"]),
               "generate_s": round(gen), "export_s": round(export_s), "total_s": round(gen + export_s),
               "passes": passes, "ai": len(groups.get("ai", [])), "export": len(export_rows),
               "export_mp": round(exp_mp, 1), "provider": provider, **{k: v for k, v in r.items()}}
    summary["pass_s"] = round(r["pass_s"], 2)
    summary["tail_s"] = round(r["tail_s"], 1)
    return per, summary


def learn(cfg: dict, provider: str, measured: dict) -> None:
    """Mémorise les durées mesurées de ce lancement pour les prochaines estimations.

    measured : {"ai": (secondes, passages, mégapixels), "ai_tail": secondes, "local": (secondes, mégapixels),
    "export": (secondes, mégapixels)} ; une mesure trop courte pour être fiable est ignorée."""
    if provider == "mock":
        measured = {k: v for k, v in measured.items() if k == "export"}
    t = load_timings()
    changed = False

    def mix(cur, val: float) -> float:
        return val if not cur else (1 - LEARN) * float(cur) + LEARN * val

    ai = measured.get("ai")
    if ai and ai[0] >= 5:
        secs, n, mp = ai
        if provider == "comfyui" and n >= 1:
            fam = family_of(cfg)
            pass_s = t.setdefault("pass_s", {})
            pass_s[fam] = round(mix(pass_s.get(fam), secs / n), 2)
            if measured.get("ai_tail") is not None:
                tails = t.setdefault("tail_s", {})
                tails[fam] = round(mix(tails.get(fam), float(measured["ai_tail"])), 1)
            changed = True
        elif provider == "esrgan" and mp > 0:
            t["esrgan_s_per_mp"] = round(mix(t.get("esrgan_s_per_mp"), secs / mp), 3)
            changed = True
    for key, name in (("local", "local_s_per_mp"), ("export", "export_s_per_mp")):
        m = measured.get(key)
        if m and m[0] >= 5 and m[1] >= 1:
            t[name] = round(mix(t.get(name), m[0] / m[1]), 3)
            changed = True
    if changed:
        _save(t)


def fmt(seconds: float | None) -> str:
    """Durée lisible et arrondie (une estimation : pas de secondes au-delà d'une minute)."""
    if seconds is None:
        return "—"
    s = max(0, int(round(seconds)))
    if s < 60:
        return tr("{s} s", s=s)
    m = int(round(s / 60))
    if m < 60:
        return tr("{m} min", m=m)
    h, m = divmod(m, 60)
    if h < 48:
        return tr("{h} h {m:02d}", h=h, m=m)
    d, h = divmod(h, 24)
    return tr("{d} j {h} h", d=d, h=h)


def describe(est: dict) -> str:
    """Ligne de console de l'estimation annoncée au début de la génération."""
    parts = []
    if est.get("passes"):
        src = tr("mesuré sur ce PC") if est.get("pass_learned") else tr("valeur indicative, mesurée pendant ce "
                                                                        "lancement")
        parts.append(tr("génération ~{gen} : {passes} passage(s) ComfyUI × ~{per} s ({src})",
                        gen=fmt(est.get("generate_s")), passes=est["passes"], per=round(float(est.get("pass_s") or 0), 1), src=src))
    elif est.get("generate_s"):
        parts.append(tr("génération ~{gen}", gen=fmt(est.get("generate_s"))))
    if est.get("export"):
        parts.append(tr("contrôle qualité et export ~{t} ({n} texture(s), {mp} Mpx)", t=fmt(est.get("export_s")),
                        n=est["export"], mp=est.get("export_mp")))
    return (tr("[ESTIMATION] ≈ {total} au total", total=fmt(est.get("total_s")))
            + (" — " + " ; ".join(parts) if parts else "")
            + tr(" — recalée pendant le traitement"))
