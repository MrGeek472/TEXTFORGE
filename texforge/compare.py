"""Données du panneau avant / après de l'interface : textures traitées d'un projet, image d'origine,
résultat à montrer à côté et explication de ce qui a été fait. Sans dépendance à Qt (testable)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .i18n import tr

DONE_STATUSES = ("done", "fallback")


@dataclass
class CompareItem:
    id: str
    name: str
    src_path: str
    right_path: str
    right_kind: str          # ai | ai_rejected | esrgan | local | ui
    right_label: str         # légende au-dessus de l'image de droite
    info: str                # ligne d'explication sous les images
    status: str
    updated: float
    src_size: tuple[int, int]

    @property
    def is_ai(self) -> bool:
        return self.right_kind in ("ai", "ai_rejected", "esrgan")


def _db_rows(db: Path) -> list[dict]:
    """Lecture seule : le traitement peut écrire dans la base pendant que l'interface la lit."""
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM textures WHERE status IN (?, ?) ORDER BY updated DESC",
                           DONE_STATUSES).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out = []
    for row in rows:
        d = dict(row)
        for k in ("analysis", "metrics"):
            try:
                d[k] = json.loads(d[k]) if d.get(k) else {}
            except ValueError:
                d[k] = {}
        out.append(d)
    return out


def _qc_by_id(work: Path) -> dict[str, dict]:
    try:
        rep = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {t.get("id"): t for t in rep.get("textures", []) if isinstance(t, dict)}


def _qc_text(entry: dict | None) -> str:
    if not entry:
        return ""
    after = (entry.get("metrics") or {}).get("after") or {}
    from .qc import LABELS
    decision = entry.get("decision", "")
    label = tr(LABELS[decision]) if decision in LABELS else (entry.get("label") or decision or "?")
    parts = [tr("QC : {label}", label=label)]
    if "ssim" in after:
        parts.append(f"SSIM {after['ssim']:.2f}")
    if "delta_e" in after:
        parts.append(f"ΔE {after['delta_e']:.1f}")
    gain = (entry.get("metrics") or {}).get("after", {}).get("detail_gain")
    if gain is not None:
        parts.append(tr("détail ×{gain:.2f}", gain=gain))
    return " · ".join(parts)


def _existing(p: str | None) -> str:
    return p if p and Path(p).exists() else ""


def describe(r: dict, qc: dict | None = None) -> CompareItem | None:
    """Ce que le panneau montre pour une texture traitée ; None si aucune image n'est sur le disque."""
    m = r.get("metrics") or {}
    an = r.get("analysis") or {}
    method = str(m.get("method", ""))
    ai, result = _existing(r.get("ai_path")), _existing(r.get("result_path"))
    material = an.get("material") or ""
    reason = an.get("route_reason") or ""
    if ai and method == "esrgan":
        kind, path, label = "esrgan", ai, tr("Real-ESRGAN (mode réaliste)")
    elif ai and (r.get("status") == "fallback" or r.get("choice") == "local"):
        kind, path, label = "ai_rejected", ai, tr("Essai IA écarté — le fichier exporté est la version locale")
    elif ai:
        kind, path, label = "ai", ai, tr("IA (ComfyUI)")
    elif result and method.startswith("ui:"):
        kind, path, label = "ui", result, tr("Agrandissement UI ({engine}), sans IA générative",
                                             engine=method.split(":", 1)[1])
    elif result:
        engine = method.split(":", 1)[1] if ":" in method else "lanczos"
        kind, path, label = "local", result, tr("Agrandissement sans IA ({engine})", engine=engine)
    else:
        return None
    bits = []
    if kind == "ai":
        attempt = m.get("attempt")
        bits.append(tr("matériau {material}", material=material) if material else "")
        if attempt:
            bits.append(tr("essai {n}", n=attempt))
    elif kind == "ai_rejected":
        bits.append(tr("raison : {why}", why=(r.get("error") or "?")[:160]))
    elif reason:
        bits.append(tr("routage : {why}", why=reason[:160]))
    q = _qc_text(qc)
    if q:
        bits.append(q)
    w, h = int(r.get("width") or 0), int(r.get("height") or 0)
    return CompareItem(r["id"], r["name"], r.get("src_path") or "", path, kind, label,
                       " — ".join(b for b in bits if b), r.get("status", ""), float(r.get("updated") or 0), (w, h))


def list_items(project_dir: Path) -> list[CompareItem]:
    """Textures traitées du projet, les plus récentes d'abord (dossier de projet de l'interface)."""
    work = Path(project_dir) / "work"
    qc = _qc_by_id(work)
    out = []
    for r in _db_rows(work / "texforge_state.db"):
        it = describe(r, qc.get(r["id"]))
        if it is not None:
            out.append(it)
    return out


def load_rgba(path: str | Path) -> np.ndarray:
    """Image en RGBA 8 bits (DDS compris) pour l'affichage."""
    from .imaging import load_texture
    return np.ascontiguousarray(np.asarray(load_texture(Path(path)).image.convert("RGBA")))
