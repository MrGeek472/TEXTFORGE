"""Événements de l'orchestrateur : console lisible, journal JSONL, flux JSON pour Node.js.

Chaque événement est une ligne JSON {"ts", "event", ...} :
  - toujours ajoutée à <travail>/events.jsonl (traçabilité complète) ;
  - écrite sur la sortie standard si json_stdout=True (pont Node.js : une ligne = un événement) ;
  - résumée en une ligne lisible sur la sortie d'erreur (tableau de bord console).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .i18n import N_, tr

EVENT_TYPES = (
    "run_start", "stage_start", "stage_end", "routed", "ui_done", "local_done", "comfy_queued", "comfy_progress",
    "comfy_done", "fallback", "retry", "failed", "gpu_free", "gpu_status", "server_down", "server_back",
    "stopped", "run_end", "info", "qc_start", "qc_done", "qc_end",
    "pipeline_start", "pipeline_stage", "progress", "pipeline_end", "esrgan_done",
)
_QC_TAG = {"validated": "[QC] ✔", "corrected": "[QC] ✎", "rejected": "[QC] ✘", "local": "[QC] ·",
           "measured": "[QC] =", "error": "[QC] !"}


def _fmt(rec: dict[str, Any]) -> str | None:
    """Ligne lisible d'un événement, dans la langue active (les champs libres comme `message` arrivent
    déjà dans la langue du processus qui les a produits)."""
    e = rec["event"]
    n = rec.get("name", "")
    pos = f" ({rec['index']}/{rec['total']})" if rec.get("total") and rec.get("index") else ""
    if e == "run_start":
        return tr("== TEXFORGE : {total} texture(s) à traiter — moteur IA : {provider}",
                  total=rec.get("total", 0), provider=rec.get("provider", "?"))
    if e == "stage_start":
        return tr("── Étape {stage} : {count} texture(s)", stage=rec.get("stage", "").upper(), count=rec.get("count", 0))
    if e == "stage_end":
        return tr("── Fin {stage} : {summary}", stage=rec.get("stage", "").upper(), summary=rec.get("summary", ""))
    if e == "routed":
        return (f"[ROUTE] {n} → {rec.get('route', '').upper()} ({rec.get('reason', '')})"
                if rec.get("verbose") else None)
    if e == "ui_done":
        return tr("[UI] {name} traité localement ({engine}, ×{k}){pos}", name=n, engine=rec.get("engine", ""),
                  k=rec.get("k", "?"), pos=pos)
    if e == "local_done":
        return tr("[LOCAL] {name} agrandi ({engine}, ×{k}){pos}", name=n, engine=rec.get("engine", "lanczos"),
                  k=rec.get("k", "?"), pos=pos)
    if e == "comfy_queued":
        return tr("[COMFYUI] {name} en file{pos} — {material}, tiling {tiling}", name=n, pos=pos,
                  material=rec.get("material", ""), tiling=rec.get("tiling", ""))
    if e == "comfy_done":
        return tr("[COMFYUI] {name} validé{pos} — structure {structure}, essai {attempt}, {seconds:.1f} s", name=n,
                  pos=pos, structure=rec.get("structure", "?"), attempt=rec.get("attempt", "?"),
                  seconds=float(rec.get("seconds", 0) or 0))
    if e == "esrgan_done":
        return tr("[ESRGAN] {name} agrandi ×{k} — {material}, part prudente {guard:.0%}, grain {grain:.1f}{pos}",
                  name=n, k=rec.get("k", "?"), material=rec.get("material", ""),
                  guard=float(rec.get("guard") or 0), grain=float(rec.get("grain") or 0), pos=pos)
    if e == "fallback":
        return tr("[REPLI] {name} → {engine} ({reason}){pos}", name=n, engine=rec.get("engine", ""),
                  reason=rec.get("reason", ""), pos=pos)
    if e == "retry":
        return tr("[RETRY] {name} : {message} — nouvel essai {attempt}", name=n, message=rec.get("message", ""),
                  attempt=rec.get("attempt", ""))
    if e == "failed":
        return tr("[ERREUR] {name} : {message} [{kind}]", name=n, message=rec.get("message", ""),
                  kind=rec.get("kind", ""))
    if e == "gpu_free":
        free = rec.get("vram_free_gb")
        return (tr("[GPU] VRAM libérée ({reason})", reason=rec.get("reason", ""))
                + (tr(" — libre avant : {gb:.1f} Go", gb=free) if free is not None else ""))
    if e == "gpu_status":
        return None
    if e == "server_down":
        return tr("[COMFYUI] serveur injoignable — {message}", message=rec.get("message", ""))
    if e == "server_back":
        return tr("[COMFYUI] serveur de nouveau disponible")
    if e == "stopped":
        return tr("■ Arrêt : {message}", message=rec.get("message", ""))
    if e == "run_end":
        return tr("== Terminé en {seconds:.0f} s — {summary}", seconds=float(rec.get("seconds", 0) or 0),
                  summary=rec.get("summary", ""))
    if e == "info":
        return rec.get("message", "")
    if e == "progress":
        return None
    if e == "pipeline_start":
        lora = tr(", LoRA {lora}", lora=rec["lora"]) if rec.get("lora") else ""
        return (tr("[INFO] Chaîne complète — entrée {input}", input=rec.get("input", "")) + "\n"
                + tr("[INFO] sortie {output} — moteur {provider}, modèle {model}, checkpoint {checkpoint}",
                     output=rec.get("output", ""), provider=rec.get("provider", ""), model=rec.get("model", ""),
                     checkpoint=rec.get("checkpoint", ""))
                + lora + "\n" + tr("[INFO] projet : {work}", work=rec.get("work", ""))
                + ("\n" + tr("[INFO] TEXFORGE {version} lancé depuis {program}", version=rec["version"],
                              program=rec.get("program", "?")) if rec.get("version") else ""))
    if e == "pipeline_stage":
        label = rec.get("label", rec.get("stage", ""))
        return tr("══ ÉTAPE {index}/{total} : {label}", index=rec.get("index", "?"), total=rec.get("total", "?"),
                  label=label)
    if e == "pipeline_end":
        heads = {"ok": N_("✔ Pipeline terminé"), "errors": N_("⚠ Pipeline terminé avec des échecs"),
                 "stopped": N_("■ Pipeline arrêté"), "fatal": N_("✗ Pipeline interrompu")}
        head = tr(heads.get(rec.get("status"), N_("Fin")))
        line = tr("══ {head} en {seconds:.0f} s", head=head, seconds=float(rec.get("seconds", 0) or 0))
        if rec.get("message"):
            line += f" — {rec['message']}"
        if rec.get("summary"):
            line += f"\n   {rec['summary']}"
        return line
    if e == "qc_start":
        what = {"specialk": N_("DDS Special K"), "loose": N_("DDS en fichiers séparés"), "png": "PNG",
                "remix": N_("PNG RTX Remix"), "none": N_("sans export")}
        target = what.get(rec.get("target"), "?")
        return (tr("== Contrôle qualité : {total} texture(s) — {target}", total=rec.get("total", 0), target=tr(target))
                + (tr(", encodeur {encoder}", encoder=rec["encoder"]) if rec.get("encoder") not in (None, "-") else "")
                + ("" if rec.get("qc", True) else tr(" (QC désactivé)")))
    if e == "qc_done":
        d = rec.get("decision", "")
        line = f"{_QC_TAG.get(d, '[QC]')} {n} {rec.get('label', d)}{pos}"
        if rec.get("summary"):
            line += f" — {rec['summary']}"
        if rec.get("format"):
            line += tr(" → {format}, {levels} mip(s)", format=rec["format"], levels=rec.get("levels", 1))
        if d == "corrected" and rec.get("fixes"):
            line += tr(" | correctifs : {fixes}", fixes=", ".join(rec["fixes"]))
        if d in ("rejected", "error") and rec.get("reasons"):
            line += f" | {'; '.join(rec['reasons'][:2])}"
            if d == "rejected":
                line += tr(" → version locale conservée")
        if rec.get("warnings"):
            line += f" | ⚠ {'; '.join(rec['warnings'][:2])}"
        return line
    if e == "qc_end":
        line = tr("== QC terminé en {seconds:.0f} s — {summary}", seconds=float(rec.get("seconds", 0) or 0),
                  summary=rec.get("summary", ""))
        if rec.get("memory_warning"):
            line += f"\n⚠ {rec['memory_warning']}"
        if rec.get("report"):
            line += "\n" + tr("Rapport : {path}", path=rec["report"])
        return line
    return None


class EventBus:
    def __init__(self, work_dir: str | Path, *, json_stdout: bool = False, console: bool = True,
                 verbose: bool = False, stream: TextIO | None = None) -> None:
        self.path = Path(work_dir) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")
        self.json_stdout = json_stdout
        self.console = console
        self.verbose = verbose
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._last_progress = 0.0
        self.counts: dict[str, int] = {}

    def emit(self, event: str, **data: Any) -> None:
        if event == "comfy_progress":         # limité à 2 par seconde
            now = time.monotonic()
            if now - self._last_progress < 0.5:
                return
            self._last_progress = now
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **data}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            self.counts[event] = self.counts.get(event, 0) + 1
            if event != "comfy_progress":
                self._file.write(line + "\n")
                self._file.flush()
            if self.json_stdout:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            if self.console:
                text = _fmt({**rec, "verbose": self.verbose})
                if text:
                    try:
                        from tqdm import tqdm
                        tqdm.write(text, file=self.stream)
                    except Exception:
                        print(text, file=self.stream, flush=True)

    def close(self) -> None:
        with self._lock:
            self._file.close()
