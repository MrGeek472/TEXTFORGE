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

EVENT_TYPES = (
    "run_start", "stage_start", "stage_end", "routed", "ui_done", "local_done", "comfy_queued", "comfy_progress",
    "comfy_done", "fallback", "retry", "failed", "gpu_free", "gpu_status", "server_down", "server_back",
    "stopped", "run_end", "info", "qc_start", "qc_done", "qc_end",
    "pipeline_start", "pipeline_stage", "progress", "pipeline_end",
)
_QC_TAG = {"validated": "[QC] ✔", "corrected": "[QC] ✎", "rejected": "[QC] ✘", "local": "[QC] ·",
           "measured": "[QC] =", "error": "[QC] !"}


def _fmt(rec: dict[str, Any]) -> str | None:
    e = rec["event"]
    n = rec.get("name", "")
    pos = f" ({rec['index']}/{rec['total']})" if rec.get("total") and rec.get("index") else ""
    if e == "run_start":
        return f"== TEXFORGE : {rec.get('total', 0)} texture(s) à traiter — moteur IA : {rec.get('provider', '?')}"
    if e == "stage_start":
        return f"── Étape {rec.get('stage', '').upper()} : {rec.get('count', 0)} texture(s)"
    if e == "stage_end":
        return f"── Fin {rec.get('stage', '').upper()} : {rec.get('summary', '')}"
    if e == "routed":
        return f"[ROUTE] {n} → {rec.get('route', '').upper()} ({rec.get('reason', '')})" if rec.get("verbose") else None
    if e == "ui_done":
        return f"[UI] {n} traité localement ({rec.get('engine', '')}, ×{rec.get('k', '?')}){pos}"
    if e == "local_done":
        return f"[LOCAL] {n} agrandi ({rec.get('engine', 'lanczos')}, ×{rec.get('k', '?')}){pos}"
    if e == "comfy_queued":
        return f"[COMFYUI] {n} en file{pos} — {rec.get('material', '')}, tiling {rec.get('tiling', '')}"
    if e == "comfy_done":
        return (f"[COMFYUI] {n} validé{pos} — structure {rec.get('structure', '?')}, essai {rec.get('attempt', '?')}, "
                f"{rec.get('seconds', 0):.1f} s")
    if e == "fallback":
        return f"[REPLI] {n} → {rec.get('engine', '')} ({rec.get('reason', '')}){pos}"
    if e == "retry":
        return f"[RETRY] {n} : {rec.get('message', '')} — nouvel essai {rec.get('attempt', '')}"
    if e == "failed":
        return f"[ERREUR] {n} : {rec.get('message', '')} [{rec.get('kind', '')}]"
    if e == "gpu_free":
        free = rec.get("vram_free_gb")
        return f"[GPU] VRAM libérée ({rec.get('reason', '')})" + (f" — libre avant : {free:.1f} Go" if free is not None else "")
    if e == "gpu_status":
        return None
    if e == "server_down":
        return f"[COMFYUI] serveur injoignable — {rec.get('message', '')}"
    if e == "server_back":
        return "[COMFYUI] serveur de nouveau disponible"
    if e == "stopped":
        return f"■ Arrêt : {rec.get('message', '')}"
    if e == "run_end":
        return f"== Terminé en {rec.get('seconds', 0):.0f} s — {rec.get('summary', '')}"
    if e == "info":
        return rec.get("message", "")
    if e == "progress":
        return None
    if e == "pipeline_start":
        return (f"[INFO] Chaîne complète — entrée {rec.get('input', '')}\n"
                f"[INFO] sortie {rec.get('output', '')} — moteur {rec.get('provider', '')}, modèle {rec.get('model', '')}, "
                f"checkpoint {rec.get('checkpoint', '')}" + (f", LoRA {rec['lora']}" if rec.get("lora") else "")
                + f"\n[INFO] projet : {rec.get('work', '')}")
    if e == "pipeline_stage":
        return f"══ ÉTAPE {rec.get('index', '?')}/{rec.get('total', '?')} : {rec.get('label', rec.get('stage', ''))}"
    if e == "pipeline_end":
        head = {"ok": "✔ Pipeline terminé", "errors": "⚠ Pipeline terminé avec des échecs",
                "stopped": "■ Pipeline arrêté", "fatal": "✗ Pipeline interrompu"}.get(rec.get("status"), "Fin")
        line = f"══ {head} en {rec.get('seconds', 0):.0f} s"
        if rec.get("message"):
            line += f" — {rec['message']}"
        if rec.get("summary"):
            line += f"\n   {rec['summary']}"
        return line
    if e == "qc_start":
        what = {"specialk": "DDS Special K", "png": "PNG", "remix": "PNG RTX Remix", "none": "sans export"}
        return (f"== Contrôle qualité : {rec.get('total', 0)} texture(s) — {what.get(rec.get('target'), '?')}"
                + (f", encodeur {rec['encoder']}" if rec.get("encoder") not in (None, "-") else "")
                + ("" if rec.get("qc", True) else " (QC désactivé)"))
    if e == "qc_done":
        d = rec.get("decision", "")
        line = f"{_QC_TAG.get(d, '[QC]')} {n} {rec.get('label', d)}{pos}"
        if rec.get("summary"):
            line += f" — {rec['summary']}"
        if rec.get("format"):
            line += f" → {rec['format']}, {rec.get('levels', 1)} mip(s)"
        if d == "corrected" and rec.get("fixes"):
            line += f" | correctifs : {', '.join(rec['fixes'])}"
        if d in ("rejected", "error") and rec.get("reasons"):
            line += f" | {'; '.join(rec['reasons'][:2])}"
            if d == "rejected":
                line += " → version locale conservée"
        if rec.get("warnings"):
            line += f" | ⚠ {'; '.join(rec['warnings'][:2])}"
        return line
    if e == "qc_end":
        line = f"== QC terminé en {rec.get('seconds', 0):.0f} s — {rec.get('summary', '')}"
        if rec.get("memory_warning"):
            line += f"\n⚠ {rec['memory_warning']}"
        if rec.get("report"):
            line += f"\nRapport : {rec['report']}"
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
