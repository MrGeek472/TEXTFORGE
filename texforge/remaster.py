"""Chaîne complète de remaster en une seule commande, pilotée par un fichier de réglages.

    python -m texforge remaster --settings reglages.json [--json-events]
    python -m texforge remaster --input C:/dump --output C:/Jeu/SK_Res/inject/textures

Étapes, dans un seul processus et un seul flux d'événements :
  1. scan       analyse et routage (UI / matériaux ComfyUI / local / gardées)
  2. generate   UI (Real-ESRGAN), local (Lanczos), ComfyUI headless (ControlNet Tile, raccord)
  3. export     contrôle qualité (DA, raccord, alpha), correctifs, DDS + mipmaps, qc_report.json

C'est le moteur de l'interface graphique (app_gui.py) : elle écrit un fichier de réglages, lance
cette commande dans un processus séparé et lit les événements JSON. Le même fichier permet de
relancer exactement le même traitement en ligne de commande ou depuis Node.js.

Les réglages sont traduits en un vrai `texforge.toml` dans le dossier de projet : toutes les
commandes habituelles (review, failures, retry, qc…) fonctionnent ensuite sur ce projet.

Événements ajoutés à ceux de l'orchestrateur et du QC :
  pipeline_start, pipeline_stage {stage, index, total, label}, progress {fraction, stage, done,
  total, eta_s}, pipeline_end {code, status, summary, report, output}.
Codes de sortie : 0 terminé, 1 terminé avec des échecs, 2 arrêt fatal, 3 arrêté à la demande.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .config import DEFAULTS, load_config
from .events import EventBus

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = {
    # famille : (workflow principal, workflow pour les matériaux à joints)
    "sdxl": ("workflows/texforge_sdxl_tile_api.json", "workflows/texforge_sdxl_tile_canny_api.json"),
    "sd15": ("workflows/texforge_sd15_tile_api.json", ""),
}
STAGES = ("scan", "generate", "export")
STAGE_LABELS = {"scan": "Scan et routage", "generate": "Génération (UI, local, ComfyUI)",
                "export": "Contrôle qualité et export DDS"}
EXIT_OK, EXIT_ERRORS, EXIT_FATAL, EXIT_STOPPED = 0, 1, 2, 3


# ------------------------------------------------------------------ réglages

@dataclass
class RemasterSettings:
    input_dir: str = ""
    output_dir: str = ""
    work_dir: str = ""              # vide : dossier de projet par défaut (données locales de l'utilisateur)
    target: str = "specialk"        # specialk (DDS) | remix (PNG RTX Remix) | png
    # moteur IA
    provider: str = "comfyui"       # comfyui | mock (simulation sans GPU)
    server: str = "http://127.0.0.1:8188"
    model_family: str = "sdxl"      # sdxl | sd15 | custom
    workflow: str = ""              # famille custom : workflow au format API
    workflow_edges: str = ""
    checkpoint: str = ""            # vide : celui du workflow
    lora: str = ""
    lora_strength: float = 0.8
    # génération
    denoise: float = 0.42
    cn_strength: float = 0.75
    per_material: bool = True       # les curseurs décalent les préréglages par matériau au lieu de les remplacer
    scale: int = 4
    max_side: int = 2048
    limit: int = 0                  # 0 : toutes les textures ComfyUI
    redo_stale: bool = True         # régénérer les textures produites avec d'autres réglages
    # options et routage
    route_ui: bool = True
    seamless: bool = True
    force_seamless: bool = False
    mipmaps: bool = True
    qc: bool = True
    qc_autofix: bool = True
    # sortie DDS
    dds_format: str = "match"       # match | auto | dxt1 | dxt5 | bc7 | bgra8
    encoder: str = "auto"           # auto | texconv | builtin
    texconv_exe: str = ""
    realesrgan_exe: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "RemasterSettings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        errs = []
        if not self.input_dir or not Path(self.input_dir).is_dir():
            errs.append(f"dossier d'entrée introuvable : {self.input_dir or '(vide)'}")
        if not self.output_dir:
            errs.append("dossier de sortie non renseigné")
        elif Path(self.input_dir).resolve() == Path(self.output_dir).resolve():
            errs.append("le dossier de sortie doit être différent du dossier d'entrée")
        if self.model_family == "custom" and not Path(self.workflow).is_file():
            errs.append(f"workflow personnalisé introuvable : {self.workflow or '(vide)'}")
        if self.model_family not in (*FAMILIES, "custom"):
            errs.append(f"famille de modèle inconnue : {self.model_family}")
        if self.dds_format not in ("match", "auto", "dxt1", "dxt5", "bc7", "bgra8"):
            errs.append(f"format DDS inconnu : {self.dds_format}")
        return errs


def default_work_dir(input_dir: str) -> Path:
    """Dossier de projet propre à un dossier d'entrée, hors du jeu et hors du dossier d'injection :
    %LOCALAPPDATA%\\TEXFORGE\\projets\\<nom>-<empreinte> (ou ~/.local/share/texforge/projets)."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "TEXFORGE"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "texforge"
    src = str(Path(input_dir).resolve()) if input_dir else "sans-dossier"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(src).name or "projet")[:40]
    return base / "projets" / f"{slug}-{hashlib.sha1(src.lower().encode()).hexdigest()[:8]}"


def work_dir_of(s: RemasterSettings) -> Path:
    return Path(s.work_dir) if s.work_dir else default_work_dir(s.input_dir)


# ------------------------------------------------------------------ réglages -> texforge.toml

def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)          # chaîne JSON = chaîne TOML de base valide
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"valeur non sérialisable en TOML : {v!r}")


def _toml(sections: dict, rules: list[dict]) -> str:
    out = ["# Généré par TEXFORGE (interface graphique / commande remaster). Modifiable à la main ;",
           "# il sera réécrit au prochain lancement depuis l'interface.", ""]
    for name, table in sections.items():
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subs = {k: v for k, v in table.items() if isinstance(v, dict)}
        if scalars:
            out.append(f"[{name}]")
            out += [f"{k} = {_toml_value(v)}" for k, v in scalars.items()]
            out.append("")
        for sub, t in subs.items():
            out.append(f"[{name}.{sub}]")
            out += [f"{k} = {_toml_value(v)}" for k, v in t.items()]
            out.append("")
    for r in rules:
        out.append("[[rules]]")
        out += [f"{k} = {_toml_value(v)}" for k, v in r.items()]
        out.append("")
    return "\n".join(out)


def material_presets(s: RemasterSettings) -> dict:
    """Denoise et force du ControlNet par matériau. En mode « par matériau », les curseurs
    décalent tous les préréglages de l'écart choisi par rapport au préréglage générique
    (la peau reste plus prudente que la pierre) ; sinon ils s'appliquent tels quels."""
    base = copy.deepcopy(DEFAULTS["materials"])
    g = base.get("generic", {})
    d_off = s.denoise - float(g.get("denoise", 0.42))
    c_off = s.cn_strength - float(g.get("cn_strength", 0.75))
    out = {}
    for name, p in base.items():
        p = dict(p)
        if s.per_material:
            p["denoise"] = round(min(0.85, max(0.10, float(p.get("denoise", 0.42)) + d_off)), 3)
            p["cn_strength"] = round(min(1.3, max(0.2, float(p.get("cn_strength", 0.75)) + c_off)), 3)
        else:
            p["denoise"], p["cn_strength"] = round(s.denoise, 3), round(s.cn_strength, 3)
        out[name] = p
    return out


def write_project_config(s: RemasterSettings) -> Path:
    work = work_dir_of(s)
    work.mkdir(parents=True, exist_ok=True)
    if s.model_family == "custom":
        wf, wf_edges = s.workflow, s.workflow_edges
    else:
        wf, wf_edges = FAMILIES[s.model_family]
    bc7 = s.dds_format == "bc7"
    sections = {
        "paths": {"inputs": [str(Path(s.input_dir).resolve())], "work": "./work"},
        "output": {"scale": int(s.scale), "max_side": int(s.max_side), "dds_format": s.dds_format,
                   # sans génération automatique : seulement si l'original avait des mipmaps
                   "mipmaps": "always" if s.mipmaps else "match"},
        "provider": {"name": s.provider},
        "comfyui": {"server": s.server, "workflow": wf, "workflow_edges": wf_edges, "checkpoint": s.checkpoint,
                    "lora": s.lora, "lora_strength": float(s.lora_strength)},
        "routing": {"enabled": bool(s.route_ui)},
        "ui_upscale": {"realesrgan_exe": s.realesrgan_exe},
        "local": {"esrgan_exe": s.realesrgan_exe},
        "qc": {"enabled": bool(s.qc), "auto_fix": bool(s.qc_autofix), "encoder": s.encoder,
               "texconv_exe": s.texconv_exe, "target_api": "d3d11" if bc7 else "d3d9", "allow_bc7": bc7},
        "materials": material_presets(s),
    }
    rules = []
    if not s.seamless:
        rules.append({"match": "*", "tile": False})
    elif s.force_seamless:
        rules.append({"match": "*", "tile": True})
    path = work / "texforge.toml"
    path.write_text(_toml(sections, rules), encoding="utf-8")
    return path


# ------------------------------------------------------------------ progression

class ProgressBus(EventBus):
    """Bus d'événements qui calcule aussi l'avancement global du lot (événement « progress ») :
    l'interface n'a qu'à l'afficher. Poids des étapes : scan 10 %, génération 60 %, export 30 %."""

    WEIGHTS = {"scan": 0.10, "generate": 0.60, "export": 0.30}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage = "scan"
        self.done = 0
        self.total = 0
        self.t_stage = time.monotonic()
        self._last = 0.0
        self.fatal: str | None = None        # workflow refusé par ComfyUI : la suite n'aurait pas de sens

    def set_stage(self, stage: str) -> None:
        self.stage, self.done, self.total, self.t_stage = stage, 0, 0, time.monotonic()
        self._progress(force=True)

    def scan_progress(self, i: int, n: int) -> None:
        self.done, self.total = i, n
        self._progress()

    def emit(self, event: str, **data) -> None:
        super().emit(event, **data)
        if event == "failed" and data.get("kind") == "workflow" and self.fatal is None:
            self.fatal = str(data.get("message", "workflow refusé"))
        if event == "run_start":
            self.total, self.done, self.t_stage = int(data.get("total", 0)), 0, time.monotonic()
        elif event in ("ui_done", "local_done", "comfy_done", "fallback") or \
                (event == "failed" and self.stage == "generate"):
            self.done += 1
        elif event == "qc_start":
            self.total, self.done, self.t_stage = int(data.get("total", 0)), 0, time.monotonic()
        elif event == "qc_done":
            self.done = int(data.get("index", self.done + 1))
        else:
            return
        self._progress(force=self.done >= self.total)

    def _progress(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < 0.2:
            return
        self._last = now
        done, total = min(self.done, self.total), self.total
        frac = done / total if total else 0.0
        before = sum(self.WEIGHTS[s] for s in STAGES[:STAGES.index(self.stage)])
        overall = before + self.WEIGHTS[self.stage] * frac
        eta = None
        if done >= 2 and total > done:
            eta = round((now - self.t_stage) / done * (total - done))
        super().emit("progress", fraction=round(min(overall, 1.0), 4), stage=self.stage, done=done, total=total,
                     eta_s=eta)

    def finish(self) -> None:
        super().emit("progress", fraction=1.0, stage="export", done=self.total, total=self.total, eta_s=0)


class _Stopped(Exception):
    pass


# ------------------------------------------------------------------ chaîne complète

def run_remaster(s: RemasterSettings, bus: ProgressBus) -> int:
    from .ddsbuild import EncoderError
    from .orchestrator import Orchestrator
    from .pipeline import open_state, scan
    from .postprocess import PostProcessor, items_from_state
    from .providers import FatalError

    t0 = time.monotonic()
    cfg = load_config(write_project_config(s))
    stop_file = Path(cfg["_work"]) / "STOP"
    gen: dict = {}
    qc_sum: dict = {}

    def end(code: int, message: str = "") -> int:
        status = {EXIT_OK: "ok", EXIT_ERRORS: "errors", EXIT_FATAL: "fatal", EXIT_STOPPED: "stopped"}[code]
        if code in (EXIT_OK, EXIT_ERRORS):
            bus.finish()
        parts = []
        if gen:
            parts.append(f"{gen.get('ui', 0)} UI, {gen.get('local', 0)} local, {gen.get('ai', 0)} ComfyUI, "
                         f"{gen.get('fallback', 0)} replis, {gen.get('errors', 0)} échecs")
        if qc_sum:
            parts.append(f"QC : {qc_sum.get('validated', 0)} validées, {qc_sum.get('corrected', 0)} corrigées, "
                         f"{qc_sum.get('rejected', 0)} rejetées — {qc_sum.get('written', 0)} fichier(s) écrit(s)")
        bus.emit("pipeline_end", code=code, status=status, message=message, summary=" | ".join(parts),
                 generation=gen, qc=qc_sum, report=qc_sum.get("report"), output=str(Path(s.output_dir).resolve()),
                 work=str(work_dir_of(s)), seconds=round(time.monotonic() - t0, 1))
        try:
            stop_file.unlink()
        except FileNotFoundError:
            pass
        return code

    family = s.model_family if s.model_family != "custom" else f"personnalisé ({Path(s.workflow).name})"
    bus.emit("pipeline_start", input=str(Path(s.input_dir).resolve()), output=str(Path(s.output_dir).resolve()),
             work=str(work_dir_of(s)), provider=s.provider, model=family, checkpoint=s.checkpoint or "(workflow)",
             lora=s.lora or "", config=str(Path(cfg["_base"]) / "texforge.toml"))

    # 1. scan et routage
    bus.set_stage("scan")
    bus.emit("pipeline_stage", stage="scan", index=1, total=len(STAGES), label=STAGE_LABELS["scan"])

    def on_scan(i: int, n: int, name: str) -> None:
        if stop_file.exists():
            raise _Stopped()
        bus.scan_progress(i, n)

    try:
        counts = scan(cfg, log=lambda m: bus.emit("info", message=f"[INFO] {m.strip()}"), on_progress=on_scan)
    except _Stopped:
        bus.emit("stopped", message="arrêt demandé pendant le scan")
        return end(EXIT_STOPPED)
    routes = ", ".join(f"{counts.get('route_' + r, 0)} {lbl}" for r, lbl in
                       (("ui", "UI"), ("comfyui", "ComfyUI"), ("local", "local"), ("keep", "gardées")))
    bus.emit("info", message=f"[INFO] {counts.get('seen', 0)} texture(s) analysée(s) — {counts['new']} nouvelle(s), "
                             f"{counts['changed']} modifiée(s), {counts['errors']} illisible(s) — routage : {routes}")

    # 2. génération
    if stop_file.exists():
        bus.emit("stopped", message="arrêt demandé avant la génération")
        return end(EXIT_STOPPED)
    bus.set_stage("generate")
    bus.emit("pipeline_stage", stage="generate", index=2, total=len(STAGES), label=STAGE_LABELS["generate"])
    orch = Orchestrator(cfg, bus, provider_name=s.provider)
    try:
        gen = orch.run(limit=s.limit or None, redo_stale=s.redo_stale)
    except FatalError as exc:
        bus.emit("failed", name="pipeline", stage="generate", kind="fatal", message=str(exc))
        return end(EXIT_FATAL, str(exc))
    if bus.fatal:
        # nœud ou modèle absent côté ComfyUI : exporter d'anciens résultats induirait en erreur
        return end(EXIT_FATAL, f"workflow refusé par ComfyUI ({bus.fatal.splitlines()[0][:160]}) — vérifie le "
                               f"checkpoint, la LoRA et les nœuds personnalisés, puis relance")
    if orch.runner.stop.is_set():
        return end(EXIT_STOPPED)

    # 3. contrôle qualité et export
    bus.set_stage("export")
    bus.emit("pipeline_stage", stage="export", index=3, total=len(STAGES), label=STAGE_LABELS["export"])
    out_dir = Path(s.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = open_state(cfg)
    try:
        pp = PostProcessor(cfg, bus, target=s.target, out_dir=out_dir, qc=s.qc, fix=s.qc_autofix,
                           encoder=s.encoder or None, state=state)
        rep = pp.run(items_from_state(cfg, state))
    except EncoderError as exc:
        bus.emit("failed", name="pipeline", stage="export", kind="fatal", message=str(exc))
        return end(EXIT_FATAL, str(exc))
    finally:
        state.close()
    qc_sum = rep["summary"]
    if qc_sum.get("stopped"):
        return end(EXIT_STOPPED)
    failed = gen.get("errors", 0) + qc_sum.get("error", 0)
    return end(EXIT_ERRORS if failed else EXIT_OK)


# ------------------------------------------------------------------ ligne de commande

def main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="texforge remaster", description="chaîne complète : scan, génération, QC, DDS")
    p.add_argument("--settings", help="fichier de réglages JSON (écrit par l'interface graphique)")
    p.add_argument("--input", help="dossier des textures extraites (remplace celui des réglages)")
    p.add_argument("--output", help="dossier de sortie DDS (remplace celui des réglages)")
    p.add_argument("--provider", choices=["comfyui", "mock"])
    p.add_argument("--json-events", action="store_true", help="un événement JSON par ligne sur la sortie standard")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    data = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else {}
    s = RemasterSettings.from_dict(data)
    if a.input:
        s.input_dir = a.input
    if a.output:
        s.output_dir = a.output
    if a.provider:
        s.provider = a.provider
    errs = s.validate()
    if errs:
        for e in errs:
            print(f"✗ {e}", file=sys.stderr)
        return EXIT_FATAL
    work = work_dir_of(s)
    work.mkdir(parents=True, exist_ok=True)
    (work / "remaster_settings.json").write_text(json.dumps(s.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    bus = ProgressBus(work / "work", json_stdout=a.json_events, console=not a.quiet)
    try:
        return run_remaster(s, bus)
    except KeyboardInterrupt:
        return EXIT_STOPPED
    finally:
        bus.close()
