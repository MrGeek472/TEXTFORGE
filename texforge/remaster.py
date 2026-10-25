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

from . import __version__
from .config import DEFAULTS, load_config
from .events import EventBus
from .i18n import N_, tr

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = {
    # famille : (workflow principal, workflow pour les matériaux à joints)
    "sdxl": ("workflows/texforge_sdxl_tile_api.json", "workflows/texforge_sdxl_tile_canny_api.json"),
    "sd15": ("workflows/texforge_sd15_tile_api.json", ""),
    # Z-Image Turbo (UNet + Qwen3 4B + VAE séparés) avec son ControlNet Tile (« model patch ») ; sans ce
    # fichier, TEXFORGE retire le ControlNet du workflow et passe en img2img doux
    "zimage": ("workflows/texforge_zimage_turbo_tile_api.json", ""),
}
# Échelle du denoise des préréglages (réglés pour SDXL + ControlNet Tile) : avec / sans ControlNet Tile.
# Z-Image + Tile supporte un denoise plus fort ; sans ControlNet, il redessinerait les formes.
DENOISE_SCALE = {"zimage": (1.3, 0.75)}
STAGES = ("scan", "generate", "export")
STAGE_LABELS = {"scan": N_("Scan et routage"), "generate": N_("Génération (UI, local, ComfyUI)"),
                "export": N_("Contrôle qualité et export DDS")}          # tr(STAGE_LABELS[…]) à l'affichage
EXIT_OK, EXIT_ERRORS, EXIT_FATAL, EXIT_STOPPED = 0, 1, 2, 3


# ------------------------------------------------------------------ réglages

@dataclass
class RemasterSettings:
    input_dir: str = ""
    output_dir: str = ""
    work_dir: str = ""              # vide : dossier de projet par défaut (données locales de l'utilisateur)
    target: str = "specialk"        # specialk (DDS) | loose (DDS, chemins conservés : Bethesda, mods) | remix | png
    # moteur IA
    provider: str = "comfyui"       # comfyui | esrgan (sans ComfyUI : Real-ESRGAN réaliste) | mock (simulation)
    server: str = "http://127.0.0.1:8188"
    model_family: str = "sdxl"      # sdxl | sd15 | zimage | custom
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
    force_ai: bool = False          # tout par ComfyUI (sauf normal maps, cartes de données, polices)
    tiled: bool = True              # ComfyUI : génération par morceaux à la taille finale
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
    esrgan_detail: float = 1.0      # mode esrgan : micro-détail (0 = aucun grain ajouté, 1 = dose mesurée)
    # archives BA2 (Fallout 4, Starfield) : textures extraites, motifs séparés par « ; » (vide : toutes)
    ba2_filter: str = "*_d.dds"

    @classmethod
    def from_dict(cls, d: dict) -> "RemasterSettings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        errs = []
        if not self.input_dir or not Path(self.input_dir).is_dir():
            errs.append(tr("dossier d'entrée introuvable : {path}", path=self.input_dir or tr("(vide)")))
        if not self.output_dir:
            errs.append(tr("dossier de sortie non renseigné"))
        elif Path(self.input_dir).resolve() == Path(self.output_dir).resolve():
            errs.append(tr("le dossier de sortie doit être différent du dossier d'entrée"))
        if self.model_family == "custom" and not Path(self.workflow).is_file():
            errs.append(tr("workflow personnalisé introuvable : {path}", path=self.workflow or tr("(vide)")))
        if self.model_family not in (*FAMILIES, "custom"):
            errs.append(tr("famille de modèle inconnue : {family}", family=self.model_family))
        if self.provider not in ("comfyui", "esrgan", "mock"):
            errs.append(tr("moteur inconnu : {name}", name=self.provider))
        if self.target not in ("specialk", "loose", "remix", "png"):
            errs.append(tr("cible inconnue : {target}", target=self.target))
        if self.dds_format not in ("match", "auto", "dxt1", "dxt5", "bc7", "bgra8"):
            errs.append(tr("format DDS inconnu : {fmt}", fmt=self.dds_format))
        # ces outils sont lancés des centaines de fois : jamais un autre programme (l'exe du jeu…)
        if self.texconv_exe and not Path(self.texconv_exe).stem.lower().startswith("texconv"):
            errs.append(tr("le champ texconv désigne « {name} », pas texconv.exe : vide-le ou choisis texconv.exe",
                           name=Path(self.texconv_exe).name))
        if self.realesrgan_exe and "realesrgan" not in Path(self.realesrgan_exe).stem.lower():
            errs.append(tr("le champ Real-ESRGAN désigne « {name} », pas realesrgan-ncnn-vulkan.exe",
                           name=Path(self.realesrgan_exe).name))
        return errs


def default_work_dir(input_dir: str) -> Path:
    """Dossier de projet propre à un dossier d'entrée, hors du jeu et hors du dossier d'injection :
    %LOCALAPPDATA%\\TEXFORGE\\projets\\<nom>-<empreinte> (ou ~/.local/share/texforge/projets)."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "TEXFORGE"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "texforge"
    src = str(Path(input_dir).resolve()) if input_dir else "sans-dossier"
    digest = hashlib.sha1(src.lower().encode()).hexdigest()[:8]
    projects = base / "projets"
    legacy = projects / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', Path(src).name or 'projet')[:40]}-{digest}"
    if legacy.is_dir():                     # projet existant : on le garde (état, résultats, reprise)
        return legacy
    # dump Special K / capture Remix : le dossier s'appelle toujours « textures », on nomme d'après le jeu
    from .detect import game_dir_of, game_name
    game = game_dir_of(src)
    name = game_name(game / "x") if game is not None else Path(src).name
    if game is None and Path(src).name.lower() == "data":       # Fallout 4, Starfield… : nommé d'après le jeu
        name = Path(src).parent.name or name
    return projects / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', name or 'projet')[:40]}-{digest}"


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


def usable_esrgan(raw: str) -> str:
    """Real-ESRGAN réglé s'il existe encore ; sinon celui que TEXFORGE trouve (tools\\, PATH) ; sinon vide.
    Un chemin périmé (dossier déplacé) ne doit pas faire retomber en silence sur Lanczos."""
    p = Path(raw) if raw else None
    if p is not None and p.is_file() and "realesrgan" in p.stem.lower():
        return str(p)
    from .esrgan import find_realesrgan
    found = find_realesrgan()
    return str(found) if found else ""


def write_project_config(s: RemasterSettings) -> Path:
    work = work_dir_of(s)
    work.mkdir(parents=True, exist_ok=True)
    if s.model_family == "custom":
        wf, wf_edges = s.workflow, s.workflow_edges
    else:
        wf, wf_edges = FAMILIES[s.model_family]
    esrgan_exe = usable_esrgan(s.realesrgan_exe)
    bc7 = s.dds_format == "bc7"
    d3d11 = bc7 or s.target == "loose"          # Fallout 4 et Starfield : Direct3D 11, BC7 lisible
    sections = {
        "paths": {"inputs": [str(Path(s.input_dir).resolve())], "work": "./work"},
        "output": {"scale": int(s.scale), "max_side": int(s.max_side), "dds_format": s.dds_format,
                   # sans génération automatique : seulement si l'original avait des mipmaps
                   "mipmaps": "always" if s.mipmaps else "match"},
        "provider": {"name": s.provider},
        "comfyui": {"server": s.server, "workflow": wf, "workflow_edges": wf_edges, "checkpoint": s.checkpoint,
                    "lora": s.lora, "lora_strength": float(s.lora_strength), "tiled": bool(s.tiled),
                    "denoise_scale": DENOISE_SCALE.get(s.model_family, (1.0, 1.0))[0],
                    "denoise_scale_no_cn": DENOISE_SCALE.get(s.model_family, (1.0, 1.0))[1],
                    "tile_size": 512 if s.model_family == "sd15" else 1024},
        "routing": {"enabled": bool(s.route_ui), "force_ai": bool(s.force_ai)},
        "ui_upscale": {"realesrgan_exe": esrgan_exe},
        "local": {"esrgan_exe": esrgan_exe},
        "esrgan": {"detail": float(s.esrgan_detail)},
        "qc": {"enabled": bool(s.qc), "auto_fix": bool(s.qc_autofix), "encoder": s.encoder,
               "texconv_exe": s.texconv_exe, "target_api": "d3d11" if d3d11 else "d3d9", "allow_bc7": d3d11},
        "ba2": {"filter": s.ba2_filter},
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
            self.fatal = str(data.get("message", tr("workflow refusé")))
        if event == "run_start":
            self.total, self.done, self.t_stage = int(data.get("total", 0)), 0, time.monotonic()
        elif event in ("ui_done", "local_done", "comfy_done", "esrgan_done", "fallback") or \
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
    from .ba2 import Ba2Error, data_folder, extract_textures
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
            parts.append(tr("{ui} UI, {local} local, {ai} {engine}, {fallback} replis, {errors} échecs",
                            ui=gen.get("ui", 0), local=gen.get("local", 0), ai=gen.get("ai", 0),
                            engine="Real-ESRGAN" if s.provider == "esrgan" else "ComfyUI" if s.provider == "comfyui"
                            else tr("simulation"), fallback=gen.get("fallback", 0), errors=gen.get("errors", 0)))
        if qc_sum:
            parts.append(tr("QC : {validated} validées, {corrected} corrigées, {rejected} rejetées — "
                            "{written} fichier(s) écrit(s)", validated=qc_sum.get("validated", 0),
                            corrected=qc_sum.get("corrected", 0), rejected=qc_sum.get("rejected", 0),
                            written=qc_sum.get("written", 0)))
        bus.emit("pipeline_end", code=code, status=status, message=message, summary=" | ".join(parts),
                 generation=gen, qc=qc_sum, report=qc_sum.get("report"), output=str(Path(s.output_dir).resolve()),
                 work=str(work_dir_of(s)), seconds=round(time.monotonic() - t0, 1))
        try:
            stop_file.unlink()
        except FileNotFoundError:
            pass
        return code

    family = s.model_family if s.model_family != "custom" else tr("personnalisé ({name})", name=Path(s.workflow).name)
    checkpoint = s.checkpoint or "(workflow)"
    if s.provider == "esrgan":
        family, checkpoint = "realesrgan-x4plus + realesr-animevideov3", "—"
    bus.emit("pipeline_start", input=str(Path(s.input_dir).resolve()), output=str(Path(s.output_dir).resolve()),
             work=str(work_dir_of(s)), provider=s.provider, model=family, checkpoint=checkpoint,
             lora=s.lora or "", config=str(Path(cfg["_base"]) / "texforge.toml"),
             version=__version__, program=str(Path(__file__).resolve().parent.parent))

    if s.provider == "comfyui" and s.tiled and not cfg["local"].get("esrgan_exe"):
        bus.emit("info", message=tr("[INFO] ⚠ Real-ESRGAN introuvable : les morceaux partiront d'un agrandissement "
                                    "Lanczos flou et l'IA ajoutera peu de détail. Bouton « Installer Real-ESRGAN »."))
    elif s.realesrgan_exe and cfg["local"].get("esrgan_exe") != s.realesrgan_exe:
        bus.emit("info", message=tr("[INFO] Real-ESRGAN réglé introuvable ({old}) : {new} utilisé.",
                                    old=s.realesrgan_exe, new=cfg["local"]["esrgan_exe"]))

    # 1. scan et routage
    bus.set_stage("scan")
    bus.emit("pipeline_stage", stage="scan", index=1, total=len(STAGES), label=tr(STAGE_LABELS["scan"]))
    ba2_dir = data_folder(s.input_dir)
    if ba2_dir is not None:
        # Fallout 4, Starfield : textures lues dans les archives BA2 (les fichiers séparés ne sont pas repris)
        dest = work_dir_of(s) / "ba2"            # hors de work/ : le scan ignore ce que TEXFORGE y produit
        bus.emit("info", message=tr("[BA2] archives de textures dans {path} — filtre : {filter}", path=ba2_dir,
                                    filter=s.ba2_filter or tr("toutes")))
        try:
            ex = extract_textures(ba2_dir, dest, include=s.ba2_filter, scale=s.scale, max_side=s.max_side,
                                  limit=s.limit, log=lambda m: bus.emit("info", message=m),
                                  on_progress=lambda i, n: bus.scan_progress(i, n))
        except (Ba2Error, OSError) as exc:
            bus.emit("failed", name="pipeline", stage="scan", kind="fatal", message=str(exc))
            return end(EXIT_FATAL, str(exc))
        bus.emit("info", message=tr("[BA2] {selected} texture(s) à traiter sur {listed} ({matched} selon le filtre ; "
                                    "{maxed} déjà à la taille maximale, {loose} remplacées par un fichier séparé, "
                                    "{cubemaps} cubemap(s) ignorées) — {written} extraite(s), {kept} déjà là",
                                    **ex))
        if not ex["selected"]:
            return end(EXIT_FATAL, tr("aucune texture des archives BA2 ne correspond au filtre « {filter} »",
                                      filter=s.ba2_filter))
        cfg["_inputs"] = [str(dest)]

    def on_scan(i: int, n: int, name: str) -> None:
        if stop_file.exists():
            raise _Stopped()
        bus.scan_progress(i, n)

    try:
        counts = scan(cfg, log=lambda m: bus.emit("info", message=f"[INFO] {m.strip()}"), on_progress=on_scan)
    except _Stopped:
        bus.emit("stopped", message=tr("arrêt demandé pendant le scan"))
        return end(EXIT_STOPPED)
    routes = ", ".join(f"{counts.get('route_' + r, 0)} {tr(lbl)}" for r, lbl in
                       (("ui", "UI"), ("comfyui", "ComfyUI"), ("local", N_("local")), ("keep", N_("gardées"))))
    bus.emit("info", message=tr("[INFO] {seen} texture(s) analysée(s) — {new} nouvelle(s), {changed} modifiée(s), "
                                "{errors} illisible(s) — routage : {routes}", seen=counts.get("seen", 0),
                                new=counts["new"], changed=counts["changed"], errors=counts["errors"], routes=routes))
    if counts.get("skin_local"):
        bus.emit("info", message=tr("[INFO] {n} texture(s) de peau ou de visage : agrandie(s) sans IA générative "
                                    "(l'IA redessinerait les traits)", n=counts["skin_local"]))
    if counts.get("missing"):
        bus.emit("info", message=tr("[INFO] {n} texture(s) d'un scan précédent introuvable(s) (fichier supprimé ou "
                                    "renommé) : mise(s) de côté", n=counts["missing"]))
    if counts.get("seen", 0) == 0:
        # rien à traiter : ne pas annoncer un succès ni écrire un rapport vide dans le dossier de sortie
        from .detect import check_input, empty_input_help
        why = empty_input_help(s.input_dir, check_input(s.input_dir, stop_at_image=False))
        bus.emit("failed", name="pipeline", stage="scan", kind="fatal", message=why)
        return end(EXIT_FATAL, tr("aucune texture à traiter dans le dossier d'entrée"))

    # 2. génération
    if stop_file.exists():
        bus.emit("stopped", message=tr("arrêt demandé avant la génération"))
        return end(EXIT_STOPPED)
    bus.set_stage("generate")
    bus.emit("pipeline_stage", stage="generate", index=2, total=len(STAGES), label=tr(STAGE_LABELS["generate"]))
    orch = Orchestrator(cfg, bus, provider_name=s.provider)
    try:
        gen = orch.run(limit=s.limit or None, redo_stale=s.redo_stale)
    except FatalError as exc:
        bus.emit("failed", name="pipeline", stage="generate", kind="fatal", message=str(exc))
        return end(EXIT_FATAL, str(exc))
    if bus.fatal:
        # nœud ou modèle absent côté ComfyUI : exporter d'anciens résultats induirait en erreur
        return end(EXIT_FATAL, tr("workflow refusé par ComfyUI ({why}) — vérifie le checkpoint, la LoRA et les "
                                  "nœuds personnalisés, puis relance", why=bus.fatal.splitlines()[0][:160]))
    if orch.runner.stop.is_set():
        return end(EXIT_STOPPED)

    # 3. contrôle qualité et export
    bus.set_stage("export")
    bus.emit("pipeline_stage", stage="export", index=3, total=len(STAGES), label=tr(STAGE_LABELS["export"]))
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
    if gen.get("server_unreachable"):          # rien n'est passé par ComfyUI : ne pas afficher un succès
        return end(EXIT_ERRORS, tr("ComfyUI injoignable : matières agrandies sans IA générative"))
    return end(EXIT_ERRORS if failed else EXIT_OK)


# ------------------------------------------------------------------ ligne de commande

def main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="texforge remaster", description="chaîne complète : scan, génération, QC, DDS")
    p.add_argument("--settings", help="fichier de réglages JSON (écrit par l'interface graphique)")
    p.add_argument("--input", help="dossier des textures extraites (remplace celui des réglages)")
    p.add_argument("--output", help="dossier de sortie DDS (remplace celui des réglages)")
    p.add_argument("--provider", choices=["comfyui", "esrgan", "mock"])
    p.add_argument("--target", choices=["specialk", "loose", "remix", "png"])
    p.add_argument("--ba2-filter", help="archives BA2 : textures à extraire (motifs séparés par « ; », vide : toutes)")
    p.add_argument("--json-events", action="store_true", help="un événement JSON par ligne sur la sortie standard")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    data = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else {}
    s = RemasterSettings.from_dict(data)
    if a.input:
        # comme dans l'interface : dossier d'un jeu -> son dump Special K, son profil SKIF, ses textures
        from .detect import locate_textures
        loc = locate_textures(a.input)
        s.input_dir = loc.path if loc.kind != "none" else a.input
        if loc.changed and loc.kind != "none":
            print(tr("[INFO] Dossier de textures : {path} ({count} fichier(s))", path=loc.path, count=loc.count),
                  file=sys.stderr)
        if loc.inject and not a.output and not s.output_dir:
            s.output_dir = loc.inject
        if loc.kind == "ba2":
            from .ba2 import mod_output_for
            if not a.target:
                s.target = "loose"
            if not a.output and not s.output_dir:
                s.output_dir = str(mod_output_for(loc.path))
    if a.output:
        s.output_dir = a.output
    if a.provider:
        s.provider = a.provider
    if a.target:
        s.target = a.target
    if a.ba2_filter is not None:
        s.ba2_filter = a.ba2_filter
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
