"""TEXFORGE — interface graphique (PySide6).

    pythonw app_gui.py          (ou texforge_gui.bat)

Architecture
------------
L'interface ne fait aucun calcul lourd. Au lancement, elle écrit un fichier de réglages et démarre
la chaîne complète (`python -m texforge remaster --settings … --json-events`) dans un PROCESSUS
séparé, supervisé par un QThread :

    MainWindow (thread Qt)  ◄── signaux ──  PipelineWorker (QThread)  ◄── JSON/ligne ──  processus texforge
          │                                         │                                    scan → génération →
          │  STOP (arrêt propre)                    └─ fichier STOP / arrêt forcé ──────► QC → DDS
          └─ QThreadPool : test du serveur ComfyUI, liste des modèles, galerie, interruption ComfyUI

Pourquoi un processus plutôt que le calcul dans le QThread : en Python, un thread partage le GIL
avec l'interface ; un processus séparé garde l'interface fluide quoi qu'il arrive (analyse numpy,
DDS, WebSocket), un plantage du traitement ne ferme pas la fenêtre, et l'arrêt forcé est toujours
possible. Le QThread se contente de lire les événements et de les transmettre par signaux.

Annulation en deux temps :
  1. « ANNULER » : fichier STOP lu entre deux textures ; les textures en cours se terminent,
     l'état est enregistré, relancer reprend où l'on s'était arrêté ;
  2. « FORCER L'ARRÊT » : arrêt du processus et de ses enfants, puis interruption de la file
     ComfyUI (sinon le GPU continuerait les prompts déjà envoyés).
"""

from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("PySide6") is None:          # message clair plutôt qu'une trace Python
    sys.stderr.write("PySide6 manque / PySide6 is missing : pip install -r requirements.txt (ou/or pip install PySide6)\n")
    raise SystemExit(1)

from PySide6.QtCore import (QDateTime, QLibraryInfo, QLocale, QObject, QPointF, QRunnable, QSettings, Qt,  # noqa: E402
                            QThread, QThreadPool, QTimer, QTranslator, QUrl, Signal, Slot)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QFontDatabase, QGuiApplication, QIcon,  # noqa: E402
                           QPainter, QPalette, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,  # noqa: E402
                               QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget)

from texforge import __version__  # noqa: E402
from texforge.ba2 import mod_output_for  # noqa: E402
from texforge.detect import Candidate, Located, check_input, empty_input_help, inject_for, locate_textures  # noqa: E402
from texforge.detect import detect as find_texture_dirs  # noqa: E402
from texforge.esrgan import INSTALL_DIR, RELEASES, find_realesrgan, install_realesrgan, platform_key  # noqa: E402
from texforge.events import _fmt as format_event  # noqa: E402
from texforge.i18n import LANGUAGES, N_, get_lang, set_lang, tr  # noqa: E402
from texforge.i18n import detect as detect_language  # noqa: E402
from texforge.qc import LABELS as QC_LABELS  # noqa: E402
from texforge.remaster import (EXIT_ERRORS, EXIT_FATAL, EXIT_OK, EXIT_STOPPED, STAGE_LABELS,  # noqa: E402
                               RemasterSettings, material_presets, work_dir_of)

APP_NAME = "TEXFORGE"

# ===================================================================== thème

C = {
    "bg0": "#14161b", "bg1": "#1b1e24", "bg2": "#23272f", "bg3": "#2c313b", "line": "#343a46",
    "text": "#d9dce3", "muted": "#8a93a3", "dim": "#626a78",
    "accent": "#ff8a1f", "accent_hi": "#ffa24d", "accent_lo": "#c96510",
    "blue": "#4aa8ff", "cyan": "#45d0db", "violet": "#b38cff", "green": "#3fcf8e", "lime": "#b5d94a",
    "yellow": "#f2c14e", "orange": "#ff9a4d", "red": "#ff5d5d",
}

QSS = f"""
* {{ color: {C['text']}; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {C['bg0']}; }}
QWidget#central, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {C['bg0']}; }}
QToolTip {{ background: {C['bg3']}; color: {C['text']}; border: 1px solid {C['line']}; padding: 6px; }}

QGroupBox {{
    background: {C['bg1']}; border: 1px solid {C['line']}; border-radius: 6px;
    margin-top: 14px; padding: 12px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: {C['accent']}; font-weight: 600; letter-spacing: 1px;
}}
QLabel#hint {{ color: {C['muted']}; font-size: 9pt; }}
QLabel#warn {{ color: {C['yellow']}; font-size: 9pt; }}
QLabel#value {{ color: {C['accent_hi']}; font-family: Consolas, 'Cascadia Mono', monospace; min-width: 38px; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {C['bg2']}; border: 1px solid {C['line']}; border-radius: 4px; padding: 4px 6px;
    selection-background-color: {C['accent_lo']};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {C['accent']}; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {C['dim']}; }}
QComboBox QAbstractItemView {{ background: {C['bg2']}; border: 1px solid {C['line']};
    selection-background-color: {C['bg3']}; selection-color: {C['accent_hi']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}

QPushButton, QToolButton {{
    background: {C['bg3']}; border: 1px solid {C['line']}; border-radius: 4px; padding: 5px 12px;
}}
QPushButton:hover, QToolButton:hover {{ border-color: {C['accent']}; color: {C['accent_hi']}; }}
QPushButton:pressed {{ background: {C['bg2']}; }}
QPushButton:disabled {{ color: {C['dim']}; border-color: {C['bg3']}; }}

QPushButton#launch {{
    font-size: 15pt; font-weight: 800; letter-spacing: 2px; padding: 14px; border: none; border-radius: 6px;
    color: #1a1006;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {C['accent_hi']}, stop:1 {C['accent_lo']});
}}
QPushButton#launch:hover {{ background: {C['accent_hi']}; color: #1a1006; }}
QPushButton#launch[mode="stop"] {{
    color: white; background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0484b, stop:1 #9e2226);
}}
QPushButton#launch[mode="kill"] {{ color: white; background: #6f1215; border: 2px solid {C['red']}; }}
QPushButton#launch:disabled {{ background: {C['bg3']}; color: {C['dim']}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid {C['line']};
    background: {C['bg2']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QCheckBox:disabled {{ color: {C['dim']}; }}

QSlider::groove:horizontal {{ height: 4px; background: {C['bg3']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C['text']}; width: 14px; height: 14px; margin: -6px 0;
    border-radius: 7px; border: 2px solid {C['accent']}; }}
QSlider::handle:horizontal:hover {{ background: {C['accent_hi']}; }}
QSlider::sub-page:horizontal:disabled {{ background: {C['line']}; }}
QSlider::handle:horizontal:disabled {{ background: {C['bg3']}; border-color: {C['line']}; }}

QProgressBar {{
    background: {C['bg2']}; border: 1px solid {C['line']}; border-radius: 5px; height: 22px;
    text-align: center; font-weight: 600;
}}
QProgressBar::chunk {{ border-radius: 4px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {C['accent_lo']}, stop:1 {C['accent_hi']}); }}

QPlainTextEdit#console {{
    background: #0e1013; border: 1px solid {C['line']}; border-radius: 4px; padding: 6px;
    selection-background-color: {C['bg3']};
}}
QTableWidget {{ background: {C['bg1']}; gridline-color: {C['line']}; border: 1px solid {C['line']};
    alternate-background-color: {C['bg2']}; }}
QHeaderView::section {{ background: {C['bg3']}; border: none; border-right: 1px solid {C['line']}; padding: 5px; }}
QSplitter::handle {{ background: {C['bg0']}; width: 6px; }}
QScrollBar:vertical {{ background: {C['bg0']}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C['bg3']}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: {C['bg0']}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {C['bg3']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QFrame#header {{ background: {C['bg1']}; border-bottom: 1px solid {C['line']}; }}
QLabel#title {{ font-size: 17pt; font-weight: 800; letter-spacing: 3px; color: {C['text']}; }}
QLabel#subtitle {{ color: {C['muted']}; }}
QLabel#pill {{ border-radius: 9px; padding: 3px 10px; font-size: 9pt; font-weight: 600;
    background: {C['bg3']}; color: {C['muted']}; }}
QLabel#pill[state="ok"] {{ background: #12352a; color: {C['green']}; }}
QLabel#pill[state="ko"] {{ background: #3a1517; color: {C['red']}; }}
QLabel#pill[state="busy"] {{ background: #3a2a10; color: {C['yellow']}; }}
QLabel#stage {{ border-radius: 4px; padding: 4px 10px; background: {C['bg2']}; color: {C['dim']}; font-weight: 600; }}
QLabel#stage[state="active"] {{ background: {C['accent_lo']}; color: white; }}
QLabel#stage[state="done"] {{ background: #12352a; color: {C['green']}; }}
QLabel#chip {{ background: {C['bg2']}; border: 1px solid {C['line']}; border-radius: 4px; padding: 3px 8px;
    font-family: Consolas, 'Cascadia Mono', monospace; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.Window, C["bg0"]), (QPalette.WindowText, C["text"]), (QPalette.Base, C["bg2"]),
                      (QPalette.AlternateBase, C["bg1"]), (QPalette.Text, C["text"]), (QPalette.Button, C["bg3"]),
                      (QPalette.ButtonText, C["text"]), (QPalette.Highlight, C["accent_lo"]),
                      (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, C["bg3"]),
                      (QPalette.ToolTipText, C["text"]), (QPalette.PlaceholderText, C["dim"])):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(QSS + _check_mark_qss())


def _check_mark_qss() -> str:
    """Coche des cases à cocher : dessinée une fois dans un PNG temporaire (une feuille de style Qt
    ne sait pas dessiner une coche elle-même)."""
    import tempfile
    path = Path(tempfile.gettempdir()) / "texforge_check.png"
    pm = QPixmap(32, 32)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = p.pen()
    pen.setColor(QColor("#1a1006"))
    pen.setWidth(5)
    p.setPen(pen)
    p.drawPolyline([QPointF(7, 17), QPointF(13, 23), QPointF(25, 9)])
    p.end()
    if not pm.save(str(path), "PNG"):
        return ""
    return f"QCheckBox::indicator:checked {{ image: url({path.as_posix()}); }}"


def app_icon() -> QIcon:
    """Icône dessinée à la volée (aucun fichier à livrer)."""
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(C["accent"]))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setBrush(QColor("#1a1006"))
    for x, y in ((14, 14), (36, 14), (14, 36)):          # damier : une texture, un morceau remasterisé
        p.drawRect(x, y, 14, 14)
    p.setBrush(QColor("#ffffff"))
    p.drawRect(36, 36, 14, 14)
    p.end()
    return QIcon(pm)


def mono_font(size: int = 10) -> QFont:
    for fam in ("Cascadia Mono", "Consolas", "JetBrains Mono", "DejaVu Sans Mono"):
        if fam in QFontDatabase.families():
            return QFont(fam, size)
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    f.setPointSize(size)
    return f


def repolish(w: QWidget) -> None:
    """Réapplique la feuille de style après changement d'une propriété dynamique."""
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


# ===================================================================== outils système

def python_for_child() -> str:
    """Interpréteur du processus de traitement. Avec pythonw.exe (lancement sans console), on prend
    python.exe à côté : ses flux standard sont fiables pour lire les événements."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def kill_tree(proc: subprocess.Popen) -> None:
    """Arrête le processus de traitement ET ses enfants (Real-ESRGAN, texconv…)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def http_json(url: str, data: dict | None = None, timeout: float = 4.0):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"},
                                 method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}


def combo_options(info: dict, node: str, field: str) -> list[str]:
    """Liste des valeurs d'un menu déroulant d'un nœud ComfyUI (anciens et nouveaux formats /object_info)."""
    try:
        spec = info[node]["input"]["required"][field]
    except (KeyError, TypeError):
        return []
    first = spec[0] if spec else None
    if isinstance(first, list):
        return [str(x) for x in first]
    if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return [str(x) for x in spec[1].get("options", [])]
    return []


def fmt_duration(s: float | None) -> str:
    if s is None:
        return "—"
    s = int(s)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def open_path(path: Path | str) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


# ===================================================================== travail en arrière-plan

class PipelineWorker(QObject):
    """Supervise le processus `texforge remaster`. Vit dans un QThread : la lecture bloquante de la
    sortie du processus n'atteint jamais la boucle d'événements de l'interface."""

    event = Signal(object)          # un événement JSON décodé (dict)
    stderr_line = Signal(str)       # avertissements et traces Python du processus
    finished = Signal(int)          # code de sortie

    def __init__(self, settings_file: Path, work_dir: Path) -> None:
        super().__init__()
        self.settings_file = settings_file
        self.stop_file = work_dir / "work" / "STOP"
        self.proc: subprocess.Popen | None = None
        self._killed = False
        self.lang = get_lang()                  # messages du traitement dans la langue de l'interface

    @Slot()
    def run(self) -> None:
        cmd = [python_for_child(), "-u", "-m", "texforge", "remaster", "--settings", str(self.settings_file),
               "--json-events", "--quiet"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", TEXFORGE_LANG=self.lang,
                   PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
        extra: dict = {}
        if os.name == "nt":
            extra["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | \
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            extra["start_new_session"] = True          # groupe de processus : arrêt forcé des enfants
        try:
            self.stop_file.unlink(missing_ok=True)
            self.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1, **extra)
        except OSError as exc:
            self.stderr_line.emit(tr("lancement impossible : {exc}", exc=exc))
            self.finished.emit(EXIT_FATAL)
            return
        pump = threading.Thread(target=self._pump_stderr, daemon=True)
        pump.start()
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.event.emit(json.loads(line))
            except json.JSONDecodeError:
                self.stderr_line.emit(line)
        code = self.proc.wait()
        pump.join(timeout=2)
        if self._killed and code not in (EXIT_OK, EXIT_ERRORS):
            code = EXIT_STOPPED
        self.finished.emit(code)

    def _pump_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            if line.strip():
                self.stderr_line.emit(line.rstrip())

    # appelés depuis le thread de l'interface
    def request_stop(self) -> None:
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text("stop", encoding="utf-8")

    def kill(self) -> None:
        self._killed = True
        if self.proc is not None:
            kill_tree(self.proc)


class _ProgressBridge(QObject):
    """Progression d'une tâche de fond (téléchargement) vers le thread de l'interface."""
    step = Signal(int, int)


class _TaskSignals(QObject):
    done = Signal(object)


class Task(QRunnable):
    """Petite tâche d'arrière-plan (QThreadPool) dont le résultat revient par signal."""

    def __init__(self, fn, *args) -> None:
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            res = self.fn(*self.args)
        except Exception as exc:          # le résultat porte l'erreur, l'interface décide
            res = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            self.signals.done.emit(res)
        except RuntimeError:              # fenêtre fermée pendant la tâche : plus personne à prévenir
            pass


def probe_comfyui(server: str) -> dict:
    """État du serveur et modèles disponibles (checkpoints, LoRA)."""
    base = server.rstrip("/")
    try:
        stats = http_json(f"{base}/system_stats", timeout=3)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "error": str(getattr(exc, "reason", exc))}
    dev = (stats.get("devices") or [{}])[0]
    out = {"ok": True, "version": (stats.get("system") or {}).get("comfyui_version", "?"),
           "gpu": re.sub(r"^cuda:\d+\s*", "", str(dev.get("name", "?"))).split(" : ")[0].strip(),
           "vram_free": dev.get("vram_free"), "vram_total": dev.get("vram_total"), "checkpoints": [], "loras": []}
    for node, field, key in (("CheckpointLoaderSimple", "ckpt_name", "checkpoints"), ("LoraLoader", "lora_name", "loras")):
        try:
            out[key] = combo_options(http_json(f"{base}/object_info/{node}", timeout=5), node, field)
        except (urllib.error.URLError, OSError, ValueError):
            pass
    return out


def interrupt_comfyui(server: str) -> str:
    base = server.rstrip("/")
    msgs = []
    for path, data, what in (("/interrupt", {}, tr("génération en cours interrompue")),
                             ("/queue", {"clear": True}, tr("file d'attente vidée"))):
        try:
            http_json(base + path, data, timeout=3)
            msgs.append(what)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            msgs.append(f"{path} : {getattr(exc, 'reason', exc)}")
    return "ComfyUI : " + ", ".join(msgs)


def build_gallery(work_dir: Path) -> dict:
    from texforge.config import load_config
    from texforge.review import build_review
    cfg = load_config(work_dir / "texforge.toml")
    return {"page": str(build_review(cfg))}


# ===================================================================== composants

class PathRow(QWidget):
    """Champ texte + bouton « Parcourir » (dossier ou fichier)."""

    changed = Signal(str)
    picked = Signal(str)            # chemin choisi avec « Parcourir » ou saisi puis validé

    def __init__(self, placeholder: str, *, file_filter: str | None = None, dialog_title: str = "",
                 extra: QWidget | None = None) -> None:
        super().__init__()
        self.file_filter = file_filter
        self.dialog_title = dialog_title
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        self.edit.editingFinished.connect(self._edited)
        btn = QPushButton(tr("Parcourir…"))
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        if extra is not None:
            lay.addWidget(extra)
        lay.addWidget(btn)

    def _browse(self) -> None:
        start = self.text() or str(Path.home())
        if self.file_filter:
            path, _ = QFileDialog.getOpenFileName(self, self.dialog_title, start, self.file_filter)
        else:
            path = QFileDialog.getExistingDirectory(self, self.dialog_title, start)
        if path:
            self.edit.setText(str(Path(path)))
            self.picked.emit(self.text())

    def _edited(self) -> None:
        if self.edit.isModified():                # saisi ou collé à la main (pas un setText)
            self.edit.setModified(False)
            self.picked.emit(self.text())

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str) -> None:
        self.edit.setText(t)


class ValueSlider(QWidget):
    """Curseur à valeur décimale, avec la valeur affichée en temps réel."""

    valueChanged = Signal(float)

    def __init__(self, lo: float, hi: float, step: float, value: float, decimals: int = 2) -> None:
        super().__init__()
        self.lo, self.step, self.decimals = lo, step, decimals
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, round((hi - lo) / step))
        self.slider.setPageStep(max(1, round(0.05 / step)))
        self.label = QLabel()
        self.label.setObjectName("value")
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._changed)
        self.setValue(value)
        self._changed()

    def _changed(self, *_):
        v = self.value()
        self.label.setText(f"{v:.{self.decimals}f}")
        self.valueChanged.emit(v)

    def value(self) -> float:
        return round(self.lo + self.slider.value() * self.step, 4)

    def setValue(self, v: float) -> None:
        self.slider.setValue(round((float(v) - self.lo) / self.step))


class LogConsole(QPlainTextEdit):
    """Console de suivi colorée. QPlainTextEdit (cousin optimisé de QTextEdit) : des dizaines de milliers
    de lignes sans ralentir, limite de blocs, défilement auto sauf si l'utilisateur remonte."""

    # étiquettes des deux langues : un journal peut mêler des lignes d'avant et d'après un changement
    COLORS = [("[ERREUR]", C["red"]), ("[ERROR]", C["red"]), ("[QC] ✘", C["orange"]), ("[QC] !", C["red"]),
              ("[QC] ✎", C["lime"]), ("[QC] ✔", C["green"]), ("[QC] ·", C["muted"]), ("[QC] =", C["blue"]),
              ("[UI]", C["cyan"]), ("[LOCAL]", "#5fb3a1"), ("[COMFYUI]", C["violet"]), ("[ESRGAN]", C["violet"]), ("[GPU]", C["yellow"]),
              ("[RETRY]", C["yellow"]), ("[REPLI]", C["orange"]), ("[FALLBACK]", C["orange"]), ("[DDS]", C["blue"]),
              ("[STDERR]", C["orange"]), ("⚠", C["yellow"]), ("══", C["accent_hi"]), ("==", C["text"]),
              ("──", C["muted"]), ("■", C["yellow"]), ("[INFO]", C["muted"])]
    ALERT = ("[ERREUR]", "[ERROR]", "[QC] ✘", "[QC] !", "[REPLI]", "[FALLBACK]", "[RETRY]", "[STDERR]", "⚠", "■",
             "══")

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("console")
        self.setReadOnly(True)
        self.setFont(mono_font(10))
        self.setMaximumBlockCount(20000)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.lines: deque[str] = deque(maxlen=100000)
        self.only_alerts = False

    def _color(self, text: str) -> str:
        s = text.lstrip()
        for prefix, col in self.COLORS:
            if s.startswith(prefix) or (prefix in ("⚠",) and prefix in s[:3]):
                return col
        return C["text"]

    def _passes(self, text: str) -> bool:
        return not self.only_alerts or text.lstrip().startswith(self.ALERT) or "⚠" in text

    def add(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            stamp = time.strftime("%H:%M:%S")
            full = f"{stamp}  {line}"
            self.lines.append(full)
            if self._passes(line):
                self._append(full, line)

    def _append(self, full: str, line: str) -> None:
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.appendHtml(f'<span style="color:{C["dim"]}">{html.escape(full[:8])}</span>'
                        f'<span style="color:{self._color(line)}; white-space:pre">  {html.escape(full[10:])}</span>')
        if at_bottom:
            bar.setValue(bar.maximum())

    def restore(self, lines) -> None:
        """Reprend le journal d'une fenêtre précédente (changement de langue)."""
        for full in lines:
            self.lines.append(full)
            if self._passes(full[10:]):
                self._append(full, full[10:])

    def set_filter(self, only_alerts: bool) -> None:
        self.only_alerts = only_alerts
        self.clear()
        for full in self.lines:
            if self._passes(full[10:]):
                self._append(full, full[10:])

    def wipe(self) -> None:
        self.lines.clear()
        self.clear()


class QcReportDialog(QDialog):
    """Rapport QC lisible : une ligne par texture, triable, filtrable par décision."""

    COLORS = {"validated": C["green"], "corrected": C["lime"], "rejected": C["orange"], "local": C["muted"],
              "measured": C["blue"], "error": C["red"]}

    def __init__(self, report_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Rapport QC — {name}", name=report_path.parent.name))
        self.resize(1100, 640)
        self.report_path = report_path
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        self.rows = rep.get("textures", [])
        s = rep.get("summary", {})
        lay = QVBoxLayout(self)

        def part(n, text: str, col: str | None = None) -> str:
            t = f"{n} {text}"
            return f"<span style='color:{col}'>{t}</span>" if col else t
        head = QLabel(f"<b>{s.get('total', 0)}</b> {tr('textures')} — "
                      + ", ".join([part(s.get("validated", 0), tr("validées"), C["green"]),
                                   part(s.get("corrected", 0), tr("corrigées"), C["lime"]),
                                   part(s.get("rejected", 0), tr("rejetées"), C["orange"]),
                                   part(s.get("local", 0), tr("locales")),
                                   part(s.get("error", 0), tr("erreurs"), C["red"])])
                      + f" — DDS {s.get('dds_mb', 0)} {tr('Mo')} — {rep.get('generated', '')}")
        bar = QHBoxLayout()
        bar.addWidget(head, 1)
        self.filter = QComboBox()
        self.filter.addItems([tr("Toutes"), tr("Rejetées et erreurs"), tr("Corrigées"), tr("Avec avertissement")])
        self.filter.currentIndexChanged.connect(self._fill)
        bar.addWidget(self.filter)
        btn = QPushButton(tr("Ouvrir le JSON"))
        btn.clicked.connect(lambda: open_path(self.report_path))
        bar.addWidget(btn)
        lay.addLayout(bar)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([tr("Texture"), tr("Décision"), tr("Raisons / correctifs / avertissements"),
                                              tr("Format"), tr("Mips"), "PSNR"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        for col in (0, 1, 3, 4, 5):
            h.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        lay.addWidget(self.table)
        self._fill()

    def _fill(self) -> None:
        mode = self.filter.currentIndex()
        rows = [r for r in self.rows if mode == 0 or (mode == 1 and r["decision"] in ("rejected", "error"))
                or (mode == 2 and r["decision"] == "corrected") or (mode == 3 and r.get("warnings"))]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            out = r.get("output", {})
            text = "; ".join(r.get("reasons", []) + [f"✎ {f}" for f in r.get("fixes", [])]
                             + [f"⚠ {w}" for w in r.get("warnings", [])])
            label = tr(QC_LABELS[r["decision"]]) if r.get("decision") in QC_LABELS else r.get("label", "")
            cells = [r["name"], label, text, out.get("format", ""),
                     str(out.get("levels", "")), f"{out['psnr_l0']:.1f}" if out.get("psnr_l0") else ""]
            for j, val in enumerate(cells):
                it = QTableWidgetItem(val)
                if j == 1:
                    it.setForeground(QColor(self.COLORS.get(r["decision"], C["text"])))
                it.setToolTip(val)
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)


# ===================================================================== fenêtre principale

# libellés traduits à l'affichage : tr(libellé)
FAMILIES = [("sdxl", N_("SDXL — ControlNet Tile (+ Canny pour les joints)")),
            ("sd15", N_("SD 1.5 — ControlNet Tile (moins de VRAM)")),
            ("custom", N_("Workflow personnalisé (export API)"))]
DDS_FORMATS = [("match", N_("Identique à l'original (recommandé)")),
               ("auto", N_("Auto : BC1/DXT1 opaque · BC3/DXT5 alpha")),
               ("dxt1", N_("BC1 / DXT1 (opaque, alpha 1 bit → DXT5)")),
               ("dxt5", N_("BC3 / DXT5 (alpha)")),
               ("bc7", N_("BC7 haute qualité — Direct3D 11 uniquement")),
               ("bgra8", N_("BGRA8 sans compression"))]
TARGETS = [("specialk", N_("DDS — injection Special K")),
           ("loose", N_("DDS — fichiers séparés (Fallout 4, Starfield, mods)")), ("remix", N_("PNG — RTX Remix"))]
ENCODERS = [("auto", N_("Auto (texconv s'il est trouvé)")), ("texconv", N_("texconv (DirectXTex)")),
            ("builtin", N_("Intégré"))]
NO_CKPT, NO_LORA = N_("(celui du workflow)"), N_("(aucune)")
STAGES = ["scan", "generate", "export"]
STAGE_SHORT = {"scan": N_("1 · SCAN & ROUTAGE"), "generate": N_("2 · GÉNÉRATION"), "export": N_("3 · QC & DDS")}

# traductions Qt (boutons Oui / Non, menus contextuels…) installées pour la langue choisie
_QT_TRANSLATORS: list[QTranslator] = []
_ACTIVE: dict = {}                       # fenêtre principale en cours


def apply_language(app: QApplication, code: str | None) -> str:
    code = set_lang(code)
    for t in _QT_TRANSLATORS:
        app.removeTranslator(t)
    _QT_TRANSLATORS.clear()
    path = QLibraryInfo.path(QLibraryInfo.TranslationsPath)
    t = QTranslator(app)
    if t.load(QLocale(code), "qtbase", "_", path):
        app.installTranslator(t)
        _QT_TRANSLATORS.append(t)
    QLocale.setDefault(QLocale(code))
    return code


class MainWindow(QMainWindow):
    def __init__(self, log: list[str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__} — {tr('remaster de textures')}")
        self.setWindowIcon(app_icon())
        self.resize(1360, 860)
        self.setMinimumSize(1040, 700)
        self.qs = QSettings(APP_NAME, APP_NAME)
        self.pool = QThreadPool.globalInstance()
        self.thread: QThread | None = None
        self.worker: PipelineWorker | None = None
        self.stop_phase = 0                      # 0 : en marche, 1 : arrêt propre demandé
        self.launching = False                   # test du serveur en cours avant un lancement
        self.t_start = 0.0
        self.last_end: dict = {}
        self.counts: dict[str, int] = {}
        self._tasks: set[Task] = set()
        self._installing = False                 # téléchargement de Real-ESRGAN en cours
        self._alive = True                       # faux une fois fermée : les tâches en retard n'y touchent plus
        self._switching = False                  # fermeture pour changement de langue

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        body = QVBoxLayout()
        body.setContentsMargins(12, 10, 12, 12)
        body.setSpacing(10)
        root.addLayout(body, 1)
        body.addWidget(self._build_folders())
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self._build_settings())
        self.splitter.addWidget(self._build_run_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([470, 890])
        body.addWidget(self.splitter, 1)
        self.statusBar().setStyleSheet(f"color: {C['muted']};")

        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick)

        self._load_settings()
        self._refresh_dependent()
        if log:
            self.console.restore(log)
            self.console.add(tr("[INFO] Langue : {name}", name=LANGUAGES[get_lang()]))
        else:
            self.console.add(tr("[INFO] TEXFORGE {version} prêt — choisis le dossier des textures extraites, "
                                "puis LANCER LE PIPELINE.", version=__version__))
        self.candidates: list[Candidate] = []
        self._detecting = False
        self._locate_seq = 0                     # recherche du dossier de textures en cours
        self._fill_detect_menu()
        QTimer.singleShot(0, lambda: self.start_detection(auto_fill=True))
        QTimer.singleShot(300, self.test_server)

    # ----------------------------------------------------------------- construction
    def _build_header(self) -> QWidget:
        f = QFrame()
        f.setObjectName("header")
        lay = QHBoxLayout(f)
        lay.setContentsMargins(16, 10, 16, 10)
        icon = QLabel()
        icon.setPixmap(app_icon().pixmap(34, 34))
        title = QLabel("TEXFORGE")
        title.setObjectName("title")
        sub = QLabel(tr("Remaster IA local des textures DirectX 9 et antérieures — ComfyUI · ControlNet Tile · QC · DDS"))
        sub.setObjectName("subtitle")
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(title)
        col.addWidget(sub)
        self.server_pill = QLabel(tr("ComfyUI : non testé"))
        self.server_pill.setObjectName("pill")
        globe = QLabel("🌐")
        globe.setToolTip(tr("Langue de l'interface"))
        self.lang_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.lang_combo.addItem(name, code)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(get_lang())))
        self.lang_combo.setToolTip(tr("Langue de l'interface et des messages du traitement"))
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        lay.addWidget(icon)
        lay.addSpacing(8)
        lay.addLayout(col)
        lay.addStretch(1)
        lay.addWidget(globe)
        lay.addWidget(self.lang_combo)
        lay.addSpacing(12)
        lay.addWidget(self.server_pill)
        return f

    def _build_folders(self) -> QWidget:
        box = QGroupBox(tr("DOSSIERS"))
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        self.btn_detect = QToolButton()
        self.btn_detect.setText(tr("🔍 Détecter"))
        self.btn_detect.setPopupMode(QToolButton.InstantPopup)
        self.btn_detect.setToolTip(tr("Dossiers de textures extraites trouvés sur ce PC (Special K, RTX Remix),\n"
                                      "du plus récent au plus ancien"))
        self.detect_menu = QMenu(self.btn_detect)
        self.btn_detect.setMenu(self.detect_menu)
        self.input_row = PathRow(tr("Textures extraites, ex. C:\\Jeux\\MonJeu\\SK_Res\\dump\\textures"),
                                 dialog_title=tr("Dossier des textures extraites"), extra=self.btn_detect)
        self.output_row = PathRow(tr("Sortie DDS, ex. C:\\Jeux\\MonJeu\\SK_Res\\inject\\textures"),
                                  dialog_title=tr("Dossier de sortie"))
        self.input_row.setToolTip(tr("Dossier des textures du jeu (DDS, PNG, TGA…), sous-dossiers compris.\n"
                                     "Rempli automatiquement avec le dump Special K le plus récent. Choisir le\n"
                                     "dossier d'un jeu suffit : son dump Special K (à côté de l'exécutable ou dans\n"
                                     "son profil SKIF), sa capture RTX Remix ou le dossier qui regroupe ses\n"
                                     "textures est retrouvé."))
        self.output_row.setToolTip(tr("Les DDS finaux y sont écrits sous le nom d'origine, avec qc_report.json.\n"
                                      "Pour Special K : SK_Res\\inject\\textures du jeu."))
        self.input_row.changed.connect(lambda _t: self._refresh_project())
        self.input_row.picked.connect(self._on_input_picked)
        self.input_status = QLabel()
        self.input_status.setObjectName("hint")
        self.input_status.setWordWrap(True)
        self.input_status.setVisible(False)
        input_box = QWidget()
        ib = QVBoxLayout(input_box)
        ib.setContentsMargins(0, 0, 0, 0)
        ib.setSpacing(2)
        ib.addWidget(self.input_row)
        ib.addWidget(self.input_status)
        grid.addWidget(QLabel(tr("Dossier d'entrée")), 0, 0, Qt.AlignTop)
        grid.addWidget(input_box, 0, 1)
        grid.addWidget(QLabel(tr("Dossier de sortie")), 1, 0)
        grid.addWidget(self.output_row, 1, 1)
        proj = QHBoxLayout()
        self.project_label = QLabel()
        self.project_label.setObjectName("hint")
        self.project_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        b_open = QToolButton()
        b_open.setText(tr("Ouvrir"))
        b_open.clicked.connect(lambda: open_path(self._work_dir()) if self._work_dir().exists() else None)
        b_cmd = QToolButton()
        b_cmd.setText(tr("Copier la commande"))
        b_cmd.setToolTip(tr("Commande équivalente en ligne de commande (après un premier lancement)"))
        b_cmd.clicked.connect(self._copy_command)
        proj.addWidget(self.project_label, 1)
        proj.addWidget(b_open)
        proj.addWidget(b_cmd)
        self.ba2_filter = QLineEdit("*_d.dds")
        self.ba2_filter.setPlaceholderText(tr("vide : toutes les textures des archives"))
        self.ba2_filter.setToolTip(tr("Textures à extraire des archives BA2 (Fallout 4, Starfield), motifs séparés "
                                      "par « ; ».\n« *_d.dds » : textures de couleur seulement (les cartes de relief "
                                      "_n et spéculaires _s\nsont des données, agrandies sans IA et beaucoup plus "
                                      "lourdes).\nAvec un « / », le motif porte sur le chemin : "
                                      "« textures/architecture/*_d.dds »."))
        self.ba2_label = QLabel(tr("Archives BA2"))
        grid.addWidget(self.ba2_label, 2, 0)
        grid.addWidget(self.ba2_filter, 2, 1)
        self.ba2_label.setVisible(False)
        self.ba2_filter.setVisible(False)
        grid.addWidget(QLabel(tr("Projet")), 3, 0)
        grid.addLayout(proj, 3, 1)
        self.folders_box = box
        return box

    def _build_settings(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)
        lay.addWidget(self._group_engine())
        lay.addWidget(self._group_generation())
        lay.addWidget(self._group_dds())
        lay.addWidget(self._group_options())
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for combo in inner.findChildren(QComboBox):          # un libellé long ne doit pas élargir la colonne
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
        for combo in (self.scale, self.max_side):
            combo.setMinimumContentsLength(6)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(400)
        self.settings_panel = scroll
        return scroll

    def _group_engine(self) -> QGroupBox:
        box = QGroupBox(tr("MODÈLE / CHECKPOINT COMFYUI"))
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)
        self.provider = QComboBox()
        self.provider.addItem(tr("ComfyUI local (headless)"), "comfyui")
        self.provider.addItem(tr("Real-ESRGAN — sans ComfyUI (rendu réaliste)"), "esrgan")
        self.provider.addItem(tr("Simulation sans GPU (test de la chaîne)"), "mock")
        self.provider.setToolTip(tr("ComfyUI : remaster génératif (SDXL + ControlNet Tile), la matière est réinventée.\n"
                                    "Real-ESRGAN : sans ComfyUI ; agrandissement net, couleurs de l'original,\n"
                                    "grain de matière dosé d'après la texture, garde contre les motifs inventés.\n"
                                    "Choisi automatiquement si ComfyUI ne répond pas (tant que tu n'imposes pas de moteur)."))
        self.provider.currentIndexChanged.connect(self._refresh_dependent)
        self.provider.activated.connect(self._engine_chosen)
        form.addRow(tr("Moteur"), self.provider)
        esr = QHBoxLayout()
        self.esrgan_status = QLabel()
        self.esrgan_status.setObjectName("hint")
        self.esrgan_status.setWordWrap(True)
        self.btn_esrgan = QPushButton(tr("Installer Real-ESRGAN"))
        self.btn_esrgan.setToolTip(tr("Télécharge realesrgan-ncnn-vulkan (version officielle de GitHub, xinntao/Real-ESRGAN,\n"
                                      "empreinte SHA-256 vérifiée) dans le dossier tools de TEXFORGE."))
        self.btn_esrgan.clicked.connect(lambda: self.install_esrgan())
        esr.addWidget(self.esrgan_status, 1)
        esr.addWidget(self.btn_esrgan)
        form.addRow("Real-ESRGAN", esr)
        srv = QHBoxLayout()
        self.server = QLineEdit("http://127.0.0.1:8188")
        self.server.editingFinished.connect(self.test_server)
        self.btn_test = QPushButton(tr("Tester"))
        self.btn_test.clicked.connect(self.test_server)
        srv.addWidget(self.server, 1)
        srv.addWidget(self.btn_test)
        form.addRow(tr("Serveur"), srv)
        self.family = QComboBox()
        for key, label in FAMILIES:
            self.family.addItem(tr(label), key)
        self.family.currentIndexChanged.connect(self._refresh_dependent)
        form.addRow(tr("Famille"), self.family)
        self.workflow_row = PathRow("workflow_api.json", file_filter=tr("Workflow API (*.json)"),
                                    dialog_title=tr("Workflow ComfyUI (format API)"))
        self.workflow_edges_row = PathRow(tr("facultatif : variante pour briques et carrelage"),
                                          file_filter=tr("Workflow API (*.json)"), dialog_title=tr("Workflow « joints »"))
        form.addRow(tr("Workflow"), self.workflow_row)
        form.addRow(tr("Workflow joints"), self.workflow_edges_row)
        ck = QHBoxLayout()
        self.checkpoint = QComboBox()
        self.checkpoint.setEditable(True)
        self.checkpoint.addItem(tr(NO_CKPT))
        self.checkpoint.setToolTip(tr("Checkpoint chargé par le nœud TEXFORGE_CHECKPOINT. « Tester » remplit la liste\n"
                                      "avec les modèles réellement présents dans ComfyUI."))
        btn_ref = QToolButton()
        btn_ref.setText("⟳")
        btn_ref.setToolTip(tr("Relire la liste des modèles sur le serveur"))
        btn_ref.clicked.connect(self.test_server)
        ck.addWidget(self.checkpoint, 1)
        ck.addWidget(btn_ref)
        form.addRow("Checkpoint", ck)
        self.lora = QComboBox()
        self.lora.setEditable(True)
        self.lora.addItem(tr(NO_LORA))
        self.lora.setToolTip(tr("LoRA insérée après le checkpoint (photoréalisme, matière…). Le nœud est ajouté\n"
                                "automatiquement au workflow ; le raccord circulaire reste actif."))
        self.lora.currentTextChanged.connect(self._refresh_dependent)
        form.addRow("LoRA", self.lora)
        self.lora_strength = ValueSlider(0.0, 1.5, 0.05, 0.8)
        form.addRow(tr("Force LoRA"), self.lora_strength)
        self.engine_box = box
        return box

    def _group_generation(self) -> QGroupBox:
        box = QGroupBox(tr("PHOTORÉALISME / FIDÉLITÉ À LA DA"))
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)
        self.denoise = ValueSlider(0.20, 0.60, 0.01, 0.42)
        self.denoise.setToolTip(tr("Part de l'image réinventée par le modèle. Bas : fidèle à l'original ;\n"
                                   "haut : plus de matière et de détail, plus de risque de trahir la DA."))
        self.cn = ValueSlider(0.30, 1.20, 0.05, 0.75)
        self.cn.setToolTip(tr("Force du ControlNet Tile : attache le rendu aux formes et aux couleurs de l'original."))
        self.denoise.valueChanged.connect(lambda _v: self._update_presets())
        self.cn.valueChanged.connect(lambda _v: self._update_presets())
        form.addRow("Denoise", self.denoise)
        form.addRow("ControlNet Tile", self.cn)
        self.esrgan_detail = ValueSlider(0.0, 1.5, 0.05, 1.0)
        self.esrgan_detail.setToolTip(tr("Mode Real-ESRGAN : grain de matière ajouté là où le modèle a lissé.\n"
                                         "0 = aucun ; 1 = dose mesurée d'après la texture ; 1,5 = appuyé."))
        form.addRow(tr("Micro-détail"), self.esrgan_detail)
        self.per_material = QCheckBox(tr("Ajuster par matériau"))
        self.per_material.setChecked(True)
        self.per_material.setToolTip(tr("Coché : les curseurs décalent les préréglages de chaque matériau.\n"
                                        "Décoché : la même valeur pour toutes les textures."))
        self.per_material.toggled.connect(lambda _c: self._update_presets())
        form.addRow("", self.per_material)
        self.presets_label = QLabel()
        self.presets_label.setObjectName("hint")
        self.presets_label.setWordWrap(True)
        form.addRow("", self.presets_label)
        row = QHBoxLayout()
        self.scale = QComboBox()
        self.scale.addItem("×2", 2)
        self.scale.addItem("×4", 4)
        self.scale.setCurrentIndex(1)
        self.max_side = QComboBox()
        for v in (1024, 2048, 4096):
            self.max_side.addItem(tr("max {v} px", v=v), v)
        self.max_side.setCurrentIndex(1)
        self.max_side.setToolTip(tr("Côté maximal. Jeux 32 bits : 2048 ou moins (mémoire du processus du jeu)."))
        row.addWidget(self.scale)
        row.addWidget(self.max_side)
        form.addRow(tr("Agrandissement"), row)
        self.limit = QSpinBox()
        self.limit.setRange(0, 100000)
        self.limit.setSpecialValueText(tr("toutes"))
        self.limit.setSuffix(tr(" textures IA"))
        self.limit.setToolTip(tr("Pour un premier essai : limite le nombre de matières traitées par ComfyUI ou "
                                 "Real-ESRGAN (0 = toutes)."))
        form.addRow(tr("Lot d'essai"), self.limit)
        return box

    def _group_dds(self) -> QGroupBox:
        box = QGroupBox(tr("DDS FORMAT && COMPRESSION"))
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)
        self.target = QComboBox()
        for key, label in TARGETS:
            self.target.addItem(tr(label), key)
        self.target.currentIndexChanged.connect(self._refresh_dependent)
        form.addRow(tr("Cible"), self.target)
        self.dds_format = QComboBox()
        for key, label in DDS_FORMATS:
            self.dds_format.addItem(tr(label), key)
        self.dds_format.setToolTip(tr("« Identique à l'original » : le jeu (ou l'injecteur) attend ce format.\n"
                                      "BC7 n'est lisible qu'en Direct3D 11 (dgVoodoo2 + Special K)."))
        self.dds_format.currentIndexChanged.connect(self._refresh_dependent)
        form.addRow(tr("Format"), self.dds_format)
        self.bc7_warn = QLabel(tr("BC7 n'existe pas en DirectX 9/10 : à n'utiliser qu'avec dgVoodoo2 + Special K "
                                  "(Direct3D 11), et texconv."))
        self.bc7_warn.setObjectName("warn")
        self.bc7_warn.setWordWrap(True)
        form.addRow("", self.bc7_warn)
        self.encoder = QComboBox()
        for key, label in ENCODERS:
            self.encoder.addItem(tr(label), key)
        self.encoder.currentIndexChanged.connect(self._refresh_dependent)
        form.addRow(tr("Encodeur"), self.encoder)
        self.texconv_row = PathRow(tr("texconv.exe (sinon tools\\texconv.exe ou PATH)"), file_filter="texconv (*.exe)",
                                   dialog_title="texconv.exe")
        self.texconv_row.changed.connect(lambda _t: self._refresh_dependent())
        form.addRow("texconv", self.texconv_row)
        self.texconv_status = QLabel()
        self.texconv_status.setObjectName("hint")
        self.texconv_status.setWordWrap(True)
        form.addRow("", self.texconv_status)
        self.dds_box = box
        return box

    def _group_options(self) -> QGroupBox:
        box = QGroupBox(tr("OPTIONS && ROUTAGE"))
        lay = QVBoxLayout(box)
        lay.setSpacing(4)

        def hint(text: str) -> QLabel:
            lb = QLabel(text)
            lb.setObjectName("hint")
            lb.setWordWrap(True)
            lb.setContentsMargins(26, 0, 0, 6)
            return lb

        def indent(widget: QWidget) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(26, 0, 0, 4)
            h.addWidget(widget)
            return w

        self.route_ui = QCheckBox(tr("Routage automatique HUD / UI"))
        self.route_ui.setToolTip(tr("Polices, icônes, HUD et cartes partent vers Real-ESRGAN / Waifu2x ; les matières\n"
                                    "vers ComfyUI. Décoché : tout ce qui n'est pas une donnée part vers ComfyUI."))
        self.realesrgan_row = PathRow(tr("realesrgan-ncnn-vulkan.exe (sinon Lanczos)"),
                                      file_filter=tr("Programme (*.exe)"), dialog_title="realesrgan-ncnn-vulkan.exe")
        self.seamless = QCheckBox(tr("Seamless tiling"))
        self.seamless.setToolTip(tr("Les textures détectées comme répétables sont générées avec un raccord circulaire,\n"
                                    "puis vérifiées au QC. Les icônes, atlas et décals ne sont pas concernés."))
        self.force_seamless = QCheckBox(tr("Forcer sur toutes les textures (déconseillé)"))
        self.force_seamless.setToolTip(tr("Le raccord circulaire fait se toucher les bords opposés : sur un atlas,\n"
                                          "le contenu d'un bord déborde sur l'autre. À réserver à un dossier de matières."))
        self.mipmaps = QCheckBox(tr("Génération automatique des mipmaps"))
        self.mipmaps.setToolTip(tr("Décoché : mipmaps seulement si l'original en avait."))
        self.qc = QCheckBox(tr("Contrôle qualité SSIM / préservation de la DA"))
        self.qc.setToolTip(tr("Structure, couleurs (ΔE2000), palette, raccord et transparence comparés à l'original\n"
                              "ramené à sa résolution. Échec : correctif automatique ou version locale conservée."))
        self.qc_fix = QCheckBox(tr("Correctifs automatiques"))
        self.qc_fix.setToolTip(tr("Verrou couleur basses fréquences et fondu des raccords avant de rejeter une texture."))
        self.redo = QCheckBox(tr("Régénérer si les réglages ont changé"))
        self.redo.setToolTip(tr("Après un changement de modèle, de denoise ou de ControlNet, les textures déjà\n"
                                "générées sont refaites. Les choix faits à la main dans la galerie sont conservés."))
        for cb in (self.route_ui, self.seamless, self.mipmaps, self.qc, self.qc_fix, self.redo):
            cb.setChecked(True)
        self.route_ui.toggled.connect(self._refresh_dependent)
        self.seamless.toggled.connect(self._refresh_dependent)
        self.qc.toggled.connect(self._refresh_dependent)
        lay.addWidget(self.route_ui)
        lay.addWidget(hint(tr("Upscale local rapide (Real-ESRGAN / Waifu2x), sans IA générative : "
                              "les textes restent lisibles.")))
        lay.addWidget(indent(self.realesrgan_row))
        lay.addWidget(self.seamless)
        lay.addWidget(hint(tr("Raccord parfait des bords sur les textures répétables détectées (sols, murs, terrains).")))
        lay.addWidget(indent(self.force_seamless))
        lay.addWidget(self.mipmaps)
        lay.addWidget(hint(tr("Chaîne complète filtrée en lumière linéaire, couverture alpha conservée "
                              "(feuillage, grilles).")))
        lay.addWidget(self.qc)
        lay.addWidget(hint(tr("Structure, couleurs, palette, raccord et transparence comparés à l'original.")))
        lay.addWidget(indent(self.qc_fix))
        lay.addWidget(self.redo)
        return box

    def _build_run_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(8)
        self.btn_launch = QPushButton(tr("▶  LANCER LE PIPELINE"))
        self.btn_launch.setObjectName("launch")
        self.btn_launch.setMinimumHeight(58)
        self.btn_launch.setCursor(Qt.PointingHandCursor)
        self.btn_launch.clicked.connect(self._on_launch_clicked)
        lay.addWidget(self.btn_launch)

        stages = QHBoxLayout()
        self.stage_labels = {}
        for st in STAGES:
            lb = QLabel(tr(STAGE_SHORT[st]))
            lb.setObjectName("stage")
            lb.setAlignment(Qt.AlignCenter)
            stages.addWidget(lb, 1)
            self.stage_labels[st] = lb
        lay.addLayout(stages)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat(tr("En attente"))
        lay.addWidget(self.progress)
        info = QHBoxLayout()
        self.progress_detail = QLabel("—")
        self.progress_detail.setObjectName("hint")
        self.time_label = QLabel("")
        self.time_label.setObjectName("hint")
        info.addWidget(self.progress_detail, 1)
        info.addWidget(self.time_label)
        lay.addLayout(info)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.chips = {}
        for key, label, col in (("ui", "UI", C["cyan"]), ("local", "LOCAL", "#5fb3a1"), ("ai", "COMFYUI", C["violet"]),
                                ("fallback", tr("REPLIS"), C["orange"]), ("validated", "QC ✔", C["green"]),
                                ("corrected", "QC ✎", C["lime"]), ("rejected", "QC ✘", C["orange"]),
                                ("errors", tr("ERREURS"), C["red"])):
            lb = QLabel()
            lb.setObjectName("chip")
            lb.setProperty("label", label)
            lb.setProperty("color", col)
            chips.addWidget(lb)
            self.chips[key] = lb
        chips.addStretch(1)
        lay.addLayout(chips)
        self._reset_counts()

        tools = QHBoxLayout()
        title = QLabel(tr("CONSOLE"))
        title.setStyleSheet(f"color: {C['accent']}; font-weight: 600; letter-spacing: 1px;")
        self.log_filter = QComboBox()
        self.log_filter.addItems([tr("Tout afficher"), tr("Alertes et étapes seulement")])
        self.log_filter.currentIndexChanged.connect(lambda i: self.console.set_filter(i == 1))
        b_clear = QToolButton()
        b_clear.setText(tr("Effacer"))
        b_save = QToolButton()
        b_save.setText(tr("Enregistrer…"))
        tools.addWidget(title)
        tools.addStretch(1)
        tools.addWidget(self.log_filter)
        tools.addWidget(b_clear)
        tools.addWidget(b_save)
        lay.addLayout(tools)
        self.console = LogConsole()
        b_clear.clicked.connect(self.console.wipe)
        b_save.clicked.connect(self._save_log)
        lay.addWidget(self.console, 1)

        res = QHBoxLayout()
        self.btn_out = QPushButton(tr("📂  Ouvrir la sortie"))
        self.btn_report = QPushButton(tr("📋  Rapport QC"))
        self.btn_gallery = QPushButton(tr("🖼  Galerie avant / après"))
        self.btn_out.clicked.connect(lambda: open_path(self.output_row.text()) if self.output_row.text() else None)
        self.btn_report.clicked.connect(self._show_report)
        self.btn_gallery.clicked.connect(self._open_gallery)
        for b in (self.btn_out, self.btn_report, self.btn_gallery):
            res.addWidget(b)
        res.addStretch(1)
        lay.addLayout(res)
        return w

    # ----------------------------------------------------------------- réglages <-> interface
    def collect(self) -> RemasterSettings:
        ckpt = self.checkpoint.currentText().strip()
        lora = self.lora.currentText().strip()
        return RemasterSettings(
            input_dir=self.input_row.text(), output_dir=self.output_row.text(),
            target=self.target.currentData(), provider=self.provider.currentData(), server=self.server.text().strip(),
            model_family=self.family.currentData(), workflow=self.workflow_row.text(),
            workflow_edges=self.workflow_edges_row.text(),
            checkpoint="" if ckpt in ("", tr(NO_CKPT)) else ckpt, lora="" if lora in ("", tr(NO_LORA)) else lora,
            lora_strength=self.lora_strength.value(), denoise=self.denoise.value(), cn_strength=self.cn.value(),
            per_material=self.per_material.isChecked(), scale=self.scale.currentData(),
            max_side=self.max_side.currentData(), limit=self.limit.value(), redo_stale=self.redo.isChecked(),
            route_ui=self.route_ui.isChecked(), seamless=self.seamless.isChecked(),
            force_seamless=self.force_seamless.isChecked() and self.seamless.isChecked(),
            mipmaps=self.mipmaps.isChecked(), qc=self.qc.isChecked(), qc_autofix=self.qc_fix.isChecked(),
            dds_format=self.dds_format.currentData(), encoder=self.encoder.currentData(),
            texconv_exe=self.texconv_row.text(), realesrgan_exe=self.realesrgan_row.text(),
            esrgan_detail=self.esrgan_detail.value(), ba2_filter=self.ba2_filter.text().strip())

    def apply(self, s: RemasterSettings) -> None:
        def select(combo: QComboBox, data) -> None:
            i = combo.findData(data)
            if i >= 0:
                combo.setCurrentIndex(i)

        def text_combo(combo: QComboBox, value: str, empty: str) -> None:
            value = value or empty
            if combo.findText(value) < 0:
                combo.addItem(value)
            combo.setCurrentText(value)

        self.input_row.setText(s.input_dir)
        self.output_row.setText(s.output_dir)
        self.ba2_filter.setText(s.ba2_filter)
        select(self.target, s.target)
        select(self.provider, s.provider)
        self.server.setText(s.server)
        select(self.family, s.model_family)
        self.workflow_row.setText(s.workflow)
        self.workflow_edges_row.setText(s.workflow_edges)
        text_combo(self.checkpoint, s.checkpoint, tr(NO_CKPT))
        text_combo(self.lora, s.lora, tr(NO_LORA))
        self.lora_strength.setValue(s.lora_strength)
        self.denoise.setValue(s.denoise)
        self.cn.setValue(s.cn_strength)
        self.per_material.setChecked(s.per_material)
        select(self.scale, s.scale)
        select(self.max_side, s.max_side)
        self.limit.setValue(s.limit)
        self.redo.setChecked(s.redo_stale)
        self.route_ui.setChecked(s.route_ui)
        self.seamless.setChecked(s.seamless)
        self.force_seamless.setChecked(s.force_seamless)
        self.mipmaps.setChecked(s.mipmaps)
        self.qc.setChecked(s.qc)
        self.qc_fix.setChecked(s.qc_autofix)
        select(self.dds_format, s.dds_format)
        select(self.encoder, s.encoder)
        self.texconv_row.setText(s.texconv_exe)
        self.realesrgan_row.setText(s.realesrgan_exe)
        self.esrgan_detail.setValue(s.esrgan_detail)

    def _load_settings(self) -> None:
        raw = self.qs.value("settings_json", "")
        if raw:
            try:
                self.apply(RemasterSettings.from_dict(json.loads(raw)))
            except (ValueError, TypeError):
                pass
        geo = self.qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        sizes = self.qs.value("splitter")
        if sizes:
            try:
                self.splitter.setSizes([int(x) for x in sizes])
            except (TypeError, ValueError):
                pass

    def _save_settings(self) -> None:
        self.qs.setValue("settings_json", json.dumps(self.collect().to_dict(), ensure_ascii=False))
        self.qs.setValue("geometry", self.saveGeometry())
        self.qs.setValue("splitter", self.splitter.sizes())

    def _work_dir(self) -> Path:
        return work_dir_of(self.collect())

    def _refresh_project(self) -> None:
        self.project_label.setText(str(self._work_dir()) if self.input_row.text() else "—")

    # ----------------------------------------------------------------- détection du dossier d'entrée
    def start_detection(self, auto_fill: bool = False) -> None:
        """Recherche des dumps Special K / captures RTX Remix en arrière-plan."""
        if self._detecting:
            return
        self._detecting = True
        self._fill_detect_menu()
        history = [p for p in (self.qs.value("recent_inputs") or []) if p]
        if self.input_row.text():
            history.append(self.input_row.text())
        self._run_task(lambda h: find_texture_dirs(history=h), lambda r: self._on_detected(r, auto_fill), history)

    def _on_detected(self, res, auto_fill: bool) -> None:
        self._detecting = False
        if isinstance(res, dict):                        # erreur de la tâche : la saisie manuelle reste possible
            self.console.add(tr("[INFO] Détection du dossier de textures impossible : {err}", err=res.get("error")))
            res = []
        self.candidates = list(res)
        self._fill_detect_menu()
        cur = self.input_row.text()
        if not auto_fill or (cur and Path(cur).is_dir()):
            if auto_fill and cur:
                self._locate(cur, adopt=False)           # dossier repris des réglages : état seulement
            return
        if self.candidates:
            self._use_candidate(self.candidates[0], auto=True)
            if len(self.candidates) > 1:
                self.console.add(tr("[INFO] {n} autre(s) jeu(x) avec des textures extraites : bouton « 🔍 Détecter ».",
                                    n=len(self.candidates) - 1))
        else:
            self.console.add(tr("[INFO] Aucun dump de textures trouvé. Dans Special K (Ctrl+Maj+Retour arrière), "
                                "active le dump des textures, joue, puis « 🔍 Détecter » → Rechercher à nouveau. "
                                "Le dossier d'installation seul ne marche que si le jeu y range ses textures en fichiers séparés."))

    def _fill_detect_menu(self) -> None:
        m = self.detect_menu
        m.clear()
        if self._detecting:
            self.btn_detect.setText(tr("🔍 Recherche…"))
            m.addAction(tr("Recherche en cours…")).setEnabled(False)
            return
        n = len(self.candidates)
        self.btn_detect.setText(tr("🔍 Détecter ({n})", n=n) if n else tr("🔍 Détecter"))
        cur = os.path.normcase(os.path.abspath(self.input_row.text())) if self.input_row.text() else ""
        for c in self.candidates:
            when = QLocale().toString(QDateTime.fromSecsSinceEpoch(int(c.newest)), QLocale.ShortFormat) if c.newest else "—"
            kind = "Special K" if c.kind == "specialk" else "RTX Remix"
            act = m.addAction(tr("{game} — {count} texture(s) · {when} · {kind}", game=c.game, count=c.count,
                                 when=when, kind=kind))
            act.setToolTip(f"{c.path}\n{tr(c.source)}")
            act.setCheckable(True)
            act.setChecked(os.path.normcase(os.path.abspath(c.path)) == cur)
            act.triggered.connect(lambda _checked=False, c=c: self._use_candidate(c))
        if not n:
            m.addAction(tr("Aucun dump trouvé : active le dump des textures dans Special K")).setEnabled(False)
        m.setToolTipsVisible(True)
        m.addSeparator()
        m.addAction(tr("Rechercher à nouveau"), lambda: self.start_detection(auto_fill=False))

    def _use_candidate(self, c: Candidate, auto: bool = False) -> None:
        self.input_row.setText(c.path)
        if auto:
            self.console.add(tr("[INFO] Dossier de textures détecté : {game} — {count} texture(s) ({path})",
                                game=c.game, count=c.count, path=c.path))
        else:
            self.console.add(tr("[INFO] Dossier de textures : {game} — {count} texture(s) ({path})",
                                game=c.game, count=c.count, path=c.path))
        self._show_input_status(Located(c.path, c.kind, c.count, c.path, c.game, c.inject, c.source))
        self._follow_input(c)
        self._fill_detect_menu()

    def _follow_input(self, c: Candidate) -> None:
        """Cible et sortie qui vont avec le dossier choisi : capture RTX Remix -> cible PNG Remix ;
        dump Special K -> cible DDS et dossier d'injection du même jeu, si la sortie est vide ou déjà le
        dossier d'injection d'un jeu (une autre sortie choisie à la main n'est jamais remplacée)."""
        want = "remix" if c.kind == "remix" else "specialk"
        if self.target.currentData() != want:
            self.target.setCurrentIndex(self.target.findData(want))
            self.console.add(tr("[INFO] Cible : {target}", target=self.target.currentText()))
        if c.kind != "specialk" or not c.inject:
            return
        out = self.output_row.text()
        is_inject = [x.lower() for x in Path(out).parts[-3:]] == ["sk_res", "inject", "textures"] if out else False
        if (not out or is_inject) and os.path.normcase(os.path.abspath(out or ".")) != os.path.normcase(c.inject):
            self.output_row.setText(c.inject)
            self.console.add(tr("[INFO] Sortie : dossier d'injection Special K du jeu ({path})", path=c.inject))

    def _on_input_picked(self, path: str) -> None:
        """Dossier choisi ou saisi à la main : on cherche le dossier des textures qu'il contient
        (dump Special K du jeu, profil SKIF, capture RTX Remix, dossier des textures du jeu)."""
        if path:
            self._locate(path, adopt=True)

    def _locate(self, path: str, adopt: bool) -> None:
        self._locate_seq += 1
        seq = self._locate_seq
        self.input_status.setText(tr("🔍 Recherche des textures dans ce dossier…"))
        self.input_status.setVisible(True)
        self._run_task(locate_textures, lambda r: self._on_located(r, seq, adopt), path)

    def _on_located(self, r, seq: int, adopt: bool) -> None:
        if seq != self._locate_seq:
            return                                       # un autre dossier a été choisi entre-temps
        if isinstance(r, dict):                          # erreur de la tâche
            self.input_status.setText(tr("⚠ Recherche impossible : {err}", err=r.get("error")))
            return
        if os.path.normcase(os.path.abspath(self.input_row.text() or ".")) != os.path.normcase(os.path.abspath(r.chosen)):
            return                                       # champ modifié pendant la recherche
        if adopt and r.changed and r.kind != "none":
            self.input_row.setText(r.path)
            self.console.add(tr("[INFO] Dossier choisi : {chosen}\n   → textures : {path} ({count} fichier(s), {source})",
                                chosen=r.chosen, path=r.path, count=r.count, source=tr(r.source)))
        elif adopt and r.kind != "none":
            self.console.add(tr("[INFO] Dossier de textures : {path} ({count} fichier(s))", path=r.path, count=r.count))
        if r.kind == "none" and r.note:
            self.console.add("⚠ " + r.note.replace("\n", "\n   "))
        self._show_input_status(r)
        if adopt and r.kind in ("specialk", "remix"):
            self._follow_input(Candidate(r.path, r.kind, r.game, r.count, 0.0, r.inject, r.source))
        elif adopt and r.kind == "ba2":
            self._follow_ba2(r)
        elif adopt:
            # sortie restée sur le dossier d'injection d'un autre jeu : elle ne vaut plus rien ici
            out = self.output_row.text()
            if out and self._is_inject_dir(out) and os.path.normcase(os.path.abspath(out)) != os.path.normcase(
                    os.path.abspath(r.inject or ".")):
                self.output_row.setText(r.inject)
                self.console.add(tr("[INFO] Sortie {action} : elle visait le dossier d'injection d'un autre jeu",
                                    action=tr("réglée sur {path}", path=r.inject) if r.inject else tr("vidée")))
        self._fill_detect_menu()

    def _follow_ba2(self, r: Located) -> None:
        """Archives BA2 : cible « fichiers séparés », sortie dans un dossier de mod à côté de Data."""
        if self.target.currentData() != "loose":
            self.target.setCurrentIndex(self.target.findData("loose"))
            self.console.add(tr("[INFO] Cible : {target}", target=self.target.currentText()))
        out = self.output_row.text()
        if not out or self._is_inject_dir(out):
            mod = str(mod_output_for(r.path))
            self.output_row.setText(mod)
            self.console.add(tr("[INFO] Sortie : {path}\n   À la fin, copie le dossier Textures qu'il contient dans "
                                "Data (ou installe ce dossier comme un mod avec Mod Organizer 2 ou Vortex).",
                                path=mod))
        self.console.add(tr("[INFO] Pour que le jeu préfère ces fichiers aux archives : dans Documents\\My Games\\"
                            "<jeu>\\<jeu>Custom.ini, section [Archive], bInvalidateOlderFiles=1 et "
                            "sResourceDataDirsFinal= (vide)."))

    def _show_input_status(self, r: Located) -> None:
        if r.kind == "none":
            short = r.summary or (r.note or tr("Aucune texture lisible.")).split("\n")[0]
            text = "⚠ " + short[:1].upper() + short[1:] + tr(" (détails dans la console)")
        else:
            what = {"specialk": tr("dump Special K"), "remix": tr("capture RTX Remix"),
                    "loose": tr("textures du jeu en fichiers séparés"),
                    "ba2": tr("archives BA2 ; extraites selon le filtre ci-dessous")}[r.kind]
            text = tr("✓ {count} texture(s) — {what}", count=r.count, what=what)
            if r.source == "profil Special K":
                text += tr(" (profil global SKIF)")
        self.input_status.setText(text)
        self.input_status.setVisible(True)
        self.ba2_label.setVisible(r.kind == "ba2")
        self.ba2_filter.setVisible(r.kind == "ba2")

    def _refresh_dependent(self, *_):
        """Active ou masque les réglages qui dépendent d'autres réglages."""
        comfy = self.provider.currentData() == "comfyui"
        esrgan = self.provider.currentData() == "esrgan"
        custom = self.family.currentData() == "custom"
        for w in (self.family, self.checkpoint, self.lora):
            w.setEnabled(comfy)
        for w in (self.server, self.btn_test):              # en mode Real-ESRGAN : savoir si ComfyUI est là
            w.setEnabled(comfy or esrgan)
        for w in (self.denoise, self.cn, self.per_material):
            w.setEnabled(not esrgan)
        self.esrgan_detail.setEnabled(esrgan)
        self._update_esrgan_status()
        if getattr(self, "chips", None) and self.worker is None:
            self.chips["ai"].setProperty("label", "ESRGAN" if esrgan else "COMFYUI")
            self._paint_chip("ai")
        self.workflow_row.setVisible(custom)
        self.workflow_edges_row.setVisible(custom)
        form = self.engine_box.layout()
        for w in (self.workflow_row, self.workflow_edges_row):
            lab = form.labelForField(w)
            if lab:
                lab.setVisible(custom)
        has_lora = self.lora.currentText().strip() not in ("", tr(NO_LORA))
        self.lora_strength.setEnabled(comfy and has_lora)
        dds = self.target.currentData() in ("specialk", "loose")
        for w in (self.dds_format, self.encoder, self.texconv_row):
            w.setEnabled(dds)
        self.bc7_warn.setVisible(dds and self.dds_format.currentData() == "bc7")
        self.force_seamless.setEnabled(self.seamless.isChecked())
        self.qc_fix.setEnabled(self.qc.isChecked())
        self.realesrgan_row.setEnabled(self.route_ui.isChecked())
        self._update_texconv_status()
        self._update_presets()
        self._refresh_project()

    def _esrgan_exe(self):
        return find_realesrgan({"ui_upscale": {"realesrgan_exe": self.realesrgan_row.text()}, "_base": str(ROOT)})

    def _update_esrgan_status(self) -> None:
        if getattr(self, "_installing", False):
            return
        exe = self._esrgan_exe()
        if exe:
            self.esrgan_status.setText(tr("✓ installé : {path}", path=exe))
        elif self.provider.currentData() == "esrgan":
            self.esrgan_status.setText(tr("⚠ absent : nécessaire pour ce moteur"))
        else:
            self.esrgan_status.setText(tr("absent (sert aussi aux HUD et aux replis)"))
        self.btn_esrgan.setVisible(exe is None)

    def install_esrgan(self, then=None) -> None:
        """Téléchargement en arrière-plan ; `then` est appelé une fois installé."""
        if getattr(self, "_installing", False):
            return
        self._installing = True
        self.btn_esrgan.setEnabled(False)
        self.btn_launch.setEnabled(False)
        name, size, _sha = RELEASES[platform_key()]
        self.console.add(tr("[INFO] Téléchargement de Real-ESRGAN ({name}, {mb:.0f} Mo) depuis GitHub…",
                            name=name, mb=size / 2 ** 20))
        bridge = _ProgressBridge()
        bridge.step.connect(lambda done, total: self.esrgan_status.setText(
            tr("téléchargement : {pct} %", pct=int(done * 100 / max(total, 1)))))
        self._esrgan_bridge = bridge                      # garde l'objet en vie pendant la tâche

        def work():
            return {"exe": str(install_realesrgan(INSTALL_DIR, lambda d, t: bridge.step.emit(d, t)))}

        def done(r: dict) -> None:
            self._installing = False
            self.btn_esrgan.setEnabled(True)
            self.btn_launch.setEnabled(True)
            if "error" in r:
                self.console.add(tr("[ERREUR] installation de Real-ESRGAN : {err}", err=r["error"]))
                QMessageBox.warning(self, APP_NAME, tr("Installation de Real-ESRGAN impossible :\n\n{err}", err=r["error"]))
            else:
                self.console.add(tr("[INFO] Real-ESRGAN installé : {path}", path=r["exe"]))
                if not self.realesrgan_row.text():           # chemin explicite pour le processus de traitement
                    self.realesrgan_row.setText(r["exe"])
            self._update_esrgan_status()
            if "error" not in r and then is not None:
                then()
        self._run_task(work, done)

    def _engine_chosen(self, _index: int) -> None:
        """Choix fait à la main : plus de bascule automatique ComfyUI <-> Real-ESRGAN."""
        self.qs.setValue("engine_chosen", True)

    def _auto_engine(self, comfy_ok: bool) -> None:
        """Tant que l'utilisateur n'a pas imposé de moteur : Real-ESRGAN sans ComfyUI, ComfyUI dès qu'il répond."""
        if str(self.qs.value("engine_chosen", "false")).lower() in ("true", "1") or self.worker is not None:
            return
        cur = self.provider.currentData()
        if cur == "comfyui" and not comfy_ok:
            self.provider.setCurrentIndex(self.provider.findData("esrgan"))
            self.console.add(tr("[INFO] ComfyUI ne répond pas : moteur Real-ESRGAN (rendu réaliste, sans IA "
                                "générative). Lance ComfyUI puis « Tester » pour revenir au remaster génératif."))
        elif cur == "esrgan" and comfy_ok:
            self.provider.setCurrentIndex(self.provider.findData("comfyui"))
            self.console.add(tr("[INFO] ComfyUI répond : moteur ComfyUI (remaster génératif)."))

    def _update_texconv_status(self) -> None:
        from texforge.ddsbuild import find_texconv, is_texconv_name
        raw = self.texconv_row.text().strip()
        cfg = {"qc": {"texconv_exe": raw}, "_base": str(ROOT)}
        exe = find_texconv(cfg)
        if raw and not is_texconv_name(raw):
            self.texconv_status.setText(tr("✗ « {name} » n'est pas texconv.exe : vide ce champ ou choisis texconv.exe.",
                                           name=Path(raw).name))
        elif self.encoder.currentData() == "builtin":
            self.texconv_status.setText(tr("Encodeur intégré : DXT1, DXT3, DXT5 (pas de BC7)."))
        elif exe:
            self.texconv_status.setText(tr("✓ texconv trouvé : {exe}", exe=exe))
        else:
            self.texconv_status.setText(tr("texconv absent : l'encodeur intégré prendra le relais (DXT1/3/5)."))

    def _update_presets(self) -> None:
        s = self.collect()
        p = material_presets(s)
        names = (("stone", N_("pierre")), ("masonry", N_("maçonnerie")), ("metal", N_("métal")), ("wood", N_("bois")),
                 ("fabric", N_("tissu")), ("vegetation", N_("végétation")), ("skin", N_("peau")))
        txt = " · ".join(f"{tr(nm)} {p[k]['denoise']:.2f}/{p[k]['cn_strength']:.2f}" for k, nm in names if k in p)
        self.presets_label.setText(tr("denoise / ControlNet effectifs : {txt}", txt=txt))

    # ----------------------------------------------------------------- serveur ComfyUI
    def _run_task(self, fn, cb, *args) -> None:
        t = Task(fn, *args)
        self._tasks.add(t)                               # garde une référence jusqu'au retour

        def done(res, t=t):
            self._tasks.discard(t)
            if self._alive:                              # fenêtre remplacée ou fermée entre-temps
                cb(res)
        t.signals.done.connect(done)
        self.pool.start(t)

    def _set_pill(self, text: str, state: str) -> None:
        self.server_pill.setText(text)
        self.server_pill.setProperty("state", state)
        repolish(self.server_pill)

    @Slot()
    def test_server(self) -> None:
        if self.provider.currentData() == "mock":
            self._set_pill(tr("Simulation : ComfyUI non utilisé"), "")
            return
        self._set_pill(tr("ComfyUI : test…"), "busy")
        self._run_task(probe_comfyui, lambda r: self._on_probe(r, auto=True), self.server.text().strip())

    def _on_probe(self, r: dict, then=None, auto: bool = False) -> None:
        if auto:
            self._auto_engine(bool(r.get("ok")))
        if not r.get("ok"):
            if self.provider.currentData() == "esrgan":
                self._set_pill(tr("● Real-ESRGAN (sans ComfyUI)"), "")
            else:
                self._set_pill(tr("● ComfyUI injoignable"), "ko")
            self.server_pill.setToolTip(str(r.get("error", "")))
        else:
            free = r.get("vram_free")
            vram = tr(", {gb:.1f} Go libres", gb=free / 2 ** 30) if free else ""
            self._set_pill(f"● ComfyUI {r.get('version', '')} — {r.get('gpu', '?')}{vram}", "ok")
            self.server_pill.setToolTip("")
            for combo, items, empty in ((self.checkpoint, r.get("checkpoints", []), tr(NO_CKPT)),
                                        (self.lora, r.get("loras", []), tr(NO_LORA))):
                if not items:
                    continue
                cur = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItem(empty)
                combo.addItems(items)
                combo.setCurrentText(cur if cur else empty)
                combo.blockSignals(False)
        if then:
            then(r)

    # ----------------------------------------------------------------- lancement
    def _on_launch_clicked(self) -> None:
        if self.launching:                       # double clic pendant le test du serveur
            return
        if self.worker is None:
            self.launch()
        elif self.stop_phase == 0:
            self.request_stop()
        else:
            self.force_stop()

    def launch(self) -> None:
        s = self.collect()
        errs = s.validate()
        out, inp = Path(s.output_dir or "."), Path(s.input_dir or ".")
        if s.output_dir and s.input_dir and inp.resolve() in out.resolve().parents:
            errs.append(tr("le dossier de sortie est à l'intérieur du dossier d'entrée : les DDS produits seraient "
                           "rescannés comme des textures du jeu"))
        if errs:
            QMessageBox.warning(self, APP_NAME, tr("Impossible de lancer :") + "\n\n• " + "\n• ".join(errs))
            return
        # dossier sans aucune texture lisible (installation d'un jeu aux textures empaquetées…) :
        # le traitement ne produirait rien, on explique tout de suite quoi faire
        chk = check_input(s.input_dir, max_entries=20_000)
        if chk.images == 0 and chk.complete:
            QMessageBox.warning(self, APP_NAME, empty_input_help(s.input_dir, chk))
            return
        if s.target == "specialk" and not self._is_inject_dir(s.output_dir):
            sugg = inject_for(s.input_dir)
            ok = QMessageBox.question(
                self, APP_NAME, tr("Special K ne charge les textures remplacées que depuis "
                                   "<jeu>\\SK_Res\\inject\\textures.\n\nDossier de sortie choisi :\n{out}",
                                   out=s.output_dir)
                + (tr("\n\nDossier attendu pour ce dump :\n{inject}", inject=sugg) if sugg else "")
                + "\n\n" + tr("Écrire les DDS dans ce dossier quand même ?")) == QMessageBox.Yes
            if not ok:
                return
        if s.dds_format == "bc7" and s.target == "specialk":
            ok = QMessageBox.question(
                self, APP_NAME, tr("BC7 n'est lisible qu'en Direct3D 11.\n\nLe jeu doit passer par dgVoodoo2 + "
                                   "Special K, et texconv doit être installé (sinon BC3). Continuer ?")) == QMessageBox.Yes
            if not ok:
                return
        if s.provider == "comfyui":
            self.launching = True
            self.btn_launch.setEnabled(False)
            self._set_pill(tr("ComfyUI : test…"), "busy")
            self._run_task(probe_comfyui, lambda r: self._on_probe(r, then=lambda rr: self._after_probe(s, rr)),
                           s.server)
        elif s.provider == "esrgan":
            self._start_esrgan(s)
        else:
            self._start(s)

    @staticmethod
    def _is_inject_dir(path: str) -> bool:
        parts = [x.lower() for x in Path(path).parts]
        return any(a == "sk_res" and b == "inject" for a, b in zip(parts, parts[1:], strict=False))

    def _start_esrgan(self, s: RemasterSettings) -> None:
        """Mode sans ComfyUI : Real-ESRGAN doit être là ; sinon on propose de l'installer, puis on lance."""
        if self._esrgan_exe() is not None:
            self._start(s)
            return
        name, size, _sha = RELEASES[platform_key()]
        ok = QMessageBox.question(
            self, APP_NAME, tr("Real-ESRGAN n'est pas installé.\n\nLe télécharger maintenant ? ({mb:.0f} Mo, version "
                               "officielle de GitHub, xinntao/Real-ESRGAN, empreinte vérifiée)\n\n{name}",
                               mb=size / 2 ** 20, name=name)) == QMessageBox.Yes
        if ok:
            # réglages relus après l'installation : ils contiennent le chemin de Real-ESRGAN
            self.install_esrgan(then=lambda: self._start(self._with_provider(s.provider)) if self.worker is None else None)

    def _with_provider(self, provider: str) -> RemasterSettings:
        s = self.collect()
        s.provider = provider
        return s

    def _after_probe(self, s: RemasterSettings, r: dict) -> None:
        self.launching = False
        self.btn_launch.setEnabled(True)
        if not r.get("ok"):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(APP_NAME)
            box.setText(tr("ComfyUI ne répond pas à {server}.", server=s.server))
            box.setInformativeText(tr("« Utiliser Real-ESRGAN » : rendu réaliste sans ComfyUI (agrandissement net, "
                                      "couleurs\nde l'original, grain de matière), sans IA générative.\n\n"
                                      "« Attendre ComfyUI » : lance-le (run_nvidia_gpu.bat ou python main.py "
                                      "--port 8188) ;\nTEXFORGE l'attend 2 minutes."))
            b_esr = box.addButton(tr("Utiliser Real-ESRGAN"), QMessageBox.AcceptRole)
            b_wait = box.addButton(tr("Attendre ComfyUI"), QMessageBox.ActionRole)
            box.addButton(tr("Annuler"), QMessageBox.RejectRole)
            box.setDefaultButton(b_esr)
            box.exec()
            if box.clickedButton() is b_esr:
                s.provider = "esrgan"
                self.provider.setCurrentIndex(self.provider.findData("esrgan"))
                self._start_esrgan(s)
                return
            if box.clickedButton() is not b_wait:
                return
        self._start(s)

    def _start(self, s: RemasterSettings) -> None:
        self._save_settings()
        recent = [p for p in (self.qs.value("recent_inputs") or []) if p != s.input_dir]
        self.qs.setValue("recent_inputs", [s.input_dir, *recent][:15])
        work = work_dir_of(s)
        work.mkdir(parents=True, exist_ok=True)
        settings_file = work / "gui_settings.json"
        settings_file.write_text(json.dumps(s.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        self.current = s
        self.chips["ai"].setProperty("label", "ESRGAN" if s.provider == "esrgan" else "COMFYUI")
        self._reset_counts()
        for st in STAGES:
            self._stage_state(st, "")
        self.progress.setValue(0)
        self.progress.setFormat(tr("Démarrage…"))
        self.progress_detail.setText("—")
        self.console.add(tr("══ Lancement — {src} → {dst}", src=s.input_dir, dst=s.output_dir))
        self._set_running(True)
        self.t_start = time.monotonic()
        self.clock.start()

        if self.thread is not None and self.thread.isRunning():     # lancement précédent pas encore refermé
            self.thread.wait(3000)
        self.thread = QThread(self)
        self.worker = PipelineWorker(settings_file, work)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.event.connect(self._on_event)
        self.worker.stderr_line.connect(self._on_stderr)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    def request_stop(self) -> None:
        if self.worker is None:
            return
        self.worker.request_stop()
        self.stop_phase = 1
        self.btn_launch.setText(tr("■  FORCER L'ARRÊT"))
        self.btn_launch.setProperty("mode", "kill")
        repolish(self.btn_launch)
        self.console.add(tr("■ Arrêt demandé : les textures en cours se terminent, l'état est enregistré "
                            "(relancer reprend où l'on s'est arrêté)."))

    def force_stop(self) -> None:
        if self.worker is None:
            return
        if QMessageBox.question(self, APP_NAME, tr("Arrêter immédiatement le traitement ?\n\nLes textures en cours "
                                                   "sont perdues (elles seront refaites au prochain lancement) et la "
                                                   "file ComfyUI est vidée.")) != QMessageBox.Yes:
            return
        self.console.add(tr("■ Arrêt forcé du traitement."))
        self.worker.kill()
        if getattr(self, "current", None) and self.current.provider == "comfyui":
            self._run_task(interrupt_comfyui, lambda msg: self.console.add(f"[INFO] {msg}"), self.current.server)

    def _set_running(self, running: bool) -> None:
        self.settings_panel.setEnabled(not running)
        self.folders_box.setEnabled(not running)
        for b in (self.btn_report, self.btn_gallery):
            b.setEnabled(not running)
        self.lang_combo.setEnabled(not running)          # la langue se change entre deux traitements
        self.stop_phase = 0
        if running:
            self.btn_launch.setText(tr("■  ANNULER"))
            self.btn_launch.setProperty("mode", "stop")
        else:
            self.btn_launch.setText(tr("▶  LANCER LE PIPELINE"))
            self.btn_launch.setProperty("mode", "")
        repolish(self.btn_launch)

    # ----------------------------------------------------------------- événements du traitement
    @Slot(object)
    def _on_event(self, ev: dict) -> None:
        e = ev.get("event")
        if e == "progress":
            self._on_progress(ev)
            return
        if e == "comfy_progress":
            self.progress_detail.setText(tr("ComfyUI : {msg}", msg=ev.get("message", "")))
            return
        if e == "pipeline_stage":
            for st in STAGES:
                idx, cur = STAGES.index(st), STAGES.index(ev["stage"])
                self._stage_state(st, "done" if idx < cur else "active" if idx == cur else "")
        elif e == "pipeline_end":
            self.last_end = ev
            if ev.get("status") in ("ok", "errors"):
                for st in STAGES:
                    self._stage_state(st, "done")
        self._count(e, ev)
        text = format_event(ev)
        if text:
            self.console.add(text)

    def _count(self, e: str, ev: dict) -> None:
        key = {"ui_done": "ui", "local_done": "local", "comfy_done": "ai", "esrgan_done": "ai",
               "fallback": "fallback", "failed": "errors"}.get(e)
        if e == "qc_done" and ev.get("decision") in ("validated", "corrected", "rejected"):
            key = ev["decision"]
        if key:
            self.counts[key] += 1
            self._paint_chip(key)

    def _on_progress(self, ev: dict) -> None:
        frac = float(ev.get("fraction", 0))
        self.progress.setValue(round(frac * 1000))
        stage = ev.get("stage", "")
        label = tr(STAGE_LABELS[stage]) if stage in STAGE_LABELS else stage
        done, total = ev.get("done", 0), ev.get("total", 0)
        self.progress.setFormat(f"{frac * 100:.0f} %  —  {label}  {done}/{total}")
        eta = ev.get("eta_s")
        self.progress_detail.setText(tr("{label} : {done}/{total}", label=label, done=done, total=total)
                                     + (tr(" — reste ~{t} pour cette étape", t=fmt_duration(eta)) if eta else ""))

    @Slot(str)
    def _on_stderr(self, line: str) -> None:
        if "%|" in line:            # barres tqdm éventuelles
            return
        self.console.add(f"[STDERR] {line}")

    @Slot(int)
    def _on_finished(self, code: int) -> None:
        self.clock.stop()
        elapsed = fmt_duration(time.monotonic() - self.t_start)
        status = {EXIT_OK: (tr("Pipeline terminé"), "ok"),
                  EXIT_ERRORS: (tr("Terminé avec des échecs (voir la console)"), "warn"),
                  EXIT_STOPPED: (tr("Arrêté — relancer reprend où l'on s'est arrêté"), "stop"),
                  EXIT_FATAL: (tr("Interrompu par une erreur (voir la console)"), "fatal")}
        text, kind = status.get(code, (tr("Processus terminé avec le code {code}", code=code), "fatal"))
        if code == EXIT_OK:
            self.progress.setValue(1000)
            self.progress.setFormat(tr("100 %  —  terminé"))
        elif code == EXIT_STOPPED:
            self.progress.setFormat(tr("{p:.0f} %  —  arrêté", p=self.progress.value() / 10))
        else:
            self.progress.setFormat(f"{self.progress.value() / 10:.0f} %  —  {text}")
        if code not in (EXIT_OK, EXIT_ERRORS, EXIT_STOPPED, EXIT_FATAL):
            self.console.add(tr("[ERREUR] le processus s'est terminé avec le code {code}", code=code))
        self.statusBar().showMessage(f"{text} — {elapsed}", 0)
        self.worker = None                       # le QThread se referme de lui-même (thread.quit)
        self._set_running(False)
        QApplication.alert(self)
        if code in (EXIT_OK, EXIT_ERRORS) and self.last_end.get("report"):
            self.console.add(tr("[INFO] Rapport QC : {path}", path=self.last_end["report"]))

    def _tick(self) -> None:
        self.time_label.setText(f"⏱ {fmt_duration(time.monotonic() - self.t_start)}")

    # ----------------------------------------------------------------- affichage
    def _stage_state(self, stage: str, state: str) -> None:
        lb = self.stage_labels[stage]
        lb.setProperty("state", state)
        repolish(lb)

    def _reset_counts(self) -> None:
        self.counts = {k: 0 for k in self.chips}
        for k in self.chips:
            self._paint_chip(k)

    def _paint_chip(self, key: str) -> None:
        lb = self.chips[key]
        n = self.counts.get(key, 0)
        col = lb.property("color") if n else C["dim"]
        lb.setText(f"<span style='color:{C['muted']}'>{lb.property('label')}</span> "
                   f"<b style='color:{col}'>{n}</b>")

    # ----------------------------------------------------------------- résultats
    def _report_path(self) -> Path | None:
        cands = [self._work_dir() / "work" / "qc_report.json"]
        if self.output_row.text():
            cands.insert(0, Path(self.output_row.text()) / "qc_report.json")
        return next((p for p in cands if p.is_file()), None)

    def _show_report(self) -> None:
        p = self._report_path()
        if p is None:
            QMessageBox.information(self, APP_NAME, tr("Aucun rapport QC pour ce dossier de sortie : "
                                                       "lance d'abord le pipeline."))
            return
        QcReportDialog(p, self).exec()

    def _open_gallery(self) -> None:
        work = self._work_dir()
        if not (work / "texforge.toml").exists():
            QMessageBox.information(self, APP_NAME, tr("Aucun projet pour ce dossier d'entrée : "
                                                       "lance d'abord le pipeline."))
            return
        self.console.add(tr("[INFO] Génération de la galerie avant / après…"))
        self.btn_gallery.setEnabled(False)

        def done(r: dict) -> None:
            self.btn_gallery.setEnabled(self.worker is None)
            if "error" in r:
                self.console.add(tr("[ERREUR] galerie : {err}", err=r["error"]))
            else:
                self.console.add(tr("[INFO] Galerie : {page}", page=r["page"]))
                open_path(r["page"])
        self._run_task(build_gallery, done, work)

    def _copy_command(self) -> None:
        f = self._work_dir() / "gui_settings.json"
        cmd = f'python -m texforge --lang {get_lang()} remaster --settings "{f}"'
        QGuiApplication.clipboard().setText(cmd)
        self.statusBar().showMessage(tr("Copié : {cmd}", cmd=cmd), 6000)

    def _save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr("Enregistrer le journal"),
                                              str(Path.home() / tr("texforge_journal.txt")), tr("Texte (*.txt)"))
        if path:
            Path(path).write_text("\n".join(self.console.lines) + "\n", encoding="utf-8")

    # ----------------------------------------------------------------- langue
    def _on_lang_changed(self, _index: int) -> None:
        code = self.lang_combo.currentData()
        if code == get_lang():
            return
        if self.worker is not None or self.launching:
            self.lang_combo.blockSignals(True)
            self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(get_lang())))
            self.lang_combo.blockSignals(False)
            QMessageBox.information(self, APP_NAME, tr("La langue se change une fois le traitement terminé."))
            return
        self._save_settings()
        self.qs.setValue("lang", code)
        # après le retour du signal : la fenêtre (et ce menu) va être remplacée
        QTimer.singleShot(0, lambda: switch_language(self, code))

    # ----------------------------------------------------------------- fermeture
    def closeEvent(self, event) -> None:
        if self.worker is not None:
            if QMessageBox.question(self, APP_NAME, tr("Un traitement est en cours. L'arrêter et quitter ?\n\n"
                                                       "L'état est enregistré : relancer reprendra où l'on s'est "
                                                       "arrêté.")) != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.kill()
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait(5000)
        self._save_settings()
        self._alive = False
        if not self._switching:                  # au changement de langue, la nouvelle fenêtre reste ouverte
            self.pool.waitForDone(3000)
        event.accept()


def initial_language() -> str:
    """Premier lancement : TEXFORGE_LANG, sinon la langue du système vue par Qt (réglages régionaux de
    Windows ou de macOS, variables LANG/LC_* sous Linux)."""
    env = os.environ.get("TEXFORGE_LANG", "").strip().lower()[:2]
    if env in LANGUAGES:
        return env
    sys_lang = QLocale.system().name()[:2].lower()
    return sys_lang if sys_lang in LANGUAGES else detect_language()


def switch_language(old: MainWindow, code: str) -> MainWindow:
    """Reconstruit la fenêtre dans la nouvelle langue (réglages, taille et journal conservés).
    Reconstruire plutôt que retraduire chaque widget : aucun libellé ne peut être oublié."""
    app = QApplication.instance()
    apply_language(app, code)
    new = MainWindow(log=list(old.console.lines))
    new.setGeometry(old.geometry())
    new.splitter.setSizes(old.splitter.sizes())
    new.log_filter.setCurrentIndex(old.log_filter.currentIndex())
    if old.isMaximized():
        new.showMaximized()
    else:
        new.show()
    old._switching = True
    old.close()
    old.deleteLater()
    _ACTIVE["window"] = new                          # référence Python : une fenêtre sans parent serait libérée
    return new


def main() -> int:
    QApplication.setApplicationName(APP_NAME)
    QApplication.setOrganizationName(APP_NAME)
    app = QApplication(sys.argv)
    lang = QSettings(APP_NAME, APP_NAME).value("lang", "") or initial_language()
    apply_language(app, str(lang))
    apply_theme(app)
    app.setWindowIcon(app_icon())
    win = MainWindow()
    win.show()
    _ACTIVE["window"] = win
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
