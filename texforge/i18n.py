"""Langue de l'interface et des messages.

Le texte source reste en français dans le code :

    tr("Dossier de sortie")                       -> "Output folder" si la langue active est l'anglais
    tr("{n} texture(s) à traiter", n=12)          -> "12 texture(s) to process"

Une chaîne absente du catalogue s'affiche en français : une traduction manquante ne casse rien.
N_("…") marque une chaîne traduite plus tard (constantes de module) ; elle est renvoyée telle quelle.

Langue active, dans l'ordre : set_lang() (interface graphique, option --lang), variable
d'environnement TEXFORGE_LANG, langue du système (français si elle est française, anglais sinon).

Ajouter une langue : copier texforge/locales/en.py vers <code>.py, traduire les valeurs, ajouter
l'entrée dans LANGUAGES. `python -m texforge selftest` signale les chaînes manquantes.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import string
from pathlib import Path

LANGUAGES: dict[str, str] = {"fr": "Français", "en": "English"}     # code -> nom affiché dans le sélecteur
SOURCE = "fr"

_lang: str | None = None
_catalog: dict[str, str] = {}


def detect() -> str:
    """Langue par défaut : TEXFORGE_LANG, sinon celle du système."""
    env = os.environ.get("TEXFORGE_LANG", "").strip().lower()[:2]
    if env in LANGUAGES:
        return env
    return system_lang()


def system_lang() -> str:
    """Langue du système : interface de Windows, sinon variables POSIX (LC_ALL > LC_MESSAGES > LANG),
    sinon la locale de Python. Français si elle est française, anglais sinon."""
    loc = ""
    if os.name == "nt":
        try:
            import ctypes
            primary = ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF
            loc = {0x0C: "fr"}.get(primary, "en")
        except (AttributeError, OSError):
            loc = ""
    if not loc:
        loc = os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES") or os.environ.get("LANG") or ""
    if not loc or loc.upper() in ("C", "POSIX") or loc.upper().startswith("C."):
        import locale
        try:
            loc = locale.getlocale()[0] or ""
        except ValueError:
            loc = ""
    return "fr" if loc.lower().startswith(("fr", "french")) else "en"


def load_catalog(code: str) -> dict[str, str]:
    if code == SOURCE:
        return {}
    try:
        mod = importlib.import_module(f"texforge.locales.{code}")
    except ImportError:
        return {}
    return dict(getattr(mod, "MESSAGES", {}))


def set_lang(code: str | None) -> str:
    """Change la langue active (code inconnu : langue détectée). Renvoie le code retenu."""
    global _lang, _catalog
    code = (code or "").strip().lower()[:2]
    if code not in LANGUAGES:
        code = detect()
    _lang, _catalog = code, load_catalog(code)
    os.environ["TEXFORGE_LANG"] = code          # transmis aux processus enfants
    return code


def get_lang() -> str:
    if _lang is None:
        set_lang(detect())
    return _lang  # type: ignore[return-value]


def tr(text: str, /, **kw) -> str:
    """Traduction de `text` (modèle str.format si des arguments nommés sont donnés)."""
    if _lang is None:
        get_lang()
    out = _catalog.get(text, text)
    if not kw:
        return out
    try:
        return out.format(**kw)
    except (KeyError, IndexError, ValueError):       # traduction défectueuse : texte d'origine
        return text.format(**kw)


def N_(text: str) -> str:
    """Marque une chaîne à traduire au moment de l'affichage (tr(constante))."""
    return text


# ------------------------------------------------------------------ contrôle des catalogues

def _fields(s: str) -> list[str]:
    """Champs {nom} et marqueurs %s / %d / %.1f (journal logging) d'un modèle : la traduction doit garder
    les mêmes (les %-marqueurs dans le même ordre)."""
    try:
        named = sorted({f for _lit, f, _spec, _conv in string.Formatter().parse(s) if f})
    except ValueError:
        return ["<modèle invalide>"]
    return named + re.findall(r"%[-+ #0]*\d*(?:\.\d+)?[sdfrxi]", s)


def source_strings(root: Path) -> dict[str, str]:
    """Chaînes passées littéralement à tr() / N_() dans le code : {texte: fichier:ligne}."""
    found: dict[str, str] = {}
    files = [root / "app_gui.py", root / "gui_compare.py", *sorted((root / "texforge").glob("*.py"))]
    for f in files:
        if not f.is_file():
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("tr", "N_")
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                found.setdefault(node.args[0].value, f"{f.name}:{node.lineno}")
    return found


def check_catalogs(root: Path) -> list[str]:
    """Problèmes des catalogues : chaînes manquantes, champs {…} différents, entrées orphelines."""
    src = source_strings(root)
    problems: list[str] = []
    for code in LANGUAGES:
        if code == SOURCE:
            continue
        cat = load_catalog(code)
        for text, where in src.items():
            if text not in cat:
                problems.append(f"[{code}] traduction manquante ({where}) : {text!r}")
            elif _fields(cat[text]) != _fields(text):
                problems.append(f"[{code}] champs différents ({where}) : {text!r} -> {cat[text]!r}")
        for text in cat:
            if text not in src:
                problems.append(f"[{code}] entrée inutilisée : {text!r}")
    return problems


__all__ = ["LANGUAGES", "detect", "system_lang", "set_lang", "get_lang", "tr", "N_", "check_catalogs",
           "source_strings"]
