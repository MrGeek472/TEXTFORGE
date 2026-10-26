"""Mise à jour de TEXFORGE depuis son dépôt GitHub (MrGeek472/TEXTFORGE).

Vérification : dernier commit de la branche par défaut (API GitHub) et version lue dans son
texforge/__init__.py, comparée à celle installée sur le disque. Seule une version plus récente est
proposée : une installation plus récente que le dépôt (version pas encore publiée) reste telle quelle.

Mise à jour : archive de ce commit précis, contrôlée avant d'écrire quoi que ce soit — chemins sûrs,
version attendue, compilation de chaque fichier Python, modules importés présents dans l'archive,
dépendances (pip si requirements.txt a changé), import d'essai de la nouvelle version dans un processus
à part. Ensuite seulement, les fichiers qui diffèrent sont remplacés un par un (écriture puis
remplacement atomique), après copie des originaux dans une sauvegarde ; une erreur d'écriture remet tout
en place. Les fichiers locaux absents du dépôt ne sont jamais supprimés. Retour arrière : rollback().
"""

from __future__ import annotations

import ast
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from . import __version__
from .i18n import tr

REPO = "MrGeek472/TEXTFORGE"
ROOT = Path(__file__).resolve().parent.parent       # dossier d'installation (app_gui.py, texforge/…)
SKIP_DIRS = {"__pycache__", ".git", ".github", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
SKIP_FILES = {".gitignore", ".gitattributes"}
SKIP_SUFFIXES = (".pyc", ".pyo")
ENTRY_POINTS = ("texforge/__init__.py", "texforge/__main__.py", "app_gui.py")
MAX_ARCHIVE = 300 * 2 ** 20
KEEP_BACKUPS = 3
NETWORK_ERRORS = (OSError, http.client.HTTPException, ValueError)


class UpdateError(Exception):
    """Mise à jour refusée ou impossible ; l'installation n'a pas été modifiée (ou a été restaurée)."""


# ------------------------------------------------------------------ réglages (variables d'environnement : tests)

def repo() -> str:
    return os.environ.get("TEXFORGE_UPDATE_REPO") or REPO


def _base(kind: str, default: str) -> str:
    return (os.environ.get(f"TEXFORGE_UPDATE_{kind}") or default).rstrip("/")


def install_root() -> Path:
    env = os.environ.get("TEXFORGE_UPDATE_ROOT")
    return Path(env).resolve() if env else ROOT


def data_dir() -> Path:
    if os.environ.get("TEXFORGE_UPDATE_DIR"):
        return Path(os.environ["TEXFORGE_UPDATE_DIR"])
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "TEXFORGE"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "texforge"
    return base / "mises_a_jour"


def repo_url() -> str:
    return f"https://github.com/{repo()}"


# ------------------------------------------------------------------ versions

def vtuple(v: str | None) -> tuple[int, ...]:
    parts = re.findall(r"\d+", v or "")
    return tuple(int(p) for p in parts[:4]) or (0,)


def version_in(text: str) -> str | None:
    m = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", text, re.M)
    return m.group(1) if m else None


def installed_version(root: Path | None = None) -> str:
    """Version des fichiers sur le disque (après une mise à jour, plus récente que celle qui tourne)."""
    try:
        text = ((root or install_root()) / "texforge" / "__init__.py").read_text(encoding="utf-8")
    except OSError:
        return __version__
    return version_in(text) or __version__


# ------------------------------------------------------------------ réseau

def _open(url: str, timeout: float, accept: str | None = None):
    headers = {"User-Agent": f"TEXFORGE/{__version__}"}
    if accept:
        headers["Accept"] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def _get(url: str, timeout: float, accept: str | None = None) -> bytes:
    with _open(url, timeout, accept) as r:
        return r.read(4 * 2 ** 20)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (403, 429):
            return tr("GitHub limite les requêtes, nouvel essai plus tard")
        if exc.code == 404:
            return tr("dépôt {repo} introuvable ou privé", repo=repo())
        return f"HTTP {exc.code}"
    return str(getattr(exc, "reason", None) or exc)


def check(timeout: float = 15.0) -> dict:
    """État de la mise à jour : versions installée, en cours et du dépôt, commit, message, erreur éventuelle."""
    root = install_root()
    local = installed_version(root)
    out = {"repo": repo(), "url": repo_url(), "local": local, "running": __version__, "remote": None,
           "available": False, "newer_local": False, "sha": None, "date": "", "message": "", "error": "",
           "git": (root / ".git").exists(), "restart": vtuple(local) > vtuple(__version__),
           "backup": (last_backup(root) or {}).get("from")}
    try:
        j = json.loads(_get(f"{_base('API', 'https://api.github.com')}/repos/{repo()}/commits/HEAD", timeout,
                            "application/vnd.github+json"))
        out["sha"] = str(j["sha"])
        commit = j.get("commit") or {}
        out["date"] = str((commit.get("committer") or {}).get("date") or "")
        lines = str(commit.get("message") or "").strip().splitlines()
        out["message"] = lines[0][:200] if lines else ""
    except (*NETWORK_ERRORS, KeyError, TypeError):
        pass                                  # API limitée ou injoignable : version lue sur la branche par défaut
    try:
        text = _get(f"{_base('RAW', 'https://raw.githubusercontent.com')}/{repo()}/{out['sha'] or 'HEAD'}"
                    f"/texforge/__init__.py", timeout).decode("utf-8", "replace")
    except NETWORK_ERRORS as exc:
        out["error"] = tr("vérification impossible : {why}", why=_reason(exc))
        return out
    remote = version_in(text)
    if not remote:
        out["error"] = tr("version introuvable dans le dépôt {repo}", repo=repo())
        return out
    out["remote"] = remote
    out["available"] = vtuple(remote) > vtuple(local)
    out["newer_local"] = vtuple(remote) < vtuple(local)
    return out


def _download(url: str, dest: Path, progress: Callable[[int, int], None] | None) -> None:
    got = 0
    try:
        with _open(url, 60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got > MAX_ARCHIVE:
                    raise UpdateError(tr("archive du dépôt trop grosse (plus de {mb} Mo) : mise à jour annulée",
                                         mb=MAX_ARCHIVE // 2 ** 20))
                if progress:
                    progress(got, total)
    except NETWORK_ERRORS as exc:
        raise UpdateError(tr("téléchargement impossible : {why}", why=_reason(exc))) from exc


# ------------------------------------------------------------------ contrôle de l'archive

def _extract(archive: Path, dest: Path) -> dict[str, Path]:
    """Fichiers de l'archive GitHub (dossier racine « dépôt-commit/ » retiré), chemins vérifiés."""
    try:
        z = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise UpdateError(tr("archive du dépôt illisible : {err}", err=exc)) from exc
    files: dict[str, Path] = {}
    with z:
        members = [m for m in z.infolist() if not m.is_dir()]
        names = [m.filename.replace("\\", "/") for m in members]
        tops = {n.split("/", 1)[0] for n in names}
        prefix = f"{tops.pop()}/" if len(tops) == 1 and all("/" in n for n in names) else ""
        for m, n in zip(members, names, strict=True):
            rel = n[len(prefix):]
            parts = rel.split("/")
            if not rel or rel.startswith("/") or ".." in parts or ":" in parts[0]:
                raise UpdateError(tr("chemin refusé dans l'archive : {name}", name=n))
            if (m.external_attr >> 16) & 0o170000 == 0o120000:
                continue                                   # lien symbolique : ignoré
            if any(p in SKIP_DIRS for p in parts[:-1]) or parts[-1] in SKIP_FILES or rel.endswith(SKIP_SUFFIXES):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            files[rel] = out
    if not files:
        raise UpdateError(tr("archive du dépôt vide"))
    return files


def _optional_imports(tree: ast.AST) -> set[int]:
    """Imports placés dans un try … except ImportError : facultatifs, non exigés dans l'archive."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            names = set()
            for h in node.handlers:
                t = h.type
                for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                    if isinstance(e, ast.Name):
                        names.add(e.id)
            if names & {"ImportError", "ModuleNotFoundError", "Exception"}:
                for stmt in node.body:
                    skip.update(id(n) for n in ast.walk(stmt))
    return skip


def missing_imports(files: dict[str, Path], root: Path) -> list[tuple[str, str]]:
    """(fichier, module) des imports de modules du projet absents de l'archive : un dépôt incomplet
    mélangerait des fichiers neufs et anciens."""
    def exists(mod: str) -> bool:
        p = mod.replace(".", "/")
        return f"{p}.py" in files or f"{p}/__init__.py" in files

    tops = {r[:-3] for r in files if "/" not in r and r.endswith(".py")}
    tops |= {r.split("/")[0] for r in files if r.count("/") == 1 and r.endswith("/__init__.py")}
    try:
        tops |= {p.stem for p in root.glob("*.py")} | {p.parent.name for p in root.glob("*/__init__.py")}
    except OSError:
        pass
    missing: list[tuple[str, str]] = []
    for rel, path in sorted(files.items()):
        if not rel.endswith(".py"):
            continue
        tree = ast.parse(path.read_bytes(), rel)
        skip = _optional_imports(tree)
        pkg = rel[:-3].split("/")[:-1] if not rel.endswith("/__init__.py") else rel.split("/")[:-1]
        for node in ast.walk(tree):
            if id(node) in skip:
                continue
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in tops and not exists(a.name):
                        missing.append((rel, a.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    mod = node.module or ""
                    if mod.split(".")[0] in tops and not exists(mod):
                        missing.append((rel, mod))
                elif node.module:
                    base = pkg[:len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                    mod = ".".join([*base, node.module])
                    if not exists(mod):
                        missing.append((rel, mod))
    return missing


def _validate(new: Path, files: dict[str, Path], remote: str, root: Path) -> None:
    absent = [e for e in ENTRY_POINTS if e not in files]
    if absent:
        raise UpdateError(tr("dépôt incomplet, fichier(s) absent(s) : {files} — rien n'a été modifié",
                             files=", ".join(absent)))
    got = version_in(files["texforge/__init__.py"].read_text(encoding="utf-8", errors="replace"))
    if got != remote:
        raise UpdateError(tr("l'archive contient la version {got} au lieu de {want} — rien n'a été modifié",
                             got=got, want=remote))
    for rel, path in sorted(files.items()):
        if rel.endswith(".py"):
            try:
                compile(path.read_bytes(), rel, "exec")
            except SyntaxError as exc:
                raise UpdateError(tr("erreur de syntaxe dans {file} (ligne {line}) : version du dépôt cassée, rien "
                                     "n'a été modifié", file=rel, line=exc.lineno)) from exc
    miss = missing_imports(files, root)
    if miss:
        shown = ", ".join(f"{m.replace('.', '/')}.py ({tr('importé par {file}', file=f)})" for f, m in miss[:4])
        raise UpdateError(tr("dépôt incomplet, module(s) absent(s) : {list} — rien n'a été modifié. Publie tous "
                             "les fichiers de la version sur GitHub.", list=shown))


def _python() -> str:
    """Interpréteur avec console pour pip et l'import d'essai (pythonw.exe n'a pas de sortie)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def _run(cmd: list[str], cwd: Path, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _requirements(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return sorted(ln.split("#", 1)[0].strip().lower() for ln in lines if ln.split("#", 1)[0].strip())


def _tail(text: str, n: int = 3) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[-400:]


def _pip_install(req: Path, python: str, say: Callable[[str], None]) -> None:
    cmd = [python, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)]
    try:
        r = _run(cmd, req.parent, 900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateError(tr("installation des dépendances impossible : {err}", err=exc)) from exc
    if r.returncode != 0:
        raise UpdateError(tr("installation des dépendances impossible (pip, code {code}) : {tail}\nÀ lancer à la "
                             "main : {cmd}", code=r.returncode, tail=_tail(r.stderr or r.stdout),
                             cmd="python -m pip install -r requirements.txt"))
    say(tr("[MAJ] Dépendances à jour."))


def _smoke_import(new: Path, python: str) -> None:
    """La nouvelle version s'importe-t-elle (modules, dépendances, code exécuté au chargement) ?"""
    code = ("import sys, importlib.util; sys.path.insert(0, sys.argv[1]); import texforge.__main__; "
            "importlib.util.find_spec('PySide6') and __import__('app_gui'); print(texforge.__version__)")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1")
    try:
        r = _run([python, "-c", code, str(new)], new, 180, env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateError(tr("import d'essai impossible : {err} — rien n'a été modifié", err=exc)) from exc
    if r.returncode != 0:
        raise UpdateError(tr("la nouvelle version ne démarre pas (import d'essai) : {err} — rien n'a été modifié",
                             err=_tail(r.stderr or r.stdout)))


# ------------------------------------------------------------------ installation, sauvegarde, retour arrière

def _write_manifest(folder: Path, manifest: dict) -> None:
    tmp = folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, folder / "manifest.json")


def _put(src: Path, dst: Path) -> None:
    """Remplacement atomique d'un fichier (copie à côté, puis os.replace)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".texforge-maj")
    shutil.copyfile(src, part)
    if dst.exists():
        try:
            shutil.copymode(dst, part)
        except OSError:
            pass
    os.replace(part, dst)


def _restore(root: Path, folder: Path, manifest: dict) -> None:
    for rel in manifest.get("replaced", []):
        b = folder / "fichiers" / rel
        if b.is_file():
            _put(b, root / rel)
    for rel in manifest.get("added", []):
        (root / rel).unlink(missing_ok=True)
        parent = (root / rel).parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()                       # dossiers créés par la mise à jour, restés vides
            parent = parent.parent


def _install(files: dict[str, Path], root: Path, info: dict, local: str, backup: Path,
             say: Callable[[str], None]) -> dict:
    plan: list[tuple[str, Path, Path, bool]] = []
    unchanged = 0
    for rel, src in sorted(files.items()):
        dst = root / rel
        try:
            dst.resolve().relative_to(root)
        except ValueError as exc:
            raise UpdateError(tr("chemin refusé dans l'archive : {name}", name=rel)) from exc
        if dst.is_file():
            if dst.read_bytes() == src.read_bytes():
                unchanged += 1
                continue
            plan.append((rel, src, dst, True))
        elif dst.exists():
            raise UpdateError(tr("{path} existe déjà et n'est pas un fichier : rien n'a été modifié", path=dst))
        else:
            plan.append((rel, src, dst, False))
    result = {"from": local, "to": info["remote"], "replaced": 0, "added": 0, "unchanged": unchanged, "backup": ""}
    if not plan:
        return result
    backup.mkdir(parents=True, exist_ok=True)
    manifest = {"from": local, "to": info["remote"], "sha": info.get("sha"), "repo": repo(), "root": str(root),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"), "replaced": [], "added": [], "complete": False,
                "rolled_back": False}
    try:
        for rel, src, dst, existed in plan:
            if existed:
                b = backup / "fichiers" / rel
                b.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, b)
                manifest["replaced"].append(rel)
            else:
                manifest["added"].append(rel)
            _write_manifest(backup, manifest)       # avant l'écriture : un arrêt brutal reste réparable
            _put(src, dst)
    except OSError as exc:
        try:
            _restore(root, backup, manifest)
            manifest["rolled_back"] = True
            _write_manifest(backup, manifest)
        except OSError as exc2:
            raise UpdateError(tr("écriture impossible ({err}) et restauration incomplète ({err2}) : sauvegarde "
                                 "dans {path}", err=exc, err2=exc2, path=backup)) from exc
        raise UpdateError(tr("écriture impossible ({err}) : fichiers d'origine restaurés", err=exc)) from exc
    manifest["complete"] = True
    _write_manifest(backup, manifest)
    say(tr("[MAJ] {n} fichier(s) remplacé(s), {a} ajouté(s), {u} identique(s) ; originaux sauvegardés dans {path}",
           n=len(manifest["replaced"]), a=len(manifest["added"]), u=unchanged, path=backup))
    result.update(replaced=len(manifest["replaced"]), added=len(manifest["added"]), backup=str(backup))
    return result


def _prune(work: Path) -> None:
    """Garde les dernières sauvegardes ; efface les dossiers temporaires d'essais interrompus."""
    try:
        backups = sorted((p for p in work.glob("sauvegarde-*") if p.is_dir()), key=lambda p: p.name, reverse=True)
        for old in backups[KEEP_BACKUPS:]:
            shutil.rmtree(old, ignore_errors=True)
        for tmp in work.glob("maj-*"):
            if tmp.is_dir() and time.time() - tmp.stat().st_mtime > 86400:
                shutil.rmtree(tmp, ignore_errors=True)
    except OSError:
        pass


def apply(info: dict, progress: Callable[[int, int], None] | None = None,
          log: Callable[[str], None] | None = None, root: Path | None = None) -> dict:
    """Installe la version du dépôt décrite par check(). Rien n'est écrit tant que l'archive n'a pas passé
    tous les contrôles ; renvoie {"from", "to", "replaced", "added", "unchanged", "backup", "pip"}."""
    root = (root or install_root()).resolve()
    say = log or (lambda _m: None)
    if (root / ".git").exists():
        raise UpdateError(tr("installation suivie par git ({root}) : mets-la à jour avec « git pull »", root=root))
    local = installed_version(root)
    remote = info.get("remote")
    if not remote or vtuple(remote) <= vtuple(local):
        raise UpdateError(tr("aucune version plus récente à installer (installée {local}, dépôt {remote})",
                             local=local, remote=remote or "?"))
    work = data_dir()
    work.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="maj-", dir=work))
    python = _python()
    try:
        say(tr("[MAJ] Téléchargement de TEXFORGE {v} depuis GitHub ({repo})…", v=remote, repo=repo()))
        archive = tmp / "depot.zip"
        _download(f"{_base('CODELOAD', 'https://codeload.github.com')}/{repo()}/zip/{info.get('sha') or 'HEAD'}",
                  archive, progress)
        new = tmp / "depot"
        files = _extract(archive, new)
        say(tr("[MAJ] Contrôle de la nouvelle version ({n} fichiers)…", n=len(files)))
        _validate(new, files, remote, root)
        pip = _requirements(new / "requirements.txt") != _requirements(root / "requirements.txt") and \
            (new / "requirements.txt").is_file()
        if pip:
            say(tr("[MAJ] Dépendances modifiées : installation avec pip…"))
            _pip_install(new / "requirements.txt", python, say)
        say(tr("[MAJ] Import d'essai de la nouvelle version…"))
        _smoke_import(new, python)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result = _install(files, root, info, local, work / f"sauvegarde-{stamp}-{local}", say)
        result["pip"] = pip
        _prune(work)
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def last_backup(root: Path | None = None) -> dict | None:
    """Dernière mise à jour complète de cette installation, restaurable (manifeste + dossier)."""
    root = (root or install_root()).resolve()
    try:
        folders = sorted(data_dir().glob("sauvegarde-*/manifest.json"), key=lambda p: p.parent.name, reverse=True)
    except OSError:
        return None
    for mf in folders:
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if Path(m.get("root", "")).resolve() == root and m.get("complete") and not m.get("rolled_back"):
            return {**m, "dir": str(mf.parent)}
    return None


def rollback(root: Path | None = None) -> dict:
    """Remet les fichiers d'avant la dernière mise à jour (et retire ceux qu'elle avait ajoutés)."""
    root = (root or install_root()).resolve()
    m = last_backup(root)
    if m is None:
        raise UpdateError(tr("aucune sauvegarde de mise à jour à restaurer"))
    cur = installed_version(root)
    if cur != m["to"]:
        raise UpdateError(tr("la version installée ({cur}) n'est plus celle de la dernière mise à jour ({to}) : "
                             "retour arrière annulé", cur=cur, to=m["to"]))
    folder = Path(m.pop("dir"))
    try:
        _restore(root, folder, m)
    except OSError as exc:
        raise UpdateError(tr("restauration incomplète : {err} (sauvegarde dans {path})", err=exc, path=folder)) from exc
    m["rolled_back"] = True
    _write_manifest(folder, m)
    return {"from": m["to"], "to": m["from"], "restored": len(m.get("replaced", [])),
            "removed": len(m.get("added", []))}


def restart_command(root: Path | None = None) -> list[str]:
    """Commande qui relance l'interface (même interpréteur : pythonw.exe garde l'absence de console)."""
    return [sys.executable, str((root or install_root()) / "app_gui.py")]
