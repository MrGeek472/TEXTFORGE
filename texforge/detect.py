"""Détection automatique des dossiers de textures extraites.

Dossiers reconnus (rien n'est lancé, on lit seulement le disque et le registre) :
  * Special K, injection locale   : <jeu>\\…\\SK_Res\\dump\\textures (à côté de l'exécutable) ;
  * Special K, injection globale  : <Special K>\\Profiles\\<jeu>\\SK_Res\\dump\\textures ;
  * RTX Remix                     : <jeu>\\…\\rtx-remix\\captures (\\textures s'il existe).

Jeux parcourus : bibliothèques Steam (libraryfolders.vdf), manifestes Epic, GOG et programmes
installés (registre Windows), dossiers usuels de chaque disque local (X:\\Games, X:\\Jeux,
X:\\GOG Games…), dossiers de TEXFORGE_SEARCH (séparés par « ; ») et dossiers déjà utilisés.

Chaque dossier de jeu est exploré sur 4 niveaux au plus, avec un plafond de dossiers visités :
la recherche reste de l'ordre de la seconde même avec des centaines de jeux installés.
Résultat : candidats triés du dump le plus récent au plus ancien.
"""

from __future__ import annotations

import heapq
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .i18n import N_, tr
from .imaging import IMAGE_EXTS

MARKERS = {"sk_res": "specialk", "rtx-remix": "remix"}
# dossiers sans intérêt ou très volumineux : jamais explorés
SKIP_DIRS = {".git", ".svn", "__pycache__", "node_modules", "$recycle.bin", "system volume information", "windows",
             "winsxs", "_commonredist", "commonredist", "redist", "redistributables", "directx", "vcredist",
             "dotnet", "logs", "crashreports", "screenshots", "shadercache", "movies", "videos", "music",
             "steamworks shared", "steamvr"}
# noms de dossiers d'exécutable : on remonte au-dessus pour nommer le jeu
GENERIC_DIRS = {"bin", "bin32", "bin64", "binaries", "win32", "win64", "x86", "x64", "game", "system", "exe",
                "retail", "release", "shipping"}
COMMON_LIBRARIES = ["Games", "Jeux", "GOG Games", "Epic Games", "SteamLibrary\\steamapps\\common",
                    "Steam\\steamapps\\common", "Games\\Steam\\steamapps\\common",
                    "Program Files (x86)\\Steam\\steamapps\\common", "Program Files (x86)\\GOG Galaxy\\Games",
                    "Program Files\\Epic Games", "Program Files (x86)\\Origin Games", "Program Files\\EA Games",
                    "Program Files (x86)\\Ubisoft\\Ubisoft Game Launcher\\games"]


@dataclass
class Candidate:
    path: str            # dossier à donner en entrée
    kind: str            # specialk | remix
    game: str            # nom affiché
    count: int           # textures trouvées (sous-dossiers compris)
    newest: float        # date de la texture la plus récente (0 si aucune)
    inject: str = ""     # Special K : dossier d'injection correspondant (SK_Res\inject\textures)
    source: str = ""     # origine (texte français, tr() à l'affichage) : Steam, Epic, profil Special K…

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ outils disque

def _norm(p: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _child(d: Path, name: str) -> Path | None:
    """Sous-dossier `name` sans tenir compte de la casse (systèmes de fichiers sensibles à la casse)."""
    direct = d / name
    if direct.is_dir():
        return direct
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.lower() == name.lower() and e.is_dir():
                    return Path(e.path)
    except OSError:
        pass
    return None


def _subdirs(d: Path):
    try:
        with os.scandir(d) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        yield e
                except OSError:
                    continue
    except OSError:
        return


def has_images(d: Path, limit: int = 4000) -> bool:
    """Le dossier contient-il directement des images (sans descendre) ?"""
    try:
        with os.scandir(d) as it:
            for i, e in enumerate(it):
                if i >= limit:
                    break
                if os.path.splitext(e.name)[1].lower() in IMAGE_EXTS:
                    return True
    except OSError:
        pass
    return False


def texture_stats(d: Path, max_files: int = 300_000, max_dirs: int = 20_000) -> tuple[int, float]:
    """(nombre d'images, date de la plus récente), sous-dossiers compris."""
    count, newest, dirs = 0, 0.0, 0
    stack = [d]
    while stack and dirs < max_dirs and count < max_files:
        cur = stack.pop()
        dirs += 1
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif os.path.splitext(e.name)[1].lower() in IMAGE_EXTS:
                            count += 1
                            newest = max(newest, e.stat().st_mtime)
                    except OSError:
                        continue
        except OSError:
            continue
    return count, newest


# exploration prioritaire : SK_Res et rtx-remix sont à côté de l'exécutable
EXE_HINTS = {"bin", "bin32", "bin64", "binaries", "win32", "win64", "x86", "x64", "system", "game", "retail",
             "shipping", "release", "exe"}
HEAVY_HINTS = {"engine", "content", "plugins", "data", "assets", "resources", "media", "sound", "sounds", "audio",
               "localization", "maps", "levels", "packs", "paks", "textures", "models", "shaders", "docs", "manual",
               "extras", "mods", "saves", "config", "cache"}


def find_markers(root: Path, depth: int = 4, budget: int = 800) -> list[Path]:
    """Dossiers SK_Res / rtx-remix sous `root` : exploration au plus `depth` niveaux et `budget` dossiers,
    les dossiers d'exécutable (Binaries, Win64, Bin…) d'abord, les dossiers de données en dernier."""
    found: list[Path] = []
    heap: list[tuple[int, int, Path, int]] = [(0, 0, root, 0)]
    order = visited = 0
    while heap and visited < budget:
        _prio, _n, cur, lvl = heapq.heappop(heap)
        visited += 1
        for e in _subdirs(cur):
            low = e.name.lower()
            if low in MARKERS:
                found.append(Path(e.path))
            elif lvl + 1 < depth and low not in SKIP_DIRS and not low.startswith("."):
                prio = (lvl + 1) * 10 - (15 if low in EXE_HINTS else 0) + (15 if low in HEAVY_HINTS else 0)
                order += 1
                heapq.heappush(heap, (prio, order, Path(e.path), lvl + 1))
    return found


def game_name(marker: Path, root: Path | None = None) -> str:
    """Nom du jeu : dossier racine fourni, sinon premier parent au nom non générique."""
    if root is not None and _norm(root) != _norm(marker.parent) and root.name and root.name.lower() not in GENERIC_DIRS:
        return root.name
    p = marker.parent
    while p.name and p.name.lower() in GENERIC_DIRS and p.parent != p:
        p = p.parent
    return p.name or str(p)


def from_marker(marker: Path, game: str, source: str) -> Candidate | None:
    """Dossier de textures correspondant à un SK_Res ou un rtx-remix (None s'il n'y en a pas)."""
    kind = MARKERS.get(marker.name.lower())
    if kind == "specialk":
        dump = _child(marker, "dump")
        tex = _child(dump, "textures") if dump else None
        if tex is None:
            return None
        inject_root = _child(marker, "inject")
        inj = (_child(inject_root, "textures") if inject_root else None) or marker / "inject" / "textures"
        count, newest = texture_stats(tex)
        return Candidate(str(tex), kind, game, count, newest, str(inj), source)
    if kind == "remix":
        cap = _child(marker, "captures")
        if cap is None:
            return None
        tex = _child(cap, "textures") or cap
        count, newest = texture_stats(tex)
        return Candidate(str(tex), kind, game, count, newest, "", source)
    return None


# ------------------------------------------------------------------ où chercher

def _registry_values(hive_path: list[tuple[int, str]], fields: tuple[str, ...]) -> list[dict]:
    """Sous-clés de registre (Windows) : valeurs demandées de chaque sous-clé."""
    try:
        import winreg
    except ImportError:
        return []
    out = []
    for hive, path in hive_path:
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                key = winreg.OpenKey(hive, path, 0, winreg.KEY_READ | view)
            except OSError:
                continue
            with key:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(key, i)
                    except OSError:
                        break
                    i += 1
                    try:
                        with winreg.OpenKey(key, sub, 0, winreg.KEY_READ | view) as sk:
                            row = {}
                            for f in fields:
                                try:
                                    row[f] = str(winreg.QueryValueEx(sk, f)[0])
                                except OSError:
                                    pass
                            out.append(row)
                    except OSError:
                        continue
    return out


def _reg_value(hive: int, path: str, name: str) -> str:
    try:
        import winreg
        with winreg.OpenKey(hive, path) as k:
            return str(winreg.QueryValueEx(k, name)[0])
    except (ImportError, OSError):
        return ""


def parse_libraryfolders(text: str) -> list[str]:
    """Chemins des bibliothèques Steam d'un libraryfolders.vdf (formats récent et ancien)."""
    paths = re.findall(r'"path"\s+"([^"]+)"', text)
    paths += [m for m in re.findall(r'^\s*"\d+"\s+"([^"]+)"\s*$', text, re.M)]
    return [p.replace("\\\\", "\\") for p in paths]


def steam_games() -> list[tuple[Path, str, str]]:
    roots: list[Path] = []
    if os.name == "nt":
        import winreg
        for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
            for name in ("SteamPath", "InstallPath"):
                v = _reg_value(hive, key, name)
                if v:
                    roots.append(Path(v))
        roots.append(Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam")
    else:
        home = Path.home()
        roots += [home / ".local/share/Steam", home / ".steam/steam", home / ".var/app/com.valvesoftware.Steam/.local/share/Steam",
                  home / "Library/Application Support/Steam"]
    libraries: list[Path] = []
    for r in roots:
        vdf = r / "steamapps" / "libraryfolders.vdf"
        if vdf.is_file():
            try:
                libraries += [Path(p) for p in parse_libraryfolders(vdf.read_text(encoding="utf-8", errors="replace"))]
            except OSError:
                pass
        libraries.append(r)
    out, seen = [], set()
    for lib in libraries:
        apps = lib / "steamapps"
        common = apps / "common"
        if _norm(common) in seen or not common.is_dir():
            continue
        seen.add(_norm(common))
        names: dict[str, str] = {}
        for acf in apps.glob("appmanifest_*.acf"):
            try:
                t = acf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            m_dir, m_name = re.search(r'"installdir"\s+"([^"]+)"', t), re.search(r'"name"\s+"([^"]+)"', t)
            if m_dir and m_name:
                names[m_dir.group(1).lower()] = m_name.group(1)
        for e in _subdirs(common):
            out.append((Path(e.path), names.get(e.name.lower(), e.name), N_("Steam")))
    return out


def epic_games() -> list[tuple[Path, str, str]]:
    base = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
    out = []
    for item in (base.glob("*.item") if base.is_dir() else []):
        try:
            d = json.loads(item.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if d.get("InstallLocation"):
            out.append((Path(d["InstallLocation"]), str(d.get("DisplayName") or Path(d["InstallLocation"]).name), N_("Epic")))
    return out


def gog_games() -> list[tuple[Path, str, str]]:
    if os.name != "nt":
        return []
    import winreg
    rows = _registry_values([(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\GOG.com\Games")], ("path", "gameName"))
    return [(Path(r["path"]), r.get("gameName") or Path(r["path"]).name, N_("GOG")) for r in rows if r.get("path")]


def installed_programs() -> list[tuple[Path, str, str]]:
    """Programmes installés (registre « Désinstaller ») : jeux en boîte, installeurs autonomes."""
    if os.name != "nt":
        return []
    import winreg
    path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    rows = _registry_values([(winreg.HKEY_LOCAL_MACHINE, path), (winreg.HKEY_CURRENT_USER, path)],
                            ("InstallLocation", "DisplayName", "Publisher"))
    out = []
    windir = _norm(os.environ.get("WINDIR", r"C:\Windows"))
    for r in rows:
        loc = (r.get("InstallLocation") or "").strip().strip('"')
        if not loc or len(Path(loc).parts) < 2 or _norm(loc).startswith(windir):
            continue
        if (r.get("Publisher") or "").lower().startswith(("microsoft", "nvidia", "intel", "amd", "advanced micro")):
            continue
        out.append((Path(loc), r.get("DisplayName") or Path(loc).name, N_("programme installé")))
    return out


def fixed_drives() -> list[Path]:
    if os.name != "nt":
        return []
    try:
        import ctypes
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        drives = []
        for i in range(26):
            if mask & (1 << i):
                root = f"{chr(65 + i)}:\\"
                if ctypes.windll.kernel32.GetDriveTypeW(root) == 3:          # DRIVE_FIXED : ni réseau ni DVD
                    drives.append(Path(root))
        return drives
    except (AttributeError, OSError):
        return [Path("C:\\")]


def common_libraries() -> list[tuple[Path, str, str]]:
    out = []
    for drive in fixed_drives():
        for rel in COMMON_LIBRARIES:
            lib = drive / rel
            if lib.is_dir():
                out += [(Path(e.path), e.name, N_("dossier de jeux")) for e in _subdirs(lib)]
    return out


def documents_dir() -> Path:
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:    # CSIDL_PERSONAL
                return Path(buf.value)
        except (AttributeError, OSError):
            pass
    return Path.home() / "Documents"


def specialk_profiles() -> list[tuple[Path, str, str]]:
    """Profils Special K (injection globale) : le SK_Res du jeu est dans son profil."""
    bases = [documents_dir() / "My Mods" / "SpecialK" / "Profiles"]
    if os.name == "nt":
        bases += [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Special K" / "Profiles",
                  Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Special K" / "Profiles"]
    out = []
    for b in bases:
        if b.is_dir():
            out += [(Path(e.path), e.name, N_("profil Special K")) for e in _subdirs(b)]
    return out


def extra_roots(paths) -> list[tuple[Path, str, str]]:
    """Dossiers ajoutés : TEXFORGE_SEARCH et dossiers déjà utilisés (jeu, puis bibliothèque parente)."""
    out = []
    env = [p for p in os.environ.get("TEXFORGE_SEARCH", "").split(os.pathsep) if p.strip()]
    for lib in env:
        lp = Path(lib.strip())
        if lp.is_dir():
            out += [(Path(e.path), e.name, "TEXFORGE_SEARCH") for e in _subdirs(lp)]
            out.append((lp, "", "TEXFORGE_SEARCH"))           # nom vide : déduit du dossier trouvé
    for p in paths or []:
        game = game_dir_of(p)
        if game is not None and game.is_dir():
            out.append((game, "", N_("déjà utilisé")))
            out += [(Path(e.path), e.name, N_("déjà utilisé")) for e in _subdirs(game.parent)]
    return out


def game_dir_of(path: str | Path) -> Path | None:
    """Dossier qui contient le SK_Res / rtx-remix d'un chemin d'entrée (None si aucun)."""
    p = Path(path)
    for i, part in enumerate(p.parts):
        if part.lower() in MARKERS:
            return Path(*p.parts[:i]) if i else None
    return None


# ------------------------------------------------------------------ recherche

def detect(history=(), *, time_budget: float = 25.0, sources=None) -> list[Candidate]:
    """Dossiers de textures extraites trouvés sur cette machine, du plus récent au plus ancien.
    `sources` : liste de fonctions renvoyant (dossier, nom du jeu, origine) — pour les tests."""
    t0 = time.monotonic()
    providers = sources if sources is not None else [
        specialk_profiles, steam_games, epic_games, gog_games, common_libraries, installed_programs]
    roots: list[tuple[Path, str, str]] = list(extra_roots(history))         # (dossier, nom du jeu ou "", origine)
    for fn in providers:
        try:
            roots += fn()
        except Exception:                 # une source illisible ne doit pas empêcher les autres
            continue
    found: dict[str, Candidate] = {}
    seen_roots: set[str] = set()
    for root, name, source in roots:
        if time.monotonic() - t0 > time_budget:
            break
        key = _norm(root)
        if key in seen_roots or not root.is_dir():
            continue
        seen_roots.add(key)
        for marker in find_markers(root):
            if _norm(marker) in found:
                continue
            if not name:
                game = game_name(marker)
            elif name == root.name:
                game = game_name(marker, root)
            else:
                game = name
            c = from_marker(marker, game, source)
            if c is not None and c.count > 0:
                found.setdefault(_norm(c.path), c)
    return sorted(found.values(), key=lambda c: (c.newest, c.count), reverse=True)


def resolve_input(path: str | Path) -> Candidate | None:
    """Dossier choisi à la main : si c'est un dossier de jeu (ou SK_Res, dump, rtx-remix) plutôt que le
    dossier des textures, renvoie le dossier de textures qu'il contient. None : rien à changer."""
    p = Path(path)
    if not p.is_dir() or has_images(p):
        return None
    low = p.name.lower()
    root: Path | None = None                    # dossier de jeu choisi : il donne son nom
    if low in MARKERS:
        markers = [p]
    elif low == "dump" and p.parent.name.lower() == "sk_res":
        markers = [p.parent]
    elif low == "captures" and p.parent.name.lower() == "rtx-remix":
        markers = [p.parent]
    else:
        markers, root = find_markers(p, depth=4, budget=1500), p
    cands = [c for c in (from_marker(m, game_name(m, root), N_("dossier choisi")) for m in markers) if c is not None]
    if not cands:
        return None
    return max(cands, key=lambda c: (c.count > 0, c.newest, c.count))


# ------------------------------------------------------------------ dossier d'entrée sans textures

# Archives où les jeux rangent leurs textures : illisibles sans extraction (moteur si l'extension le dit)
PACKED_EXTS = {".assets": "Unity", ".ress": "Unity", ".resource": "Unity", ".unity3d": "Unity", ".bundle": "Unity",
               ".utoc": "Unreal Engine", ".ucas": "Unreal Engine", ".upk": "Unreal Engine 3",
               ".vpk": "Source", ".gcf": "Source", ".bsa": "Bethesda", ".ba2": "Bethesda", ".rpf": "RAGE",
               ".forge": "Anvil", ".pk3": "id Tech", ".pk4": "id Tech", ".pak": "", ".big": "", ".arc": "",
               ".pck": "", ".wad": "", ".cpk": "", ".gpk": "", ".000": ""}     # .000 : bigfile.000, .001…


NUMBERED_ARCHIVE = re.compile(r"^\.\d{3}$")       # bigfile.000, data.001… (archives en plusieurs parties)
FEW_IMAGES = 32         # moins que ça dans un dossier de jeu : icônes, lanceur, captures — pas les textures du jeu


def graphics_api(folder: str | Path) -> str:
    """API graphique du jeu d'après les DLL que citent ses exécutables (« d3d9 », « d3d11 », « d3d12 »,
    « opengl »… ; "" si inconnue). Un exécutable protégé ou compressé ne dit rien."""
    exes: list[tuple[int, Path]] = []
    stack = [(Path(folder), 0)]
    while stack:
        d, lvl = stack.pop()
        try:
            for e in os.scandir(d):
                if e.is_dir(follow_symlinks=False) and lvl < 3 and e.name.lower() not in SKIP_DIRS:
                    stack.append((Path(e.path), lvl + 1))
                elif e.name.lower().endswith(".exe") and not e.name.lower().startswith(("unins", "setup", "vc_redist",
                                                                                      "dxsetup", "crashreport")):
                    exes.append((e.stat().st_size, Path(e.path)))
        except OSError:
            continue
    marks = {b"d3d8.dll": "d3d8", b"d3d9.dll": "d3d9", b"d3d10.dll": "d3d10", b"d3d11.dll": "d3d11",
             b"d3d12.dll": "d3d12", b"opengl32.dll": "opengl", b"vulkan-1.dll": "vulkan"}
    found: set[str] = set()
    for _size, exe in sorted(exes, reverse=True)[:3]:
        try:
            with open(exe, "rb") as f:
                tail = b""
                for _ in range(64):                        # 64 Mo au plus par exécutable
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    low = (tail + chunk).lower()
                    found |= {api for m, api in marks.items() if m in low}
                    tail = chunk[-16:]
        except OSError:
            continue
    if {"d3d11", "d3d12"} <= found:
        return "d3d11/d3d12"
    for api in ("d3d12", "d3d11", "d3d10", "d3d9", "d3d8", "vulkan", "opengl"):
        if api in found:
            return api
    return ""


@dataclass
class InputCheck:
    images: int                  # textures lisibles trouvées (arrêt à la première si demandé)
    packed: dict                 # extension d'archive -> nombre de fichiers
    engine: str                  # moteur reconnu ("" si inconnu)
    exe: bool                    # un exécutable : c'est un dossier d'installation de jeu
    complete: bool               # tout le dossier a été parcouru


def check_input(path: str | Path, max_entries: int = 60_000, stop_at_image: bool = True) -> InputCheck:
    """Le dossier d'entrée contient-il des textures lisibles ? Sinon, qu'y a-t-il à la place
    (archives d'un moteur, exécutable) : de quoi expliquer pourquoi le traitement serait vide."""
    images, seen, exe, unity = 0, 0, False, False
    packed: dict[str, int] = {}
    stack = [Path(path)]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    seen += 1
                    if seen > max_entries:
                        return InputCheck(images, packed, _engine(packed, unity), exe, False)
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name.lower() not in SKIP_DIRS:
                                stack.append(Path(e.path))
                            unity = unity or e.name.endswith("_Data")
                            continue
                    except OSError:
                        continue
                    ext = os.path.splitext(e.name)[1].lower()
                    n_ba2 = _ba2_textures(e.path) if ext == ".ba2" else 0
                    if ext in IMAGE_EXTS or n_ba2:
                        images += max(1, n_ba2)
                        if stop_at_image:
                            return InputCheck(images, packed, _engine(packed, unity), exe, False)
                    elif ext in PACKED_EXTS or NUMBERED_ARCHIVE.match(ext):
                        key = ext if ext in PACKED_EXTS else ".000"
                        packed[key] = packed.get(key, 0) + 1
                    elif ext == ".exe":
                        exe = True
        except OSError:
            continue
    return InputCheck(images, packed, _engine(packed, unity), exe, True)


def _ba2_textures(path: str) -> int:
    from .ba2 import texture_count
    return texture_count(path)


def _engine(packed: dict, unity: bool) -> str:
    if unity and any(PACKED_EXTS[x] == "Unity" for x in packed):
        return "Unity"
    named = sorted(((n, PACKED_EXTS[x]) for x, n in packed.items() if PACKED_EXTS[x] and PACKED_EXTS[x] != "Unity"),
                   reverse=True)
    return named[0][1] if named else ""


def empty_input_help(path: str | Path, chk: InputCheck) -> str:
    """Explication, dans la langue active, d'un dossier d'entrée sans aucune texture lisible."""
    from .unreal import unreal_help, unreal_info
    ui = unreal_info(path)
    if ui is not None:
        return unreal_help(ui)[1]
    lines = [] if chk.images else [tr("Aucune texture lisible dans « {path} » : TEXFORGE lit des fichiers .dds, "
                                      ".png, .tga, .bmp ou .jpg séparés.", path=path)]
    if chk.packed:
        top = sorted(chk.packed.items(), key=lambda kv: -kv[1])[:3]
        archives = ", ".join(f"{n} {({'.ress': '.resS', '.000': '.000/.001…'}).get(ext, ext)}" for ext, n in top)
        if chk.engine:
            lines.append(tr("Ce jeu ({engine}) range ses textures dans des archives ({archives}).",
                            engine=chk.engine, archives=archives))
        else:
            lines.append(tr("Ce jeu range ses textures dans des archives ({archives}).", archives=archives))
    elif chk.exe:
        lines.append(tr("C'est le dossier d'installation du jeu : ses textures n'y sont pas en fichiers séparés."))
    lines.append(tr("Il faut d'abord les extraire pendant que le jeu tourne : lance-le avec Special K "
                    "(Ctrl+Maj+Retour arrière), active le dump des textures, joue quelques minutes, puis choisis "
                    "le dossier SK_Res\\dump\\textures (bouton « 🔍 Détecter »)."))
    api = graphics_api(path) if chk.exe else ""
    if api in ("d3d8", "d3d9"):
        lines.append(tr("Ce jeu est en Direct3D {v} : Special K y gère mal les textures. Fais-le d'abord passer en "
                        "Direct3D 11 avec dgVoodoo2 (section 2 du README), puis fais le dump.", v=api[3:]))
    elif api in ("d3d12", "vulkan", "opengl"):
        lines.append(tr("Ce jeu semble utiliser {api} : Special K n'y remplace pas les textures (Direct3D 11 "
                        "seulement, Direct3D 9 en partie).", api={"d3d12": "Direct3D 12", "vulkan": "Vulkan",
                                                                    "opengl": "OpenGL"}[api]))
    return "\n".join(lines)


def inject_for(input_path: str | Path) -> str:
    """Special K : dossier d'injection qui correspond à un dossier de dump (sinon "")."""
    p = Path(input_path)
    parts = [x.lower() for x in p.parts]
    for i in range(len(parts) - 2):
        if parts[i] == "sk_res" and parts[i + 1] == "dump" and parts[i + 2] == "textures":
            sk = Path(*p.parts[:i + 1])
            inject_root = _child(sk, "inject")
            return str((_child(inject_root, "textures") if inject_root else None) or sk / "inject" / "textures")
    return ""


# ------------------------------------------------------------------ dossier choisi -> dossier des textures

@dataclass
class Located:
    path: str                 # dossier d'entrée à utiliser
    kind: str                 # specialk | remix | ba2 (archives Bethesda) | loose (fichiers séparés) | none
    count: int                # textures lisibles sous `path` (sous-dossiers compris)
    chosen: str               # dossier choisi par l'utilisateur
    game: str = ""
    inject: str = ""          # Special K : dossier d'injection associé
    source: str = ""          # d'où vient le dossier (texte français, tr() à l'affichage)
    note: str = ""            # explication quand rien d'utilisable n'est trouvé (langue active)
    summary: str = ""         # la même, en quelques mots (ligne d'état sous le champ)

    @property
    def changed(self) -> bool:
        return _norm(self.path) != _norm(self.chosen)


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _game_keys(d: Path) -> set[str]:
    """Noms sous lesquels Special K peut avoir rangé le profil d'un jeu : dossier, exécutables."""
    keys = {_key(d.name)}
    for sub in (d, *[Path(e.path) for e in _subdirs(d)][:40]):
        try:
            with os.scandir(sub) as it:
                keys |= {_key(e.name[:-4]) for e in it if e.name.lower().endswith(".exe")}
        except OSError:
            continue
    return {k for k in keys if len(k) >= 4 and k not in {"unins000", "uninstall", "setup", "launcher", "crashreporter"}}


def _profile_for(d: Path, profiles) -> Candidate | None:
    """Profil Special K global (injection par SKIF) dont le nom correspond au jeu choisi."""
    keys = _game_keys(d)
    best = None
    for prof, name, source in profiles:
        pk = _key(name)
        if not pk or not any(pk == k or (min(len(pk), len(k)) >= 6 and (pk in k or k in pk)) for k in keys):
            continue
        marker = _child(prof, "SK_Res")
        c = from_marker(marker, d.name, source) if marker else None
        if c is not None and (best is None or (c.count, c.newest) > (best.count, best.newest)):
            best = c
    return best


def _texture_tree(root: Path, max_entries: int, time_budget: float) -> tuple[dict, dict, InputCheck]:
    """Textures directement dans chaque dossier, et totaux par sous-arbre (dumps et sorties exclus)."""
    t0 = time.monotonic()
    direct: dict[Path, int] = {}
    children: dict[Path, list[Path]] = {}
    packed: dict[str, int] = {}
    seen, exe, unity, complete = 0, False, False, True
    stack = [root]
    while stack:
        cur = stack.pop()
        n, kids = 0, []
        try:
            with os.scandir(cur) as it:
                for e in it:
                    seen += 1
                    try:
                        if e.is_dir(follow_symlinks=False):
                            low = e.name.lower()
                            unity = unity or e.name.endswith("_Data")
                            # SK_Res et rtx-remix sont traités à part ; leurs sorties ne sont jamais une entrée
                            if low not in SKIP_DIRS and low not in MARKERS and not low.startswith("."):
                                kids.append(Path(e.path))
                            continue
                    except OSError:
                        continue
                    ext = os.path.splitext(e.name)[1].lower()
                    if ext in IMAGE_EXTS:
                        n += 1
                    elif ext in PACKED_EXTS or NUMBERED_ARCHIVE.match(ext):
                        key = ext if ext in PACKED_EXTS else ".000"
                        packed[key] = packed.get(key, 0) + 1
                    elif ext == ".exe":
                        exe = True
        except OSError:
            pass
        direct[cur], children[cur] = n, kids
        if seen > max_entries or time.monotonic() - t0 > time_budget:
            complete = False
            break
        stack.extend(kids)
    total: dict[Path, int] = {}

    def subtotal(d: Path) -> int:          # itératif : pas de limite de récursion sur les arbres profonds
        order, todo = [], [d]
        while todo:
            x = todo.pop()
            order.append(x)
            todo.extend(children.get(x, []))
        for x in reversed(order):
            total[x] = direct.get(x, 0) + sum(total.get(c, 0) for c in children.get(x, []))
        return total[d]

    count = subtotal(root)
    return direct, total, InputCheck(count, packed, _engine(packed, unity), exe, complete)


def locate_textures(path: str | Path, *, profiles=None, max_entries: int = 150_000,
                    time_budget: float = 8.0) -> Located:
    """Dossier des textures correspondant au dossier choisi : dump Special K du jeu (à côté de
    l'exécutable ou dans son profil global), capture RTX Remix, ou sous-dossier qui regroupe les
    textures du jeu. `profiles` : profils Special K (pour les tests ; sinon ceux de la machine)."""
    p = Path(path)
    chosen = str(p)
    if not p.is_dir():
        return Located(chosen, "none", 0, chosen, note=tr("Dossier introuvable : {path}", path=chosen),
                       summary=tr("dossier introuvable"))
    parts = [x.lower() for x in p.parts]
    # 1. dossier d'injection (sortie de TEXFORGE) choisi par erreur : on revient au dump du même SK_Res
    for i in range(len(parts) - 1):
        if parts[i] == "sk_res" and parts[i + 1] == "inject":
            c = from_marker(Path(*p.parts[:i + 1]), game_name(Path(*p.parts[:i + 1])), N_("dump du même jeu"))
            if c is not None and c.count > 0:
                return Located(c.path, c.kind, c.count, chosen, c.game, c.inject, c.source)
            return Located(chosen, "none", 0, chosen, p.name, chosen, "",
                           tr("C'est un dossier d'injection Special K : la sortie de TEXFORGE, pas les textures du "
                              "jeu. Choisis le dossier du jeu ou son SK_Res\\dump\\textures."),
                           tr("dossier d'injection (sortie de TEXFORGE), pas les textures du jeu"))
    # 2. dans un dump Special K ou une capture RTX Remix (ou l'un de leurs sous-dossiers) : on garde
    inside = any((parts[i], parts[i + 1]) in (("sk_res", "dump"), ("rtx-remix", "captures")) for i in range(len(parts) - 1))
    if inside:
        count, _newest = texture_stats(p)
        inj = inject_for(p)
        if count > 0:
            return Located(chosen, "specialk" if "sk_res" in parts else "remix", count, chosen, p.name, inj,
                           N_("dossier choisi"))
        return Located(chosen, "none", 0, chosen, p.name, inj, "",
                       tr("Le dump Special K de ce jeu existe mais est vide ({path}) : active le dump des textures, "
                          "joue quelques minutes, puis choisis à nouveau le dossier.", path=chosen),
                       tr("dump Special K encore vide : active le dump des textures et joue"))
    # 3. dossier de jeu, SK_Res, rtx-remix : dump ou capture qu'il contient
    markers = [p] if p.name.lower() in MARKERS else ([] if inside else find_markers(p, depth=4, budget=1500))
    cands = [c for c in (from_marker(m, game_name(m, None if p.name.lower() in MARKERS else p), N_("dossier choisi"))
                         for m in markers) if c is not None]
    found = max(cands, key=lambda c: (c.count > 0, c.newest, c.count), default=None)
    if found is not None and found.count > 0:
        return Located(found.path, found.kind, found.count, chosen, found.game, found.inject, found.source)
    # 4. jeu injecté par SKIF : son SK_Res est dans le profil Special K global
    prof = _profile_for(p, specialk_profiles() if profiles is None else profiles)
    if prof is not None and prof.count > 0:
        return Located(prof.path, "specialk", prof.count, chosen, prof.game, prof.inject, prof.source)
    # 5. Fallout 4, Starfield : archives BA2 de textures dans le dossier ou dans son Data
    from .ba2 import data_folder, texture_archives, texture_count
    bdir = data_folder(p)
    if bdir is not None:
        arcs = texture_archives(bdir)
        game = (bdir.parent if bdir.name.lower() == "data" else bdir).name
        return Located(str(bdir), "ba2", sum(texture_count(a) for a in arcs), chosen, game, "",
                       N_("archives BA2"))
    # 6. textures du jeu en fichiers séparés : le sous-dossier qui les regroupe
    _direct, total, chk = _texture_tree(p, max_entries, time_budget)
    few = chk.exe and chk.complete and 0 < total.get(p, 0) < FEW_IMAGES
    if total.get(p, 0) > 0 and not few:
        cur = p
        while chk.complete:              # parcours incomplet (disque entier…) : on ne devine pas
            kids = [k for k in total if k.parent == cur and total[k] > 0]
            best = max(kids, key=lambda k: total[k], default=None)
            # on descend tant que le reste (lanceur, captures, icônes) pèse moins de 2 % : un dossier
            # d'interface ou de textures voisin, lui, n'est jamais écarté
            if best is None or total[cur] - total[best] > 0.02 * total[cur]:
                break
            cur = best
        return Located(str(cur), "loose", total[cur], chosen, p.name, "", N_("textures du jeu"))
    # 7. rien de lisible : pourquoi, et quoi faire
    empty_dump = found or prof
    note = empty_input_help(p, chk)
    if chk.packed:
        summary = (tr("aucune texture lisible : le jeu ({engine}) les range dans des archives — dump Special K "
                      "nécessaire", engine=chk.engine) if chk.engine else
                   tr("aucune texture lisible : le jeu les range dans des archives — dump Special K nécessaire"))
    else:
        summary = tr("aucune texture lisible dans ce dossier")
    if few:
        examples = []
        for f in p.rglob("*"):
            if f.suffix.lower() in IMAGE_EXTS and f.is_file():
                examples.append(f.name)
                if len(examples) >= 3:
                    break
        note = tr("Ce dossier de jeu ne contient que {n} image(s) séparée(s) ({examples}) : des icônes ou des "
                  "fichiers annexes, pas les textures du jeu.", n=total[p], examples=", ".join(examples)) + "\n" + note
        summary = tr("{n} image(s) annexe(s) seulement : les textures du jeu sont ailleurs — dump Special K nécessaire",
                     n=total[p])
    from .unreal import unreal_help, unreal_info
    ui = unreal_info(p)
    if ui is not None:
        summary, note = unreal_help(ui)
    if empty_dump is not None:
        note = tr("Le dump Special K de ce jeu existe mais est vide ({path}) : active le dump des textures, joue "
                  "quelques minutes, puis choisis à nouveau le dossier.", path=empty_dump.path)
        summary = tr("dump Special K encore vide : active le dump des textures et joue")
    return Located(chosen, "none", 0, chosen, p.name, empty_dump.inject if empty_dump else "", "", note, summary)


__all__ = ["Candidate", "detect", "resolve_input", "inject_for", "texture_stats", "find_markers",
           "parse_libraryfolders", "game_dir_of", "has_images", "InputCheck", "check_input", "empty_input_help",
           "Located", "locate_textures", "graphics_api", "FEW_IMAGES"]
