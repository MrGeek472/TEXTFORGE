"""Jeux Unreal Engine : reconnaître le moteur et dire ce qui est possible pour leurs textures.

TEXFORGE ne lit pas les archives Unreal. Ce module lit seulement leurs en-têtes, sans rien
déchiffrer ni décompresser, pour expliquer la situation :
  * Unreal Engine 4 et 5 : archives .pak (pied de fichier : version, index chiffré, méthodes de
    compression) et conteneurs IoStore .utoc/.ucas (en-tête : drapeaux compressé, chiffré, signé) ;
  * Unreal Engine 3 : paquets .upk/.xxx et caches .tfc dans CookedPC*.
Les archives des jeux récents sont presque toujours chiffrées (AES) : les ouvrir supposerait de
contourner la protection du jeu, ce que TEXFORGE ne fait pas. La voie qui reste est la capture en jeu
avec Special K, qui ne sait remplacer des textures qu'en Direct3D 11 (et en partie en Direct3D 9).
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import tr

PAK_MAGIC = 0x5A6F12E1
UTOC_MAGIC = b"-==--==--==--==-"
UE3_DIRS = ("cookedpc", "cookedpcconsole", "cookedpcserver", "cookedpcconsole_fr", "cookedwin32", "cookedwin64")


@dataclass
class UnrealInfo:
    root: Path                          # dossier du jeu choisi
    project: str                        # dossier du projet (celui qui contient Content/Paks), ou « UE3 »
    generation: str                     # UE3 | UE4 | UE5 | UE4/UE5
    exe_dir: Path | None                # dossier de l'exécutable : là que va Special K (injection locale)
    paks: int = 0
    containers: int = 0                 # conteneurs IoStore (.utoc)
    encrypted: bool = False
    signed: bool = False
    compression: set[str] = field(default_factory=set)
    pak_version: int = 0
    d3d12: bool = False                 # le jeu embarque Direct3D 12 (dossier D3D12, Agility SDK)


def pak_footer(path: str | Path) -> dict | None:
    """Pied d'une archive .pak (UE4/UE5) : version, index chiffré, méthodes de compression."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 512))
            tail = f.read()
    except OSError:
        return None
    i = tail.rfind(struct.pack("<I", PAK_MAGIC))
    if i < 1 or i + 24 > len(tail):
        return None
    version = struct.unpack_from("<I", tail, i + 4)[0]
    if not 1 <= version <= 20:
        return None
    names = []
    rest = tail[i + 4 + 4 + 8 + 8 + 20:]
    for k in range(0, len(rest) - 31, 32):
        n = rest[k:k + 32].split(b"\0")[0]
        if n and all(32 <= c < 127 for c in n):
            names.append(n.decode("ascii"))
    return {"version": version, "encrypted_index": version >= 4 and tail[i - 1] == 1, "compression": names}


def utoc_header(path: str | Path) -> dict | None:
    """En-tête d'un conteneur IoStore .utoc : version et drapeaux (compressé, chiffré, signé)."""
    try:
        with open(path, "rb") as f:
            h = f.read(144)
            if len(h) < 96 or h[:16] != UTOC_MAGIC:
                return None
            hsize, entries, blocks, block_entry, n_meth, meth_len = struct.unpack_from("<6I", h, 20)
            flags = h[80]
            methods = []
            if 0 < n_meth <= 8 and 0 < meth_len <= 64 and block_entry == 12:
                f.seek(hsize + entries * 12 + entries * 10 + blocks * block_entry)
                raw = f.read(n_meth * meth_len)
                methods = [raw[k * meth_len:(k + 1) * meth_len].split(b"\0")[0].decode("ascii", "replace")
                           for k in range(n_meth)]
    except (OSError, struct.error):
        return None
    return {"version": h[16], "compressed": bool(flags & 1), "encrypted": bool(flags & 2),
            "signed": bool(flags & 4), "compression": [m for m in methods if re.fullmatch(r"[A-Za-z0-9_]{2,}", m)]}


def _child(d: Path, name: str) -> Path | None:
    try:
        for e in os.scandir(d):
            if e.is_dir() and e.name.lower() == name.lower():
                return Path(e.path)
    except OSError:
        pass
    return None


def _subdirs(d: Path) -> list[Path]:
    try:
        return [Path(e.path) for e in os.scandir(d) if e.is_dir(follow_symlinks=False)]
    except OSError:
        return []


def _paks_dir(project: Path) -> Path | None:
    content = _child(project, "Content")
    return _child(content, "Paks") if content else None


def unreal_info(path: str | Path) -> UnrealInfo | None:
    """Jeu Unreal reconnu dans `path` (dossier du jeu, de son projet, ou l'un de leurs sous-dossiers)."""
    p = Path(path)
    if not p.is_dir():
        return None
    # dossier choisi dans le projet (Content, Paks, Binaries…) : on remonte jusqu'au projet
    cands = [p, *list(p.parents)[:3]]
    projects = [c for c in cands if _paks_dir(c) is not None]
    if not projects:
        projects = [d for d in _subdirs(p) if _paks_dir(d) is not None]
        projects += [d2 for d in _subdirs(p) for d2 in _subdirs(d) if _paks_dir(d2) is not None][:4]
    if projects:
        proj = projects[0]
        paks_dir = _paks_dir(proj)
        assert paks_dir is not None
        info = UnrealInfo(p, proj.name, "UE4/UE5", None)
        for f in sorted(paks_dir.rglob("*")):
            low = f.name.lower()
            if low.endswith(".pak"):
                info.paks += 1
                foot = pak_footer(f)
                if foot:
                    info.pak_version = max(info.pak_version, foot["version"])
                    info.encrypted |= foot["encrypted_index"]
                    info.compression |= {c for c in foot["compression"] if c}
                if f.with_suffix(".sig").exists():
                    info.signed = True
            elif low.endswith(".utoc") and low != "global.utoc":
                info.containers += 1
                head = utoc_header(f)
                if head:
                    info.encrypted |= head["encrypted"]
                    info.signed |= head["signed"]
                    info.compression |= set(head["compression"])
        if not info.paks and not info.containers:
            return None
        # IoStore : défaut d'Unreal 5 (existe aussi en 4.26/4.27) ; pak v11 : 4.26 et plus
        info.generation = "UE5" if info.containers else ("UE4" if info.pak_version and info.pak_version < 11
                                                         else "UE4/UE5")
        win64 = _child(_child(proj, "Binaries") or proj, "Win64")
        info.exe_dir = win64
        info.d3d12 = bool(win64 and _child(win64, "D3D12"))
        return info
    # Unreal Engine 3 : CookedPC* avec des .upk / .xxx / .tfc
    level1 = _subdirs(p)
    for d in [p, *level1, *[d2 for d1 in level1 for d2 in _subdirs(d1)]][:200]:
        if d.name.lower() in UE3_DIRS:
            try:
                has = any(e.name.lower().endswith((".upk", ".xxx", ".tfc", ".u")) for e in os.scandir(d))
            except OSError:
                has = False
            if has:
                game = d.parent
                bins = _child(game, "Binaries") or _child(game.parent, "Binaries")
                exe_dir = (_child(bins, "Win64") or _child(bins, "Win32")) if bins else None
                return UnrealInfo(p, "UE3", "UE3", exe_dir, paks=sum(1 for _ in d.glob("*.upk")))
    return None


def unreal_help(info: UnrealInfo) -> tuple[str, str]:
    """(résumé en quelques mots, explication détaillée), dans la langue active."""
    where = str(info.exe_dir) if info.exe_dir else tr("le dossier de l'exécutable du jeu")
    if info.generation == "UE3":
        summary = tr("Unreal Engine 3 : capture Special K nécessaire (dgVoodoo2 si le jeu est en Direct3D 9)")
        note = "\n".join([
            tr("Jeu Unreal Engine 3 : ses textures sont rangées dans des paquets .upk et des caches .tfc, que "
               "TEXFORGE ne lit pas."),
            tr("Voie possible : capture en jeu avec Special K. Si le jeu est en Direct3D 9, passe-le d'abord en "
               "Direct3D 11 avec dgVoodoo2 (section 2 du README), puis active le dump des textures."),
            tr("Une fois le dump fait, choisis à nouveau ce dossier : TEXFORGE trouvera SK_Res\\dump\\textures "
               "dans {where}.", where=where)])
        return summary, note
    container = ".utoc/.ucas (IoStore)" if info.containers else ".pak"
    traits = []
    if info.encrypted:
        traits.append(tr("chiffrées (AES)"))
    if info.compression:
        traits.append(tr("compressées ({methods})", methods=", ".join(sorted(info.compression))))
    if info.signed:
        traits.append(tr("signées"))
    generation = {"UE4": "4", "UE5": "5"}.get(info.generation, tr("4 ou 5"))
    lines = [tr("Jeu Unreal Engine {generation} : ses textures sont dans des archives {container}{traits}.",
                generation=generation, container=container, traits=" " + ", ".join(traits) if traits else "")]
    if info.encrypted:
        lines.append(tr("TEXFORGE n'ouvre pas les archives chiffrées : il faudrait contourner la protection du jeu."))
        summary = tr("Unreal Engine, archives chiffrées : capture Special K en Direct3D 11 nécessaire")
    else:
        lines.append(tr("TEXFORGE ne lit pas encore les textures des fichiers .uasset d'Unreal."))
        summary = tr("Unreal Engine : capture Special K en Direct3D 11 nécessaire")
    lines += [
        tr("Voie possible : capture en jeu avec Special K, qui ne remplace les textures qu'en Direct3D 11 :"),
        tr("1. Installe Special K pour ce jeu (SKIF, ou dxgi.dll dans {where}).", where=where),
        tr("2. Force Direct3D 11 : Steam → clic droit sur le jeu → Propriétés → Options de lancement : -dx11"),
        tr("3. Active le dump des textures, joue, puis choisis à nouveau ce dossier : TEXFORGE trouvera "
           "SK_Res\\dump\\textures dans {where}.", where=where),
    ]
    if info.d3d12:
        lines.append(tr("Ce jeu embarque Direct3D 12 : s'il refuse -dx11 ou ne démarre pas ainsi, ni la capture ni le "
                        "remplacement ne sont possibles."))
    lines.append(tr("Limite : Unreal charge ses textures par morceaux pendant le jeu (streaming) ; la capture "
                    "peut être partielle, à vérifier jeu par jeu."))
    return summary, "\n".join(lines)


__all__ = ["UnrealInfo", "unreal_info", "unreal_help", "pak_footer", "utoc_header"]
