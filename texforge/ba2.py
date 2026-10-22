"""Archives BA2 de Bethesda (Fallout 4, Fallout 76, Starfield) : lecture des textures.

Format (petit-boutiste) :
  en-tête   "BTDX", version u32 (1 : Fallout 4 et 76 ; 7 et 8 : Fallout 4 « next-gen » ; 2 : Starfield,
            fichiers ; 3 : Starfield, textures), type sur 4 caractères ("GNRL" fichiers, "DX10" textures),
            nombre de fichiers u32, position de la table des noms u64. Versions 2 et 3 : 8 octets de plus ;
            version 3 : méthode de compression u32 (0 zlib, 3 LZ4 en blocs).
  DX10      par texture : hachages du nom, de l'extension et du dossier (3 × u32), u8, nombre de morceaux
            u8, taille d'un en-tête de morceau u16 (24), hauteur u16, largeur u16, mips u8, format DXGI u8,
            drapeaux u16 (2049 : cubemap) ; puis par morceau : position u64, taille compressée u32
            (0 : stocké tel quel), taille u32, premier et dernier mip u16, 0xBAADF00D.
  GNRL      par fichier : 3 × u32, drapeaux u32, position u64, taille compressée u32, taille u32, 0xBAADF00D.
  noms      pour chaque fichier, dans l'ordre : longueur u16 puis le chemin (textures\\...\\x_d.dds).

Les morceaux d'une texture sont ses mips, du plus grand au plus petit, bout à bout : le fichier DDS se
reconstruit en plaçant un en-tête devant. Les cubemaps sont listées mais pas extraites (le pipeline ne
traite que les textures 2D).
"""

from __future__ import annotations

import fnmatch
import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .i18n import tr

MAGIC = b"BTDX"
VERSIONS = {1: "Fallout 4", 2: "Starfield", 3: "Starfield", 7: "Fallout 4 (next-gen)", 8: "Fallout 4 (next-gen)"}
BAADFOOD = 0xBAADF00D
CUBEMAP_FLAGS = 2049
MANIFEST = ".texforge_ba2.json"

# DDS
DDSD = 0x1 | 0x2 | 0x4 | 0x1000          # CAPS | HEIGHT | WIDTH | PIXELFORMAT
DDSD_MIPMAPCOUNT, DDSD_PITCH, DDSD_LINEARSIZE = 0x20000, 0x8, 0x80000
DDPF_ALPHAPIXELS, DDPF_ALPHA, DDPF_FOURCC, DDPF_RGB, DDPF_LUMINANCE = 0x1, 0x2, 0x4, 0x40, 0x20000
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
DDSCAPS2_CUBEMAP_ALL = 0x200 | 0xFC00

# Formats DXGI : en-tête « classique » quand il existe (lu par tous les outils), sinon en-tête DX10
_LEGACY_FOURCC = {71: b"DXT1", 74: b"DXT3", 77: b"DXT5", 80: b"ATI1", 83: b"ATI2"}
_LEGACY_MASKS = {   # dxgi -> (drapeaux, bits, R, G, B, A)
    28: (DDPF_RGB | DDPF_ALPHAPIXELS, 32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000),
    87: (DDPF_RGB | DDPF_ALPHAPIXELS, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
    88: (DDPF_RGB, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0),
    85: (DDPF_RGB, 16, 0xF800, 0x07E0, 0x001F, 0),
    86: (DDPF_RGB | DDPF_ALPHAPIXELS, 16, 0x7C00, 0x03E0, 0x001F, 0x8000),
    61: (DDPF_LUMINANCE, 8, 0xFF, 0, 0, 0),
    65: (DDPF_ALPHA, 8, 0, 0, 0, 0xFF),
}
# octets par bloc 4x4 (compressés) ou par pixel (non compressés)
_BLOCK = {70: 8, 71: 8, 72: 8, 79: 8, 80: 8, 81: 8, 73: 16, 74: 16, 75: 16, 76: 16, 77: 16, 78: 16, 82: 16, 83: 16,
          84: 16, 94: 16, 95: 16, 96: 16, 97: 16, 98: 16, 99: 16}
_PIXEL = {28: 4, 29: 4, 87: 4, 88: 4, 91: 4, 93: 4, 85: 2, 86: 2, 115: 2, 61: 1, 65: 1, 49: 2, 10: 8, 2: 16}
# format DXGI d'un DDS lu par TEXFORGE (pour écrire une archive de test)
_DXGI_OF = {"DXT1": 71, "DXT3": 74, "DXT5": 77, "BC1": 71, "BC1_SRGB": 72, "BC2": 74, "BC2_SRGB": 75, "BC3": 77,
            "BC3_SRGB": 78, "BC4": 80, "BC5": 83, "BC7": 98, "BC7_SRGB": 99, "BGRA8": 87, "BGRX8": 88,
            "RGBA8": 28, "DX10_RGBA8": 28, "RGBA8_SRGB": 29, "DX10_BGRA8": 87, "BGRA8_SRGB": 91, "DX10_BGRX8": 88}


class Ba2Error(Exception):
    pass


@dataclass
class Chunk:
    offset: int
    packed: int          # 0 : stocké sans compression
    size: int
    first_mip: int
    last_mip: int


@dataclass
class Ba2Texture:
    name: str            # chemin interne, séparateur « / », casse d'origine
    width: int
    height: int
    mips: int
    dxgi: int
    cubemap: bool
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(c.size for c in self.chunks)


def header_info(path: str | Path) -> tuple[int, str, int]:
    """(version, type, nombre de fichiers) d'une archive, sans lire le reste. Lève Ba2Error."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError as exc:
        raise Ba2Error(str(exc)) from exc
    if len(head) < 24 or head[:4] != MAGIC:
        raise Ba2Error(tr("pas une archive BA2 : {path}", path=Path(path).name))
    version, kind, count = struct.unpack_from("<I4sI", head, 4)
    return version, kind.decode("ascii", "replace"), count


def texture_count(path: str | Path) -> int:
    """Nombre de textures d'une archive BA2 de textures (0 pour une archive de fichiers ou illisible)."""
    try:
        _v, kind, count = header_info(path)
    except Ba2Error:
        return 0
    return count if kind == "DX10" else 0


class Ba2Archive:
    """Archive ouverte en lecture. `textures` : textures 2D et cubemaps d'une archive DX10."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.f = open(self.path, "rb")
        try:
            self._read_index()
        except (struct.error, UnicodeDecodeError) as exc:
            self.f.close()
            raise Ba2Error(tr("archive BA2 illisible ({name}) : {exc}", name=self.path.name, exc=exc)) from exc
        except Exception:
            self.f.close()
            raise

    def __enter__(self) -> "Ba2Archive":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self.f.close()

    # ------------------------------------------------------------ index
    def _read_index(self) -> None:
        f = self.f
        head = f.read(24)
        if len(head) < 24 or head[:4] != MAGIC:
            raise Ba2Error(tr("pas une archive BA2 : {path}", path=self.path.name))
        self.version, kind, count, names_at = struct.unpack_from("<I4sIQ", head, 4)
        self.kind = kind.decode("ascii", "replace")
        if self.version not in VERSIONS:
            raise Ba2Error(tr("version BA2 {v} inconnue ({name})", v=self.version, name=self.path.name))
        self.compression = "zlib"
        if self.version in (2, 3):
            f.read(8)
        if self.version == 3:
            method = struct.unpack("<I", f.read(4))[0]
            self.compression = {0: "zlib", 3: "lz4"}.get(method, f"#{method}")
        self.count = count
        self.textures: list[Ba2Texture] = []
        self.files: list[tuple[str, int, int, int]] = []          # GNRL : (nom, position, compressé, taille)
        recs = []
        if self.kind == "DX10":
            for _ in range(count):
                b = f.read(24)
                (_nh, _ext, _dh, _u, nchunks, chdr, h, w, mips, dxgi, flags) = struct.unpack("<I4sIBBHHHBBH", b)
                if chdr != 24:
                    raise Ba2Error(tr("archive BA2 corrompue ({name})", name=self.path.name))
                chunks = []
                for _c in range(nchunks):
                    off, packed, size, m0, m1, food = struct.unpack("<QIIHHI", f.read(24))
                    if food != BAADFOOD:
                        raise Ba2Error(tr("archive BA2 corrompue ({name})", name=self.path.name))
                    chunks.append(Chunk(off, packed, size, m0, m1))
                recs.append((w, h, mips, dxgi, flags == CUBEMAP_FLAGS, chunks))
        elif self.kind == "GNRL":
            for _ in range(count):
                (_nh, _ext, _dh, _fl, off, packed, size, food) = struct.unpack("<I4sIIQIII", f.read(36))
                if food != BAADFOOD:
                    raise Ba2Error(tr("archive BA2 corrompue ({name})", name=self.path.name))
                recs.append((off, packed, size))
        else:
            raise Ba2Error(tr("type d'archive BA2 « {kind} » non géré ({name})", kind=self.kind, name=self.path.name))
        names = self._read_names(names_at, count)
        if self.kind == "DX10":
            for name, (w, h, mips, dxgi, cube, chunks) in zip(names, recs, strict=True):
                self.textures.append(Ba2Texture(name, w, h, mips, dxgi, cube, chunks))
        else:
            self.files = [(n, *r) for n, r in zip(names, recs, strict=True)]

    def _read_names(self, at: int, count: int) -> list[str]:
        if not at:
            return [f"sans_nom_{i:05d}.dds" for i in range(count)]
        self.f.seek(at)
        names = []
        for _ in range(count):
            n = struct.unpack("<H", self.f.read(2))[0]
            raw = self.f.read(n)
            try:
                s = raw.decode("utf-8")
            except UnicodeDecodeError:
                s = raw.decode("cp1252", "replace")
            names.append(s.replace("\\", "/"))
        return names

    # ------------------------------------------------------------ données
    def _unpack(self, c: Chunk) -> bytes:
        self.f.seek(c.offset)
        if not c.packed:
            data = self.f.read(c.size)
        else:
            raw = self.f.read(c.packed)
            if self.compression == "lz4":
                try:
                    import lz4.block
                except ImportError as exc:
                    raise Ba2Error(tr("archive compressée en LZ4 (Starfield) : installe le module lz4 "
                                      "(pip install lz4)")) from exc
                data = lz4.block.decompress(raw, uncompressed_size=c.size)
            elif self.compression == "zlib":
                data = zlib.decompress(raw)
            else:
                raise Ba2Error(tr("compression BA2 {method} non gérée ({name})", method=self.compression,
                                  name=self.path.name))
        if len(data) != c.size:
            raise Ba2Error(tr("morceau tronqué dans {name}", name=self.path.name))
        return data

    def read_dds(self, tex: Ba2Texture) -> bytes:
        """Le fichier DDS complet d'une texture (en-tête reconstruit + mips)."""
        return dds_header(tex.width, tex.height, tex.mips, tex.dxgi, tex.cubemap) + b"".join(
            self._unpack(c) for c in tex.chunks)

    def read_file(self, name: str) -> bytes:
        for n, off, packed, size in self.files:
            if n.lower() == name.lower():
                return self._unpack(Chunk(off, packed, size, 0, 0))
        raise KeyError(name)


def _level0_bytes(w: int, h: int, dxgi: int) -> tuple[int, bool]:
    if dxgi in _BLOCK:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * _BLOCK[dxgi], True
    return w * h * _PIXEL.get(dxgi, 4), False


def dds_header(w: int, h: int, mips: int, dxgi: int, cubemap: bool = False) -> bytes:
    """En-tête DDS d'une texture d'archive : FourCC ou masques classiques quand le format en a,
    extension DX10 sinon (sRGB, BC7…)."""
    size0, compressed = _level0_bytes(w, h, dxgi)
    flags = DDSD | (DDSD_LINEARSIZE if compressed else DDSD_PITCH)
    pitch = size0 if compressed else w * _PIXEL.get(dxgi, 4)
    caps, caps2 = DDSCAPS_TEXTURE, 0
    if mips > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_MIPMAP | DDSCAPS_COMPLEX
    if cubemap:
        caps |= DDSCAPS_COMPLEX
        caps2 = DDSCAPS2_CUBEMAP_ALL
    dx10 = b""
    if dxgi in _LEGACY_FOURCC:
        pf = struct.pack("<II4s5I", 32, DDPF_FOURCC, _LEGACY_FOURCC[dxgi], 0, 0, 0, 0, 0)
    elif dxgi in _LEGACY_MASKS:
        pflags, bits, r, g, b, a = _LEGACY_MASKS[dxgi]
        pf = struct.pack("<II4s5I", 32, pflags, b"\0\0\0\0", bits, r, g, b, a)
    else:
        pf = struct.pack("<II4s5I", 32, DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0)
        dx10 = struct.pack("<IIIII", dxgi, 3, 4 if cubemap else 0, 1, 0)
    head = struct.pack("<7I44x", 124, flags, h, w, pitch, 0, max(1, mips)) + pf + struct.pack("<5I", caps, caps2, 0, 0, 0)
    return b"DDS " + head + dx10


# ------------------------------------------------------------------ dossier de jeu

def texture_archives(folder: str | Path) -> list[Path]:
    """Archives BA2 de textures (type DX10) directement dans `folder` (le dossier Data d'un jeu),
    archives principales du jeu d'abord : une archive chargée après remplace une texture de même chemin."""
    d = Path(folder)
    try:
        cands = [Path(e.path) for e in os.scandir(d) if e.is_file() and e.name.lower().endswith(".ba2")]
    except OSError:
        return []
    found = [p for p in cands if texture_count(p) > 0]

    def order(p: Path) -> tuple[int, str]:
        low = p.name.lower()
        return (0 if low.startswith(("fallout4 - ", "starfield - ", "seventysix - ")) else 1, low)
    return sorted(found, key=order)


def data_folder(folder: str | Path) -> Path | None:
    """Dossier qui contient les archives de textures : le dossier choisi, ou son sous-dossier Data."""
    p = Path(folder)
    for cand in (p, p / "Data", p / "data"):
        if cand.is_dir() and texture_archives(cand):
            return cand
    return None


def mod_output_for(data_dir: str | Path) -> Path:
    """Sortie proposée : un dossier de mod à côté de Data (Textures\\… dedans), à copier dans Data
    ou à installer avec un gestionnaire de mods. Jamais dans Data : TEXFORGE n'écrase rien du jeu."""
    d = Path(data_dir)
    root = d.parent if d.name.lower() == "data" else d
    return root / "TEXFORGE_Remaster"


def matches(name: str, patterns: list[str]) -> bool:
    """Motifs séparés par « ; » : avec un « / », comparés au chemin interne, sinon au nom du fichier.
    Aucun motif : tout est retenu."""
    if not patterns:
        return True
    low = name.lower()
    base = low.rsplit("/", 1)[-1]
    for pat in patterns:
        p = pat.strip().lower().replace("\\", "/")
        if p and fnmatch.fnmatch(low if "/" in p else base, p):
            return True
    return False


def safe_rel(name: str) -> str | None:
    """Chemin interne utilisable sous le dossier d'extraction (ni absolu, ni « .. »)."""
    parts = [x for x in name.replace("\\", "/").split("/") if x not in ("", ".")]
    if not parts or any(x == ".." or ":" in x for x in parts):
        return None
    return "/".join(parts)


def parse_patterns(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").replace(",", ";").split(";") if p.strip()]


def extract_textures(data_dir: str | Path, dest: str | Path, *, include: str = "", skip_loose: bool = True,
                     scale: int = 0, max_side: int = 0, limit: int = 0,
                     log: Callable[[str], None] | None = None,
                     on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Extrait les textures des archives BA2 de `data_dir` vers `dest`, chemins internes conservés
    (dest/Textures/...). Ne sont retenues que les textures qui passent le filtre `include`, qui ne
    sont pas remplacées par un fichier séparé dans `data_dir` (mod : le jeu affiche celui-là) et que
    `scale`/`max_side` permettent d'agrandir. `limit` : échantillon stable et varié (lot d'essai).
    Une texture déjà extraite de la même archive n'est pas réécrite."""
    from .imaging import output_scale
    log = log or (lambda _m: None)
    data_dir, dest = Path(data_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    pats = parse_patterns(include)
    man_path = dest / MANIFEST
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    counts = {"archives": 0, "listed": 0, "matched": 0, "selected": 0, "written": 0, "kept": 0, "cubemaps": 0,
              "loose": 0, "maxed": 0, "errors": 0}
    chosen: dict[str, tuple[Path, Ba2Texture]] = {}
    archives = texture_archives(data_dir)
    for arc in archives:
        try:
            with Ba2Archive(arc) as a:
                counts["archives"] += 1
                counts["listed"] += len(a.textures)
                n = 0
                for t in a.textures:
                    if not matches(t.name, pats):
                        continue
                    counts["matched"] += 1
                    if t.cubemap:
                        counts["cubemaps"] += 1
                        continue
                    if safe_rel(t.name) is None:
                        counts["errors"] += 1
                        continue
                    if skip_loose and (data_dir / t.name).is_file():
                        counts["loose"] += 1
                        continue
                    if max_side and scale and output_scale(t.width, t.height, scale, max_side) <= 1:
                        counts["maxed"] += 1              # déjà à la taille maximale : rien à gagner
                        continue
                    chosen[t.name.lower()] = (arc, t)          # archive chargée plus tard : elle l'emporte
                    n += 1
                log(tr("[BA2] {name} : {total} texture(s), {n} retenue(s)", name=arc.name, total=len(a.textures), n=n))
        except Ba2Error as exc:
            counts["errors"] += 1
            log(tr("[BA2] {name} ignorée : {exc}", name=arc.name, exc=exc))
    if limit and len(chosen) > limit:
        keep = sorted(chosen, key=name_hash)[:limit]        # échantillon réparti dans tout le jeu, toujours le même
        chosen = {k: chosen[k] for k in keep}
    counts["selected"] = len(chosen)
    by_arc: dict[Path, list[Ba2Texture]] = {}
    for arc, t in chosen.values():
        by_arc.setdefault(arc, []).append(t)
    done = 0
    for arc, texs in by_arc.items():
        st = arc.stat()
        stamp = {"arc": arc.name, "size": st.st_size, "mtime": st.st_mtime_ns}
        with Ba2Archive(arc) as a:
            for t in sorted(texs, key=lambda x: x.chunks[0].offset if x.chunks else 0):   # lecture dans l'ordre
                done += 1
                rel = safe_rel(t.name)
                out = dest / rel
                key = t.name.lower()
                if manifest.get(key) == {**stamp, "len": t.size, "path": rel} and out.is_file():
                    counts["kept"] += 1
                else:
                    try:
                        data = a.read_dds(t)
                    except (Ba2Error, zlib.error, OSError) as exc:
                        counts["errors"] += 1
                        log(tr("[BA2] {name} illisible : {exc}", name=t.name, exc=exc))
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".part")
                    tmp.write_bytes(data)
                    os.replace(tmp, out)
                    manifest[key] = {**stamp, "len": t.size, "path": rel}
                    counts["written"] += 1
                if on_progress is not None:
                    on_progress(done, len(chosen))
            man_path.write_text(json.dumps(manifest), encoding="utf-8")      # reprise possible après un arrêt
    # textures qui ne sont plus retenues (filtre changé) : retirées du dossier d'extraction
    for key in [k for k in manifest if k not in chosen]:
        entry = manifest.pop(key)
        rel = entry.get("path") if isinstance(entry, dict) else None
        if rel and safe_rel(rel) == rel:
            try:
                (dest / rel).unlink()
            except OSError:
                pass
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    return counts


# ------------------------------------------------------------------ écriture (tests, archives de démonstration)

def _crc_table() -> list[int]:
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        t.append(c)
    return t


_CRC = _crc_table()


def name_hash(s: str) -> int:
    """Hachage des noms BA2 : CRC32 sans inversion finale, en minuscules, séparateur « \\ »."""
    r = 0
    for ch in s.lower().replace("/", "\\").encode("latin-1", "ignore"):
        if ch > 127:
            continue
        r = (r >> 8) ^ _CRC[(r ^ ch) & 0xFF]
    return r


def write_texture_archive(path: str | Path, files: list[tuple[str, bytes]], *, version: int = 1,
                          compression: str = "zlib") -> None:
    """Archive DX10 à partir de fichiers DDS (chemin interne, contenu) : un morceau par mip pour les
    mips de 512 px et plus, les petits mips regroupés dans un dernier morceau. Sert aux tests."""
    from . import dds as ddsmod
    recs, blobs, names = [], [], []
    for name, data in files:
        info = ddsmod.read_info(data)
        dxgi = struct.unpack_from("<I", data, 128)[0] if info.dx10 else _DXGI_OF.get(info.format)
        if dxgi is None:
            raise Ba2Error(f"format {info.format}")
        payload = data[148 if info.dx10 else 128:]
        sizes, w, h = [], info.width, info.height
        for _ in range(max(1, info.mipmaps)):
            sizes.append(_level0_bytes(w, h, dxgi)[0])
            w, h = max(1, w // 2), max(1, h // 2)
        groups, i = [], 0
        while i < len(sizes):
            big = max(info.width >> i, info.height >> i) >= 512 and len(groups) < 3
            j = i + 1 if big else len(sizes)
            groups.append((i, j))
            i = j
        chunks, pos = [], 0
        for a_, b_ in groups:
            n = sum(sizes[a_:b_])
            raw = payload[pos:pos + n]
            pos += n
            if compression == "zlib":
                packed = zlib.compress(raw, 6)
            elif compression == "lz4":
                import lz4.block
                packed = lz4.block.compress(raw, store_size=False)
            else:
                packed = b""
            chunks.append((raw, packed, a_, b_ - 1))
        folder, _, fname = name.replace("/", "\\").rpartition("\\")
        stem, _, ext = fname.rpartition(".")
        recs.append((name_hash(stem), ext.lower().encode()[:4].ljust(4, b"\0"), name_hash(folder),
                     info.height, info.width, max(1, info.mipmaps), dxgi, chunks))
        names.append(name.replace("/", "\\").encode("utf-8"))
    head_len = 24 + (8 if version in (2, 3) else 0) + (4 if version == 3 else 0)
    table_len = sum(24 + 24 * len(r[7]) for r in recs)
    offset = head_len + table_len
    table = bytearray()
    for nh, ext, dh, h, w, mips, dxgi, chunks in recs:
        table += struct.pack("<I4sIBBHHHBBH", nh, ext, dh, 0, len(chunks), 24, h, w, mips, dxgi, 2048)
        for raw, packed, m0, m1 in chunks:
            body = packed or raw
            table += struct.pack("<QIIHHI", offset, len(packed), len(raw), m0, m1, BAADFOOD)
            blobs.append(body)
            offset += len(body)
    names_at = offset
    head = MAGIC + struct.pack("<I4sIQ", version, b"DX10", len(recs), names_at)
    if version in (2, 3):
        head += b"\0" * 8
    if version == 3:
        head += struct.pack("<I", 3 if compression == "lz4" else 0)
    with open(path, "wb") as f:
        f.write(head)
        f.write(table)
        for b in blobs:
            f.write(b)
        for n in names:
            f.write(struct.pack("<H", len(n)) + n)


__all__ = ["Ba2Archive", "Ba2Error", "Ba2Texture", "header_info", "texture_count", "texture_archives",
           "data_folder", "mod_output_for", "extract_textures", "matches", "parse_patterns", "safe_rel", "dds_header", "write_texture_archive",
           "name_hash"]
