"""Construction des DDS finaux : choix du format, chaîne de mipmaps, compression, vérification.

Répartition des rôles :
- la chaîne de mipmaps est calculée ici (filtrage en lumière linéaire pour les couleurs,
  addressage en boucle pour les textures répétables, couverture alpha conservée pour les
  découpes, renormalisation des normal maps) ;
- la compression BC de chaque niveau est confiée à texconv (DirectXTex, Microsoft) quand il
  est installé, sinon à l'encodeur intégré (Pillow : DXT1/DXT5, DXT3 écrit ici) ;
- le conteneur DDS est assemblé ici (en-tête legacy lisible par D3D9, ou DX10 si l'original
  en avait un), puis relu niveau par niveau et comparé à la chaîne de référence.

texconv ne reçoit donc qu'un niveau à la fois (-m 1) : son comportement de génération de
mipmaps n'entre jamais en jeu, et le résultat ne dépend pas de sa version.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import dds
from .imaging import Analysis, bleed_rgb
from .metrics import linear_to_srgb, resize_f32, srgb_to_linear

LANCZOS = Image.Resampling.LANCZOS
BOX = Image.Resampling.BOX
TEXCONV_FORMAT = {"DXT1": "BC1_UNORM", "DXT3": "BC2_UNORM", "DXT5": "BC3_UNORM", "BC7": "BC7_UNORM"}
DATA_FORMATS = {"L8", "A8", "A8L8"}


# ------------------------------------------------------------------ plan

@dataclass
class DDSPlan:
    fmt: str                     # nom écrit (DXT1, BC3_SRGB, BGRA8…)
    codec: str                   # DXT1 | DXT3 | DXT5 | BC7 | disposition non compressée
    levels: int
    width: int
    height: int
    color_space: str             # srgb | linear | normal
    alpha_mode: str              # none | test (découpe) | blend (dégradé)
    tile_x: bool = False
    tile_y: bool = False
    alpha_ref: float = 0.5
    notes: list[str] = field(default_factory=list)

    @property
    def compressed(self) -> bool:
        return self.codec in dds.BLOCK_BYTES


def _srgb_variant(fmt: str, srgb: bool, dx10: bool) -> str:
    base = {"DXT1": "BC1", "DXT3": "BC2", "DXT5": "BC3", "BC7": "BC7"}.get(fmt, fmt)
    if not dx10:
        return fmt
    return f"{base}_SRGB" if srgb else base


def choose_format(src_fmt: str, src_is_dds: bool, an: Analysis, cfg: dict, *, data: bool,
                  can_bc7: bool, override: str | None = None) -> tuple[str, list[str]]:
    """Format de sortie. « match » garde le format de l'original (le jeu ou l'injecteur l'attend) ;
    « auto » choisit selon la transparence ; les autres valeurs imposent un format.
    `override` : format imposé par une règle [[rules]] pour cette texture."""
    out_cfg, qc = cfg["output"], cfg["qc"]
    policy = str(override or out_cfg.get("dds_format", "match")).lower()
    api = str(qc.get("target_api", "d3d9")).lower()
    notes: list[str] = []
    src_dx10 = src_is_dds and dds.needs_dx10(src_fmt)
    src_srgb = src_fmt.endswith("_SRGB")
    alpha = an.alpha_class

    def auto() -> str:
        if alpha == "none":
            f = "DXT1"
        elif alpha == "binary" and qc.get("dxt1_alpha", False):
            f = "DXT1"
        else:
            f = "DXT5"
        return _srgb_variant(f, src_srgb, src_dx10)

    if policy == "match" or (data and policy != "bgra8"):
        if src_is_dds and (src_fmt in dds.WRITABLE_FORMATS or src_fmt in dds.DX10_WRITE):
            fmt = src_fmt
        elif src_is_dds:
            fmt = auto()
            notes.append(f"format d'origine {src_fmt} non inscriptible : {fmt}")
        else:
            # original PNG/TGA (format du jeu inconnu) : sans perte, lisible par D3D9
            fmt = "BGRA8"
            notes.append("original non DDS : BGRA8 sans perte (dds_format = \"auto\" pour compresser en DXT)")
    elif policy == "auto":
        fmt = auto()
    elif policy == "dxt1":
        fmt = "DXT1" if alpha == "none" else "DXT5"
    elif policy == "dxt5":
        fmt = "DXT5"
    elif policy == "bgra8":
        fmt = "BGRA8"
    elif policy == "bc7":
        fmt = _srgb_variant("BC7", src_srgb, True)
    else:
        raise ValueError(f"dds_format inconnu : {policy}")

    # Dégradé d'alpha : DXT1 ne stocke qu'un alpha tout-ou-rien
    if alpha == "gradient" and dds.codec_of(fmt) == "DXT1":
        fmt = {"DXT1": "DXT5", "BC1": "BC3", "BC1_SRGB": "BC3_SRGB"}[fmt]
        notes.append("alpha en dégradé : DXT1 → DXT5")

    # BC7 : Direct3D 11 uniquement (dgVoodoo2 + Special K), et seulement avec texconv
    if dds.codec_of(fmt) == "BC7":
        if api != "d3d11" or not qc.get("allow_bc7", False):
            fmt = _srgb_variant("DXT5" if alpha != "none" else "DXT1", fmt.endswith("_SRGB"), src_dx10)
            notes.append("BC7 refusé (Direct3D 9/10 ne le lit pas ; target_api = \"d3d11\" et allow_bc7 = true "
                         "pour dgVoodoo2 + Special K)")
        elif not can_bc7:
            fmt = _srgb_variant("DXT5", fmt.endswith("_SRGB"), True)
            notes.append("BC7 impossible sans texconv : BC3")

    # Direct3D 9 natif : pas d'en-tête DX10
    if api == "d3d9" and dds.needs_dx10(fmt) and not src_dx10:
        legacy = dds.LEGACY_OF.get(fmt, "BGRA8")
        notes.append(f"{fmt} demande un en-tête DX10 : {legacy} (D3D9)")
        fmt = legacy
    return fmt, notes


def plan_for(img: Image.Image, an: Analysis, src_fmt: str, src_is_dds: bool, src_mips: int, cfg: dict, *,
             data: bool, can_bc7: bool, override: str | None = None) -> DDSPlan:
    qc = cfg["qc"]
    fmt, notes = choose_format(src_fmt, src_is_dds, an, cfg, data=data, can_bc7=can_bc7, override=override)
    codec = dds.codec_of(fmt)
    w, h = img.size
    policy = str(cfg["output"].get("mipmaps", "always")).lower()
    if policy == "never":
        levels = 1
    elif policy == "match" and (src_mips or 1) <= 1:
        levels = 1
    else:
        levels = dds.mip_count_full(w, h)
    if an.normal_map:
        space = "normal"
    elif data or src_fmt in DATA_FORMATS:
        space = "linear"
    else:
        space = str(qc.get("mip_gamma", "srgb"))
    alpha_mode = {"none": "none", "binary": "test", "gradient": "blend"}[an.alpha_class]
    if codec in ("BGRX8", "BGR8", "R5G6B5", "X1R5G5B5", "L8") and alpha_mode != "none":
        notes.append(f"{fmt} sans alpha : transparence perdue")
    return DDSPlan(fmt, codec, levels, w, h, space, alpha_mode, an.tile_x, an.tile_y,
                   float(qc.get("alpha_ref", 0.5)), notes)


def fit_block_size(img: Image.Image, plan_codec: str) -> tuple[Image.Image, str | None]:
    """D3D9 exige des côtés multiples de 4 pour un niveau 0 compressé."""
    if plan_codec not in dds.BLOCK_BYTES:
        return img, None
    w, h = img.size
    W, H = max(4, round(w / 4) * 4), max(4, round(h / 4) * 4)
    if (W, H) == (w, h):
        return img, None
    return img.resize((W, H), LANCZOS), f"{w}x{h} → {W}x{H} (multiple de 4 exigé par DXT)"


# ------------------------------------------------------------------ chaîne de mipmaps

def _coverage(a: np.ndarray, ref: float) -> float:
    return float((a >= ref).mean())


def _scale_to_coverage(a: np.ndarray, ref: float, target: float) -> np.ndarray:
    """Remappe l'alpha d'un niveau pour retrouver la couverture du niveau 0 au seuil du test
    alpha (sans cela, grillages et feuillages « fondent » à distance). On cherche le seuil t
    qui donne la couverture voulue, puis une courbe monotone envoie t sur `ref` en gardant
    0 -> 0 et 1 -> 1 : les texels pleins restent pleins, seuls les bords bougent."""
    t = float(np.quantile(a, 1.0 - target))
    t = min(max(t, 1e-3), 1 - 1e-3)
    return np.where(a < t, a * (ref / t), ref + (a - t) * ((1 - ref) / (1 - t))).astype(np.float32)


def build_chain(img: Image.Image, plan: DDSPlan, mip_filter: str = "lanczos") -> list[np.ndarray]:
    """Niveaux RGBA uint8, du plus grand au 1x1. Le niveau 0 est l'image elle-même, sans
    aller-retour de quantification ; les suivants sont filtrés en cascade en flottants."""
    base = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    if plan.alpha_mode == "test":
        # couleurs étendues sous les zones transparentes : pas de liseré sombre dans les mips
        bled = np.asarray(bleed_rgb(img), dtype=np.uint8)
        base = np.concatenate([bled[..., :3], base[..., 3:4]], axis=-1)
    chain = [base]
    if plan.levels <= 1:
        return chain
    f = base.astype(np.float32) / 255.0
    a = f[..., 3]
    if plan.color_space == "srgb":
        c = srgb_to_linear(f[..., :3])
    elif plan.color_space == "normal":
        c = f[..., :3] * 2.0 - 1.0
    else:
        c = f[..., :3].copy()
    if plan.alpha_mode == "blend":
        c = c * a[..., None]                      # alpha prémultiplié : pas de couleur fantôme
    target_cov = _coverage(a, plan.alpha_ref) if plan.alpha_mode == "test" else None
    resample = LANCZOS if mip_filter == "lanczos" else BOX
    W, H = plan.width, plan.height
    for i in range(1, plan.levels):
        size = dds.level_dims(W, H, i)
        c = resize_f32(c, size, resample, plan.tile_x, plan.tile_y)
        a = np.clip(resize_f32(a, size, BOX, plan.tile_x, plan.tile_y), 0, 1)
        col = c
        if plan.alpha_mode == "blend":
            col = np.where(a[..., None] > 1e-4, c / np.maximum(a[..., None], 1e-4), 0)
        if plan.color_space == "srgb":
            rgb = linear_to_srgb(col)
        elif plan.color_space == "normal":
            n = col / np.maximum(np.linalg.norm(col, axis=-1, keepdims=True), 1e-6)
            rgb = (n + 1.0) * 0.5
        else:
            rgb = np.clip(col, 0, 1)
        alpha = a
        if target_cov is not None and 0.0 < target_cov < 1.0:
            alpha = _scale_to_coverage(a, plan.alpha_ref, target_cov)
            # marge autour du seuil : la quantification DXT5 (8 niveaux par bloc) ne doit pas
            # faire basculer les texels qui frôlent le seuil du test alpha
            m = 12.0 / 255.0
            near = np.abs(alpha - plan.alpha_ref) < m
            alpha = np.where(near, np.where(alpha >= plan.alpha_ref, plan.alpha_ref + m, plan.alpha_ref - m), alpha)
        if plan.alpha_mode == "none":
            alpha = np.ones_like(a)
        lvl = np.concatenate([rgb, alpha[..., None]], axis=-1)
        chain.append(np.clip(lvl * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if plan.codec == "DXT1" and plan.alpha_mode != "none":
        # DXT1 : alpha sur 1 bit, seuillé ici au seuil du test alpha (indépendant de l'encodeur)
        thr = round(plan.alpha_ref * 255)
        for lvl in chain:
            lvl[..., 3] = np.where(lvl[..., 3] >= thr, 255, 0)
    return chain


# ------------------------------------------------------------------ encodeurs

class EncoderError(Exception):
    pass


Job = tuple[str, np.ndarray, str]          # (clé, niveau RGBA uint8, codec)


class BuiltinEncoder:
    name = "builtin"
    block_codecs = {"DXT1", "DXT3", "DXT5"}

    def encode(self, jobs: list[Job]) -> dict[str, bytes]:
        out = {}
        for key, rgba, codec in jobs:
            if codec in dds.BLOCK_BYTES:
                if codec not in self.block_codecs:
                    raise EncoderError(f"encodeur intégré : {codec} non géré")
                out[key] = dds.encode_blocks(rgba, codec)
            else:
                out[key] = dds._encode_level_uncompressed(Image.fromarray(rgba, "RGBA"), codec)
        return out


def find_texconv(cfg: dict) -> Path | None:
    raw = str(cfg["qc"].get("texconv_exe", "") or "")
    cands: list[Path] = []
    if raw:
        p = Path(raw)
        cands.append(p if p.is_absolute() else Path(cfg.get("_base", ".")) / p)
    found = shutil.which("texconv") or shutil.which("texconv.exe")
    if found:
        cands.append(Path(found))
    root = Path(__file__).resolve().parent.parent
    cands += [root / "tools" / "texconv.exe", root / "texconv.exe"]
    return next((c for c in cands if c.exists()), None)


class TexconvEncoder:
    """Compression BC par texconv, un niveau par fichier, tous les niveaux d'un même format
    en un seul lancement. Les formats non compressés restent à l'encodeur intégré."""

    name = "texconv"
    block_codecs = {"DXT1", "DXT3", "DXT5", "BC7"}

    def __init__(self, exe: Path, cfg: dict, log: Callable[[str], None] | None = None) -> None:
        self.exe = exe
        self.qc = cfg["qc"]
        self.log = log or (lambda _m: None)
        self.tmp_root = Path(cfg["_work"]) / "tmp_texconv"
        self.builtin = BuiltinEncoder()

    def encode(self, jobs: list[Job]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        plain = [j for j in jobs if j[2] not in dds.BLOCK_BYTES]
        out.update(self.builtin.encode(plain))
        by_codec: dict[str, list[Job]] = {}
        for j in jobs:
            if j[2] in dds.BLOCK_BYTES:
                by_codec.setdefault(j[2], []).append(j)
        for codec, group in by_codec.items():
            out.update(self._run(codec, group))
        return out

    def _run(self, codec: str, jobs: list[Job]) -> dict[str, bytes]:
        run_dir = self.tmp_root / f"{codec}_{uuid.uuid4().hex[:8]}"
        in_dir, out_dir = run_dir / "in", run_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        names: dict[str, tuple[str, int, int]] = {}
        try:
            files = []
            for i, (key, rgba, _c) in enumerate(jobs):
                arr = dds._pad4(rgba)
                h, w = arr.shape[:2]
                stem = f"l{i:06d}"
                payload = dds._encode_level_uncompressed(Image.fromarray(arr, "RGBA"), "RGBA8")
                (in_dir / f"{stem}.dds").write_bytes(dds.make_header("RGBA8", w, h, 1) + payload)
                names[stem] = (key, rgba.shape[1], rgba.shape[0])
                files.append(str(in_dir / f"{stem}.dds"))
            base = [str(self.exe), "-nologo", "-y", "-m", "1", "-f", TEXCONV_FORMAT[codec], "-o", str(out_dir)]
            if codec == "DXT1":
                base += ["-at", f"{float(self.qc.get('alpha_ref', 0.5)):.3f}"]
            base += [str(a) for a in self.qc.get("texconv_args", [])]
            timeout = float(self.qc.get("texconv_timeout", 600))
            t0 = time.monotonic()
            for chunk in _chunk_args(files, budget=30000 - sum(len(a) + 1 for a in base)):
                try:
                    proc = subprocess.run(base + chunk, capture_output=True, timeout=timeout, cwd=str(run_dir))
                except subprocess.TimeoutExpired as exc:
                    raise EncoderError(f"texconv : délai de {timeout:.0f} s dépassé") from exc
                except OSError as exc:
                    raise EncoderError(f"texconv : lancement impossible ({exc})") from exc
                if proc.returncode != 0:
                    tail = (proc.stdout + proc.stderr).decode("utf-8", "replace").strip().splitlines()[-3:]
                    raise EncoderError(f"texconv code {proc.returncode} : {' | '.join(tail)}")
            self.log(f"[DDS] texconv {TEXCONV_FORMAT[codec]} : {len(jobs)} niveau(x) en {time.monotonic() - t0:.1f} s")
            produced = {p.stem.lower(): p for p in out_dir.iterdir() if p.suffix.lower() == ".dds"}
            result: dict[str, bytes] = {}
            for stem, (key, w, h) in names.items():
                f = produced.get(stem)
                if f is None:
                    raise EncoderError(f"texconv : sortie absente pour {stem}")
                _info, got, levels = dds.split_levels(f.read_bytes())
                if got != codec:
                    raise EncoderError(f"texconv : format {got} reçu au lieu de {codec}")
                result[key] = levels[0][2]
            return result
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


def _chunk_args(files: list[str], budget: int) -> list[list[str]]:
    chunks, cur, size = [], [], 0
    for f in files:
        if cur and size + len(f) + 1 > max(2000, budget):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += len(f) + 1
    if cur:
        chunks.append(cur)
    return chunks


def make_encoder(cfg: dict, name: str | None = None, log: Callable[[str], None] | None = None):
    choice = (name or str(cfg["qc"].get("encoder", "auto"))).lower()
    if choice == "builtin":
        return BuiltinEncoder()
    exe = find_texconv(cfg)
    if exe is None:
        if choice == "texconv":
            raise EncoderError("texconv introuvable : renseigne [qc] texconv_exe ou place texconv.exe dans tools/")
        return BuiltinEncoder()
    return TexconvEncoder(exe, cfg, log)


# ------------------------------------------------------------------ assemblage et vérification

def assemble(plan: DDSPlan, chain: list[np.ndarray], payloads: list[bytes]) -> bytes:
    return dds.write_dds(plan.fmt, plan.width, plan.height, payloads)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse <= 1e-9 else round(10 * np.log10(255.0 ** 2 / mse), 2)


def verify(data: bytes, plan: DDSPlan, chain: list[np.ndarray]) -> dict:
    """Relit le DDS écrit, décode chaque niveau et le compare à la chaîne de référence :
    dimensions, nombre de niveaux, PSNR, couverture alpha, luminosité moyenne (une chaîne
    filtrée en gamma assombrit les niveaux éloignés)."""
    res = {"ok": True, "errors": [], "psnr_l0": None, "psnr_min": None,
           "coverage_drift": None, "brightness_drift": None}
    try:
        info, codec, levels = dds.split_levels(data)
    except dds.DDSError as exc:
        res.update(ok=False, errors=[f"relecture : {exc}"])
        return res
    if (info.width, info.height) != (plan.width, plan.height) or len(levels) != len(chain):
        res["errors"].append(f"en-tête {info.width}x{info.height}/{len(levels)} niv. au lieu de "
                             f"{plan.width}x{plan.height}/{len(chain)}")
    if codec != plan.codec:
        res["errors"].append(f"format relu {codec} au lieu de {plan.codec}")
    psnrs, cov, bright = [], [], []
    thr = plan.alpha_ref * 255
    ref_cov = float((chain[0][..., 3] >= thr).mean()) if plan.alpha_mode == "test" else None
    ref_lum = None
    for i, ((w, h, payload), ref) in enumerate(zip(levels, chain)):
        try:
            dec = dds.decode_level(codec, w, h, payload)
        except Exception as exc:
            res["errors"].append(f"niveau {i} illisible : {exc}")
            break
        if dec.shape[:2] != ref.shape[:2]:
            res["errors"].append(f"niveau {i} : taille {dec.shape[1]}x{dec.shape[0]}")
            break
        m = ref[..., 3] >= thr if plan.alpha_mode != "none" else np.ones(ref.shape[:2], bool)
        if m.sum() >= 16:
            p = _psnr(dec[..., :3][m], ref[..., :3][m])
            psnrs.append(p)
            if i == 0:
                res["psnr_l0"] = p
            if plan.alpha_mode == "none" and plan.color_space == "srgb" and min(w, h) >= 8:
                # luminosité moyenne en lumière linéaire : une chaîne filtrée en gamma s'assombrit
                lin = float(srgb_to_linear(dec[..., :3].astype(np.float32) / 255.0).mean())
                if ref_lum is None:
                    ref_lum = lin
                elif ref_lum > 0.01:
                    bright.append(abs(lin - ref_lum) / ref_lum)
        if ref_cov is not None and min(w, h) >= 16:      # en dessous, la couverture varie par pas de 1/256
            cov.append(abs(float((dec[..., 3] >= thr).mean()) - ref_cov))
    if psnrs:
        res["psnr_min"] = min(psnrs)
    if cov:
        res["coverage_drift"] = round(max(cov), 4)
    if bright:
        res["brightness_drift"] = round(max(bright), 4)
    res["ok"] = not res["errors"]
    return res


@dataclass
class DDSOutput:
    data: bytes
    plan: DDSPlan
    encoder: str
    check: dict


def encode_many(items: dict[str, tuple[DDSPlan, list[np.ndarray]]], encoder, *, verify_output: bool = True,
                log: Callable[[str], None] | None = None) -> dict[str, DDSOutput | Exception]:
    """Encode un lot de textures en un minimum de lancements d'encodeur. En cas d'échec de
    texconv, l'encodeur intégré reprend les formats qu'il connaît."""
    log = log or (lambda _m: None)
    jobs: list[Job] = []
    for tid, (plan, chain) in items.items():
        for i, lvl in enumerate(chain):
            jobs.append((f"{tid}\x00{i}", lvl, plan.codec))
    used = encoder.name
    try:
        encoded = encoder.encode(jobs)
    except EncoderError as exc:
        if isinstance(encoder, BuiltinEncoder):
            return {tid: exc for tid in items}
        log(f"[DDS] {exc} — repli sur l'encodeur intégré")
        used = "builtin (repli)"
        builtin = BuiltinEncoder()
        # BC7 n'a pas d'encodeur intégré : même famille DX10, en BC3
        for tid, (plan, _chain) in items.items():
            if plan.codec == "BC7":
                plan.fmt = "BC3_SRGB" if plan.fmt.endswith("_SRGB") else "BC3"
                plan.codec = "DXT5"
                plan.notes.append("texconv indisponible : BC3 au lieu de BC7")
        jobs = [(key, lvl, items[key.split("\x00")[0]][0].codec) for key, lvl, _c in jobs]
        encoded = builtin.encode(jobs)
    out: dict[str, DDSOutput | Exception] = {}
    for tid, (plan, chain) in items.items():
        try:
            payloads = [encoded[f"{tid}\x00{i}"] for i in range(len(chain))]
        except KeyError:
            out[tid] = EncoderError(f"aucun encodeur disponible pour {plan.codec}")
            continue
        try:
            data = assemble(plan, chain, payloads)
        except dds.DDSError as exc:
            out[tid] = exc
            continue
        check = verify(data, plan, chain) if verify_output else {"ok": True, "errors": []}
        out[tid] = DDSOutput(data, plan, used, check)
    return out


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
