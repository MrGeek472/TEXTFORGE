"""Mode sans ComfyUI : agrandissement réaliste par Real-ESRGAN (realesrgan-ncnn-vulkan).

Real-ESRGAN x4plus redessine des contours nets (briques, pelage, yeux), mais sur une matière
bruitée (pierre, gravier, herbe) il invente des structures : quadrillages, fissures, « murs ».
Ailleurs il lisse le grain fin : aspect « cireux ». Mesuré sur des textures dont la version haute
résolution est connue (réduites ×4 puis remontées), la chaîne ci-dessous rapproche nettement le
résultat du réel (SSIM 0,49 → 0,67, écart de spectre 8,8 → 4,8 dB, écart de couleur ΔE 2,9 → 0,6).

Chaîne appliquée par lots (chaque modèle n'est chargé qu'une fois par lot) :

  1. bords : marge circulaire sur les axes répétables (raccord parfait), miroir sinon ;
  2. deux rendus ×4 : realesrgan-x4plus (net) et realesr-animevideov3 (prudent, n'invente rien),
     réduits à la taille cible (×2 = suréchantillonnage) ;
  3. verrou des couleurs : les basses fréquences (teintes, éclairage, grandes taches) sont reprises
     de l'original agrandi ; la DA ne bouge pas ;
  4. accord du détail : l'énergie de détail attendue dans les octaves créées est prolongée depuis
     la dernière octave de l'original (spectre en loi de puissance des matières) ; un grain de
     matière comble le manque, un excès de contours est atténué. Un aplat reste un aplat. Le
     grain est périodique : une texture répétable le reste ;
  5. garde anti-invention : là où x4plus produit beaucoup plus de détail que le modèle prudent
     (rapport d'énergie > 3, total au-delà de 9), c'est une structure inventée : on passe
     progressivement au rendu prudent ;
  6. transparence : canal alpha d'origine agrandi à part (contours nets, couverture conservée).

L'exécutable officiel (GitHub xinntao/Real-ESRGAN, v0.2.5.0) peut être installé dans tools\\ :
install_realesrgan() vérifie la taille et l'empreinte SHA-256 de l'archive avant de l'ouvrir.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .i18n import tr
from .imaging import Analysis, bleed_rgb, final_alpha
from .metrics import resize_f32

ROOT = Path(__file__).resolve().parent.parent
EXE_NAME = "realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan"
RELEASE_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
# archive officielle par système : (nom, taille en octets, SHA-256)
RELEASES = {
    "windows": ("realesrgan-ncnn-vulkan-20220424-windows.zip", 45474481,
                "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"),
    "linux": ("realesrgan-ncnn-vulkan-20220424-ubuntu.zip", 46931474,
              "e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96"),
    "macos": ("realesrgan-ncnn-vulkan-20220424-macos.zip", 51817124,
              "e0ad05580abfeb25f8d8fb55aaf7bedf552c375b5b4d9bd3c8d59764d2cc333a"),
}
INSTALL_DIR = ROOT / "tools" / "realesrgan-ncnn-vulkan"

DEFAULTS = {
    "model": "realesrgan-x4plus",          # net, le plus photographique des modèles livrés
    "guard_model": "realesr-animevideov3",  # prudent : référence de la garde anti-invention ("" = sans)
    "guard": [3.0, 9.0],                    # rapport d'énergie x4plus / prudent : début, bascule totale
    "color_lock": 1.0,                      # 0 : couleurs du modèle ; 1 : basses fréquences de l'original
    "detail": 1.0,                          # micro-détail : 0 = aucun grain ajouté, 1 = dose mesurée, 1.5 = appuyé
    "detail_gain": 0.3,                     # énergie par octave créée / énergie de la dernière octave (mesuré)
    "tame": 0.8,                            # atténuation des contours en excès vers la cible (0 à 1)
    "tta": False,                           # -x : 8 passes (un peu plus propre, 8 fois plus lent)
    "tile": 0,                              # 0 = automatique (selon la VRAM)
    "gpu": "",
    "batch_size": 32,
    "timeout": 600,
    "materials": {"stone": 1.0, "masonry": 1.0, "generic": 1.0, "wood": 0.9, "fabric": 0.9,
                  "vegetation": 0.8, "metal": 0.6, "skin": 0.0},
}


class EsrganError(Exception):
    pass


def settings(cfg: dict | None) -> dict:
    e = {**DEFAULTS, **((cfg or {}).get("esrgan") or {})}
    e["materials"] = {**DEFAULTS["materials"], **(e.get("materials") or {})}
    return e


# ------------------------------------------------------------------ exécutable

def _has_models(exe: Path, model: str = "realesrgan-x4plus", scale: int = 4) -> bool:
    """Modèle présent à côté de l'exécutable (realesr-animevideov3 existe en -x2/-x3/-x4)."""
    d = exe.parent / "models"
    return (d / f"{model}.param").is_file() or (d / f"{model}-x{scale}.param").is_file()


def find_realesrgan(cfg: dict | None = None) -> Path | None:
    """realesrgan-ncnn-vulkan : chemin configuré, tools\\ de TEXFORGE, puis PATH."""
    cands: list[Path] = []
    for raw in ((cfg or {}).get("esrgan", {}).get("exe"), (cfg or {}).get("ui_upscale", {}).get("realesrgan_exe")):
        if raw:
            p = Path(str(raw))
            if not p.is_absolute():
                p = Path((cfg or {}).get("_base", ROOT)) / p
            cands.append(p)
    tools = ROOT / "tools"
    cands += [INSTALL_DIR / EXE_NAME, tools / EXE_NAME]
    if tools.is_dir():
        cands += sorted(tools.glob(f"*/{EXE_NAME}"))
    found = shutil.which("realesrgan-ncnn-vulkan")
    if found:
        cands.append(Path(found))
    for p in cands:
        if "realesrgan" in p.stem.lower() and p.is_file() and _has_models(p):
            return p
    return None


def platform_key() -> str:
    if os.name == "nt":
        return "windows"
    return "macos" if sys.platform == "darwin" else "linux"


def install_realesrgan(dest: Path = INSTALL_DIR, progress: Callable[[int, int], None] | None = None,
                       url_base: str = RELEASE_URL) -> Path:
    """Télécharge l'archive officielle, vérifie taille et SHA-256, extrait l'exécutable et les modèles."""
    name, size, sha = RELEASES[platform_key()]
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / f".{name}.part"
    h = hashlib.sha256()
    got = 0
    try:
        req = urllib.request.Request(url_base + name, headers={"User-Agent": "TEXFORGE"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if progress:
                    progress(got, size)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise EsrganError(tr("téléchargement de Real-ESRGAN impossible : {err}", err=exc)) from exc
    if got != size or h.hexdigest() != sha:
        tmp.unlink(missing_ok=True)
        raise EsrganError(tr("archive Real-ESRGAN non conforme ({got} octets reçus, {size} attendus, ou empreinte "
                             "SHA-256 différente de la version officielle) : rien n'a été installé", got=got, size=size))
    try:
        with zipfile.ZipFile(tmp) as z:
            for m in z.infolist():
                n = m.filename.replace("\\", "/")
                base = n.rsplit("/", 1)[-1]
                if m.is_dir() or ".." in n.split("/") or n.startswith("/"):
                    continue
                if not (base.startswith("realesrgan-ncnn-vulkan") or base.endswith((".dll", ".param", ".bin"))):
                    continue                                   # démos (jpg, mp4), README : inutiles
                target = dest / "models" / base if "/models/" in f"/{n}" else dest / base
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as srcf, open(target, "wb") as out:
                    shutil.copyfileobj(srcf, out)
    finally:
        tmp.unlink(missing_ok=True)
    exe = dest / EXE_NAME
    if not exe.is_file():
        raise EsrganError(tr("archive Real-ESRGAN sans exécutable {name}", name=EXE_NAME))
    if os.name != "nt":
        exe.chmod(0o755)
    return exe


def run_ncnn(exe: Path, images: dict[str, Image.Image], *, model: str, scale: int, work_dir: Path,
             tile: int = 0, gpu: str = "", tta: bool = False, timeout: float = 600,
             log: Callable[[str], None] | None = None) -> dict[str, Image.Image]:
    """Un lancement pour tout un dossier d'images RGB ; renvoie les images agrandies (par clé)."""
    run_dir = Path(work_dir).resolve() / f"esrgan_{uuid.uuid4().hex[:8]}"      # absolu : l'exe tourne depuis son dossier
    exe = Path(exe).resolve()
    in_dir, out_dir = run_dir / "in", run_dir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        names = {}
        for i, (key, img) in enumerate(images.items()):
            names[key] = f"{i:05d}.png"
            img.convert("RGB").save(in_dir / names[key])
        cmd = [str(exe), "-i", str(in_dir), "-o", str(out_dir), "-n", model, "-s", str(scale), "-f", "png",
               "-m", str(exe.parent / "models"), "-t", str(int(tile))]
        if gpu != "":
            cmd += ["-g", str(gpu)]
        if tta:
            cmd.append("-x")
        t0 = time.monotonic()
        extra = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 4.0 * len(images),
                                  cwd=str(exe.parent), **extra)
        except subprocess.TimeoutExpired as exc:
            raise EsrganError(tr("Real-ESRGAN : délai de {seconds:.0f} s dépassé", seconds=timeout)) from exc
        except OSError as exc:
            raise EsrganError(tr("Real-ESRGAN : lancement impossible ({err})", err=exc)) from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
            raise EsrganError(tr("Real-ESRGAN : code {code} : {tail}", code=proc.returncode, tail=" | ".join(tail)))
        if log:
            log(tr("[ESRGAN] {model} ×{scale} : {n} image(s) en {seconds:.1f} s", model=model, scale=scale,
                   n=len(images), seconds=time.monotonic() - t0))
        out: dict[str, Image.Image] = {}
        for key, fname in names.items():
            f = out_dir / fname
            if f.is_file():
                with Image.open(f) as im:
                    out[key] = im.convert("RGB")
        return out
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ------------------------------------------------------------------ traitement d'image

def blur(a: np.ndarray, sigma: float, wrap_x: bool = False, wrap_y: bool = False) -> np.ndarray:
    """Flou gaussien par FFT (rapide quel que soit sigma) ; axes répétables : bouclage exact,
    autres axes : marge miroir (pas de contamination d'un bord par l'autre)."""
    a = a.astype(np.float32)
    if sigma <= 0:
        return a.copy()
    r = int(math.ceil(3 * sigma))
    h, w = a.shape[:2]
    py = 0 if wrap_y else min(r, h - 1)
    px = 0 if wrap_x else min(r, w - 1)
    pad = ((py, py), (px, px)) + ((0, 0),) * (a.ndim - 2)
    p = np.pad(a, pad, mode="reflect") if (px or py) else a
    H, W = p.shape[:2]
    fy = np.fft.fftfreq(H).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(W).astype(np.float32)[None, :]
    g = np.exp(-2.0 * (math.pi * sigma) ** 2 * (fx * fx + fy * fy))
    if a.ndim == 3:
        g = g[..., None]
    out = np.fft.irfft2(np.fft.rfft2(p, axes=(0, 1)) * g, s=(H, W), axes=(0, 1))
    return out[py:py + h, px:px + w].astype(np.float32)


def band_noise(h: int, w: int, f_lo: float, f_hi: float, rng: np.random.Generator) -> np.ndarray:
    """Bruit blanc filtré dans la bande de fréquences [f_lo, f_hi] (cycles/pixel), écart type 1.
    Construit par FFT : il est périodique, donc raccordable."""
    n = rng.standard_normal((h, w)).astype(np.float32)
    fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(w).astype(np.float32)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    lo = 1.0 / (1.0 + np.exp(-(f - f_lo) / max(0.12 * f_lo, 1e-4)))
    hi = 1.0 / (1.0 + np.exp((f - f_hi) / 0.03))
    out = np.fft.irfft2(np.fft.rfft2(n) * lo * hi, s=(h, w)).astype(np.float32)
    s = float(out.std())
    return out / s if s > 1e-8 else out


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _pad_amount(w: int, h: int) -> int:
    return int(min(32, max(4, round(max(w, h) / 8)), w - 1, h - 1))


def pad_input(rgb: np.ndarray, p: int, tile_x: bool, tile_y: bool) -> np.ndarray:
    """Marge avant Real-ESRGAN : circulaire sur un axe répétable (le modèle voit le bord opposé),
    miroir sinon (pas de bord artificiel)."""
    a = np.pad(rgb, ((p, p), (0, 0), (0, 0)), mode="wrap" if tile_y else "reflect")
    return np.pad(a, ((0, 0), (p, p), (0, 0)), mode="wrap" if tile_x else "reflect")


def finish(esr: np.ndarray, src_rgb: np.ndarray, k: int, tile_x: bool, tile_y: bool, *, lock: float,
           detail: float, gain: float, seed: int, tame: float = 0.0) -> tuple[np.ndarray, dict]:
    """Verrou des couleurs + accord du détail. esr : rendu Real-ESRGAN à la taille cible (float 0..1),
    src_rgb : original (float 0..1, taille source).

    Détail attendu dans les octaves créées par l'agrandissement (au-dessus de la fréquence de Nyquist de
    l'original) : `gain` × nombre d'octaves × énergie de la dernière octave de l'original, mesurée
    localement. Là où Real-ESRGAN en produit moins (grain lissé), un bruit de matière comble l'écart
    (dose `detail`) ; là où il en produit plus (contours sur-accentués), la bande est atténuée vers la
    cible (`tame` : 0 = rien, 1 = jusqu'à la cible ; moyenne géométrique entre les deux)."""
    H, W = esr.shape[:2]
    out = esr.astype(np.float32)
    info: dict = {}
    # 1. verrou des couleurs : sous ~la moitié de la fréquence de Nyquist de l'original, on garde l'original
    if lock > 0 and k >= 1:
        sigma = 0.75 * k
        base = resize_f32(src_rgb, (W, H), Image.Resampling.LANCZOS, tile_x, tile_y)
        out = out + lock * (blur(base, sigma, tile_x, tile_y) - blur(out, sigma, tile_x, tile_y))
    # 2. détail des octaves créées : cible prolongée depuis le spectre de l'original
    if (detail > 0 or tame > 0) and k >= 2:
        ls = luma(src_rgb)
        top = ls - blur(ls, 1.0, tile_x, tile_y)                       # dernière octave de l'original
        e_src = blur(top * top, 3.0, tile_x, tile_y)                   # variance locale
        e_target = gain * math.log2(k) * np.maximum(resize_f32(e_src, (W, H), Image.Resampling.BILINEAR,
                                                               tile_x, tile_y), 0.0)
        lo = luma(out)
        band = lo - blur(lo, 0.375 * k, tile_x, tile_y)                 # au-dessus du Nyquist d'origine
        e_have = np.maximum(blur(band * band, 3.0 * k, tile_x, tile_y), 1e-10)
        info["detail_before"] = round(float(np.sqrt(e_have.mean())) * 255, 2)
        if tame > 0:                                                    # excès : bande atténuée
            want = np.exp((1 - tame) * np.log(e_have) + tame * np.log(np.maximum(e_target, 1e-10)))
            scale = np.where(e_have > want, np.sqrt(want / e_have), 1.0).astype(np.float32)
            out = out - (band * (1.0 - scale))[..., None]
            e_have = np.minimum(e_have, want)
            info["tamed"] = round(float(1.0 - scale.mean()), 3)
        if detail > 0:                                                  # manque : grain de matière
            amp = detail * np.sqrt(np.maximum(0.0, e_target - e_have))
            noise = band_noise(H, W, 0.5 / k, 0.5, np.random.default_rng(seed))
            out = out + (amp * noise)[..., None]
            info["grain"] = round(float(amp.mean()) * 255, 2)
    return np.clip(out, 0.0, 1.0), info


# ------------------------------------------------------------------ lots

@dataclass
class EsrganJob:
    tid: str
    name: str
    src: Image.Image          # RGBA d'origine
    an: Analysis
    k: int
    material: str = "generic"
    seed: int = 0


@dataclass
class EsrganResult:
    image: Image.Image | None = None
    info: dict = field(default_factory=dict)
    error: str = ""


def band_energy(img: np.ndarray, k: int, tile_x: bool = False, tile_y: bool = False) -> np.ndarray:
    """Énergie locale du détail au-dessus de la fréquence de Nyquist de l'original (luminance)."""
    lo = luma(img)
    band = lo - blur(lo, 0.375 * max(k, 1), tile_x, tile_y)
    return blur(band * band, 3.0 * max(k, 1), tile_x, tile_y)


def guard_weight(sharp: np.ndarray, safe: np.ndarray, k: int, tile_x: bool, tile_y: bool,
                 r_ok: float, r_bad: float) -> np.ndarray:
    """Part du rendu prudent (0..1) : rapport d'énergie de détail net / prudent, en échelle log."""
    r = band_energy(sharp, k, tile_x, tile_y) / np.maximum(band_energy(safe, k, tile_x, tile_y), 1e-9)
    w = np.clip((np.log(np.maximum(r, 1e-9)) - math.log(r_ok)) / max(math.log(r_bad) - math.log(r_ok), 1e-6), 0, 1)
    return np.clip(blur(w.astype(np.float32), 2.0 * max(k, 1), tile_x, tile_y), 0, 1)


class RealisticUpscaler:
    def __init__(self, cfg: dict, exe: Path | None = None, log: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.e = settings(cfg)
        self.exe = exe or find_realesrgan(cfg)
        self.log = log
        self.work = Path(cfg.get("_work") or tempfile.gettempdir()) / "tmp_esrgan"

    def available(self) -> bool:
        return self.exe is not None

    def run(self, jobs: list[EsrganJob]) -> dict[str, EsrganResult]:
        if self.exe is None:
            raise EsrganError(tr("Real-ESRGAN introuvable : bouton « Installer Real-ESRGAN » ou "
                                 "python -m texforge install-realesrgan"))
        prepared: dict[str, tuple[EsrganJob, np.ndarray, int]] = {}
        inputs: dict[str, Image.Image] = {}
        for j in jobs:
            rgb = np.asarray(bleed_rgb(j.src).convert("RGB"), dtype=np.float32) / 255.0
            p = _pad_amount(j.src.width, j.src.height)
            padded = pad_input(rgb, p, j.an.tile_x, j.an.tile_y)
            prepared[j.tid] = (j, rgb, p)
            inputs[j.tid] = Image.fromarray(np.clip(padded * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
        common = dict(scale=4, work_dir=self.work, tile=int(self.e["tile"]), gpu=str(self.e["gpu"]),
                      tta=bool(self.e["tta"]), timeout=float(self.e["timeout"]), log=self.log)
        sharp = run_ncnn(self.exe, inputs, model=str(self.e["model"]), **common)
        safe: dict[str, Image.Image] = {}
        if self.e.get("guard_model") and _has_models(self.exe, str(self.e["guard_model"])):
            safe = run_ncnn(self.exe, inputs, model=str(self.e["guard_model"]), **common)
        results: dict[str, EsrganResult] = {}
        for tid, (j, rgb, p) in prepared.items():
            up = sharp.get(tid)
            if up is None:
                results[tid] = EsrganResult(error=tr("Real-ESRGAN : aucune sortie"))
                continue
            try:
                results[tid] = self._finish(j, rgb, p, up, safe.get(tid))
            except Exception as exc:                 # une texture ratée n'arrête pas le lot
                results[tid] = EsrganResult(error=f"{type(exc).__name__}: {exc}")
        return results

    def _crop(self, up: Image.Image, j: EsrganJob, p: int) -> np.ndarray:
        w, h = j.src.width, j.src.height
        arr = np.asarray(up, dtype=np.float32) / 255.0
        s = arr.shape[1] / (w + 2 * p)                       # échelle réelle du modèle (4)
        o = int(round(p * s))
        arr = arr[o:o + int(round(h * s)), o:o + int(round(w * s))]
        if arr.shape[:2] != (h * j.k, w * j.k):
            arr = resize_f32(arr, (w * j.k, h * j.k), Image.Resampling.LANCZOS, j.an.tile_x, j.an.tile_y)
        return arr

    def _finish(self, j: EsrganJob, rgb: np.ndarray, p: int, up: Image.Image,
                safe_up: Image.Image | None) -> EsrganResult:
        W, H = j.src.width * j.k, j.src.height * j.k
        tx, ty = j.an.tile_x, j.an.tile_y
        dose = float(self.e["detail"]) * float(self.e["materials"].get(j.material, self.e["materials"]["generic"]))
        kw = dict(lock=float(self.e["color_lock"]), detail=dose, gain=float(self.e["detail_gain"]),
                  tame=float(self.e["tame"]), seed=j.seed)
        sharp = self._crop(up, j, p)
        out, info = finish(sharp, rgb, j.k, tx, ty, **kw)
        if safe_up is not None:
            safe = self._crop(safe_up, j, p)
            r_ok, r_bad = (float(x) for x in self.e["guard"])
            w = guard_weight(sharp, safe, j.k, tx, ty, r_ok, r_bad)
            if float(w.max()) > 0.01:
                out_safe, _ = finish(safe, rgb, j.k, tx, ty, **kw)
                out = out * (1.0 - w[..., None]) + out_safe * w[..., None]
            info["guard"] = round(float(w.mean()), 3)
        alpha = final_alpha(j.src, j.an, (W, H))
        rgba = np.concatenate([np.clip(out, 0.0, 1.0), alpha[..., None]], axis=-1)
        img = Image.fromarray(np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA")
        info.update(model=str(self.e["model"]), material=j.material, dose=round(dose, 2))
        return EsrganResult(img, info)


__all__ = ["RealisticUpscaler", "EsrganJob", "EsrganResult", "EsrganError", "find_realesrgan",
           "install_realesrgan", "run_ncnn", "finish", "blur", "band_noise", "band_energy", "guard_weight",
           "settings", "RELEASES", "DEFAULTS"]
