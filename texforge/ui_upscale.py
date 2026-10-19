"""Agrandissement local rapide, sans IA générative.

Moteurs, essayés dans l'ordre de la configuration [ui_upscale] engines :
  realesrgan  realesrgan-ncnn-vulkan.exe (modèle léger realesr-animevideov3 par défaut)
  waifu2x     waifu2x-ncnn-vulkan.exe
  nearest     pixels doublés (pixel art, polices bitmap nettes)
  lanczos     Lanczos + légère accentuation, en Python : toujours disponible, dernier recours

Les exécutables ncnn traitent un DOSSIER entier en un seul lancement (le modèle n'est
chargé qu'une fois) : un lot de 200 icônes prend quelques secondes. La transparence est
agrandie à part, comme une image en niveaux de gris passée dans le même moteur, ce qui
garde des contours de glyphes nets.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from .imaging import LANCZOS, Analysis, bleed_rgb, local_upscale
from .validate import evaluate

ENGINES = ("realesrgan", "waifu2x", "nearest", "lanczos")
# Échelles acceptées par chaque modèle ncnn
REALESRGAN_SCALES = {"realesr-animevideov3": (2, 3, 4), "realesrgan-x4plus": (4,), "realesrgan-x4plus-anime": (4,)}
WAIFU2X_SCALES = (1, 2, 4, 8, 16, 32)


class EngineError(Exception):
    pass


@dataclass
class UiJob:
    tid: str
    name: str
    src: Image.Image          # RGBA d'origine
    k: int                    # facteur d'agrandissement voulu
    an: Analysis
    engine: str = ""          # moteur imposé par une règle (sinon la chaîne configurée)

    @property
    def target(self) -> tuple[int, int]:
        return self.src.width * self.k, self.src.height * self.k


@dataclass
class UiResult:
    image: Image.Image | None = None
    engine: str = ""
    errors: list[str] = field(default_factory=list)


def _pick_scale(allowed: tuple[int, ...], k: int) -> int:
    for s in sorted(allowed):
        if s >= k:
            return s
    return max(allowed)


class UiUpscaler:
    def __init__(self, cfg: dict, log: Callable[[str], None] | None = None, *, model: str | None = None,
                 engines: list[str] | None = None) -> None:
        self.cfg = cfg
        self.u = cfg["ui_upscale"]
        self.log = log or (lambda _m: None)
        self.model = model or str(self.u.get("realesrgan_model", "realesr-animevideov3"))
        chain = engines or list(self.u.get("engines", ["realesrgan", "waifu2x", "lanczos"]))
        self.chain = [e for e in chain if e in ENGINES]
        if "lanczos" not in self.chain:
            self.chain.append("lanczos")      # filet de sécurité
        self.tmp_root = Path(cfg["_work"]) / "tmp_upscale"

    # ------------------------------------------------------------ disponibilité
    def _exe(self, engine: str) -> Path | None:
        key = {"realesrgan": "realesrgan_exe", "waifu2x": "waifu2x_exe"}.get(engine)
        if not key:
            return None
        raw = str(self.u.get(key, "") or "")
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = Path(self.cfg["_base"]) / p
        return p if p.exists() else None

    def available(self) -> list[str]:
        return [e for e in self.chain if e in ("nearest", "lanczos") or self._exe(e) is not None]

    # ------------------------------------------------------------ traitement
    def run(self, jobs: list[UiJob], validate: bool = True) -> dict[str, UiResult]:
        """Traite un lot ; chaque texture descend la chaîne de moteurs jusqu'à réussir."""
        results = {j.tid: UiResult() for j in jobs}
        pending = list(jobs)

        def chain_for(j: UiJob) -> list[str]:
            return [j.engine, "lanczos"] if j.engine in ENGINES else self.chain

        order: list[str] = []
        for j in jobs:
            for e in chain_for(j):
                if e not in order:
                    order.append(e)
        order.sort(key=lambda e: e == "lanczos")      # lanczos toujours en dernier
        for engine in order:
            if not pending:
                break
            todo = [j for j in pending if engine in chain_for(j)]
            if not todo:
                continue
            if engine in ("realesrgan", "waifu2x") and self._exe(engine) is None:
                for j in todo:
                    results[j.tid].errors.append(f"{engine} : exécutable non configuré")
                continue
            try:
                produced = self._run_engine(engine, todo)
            except EngineError as exc:
                self.log(f"[UI] moteur {engine} en échec pour le lot : {exc}")
                for j in todo:
                    results[j.tid].errors.append(f"{engine} : {exc}")
                continue
            still: list[UiJob] = []
            for j in pending:
                img = produced.get(j.tid)
                if img is None:
                    if j in todo:
                        results[j.tid].errors.append(f"{engine} : aucune sortie")
                    still.append(j)
                    continue
                if validate and engine != "lanczos":
                    # un graphisme d'interface ne se répète pas : pas de contrôle de raccord
                    m = evaluate(j.src, img, dc_replace(j.an, tile_x=False, tile_y=False), self.cfg)
                    if not m["passed"]:
                        results[j.tid].errors.append(f"{engine} : rejeté ({'; '.join(m['reasons'])})")
                        still.append(j)
                        continue
                results[j.tid].image = img
                results[j.tid].engine = engine
            pending = still
        return results

    def upscale_one(self, src: Image.Image, an: Analysis, k: int, name: str = "texture") -> tuple[Image.Image, str]:
        """Une seule texture (repli d'une génération ComfyUI ratée)."""
        job = UiJob(tid="one", name=name, src=src.convert("RGBA"), k=k, an=an)
        res = self.run([job], validate=False)["one"]
        if res.image is None:
            raise EngineError("; ".join(res.errors))
        return res.image, res.engine

    def _run_engine(self, engine: str, jobs: list[UiJob]) -> dict[str, Image.Image]:
        if engine == "lanczos":
            return {j.tid: self._lanczos(j) for j in jobs}
        if engine == "nearest":
            return {j.tid: j.src.resize(j.target, Image.Resampling.NEAREST) for j in jobs}
        return self._run_ncnn(engine, jobs)

    def _lanczos(self, j: UiJob) -> Image.Image:
        out = local_upscale(j.src, j.k, j.an, self.cfg, use_esrgan=False)
        rgb = out.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=1))
        rgba = rgb.convert("RGBA")
        rgba.putalpha(out.getchannel("A"))
        return rgba

    def _run_ncnn(self, engine: str, jobs: list[UiJob]) -> dict[str, Image.Image]:
        exe = self._exe(engine)
        assert exe is not None
        kmax = max(j.k for j in jobs)
        if engine == "realesrgan":
            scale = _pick_scale(REALESRGAN_SCALES.get(self.model, (4,)), kmax)
        else:
            scale = _pick_scale(WAIFU2X_SCALES, kmax)
        run_dir = self.tmp_root / f"{engine}_{uuid.uuid4().hex[:8]}"
        in_dir, out_dir = run_dir / "in", run_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        index: dict[str, UiJob] = {}
        try:
            for i, j in enumerate(jobs):
                key = f"{i:05d}"
                index[key] = j
                bleed_rgb(j.src).convert("RGB").save(in_dir / f"{key}.png")
                if j.an.alpha_class != "none":
                    j.src.getchannel("A").convert("RGB").save(in_dir / f"{key}_a.png")
            cmd = [str(exe), "-i", str(in_dir), "-o", str(out_dir), "-s", str(scale), "-f", "png",
                   "-t", str(int(self.u.get("tile", 256)))]
            if engine == "realesrgan":
                cmd += ["-n", self.model]
            else:
                cmd += ["-n", str(int(self.u.get("waifu2x_noise", 1)))]
                if self.u.get("waifu2x_model"):
                    cmd += ["-m", str(self.u["waifu2x_model"])]
            if str(self.u.get("gpu", "")) != "":
                cmd += ["-g", str(self.u["gpu"])]
            timeout = float(self.u.get("timeout", 120)) + 2.0 * len(jobs)
            t0 = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                                      cwd=str(exe.parent))   # les modèles ncnn sont cherchés à côté de l'exe
            except subprocess.TimeoutExpired as exc:
                raise EngineError(f"délai de {timeout:.0f} s dépassé") from exc
            except OSError as exc:
                raise EngineError(f"lancement impossible : {exc}") from exc
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
                raise EngineError(f"code {proc.returncode} : {' | '.join(tail)}")
            self.log(f"[UI] {engine} ({self.model if engine == 'realesrgan' else 'waifu2x'}, ×{scale}) : "
                     f"{len(jobs)} image(s) en {time.monotonic() - t0:.1f} s")
            produced: dict[str, Image.Image] = {}
            for key, j in index.items():
                f = out_dir / f"{key}.png"
                if not f.exists():
                    continue
                with Image.open(f) as im:
                    rgb = im.convert("RGB")
                if rgb.size != j.target:
                    rgb = rgb.resize(j.target, LANCZOS)
                produced[j.tid] = self._with_alpha(rgb, j, out_dir / f"{key}_a.png")
            return produced
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    @staticmethod
    def _with_alpha(rgb: Image.Image, j: UiJob, alpha_file: Path) -> Image.Image:
        out = rgb.convert("RGBA")
        if j.an.alpha_class == "none":
            return out
        if alpha_file.exists():
            with Image.open(alpha_file) as im:
                alpha = im.convert("L")
            if alpha.size != j.target:
                alpha = alpha.resize(j.target, LANCZOS)
        else:
            alpha = j.src.getchannel("A").resize(j.target, LANCZOS)
        if j.an.alpha_class == "binary":
            alpha = Image.fromarray(np.where(np.asarray(alpha) >= 128, 255, 0).astype(np.uint8), "L")
        out.putalpha(alpha)
        return out
