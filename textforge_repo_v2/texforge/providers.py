"""Fournisseurs de génération : ComfyUI (local, par défaut), Mock (gratuit, pour tester
toute la chaîne) et Higgsfield (cloud payant, optionnel : higgsfield-client n'est importé
que si ce fournisseur est choisi)."""

from __future__ import annotations

import copy

import hashlib
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from .imaging import LANCZOS, parse_ratio


class ProviderError(Exception):
    """Échec définitif pour cette requête (refus, contenu bloqué, paramètres invalides)."""


class TransientError(Exception):
    """Échec temporaire (réseau, délai dépassé) : la requête peut être reprise."""


class FatalError(Exception):
    """Inutile de continuer (clé invalide, crédits épuisés)."""


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockProvider:
    """Imite un modèle d'édition : renforce le détail, ajoute un grain de matière.
    Aucun réseau, aucun coût. Option mock.shift pour simuler un décalage (tests)."""

    name = "mock"
    paid = False

    def __init__(self, cfg: dict, state=None):
        self.cfg = cfg
        self.ratios = cfg["higgsfield"]["aspect_ratios"]
        self.shift = float(cfg.get("mock", {}).get("shift", 0.0))

    def generate(self, image: Image.Image, refs: list[Path], prompt: str, negative: str,
                 aspect_ratio: str, seed: int, on_submitted: Callable[[str], None]) -> tuple[Image.Image, str]:
        rid = f"mock-{uuid.uuid4().hex[:12]}"
        on_submitted(rid)
        rng = np.random.default_rng(seed)
        t = parse_ratio(aspect_ratio)
        # Dimensions de sortie différentes de l'entrée, comme un vrai modèle
        out_w = 1184 if t >= 1 else round(1184 * t)
        out_h = round(1184 / t) if t >= 1 else 1184
        img = image.convert("RGB").resize((out_w, out_h), LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=120, threshold=2))
        arr = np.asarray(img, dtype=np.float32)
        grain = rng.normal(0, 6, arr.shape[:2])[..., None]
        arr = arr + grain + np.array([4.0, 1.0, -3.0])
        if self.shift:
            arr = np.roll(arr, (round(out_h * self.shift), round(out_w * self.shift)), axis=(0, 1))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), f"mock://{rid}"

    def resume(self, request_id: str):
        raise ProviderError("une requête mock ne peut pas être reprise")


class HiggsfieldProvider:
    name = "higgsfield"
    paid = True

    def __init__(self, cfg: dict, state):
        try:
            import higgsfield_client
            import httpx
        except ImportError as exc:
            raise FatalError("le paquet higgsfield-client n'est pas installé : pip install -r requirements.txt") from exc
        self._hf = higgsfield_client
        self._httpx = httpx
        self.cfg = cfg
        self.hcfg = cfg["higgsfield"]
        self.pcfg = cfg["provider"]
        self.state = state
        self.ratios = self.hcfg["aspect_ratios"]
        key = (self.hcfg.get("api_key") or "").strip()
        if not key:
            key = os.getenv("HF_KEY") or ""
            if not key and os.getenv("HF_API_KEY") and os.getenv("HF_API_SECRET"):
                key = f"{os.getenv('HF_API_KEY')}:{os.getenv('HF_API_SECRET')}"
        if not key or ":" not in key:
            raise FatalError(
                "clé Higgsfield absente. Crée une clé sur console.higgsfield.ai puis définis la variable "
                "d'environnement HF_KEY=\"id:secret\" (ou HF_API_KEY et HF_API_SECRET).")
        self.client = higgsfield_client.SyncClient(api_key=key, timeout=120.0)
        self.model = self.hcfg["model"]

    # ------------------------------------------------------------ erreurs
    def _classify(self, exc: Exception) -> Exception:
        cause = exc.__cause__
        code = getattr(getattr(cause, "response", None), "status_code", None)
        msg = str(exc) or exc.__class__.__name__
        if code in (401, 403):
            return FatalError(f"accès refusé par Higgsfield ({code}) : {msg}")
        if code == 402 or "credit" in msg.lower() or "balance" in msg.lower():
            return FatalError(f"crédits insuffisants ({code}) : {msg}")
        if code in (400, 404, 422):
            return ProviderError(f"requête refusée ({code}) : {msg}")
        return TransientError(f"erreur Higgsfield ({code}) : {msg}")

    # ------------------------------------------------------------ envois
    def _upload_bytes(self, data: bytes, content_type: str) -> str:
        sha = hashlib.sha256(data).hexdigest()
        cached = self.state.get_upload(sha) if self.state else None
        if cached:
            return cached
        try:
            url = self.client.upload(data, content_type)
        except self._hf.HiggsfieldClientError as exc:
            raise self._classify(exc) from exc
        except self._httpx.TransportError as exc:
            raise TransientError(f"envoi impossible : {exc}") from exc
        if self.state:
            self.state.put_upload(sha, url)
        return url

    def _ref_urls(self, refs: list[Path]) -> list[str]:
        urls = []
        for p in refs[: max(0, int(self.hcfg.get("max_images", 3)) - 1)]:
            img = Image.open(p)
            img.load()
            urls.append(self._upload_bytes(_png_bytes(img.convert("RGB")), "image/png"))
        return urls

    def build_args(self, prompt: str, negative: str, aspect_ratio: str, seed: int, urls: list[str]) -> dict:
        args = dict(self.hcfg.get("extra_args", {}))
        args["prompt"] = prompt
        args[self.hcfg.get("image_field", "image_urls")] = urls
        if aspect_ratio in self.ratios:
            args["aspect_ratio"] = aspect_ratio
        if self.hcfg.get("supports_seed", True):
            args["seed"] = int(seed) % 2147483647
        if self.hcfg.get("supports_negative", True) and negative:
            args["negative_prompt"] = negative
        return args

    # ------------------------------------------------------------ génération
    def generate(self, image: Image.Image, refs: list[Path], prompt: str, negative: str,
                 aspect_ratio: str, seed: int, on_submitted: Callable[[str], None]) -> tuple[Image.Image, str]:
        input_url = self._upload_bytes(_png_bytes(image.convert("RGB")), "image/png")
        urls = [input_url] + self._ref_urls(refs)
        args = self.build_args(prompt, negative, aspect_ratio, seed, urls)
        try:
            controller = self.client.submit(self.model, args)
        except self._hf.HiggsfieldClientError as exc:
            raise self._classify(exc) from exc
        except self._httpx.TransportError as exc:
            raise TransientError(f"soumission impossible : {exc}") from exc
        on_submitted(controller.request_id)
        return self._wait(controller)

    def resume(self, request_id: str) -> tuple[Image.Image, str]:
        controller = self.client.get_request_controller(request_id)
        return self._wait(controller)

    def _wait(self, controller) -> tuple[Image.Image, str]:
        hf = self._hf
        deadline = time.time() + float(self.pcfg.get("timeout", 900))
        interval = max(1.0, float(self.pcfg.get("poll_interval", 3.0)))
        net_errors = 0
        while True:
            try:
                st = controller.status()
                net_errors = 0
            except hf.HiggsfieldClientError as exc:
                raise self._classify(exc) from exc
            except self._httpx.TransportError as exc:
                net_errors += 1
                if net_errors >= 5:
                    raise TransientError(f"réseau indisponible pendant l'attente : {exc}") from exc
                time.sleep(interval * net_errors)
                continue
            if isinstance(st, hf.Completed):
                break
            if isinstance(st, hf.NSFW):
                raise ProviderError("résultat bloqué par le filtre de contenu (NSFW)")
            if isinstance(st, (hf.Failed, hf.Cancelled)):
                raise ProviderError(f"génération en échec ({st.__class__.__name__})")
            if time.time() > deadline:
                raise TransientError("délai dépassé ; la requête sera reprise au prochain lancement")
            time.sleep(interval)
        try:
            data = controller.get()
        except hf.HiggsfieldClientError as exc:
            raise self._classify(exc) from exc
        except self._httpx.TransportError as exc:
            raise TransientError(f"lecture du résultat impossible : {exc}") from exc
        images = data.get("images") or []
        if not images or not images[0].get("url"):
            raise ProviderError(f"réponse sans image : {str(data)[:200]}")
        url = images[0]["url"]
        try:
            r = self._httpx.get(url, timeout=180.0, follow_redirects=True)
            r.raise_for_status()
        except Exception as exc:
            raise TransientError(f"téléchargement du résultat impossible : {exc}") from exc
        img = Image.open(io.BytesIO(r.content))
        img.load()
        return img.convert("RGB"), url


class ServerDown(FatalError):
    """ComfyUI injoignable : l'orchestrateur attend son retour, puis applique le repli."""


class WorkflowFatal(FatalError):
    """Workflow refusé par ComfyUI (nœud ou modèle absent) : inutile d'insister."""


def resolve_workflow_path(cfg: dict, raw: str) -> Path:
    """Chemin relatif : cherché à côté de la configuration, puis dans le dossier de TEXFORGE."""
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (Path(cfg["_base"]), Path(__file__).resolve().parent.parent):
        if (base / p).exists():
            return base / p
    return Path(cfg["_base"]) / p


class ComfyUIProvider:
    """Génération locale par ComfyUI, via comfyui_headless_client (asynchrone).

    Le client tourne dans une boucle asyncio dédiée (thread de fond) ; les threads de
    TEXFORGE lui soumettent leurs requêtes. Chaque texture reçoit ses propres surcharges :
    prompt du matériau, denoise, force du ControlNet, mode de raccord, graine.
    """

    name = "comfyui"
    paid = False
    native_tiling = True          # le raccord est géré dans le workflow (padding circulaire)
    ratios = ["1:1"]

    def __init__(self, cfg: dict, state=None, on_progress: Callable[[str], None] | None = None):
        import asyncio
        import threading
        from . import comfyui_headless_client as chc
        self.cfg = cfg
        self.c = cfg["comfyui"]
        self.chc = chc
        self._asyncio = asyncio
        self._warned: set[str] = set()
        try:
            self.wf_main = customize_workflow(chc.ComfyWorkflow.from_file(
                resolve_workflow_path(cfg, self.c["workflow"])), self.c, chc)
            edges_raw = str(self.c.get("workflow_edges") or "")
            self.wf_edges = (customize_workflow(chc.ComfyWorkflow.from_file(
                resolve_workflow_path(cfg, edges_raw)), self.c, chc) if edges_raw else None)
        except chc.WorkflowError as exc:
            raise WorkflowFatal(f"workflow ComfyUI invalide : {exc}") from exc
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="comfyui-loop", daemon=True)
        self.thread.start()
        self.backend = chc.ComfyUIBackend(str(self.c["server"]), self.wf_main, timeout=float(self.c["timeout"]),
                                          max_in_flight=int(self.c.get("max_in_flight", 2)))
        self.backend.on_progress = on_progress
        self._call(self.backend.start())
        try:
            self.info = self.check_ready()
        except BaseException:
            self.close()          # sinon la boucle et le WebSocket de cet essai restent ouverts
            raise

    # ------------------------------------------------------------ outils
    def _call(self, coro, timeout: float | None = None):
        fut = self._asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout)
        except BaseException:
            fut.cancel()
            raise

    def check_ready(self) -> str:
        try:
            return self._call(self.backend.check_ready(), timeout=30)
        except (self.chc.ServerUnavailable, self.chc.TextureError) as exc:
            raise ServerDown(str(exc)) from exc

    def vram_free(self) -> int | None:
        try:
            return self._call(self.backend.vram_free_bytes(), timeout=30)
        except Exception:
            return None

    def free_vram(self) -> None:
        try:
            self._call(self.backend.free_vram(), timeout=30)
        except self.chc.ServerUnavailable as exc:
            raise ServerDown(str(exc)) from exc

    def close(self) -> None:
        try:
            self._call(self.backend.close(), timeout=15)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    def cancel_pending(self) -> None:
        try:
            self._call(self.backend.cancel_pending(), timeout=10)
        except Exception:
            pass

    # ------------------------------------------------------------ contexte par texture
    def _has(self, wf, title: str) -> bool:
        try:
            wf.resolve_node(title)
            return True
        except self.chc.WorkflowError:
            if title not in self._warned:
                self._warned.add(title)
                print(f"  ! nœud {title} absent du workflow : réglage ignoré", file=sys.stderr)
            return False

    def build_context(self, rec: dict, an) -> dict:
        """Matériau, raccord et variante de workflow d'une texture."""
        material = (rec.get("analysis") or {}).get("material") or "generic"
        preset = {**self.cfg["materials"].get("generic", {}), **self.cfg["materials"].get(material, {})}
        tiling = ("enable" if an.tile_x and an.tile_y else "x_only" if an.tile_x
                  else "y_only" if an.tile_y else "disable")
        hint = (rec.get("hint") or "").strip()
        body = ", ".join(x for x in (hint, str(preset.get("prompt", ""))) if x)
        prefix = self.c["prompt_prefix_tiled"] if tiling != "disable" else self.c["prompt_prefix"]
        edges = bool(preset.get("edges")) and self.wf_edges is not None
        return {"material": material, "tiling": tiling, "edges": edges,
                "prompt": f"{prefix}{body}{self.c['prompt_suffix']}", "negative": str(self.c["negative"]),
                "denoise": float(preset.get("denoise", 0.42)), "cn_strength": float(preset.get("cn_strength", 0.75)),
                "stem": rec.get("name") or "texture"}

    # ------------------------------------------------------------ génération
    def generate(self, image: Image.Image, refs: list[Path], prompt: str, negative: str,
                 aspect_ratio: str, seed: int, on_submitted: Callable[[str], None],
                 context: dict | None = None) -> tuple[Image.Image, str]:
        ctx = context or {}
        wf = self.wf_edges if ctx.get("edges") and self.wf_edges is not None else self.wf_main
        wanted = {
            ("TEXFORGE_PROMPT", "text"): ctx.get("prompt", prompt),
            ("TEXFORGE_NEGATIVE", "text"): ctx.get("negative", negative),
            ("TEXFORGE_SAMPLER", "seed"): int(seed),
            ("TEXFORGE_SAMPLER", "denoise"): ctx.get("denoise"),
            ("TEXFORGE_CN_TILE", "strength"): ctx.get("cn_strength"),
            ("TEXFORGE_TILING", "tiling"): ctx.get("tiling"),
            ("TEXFORGE_TILING_VAE", "tiling"): ctx.get("tiling"),
        }
        overrides = {k: v for k, v in wanted.items() if v is not None and self._has(wf, k[0])}
        on_submitted(f"local-{uuid.uuid4().hex[:12]}")
        try:
            blobs = self._call(self.backend.generate(image.convert("RGB"), str(ctx.get("stem", "texture")),
                                                     overrides=overrides, workflow=wf),
                               timeout=float(self.c["timeout"]) + 120)
        except self.chc.ServerUnavailable as exc:
            raise ServerDown(str(exc)) from exc
        except self.chc.WorkflowError as exc:
            raise WorkflowFatal(str(exc)) from exc
        except self.chc.TextureError as exc:
            if exc.kind in ("oom", "timeout", "interrupted", "http"):
                raise TransientError(f"[{exc.kind}] {exc}") from exc
            raise ProviderError(f"[{exc.kind}] {exc}") from exc
        except TimeoutError as exc:
            raise TransientError(f"[timeout] pas de réponse de la boucle ComfyUI : {exc}") from exc
        img = Image.open(io.BytesIO(blobs[0]))
        img.load()
        return img.convert("RGB"), f"comfyui://{ctx.get('stem', 'texture')}"

    def resume(self, request_id: str):
        raise ProviderError("les requêtes locales ne se reprennent pas")


def customize_workflow(wf, c: dict, chc):
    """Applique au graphe les choix faits dans l'interface ou la configuration :
    checkpoint (nœud TEXFORGE_CHECKPOINT), modèle ControlNet (TEXFORGE_CN_TILE_MODEL) et LoRA.
    La LoRA est insérée entre le checkpoint et tout ce qui consommait son MODEL et son CLIP
    (raccord circulaire compris) ; un nœud TEXFORGE_LORA déjà présent est simplement réglé."""
    ckpt, cn, lora = (str(c.get(k) or "").strip() for k in ("checkpoint", "controlnet_model", "lora"))
    if not (ckpt or cn or lora):
        return wf
    graph = copy.deepcopy(wf.graph)

    def by_title(title: str, class_types: tuple[str, ...]) -> str | None:
        hits = [n for n, node in graph.items() if str((node.get("_meta") or {}).get("title", "")).lower() == title.lower()]
        if not hits:
            hits = [n for n, node in graph.items() if node.get("class_type") in class_types]
        return hits[0] if len(hits) == 1 else None

    ck = by_title("TEXFORGE_CHECKPOINT", ("CheckpointLoaderSimple", "CheckpointLoader"))
    if ckpt:
        if ck is None:
            raise chc.WorkflowError("checkpoint choisi mais aucun nœud TEXFORGE_CHECKPOINT unique dans le workflow")
        graph[ck]["inputs"]["ckpt_name"] = ckpt
    if cn:
        cid = by_title("TEXFORGE_CN_TILE_MODEL", ("ControlNetLoader",))
        if cid is None:
            raise chc.WorkflowError("modèle ControlNet choisi mais aucun nœud TEXFORGE_CN_TILE_MODEL unique")
        graph[cid]["inputs"]["control_net_name"] = cn
    if lora:
        strength = float(c.get("lora_strength", 0.8))
        existing = by_title("TEXFORGE_LORA", ())
        if existing is not None:
            graph[existing]["inputs"].update(lora_name=lora, strength_model=strength, strength_clip=strength)
        else:
            if ck is None:
                raise chc.WorkflowError("LoRA choisie mais aucun nœud TEXFORGE_CHECKPOINT unique pour l'insérer")
            numeric = [int(n) for n in graph if str(n).isdigit()]
            lid = str(max(numeric, default=0) + 1)
            for node in graph.values():
                for key, val in node["inputs"].items():
                    if isinstance(val, list) and len(val) == 2 and str(val[0]) == ck and val[1] in (0, 1):
                        node["inputs"][key] = [lid, val[1]]
            graph[lid] = {"class_type": "LoraLoader", "_meta": {"title": "TEXFORGE_LORA"},
                          "inputs": {"model": [ck, 0], "clip": [ck, 1], "lora_name": lora,
                                     "strength_model": strength, "strength_clip": strength}}
    return chc.ComfyWorkflow(graph, wf.load_node_id, list(wf.output_node_ids), wf.preview_only)


def make_provider(cfg: dict, state, name: str | None = None, **kwargs):
    name = (name or cfg["provider"]["name"]).lower()
    if name == "mock":
        return MockProvider(cfg, state)
    if name == "comfyui":
        return ComfyUIProvider(cfg, state, **kwargs)
    if name == "higgsfield":
        return HiggsfieldProvider(cfg, state)
    raise FatalError(f"fournisseur inconnu : {name} (attendu : comfyui, mock ou higgsfield)")
