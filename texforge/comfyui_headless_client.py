#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comfyui_headless_client.py — traitement par lot de textures pour TEXFORGE.

Deux moteurs, même interface :
  * ComfyUI (défaut)          http://127.0.0.1:8188  workflow au format API, suivi par WebSocket
  * Stable Diffusion WebUI    http://127.0.0.1:7860  API AUTOMATIC1111 / Forge (/sdapi/v1/img2img)

Exemples :
  python comfyui_headless_client.py entree/ sortie/ --workflow texforge_api.json
  python comfyui_headless_client.py entree/ sortie/ --workflow texforge_api.json --set 6.text="mossy stone"
  python comfyui_headless_client.py --example-sd-params > sd_params.json
  python comfyui_headless_client.py entree/ sortie/ --backend sd --workflow sd_params.json --sd-tiling

Utilisation depuis Python :
  from comfyui_headless_client import process_texture_batch
  report = process_texture_batch("entree", "sortie", "texforge_api.json")

Dépendances : Python 3.10+ (3.11 recommandé) — pip install aiohttp tqdm pillow
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import base64
import contextlib
import copy
import hashlib
import io
import json
import logging
import random
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import aiohttp
from PIL import Image
from tqdm import tqdm

# --------------------------------------------------------------------------- constantes

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_SD_URL = "http://127.0.0.1:7860"
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds", ".webp", ".tif", ".tiff"})
UPLOAD_SUBFOLDER = "texforge"          # sous-dossier de ComfyUI/input et ComfyUI/output
INPUT_TITLE_TAG = "TEXFORGE_INPUT"     # titre à donner au LoadImage si le workflow en contient plusieurs
OUTPUT_TITLE_TAG = "TEXFORGE_OUTPUT"   # si présent, seul ce SaveImage est récupéré
ERROR_LOG_NAME = "comfy_errors.log"

LOAD_IMAGE_TYPES = ("LoadImage",)
SAVE_IMAGE_TYPES = ("SaveImage",)
PREVIEW_IMAGE_TYPES = ("PreviewImage",)

BackendName = Literal["comfyui", "sd"]
FreePolicy = Literal["auto", "always", "never"]
ProgressCallback = Callable[[str], None]
Overrides = dict[tuple[str, str], Any]   # (id du nœud, nom de l'entrée) -> valeur

log = logging.getLogger("texforge.headless")

# Paramètres img2img d'exemple pour la WebUI (affichés par --example-sd-params)
SD_EXAMPLE_PARAMS: dict[str, Any] = {
    "prompt": "photorealistic high resolution texture, detailed physically based surface, "
              "flat orthographic view, same layout",
    "negative_prompt": "perspective, frame, border, text, watermark, blurry, cartoon, new objects",
    "denoising_strength": 0.35,
    "steps": 25,
    "cfg_scale": 6.0,
    "sampler_name": "DPM++ 2M",
    "scheduler": "Karras",
    "seed": -1,
    "batch_size": 1,
    "n_iter": 1,
    "override_settings": {},
    "override_settings_restore_afterwards": True,
}


# --------------------------------------------------------------------------- erreurs

class TextureError(Exception):
    """Échec limité à une texture : consigné dans le journal, le lot continue."""

    def __init__(self, message: str, *, kind: str = "error", details: str = "") -> None:
        super().__init__(message)
        # oom | cuda | workflow | timeout | image | http | interrupted | error
        self.kind = kind
        self.details = details   # trace complète (traceback ComfyUI, corps HTTP…)

    @property
    def is_oom(self) -> bool:
        return self.kind == "oom"


class WorkflowError(TextureError):
    """Workflow ou paramètres refusés (nœud manquant, modèle absent, JSON invalide…)."""

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, kind="workflow", details=details)


class ServerUnavailable(Exception):
    """Serveur injoignable ou connexion perdue."""


def classify_failure(text: str) -> str:
    """Range un message d'erreur serveur dans une catégorie exploitable."""
    t = text.lower()
    if "out of memory" in t or "outofmemory" in t or "allocation on device" in t:
        return "oom"
    if "cuda" in t or "cudnn" in t or "cublas" in t:
        return "cuda"
    return "error"


def parse_json(body: bytes, what: str) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TextureError(f"réponse JSON invalide pour {what} : {body[:200]!r}", kind="http") from exc


def _gb(n: Any) -> str:
    return f"{n / 1024 ** 3:.1f}" if isinstance(n, (int, float)) else "?"


# --------------------------------------------------------------------------- images

def load_source_image(path: Path) -> Image.Image:
    """Ouvre une texture (PNG, DDS DXT1/3/5, TGA, BMP…) en RGB ou RGBA."""
    try:
        with Image.open(path) as im:
            im.load()
            alpha = "A" in im.getbands() or im.info.get("transparency") is not None
            return im.convert("RGBA" if alpha else "RGB")
    except Exception as exc:  # fichier corrompu ou format non géré par Pillow
        raise TextureError(f"image illisible : {exc}", kind="image") from exc


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def has_alpha(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    lowest = img.getchannel("A").getextrema()[0]
    return isinstance(lowest, (int, float)) and lowest < 255


def save_outputs(blobs: list[bytes], source: Image.Image, dest: Path, keep_alpha: bool) -> list[Path]:
    """Décode les images générées, recolle l'alpha d'origine (les modèles ne produisent
    que du RGB), écrit en PNG de façon atomique. Appelée dans un thread."""
    paths: list[Path] = []
    for i, blob in enumerate(blobs):
        try:
            with Image.open(io.BytesIO(blob)) as im:
                im.load()
                out = im.convert("RGB")
        except Exception as exc:
            raise TextureError(f"image renvoyée illisible : {exc}", kind="http") from exc
        if keep_alpha and has_alpha(source):
            out = out.convert("RGBA")
            out.putalpha(source.getchannel("A").resize(out.size, Image.Resampling.LANCZOS))
        target = dest if i == 0 else dest.with_name(f"{dest.stem}_{i + 1}{dest.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        out.save(tmp, format="PNG")
        tmp.replace(target)
        paths.append(target)
    return paths


def safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80] or "texture"


def fit_size(size: tuple[int, int], long_side: int, multiple: int = 64) -> tuple[int, int]:
    """Taille de génération : côté long = long_side, côtés multiples de 64."""
    w, h = size
    scale = long_side / max(w, h)

    def snap(v: float) -> int:
        return max(multiple, int(round(v * scale / multiple)) * multiple)

    return snap(w), snap(h)


# --------------------------------------------------------------------------- workflow ComfyUI

@dataclass
class ComfyWorkflow:
    """Workflow ComfyUI au format API + nœuds où injecter l'entrée et le nom de sortie."""

    graph: dict[str, dict[str, Any]]
    load_node_id: str
    output_node_ids: list[str]
    preview_only: bool = False

    @classmethod
    def from_file(cls, path: Path, load_node_id: str | None = None,
                  output_node_ids: list[str] | None = None) -> ComfyWorkflow:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise WorkflowError(f"workflow introuvable : {path}") from None
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"JSON invalide dans {path} (ligne {exc.lineno}) : {exc.msg}") from None
        if isinstance(raw, dict) and isinstance(raw.get("prompt"), dict):
            raw = raw["prompt"]   # accepte aussi un corps de requête /prompt complet
        if isinstance(raw, dict) and "nodes" in raw and "links" in raw:
            raise WorkflowError("ce fichier est au format « interface » de ComfyUI : "
                                "réexporte-le avec Workflow → Export (API)")
        if not isinstance(raw, dict) or not raw:
            raise WorkflowError("workflow vide ou mal formé (objet JSON attendu)")
        for nid, node in raw.items():
            if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
                raise WorkflowError(f"nœud {nid!r} mal formé : « class_type » et « inputs » attendus (format API)")
        graph: dict[str, dict[str, Any]] = raw

        def title(nid: str) -> str:
            return str((graph[nid].get("_meta") or {}).get("title", ""))

        # Nœud d'entrée
        loads = [nid for nid, n in graph.items() if n["class_type"] in LOAD_IMAGE_TYPES]
        if load_node_id is not None:
            if load_node_id not in graph:
                raise WorkflowError(f"nœud d'entrée {load_node_id!r} absent du workflow")
            chosen = load_node_id
        elif len(loads) == 1:
            chosen = loads[0]
        elif not loads:
            raise WorkflowError("aucun nœud LoadImage dans le workflow")
        else:
            tagged = [nid for nid in loads if INPUT_TITLE_TAG.lower() in title(nid).lower()]
            if len(tagged) != 1:
                raise WorkflowError(f"plusieurs nœuds LoadImage ({', '.join(loads)}) : renomme celui de la "
                                    f"texture « {INPUT_TITLE_TAG} » ou passe --load-node ID")
            chosen = tagged[0]

        # Nœuds de sortie
        preview_only = False
        if output_node_ids:
            missing = [n for n in output_node_ids if n not in graph]
            if missing:
                raise WorkflowError(f"nœud(s) de sortie absent(s) : {', '.join(missing)}")
            outs = list(output_node_ids)
        else:
            saves = [nid for nid, n in graph.items() if n["class_type"] in SAVE_IMAGE_TYPES]
            tagged_out = [nid for nid in saves if title(nid).lower() == OUTPUT_TITLE_TAG.lower()]
            outs = tagged_out or saves
            if not outs:
                outs = [nid for nid, n in graph.items() if n["class_type"] in PREVIEW_IMAGE_TYPES]
                preview_only = True
                if not outs:
                    raise WorkflowError("aucun nœud SaveImage (ni PreviewImage) : le workflow ne produit rien")
                log.warning("pas de SaveImage : lecture depuis le dossier temporaire de ComfyUI (PreviewImage)")
        return cls(graph, chosen, outs, preview_only)

    def build(self, image_name: str, filename_prefix: str, overrides: Overrides | None = None,
              randomize_seed: bool = False) -> dict[str, dict[str, Any]]:
        """Copie du graphe avec l'image source, le préfixe de sortie et les surcharges injectés."""
        graph = copy.deepcopy(self.graph)
        graph[self.load_node_id]["inputs"]["image"] = image_name
        for nid in self.output_node_ids:
            node = graph[nid]
            if node["class_type"] in SAVE_IMAGE_TYPES or "filename_prefix" in node["inputs"]:
                node["inputs"]["filename_prefix"] = filename_prefix
        for (ref, key), value in (overrides or {}).items():
            graph[self.resolve_node(ref)]["inputs"][key] = value
        if randomize_seed:
            for node in graph.values():
                for key in ("seed", "noise_seed"):
                    if type(node["inputs"].get(key)) is int:   # pas un lien [id, sortie]
                        node["inputs"][key] = random.randint(0, 2 ** 32 - 1)
        return graph

    def resolve_node(self, ref: str) -> str:
        """Identifiant d'un nœud désigné par son id (« 40 ») ou par son titre (« TEXFORGE_SAMPLER »)."""
        if ref in self.graph:
            return ref
        hits = [nid for nid, node in self.graph.items()
                if str((node.get("_meta") or {}).get("title", "")).lower() == ref.lower()]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise WorkflowError(f"--set : plusieurs nœuds portent le titre {ref!r} ({', '.join(hits)})")
        raise WorkflowError(f"--set : aucun nœud d'id ou de titre {ref!r} dans le workflow")

    def node_label(self, nid: Any) -> str:
        node = self.graph.get(str(nid)) or {}
        return str((node.get("_meta") or {}).get("title") or node.get("class_type") or nid)


# --------------------------------------------------------------------------- moteurs

class HttpBackend(abc.ABC):
    """Socle commun : session HTTP, correspondance des erreurs réseau, progression."""

    label = "serveur"
    max_concurrency = 1

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.on_progress: ProgressCallback | None = None
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("moteur non démarré (appeler start())")
        return self._session

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=120))

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> tuple[int, bytes]:
        try:
            async with self.session.request(method, self.base_url + path, **kwargs) as resp:
                return resp.status, await resp.read()
        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as exc:
            raise ServerUnavailable(f"{self.label} injoignable ({self.base_url}) : {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise TextureError(f"{method} {path} : délai réseau dépassé", kind="timeout") from exc
        except aiohttp.ClientPayloadError as exc:
            raise TextureError(f"{method} {path} : réponse tronquée ({exc})", kind="http") from exc

    def _emit(self, text: str) -> None:
        if self.on_progress is not None:
            self.on_progress(text)

    @abc.abstractmethod
    async def check_ready(self) -> str: ...

    @abc.abstractmethod
    async def vram_free_bytes(self) -> int | None: ...

    @abc.abstractmethod
    async def free_vram(self) -> None: ...

    @abc.abstractmethod
    async def generate(self, source: Image.Image, stem: str) -> list[bytes]: ...

    @abc.abstractmethod
    async def cancel_pending(self) -> None: ...


@dataclass
class _PromptTracker:
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: dict[str, Any] | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    node: str | None = None
    workflow: Any = None

    def finish(self, error: dict[str, Any] | None = None) -> None:
        if not self.done.is_set():
            self.error = error
            self.done.set()


class ComfyUIBackend(HttpBackend):
    """ComfyUI : envoi de l'image (/upload/image), soumission (/prompt), suivi WebSocket
    (/ws), lecture des résultats (/history + /view). Repli sur l'historique si le
    WebSocket tombe."""

    label = "ComfyUI"

    def __init__(self, base_url: str, workflow: ComfyWorkflow, *, timeout: float = 600.0,
                 overrides: Overrides | None = None, randomize_seed: bool = False,
                 max_in_flight: int = 2) -> None:
        super().__init__(base_url, timeout)
        self.workflow = workflow
        self.overrides = overrides or {}
        self.randomize_seed = randomize_seed
        # 2 requêtes en file : le GPU enchaîne pendant l'envoi/téléchargement de la suivante
        self.max_concurrency = max(1, max_in_flight)
        self.client_id = uuid.uuid4().hex
        self._ws_task: asyncio.Task[None] | None = None
        self._ws_connected = asyncio.Event()
        self._closing = False
        self._trackers: dict[str, _PromptTracker] = {}
        self._retired: set[str] = set()
        self._inflight: set[str] = set()
        self._queue_remaining: int | None = None

    # ---------------------------------------------------------------- cycle de vie
    async def start(self) -> None:
        await super().start()
        self._closing = False
        self._ws_task = asyncio.create_task(self._ws_loop(), name="comfyui-ws")

    async def close(self) -> None:
        self._closing = True
        if self._ws_task is not None:
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._ws_task
            self._ws_task = None
        await super().close()

    async def check_ready(self) -> str:
        status, body = await self._request("GET", "/system_stats")
        if status != 200:
            raise ServerUnavailable(f"/system_stats a répondu {status} : est-ce bien ComfyUI ?")
        try:
            stats = parse_json(body, "/system_stats")
        except TextureError as exc:
            raise ServerUnavailable(str(exc)) from exc
        dev = (stats.get("devices") or [{}])[0]
        version = (stats.get("system") or {}).get("comfyui_version", "?")
        # laisse au WebSocket le temps de s'ouvrir avant la première requête
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._ws_connected.wait(), timeout=5)
        ws = "WebSocket connecté" if self._ws_connected.is_set() else "WebSocket indisponible (suivi par sondage)"
        return (f"ComfyUI {version} — {dev.get('name', 'GPU ?')}, VRAM libre "
                f"{_gb(dev.get('vram_free'))}/{_gb(dev.get('vram_total'))} Go — {ws}")

    async def vram_free_bytes(self) -> int | None:
        status, body = await self._request("GET", "/system_stats")
        if status != 200:
            return None
        free = ((parse_json(body, "/system_stats").get("devices") or [{}])[0]).get("vram_free")
        return int(free) if isinstance(free, (int, float)) else None

    async def free_vram(self) -> None:
        """Décharge les modèles et vide le cache CUDA (appliqué par ComfyUI entre deux requêtes)."""
        status, _ = await self._request("POST", "/free", json={"unload_models": True, "free_memory": True})
        if status != 200:
            log.warning("/free a répondu %s", status)

    # ---------------------------------------------------------------- génération
    async def generate(self, source: Image.Image, stem: str, *, overrides: Overrides | None = None,
                       workflow: ComfyWorkflow | None = None) -> list[bytes]:
        """overrides : surcharges propres à cette texture (fusionnées avec celles du moteur) ;
        workflow : variante de workflow pour cette texture (sinon celui du moteur)."""
        wf = workflow or self.workflow
        png = png_bytes(source)
        safe = safe_stem(stem)
        image_ref = await self._upload(png, f"{safe}_{hashlib.sha1(png).hexdigest()[:10]}.png")
        merged = {**self.overrides, **(overrides or {})}
        graph = wf.build(image_ref, f"{UPLOAD_SUBFOLDER}/{safe}", merged, self.randomize_seed)
        pid = await self._submit(graph)
        self._tracker(pid).workflow = wf
        self._inflight.add(pid)
        try:
            tracker = await self._wait(pid)
            refs = await self._collect_outputs(pid, tracker, wf)
            images = [await self._download(ref) for ref in refs]
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._cancel_prompt(pid), timeout=5)
            raise
        finally:
            self._inflight.discard(pid)
            self._trackers.pop(pid, None)
            self._retired.add(pid)
        if not images:
            raise TextureError("le workflow n'a produit aucune image", kind="workflow")
        return images

    async def cancel_pending(self) -> None:
        for pid in list(self._inflight):
            await self._cancel_prompt(pid)

    async def _upload(self, png: bytes, filename: str) -> str:
        form = aiohttp.FormData()
        form.add_field("image", png, filename=filename, content_type="image/png")
        form.add_field("type", "input")
        form.add_field("subfolder", UPLOAD_SUBFOLDER)
        form.add_field("overwrite", "true")
        status, body = await self._request("POST", "/upload/image", data=form)
        if status != 200:
            raise TextureError(f"envoi de l'image refusé ({status}) : {body[:200]!r}", kind="http")
        info = parse_json(body, "/upload/image")
        name, sub = info.get("name"), info.get("subfolder") or ""
        if not name:
            raise TextureError("réponse /upload/image sans nom de fichier", kind="http")
        # LoadImage attend un chemin relatif au dossier input de ComfyUI
        return f"{sub}/{name}" if sub else str(name)

    async def _submit(self, graph: dict[str, Any]) -> str:
        # identifiant choisi côté client : le suivi commence avant même la réponse
        wanted = str(uuid.uuid4())
        self._tracker(wanted)
        status, body = await self._request(
            "POST", "/prompt", json={"prompt": graph, "client_id": self.client_id, "prompt_id": wanted})
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if status != 200 or not isinstance(data, dict):
            self._trackers.pop(wanted, None)
            details = json.dumps(data, ensure_ascii=False, indent=1) if data else body.decode("utf-8", "replace")
            raise WorkflowError(describe_prompt_error(status, data, body), details=details)
        if data.get("node_errors"):
            log.warning("sorties partiellement invalides : %s", describe_prompt_error(status, data, body))
        pid = data.get("prompt_id")
        if not isinstance(pid, str):
            self._trackers.pop(wanted, None)
            raise TextureError("réponse /prompt sans prompt_id", kind="http")
        if pid != wanted:   # ancienne version de ComfyUI qui ignore l'identifiant fourni
            self._trackers[pid] = self._trackers.pop(wanted) if wanted in self._trackers else _PromptTracker()
        return pid

    async def _wait(self, pid: str) -> _PromptTracker:
        tracker = self._tracker(pid)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        last_poll = loop.time()
        while not tracker.done.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                await self._cancel_prompt(pid)
                raise TextureError(f"délai de {self.timeout:.0f} s dépassé (requête annulée)", kind="timeout")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(tracker.done.wait(), timeout=min(2.0, remaining))
            if tracker.done.is_set():
                break
            # Repli : si le WebSocket est coupé (ou par sécurité toutes les 20 s), l'historique fait foi
            if not self._ws_connected.is_set() or loop.time() - last_poll > 20:
                last_poll = loop.time()
                entry = await self._history_entry(pid)
                if entry is not None:
                    st = entry.get("status") or {}
                    if st.get("status_str") == "error":
                        tracker.finish(error=error_from_history(st))
                    elif st.get("completed") or entry.get("outputs"):
                        tracker.outputs.update(entry.get("outputs") or {})
                        tracker.finish()
        if tracker.error is not None:
            raise texture_error_from_comfy(tracker.error)
        return tracker

    async def _history_entry(self, pid: str) -> dict[str, Any] | None:
        status, body = await self._request("GET", f"/history/{pid}")
        if status != 200:
            return None
        data = parse_json(body, "/history")
        entry = data.get(pid) if isinstance(data, dict) else None
        return entry if isinstance(entry, dict) else None

    async def _collect_outputs(self, pid: str, tracker: _PromptTracker,
                               wf: ComfyWorkflow | None = None) -> list[dict[str, str]]:
        wf = wf or self.workflow
        outputs: dict[str, Any] = {}
        for _ in range(10):   # l'historique est écrit juste après le signal de fin
            entry = await self._history_entry(pid)
            if entry and entry.get("outputs"):
                outputs = entry["outputs"]
                break
            await asyncio.sleep(0.3)
        if not outputs:
            outputs = tracker.outputs   # repli : messages « executed » reçus par WebSocket
        wanted_type = "temp" if wf.preview_only else "output"
        refs: list[dict[str, str]] = []
        for nid in wf.output_node_ids:
            for img in (outputs.get(nid) or {}).get("images", []):
                if isinstance(img, dict) and img.get("filename") and img.get("type", wanted_type) == wanted_type:
                    refs.append({"filename": str(img["filename"]), "subfolder": str(img.get("subfolder", "")),
                                 "type": str(img.get("type", wanted_type))})
        return refs

    async def _download(self, ref: dict[str, str]) -> bytes:
        status, body = await self._request("GET", "/view", params=ref)
        if status != 200 or not body:
            raise TextureError(f"téléchargement de {ref['filename']} impossible ({status})", kind="http")
        return body

    async def _cancel_prompt(self, pid: str) -> None:
        """Retire la requête de la file d'attente, ou interrompt son exécution."""
        with contextlib.suppress(Exception):
            status, body = await self._request("GET", "/queue")
            queue = parse_json(body, "/queue") if status == 200 else {}
            running = {item[1] for item in queue.get("queue_running", []) if len(item) > 1}
            pending = {item[1] for item in queue.get("queue_pending", []) if len(item) > 1}
            if pid in pending:
                await self._request("POST", "/queue", json={"delete": [pid]})
            if pid in running:
                await self._request("POST", "/interrupt", json={"prompt_id": pid})

    # ---------------------------------------------------------------- WebSocket
    def _tracker(self, pid: str) -> _PromptTracker:
        tracker = self._trackers.get(pid)
        if tracker is None:
            tracker = self._trackers[pid] = _PromptTracker()
        return tracker

    async def _ws_loop(self) -> None:
        ws_url = re.sub(r"^http", "ws", self.base_url) + f"/ws?clientId={self.client_id}"
        backoff = 1.0
        while not self._closing:
            try:
                async with self.session.ws_connect(ws_url, heartbeat=20, max_msg_size=0) as ws:
                    self._ws_connected.set()
                    backoff = 1.0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._on_ws_message(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        # BINARY : aperçus du sampler, ignorés
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug("WebSocket : %s", exc)
            finally:
                self._ws_connected.clear()
            if self._closing:
                return
            log.debug("WebSocket fermé, reconnexion dans %.0f s", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def _on_ws_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        data = msg.get("data") or {}
        if mtype == "status":
            remaining = ((data.get("status") or {}).get("exec_info") or {}).get("queue_remaining")
            if isinstance(remaining, int):
                self._queue_remaining = remaining
            return
        pid = data.get("prompt_id")
        if not isinstance(pid, str) or pid in self._retired:
            return
        tracker = self._tracker(pid)
        if mtype == "executing":
            node = data.get("node")
            if node is None:
                tracker.finish()
            else:
                tracker.node = str(node)
                self._progress((tracker.workflow or self.workflow).node_label(node))
        elif mtype == "progress":
            label = (tracker.workflow or self.workflow).node_label(data.get("node") or tracker.node)
            self._progress(f"{label} {data.get('value')}/{data.get('max')}")
        elif mtype == "executed":
            node, output = data.get("node"), data.get("output")
            if node is not None and isinstance(output, dict):
                tracker.outputs[str(node)] = output
        elif mtype == "execution_success":
            tracker.finish()
        elif mtype == "execution_error":
            tracker.finish(error=data)
        elif mtype == "execution_interrupted":
            tracker.finish(error={"exception_type": "Interrupted", "exception_message": "exécution interrompue",
                                  **data})

    def _progress(self, text: str) -> None:
        queue = f" · file {self._queue_remaining}" if self._queue_remaining else ""
        self._emit(text + queue)


def describe_prompt_error(status: int, data: Any, body: bytes) -> str:
    """Résumé lisible d'un refus de /prompt (validation du graphe)."""
    if not isinstance(data, dict):
        return f"/prompt a répondu {status} : {body[:300].decode('utf-8', 'replace')}"
    parts: list[str] = []
    err = data.get("error")
    if isinstance(err, dict):
        parts.append(str(err.get("message") or err.get("type") or "erreur"))
        if err.get("details"):
            parts.append(str(err["details"]))
    elif err:
        parts.append(str(err))
    for nid, node_err in (data.get("node_errors") or {}).items():
        for e in (node_err or {}).get("errors", []):
            parts.append(f"nœud {nid} ({node_err.get('class_type', '?')}) : "
                         f"{e.get('message', '')} {e.get('details', '')}".strip())
    return f"workflow refusé ({status}) : " + " | ".join(parts or ["raison inconnue"])


def error_from_history(status: dict[str, Any]) -> dict[str, Any]:
    for item in status.get("messages") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "execution_error":
            return dict(item[1])
    return {"exception_type": "Error", "exception_message": "échec signalé dans l'historique (voir console ComfyUI)"}


def texture_error_from_comfy(err: dict[str, Any]) -> TextureError:
    etype = str(err.get("exception_type") or "Error")
    emsg = str(err.get("exception_message") or "").strip()
    node = err.get("node_id")
    where = f" — nœud {node} ({err.get('node_type', '?')})" if node else ""
    tb = err.get("traceback")
    details = "".join(tb) if isinstance(tb, list) else str(tb or "")
    kind = "interrupted" if etype == "Interrupted" else classify_failure(f"{etype} {emsg}")
    return TextureError(f"{etype}: {emsg}{where}", kind=kind, details=details)


class SDWebUIBackend(HttpBackend):
    """API Stable Diffusion WebUI (AUTOMATIC1111 / Forge, lancée avec --api) :
    /sdapi/v1/img2img, progression par /sdapi/v1/progress."""

    label = "SD WebUI"
    max_concurrency = 1   # la WebUI traite une génération à la fois

    def __init__(self, base_url: str, params: dict[str, Any], *, timeout: float = 600.0,
                 long_side: int = 1024, tiling: bool = False, randomize_seed: bool = False) -> None:
        super().__init__(base_url, timeout)
        self.params = params
        self.long_side = long_side
        self.tiling = tiling
        self.randomize_seed = randomize_seed
        self._needs_reload = False

    @staticmethod
    def load_params(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise WorkflowError(f"paramètres introuvables : {path}") from None
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"JSON invalide dans {path} (ligne {exc.lineno}) : {exc.msg}") from None
        if not isinstance(data, dict):
            raise WorkflowError("paramètres img2img : objet JSON attendu")
        if any(isinstance(v, dict) and "class_type" in v for v in data.values()):
            raise WorkflowError("ce fichier est un workflow ComfyUI : utilise --backend comfyui")
        data.pop("init_images", None)
        return data

    async def check_ready(self) -> str:
        status, _ = await self._request("GET", "/sdapi/v1/progress?skip_current_image=true")
        if status == 404:
            raise ServerUnavailable("API absente : relance la WebUI avec l'option --api")
        if status != 200:
            raise ServerUnavailable(f"/sdapi/v1/progress a répondu {status}")
        model = "?"
        with contextlib.suppress(Exception):
            st, body = await self._request("GET", "/sdapi/v1/options")
            if st == 200:
                model = str(json.loads(body).get("sd_model_checkpoint", "?"))
        free = await self.vram_free_bytes()
        return f"SD WebUI — modèle {model}, VRAM libre {_gb(free)} Go"

    async def vram_free_bytes(self) -> int | None:
        with contextlib.suppress(TextureError):
            status, body = await self._request("GET", "/sdapi/v1/memory")
            if status == 200:
                free = ((parse_json(body, "/memory").get("cuda") or {}).get("system") or {}).get("free")
                return int(free) if isinstance(free, (int, float)) else None
        return None

    async def free_vram(self) -> None:
        """Décharge le checkpoint ; il sera rechargé avant la génération suivante."""
        status, _ = await self._request("POST", "/sdapi/v1/unload-checkpoint")
        if status == 200:
            self._needs_reload = True
        else:
            log.warning("/sdapi/v1/unload-checkpoint a répondu %s", status)

    async def generate(self, source: Image.Image, stem: str) -> list[bytes]:
        if self._needs_reload:
            status, _ = await self._request("POST", "/sdapi/v1/reload-checkpoint")
            if status != 200:
                raise TextureError(f"rechargement du modèle impossible ({status})", kind="http")
            self._needs_reload = False
        payload = copy.deepcopy(self.params)
        payload["init_images"] = [base64.b64encode(png_bytes(source.convert("RGB"))).decode("ascii")]
        if "width" not in payload or "height" not in payload:
            payload["width"], payload["height"] = fit_size(source.size, self.long_side)
        if self.tiling:
            payload["tiling"] = True   # texture raccordable sur les bords
        if self.randomize_seed:
            payload["seed"] = random.randint(0, 2 ** 31 - 1)
        payload.setdefault("send_images", True)
        payload.setdefault("save_images", False)
        expected = max(1, int(payload.get("batch_size", 1)) * int(payload.get("n_iter", 1)))

        poller = asyncio.create_task(self._poll_progress())
        try:
            status, body = await asyncio.wait_for(
                self._request("POST", "/sdapi/v1/img2img", json=payload,
                              timeout=aiohttp.ClientTimeout(total=None, sock_connect=10,
                                                            sock_read=self.timeout + 30)),
                timeout=self.timeout)
        except asyncio.TimeoutError:
            await self.cancel_pending()
            raise TextureError(f"délai de {self.timeout:.0f} s dépassé (génération interrompue)",
                               kind="timeout") from None
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.cancel_pending(), timeout=5)
            raise
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller

        if status != 200:
            raise sd_error(status, body)
        images = parse_json(body, "/sdapi/v1/img2img").get("images") or []
        if not images:
            raise TextureError("la WebUI n'a renvoyé aucune image", kind="http")
        out: list[bytes] = []
        for b64 in images[:expected]:   # au-delà : cartes ControlNet ajoutées par les extensions
            if "," in b64[:100]:
                b64 = b64.split(",", 1)[1]
            out.append(base64.b64decode(b64))
        return out

    async def cancel_pending(self) -> None:
        with contextlib.suppress(Exception):
            await self._request("POST", "/sdapi/v1/interrupt")

    async def _poll_progress(self) -> None:
        while True:
            await asyncio.sleep(0.8)
            with contextlib.suppress(Exception):
                status, body = await self._request("GET", "/sdapi/v1/progress?skip_current_image=true")
                if status != 200:
                    continue
                d = json.loads(body)
                state = d.get("state") or {}
                step = (f" · pas {state.get('sampling_step')}/{state.get('sampling_steps')}"
                        if state.get("sampling_steps") else "")
                eta = d.get("eta_relative")
                self._emit(f"{float(d.get('progress') or 0) * 100:.0f}%{step}"
                           + (f" · reste ~{eta:.0f} s" if isinstance(eta, (int, float)) and eta > 0 else ""))


def sd_error(status: int, body: bytes) -> TextureError:
    text = body[:20000].decode("utf-8", "replace")
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict):
        msg = " — ".join(str(data[k]) for k in ("error", "detail", "errors") if data.get(k)) or text[:300]
    else:
        msg = text[:300]
    if status == 422:
        return WorkflowError(f"paramètres refusés par la WebUI (422) : {msg}", details=text)
    if status == 404:
        return WorkflowError("point d'accès introuvable (404) : lance la WebUI avec --api", details=text)
    return TextureError(f"erreur WebUI ({status}) : {msg}", kind=classify_failure(msg), details=text)


# --------------------------------------------------------------------------- lot

@dataclass
class TextureResult:
    source: Path
    outputs: list[Path] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


@dataclass
class BatchReport:
    results: list[TextureResult] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    aborted: str | None = None
    error_log: Path | None = None
    seconds: float = 0.0

    @property
    def succeeded(self) -> list[TextureResult]:
        return [r for r in self.results if r.error is None]

    @property
    def failed(self) -> list[TextureResult]:
        return [r for r in self.results if r.error is not None]


def collect_textures(root: Path, recursive: bool) -> list[Path]:
    pattern = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in pattern if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def plan_outputs(sources: list[Path], in_root: Path, out_root: Path) -> list[tuple[Path, Path]]:
    """Chemin de sortie de chaque texture : même arborescence, extension .png.
    Si deux sources ont le même nom (rock.dds et rock.png), toutes deux gardent leur
    extension d'origine dans le nom (rock.dds.png, rock.png.png), quel que soit l'ordre."""
    plain = {src: out_root / src.relative_to(in_root).with_suffix(".png") for src in sources}
    counts: dict[Path, int] = {}
    for out in plain.values():
        counts[out] = counts.get(out, 0) + 1
    plan: list[tuple[Path, Path]] = []
    for src, out in plain.items():
        if counts[out] > 1:
            rel = src.relative_to(in_root)
            out = out_root / rel.parent / f"{rel.name}.png"
        plan.append((src, out))
    return plan


def _open_error_log(path: Path) -> tuple[logging.Logger, logging.Handler]:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("texforge.headless.errors")
    logger.propagate = False
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger, handler


def _log_failure(errlog: logging.Logger, src: Path, exc: TextureError | str, engine: str) -> None:
    if isinstance(exc, TextureError):
        entry = f"{engine} | {exc.kind} | {src} | {exc}"
        if exc.details:
            entry += "\n    " + exc.details.strip().replace("\n", "\n    ")
    else:
        entry = f"{engine} | {src} | {exc}"
    errlog.error(entry)


async def _wait_for_server(engine: HttpBackend, seconds: float) -> bool:
    """Le serveur a peut-être planté (erreur CUDA) : on lui laisse le temps de redémarrer."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            await engine.check_ready()
            return True
        except (ServerUnavailable, TextureError):
            await asyncio.sleep(3)
    return False


async def _maybe_free_vram(engine: HttpBackend, policy: FreePolicy, threshold_gb: float) -> None:
    if policy == "never":
        return
    try:
        if policy == "auto":
            free = await engine.vram_free_bytes()
            if free is None or free >= threshold_gb * 1024 ** 3:
                return
            log.info("VRAM libre %.1f Go < %.1f Go : libération", free / 1024 ** 3, threshold_gb)
        await engine.free_vram()
    except (ServerUnavailable, TextureError) as exc:
        log.warning("libération de la VRAM impossible : %s", exc)


async def _process_one(engine: HttpBackend, src: Path, out: Path, keep_alpha: bool) -> list[Path]:
    source = await asyncio.to_thread(load_source_image, src)
    try:
        blobs = await engine.generate(source, src.stem)
    except TextureError as exc:
        if not exc.is_oom:
            raise
        log.warning("%s : mémoire GPU saturée, libération de la VRAM puis nouvel essai", src.name)
        await engine.free_vram()
        blobs = await engine.generate(source, src.stem)
    return await asyncio.to_thread(save_outputs, blobs, source, out, keep_alpha)


async def _run_jobs(engine: HttpBackend, jobs: list[tuple[Path, Path]], report: BatchReport,
                    errlog: logging.Logger, *, batch_size: int, free_vram: FreePolicy,
                    vram_threshold_gb: float, keep_alpha: bool, max_consecutive_failures: int,
                    show_progress: bool) -> None:
    bar = tqdm(total=len(jobs), unit="tex", desc=engine.label, dynamic_ncols=True, disable=not show_progress)
    engine.on_progress = lambda text: bar.set_postfix_str(text[:70], refresh=True)
    sem = asyncio.Semaphore(engine.max_concurrency)
    abort = asyncio.Event()
    abort_reason: list[str] = []
    consecutive = 0

    def stop(reason: str) -> None:
        if not abort.is_set():
            abort_reason.append(reason)
            abort.set()

    async def one(src: Path, out: Path) -> None:
        nonlocal consecutive
        async with sem:
            if abort.is_set():
                return
            result = TextureResult(src)
            t0 = time.monotonic()
            try:
                result.outputs = await _process_one(engine, src, out, keep_alpha)
                consecutive = 0
            except asyncio.CancelledError:
                result.error = "annulé"
                raise
            except ServerUnavailable as exc:
                result.error = str(exc)
                _log_failure(errlog, src, str(exc), engine.label)
                log.error("✗ %s : %s", src.name, exc)
                log.warning("serveur injoignable, attente de son retour (60 s max)…")
                if not await _wait_for_server(engine, 60):
                    stop(f"{engine.label} injoignable : {exc}")
            except TextureError as exc:
                result.error = str(exc)
                _log_failure(errlog, src, exc, engine.label)
                log.error("✗ %s : %s", src.name, exc)
                consecutive += 1
            except Exception as exc:   # bug inattendu : consigné, le lot continue
                result.error = f"{type(exc).__name__}: {exc}"
                _log_failure(errlog, src, TextureError(result.error, details=traceback.format_exc()), engine.label)
                log.error("✗ %s : %s", src.name, result.error)
                consecutive += 1
            finally:
                result.seconds = time.monotonic() - t0
                report.results.append(result)
                bar.update(1)
            if max_consecutive_failures and consecutive >= max_consecutive_failures:
                stop(f"{consecutive} échecs consécutifs : problème probable de workflow ou de serveur")

    try:
        for start in range(0, len(jobs), batch_size):
            if abort.is_set():
                break
            chunk = jobs[start:start + batch_size]
            await asyncio.gather(*(one(src, out) for src, out in chunk))
            if not abort.is_set() and start + batch_size < len(jobs):
                await _maybe_free_vram(engine, free_vram, vram_threshold_gb)
    except asyncio.CancelledError:
        log.warning("■ interruption : annulation des requêtes en cours…")
        with contextlib.suppress(Exception):
            await asyncio.wait_for(engine.cancel_pending(), timeout=5)
        report.aborted = "interrompu par l'utilisateur"
        raise
    finally:
        engine.on_progress = None
        bar.close()
    if abort_reason:
        report.aborted = abort_reason[0]
        errlog.error("ARRÊT DU LOT | %s", abort_reason[0])


def make_backend(backend: BackendName, workflow_json_path: str | Path, server_url: str | None, *,
                 timeout: float, max_in_flight: int, overrides: Overrides | None,
                 load_node_id: str | None, output_node_ids: list[str] | None, randomize_seed: bool,
                 sd_long_side: int, sd_tiling: bool) -> HttpBackend:
    if backend == "comfyui":
        wf = ComfyWorkflow.from_file(Path(workflow_json_path), load_node_id, output_node_ids)
        wf.build("test.png", "test", overrides)   # valide les --set avant de lancer le lot
        return ComfyUIBackend(server_url or DEFAULT_COMFY_URL, wf, timeout=timeout, overrides=overrides,
                              randomize_seed=randomize_seed, max_in_flight=max_in_flight)
    if backend == "sd":
        params = SDWebUIBackend.load_params(Path(workflow_json_path))
        return SDWebUIBackend(server_url or DEFAULT_SD_URL, params, timeout=timeout, long_side=sd_long_side,
                              tiling=sd_tiling, randomize_seed=randomize_seed)
    raise ValueError(f"moteur inconnu : {backend}")


async def process_texture_batch_async(
    input_folder: str | Path,
    output_folder: str | Path,
    workflow_json_path: str | Path,
    *,
    backend: BackendName = "comfyui",
    server_url: str | None = None,
    batch_size: int = 8,
    free_vram: FreePolicy = "auto",
    vram_threshold_gb: float = 1.5,
    timeout: float = 600.0,
    max_in_flight: int = 2,
    overwrite: bool = False,
    recursive: bool = False,
    keep_alpha: bool = True,
    overrides: Overrides | None = None,
    load_node_id: str | None = None,
    output_node_ids: list[str] | None = None,
    randomize_seed: bool = False,
    sd_long_side: int = 1024,
    sd_tiling: bool = False,
    max_consecutive_failures: int = 10,
    error_log: str | Path | None = None,
    show_progress: bool = True,
) -> BatchReport:
    """Version asynchrone de process_texture_batch (mêmes paramètres)."""
    t_start = time.monotonic()
    in_root = Path(input_folder).resolve()
    out_root = Path(output_folder).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(error_log).resolve() if error_log else out_root / ERROR_LOG_NAME
    report = BatchReport(error_log=log_path)
    errlog, handler = _open_error_log(log_path)
    try:
        if not in_root.is_dir():
            report.aborted = f"dossier d'entrée introuvable : {in_root}"
            errlog.error("CONFIG | %s", report.aborted)
            log.error(report.aborted)
            return report
        try:
            engine = make_backend(backend, workflow_json_path, server_url, timeout=timeout,
                                  max_in_flight=max_in_flight, overrides=overrides, load_node_id=load_node_id,
                                  output_node_ids=output_node_ids, randomize_seed=randomize_seed,
                                  sd_long_side=sd_long_side, sd_tiling=sd_tiling)
        except WorkflowError as exc:
            report.aborted = f"workflow invalide : {exc}"
            errlog.error("CONFIG | %s | %s", workflow_json_path, exc)
            log.error(report.aborted)
            return report

        jobs: list[tuple[Path, Path]] = []
        for src, out in plan_outputs(collect_textures(in_root, recursive), in_root, out_root):
            if out.exists() and not overwrite:
                report.skipped.append(src)
            else:
                jobs.append((src, out))
        if not jobs:
            log.info("rien à faire : %d texture(s) déjà traitée(s) (--overwrite pour refaire)", len(report.skipped))
            return report

        await engine.start()
        try:
            try:
                info = await engine.check_ready()
            except (ServerUnavailable, TextureError) as exc:
                report.aborted = str(exc)
                errlog.error("SERVEUR | %s", exc)
                log.error("%s", exc)
                return report
            log.info("%s", info)
            log.info("%d texture(s) à traiter, %d déjà faite(s)", len(jobs), len(report.skipped))
            await _run_jobs(engine, jobs, report, errlog, batch_size=max(1, batch_size), free_vram=free_vram,
                            vram_threshold_gb=vram_threshold_gb, keep_alpha=keep_alpha,
                            max_consecutive_failures=max_consecutive_failures, show_progress=show_progress)
        finally:
            await engine.close()
        return report
    finally:
        report.seconds = time.monotonic() - t_start
        errlog.removeHandler(handler)
        handler.close()


def process_texture_batch(input_folder: str | Path, output_folder: str | Path,
                          workflow_json_path: str | Path, **kwargs: Any) -> BatchReport:
    """Traite toutes les textures de input_folder et écrit les PNG dans output_folder.

    workflow_json_path : workflow ComfyUI au format API (backend="comfyui", défaut), ou
    paramètres img2img JSON (backend="sd"). Options : voir process_texture_batch_async.
    Les échecs sont consignés dans output_folder/comfy_errors.log ; le lot continue.
    """
    return asyncio.run(process_texture_batch_async(input_folder, output_folder, workflow_json_path, **kwargs))


# --------------------------------------------------------------------------- ligne de commande

class _TqdmHandler(logging.Handler):
    """Affiche les messages au-dessus de la barre de progression sans la casser."""

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            tqdm.write(self.format(record), file=sys.stderr)


def _setup_console_logging(verbose: bool) -> None:
    handler = _TqdmHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("texforge")
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False
    logging.getLogger("texforge.headless.errors").propagate = False


def parse_set_args(items: list[str]) -> Overrides:
    """--set 6.text="mossy stone" --set 3.steps=30 -> {("6","text"): "mossy stone", ("3","steps"): 30}"""
    out: Overrides = {}
    for item in items:
        key, sep, raw = item.partition("=")
        node, dot, inp = key.partition(".")
        if not sep or not dot or not node.strip() or not inp.strip():
            raise SystemExit(f"--set attend NŒUD.ENTRÉE=VALEUR, reçu : {item}")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[(node.strip(), inp.strip())] = value
    return out


EPILOG = """\
Exporter un workflow ComfyUI au format API :
  menu Workflow → Export (API). Anciennes versions : Paramètres → activer
  « Dev mode », puis bouton « Save (API Format) ».
  Le workflow doit contenir un LoadImage (titre TEXFORGE_INPUT s'il y en a
  plusieurs) et un SaveImage.

Lancer ComfyUI sans ouvrir le navigateur :
  python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch

Lancer la WebUI A1111/Forge avec l'API :
  webui-user.bat avec  set COMMANDLINE_ARGS=--api --nowebui
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comfyui_headless_client",
        description="Traitement par lot de textures via ComfyUI (headless) ou l'API Stable Diffusion WebUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    p.add_argument("input", nargs="?", help="dossier des textures sources")
    p.add_argument("output", nargs="?", help="dossier de sortie (PNG)")
    p.add_argument("-w", "--workflow", help="ComfyUI : workflow format API ; SD : paramètres img2img (JSON)")
    p.add_argument("--backend", choices=["comfyui", "sd"], default="comfyui", help="moteur (défaut : comfyui)")
    p.add_argument("--server", help=f"URL du serveur (défaut : {DEFAULT_COMFY_URL} ou {DEFAULT_SD_URL})")
    p.add_argument("--batch-size", type=int, default=8, help="textures par lot entre deux contrôles de VRAM (8)")
    p.add_argument("--free-vram", choices=["auto", "always", "never"], default="auto",
                   help="libération de la VRAM entre les lots (auto : sous le seuil)")
    p.add_argument("--vram-threshold", type=float, default=1.5, metavar="GO", help="seuil du mode auto (1.5 Go)")
    p.add_argument("--timeout", type=float, default=600, metavar="S", help="délai max par texture (600 s)")
    p.add_argument("--max-in-flight", type=int, default=2, help="ComfyUI : requêtes en file simultanées (2)")
    p.add_argument("--overwrite", action="store_true", help="refaire les textures déjà présentes en sortie")
    p.add_argument("-r", "--recursive", action="store_true", help="inclure les sous-dossiers")
    p.add_argument("--no-alpha", action="store_true", help="ne pas recoller la transparence d'origine")
    p.add_argument("--set", action="append", default=[], metavar="NŒUD.ENTRÉE=VALEUR",
                   help="ComfyUI : surcharge une entrée ; NŒUD = id ou titre (répétable), "
                        "ex. --set TEXFORGE_SAMPLER.denoise=0.4 --set TEXFORGE_TILING.tiling=disable")
    p.add_argument("--load-node", metavar="ID", help="ComfyUI : id du LoadImage à alimenter")
    p.add_argument("--output-node", action="append", metavar="ID", help="ComfyUI : id du nœud de sortie (répétable)")
    p.add_argument("--random-seed", action="store_true", help="graine aléatoire à chaque texture")
    p.add_argument("--sd-long-side", type=int, default=1024, help="SD : côté long de génération (1024)")
    p.add_argument("--sd-tiling", action="store_true", help="SD : génération raccordable (textures répétées)")
    p.add_argument("--max-failures", type=int, default=10, help="arrêt après N échecs consécutifs (0 = jamais)")
    p.add_argument("--error-log", metavar="FICHIER", help=f"journal d'erreurs (défaut : SORTIE/{ERROR_LOG_NAME})")
    p.add_argument("--example-sd-params", action="store_true", help="affiche un fichier de paramètres SD d'exemple")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_summary(report: BatchReport) -> None:
    minutes, seconds = divmod(int(report.seconds), 60)
    print(f"\nTerminé en {minutes} min {seconds:02d} s : {len(report.succeeded)} réussie(s), "
          f"{len(report.skipped)} déjà faite(s), {len(report.failed)} échec(s).")
    if report.failed:
        for r in report.failed[:10]:
            print(f"  ✗ {r.source.name} : {r.error}")
        if len(report.failed) > 10:
            print(f"  … et {len(report.failed) - 10} autre(s)")
    if report.failed or report.aborted:
        print(f"Détails : {report.error_log}")
    if report.aborted:
        print(f"Lot arrêté : {report.aborted}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.example_sd_params:
        print(json.dumps(SD_EXAMPLE_PARAMS, indent=2, ensure_ascii=False))
        return 0
    if not (args.input and args.output and args.workflow):
        parser.error("arguments requis : ENTRÉE SORTIE --workflow FICHIER")
    _setup_console_logging(args.verbose)
    try:
        report = process_texture_batch(
            args.input, args.output, args.workflow,
            backend=args.backend, server_url=args.server, batch_size=args.batch_size,
            free_vram=args.free_vram, vram_threshold_gb=args.vram_threshold, timeout=args.timeout,
            max_in_flight=args.max_in_flight, overwrite=args.overwrite, recursive=args.recursive,
            keep_alpha=not args.no_alpha, overrides=parse_set_args(args.set), load_node_id=args.load_node,
            output_node_ids=args.output_node, randomize_seed=args.random_seed,
            sd_long_side=args.sd_long_side, sd_tiling=args.sd_tiling,
            max_consecutive_failures=args.max_failures, error_log=args.error_log)
    except KeyboardInterrupt:
        print("\n■ interrompu : les requêtes en cours ont été annulées côté serveur.")
        return 130
    _print_summary(report)
    if report.aborted:
        return 2
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
