"""Fournisseurs de génération : ComfyUI (local, par défaut), Mock (gratuit, pour tester
toute la chaîne) et Higgsfield (cloud payant, optionnel : higgsfield-client n'est importé
que si ce fournisseur est choisi)."""

from __future__ import annotations

import copy

import hashlib
import io
import os
import sys
import re
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from .i18n import N_, tr
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
        raise ProviderError(tr("une requête mock ne peut pas être reprise"))


class HiggsfieldProvider:
    name = "higgsfield"
    paid = True

    def __init__(self, cfg: dict, state):
        try:
            import higgsfield_client
            import httpx
        except ImportError as exc:
            raise FatalError(tr("le paquet higgsfield-client n'est pas installé : pip install -r requirements.txt")) from exc
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
                tr("clé Higgsfield absente. Crée une clé sur console.higgsfield.ai puis définis la variable "
                "d'environnement HF_KEY=\"id:secret\" (ou HF_API_KEY et HF_API_SECRET)."))
        self.client = higgsfield_client.SyncClient(api_key=key, timeout=120.0)
        self.model = self.hcfg["model"]

    # ------------------------------------------------------------ erreurs
    def _classify(self, exc: Exception) -> Exception:
        cause = exc.__cause__
        code = getattr(getattr(cause, "response", None), "status_code", None)
        msg = str(exc) or exc.__class__.__name__
        if code in (401, 403):
            return FatalError(tr("accès refusé par Higgsfield ({code}) : {msg}", code=code, msg=msg))
        if code == 402 or "credit" in msg.lower() or "balance" in msg.lower():
            return FatalError(tr("crédits insuffisants ({code}) : {msg}", code=code, msg=msg))
        if code in (400, 404, 422):
            return ProviderError(tr("requête refusée ({code}) : {msg}", code=code, msg=msg))
        return TransientError(tr("erreur Higgsfield ({code}) : {msg}", code=code, msg=msg))

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
            raise TransientError(tr("envoi impossible : {exc}", exc=exc)) from exc
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
            raise TransientError(tr("soumission impossible : {exc}", exc=exc)) from exc
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
                    raise TransientError(tr("réseau indisponible pendant l'attente : {exc}", exc=exc)) from exc
                time.sleep(interval * net_errors)
                continue
            if isinstance(st, hf.Completed):
                break
            if isinstance(st, hf.NSFW):
                raise ProviderError(tr("résultat bloqué par le filtre de contenu (NSFW)"))
            if isinstance(st, (hf.Failed, hf.Cancelled)):
                raise ProviderError(tr("génération en échec ({name})", name=st.__class__.__name__))
            if time.time() > deadline:
                raise TransientError(tr("délai dépassé ; la requête sera reprise au prochain lancement"))
            time.sleep(interval)
        try:
            data = controller.get()
        except hf.HiggsfieldClientError as exc:
            raise self._classify(exc) from exc
        except self._httpx.TransportError as exc:
            raise TransientError(tr("lecture du résultat impossible : {exc}", exc=exc)) from exc
        images = data.get("images") or []
        if not images or not images[0].get("url"):
            raise ProviderError(tr("réponse sans image : {data}", data=str(data)[:200]))
        url = images[0]["url"]
        try:
            r = self._httpx.get(url, timeout=180.0, follow_redirects=True)
            r.raise_for_status()
        except Exception as exc:
            raise TransientError(tr("téléchargement du résultat impossible : {exc}", exc=exc)) from exc
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
    local_gpu = True              # file locale : groupes par workflow et par raccord, préréglages par matériau
    native_tiling = True          # le raccord est géré dans le workflow (padding circulaire)
    supports_tiles = True         # génération par morceaux à la taille finale (comfyui.tiled)
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
        self._fit_lock = threading.Lock()
        self._missing: set[str] = set()       # types de nœuds absents du serveur (contournés)
        self._edges_off = False               # workflow Tile+Canny inutilisable sur ce serveur
        self.on_note: Callable[[str], None] | None = None
        self.on_pass: Callable[[str | None], None] | None = None   # passage terminé (id de la texture)
        try:
            self.wf_main = customize_workflow(chc.ComfyWorkflow.from_file(
                resolve_workflow_path(cfg, self.c["workflow"])), self.c, chc)
            edges_raw = str(self.c.get("workflow_edges") or "")
            self.wf_edges = (customize_workflow(chc.ComfyWorkflow.from_file(
                resolve_workflow_path(cfg, edges_raw)), self.c, chc) if edges_raw else None)
        except chc.WorkflowError as exc:
            raise WorkflowFatal(tr("workflow ComfyUI invalide : {exc}", exc=exc)) from exc
        self._orig_main, self._orig_edges = self.wf_main, self.wf_edges
        titles = {str((n.get("_meta") or {}).get("title", "")) for n in self.wf_main.graph.values()}
        classes = {n.get("class_type") for n in self.wf_main.graph.values()}
        # raccord circulaire seulement si le workflow a le nœud (sans lui : bords prolongés par TEXFORGE) ;
        # réglages sans nœud correspondant (Z-Image : ni ControlNet ni raccord) ignorés sans avertissement
        self.native_tiling = "SeamlessTile" in classes
        self._warned |= {t for t in ("TEXFORGE_CN_TILE", "TEXFORGE_TILING", "TEXFORGE_TILING_VAE") if t not in titles}
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="comfyui-loop", daemon=True)
        self.thread.start()
        self.backend = chc.ComfyUIBackend(str(self.c["server"]), self.wf_main, timeout=float(self.c["timeout"]),
                                          max_in_flight=int(self.c.get("max_in_flight", 2)))
        self.backend.on_progress = on_progress
        self._call(self.backend.start())
        self.notes: list[str] = []
        try:
            self.info = self.check_ready()
            self._fit_to_server()
        except BaseException:
            self.close()          # sinon la boucle et le WebSocket de cet essai restent ouverts
            raise

    def _fit_to_server(self) -> None:
        """Workflow adapté aux nœuds installés sur ce serveur : les nœuds facultatifs absents sont
        contournés (TEXFORGE fait le travail lui-même), un nœud indispensable absent arrête tout de
        suite, avec la liste, au lieu d'un refus texture par texture."""
        self._fit_models()
        wanted = self._workflow_classes()
        classes = None
        for ask in (lambda: self.backend.node_classes(), lambda: self.backend.probe_node_classes(wanted)):
            try:
                classes = self._call(ask(), timeout=60)
            except Exception:
                classes = None
            if classes:
                break
        if not classes:
            # liste illisible : ComfyUI jugera ; un nœud facultatif refusé sera contourné à ce moment-là
            self.notes.append(tr("liste des nœuds installés illisible (/object_info) : workflow envoyé tel quel ; "
                                 "un nœud facultatif refusé par ComfyUI sera contourné à la première texture."))
            self._set_denoise_scale()
            return
        self.notes += self._refit(wanted - classes)

    def _set_denoise_scale(self) -> None:
        cn = any(n.get("class_type") in CONTROLNET_APPLY for n in self.wf_main.graph.values())
        self.denoise_scale = float(self.c.get("denoise_scale" if cn else "denoise_scale_no_cn", 1.0) or 1.0)

    def _family(self) -> str:
        name = Path(str(self.c.get("workflow") or "")).name.lower()
        return next((f for f in ("sdxl", "sd15", "zimage") if f in name), "")

    def _fit_models(self) -> None:
        """Fichiers de modèles du workflow comparés à ceux de ComfyUI, avant toute texture. Un modèle par défaut
        absent est remplacé par un équivalent présent (autre checkpoint SDXL, autre précision du même modèle) ;
        sinon : workflow Tile+Canny désactivé, ou arrêt immédiat avec ce qui manque et où le mettre."""
        cache: dict[tuple[str, str], list[str] | None] = {}

        def options(cls: str, field: str) -> list[str] | None:
            if (cls, field) not in cache:
                try:
                    cache[(cls, field)] = self._call(self.backend.input_options(cls, field), timeout=30)
                except Exception:
                    cache[(cls, field)] = None
            return cache[(cls, field)]

        family = self._family()
        chosen = {"ckpt_name": self.c.get("checkpoint"), "control_net_name": self.c.get("controlnet_model"),
                  "lora_name": self.c.get("lora")}
        missing: list[tuple[str, str, str, list[str]]] = []
        for role in ("main", "edges"):
            wf = self._orig_main if role == "main" else self._orig_edges
            if wf is None:
                continue
            graph = None
            for nid, node in wf.graph.items():
                if graph is not None and nid not in graph:
                    continue                      # retiré avec le ControlNet « model patch »
                cls = str(node.get("class_type"))
                field = LOADER_FIELDS.get(cls)
                val = node["inputs"].get(field) if field else None
                if not isinstance(val, str):
                    continue
                avail = options(cls, field)
                if avail is None or val in avail:
                    continue                      # liste illisible : ComfyUI jugera
                pick = None if str(chosen.get(field) or "").strip() else substitute_model(field, val, avail, family)
                if not pick and cls == "ModelPatchLoader" and role == "main":
                    graph = drop_model_patches(graph or wf.graph)
                    self.notes.append(tr("ControlNet Tile « {val} » absent de ComfyUI (models\\model_patches) : "
                                         "img2img sans ControlNet, denoise réduit. Avec lui, l'IA peut ajouter bien "
                                         "plus de détail sans redessiner les formes.", val=val))
                    continue
                if pick:
                    graph = graph or copy.deepcopy(wf.graph)
                    graph[nid]["inputs"][field] = pick
                    note = tr("modèle « {val} » absent de ComfyUI : « {pick} » utilisé à la place.", val=val, pick=pick)
                    if note not in self.notes:            # même modèle dans les deux workflows : une seule remarque
                        self.notes.append(note)
                elif role == "edges":
                    self._orig_edges = self.wf_edges = None
                    self._edges_off = True
                    self.notes.append(tr("workflow Tile+Canny désactivé : modèle « {val} » absent de ComfyUI (dossier "
                                         "models\\{folder}) ; toutes les matières passent par le workflow Tile.",
                                         val=val, folder=MODEL_DIRS.get(field, "")))
                    graph = None
                    break
                else:
                    missing.append((cls, field, val, avail))
            if graph is not None:
                new = self.chc.ComfyWorkflow(graph, wf.load_node_id, wf.output_node_ids, wf.preview_only)
                if role == "main":
                    self._orig_main = self.wf_main = new
                else:
                    self._orig_edges = self.wf_edges = new
        if not missing:
            return
        parts = []
        for _cls, field, val, avail in missing:
            have = (tr("dossier vide") if not avail else
                    tr("présents : {names}", names=", ".join(avail[:5]) + (" …" if len(avail) > 5 else "")))
            src = MODEL_SOURCES.get(val)
            parts.append(tr("« {val} » (models\\{folder}, {have})", val=val, folder=MODEL_DIRS.get(field, "?"),
                            have=have) + (f" — {tr(src)}" if src else ""))
        hint = ""
        if any(f == "ckpt_name" for _c, f, _v, _a in missing):
            unets = [u for u in (options("UNETLoader", "unet_name") or []) if "z_image" in u.lower() or "z-image" in u.lower()]
            if unets:
                hint = tr("ton ComfyUI n'a pas de checkpoint pour ce workflow mais il a Z-Image Turbo ({name}) : "
                          "choisis la famille « Z-Image Turbo » dans TEXFORGE, sans rien télécharger. ", name=unets[0])
        raise WorkflowFatal(hint + tr("modèles absents de ComfyUI : {items}. Après un ajout, redémarre ComfyUI puis "
                                      "relance.", items=" ; ".join(parts)))

    def _workflow_classes(self) -> set[str]:
        return {str(n.get("class_type")) for wf in (self._orig_main, self._orig_edges) if wf is not None
                for n in wf.graph.values()}

    def _refit(self, missing: set[str]) -> list[str]:
        """Workflows refaits depuis les originaux pour un serveur sans les nœuds `missing` ; renvoie les
        remarques nouvelles. Lève WorkflowFatal si un nœud indispensable du workflow principal manque."""
        if missing & {"ModelPatchLoader", "ZImageFunControlnet"}:
            # ComfyUI trop ancien pour le ControlNet de Z-Image : workflow sans lui plutôt qu'un arrêt
            wf = self._orig_main
            self._orig_main = self.chc.ComfyWorkflow(drop_model_patches(wf.graph), wf.load_node_id,
                                                     wf.output_node_ids, wf.preview_only)
            missing = missing - {"ModelPatchLoader", "ZImageFunControlnet"}
        available = self._workflow_classes() - missing
        main, b, required = fit_workflow(self._orig_main, available, self.chc)
        if required:
            packs = sorted({NODE_PACKS[c] for c in required if c in NODE_PACKS})
            raise WorkflowFatal(tr("nœuds absents de ComfyUI : {nodes}{packs}. Installe-les avec ComfyUI Manager "
                                   "(Custom Nodes Manager) ou mets ComfyUI à jour, redémarre-le, puis relance.",
                                   nodes=", ".join(sorted(required)),
                                   packs=f" ({', '.join(packs)})" if packs else ""))
        notes: list[str] = []
        bypassed = set(b)
        edges = None
        if self._orig_edges is not None:
            wf_edges, b, r = fit_workflow(self._orig_edges, available, self.chc)
            if r:
                # workflow « joints » inutilisable ici : toutes les matières passent par le workflow principal
                if not self._edges_off:
                    notes.append(tr("workflow Tile+Canny désactivé, nœuds absents : {nodes} ; toutes les "
                                    "matières passent par le workflow Tile.", nodes=", ".join(r)))
                self._edges_off = True
            else:
                edges = wf_edges
                bypassed |= set(b)
        new = bypassed - self._missing
        self.wf_main, self.wf_edges = main, edges
        self._missing |= missing
        if new & TILING_CLASSES:
            self.native_tiling = False                 # TEXFORGE prolonge les bords en boucle avant l'envoi
            self._warned |= {"TEXFORGE_TILING", "TEXFORGE_TILING_VAE"}
            notes.append(tr("nœuds de raccord absents ({nodes}, pack ComfyUI-seamless-tiling) : TEXFORGE "
                            "prolonge lui-même les textures répétables en boucle avant la génération. Raccord "
                            "natif : ComfyUI Manager → Custom Nodes Manager → « ComfyUI-seamless-tiling ».",
                            nodes=", ".join(sorted(bypassed & TILING_CLASSES))))
        if "ColorTransfer" in new:
            notes.append(tr("nœud ColorTransfer absent : couleurs recalées par TEXFORGE après la génération."))
        if "OffsetImage" in new:
            notes.append(tr("nœud OffsetImage absent : aperçu du raccord dans ComfyUI désactivé (sans effet "
                            "sur le résultat)."))
        self._set_denoise_scale()
        return notes

    def _note(self, text: str) -> None:
        self.notes.append(text)
        if self.on_note is not None:
            self.on_note(text)
        else:
            print(f"[COMFYUI] ⚠ {text}", file=sys.stderr)

    def _missing_after_refusal(self, exc, wf) -> bool:
        """Refus « nœud absent » d'un nœud facultatif : workflows refaits sans lui (et sans les autres
        nœuds de son pack). False si le refus a une autre cause, ou concerne un nœud indispensable."""
        cls = self.chc.missing_node_class(exc, wf.graph)
        if not cls or cls not in BYPASS:
            return False
        with self._fit_lock:
            if cls not in self._missing:
                pack = NODE_PACKS.get(cls)
                mates = {c for c, p in NODE_PACKS.items() if pack and p == pack} & self._workflow_classes()
                for text in self._refit(self._missing | {cls} | mates):
                    self._note(text)
        return True

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
                print(tr("  ! nœud {title} absent du workflow : réglage ignoré", title=title), file=sys.stderr)
            return False

    def build_context(self, rec: dict, an) -> dict:
        """Matériau, raccord et variante de workflow d'une texture."""
        material = (rec.get("analysis") or {}).get("material") or "generic"
        preset = {**self.cfg["materials"].get("generic", {}), **self.cfg["materials"].get(material, {})}
        tiling = ("enable" if an.tile_x and an.tile_y else "x_only" if an.tile_x
                  else "y_only" if an.tile_y else "disable")
        style = str(self.c.get("style") or "faithful")
        prompt = compose_prompt(self.c, (rec.get("hint") or "").strip(), str(preset.get("prompt", "")),
                                tiling != "disable")
        edges = bool(preset.get("edges")) and self.wf_edges is not None and style != "photoreal"
        top = 1.0 if style == "photoreal" else 0.9
        denoise = min(top, max(0.08, float(preset.get("denoise", 0.42)) * getattr(self, "denoise_scale", 1.0)))
        return {"material": material, "tiling": tiling, "edges": edges, "style": style,
                "prompt": prompt, "negative": str(self.c["negative"]),
                "denoise": round(denoise, 3), "cn_strength": float(preset.get("cn_strength", 0.75)),
                # affinage par morceaux après la passe globale : assez pour le grain, pas pour réinventer
                "refine_denoise": round(min(0.45, max(0.25, denoise * 0.5)), 3),
                "denoise_ladder": [float(f) for f in (self.c.get("denoise_ladder") or [])],
                "stem": rec.get("name") or "texture"}

    # ------------------------------------------------------------ génération
    def generate(self, image: Image.Image, refs: list[Path], prompt: str, negative: str,
                 aspect_ratio: str, seed: int, on_submitted: Callable[[str], None],
                 context: dict | None = None) -> tuple[Image.Image, str]:
        ctx = context or {}
        on_submitted(f"local-{uuid.uuid4().hex[:12]}")
        top = float(ctx.get("denoise") or 0.42)
        ladder = [float(f) for f in ctx.get("denoise_ladder") or []]
        # denoise automatique : un palier plus bas à chaque nouvel essai de la texture
        steps = [min(1.0, top * f) for f in ladder[min(int(ctx.get("attempt", 0)), len(ladder) - 1):]] if ladder else [top]
        ctx = {**ctx, "denoise": round(steps[0], 3)}
        tiles = ctx.get("tiles")
        if tiles and max(image.size) > int(tiles["size"]):
            if ctx.get("style") == "photoreal":
                # passe globale : le modèle voit toute l'image (sujet, matières, lumière) et réinvente au denoise
                # demandé ; les morceaux ne font ensuite qu'affiner, sinon chacun invente son propre motif
                image, d = self._global_pass(image, ctx, prompt, negative, seed, steps)
                ctx = {**ctx, "denoise": round(min(0.45, max(0.25, d * 0.5)), 3)}
            elif len(steps) > 1:
                # fidèle : le palier se choisit sur un essai à 1 mégapixel (quelques secondes) plutôt que sur
                # une génération complète par morceaux (plusieurs minutes)
                ctx = {**ctx, "denoise": self._probe_denoise(image, ctx, prompt, negative, seed, steps)}
            return self._generate_tiled(image, ctx, prompt, negative, seed, int(tiles["size"]),
                                        int(tiles["overlap"])), f"comfyui://{ctx.get('stem', 'texture')}"
        return self._generate_one(image, ctx, prompt, negative, seed), f"comfyui://{ctx.get('stem', 'texture')}"

    def _faithful_enough(self, ref: Image.Image, out: Image.Image) -> bool:
        """Validation de la génération (même seuils que pour la texture finale) sur une image réduite."""
        from .imaging import Analysis
        from .validate import evaluate
        w, h = ref.size
        an = Analysis(w, h, "none", False, False, 0.1, 0.1, False, "ai", "", "")
        return bool(evaluate(ref.convert("RGBA"), out.convert("RGBA").resize((w, h), Image.LANCZOS), an,
                             self.cfg).get("passed"))

    def _probe_denoise(self, image: Image.Image, ctx: dict, prompt: str, negative: str, seed: int,
                       steps: list[float]) -> float:
        small = _megapixel(image)
        for i, d in enumerate(steps):
            if self.backend.on_progress is not None:
                self.backend.on_progress(tr("{stem} : essai du denoise {d:.2f}", stem=ctx.get("stem", "texture"), d=d))
            out = self._generate_one(small, {**ctx, "denoise": round(d, 3), "tiling": "disable",
                                             "stem": f"{ctx.get('stem', 'texture')}_p{i}"}, prompt, negative, seed,
                                     tiled=True)
            if self._faithful_enough(small, out):
                return round(d, 3)
        return round(steps[-1], 3)

    def _global_pass(self, image: Image.Image, ctx: dict, prompt: str, negative: str, seed: int,
                     steps: list[float] | None = None) -> tuple[Image.Image, float]:
        """Image entière à environ 1 mégapixel, générée au denoise demandé, puis remise à la taille de `image`
        (Real-ESRGAN si disponible, sinon Lanczos) pour servir de base aux morceaux. Avec plusieurs paliers
        (denoise automatique), le premier qui passe le garde-fou est gardé. Renvoie (base, denoise retenu)."""
        W, H = image.size
        small = _megapixel(image)
        steps = steps or [float(ctx.get("denoise") or 0.8)]
        for i, d in enumerate(steps):
            if self.backend.on_progress is not None:
                self.backend.on_progress(tr("{stem} : passe globale", stem=ctx.get("stem", "texture"))
                                         + (f" ({d:.2f})" if len(steps) > 1 else ""))
            out = self._generate_one(small, {**ctx, "denoise": round(d, 3), "tiling": "disable",
                                             "stem": f"{ctx.get('stem', 'texture')}_g{i}"}, prompt, negative, seed,
                                     tiled=True)
            if len(steps) == 1 or i == len(steps) - 1 or self._faithful_enough(small, out):
                break
        from .imaging import _esrgan
        lc = self.cfg.get("local", {})
        up = _esrgan(out, lc.get("esrgan_exe", ""), lc.get("esrgan_model", "realesrgan-x4plus"))
        return (up or out).convert("RGB").resize((W, H), Image.LANCZOS), d

    def _generate_tiled(self, image: Image.Image, ctx: dict, prompt: str, negative: str, seed: int,
                        size: int, overlap: int) -> Image.Image:
        """Image déjà à sa taille finale, générée morceau par morceau puis fondue : chaque morceau garde la
        résolution native du modèle, donc le détail ajouté l'est à la taille finale."""
        from .imaging import tile_starts, tile_weights
        arr = np.asarray(image.convert("RGB"))
        H, W = arr.shape[:2]
        ys, xs = tile_starts(H, size, overlap), tile_starts(W, size, overlap)
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        n = len(ys) * len(xs)
        stem = str(ctx.get("stem", "texture"))
        # raccord géré par la marge en boucle et le fondu final : jamais de padding circulaire par morceau
        tctx = {**ctx, "tiling": "disable"}
        for i, (y, x) in enumerate(((y, x) for y in ys for x in xs), 1):
            th, tw = min(size, H - y), min(size, W - x)
            piece = Image.fromarray(np.ascontiguousarray(arr[y:y + th, x:x + tw]), "RGB")
            if self.backend.on_progress is not None:
                self.backend.on_progress(tr("{stem} : morceau {i}/{n}", stem=stem, i=i, n=n))
            out = self._generate_one(piece, {**tctx, "stem": f"{stem}_m{i:03d}"}, prompt, negative, seed,
                                     tiled=True)
            if out.size != (tw, th):
                out = out.resize((tw, th), Image.LANCZOS)
            wgt = tile_weights(th, tw, overlap, (y > 0, y + th < H, x > 0, x + tw < W))
            acc[y:y + th, x:x + tw] += np.asarray(out, np.float32) * wgt
            wsum[y:y + th, x:x + tw] += wgt
        out = np.clip(acc / np.maximum(wsum, 1e-6) + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")

    def _generate_one(self, image: Image.Image, ctx: dict, prompt: str, negative: str, seed: int,
                      tiled: bool = False) -> Image.Image:
        wanted = {
            ("TEXFORGE_PROMPT", "text"): ctx.get("prompt", prompt),
            ("TEXFORGE_NEGATIVE", "text"): ctx.get("negative", negative),
            ("TEXFORGE_SAMPLER", "seed"): int(seed),
            ("TEXFORGE_SAMPLER", "denoise"): ctx.get("denoise"),
            ("TEXFORGE_CN_TILE", "strength"): ctx.get("cn_strength"),
            ("TEXFORGE_TILING", "tiling"): ctx.get("tiling"),
            ("TEXFORGE_TILING_VAE", "tiling"): ctx.get("tiling"),
        }
        native = self.native_tiling and not tiled   # préparation de l'image reçue (pipeline.do_ai)
        last: Exception | None = None
        for _ in range(len(BYPASS) + 1):
            wf = self.wf_edges if ctx.get("edges") and self.wf_edges is not None else self.wf_main
            overrides = {k: v for k, v in wanted.items() if v is not None and self._has(wf, k[0])}
            try:
                blobs = self._call(self.backend.generate(image.convert("RGB"), str(ctx.get("stem", "texture")),
                                                         overrides=overrides, workflow=wf),
                                   timeout=float(self.c["timeout"]) + 120)
                break
            except self.chc.ServerUnavailable as exc:
                raise ServerDown(str(exc)) from exc
            except self.chc.WorkflowError as exc:
                if not self._missing_after_refusal(exc, wf):
                    cls = self.chc.missing_node_class(exc, wf.graph)
                    if cls:
                        pack = f" ({NODE_PACKS[cls]})" if cls in NODE_PACKS else ""
                        raise WorkflowFatal(tr("{exc} — type de nœud absent : {cls}{pack}. Installe-le avec ComfyUI "
                                               "Manager (Custom Nodes Manager) ou mets ComfyUI à jour, redémarre-le, "
                                               "puis relance.",
                                               exc=exc, cls=cls, pack=pack)) from exc
                    raise WorkflowFatal(str(exc)) from exc
                if native and not self.native_tiling:
                    # image préparée pour le raccord natif : à refaire, bords prolongés par TEXFORGE
                    raise TransientError(tr("nœuds de raccord absents de ComfyUI : contournés, texture relancée "
                                            "avec le raccord préparé par TEXFORGE")) from exc
                last = exc
                continue                     # autre nœud facultatif contourné : nouvel envoi
            except self.chc.TextureError as exc:
                if exc.kind in ("oom", "timeout", "interrupted", "http"):
                    raise TransientError(f"[{exc.kind}] {exc}") from exc
                raise ProviderError(f"[{exc.kind}] {exc}") from exc
            except TimeoutError as exc:
                raise TransientError(tr("[timeout] pas de réponse de la boucle ComfyUI : {exc}", exc=exc)) from exc
        else:
            raise WorkflowFatal(str(last))
        if self.on_pass is not None:
            self.on_pass(ctx.get("tid"))
        img = Image.open(io.BytesIO(blobs[0]))
        img.load()
        return img.convert("RGB")

    def resume(self, request_id: str):
        raise ProviderError(tr("les requêtes locales ne se reprennent pas"))


# Chargeurs de modèles : champ du fichier et dossier de ComfyUI (models\<dossier>)
LOADER_FIELDS = {"CheckpointLoaderSimple": "ckpt_name", "CheckpointLoader": "ckpt_name",
                 "ControlNetLoader": "control_net_name", "UNETLoader": "unet_name", "CLIPLoader": "clip_name",
                 "VAELoader": "vae_name", "LoraLoader": "lora_name", "LoraLoaderModelOnly": "lora_name",
                 "ModelPatchLoader": "name", "UpscaleModelLoader": "model_name"}
MODEL_DIRS = {"ckpt_name": "checkpoints", "control_net_name": "controlnet", "unet_name": "diffusion_models",
              "clip_name": "text_encoders", "vae_name": "vae", "lora_name": "loras", "name": "model_patches",
              "model_name": "upscale_models"}
MODEL_SOURCES = {
    "juggernautXL_ragnarok.safetensors": N_("n'importe quel checkpoint SDXL convient, ex. huggingface.co/RunDiffusion/"
                                            "Juggernaut-XL-v9 (7,1 Go)"),
    "xinsir-controlnet-tile-sdxl-1.0.safetensors": N_("huggingface.co/xinsir/controlnet-tile-sdxl-1.0 "
                                                      "(diffusion_pytorch_model.safetensors, 2,5 Go, à renommer ainsi)"),
    "xinsir-controlnet-union-sdxl-1.0-promax.safetensors": N_("huggingface.co/xinsir/controlnet-union-sdxl-1.0 "
                                                              "(diffusion_pytorch_model_promax.safetensors, 2,5 Go, à "
                                                              "renommer ainsi)"),
    "realisticVisionV60B1_v51VAE.safetensors": N_("n'importe quel checkpoint SD 1.5 convient"),
    "control_v11f1e_sd15_tile.pth": N_("huggingface.co/lllyasviel/ControlNet-v1-1"),
    "z_image_turbo_bf16.safetensors": N_("huggingface.co/Comfy-Org/z_image_turbo"),
    "qwen_3_4b.safetensors": N_("huggingface.co/Comfy-Org/z_image_turbo"),
    "ae.safetensors": N_("huggingface.co/Comfy-Org/z_image_turbo"),
}
_PRECISION = re.compile(r"[_-](bf16|fp16|fp32|fp8(_e4m3fn|_e5m2)?(_scaled)?|int8|q\d(_k)?(_[sm])?|q8_0)$")


def _model_stem(name: str) -> str:
    stem = re.sub(r"\.(safetensors|ckpt|pth|pt|bin|gguf|sft)$", "", Path(name.replace("\\", "/")).name.lower())
    return _PRECISION.sub("", stem)


def substitute_model(field: str, wanted: str, available: list[str], family: str) -> str | None:
    """Équivalent présent d'un modèle par défaut absent, ou None : checkpoint de la même famille (SDXL / SD 1.5),
    ControlNet du même type (tile, union, canny) et de la même famille, ou autre précision du même modèle."""
    low = {a: a.lower() for a in available}
    if field == "ckpt_name":
        if family == "sdxl":
            cands = [a for a, s in low.items() if "xl" in s and "refiner" not in s]
        elif family == "sd15":
            cands = [a for a, s in low.items() if "xl" not in s]
        else:
            cands = []
        return sorted(cands, key=lambda a: ("juggernaut" not in low[a], low[a]))[0] if cands else None
    if field == "control_net_name":
        kind = next((k for k in ("tile", "union", "canny") if k in wanted.lower()), "")
        if not kind:
            return None
        cands = [a for a, s in low.items() if kind in s and (family not in ("sdxl", "sd15") or ("xl" in s) == (family == "sdxl"))]
        return sorted(cands)[0] if cands else None
    if field in ("unet_name", "clip_name", "vae_name", "name"):
        cands = [a for a in available if _model_stem(a) == _model_stem(wanted)]
        return sorted(cands)[0] if cands else None
    return None


# Nœuds facultatifs du workflow : sortie = entrée indiquée quand le serveur ne les a pas
BYPASS = {"SeamlessTile": "model", "MakeCircularVAE": "vae", "ColorTransfer": "image_target", "OffsetImage": "pixels",
          "ZImageFunControlnet": "model"}
CONTROLNET_APPLY = {"ControlNetApplyAdvanced", "ControlNetApply", "ZImageFunControlnet", "QwenImageDiffsynthControlnet"}


def compose_prompt(c: dict, hint: str, material_prompt: str, tiled: bool) -> str:
    """Prompt d'une texture : préfixe du rendu, indice de la règle, « Prompt à donner », prompt du matériau
    (rendu Fidèle seulement), suffixe. Avec « prompt seul », le texte de l'utilisateur est envoyé tel quel."""
    user = " ".join(str(c.get("subject") or "").split())          # retours à la ligne -> espaces
    if c.get("prompt_only") and user:
        return ", ".join(x for x in (hint, user) if x)
    # photoréaliste : pas de prompt de matériau (détection faite pour des textures, trompeuse sur une image)
    material = "" if str(c.get("style") or "faithful") == "photoreal" else material_prompt
    body = ", ".join(x for x in (hint, user, material) if x) or "detailed surface"
    prefix = c.get("prompt_prefix_tiled", "") if tiled else c.get("prompt_prefix", "")
    return f"{prefix}{body}{c.get('prompt_suffix', '')}"


def _megapixel(image: Image.Image, pixels: int = 1024 * 1024) -> Image.Image:
    """Réduction à environ 1 mégapixel (résolution native des modèles) pour une passe sur l'image entière."""
    W, H = image.size
    s = min(1.0, (pixels / (W * H)) ** 0.5)
    return image.convert("RGB").resize((max(64, round(W * s)), max(64, round(H * s))), Image.LANCZOS)


def drop_model_patches(graph: dict) -> dict:
    """Retire les ControlNet « model patch » (Z-Image) d'un graphe : chargeur supprimé, nœud d'application
    contourné (ce qui lisait son MODEL lit celui d'avant). Sert quand le fichier ou les nœuds manquent."""
    g = copy.deepcopy(graph)
    for nid in [n for n, node in g.items() if node.get("class_type") == "ModelPatchLoader"]:
        for aid in [a for a, node in g.items() if any(isinstance(v, list) and len(v) == 2 and str(v[0]) == str(nid)
                                                       for v in node["inputs"].values())]:
            src = g[aid]["inputs"].get(BYPASS.get(g[aid].get("class_type"), ""))
            if isinstance(src, list) and len(src) == 2:
                for other in g.values():
                    for key, val in other["inputs"].items():
                        if isinstance(val, list) and len(val) == 2 and str(val[0]) == str(aid):
                            other["inputs"][key] = list(src)
            del g[aid]
        del g[nid]
    return g
TILING_CLASSES = {"SeamlessTile", "MakeCircularVAE"}
NODE_PACKS = {"SeamlessTile": "ComfyUI-seamless-tiling", "MakeCircularVAE": "ComfyUI-seamless-tiling",
              "OffsetImage": "ComfyUI-seamless-tiling"}


def fit_workflow(wf, available: set[str], chc):
    """(workflow, nœuds contournés, nœuds indispensables absents) pour un serveur qui ne connaît que
    `available`. Un nœud contourné est retiré ; ce qui lisait sa sortie lit son entrée à la place."""
    graph = copy.deepcopy(wf.graph)
    bypassed: list[str] = []
    required: list[str] = []
    for nid in list(graph):
        node = graph[nid]
        ct = node.get("class_type")
        if ct in available:
            continue
        src = node["inputs"].get(BYPASS.get(ct, ""))
        if not (isinstance(src, list) and len(src) == 2):
            required.append(ct)
            continue
        for other in graph.values():
            for key, val in other["inputs"].items():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == str(nid):
                    other["inputs"][key] = list(src)
        del graph[nid]
        bypassed.append(ct)
    if not bypassed:
        return wf, [], sorted(set(required))
    if "OffsetImage" in bypassed:              # l'aperçu du raccord n'a plus rien à montrer
        for nid in [n for n, node in graph.items()
                    if str((node.get("_meta") or {}).get("title", "")) == "TEXFORGE_SEAM_PREVIEW"]:
            del graph[nid]
    outs = [o for o in wf.output_node_ids if o in graph]
    return chc.ComfyWorkflow(graph, wf.load_node_id, outs, wf.preview_only), sorted(set(bypassed)), sorted(set(required))


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
            raise chc.WorkflowError(tr("checkpoint choisi mais aucun nœud TEXFORGE_CHECKPOINT unique dans le workflow"))
        graph[ck]["inputs"]["ckpt_name"] = ckpt
    if cn:
        cid = by_title("TEXFORGE_CN_TILE_MODEL", ("ControlNetLoader",))
        if cid is None:
            raise chc.WorkflowError(tr("modèle ControlNet choisi mais aucun nœud TEXFORGE_CN_TILE_MODEL unique"))
        graph[cid]["inputs"]["control_net_name"] = cn
    if lora:
        strength = float(c.get("lora_strength", 0.8))
        existing = by_title("TEXFORGE_LORA", ())
        if existing is not None:
            graph[existing]["inputs"].update(lora_name=lora, strength_model=strength, strength_clip=strength)
        else:
            if ck is None:
                raise chc.WorkflowError(tr("LoRA choisie mais aucun nœud TEXFORGE_CHECKPOINT unique pour l'insérer"))
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
    raise FatalError(tr("fournisseur inconnu : {name} (attendu : comfyui, esrgan, mock ou higgsfield)", name=name))
