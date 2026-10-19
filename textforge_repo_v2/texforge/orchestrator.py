"""Orchestrateur : enchaîne les trois pipelines et pilote le GPU.

    1. UI       graphismes plats -> Real-ESRGAN / Waifu2x / Lanczos, par lots (un lancement par lot)
    2. LOCAL    données techniques -> Lanczos
    3. COMFYUI  matériaux -> file ComfyUI, par groupes de workflow et lots de VRAM

Garanties :
  - une texture en échec n'arrête pas le lot : elle est consignée (table failures +
    events.jsonl), retentée si l'échec est passager (OOM, délai, coupure), sinon remplacée
    par un repli (Real-ESRGAN général, puis Lanczos) ;
  - la VRAM est contrôlée entre les lots et libérée au changement de workflow, sur OOM,
    sous le seuil configuré et toutes les N textures ;
  - si ComfyUI tombe, l'orchestrateur attend son retour, puis applique le repli ;
  - un fichier STOP dans le dossier de travail arrête proprement l'exécution (pont Node.js).
"""

from __future__ import annotations

import fnmatch
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .events import EventBus
from .imaging import local_upscale, save_png
from .pipeline import BudgetReached, Runner, analysis_of, style_refs
from .providers import FatalError, ServerDown, WorkflowFatal, make_provider
from .ui_upscale import UiJob, UiUpscaler

STAGES = ("ui", "local", "ai")


def _kind_of(message: str) -> str:
    m = message.lower()
    for kind in ("oom", "timeout", "interrupted", "http", "cuda", "workflow"):
        if f"[{kind}]" in m:
            return kind
    if "out of memory" in m:
        return "oom"
    if "structure" in m or "décalage" in m or "raccord" in m or "zone modifiée" in m or "couleur locale" in m:
        return "validation"
    return "error"


class Orchestrator:
    def __init__(self, cfg: dict, events: EventBus, *, provider_name: str | None = None, provider=None) -> None:
        self.cfg = cfg
        self.ev = events
        self.runner = Runner(cfg, provider_name, log=lambda m: events.emit("info", message=str(m).strip()),
                             provider=provider)
        self.state = self.runner.state
        self.provider_name = (provider_name or cfg["provider"]["name"]).lower()
        self.stop_file = Path(cfg["_work"]) / "STOP"
        self.stats: Counter = Counter()
        self._lock = threading.Lock()
        self._sent = 0
        self._positions: dict[str, int] = {}
        self._since_free = 0
        self.provider: Any = None
        self._bar = None

    # ------------------------------------------------------------------ outils
    def stop_requested(self) -> bool:
        if self.stop_file.exists() and not self.runner.stop.is_set():
            self.runner.stop.set()
            self.ev.emit("stopped", message="fichier STOP détecté")
        return self.runner.stop.is_set()

    def _fail(self, rec: dict, stage: str, kind: str, message: str, action: str, details: str = "") -> None:
        route = (rec.get("analysis") or {}).get("route", "")
        self.state.add_failure(rec["id"], stage, route, kind, message, details, action)
        if action == "abandon":
            self.ev.emit("failed", texture=rec["id"], name=rec["name"], stage=stage, kind=kind, message=message)

    def _chunks(self, items: list, size: int) -> list[list]:
        size = max(1, int(size))
        return [items[i:i + size] for i in range(0, len(items), size)]

    # ------------------------------------------------------------------ exécution
    def run(self, *, only: str | None = None, limit: int | None = None, stages: tuple[str, ...] = STAGES,
            redo_stale: bool = False) -> dict:
        t0 = time.monotonic()
        if self.stop_file.exists():
            self.stop_file.unlink()
        rows = self.state.all("status='pending' OR status='error'")
        if redo_stale:
            rows += [r for r in self.state.all("status IN ('done','fallback')")
                     if r["mode"] == "ai" and r.get("fingerprint") != self.runner.fp and not r.get("reviewed")]
        if only:
            pat = only.lower()
            rows = [r for r in rows if fnmatch.fnmatch(r["name"].lower(), pat) or fnmatch.fnmatch(r["id"].lower(), pat)]
        by_mode: dict[str, list[dict]] = {m: [] for m in ("ui", "upscale", "ai")}
        for r in rows:
            if r["mode"] in by_mode:
                by_mode[r["mode"]].append(r)
        ai_rows = by_mode["ai"][:limit] if limit is not None else by_mode["ai"]
        total = len(by_mode["ui"]) * ("ui" in stages) + len(by_mode["upscale"]) * ("local" in stages) + \
            len(ai_rows) * ("ai" in stages)
        self.ev.emit("run_start", total=total, provider=self.provider_name, ui=len(by_mode["ui"]),
                     local=len(by_mode["upscale"]), ai=len(ai_rows))
        try:
            if "ui" in stages and not self.stop_requested():
                self._stage_ui(by_mode["ui"])
            if "local" in stages and not self.stop_requested():
                self._stage_local(by_mode["upscale"])
            if "ai" in stages and not self.stop_requested():
                self._stage_ai(ai_rows)
        except KeyboardInterrupt:
            self.runner.stop.set()
            if self.provider is not None and hasattr(self.provider, "cancel_pending"):
                self.provider.cancel_pending()
            self.ev.emit("stopped", message="interruption clavier")
            raise
        finally:
            if self.provider is not None and hasattr(self.provider, "close"):
                self.provider.close()
            if self._bar is not None:
                self._bar.close()
            s = self.stats
            summary = (f"{s['ui']} UI, {s['local']} local, {s['ai']} ComfyUI validées, {s['fallback']} replis, "
                       f"{s['retried']} reprises, {s['errors']} échecs")
            self.ev.emit("run_end", seconds=round(time.monotonic() - t0, 1), summary=summary, **dict(s))
            self.state.close()
        result = dict(self.stats)
        for key in ("ui", "local", "ai", "fallback", "transient", "errors", "retried"):
            result.setdefault(key, 0)
        result.setdefault("budget", False)
        result["requests_used"] = self.runner.budget.used
        return result

    # ------------------------------------------------------------------ étape UI
    def _stage_ui(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.ev.emit("stage_start", stage="ui", count=len(rows))
        up = UiUpscaler(self.cfg, log=lambda m: self.ev.emit("info", message=m))
        self.ev.emit("info", message=f"[UI] moteurs disponibles : {', '.join(up.available())}")
        index = 0
        for chunk in self._chunks(rows, self.cfg["ui_upscale"].get("batch_size", 64)):
            if self.stop_requested():
                break
            jobs: list[UiJob] = []
            recs = {}
            for r in chunk:
                try:
                    src, an, k = self.runner._load(r)
                except Exception as exc:
                    self.state.update(r["id"], status="error", error=f"image illisible : {exc}")
                    self._fail(r, "ui", "image", f"image illisible : {exc}", "abandon")
                    self.stats["errors"] += 1
                    continue
                engine = str((r.get("analysis") or {}).get("ui_engine", ""))
                jobs.append(UiJob(r["id"], r["name"], src.convert("RGBA"), k, an, engine=engine))
                recs[r["id"]] = r
            results = up.run(jobs)
            for j in jobs:
                index += 1
                r, res = recs[j.tid], results[j.tid]
                if res.image is None:
                    msg = "; ".join(res.errors) or "aucun moteur n'a abouti"
                    self.state.update(j.tid, status="error", error=msg)
                    self._fail(r, "ui", "engine", msg, "abandon")
                    self.stats["errors"] += 1
                    continue
                p = self.runner._paths(j.tid)
                save_png(res.image, p["result"])
                save_png(res.image, p["local"])
                self.state.update(j.tid, status="done", choice="local", result_path=str(p["result"]),
                                  local_path=str(p["local"]), error=None, fingerprint=self.runner.fp,
                                  metrics={"method": f"ui:{res.engine}", "k": j.k})
                if res.errors:     # un moteur précédent a échoué : tracé, mais résolu
                    self._fail(r, "ui", "engine", "; ".join(res.errors), "fallback")
                self.stats["ui"] += 1
                self.ev.emit("ui_done", texture=j.tid, name=j.name, engine=res.engine, k=j.k,
                             kind=(r.get("analysis") or {}).get("ui_kind", ""), index=index, total=len(rows))
        self.ev.emit("stage_end", stage="ui", summary=f"{self.stats['ui']} texture(s) UI traitée(s)")

    # ------------------------------------------------------------------ étape locale
    def _stage_local(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.ev.emit("stage_start", stage="local", count=len(rows))
        for i, r in enumerate(rows, 1):
            if self.stop_requested():
                break
            try:
                self.runner.do_local(r)
                self.stats["local"] += 1
                k = (self.state.get(r["id"]) or {}).get("metrics", {}).get("k")
                self.ev.emit("local_done", texture=r["id"], name=r["name"], engine="lanczos", k=k,
                             index=i, total=len(rows))
            except Exception as exc:
                self.state.update(r["id"], status="error", error=str(exc))
                self._fail(r, "local", "error", str(exc), "abandon")
                self.stats["errors"] += 1
        self.ev.emit("stage_end", stage="local", summary=f"{self.stats['local']} texture(s) agrandie(s)")

    # ------------------------------------------------------------------ étape IA
    def _open_provider(self) -> bool:
        if self.runner._injected is not None:
            self.provider = self.runner._injected
            return True
        kwargs = {}
        if self.provider_name == "comfyui":
            kwargs["on_progress"] = self._on_progress
        deadline = time.monotonic() + float(self.cfg["comfyui"].get("server_wait", 120))
        while True:
            try:
                self.provider = make_provider(self.cfg, self.state, self.provider_name, **kwargs)
                info = getattr(self.provider, "info", "")
                if info:
                    self.ev.emit("info", message=f"[COMFYUI] {info}")
                return True
            except ServerDown as exc:
                policy = self.cfg["comfyui"].get("on_server_down", "wait")
                self.ev.emit("server_down", message=str(exc), policy=policy)
                if policy != "wait" or time.monotonic() > deadline or self.stop_requested():
                    return False
                time.sleep(5)

    def _on_progress(self, text: str) -> None:
        self.ev.emit("comfy_progress", message=text)
        if self._bar is not None:
            self._bar.set_postfix_str(text[:60], refresh=True)

    def _stage_ai(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.ev.emit("stage_start", stage="comfyui" if self.provider_name == "comfyui" else self.provider_name,
                     count=len(rows))
        c = self.cfg["comfyui"]
        fb = UiUpscaler(self.cfg, model=str(c.get("fallback_model", "realesrgan-x4plus")),
                        engines=list(c.get("fallback_engines", ["realesrgan", "lanczos"])))
        self._fallback = fb
        self.runner.fallback_image = lambda src, an, k: fb.upscale_one(src, an, k)
        try:
            ok = self._open_provider()
        except WorkflowFatal as exc:
            self.ev.emit("failed", name="(workflow)", kind="workflow", message=str(exc))
            self.stats["errors"] += 1
            return
        if not ok:
            policy = c.get("on_server_down", "wait")
            if policy == "abort":
                self.ev.emit("stopped", message="ComfyUI injoignable : textures laissées en attente")
                return
            self._fallback_many(rows, "ComfyUI injoignable")
            return
        self.runner.provider = self.provider
        self.runner.refs = style_refs(self.cfg) if self.provider_name == "higgsfield" else []
        local_gpu = getattr(self.provider, "native_tiling", False)
        try:
            from tqdm import tqdm
            self._bar = tqdm(total=len(rows), desc=self.provider.name, unit="tex", disable=None, leave=False)
        except Exception:
            self._bar = None

        # Groupes : même workflow puis même mode de raccord -> moins de rechargements de modèles
        if local_gpu:
            contexts = {r["id"]: self.provider.build_context(r, analysis_of(r)) for r in rows}
            keyed: dict[tuple, list[dict]] = {}
            for r in rows:
                ctx = contexts[r["id"]]
                keyed.setdefault((ctx["edges"], ctx["tiling"]), []).append(r)
            groups = sorted(keyed.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        else:
            contexts = {}
            groups = [((False, ""), rows)]
        workers = int(c.get("max_in_flight", 2)) if local_gpu else int(self.cfg["provider"].get("workers", 3))
        batch_size = int(c.get("batch_size", 12)) if local_gpu else len(rows)
        total = len(rows)
        retries: Counter = Counter()
        consecutive = 0
        prev_edges = None

        for (edges, tiling), grows in groups:
            if self.stop_requested():
                break
            if (local_gpu and prev_edges is not None and edges != prev_edges and self._since_free > 0
                    and c.get("free_between_groups", True)):      # inutile si la VRAM vient d'être libérée
                self._free("changement de workflow (Tile ↔ Tile+Canny)")
            prev_edges = edges
            self.ev.emit("info", message=f"[COMFYUI] groupe : workflow {'Tile+Canny' if edges else 'Tile'}, "
                                         f"raccord {tiling or '-'}, {len(grows)} texture(s)")
            queue = list(grows)
            while queue and not self.stop_requested():
                chunk, queue = queue[:batch_size], queue[batch_size:]
                outcomes = self._run_chunk(chunk, workers, total, contexts)
                requeue: list[dict] = []
                server_down = False
                oom = False
                for rec, outcome, info, secs in outcomes:
                    if outcome == "ai":
                        consecutive = 0
                        self.stats["ai"] += 1
                        cur = self.state.get(rec["id"]) or {}
                        m = cur.get("metrics") or {}
                        self.ev.emit("comfy_done", texture=rec["id"], name=rec["name"], structure=m.get("structure"),
                                     attempt=m.get("attempt"), seconds=round(secs, 1),
                                     material=contexts.get(rec["id"], {}).get("material", ""))
                    elif outcome == "fallback":
                        consecutive = 0
                        self.stats["fallback"] += 1
                        cur = self.state.get(rec["id"]) or {}
                        reason = cur.get("error") or ""
                        engine = (cur.get("metrics") or {}).get("fallback_engine", "lanczos")
                        self._fail(rec, "comfyui", _kind_of(reason), reason, "fallback")
                        self.ev.emit("fallback", texture=rec["id"], name=rec["name"], engine=engine, reason=reason[:160])
                    elif outcome == "transient":
                        cur = self.state.get(rec["id"]) or {}
                        msg = cur.get("error") or info
                        kind = _kind_of(msg)
                        oom = oom or kind == "oom"
                        retries[rec["id"]] += 1
                        if retries[rec["id"]] <= int(c.get("max_transient_retries", 2)):
                            self.stats["retried"] += 1
                            self._fail(rec, "comfyui", kind, msg, "retry")
                            self.ev.emit("retry", texture=rec["id"], name=rec["name"], message=msg[:160],
                                         attempt=retries[rec["id"]], kind=kind)
                            requeue.append(self.state.get(rec["id"]) or rec)
                        else:
                            self._fallback_many([rec], f"échecs passagers répétés : {msg}")
                    elif outcome == "server_down":
                        server_down = True
                        requeue.append(rec)
                    elif outcome == "workflow":
                        self._fail(rec, "comfyui", "workflow", info, "abandon")
                        self.ev.emit("stopped", message=f"workflow refusé par ComfyUI, étape arrêtée : {info}")
                        self.stats["errors"] += 1
                        return
                    elif outcome == "budget":
                        self.stats["budget"] = True
                        self.ev.emit("stopped", message="plafond de requêtes payantes atteint")
                        return
                    elif outcome == "stopped":
                        pass
                    else:   # error / fatal
                        consecutive += 1
                        self.stats["errors"] += 1
                        self._fail(rec, "comfyui", _kind_of(info), info, "abandon")
                if self._bar is not None:
                    self._bar.update(sum(1 for _, o, _, _ in outcomes if o in ("ai", "fallback", "error", "fatal")))
                if consecutive >= int(c.get("max_consecutive_failures", 8)):
                    self.ev.emit("stopped", message=f"{consecutive} échecs consécutifs : vérifie ComfyUI et le workflow")
                    return
                if server_down:
                    if self._wait_server():
                        queue = requeue + queue
                        continue
                    if c.get("on_server_down", "wait") == "abort":
                        self.ev.emit("stopped", message="ComfyUI injoignable : textures restantes laissées en attente")
                        return
                    seen: set[str] = set()
                    rest: list[dict] = []
                    for _, gr in groups:
                        for r in gr:
                            cur = self.state.get(r["id"])
                            if cur and cur["status"] in ("pending", "error") and r["id"] not in seen:
                                seen.add(r["id"])
                                rest.append(cur)
                    self._fallback_many(rest, "ComfyUI injoignable")
                    return
                queue = queue + requeue
                if oom and local_gpu:
                    self._free("mémoire saturée (OOM) avant nouvel essai")
                elif local_gpu:
                    self._housekeeping(len(chunk))
        self.ev.emit("stage_end", stage="comfyui",
                     summary=f"{self.stats['ai']} validée(s), {self.stats['fallback']} repli(s), "
                             f"{self.stats['retried']} reprise(s)")

    def _run_chunk(self, chunk: list[dict], workers: int, total: int,
                   contexts: dict[str, dict]) -> list[tuple[dict, str, str, float]]:
        results: list[tuple[dict, str, str, float]] = []

        def work(rec: dict) -> tuple[dict, str, str, float]:
            if self.stop_requested():
                return rec, "stopped", "", 0.0
            with self._lock:      # une texture reprise garde son numéro
                if rec["id"] not in self._positions:
                    self._sent += 1
                    self._positions[rec["id"]] = self._sent
                n = self._positions[rec["id"]]
            ctx = contexts.get(rec["id"], {})
            self.ev.emit("comfy_queued", texture=rec["id"], name=rec["name"], index=n, total=total,
                         material=ctx.get("material", (rec.get("analysis") or {}).get("material", "")),
                         tiling=ctx.get("tiling", ""), workflow="edges" if ctx.get("edges") else "tile")
            t0 = time.monotonic()
            try:
                outcome = self.runner.do_ai(rec)
                return rec, outcome, "", time.monotonic() - t0
            except ServerDown as exc:
                self.state.update(rec["id"], error=str(exc))
                return rec, "server_down", str(exc), time.monotonic() - t0
            except WorkflowFatal as exc:
                return rec, "workflow", str(exc), time.monotonic() - t0
            except BudgetReached:
                return rec, "budget", "", time.monotonic() - t0
            except FatalError as exc:
                self.state.update(rec["id"], status="error", error=str(exc))
                return rec, "fatal", str(exc), time.monotonic() - t0
            except Exception as exc:   # bug inattendu sur une texture : consigné, le lot continue
                msg = f"{type(exc).__name__}: {exc}"
                self.state.update(rec["id"], status="error", error=msg)
                return rec, "error", msg, time.monotonic() - t0

        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = [pool.submit(work, r) for r in chunk]
        pending = set(futures)
        try:
            while pending:       # attente par tranches : Ctrl+C reste réactif sous Windows
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for f in done:
                    results.append(f.result())
        except KeyboardInterrupt:
            self.runner.stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return results

    # ------------------------------------------------------------------ GPU
    def _free(self, reason: str) -> None:
        if self.provider is None or not hasattr(self.provider, "free_vram"):
            return
        free = self.provider.vram_free() if hasattr(self.provider, "vram_free") else None
        try:
            self.provider.free_vram()
            self._since_free = 0
            self.ev.emit("gpu_free", reason=reason, vram_free_gb=None if free is None else round(free / 1024 ** 3, 2))
        except ServerDown as exc:
            self.ev.emit("server_down", message=f"libération de la VRAM impossible : {exc}")

    def _housekeeping(self, processed: int) -> None:
        c = self.cfg["comfyui"]
        self._since_free += processed
        every = int(c.get("free_every", 0) or 0)
        if every and self._since_free >= every:
            self._free(f"{self._since_free} textures depuis la dernière libération")
            return
        policy = c.get("free_vram", "auto")
        if policy == "always":
            self._free("fin de lot")
        elif policy == "auto":
            free = self.provider.vram_free()
            if free is not None:
                self.ev.emit("gpu_status", vram_free_gb=round(free / 1024 ** 3, 2))
                if free < float(c.get("vram_threshold_gb", 1.5)) * 1024 ** 3:
                    self._free(f"VRAM libre {free / 1024 ** 3:.1f} Go sous le seuil")

    def _wait_server(self) -> bool:
        c = self.cfg["comfyui"]
        if c.get("on_server_down", "wait") != "wait":
            return False
        self.ev.emit("server_down", message=f"attente du redémarrage ({c.get('server_wait', 120)} s max)")
        deadline = time.monotonic() + float(c.get("server_wait", 120))
        while time.monotonic() < deadline and not self.stop_requested():
            time.sleep(5)
            try:
                self.provider.check_ready()
                self.ev.emit("server_back")
                return True
            except ServerDown:
                continue
        return False

    # ------------------------------------------------------------------ repli
    def _fallback_many(self, rows: list[dict], reason: str) -> None:
        if not rows:
            return
        fb: UiUpscaler = getattr(self, "_fallback", None) or UiUpscaler(self.cfg)
        jobs, recs = [], {}
        for r in rows:
            try:
                src, an, k = self.runner._load(r)
            except Exception as exc:
                self._fail(r, "comfyui", "image", f"image illisible : {exc}", "abandon")
                self.stats["errors"] += 1
                continue
            jobs.append(UiJob(r["id"], r["name"], src.convert("RGBA"), k, an))
            recs[r["id"]] = r
        results = fb.run(jobs, validate=False)
        for j in jobs:
            r, res = recs[j.tid], results[j.tid]
            img, engine = res.image, res.engine
            if img is None:      # dernier recours absolu
                img, engine = local_upscale(j.src, j.k, j.an, self.cfg, use_esrgan=False), "lanczos"
            p = self.runner._paths(j.tid)
            save_png(img, p["result"])
            save_png(img, p["local"])
            self.state.update(j.tid, status="fallback", choice="local", result_path=str(p["result"]),
                              local_path=str(p["local"]), error=reason, fingerprint=self.runner.fp,
                              metrics={"method": f"fallback:{engine}", "k": j.k, "fallback_engine": engine})
            self._fail(r, "comfyui", _kind_of(reason) if "[" in reason else "server", reason, "fallback")
            self.stats["fallback"] += 1
            self.ev.emit("fallback", texture=j.tid, name=j.name, engine=engine, reason=reason[:160])
