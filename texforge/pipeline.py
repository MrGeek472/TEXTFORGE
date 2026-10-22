"""Orchestration : scan -> estimation -> génération -> revue -> export."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from PIL import Image

from .i18n import tr
from .config import DEFAULT_PROMPT, result_fingerprint
from .imaging import (IMAGE_EXTS, Analysis, analyze, finalize_ai, load_texture, local_upscale,
                      output_scale, prepare_for_model, prepare_native, save_png)
from .routing import decide
from .providers import FatalError, ProviderError, TransientError, make_provider
from .state import State
from .validate import evaluate

Log = Callable[[str], None]


# ------------------------------------------------------------------ outils

def open_state(cfg: dict) -> State:
    return State(Path(cfg["_work"]) / "texforge_state.db")


def safe_name(tid: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", tid)[-90:]
    return f"{base}_{hashlib.sha1(tid.encode()).hexdigest()[:8]}"


def root_label(root: Path) -> str:
    return f"{root.name or 'racine'}-{hashlib.sha1(str(root).lower().encode()).hexdigest()[:6]}"


MODES = ("ai", "ui", "upscale", "keep")


def apply_rules(cfg: dict, name: str, rel: str, auto_mode: str) -> tuple[str, str, bool | None, dict]:
    """Première règle dont le motif correspond : (mode, indice de prompt, répétable forcé ou None,
    champs supplémentaires : material, engine)."""
    for rule in cfg.get("rules", []) or []:
        pat = str(rule.get("match", "")).lower()
        if pat and (fnmatch.fnmatch(name.lower(), pat) or fnmatch.fnmatch(rel.lower(), pat)):
            mode = str(rule.get("mode", auto_mode)).lower()
            if mode not in MODES:
                mode = auto_mode
            tile = rule.get("tile")
            extra = {k: str(rule[k]) for k in ("material", "engine", "dds_format") if rule.get(k)}
            extra["rule"] = str(rule.get("match"))
            return mode, str(rule.get("hint", "")), (bool(tile) if tile is not None else None), extra
    return auto_mode, "", None, {}


def build_prompt(cfg: dict, hint: str, n_refs: int) -> str:
    template = cfg["prompt"].get("template") or DEFAULT_PROMPT
    hint_s = f", the surface is {hint}" if hint else ""
    if n_refs >= 2:
        refs_s = (" Images 2 and 3 are style references only: borrow their material look and level of "
                  "detail, never their content or layout.")
    elif n_refs == 1:
        refs_s = (" Image 2 is a style reference only: borrow its material look and level of detail, "
                  "never its content or layout.")
    else:
        refs_s = ""
    return (template.replace("{style}", cfg["prompt"].get("style", ""))
            .replace("{hint}", hint_s).replace("{refs}", refs_s))


def style_refs(cfg: dict) -> list[Path]:
    out = []
    for p in cfg["higgsfield"].get("style_references", []) or []:
        pp = Path(p)
        pp = pp if pp.is_absolute() else Path(cfg["_base"]) / pp
        if not pp.exists():
            raise FatalError(tr("image de référence introuvable : {path}", path=pp))
        out.append(pp)
    return out


def analysis_of(rec: dict) -> Analysis:
    d = dict(rec["analysis"])
    d.pop("error", None)
    return Analysis(**{k: d[k] for k in Analysis.__dataclass_fields__ if k in d})   # champs récents : défaut


# ------------------------------------------------------------------ scan

def scan(cfg: dict, log: Log = print, on_progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Analyse et routage. `on_progress(i, total, nom)` est appelé avant chaque fichier ; il peut
    lever une exception pour interrompre (chaque texture est enregistrée dès qu'elle est analysée)."""
    state = open_state(cfg)
    work = Path(cfg["_work"]).resolve()
    counts = {"new": 0, "changed": 0, "same": 0, "errors": 0}
    seen = 0
    roots = []
    for root_s in cfg["_inputs"]:
        root = Path(root_s)
        if not root.exists():
            log(tr("  ! dossier introuvable : {root}", root=root))
            continue
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                       and work not in p.resolve().parents)
        roots.append((root, files))
    grand_total = sum(len(f) for _r, f in roots)
    for root, files in roots:
        label = root_label(root)
        for path in files:
            if on_progress is not None:
                try:
                    on_progress(seen, grand_total, path.name)
                except BaseException:
                    state.close()
                    raise
            seen += 1
            rel = path.relative_to(root).as_posix()
            tid = f"{label}/{rel}"
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            old = state.get(tid)
            rec = {"id": tid, "src_path": str(path), "src_sha": sha, "name": path.stem}
            try:
                src = load_texture(path)
                an = analyze(src.image, cfg)
                an_d = an.to_dict()
                auto_mode = an.mode
                if cfg.get("routing", {}).get("enabled", True):
                    d = decide(src.image, an, path.stem, cfg)
                    auto_mode = d.mode
                    an_d.update(route=d.route, category=d.category, ui_kind=d.ui_kind, material=d.material,
                                ui_score=d.ui_score, route_reason=d.reason, features=d.features)
                if src.fmt in ("BC4", "BC5") and auto_mode != "keep":
                    # un ou deux canaux : jamais une image en couleur (relief, spéculaire, masque)
                    auto_mode = "upscale"
                    an_d.update(route="local", category="data",
                                route_reason=tr("format {fmt} : carte de données, agrandie sans IA", fmt=src.fmt))
                rec.update(fmt=src.fmt, mipmaps=src.mipmaps, width=src.image.width, height=src.image.height)
            except Exception as exc:
                counts["errors"] += 1
                an_d = {"error": str(exc)}
                rec.update(fmt="?", mipmaps=0, width=0, height=0)
            if "error" in an_d:
                mode, hint = "keep", ""
            else:
                mode, hint, tile, extra = apply_rules(cfg, path.name, rel, auto_mode)
                if tile is not None:
                    an_d["tile_x"] = an_d["tile_y"] = tile
                if extra.get("material"):
                    an_d["material"] = extra["material"]
                if extra.get("engine"):
                    an_d["ui_engine"] = extra["engine"]
                if extra.get("rule"):
                    an_d["route_reason"] = tr("règle « {rule} »", rule=extra['rule'])
                an_d["route"] = {"ai": "comfyui", "ui": "ui", "upscale": "local", "keep": "keep"}[mode]
            counts["route_" + an_d.get("route", "keep")] = counts.get("route_" + an_d.get("route", "keep"), 0) + 1
            rec.update(analysis=an_d, mode=mode, hint=hint)
            kind = state.upsert_scan(rec)
            counts[kind] += 1
            cur = state.get(tid)
            if "error" in an_d:
                state.update(tid, status="kept", error=an_d["error"])
            elif mode == "keep":
                state.update(tid, status="kept")
            elif old and old["mode"] != mode and cur["status"] != "pending":
                state.update(tid, status="pending", attempts=0, result_path=None, ai_path=None,
                             best_score=None, metrics=None, choice="auto", reviewed=0)
            elif cur["status"] == "kept":
                state.update(tid, status="pending")
    counts["seen"] = seen
    state.close()
    return counts


def summary(cfg: dict) -> dict:
    state = open_state(cfg)
    rows = state.all()
    by_mode: dict = {}
    by_status: dict = {}
    for r in rows:
        by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    out = {"total": len(rows), "by_mode": by_mode, "by_status": by_status,
           "requests": state.count_requests(),
           "ai_pending": sum(1 for r in rows if r["mode"] == "ai" and r["status"] == "pending"),
           "by_route": {}}
    for r in rows:
        route = (r.get("analysis") or {}).get("route", "?")
        out["by_route"][route] = out["by_route"].get(route, 0) + 1
    state.close()
    return out


# ------------------------------------------------------------------ budget

class BudgetReached(Exception):
    pass


class Budget:
    def __init__(self, limit: int):
        self.limit = max(0, int(limit))
        self.used = 0
        self._lock = threading.Lock()

    def reserve(self) -> None:
        with self._lock:
            if self.limit and self.used >= self.limit:
                raise BudgetReached()
            self.used += 1

    def release(self) -> None:
        with self._lock:
            self.used = max(0, self.used - 1)


# ------------------------------------------------------------------ traitement

class Runner:
    def __init__(self, cfg: dict, provider_name: str | None, log: Log, provider=None):
        self.cfg = cfg
        self.log = log
        self.state = open_state(cfg)
        self.work = Path(cfg["_work"])
        self._injected = provider
        self.provider = None
        self.provider_name = provider_name or cfg["provider"]["name"]
        self.fp = result_fingerprint({**cfg, "provider": {**cfg["provider"], "name": self.provider_name}})
        self.stop = threading.Event()
        self.fatal: str | None = None
        self.budget = Budget(cfg["provider"].get("max_requests", 0))
        self.refs: list[Path] = []
        # Repli d'une génération ratée : (src, analyse, k) -> (image, moteur). Défaut : Lanczos.
        self.fallback_image = None

    def _paths(self, tid: str) -> dict:
        n = safe_name(tid)
        return {"result": self.work / "results" / f"{n}.png", "ai": self.work / "ai" / f"{n}.png",
                "local": self.work / "local" / f"{n}.png", "prep": self.work / "prep" / f"{n}.png"}

    def _load(self, rec: dict):
        src = load_texture(Path(rec["src_path"])).image
        an = analysis_of(rec)
        k = output_scale(src.width, src.height, self.cfg["output"]["scale"], self.cfg["output"]["max_side"])
        return src, an, k

    # ------------------------------------------------ agrandissement local
    def do_local(self, rec: dict) -> str:
        src, an, k = self._load(rec)
        p = self._paths(rec["id"])
        out = local_upscale(src, k, an, self.cfg)
        save_png(out, p["local"])
        save_png(out, p["result"])
        self.state.update(rec["id"], status="done", choice="local", result_path=str(p["result"]),
                          local_path=str(p["local"]), metrics={"k": k, "method": "local"},
                          fingerprint=self.fp, error=None)
        return "local"

    # ------------------------------------------------ génération IA
    def do_ai(self, rec: dict) -> str:
        tid = rec["id"]
        src, an, k = self._load(rec)
        p = self._paths(tid)
        local = local_upscale(src, k, an, self.cfg)
        save_png(local, p["local"])
        if getattr(self.provider, "native_tiling", False):
            model_in, prep = prepare_native(src, an)
        else:
            model_in, prep = prepare_for_model(src, an, self.cfg, self.provider.ratios)
        save_png(model_in, p["prep"])
        context = self.provider.build_context(rec, an) if hasattr(self.provider, "build_context") else None
        extra_kw = {"context": context} if context is not None else {}
        prompt = (context or {}).get("prompt") or build_prompt(self.cfg, rec.get("hint") or "", len(self.refs))
        negative = (context or {}).get("negative") or self.cfg["prompt"].get("negative", "")
        max_attempts = max(1, int(self.cfg["provider"].get("max_attempts", 3)))
        base_seed = int(self.cfg["provider"].get("base_seed", 1234)) + int(hashlib.sha1(tid.encode()).hexdigest()[:6], 16) % 10000

        best: dict | None = None
        if rec.get("ai_path") and rec.get("metrics") and rec["metrics"].get("method") == "ai":
            if Path(rec["ai_path"]).exists() and rec.get("fingerprint") == self.fp:
                best = {"metrics": rec["metrics"], "path": rec["ai_path"]}
        errors: list[str] = []

        def consider(out_img: Image.Image, attempt: int, req_prep: dict) -> dict:
            nonlocal best
            from .imaging import PrepInfo
            final, extra = finalize_ai(out_img, src, an, PrepInfo.from_dict(req_prep), k, self.cfg)
            m = evaluate(src, final, an, self.cfg)
            m.update(extra)
            m.update({"k": k, "method": "ai", "attempt": attempt + 1})
            if best is None or m["score"] > best["metrics"]["score"]:
                save_png(final, p["ai"])
                best = {"metrics": m, "path": str(p["ai"])}
                self.state.update(tid, ai_path=str(p["ai"]), best_score=m["score"], metrics=m)
            return m

        # 1) reprise des requêtes payées lors d'une exécution interrompue
        attempts = int(rec.get("attempts") or 0)
        for req in self.state.requests_for(tid):
            if req["status"] != "submitted" or self.stop.is_set():
                continue
            if not self.provider.paid:
                self.state.finish_request(req["request_id"], "failed")
                continue
            self.log(tr("    reprise de la requête {rid}… ({name})", rid=req["request_id"][:8], name=rec["name"]))
            try:
                out_img, url = self.provider.resume(req["request_id"])
            except ProviderError as exc:
                self.state.finish_request(req["request_id"], "failed")
                errors.append(str(exc))
                continue
            except TransientError as exc:
                self.state.update(tid, error=str(exc))
                return "transient"
            self.state.finish_request(req["request_id"], "completed", url)
            m = consider(out_img, req["attempt"], req["prep"])
            attempts = max(attempts, req["attempt"] + 1)
            self.state.update(tid, attempts=attempts)
            if m["passed"]:
                break

        # 2) nouvelles tentatives
        attempt = attempts
        while (best is None or not best["metrics"]["passed"]) and attempt < max_attempts:
            if self.stop.is_set():
                return "stopped"
            seed = base_seed + attempt * 7919
            reserved = False
            if self.provider.paid:
                self.budget.reserve()        # lève BudgetReached
                reserved = True
            submitted: dict = {}

            def on_submitted(rid: str, _a=attempt, _s=seed) -> None:
                submitted["id"] = rid
                self.state.add_request(rid, tid, _a, _s, prep.to_dict())

            try:
                out_img, url = self.provider.generate(model_in, self.refs, prompt, negative,
                                                      prep.aspect_ratio, seed, on_submitted, **extra_kw)
            except ProviderError as exc:
                if "id" in submitted:
                    self.state.finish_request(submitted["id"], "failed")
                elif reserved:
                    self.budget.release()
                errors.append(str(exc))
                attempt += 1
                self.state.update(tid, attempts=attempt, error=str(exc))
                continue
            except TransientError as exc:
                if "id" not in submitted and reserved:
                    self.budget.release()
                self.state.update(tid, error=str(exc))
                return "transient"
            except FatalError:
                if "id" not in submitted and reserved:
                    self.budget.release()
                raise
            self.state.finish_request(submitted.get("id", ""), "completed", url)
            consider(out_img, attempt, prep.to_dict())
            attempt += 1
            self.state.update(tid, attempts=attempt)

        if best is not None and best["metrics"]["passed"]:
            save_png(Image.open(best["path"]), p["result"])
            self.state.update(tid, status="done", choice="ai", result_path=str(p["result"]),
                              local_path=str(p["local"]), fingerprint=self.fp, error=None)
            return "ai"
        engine = "lanczos"
        if self.fallback_image is not None:
            try:
                local, engine = self.fallback_image(src, an, k)
                save_png(local, p["local"])
            except Exception as exc:   # le repli amélioré a échoué : Lanczos déjà calculé
                errors.append(tr("repli {exc}", exc=exc))
        save_png(local, p["result"])
        reason = "; ".join(best["metrics"]["reasons"]) if best else ("; ".join(errors) or tr("aucun résultat"))
        self.state.update(tid, status="fallback", choice="local", result_path=str(p["result"]),
                          local_path=str(p["local"]), fingerprint=self.fp, error=reason,
                          metrics={**(best["metrics"] if best else {}), "fallback_engine": engine, "k": k})
        return "fallback"

    # ------------------------------------------------ boucle principale
    def run(self, limit: int | None = None, only: str | None = None, redo_stale: bool = False,
            local_only: bool = False) -> dict:
        rows = self.state.all("status='pending' OR status='error'")
        if redo_stale:
            rows += [r for r in self.state.all("status IN ('done','fallback')")
                     if r["mode"] == "ai" and r.get("fingerprint") != self.fp and r.get("choice") == "auto"]
        if only:
            pat = only.lower()
            rows = [r for r in rows if fnmatch.fnmatch(r["name"].lower(), pat) or fnmatch.fnmatch(r["id"].lower(), pat)]
        stale = [r for r in self.state.all("status IN ('done','fallback')")
                 if r["mode"] == "ai" and r.get("fingerprint") not in (None, self.fp)]
        if stale and not redo_stale:
            self.log(tr("  ℹ {n} texture(s) traitée(s) avec d'anciens réglages (relance avec --redo-stale pour "
                        "les refaire)",
                        n=len(stale)))
        local_rows = [r for r in rows if r["mode"] == "upscale"]
        ai_rows = [r for r in rows if r["mode"] == "ai"]
        if limit is not None:
            ai_rows = ai_rows[:limit]
        stats = {"local": 0, "ai": 0, "fallback": 0, "transient": 0, "errors": 0, "budget": False}

        for r in local_rows:
            try:
                self.do_local(r)
                stats["local"] += 1
            except Exception as exc:
                stats["errors"] += 1
                self.state.update(r["id"], status="error", error=str(exc))
                self.log(tr("  ✗ {name} : {exc}", name=r['name'], exc=exc))
        if local_rows:
            self.log(tr("  agrandissement local : {local} texture(s)", local=stats['local']))

        if local_only or not ai_rows:
            self.state.close()
            return stats

        self.provider = self._injected or make_provider(self.cfg, self.state, self.provider_name)
        self.refs = style_refs(self.cfg)
        total = len(ai_rows)
        done = 0
        workers = max(1, int(self.cfg["provider"].get("workers", 3)))
        self.log(tr("  génération IA ({name}) : {total} texture(s), {workers} en parallèle",
                    name=self.provider.name, total=total, workers=workers)
                 + (tr(", plafond {limit} requêtes", limit=self.budget.limit) if self.provider.paid and self.budget.limit else ""))
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(self._safe_ai, r): r for r in ai_rows}
        pending = set(futures)
        try:
            # Attente par tranches d'une seconde : Ctrl+C reste réactif sous Windows
            while pending:
                finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    r = futures[fut]
                    done += 1
                    outcome, info = fut.result()
                    if outcome == "budget":
                        stats["budget"] = True
                        continue
                    if outcome in stats:
                        stats[outcome] += 1
                    cur = self.state.get(r["id"]) or {}
                    m = cur.get("metrics") or {}
                    tag = {"ai": tr("✓ IA"), "fallback": tr("↺ repli local"), "transient": tr("… à reprendre"),
                           "errors": tr("✗ erreur"), "stopped": tr("■ arrêt")}.get(outcome, outcome)
                    detail = tr("structure {structure}, essais {attempts}",
                                structure=m.get("structure", "-"), attempts=cur.get("attempts", 0)) if m else (info or "")
                    if outcome == "fallback" and cur.get("error"):
                        detail += f" — {cur['error'][:90]}"
                    self.log(tr("  [{done}/{total}] {name} : {tag} ({detail})",
                                done=done, total=total, name=r['name'], tag=tag, detail=detail))
        except KeyboardInterrupt:
            self.stop.set()
            self.log(tr("  ■ interruption : les requêtes déjà lancées seront reprises au prochain run"))
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        if self.fatal:
            stats["fatal"] = self.fatal
        stats["requests_used"] = self.budget.used
        self.state.close()
        return stats

    def _safe_ai(self, rec: dict) -> tuple[str, str]:
        if self.stop.is_set():
            return "stopped", tr("arrêt demandé")
        try:
            return self.do_ai(rec), ""
        except BudgetReached:
            return "budget", ""
        except FatalError as exc:
            self.fatal = str(exc)
            self.stop.set()
            return "errors", str(exc)
        except Exception as exc:  # erreur inattendue sur une texture : on continue les autres
            self.state.update(rec["id"], status="error", error=f"{type(exc).__name__}: {exc}")
            return "errors", f"{type(exc).__name__}: {exc}"


def run(cfg: dict, provider_name: str | None = None, limit: int | None = None, only: str | None = None,
        redo_stale: bool = False, local_only: bool = False, log: Log = print, events=None, provider=None) -> dict:
    """Exécution complète par l'orchestrateur : UI, local, puis IA (ComfyUI par défaut)."""
    from .events import EventBus
    from .orchestrator import Orchestrator
    bus = events or EventBus(cfg["_work"], console=False)
    stages = ("ui", "local") if local_only else ("ui", "local", "ai")
    try:
        return Orchestrator(cfg, bus, provider_name=provider_name, provider=provider).run(
            only=only, limit=limit, stages=stages, redo_stale=redo_stale)
    finally:
        if events is None:
            bus.close()


# ------------------------------------------------------------------ revue

def apply_review(cfg: dict, decisions_path: Path, log: Log = print) -> dict:
    data = json.loads(decisions_path.read_text(encoding="utf-8"))
    choices = data.get("choices", data)
    state = open_state(cfg)
    counts = {"ai": 0, "local": 0, "redo": 0, "ignored": 0}
    for tid, choice in choices.items():
        rec = state.get(tid)
        if not rec:
            counts["ignored"] += 1
            continue
        if choice == "ai" and rec.get("ai_path") and Path(rec["ai_path"]).exists():
            save_png(Image.open(rec["ai_path"]), Path(rec["result_path"]))
            state.update(tid, choice="ai", status="done", reviewed=1)
            counts["ai"] += 1
        elif choice == "local" and rec.get("local_path") and Path(rec["local_path"]).exists():
            save_png(Image.open(rec["local_path"]), Path(rec["result_path"]))
            state.update(tid, choice="local", status="done", reviewed=1)
            counts["local"] += 1
        elif choice == "redo":
            state.update(tid, status="pending", attempts=0, choice="auto", best_score=None,
                         metrics=None, ai_path=None, error=None, reviewed=0)
            counts["redo"] += 1
        else:
            counts["ignored"] += 1
    state.close()
    return counts


# ------------------------------------------------------------------ export

def export(cfg: dict, target: str, out_dir: Path, log: Log = print, *, qc: bool = True, fix: bool | None = None,
           encoder: str | None = None, events=None, only: str | None = None) -> dict:
    """Contrôle qualité puis écriture des textures finales (DDS avec mipmaps, PNG ou RTX Remix).
    Détail dans postprocess.py ; rapport écrit dans <sortie>/qc_report.json."""
    from .events import EventBus
    from .postprocess import PostProcessor, items_from_state
    state = open_state(cfg)
    bus = events or EventBus(cfg["_work"], console=False)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pp = PostProcessor(cfg, bus, target=target, out_dir=out_dir, qc=qc, fix=fix, encoder=encoder, state=state)
        report = pp.run(items_from_state(cfg, state, only))
    finally:
        state.close()
        if events is None:
            bus.close()
    s = report["summary"]
    lines = [f"{t['name']}\t{t.get('output', {}).get('format', '')}\t{t['decision']}\t{'; '.join(t.get('reasons', []))[:120]}"
             for t in report["textures"]]
    (out_dir / "texforge_export.txt").write_text(
        f"TEXFORGE export ({target}) — {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return {"written": s["written"], "skipped": s["total"] - s["written"], "out": str(out_dir), "summary": s,
            "report": s.get("report")}
