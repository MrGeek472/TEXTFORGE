"""Post-traitement final : contrôle qualité, correctifs, DDS avec mipmaps, rapport.

Deux points d'entrée :
- `PostProcessor.from_state(...)` : textures du projet TEXFORGE (base d'état), utilisé par
  `texforge qc` et `texforge export` ;
- `items_from_folders(...)` : n'importe quel dossier de sorties ComfyUI apparié à un dossier
  d'originaux (commande `texforge postprocess`), sans scan préalable.

Déroulé par lots (`[qc] chunk`) : QC + chaîne de mipmaps en parallèle (threads), puis un seul
lancement de l'encodeur par format pour tout le lot, puis relecture et écriture.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import ddsbuild
from .events import EventBus
from .imaging import IMAGE_EXTS, Analysis, SourceTexture, analyze, load_texture, output_scale, save_png
from .i18n import tr
from .qc import QC_VERSION, QCItem, QCResult, label_of, run_qc, summarize, to_target

REPORT_NAME = "qc_report.json"


@dataclass
class ExportItem:
    tid: str
    name: str                       # nom de fichier sans extension (sert à l'injection)
    src_path: Path
    an: Analysis
    candidate_path: Path | None     # image à contrôler (None : seulement l'agrandissement local)
    source: str                     # ai | ui | local | fallback
    k: int
    out_rel: str                    # chemin relatif de sortie, sans extension
    fallback_path: Path | None = None
    reviewed: bool = False
    data: bool = False
    route: str = ""
    row: dict | None = None         # ligne de la base d'état (mode projet)
    dds_format: str | None = None   # format imposé par une règle [[rules]]
    src_rel: str = ""               # chemin sous le dossier d'entrée (cible « loose » : conservé à l'identique)


# ------------------------------------------------------------------ sources des textures

def qc_fingerprint(cfg: dict) -> str:
    keys = {k: v for k, v in cfg["qc"].items() if not k.startswith(("encoder", "texconv", "workers", "chunk"))}
    rel = {"qc": keys, "quality": cfg["quality"], "scale": cfg["output"]["scale"], "max_side": cfg["output"]["max_side"],
           "version": QC_VERSION}
    return hashlib.sha256(json.dumps(rel, sort_keys=True, default=str).encode()).hexdigest()[:12]


def items_from_state(cfg: dict, state, only: str | None = None) -> list[ExportItem]:
    from fnmatch import fnmatch

    from .pipeline import analysis_of, apply_rules
    items = []
    for r in state.all("status IN ('done','fallback')"):
        if not r.get("result_path") or not Path(r["result_path"]).exists():
            continue
        if only and not (fnmatch(r["name"], only) or fnmatch(r["id"], only)):
            continue
        an_d = r.get("analysis") or {}
        an = analysis_of(r)
        route = an_d.get("route", "")
        if route == "ui":
            # un élément d'interface s'affiche en sprite : pas de contrôle de raccord (comme le pipeline UI)
            an = dc_replace(an, tile_x=False, tile_y=False)
        if r["status"] == "fallback":
            source = "fallback"
        elif r.get("choice") == "ai":
            source = "ai"
        elif route == "ui":
            source = "ui"
        else:
            source = "local"
        k = int((r.get("metrics") or {}).get("k") or 0)
        if not k:
            k = output_scale(r["width"], r["height"], cfg["output"]["scale"], cfg["output"]["max_side"])
        local = Path(r["local_path"]) if r.get("local_path") else None
        fb = local if source == "ai" and local and local.exists() else None
        label, _, rel = r["id"].partition("/")
        # règles relues à l'export : un dds_format ajouté après le scan s'applique sans rescanner
        _m, _h, _t, extra = apply_rules(cfg, Path(r["src_path"]).name, rel, r.get("mode") or "ai")
        items.append(ExportItem(
            tid=r["id"], name=r["name"], src_path=Path(r["src_path"]), an=an,
            candidate_path=Path(r["result_path"]), source=source, k=k,
            out_rel=str(Path(label) / Path(rel).with_suffix("")), fallback_path=fb,
            reviewed=bool(r.get("reviewed")), data=bool(an.normal_map or an_d.get("category") == "data"),
            route=route, row=r, dds_format=extra.get("dds_format"), src_rel=rel))
    return items


_COMFY_SUFFIX = re.compile(r"(_\d{4,6}_?)+$")


def _norm(stem: str) -> str:
    s = stem.lower()
    s = _COMFY_SUFFIX.sub("", s)
    for pre in ("texforge_", "comfyui_", "output_"):
        if s.startswith(pre):
            s = s[len(pre):]
    return s


def items_from_folders(cfg: dict, originals: Path, generated: Path, *, include_unmatched: bool = False,
                       log: Callable[[str], None] = print) -> list[ExportItem]:
    """Apparie chaque original à la sortie ComfyUI de même nom (préfixes et numéros ajoutés par
    SaveImage ignorés : TEXFORGE_Wall_04_00001_.png ↔ Wall_04.dds)."""
    from .pipeline import apply_rules
    from .routing import decide
    gens: dict[str, Path] = {}
    for p in sorted(generated.rglob("*")):
        if p.suffix.lower() in (".png", ".tga", ".jpg", ".jpeg", ".webp") and p.is_file():
            gens.setdefault(_norm(p.stem), p)
    items, missing = [], 0
    for p in sorted(originals.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        rel = p.relative_to(originals)
        cand = gens.get(_norm(p.stem))
        if cand is None:
            # dernier recours : un nom généré qui contient le nom de l'original
            key = p.stem.lower()
            hits = [g for n, g in gens.items() if key in n]
            cand = hits[0] if len(hits) == 1 else None
        if cand is None and not include_unmatched:
            missing += 1
            continue
        try:
            tex = load_texture(p)
        except Exception as exc:
            log(tr("  ! {name} illisible : {err}", name=rel, err=exc))
            continue
        an = analyze(tex.image, cfg)
        route = decide(tex.image, an, p.stem, cfg)
        _m, _h, tile, extra = apply_rules(cfg, p.name, str(rel).replace("\\", "/"), route.mode)
        if tile is not None:
            an.tile_x = an.tile_y = tile
        k = output_scale(tex.image.width, tex.image.height, cfg["output"]["scale"], cfg["output"]["max_side"])
        items.append(ExportItem(
            tid=str(rel).replace("\\", "/"), name=p.stem, src_path=p, an=an, candidate_path=cand,
            source="ai" if cand else "local", k=k, out_rel=str(rel.with_suffix("")),
            data=bool(an.normal_map or route.category == "data"), route=route.route,
            dds_format=extra.get("dds_format"), src_rel=str(rel).replace("\\", "/")))
    if missing:
        log(tr("  {n} original(aux) sans image générée correspondante (ignorés ; --include-unmatched pour "
               "les exporter en agrandissement local)", n=missing))
    return items


# ------------------------------------------------------------------ traitement

@dataclass
class _Prepared:
    item: ExportItem
    src: SourceTexture | None = None
    qc: QCResult | None = None
    plan: ddsbuild.DDSPlan | None = None
    chain: list | None = None
    notes: list | None = None
    error: str | None = None
    cached: bool = False


class PostProcessor:
    def __init__(self, cfg: dict, events: EventBus, *, target: str = "none", out_dir: Path | None = None,
                 qc: bool = True, fix: bool | None = None, encoder: str | None = None, state=None,
                 write_state: bool = True) -> None:
        if target not in ("none", "specialk", "loose", "png", "remix"):
            raise ValueError(tr("cible inconnue : {target}", target=target))
        self.cfg = cfg
        self.ev = events
        self.target = target
        self.out_dir = out_dir
        self.do_qc = qc and bool(cfg["qc"].get("enabled", True))
        if fix is not None:
            cfg["qc"]["auto_fix"] = bool(fix)
        self.state = state
        self.write_state = write_state and state is not None
        self.encoder = ddsbuild.make_encoder(cfg, encoder, log=lambda m: self.ev.emit("info", message=m)) \
            if target in ("specialk", "loose") else None
        self.fp = qc_fingerprint(cfg)
        self.stop_file = Path(cfg["_work"]) / "STOP"
        self.used_names: dict[str, str] = {}
        self._lock = threading.Lock()

    # .................................................................. un élément
    def _prepare(self, it: ExportItem) -> _Prepared:
        pr = _Prepared(it)
        try:
            pr.src = load_texture(it.src_path)
            cand = Image.open(it.candidate_path).convert("RGBA") if it.candidate_path else None
            res = self._cached(it, cand)
            if res is not None:
                pr.qc, pr.cached = res, True
            elif self.do_qc:
                fb = (lambda p=it.fallback_path: Image.open(p).convert("RGBA")) if it.fallback_path else None
                pr.qc = run_qc(QCItem(it.tid, it.name, pr.src.image, it.an, cand, it.source, it.k, fb,
                                      it.reviewed, it.data), self.cfg)
            else:
                img, notes = to_target(cand, pr.src.image, it.an, it.k) if cand else (None, [])
                if img is None:
                    raise ValueError(tr("aucune image à exporter"))
                pr.qc = QCResult(img, "measured", it.source, warnings=notes)
            if self.target in ("specialk", "loose"):
                fmt, _n = ddsbuild.choose_format(pr.src.fmt, pr.src.is_dds, it.an, self.cfg, data=it.data,
                                                 can_bc7=self._can_bc7(), override=it.dds_format)
                img, note = ddsbuild.fit_block_size(pr.qc.image, ddsbuild.dds.codec_of(fmt))
                plan = ddsbuild.plan_for(img, it.an, pr.src.fmt, pr.src.is_dds, pr.src.mipmaps, self.cfg,
                                         data=it.data, can_bc7=self._can_bc7(), override=it.dds_format)
                if note:
                    plan.notes.append(note)
                pr.plan = plan
                pr.chain = ddsbuild.build_chain(img, plan, str(self.cfg["qc"].get("mip_filter", "lanczos")))
        except Exception as exc:          # une texture illisible n'arrête pas le lot
            pr.error = f"{type(exc).__name__}: {exc}"
        return pr

    def _can_bc7(self) -> bool:
        return self.encoder is not None and "BC7" in getattr(self.encoder, "block_codecs", set())

    def _cached(self, it: ExportItem, cand: Image.Image | None) -> QCResult | None:
        """Résultat déjà contrôlé avec les mêmes réglages et le même fichier : pas de nouveau calcul."""
        if it.row is None or cand is None or it.candidate_path is None:
            return None
        q = (it.row.get("metrics") or {}).get("qc")
        if not q or q.get("fp") != self.fp or q.get("mtime_ns") != it.candidate_path.stat().st_mtime_ns:
            return None
        if cand.size != (it.an.width * it.k, it.an.height * it.k):
            return None
        return QCResult(cand, q["decision"], q.get("source", it.source), reasons=q.get("reasons", []),
                        fixes=q.get("fixes", []), warnings=q.get("warnings", []),
                        before=q.get("before", {}), after=q.get("after", {}))

    # .................................................................. lot complet
    def run(self, items: list[ExportItem]) -> dict:
        t0 = time.monotonic()
        total = len(items)
        enc_name = self.encoder.name if self.encoder else "-"
        self.ev.emit("qc_start", total=total, target=self.target, encoder=enc_name, qc=self.do_qc)
        entries: list[dict] = []
        counts = {k: 0 for k in ("validated", "corrected", "rejected", "local", "measured", "error")}
        dds_bytes, orig_bytes, written = 0, 0, 0
        chunk = max(1, int(self.cfg["qc"].get("chunk", 32)))
        workers = max(1, int(self.cfg["qc"].get("workers", 4)))
        index = 0
        stopped = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, total, chunk):
                if self.stop_file.exists():
                    stopped = True
                    self.ev.emit("stopped", message=tr("fichier STOP détecté"))
                    break
                batch = list(pool.map(self._prepare, items[start:start + chunk]))
                outputs = self._encode(batch)
                for pr in batch:
                    index += 1
                    entry = self._finish(pr, outputs.get(pr.item.tid), index, total)
                    entries.append(entry)
                    counts[entry["decision"]] = counts.get(entry["decision"], 0) + 1
                    if entry.get("output", {}).get("bytes"):
                        dds_bytes += entry["output"]["bytes"]
                        written += 1
                    elif entry.get("output", {}).get("file"):
                        written += 1
                    try:
                        orig_bytes += pr.item.src_path.stat().st_size
                    except OSError:
                        pass
        summary = {"total": total, "processed": index, "written": written, **counts,
                   "warnings": sum(1 for e in entries if e.get("warnings")),
                   "dds_mb": round(dds_bytes / 2 ** 20, 1), "original_mb": round(orig_bytes / 2 ** 20, 1),
                   "seconds": round(time.monotonic() - t0, 1), "stopped": stopped}
        budget = float(self.cfg["qc"].get("memory_budget_mb", 0) or 0)
        if budget and summary["dds_mb"] > budget:
            summary["memory_warning"] = tr("{mb} Mo de textures pour un budget de {budget:.0f} Mo : risque de "
                                           "plantage d'un jeu 32 bits (max_side ou scale à réduire)",
                                           mb=summary["dds_mb"], budget=budget)
        report = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "target": self.target, "encoder": enc_name,
            "qc_enabled": self.do_qc, "fingerprint": self.fp,
            "thresholds": {**{k: v for k, v in self.cfg["qc"].items() if not k.startswith("texconv")},
                           "quality": self.cfg["quality"]},
            "summary": summary, "textures": entries,
        }
        paths = [Path(self.cfg["_work"]) / REPORT_NAME]
        if self.out_dir is not None:
            paths.append(self.out_dir / REPORT_NAME)
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        summary["report"] = str(paths[-1])
        self.ev.emit("qc_end", summary=_summary_text(summary), **{k: v for k, v in summary.items() if k != "seconds"},
                     seconds=summary["seconds"])
        return report

    def _encode(self, batch: list[_Prepared]) -> dict:
        if self.target not in ("specialk", "loose"):
            return {}
        todo = {pr.item.tid: (pr.plan, pr.chain) for pr in batch if pr.error is None and pr.plan is not None}
        if not todo:
            return {}
        return ddsbuild.encode_many(todo, self.encoder, verify_output=bool(self.cfg["qc"].get("verify", True)),
                                    log=lambda m: self.ev.emit("info", message=m))

    # .................................................................. écriture et rapport
    def _finish(self, pr: _Prepared, out, index: int, total: int) -> dict:
        it = pr.item
        entry: dict = {"id": it.tid, "name": it.name, "route": it.route, "source": it.source}
        if pr.error is not None:
            entry.update(decision="error", label=label_of("error"), reasons=[pr.error], fixes=[], warnings=[])
            self._fail(it, "qc", "error", pr.error, "abandon")
            self.ev.emit("failed", texture=it.tid, name=it.name, stage="qc", kind="error", message=pr.error)
            return entry
        r = pr.qc
        assert r is not None
        entry.update(decision=r.decision, label=r.label, reasons=r.reasons, fixes=r.fixes, warnings=list(r.warnings),
                     metrics={"before": r.before, "after": r.after}, cached=pr.cached)
        if self.write_state and not pr.cached:
            self._update_state(it, r)
        output: dict = {}
        if self.target in ("specialk", "loose"):
            output = self._write_dds(it, pr, out, entry)
        elif self.target in ("png", "remix"):
            output = self._write_png(it, r.image)
        if output:
            entry["output"] = output
        ev = {"texture": it.tid, "name": it.name, "decision": r.decision, "label": r.label,
              "reasons": r.reasons[:4], "fixes": r.fixes, "warnings": entry["warnings"][:4],
              "summary": _short(r),
              "index": index, "total": total, "source": r.source}
        if output.get("format"):
            ev.update(format=output["format"], levels=output["levels"], encoder=output.get("encoder"),
                      psnr=output.get("psnr_min"))
        if entry["decision"] == "error":
            ev["decision"], ev["label"] = "error", label_of("error")
        self.ev.emit("qc_done", **ev)
        return entry

    def _write_dds(self, it: ExportItem, pr: _Prepared, out, entry: dict) -> dict:
        assert self.out_dir is not None and pr.plan is not None
        fname = f"{it.name}.dds"
        if self.target == "loose" and it.src_rel:
            # même chemin que dans le jeu (Textures/Architecture/…) : le fichier remplace celui de l'archive
            rel = Path(it.src_rel)
            fname = (rel if rel.suffix.lower() == ".dds" else rel.with_suffix(".dds")).as_posix()
        with self._lock:
            prev = self.used_names.get(fname.lower())
            if prev is not None:
                msg = tr("nom en double : {file} ({a} / {b})", file=fname, a=it.tid, b=prev)
                entry["warnings"].append(msg)
                return {"skipped": msg}
            self.used_names[fname.lower()] = it.tid
        if out is None or isinstance(out, Exception):
            msg = f"DDS : {out}" if out is not None else tr("DDS : aucun résultat d'encodage")
            entry.update(decision="error", label=label_of("error"))
            entry["reasons"] = entry.get("reasons", []) + [msg]
            self._fail(it, "dds", "dds", msg, "abandon")
            self.ev.emit("failed", texture=it.tid, name=it.name, stage="dds", kind="dds", message=msg)
            return {"error": msg}
        chk = out.check
        if not chk.get("ok", True):
            msg = tr("DDS relu invalide : {errors}", errors="; ".join(chk.get("errors", [])))
            entry.update(decision="error", label=label_of("error"))
            entry["reasons"] = entry.get("reasons", []) + [msg]
            self._fail(it, "dds", "dds", msg, "abandon")
            self.ev.emit("failed", texture=it.tid, name=it.name, stage="dds", kind="dds", message=msg)
            return {"error": msg}
        min_psnr = float(self.cfg["qc"].get("min_psnr", 28))
        if chk.get("psnr_l0") is not None and chk["psnr_l0"] < min_psnr:
            entry["warnings"].append(tr("compression {fmt} : PSNR {psnr:.1f} dB < {lim:.0f} (dégradé, texte ou UI : "
                                        "dds_format = \"bgra8\" pour cette texture)", fmt=out.plan.fmt,
                                        psnr=chk["psnr_l0"], lim=min_psnr))
        if chk.get("coverage_drift") is not None and chk["coverage_drift"] > 0.05:
            entry["warnings"].append(tr("couverture alpha des mipmaps : écart {v:.2f}", v=chk["coverage_drift"]))
        if chk.get("brightness_drift") is not None and chk["brightness_drift"] > 0.05:
            entry["warnings"].append(tr("luminosité des mipmaps : écart {v:.0%}", v=chk["brightness_drift"]))
        path = self.out_dir / fname
        ddsbuild.write_atomic(path, out.data)
        return {"file": fname, "format": out.plan.fmt, "levels": out.plan.levels,
                "size": [out.plan.width, out.plan.height], "bytes": len(out.data), "encoder": out.encoder,
                "color_space": out.plan.color_space, "alpha": out.plan.alpha_mode,
                "psnr_l0": chk.get("psnr_l0"), "psnr_min": chk.get("psnr_min"),
                "coverage_drift": chk.get("coverage_drift"), "brightness_drift": chk.get("brightness_drift"),
                "notes": out.plan.notes}

    def _write_png(self, it: ExportItem, img: Image.Image) -> dict:
        assert self.out_dir is not None
        if self.target == "remix":
            fname = f"{it.name}.png"
            with self._lock:
                if fname.lower() in self.used_names:
                    return {"skipped": tr("nom en double : {file}", file=fname)}
                self.used_names[fname.lower()] = it.tid
            dest = self.out_dir / fname
        else:
            dest = self.out_dir / f"{it.out_rel}.png"
        save_png(img, dest)
        return {"file": str(dest.relative_to(self.out_dir)), "size": [img.width, img.height]}

    def _update_state(self, it: ExportItem, r: QCResult) -> None:
        st, row = self.state, it.row
        if row is None or r.decision == "measured":
            return
        res_path = Path(row["result_path"])
        if r.decision in ("corrected", "rejected", "validated", "local"):
            cur = np.asarray(Image.open(res_path).convert("RGBA"))
            if cur.shape != np.asarray(r.image).shape or not np.array_equal(cur, np.asarray(r.image)):
                save_png(r.image, res_path)
        metrics = dict(row.get("metrics") or {})
        metrics["qc"] = {"decision": r.decision, "source": r.source, "reasons": r.reasons, "fixes": r.fixes,
                         "warnings": r.warnings, "before": r.before, "after": r.after, "fp": self.fp,
                         "mtime_ns": res_path.stat().st_mtime_ns}
        fields: dict = {"metrics": metrics}
        if r.decision == "rejected":
            msg = "QC : " + "; ".join(r.reasons[:3])
            fields.update(status="fallback", choice="local", error=msg)
            self._fail(it, "qc", "qc", msg, "fallback", json.dumps(r.before, ensure_ascii=False)[:4000])
        st.update(it.tid, **fields)

    def _fail(self, it: ExportItem, stage: str, kind: str, message: str, action: str, details: str = "") -> None:
        if self.state is not None and it.row is not None:
            self.state.add_failure(it.tid, stage, it.route, kind, message, details, action)


def _short(r: QCResult) -> str:
    """Mesures affichées : celles de l'image exportée, sauf en cas de rejet (celles du rendu refusé)."""
    m = r.before if r.decision == "rejected" else (r.after or r.before)
    return summarize(m) if m else ""


def _summary_text(s: dict) -> str:
    parts = [tr("{n} validée(s)", n=s["validated"]), tr("{n} corrigée(s)", n=s["corrected"]),
             tr("{n} rejetée(s)", n=s["rejected"]), tr("{n} locale(s)", n=s["local"])]
    if s.get("measured"):
        parts.append(tr("{n} mesurée(s) (choix manuel ou QC désactivé)", n=s["measured"]))
    if s.get("error"):
        parts.append(tr("{n} erreur(s)", n=s["error"]))
    txt = ", ".join(parts)
    if s.get("dds_mb"):
        ratio = f", ×{s['dds_mb'] / s['original_mb']:.1f}" if s.get("original_mb") else ""
        txt += tr(" — DDS {mb} Mo (originaux {orig} Mo{ratio})", mb=s["dds_mb"], orig=s["original_mb"], ratio=ratio)
    return txt
