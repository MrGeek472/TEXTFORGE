"""Interface en ligne de commande : python -m texforge <commande>"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .config import EXAMPLE_TOML, ConfigError, load_config


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _cfg(args) -> dict:
    return load_config(Path(args.config).resolve())


def cmd_init(args) -> int:
    folder = Path(args.folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cfg_path = folder / "texforge.toml"
    if cfg_path.exists():
        print(f"existe déjà : {cfg_path}")
    else:
        cfg_path.write_text(EXAMPLE_TOML, encoding="utf-8")
        print(f"configuration créée : {cfg_path}")
    (folder / "entree").mkdir(exist_ok=True)
    print("Étapes suivantes :")
    print("  1. Ouvre texforge.toml et indique dans [paths] inputs le dossier des textures extraites")
    print("  2. python -m texforge scan")
    print("  3. python -m texforge route --report routage.csv   (vérifier le routage)")
    print("  4. python -m texforge run")
    return 0


def cmd_doctor(args) -> int:
    ok = True
    print(f"TEXFORGE {__version__} — Python {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("  ✗ Python 3.11 ou plus récent requis")
        ok = False
    try:
        import io
        import PIL
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGBA", (8, 8)).save(buf, format="DDS", pixel_format="DXT5")
        print(f"  ✓ numpy, Pillow {PIL.__version__} (écriture DXT disponible)")
    except Exception as exc:
        print(f"  ✗ numpy/Pillow : {exc} — lance : pip install -r requirements.txt")
        ok = False
    for mod, why in (("aiohttp", "client ComfyUI"), ("tqdm", "barres de progression")):
        try:
            __import__(mod)
            print(f"  ✓ {mod} ({why})")
        except ImportError:
            print(f"  ✗ {mod} absent ({why}) — pip install -r requirements.txt")
            ok = False
    try:
        cfg = _cfg(args)
    except ConfigError as exc:
        print(f"  ✗ {exc}")
        return 1
    print(f"  ✓ configuration lue : {Path(args.config).resolve()}")
    for p in cfg["_inputs"]:
        print(f"  {'✓' if Path(p).exists() else '✗'} entrée : {p}")
    print(f"  · dossier de travail : {cfg['_work']}")
    print(f"  · fournisseur IA : {cfg['provider']['name']}")
    from .ui_upscale import UiUpscaler
    up = UiUpscaler(cfg)
    print(f"  · moteurs UI disponibles : {', '.join(up.available())}"
          + ("" if len(up.available()) > 1 else "  (configure realesrgan_exe pour un meilleur résultat)"))
    from .ddsbuild import find_texconv
    tx = find_texconv(cfg)
    if tx:
        print(f"  ✓ texconv : {tx} (BC1/BC2/BC3, BC7 pour Direct3D 11)")
    else:
        print("  · texconv absent : encodeur DDS intégré (DXT1/DXT3/DXT5). Pour la meilleure compression :")
        print("    texconv.exe (github.com/microsoft/DirectXTex/releases) dans tools\\ ou [qc] texconv_exe")
    print(f"  · DDS : format {cfg['output']['dds_format']}, mipmaps {cfg['output']['mipmaps']}, "
          f"API cible {cfg['qc']['target_api']}")
    if cfg["provider"]["name"] == "comfyui":
        from .providers import ServerDown, WorkflowFatal, make_provider
        try:
            prov = make_provider(cfg, None, "comfyui")
            print(f"  ✓ {prov.info}")
            print(f"  ✓ workflows : {cfg['comfyui']['workflow']}"
                  + (f", {cfg['comfyui']['workflow_edges']}" if prov.wf_edges else ""))
            prov.close()
        except ServerDown as exc:
            print(f"  ✗ ComfyUI injoignable : {exc}")
            print("    lance : python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch")
            ok = False
        except WorkflowFatal as exc:
            print(f"  ✗ {exc}")
            ok = False
    elif cfg["provider"]["name"] == "higgsfield":
        has_key = bool(os.getenv("HF_KEY") or (os.getenv("HF_API_KEY") and os.getenv("HF_API_SECRET"))
                       or cfg["higgsfield"].get("api_key"))
        print(f"  {'✓' if has_key else '✗'} clé Higgsfield {'trouvée' if has_key else 'absente'}")
    return 0 if ok else 1


ROUTE_LABELS = {"ui": "UI (Real-ESRGAN/Waifu2x)", "comfyui": "ComfyUI (matériaux)",
                "local": "local (Lanczos)", "keep": "gardées telles quelles"}


def cmd_scan(args) -> int:
    from .pipeline import scan, summary
    cfg = _cfg(args)
    c = scan(cfg)
    print(f"{c['seen']} fichier(s) : {c['new']} nouveau(x), {c['changed']} modifié(s), "
          f"{c['same']} inchangé(s), {c['errors']} illisible(s)")
    s = summary(cfg)
    for route, n in sorted(s["by_route"].items()):
        print(f"  {ROUTE_LABELS.get(route, route):28s} {n}")
    print("Détail du routage : python -m texforge route --report routage.csv")
    return 0


def cmd_route(args) -> int:
    import csv
    from .pipeline import open_state
    cfg = _cfg(args)
    st = open_state(cfg)
    rows = st.all()
    st.close()
    counts: dict = {}
    for r in rows:
        a = r.get("analysis") or {}
        key = (a.get("route", "?"), a.get("ui_kind") or a.get("material") or a.get("category", ""))
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(rows)} texture(s)")
    for (route, sub), n in sorted(counts.items()):
        print(f"  {ROUTE_LABELS.get(route, route):28s} {sub:12s} {n}")
    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["texture", "route", "type", "matériau", "score_ui", "raison", "taille", "raccord", "statut"])
            for r in rows:
                a = r.get("analysis") or {}
                w.writerow([r["id"], a.get("route", ""), a.get("ui_kind", "") or a.get("category", ""),
                            a.get("material", ""), a.get("ui_score", ""), a.get("route_reason", ""),
                            f"{r['width']}x{r['height']}", f"{int(bool(a.get('tile_x')))}{int(bool(a.get('tile_y')))}",
                            r["status"]])
        print(f"Rapport : {Path(args.report).resolve()}  (ouvrable dans Excel)")
    return 0


def cmd_estimate(args) -> int:
    from .pipeline import summary
    cfg = _cfg(args)
    s = summary(cfg)
    n = s["ai_pending"]
    if cfg["provider"]["name"] == "comfyui":
        print(f"Textures pour ComfyUI : {n}")
        print(f"Durée indicative sur une RTX 5060 Ti (SDXL, 1 MP, 28 pas) : {n * 15 / 60:.0f} à {n * 30 / 60:.0f} min "
              "(15 à 30 s par texture ; à mesurer sur ton premier lot)")
        print(f"Textures UI : {s['by_route'].get('ui', 0)} — quelques secondes par lot de 64 avec Real-ESRGAN")
        return 0
    att = int(cfg["provider"]["max_attempts"])
    price = float(cfg["provider"].get("price_per_request_usd", 0) or 0)
    print(f"Textures à générer par IA : {n} — requêtes : {n} à {n * att}, réaliste ≈ {round(n * 1.5)}")
    if price > 0:
        print(f"Coût : {n * price:.2f} $ à {n * att * price:.2f} $")
    return 0


def cmd_run(args) -> int:
    from .events import EventBus
    from .pipeline import run
    from .providers import FatalError
    cfg = _cfg(args)
    if args.max_requests is not None:
        cfg["provider"]["max_requests"] = args.max_requests
    if args.workers is not None:
        cfg["provider"]["workers"] = args.workers
        cfg["comfyui"]["max_in_flight"] = args.workers
    out = sys.stderr if args.json_events else sys.stdout
    bus = EventBus(cfg["_work"], json_stdout=args.json_events, console=not args.quiet, verbose=args.verbose)
    stages = ("ui", "local", "ai") if not args.stage else tuple(args.stage)
    try:
        if args.stage:
            from .orchestrator import Orchestrator
            stats = Orchestrator(cfg, bus, provider_name=args.provider).run(
                only=args.only, limit=args.limit, stages=stages, redo_stale=args.redo_stale)
        else:
            stats = run(cfg, provider_name=args.provider, limit=args.limit, only=args.only,
                        redo_stale=args.redo_stale, local_only=args.local_only, events=bus)
    except FatalError as exc:
        print(f"✗ arrêt : {exc}", file=out)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        bus.close()
    if stats.get("budget"):
        print("Plafond de requêtes atteint : relance la commande pour continuer.", file=out)
    print(f"Journal complet : {Path(cfg['_work']) / 'events.jsonl'}", file=out)
    return 1 if stats.get("errors") else 0


def cmd_failures(args) -> int:
    import json
    from .pipeline import open_state
    cfg = _cfg(args)
    st = open_state(cfg)
    rows = st.failures(kind=args.kind)
    names = {r["id"]: r for r in st.all()}
    st.close()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    if not rows:
        print("Aucun échec consigné.")
        return 0
    by_kind: dict = {}
    for f in rows:
        by_kind.setdefault((f["kind"], f["action"]), []).append(f)
    for (kind, action), items in sorted(by_kind.items()):
        print(f"{kind:11s} → {action:8s} : {len(items)}")
    print()
    for f in rows[-args.last:]:
        cur = names.get(f["texture_id"], {})
        print(f"  {cur.get('name', f['texture_id'])} [{f['stage']}/{f['kind']} → {f['action']}] "
              f"statut {cur.get('status', '?')} : {f['message'][:110]}")
    return 0


def cmd_retry(args) -> int:
    from .pipeline import open_state
    cfg = _cfg(args)
    st = open_state(cfg)
    targets: set[str] = set()
    statuses = ("error", "fallback") if args.include_fallback else ("error",)
    for r in st.all():
        if r["status"] in statuses:
            if args.kind and not any(f["kind"] == args.kind for f in st.failures(tid=r["id"])):
                continue
            if args.route and (r.get("analysis") or {}).get("route") != args.route:
                continue
            targets.add(r["id"])
    for tid in targets:
        st.update(tid, status="pending", attempts=0, error=None, choice="auto", reviewed=0)
    st.close()
    print(f"{len(targets)} texture(s) remise(s) en file. Relance : python -m texforge run")
    return 0


def cmd_stop(args) -> int:
    cfg = _cfg(args)
    f = Path(cfg["_work"]) / "STOP"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("stop", encoding="utf-8")
    print(f"Arrêt demandé ({f}) : l'exécution en cours s'arrête après les textures en vol.")
    return 0


def cmd_status(args) -> int:
    from .pipeline import summary
    s = summary(_cfg(args))
    print(f"{s['total']} texture(s) — requêtes enregistrées : {s['requests']}")
    for k, v in sorted(s["by_status"].items()):
        print(f"  {k:10s} {v}")
    return 0


def cmd_review(args) -> int:
    from .review import build_review
    page = build_review(_cfg(args))
    print(f"Galerie : {page}")
    if not args.no_open:
        webbrowser.open(page.as_uri())
    return 0


def cmd_apply_review(args) -> int:
    from .pipeline import apply_review
    c = apply_review(_cfg(args), Path(args.file))
    print(f"Choix appliqués : {c['ai']} IA, {c['local']} local, {c['redo']} à regénérer, {c['ignored']} ignoré(s)")
    if c["redo"]:
        print("Relance « run » pour regénérer les textures marquées.")
    return 0


def _qc_bus(args, cfg):
    from .events import EventBus
    return EventBus(cfg["_work"], json_stdout=getattr(args, "json_events", False),
                    console=not getattr(args, "quiet", False))


def _apply_qc_args(cfg: dict, args) -> None:
    if getattr(args, "workers", None):
        cfg["qc"]["workers"] = args.workers
    if getattr(args, "target_api", None):
        cfg["qc"]["target_api"] = args.target_api
    if getattr(args, "dds_format", None):
        cfg["output"]["dds_format"] = args.dds_format


def cmd_qc(args) -> int:
    """Contrôle qualité seul (pas d'export) : met à jour les résultats et le rapport."""
    from .pipeline import open_state
    from .postprocess import PostProcessor, items_from_state
    cfg = _cfg(args)
    _apply_qc_args(cfg, args)
    bus = _qc_bus(args, cfg)
    state = open_state(cfg)
    try:
        rep = PostProcessor(cfg, bus, target="none", fix=not args.no_fix, state=state).run(
            items_from_state(cfg, state, args.only))
    finally:
        state.close()
        bus.close()
    s = rep["summary"]
    return 1 if s.get("error") else 0


def cmd_export(args) -> int:
    from .ddsbuild import EncoderError
    from .pipeline import export
    cfg = _cfg(args)
    _apply_qc_args(cfg, args)
    bus = _qc_bus(args, cfg)
    out = sys.stderr if args.json_events else sys.stdout
    try:
        r = export(cfg, args.target, Path(args.out).resolve(), qc=not args.no_qc, fix=not args.no_fix,
                   encoder=args.encoder, events=bus, only=args.only)
    except EncoderError as exc:
        print(f"✗ {exc}", file=out)
        return 2
    finally:
        bus.close()
    print(f"{r['written']} fichier(s) écrit(s) dans {r['out']} ({r['skipped']} ignoré(s))", file=out)
    if args.target == "specialk":
        print("Special K : ce dossier doit être SK_Res\\inject\\textures du jeu (ou y copier les fichiers),", file=out)
        print("puis activer Inject=true dans la section [Textures.D3D11] du fichier de configuration de Special K.",
              file=out)
    return 1 if r["summary"].get("error") else 0


def cmd_postprocess(args) -> int:
    """Mode autonome : un dossier d'originaux + un dossier de sorties ComfyUI -> DDS contrôlés."""
    import copy

    from .config import DEFAULTS
    from .postprocess import PostProcessor, items_from_folders
    cfgp = Path(args.config)
    if cfgp.exists():
        cfg = load_config(cfgp)
    else:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["_base"] = str(Path.cwd())
    out_dir = Path(args.out).resolve()
    cfg["_work"] = str(out_dir.parent / f"{out_dir.name}_texforge_qc")     # hors du dossier d'injection
    _apply_qc_args(cfg, args)
    if args.scale:
        cfg["output"]["scale"] = args.scale
    log_out = sys.stderr if args.json_events else sys.stdout
    items = items_from_folders(cfg, Path(args.originals).resolve(), Path(args.generated).resolve(),
                               include_unmatched=args.include_unmatched,
                               log=lambda m: print(m, file=log_out))
    if not items:
        print("✗ aucune paire original / image générée trouvée", file=log_out)
        return 1
    bus = _qc_bus(args, cfg)
    try:
        rep = PostProcessor(cfg, bus, target=args.target, out_dir=out_dir, fix=not args.no_fix,
                            encoder=args.encoder).run(items)
    finally:
        bus.close()
    return 1 if rep["summary"].get("error") else 0


def cmd_selftest(args) -> int:
    from .selftest import main as selftest_main
    return selftest_main()


def main(argv=None) -> int:
    _utf8_console()
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] == "remaster":          # chaîne complète (interface graphique, scripts)
        from .remaster import main as remaster_main
        return remaster_main(args_in[1:])
    p = argparse.ArgumentParser(prog="texforge", description="Remaster des textures de jeux anciens : UI locale + ComfyUI")
    p.add_argument("--config", default="texforge.toml", help="fichier de configuration (défaut : ./texforge.toml)")
    p.add_argument("--version", action="version", version=f"TEXFORGE {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="créer une configuration d'exemple")
    s.add_argument("folder", nargs="?", default=".")
    s.set_defaults(fn=cmd_init)
    sub.add_parser("doctor", help="vérifier l'installation").set_defaults(fn=cmd_doctor)
    sub.add_parser("scan", help="analyser et router les textures d'entrée").set_defaults(fn=cmd_scan)
    s = sub.add_parser("route", help="résumé du routage (UI / ComfyUI / local)")
    s.add_argument("--report", metavar="FICHIER.csv", help="écrire le détail par texture")
    s.set_defaults(fn=cmd_route)
    sub.add_parser("estimate", help="estimer la durée (ComfyUI) ou le coût (cloud)").set_defaults(fn=cmd_estimate)
    s = sub.add_parser("run", help="traiter : UI, local, puis ComfyUI")
    s.add_argument("--provider", choices=["comfyui", "mock", "higgsfield"], help="remplace le fournisseur de la config")
    s.add_argument("--stage", action="append", choices=["ui", "local", "ai"], help="limiter à une étape (répétable)")
    s.add_argument("--json-events", action="store_true", help="événements JSON sur la sortie standard (pont Node.js)")
    s.add_argument("--verbose", action="store_true", help="afficher aussi le routage de chaque texture")
    s.add_argument("--quiet", action="store_true", help="pas de tableau de bord console")
    s.add_argument("--limit", type=int, help="nombre maximal de textures IA à traiter")
    s.add_argument("--only", help="motif de nom (ex. *1A2B*)")
    s.add_argument("--max-requests", type=int, help="plafond de requêtes payantes pour cette exécution")
    s.add_argument("--workers", type=int, help="requêtes en parallèle")
    s.add_argument("--redo-stale", action="store_true", help="refaire les textures générées avec d'anciens réglages")
    s.add_argument("--local-only", action="store_true", help="seulement l'agrandissement local (gratuit)")
    s.set_defaults(fn=cmd_run)
    sub.add_parser("status", help="état d'avancement").set_defaults(fn=cmd_status)
    s = sub.add_parser("failures", help="échecs consignés (traçabilité)")
    s.add_argument("--kind", help="filtrer : oom, timeout, server, workflow, cuda, validation, engine, image, qc, dds, error")
    s.add_argument("--last", type=int, default=30, help="nombre de lignes détaillées (30)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_failures)
    s = sub.add_parser("retry", help="remettre en file les textures en échec")
    s.add_argument("--kind", help="seulement ce type d'échec")
    s.add_argument("--route", choices=["ui", "comfyui", "local"], help="seulement cette route")
    s.add_argument("--include-fallback", action="store_true", help="refaire aussi les textures passées en repli")
    s.set_defaults(fn=cmd_retry)
    sub.add_parser("stop", help="arrêter proprement une exécution en cours").set_defaults(fn=cmd_stop)
    s = sub.add_parser("review", help="galerie avant/après")
    s.add_argument("--no-open", action="store_true", help="ne pas ouvrir le navigateur")
    s.set_defaults(fn=cmd_review)
    s = sub.add_parser("apply-review", help="appliquer les choix exportés depuis la galerie")
    s.add_argument("file")
    s.set_defaults(fn=cmd_apply_review)
    s = sub.add_parser("qc", help="contrôle qualité des résultats (sans export)")
    s.add_argument("--only", help="motif de nom (ex. *1A2B*)")
    s.add_argument("--no-fix", action="store_true", help="mesurer et rejeter, sans correctif automatique")
    s.add_argument("--workers", type=int, help="textures contrôlées en parallèle")
    s.add_argument("--json-events", action="store_true", help="événements JSON sur la sortie standard (pont Node.js)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_qc)
    s = sub.add_parser("export", help="contrôle qualité puis écriture des textures finales (DDS + mipmaps)")
    s.add_argument("--target", choices=["specialk", "png", "remix"], required=True)
    s.add_argument("--out", required=True, help="dossier de sortie")
    s.add_argument("--encoder", choices=["auto", "texconv", "builtin"], help="compresseur DDS (défaut : [qc] encoder)")
    s.add_argument("--dds-format", choices=["match", "auto", "dxt1", "dxt5", "bgra8", "bc7"],
                   help="remplace [output] dds_format")
    s.add_argument("--target-api", choices=["d3d9", "d3d10", "d3d11"], help="remplace [qc] target_api")
    s.add_argument("--no-qc", action="store_true", help="exporter sans contrôle qualité")
    s.add_argument("--no-fix", action="store_true", help="contrôle sans correctif : échec = version locale")
    s.add_argument("--only", help="motif de nom (ex. *1A2B*)")
    s.add_argument("--workers", type=int)
    s.add_argument("--json-events", action="store_true", help="événements JSON sur la sortie standard (pont Node.js)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_export)
    sub.add_parser("remaster", help="chaîne complète : scan, génération, QC, DDS (texforge remaster -h)",
                   add_help=False)
    s = sub.add_parser("postprocess", help="mode autonome : originaux + sorties ComfyUI -> DDS contrôlés")
    s.add_argument("--originals", required=True, help="dossier des textures d'origine (DDS, PNG…)")
    s.add_argument("--generated", required=True, help="dossier des images générées (sortie ComfyUI)")
    s.add_argument("--out", required=True, help="dossier de sortie (DDS + qc_report.json)")
    s.add_argument("--target", choices=["specialk", "png", "remix"], default="specialk")
    s.add_argument("--encoder", choices=["auto", "texconv", "builtin"])
    s.add_argument("--dds-format", choices=["match", "auto", "dxt1", "dxt5", "bgra8", "bc7"])
    s.add_argument("--target-api", choices=["d3d9", "d3d10", "d3d11"])
    s.add_argument("--scale", type=int, help="facteur d'agrandissement cible (défaut : [output] scale)")
    s.add_argument("--include-unmatched", action="store_true", help="exporter aussi les originaux sans image générée")
    s.add_argument("--no-fix", action="store_true")
    s.add_argument("--workers", type=int)
    s.add_argument("--json-events", action="store_true")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_postprocess)
    sub.add_parser("selftest", help="test complet avec textures synthétiques (gratuit)").set_defaults(fn=cmd_selftest)

    args = p.parse_args(args_in)
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
