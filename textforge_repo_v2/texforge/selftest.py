"""Test complet sans réseau ni coût : textures synthétiques -> scan -> génération
(fournisseur mock) -> validation -> reprise -> export DDS/PNG -> relecture."""

from __future__ import annotations

import json

import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from . import dds
from .config import EXAMPLE_TOML, load_config
from .imaging import wrap_seam_ratio, to_f32
from .pipeline import Runner, export, open_state, run, scan, summary
from .providers import MockProvider


def _periodic_noise(h: int, w: int, rng, terms: int = 10) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w), np.float32)
    for _ in range(terms):
        fx, fy = rng.integers(1, 6), rng.integers(1, 6)
        ph = rng.random() * 6.28
        out += np.sin(2 * np.pi * (fx * x / w + fy * y / h) + ph)
    return out / terms


def make_textures(folder: Path) -> dict:
    rng = np.random.default_rng(7)
    folder.mkdir(parents=True, exist_ok=True)
    # 1. briques répétables, DXT1 avec mipmaps
    h = w = 64
    y, x = np.mgrid[0:h, 0:w]
    row = y // 16
    mortar = ((y % 16) < 2) | (((x + (row % 2) * 16) % 32) < 2)
    n = _periodic_noise(h, w, rng)
    grain = rng.normal(0, 9, (h, w))       # grain de matière, comme une vraie texture
    rgb = np.stack([150 + 30 * n + grain, 70 + 15 * n + grain, 50 + 10 * n + grain], -1)
    rgb[mortar] = [190, 185, 175]
    bricks = np.concatenate([rgb, np.full((h, w, 1), 255)], -1).clip(0, 255).astype(np.uint8)
    (folder / "Compressed_BRICKS01.dds").write_bytes(dds.encode_dds(Image.fromarray(bricks, "RGBA"), "DXT1", 99))
    # 2. feuillage découpé (alpha binaire), DXT5 avec mipmaps
    leaf = np.zeros((64, 64, 4), np.uint8)
    leaf[..., 0], leaf[..., 1], leaf[..., 2] = 40, 120, 30
    for cx, cy, r in ((20, 20, 12), (44, 26, 14), (30, 46, 13)):
        m = (x - cx) ** 2 + (y - cy) ** 2 < r * r
        leaf[m, 3] = 255
        leaf[m, 1] = 140 + (x[m] % 7) * 6
    (folder / "Compressed_LEAVES02.dds").write_bytes(dds.encode_dds(Image.fromarray(leaf, "RGBA"), "DXT5", 99))
    # 3. vitre (alpha en dégradé), BGRA8 sans mipmaps
    gy, gx = np.mgrid[0:32, 0:32]
    glass = np.stack([120 + gx * 3, 160 + gy * 2, np.full_like(gx, 210), 60 + gx * 5], -1).clip(0, 255).astype(np.uint8)
    (folder / "Uncompressed_GLASS03.dds").write_bytes(dds.encode_dds(Image.fromarray(glass, "RGBA"), "BGRA8", 1))
    # 4. minuscule -> gardée
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(folder / "tiny_lut.png")
    # 5. bande très allongée -> agrandissement local
    strip = (np.stack([np.tile(np.arange(128) * 2, (16, 1))] * 3 + [np.full((16, 128), 255)], -1)).astype(np.uint8)
    Image.fromarray(strip, "RGBA").save(folder / "strip_hud.png")
    # 6. carte de relief -> agrandissement local
    nm = np.stack([128 + 20 * n, 128 - 20 * n, np.full((64, 64), 240), np.full((64, 64), 255)], -1).clip(0, 255).astype(np.uint8)
    (folder / "Compressed_NORMAL04.dds").write_bytes(dds.encode_dds(Image.fromarray(nm, "RGBA"), "DXT5", 99))
    # 7. panneau non répétable 64x32 (PNG)
    sign = np.zeros((32, 64, 4), np.uint8)
    sign[..., 3] = 255
    sx = np.arange(64)[None, :]
    sign[..., 0], sign[..., 1], sign[..., 2] = 60 + sx * 2, 40 + sx, 20 + (sx // 2)   # dégradé : bords différents
    sign[6:26, 8:56, :3] = [220, 210, 160]
    sign[12:20, 14:30, :3] = [30, 30, 120]
    sign[10:22, 36:50, :3] = [160, 30, 30]
    Image.fromarray(sign, "RGBA").save(folder / "sign_shop.png")
    # 8. format 16 bits R5G6B5
    r565 = bricks.copy()
    r565[..., :3] = np.roll(bricks[..., :3], 5, axis=1) // 2 + 60
    (folder / "Uncompressed_R565.dds").write_bytes(dds.encode_dds(Image.fromarray(r565, "RGBA"), "R5G6B5", 99))
    # 9. carrelage répétable avec le joint d'un seul côté (bord gauche et haut)
    tiles = np.zeros((64, 64, 4), np.uint8)
    tiles[..., 3] = 255
    tiles[..., :3] = (np.stack([200 + 15 * n + grain, 195 + 15 * n + grain, 185 + 15 * n + grain], -1)).clip(0, 255).astype(np.uint8)
    tiles[:, :3, :3] = [70, 70, 75]
    tiles[:3, :, :3] = [70, 70, 75]
    (folder / "Compressed_TILES05.dds").write_bytes(dds.encode_dds(Image.fromarray(tiles, "RGBA"), "DXT1", 99))
    # 10. dump D3D11 typique (jeu DX9 passé par dgVoodoo) : BC3 sRGB avec en-tête DX10
    dx = np.roll(bricks, 9, axis=0).copy()
    dx[..., 2] = np.clip(dx[..., 2].astype(int) + 40, 0, 255)
    (folder / "Compressed_DX10SRGB.dds").write_bytes(dds.encode_dds(Image.fromarray(dx, "RGBA"), "BC3_SRGB", 99))
    # 11. roche non répétable (bruit aléatoire lissé + grain)
    field = rng.normal(0, 1, (16, 16))
    field = np.asarray(Image.fromarray(((field - field.min()) / np.ptp(field) * 255).astype(np.uint8)).resize((64, 64), Image.Resampling.BICUBIC), np.float32) / 255
    rock = np.stack([95 + 70 * field, 90 + 65 * field, 82 + 60 * field], -1) + rng.normal(0, 10, (64, 64, 1))
    rock = np.concatenate([rock, np.full((64, 64, 1), 255)], -1).clip(0, 255).astype(np.uint8)
    (folder / "Compressed_ROCK06.dds").write_bytes(dds.encode_dds(Image.fromarray(rock, "RGBA"), "DXT1", 99))
    return {"ai": {"Compressed_BRICKS01", "Compressed_LEAVES02", "Uncompressed_R565", "Compressed_TILES05",
                   "Compressed_DX10SRGB", "Compressed_ROCK06"},
            "ui": {"sign_shop", "strip_hud"},       # strip_hud : routé par son nom (« hud »)
            "upscale": {"Compressed_NORMAL04", "Uncompressed_GLASS03"}, "keep": {"tiny_lut"}}


class _PaidMock(MockProvider):
    """Mock qui se comporte comme un fournisseur payant pour tester la reprise."""
    paid = True

    def __init__(self, cfg, state=None):
        super().__init__(cfg, state)
        self.generated = 0
        self.resumed = 0
        self._last_input = None

    def generate(self, image, refs, prompt, negative, aspect_ratio, seed, on_submitted):
        self.generated += 1
        return super().generate(image, refs, prompt, negative, aspect_ratio, seed, on_submitted)

    def resume(self, request_id):
        self.resumed += 1
        img = Image.new("RGB", (1024, 1024), (128, 128, 128))
        return img, "mock://resumed"


def _check(cond: bool, msg: str, failures: list) -> None:
    print(f"  {'✓' if cond else '✗'} {msg}")
    if not cond:
        failures.append(msg)


def main() -> int:
    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="texforge_selftest_"))
    try:
        inp = tmp / "entree"
        expected = make_textures(inp)
        (tmp / "texforge.toml").write_text(
            EXAMPLE_TOML.replace('name = "comfyui"', 'name = "mock"'), encoding="utf-8")
        cfg = load_config(tmp / "texforge.toml")

        print("1. Analyse")
        c = scan(cfg, log=lambda *_: None)
        _check(c["seen"] == 11 and c["errors"] == 0, f"11 textures lues sans erreur ({c})", failures)
        st = open_state(cfg)
        modes = {r["name"]: r["mode"] for r in st.all()}
        an = {r["name"]: r["analysis"] for r in st.all()}
        st.close()
        for mode, names in expected.items():
            for nm in names:
                _check(modes.get(nm) == mode, f"{nm} -> {mode} (obtenu {modes.get(nm)})", failures)
        _check(an["Compressed_BRICKS01"]["tile_x"] and an["Compressed_BRICKS01"]["tile_y"], "briques détectées répétables", failures)
        _check(not an["sign_shop"]["tile_x"], "panneau détecté non répétable", failures)
        _check(an["Compressed_TILES05"]["tile_x"] and an["Compressed_TILES05"]["tile_y"], "carrelage (joint au bord) détecté répétable", failures)
        _check(an["Compressed_LEAVES02"]["alpha_class"] == "binary", "feuillage : alpha binaire", failures)
        _check(an["Uncompressed_GLASS03"]["alpha_class"] == "gradient", "vitre : alpha en dégradé", failures)

        print("2. Génération (mock)")
        stats = run(cfg, log=lambda *_: None)
        _check(stats["ai"] == 6 and stats["fallback"] == 0, f"6 textures IA validées ({stats})", failures)
        _check(stats["local"] == 2, "2 textures agrandies localement (relief, dégradé lisse)", failures)
        _check(stats["ui"] == 2, "2 graphismes plats traités par le pipeline UI (panneau, HUD)", failures)
        ev = (Path(cfg["_work"]) / "events.jsonl").read_text(encoding="utf-8")
        _check('"event": "ui_done"' in ev and '"event": "comfy_done"' in ev and '"event": "run_end"' in ev,
               "événements UI, IA et fin consignés dans events.jsonl", failures)
        st = open_state(cfg)
        recs = {r["name"]: r for r in st.all()}
        req_before = st.count_requests()
        st.close()
        b = recs["Compressed_BRICKS01"]
        img = Image.open(b["result_path"])
        _check(img.size == (256, 256), f"briques agrandies x4 ({img.size})", failures)
        seam = max(wrap_seam_ratio(to_f32(img), 0), wrap_seam_ratio(to_f32(img), 1))
        _check(seam < 1.8, f"briques toujours répétables après génération (raccord {seam:.2f})", failures)
        leaves = np.asarray(Image.open(recs["Compressed_LEAVES02"]["result_path"]))
        _check(set(np.unique(leaves[..., 3])) <= {0, 255}, "feuillage : alpha resté binaire", failures)
        _check(Image.open(recs["sign_shop"]["result_path"]).size == (256, 128), "panneau 64x32 -> 256x128", failures)
        _check(str(recs["sign_shop"]["metrics"].get("method", "")).startswith("ui:"),
               f"panneau traité sans IA générative ({recs['sign_shop']['metrics'].get('method')})", failures)

        print("3. Reprise sans doublon")
        stats2 = run(cfg, log=lambda *_: None)
        st = open_state(cfg)
        _check(stats2["ai"] == 0 and st.count_requests() == req_before, "2e exécution : aucune nouvelle requête", failures)
        # Requête « payée » restée en suspens : doit être reprise, pas relancée
        tid = recs["Compressed_ROCK06"]["id"]
        st.update(tid, status="pending", attempts=0, ai_path=None, metrics=None, best_score=None)
        from .imaging import PrepInfo
        prep = PrepInfo(64, 64, 0, 0, 64, 64, "1:1", "stretch", (0, 0, 1, 1), 1024, 1024)
        st.add_request("req-en-suspens", tid, 0, 1, prep.to_dict())
        st.close()
        paid = _PaidMock(cfg)
        Runner(cfg, None, lambda *_: None, provider=paid).run()
        _check(paid.resumed == 1, "requête en suspens reprise (pas repayée)", failures)
        _check(paid.generated >= 1, "image grise de reprise rejetée puis nouvel essai", failures)

        print("4. Rejet d'un résultat décalé")
        cfg_bad = load_config(tmp / "texforge.toml")
        cfg_bad["mock"]["shift"] = 0.12
        cfg_bad["provider"]["max_attempts"] = 1
        cfg_bad["_work"] = str(tmp / "work_rejet")
        scan(cfg_bad, log=lambda *_: None)
        s_bad = run(cfg_bad, only="Compressed_ROCK06", log=lambda *_: None)
        _check(s_bad["fallback"] == 1, f"résultat décalé refusé, repli local ({s_bad})", failures)
        st = open_state(cfg_bad)
        fails = st.failures()
        st.close()
        _check(any(f["kind"] == "validation" and f["action"] == "fallback" for f in fails),
               "rejet tracé dans la table des échecs (validation → repli)", failures)

        print("5. Plafond de requêtes")
        cfg_b = load_config(tmp / "texforge.toml")
        cfg_b["_work"] = str(tmp / "work_budget")
        cfg_b["provider"]["max_requests"] = 2
        scan(cfg_b, log=lambda *_: None)
        paid2 = _PaidMock(cfg_b)
        sb = Runner(cfg_b, None, lambda *_: None, provider=paid2).run()
        _check(paid2.generated == 2 and sb.get("budget"), f"arrêt à 2 requêtes ({paid2.generated})", failures)

        print("6. Export")
        out = tmp / "SK_Res" / "inject" / "textures"
        r = export(cfg, "specialk", out, log=lambda *_: None)
        _check(r["written"] == 10, f"10 fichiers DDS écrits ({r['summary']})", failures)
        rep = json.loads((out / "qc_report.json").read_text(encoding="utf-8"))
        for t in rep["textures"]:
            print(f"     QC {t['name']:24s} {t['decision']:10s} {'; '.join(t['reasons'])[:70]} "
                  f"{'| ' + '; '.join(t['warnings'])[:60] if t['warnings'] else ''}")
        checks = {"Compressed_BRICKS01.dds": ("DXT1", 256, 9), "Compressed_LEAVES02.dds": ("DXT5", 256, 9),
                  "Uncompressed_GLASS03.dds": ("BGRA8", 128, 8), "Uncompressed_R565.dds": ("R5G6B5", 256, 9),
                  "sign_shop.dds": ("BGRA8", 256, 9), "Compressed_NORMAL04.dds": ("DXT5", 256, 9),
                  "Compressed_DX10SRGB.dds": ("BC3_SRGB", 256, 9)}
        for fname, (fmt, width, mips) in checks.items():
            data = (out / fname).read_bytes()
            img, info = dds.load_dds(data)
            _check((info.format, info.width, info.mipmaps) == (fmt, width, mips),
                   f"{fname} : {info.format} {info.width}px {info.mipmaps} niveaux", failures)
        _check(not (out / "tiny_lut.dds").exists(), "texture minuscule non exportée (originale conservée)", failures)
        by = {t["name"]: t for t in rep["textures"]}
        _check(not any(t["decision"] == "error" for t in rep["textures"]), "aucune erreur QC ni DDS", failures)
        lv = by["Compressed_LEAVES02"]["output"]
        _check(lv["coverage_drift"] is not None and lv["coverage_drift"] < 0.03,
               f"feuillage : couverture alpha conservée dans les mipmaps (écart {lv['coverage_drift']})", failures)
        bo = by["Compressed_BRICKS01"]["output"]
        _check(bo["brightness_drift"] is not None and bo["brightness_drift"] < 0.05,
               f"briques : luminosité stable d'un mip à l'autre (écart {bo['brightness_drift']})", failures)
        nmap = by["Compressed_NORMAL04"]["output"]
        _check(nmap["color_space"] == "normal", "normal map : mipmaps renormalisées", failures)
        r2 = export(cfg, "png", tmp / "png_out", log=lambda *_: None)
        _check(r2["written"] == 10, "export PNG", failures)

        print("6b. Contrôle qualité : défauts injectés")
        from .events import EventBus
        from .postprocess import PostProcessor, items_from_state
        st = open_state(cfg)
        recs = {rr["name"]: rr for rr in st.all()}
        # teinte bleue sur les briques (dérive de couleur : à corriger)
        im = np.asarray(Image.open(recs["Compressed_BRICKS01"]["result_path"]).convert("RGBA")).astype(np.float32)
        im[..., :3] *= [0.8, 0.95, 1.3]
        Image.fromarray(im.clip(0, 255).astype(np.uint8), "RGBA").save(recs["Compressed_BRICKS01"]["result_path"])
        # objet inventé au milieu de la roche (hallucination : à rejeter)
        im = np.asarray(Image.open(recs["Compressed_ROCK06"]["result_path"]).convert("RGBA")).copy()
        H, W = im.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        im[(yy - H * 0.45) ** 2 + (xx - W * 0.5) ** 2 < (H * 0.15) ** 2, :3] = [20, 15, 15]
        Image.fromarray(im, "RGBA").save(recs["Compressed_ROCK06"]["result_path"])
        # choix manuel en galerie + teinte : mesuré, jamais modifié
        st.update(recs["Uncompressed_R565"]["id"], reviewed=1)
        im = np.asarray(Image.open(recs["Uncompressed_R565"]["result_path"]).convert("RGBA")).astype(np.float32)
        im[..., :3] *= [1.3, 1.0, 0.7]
        Image.fromarray(im.clip(0, 255).astype(np.uint8), "RGBA").save(recs["Uncompressed_R565"]["result_path"])
        bus = EventBus(cfg["_work"], console=False)
        rep2 = PostProcessor(cfg, bus, target="none", state=st).run(items_from_state(cfg, st))
        by2 = {t["name"]: t for t in rep2["textures"]}
        b2, k2, m2 = by2["Compressed_BRICKS01"], by2["Compressed_ROCK06"], by2["Uncompressed_R565"]
        _check(b2["decision"] == "corrected", f"teinte bleue corrigée ({b2['decision']} : {'; '.join(b2['reasons'][:2])})",
               failures)
        _check(b2["decision"] != "corrected" or b2["metrics"]["after"]["delta_e"] < 3,
               f"couleurs ramenées à l'original (ΔE {b2['metrics']['before'].get('delta_e')} → "
               f"{b2['metrics']['after'].get('delta_e')})", failures)
        _check(k2["decision"] == "rejected", f"objet inventé rejeté ({'; '.join(k2['reasons'][:2])})", failures)
        row = st.get(recs["Compressed_ROCK06"]["id"])
        _check(row["status"] == "fallback" and any(f["kind"] == "qc" for f in st.failures(tid=row["id"])),
               "rejet tracé : statut repli, échec « qc » (texforge retry --kind qc)", failures)
        _check(m2["decision"] == "measured" and any("aurait refusé" in w for w in m2["warnings"]),
               f"choix manuel respecté : mesuré, signalé, non modifié ({m2['decision']})", failures)
        rep3 = PostProcessor(cfg, bus, target="none", state=st).run(items_from_state(cfg, st))
        cached = sum(1 for t in rep3["textures"] if t.get("cached"))
        _check(cached >= 8, f"2e contrôle : résultats déjà vérifiés réutilisés ({cached}/{len(rep3['textures'])})",
               failures)
        bus.close()
        st.close()

        print("7. Galerie")
        from .review import build_review
        page = build_review(cfg)
        html = page.read_text(encoding="utf-8")
        _check(page.exists() and "Compressed_BRICKS01" in html, "galerie générée", failures)
        s = summary(cfg)
        _check(s["total"] == 11, "résumé cohérent", failures)
    except Exception:
        traceback.print_exc()
        failures.append("exception")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        print(f"SELFTEST : {len(failures)} échec(s)")
        return 1
    print("SELFTEST OK")
    return 0
