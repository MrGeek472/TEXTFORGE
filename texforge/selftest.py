"""Test complet sans réseau ni coût : textures synthétiques -> scan -> génération
(fournisseur mock) -> validation -> reprise -> export DDS/PNG -> relecture."""

from __future__ import annotations

import json

import shutil
import struct
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import dds
from .config import EXAMPLE_TOML, load_config
from .events import _fmt
from .i18n import LANGUAGES, check_catalogs, get_lang, tr
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


def _fake_ncnn(exe, images, *, model, scale, work_dir, log=None, **_kw):
    """Remplace realesrgan-ncnn-vulkan : x4plus = net + quadrillage inventé sur la roche (le défaut réel
    de ce modèle sur les matières bruitées) ; animevideov3 = prudent (lisse)."""
    out = {}
    for key, img in images.items():
        big = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
        arr = np.asarray(big, dtype=np.float32)
        if model == "realesrgan-x4plus":
            arr = arr + 0.8 * (arr - np.asarray(big.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32))
            if "ROCK" in key:
                yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
                arr[(yy % 6 == 0) | (xx % 6 == 0)] *= 0.55
        out[key] = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return out


def _selftest_esrgan(tmp: Path, inp: Path, failures: list) -> None:
    from . import esrgan
    from .events import EventBus
    from .orchestrator import Orchestrator
    from .postprocess import PostProcessor, items_from_state
    from .qc import measure as _m  # noqa: F401  (le contrôle qualité tourne comme pour ComfyUI)
    base = tmp / "esrgan"
    base.mkdir()
    (base / "texforge.toml").write_text(EXAMPLE_TOML.replace("inputs = ['./entree']", f"inputs = ['{inp.as_posix()}']")
                                        .replace('name = "comfyui"', 'name = "esrgan"'), encoding="utf-8")
    cfg = load_config(base / "texforge.toml")
    scan(cfg, log=lambda *_: None)
    real = (esrgan.run_ncnn, esrgan.find_realesrgan, esrgan._has_models)
    esrgan.run_ncnn = _fake_ncnn
    esrgan.find_realesrgan = lambda cfg=None: Path("realesrgan-ncnn-vulkan")
    esrgan._has_models = lambda exe, model="", scale=4: True
    try:
        bus = EventBus(cfg["_work"], console=False)
        stats = Orchestrator(cfg, bus, provider_name="esrgan").run()
        st = open_state(cfg)
        rows = {r["name"]: r for r in st.all()}
        _check(stats["ai"] == 6 and stats["errors"] == 0, f"6 matières agrandies par Real-ESRGAN ({stats['ai']})", failures)
        rock = (rows["Compressed_ROCK06"].get("metrics") or {})
        brick = (rows["Compressed_BRICKS01"].get("metrics") or {})
        _check(rock.get("guard", 0) > 0.5 > brick.get("guard", 1),
               f"garde anti-invention : roche quadrillée {rock.get('guard')} > briques {brick.get('guard')}", failures)
        ev = (Path(cfg["_work"]) / "events.jsonl").read_text(encoding="utf-8")
        _check(ev.count('"event": "esrgan_done"') == 6, "événements esrgan_done", failures)
        rep = PostProcessor(cfg, bus, target="none", state=st).run(items_from_state(cfg, st))
        dec = {t["name"]: t["decision"] for t in rep["textures"]}
        ok = all(dec[n] in ("validated", "corrected") for n in ("Compressed_BRICKS01", "Compressed_ROCK06",
                                                                 "Compressed_TILES05", "Compressed_LEAVES02"))
        _check(ok, f"contrôle qualité : rendus acceptés ({dec})", failures)
        bt = next(t for t in rep["textures"] if t["name"] == "Compressed_BRICKS01")
        seam = bt["metrics"]["after"].get("seam", {})
        _check(seam and all(v["full"] <= v["limit_full"] for v in seam.values()), "briques : raccord conservé", failures)
        st.close()
        bus.close()
    finally:
        esrgan.run_ncnn, esrgan.find_realesrgan, esrgan._has_models = real
    from .esrgan import band_noise, blur
    n = band_noise(64, 64, 0.125, 0.5, np.random.default_rng(1))
    across = float(np.abs(n[:, 0] - n[:, -1]).mean())           # de part et d'autre du bord
    inside = float(np.abs(n[:, 1:] - n[:, :-1]).mean())         # entre colonnes voisines
    _check(abs(float(n.std()) - 1) < 1e-3 and across < 1.3 * inside,
           f"grain de matière : écart type 1, périodique (bord {across:.2f} / intérieur {inside:.2f})", failures)
    a = np.random.default_rng(2).random((32, 32)).astype(np.float32)
    _check(abs(float(blur(a, 2.0, True, True).mean()) - float(a.mean())) < 1e-4, "flou FFT : moyenne conservée", failures)


def _selftest_locate(lib: Path, dump: Path, failures: list) -> None:
    """Dossier de départ choisi à la main -> dossier des textures."""
    from .detect import locate_textures

    def mk(d: Path, n: int, ext: str = ".dds") -> None:
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"t{i}{ext}").write_bytes(b"x")

    r = locate_textures(lib / "Vieux Jeu", profiles=[])
    _check(r.kind == "specialk" and Path(r.path) == dump and r.changed and r.inject, "jeu -> son dump Special K", failures)
    inj = dump.parent.parent / "inject" / "textures"
    inj.mkdir(parents=True, exist_ok=True)
    r = locate_textures(inj, profiles=[])
    _check(Path(r.path) == dump, "dossier d'injection choisi par erreur -> dump du même jeu", failures)
    loose = lib / "Jeu Fichiers"
    mk(loose / "Data" / "Textures", 120)
    mk(loose / "Data" / "UI", 8, ".tga")
    mk(loose / "Launcher", 2, ".png")
    r = locate_textures(loose, profiles=[])
    _check(r.kind == "loose" and Path(r.path) == loose / "Data" and r.count == 128,
           f"textures en fichiers -> dossier qui les regroupe, interface comprise ({r.path}, {r.count})", failures)
    unity = lib / "Jeu Unity"
    (unity / "Jeu_Data").mkdir(parents=True)
    (unity / "Jeu.exe").write_bytes(b"x")
    (unity / "Jeu_Data" / "sharedassets0.assets").write_bytes(b"x")
    r = locate_textures(unity, profiles=[])
    _check(r.kind == "none" and not r.changed and "Unity" in r.summary, f"Unity sans dump : explication ({r.summary})",
           failures)
    prof = lib / "_profils" / "Jeu Unity"
    mk(prof / "SK_Res" / "dump" / "textures", 5)
    r = locate_textures(unity, profiles=[(prof, prof.name, "profil Special K")])
    _check(r.kind == "specialk" and Path(r.path) == prof / "SK_Res" / "dump" / "textures"
           and Path(r.inject) == prof / "SK_Res" / "inject" / "textures", "jeu injecté par SKIF -> dump de son profil",
           failures)


def _selftest_ba2(tmp: Path, failures: list) -> None:
    """Fallout 4 simulé : textures lues dans les archives BA2, rendues en fichiers séparés au même chemin."""
    from . import ba2
    from .detect import locate_textures
    from .remaster import ProgressBus, RemasterSettings, run_remaster

    rng = np.random.default_rng(3)

    def tex(w: int, h: int, tint, alpha: bool = False) -> Image.Image:
        y, x = np.mgrid[0:h, 0:w]
        base = 0.5 + 0.25 * np.sin(x / 3.0) * np.cos(y / 5.0) + rng.normal(0, 0.06, (h, w))
        rgb = np.clip(base[..., None] * np.array(tint, np.float32), 0, 1)
        a = np.where((x // 8 + y // 8) % 2 == 0, 1.0, 0.0) if alpha else np.ones((h, w))
        return Image.fromarray((np.dstack([rgb, a]) * 255).astype(np.uint8), "RGBA")

    game = tmp / "Jeux" / "Fallout 4"
    data = game / "Data"
    (data / "Textures" / "Props").mkdir(parents=True)
    (game / "Fallout4.exe").write_bytes(b"MZ")
    wall_base, wall_mod = tex(64, 64, (0.9, 0.5, 0.4)), tex(64, 64, (0.3, 0.5, 0.9))
    base = [("Textures/Architecture/Wall01_d.DDS", dds.encode_dds(wall_base, "DXT1", 7)),
            ("Textures/Architecture/Wall01_n.DDS", dds.encode_dds(tex(64, 64, (0.5, 0.5, 1.0)), "DXT5", 7)),
            ("Textures/Props/Box_d.DDS", dds.encode_dds(tex(32, 32, (0.6, 0.6, 0.6), True), "DXT5", 6)),
            ("Textures/Props/Override_d.DDS", dds.encode_dds(tex(64, 64, (0.2, 0.8, 0.2)), "DXT1", 7))]
    ba2.write_texture_archive(data / "Fallout4 - Textures1.ba2", base, version=7)
    ba2.write_texture_archive(data / "MonMod - Textures.ba2",
                              [("Textures/Architecture/Wall01_d.DDS", dds.encode_dds(wall_mod, "DXT1", 7))], version=1)
    (data / "Fallout4 - Meshes.ba2").write_bytes(b"BTDX" + struct.pack("<I4sIQ", 1, b"GNRL", 0, 0))
    (data / "Textures" / "Props" / "Override_d.DDS").write_bytes(base[3][1])          # fichier séparé d'un mod

    with ba2.Ba2Archive(data / "Fallout4 - Textures1.ba2") as a:
        ok = [a.read_dds(t)[128:] == d[128:] for t, (_n, d) in zip(a.textures, base, strict=True)]
    _check(all(ok) and ba2.texture_archives(data) == [data / "Fallout4 - Textures1.ba2", data / "MonMod - Textures.ba2"],
           "archive BA2 (version 7) relue à l'identique ; archive de fichiers (GNRL) ignorée", failures)
    try:
        import lz4.block  # noqa: F401
        sf = tmp / "starfield.ba2"
        ba2.write_texture_archive(sf, base[:1], version=3, compression="lz4")
        with ba2.Ba2Archive(sf) as a:
            _check(a.compression == "lz4" and a.read_dds(a.textures[0])[128:] == base[0][1][128:],
                   "archive Starfield (version 3, LZ4) relue à l'identique", failures)
    except ImportError:
        print("  · module lz4 absent : archive Starfield LZ4 non testée")

    loc = locate_textures(game, profiles=[])
    _check(loc.kind == "ba2" and Path(loc.path) == data and loc.count == 5,
           f"dossier du jeu -> archives BA2 de Data ({loc.kind}, {loc.count} textures)", failures)
    out = tmp / "TEXFORGE_Remaster"
    s = RemasterSettings(input_dir=str(data), output_dir=str(out), work_dir=str(tmp / "projet_ba2"), target="loose",
                         provider="mock", scale=4, max_side=2048, mipmaps=True)
    bus = ProgressBus(tmp / "projet_ba2" / "work", console=False)
    try:
        code = run_remaster(s, bus)
    finally:
        bus.close()
    wall = out / "Textures" / "Architecture" / "Wall01_d.DDS"
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.DDS"))
    _check(code == 0 and files == ["Textures/Architecture/Wall01_d.DDS", "Textures/Props/Box_d.DDS"],
           f"sortie en fichiers séparés au chemin du jeu ; _n filtrée, fichier séparé du mod respecté ({code}, {files})",
           failures)
    if wall.exists():
        img, info = dds.load_dds(wall.read_bytes())
        small = np.asarray(img.convert("RGB").resize((64, 64), Image.Resampling.BOX), np.float32)
        d_mod = np.abs(small - np.asarray(wall_mod.convert("RGB"), np.float32)).mean()
        d_base = np.abs(small - np.asarray(wall_base.convert("RGB"), np.float32)).mean()
        _check(img.size == (256, 256) and info.mipmaps == 9 and d_mod < d_base,
               f"texture agrandie ×4 avec mipmaps, version du mod (chargée après) ({img.size}, {info.mipmaps} mips, "
               f"écart {d_mod:.0f} vs {d_base:.0f})", failures)


def _selftest_unreal(tmp: Path, failures: list) -> None:
    """Jeux Unreal : moteur reconnu à partir des en-têtes, explication au lieu d'un dossier « vide »."""
    from .detect import locate_textures
    from .unreal import PAK_MAGIC, UTOC_MAGIC, unreal_info

    def pak(path: Path, version: int, encrypted: bool, methods=("Oodle",)) -> None:
        names = b"".join(m.encode().ljust(32, b"\0") for m in methods)
        foot = (b"\0" * 16 + bytes([1 if encrypted else 0]) + struct.pack("<IIQQ", PAK_MAGIC, version, 0, 0)
                + b"\0" * 20 + names)
        path.write_bytes(b"\0" * 64 + foot)

    def utoc(path: Path, flags: int) -> None:
        h = bytearray(144)
        h[:16] = UTOC_MAGIC
        h[16] = 5
        struct.pack_into("<6I", h, 20, 144, 0, 0, 12, 0, 32)
        h[80] = flags
        path.write_bytes(bytes(h))

    ue5 = tmp / "Unreal" / "Jeu UE5"
    paks = ue5 / "Projet" / "Content" / "Paks"
    paks.mkdir(parents=True)
    (ue5 / "Projet" / "Binaries" / "Win64" / "D3D12").mkdir(parents=True)
    pak(paks / "Projet-Windows.pak", 11, True)
    utoc(paks / "Projet-Windows.utoc", 1 | 2 | 4 | 8)
    (paks / "Projet-Windows.ucas").write_bytes(b"\0" * 64)
    info = unreal_info(ue5)
    _check(info is not None and info.generation == "UE5" and info.encrypted and info.signed and info.d3d12
           and "Oodle" in info.compression, f"Unreal 5 : IoStore chiffré et signé reconnu ({info})", failures)
    r = locate_textures(ue5, profiles=[])
    _check(r.kind == "none" and "Unreal" in r.summary and "-dx11" in r.note and "Direct3D 12" in r.note,
           f"Unreal 5 : explication (capture Special K en Direct3D 11) ({r.summary})", failures)
    ue4 = tmp / "Unreal" / "Jeu UE4"
    p4 = ue4 / "Jeu" / "Content" / "Paks"
    p4.mkdir(parents=True)
    pak(p4 / "Jeu-WindowsNoEditor.pak", 8, False, ("Zlib",))
    info = unreal_info(p4)
    _check(info is not None and info.generation == "UE4" and not info.encrypted and info.compression == {"Zlib"},
           "Unreal 4 : archive .pak non chiffrée reconnue depuis le dossier Paks", failures)
    ue3 = tmp / "Unreal" / "Jeu UE3"
    (ue3 / "Game" / "CookedPC").mkdir(parents=True)
    (ue3 / "Game" / "CookedPC" / "Map01.upk").write_bytes(b"\xc1\x83\x2a\x9e")
    r = locate_textures(ue3, profiles=[])
    _check(r.kind == "none" and "Unreal Engine 3" in r.summary, f"Unreal 3 : paquets .upk reconnus ({r.summary})",
           failures)


def _selftest_tiles(cfg: dict, failures: list) -> None:
    """« Tout par l'IA » (routage), génération par morceaux (découpe, fondu, raccord) et données du
    panneau avant / après."""
    import copy
    from types import SimpleNamespace

    from .compare import describe
    from .imaging import (Analysis, analyze, finalize_ai, local_upscale_engine, prepare_tiled, tile_plan,
                          tile_starts, tile_weights)
    from .providers import ComfyUIProvider
    from .routing import decide

    c = copy.deepcopy(cfg)
    flat = Image.new("RGBA", (128, 128), (120, 110, 100, 255))
    an = analyze(flat, c)
    normal = np.zeros((128, 128, 4), np.uint8)
    normal[..., 0], normal[..., 1], normal[..., 2], normal[..., 3] = 128, 128, 255, 255
    nimg = Image.fromarray(normal, "RGBA")
    nan = analyze(nimg, c)
    before = (decide(flat, an, "wall", c).route, decide(nimg, nan, "wall_n", c).route)
    c["routing"]["force_ai"] = True
    after = (decide(flat, an, "wall", c).route, decide(nimg, nan, "wall_n", c).route)
    _check(before == ("local", "local") and after == ("comfyui", "local"),
           f"tout par l'IA : aplat envoyé à ComfyUI, normal map gardée hors IA ({before} -> {after})", failures)

    starts = tile_starts(1280, 512, 128)
    cover = np.zeros(1280)
    for s in starts:
        cover[s:s + 512] += 1
    _check(starts[0] == 0 and starts[-1] == 768 and cover.min() >= 1 and
           all(b - a <= 512 - 128 for a, b in zip(starts, starts[1:], strict=False)),
           f"découpe : tout couvert, recouvrement d'au moins 128 px ({starts})", failures)

    # texture répétable : morceaux générés avec un décalage de teinte différent chacun -> fondu sans marche
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:64, 0:64] / 64.0
    tex = np.zeros((64, 64, 3))
    for _ in range(6):                             # motif périodique : se raccorde exactement
        kx, ky, ph = rng.integers(1, 4), rng.integers(1, 4), rng.uniform(0, 6.3)
        tex += np.sin(2 * np.pi * (kx * xx + ky * yy) + ph)[..., None] * rng.uniform(5, 15, 3)
    tex = np.clip(tex + 128, 0, 255).astype(np.uint8)
    src = Image.fromarray(np.concatenate([tex, np.full((64, 64, 1), 255, np.uint8)], axis=-1), "RGBA")
    an_t = Analysis(64, 64, "none", True, True, 0.1, 0.1, False, "ai", "", "")
    c2 = copy.deepcopy(cfg)
    c2["comfyui"].update(tiled=True, tile_size=128, tile_overlap=32)
    plan = tile_plan(src, 8, c2)
    base, _e = local_upscale_engine(src, 8, an_t, c2)
    model_in, prep = prepare_tiled(src, an_t, 8, base, plan[1] if plan else 32)
    calls = []

    def fake_one(piece, ctx, prompt, negative, seed, tiled=False):
        calls.append(ctx["tiling"])
        a = np.asarray(piece, np.float32) + (len(calls) % 3) * 6.0      # chaque morceau un peu différent
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    stub = SimpleNamespace(backend=SimpleNamespace(on_progress=None), _generate_one=fake_one)
    out = ComfyUIProvider._generate_tiled(stub, model_in, {"stem": "t", "tiling": "enable"}, "", "", 1,
                                          plan[0] if plan else 128, plan[1] if plan else 32)
    final, _x = finalize_ai(out, src, an_t, prep, 8, c2)
    f = np.asarray(final, np.float32)[..., :3]
    step = np.abs(np.diff(f, axis=1)).mean(axis=2)
    ref_step = np.abs(np.diff(np.asarray(base, np.float32)[..., :3], axis=1)).mean(axis=2)
    wrap_jump = float(np.abs(f[:, 0] - f[:, -1]).mean())
    _check(plan is not None and final.size == (512, 512) and set(calls) == {"disable"}
           and step.max() <= ref_step.max() + 8 and wrap_jump <= ref_step.mean() * 4 + 3,
           f"par morceaux : {len(calls)} morceaux sans raccord circulaire, fondus sans marche "
           f"(saut max {step.max():.1f} vs {ref_step.max():.1f}), bords qui se raccordent ({wrap_jump:.1f})", failures)
    _check(tile_plan(Image.new("RGB", (256, 256)), 4, c2) is not None and tile_plan(Image.new("RGB", (32, 32)), 4, c2)
           is None and tile_plan(src, 8, cfg) is None,
           "par morceaux seulement si la taille finale dépasse un morceau et si l'option est active", failures)
    w = tile_weights(128, 128, 32, (True, False, False, True))
    _check(w[0, 64, 0] < 0.05 and w[127, 64, 0] == 1.0 and w[64, 127, 0] < 0.05 and w[64, 0, 0] == 1.0,
           "fondu seulement du côté des morceaux voisins", failures)

    # modèles : équivalent présent d'un modèle par défaut absent, famille Z-Image Turbo
    from . import comfyui_headless_client as chc
    from .providers import substitute_model
    from .providers import drop_model_patches
    from .remaster import DENOISE_SCALE, FAMILIES
    picks = [substitute_model("ckpt_name", "juggernautXL_ragnarok.safetensors",
                              ["old_sd15.safetensors", "sdxl_refiner_1.0.safetensors", "dreamshaperXL.safetensors"], "sdxl"),
             substitute_model("ckpt_name", "juggernautXL_ragnarok.safetensors", [], "sdxl"),
             substitute_model("unet_name", "z_image_turbo_bf16.safetensors", ["z_image_turbo_fp8_e4m3fn.safetensors"],
                              "zimage"),
             substitute_model("control_net_name", "xinsir-controlnet-tile-sdxl-1.0.safetensors",
                              ["control_v11f1e_sd15_tile.pth"], "sdxl")]
    _check(picks == ["dreamshaperXL.safetensors", None, "z_image_turbo_fp8_e4m3fn.safetensors", None],
           f"modèle par défaut absent : équivalent de la même famille seulement ({picks})", failures)
    zwf = chc.ComfyWorkflow.from_file(Path(__file__).resolve().parent.parent / FAMILIES["zimage"][0])
    zcls = {n["class_type"] for n in zwf.graph.values()}
    plain = drop_model_patches(zwf.graph)
    ks = next(n for n in plain.values() if n["class_type"] == "KSampler")
    _check({"UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingAuraFlow", "ModelPatchLoader",
            "ZImageFunControlnet"} <= zcls and "SeamlessTile" not in zcls and zwf.output_node_ids
           and not {"ModelPatchLoader", "ZImageFunControlnet"} & {n["class_type"] for n in plain.values()}
           and ks["inputs"]["model"][0] in plain and DENOISE_SCALE["zimage"][0] > 1 > DENOISE_SCALE["zimage"][1],
           "famille Z-Image Turbo : ControlNet Tile branché, retiré proprement si son fichier manque, denoise "
           "plus fort avec lui, plus doux sans lui", failures)

    # rendu photoréaliste créatif : prompts photo (sans matériau), contrôle qualité réduit, passe globale puis morceaux
    from .remaster import RemasterSettings, write_project_config
    tmpw = Path(tempfile.mkdtemp(prefix="texforge_photo_"))
    try:
        cfgp = load_config(write_project_config(RemasterSettings(
            input_dir=str(tmpw), output_dir=str(tmpw / "out"), work_dir=str(tmpw / "proj"), model_family="zimage",
            style="photoreal", subject="green armored soldier, gold visor", denoise=0.85)))
    finally:
        shutil.rmtree(tmpw, ignore_errors=True)
    prov = SimpleNamespace(cfg=cfgp, c=cfgp["comfyui"], wf_edges=None, denoise_scale=cfgp["comfyui"]["denoise_scale"])
    ctx = ComfyUIProvider.build_context(prov, {"name": "img", "analysis": {"material": "fabric"}},
                                        Analysis(64, 64, "none", False, False, 0.1, 0.1, False, "ai", "", ""))
    _check(ctx["style"] == "photoreal" and "weave" not in ctx["prompt"] and "gold visor" in ctx["prompt"]
           and "photorealistic" in ctx["prompt"] and abs(ctx["denoise"] - 0.85) < 1e-6 and 0.25 <= ctx["refine_denoise"] <= 0.45
           and cfgp["qc"]["min_ssim"] < 0.3 and cfgp["quality"]["min_structure"] < 0.5,
           f"photoréaliste : prompt sans matériau avec la description, denoise {ctx['denoise']} tel quel, "
           f"affinage {ctx['refine_denoise']}, contrôle qualité assoupli", failures)
    seen = []

    def fake_one2(piece, c2, prompt, negative, seed, tiled=False):
        seen.append((piece.size, c2["denoise"], c2["tiling"]))
        return piece.convert("RGB")
    stub2 = SimpleNamespace(backend=SimpleNamespace(on_progress=None), _generate_one=fake_one2, cfg=cfgp,
                            _generate_tiled=lambda *a, **k: ComfyUIProvider._generate_tiled(stub2, *a, **k),
                            _global_pass=lambda *a, **k: ComfyUIProvider._global_pass(stub2, *a, **k))
    big = Image.new("RGB", (2304, 2304), (90, 120, 80))
    res, _u = ComfyUIProvider.generate(stub2, big, [], "", "", "1:1", 3, lambda _r: None,
                                       context={**ctx, "tiles": {"size": 1024, "overlap": 128}})
    _check(res.size == big.size and seen[0][0][0] <= 1024 and seen[0][1] == ctx["denoise"]
           and all(d == ctx["refine_denoise"] and tl == "disable" for _s, d, tl in seen[1:]) and len(seen) == 1 + 9,
           f"photoréaliste : 1 passe globale à {ctx['denoise']} sur l'image entière ({seen[0][0]}), puis "
           f"{len(seen) - 1} morceaux à {ctx['refine_denoise']}", failures)

    # « Prompt à donner » : ajouté au prompt automatique, ou envoyé seul
    from .config import DEFAULTS as _D
    from .providers import compose_prompt
    base_c = {**_D["comfyui"], "subject": "scratched green armor,\n gold visor"}
    pa = compose_prompt(base_c, "mossy", "worn painted metal", False)
    pb = compose_prompt({**base_c, "prompt_only": True}, "mossy", "worn painted metal", True)
    pc = compose_prompt({**base_c, "style": "photoreal"}, "", "woven fabric", False)
    _check(pa.startswith(_D["comfyui"]["prompt_prefix"]) and "scratched green armor, gold visor" in pa
           and "worn painted metal" in pa and pb == "mossy, scratched green armor, gold visor" and "woven" not in pc,
           f"prompt à donner : ajouté (« {pa[:60]}… »), seul (« {pb} »), sans matériau en photoréaliste", failures)

    # denoise automatique : maximum du rendu, un palier plus bas seulement si la validation refuse
    from .remaster import AUTO_DENOISE, material_presets as presets_of
    pf = presets_of(RemasterSettings(model_family="sdxl", auto_denoise=True, denoise=0.3))
    pp = presets_of(RemasterSettings(model_family="sdxl", style="photoreal", auto_denoise=True, denoise=0.3))
    _check(abs(pf["generic"]["denoise"] - AUTO_DENOISE["faithful"][0]) < 1e-6 and abs(pp["generic"]["denoise"] - 1.0) < 1e-6,
           f"denoise automatique : curseur ignoré, maximum du rendu (fidèle {pf['generic']['denoise']}, "
           f"photoréaliste {pp['generic']['denoise']})", failures)
    calls3 = []

    def fake_one3(piece, c3, prompt, negative, seed, tiled=False):
        calls3.append((c3["stem"][-3:], c3["denoise"], piece.size))
        return piece.convert("RGB")
    verdicts = iter([False, True])                 # 1er palier refusé (texture transformée), 2e accepté
    stub3 = SimpleNamespace(backend=SimpleNamespace(on_progress=None), _generate_one=fake_one3, cfg=cfgp,
                            _faithful_enough=lambda ref, out: next(verdicts),
                            _probe_denoise=lambda *a, **k: ComfyUIProvider._probe_denoise(stub3, *a, **k),
                            _generate_tiled=lambda *a, **k: ComfyUIProvider._generate_tiled(stub3, *a, **k))
    fctx = {"style": "faithful", "denoise": 0.6, "denoise_ladder": AUTO_DENOISE["faithful"][1], "stem": "t",
            "tiling": "disable", "tiles": {"size": 1024, "overlap": 128}, "attempt": 0}
    ComfyUIProvider.generate(stub3, Image.new("RGB", (2304, 2304), (90, 90, 90)), [], "", "", "1:1", 1,
                             lambda _r: None, context=fctx)
    probes = [c for c in calls3 if "_p" in c[0]]
    tiles_d = {c[1] for c in calls3 if "_p" not in c[0]}
    _check([round(c[1], 3) for c in probes] == [0.6, 0.492] and tiles_d == {0.492}
           and all(max(c[2]) <= 1024 for c in probes),
           f"fidèle automatique : essais à 1 Mpx {[round(c[1], 3) for c in probes]}, morceaux au premier accepté "
           f"{sorted(tiles_d)}", failures)
    calls3.clear()
    ComfyUIProvider.generate(stub3, Image.new("RGB", (512, 512)), [], "", "", "1:1", 1, lambda _r: None,
                             context={**fctx, "tiles": None, "attempt": 2})
    _check([round(c[1], 3) for c in calls3] == [0.408],
           f"sans morceaux : le 3e essai de la texture passe au 3e palier ({[c[1] for c in calls3]})", failures)

    tmpf = Path(tempfile.mkdtemp(prefix="texforge_cmp_"))
    try:
        for n in ("ai", "res"):
            Image.new("RGB", (8, 8)).save(tmpf / f"{n}.png")
        rec = {"id": "x/a", "name": "a", "src_path": "a.dds", "status": "done", "choice": "auto", "width": 2,
               "height": 2, "updated": 1.0, "analysis": {"material": "metal"}, "metrics": {"method": "ai", "attempt": 1},
               "ai_path": str(tmpf / "ai.png"), "result_path": str(tmpf / "res.png")}
        kinds = [describe(rec).right_kind,
                 describe({**rec, "status": "fallback", "error": "QC"}).right_kind,
                 describe({**rec, "ai_path": None, "metrics": {"method": "local:realesrgan-x4plus"}}).right_kind,
                 describe({**rec, "ai_path": None, "result_path": None})]
        _check(kinds == ["ai", "ai_rejected", "local", None],
               f"panneau avant / après : image de droite et légende selon le traitement ({kinds})", failures)
    finally:
        shutil.rmtree(tmpf, ignore_errors=True)


def _selftest_robustness(tmp: Path, failures: list) -> None:
    """Dossier de jeu avec quelques images annexes, fichier source disparu, ComfyUI sans nœuds perso."""
    import os

    from . import comfyui_headless_client as chc
    from .detect import locate_textures
    from .providers import fit_workflow

    game = tmp / "Robustesse" / "Vieux Jeu D3D9"
    game.mkdir(parents=True)
    (game / "game.exe").write_bytes(b"MZ" + os.urandom(4096) + b"KERNEL32.dll\0d3d9.dll\0" + os.urandom(512))
    for i in range(4):
        (game / f"bigfile.{i:03d}").write_bytes(os.urandom(256))
    Image.new("RGBA", (256, 256), (90, 60, 40, 255)).save(game / "Icone.png")
    r = locate_textures(game, profiles=[])
    _check(r.kind == "none" and "Icone.png" in r.note and "Direct3D 9" in r.note and ".000" in r.note,
           f"dossier de jeu avec une seule icône : pas pris pour les textures ({r.summary})", failures)

    inp = tmp / "Robustesse" / "dump"
    inp.mkdir()
    for nm in ("a", "b"):
        Image.fromarray((np.random.default_rng(len(nm)).random((64, 64, 3)) * 255).astype(np.uint8)).save(inp / f"{nm}.png")
    base = tmp / "Robustesse" / "projet"
    base.mkdir()
    (base / "texforge.toml").write_text(EXAMPLE_TOML.replace("inputs = ['./entree']", f"inputs = ['{inp.as_posix()}']")
                                        .replace('name = "comfyui"', 'name = "mock"'), encoding="utf-8")
    cfg = load_config(base / "texforge.toml")
    scan(cfg, log=lambda *_: None)
    (inp / "b.png").unlink()
    c = scan(cfg, log=lambda *_: None)
    st = open_state(cfg)
    status = {r["name"]: r["status"] for r in st.all()}
    st.close()
    _check(c.get("missing") == 1 and status.get("b") == "missing" and status.get("a") != "missing",
           f"fichier source disparu : mis de côté au lieu d'être envoyé à la génération ({status})", failures)

    # agrandissement local fait par Lanczos faute de Real-ESRGAN : refait une fois quand Real-ESRGAN est réglé
    from .pipeline import Runner, local_to_redo
    cfg["local"]["esrgan_exe"] = ""
    run = Runner(cfg, "mock", lambda *_: None)
    rec_a = next(r for r in run.state.all() if r["name"] == "a")
    run.state.update(rec_a["id"], mode="upscale")
    run.do_local(run.state.get(rec_a["id"]))
    m1 = (run.state.get(rec_a["id"]) or {}).get("metrics") or {}
    fake = tmp / "Robustesse" / "outils" / ("realesrgan-ncnn-vulkan" + (".exe" if os.name == "nt" else ""))
    fake.parent.mkdir()
    fake.write_bytes(b"")                          # présent mais inutilisable : Real-ESRGAN échouera
    cfg["local"]["esrgan_exe"] = str(fake)
    redo1 = [r["name"] for r in local_to_redo(run.state, cfg)]
    run.do_local(run.state.get(rec_a["id"]))
    m2 = (run.state.get(rec_a["id"]) or {}).get("metrics") or {}
    redo2 = [r["name"] for r in local_to_redo(run.state, cfg)]
    run.state.close()
    cfg["local"]["esrgan_exe"] = ""
    _check(m1.get("method") == "local:lanczos" and redo1 == ["a"] and m2.get("esrgan") == str(fake) and redo2 == [],
           f"agrandi par Lanczos faute de Real-ESRGAN : refait une fois quand il est réglé, sans boucle "
           f"({m1.get('method')}, {redo1}, {redo2})", failures)

    wf = chc.ComfyWorkflow.from_file(Path(__file__).resolve().parent.parent / "workflows" / "texforge_sdxl_tile_api.json")
    classes = {n["class_type"] for n in wf.graph.values()}
    vanilla, bypassed, required = fit_workflow(wf, classes - {"SeamlessTile", "MakeCircularVAE", "OffsetImage",
                                                              "ColorTransfer"}, chc)
    dangling = [k for n in vanilla.graph.values() for k, v in n["inputs"].items()
                if isinstance(v, list) and len(v) == 2 and str(v[0]) not in vanilla.graph]
    _check(not required and set(bypassed) == {"SeamlessTile", "MakeCircularVAE", "OffsetImage", "ColorTransfer"}
           and not dangling and vanilla.output_node_ids,
           f"ComfyUI sans nœuds personnalisés : workflow contourné proprement ({bypassed})", failures)
    _w, _b, required = fit_workflow(wf, classes - {"ControlNetApplyAdvanced"}, chc)
    _check(required == ["ControlNetApplyAdvanced"], "nœud indispensable absent : signalé avant tout envoi", failures)

    # liste des nœuds illisible : le refus de /prompt suffit à reconnaître le nœud absent
    def refusal(err: dict) -> chc.WorkflowError:
        data = {"error": err, "node_errors": {}}
        return chc.WorkflowError("workflow refusé (400)", details=json.dumps(data))
    recent = refusal({"type": "missing_node_type", "message": "Node 'TEXFORGE_TILING' not found. The custom node "
                      "may not be installed.", "details": "Node ID '#3'",
                      "extra_info": {"node_id": "3", "class_type": "SeamlessTile", "node_title": "TEXFORGE_TILING"}})
    bare = refusal({"type": "missing_node_type", "message": "Node 'TEXFORGE_TILING_VAE' not found.", "details": "",
                    "extra_info": {}})
    old = refusal({"type": "invalid_prompt", "message": "Cannot execute because node ColorTransfer does not exist.",
                   "details": "Node ID '#40'", "extra_info": {}})
    other = refusal({"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                     "details": "", "extra_info": {}})
    got = [chc.missing_node_class(e, wf.graph) for e in (recent, bare, old, other)]
    _check(got == ["SeamlessTile", "MakeCircularVAE", "ColorTransfer", None],
           f"refus « nœud absent » de ComfyUI reconnu (récent, sans détail, ancien, autre refus) : {got}", failures)


def _selftest_normals(cfg: dict, failures: list) -> None:
    """Normal maps rangées autrement qu'en XYZ : jamais vues comme des couleurs, ni envoyées à une IA,
    canaux conservés à l'agrandissement et dans les mipmaps (pas de prémultiplication de l'alpha)."""
    from .ddsbuild import build_chain, plan_for
    from .imaging import analyze, local_upscale
    from .qc import QCItem, run_qc
    from .routing import decide

    rng = np.random.default_rng(11)
    hgt = _periodic_noise(64, 64, rng)
    gx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 6
    gy = (np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)) * 6
    nrm = np.stack([-gx, -gy, np.ones_like(hgt)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    X, Y = nrm[..., 0] * 0.5 + 0.5, nrm[..., 1] * 0.5 + 0.5
    one, zero = np.ones_like(X), np.zeros_like(X)
    h01 = (hgt - hgt.min()) / (hgt.max() - hgt.min())
    cases = {  # nom : (canaux RGBA, rangement attendu, canaux de X et Y, format d'origine)
        "ag": (np.stack([one, Y, one, X], -1), "ag", (3, 1), "DXT5"),
        "rg": (np.stack([X, Y, zero, one], -1), "rg", (0, 1), "BC5"),
        "xyz_hauteur": (np.concatenate([nrm * 0.5 + 0.5, h01[..., None]], -1), "xyz", (0, 1), "DXT5"),
    }
    for nm, (arr, want, (cx, cy), fmt) in cases.items():
        img = Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGBA")
        an = analyze(img, cfg)
        _check(an.normal_packing == want and an.normal_map, f"{nm} : normal map « {an.normal_packing} »", failures)
        route = decide(img, an, "3fa9c1d2", cfg).route
        _check(route == "local", f"{nm} : agrandie sans IA (route {route})", failures)
        up = local_upscale(img, 4, an, cfg)
        u = np.asarray(up, dtype=np.float32) / 255
        down = u.reshape(64, 4, 64, 4, 4).mean(axis=(1, 3))
        err = max(float(np.abs(down[..., cx] - arr[..., cx]).mean()), float(np.abs(down[..., cy] - arr[..., cy]).mean()))
        _check(up.size == (256, 256) and err < 0.02, f"{nm} : X et Y conservés à l'agrandissement (écart {err:.3f})",
               failures)
        if want == "ag":
            _check(u[..., 0].std() < 0.02 and u[..., 2].std() < 0.02, "ag : rouge et bleu restés vides", failures)
        plan = plan_for(up, an, fmt, True, 9, cfg, data=True, can_bc7=False)
        if want == "ag":
            _check(plan.alpha_mode == "data" and plan.color_space == "linear",
                   f"ag : mipmaps linéaires, alpha = donnée ({plan.color_space}, {plan.alpha_mode})", failures)
        if want == "rg":
            _check(plan.fmt == "BGRA8", f"rg : BC5 réécrit sans perte ({plan.fmt}, pas DXT1)", failures)
        chain = build_chain(up, plan)
        l1 = chain[1].astype(np.float32) / 255
        ref = u.reshape(128, 2, 128, 2, 4).mean(axis=(1, 3))
        e1 = max(float(np.abs(l1[..., cx] - ref[..., cx]).mean()), float(np.abs(l1[..., cy] - ref[..., cy]).mean()))
        _check(e1 < 0.03, f"{nm} : mipmap 1 fidèle à X et Y (écart {e1:.3f})", failures)
        res = run_qc(QCItem("n-" + nm, nm, img, an, up, "local", 4, data=True), cfg)
        _check(res.decision in ("validated", "local") and not res.reasons,
               f"{nm} : contrôle qualité sans faux échec ({res.decision} {'; '.join(res.reasons)[:80]})", failures)


def _selftest_eta(tmp: Path, cfg: dict, failures: list) -> None:
    """Estimation du temps : passages ComfyUI comptés comme le moteur les fera, recalage en cours de lot,
    durées apprises d'un lancement à l'autre."""
    import copy

    from . import eta
    from .remaster import ProgressBus

    c = copy.deepcopy(cfg)
    c["output"].update(scale=4, max_side=4096)
    c["comfyui"].update(tiled=True, tile_size=1024, tile_overlap=128, style="faithful", denoise_ladder=[])

    def row(w, h, tile=False, tid="t"):
        return {"id": tid, "width": w, "height": h, "analysis": {"tile_x": tile, "tile_y": tile}}

    counts = [eta.passes_for(row(1024, 1024), c), eta.passes_for(row(1024, 1024, True), c),
              eta.passes_for(row(2048, 1024), c), eta.passes_for(row(256, 256), c)]
    photo = copy.deepcopy(c)
    photo["comfyui"]["style"] = "photoreal"
    auto = copy.deepcopy(c)
    auto["comfyui"]["denoise_ladder"] = [1.0, 0.82, 0.68]
    flat = copy.deepcopy(c)
    flat["comfyui"]["tiled"] = False
    extra = [eta.passes_for(row(1024, 1024), photo), eta.passes_for(row(1024, 1024), auto),
             eta.passes_for(row(1024, 1024), flat)]
    _check(counts == [25, 25, 15, 1] and extra == [26, 26, 1],
           f"passages comptés comme le moteur : 4096² = 25 morceaux, 4096×2048 = 15, petite texture 1 ; "
           f"+1 passe globale ou essai du denoise ; sans morceaux 1 ({counts}, {extra})", failures)

    old = __import__("os").environ.get("TEXFORGE_TIMINGS")
    __import__("os").environ["TEXFORGE_TIMINGS"] = str(tmp / "timings_eta.json")
    try:
        per, est = eta.plan(c, "comfyui", {"ai": [row(1024, 1024, tid="a"), row(256, 256, tid="b")],
                                           "local": [row(512, 512, tid="l")]}, [row(1024, 1024, tid="a")])
        default = eta.DEFAULT_PASS_S[eta.family_of(c)]
        _check(est["passes"] == 26 and abs(est["ai_s"] - 26 * default) <= 1 and est["export"] == 1
               and est["total_s"] == est["generate_s"] + est["export_s"] and not est["pass_learned"],
               f"estimation : {est['passes']} passages × {default} s par défaut, export de 16,8 Mpx "
               f"({eta.describe(est)})", failures)

        bus = ProgressBus(tmp / "eta_work", console=False)
        t = [0.0]
        bus.clock = lambda: t[0]
        bus.set_stage("generate")
        bus.set_plan({"a": ("ai", 100.0, 10, 16.8), "b": ("ai", 100.0, 10, 16.8), "l": ("local", 2.0, 0, 4.0)},
                     {"export_s": 20, "passes": 20, "pass_s": 10.0})
        r0 = bus.remaining_s()                    # rien de commencé : 2 + 200 + 20 prévus
        bus.emit("stage_start", stage="local", count=1)
        t[0] = 1.0
        bus.emit("local_done", texture="l", name="l")
        bus.emit("stage_start", stage="comfyui", count=2)
        for tid, at in (("a", 16.0), ("b", 31.0)):
            t[0] = at
            bus.pass_done(tid)
        r1 = bus.remaining_s()                    # (30 s + 4 × 10 s prévus) / 6 = 11,7 s par passage, 18 à faire
        for i in range(19):                       # les deux textures avancent ensemble dans la file
            t[0] += 5.0
            bus.pass_done("a" if i < 11 else "b")
        bus.emit("comfy_done", texture="a", name="a")
        r2 = bus.remaining_s()                    # a : 12 passages pour 10 prévus (×1,2) ; b en a fait 9 sur 12
        t[0] += 10.0
        r3 = bus.remaining_s()
        for _ in range(3):
            t[0] += 5.0
            bus.pass_done("b")
        t[0] += 9.0                               # fin du lot : fondu, contrôle et PNG après le dernier passage
        bus.emit("comfy_done", texture="b", name="b")
        m = bus.measured()
        bus.close()
        got = [round(x) for x in (r0, r1, r2, r3)]
        _check(got == [222, 236, 39, 41],
               f"temps restant recalé sur les passages réels (textures en parallèle dans la file, essais en plus "
               f"compris) : {got}", failures)
        eta.learn(c, "comfyui", m)
        first = eta.rates(c)
        eta.learn(c, "comfyui", {"ai": (300.0, 10, 16.8)})
        second = eta.rates(c)
        eta.learn(c, "mock", {"ai": (999.0, 1, 1.0)})
        want1 = round(150 / (24 / 1.2), 2)
        want2 = round(0.5 * want1 + 0.5 * 30.0, 2)
        _per2, est2 = eta.plan(c, "comfyui", {"ai": [row(1024, 1024, tid="a")]}, [])
        _check(first["pass_learned"] and first["pass_s"] == want1 and first["tail_s"] == 9.0
               and second["pass_s"] == want2 and eta.rates(c)["pass_s"] == want2
               and abs(est2["ai_s"] - (25 * want2 + 9.0)) <= 1,
               f"durées apprises : passage prévu {first['pass_s']} s (essais en plus compris), fin du dernier lot "
               f"{first['tail_s']} s, moyenne glissante {second['pass_s']} s, simulation ignorée", failures)
        _check([eta.fmt(x) for x in (42, 125, 3725, 3 * 86400 + 7200)] == [tr("{s} s", s=42), tr("{m} min", m=2),
                                                                           tr("{h} h {m:02d}", h=1, m=2),
                                                                           tr("{d} j {h} h", d=3, h=2)],
               "durées lisibles et arrondies", failures)
    finally:
        if old is None:
            __import__("os").environ.pop("TEXFORGE_TIMINGS", None)
        else:
            __import__("os").environ["TEXFORGE_TIMINGS"] = old

    ev = tmp / "projet_ba2" / "work" / "events.jsonl"
    if ev.exists():
        lines = [json.loads(x) for x in ev.read_text(encoding="utf-8").splitlines() if x.strip()]
        ests = [x for x in lines if x.get("event") == "estimate"]
        after = lines[lines.index(ests[0]) + 1:] if ests else []
        rest = [x.get("eta_total_s") for x in after if x.get("event") == "progress"]
        _check(len(ests) == 1 and ests[0].get("export", 0) >= 2 and rest and all(r is not None for r in rest)
               and rest[-1] == 0,
               f"chaîne complète : estimation annoncée une fois (génération + export), temps restant dans chaque "
               f"avancement qui suit ({len(ests)}, {len(rest)} avancements, dernier {rest[-1:]})", failures)


def _selftest_stale(tmp: Path, inp: Path, failures: list) -> None:
    """Réglages changés entre deux lancements : chaque texture IA repart avec tous ses essais."""
    from .remaster import ProgressBus, RemasterSettings, run_remaster
    from .state import State

    work = tmp / "projet_reglages"
    seen = []
    for d in (0.42, 0.50, 0.58, 0.66):
        s = RemasterSettings(input_dir=str(inp), output_dir=str(tmp / "sortie_reglages"), work_dir=str(work),
                             provider="mock", target="png", denoise=d)
        bus = ProgressBus(work / "work", console=False)
        try:
            code = run_remaster(s, bus)
        finally:
            bus.close()
        st = State(work / "work" / "texforge_state.db")
        seen.append((code, sorted({(r["status"], r["attempts"]) for r in st.all() if r["mode"] == "ai"})))
        st.close()
    _check(all(c == 0 and rows == [("done", 1)] for c, rows in seen),
           f"denoise changé 3 fois : chaque texture IA regénérée dès le 1er essai, aucune passée en agrandissement "
           f"simple faute d'essais ({seen[-1]})", failures)


class FakeGitHub:
    """Faux GitHub local (API commits/HEAD, fichiers bruts, archive zip d'un commit) servant un dossier ;
    sert aussi aux tests de l'interface. Usage : with FakeGitHub(dossier, "1.2.3") as gh: gh.env()…"""

    def __init__(self, folder: Path, message: str = "Nouvelle version", repo: str = "test/TEXTFORGE") -> None:
        import hashlib
        import http.server
        import io
        import threading
        import zipfile
        self.repo, self.folder, self.message = repo, Path(folder), message
        self.sha = hashlib.sha1(str(folder).encode()).hexdigest()
        self.extra: dict[str, bytes] = {}          # entrées ajoutées telles quelles (chemins piégés)
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == f"/api/repos/{outer.repo}/commits/HEAD":
                    body = json.dumps({"sha": outer.sha, "commit": {"message": outer.message + "\n\ndétail",
                                       "committer": {"date": "2026-10-26T12:00:00Z"}}}).encode()
                elif path.startswith(f"/raw/{outer.repo}/"):
                    rel = path.split("/", 5)[-1]
                    f = outer.folder / rel
                    if not f.is_file():
                        self.send_error(404)
                        return
                    body = f.read_bytes()
                elif path.startswith(f"/codeload/{outer.repo}/zip/"):
                    buf = io.BytesIO()
                    top = f"TEXTFORGE-{outer.sha}/"
                    with zipfile.ZipFile(buf, "w") as z:
                        for f in sorted(outer.folder.rglob("*")):
                            if f.is_file():
                                z.write(f, top + f.relative_to(outer.folder).as_posix())
                        for name, data in outer.extra.items():
                            z.writestr(top + name, data)
                    body = buf.getvalue()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def env(self) -> dict:
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        return {"TEXFORGE_UPDATE_REPO": self.repo, "TEXFORGE_UPDATE_API": f"{base}/api",
                "TEXFORGE_UPDATE_RAW": f"{base}/raw", "TEXFORGE_UPDATE_CODELOAD": f"{base}/codeload"}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


def _project_copy(dest: Path, version: str) -> Path:
    """Copie de l'installation (code seulement) avec une autre version."""
    src = Path(__file__).resolve().parent.parent
    dest.mkdir(parents=True)
    for name in ("app_gui.py", "gui_compare.py", "requirements.txt"):
        if (src / name).is_file():
            shutil.copy2(src / name, dest / name)
    shutil.copytree(src / "texforge", dest / "texforge", ignore=shutil.ignore_patterns("__pycache__"))
    init = dest / "texforge" / "__init__.py"
    import re as _re
    init.write_text(_re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', init.read_text(encoding="utf-8")),
                    encoding="utf-8")
    from .updater import write_release
    write_release(dest)
    return dest


def _digest(folder: Path) -> dict:
    import hashlib
    return {f.relative_to(folder).as_posix(): hashlib.sha1(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob("*")) if f.is_file() and "__pycache__" not in f.parts}


def _selftest_update(tmp: Path, failures: list) -> None:
    """Mise à jour depuis GitHub (faux serveur local) : détection, refus d'un dépôt incomplet ou cassé,
    installation contrôlée, sauvegarde, retour arrière."""
    import os

    from . import __version__, updater

    base = tmp / "maj"
    root = _project_copy(base / "installation", __version__)
    good = _project_copy(base / "depot", "99.0.0")
    (good / "texforge" / "nouveau_module.py").write_text("VALEUR = 1\n", encoding="utf-8")
    eta = good / "texforge" / "eta.py"
    eta.write_text(eta.read_text(encoding="utf-8") + "\n# modifié dans le dépôt\n", encoding="utf-8")
    updater.write_release(good)
    before = _digest(root)
    saved = {k: os.environ.get(k) for k in ("TEXFORGE_UPDATE_REPO", "TEXFORGE_UPDATE_API", "TEXFORGE_UPDATE_RAW",
                                             "TEXFORGE_UPDATE_CODELOAD", "TEXFORGE_UPDATE_DIR", "TEXFORGE_UPDATE_ROOT")}
    os.environ["TEXFORGE_UPDATE_DIR"] = str(base / "donnees")
    os.environ["TEXFORGE_UPDATE_ROOT"] = str(root)
    try:
        with FakeGitHub(good, "Ajoute l'estimation du temps") as gh:
            os.environ.update(gh.env())
            info = updater.check()
            _check(info["available"] and info["remote"] == "99.0.0" and info["sha"] == gh.sha
                   and info["message"] == "Ajoute l'estimation du temps" and not info["error"],
                   f"dépôt plus récent détecté ({info['local']} -> {info['remote']}, « {info['message']} »)", failures)

            def refused(prepare, undo, expect: str, label: str) -> None:
                prepare()
                try:
                    updater.apply(updater.check())
                    ok, why = False, "installée"
                except updater.UpdateError as exc:
                    ok, why = expect in str(exc), str(exc)
                finally:
                    undo()
                _check(ok and _digest(root) == before, f"{label} : refusé, rien de modifié ({why[:110]})", failures)

            gc = good / "gui_compare.py"
            data = gc.read_bytes()
            refused(gc.unlink, lambda: gc.write_bytes(data), "gui_compare", "dépôt incomplet (gui_compare.py absent)")
            bad = good / "texforge" / "casse.py"
            refused(lambda: (bad.write_text("def f(:\n", encoding="utf-8"), updater.write_release(good)),
                    lambda: (bad.unlink(), updater.write_release(good)), "casse.py",
                    "erreur de syntaxe dans un fichier de la version")
            refused(lambda: gh.extra.update({"../evil.py": b"x = 1\n"}), gh.extra.clear, "evil.py",
                    "chemin sortant du dossier dans l'archive")
            app = good / "app_gui.py"
            app_data = app.read_bytes()
            refused(lambda: app.write_bytes(app_data + b"\n# ancien fichier\n"), lambda: app.write_bytes(app_data),
                    "app_gui.py", "envoi partiel sur GitHub (app_gui.py ancien, texforge/ neuf)")

            pip_calls = []
            real_pip = updater._pip_install
            updater._pip_install = lambda req, py, say: pip_calls.append(req.name)
            req = good / "requirements.txt"
            req.write_text(req.read_text(encoding="utf-8") + "rich>=13\n", encoding="utf-8")
            updater.write_release(good)
            try:
                logs: list[str] = []
                res = updater.apply(updater.check(), log=logs.append)
            finally:
                updater._pip_install = real_pip
            after = _digest(root)
            _check(res["to"] == "99.0.0" and res["added"] == 1 and res["replaced"] == 4
                   and updater.installed_version(root) == "99.0.0" and after == _digest(good)
                   and pip_calls == ["requirements.txt"],
                   f"installée : {res['replaced']} fichiers remplacés (version, eta.py, requirements, liste de la version), "
                   f"{res['added']} ajouté, dépendances installées par pip, copie identique au dépôt", failures)
            again = updater.check()
            _check(again["restart"] and not again["available"] and again["backup"] == __version__,
                   "après installation : redémarrage demandé, plus rien à installer, retour arrière proposé", failures)
            back = updater.rollback()
            _check(back["to"] == __version__ and _digest(root) == before and updater.last_backup() is None,
                   f"retour arrière : installation d'origine à l'identique ({back['restored']} restaurés, "
                   f"{back['removed']} retiré)", failures)
        old = _project_copy(base / "depot_ancien", "0.0.1")
        with FakeGitHub(old) as gh:
            os.environ.update(gh.env())
            info = updater.check()
            _check(not info["available"] and info["newer_local"],
                   "dépôt plus ancien que l'installation (0.0.1) : rien proposé, pastille grise", failures)
        os.environ["TEXFORGE_UPDATE_API"] = os.environ["TEXFORGE_UPDATE_RAW"] = "http://127.0.0.1:9/x"
        info = updater.check(timeout=2)
        _check(bool(info["error"]) and not info["available"], f"GitHub injoignable : {info['error'][:80]}", failures)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeHTTP:
    """Petit serveur local de test : handler(méthode, chemin, corps JSON) -> (code, objet JSON)."""

    def __init__(self, handler) -> None:
        import http.server
        import threading
        outer = self
        self.calls: list[tuple[str, str, object]] = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _go(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                outer.calls.append((method, self.path, body))
                code, obj = handler(method, self.path, body)
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._go("GET")

            def do_POST(self):
                self._go("POST")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _selftest_prompt(failures: list) -> None:
    """Prompt détaillé en anglais : ComfyUI (TextGenerate + Qwen3-4B de Z-Image), repli LM Studio / Ollama."""
    import os

    from . import prompt_enhance as pe

    answer = '<think></think>\nHere is the prompt:\nPrompt: "Worn green armor plates, deep scratches, chipped paint, ' \
             'gold reflective visor, fine dust in the seams"'

    def comfy(with_textgen: bool):
        def handler(method, path, body):
            if path.startswith("/object_info/"):
                cls = path.rsplit("/", 1)[-1]
                if cls == "CLIPLoader":
                    return 200, {cls: {"input": {"required": {"clip_name": [["clip_l.safetensors",
                                                                             "qwen_3_4b.safetensors"]],
                                                              "type": [["stable_diffusion", "lumina2"]]}}}}
                if cls == "TextGenerate" and not with_textgen:
                    return 200, {}
                return 200, {cls: {"input": {}}}
            if path == "/prompt":
                return 200, {"prompt_id": "p1", "number": 1}
            if path == "/history/p1":
                return 200, {"p1": {"status": {"status_str": "success", "completed": True},
                                    "outputs": {"4": {"text": [answer]}}}}
            return 404, {}
        return handler

    srv = _FakeHTTP(comfy(True))
    saved = os.environ.get("TEXFORGE_LLM_URL")
    os.environ["TEXFORGE_LLM_URL"] = "http://127.0.0.1:9/v1"           # pas de LM Studio pendant le test
    try:
        res = pe.enhance("armure verte rayée, visière dorée", srv.url, "faithful", "zimage", seed=7)
        graph = next(b for m, pth, b in srv.calls if pth == "/prompt")["prompt"]
        gen = graph["3"]["inputs"]
        _check(res["prompt"].startswith("Worn green armor") and '"' not in res["prompt"]
               and "Prompt:" not in res["prompt"] and "Qwen3-4B" in res["engine"],
               f"ComfyUI : prompt détaillé nettoyé ({res['words']} mots, {res['engine']})", failures)
        _check(graph["1"]["inputs"] == {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}
               and gen["system_prompt"] == ["2", 0] and gen["sampling_mode"] == "on"
               and gen["sampling_mode.seed"] == 7 and gen["thinking"] is False
               and "English" in graph["2"]["inputs"]["value"] and "70 to 130 words" in graph["2"]["inputs"]["value"]
               and graph["4"]["inputs"]["source"] == ["3", 0],
               "graphe ComfyUI : encodeur Qwen3-4B de Z-Image, TextGenerate, consignes en système, PreviewAny",
               failures)
        short = pe.system_prompt("photoreal", "sdxl")
        _check("30 to 60 words" in short and "photograph" in short and "albedo" not in short
               and "albedo" in pe.system_prompt("faithful", "zimage"),
               "consignes : courtes pour SDXL (CLIP), photo en photoréaliste, éclairage neutre en fidèle", failures)
        srv.close()

        srv = _FakeHTTP(comfy(False))
        llm = _FakeHTTP(lambda m, pth, b: (200, {"data": [{"id": "nomic-embed"}, {"id": "qwen3-8b"}]})
                        if pth == "/v1/models" else (200, {"choices": [{"message": {"content": answer}}]}))
        os.environ["TEXFORGE_LLM_URL"] = f"{llm.url}/v1"
        res = pe.enhance("mur de briques", srv.url, "faithful", "sdxl")
        chat = next(b for m, pth, b in llm.calls if pth == "/v1/chat/completions")
        _check(res["prompt"].startswith("Worn green armor") and "qwen3-8b" in res["engine"]
               and chat["model"] == "qwen3-8b" and chat["messages"][0]["role"] == "system"
               and chat["messages"][1]["content"] == "mur de briques",
               f"ComfyUI sans TextGenerate : repli sur un serveur compatible OpenAI ({res['engine']})", failures)
        llm.close()
        os.environ["TEXFORGE_LLM_URL"] = "http://127.0.0.1:9/v1"
        try:
            pe.enhance("mur", srv.url)
            ok, msg = False, ""
        except pe.EnhanceError as exc:
            ok, msg = True, str(exc)
        _check(ok and "TextGenerate" in msg and "LM Studio" in msg,
               "aucun modèle de langage : message qui dit quoi installer", failures)
    finally:
        srv.close()
        if saved is None:
            os.environ.pop("TEXFORGE_LLM_URL", None)
        else:
            os.environ["TEXFORGE_LLM_URL"] = saved


def _selftest_security(tmp: Path, failures: list) -> None:
    """Attaques simulées : fichier modifié après publication, fichiers en trop dans le dépôt, dépendance
    venue d'ailleurs, adresse de mise à jour détournée, archives piégées."""
    import os
    import zlib

    from . import __version__, ba2, updater
    from .review import render_page

    base = tmp / "securite"
    keys = ("TEXFORGE_UPDATE_REPO", "TEXFORGE_UPDATE_API", "TEXFORGE_UPDATE_RAW", "TEXFORGE_UPDATE_CODELOAD",
            "TEXFORGE_UPDATE_DIR", "TEXFORGE_UPDATE_ROOT")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["TEXFORGE_UPDATE_DIR"] = str(base / "donnees")
    try:
        repo = _project_copy(base / "depot", "99.1.0")
        (repo / "notes_perso.txt").write_text("à ne pas publier", encoding="utf-8")
        (repo / "texforge" / "release_key.txt").write_text("ancienne clé publique\n", encoding="utf-8")
        pub = updater.publish(base / "publication", root=repo)
        out = Path(pub["dest"])
        _check(not (out / "README.md").exists() and not (out / "notes_perso.txt").exists()
               and not (out / "texforge" / "release_key.txt").exists() and (out / updater.RELEASE_FILE).is_file()
               and (out / "app_gui.py").is_file(),
               f"dossier à publier : seulement les fichiers de la version ({pub['files']}), ni notes ni ancienne clé",
               failures)
        root = _project_copy(base / "installation", __version__)
        (root / "texforge" / "release_key.txt").write_text("ancienne clé publique\n", encoding="utf-8")   # 0.5.7
        os.environ["TEXFORGE_UPDATE_ROOT"] = str(root)
        before = _digest(root)

        eta = out / "texforge" / "eta.py"
        orig = eta.read_bytes()
        eta.write_bytes(orig + b"\nimport os\n")
        try:
            with FakeGitHub(out) as gh:
                os.environ.update(gh.env())
                updater.apply(updater.check())
            ok_, why = False, "INSTALLÉE"
        except updater.UpdateError as exc:
            ok_, why = "eta.py" in str(exc), str(exc)
        finally:
            eta.write_bytes(orig)
        _check(ok_ and _digest(root) == before, f"fichier modifié après la publication : refusé ({why[:80]})", failures)

        with FakeGitHub(out) as gh:
            (out / "texforge" / "porte_derobee.py").write_text("raise SystemExit('code pirate')\n", encoding="utf-8")
            (out / "sitecustomize.py").write_text("raise SystemExit('code pirate')\n", encoding="utf-8")
            (out / "texforge" / "release.sig").write_text("00\n", encoding="utf-8")
            os.environ.update(gh.env())
            res = updater.apply(updater.check())
        _check(res["to"] == "99.1.0" and updater.installed_version(root) == "99.1.0"
               and not (root / "texforge" / "porte_derobee.py").exists() and not (root / "sitecustomize.py").exists()
               and not (root / "texforge" / "release.sig").exists(),
               "ancien fichier de clé de la 0.5.7 présent : ignoré, mise à jour installée ; fichiers hors de la "
               "liste ignorés (ni installés ni importés)", failures)
        updater.rollback(root)

        plain_root = _project_copy(base / "installation_b", __version__)
        plain_repo = _project_copy(base / "depot_b", "99.2.0")
        req = plain_repo / "requirements.txt"
        req.write_text(req.read_text(encoding="utf-8") + "--extra-index-url http://pirate.example/simple\n",
                       encoding="utf-8")
        updater.write_release(plain_repo)
        os.environ["TEXFORGE_UPDATE_ROOT"] = str(plain_root)
        before2 = _digest(plain_root)
        try:
            with FakeGitHub(plain_repo) as gh:
                os.environ.update(gh.env())
                updater.apply(updater.check())
            ok_, why = False, "installée"
        except updater.UpdateError as exc:
            ok_, why = "extra-index-url" in str(exc), str(exc)
        _check(ok_ and _digest(plain_root) == before2,
               f"dépendance venue d'un autre index que PyPI : refusée ({why[:70]})", failures)

        os.environ["TEXFORGE_UPDATE_API"] = "http://pirate.example/api"
        _check(updater._base("API", "https://api.github.com") == "https://api.github.com",
               "adresse de mise à jour détournée en HTTP : ignorée (HTTPS ou machine locale seulement)", failures)
        real_max = updater.MAX_EXTRACTED
        updater.MAX_EXTRACTED = 1000
        try:
            os.environ["TEXFORGE_UPDATE_ROOT"] = str(root)
            with FakeGitHub(out) as gh:
                os.environ.update(gh.env())
                updater.apply(updater.check())
            ok_ = False
        except updater.UpdateError as exc:
            ok_ = "anormale" in str(exc) or "abnormal" in str(exc)
        finally:
            updater.MAX_EXTRACTED = real_max
        _check(ok_, "archive démesurée une fois décompressée (bombe) : refusée", failures)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # archives BA2 piégées (mods téléchargés) : refus propre, sans allouer des gigaoctets
    from . import dds as _dds
    img = Image.new("RGBA", (64, 64), (90, 120, 30, 255))
    arc = base / "piege.ba2"
    ba2.write_texture_archive(arc, [("Textures/Piege_d.DDS", _dds.encode_dds(img, "DXT1", 1))], version=7)
    raw = bytearray(arc.read_bytes())
    struct.pack_into("<I", raw, 24 + 24 + 12, 0xFFFFFFF0)              # taille du 1er morceau : 4 Go annoncés
    arc.write_bytes(bytes(raw))
    try:
        with ba2.Ba2Archive(arc) as a:
            a.read_dds(a.textures[0])
        ok_ = False
    except ba2.Ba2Error:
        ok_ = True
    bomb = base / "bombe.bin"
    bomb.write_bytes(zlib.compress(b"\0" * 20_000_000))
    a = ba2.Ba2Archive.__new__(ba2.Ba2Archive)
    a.path, a.f, a.compression = bomb, open(bomb, "rb"), "zlib"
    try:
        a._unpack(ba2.Chunk(0, bomb.stat().st_size, 1000, 0, 0))
        ok2 = False
    except ba2.Ba2Error:
        ok2 = True
    finally:
        a.f.close()
    _check(ok_ and ok2 and ba2.safe_rel("Textures/../../Démarrage/x.exe") is None and ba2.safe_rel("C:/x") is None,
           "BA2 piégée : taille démesurée et bombe zlib refusées, chemins sortants bloqués", failures)
    page = render_page([{"id": "a", "name": "</script><script>alert(1)</script>", "status": "done", "mode": "ai",
                         "choice": "auto", "fmt": "DXT1", "w": 1, "h": 1, "metrics": {}, "error": "<img src=x>",
                         "imgs": {}}])
    _check("</script><script>alert(1)" not in page, "galerie HTML : noms de textures piégés neutralisés", failures)


def main() -> int:
    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="texforge_selftest_"))
    import os
    saved_timings = os.environ.get("TEXFORGE_TIMINGS")
    os.environ["TEXFORGE_TIMINGS"] = str(tmp / "timings.json")    # durées des lancements de test : à part
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
        _check(an["Compressed_NORMAL04"].get("normal_packing") == "xyz", "relief : normal map XYZ", failures)
        _check(not any(a.get("normal_packing") for n, a in an.items() if n != "Compressed_NORMAL04"),
               "aucune autre texture prise pour une normal map (vitre bleutée comprise)", failures)

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
        refused = tr("le QC aurait refusé : {reasons}", reasons="")
        _check(m2["decision"] == "measured" and any(w.startswith(refused) for w in m2["warnings"]),
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

        print(f"8. Langues (active : {get_lang()})")
        problems = check_catalogs(Path(__file__).resolve().parent.parent)
        for pb in problems[:10]:
            print(f"     {pb}")
        _check(not problems, f"catalogues complets ({', '.join(LANGUAGES)}) : {len(problems)} problème(s)", failures)
        lines = [_fmt(json.loads(x)) for x in (Path(cfg["_work"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        _check(all(x is None or isinstance(x, str) for x in lines), f"{len(lines)} événements mis en forme", failures)
        _check(tr("Exporter mes choix (JSON)") in html, "galerie dans la langue active", failures)

        print("9. Détection du dossier d'entrée")
        from .detect import detect, inject_for, resolve_input
        lib = tmp / "MesJeux"
        dump = lib / "Vieux Jeu" / "Bin" / "SK_Res" / "dump" / "textures"
        shutil.copytree(inp, dump)
        (lib / "Autre Jeu" / "SK_Res" / "dump" / "textures").mkdir(parents=True)        # dump vide : ignoré
        found = detect(sources=[lambda: [(p, p.name, "test") for p in lib.iterdir()]], history=[])
        _check(len(found) == 1 and found[0].game == "Vieux Jeu" and found[0].count == 11,
               f"dump Special K trouvé ({[(c.game, c.count) for c in found]})", failures)
        _check(found and Path(found[0].inject) == dump.parent.parent / "inject" / "textures"
               and inject_for(dump) == found[0].inject, "dossier d'injection associé", failures)
        r = resolve_input(lib / "Vieux Jeu")
        _check(r is not None and Path(r.path) == dump, "dossier du jeu -> ses textures extraites", failures)
        _check(resolve_input(dump) is None and resolve_input(inp) is None, "dossier de textures laissé tel quel", failures)
        _selftest_locate(lib, dump, failures)

        print("10. Mode sans ComfyUI (Real-ESRGAN simulé)")
        _selftest_esrgan(tmp, inp, failures)

        print("11. Normal maps rangées autrement (Unity DXT5nm, BC5, hauteur en alpha)")
        _selftest_normals(cfg, failures)

        print("12. Archives BA2 (Fallout 4, Starfield)")
        _selftest_ba2(tmp, failures)

        print("13. Jeux Unreal (reconnaissance, sans déchiffrement)")
        _selftest_unreal(tmp, failures)

        print("14. Robustesse (dossier de jeu, fichier disparu, ComfyUI sans nœuds personnalisés)")
        _selftest_robustness(tmp, failures)

        print("15. Tout par l'IA, génération par morceaux, panneau avant / après")
        _selftest_tiles(cfg, failures)

        print("16. Estimation du temps du lancement")
        _selftest_eta(tmp, cfg, failures)

        print("17. Réglages changés entre deux lancements")
        _selftest_stale(tmp, inp, failures)

        print("18. Mise à jour depuis GitHub (faux serveur local)")
        _selftest_update(tmp, failures)

        print("19. Prompt détaillé en anglais (aperçu)")
        _selftest_prompt(failures)

        print("20. Sécurité : attaques simulées")
        _selftest_security(tmp, failures)
    except Exception:
        traceback.print_exc()
        failures.append("exception")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if saved_timings is None:
            os.environ.pop("TEXFORGE_TIMINGS", None)
        else:
            os.environ["TEXFORGE_TIMINGS"] = saved_timings
    print()
    if failures:
        print(f"SELFTEST : {len(failures)} échec(s)")
        return 1
    print("SELFTEST OK")
    return 0
