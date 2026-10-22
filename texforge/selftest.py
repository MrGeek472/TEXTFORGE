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
