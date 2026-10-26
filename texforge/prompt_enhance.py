"""Prompt détaillé en anglais à partir de la zone « Prompt à donner » (aperçu avant de l'utiliser).

Moteur, dans cet ordre :
1. ComfyUI lui-même : nœud TextGenerate (ComfyUI récent) avec un encodeur de texte qui est un vrai modèle
   de langage, en priorité Qwen3-4B, celui de Z-Image Turbo (rien à télécharger s'il est déjà là) ;
2. un serveur local compatible OpenAI : LM Studio (port 1234) ou Ollama (port 11434).
Tout reste sur le PC. Le texte produit décrit la texture finie (matières, usure, micro-détail, couleurs),
en anglais, à la longueur que la famille de modèles exploite (CLIP de SDXL : courte ; Qwen de Z-Image : longue).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Callable

from .i18n import tr

# encodeurs de texte de ComfyUI capables de générer du texte : (motif du fichier, type de CLIPLoader, nom affiché)
ENCODERS = (("qwen_3_4b", "lumina2", "Qwen3-4B"), ("qwen3_4b", "lumina2", "Qwen3-4B"),
            ("qwen3.5", "stable_diffusion", "Qwen3.5"))
LOCAL_LLM = (("http://127.0.0.1:1234/v1", "LM Studio"), ("http://127.0.0.1:11434/v1", "Ollama"))
MAX_TOKENS = 360

BASE_RULES = """You write prompts for an image diffusion model that remasters textures of old video games.
Rewrite the user's request as ONE detailed English prompt that describes the finished texture.
Rules:
- Write in English, whatever the language of the request. Keep every element the user asked for, never contradict it, never add objects, characters, text, logos, watermarks or borders they did not ask for.
- Be concrete about materials: substance, surface finish, micro-detail (scratches, pores, grain, fibres, chips, cracks, rust, dust, stitching), wear and age, colour nuances and variations.
- {style}
- {length}
- Output only the prompt itself: no title, no quotes, no list, no explanation, no Markdown."""

STYLE_RULES = {
    "faithful": "It is a flat texture map seen straight on (orthographic, top-down), applied to a 3D model by the "
                "game engine: no scene, no perspective, no background, no camera terms. Lighting stays neutral and "
                "even (diffuse albedo, no baked shadows, no specular highlights, no dramatic light) because the "
                "engine lights the texture. Keep the original colours and layout; add detail, not new content.",
    "photoreal": "The result must look like a high-end photograph of the real thing: physically accurate "
                 "materials, natural light, depth, sharp focus, high dynamic range, realistic imperfections. Keep "
                 "the subject, composition and colours of the original texture.",
}
LENGTH_RULES = {
    "short": "Write 30 to 60 words as comma-separated descriptive phrases, most important first (the model's text "
             "encoder only reads about 75 words).",
    "long": "Write one paragraph of 70 to 130 words in natural descriptive language.",
}


class EnhanceError(Exception):
    """Aucun moteur n'a pu produire de prompt (raisons de chacun dans le message)."""


class _Unavailable(Exception):
    """Ce moteur-là ne convient pas ou a échoué : on passe au suivant."""


def system_prompt(style: str = "faithful", family: str = "zimage") -> str:
    length = "short" if family in ("sdxl", "sd15") else "long"
    return BASE_RULES.format(style=STYLE_RULES.get(style, STYLE_RULES["faithful"]), length=LENGTH_RULES[length])


def clean(text: str) -> str:
    """Retire raisonnement, étiquettes (« Prompt: »), guillemets et mise en forme ; une seule ligne."""
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    text = text.split("</think>")[-1]
    lines = []
    for ln in text.splitlines():
        ln = ln.strip().strip("`").strip()
        if not ln or re.fullmatch(r"(?i)(here is|here's|sure|okay|ok)\b.*:\s*", ln):
            continue
        ln = re.sub(r"(?i)^(\*\*)?(enhanced |detailed |final |improved )?prompt(\*\*)?\s*:\s*", "", ln)
        ln = re.sub(r"^[-*•]\s+|^#+\s*", "", ln)
        lines.append(ln)
    out = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'“”«»":
        out = out[1:-1].strip()
    return out.strip("\"“”«» ")


def _http(url: str, data: dict | None = None, timeout: float = 10.0):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def _options(spec: dict, cls: str, field: str) -> list[str]:
    inputs = (spec.get(cls) or {}).get("input") or {}
    for group in ("required", "optional"):
        val = (inputs.get(group) or {}).get(field)
        if not val:
            continue
        first = val[0]
        if isinstance(first, list):
            return [str(x) for x in first]
        if first == "COMBO" and len(val) > 1 and isinstance(val[1], dict):
            return [str(x) for x in val[1].get("options", [])]
    return []


def pick_encoder(names: list[str]) -> tuple[str, str, str] | None:
    for pattern, ctype, label in ENCODERS:
        for n in names:
            if pattern in n.lower():
                return n, ctype, label
    return None


def comfy_graph(encoder: str, ctype: str, text: str, system: str, seed: int, with_primitive: bool = True) -> dict:
    gen = {"clip": ["1", 0], "prompt": text, "max_length": MAX_TOKENS, "sampling_mode": "on",
           "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95,
           "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed,
           "thinking": False, "use_default_template": True}
    graph = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": encoder, "type": ctype}},
             "3": {"class_type": "TextGenerate", "inputs": gen},
             "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}}
    if with_primitive:
        graph["2"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": system}}
        gen["system_prompt"] = ["2", 0]
    else:                                             # ComfyUI sans ce nœud : consignes dans la demande
        gen["prompt"] = f"{system}\n\nUser request: {text}"
    return graph


def _via_comfyui(server: str, text: str, system: str, seed: int, timeout: float,
                 on_status: Callable[[str], None]) -> dict:
    base = server.rstrip("/")
    try:
        spec = {}
        for cls in ("TextGenerate", "PreviewAny", "PrimitiveStringMultiline", "CLIPLoader"):
            spec.update(_http(f"{base}/object_info/{cls}", timeout=5) or {})
    except (OSError, ValueError) as exc:
        raise _Unavailable(tr("ComfyUI injoignable ({err})", err=getattr(exc, "reason", exc))) from exc
    if "TextGenerate" not in spec or "PreviewAny" not in spec:
        raise _Unavailable(tr("ComfyUI sans le nœud TextGenerate (version trop ancienne : mets ComfyUI à jour)"))
    enc = pick_encoder(_options(spec, "CLIPLoader", "clip_name"))
    if enc is None:
        raise _Unavailable(tr("aucun encodeur de texte Qwen dans ComfyUI (models\\text_encoders) ; celui de Z-Image "
                              "Turbo, qwen_3_4b.safetensors, convient"))
    name, ctype, label = enc
    graph = comfy_graph(name, ctype, text, system, seed, "PrimitiveStringMultiline" in spec)
    on_status(tr("{model} dans ComfyUI : génération du prompt (le premier appel charge le modèle)…", model=label))
    try:
        res = _http(f"{base}/prompt", {"prompt": graph, "client_id": f"texforge-{uuid.uuid4().hex[:8]}"}, timeout=15)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode("utf-8", "replace")
        raise _Unavailable(tr("ComfyUI a refusé la demande : {err}", err=detail or exc)) from exc
    except (OSError, ValueError) as exc:
        raise _Unavailable(tr("ComfyUI injoignable ({err})", err=getattr(exc, "reason", exc))) from exc
    pid = (res or {}).get("prompt_id")
    if not pid:
        raise _Unavailable(tr("ComfyUI a refusé la demande : {err}", err=res))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            hist = (_http(f"{base}/history/{pid}", timeout=10) or {}).get(pid)
        except (OSError, ValueError):
            hist = None
        if hist:
            status = hist.get("status") or {}
            if status.get("status_str") == "error":
                msg = next((m[1].get("exception_message", "") for m in status.get("messages", [])
                            if isinstance(m, list) and len(m) > 1 and m[0] == "execution_error"), "")
                raise _Unavailable(tr("échec dans ComfyUI : {err}", err=str(msg).strip()[:300] or "?"))
            out = ((hist.get("outputs") or {}).get("4") or {}).get("text") or []
            if out or status.get("completed"):
                return {"text": out[0] if out else "", "engine": tr("{model} (ComfyUI)", model=label)}
        time.sleep(0.5)
    raise _Unavailable(tr("pas de réponse de ComfyUI en {s} s", s=int(timeout)))


def _llm_servers() -> list[tuple[str, str]]:
    env = os.environ.get("TEXFORGE_LLM_URL", "").strip()
    if env:
        return [(u.strip().rstrip("/"), u.strip()) for u in env.split(",") if u.strip()]
    return list(LOCAL_LLM)


def _via_openai(base: str, label: str, text: str, system: str, seed: int, timeout: float,
                on_status: Callable[[str], None]) -> dict:
    try:
        models = [str(m.get("id")) for m in (_http(f"{base}/models", timeout=2) or {}).get("data", [])
                  if m.get("id") and "embed" not in str(m.get("id")).lower()]
    except (OSError, ValueError, AttributeError) as exc:
        raise _Unavailable(tr("{name} non lancé", name=label)) from exc
    model = os.environ.get("TEXFORGE_LLM_MODEL") or (models[0] if models else "")
    if not model:
        raise _Unavailable(tr("{name} sans modèle", name=label))
    on_status(tr("{name} ({model}) : génération du prompt…", name=label, model=model))
    try:
        res = _http(f"{base}/chat/completions",
                    {"model": model, "temperature": 0.7, "max_tokens": MAX_TOKENS, "seed": seed, "stream": False,
                     "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]},
                    timeout=timeout)
        content = res["choices"][0]["message"]["content"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise _Unavailable(tr("{name} : échec ({err})", name=label, err=getattr(exc, "reason", exc))) from exc
    return {"text": str(content or ""), "engine": f"{label} — {model}"}


def enhance(text: str, server: str, style: str = "faithful", family: str = "zimage", seed: int | None = None,
            timeout: float = 240.0, on_status: Callable[[str], None] | None = None) -> dict:
    """Prompt détaillé en anglais : {"prompt", "engine", "seconds", "words"}. Lève EnhanceError si aucun moteur."""
    text = (text or "").strip()
    if not text:
        raise EnhanceError(tr("écris d'abord quelques mots dans « Prompt à donner »"))
    say = on_status or (lambda _m: None)
    seed = random.randrange(1, 2 ** 31) if seed is None else int(seed)
    system = system_prompt(style, family)
    t0 = time.monotonic()
    why: list[str] = []
    attempts = [lambda: _via_comfyui(server, text, system, seed, timeout, say)] if server else []
    attempts += [lambda b=b, n=n: _via_openai(b, n, text, system, seed, timeout, say) for b, n in _llm_servers()]
    for attempt in attempts:
        try:
            res = attempt()
        except _Unavailable as exc:
            why.append(str(exc))
            continue
        prompt = clean(res["text"])
        if not prompt:
            why.append(tr("{engine} : réponse vide", engine=res["engine"]))
            continue
        return {"prompt": prompt, "engine": res["engine"], "seconds": round(time.monotonic() - t0, 1),
                "words": len(prompt.split())}
    raise EnhanceError(tr("aucun modèle de langage disponible pour détailler le prompt :\n- {why}\n\nIl faut soit "
                          "ComfyUI à jour avec l'encodeur Qwen3-4B de Z-Image Turbo (qwen_3_4b.safetensors), soit "
                          "LM Studio ou Ollama lancés avec un modèle.", why="\n- ".join(why)))
