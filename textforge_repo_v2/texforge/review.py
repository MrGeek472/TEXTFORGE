"""Galerie de revue avant/après (page HTML locale, aucun serveur nécessaire).
Les choix sont exportés en JSON puis appliqués avec : texforge apply-review."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .imaging import load_texture, thumbnail
from .pipeline import open_state, safe_name


def build_review(cfg: dict) -> Path:
    state = open_state(cfg)
    rows = [r for r in state.all() if r["status"] in ("done", "fallback", "error")]
    out_dir = Path(cfg["_work"]) / "review"
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for r in rows:
        n = safe_name(r["id"])
        entry = {"id": r["id"], "name": r["name"], "status": r["status"], "mode": r["mode"],
                 "choice": r.get("choice") or "auto", "fmt": r["fmt"], "w": r["width"], "h": r["height"],
                 "metrics": r.get("metrics") or {}, "error": r.get("error") or "", "imgs": {}}
        try:
            src = load_texture(Path(r["src_path"])).image
            o = img_dir / f"{n}_o.png"
            thumbnail(src, 512).save(o)
            entry["imgs"]["original"] = f"img/{o.name}"
        except Exception:
            pass
        for key, field in (("local", "local_path"), ("ai", "ai_path"), ("final", "result_path")):
            pth = r.get(field)
            if pth and Path(pth).exists():
                t = img_dir / f"{n}_{key}.png"
                thumbnail(Image.open(pth), 512).save(t)
                entry["imgs"][key] = f"img/{t.name}"
        cards.append(entry)
    state.close()
    page = out_dir / "index.html"
    page.write_text(_HTML.replace("__DATA__", json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")),
                    encoding="utf-8")
    return page


_HTML = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TEXFORGE — Revue</title>
<style>
:root{--bg:#121316;--panel:#1b1d22;--line:#2b2e36;--ink:#e8e9ec;--mute:#9298a5;--ok:#5fc48a;--warn:#e0a84f;--bad:#e06a5f;--acc:#7aa2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:rgba(18,19,22,.95);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
h1{font-size:16px;margin:0 12px 0 0;letter-spacing:.04em}
.chip{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:5px 12px;cursor:pointer;font-size:13px}
.chip.on{border-color:var(--acc);color:var(--acc)}
.spacer{flex:1}
button.main{background:var(--acc);color:#0b0d12;border:0;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer}
#count{color:var(--mute);font-size:13px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.cmp{position:relative;aspect-ratio:1/1;background:repeating-conic-gradient(#2a2c33 0% 25%,#202228 0% 50%) 50%/16px 16px;overflow:hidden;cursor:ew-resize}
.cmp img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.cmp img.orig{image-rendering:pixelated}
.cmp .top{clip-path:inset(0 0 0 var(--cut,50%))}
.cmp .bar{position:absolute;top:0;bottom:0;left:var(--cut,50%);width:2px;background:#fff;box-shadow:0 0 6px #000}
.cmp .lab{position:absolute;bottom:6px;font-size:11px;background:rgba(0,0,0,.6);padding:2px 6px;border-radius:4px}
.cmp .l{left:6px}.cmp .r{right:6px}
.body{padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.name{font-family:ui-monospace,Consolas,monospace;font-size:12px;word-break:break-all}
.meta{display:flex;flex-wrap:wrap;gap:6px;font-size:12px;color:var(--mute)}
.badge{border-radius:6px;padding:1px 7px;font-size:12px;font-weight:600}
.b-done{background:rgba(95,196,138,.15);color:var(--ok)}.b-fallback{background:rgba(224,168,79,.15);color:var(--warn)}.b-error{background:rgba(224,106,95,.15);color:var(--bad)}
.err{color:var(--warn);font-size:12px}
.views{display:flex;gap:6px}
.views button{flex:1;background:transparent;border:1px solid var(--line);color:var(--mute);border-radius:6px;padding:4px;cursor:pointer;font-size:12px}
.views button.on{color:var(--ink);border-color:var(--acc)}
.views button:disabled{opacity:.35;cursor:default}
select{width:100%;background:#111317;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px}
.changed{outline:2px solid var(--acc)}
@media (max-width:600px){main{grid-template-columns:1fr;padding:12px}}
</style></head><body>
<header>
  <h1>TEXFORGE · revue</h1>
  <span class="chip on" data-f="all">Toutes</span>
  <span class="chip" data-f="done">Validées</span>
  <span class="chip" data-f="fallback">Replis locaux</span>
  <span class="chip" data-f="error">Erreurs</span>
  <span class="spacer"></span><span id="count"></span>
  <button class="main" id="save">Exporter mes choix (JSON)</button>
</header>
<main id="grid"></main>
<script>
const DATA = __DATA__;
const choices = {};
let filter = 'all';
const LABEL = {done:'Validée',fallback:'Repli local',error:'Erreur'};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function card(d){
  const m=d.metrics||{};
  const el=document.createElement('div');el.className='card';el.dataset.status=d.status;
  const right=d.imgs.final||d.imgs.ai||d.imgs.local||'';
  el.innerHTML=`
   <div class="cmp" style="--cut:50%">
     ${d.imgs.original?`<img class="orig" src="${d.imgs.original}" alt="">`:''}
     ${right?`<img class="top" src="${right}" alt="">`:''}
     <div class="bar"></div><span class="lab l">Original</span><span class="lab r rl">Final</span>
   </div>
   <div class="body">
     <div class="name">${esc(d.name)}</div>
     <div class="meta"><span class="badge b-${d.status}">${d.mode==='upscale'&&d.status==='done'?'Agrandie localement':(LABEL[d.status]||d.status)}</span>
       <span>${esc(d.fmt)} · ${d.w}×${d.h}</span>
       ${m.structure!=null?`<span>structure ${m.structure}</span>`:''}
       ${m.shift!=null?`<span>décalage ${m.shift}</span>`:''}
       ${m.seam!=null?`<span>raccord ${m.seam}</span>`:''}
       ${m.attempt?`<span>essai ${m.attempt}</span>`:''}</div>
     ${d.error?`<div class="err">${esc(d.error)}</div>`:''}
     <div class="views">
       <button data-v="final" ${d.imgs.final?'':'disabled'}>Final</button>
       <button data-v="ai" ${d.imgs.ai?'':'disabled'}>IA</button>
       <button data-v="local" ${d.imgs.local?'':'disabled'}>Local</button>
     </div>
     <select>
       <option value="">Garder le choix actuel (${d.choice==='ai'?'IA':d.choice==='local'?'local':'auto'})</option>
       <option value="ai" ${d.imgs.ai?'':'disabled'}>Utiliser la version IA</option>
       <option value="local" ${d.imgs.local?'':'disabled'}>Utiliser la version locale</option>
       <option value="redo">Regénérer (nouvelle requête payante)</option>
     </select>
   </div>`;
  const cmp=el.querySelector('.cmp');
  const move=e=>{const r=cmp.getBoundingClientRect();const x=((e.touches?e.touches[0].clientX:e.clientX)-r.left)/r.width;cmp.style.setProperty('--cut',(Math.min(1,Math.max(0,x))*100)+'%')};
  cmp.addEventListener('mousemove',move);cmp.addEventListener('touchmove',move,{passive:true});
  el.querySelectorAll('.views button').forEach(b=>{
    if(b.dataset.v==='final')b.classList.add('on');
    b.onclick=()=>{const top=el.querySelector('.top');if(!top||b.disabled)return;top.src=d.imgs[b.dataset.v];
      el.querySelectorAll('.views button').forEach(x=>x.classList.toggle('on',x===b));
      el.querySelector('.rl').textContent=b.textContent;};
  });
  el.querySelector('select').onchange=e=>{const v=e.target.value;if(v)choices[d.id]=v;else delete choices[d.id];
    el.classList.toggle('changed',!!v);updateCount();};
  return el;
}
const grid=document.getElementById('grid');
DATA.forEach(d=>grid.appendChild(card(d)));
function apply(){grid.querySelectorAll('.card').forEach(c=>{c.style.display=(filter==='all'||c.dataset.status===filter)?'':'none'});updateCount()}
function updateCount(){const vis=[...grid.children].filter(c=>c.style.display!=='none').length;
  document.getElementById('count').textContent=`${vis} texture(s) · ${Object.keys(choices).length} choix modifié(s)`}
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{filter=c.dataset.f;document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===c));apply()});
document.getElementById('save').onclick=()=>{
  const blob=new Blob([JSON.stringify({choices},null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='texforge_choix.json';a.click();
};
apply();
</script></body></html>
"""
