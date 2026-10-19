// example.mjs — exemple d'orchestrateur Node.js branché sur TEXFORGE + ComfyUI local.
//
//   node bridge/example.mjs <dossier TEXFORGE> <dossier projet> [dossier d'export DDS]
//
// Déroulé : analyse et routage -> traitement (UI, local, ComfyUI) avec tableau de bord
// -> si des textures ont échoué pour une cause passagère (OOM, délai, serveur), une
// seconde passe automatique -> contrôle qualité + DDS avec mipmaps -> résumé final.

import { TexforgeBridge } from './texforge-bridge.mjs';

const [texforgeDir, projectDir] = process.argv.slice(2);
if (!texforgeDir || !projectDir) {
  console.error('usage : node example.mjs <dossier TEXFORGE> <dossier projet>');
  process.exit(64);
}

const tf = new TexforgeBridge({ texforgeDir, projectDir });
const t0 = Date.now();
const stamp = () => `${((Date.now() - t0) / 1000).toFixed(1).padStart(6)} s`;

// Tableau de bord : une ligne par événement utile
tf.on('run_start', (e) => console.log(`${stamp()}  ▶ ${e.total} texture(s) : ${e.ui} UI, ${e.local} local, ${e.ai} ComfyUI`));
tf.on('ui_done', (e) => console.log(`${stamp()}  [UI] ${e.name} traité localement (${e.engine}, ×${e.k}) (${e.index}/${e.total})`));
tf.on('local_done', (e) => console.log(`${stamp()}  [LOCAL] ${e.name} agrandi (×${e.k})`));
tf.on('comfy_queued', (e) => console.log(`${stamp()}  [COMFYUI] ${e.name} en file (${e.index}/${e.total}) — ${e.material}`));
tf.on('comfy_done', (e) => console.log(`${stamp()}  [COMFYUI] ${e.name} validé (structure ${e.structure}, ${e.seconds} s)`));
tf.on('fallback', (e) => console.log(`${stamp()}  [REPLI] ${e.name} → ${e.engine} : ${e.reason}`));
tf.on('retry', (e) => console.log(`${stamp()}  [RETRY] ${e.name} (${e.kind}), essai ${e.attempt}`));
tf.on('failed', (e) => console.log(`${stamp()}  [ERREUR] ${e.name} : ${e.message}`));
tf.on('gpu_free', (e) => console.log(`${stamp()}  [GPU] VRAM libérée : ${e.reason}`));
tf.on('server_down', (e) => console.log(`${stamp()}  [COMFYUI] serveur injoignable : ${e.message}`));
tf.on('qc_done', (e) => {
  const tag = { validated: '✔', corrected: '✎', rejected: '✘', local: '·', measured: '=', error: '!' }[e.decision] ?? '?';
  const what = e.decision === 'corrected' ? ` (${e.fixes.join(', ')})`
    : ['rejected', 'error'].includes(e.decision) ? ` (${e.reasons.slice(0, 1).join('')})` : '';
  console.log(`${stamp()}  [QC] ${tag} ${e.name} ${e.label}${what}${e.format ? ` → ${e.format}, ${e.levels} mips` : ''} (${e.index}/${e.total})`);
});
tf.on('warning', (w) => console.warn(`⚠ ${w}`));
// tf.on('log', (line) => console.error(line));   // tableau de bord Python brut, si besoin

// Arrêt propre sur Ctrl+C : TEXFORGE finit les textures en vol puis s'arrête
let job;
process.on('SIGINT', async () => {
  console.log('\n■ arrêt demandé…');
  await job?.stop();
  process.exit(130);
});

try {
  console.log(await tf.scan());
  job = tf.run();
  let summary = await job.done;

  // Seconde passe automatique pour les échecs passagers
  const transient = (await tf.failures()).filter((f) => ['oom', 'timeout', 'server', 'http'].includes(f.kind)
    && f.action !== 'retry');
  if (transient.length) {
    console.log(`${stamp()}  ↻ ${transient.length} échec(s) passager(s) : seconde passe`);
    for (const kind of new Set(transient.map((f) => f.kind))) await tf.retry({ includeFallback: true, kind });
    job = tf.run({ stages: ['ai'] });
    summary = await job.done;
  }

  const end = summary.runEnd ?? {};
  console.log(`\n${stamp()}  ✔ génération terminée (code ${summary.code}) — ${end.summary ?? ''}`);

  // Contrôle qualité + DDS avec mipmaps, prêts pour Special K
  const outDir = process.argv[4] ?? `${projectDir}/export_dds`;
  job = tf.export({ target: 'specialk', out: outDir });
  const exp = await job.done;
  const q = exp.runEnd ?? {};
  console.log(`${stamp()}  ✔ export (code ${exp.code}) — ${q.summary ?? ''}`);

  const report = await tf.report(outDir);
  const toCheck = report.textures.filter((t) => ['rejected', 'error'].includes(t.decision) || t.warnings.length);
  if (summary.failures.length || toCheck.length) {
    console.log('  Textures à vérifier :');
    for (const f of summary.failures) console.log(`   - ${f.name} [${f.event}] ${f.reason ?? f.message ?? ''}`);
    for (const t of toCheck) {
      console.log(`   - ${t.name} [QC ${t.label}] ${[...t.reasons, ...t.warnings].slice(0, 2).join(' ; ')}`);
    }
  }
  process.exit(Math.max(summary.code, exp.code));
} catch (err) {
  console.error(`✗ ${err.message}`);
  process.exit(2);
}
