// texforge-bridge.mjs — pilote TEXFORGE (Python) depuis un orchestrateur Node.js.
//
// Principe : Node lance `python -m texforge run --json-events`. La sortie standard ne
// contient qu'une ligne JSON par événement ; le tableau de bord lisible part sur la
// sortie d'erreur. Aucun serveur supplémentaire, aucun port à ouvrir.
//
//   import { TexforgeBridge } from './texforge-bridge.mjs';
//   const tf = new TexforgeBridge({ texforgeDir: 'C:/Outils/TEXFORGE', projectDir: 'C:/Remaster/TombRaider' });
//   tf.on('comfy_done', (e) => console.log(e.name, e.structure));
//   const job = tf.run({ scan: true });
//   const summary = await job.done;
//   tf.on('qc_done', (e) => console.log(e.name, e.decision, e.format));
//   await tf.export({ out: 'C:/Jeux/MonJeu/SK_Res/inject/textures' }).done;   // QC + DDS + qc_report.json
//
// Node.js 18 ou plus récent. Aucune dépendance.

import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import path from 'node:path';
import readline from 'node:readline';

const DEFAULT_PYTHON = process.platform === 'win32' ? 'python' : 'python3';

export class TexforgeBridge extends EventEmitter {
  /**
   * @param {object} opts
   * @param {string} opts.texforgeDir  dossier qui contient le paquet `texforge/`
   * @param {string} opts.projectDir   dossier du projet (celui de texforge.toml)
   * @param {string} [opts.config]     nom du fichier de configuration (texforge.toml)
   * @param {string} [opts.python]     exécutable Python (python, python3, py)
   */
  constructor({ texforgeDir, projectDir, config = 'texforge.toml', python = DEFAULT_PYTHON } = {}) {
    super();
    if (!texforgeDir || !projectDir) throw new Error('texforgeDir et projectDir sont requis');
    this.texforgeDir = path.resolve(texforgeDir);
    this.projectDir = path.resolve(projectDir);
    this.config = path.resolve(this.projectDir, config);
    this.python = python;
    this.child = null;
  }

  // ------------------------------------------------------------------ bas niveau
  #spawn(args) {
    const env = {
      ...process.env,
      PYTHONPATH: [this.texforgeDir, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',
    };
    return spawn(this.python, ['-m', 'texforge', '--config', this.config, ...args], {
      cwd: this.projectDir, env, windowsHide: true,
    });
  }

  /** Commande courte : renvoie { code, stdout, stderr }. */
  exec(args) {
    return new Promise((resolve, reject) => {
      const child = this.#spawn(args);
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (d) => { stdout += d; });
      child.stderr.on('data', (d) => { stderr += d; });
      child.on('error', reject);
      child.on('close', (code) => resolve({ code, stdout, stderr }));
    });
  }

  // ------------------------------------------------------------------ commandes
  /** Analyse et routage des textures du dossier d'entrée. */
  async scan() {
    const r = await this.exec(['scan']);
    if (r.code !== 0) throw new Error(`scan en échec (${r.code}) : ${r.stderr || r.stdout}`);
    return r.stdout;
  }

  /** Échecs consignés (table failures), filtrables par type. */
  async failures({ kind } = {}) {
    const r = await this.exec(['failures', '--json', ...(kind ? ['--kind', kind] : [])]);
    if (r.code !== 0) throw new Error(`failures en échec (${r.code}) : ${r.stderr}`);
    return JSON.parse(r.stdout || '[]');
  }

  /** Remet en file les textures en échec (et les replis si includeFallback). */
  async retry({ kind, route, includeFallback = false } = {}) {
    const args = ['retry'];
    if (kind) args.push('--kind', kind);
    if (route) args.push('--route', route);
    if (includeFallback) args.push('--include-fallback');
    const r = await this.exec(args);
    if (r.code !== 0) throw new Error(`retry en échec (${r.code}) : ${r.stderr}`);
    return r.stdout.trim();
  }

  /**
   * Lance le traitement. Émet 'event' pour chaque événement, puis l'événement sous son
   * propre nom ('ui_done', 'comfy_queued', 'comfy_done', 'fallback', 'retry', 'failed',
   * 'gpu_free', 'server_down', 'run_end'…), et 'log' pour chaque ligne du tableau de bord.
   * @returns {{ done: Promise<object>, stop: (timeoutMs?: number) => Promise<void> }}
   */
  run({ scan = false, stages, only, limit, provider } = {}) {
    const args = ['run', '--json-events'];
    for (const s of stages ?? []) args.push('--stage', s);
    if (only) args.push('--only', only);
    if (limit) args.push('--limit', String(limit));
    if (provider) args.push('--provider', provider);
    return this.#stream(args, scan);
  }

  /**
   * Contrôle qualité seul (structure, couleurs, raccord, alpha), correctifs compris.
   * Événements : 'qc_start', 'qc_done' (une texture : decision, reasons, fixes, warnings), 'qc_end'.
   */
  qc({ only, fix = true } = {}) {
    const args = ['qc', '--json-events'];
    if (only) args.push('--only', only);
    if (!fix) args.push('--no-fix');
    return this.#stream(args);
  }

  /**
   * Contrôle qualité puis écriture des DDS (mipmaps complètes) dans `out`.
   * @param {object} opts
   * @param {'specialk'|'png'|'remix'} [opts.target]
   * @param {string} opts.out                    dossier de sortie (ex. SK_Res/inject/textures)
   * @param {'auto'|'texconv'|'builtin'} [opts.encoder]
   * @param {string} [opts.ddsFormat]            match | auto | dxt1 | dxt5 | bgra8 | bc7
   * @param {'d3d9'|'d3d10'|'d3d11'} [opts.targetApi]
   * @param {boolean} [opts.qc]                  false : exporter sans contrôle
   * @param {boolean} [opts.fix]                 false : un échec du QC garde la version locale
   */
  export({ target = 'specialk', out, encoder, ddsFormat, targetApi, qc = true, fix = true, only } = {}) {
    if (!out) throw new Error('export : dossier de sortie requis');
    const args = ['export', '--json-events', '--target', target, '--out', path.resolve(out)];
    if (encoder) args.push('--encoder', encoder);
    if (ddsFormat) args.push('--dds-format', ddsFormat);
    if (targetApi) args.push('--target-api', targetApi);
    if (!qc) args.push('--no-qc');
    if (!fix) args.push('--no-fix');
    if (only) args.push('--only', only);
    return this.#stream(args);
  }

  /** Lit le rapport de QC le plus récent (dossier de travail, ou dossier d'export). */
  async report(dir) {
    const { readFile } = await import('node:fs/promises');
    const file = dir ? path.join(path.resolve(dir), 'qc_report.json')
      : path.join(this.projectDir, 'texforge_work', 'qc_report.json');
    return JSON.parse(await readFile(file, 'utf-8'));
  }

  // Processus long à événements JSON (run, qc, export) : { done, stop }
  #stream(args, scanFirst = false) {
    if (this.child) throw new Error('un traitement est déjà en cours');
    const start = async () => {
      if (scanFirst) await this.scan();
      const child = this.#spawn(args);
      this.child = child;
      const counts = {};
      const failures = [];
      let runEnd = null;
      readline.createInterface({ input: child.stdout }).on('line', (line) => {
        if (!line.trim()) return;
        let ev;
        try { ev = JSON.parse(line); } catch { this.emit('warning', `ligne non JSON : ${line}`); return; }
        counts[ev.event] = (counts[ev.event] ?? 0) + 1;
        if (ev.event === 'failed' || ev.event === 'fallback') failures.push(ev);
        if (ev.event === 'qc_done' && ['rejected', 'error'].includes(ev.decision)) failures.push(ev);
        if (ev.event === 'run_end' || ev.event === 'qc_end') runEnd = ev;
        this.emit('event', ev);
        this.emit(ev.event, ev);
      });
      readline.createInterface({ input: child.stderr }).on('line', (line) => this.emit('log', line));
      return new Promise((resolve, reject) => {
        child.on('error', (err) => { this.child = null; reject(err); });
        child.on('close', (code) => {
          this.child = null;
          const summary = { code, ok: code === 0, counts, failures, runEnd };
          if (code === 2) reject(Object.assign(new Error(`arrêt fatal de TEXFORGE (code 2)`), { summary }));
          else resolve(summary);
        });
      });
    };
    const done = start();
    const stop = async (timeoutMs = 120_000) => {
      // Arrêt propre : fichier STOP lu entre deux textures, puis arrêt forcé si besoin.
      await this.exec(['stop']);
      const child = this.child;
      if (!child) return;
      const timer = setTimeout(() => child.kill(), timeoutMs);
      await done.catch(() => {});
      clearTimeout(timer);
    };
    return { done, stop };
  }
}
