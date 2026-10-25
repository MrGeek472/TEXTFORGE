"""État persistant (SQLite) : chaque texture, chaque requête payante, chaque résultat.
Permet la reprise après interruption sans repayer une génération déjà lancée."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS textures (
    id TEXT PRIMARY KEY,          -- chemin relatif unique (racine + chemin)
    src_path TEXT NOT NULL,
    src_sha TEXT NOT NULL,
    name TEXT NOT NULL,           -- nom de fichier sans extension (sert à l'injection)
    fmt TEXT, mipmaps INTEGER, width INTEGER, height INTEGER,
    analysis TEXT,                -- JSON
    mode TEXT,                    -- ai | upscale | copy | skip
    hint TEXT DEFAULT '',
    status TEXT,                  -- pending | done | fallback | copied | skipped | error | kept | missing
    choice TEXT DEFAULT 'auto',   -- auto | local | ai (décision de revue)
    reviewed INTEGER DEFAULT 0,   -- 1 : choix fait à la main dans la galerie (le QC ne le modifie pas)
    attempts INTEGER DEFAULT 0,
    best_score REAL,
    metrics TEXT,                 -- JSON
    result_path TEXT,             -- PNG final retenu
    ai_path TEXT,                 -- meilleur candidat IA (même rejeté)
    local_path TEXT,              -- version agrandie localement
    fingerprint TEXT,
    error TEXT,
    updated REAL
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    texture_id TEXT NOT NULL,
    attempt INTEGER, seed INTEGER,
    status TEXT,                  -- submitted | completed | failed
    prep TEXT,                    -- JSON (géométrie de l'image envoyée)
    output_url TEXT,
    created REAL
);
CREATE TABLE IF NOT EXISTS uploads (
    sha TEXT PRIMARY KEY, url TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    texture_id TEXT NOT NULL,
    stage TEXT,                   -- ui | local | comfyui
    route TEXT,
    kind TEXT,                    -- oom | timeout | server | workflow | cuda | validation | image | qc | dds | error
    message TEXT,
    details TEXT,
    action TEXT,                  -- retry | fallback | abandon
    created REAL
);
CREATE INDEX IF NOT EXISTS failures_texture ON failures(texture_id);
"""


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        with self._lock:
            self.db.executescript(SCHEMA)
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(textures)").fetchall()}
            if "reviewed" not in cols:      # bases créées avant le contrôle qualité
                self.db.execute("ALTER TABLE textures ADD COLUMN reviewed INTEGER DEFAULT 0")
            self.db.commit()

    # ------------------------------------------------------------ textures
    def get(self, tid: str) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM textures WHERE id=?", (tid,)).fetchone()
        return self._row(row)

    def all(self, where: str = "1=1", params: tuple = ()) -> list[dict]:
        with self._lock:
            rows = self.db.execute(f"SELECT * FROM textures WHERE {where} ORDER BY id", params).fetchall()
        return [self._row(r) for r in rows]

    def upsert_scan(self, rec: dict) -> str:
        """Insère ou met à jour une texture scannée. Renvoie 'new', 'changed' ou 'same'."""
        with self._lock:
            old = self.db.execute("SELECT src_sha FROM textures WHERE id=?", (rec["id"],)).fetchone()
            now = time.time()
            if old is None:
                self.db.execute(
                    "INSERT INTO textures (id, src_path, src_sha, name, fmt, mipmaps, width, height, analysis,"
                    " mode, hint, status, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rec["id"], rec["src_path"], rec["src_sha"], rec["name"], rec["fmt"], rec["mipmaps"],
                     rec["width"], rec["height"], json.dumps(rec["analysis"]), rec["mode"], rec.get("hint", ""),
                     "pending", now))
                self.db.commit()
                return "new"
            if old["src_sha"] != rec["src_sha"]:
                self.db.execute(
                    "UPDATE textures SET src_path=?, src_sha=?, fmt=?, mipmaps=?, width=?, height=?, analysis=?,"
                    " mode=?, hint=?, status='pending', choice='auto', reviewed=0, attempts=0, best_score=NULL, metrics=NULL,"
                    " result_path=NULL, ai_path=NULL, local_path=NULL, fingerprint=NULL, error=NULL, updated=?"
                    " WHERE id=?",
                    (rec["src_path"], rec["src_sha"], rec["fmt"], rec["mipmaps"], rec["width"], rec["height"],
                     json.dumps(rec["analysis"]), rec["mode"], rec.get("hint", ""), now, rec["id"]))
                self.db.commit()
                return "changed"
            # Même fichier : on rafraîchit seulement le chemin, le mode et l'indice (règles modifiées)
            self.db.execute("UPDATE textures SET src_path=?, mode=?, hint=?, analysis=? WHERE id=?",
                            (rec["src_path"], rec["mode"], rec.get("hint", ""), json.dumps(rec["analysis"]),
                             rec["id"]))
            self.db.commit()
            return "same"

    def update(self, tid: str, **fields) -> None:
        if not fields:
            return
        for k in ("analysis", "metrics"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = json.dumps(fields[k])
        fields["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self.db.execute(f"UPDATE textures SET {cols} WHERE id=?", (*fields.values(), tid))
            self.db.commit()

    # ------------------------------------------------------------ requêtes
    def add_request(self, request_id: str, tid: str, attempt: int, seed: int, prep: dict) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?)",
                            (request_id, tid, attempt, seed, "submitted", json.dumps(prep), None, time.time()))
            self.db.commit()

    def finish_request(self, request_id: str, status: str, output_url: str | None = None) -> None:
        with self._lock:
            self.db.execute("UPDATE requests SET status=?, output_url=? WHERE request_id=?",
                            (status, output_url, request_id))
            self.db.commit()

    def requests_for(self, tid: str) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT * FROM requests WHERE texture_id=? ORDER BY attempt", (tid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["prep"] = json.loads(d["prep"]) if d["prep"] else None
            out.append(d)
        return out

    def count_requests(self) -> int:
        with self._lock:
            return self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    # ------------------------------------------------------------ envois
    def get_upload(self, sha: str, max_age: float = 6 * 3600) -> str | None:
        """URL d'un fichier déjà envoyé, si l'envoi est récent (les liens peuvent expirer)."""
        with self._lock:
            r = self.db.execute("SELECT url, created FROM uploads WHERE sha=?", (sha,)).fetchone()
        if r and time.time() - r["created"] < max_age:
            return r["url"]
        return None

    def put_upload(self, sha: str, url: str) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO uploads VALUES (?,?,?)", (sha, url, time.time()))
            self.db.commit()

    # ------------------------------------------------------------ échecs (traçabilité)
    def add_failure(self, tid: str, stage: str, route: str, kind: str, message: str, details: str = "",
                    action: str = "") -> None:
        with self._lock:
            self.db.execute("INSERT INTO failures (texture_id, stage, route, kind, message, details, action, created)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            (tid, stage, route, kind, message[:2000], details[:20000], action, time.time()))
            self.db.commit()

    def failures(self, tid: str | None = None, kind: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM failures WHERE 1=1", []
        if tid:
            sql += " AND texture_id=?"
            params.append(tid)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        with self._lock:
            rows = self.db.execute(sql + " ORDER BY id", params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ outils
    @staticmethod
    def _row(row) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for k in ("analysis", "metrics"):
            if d.get(k):
                d[k] = json.loads(d[k])
        return d

    def close(self) -> None:
        with self._lock:
            self.db.close()
