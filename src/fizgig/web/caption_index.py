"""Disposable, persistent metadata cache for Caption Workbench directory scans."""
from __future__ import annotations

import json
import os
import sqlite3

_BATCH = 256


def signature(st):
    return ":".join(str(x) for x in (st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino))


class FolderIndex:
    """Keep file enumeration authoritative; cache only unchanged file contents."""

    def __init__(self, db_path, folder):
        self.folder = os.fspath(folder)
        self.key = os.path.normcase(os.path.abspath(folder))
        with os.scandir(folder) as entries:
            self.entries = {os.path.normcase(e.name): e for e in entries}
        self.conn = None
        self.rows = {}
        self.pending = []
        try:
            self.conn = sqlite3.connect(db_path, timeout=0.25)
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata_v1 (folder TEXT, name TEXT, kind TEXT, "
                "sig TEXT, value TEXT, PRIMARY KEY(folder, name, kind))"
            )
            self.rows = {(name, kind): (sig, value) for name, kind, sig, value in
                         self.conn.execute("SELECT name, kind, sig, value FROM metadata_v1 WHERE folder=?",
                                           (self.key,))}
        except (OSError, sqlite3.Error):
            self._disable()

    def _disable(self):
        if self.conn is not None:
            self.conn.close()
        self.conn = None
        self.rows.clear()
        self.pending.clear()

    @property
    def names(self):
        return [e.name for e in self.entries.values()]

    def read(self, name, kind, reader):
        path = os.path.join(self.folder, name)
        key = os.path.normcase(name)
        entry = self.entries.get(key)
        if entry is None:
            return None
        try:
            if not entry.is_file():
                return None
            before = signature(entry.stat())
        except OSError:
            return None
        hit = self.rows.get((key, kind))
        if hit and hit[0] == before:
            try:
                return json.loads(hit[1])
            except (TypeError, ValueError):
                pass
        try:
            value = reader(path)
        except OSError:
            return None
        # Don't persist transient read failures or files modified during reading.
        if value is not None and self.conn is not None:
            try:
                if signature(os.stat(path)) == before:
                    encoded = json.dumps(value, ensure_ascii=True)
                    self.rows[(key, kind)] = (before, encoded)
                    self.pending.append((self.key, key, kind, before, encoded))
                    if len(self.pending) >= _BATCH:
                        self.flush()
            except OSError:
                pass
        return value

    def flush(self):
        if self.conn is None or not self.pending:
            return
        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO metadata_v1(folder,name,kind,sig,value) VALUES (?,?,?,?,?)",
                    self.pending,
                )
            self.pending.clear()
        except sqlite3.Error:
            self._disable()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.conn is not None:
            self.flush()
        self._disable()
