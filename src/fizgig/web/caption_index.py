"""Disposable, persistent metadata cache for Caption Workbench directory scans."""
from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

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
        return self.read_many([(name, kind, reader)])[0]

    def read_many(self, requests, cancel=None):
        """Resolve a bounded batch, parallelizing only actual cache misses.

        Directory metadata and SQLite stay on the calling thread. Duplicate
        requests share one read; cache hits never enter the executor.
        """
        def check():
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Caption indexing canceled")

        results = [None] * len(requests)
        misses = {}
        for i, (name, kind, reader) in enumerate(requests):
            check()
            key = os.path.normcase(name)
            entry = self.entries.get(key)
            if entry is None:
                continue
            try:
                if not entry.is_file():
                    continue
                before = signature(entry.stat())
            except OSError:
                continue
            hit = self.rows.get((key, kind))
            if hit and hit[0] == before:
                try:
                    results[i] = json.loads(hit[1])
                    continue
                except (TypeError, ValueError):
                    pass
            k = (key, kind)
            if k in misses:
                misses[k][4].append(i)
            else:
                misses[k] = (os.path.join(self.folder, name), before, reader, k, [i])

        def read_file(job):
            path, _, reader, _, _ = job
            check()
            try:
                return reader(path)
            except OSError:
                return None

        def record(job, value):
            check()
            path, before, _, (key, kind), positions = job
            for i in positions:
                results[i] = value
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

        jobs = list(misses.values())
        # Small/warm batches avoid thread startup; cap cold reads to four
        # workers so a dataset scan does not saturate a training machine.
        workers = min(4, max(1, (len(jobs) + 15) // 16))
        if workers == 1:
            for job in jobs:
                record(job, read_file(job))
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="caption-index") as pool:
                for job, value in zip(jobs, pool.map(read_file, jobs)):
                    record(job, value)
        return results

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
