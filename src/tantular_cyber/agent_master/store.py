"""Private SQLite queue. No raw conversations, source text, or model responses stored."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        # Reject symlink stores; private files must not alias analyzed files.
        if path.is_symlink():
            raise ValueError("symlink state database refused")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        path.chmod(0o600)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS runs (
          id TEXT PRIMARY KEY, request TEXT, model TEXT, status TEXT, language TEXT,
          deadline REAL, calls INTEGER DEFAULT 0, steps INTEGER DEFAULT 0,
          plan TEXT, report TEXT, limits TEXT, policy TEXT);
        CREATE TABLE IF NOT EXISTS tasks (
          run_id TEXT REFERENCES runs(id), id TEXT, assignment TEXT,
          status TEXT, result TEXT, PRIMARY KEY(run_id,id));
        CREATE TABLE IF NOT EXISTS effects (
          run_id TEXT, task_id TEXT, id TEXT, tool TEXT, status TEXT, receipt TEXT,
          PRIMARY KEY(run_id,id));
        """)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def new(self, request: str, model: str, limits: dict, policy: dict) -> str:
        run = uuid.uuid4().hex[:12]
        with self.db:
            self.db.execute(
                "INSERT INTO runs(id,request,model,status,deadline,limits,policy) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    run,
                    request,
                    model,
                    "planning",
                    time.time() + limits["max_duration"],
                    json.dumps(limits),
                    json.dumps(policy),
                ),
            )
        return run

    def run(self, run: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
        if row is None:
            raise ValueError("unknown run")
        return dict(row)

    def runs(self) -> list[dict]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT id,status,model,calls,steps FROM runs ORDER BY rowid DESC LIMIT 20"
            )
        ]

    def update(self, run: str, **fields: Any) -> None:
        statements = {
            "status": "UPDATE runs SET status=? WHERE id=?",
            "language": "UPDATE runs SET language=? WHERE id=?",
            "plan": "UPDATE runs SET plan=? WHERE id=?",
            "report": "UPDATE runs SET report=? WHERE id=?",
            "calls": "UPDATE runs SET calls=? WHERE id=?",
            "steps": "UPDATE runs SET steps=? WHERE id=?",
        }
        if not fields or not set(fields) <= statements.keys():
            raise ValueError("unsupported state field")
        with self.db:
            for key, value in fields.items():
                self.db.execute(statements[key], (value, run))

    def plan(self, run: str, plan: dict) -> None:
        with self.db:
            self.db.execute(
                "UPDATE runs SET plan=?,language=?,status=? WHERE id=?",
                (json.dumps(plan), plan["language"], "running", run),
            )
            for task in plan["tasks"]:
                self.db.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,NULL)",
                    (run, task["id"], json.dumps(task), "queued"),
                )

    def tasks(self, run: str) -> list[dict]:
        return [
            dict(t)
            for t in self.db.execute("SELECT * FROM tasks WHERE run_id=? ORDER BY rowid", (run,))
        ]

    def task(self, run: str, task: str, status: str, result: dict | None = None) -> None:
        with self.db:
            self.db.execute(
                "UPDATE tasks SET status=?, result=? WHERE run_id=? AND id=?",
                (status, json.dumps(result) if result else None, run, task),
            )

    def intent(self, run: str, task: str, effect: str, tool: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO effects VALUES(?,?,?,?,?,NULL)", (run, task, effect, tool, "started")
            )

    def receipt(self, run: str, effect: str, receipt: dict) -> None:
        with self.db:
            self.db.execute(
                "UPDATE effects SET status=?,receipt=? WHERE run_id=? AND id=?",
                ("done", json.dumps(receipt), run, effect),
            )

    def receipts(self, run: str, task: str | None = None) -> list[dict]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM effects WHERE run_id=? AND (? IS NULL OR task_id=?)",
                (run, task, task),
            )
        ]

    def recover(self) -> None:
        # Never replay an in-flight tool or model request. Pending tasks remain queued.
        with self.db:
            self.db.execute(
                "UPDATE tasks SET status='blocked', result=? WHERE status='running'",
                (
                    json.dumps(
                        {
                            "status": "blocked",
                            "summary": "Interrupted task",
                            "evidence": [],
                            "artifact_paths": [],
                            "limitations": [],
                            "blocker": "Interrupted; inspect effect receipts and artifacts before "
                            "creating a replacement task. No action was replayed.",
                        }
                    ),
                ),
            )
            self.db.execute(
                "UPDATE runs SET status='interrupted' "
                "WHERE status IN ('planning','running','reporting')"
            )
