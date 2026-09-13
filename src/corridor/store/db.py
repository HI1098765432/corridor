"""Local project database.

Design:

*   SQLite holds the *index*: which analyses exist, what they were run with,
    what came out, and where the files are.  It never holds pixel data.
*   The filesystem holds the *results*: masks, CSVs, previews.  Large arrays
    belong in files that other scientific tools can open.
*   Every stage updates the row as it completes, so a crash leaves a project
    that still knows how far it got.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .. import app_meta

SCHEMA_VERSION = 1

STATUS_NEW = "new"
STATUS_RUNNING = "running"
STATUS_SEGMENTED = "segmented"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_STATUS_LABELS = {
    STATUS_NEW: "Not analysed",
    STATUS_RUNNING: "Analysing",
    STATUS_SEGMENTED: "Segmented",
    STATUS_COMPLETE: "Analysed",
    STATUS_FAILED: "Failed",
    STATUS_CANCELLED: "Stopped",
}


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


def app_data_dir() -> Path:
    """``%LOCALAPPDATA%/Corridor`` on Windows, a sensible equivalent elsewhere."""
    override = os.environ.get("CORRIDOR_DATA_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / app_meta.LOCAL_DIR_NAME


def projects_dir() -> Path:
    return app_data_dir() / "projects"


def database_path() -> Path:
    return app_data_dir() / "corridor.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ProjectRecord:
    id: str
    name: str
    source_path: str
    source_name: str
    directory: str
    created_at: str
    updated_at: str
    status: str = STATUS_NEW
    n_frames: int | None = None
    height: int | None = None
    width: int | None = None
    pixel_size_um: float | None = None
    frame_interval_min: float | None = None
    n_detections: int | None = None
    n_tracks: int | None = None
    n_warnings: int | None = None
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.directory)

    @property
    def preview_path(self) -> Path:
        return self.path / "preview.png"

    @property
    def source_exists(self) -> bool:
        return Path(self.source_path).exists()

    @property
    def has_results(self) -> bool:
        return (self.path / "tracks.csv").exists()

    def shape_text(self) -> str:
        if not self.n_frames:
            return ""
        return f"{self.n_frames} frames · {self.width}×{self.height}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    source_path        TEXT NOT NULL,
    source_name        TEXT NOT NULL,
    directory          TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    status             TEXT NOT NULL,
    n_frames           INTEGER,
    height             INTEGER,
    width              INTEGER,
    pixel_size_um      REAL,
    frame_interval_min REAL,
    n_detections       INTEGER,
    n_tracks           INTEGER,
    n_warnings         INTEGER,
    error              TEXT,
    config_json        TEXT,
    manifest_json      TEXT
);

CREATE INDEX IF NOT EXISTS projects_updated ON projects (updated_at DESC);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    """A small, synchronous SQLite wrapper. One instance per application."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        projects_dir().mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # -- lifecycle ---------------------------------------------------------
    def _migrate(self) -> None:
        with self.transaction() as cx:
            cx.executescript(_SCHEMA)
            row = cx.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                cx.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._conn:
                yield self._conn
        except sqlite3.Error:
            raise

    # -- projects ----------------------------------------------------------
    def create_project(
        self, source_path: str | Path, name: str | None = None
    ) -> ProjectRecord:
        source = Path(source_path)
        project_id = uuid.uuid4().hex[:12]
        directory = projects_dir() / project_id
        directory.mkdir(parents=True, exist_ok=True)
        now = _now()
        record = ProjectRecord(
            id=project_id,
            name=name or source.stem,
            source_path=str(source),
            source_name=source.name,
            directory=str(directory),
            created_at=now,
            updated_at=now,
            status=STATUS_NEW,
        )
        with self.transaction() as cx:
            cx.execute(
                """INSERT INTO projects
                   (id, name, source_path, source_name, directory, created_at,
                    updated_at, status)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    record.id, record.name, record.source_path, record.source_name,
                    record.directory, record.created_at, record.updated_at, record.status,
                ),
            )
        return record

    def update_project(self, record: ProjectRecord) -> None:
        record.updated_at = _now()
        with self.transaction() as cx:
            cx.execute(
                """UPDATE projects SET
                     name=?, source_path=?, source_name=?, directory=?, updated_at=?,
                     status=?, n_frames=?, height=?, width=?, pixel_size_um=?,
                     frame_interval_min=?, n_detections=?, n_tracks=?, n_warnings=?,
                     error=?, config_json=?, manifest_json=?
                   WHERE id=?""",
                (
                    record.name, record.source_path, record.source_name,
                    record.directory, record.updated_at, record.status,
                    record.n_frames, record.height, record.width,
                    record.pixel_size_um, record.frame_interval_min,
                    record.n_detections, record.n_tracks, record.n_warnings,
                    record.error,
                    json.dumps(record.config) if record.config else None,
                    json.dumps(record.manifest) if record.manifest else None,
                    record.id,
                ),
            )

    def set_status(self, project_id: str, status: str, error: str | None = None) -> None:
        with self.transaction() as cx:
            cx.execute(
                "UPDATE projects SET status=?, error=?, updated_at=? WHERE id=?",
                (status, error, _now(), project_id),
            )

    def get_project(self, project_id: str) -> ProjectRecord | None:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def recent_projects(self, limit: int = 30) -> list[ProjectRecord]:
        rows = self._conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def find_by_source(self, source_path: str | Path) -> ProjectRecord | None:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE source_path=? ORDER BY updated_at DESC LIMIT 1",
            (str(Path(source_path)),),
        ).fetchone()
        return _row_to_record(row) if row else None

    def delete_project(self, project_id: str, remove_files: bool = True) -> None:
        """Forget a project. Never touches the original microscopy file."""
        record = self.get_project(project_id)
        with self.transaction() as cx:
            cx.execute("DELETE FROM projects WHERE id=?", (project_id,))
        if remove_files and record is not None:
            directory = Path(record.directory)
            # Only ever delete inside our own managed area.
            if directory.is_dir() and projects_dir() in directory.parents:
                import shutil

                shutil.rmtree(directory, ignore_errors=True)

    # -- settings ----------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return default

    def set_setting(self, key: str, value: Any) -> None:
        with self.transaction() as cx:
            cx.execute(
                "INSERT INTO settings (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )


def _row_to_record(row: sqlite3.Row) -> ProjectRecord:
    return ProjectRecord(
        id=row["id"],
        name=row["name"],
        source_path=row["source_path"],
        source_name=row["source_name"],
        directory=row["directory"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        status=row["status"],
        n_frames=row["n_frames"],
        height=row["height"],
        width=row["width"],
        pixel_size_um=row["pixel_size_um"],
        frame_interval_min=row["frame_interval_min"],
        n_detections=row["n_detections"],
        n_tracks=row["n_tracks"],
        n_warnings=row["n_warnings"],
        error=row["error"],
        config=json.loads(row["config_json"]) if row["config_json"] else {},
        manifest=json.loads(row["manifest_json"]) if row["manifest_json"] else {},
    )
