"""SQLite 持久化层。

四类一等数据：
* cases    工况本体（名字唯一）
* versions 工况参数的不可变版本；改区参数 -> 追加新版本，旧版本与其下结果永不被覆盖
* solves   每次收敛求解的留档（k、通量、分区反应率、迭代信息、冷/热启动）
             挂在求解那一刻的 version_id 下
* jobs     异步临界搜索作业，提交时把 version_id 快照写死（参数版本锁定）

数据库文件路径由 DIFFUSION_DB 指定，随容器卷保留；开启 WAL。
所有访问过同一把进程锁，SQLite 连接 check_same_thread=False。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import numpy as np

from app.core.models import Problem, Zone

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id         TEXT PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    version_no INTEGER NOT NULL,
    params     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(case_id, version_no)
);
CREATE TABLE IF NOT EXISTS solves (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    version_id         INTEGER NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    started_from       TEXT NOT NULL,
    total_fission_rate REAL NOT NULL,
    tol_k              REAL NOT NULL,
    tol_phi            REAL NOT NULL,
    max_iter           INTEGER NOT NULL,
    k_eff              REAL NOT NULL,
    iterations         INTEGER NOT NULL,
    residual_k         REAL NOT NULL,
    residual_phi       REAL NOT NULL,
    balance_error      REAL NOT NULL,
    flux_blob          BLOB NOT NULL,
    reactions          TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    case_id      TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    version_id   INTEGER NOT NULL,
    status       TEXT NOT NULL,   -- running|succeeded|failed|canceled|interrupted
    target_zone  INTEGER NOT NULL,
    target_field TEXT NOT NULL,   -- thickness | nu_sigma_f
    low          REAL NOT NULL,
    high         REAL NOT NULL,
    k_low        REAL,
    k_high       REAL,
    best_value   REAL,
    best_k       REAL,
    steps_done   INTEGER NOT NULL DEFAULT 0,
    max_steps    INTEGER NOT NULL,
    tol_k        REAL NOT NULL,
    tol_phi      REAL NOT NULL,
    max_iter     INTEGER NOT NULL,
    error        TEXT,
    result       TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex


# ---------- 参数序列化 ----------

def problem_to_dict(problem: Problem) -> dict[str, Any]:
    return {
        "left_bc": problem.left_bc,
        "right_bc": problem.right_bc,
        "zones": [
            {"thickness": z.thickness, "d": z.d, "sigma_a": z.sigma_a,
             "nu_sigma_f": z.nu_sigma_f, "n_mesh": z.n_mesh}
            for z in problem.zones
        ],
    }


def problem_from_dict(data: dict[str, Any]) -> Problem:
    zones = tuple(Zone(**z) for z in data["zones"])
    return Problem(zones=zones,
                   left_bc=data["left_bc"] if "left_bc" in data else data["leftBc"],
                   right_bc=data["right_bc"] if "right_bc" in data else data["rightBc"])


def flux_to_blob(phi: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, phi, allow_pickle=False)
    return buf.getvalue()


def flux_from_blob(blob: bytes) -> np.ndarray:
    return np.load(BytesIO(blob), allow_pickle=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"不可序列化对象 {type(obj)!r}")


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                      default=_json_default)


class Storage:
    def __init__(self, db_path: str):
        self._path = db_path
        if db_path != ":memory:":
            import os
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def new_id() -> str:
        return new_id()

    @staticmethod
    def utcnow() -> str:
        return utcnow()

    # ---------- 工况与版本 ----------

    def create_case(self, name: str, problem: Problem) -> dict[str, Any]:
        with self._lock:
            now = utcnow()
            case_id = new_id()
            self._conn.execute(
                "INSERT INTO cases(id, name, created_at) VALUES (?,?,?)",
                (case_id, name, now))
            version_id = self._insert_version(case_id, 1, problem, now)
            self._conn.commit()
        return {"id": case_id, "name": name, "created_at": now,
                "version_id": version_id, "version_no": 1}

    def _insert_version(self, case_id: str, no: int, problem: Problem,
                        now: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO versions(case_id, version_no, params, created_at)"
            " VALUES (?,?,?,?)",
            (case_id, no, json_dumps(problem_to_dict(problem)), now))
        return int(cur.lastrowid)

    def add_version(self, case_id: str, problem: Problem) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(version_no) AS m FROM versions WHERE case_id=?",
                (case_id,)).fetchone()
            no = (row["m"] or 0) + 1
            now = utcnow()
            version_id = self._insert_version(case_id, no, problem, now)
            self._conn.commit()
        return {"version_id": version_id, "version_no": no, "created_at": now}

    def list_cases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.name, c.created_at,"
                " v.id AS version_id, v.version_no"
                " FROM cases c JOIN versions v ON v.case_id=c.id"
                " WHERE v.version_no=(SELECT MAX(version_no) FROM versions"
                " WHERE case_id=c.id) ORDER BY c.created_at").fetchall()
        return [dict(r) for r in rows]

    def _case_row(self, case_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            case = self._case_row(case_id)
            if case is None:
                return None
            versions = self._conn.execute(
                "SELECT id AS version_id, version_no, params, created_at"
                " FROM versions WHERE case_id=? ORDER BY version_no",
                (case_id,)).fetchall()
            vlist = []
            for v in versions:
                d = dict(v)
                d["params"] = json.loads(d["params"])
                d["solve_count"] = self._conn.execute(
                    "SELECT COUNT(*) FROM solves WHERE version_id=?",
                    (v["version_id"],)).fetchone()[0]
                vlist.append(d)
            current = vlist[-1]
        return {"id": case["id"], "name": case["name"],
                "created_at": case["created_at"],
                "current_version": current, "versions": vlist}

    def get_version(self, version_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT v.*, c.name AS case_name FROM versions v"
                " JOIN cases c ON c.id=v.case_id WHERE v.id=?",
                (version_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["problem"] = problem_from_dict(json.loads(d["params"]))
        return d

    def latest_version_id(self, case_id: str) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM versions WHERE case_id=?"
                " ORDER BY version_no DESC LIMIT 1", (case_id,)).fetchone()
        return None if row is None else int(row["id"])

    def previous_version_id(self, version_id: int) -> int | None:
        """取同一工况上一版（热启动找初值用）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT case_id, version_no FROM versions WHERE id=?",
                (version_id,)).fetchone()
            if row is None:
                return None
            prev = self._conn.execute(
                "SELECT id FROM versions WHERE case_id=? AND version_no<?",
                (row["case_id"], row["version_no"])).fetchone()
        return None if prev is None else int(prev["id"])

    def rename_case(self, case_id: str, name: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE cases SET name=? WHERE id=?",
                               (name, case_id))
            self._conn.commit()

    def delete_case(self, case_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM cases WHERE id=?",
                                     (case_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ---------- 求解留档 ----------

    def add_solve(self, *, case_id: str, version_id: int, started_from: str,
                  total_fission_rate: float, tol_k: float, tol_phi: float,
                  max_iter: int, k_eff: float, iterations: int,
                  residual_k: float, residual_phi: float,
                  balance_error: float, phi: np.ndarray,
                  reactions: dict[str, Any]) -> int:
        with self._lock:
            now = utcnow()
            cur = self._conn.execute(
                "INSERT INTO solves(case_id, version_id, started_from,"
                " total_fission_rate, tol_k, tol_phi, max_iter, k_eff,"
                " iterations, residual_k, residual_phi, balance_error,"
                " flux_blob, reactions, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, version_id, started_from, total_fission_rate,
                 tol_k, tol_phi, max_iter, float(k_eff), iterations,
                 float(residual_k), float(residual_phi),
                 float(balance_error), flux_to_blob(phi),
                 json_dumps(reactions), now))
            self._conn.commit()
            return int(cur.lastrowid)

    def list_solves(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id, s.version_id, s.started_from, s.k_eff,"
                " s.iterations, s.residual_k, s.residual_phi,"
                " s.balance_error, s.total_fission_rate, s.created_at,"
                " v.version_no FROM solves s JOIN versions v"
                " ON v.id=s.version_id WHERE s.case_id=?"
                " ORDER BY s.id", (case_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_solve(self, solve_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT s.*, v.version_no FROM solves s JOIN versions v"
                " ON v.id=s.version_id WHERE s.id=?", (solve_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["flux"] = flux_from_blob(d.pop("flux_blob"))
            d["reactions"] = json.loads(d["reactions"])
        return d

    def latest_solve(self, version_id: int) -> dict[str, Any] | None:
        """该版本最近一次收敛解（热启动取初值的来源）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM solves WHERE version_id=?"
                " ORDER BY id DESC LIMIT 1", (version_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["flux"] = flux_from_blob(d.pop("flux_blob"))
        return d

    # ---------- 临界搜索作业 ----------

    def create_job(self, job: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(id, case_id, version_id, status,"
                " target_zone, target_field, low, high, k_low, k_high,"
                " best_value, best_k, steps_done, max_steps, tol_k, tol_phi,"
                " max_iter, error, result, created_at, updated_at)"
                " VALUES (:id,:case_id,:version_id,:status,:target_zone,"
                ":target_field,:low,:high,:k_low,:k_high,:best_value,"
                ":best_k,:steps_done,:max_steps,:tol_k,:tol_phi,:max_iter,"
                ":error,:result,:created_at,:updated_at)", job)
            self._conn.commit()

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utcnow()
        keys = ", ".join(f"{k}=:{k}" for k in fields)
        fields["id"] = job_id
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {keys} WHERE id=:id", fields)
            self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?",
                                     (job_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_jobs(self, case_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if case_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE case_id=? ORDER BY created_at",
                    (case_id,)).fetchall()
        return [dict(r) for r in rows]

    def reset_stale_running_jobs(self) -> int:
        """服务启动时把上次残留的 running 作业标记为 interrupted。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='interrupted',"
                " error='服务重启，作业中断（不会继续写结果）',"
                f" updated_at='{utcnow()}' WHERE status='running'")
            self._conn.commit()
            return cur.rowcount
