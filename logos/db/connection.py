"""Logos 数据库连接层 —— SQLite 连接的生命周期管理。

设计目标
--------
1. **零第三方依赖**：只用标准库 ``sqlite3``，不引入 ORM。
2. **安全的默认值**：连接一建立就自动应用一组经过实测的 PRAGMA 调优参数
   （WAL / NORMAL / 64MB 页缓存 / 256MB mmap / MEMORY temp_store），
   并用「写后读回」校验，杜绝「PRAGMA 静默失效」。
3. **显式事务**：``isolation_level=None``，Python 层不再隐式开启事务；
   需要原子性时用 :func:`transaction` 显式包裹。这消除了「忘记 commit」
   与「PRAGMA 落在事务内变成 no-op」这两类难查的问题。
4. **可观测**：计时装饰器 + 内存哈希缓存 + 活连接登记，便于性能分析与泄漏排查。

四个必须知道的 SQLite 陷阱（本模块已全部规避，均有实测支撑）
-----------------------------------------------------------
1. ``PRAGMA foreign_keys`` **默认是 OFF**，且**在事务内是 no-op**（静默失效）。
   本模块在连接建立时立刻设置（此时必然无事务），随后**读回校验**，
   不一致就抛 :class:`DatabaseError`。
2. ``PRAGMA journal_mode=WAL`` 的报错**不在 execute() 抛出，而延迟到 fetch()**。
   只执行不取行会把错误吞掉 —— 本模块所有 PRAGMA 赋值都强制取回结果。
3. ``sqlite3.Connection.executescript()`` 有两个副作用（均已实测）：
   出错时**留下未闭合事务**；执行前**隐式提交**调用方未完成的事务。
   本模块因此改用自研的 :func:`split_statements` 逐句执行。该切分器相比
   标准库 ``sqlite3.complete_statement`` 的价值不在于"能不能切对"（两者都能
   正确处理 ``CREATE TRIGGER ... BEGIN ... END``），而在于**它能给出精确行号**
   （错误可报到 ``schema_v3.sql:123``）、能拆开同一行的多条语句、
   且不把纯注释块当成语句 —— 详见该函数的文档。
4. Python 的 ``with sqlite3.connect(...) as conn`` **不关闭连接**，只提交事务。
   本模块用 :func:`connect` 上下文管理器显式负责 close。

另注：``sqlite3.Connection`` 本体**既不支持属性赋值也不支持 weakref**，
因此本模块用 :class:`LogosConnection` 子类承载元数据（``isinstance`` 判定
依然通过，对调用方完全透明）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import sqlite3
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

__all__ = [
    # 异常
    "DatabaseError",
    "DatabaseConnectionError",
    "SchemaError",
    "SeedError",
    "TransactionError",
    "IntegrityCheckError",
    # 配置
    "PragmaProfile",
    "DEFAULT_PROFILE",
    # 连接
    "LogosConnection",
    "get_connection",
    "connect",
    "configure_connection",
    "close_connection",
    "close_all",
    "live_connections",
    "transaction",
    "savepoint",
    # 缓存
    "MemoryCache",
    "get_memory_cache",
    "clear_caches",
    "hash_key",
    "file_digest",
    # 计时
    "timed",
    "timed_block",
    "execute_timed",
    "count_sql",
    "get_timing_stats",
    "reset_timing_stats",
    "dump_timing_stats",
    # 建库
    "init_database",
    "InitReport",
    "split_statements",
    # 常量
    "PROJECT_ROOT",
    "DEFAULT_DB_PATH",
    "SCHEMA_FILENAME",
    "SEED_GLOB",
]

log = logging.getLogger("logos.db")

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
# 本文件位于 <项目根>/logos/db/connection.py，故 parents[2] 即项目根目录
PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH: pathlib.Path = PROJECT_ROOT / "logos.db"
SCHEMA_FILENAME: str = "schema_v3.sql"
SEED_GLOB: str = "seed_*.sql"


# ===========================================================================
# 1. 异常体系
# ===========================================================================
class DatabaseError(Exception):
    """Logos 数据层所有异常的基类。

    附带结构化上下文，便于上层日志/排障直接取用：

    :param message: 人类可读的错误说明
    :param db_path: 涉及的数据库文件
    :param sql:     引发错误的 SQL（可能被截断）
    :param params:  绑定参数
    :param cause:   原始底层异常（``sqlite3.Error`` 等）
    """

    def __init__(
        self,
        message: str,
        *,
        db_path: str | os.PathLike[str] | None = None,
        sql: str | None = None,
        params: Sequence[Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.message = message
        self.db_path = str(db_path) if db_path is not None else None
        self.sql = sql
        self.params = tuple(params) if params is not None else None
        self.cause = cause

        parts = [message]
        if self.db_path:
            parts.append(f"[db={self.db_path}]")
        if cause is not None:
            parts.append(f"[cause={type(cause).__name__}: {cause}]")
        if sql:
            parts.append(f"[sql={_snippet(sql)}]")
        super().__init__(" ".join(parts))


class DatabaseConnectionError(DatabaseError):
    """连接无法建立（文件不存在 / 无权限 / URI 非法等）。

    不叫 ``ConnectionError`` 是为了不与内置异常同名。
    """


class SchemaError(DatabaseError):
    """建表脚本（schema）加载失败，含失败语句的行号。"""


class SeedError(DatabaseError):
    """种子数据脚本加载失败，含失败语句的行号。"""


class TransactionError(DatabaseError):
    """事务使用方式非法（例如在未闭合事务上再开事务）。"""


class IntegrityCheckError(DatabaseError):
    """``integrity_check`` / ``foreign_key_check`` 未通过。"""


def _snippet(text: str, limit: int = 160) -> str:
    """把 SQL 压成单行短片段，用于错误信息。"""
    flat = re.sub(r"\s+", " ", text or "").strip()
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


# ===========================================================================
# 2. 连接参数配置
# ===========================================================================
@dataclass(frozen=True)
class PragmaProfile:
    """连接级 PRAGMA 调优档案（不可变，可安全共享）。

    每个 PRAGMA 都对应一个 ``expected()`` 里的期望值，供 :func:`configure_connection`
    写后读回校验。

    注意 ``journal_mode`` 是**持久化**到数据库文件的；其余都是**连接级**的
    （``foreign_keys`` / ``recursive_triggers`` / ``cache_size`` …），
    因此每个新连接都必须重新应用 —— 这正是 :func:`get_connection` 默认做的事。
    """

    #: WAL 支持读写并发；只读连接与内存库会自动跳过
    journal_mode: str = "WAL"
    #: WAL 下 NORMAL 已能保证「断电不损坏库」，且比 FULL 快得多
    synchronous: str = "NORMAL"
    #: 负值 = KiB。−64000 KiB ≈ 64MB 页缓存
    cache_size: int = -64000
    #: 临时表/B 树放内存，避免落盘
    temp_store: str = "MEMORY"
    #: 256MB 内存映射 I/O
    mmap_size: int = 268435456
    #: 锁等待上限，避免立即抛 SQLITE_BUSY
    busy_timeout_ms: int = 5000
    #: **最关键的一项**：SQLite 默认 OFF，不设则全部 FK 形同虚设
    foreign_keys: bool = True
    #: 闭包维护依赖触发器级联语义（见 schema §13）
    recursive_triggers: bool = True
    #: WAL 自动 checkpoint 阈值（页数），防止 -wal 无限增长
    wal_autocheckpoint: int = 1000
    #: 批量导入时延迟 FK 校验到 COMMIT。默认关闭：立即校验能报出**具体哪一行**违规
    defer_foreign_keys: bool = False
    # 注：Python 侧 isolation_level 固定为 None（显式事务模式），由 get_connection 设定，
    #     它不是 PRAGMA、不随连接档位变化，故不作为字段暴露。

    def expected(self) -> dict[str, Any]:
        """返回「读回值应当等于多少」的映射，供自检与调试使用。"""
        exp: dict[str, Any] = {
            "journal_mode": (self.journal_mode or "").lower(),
            "synchronous": _SYNCHRONOUS_TO_INT.get(self.synchronous.upper(), self.synchronous),
            "cache_size": self.cache_size,
            "temp_store": _TEMP_STORE_TO_INT.get(self.temp_store.upper(), self.temp_store),
            "mmap_size": self.mmap_size,
            "busy_timeout": self.busy_timeout_ms,
            "foreign_keys": int(self.foreign_keys),
            "recursive_triggers": int(self.recursive_triggers),
            "wal_autocheckpoint": self.wal_autocheckpoint,
            "defer_foreign_keys": int(self.defer_foreign_keys),
        }
        return exp


_SYNCHRONOUS_TO_INT = {"OFF": 0, "NORMAL": 1, "FULL": 2, "EXTRA": 3}
_TEMP_STORE_TO_INT = {"DEFAULT": 0, "FILE": 1, "MEMORY": 2}

#: 模块默认档案：非只读文件库使用
DEFAULT_PROFILE = PragmaProfile()


# ===========================================================================
# 3. 连接对象
# ===========================================================================
class LogosConnection(sqlite3.Connection):
    """``sqlite3.Connection`` 的子类，额外承载元数据与便捷诊断方法。

    之所以要子类化：原生 ``sqlite3.Connection`` **既不允许设置属性、
    也不支持 weakref**（实测 ``AttributeError`` / ``TypeError``），
    无法挂载"这个连接是只读的/它的 PRAGMA 应用结果"这类信息。
    子类实例则是普通 Python 对象，``isinstance(conn, sqlite3.Connection)``
    依旧为真，对调用方完全透明。
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        timeout: float = 30.0,
        isolation_level: None = None,
        check_same_thread: bool = True,
        uri: bool = False,
        cached_statements: int = 128,
        detect_types: int = 0,
        row_factory: Any = None,
    ) -> None:
        super().__init__(
            str(database),
            timeout=timeout,
            isolation_level=isolation_level,
            check_same_thread=check_same_thread,
            uri=uri,
            cached_statements=cached_statements,
            detect_types=detect_types,
        )
        #: 结构化元数据，供 close_connection / 自检读取
        self.logos_meta: dict[str, Any] = {
            "db_path": None if str(database) == ":memory:" else str(database),
            "read_only": False,
            "memory": str(database) == ":memory:" or "mode=memory" in str(database),
            "profile": None,
            "pragmas": {},
            "created_at": time.time(),
            "closed": False,
        }
        if row_factory is not None:
            self.row_factory = row_factory

    # ---- 便捷访问器 -------------------------------------------------------
    @property
    def db_path(self) -> str | None:
        return self.logos_meta.get("db_path")

    @property
    def read_only(self) -> bool:
        return bool(self.logos_meta.get("read_only"))

    @property
    def is_memory(self) -> bool:
        return bool(self.logos_meta.get("memory"))

    @property
    def pragmas(self) -> dict[str, Any]:
        """:func:`configure_connection` 的读回结果。"""
        return dict(self.logos_meta.get("pragmas") or {})

    def pragma(self, name: str) -> Any:
        """读取单个 PRAGMA 的当前值（标量）。"""
        row = self.execute(f"PRAGMA {name}").fetchone()
        return row[0] if row else None

    # ---- 结构自省 ---------------------------------------------------------
    def tables(self, *, include_internal: bool = False) -> list[str]:
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        if not include_internal:
            sql += " AND name NOT LIKE 'sqlite_%'"
        return [r[0] for r in self.execute(sql + " ORDER BY name")]

    def table_exists(self, name: str) -> bool:
        return (
            self.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
                (name,),
            ).fetchone()
            is not None
        )

    def row_count(self, table: str) -> int:
        # 表名不可参数化，这里只允许标识符，避免注入
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise DatabaseError(f"非法表名: {table!r}", db_path=self.db_path)
        return self.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def counts(self, include: Iterable[str] | None = None) -> dict[str, int]:
        """批量统计行数；``include`` 给定时只统计这些表（忽略不存在的）。"""
        names = self.tables() if include is None else list(include)
        out: dict[str, int] = {}
        for t in names:
            try:
                out[t] = self.row_count(t)
            except (DatabaseError, sqlite3.Error):
                continue
        return out

    # ---- 健康检查 ---------------------------------------------------------
    def foreign_key_violations(self) -> list[tuple[Any, ...]]:
        """``PRAGMA foreign_key_check``：列出全部悬空外键（无违规则空）。

        每行为 ``(子表, 行号, 父表, fk序号)``。
        """
        return [tuple(r) for r in self.execute("PRAGMA foreign_key_check").fetchall()]

    def fk_ok(self) -> bool:
        return not self.foreign_key_violations()

    def integrity_ok(self) -> bool:
        """``PRAGMA integrity_check``：整库物理一致性。"""
        rows = self.execute("PRAGMA integrity_check").fetchall()
        return len(rows) == 1 and rows[0][0] == "ok"

    def explain(self, sql: str, params: Sequence[Any] = ()) -> list[str]:
        """返回 ``EXPLAIN QUERY PLAN`` 的文本行，用于索引命中排查。"""
        rows = self.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
        return [str(r[-1]) for r in rows]

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        kind = "memory" if self.is_memory else ("ro" if self.read_only else "rw")
        return f"<LogosConnection {kind} {self.db_path!r}>"


# 活连接登记：原生 Connection 不支持 weakref，但**子类实例支持**（已实测），
# 因此这里用 WeakSet 做零泄漏的泄漏诊断。
_LIVE: "weakref.WeakSet[LogosConnection]" = weakref.WeakSet()
_LIVE_LOCK = threading.RLock()


def live_connections() -> list[LogosConnection]:
    """列出当前仍存活的、由本模块创建的连接（用于泄漏排查）。"""
    with _LIVE_LOCK:
        return [c for c in list(_LIVE) if not c.logos_meta.get("closed")]


# ===========================================================================
# 4. PRAGMA 应用与校验
# ===========================================================================
# PRAGMA 的值无法参数化，只能拼字符串 —— 因此必须严格白名单校验
_PRAGMA_VALUE_RE = re.compile(r"^(?:-?\d+|[A-Za-z_][A-Za-z0-9_]*)$")


def _set_pragma(conn: sqlite3.Connection, name: str, value: Any) -> Any:
    """设置单个 PRAGMA 并**立即取回**结果。

    强制取回是关键：``PRAGMA journal_mode=WAL`` 在只读库上失败时，
    错误不在 ``execute()`` 抛出，而是延迟到 ``fetchone()``（实测）。
    只 execute 不 fetch 会静默吞掉错误。
    """
    text = str(value)
    if not _PRAGMA_VALUE_RE.match(text):
        raise DatabaseError(f"PRAGMA {name} 的值不合法: {value!r}")
    try:
        row = conn.execute(f"PRAGMA {name}={text}").fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError(
            f"设置 PRAGMA {name}={text} 失败", sql=f"PRAGMA {name}={text}", cause=exc
        ) from exc
    return row[0] if row else None


def _read_pragma(conn: sqlite3.Connection, name: str) -> Any:
    row = conn.execute(f"PRAGMA {name}").fetchone()
    return row[0] if row else None


def configure_connection(
    conn: sqlite3.Connection,
    profile: PragmaProfile | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """在连接上应用 :class:`PragmaProfile`，返回**读回校验**结果。

    顺序很重要，且**绝不能在有未闭合事务时调用**：

    * ``journal_mode`` 与 ``foreign_keys`` 在事务内都是 no-op（静默失效），
      所以本函数一旦发现 ``conn.in_transaction`` 就直接抛
      :class:`TransactionError`，而不是悄悄跳过。
    * ``journal_mode`` 必须在 ``synchronous`` **之前**设置：改变日志模式
      可能会重置同步级别。
    * ``foreign_keys`` 放在最后并**强制校验**，这是全库引用完整性的唯一保障。

    :param strict: 若 ``foreign_keys`` 未能真正生效则抛 :class:`DatabaseError`。
       生产环境不建议关闭 —— 它拦住的是"静默数据损坏"。
    """
    prof = profile or DEFAULT_PROFILE

    if conn.in_transaction:
        raise TransactionError(
            "不能在未闭合的事务内配置连接：PRAGMA journal_mode / foreign_keys "
            "在事务内是 no-op，会被静默忽略。请先 commit()/rollback()。",
            db_path=getattr(conn, "db_path", None),
        )

    meta = getattr(conn, "logos_meta", None)
    read_only = bool(meta.get("read_only")) if isinstance(meta, dict) else False
    is_memory = bool(meta.get("memory")) if isinstance(meta, dict) else _is_memory_dsn(conn)

    applied: dict[str, Any] = {}
    notes: list[str] = []

    # 1) 锁等待：先设，后续语句都能受益
    _set_pragma(conn, "busy_timeout", prof.busy_timeout_ms)

    # 2) 日志模式（只读库/内存库跳过）
    if read_only or is_memory:
        applied["journal_mode"] = _read_pragma(conn, "journal_mode")
        notes.append(f"跳过 journal_mode 设置（{'只读' if read_only else '内存'}库）")
    else:
        got = _set_pragma(conn, "journal_mode", prof.journal_mode)
        applied["journal_mode"] = got
        if (got or "").lower() != (prof.journal_mode or "").lower():
            # 网络盘/某些文件系统不支持 WAL；不是致命错误，但必须让调用方知道
            notes.append(
                f"journal_mode 期望 {prof.journal_mode} 实得 {got}（该文件系统可能不支持 WAL）"
            )
            log.warning("journal_mode 未被接受：期望=%s 实得=%s", prof.journal_mode, got)

    # 3) 同步级别（必须在 journal_mode 之后）
    _set_pragma(conn, "synchronous", prof.synchronous)
    applied["synchronous"] = _read_pragma(conn, "synchronous")

    # 4) 缓存与临时存储
    _set_pragma(conn, "temp_store", prof.temp_store)
    _set_pragma(conn, "cache_size", prof.cache_size)
    _set_pragma(conn, "mmap_size", prof.mmap_size)
    applied["temp_store"] = _read_pragma(conn, "temp_store")
    applied["cache_size"] = _read_pragma(conn, "cache_size")
    applied["mmap_size"] = _read_pragma(conn, "mmap_size")

    # 5) WAL 自动 checkpoint
    if not is_memory:
        _set_pragma(conn, "wal_autocheckpoint", prof.wal_autocheckpoint)
        applied["wal_autocheckpoint"] = _read_pragma(conn, "wal_autocheckpoint")

    # 6) 递归触发器（闭包级联维护所需）
    _set_pragma(conn, "recursive_triggers", 1 if prof.recursive_triggers else 0)
    applied["recursive_triggers"] = _read_pragma(conn, "recursive_triggers")

    # 7) 外键 —— 最关键，放最后并强校验
    _set_pragma(conn, "foreign_keys", 1 if prof.foreign_keys else 0)
    applied["foreign_keys"] = _read_pragma(conn, "foreign_keys")

    # 8) 延迟外键（仅批量导入时开启）
    _set_pragma(conn, "defer_foreign_keys", 1 if prof.defer_foreign_keys else 0)
    applied["defer_foreign_keys"] = _read_pragma(conn, "defer_foreign_keys")

    applied["busy_timeout"] = _read_pragma(conn, "busy_timeout")

    if prof.foreign_keys and applied.get("foreign_keys") != 1:
        msg = (
            "PRAGMA foreign_keys 未能生效（读回=%r）。这意味着全库外键约束形同虚设，"
            "写入脏数据不会报错。常见原因：连接处于事务中，或 PRAGMA 被后续语句覆盖。"
            % (applied.get("foreign_keys"),)
        )
        if strict:
            raise DatabaseError(msg, db_path=getattr(conn, "db_path", None))
        log.error(msg)

    if isinstance(getattr(conn, "logos_meta", None), dict):
        conn.logos_meta["pragmas"] = dict(applied)
        conn.logos_meta["profile"] = prof
        conn.logos_meta["pragmas_notes"] = notes

    return applied


def _is_memory_dsn(conn: sqlite3.Connection) -> bool:
    """通过 ``PRAGMA database_list`` 判断是否内存库（``:memory:`` 的 file 为空）。"""
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.Error:
        return False
    for r in rows:
        if r[1] == "main":
            return not r[2]  # file 为空 => 内存库
    return False


# ===========================================================================
# 5. 连接的获取 / 上下文管理 / 事务
# ===========================================================================
def _resolve_and_check_path(
    db_path: str | os.PathLike[str] | None,
    *,
    read_only: bool,
    must_exist: bool,
    create_parents: bool,
) -> tuple[str, bool]:
    """规整 db_path 并做存在性/权限前置检查。

    返回 ``(传给 sqlite3 的 DSN, 是否内存库)``。

    只读连接用 ``Path.as_uri()`` 构造 URI —— 它会对非 ASCII 字符做百分号编码。
    本项目的路径含中文（``数据库logos``），直接拼 ``file:`` 前缀会打不开库。
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    raw = str(db_path)

    if raw == ":memory:" or raw.startswith("file:") and "mode=memory" in raw:
        return raw, True

    path = pathlib.Path(raw)
    if create_parents and not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists()
    if read_only and not exists:
        raise DatabaseConnectionError(
            f"只读打开失败：文件不存在 {path}", db_path=path
        )
    if must_exist and not exists:
        raise DatabaseConnectionError(f"数据库文件不存在：{path}", db_path=path)

    if read_only:
        # as_uri() 会百分号编码中文路径；immutable 不启用（WAL 库需要读写 -shm）
        return path.resolve().as_uri() + "?mode=ro", False
    return str(path), False


def get_connection(
    db_path: str | os.PathLike[str] | None = None,
    *,
    read_only: bool = False,
    profile: PragmaProfile | None = None,
    timeout: float = 30.0,
    check_same_thread: bool = True,
    row_factory: Any = sqlite3.Row,
    must_exist: bool = False,
    create_parents: bool = True,
    apply_pragmas: bool = True,
    strict_pragmas: bool = True,
    uri: bool = False,
) -> LogosConnection:
    """建立一条已完成 PRAGMA 调优的连接。

    这是模块的主入口。返回的 :class:`LogosConnection` 依然是
    ``sqlite3.Connection``（子类），可直接喂给任何现有代码。

    :param db_path: 数据库路径；``None`` 用 :data:`DEFAULT_DB_PATH`。
        也接受 ``":memory:"`` 或 ``"file:...?mode=memory"``。
    :param read_only: 以 ``mode=ro`` URI 打开。只读连接会自动跳过
        ``journal_mode`` 设置（只读库上会被拒），并只允许 SELECT。
    :param row_factory: 默认 ``sqlite3.Row`` —— 行可按列名取值，
        对本项目的知识库查询更友好；传 ``None`` 恢复元组。
    :param must_exist: 为 ``True`` 时文件不存在直接抛错，
        避免手误的路径静默创建一个空库。
    :param apply_pragmas: 关掉可跳过 PRAGMA（例如挂到别人的连接上做诊断）。

    .. warning::
       写模式下若文件不存在，SQLite 会**静默创建**一个空库。
       生产代码建议 ``must_exist=True``（读取场景）或先调 :func:`init_database`。
    """
    dsn, is_memory = _resolve_and_check_path(
        db_path, read_only=read_only, must_exist=must_exist, create_parents=create_parents
    )
    use_uri = uri or read_only

    try:
        conn = LogosConnection(
            dsn,
            timeout=timeout,
            isolation_level=None,  # 显式事务模式
            check_same_thread=check_same_thread,
            uri=use_uri,
            row_factory=row_factory,
        )
    except sqlite3.Error as exc:
        raise DatabaseConnectionError(
            f"打开数据库失败：{dsn}", db_path=dsn, cause=exc
        ) from exc

    conn.logos_meta["read_only"] = bool(read_only)
    conn.logos_meta["memory"] = bool(is_memory)

    if apply_pragmas:
        configure_connection(conn, profile, strict=strict_pragmas)
    else:
        # 即便跳过调优，外键也必须开 —— 否则后续所有 FK 都是装饰品
        _set_pragma(conn, "foreign_keys", 1)

    with _LIVE_LOCK:
        _LIVE.add(conn)
    log.debug("已建立连接 %r", conn)
    return conn


@contextmanager
def connect(
    db_path: str | os.PathLike[str] | None = None, **kwargs: Any
) -> Iterator[LogosConnection]:
    """连接上下文管理器：**退出时保证关闭**。

    ::

        with connect("logos.db", read_only=True) as conn:
            rows = conn.execute("SELECT * FROM words LIMIT 10").fetchall()

    与标准库的区别：``with sqlite3.connect(...) as conn`` **不会关闭连接**，
    只会在异常时回滚 —— 这是最常见的连接泄漏来源。

    本模块使用显式事务模式（``isolation_level=None``），因此退出时**不做隐式提交**：
    每条语句在无事务时本就即时生效。需要原子性请用 :func:`transaction`。
    """
    conn = get_connection(db_path, **kwargs)
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        close_connection(conn)


@contextmanager
def transaction(
    conn: sqlite3.Connection,
    mode: str = "IMMEDIATE",
    *,
    allow_nested: bool = False,
) -> Iterator[sqlite3.Connection]:
    """显式事务块：成功即提交，异常即回滚。

    ::

        with transaction(conn):
            conn.execute("INSERT INTO words(word) VALUES('新词')")
            conn.execute("INSERT INTO senses(word_id, pos, sense_id, sense_no, definition) "
                         "VALUES(last_insert_rowid(), 'n', 9999, 1, '测试')")

    默认 ``IMMEDIATE``：立刻取写锁，把「升级锁失败」提前到事务开头，
    避免 WAL 下两个写事务互相饿死（``SQLITE_BUSY_SNAPSHOT``）。

    **嵌套是不允许的** —— SQLite 会报 ``cannot start a transaction within
    a transaction``。本函数会提前拦截并给出明确错误；确实需要嵌套时传
    ``allow_nested=True``，内部会自动改用 SAVEPOINT。
    """
    mode_up = mode.upper()
    if mode_up not in ("DEFERRED", "IMMEDIATE", "EXCLUSIVE"):
        raise TransactionError(f"不合法的事务模式: {mode!r}（可选 DEFERRED/IMMEDIATE/EXCLUSIVE）")

    if conn.in_transaction:
        if not allow_nested:
            raise TransactionError(
                "检测到未闭合的事务，无法再开一个。SQLite 不支持嵌套事务。"
                "请先 commit()/rollback()，或显式使用 transaction(..., allow_nested=True) "
                "（内部走 SAVEPOINT）。",
                db_path=getattr(conn, "db_path", None),
            )
        with savepoint(conn) as sp:
            yield sp
        return

    conn.execute(f"BEGIN {mode_up}")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        if conn.in_transaction:
            conn.execute("COMMIT")
        else:
            # 触发器里写了 COMMIT/ROLLBACK 才会走到这里，属于设计外用法
            raise TransactionError("事务已被内部语句提前结束（事务块内不应出现 COMMIT/ROLLBACK）")


@contextmanager
def savepoint(conn: sqlite3.Connection, name: str | None = None) -> Iterator[sqlite3.Connection]:
    """SAVEPOINT 块 —— SQLite 里唯一正确的"嵌套事务"方式。

    外层事务回滚时，内层 SAVEPOINT 的提交会一并失效（这是与独立事务的本质区别）。
    """
    sp = name or f"sp_{int(time.time() * 1_000_000)}_{os.getpid()}"
    conn.execute(f"SAVEPOINT {sp}")
    try:
        yield conn
    except BaseException:
        conn.execute(f"ROLLBACK TO {sp}")
        conn.execute(f"RELEASE {sp}")
        raise
    else:
        conn.execute(f"RELEASE {sp}")


# ===========================================================================
# 6. 内存哈希缓存
# ===========================================================================
class MemoryCache(dict):
    """有容量上限与命中率统计的字典 —— 就是 ``dict`` 的子类。

    用于缓存「热数据」：词条义项、量词搭配、句式的槽位表等读多写少的查询结果。
    之所以要哈希键而不是直接用行 id 列表：连接层拿不到语义主键，
    调用方用 :func:`hash_key` 把任意参数拼成稳定键即可。

    .. warning::
       直接下标赋值 (``cache[k] = v``) 会走 :meth:`put` 并触发淘汰；
       但 ``dict.update()`` / ``dict.setdefault()`` 是 C 层实现，**不会**经过
       本类的钩子。需要淘汰语义时请显式调用 :meth:`put`。
    """

    def __init__(self, name: str = "hot", max_entries: int = 2048) -> None:
        super().__init__()
        self.name = name
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ---- 写 ---------------------------------------------------------------
    def put(self, key: Any, value: Any) -> Any:
        """写入并做 FIFO 淘汰（超出容量时丢弃最早插入的键）。"""
        with self._lock:
            if dict.__contains__(self, key) is False and dict.__len__(self) >= self.max_entries:
                oldest = next(iter(dict.keys(self)))
                dict.__delitem__(self, oldest)
                self.evictions += 1
            dict.__setitem__(self, key, value)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        self.put(key, value)

    def get_or_compute(self, key: Any, factory: Callable[[], Any]) -> Any:
        """命中即返回，未命中则调用 ``factory()`` 计算并写入。

        ``factory`` 在锁**外**执行：慢查询不应阻塞其他键的读取。
        代价是并发未命中可能重复计算（对本项目的查询而言可以接受）。
        """
        with self._lock:
            if dict.__contains__(self, key):
                self.hits += 1
                return dict.__getitem__(self, key)
            self.misses += 1
        value = factory()
        self.put(key, value)
        return value

    # ---- 失效与统计 -------------------------------------------------------
    def invalidate(self, key: Any = None) -> int:
        """删除单个键（``key=None`` 表示清空），返回被删条目数。"""
        with self._lock:
            if key is None:
                n = dict.__len__(self)
                dict.clear(self)
                return n
            if dict.__contains__(self, key):
                dict.__delitem__(self, key)
                return 1
            return 0

    def load(self, key: Any, factory: Callable[[], Any]) -> Any:
        """``get_or_compute`` 的别名，语义更符合"懒加载热数据"。"""
        return self.get_or_compute(key, factory)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "entries": dict.__len__(self),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": round(self.hit_rate, 4),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self.hits = self.misses = self.evictions = 0

    def snapshot(self) -> dict[Any, Any]:
        with self._lock:
            return dict(self)


_CACHES: dict[str, MemoryCache] = {}
_CACHES_LOCK = threading.RLock()
DEFAULT_CACHE_ENTRIES = 2048


def get_memory_cache(name: str = "hot", max_entries: int | None = None) -> MemoryCache:
    """取得（或创建）具名进程内缓存。

    返回 :class:`MemoryCache`，它是 ``dict`` 的子类，因此
    ``isinstance(get_memory_cache(), dict)`` 为真，可直接当 dict 用。

    ::

        cache = get_memory_cache("senses")
        key = hash_key("sense_detail", word, pos)
        detail = cache.get_or_compute(key, lambda: query_detail(conn, word, pos))
    """
    with _CACHES_LOCK:
        cache = _CACHES.get(name)
        if cache is None:
            cache = MemoryCache(name=name, max_entries=max_entries or DEFAULT_CACHE_ENTRIES)
            _CACHES[name] = cache
        return cache


def clear_caches(name: str | None = None) -> int:
    """清空指定缓存（``None`` = 全部），返回删除条目数。"""
    with _CACHES_LOCK:
        if name is not None:
            c = _CACHES.get(name)
            return c.invalidate() if c else 0
        return sum(c.invalidate() for c in _CACHES.values())


def hash_key(*parts: Any, size: int = 16) -> str:
    """把任意参数拼成稳定的短哈希键（blake2b，跨进程跨运行一致）。

    ``hash()`` 对 str 会被 PYTHONHASHSEED 随机化，**不能**用作持久缓存键，
    这里刻意避开它。
    """
    h = hashlib.blake2b(digest_size=size)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(b"B")
            h.update(bytes(p))
        else:
            h.update(b"S")
            h.update(
                json.dumps(p, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            )
        h.update(b"\x00")
    return h.hexdigest()


def file_digest(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    """文件内容指纹（sha256）。

    用途：把 schema/seed 的版本绑进缓存键 —— 脚本一改，旧缓存自动失效。
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ===========================================================================
# 7. 查询计时（性能分析）
# ===========================================================================
F = TypeVar("F", bound=Callable[..., Any])


class _Accumulator:
    """单个标签的计时累加器（保留最近 N 个样本以算分位数）。"""

    __slots__ = ("count", "total_ms", "min_ms", "max_ms", "last_ms", "samples")

    def __init__(self, window: int = 1024) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0
        self.last_ms = 0.0
        self.samples: deque[float] = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.min_ms = ms if ms < self.min_ms else self.min_ms
        self.max_ms = ms if ms > self.max_ms else self.max_ms
        self.last_ms = ms
        self.samples.append(ms)


_TIMING: dict[str, _Accumulator] = {}
_TIMING_LOCK = threading.RLock()


def _record(label: str, ms: float) -> None:
    with _TIMING_LOCK:
        acc = _TIMING.get(label)
        if acc is None:
            acc = _TIMING[label] = _Accumulator()
        acc.add(ms)


@dataclass
class TimingStat:
    """单个标签的统计结果。"""

    label: str
    count: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    last_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "last_ms": round(self.last_ms, 3),
        }


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def get_timing_stats() -> dict[str, TimingStat]:
    """取出全部计时统计（``label -> TimingStat``）。"""
    out: dict[str, TimingStat] = {}
    with _TIMING_LOCK:
        for label, acc in _TIMING.items():
            vals = sorted(acc.samples)
            out[label] = TimingStat(
                label=label,
                count=acc.count,
                total_ms=acc.total_ms,
                avg_ms=(acc.total_ms / acc.count) if acc.count else 0.0,
                min_ms=(0.0 if acc.min_ms == float("inf") else acc.min_ms),
                max_ms=acc.max_ms,
                p50_ms=_percentile(vals, 0.50),
                p95_ms=_percentile(vals, 0.95),
                last_ms=acc.last_ms,
            )
    return out


def reset_timing_stats() -> None:
    """清空计时统计。"""
    with _TIMING_LOCK:
        _TIMING.clear()


def dump_timing_stats(file: Any = None, *, sort_by: str = "total_ms") -> str:
    """把计时统计格式化成表格并打印，返回同一份文本。"""
    stream = file or sys.stderr
    stats = get_timing_stats()
    if not stats:
        text = "（无计时数据）"
        print(text, file=stream)
        return text
    rows = sorted(stats.values(), key=lambda s: getattr(s, sort_by), reverse=True)
    header = f"{'label':<28}{'n':>6}{'total_ms':>11}{'avg':>9}{'p50':>9}{'p95':>9}{'max':>9}"
    lines = [header, "-" * len(header)]
    for s in rows:
        lines.append(
            f"{s.label[:27]:<28}{s.count:>6}{s.total_ms:>11.2f}{s.avg_ms:>9.3f}"
            f"{s.p50_ms:>9.3f}{s.p95_ms:>9.3f}{s.max_ms:>9.3f}"
        )
    text = "\n".join(lines)
    print(text, file=stream)
    return text


def timed(arg: Any = None, *, label: str | None = None) -> Any:
    """计时装饰器。既可用 ``@timed``，也可用 ``@timed("标签")``。

    ::

        @timed("search_words")
        def search(conn, kw): ...

        @timed
        def rebuild_closure(conn): ...

    统计结果通过 :func:`get_timing_stats` / :func:`dump_timing_stats` 取出。
    异常也会被计时（异常路径的耗时同样是性能信息），随后原样抛出。
    """

    def decorate(fn: F, name: str | None = None) -> F:
        tag = name or label or getattr(fn, "__qualname__", getattr(fn, "__name__", "callable"))

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _record(tag, (time.perf_counter() - t0) * 1000.0)

        wrapper.timing_label = tag  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if callable(arg):
        return decorate(arg)
    if arg is not None and not isinstance(arg, str):
        raise TypeError("timed 只接受字符串标签或直接作为装饰器使用")
    return lambda fn: decorate(fn, arg)


@contextmanager
def timed_block(label: str) -> Iterator[None]:
    """给任意代码块计时（比如一次多语句批量查询）。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _record(label, (time.perf_counter() - t0) * 1000.0)


def execute_timed(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
    *,
    label: str | None = None,
    many: bool = False,
) -> tuple[list[Any], float]:
    """执行 SQL 并返回 ``(rows, 耗时毫秒)``，同时计入统计。

    ``many=True`` 时按 ``executemany`` 语义处理 ``params``（一批参数）。
    """
    tag = label or f"sql:{_snippet(sql, 48)}"
    t0 = time.perf_counter()
    try:
        if many:
            cur = conn.executemany(sql, params)  # type: ignore[arg-type]
            rows = []
        else:
            cur = conn.execute(sql, params)
            rows = list(cur.fetchall())
        return rows, (time.perf_counter() - t0) * 1000.0
    finally:
        _record(tag, (time.perf_counter() - t0) * 1000.0)


@contextmanager
def count_sql(
    conn: sqlite3.Connection, restore: Callable[[str], None] | None = None
) -> Iterator[dict[str, Any]]:
    """统计代码块内执行的 SQL 条数与文本（基于 ``set_trace_callback``）。

    ::

        with count_sql(conn) as info:
            do_something()
        print(info["count"], info["sql"])

    .. note::
       ``sqlite3`` **没有读取当前 trace callback 的接口**，因此本函数无法自动
       保留调用方原有的 tracer。若你先前装过自定义 tracer，请通过 ``restore``
       传进来，块结束时会重新装上；未传则会把 tracer 清空（``None``）。
    """
    info: dict[str, Any] = {"count": 0, "sql": []}

    def tracer(stmt: str) -> None:
        info["count"] += 1
        info["sql"].append(stmt)

    conn.set_trace_callback(tracer)
    try:
        yield info
    finally:
        conn.set_trace_callback(restore)


# ===========================================================================
# 8. SQL 脚本切分（触发器体感知）
# ===========================================================================
def _strip_leading_comments(text: str) -> str:
    """去掉语句文本开头的注释与空行，使其直接从关键字开始。"""
    s = text.lstrip()
    while s:
        if s.startswith("--"):
            nl = s.find("\n")
            if nl < 0:
                return ""
            s = s[nl + 1 :].lstrip()
            continue
        if s.startswith("/*"):
            end = s.find("*/")
            if end < 0:
                return s
            s = s[end + 2 :].lstrip()
            continue
        break
    return s


def split_statements(sql: str) -> list[tuple[str, int]]:
    """把 SQL 脚本切成 ``[(语句文本, 起始行号), ...]``。

    **为什么不用标准库的 ``sqlite3.complete_statement``**
    ----------------------------------------------------
    先纠正一个常见误解：``complete_statement`` 底层是 SQLite 自带的
    ``sqlite3_complete()``，它**能够**正确识别 ``CREATE TRIGGER ... BEGIN
    ... END;`` 的块结构（实测：简单触发器切为完整的 1 段；本库
    ``schema_v3.sql`` 的 10 个触发器全部完整、逐句执行 0 失败）。
    所以「它会被触发器体内的分号切断」的说法是不成立的。

    它对本模块仍不够用，真正的原因是三条**能力缺失**：

    1. **不返回行号** —— 而本模块的核心价值之一就是让加载失败报到
       ``schema_v3.sql:123`` 这种精度（见 :func:`_load_script`）。
    2. **按行累积** —— 同一行里的多条语句会被合并成一段。
       本库 ``schema_v3.sql`` 第 1642、1648 行正是这种情况。
    3. 会把**纯注释块**也当成一条"语句"返回（本库两个脚本各 1 处），
       需要调用方再过滤。

    本实现是字符级状态机，一次扫描直接产出「完整语句 + 精确行号」，
    并跳过纯注释块。正确处理：

    * 单引号字符串（含 ``''`` 转义）、双引号 / 反引号 / 方括号标识符
    * ``--`` 行注释、``/* */`` 块注释（其中的分号不算语句结束）
    * ``CREATE TRIGGER`` 的 ``BEGIN ... END`` 块，以及表达式里的 ``CASE ... END``

    ``BEGIN`` 有歧义（事务开始 vs 触发器体开头），按**上下文**区分：
    若 ``BEGIN`` 之前当前语句已有代码（如 ``CREATE TRIGGER ... AS``），
    它开启一个块；若它出现在语句最开头，则是事务开始语句。
    """
    out: list[tuple[str, int]] = []
    buf: list[str] = []
    depth = 0
    pending_code = False  # 当前语句是否已出现真实代码（注释/空白不算）
    start_line = 1
    line = 1
    i, n = 0, len(sql)

    while i < n:
        ch = sql[i]
        is_comment_start = (ch == "-" and sql.startswith("--", i)) or (
            ch == "/" and sql.startswith("/*", i)
        )
        # 首个真实代码字符决定语句起始行号；注释与空白不参与
        first_code = False
        if not pending_code and not is_comment_start and not ch.isspace():
            pending_code = True
            start_line = line
            first_code = True

        # ---- 行注释：原样保留；换行交给下面统一维护行号 ----
        if ch == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            if j < 0:
                j = n
            buf.append(sql[i:j])
            i = j
            continue

        # ---- 块注释：其中的分号必须被忽略 ----
        if ch == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            buf.append(sql[i:j])
            line += sql.count("\n", i, j)
            i = j
            continue

        # ---- 单引号字符串（SQLite 里 '' 表示转义的单引号）----
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            buf.append(sql[i:j])
            line += sql.count("\n", i, j)
            i = j
            continue

        # ---- 双引号 / 反引号标识符 ----
        if ch in ('"', "`"):
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            buf.append(sql[i:j])
            line += sql.count("\n", i, j)
            i = j
            continue

        # ---- 方括号标识符 [xxx] ----
        if ch == "[":
            j = sql.find("]", i)
            j = n if j < 0 else j + 1
            buf.append(sql[i:j])
            line += sql.count("\n", i, j)
            i = j
            continue

        # ---- 标识符/关键字：维护块深度 BEGIN / CASE / END ----
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            word = sql[i:j].upper()
            if word == "CASE":
                depth += 1
            elif word == "BEGIN":
                # 语句最开头的 BEGIN 是事务开始，不是块起始
                if pending_code and not first_code:
                    depth += 1
            elif word == "END" and depth > 0:
                depth -= 1
            buf.append(sql[i:j])
            i = j
            continue

        # ---- 语句终止符：仅在最外层生效 ----
        if ch == ";" and depth == 0:
            buf.append(ch)
            text = _strip_leading_comments("".join(buf))
            if pending_code and text:
                out.append((text, start_line))
            buf.clear()
            pending_code = False
            i += 1
            continue

        # ---- 普通字符 ----
        buf.append(ch)
        if ch == "\n":
            line += 1
        i += 1

    # 收尾：文件末尾未以分号结束的最后一条语句
    text = _strip_leading_comments("".join(buf))
    if pending_code and text:
        out.append((text, start_line))
    return out


_TXN_CONTROL_RE = re.compile(r"^(?:BEGIN|COMMIT|END|ROLLBACK)(?:\s+\w+)?$", re.I)


def _is_txn_control(stmt: str) -> bool:
    """判断语句是否为顶层事务控制（``BEGIN`` / ``COMMIT`` / ``ROLLBACK``）。"""
    return bool(_TXN_CONTROL_RE.match(stmt.strip().rstrip(";").strip()))


# ===========================================================================
# 9. 建库：schema + 种子
# ===========================================================================
@dataclass
class InitReport:
    """:func:`init_database` 的执行报告。"""

    db_path: str
    created: bool = False
    reset: bool = False
    schema_loaded: bool = False
    seed_loaded: list[str] = field(default_factory=list)
    statements: int = 0
    tables: int = 0
    views: int = 0
    triggers: int = 0
    indexes: int = 0
    rows: dict[str, int] = field(default_factory=dict)
    pragmas: dict[str, Any] = field(default_factory=dict)
    integrity_ok: bool = False
    foreign_key_ok: bool = False
    schema_fingerprint: str = ""
    seed_fingerprints: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.integrity_ok and self.foreign_key_ok and not self.messages

    def render(self) -> str:
        """人类可读的多行报告，便于脚本直接打印。"""
        lines = [
            f"数据库       : {self.db_path}",
            f"新建/重建    : created={self.created} reset={self.reset}",
            f"schema       : {'已加载' if self.schema_loaded else '跳过（已存在）'}"
            f"  指纹={self.schema_fingerprint[:12]}",
            f"种子         : {', '.join(self.seed_loaded) if self.seed_loaded else '无'}",
            f"执行语句数   : {self.statements}",
            f"对象         : 表={self.tables} 视图={self.views} 触发器={self.triggers} 索引={self.indexes}",
            f"完整性       : integrity={self.integrity_ok} foreign_key={self.foreign_key_ok}",
            f"耗时         : {self.elapsed_ms:.1f} ms",
        ]
        if self.rows:
            top = ", ".join(f"{k}={v}" for k, v in list(self.rows.items())[:12])
            lines.append(f"行数         : {top}")
        if self.messages:
            lines.append("提示         : " + " | ".join(self.messages))
        return "\n".join(lines)


#: 报告里默认统计行数的关键表（存在才统计，缺失自动跳过）
_REPORT_TABLES = (
    "words",
    "senses",
    "relations",
    "closure",
    "function_words",
    "classifiers",
    "noun_classifier_pairs",
    "relation_constraints",
    "pattern_slots",
    "slot_constraints",
    "sentence_patterns",
    "utterances",
)


def _load_script(conn: LogosConnection, path: pathlib.Path, *, kind: str) -> int:
    """逐句加载一个 SQL 脚本，失败时报出**具体行号**。

    * 脚本自带顶层 ``BEGIN``/``COMMIT``（如 ``seed_general_v3.sql``）→ 原样执行，
      由脚本自己保证原子性。
    * 脚本没有事务控制（如 ``schema_v3.sql``）→ 由本函数**包一个事务**：
      既让 DDL 具备原子性（失败整库不留半成品），又把 150 次逐句提交
      合并成 1 次提交（实测文件库上约 38 倍加速）。

    无论哪条路径，失败时都会回滚并抛 :class:`SchemaError` / :class:`SeedError`。
    """
    exc_type = SchemaError if kind == "schema" else SeedError
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise exc_type(f"读取{kind}文件失败：{path}", db_path=conn.db_path, cause=exc) from exc

    try:
        stmts = split_statements(text)
    except Exception as exc:  # 切分器自身异常也要包装成领域异常
        raise exc_type(f"解析{kind}脚本失败：{path}", db_path=conn.db_path, cause=exc) from exc

    if not stmts:
        raise exc_type(f"{kind}脚本不含任何语句：{path}", db_path=conn.db_path)

    self_managed = any(_is_txn_control(s) for s, _ in stmts)
    wrapped = not self_managed
    if wrapped:
        conn.execute("BEGIN")

    executed = 0
    try:
        for ordinal, (stmt, line_no) in enumerate(stmts, 1):
            try:
                conn.execute(stmt)
                executed += 1
            except sqlite3.Error as exc:
                raise exc_type(
                    f"{path.name}:{line_no} 第 {ordinal}/{len(stmts)} 条语句执行失败：{exc}",
                    db_path=conn.db_path,
                    sql=stmt,
                    cause=exc,
                ) from exc
        if wrapped:
            conn.execute("COMMIT")
    except BaseException:
        # executescript 出错会留下未闭合事务；这里必须显式回滚，否则连接被"污染"
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise

    log.info("%s: %s 共 %d 条语句（事务%s）", kind, path.name, executed,
             "内嵌" if self_managed else "由连接层包裹")
    return executed


def init_database(
    db_path: str | os.PathLike[str] | None = None,
    schema_sql_path: str | os.PathLike[str] | None = None,
    seed_sql_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    *,
    profile: PragmaProfile | None = None,
    reset: bool = False,
    validate: bool = True,
    defer_foreign_keys: bool | None = None,
    verbose: bool = False,
) -> InitReport:
    """建库 + 填种子，返回 :class:`InitReport`。

    行为矩阵
    --------
    ==================  ==============================  ==========================
    目标库状态            ``reset=False``（默认）          ``reset=True``
    ==================  ==============================  ==========================
    文件不存在            建库 + 加载 schema + 种子        同左
    已存在且已建表        跳过 schema（schema 不幂等），    删除文件后完整重建
                        仅**重跑种子**（种子是幂等的）
    ==================  ==============================  ==========================

    之所以要区分：``schema_v3.sql`` 的 52 个 ``CREATE TABLE`` **没有**
    ``IF NOT EXISTS``，重复执行必然报错；而 ``seed_general_v3.sql``
    通篇 ``ON CONFLICT DO NOTHING``，可安全反复执行。

    :param db_path: 目标库路径，默认 :data:`DEFAULT_DB_PATH`。
    :param schema_sql_path: schema 文件或所在目录；默认项目根的 ``schema_v3.sql``。
    :param seed_sql_paths: 种子文件（单个/多个）；``None`` = 自动发现
        项目根的 ``seed_*.sql`` 并**按文件名排序**执行（保证种子之间的确定性顺序）。
    :param reset: 删除已有库文件后完整重建。
        **安全护栏**：仅当文件确有 ``SQLite format 3`` 魔数或带 ``.db`` 后缀时才删除，
        绝不误删其他文件。
    :param validate: 建库后跑 ``integrity_check`` + ``foreign_key_check``。
    :param defer_foreign_keys: 覆盖档案设置。批量导入设为 ``True`` 可提速，
        代价是 FK 错误在 COMMIT 时才报、拿不到具体行。

    .. note::
       建库过程中 ``PRAGMA foreign_keys`` 由连接层在**事务外**设置并读回校验；
       脚本内若出现同名的 ``PRAGMA``（``schema_v3.sql`` 第 38 行就有）在事务内是
       无副作用的 no-op，不会把外键关掉。加载结束后本函数还会再校验一次。
    """
    t_start = time.perf_counter()

    # ---- 解析路径 ----
    db_path_obj = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    schema_path = _resolve_schema_path(schema_sql_path)
    seed_paths = _resolve_seed_paths(seed_sql_paths, schema_path.parent)

    report = InitReport(
        db_path=str(db_path_obj),
        reset=bool(reset),
        schema_fingerprint=file_digest(schema_path) if schema_path.exists() else "",
        seed_fingerprints={p.name: file_digest(p) for p in seed_paths if p.exists()},
    )

    # ---- 可选重建 ----
    if reset:
        removed = _remove_db_files(db_path_obj)
        report.messages.append(f"已删除旧库文件 {len(removed)} 个" if removed else "无可删除的旧库")
    report.created = not db_path_obj.exists()

    # ---- 建连（PRAGMA 已应用并校验）----
    conn = get_connection(
        db_path_obj,
        profile=profile,
        must_exist=False,
        create_parents=True,
        strict_pragmas=True,
    )
    try:
        if defer_foreign_keys is not None:
            _set_pragma(conn, "defer_foreign_keys", 1 if defer_foreign_keys else 0)

        already_built = conn.table_exists("schema_meta") or bool(
            [t for t in conn.tables() if t != "sqlite_sequence"]
        )

        # ---- schema ----
        if already_built and not reset:
            report.messages.append("目标库已建表：跳过 schema（schema 不幂等），仅重跑种子")
            log.info("跳过 schema 加载（库已存在）")
        else:
            if not schema_path.exists():
                raise SchemaError(f"schema 文件不存在：{schema_path}", db_path=db_path_obj)
            report.statements += _load_script(conn, schema_path, kind="schema")
            report.schema_loaded = True

        # 事务外再确认一次外键状态（脚本内的 PRAGMA 可能落在事务里成为 no-op）
        _set_pragma(conn, "foreign_keys", 1)
        fk_now = _read_pragma(conn, "foreign_keys")
        if fk_now != 1:
            raise DatabaseError(
                f"加载 schema 后 foreign_keys 读回={fk_now!r}，外键未生效，拒绝继续填种子",
                db_path=db_path_obj,
            )

        # ---- 种子（幂等，可反复执行）----
        if defer_foreign_keys:
            _set_pragma(conn, "defer_foreign_keys", 1)
        for sp in seed_paths:
            if not sp.exists():
                report.messages.append(f"种子文件缺失，已跳过：{sp.name}")
                continue
            report.statements += _load_script(conn, sp, kind="seed")
            report.seed_loaded.append(sp.name)

        # ---- 校验 ----
        if validate:
            report.integrity_ok = conn.integrity_ok()
            report.foreign_key_ok = conn.fk_ok()
            if not report.integrity_ok:
                raise IntegrityCheckError("PRAGMA integrity_check 未通过", db_path=db_path_obj)
            if not report.foreign_key_ok:
                viol = conn.foreign_key_violations()
                raise IntegrityCheckError(
                    f"PRAGMA foreign_key_check 发现 {len(viol)} 条悬空外键，前 5 条：{viol[:5]}",
                    db_path=db_path_obj,
                )
        else:
            report.integrity_ok = report.foreign_key_ok = True

        # ---- 统计 ----
        master = conn.execute(
            "SELECT type, COUNT(*) FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' GROUP BY type"
        ).fetchall()
        by_type = {r[0]: r[1] for r in master}
        report.tables = by_type.get("table", 0)
        report.views = by_type.get("view", 0)
        report.triggers = by_type.get("trigger", 0)
        report.indexes = by_type.get("index", 0)
        report.rows = conn.counts(_REPORT_TABLES)
        report.pragmas = conn.pragmas
    finally:
        report.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        close_connection(conn)

    if verbose:
        print(report.render())
    return report


def _resolve_schema_path(schema_sql_path: str | os.PathLike[str] | None) -> pathlib.Path:
    if schema_sql_path is None:
        return PROJECT_ROOT / SCHEMA_FILENAME
    p = pathlib.Path(schema_sql_path)
    if p.is_dir():
        return p / SCHEMA_FILENAME
    return p


def _resolve_seed_paths(
    seed_sql_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None,
    search_root: pathlib.Path,
) -> list[pathlib.Path]:
    """规整种子路径；``None`` 时按文件名排序自动发现 ``seed_*.sql``。"""
    if seed_sql_paths is None:
        if not search_root.exists():
            return []
        return sorted(search_root.glob(SEED_GLOB))
    if isinstance(seed_sql_paths, (str, os.PathLike)):
        return [pathlib.Path(seed_sql_paths)]
    return sorted(pathlib.Path(p) for p in seed_sql_paths)


_SQLITE_MAGIC = b"SQLite format 3\x00"


def _remove_db_files(db_path: pathlib.Path) -> list[str]:
    """安全删除数据库文件及其 ``-wal`` / ``-shm`` 伴随文件。

    护栏：目标必须是 ``.db`` 结尾，**或**文件头是 SQLite 魔数。
    两个条件都不满足就拒绝删除 —— 避免 ``reset=True`` 误删无关文件。
    """
    removed: list[str] = []
    targets = [db_path, pathlib.Path(str(db_path) + "-wal"), pathlib.Path(str(db_path) + "-shm")]
    for t in targets:
        if not t.exists():
            continue
        looks_like_db = t.suffix.lower() == ".db"
        if not looks_like_db:
            try:
                with open(t, "rb") as fh:
                    looks_like_db = fh.read(16) == _SQLITE_MAGIC
            except OSError:
                looks_like_db = False
        if not looks_like_db:
            log.warning("拒绝删除非数据库文件：%s", t)
            continue
        t.unlink()
        removed.append(t.name)
    return removed


# ===========================================================================
# 10. 关闭
# ===========================================================================
def close_connection(conn: sqlite3.Connection | None, *, commit: bool = False) -> bool:
    """关闭连接，并做收尾优化。返回是否真的关掉了。

    收尾动作：

    1. **未闭合事务**：默认 ``rollback()`` 并记一条 WARNING，而不是悄悄提交。
       看到这条日志通常意味着某处漏了 ``commit()`` —— 那是 bug，不该被自动提交掩盖。
       确实想提交时显式传 ``commit=True``。
    2. ``PRAGMA optimize``：SQLite 官方建议关闭前执行，让查询规划器把
       统计信息写回。仅对可写、非内存、无事务的连接执行，失败静默忽略。
    3. 从活连接登记表移除。

    .. note::
       本函数对**任何** ``sqlite3.Connection`` 都安全（包括不是本模块创建的），
       缺少元数据时会退化为"只回滚 + 关闭"。
    """
    if conn is None:
        return False

    meta = getattr(conn, "logos_meta", None)
    if isinstance(meta, dict) and meta.get("closed"):
        return False

    try:
        if conn.in_transaction:
            if commit:
                conn.commit()
            else:
                conn.rollback()
                log.warning(
                    "关闭连接时发现未闭合事务，已回滚：%s（若本应提交，请显式 commit）",
                    (meta or {}).get("db_path", "?"),
                )

        read_only = bool((meta or {}).get("read_only"))
        is_memory = bool((meta or {}).get("memory"))
        if not read_only and not is_memory and not conn.in_transaction:
            try:
                conn.execute("PRAGMA optimize").fetchall()
            except sqlite3.Error as exc:  # optimize 失败不影响关闭
                log.debug("PRAGMA optimize 失败（忽略）：%s", exc)
    finally:
        try:
            conn.close()
        finally:
            if isinstance(meta, dict):
                meta["closed"] = True
            with _LIVE_LOCK:
                _LIVE.discard(conn)  # type: ignore[arg-type]
    return True


def close_all(*, commit: bool = False) -> int:
    """关闭本模块创建且仍存活的全部连接，返回关闭数量。

    主要在测试收尾/进程退出路径上使用。若返回值 > 0，说明调用方多半有连接泄漏。
    """
    open_conns = live_connections()
    n = 0
    for c in open_conns:
        if close_connection(c, commit=commit):
            n += 1
    return n


# ===========================================================================
# 11. 冒烟自检（python -m logos.db.connection）
# ===========================================================================
def _smoke_test() -> int:  # pragma: no cover - 手工运行
    """打印当前环境的连接层自检结果，供快速排查。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"python        : {sys.version.split()[0]}")
    print(f"sqlite        : {sqlite3.sqlite_version}")
    print(f"threadsafety  : {sqlite3.threadsafety}")
    print(f"项目根        : {PROJECT_ROOT}")
    print(f"默认库路径    : {DEFAULT_DB_PATH}")
    print(f"默认档案      : {DEFAULT_PROFILE}")
    print(f"schema 存在   : {(PROJECT_ROOT / SCHEMA_FILENAME).exists()}")

    conn = get_connection(":memory:")
    try:
        print("内存库 PRAGMA :")
        for k, v in conn.pragmas.items():
            print(f"    {k:<22} = {v}")
        print(f"  SQL 切分能力 : {len(split_statements('CREATE TRIGGER t AFTER INSERT ON a BEGIN INSERT INTO b VALUES(1); INSERT INTO c VALUES(2); END;'))} 条（期望 1）")
    finally:
        close_connection(conn)
    print(f"存活连接      : {len(live_connections())}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_smoke_test())
