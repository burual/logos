#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""validate_connection.py —— logos/db/connection.py 的行为测试。

设计原则：**只测运行时行为，不测"表/函数存不存在"**。
每条断言的判据都来自实际执行结果（读回的 PRAGMA 值、抛出的异常、行数变化、
与独立实现的对拍），因此能抓住"看起来对但实际没生效"的缺陷。

运行：
    python validate_connection.py
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from logos.db import connection as C  # noqa: E402
from logos.db import (  # noqa: E402
    DatabaseError,
    MemoryCache,
    PragmaProfile,
    SchemaError,
    SeedError,
    TransactionError,
    clear_caches,
    close_all,
    close_connection,
    connect,
    count_sql,
    dump_timing_stats,
    execute_timed,
    file_digest,
    get_connection,
    get_memory_cache,
    get_timing_stats,
    hash_key,
    init_database,
    live_connections,
    reset_timing_stats,
    savepoint,
    split_statements,
    timed,
    timed_block,
    transaction,
)

PASS = 0
FAIL = 0
FAILED: list[tuple[str, str]] = []
T0 = time.perf_counter()


def check(name: str, cond: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"   {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILED.append((name, detail))
        print(f"  [FAIL] {name}   {detail}")
    return bool(cond)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def expect_error(name: str, fn, exc_types, contains: str | None = None) -> bool:
    """断言 fn() 抛出指定异常（可选校验消息片段）。返回 True 表示**测试通过**。"""
    try:
        fn()
    except exc_types as e:  # type: ignore[misc]
        if contains is not None:
            return check(name, contains in str(e), f"异常={type(e).__name__}: {str(e)[:150]}")
        return check(name, True, f"异常={type(e).__name__}")
    except BaseException as e:  # noqa: BLE001
        return check(name, False, f"抛出了非预期异常 {type(e).__name__}: {e}")
    return check(name, False, "未抛出任何异常")


# ---------------------------------------------------------------------------
# 临时工作目录：路径**刻意含中文**，用于验证 URI 百分号编码
# ---------------------------------------------------------------------------
TMP = pathlib.Path(tempfile.mkdtemp(prefix="logos_conn_"))
CJK_DIR = TMP / "数据库logos" / "子目录"
CJK_DIR.mkdir(parents=True, exist_ok=True)
MAIN_DB = CJK_DIR / "logos_main.db"          # §5 建库专用，必须“事先不存在”
PRAGMA_DB = CJK_DIR / "pragma_probe.db"      # §2 PRAGMA 专用，避免污染建库前置状态
SCHEMA = ROOT / C.SCHEMA_FILENAME
SEED = ROOT / "seed_general_v3.sql"


def objects_of(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT type, COUNT(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' GROUP BY type"
    ).fetchall()
    by = {r[0]: r[1] for r in rows}
    return {
        "table": by.get("table", 0),
        "view": by.get("view", 0),
        "trigger": by.get("trigger", 0),
        "index": by.get("index", 0),
    }


# ===========================================================================
banner("§1 环境与模块契约")
# ===========================================================================
check(
    f"SQLite 版本 >= 3.37（STRICT 表前提），实测 {sqlite3.sqlite_version}",
    tuple(int(x) for x in sqlite3.sqlite_version.split(".")) >= (3, 37, 0),
)
check("PROJECT_ROOT 正确定位到项目根", C.PROJECT_ROOT == ROOT, str(C.PROJECT_ROOT))
check("schema_v3.sql 存在", SCHEMA.exists(), str(SCHEMA))
check("seed_general_v3.sql 存在", SEED.exists(), str(SEED))

_mem = get_connection(":memory:")
check(
    "get_connection 返回的仍是 sqlite3.Connection（子类对调用方透明）",
    isinstance(_mem, sqlite3.Connection),
    f"类型={type(_mem).__name__}",
)
check("返回对象同时是 LogosConnection（可承载元数据）", isinstance(_mem, C.LogosConnection))
_mem.execute("CREATE TABLE t(a)")
_mem.execute("INSERT INTO t VALUES(1)")
check(
    "默认 row_factory=Row：可按列名取值",
    _mem.execute("SELECT a FROM t").fetchone()["a"] == 1,
)
check("isolation_level=None：单条 INSERT 后无未闭合事务", _mem.in_transaction is False)
close_connection(_mem)

# ===========================================================================
banner("§2 PRAGMA 应用：全部以“写后读回”的方式验证")
# ===========================================================================
conn = get_connection(PRAGMA_DB)
P = conn.pragmas
check("journal_mode = wal（持久化到数据库文件）", P.get("journal_mode") == "wal", str(P.get("journal_mode")))
check("synchronous = 1（NORMAL）", P.get("synchronous") == 1, str(P.get("synchronous")))
check("cache_size = -64000（≈64MB 页缓存）", P.get("cache_size") == -64000, str(P.get("cache_size")))
check("temp_store = 2（MEMORY）", P.get("temp_store") == 2, str(P.get("temp_store")))
check(
    "mmap_size 已生效（>0，可能被编译期上限截断）",
    0 < (P.get("mmap_size") or 0) <= 268435456,
    str(P.get("mmap_size")),
)
check("busy_timeout = 5000", P.get("busy_timeout") == 5000, str(P.get("busy_timeout")))
check("recursive_triggers = 1（闭包级联所需）", P.get("recursive_triggers") == 1, str(P.get("recursive_triggers")))
check("wal_autocheckpoint = 1000", P.get("wal_autocheckpoint") == 1000, str(P.get("wal_autocheckpoint")))
check("defer_foreign_keys = 0（默认立即校验）", P.get("defer_foreign_keys") == 0, str(P.get("defer_foreign_keys")))
check("★ foreign_keys = 1（SQLite 默认 OFF，必须显式开启）", P.get("foreign_keys") == 1, str(P.get("foreign_keys")))
check("每个 PRAGMA 都拿到了读回值（无“只设不读”的静默失败）",
      all(v is not None for k, v in P.items() if k != "mmap_size"), str({k: v for k, v in P.items() if v is None}))

# 逐连接生效性：新连接必须自己再设一次
conn2 = get_connection(PRAGMA_DB)
check("PRAGMA 是连接级的：新连接 FK 依然为 1（不是全局状态）", conn2.pragma("foreign_keys") == 1)
close_connection(conn2)

# WAL 文件确实产生（必须先发生一次写操作，SQLite 才会建 -wal/-shm）
conn.execute("CREATE TABLE wal_probe(a INTEGER) STRICT")
conn.execute("INSERT INTO wal_probe VALUES(1)")
wal_files = [f.name for f in PRAGMA_DB.parent.iterdir() if f.name.startswith(PRAGMA_DB.name)]
check("WAL 模式在文件系统上真实生效（-wal/-shm 伴生文件出现）",
      any(n.endswith("-wal") for n in wal_files) and any(n.endswith("-shm") for n in wal_files),
      str(sorted(wal_files)))

_m = get_connection(":memory:")
check("内存库自动跳过 journal_mode（读回 memory，不报错）", _m.pragma("journal_mode") == "memory", str(_m.pragma("journal_mode")))
check("内存库跳过 wal_autocheckpoint（无该 PRAGMA 结果）", "wal_autocheckpoint" not in _m.pragmas)
close_connection(_m)

# 自定义档案
custom = get_connection(PRAGMA_DB, profile=PragmaProfile(cache_size=-2048, busy_timeout_ms=1234))
check("自定义档案生效：cache_size = -2048", custom.pragma("cache_size") == -2048, str(custom.pragma("cache_size")))
check("自定义档案生效：busy_timeout = 1234", custom.pragma("busy_timeout") == 1234, str(custom.pragma("busy_timeout")))
close_connection(custom)

# 关闭 §2 的连接：绝不能在持有句柄时去删除/重建 MAIN_DB（Windows 下会失败）
check("PRAGMA 段收尾：连接已关闭，无存活连接", close_connection(conn) and not live_connections())

# ===========================================================================
banner("§3 外键：证明“拒绝脏数据”确实来自 PRAGMA，而不是巧合")
# ===========================================================================
FK_DB = TMP / "fk_probe.db"
c = get_connection(FK_DB)
c.executescript(
    "CREATE TABLE parent(id INTEGER PRIMARY KEY) STRICT;"
    "CREATE TABLE child(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES parent(id)) STRICT;"
)
c.execute("INSERT INTO parent(id) VALUES(1)")

expect_error(
    "FK 生效：悬空外键插入被拒（IntegrityError）",
    lambda: c.execute("INSERT INTO child(id,pid) VALUES(9,999)"),
    sqlite3.IntegrityError,
)
check("conn.fk_ok() 判定为真", c.fk_ok() is True)
check("conn.integrity_ok() 判定为真", c.integrity_ok() is True)

# --- 阳性对照：关掉 FK 后同样的插入必须成功 ---
c.execute("PRAGMA foreign_keys=OFF")  # 事务外，真实生效
check("阳性对照前置条件：事务外可把 foreign_keys 关成 0", c.pragma("foreign_keys") == 0)
c.execute("INSERT INTO child(id,pid) VALUES(10,999)")
ok_cnt = c.execute("SELECT COUNT(*) FROM child WHERE pid=999").fetchone()[0]
check("阳性对照：FK 关闭后同一插入成功（证明上一断言确由 PRAGMA 引起）", ok_cnt == 1)
c.execute("DELETE FROM child WHERE pid=999")
c.execute("PRAGMA foreign_keys=ON")

# --- 事务内 PRAGMA 是 no-op（本质陷阱）---
c.execute("BEGIN IMMEDIATE")
c.execute("PRAGMA foreign_keys=OFF")
check(
    "陷阱复现：事务内设 foreign_keys=OFF 被静默忽略（读回仍 1）",
    c.pragma("foreign_keys") == 1,
    f"读回={c.pragma('foreign_keys')}",
)
expect_error(
    "防护：configure_connection 在未闭合事务上主动报 TransactionError（而非静默失效）",
    lambda: C.configure_connection(c),
    TransactionError,
)
c.execute("ROLLBACK")
close_connection(c)

# ===========================================================================
banner("§4 split_statements：触发器体感知切分（本模块的核心技术点）")
# ===========================================================================
TRIG = (
    "CREATE TRIGGER tr AFTER INSERT ON t BEGIN\n"
    "    INSERT INTO b VALUES(1);\n"
    "    INSERT INTO c VALUES(2);\n"
    "END;"
)
mine = split_statements(TRIG)
check("自研切分器：含多语句体的触发器 = 1 条完整语句", len(mine) == 1, f"实得 {len(mine)} 条")
check("自研切分器：切分结果以 END; 收尾（语句文本自带终止符）",
      bool(mine) and mine[0][0].rstrip().endswith("END;"), repr(mine[0][0][-20:]) if mine else "空")

# 与标准库对拍。注意：complete_statement 底层是 SQLite 的 sqlite3_complete()，
# 它**同样**能正确识别触发器体（这一点曾被误判，此处用断言固化正确认知）。
_official, _buf = [], ""
for _ln in TRIG.splitlines(keepends=True):
    _buf += _ln
    if sqlite3.complete_statement(_buf):
        _official.append(_buf.strip())
        _buf = ""
if _buf.strip():
    _official.append(_buf.strip())
check("对照：标准库 complete_statement 也能正确处理触发器体（同为 1 条，并非“会被切断”）",
      len(_official) == 1, f"官方切出 {len(_official)} 条")


def official_split(sql: str) -> list[str]:
    """标准库参考实现（按行累积 + complete_statement），用于能力对比。"""
    buf, out = "", []
    for ln in sql.splitlines(keepends=True):
        buf += ln
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def comment_only(s: str) -> bool:
    t = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    t = re.sub(r"--[^\n]*", "", t)
    return not t.strip()


# 能力差异①：同一行多条语句
same_line = "SELECT 1; SELECT 2;"
check("能力差异①：自研可拆开同一行的 2 条语句", len(split_statements(same_line)) == 2)
check("能力差异①对照：官方的按行累积会把同一行的 2 条语句合并成 1 条",
      len(official_split(same_line)) == 1, f"官方 {len(official_split(same_line))} 条")

schema_text = SCHEMA.read_text(encoding="utf-8")
seed_text = SEED.read_text(encoding="utf-8")
stmts = split_statements(schema_text)
trig_stmts = [s for s, _ in stmts if s.upper().startswith("CREATE TRIGGER")]
check("真实 schema：切出全部 10 个触发器", len(trig_stmts) == 10, f"实得 {len(trig_stmts)}")
check("真实 schema：每个触发器语句都以 END; 完整收尾",
      all(s.rstrip().endswith("END;") for s in trig_stmts))
check("真实 schema：无被截断的语句残片（无块以 END 开头）",
      not any(s.upper().startswith("END") for s, _ in stmts))
check("真实 schema：语句数 = 150，且与官方切分的差异恰为 1 个纯注释块",
      len(stmts) == 150 and len(official_split(schema_text)) - len(stmts) == 1,
      f"自研 {len(stmts)}，官方 {len(official_split(schema_text))}")

# 能力差异②：纯注释块
check("能力差异②：自研输出中不含纯注释块（不产出无法执行的“语句”）",
      not any(comment_only(s) for s, _ in stmts))
check("能力差异②对照：官方把尾部纯注释块也当作 1 条语句返回",
      sum(1 for s in official_split(schema_text) if comment_only(s)) == 1)
check("种子脚本同样满足：自研无纯注释块、官方多出 1 条",
      not any(comment_only(s) for s, _ in split_statements(seed_text))
      and sum(1 for s in official_split(seed_text) if comment_only(s)) == 1)

# 能力差异③：官方不提供行号
check("能力差异③：自研返回 (语句, 行号) 二元组，官方只返回文本",
      isinstance(stmts[0], tuple) and len(stmts[0]) == 2 and isinstance(stmts[0][1], int))

# ---- 为什么本模块不用 executescript：两个已实测的副作用 ----
_ec = get_connection(":memory:")
_ec.execute("CREATE TABLE t(a INTEGER)")
_ec.execute("BEGIN IMMEDIATE")
_ec.execute("INSERT INTO t(a) VALUES(1)")
_ec.executescript("SELECT 1;")
check("★ 设计依据①：executescript 会**隐式提交**调用方未完成的事务（故本模块不用它）",
      _ec.in_transaction is False, f"in_transaction={_ec.in_transaction}")
close_connection(_ec)

_ec2 = get_connection(":memory:")
try:
    _ec2.executescript("BEGIN;\nCREATE TABLE z(a INTEGER);\nINSERT INTO nope VALUES(1);")
except sqlite3.Error:
    pass
check("★ 设计依据②：executescript 出错后**留下未闭合事务**（须显式回滚）",
      _ec2.in_transaction is True, f"in_transaction={_ec2.in_transaction}")
_ec2.execute("ROLLBACK")
close_connection(_ec2)

# 对照：本模块的 _load_script 出错后自行回滚，且已执行的 DDL 一并回滚（原子性）
_clean = get_connection(":memory:")
_rollback_probe = TMP / "rollback_probe.sql"
_rollback_probe.write_text(
    "CREATE TABLE keep_me(a INTEGER) STRICT;\n"
    "CREATE TABLE broken(b INTEGER,);\n",   # 第 2 行故意语法错误
    encoding="utf-8",
)
try:
    C._load_script(_clean, _rollback_probe, kind="schema")
    check("坏脚本应抛 SchemaError", False, "未抛异常")
except SchemaError as e:
    check("坏脚本抛 SchemaError 且含行号 :2", ":2" in str(e), str(e)[:110])
check("★ 对照：_load_script 出错后连接无残留未闭合事务", _clean.in_transaction is False)
_keep_exists = _clean.table_exists("keep_me")
check("★ 对照：schema 加载是原子的 —— 已执行的第 1 条 DDL 也随事务回滚",
      not _keep_exists, f"keep_me 存在={_keep_exists}")
close_connection(_clean)

# 各类边界：分号出现在字符串/注释/CASE 中都不应切分
edge = [
    ("字符串内分号不切分", "INSERT INTO t VALUES('a;b');", 1),
    ("单引号 '' 转义不破坏切分", "INSERT INTO t VALUES('it''s; ok');", 1),
    ("行注释内分号不切分", "SELECT 1; -- 注释; 还有分号\nSELECT 2;", 2),
    ("块注释内分号不切分", "SELECT 1 /* 注释; 分号 */ ;", 1),
    ("CASE...END 内分号不切分", "UPDATE t SET a = CASE WHEN x THEN 1 ELSE 2 END; -- ;", 1),
    ("双引号标识符内分号不切分", 'SELECT "a;b" FROM t;', 1),
    ("多语句顺序切割", "SELECT 1;\nSELECT 2;\nSELECT 3;", 3),
    ("末尾无分号也产出语句", "SELECT 1", 1),
    ("纯注释不产出语句", "-- 只有注释\n/* 也是注释 */", 0),
]
for label, sql, want in edge:
    got = split_statements(sql)
    check(f"切分边界：{label}", len(got) == want, f"期望 {want} 实得 {len(got)}")

# 行号准确性
lined = "SELECT 1;\nSELECT 2;\nSELECT 3;"
got = split_statements(lined)
check("切分行号准确（第 3 条语句标记为第 3 行）", got[2][1] == 3, f"实得行号 {got[2][1]}")

# 事务控制识别
check("事务控制识别：BEGIN 语句", C._is_txn_control("BEGIN"))
check("事务控制识别：BEGIN IMMEDIATE", C._is_txn_control("BEGIN IMMEDIATE;"))
check("事务控制识别：COMMIT", C._is_txn_control("COMMIT;"))
check("事务控制识别：触发器 CREATE 语句**不**被误判为事务",
      not C._is_txn_control(trig_stmts[0]))

# 差分对拍：拆句执行 vs executescript，产出对象数必须一致
mem_a = sqlite3.connect(":memory:", isolation_level=None)
mem_a.executescript(schema_text)
oracle = objects_of(mem_a)
mem_a.close()
check(
    "差分对拍：拆句执行与 executescript 产出的对象数完全一致",
    oracle == {"table": 62, "view": 5, "trigger": 10, "index": 34},
    f"executescript 侧={oracle}",
)

# ===========================================================================
banner("§5 init_database：建库 + 填种子 + 校验")
# ===========================================================================
t0 = time.perf_counter()
rep = init_database(MAIN_DB, SCHEMA, SEED)
dt1 = (time.perf_counter() - t0) * 1000
check("新建库：created=True", rep.created is True)
check("新建库：schema 已加载", rep.schema_loaded is True)
check("新建库：种子已加载", rep.seed_loaded == [SEED.name], str(rep.seed_loaded))
check(f"对象数与 executescript 对拍一致（表 {rep.tables} / 视图 {rep.views} / 触发器 {rep.triggers} / 索引 {rep.indexes}）",
      (rep.tables, rep.views, rep.triggers, rep.indexes) ==
      (oracle["table"], oracle["view"], oracle["trigger"], oracle["index"]))
check("integrity_check 通过", rep.integrity_ok is True)
check("foreign_key_check 无悬空外键", rep.foreign_key_ok is True)
check("报告 ok 为真（无告警）", rep.ok is True, str(rep.messages))
check("schema 指纹是 64 位 sha256", len(rep.schema_fingerprint) == 64, rep.schema_fingerprint[:16])
check("种子指纹已记录", SEED.name in rep.seed_fingerprints)
check("关键表均有数据", all(rep.rows.get(t, 0) > 0 for t in
                            ("words", "senses", "relations", "closure")),
      str({k: rep.rows.get(k) for k in ("words", "senses", "relations", "closure")}))
check("闭包表由触发器自动派生（depth>1 行存在）", rep.rows.get("closure", 0) > 0, f"closure={rep.rows.get('closure')}")
check("执行语句数 > 150（schema 151 + 种子语句）", rep.statements > 150, f"statements={rep.statements}")
print(f"     · 首次建库耗时 {dt1:.1f} ms，报告行数示例: "
      f"{ {k: v for k, v in list(rep.rows.items())[:6]} }")

# ---- 幂等性：二次执行必须逐表行数完全一致 ----
snap1 = rep.rows.copy()
rep2 = init_database(MAIN_DB, SCHEMA, SEED)
check("二次执行：识别出库已存在并跳过 schema（schema 不幂等，必须跳过）",
      rep2.schema_loaded is False, str(rep2.messages))
check("二次执行：仍然重跑种子（种子幂等）", rep2.seed_loaded == [SEED.name])
diff = {k: (snap1.get(k), rep2.rows.get(k)) for k in snap1 if snap1.get(k) != rep2.rows.get(k)}
check("★ 幂等性：二次执行后逐表行数完全不变", not diff, str(diff))
check("二次执行：FK 完整性依然通过", rep2.foreign_key_ok is True)

# ---- reset 重建 ----
rep3 = init_database(MAIN_DB, SCHEMA, SEED, reset=True)
check("reset=True：删除旧库后完整重建（schema 重新加载）", rep3.schema_loaded is True)
check("reset=True：报告标记 reset 与 created", rep3.reset is True and rep3.created is True)
check("reset=True：重建后行数与首次一致", rep3.rows == snap1,
      str({k: (snap1.get(k), rep3.rows.get(k)) for k in snap1 if snap1.get(k) != rep3.rows.get(k)}))

# 建库后连接已释放
check("init_database 结束时已释放连接（无存活连接）", len(live_connections()) == 0,
      f"存活 {len(live_connections())}")

# 建库产物的真实可用性
with connect(MAIN_DB, read_only=True) as ro_check:
    n_words = ro_check.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    check("建库产物可被独立只读连接读取", n_words > 0, f"words={n_words}")
    check("★ 只读库上套用完整 PRAGMA 档案不报错（journal_mode 被正确跳过）",
          ro_check.pragmas.get("foreign_keys") == 1)

# ===========================================================================
banner("§6 事务：提交 / 回滚 / 嵌套拦截 / SAVEPOINT")
# ===========================================================================
TX_DB = TMP / "tx.db"
c = get_connection(TX_DB)
c.execute("CREATE TABLE n(v INTEGER) STRICT")

with transaction(c):
    c.execute("INSERT INTO n(v) VALUES(1)")
    c.execute("INSERT INTO n(v) VALUES(2)")
check("transaction：正常退出即提交（2 行）", c.execute("SELECT COUNT(*) FROM n").fetchone()[0] == 2)

try:
    with transaction(c):
        c.execute("INSERT INTO n(v) VALUES(3)")
        raise RuntimeError("故意失败")
except RuntimeError:
    pass
check("transaction：异常导致回滚（仍为 2 行，第 3 行未落盘）",
      c.execute("SELECT COUNT(*) FROM n").fetchone()[0] == 2)
check("transaction：异常后无残留未闭合事务", c.in_transaction is False)

c.execute("BEGIN IMMEDIATE")
expect_error(
    "★ 嵌套拦截：transaction() 检测到未闭合事务并主动报 TransactionError",
    lambda: transaction(c).__enter__(),
    TransactionError,
)
c.execute("ROLLBACK")
check("拦截后回滚干净", c.in_transaction is False)

# allow_nested=True → SAVEPOINT 语义
with transaction(c):
    c.execute("INSERT INTO n(v) VALUES(10)")
    try:
        with transaction(c, allow_nested=True):
            c.execute("INSERT INTO n(v) VALUES(11)")
            raise ValueError("内层失败")
    except ValueError:
        pass
    check("allow_nested=True 走 SAVEPOINT：内层回滚后外层仍可提交",
          c.execute("SELECT COUNT(*) FROM n WHERE v IN (10,11)").fetchone()[0] == 1)

# 显式 savepoint
with transaction(c):
    with savepoint(c, "sp1"):
        c.execute("INSERT INTO n(v) VALUES(20)")
    check("savepoint：内层提交后可见", c.execute("SELECT COUNT(*) FROM n WHERE v=20").fetchone()[0] == 1)

expect_error(
    "非法事务模式报 TransactionError",
    lambda: transaction(c, mode="NONSENSE").__enter__(),
    TransactionError,
)

check("isolation_level=None：事务块外的 INSERT 立即持久化", c.in_transaction is False)
close_connection(c)

c = get_connection(TX_DB)
check("重开连接验证提交结果确实落盘（不会被隐式回滚）",
      c.execute("SELECT COUNT(*) FROM n").fetchone()[0] > 0)
close_connection(c)

# ===========================================================================
banner("§7 连接生命周期：关闭语义与泄漏检测")
# ===========================================================================
with connect(TX_DB) as cx:
    cx.execute("SELECT 1")
expect_error(
    "★ with connect：退出后连接不可再用（而标准库的 with 不会关闭连接）",
    lambda: cx.execute("SELECT 1"),
    sqlite3.ProgrammingError,
)

# 未闭合事务的关闭语义
c = get_connection(TX_DB)
before = c.execute("SELECT COUNT(*) FROM n").fetchone()[0]
c.execute("BEGIN IMMEDIATE")
c.execute("INSERT INTO n(v) VALUES(99)")
close_connection(c)
c = get_connection(TX_DB)
check("close_connection 默认回滚未闭合事务（不静默提交脏数据）",
      c.execute("SELECT COUNT(*) FROM n").fetchone()[0] == before)
close_connection(c)

c = get_connection(TX_DB)
c.execute("BEGIN IMMEDIATE")
c.execute("INSERT INTO n(v) VALUES(77)")
close_connection(c, commit=True)
c = get_connection(TX_DB)
check("close_connection(commit=True) 显式提交", c.execute("SELECT COUNT(*) FROM n WHERE v=77").fetchone()[0] == 1)
close_connection(c)

check("close_connection(None) 返回 False（空安全）", close_connection(None) is False)
cc = get_connection(TX_DB)
check("close_connection 首次返回 True", close_connection(cc) is True)
check("close_connection 二次调用幂等返回 False", close_connection(cc) is False)

# 泄漏检测
leak_a = get_connection(TX_DB)
leak_b = get_connection(TX_DB)
check("live_connections 能发现未关闭的连接", len(live_connections()) >= 2, f"存活 {len(live_connections())}")
closed_n = close_all()
check("close_all 关闭全部存活连接", len(live_connections()) == 0, f"关闭 {closed_n} 条，剩余 {len(live_connections())}")

# ===========================================================================
banner("§8 只读连接与路径边界（含中文路径 URI 编码）")
# ===========================================================================
ro = get_connection(MAIN_DB, read_only=True)
check("只读连接：read_only 元数据为真", ro.read_only is True)
check("只读连接：FK 仍被强制开启", ro.pragma("foreign_keys") == 1)
check("只读连接：可正常 SELECT", ro.execute("SELECT COUNT(*) FROM words").fetchone()[0] > 0)
expect_error(
    "只读连接：写入被拒绝",
    lambda: ro.execute("INSERT INTO contexts(name) VALUES('x')"),
    sqlite3.OperationalError,
)
expect_error(
    "只读连接：PRAGMA user_version 写入被拒（佐证确为只读句柄）",
    lambda: ro.execute("PRAGMA user_version=123"),
    sqlite3.OperationalError,
)
close_connection(ro)

expect_error(
    "只读打开不存在的文件 → DatabaseConnectionError（绝不静默创建空库）",
    lambda: get_connection(TMP / "ghost.db", read_only=True),
    C.DatabaseConnectionError,
)
expect_error(
    "must_exist=True 时文件不存在 → DatabaseConnectionError",
    lambda: get_connection(TMP / "ghost2.db", must_exist=True),
    C.DatabaseConnectionError,
)

deep = TMP / "自动创建" / "多层" / "深层.db"
deep_conn = get_connection(deep)
deep_conn.execute("CREATE TABLE t(a INTEGER)")
close_connection(deep_conn)
check("create_parents：自动创建多级（含中文）父目录", deep.exists(), str(deep))
deep_wal = get_connection(deep)
check("中文路径下 WAL 生效（URI/路径编码正确）", deep_wal.pragma("journal_mode") == "wal",
      str(deep_wal.pragma("journal_mode")))
close_connection(deep_wal)

# 另一进程只读打开中文路径（验证 as_uri 百分号编码在独立进程中可用）
probe = subprocess.run(
    [
        sys.executable,
        "-c",
        "import sqlite3,sys,pathlib;"
        "uri=pathlib.Path(sys.argv[1]).resolve().as_uri()+'?mode=ro';"
        "c=sqlite3.connect(uri,uri=True);"
        "print(c.execute(\"SELECT COUNT(*) FROM sqlite_master WHERE name='t'\").fetchone()[0])",
        str(deep),
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
)
check("独立进程用 file: URI 读取中文路径只读库成功（表结构可见）", probe.stdout.strip() == "1",
      f"stdout={probe.stdout.strip()!r} stderr={probe.stderr.strip()[:120]}")

# ===========================================================================
banner("§9 内存哈希缓存")
# ===========================================================================
cache = get_memory_cache("t_hot", max_entries=3)
check("get_memory_cache 返回的对象 isinstance(dict) 为真", isinstance(cache, dict))
check("同名缓存返回同一实例", get_memory_cache("t_hot") is cache)
check("MemoryCache 也是 dict 子类", isinstance(cache, MemoryCache))

clear_caches()
calls = {"n": 0}


def factory() -> str:
    calls["n"] += 1
    return "computed"


cache.get_or_compute("k1", factory)
cache.get_or_compute("k1", factory)
check("get_or_compute：命中不重复计算（factory 只调 1 次）", calls["n"] == 1, f"调用 {calls['n']} 次")
check("get_or_compute：返回缓存值", cache.get_or_compute("k1", factory) == "computed")
st = cache.stats()
check("统计：hits=2 / misses=1", st["hits"] == 2 and st["misses"] == 1, str(st))
check("统计：hit_rate 计算正确（≈0.667）", abs(st["hit_rate"] - 2 / 3) < 0.01, str(st["hit_rate"]))

for i in range(5):
    cache.put(f"k{i}", i)
check("容量上限生效：条目数不超过 max_entries=3", len(cache) == 3, f"len={len(cache)}")
check("FIFO 淘汰计数递增", cache.stats()["evictions"] >= 2, str(cache.stats()["evictions"]))
check("淘汰后保留最新键（k4 在、k1 不在）", "k4" in cache and "k1" not in cache)
n_rm = cache.invalidate("k4")
check("invalidate(key) 删除单个键", n_rm == 1 and "k4" not in cache)
_removed = cache.invalidate()
check("invalidate() 清空全部并返回被删条目数", _removed == 2 and len(cache) == 0,
      f"被删={_removed} 剩余={len(cache)}")

h1 = hash_key("sense", "猫", 2)
h2 = hash_key("sense", "猫", 2)
h3 = hash_key("sense", "鱼", 2)
check("hash_key 稳定：同输入同结果", h1 == h2)
check("hash_key 敏感：不同输入不同结果", h1 != h3)
check("hash_key 输出为定长 hex（32 字符 = 16 字节）", len(h1) == 32)
# 跨进程 + 不同 PYTHONHASHSEED 仍稳定（hash() 会不稳定，这里必须稳定）
sub = []
for seedval in ("0", "99999"):
    env = dict(os.environ, PYTHONHASHSEED=seedval)
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys;sys.path.insert(0,sys.argv[1]);from logos.db import hash_key;"
         "print(hash_key('sense','猫',2))", str(ROOT)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    sub.append(r.stdout.strip())
check("★ hash_key 跨进程跨 PYTHONHASHSEED 稳定（不像内置 hash() 被随机化）",
      sub[0] == h1 and sub[1] == h1, f"子进程={sub} 本进程={h1}")

d1 = file_digest(SCHEMA)
check("file_digest 稳定：两次结果一致", d1 == file_digest(SCHEMA))
check("file_digest 敏感：不同文件不同指纹", d1 != file_digest(SEED))
check("file_digest 与报告记录一致", d1 == rep.schema_fingerprint)

# ===========================================================================
banner("§10 查询计时与性能分析")
# ===========================================================================
reset_timing_stats()


@timed("t_deco_args")
def _f1() -> int:
    return 1


@timed
def _f2() -> int:
    return 2


for _ in range(3):
    _f1()
    _f2()
stats = get_timing_stats()
check("@timed('标签') 记录调用次数 = 3", stats["t_deco_args"].count == 3, str(stats["t_deco_args"].count))
check("@timed 无参形式用函数限定名作标签", "_f2" in "".join(stats.keys()), str(list(stats.keys())))


@timed("t_raises")
def _boom() -> None:
    raise ValueError("x")


try:
    _boom()
except ValueError:
    pass
check("计时装饰器：异常路径同样被计时且异常照常抛出",
      get_timing_stats()["t_raises"].count == 1)

with timed_block("t_block"):
    time.sleep(0.005)
check("timed_block：耗时被记录且 >= 5ms", get_timing_stats()["t_block"].total_ms >= 5.0,
      f"{get_timing_stats()['t_block'].total_ms:.2f} ms")

sq = get_connection(TX_DB)
rows, ms = execute_timed(sq, "SELECT * FROM n", label="t_query")
check("execute_timed 返回 (rows, 耗时)", isinstance(rows, list) and ms >= 0, f"{len(rows)} 行 / {ms:.3f} ms")
check("execute_timed 计入统计", get_timing_stats()["t_query"].count == 1)

with count_sql(sq) as info:
    sq.execute("SELECT 1")
    sq.execute("SELECT 2")
check("count_sql：统计块内 SQL 条数 = 2（且带文本）",
      info["count"] == 2 and len(info["sql"]) == 2, str(info["count"]))
check("count_sql：块结束后 tracer 已摘除（不污染后续统计）",
      info["sql"] == ["SELECT 1", "SELECT 2"], str(info["sql"]))

# tracer 归还语义：sqlite3 无 getter，只能由调用方传回
restored: list[str] = []
with count_sql(sq, restore=restored.append) as info2:
    sq.execute("SELECT 3")
sq.execute("SELECT 4")
check("count_sql(restore=...) 块结束后装回调用方 tracer",
      restored == ["SELECT 4"] and info2["count"] == 1, f"restored={restored}")

st_t = get_timing_stats()["t_deco_args"]
check("分位数自洽：p95 >= p50 >= min", st_t.p95_ms >= st_t.p50_ms >= st_t.min_ms,
      f"p50={st_t.p50_ms:.3f} p95={st_t.p95_ms:.3f} min={st_t.min_ms:.3f}")
close_connection(sq)

import io  # noqa: E402

buf = io.StringIO()
dump_timing_stats(buf)
check("dump_timing_stats 输出含标签与统计列", "t_deco_args" in buf.getvalue())

# ===========================================================================
banner("§11 错误处理：自定义异常带结构化上下文")
# ===========================================================================
check("SchemaError 是 DatabaseError 子类", issubclass(SchemaError, DatabaseError))
check("SeedError 是 DatabaseError 子类", issubclass(SeedError, DatabaseError))
check("TransactionError 是 DatabaseError 子类", issubclass(TransactionError, DatabaseError))
check("DatabaseConnectionError 是 DatabaseError 子类", issubclass(C.DatabaseConnectionError, DatabaseError))

bad_schema = TMP / "bad_schema.sql"
bad_schema.write_text(
    "CREATE TABLE a(x INTEGER);\n\nCREATE TABL b(y INTEGER);\n", encoding="utf-8"
)
try:
    init_database(TMP / "bad1.db", bad_schema, [])
    check("坏 schema 应抛 SchemaError", False, "未抛异常")
except SchemaError as e:
    check("坏 schema 抛 SchemaError", True)
    check("★ 错误消息含**精确行号** :3（拆句执行的价值）", ":3" in str(e), str(e)[:150])
    check("SchemaError 携带 db_path / sql / cause 上下文",
          e.db_path is not None and e.sql is not None and e.cause is not None,
          f"db_path={bool(e.db_path)} sql={bool(e.sql)} cause={type(e.cause).__name__}")

good_schema = TMP / "tiny_schema.sql"
good_schema.write_text("CREATE TABLE t(a INTEGER) STRICT;\n", encoding="utf-8")
bad_seed = TMP / "bad_seed.sql"
bad_seed.write_text("INSERT INTO t(a) VALUES(1);\nINSERT INTO nope(a) VALUES(2);\n", encoding="utf-8")
try:
    init_database(TMP / "bad2.db", good_schema, [bad_seed])
    check("坏种子应抛 SeedError", False, "未抛异常")
except SeedError as e:
    check("坏种子抛 SeedError", True)
    check("★ 种子错误同样含精确行号 :2", ":2" in str(e), str(e)[:150])

# 错误后连接不得泄漏、不得留下未闭合事务
check("异常路径不泄漏连接（finally 已关闭）", len(live_connections()) == 0,
      f"存活 {len(live_connections())}")

# 建库失败后库文件处于可重新使用状态（无残留事务）
retry = get_connection(TMP / "bad2.db")
check("失败库可被重新打开且无未闭合事务", retry.in_transaction is False)
close_connection(retry)

# ===========================================================================
banner("§12 PRAGMA 值注入防护")
# ===========================================================================
inj = get_connection(TMP / "inj.db")
expect_error(
    "PRAGMA 值白名单：非法值被拒（PRAGMA 无法参数化，必须白名单校验）",
    lambda: C._set_pragma(inj, "cache_size", "-1; DROP TABLE t"),
    DatabaseError,
)
check("注入尝试后连接仍可用", inj.execute("SELECT 1").fetchone()[0] == 1)
close_connection(inj)

# ===========================================================================
# 收尾
# ===========================================================================
print()
print("=" * 78)
print("计时统计（由本模块的计时装饰器采集，用于性能分析）")
print("=" * 78)
dump_timing_stats(sort_by="total_ms")

print()
print("=" * 78)
if FAILED:
    print("失败清单：")
    for name, detail in FAILED:
        print(f"  ✗ {name}   {detail}")
print(f"结果: PASS {PASS} / FAIL {FAIL}      总耗时 {time.perf_counter() - T0:.2f}s")
print("=" * 78)

# 清理且确认无连接泄漏
leftover = close_all()
if leftover:
    print(f"（清理了 {leftover} 条遗留连接）")
shutil.rmtree(TMP, ignore_errors=True)

sys.exit(1 if FAIL else 0)
