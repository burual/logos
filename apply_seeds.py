#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows 下建库 + 填种子的引导脚本（Logos v3）

用途
----
从零把八份 SQL 铺进一个 SQLite 文件库，并在结束时做一次**结构性自检**：

    L1 Schema      schema_v3.sql               52 STRICT 表 / 5 视图 / 10 触发器
    L2 Seed        seed_general_v3.sql         通用领域种子（猫吃鱼及其扩展句式）
    L2 Seed+       seed_general_v3_ext.sql     鱼的多继承 + 抽象/人造物分类骨架
    L2 SeedM       seed_medical_v3.sql         医学通用骨架 + 跨库领域链接
    L2 SeedI       seed_medical_v3_imaging.sql 影像特征子树（形态/大小/密度/强化/位置 + 结节）
    L2 SeedM+      seed_medical_v3_patch_symptom.sql          症状继承补丁（通用症状下沉到层）
    L2 SeedM++     seed_medical_v3_patch_symptom_layers.sql   症状分层补丁（炎症性疾病层 + 癫痫补边）
    L2 SeedM+++    seed_medical_v3_patch_symptom_severity.sql 症状严重度义素（sense_features）

加载顺序即上表顺序（医学库依赖 ext 建的「抽象概念 / 营养物」，且跨库链接用到
seed_general 的「鱼」「吃」；影像子树依赖医学库建的「医学概念」「信号」；
症状补丁依赖医学库建的「症状 / 疾病 / 脑部疾病」，分层补丁依赖症状补丁建的
「发热」，严重度补丁又依赖症状补丁建的 4031-4036 症状义项 —— 这些依赖关系在
种子文件里由 §0 的前置检查强制，裸 schema 上单独加载医学库 / 影像子树 /
症状补丁 / 分层补丁 / 严重度补丁都会**显式失败**而不是静默断链）。

注：种子文件在 `init_database` 里的**自动发现顺序是文件名字典序**（
`logos/db/connection.py` 的 `_resolve_seed_paths()` 用
`sorted(glob("seed_*.sql"))`）。本表的顺序与之**逐项一致**，其中三处命中
字典序的细节值得留意（改名前务必实跑 `sorted(glob(...))` 确认）：
  · `seed_medical_v3.sql` < `seed_medical_v3_imaging.sql` < `..._patch_symptom.sql`
    （同前缀后 '.'(U+002E) < '_'(U+005F)；再往后 'i'(0x69) < 'p'(0x70)）——
    影像子树不依赖任何症状补丁，故夹在两者之间是正确且安全的位次；
  · 两份 symptom 补丁之间：'l'(0x6C, layers) < 's'(0x73, severity)，位置自由；
  · 若有人把严重度补丁改名为 `..._patch_severity.sql`，'severity' 的 'e' 会排在
    'symptom' 的 'y' 之前，从而在被依赖的症状义项尚未建立时执行、静默丢 8 行。

用法
----
    python apply_seeds.py                 # 建 ROOT/logos.db（已存在则先删除）
    python apply_seeds.py path/to/x.db    # 自定义目标库
    python apply_seeds.py --keep          # 保留已有库，做增量叠加（种子文件本身幂等）

设计要点（三条，都是刻意的）
----------------------------
1. **逐条语句执行，而不是 executescript**
   `conn.executescript()` 把整段脚本一体化执行，报错时无法指出是哪一条语句出错，
   而种子文件里一条多行 `INSERT` 可能涉及几十行数据。本脚本用
   `sqlite3.complete_statement()` 切分语句后逐条 `execute`，因此报错能精确到
   「文件 + 第 N 条语句 + 语句文本 + 冲突的具体键/端点」。
   `complete_statement` 能正确识别 `CREATE TRIGGER ... BEGIN ... END;` 的触发器体
   （体内的分号不会误切）—— 这一点在本项目的 docs/CONNECTION_LAYER.md 里已实测确认。

2. **连接用 isolation_level=None（自动提交），事务交给 SQL 文件自己管**
   八份 SQL 文件都自带 `BEGIN; ... COMMIT;`。若沿用 python
   sqlite3 默认的隐式事务管理，显式 `BEGIN;` 会与驱动自动开启的事务撞车
   （"cannot start a transaction within a transaction"）。设为 None 后，
   事务边界完全由文件里的 BEGIN/COMMIT 决定，`conn.in_transaction` 也随之成为
   一个可信的收尾校验（文件漏写 COMMIT 会被立刻发现）。

3. **失败绝不静默跳过**
   任何一步出错都打印：出错文件 / 语句序号 / 语句文本 / 原始异常 / **可执行的诊断 SQL**，
   然后 rollback、关闭连接、`sys.exit(非零)`。
   - 外键冲突 → 从语句字面量反推出**缺失的父行值**（如 `relations.src_id = 999`），
     给出三条修复 SQL；
   - UNIQUE 冲突 → 从异常信息解析冲突列、从语句字面量取出键值，
     查出库中已存在的那一行，并给出幂等改写方案；
   - 编码错误 → 报出文件与字节偏移，附 PowerShell 转码命令；
   - 路径错误 → 报出搜索过的目录列表。

退出码
------
    0 成功     1 建库成功但结构自检未全绿     2 路径错误     3 执行 SQL 出错     4 编码错误
"""

from __future__ import annotations

import io
import re
import sqlite3
import sys
import pathlib

# --------------------------------------------------------------------------- #
# 0. 路径解析（pathlib，不硬编码分隔符）
# --------------------------------------------------------------------------- #

ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = ROOT / "logos.db"

# (显示名, 候选文件名) —— 依次尝试，命中第一个存在的。
# 兼容两种工程布局：SQL 放在项目根目录（本仓库现状），或放在 ROOT/db/ 下。
SQL_FILES = [
    ("L1 Schema  ", ("schema_v3.sql",)),
    ("L2 Seed    ", ("seed_general_v3.sql",)),
    ("L2 Seed+   ", ("seed_general_v3_ext.sql",)),
    ("L2 SeedM   ", ("seed_medical_v3.sql",)),
    ("L2 SeedI   ", ("seed_medical_v3_imaging.sql",)),                 # 影像特征子树（追加）
    ("L2 SeedM+  ", ("seed_medical_v3_patch_symptom.sql",)),           # 症状继承补丁（追加）
    ("L2 SeedM++ ", ("seed_medical_v3_patch_symptom_layers.sql",)),    # 症状分层补丁（追加）
    ("L2 SeedM+++", ("seed_medical_v3_patch_symptom_severity.sql",)),  # 症状严重度义素（追加）
]
SEARCH_DIRS = (ROOT, ROOT / "db", ROOT / "sql", ROOT / "database")

# 自检期望值（来自种子文件头的实测值；不一致只算 FAIL 项，不影响建库完成）
EXPECT = {
    "senses_after_all": 170,     # 27 + 27 + 39 + medical 31 + imaging 37 + patch 6 + layers 3
    "relations_after_all": 166,  # 23 + 38 + medical 40 + imaging 46 + patch 11 + layers 8
    "closure_after_all": 371,    # 全部由触发器派生（八份文件均无手工 closure INSERT）
    # 只写义素表的两个补丁：severity 8 行 / imaging 12 行
    "sense_features_after_all": 98,      # 六份文件后 78 + severity 8 + imaging 12
    "semantic_features_after_all": 22,   # schema 7 + severity 3 + imaging 12
}


def resolve_sql(name: str) -> pathlib.Path:
    """在候选目录里找 SQL 文件；找不到就抛 FileNotFoundError（含搜索过的目录）。"""
    for d in SEARCH_DIRS:
        p = d / name
        if p.is_file():
            return p
    tried = "\n".join(f"    - {d / name}" for d in SEARCH_DIRS)
    raise FileNotFoundError(f"找不到 SQL 文件 {name}（已搜索以下路径）:\n{tried}")


def read_sql(path: pathlib.Path) -> str:
    """UTF-8 读取；编码问题给出文件与字节偏移，不抛裸异常。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(
            f"[编码错误] {path.name} 不是合法 UTF-8。\n"
            f"  位置: 字节偏移 {e.start}-{e.end}\n"
            f"  原因: {e.reason}\n"
            f"  建议: 另存为「UTF-8」后重跑；PowerShell 转码命令：\n"
            f"        (Get-Content -Raw -Encoding GBK '{path}') | "
            f"Set-Content -Encoding UTF8 -NoNewline '{path}'"
        ) from None


# --------------------------------------------------------------------------- #
# 1. 语句切分
# --------------------------------------------------------------------------- #

def _comment_only(text: str) -> bool:
    """整块都是 SQL 注释（含 /* */ 块注释与 -- 行注释）时返回 True。"""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("--", "/*", "*", "*/")):
            continue
        return False
    return True


def split_statements(sql: str):
    """把脚本切成 (序号, 语句文本)；序号从 1 起。

    只有 `sqlite3.complete_statement()` 认定"已完整"的缓冲才算一条语句，
    因此触发器体内的分号不会被误当作语句边界。
    """
    buf: list[str] = []
    n = 0
    for line in sql.splitlines(keepends=True):
        buf.append(line)
        chunk = "".join(buf)
        if sqlite3.complete_statement(chunk):
            text = chunk.strip()
            if text and not _comment_only(text):
                n += 1
                yield n, text
            buf = []
    tail = "".join(buf).strip()
    if tail and not _comment_only(tail):
        n += 1
        yield n, tail


# --------------------------------------------------------------------------- #
# 2. 报错诊断（把"哪条语句"升级为"哪个键/哪个端点"）
# --------------------------------------------------------------------------- #

_UNIQUE_RE = re.compile(r"UNIQUE constraint failed:\s*(.+)", re.I)
_INSERT_RE = re.compile(
    r"\s*INSERT\s+(?:OR\s+\w+\s+)?INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*VALUES\s*(.*)",
    re.S | re.I)


def _read_tuple(s: str, i: int) -> tuple[str, int]:
    """从 s[i]=='(' 起读出一个括号组，返回 (内部文本, 下一位置)。前后不匹配则读到末尾。"""
    depth, quote, out = 0, None, []
    j = i
    while j < len(s):
        ch = s[j]
        if quote:
            if ch == quote:
                if j + 1 < len(s) and s[j + 1] == quote:     # 转义引号 '' / ""
                    out.append(ch)
                    j += 2
                    continue
                quote = None
            out.append(ch)
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "(":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return "".join(out), j + 1
            out.append(ch)
        else:
            out.append(ch)
        j += 1
    return "".join(out), j


def _split_fields(inner: str) -> list[str]:
    """按顶层逗号切分元组内部（忽略字符串内与括号内的逗号）。"""
    out: list[str] = []
    cur: list[str] = []
    depth, quote, i = 0, None, 0
    while i < len(inner):
        ch = inner[i]
        if quote:
            if ch == quote:
                if i + 1 < len(inner) and inner[i + 1] == quote:
                    cur.append(ch)
                    i += 2
                    continue
                quote = None
            cur.append(ch)
        elif ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur).strip())
    return out


def parse_insert_literals(stmt: str):
    """尽力解析 `INSERT INTO t(cols) VALUES (…),(…)` 的字面量。

    只服务于**报错诊断**：解析失败返回 None，绝不因解析失败而吞掉原异常。
    返回 (table, cols, [每行的值列表])；值保留 SQL 字面量原样（含引号）。
    """
    m = _INSERT_RE.match(stmt)
    if not m:
        return None
    table, coltext, rest = m.group(1), m.group(2), m.group(3)
    cols = [c.strip().strip('"`[]') for c in coltext.split(",") if c.strip()]
    rows: list[list[str]] = []
    i = 0
    while i < len(rest):
        while i < len(rest) and rest[i] in " \t\r\n,":
            i += 1
        if i >= len(rest) or rest[i] != "(":
            break
        inner, i = _read_tuple(rest, i)
        rows.append(_split_fields(inner))
    return table, cols, rows


def _unquote(lit: str):
    """把 SQL 字面量还原成 python 值（够诊断用即可）。"""
    s = lit.strip()
    if s.upper() == "NULL":
        return None
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        return s[1:-1].replace(s[0] * 2, s[0])
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    try:
        return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:                                     # pragma: no cover
        return False


def diag_unique(conn: sqlite3.Connection, msg: str, stmt: str) -> list[str]:
    """UNIQUE 冲突：解析冲突列 → 取出键值 → 查库中已存在的那一行。"""
    out: list[str] = []
    m = _UNIQUE_RE.search(msg)
    pairs = [x.strip() for x in m.group(1).split(",")] if m else []
    table = pairs[0].split(".")[0] if pairs else None
    keycols = [p.split(".")[-1] for p in pairs]
    if table:
        out.append(f"  · 冲突表 {table}，唯一键列 ({', '.join(keycols)})")

    parsed = parse_insert_literals(stmt)
    hits: list[tuple[tuple, list]] = []
    if parsed and table and parsed[0] == table:
        _, cols, rows = parsed
        idxs = [(k, cols.index(k)) for k in keycols if k in cols]
        if len(idxs) == len(keycols):
            for r in rows:
                key = tuple(_unquote(r[i]) for _, i in idxs)
                if any(v is None for v in key):
                    continue
                where = " AND ".join(f"{k}=?" for k, _ in idxs)
                try:
                    hit = conn.execute(
                        f"SELECT rowid FROM {table} WHERE {where}", key).fetchall()
                except sqlite3.Error:
                    hit = []
                if hit:
                    hits.append((key, hit))

    if hits:
        out.append("  · 重复键（库中已存在，本条 INSERT 撞上了它）：")
        for key, hit in hits[:10]:
            ks = ", ".join(f"{k}={v!r}" for k, v in zip(keycols, key))
            out.append(f"      {ks}      已存在 {len(hit)} 行")
        out.append(
            f"  · 修复 SQL（二选一）：\n"
            f"      -- (a) 幂等叠加（本项目约定，推荐）：\n"
            f"      INSERT INTO {table}(...) VALUES (...) ON CONFLICT DO NOTHING;\n"
            f"      -- (b) 确实要用新值覆盖旧值：\n"
            f"      INSERT INTO {table}(...) VALUES (...) "
            f"ON CONFLICT({', '.join(keycols)}) DO UPDATE SET ...;"
        )
        return out

    # 兜底：解析不出字面量时，退化为"表内是否已有重复键"的扫描
    if table and keycols and all(_col_exists(conn, table, c) for c in keycols):
        collist = ", ".join(keycols)
        try:
            dup = conn.execute(
                f"SELECT {collist}, COUNT(*) FROM {table} "
                f"GROUP BY {collist} HAVING COUNT(*) > 1 LIMIT 10").fetchall()
        except sqlite3.Error:                                 # pragma: no cover
            dup = []
        if dup:
            out.append("  · 表内已存在的重复键：")
            for r in dup:
                out.append("      " + " | ".join(str(v) for v in r))
        else:
            out.append("  · 表内没有重复键，说明冲突来自一个**已存在的单行**"
                       "（该数据的等价行已在库中）。")
        out.append(
            f"  · 修复 SQL：把该语句改成幂等形式\n"
            f"      INSERT INTO {table}(...) SELECT ... WHERE NOT EXISTS (...);\n"
            f"      -- 或  ON CONFLICT DO NOTHING / DO UPDATE"
        )
    return out


def diag_relation_endpoints(conn: sqlite3.Connection, cols: list[str],
                           rows: list[list[str]]) -> list[str]:
    """relations 的端点校验（tr_relations_validate_ai 在触发器层做，没有 FK 可查）。"""
    bad: list[tuple] = []
    for r in rows:
        d = {c: _unquote(v) for c, v in zip(cols, r)}
        for side in ("src", "dst"):
            kind, rid = d.get(f"{side}_kind"), d.get(f"{side}_id")
            if kind is None or rid is None:
                continue
            tbl = "senses" if kind == "sense" else "entity_nodes"
            col = "sense_id" if kind == "sense" else "node_id"
            try:
                ok = conn.execute(f"SELECT 1 FROM {tbl} WHERE {col}=? LIMIT 1", (rid,)).fetchone()
            except sqlite3.Error:
                ok = True
            if not ok:
                bad.append((side, kind, rid, tbl, col))
    if not bad:
        return []
    out = ["  · 端点校验失败：以下端点不存在（触发器 tr_relations_validate_ai 拒绝）"]
    for side, kind, rid, tbl, col in bad[:10]:
        out.append(f"      {side}_kind={kind!r}, {side}_id={rid!r}  →  {tbl}.{col} 无此行")

    # 按父表分组，修复 SQL 每个父表只打印一次（多个悬空 id 时不再重复同一段模板）
    grouped: dict[tuple, list] = {}
    for _side, _kind, rid, tbl, col in bad:
        grouped.setdefault((tbl, col), []).append(rid)
    out.append("  · 修复 SQL：")
    for (tbl, col), rids in list(grouped.items())[:3]:
        preview = ", ".join(repr(v) for v in rids[:5])
        out.append(f"      -- 先确认这些 id 是否真的不存在（父表 {tbl}）：")
        out.append(f"      SELECT {col} FROM {tbl} WHERE {col} IN ({preview});")
        out.append(f"      -- (a) 若应存在：在插入 relations **之前**先插入 {tbl} 的对应行；")
        out.append(f"      -- (b) 若 id 写错了：改用自然键定位，例如")
        out.append(f"      --     SELECT s.sense_id FROM senses s JOIN words w ON w.word_id=s.word_id "
                   f"WHERE w.word='<目标词>' AND s.pos='n';")
    return out


def diag_foreign_key(conn: sqlite3.Connection, stmt: str) -> list[str]:
    """外键冲突：反推缺失的父行值 + 给出修复 SQL。"""
    out: list[str] = []
    chk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if chk:
        out.append("  · PRAGMA foreign_key_check（表, rowid, 父表, fk#）：")
        for r in chk[:10]:
            out.append(f"      {r}")

    parsed = parse_insert_literals(stmt)
    if not parsed:
        out.append("  · 语句含子查询/CTE，无法从中反推键值；请按下方模板自查。")
        return out
    table, cols, rows = parsed
    try:
        fks = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    except sqlite3.Error:                                     # pragma: no cover
        return out

    if table == "relations":
        out.extend(diag_relation_endpoints(conn, cols, rows))

    if not fks:
        if table != "relations":
            out.append(f"  · {table} 未声明外键（冲突可能来自触发器内的 RAISE(ABORT)）")
        return out

    out.append(f"  · {table} 的外键声明：")
    for fk in fks:
        out.append(f"      {fk[3]} → {fk[2]}.{fk[4]}")

    missing: list[tuple] = []
    for fk in fks:
        parent, frm, to = fk[2], fk[3], fk[4]
        if frm not in cols:
            continue
        ci = cols.index(frm)
        for r in rows:
            if ci >= len(r):
                continue
            v = _unquote(r[ci])
            if v is None:
                continue
            try:
                ok = conn.execute(
                    f"SELECT 1 FROM {parent} WHERE {to}=? LIMIT 1", (v,)).fetchone()
            except sqlite3.Error:
                ok = True
            if not ok:
                missing.append((table, frm, parent, to, v))

    if missing:
        out.append("  · 具体缺失的父行：")
        for t, frm, parent, to, v in missing[:10]:
            out.append(f"      {t}.{frm} = {v!r}  →  {parent}.{to} 中不存在")
        out.append("  · 修复 SQL（三选一）：")
        for t, frm, parent, to, v in missing[:3]:
            out.append(f"      -- (a) 补建父行："
                       f" INSERT INTO {parent}({to}, ...) VALUES ({v!r}, ...);")
            out.append(f"      -- (b) 改用已存在的 id：SELECT {to} FROM {parent} LIMIT 5;")
            out.append(f"      -- (c) 若父行本应由更早的语句创建，先修那条语句（可能被跳过）。")
    else:
        out.append("  · 未能从字面量定位缺失值；请按下方模板自查：")
        for fk in fks:
            out.append(f"      SELECT {fk[3]} FROM {table} WHERE {fk[3]} IS NOT NULL "
                       f"AND {fk[3]} NOT IN (SELECT {fk[4]} FROM {fk[2]});")
    return out


def report_failure(conn: sqlite3.Connection, path: pathlib.Path, idx: int,
                   total: int, stmt: str, exc: BaseException) -> None:
    """统一失败出口：文件 / 语句序号 / 语句文本 / 异常 / 诊断 / 修复建议。"""
    head = stmt.splitlines()[0] if stmt else ""
    print("\n" + "!" * 78)
    print(f"[失败] {path.name}  第 {idx}/{total} 条语句")
    print("!" * 78)
    print(f"  语句首行: {head[:110]}")
    lines = stmt.splitlines()
    if len(lines) > 1:
        print(f"  语句共 {len(lines)} 行，前 400 字符：")
        print("  " + "-" * 70)
        for ln in stmt[:400].splitlines():
            print(f"  | {ln}")
        print("  " + "-" * 70)
    print(f"  异常: {type(exc).__name__}: {exc}")

    msg = str(exc)
    upper = msg.upper()
    diag: list[str] = []
    if isinstance(exc, sqlite3.IntegrityError):
        if "UNIQUE" in upper:
            diag = diag_unique(conn, msg, stmt)
        elif "FOREIGN KEY" in upper or "端点不存在" in msg:
            diag = diag_foreign_key(conn, stmt)
        else:
            diag = [f"  · 其它完整性违规（CHECK / NOT NULL / 触发器 RAISE）：{msg}"]
    elif isinstance(exc, sqlite3.OperationalError):
        diag = [
            "  · 语法 / 对象缺失类错误。常见原因：",
            "      - 目标表尚未创建（schema_v3.sql 未成功执行）",
            "      - 引用了 schema 中不存在的列，或 CHECK 白名单外的取值",
            "      - 把 'WITH ... INSERT' 写进了触发器体（SQLite 不允许，"
            "需改成 INSERT ... SELECT ... FROM (WITH RECURSIVE ...)）",
        ]
    if diag:
        print("  诊断：")
        for line in diag:
            print(line)


# --------------------------------------------------------------------------- #
# 3. 主流程
# --------------------------------------------------------------------------- #

def apply_file(conn: sqlite3.Connection, path: pathlib.Path) -> tuple[int, int]:
    """逐条语句执行一个 SQL 文件；返回 (语句总数, 已执行数)。失败时抛异常给上层。"""
    sql = read_sql(path)
    stmts = list(split_statements(sql))
    total = len(stmts)
    done = 0
    for idx, stmt in stmts:
        try:
            conn.execute(stmt)
            done += 1
        except (sqlite3.Error, sqlite3.Warning) as e:
            report_failure(conn, path, idx, total, stmt, e)
            raise
    return total, done


def selfcheck(conn: sqlite3.Connection) -> int:
    """结构 + 语义自检。返回 FAIL 数（0 = 全绿）。"""
    fails = 0

    def ck(name: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        if cond:
            print(f"  [ok]   {name}")
        else:
            fails += 1
            print(f"  [FAIL] {name}   {detail}")

    def one(sql: str, args=()):
        return conn.execute(sql, args).fetchone()[0]

    def sid(word: str, pos: str):
        r = conn.execute(
            "SELECT s.sense_id FROM senses s JOIN words w ON w.word_id=s.word_id "
            "WHERE w.word=? AND s.pos=? LIMIT 1", (word, pos)).fetchone()
        return r[0] if r else None

    def anc_of(word: str) -> set:
        return {r[0] for r in conn.execute(
            "SELECT ancestor FROM closure WHERE relation='is_a' AND descendant=?",
            (sid(word, "n"),))}

    # --- 完整性 ---
    ck("PRAGMA integrity_check = ok",
       conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok")
    fkv = conn.execute("PRAGMA foreign_key_check").fetchall()
    ck("PRAGMA foreign_key_check 无违规", not fkv, str(fkv[:3]))
    ck("收尾无未闭合事务（SQL 文件 BEGIN/COMMIT 配对）",
       conn.in_transaction is False, "in_transaction=True —— 缺 COMMIT")

    # --- 八份文件都进去了 ---
    n_senses, n_rel, n_clo = (one("SELECT COUNT(*) FROM senses"),
                              one("SELECT COUNT(*) FROM relations"),
                              one("SELECT COUNT(*) FROM closure"))
    ck(f"senses = {EXPECT['senses_after_all']}（实测 {n_senses}）",
       n_senses == EXPECT["senses_after_all"], f"实测 {n_senses}")
    ck(f"relations = {EXPECT['relations_after_all']}（实测 {n_rel}）",
       n_rel == EXPECT["relations_after_all"], f"实测 {n_rel}")
    ck(f"closure = {EXPECT['closure_after_all']}（实测 {n_clo}）",
       n_clo == EXPECT["closure_after_all"], f"实测 {n_clo}")
    n_sf = one("SELECT COUNT(*) FROM sense_features")
    n_mf = one("SELECT COUNT(*) FROM semantic_features")
    ck(f"sense_features = {EXPECT['sense_features_after_all']}（实测 {n_sf}）",
       n_sf == EXPECT["sense_features_after_all"], f"实测 {n_sf}")
    ck(f"semantic_features = {EXPECT['semantic_features_after_all']}（实测 {n_mf}）",
       n_mf == EXPECT["semantic_features_after_all"], f"实测 {n_mf}")

    # --- 任务一：鱼的多继承 ---
    anc = anc_of("鱼")
    ck("鱼的上位链 = 动物/食物/生物/营养物（DAG 多继承）",
       {sid("动物", "n"), sid("食物", "n"), sid("生物", "n"), sid("营养物", "n")} <= anc,
       f"实测 {sorted(anc)}")
    ck("鱼的 is_a 出边 = 2（动物 / 食物）",
       one("SELECT COUNT(*) FROM relations WHERE src_kind='sense' AND src_id=? "
           "AND relation='is_a'", (sid("鱼", "n"),)) == 2)
    props = {r[0] for r in conn.execute(
        "SELECT wn.word FROM relations r JOIN senses s ON s.sense_id=r.dst_id "
        "JOIN words wn ON wn.word_id=s.word_id "
        "WHERE r.relation='has_property' AND r.src_id=?", (sid("食物", "n"),))}
    ck("食物 has_property 可食用 / 有营养",
       {"可食用", "有营养"} <= props, f"实测 {props}")
    ck("cross_links 视图含 sense↔node 领域链接",
       one("SELECT COUNT(*) FROM cross_links WHERE relation='domain_link'") >= 1)

    # --- 任务二：分类骨架 ---
    # 注：seed_medical_v3.sql 把「医学概念」挂在「抽象概念」下，因此四份 SQL 全加载后
    #     抽象概念子树**必然**多于 ext 建的那 36 个 —— 这是设计内的扩展，不是漂移。
    #     本项只统计 ext 时代的节点（sense_id < 4000，medical 段自 4001 起），
    #     医学分支由 validate_medical.py 的 §M4 单独断言（那边口径更细）。
    MED_FLOOR = 4000
    skel: set = set()
    for root in ("抽象概念", "人造物"):
        rs = sid(root, "n")
        skel.add(rs)
        skel |= {r[0] for r in conn.execute(
            "SELECT descendant FROM closure WHERE relation='is_a' AND ancestor=?", (rs,))}
    ext_skel = {n for n in skel if n < MED_FLOOR}
    ck(f"分类骨架节点数 = 36（ext 时代节点；全量含医学分支共 {len(skel)}）",
       len(ext_skel) == 36, f"ext 时代 {len(ext_skel)}；全量 {len(skel)}")
    for w, chain in (("视图", ("数据对象", "信息人造物", "人造物")),
                     ("文档", ("信息载体", "信息人造物")),
                     ("程序", ("软件产物", "信息人造物"))):
        a = anc_of(w)
        ck(f"{w} 的祖先链 ⊇ {' → '.join(chain)}",
           {sid(x, "n") for x in chain} <= a, f"实测 {sorted(a)}")

    # --- 量词搭配 ---
    nc = {r[0] for r in conn.execute(
        "SELECT wn.word FROM noun_classifier_pairs p "
        "JOIN senses s ON s.sense_id=p.classifier_sense_id "
        "JOIN words wn ON wn.word_id=s.word_id WHERE p.noun_sense_id IN (?,?,?)",
        (sid("视图", "n"), sid("文档", "n"), sid("程序", "n")))}
    ck("视图 / 文档 / 程序 均已配量词", {"个", "份"} <= nc, f"实测 {nc}")

    # --- 医学库（seed_medical_v3.sql）---
    # 不重复 validate_medical.py 的细查，这里只验"七份 SQL 合起来形状还对"。
    ck("医学概念 ⊑ 抽象概念（医学树挂到 ext 的抽象概念下）",
       bool(one("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=? "
                "AND descendant=?", (sid("抽象概念", "n"), sid("医学概念", "n")))))
    ck("癫痫 的 is_a 出边 = 2（疾病 / 脑部疾病，多继承）",
       one("SELECT COUNT(*) FROM relations WHERE relation='is_a' AND src_kind='sense' "
           "AND src_id=?", (sid("癫痫", "n"),)) == 2)
    # has_part 的闭包行方向恒为边的反向（I3: ancestor = 边的 dst）→ 部分在 ancestor 侧
    ck("has_part 传递闭包已派生 depth>1（脑 → 大脑皮层）",
       one("SELECT COUNT(*) FROM closure WHERE relation='has_part' AND ancestor=? "
           "AND descendant=? AND depth=2", (sid("大脑皮层", "n"), sid("脑", "n"))) == 1)
    n_med_cross = one("SELECT COUNT(*) FROM cross_links r "
                      "JOIN entity_nodes n ON n.node_id=r.target_id "
                      "WHERE n.domain='medical'")
    ck("医学库跨库链接 = 4（营养学相关 / 功能类比 / 领域特化×2：脑电图信号 + 强化特征）",
       n_med_cross == 4, f"实测 {n_med_cross}")

    # --- 症状继承补丁（seed_medical_v3_patch_symptom.sql）---
    ck("通用症状义项 = 6（疼痛/发热/头痛/眩晕/乏力/意识模糊）",
       one("SELECT COUNT(*) FROM words WHERE word_id BETWEEN 4031 AND 4036") == 6)
    ck("层上症状边 = 疾病→2 + 脑部疾病→3",
       one("SELECT COUNT(*) FROM relations WHERE relation='has_symptom' "
           "AND src_kind='sense' AND src_id=?", (sid("疾病", "n"),)) == 2
       and one("SELECT COUNT(*) FROM relations WHERE relation='has_symptom' "
               "AND src_kind='sense' AND src_id=?", (sid("脑部疾病", "n"),)) == 3)
    inh_epi = {r[0] for r in conn.execute("""
        SELECT DISTINCT wd.word FROM closure cl
          JOIN relations r ON r.relation='has_symptom' AND r.src_kind='sense'
                          AND r.src_id=cl.ancestor
          JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
         WHERE cl.relation='is_a' AND cl.descendant=?""", (sid("癫痫", "n"),))}
    ck("癫痫 沿 is_a 继承到 5 种通用症状（疾病层 2 + 脑部疾病层 3）",
       {"疼痛", "乏力", "头痛", "眩晕", "意识模糊"} <= inh_epi, f"实测 {sorted(inh_epi)}")

    # --- 症状严重度义素补丁（seed_medical_v3_patch_symptom_severity.sql）---
    sev_of = {r[0]: r[1] for r in conn.execute(
        "SELECT w.word, sf.feature FROM sense_features sf "
        "JOIN senses s ON s.sense_id=sf.sense_id JOIN words w ON w.word_id=s.word_id "
        "WHERE sf.feature IN ('+severe','+moderate','+mild')")}
    ck("严重度义素 = 8 个症状各 1 行（+severe/+moderate/+mild）",
       len(sev_of) == 8, f"实测 {len(sev_of)}：{sorted(sev_of)}")
    ck("严重度分级正确（重度 疼痛/头痛/抽搐/意识模糊；中度 眩晕/乏力/记忆障碍；轻度 发热）",
       sev_of == {"疼痛": "+severe", "头痛": "+severe", "抽搐": "+severe",
                  "意识模糊": "+severe", "眩晕": "+moderate", "乏力": "+moderate",
                  "记忆障碍": "+moderate", "发热": "+mild"},
       f"实测 {sev_of}")
    sev_epi = {r[0] for r in conn.execute("""
        SELECT DISTINCT wd.word FROM relations r
          JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
          JOIN sense_features sf ON sf.sense_id=r.dst_id AND sf.feature='+severe'
         WHERE r.relation='has_symptom' AND r.src_kind='sense'
           AND ( r.src_id = ?
                 OR r.src_id IN (SELECT ancestor FROM closure
                                  WHERE relation='is_a' AND descendant=?) )""",
        (sid("癫痫", "n"), sid("癫痫", "n")))}
    ck("癫痫 的重度症状 ⊇ {疼痛, 头痛, 抽搐}（closure 继承 + 义素过滤）",
       {"疼痛", "头痛", "抽搐"} <= sev_epi, f"实测 {sorted(sev_epi)}")

    # --- 症状分层补丁（seed_medical_v3_patch_symptom_layers.sql）---
    ck("症状分层义项 = 3（炎症性疾病 / 感染性疾病 / 脑炎；4037-4039 段）",
       one("SELECT COUNT(*) FROM words WHERE word_id BETWEEN 4037 AND 4039") == 3)
    ck("脑炎 的 is_a 出边 = 2（炎症性疾病 / 脑部疾病，多继承）",
       one("SELECT COUNT(*) FROM relations WHERE relation='is_a' AND src_kind='sense' "
           "AND src_id=?", (sid("脑炎", "n"),)) == 2)
    ck("炎症性疾病 的 has_symptom 出边 = 1（发热；这正是 patch_symptom §3.7 留的扩展位）",
       one("SELECT COUNT(*) FROM relations WHERE relation='has_symptom' AND src_kind='sense' "
           "AND src_id=?", (sid("炎症性疾病", "n"),)) == 1)
    sym_ence = {r[0] for r in conn.execute("""
        SELECT DISTINCT wd.word FROM relations r
          JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
         WHERE r.relation='has_symptom' AND r.src_kind='sense'
           AND ( r.src_id = ? OR r.src_id IN (SELECT ancestor FROM closure
                                               WHERE relation='is_a' AND descendant=?) )""",
        (sid("脑炎", "n"), sid("脑炎", "n")))}
    ck("脑炎 的症状 = 7 种（炎症层 1 + 疾病层 2 + 脑部疾病层 3 + 直接 1）",
       sym_ence == {"发热", "疼痛", "乏力", "头痛", "眩晕", "意识模糊", "抽搐"},
       f"实测 {sorted(sym_ence)}")
    ck("癫痫→发热 为低置信度弱关联（confidence=0.5 = 全库 has_symptom 最低）",
       one("SELECT MIN(confidence) FROM relations WHERE relation='has_symptom'") == 0.5)

    # --- 影像特征子树（seed_medical_v3_imaging.sql）---
    ck("影像义项 = 37（4040-4076 段）",
       one("SELECT COUNT(*) FROM words WHERE word_id BETWEEN 4040 AND 4076") == 37)
    ck("影像特征 ⊑ 医学概念，且 影像特征 is_a 下位 = 32",
       bool(one("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=? "
                "AND descendant=?", (sid("医学概念", "n"), sid("影像特征", "n"))))
       and one("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=?",
               (sid("影像特征", "n"),)) == 32)
    ck("结节 ⊑ 病理结构 ⊑ 医学概念；恶性 / 良性结节 ⊑ 结节",
       one("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
           (sid("病理结构", "n"), sid("结节", "n"))) == 1
       and one("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=?",
               (sid("结节", "n"),)) == 2)
    ck("typically_has 六条边 + 逆类型 typical_of 成对登记",
       one("SELECT COUNT(*) FROM relations WHERE relation='typically_has'") == 6
       and one("SELECT COUNT(*) FROM relation_types WHERE relation='typically_has' "
               "AND inverse_of='typical_of' AND is_transitive=0") == 1
       and one("SELECT COUNT(*) FROM relation_types WHERE relation='typical_of' "
               "AND inverse_of='typically_has'") == 1)
    ck("typically_has 零闭包 / has_property 派生 2 行闭包（is_transitive 语义差异的真实行为）",
       one("SELECT COUNT(*) FROM closure WHERE relation='typically_has'") == 0
       and one("SELECT COUNT(*) FROM closure WHERE relation='has_property' AND descendant=?",
               (sid("结节", "n"),)) == 2)
    ck("影像义素 = 12 行（形状/边缘/大小各 4）",
       one("SELECT COUNT(*) FROM sense_features WHERE sense_id BETWEEN 4040 AND 4076") == 12)
    ck("跨库边 信号 --domain_specialization--> 强化特征（node 3104）",
       one("SELECT COUNT(*) FROM relations WHERE relation='domain_specialization' "
           "AND src_kind='sense' AND src_id=? AND dst_kind='node' AND dst_id=3104",
           (sid("信号", "n"),)) == 1)

    return fails


def main(argv: list[str]) -> int:
    global DB_PATH

    keep = "--keep" in argv
    positional = [a for a in argv[1:] if not a.startswith("-")]
    if positional:
        DB_PATH = pathlib.Path(positional[0]).expanduser().resolve()

    print("=" * 78)
    print("Logos 建库引导脚本  apply_seeds.py")
    print("=" * 78)
    print(f"  项目根目录 : {ROOT}")
    print(f"  目标库     : {DB_PATH}")
    print(f"  模式       : {'增量叠加（--keep，保留已有库）' if keep else '从零重建'}")

    # ---- 解析 SQL 路径（路径错误立刻退出，不进入写库阶段）----
    files: list[tuple[str, pathlib.Path]] = []
    for label, cands in SQL_FILES:
        resolved = None
        for name in cands:
            try:
                resolved = resolve_sql(name)
                break
            except FileNotFoundError:
                continue
        if resolved is None:
            print(f"\n[路径错误] 找不到 {cands[0]}")
            print("  已搜索以下目录：")
            for d in SEARCH_DIRS:
                print(f"    - {d}")
            print("  提示：本脚本用 pathlib 自动解析，不硬编码分隔符；")
            print("        请把 SQL 文件放在项目根目录或 ROOT/db/ 下。")
            return 2
        files.append((label, resolved))

    # ---- 从零重建 ----
    if not keep:
        for suffix in ("", "-wal", "-shm"):
            p = pathlib.Path(str(DB_PATH) + suffix)
            if p.exists():
                p.unlink()
                print(f"  已删除旧库文件 : {p.name}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn: sqlite3.Connection | None = None
    try:
        # isolation_level=None：显式事务交给 SQL 文件自己管（见模块 docstring 要点 2）
        conn = sqlite3.connect(str(DB_PATH), isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")         # 连接后第一件事
        conn.execute("PRAGMA recursive_triggers = ON")   # 闭包触发器依赖级联
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        print(f"  PRAGMA     : foreign_keys=ON  recursive_triggers=ON  journal_mode={mode}")
        print()

        print("应用 SQL 文件：")
        for label, path in files:
            print(f"  {label}{path.name:<26}", end="", flush=True)
            total, _done = apply_file(conn, path)
            print(f" {total:>4} 条语句  OK")

        print()
        print("行数速览：")
        for t in ("senses", "relations", "closure", "utterances", "words"):
            print(f"  {t:<12} {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>6}")
        print(f"  {'relations(按类型)':<12} " + ", ".join(
            f"{r[0]}={r[1]}" for r in conn.execute(
                "SELECT relation, COUNT(*) FROM relations GROUP BY relation ORDER BY 2 DESC")))

        print()
        print("-" * 78)
        print("自检：")
        fails = selfcheck(conn)

        print("-" * 78)
        if fails:
            print(f"结果: FAIL {fails}（库已建好，但结构自检未全绿）")
            return 1
        print("结果: 全部自检通过（FAIL 0）")
        print("DONE")
        return 0

    except sqlite3.Error as e:
        print(f"\n[中止] {type(e).__name__}: {e}")
        if conn is not None:
            try:
                conn.rollback()
                print("  已回滚未提交事务。")
            except sqlite3.Error:                             # pragma: no cover
                pass
        print("  非零退出码 3。库可能处于部分写入状态；修复后重跑本脚本（默认从零重建）。")
        return 3
    except SystemExit as e:
        print(e.code if isinstance(e.code, str) else "已中止")
        return 4
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv))
