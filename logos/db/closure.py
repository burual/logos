#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logos 传递闭包维护层（``closure`` 表）。

本模块是 **唯一** 允许直接读写 ``closure`` 表的应用层入口。它与
``schema_v3.sql`` §13.2/§13.3 的两个触发器形成互补：

===============  ==========================================  ==========================
路径             触发者                                        适用场景
===============  ==========================================  ==========================
增量（触发器）    ``tr_closure_edge_ai`` / ``_ad``             用户手工增删单条关系
增量（Python）    :func:`incremental_update`                   触发器被关（mode≠incremental）时
全量（Python）    :func:`rebuild_closure`                      批量导入后一次性重建
全量（DB 队列）   :func:`queue_rebuild` → ``tr_closure_rebuild_ai``  想在数据库内完成重建
===============  ==========================================  ==========================

方向约定（**务必先读** ``docs/CLOSURE_DIRECTION.md``）
-----------------------------------------------------
::

    relations 边：   src(下位/具体)  --relation-->  dst(上位/一般)
    closure  行：   (ancestor = 上位/一般, descendant = 下位/具体, depth)

**行的方向 = 边的反向**。因此：

* 在 ``relations`` 里，``src_id`` 是**更具体**的一端，``dst_id`` 是**更一般**的一端；
* 在 ``closure`` 里 ``ancestor`` 是**更一般**的一端，``descendant`` 是**更具体**的一端。

本模块所有 SQL 都通过「问谁的上位就约束 ``descendant``、问谁的下位就约束
``ancestor``」这一口径书写，函数级带测试（``validate_closure.py`` STEP 2）锁定。

不变量
------
:func:`validate_closure_integrals` 默认校验 I1–I5（``docs/CLOSURE_DIRECTION.md`` §5）；
``include_differential=True`` 时额外做 I6「增量 ≡ 全量重建」的差分对拍，
``include_policy=True`` 时额外做 P1/P2 两条策略一致性检查。

SQLite 事实（本机 3.53.1 实测，见 ``docs/CLOSURE_LAYER.md``）
-------------------------------------------------------------
* **触发器体内**不能写前导式 ``WITH ... INSERT``（报 ``near "INSERT": syntax error``），
  必须把递归 CTE 嵌进 ``FROM`` 子查询；而 ``INSERT ... WITH ... SELECT`` 在触发器内是合法的。
* **Python 层**无此限制，前导式 ``WITH RECURSIVE ... INSERT INTO ... SELECT`` 可正常执行，
  因此本模块的重建走前导式写法（更贴合「先算闭包、再落表」的语义）。
* GROUP BY 时使用「裸列 + ``MIN()``」，裸列取自取到最小值的那一行 →
  一次拿到「最短深度」与其对应见证边 ``first_edge``。

维护代价（实测，``bench_closure.py``）
--------------------------------------
插入单条边的闭包代价 ≈ ``(1 + |src 的下位子树|) × (1 + |dst 的上位链|)``，
根节点可达数百毫秒。**批量导入必须** :func:`bulk_import_mode_off` → 导入 →
:func:`rebuild_closure` → :func:`bulk_import_mode_on`（或直接用 :func:`bulk_import` 上下文）。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, NamedTuple, Sequence

from .connection import DatabaseError, transaction

__all__ = [
    # 异常
    "ClosureError",
    "ClosureIntegrityError",
    # 结果类型
    "ClosureEntry",
    "InheritedProperty",
    "Violation",
    "RebuildReport",
    "IncrementalReport",
    "BulkState",
    # 核心 API
    "rebuild_closure",
    "incremental_update",
    "get_ancestors",
    "get_descendants",
    "get_property_inheritance",
    "validate_closure_integrals",
    "inspect_closure_integrals",
    "bulk_import_mode_on",
    "bulk_import_mode_off",
    "bulk_import",
    # 便利函数
    "iter_ancestors",
    "iter_descendants",
    "get_ancestor_ids",
    "get_descendant_ids",
    "queue_rebuild",
    "closure_stats",
    # 常量
    "HIERARCHY_RELATION",
    "DEFAULT_PROPERTY_RELATION",
    "MAINTENANCE_MODES",
    "ACTIVE_MODES",
    "CLOSURE_TRIGGERS",
]

logger = logging.getLogger("logos.db.closure")


# ===========================================================================
# 1. 异常
# ===========================================================================
class ClosureError(DatabaseError):
    """闭包维护失败（边不存在、未知关系、无策略行等）。"""


class ClosureIntegrityError(ClosureError):
    """不变量校验未通过。

    ``violations`` 携带每一条被打破的不变量及其**具体违反行**（已截断）。
    """

    def __init__(self, message: str, violations: Sequence["Violation"] = ()) -> None:
        self.violations = list(violations)
        detail = "；".join(f"{v.invariant}({v.count}行)" for v in self.violations)
        super().__init__(f"{message} 违反：{detail}" if detail else message)


# ===========================================================================
# 2. 常量
# ===========================================================================
#: 属性继承所依赖的层级关系（``docs/CLOSURE_DIRECTION.md`` §4）
HIERARCHY_RELATION = "is_a"

#: 属性继承默认使用的属性关系
DEFAULT_PROPERTY_RELATION = "has_property"

#: ``closure_policy.mode`` 的三档取值
MAINTENANCE_MODES = ("incremental", "batch", "off")

#: 会产出并维护物化闭包的两档（``off`` 表示不物化）
ACTIVE_MODES = ("incremental", "batch")

#: 与闭包增量/删边维护相关的触发器名（drop 策略用）
CLOSURE_TRIGGERS = ("tr_closure_edge_ai", "tr_closure_edge_ad")

#: 「两侧端点都必须是义项」这一条件的口径，避免各处手写不一致
_SENSE_EDGE = "src_kind='sense' AND dst_kind='sense'"


# ===========================================================================
# 3. 结果类型
# ===========================================================================
class ClosureEntry(NamedTuple):
    """闭包查询的一行：某节点到目标节点的一个上位/下位配对。

    :param node_id:    对端节点 id（查上位时是 ancestor，查下位时是 descendant）
    :param depth:      最短路径长度
    :param relation:   关系名
    :param first_edge: 见证边 ``relations.relation_id``（弱引用，可能为 NULL）
    """

    node_id: int
    depth: int
    relation: str
    first_edge: int | None


class InheritedProperty(NamedTuple):
    """属性继承结果。

    :param property_id: 属性（``has_property`` 的 ``dst_id``）
    :param distance:    声明该属性的概念距查询概念的最短层级距离（0 = 概念自身）
    :param declared_by: 声明该属性的概念 id（距离最近的那个）
    """

    property_id: int
    distance: int
    declared_by: int


@dataclass(frozen=True)
class Violation:
    """一条被打破的不变量。

    :param invariant:   不变量编号（I1–I6 / P1–P2）
    :param description: 该不变量的含义
    :param count:       违反行总数
    :param sample:      具体违反行（最多 :data:`_DETAIL_LIMIT` 条）
    :param sql:         定位用的检查 SQL
    """

    invariant: str
    description: str
    count: int
    sample: list[tuple[Any, ...]]
    sql: str

    def render(self, limit: int = 5) -> str:
        head = f"[{self.invariant}] {self.description} —— 违反 {self.count} 行"
        rows = "\n".join(f"      {r}" for r in self.sample[:limit])
        more = "" if self.count <= min(limit, len(self.sample)) else f"\n      …（共 {self.count} 行）"
        return f"{head}\n{rows}{more}" if rows else head


@dataclass
class RebuildReport:
    """:func:`rebuild_closure` 的执行报告。"""

    rebuilt: dict[str, int] = field(default_factory=dict)
    cleared: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def total_rows(self) -> int:
        return sum(self.rebuilt.values())

    def render(self) -> str:
        lines = [f"闭包重建完成：{len(self.rebuilt)} 个关系 / {self.total_rows} 行"
                 f"（{self.elapsed_ms:.1f} ms）"]
        for rel, n in sorted(self.rebuilt.items()):
            lines.append(f"  · {rel:<16} {n:>7} 行")
        if self.cleared:
            lines.append(f"  清空（mode=off）：{', '.join(self.cleared)}")
        if self.skipped:
            lines.append(f"  跳过（无策略行）：{', '.join(self.skipped)}")
        return "\n".join(lines)


@dataclass
class IncrementalReport:
    """:func:`incremental_update` 的执行报告。"""

    relation: str
    source_id: int
    target_id: int
    first_edge: int
    #: 公式 ``(1+|src 下位子树|) × (1+|dst 上位链|)`` 给出的受影响对数上界
    affected_pairs: int
    #: 实际写入（插入或更新）的闭包行数
    written: int
    elapsed_ms: float

    def render(self) -> str:
        return (f"增量闭包 {self.relation}: ({self.source_id} → {self.target_id}) "
                f"受影响上界 {self.affected_pairs} 对，写入 {self.written} 行，"
                f"见证边 #{self.first_edge}，耗时 {self.elapsed_ms:.2f} ms")


@dataclass
class BulkState:
    """批量导入模式的状态快照，供 :func:`bulk_import_mode_on` 精确恢复。"""

    strategy: str
    #: relation -> 进入批量模式前的 ``closure_policy.mode``
    policy: dict[str, str] = field(default_factory=dict)
    #: 被 DROP 的触发器：name -> 原始 CREATE 语句（strategy='drop' 时非空）
    dropped_triggers: dict[str, str] = field(default_factory=dict)
    #: 本次涉及的关系清单（None 表示全部 active 关系）
    relations: list[str] | None = None

    def render(self) -> str:
        parts = [f"strategy={self.strategy}"]
        if self.policy:
            parts.append("policy=" + ", ".join(f"{k}:{v}" for k, v in sorted(self.policy.items())))
        if self.dropped_triggers:
            parts.append("dropped=" + ",".join(sorted(self.dropped_triggers)))
        return "BulkState(" + " ".join(parts) + ")"


_DETAIL_LIMIT = 20


# ===========================================================================
# 4. 内部工具
# ===========================================================================
def _as_tuple(row: Any) -> tuple[Any, ...]:
    """把 ``sqlite3.Row`` / tuple / list 统一成 tuple，屏蔽 row_factory 差异。"""
    if isinstance(row, tuple):
        return row
    return tuple(row)


def _policy_map(conn: Any) -> dict[str, tuple[str, int]]:
    """读出 ``closure_policy``：relation -> (mode, max_depth)。"""
    rows = conn.execute("SELECT relation, mode, max_depth FROM closure_policy").fetchall()
    return {_as_tuple(r)[0]: (_as_tuple(r)[1], int(_as_tuple(r)[2])) for r in rows}


def _relation_info(conn: Any, relation: str) -> tuple[int, int]:
    """返回 ``(is_transitive, max_depth)``；关系不存在时抛 :class:`ClosureError`。"""
    row = conn.execute(
        "SELECT is_transitive, max_depth FROM relation_types WHERE relation = ?",
        (relation,),
    ).fetchone()
    if row is None:
        raise ClosureError(f"未知关系 {relation!r}（relation_types 中不存在）")
    t = _as_tuple(row)
    return int(t[0]), int(t[1])


def _max_depth_for(conn: Any, relation: str, policy: dict[str, tuple[str, int]] | None = None) -> int:
    """取某关系的深度上限：优先 ``closure_policy.max_depth``，退回 ``relation_types.max_depth``。"""
    if policy is None:
        policy = _policy_map(conn)
    if relation in policy:
        return policy[relation][1]
    _, fallback = _relation_info(conn, relation)
    logger.warning("关系 %r 无 closure_policy 行，退回 relation_types.max_depth=%d"
                   "（docs/CLOSURE_DIRECTION.md §6.5：应显式补策略行）", relation, fallback)
    return fallback


def _transitive_relations(conn: Any) -> list[str]:
    """所有 ``is_transitive = 1`` 的关系（潜在需要物化闭包的关系）。"""
    rows = conn.execute(
        "SELECT relation FROM relation_types WHERE is_transitive = 1 ORDER BY relation"
    ).fetchall()
    return [_as_tuple(r)[0] for r in rows]


def _active_relations(conn: Any) -> list[str]:
    """所有 ``mode IN ('incremental','batch')`` 的关系（真正会被物化的关系）。"""
    rows = conn.execute(
        "SELECT relation FROM closure_policy WHERE mode <> 'off' ORDER BY relation"
    ).fetchall()
    return [_as_tuple(r)[0] for r in rows]


@contextmanager
def _atomic(conn: Any) -> Iterator[Any]:
    """原子块：无外层事务时 ``BEGIN IMMEDIATE``，有外层事务时退化为 SAVEPOINT。

    复用连接层 :func:`logos.db.connection.transaction`，保证与全项目一致的
    事务语义（``allow_nested=True`` 时内部自动改用 SAVEPOINT）。
    """
    with transaction(conn, allow_nested=True):
        yield conn


# ---------------------------------------------------------------------------
# 4.1 批量模式的状态寄存（对原生 sqlite3.Connection 也可用）
# ---------------------------------------------------------------------------
_BULK_STATE: dict[int, BulkState] = {}
try:  # pragma: no cover - 取决于运行期是否支持 weakref
    import weakref

    _BULK_STATE_WEAK: "weakref.WeakValueDictionary | None" = weakref.WeakKeyDictionary()
except Exception:  # pragma: no cover
    _BULK_STATE_WEAK = None


def _bulk_state_store(conn: Any, state: BulkState) -> None:
    if _BULK_STATE_WEAK is not None:
        try:
            _BULK_STATE_WEAK[conn] = state
            return
        except TypeError:
            pass  # 原生 sqlite3.Connection 可能不可弱引用
    _BULK_STATE[id(conn)] = state


def _bulk_state_pop(conn: Any) -> BulkState | None:
    if _BULK_STATE_WEAK is not None:
        try:
            return _BULK_STATE_WEAK.pop(conn)  # type: ignore[union-attr]
        except (KeyError, TypeError):
            pass
    return _BULK_STATE.pop(id(conn), None)


# ---------------------------------------------------------------------------
# 4.2 SQL 片段
# ---------------------------------------------------------------------------
# 全量重建：前导式 WITH RECURSIVE（Python 层合法；触发器体内不合法）。
# 语义：从每条边播种，沿 is_a 向上延伸，visited 防环，GROUP BY 取最短并取回见证边。
_REBUILD_SQL = f"""
WITH RECURSIVE closure_build(down_id, up_id, depth, first_edge, visited) AS (
    -- 播种：一条边 下位(down=src) --relation--> 上位(up=dst)
    SELECT r.src_id, r.dst_id, 1, r.relation_id,
           ',' || r.src_id || ',' || r.dst_id || ','
      FROM relations r
     WHERE r.relation = ? AND {_SENSE_EDGE}
    UNION ALL
    -- 延伸：从当前上位继续向上找更一般的节点
    SELECT cb.down_id, r.dst_id, cb.depth + 1, cb.first_edge,
           cb.visited || r.dst_id || ','
      FROM closure_build cb
      JOIN relations r ON r.src_id = cb.up_id
     WHERE r.relation = ? AND {_SENSE_EDGE}
       AND cb.depth < ?
       AND instr(cb.visited, ',' || r.dst_id || ',') = 0
)
INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
SELECT ?, cb.up_id, cb.down_id, MIN(cb.depth), cb.first_edge
  FROM closure_build cb
 WHERE cb.up_id <> cb.down_id
 GROUP BY cb.down_id, cb.up_id
"""

# 在线 oracle：与 _REBUILD_SQL 同构，但只读 relations，不复用物化闭包。
# 列名用 desc_id / anc_id 明确语义，避免 "cur/anc" 这种一眼看不出方向的命名：
#   desc_id = 边的 src 端 = 下位/具体；anc_id = 沿 src_id 向上走到的 = 上位/一般
# 输出 (descendant=desc_id, ancestor=anc_id, depth)，与 closure 表列序一致。
_ORACLE_CTE = f"""
WITH RECURSIVE w(desc_id, anc_id, d, visited) AS (
    SELECT r.src_id, r.dst_id, 1, ',' || r.src_id || ',' || r.dst_id || ','
      FROM relations r
     WHERE r.relation = ? AND {_SENSE_EDGE}
    UNION ALL
    SELECT w.desc_id, r.dst_id, w.d + 1, w.visited || r.dst_id || ','
      FROM w JOIN relations r ON r.src_id = w.anc_id
     WHERE r.relation = ? AND {_SENSE_EDGE}
       AND instr(w.visited, ',' || r.dst_id || ',') = 0
       AND w.d < ?
)
"""


# ===========================================================================
# 5. 全量重建
# ===========================================================================
def _rebuild_one(conn: Any, relation: str, max_depth: int) -> int:
    """重建单个关系的闭包：先 DELETE 再 INSERT（不使用触发器）。

    :returns: 写入的闭包行数

    .. note::
        计数用 ``Connection.total_changes`` 差值，**不能用** ``cursor.rowcount``：
        CPython 的 ``sqlite3`` 按语句首个关键字判定类型，前导式为 ``WITH`` 的
        ``WITH ... INSERT`` 不被识别为 DML，``rowcount`` 恒为 -1（实测踩坑）。
    """
    conn.execute("DELETE FROM closure WHERE relation = ?", (relation,))
    before = conn.total_changes
    conn.execute(_REBUILD_SQL, (relation, relation, max_depth, relation))
    return int(conn.total_changes - before)


def rebuild_closure(conn: Any, relation: str | None = None) -> RebuildReport:
    """全量重建闭包表（幂等，可重复调用）。

    :param conn: 数据库连接（推荐 :func:`logos.db.get_connection` 的产物）
    :param relation: ``None`` → 重建**全部** ``is_transitive = 1`` 的关系；
        给定关系名 → 只重建该关系。
    :returns: :class:`RebuildReport`

    行为
    ----
    * 对每个目标关系：``mode='off'`` 的关系**清空**其闭包行（off 即不物化，
      保留陈旧行会让 I2/方向判断静默出错）；``mode ∈ {incremental,batch}``
      的关系执行 DELETE + 递归 CTE INSERT。
    * 缺少 ``closure_policy`` 行的关系记入 ``skipped`` 并告警（§6.5：不得依赖隐式默认）。
    * 整个过程包在一个事务里（外层已有事务时走 SAVEPOINT）。
    * **不使用** ``closure_rebuild_queue`` 触发器，本函数是纯 Python 实现，
      两路径结果一致（由 I6 差分对拍锁定）。

    示例
    ----
    ::

        report = rebuild_closure(conn)                 # 全量
        report = rebuild_closure(conn, "is_a")         # 单关系
        print(report.render())
    """
    policy = _policy_map(conn)

    if relation is not None:
        _relation_info(conn, relation)  # 触发未知关系报错
        targets = [relation]
    else:
        targets = _transitive_relations(conn)

    report = RebuildReport()
    t0 = time.perf_counter()
    with _atomic(conn):
        for rel in targets:
            if rel not in policy:
                report.skipped.append(rel)
                logger.warning("重建跳过 %r：无 closure_policy 行（静默不物化的风险）", rel)
                continue
            mode, max_depth = policy[rel]
            if mode == "off":
                conn.execute("DELETE FROM closure WHERE relation = ?", (rel,))
                report.cleared.append(rel)
                continue
            report.rebuilt[rel] = _rebuild_one(conn, rel, max_depth)
    report.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info("重建完成：%d 关系 / %d 行 / %.1f ms",
                len(report.rebuilt), report.total_rows, report.elapsed_ms)
    return report


def queue_rebuild(conn: Any, relation: str) -> None:
    """走 **数据库内** 重建：往 ``closure_rebuild_queue`` 插一行，由触发器完成。

    与 :func:`rebuild_closure` 等价，区别是重建逻辑在 SQLite 侧执行，
    适合「只想给数据库发个指令」的场景（例如运维脚本、其他语言的客户端）。
    触发器 ``tr_closure_rebuild_ai`` 执行后会自动删除队列行。
    """
    conn.execute(
        "INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at) "
        "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
        (relation,),
    )


# ===========================================================================
# 6. 增量更新
# ===========================================================================
# 单条边插入后手工补闭包。与触发器 tr_closure_edge_ai 同构：
#   · lo: 从 src 出发沿"上位→下位"反向收集 src 的下位子树（含 src 自身）
#   · hi: 从 dst 出发沿"下位→上位"收集 dst 的上位链（含 dst 自身）
#   · 新配对 (ancestor=H.y, descendant=L.x)，depth = L.d + 1 + H.d
#   · 深度上限由最终 WHERE 卡口（v3.0.1 修正），lo/hi 只做剪枝
#   · UPSERT 中 first_edge 必须写在 depth 之前（SET 从左到右求值）
_INCREMENTAL_SQL = f"""
INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
SELECT ?, H.y, L.x, L.d + 1 + H.d, ?
  FROM (WITH RECURSIVE lo(x, d, w) AS (
            SELECT ?, 0, ?
            UNION
            SELECT cl.descendant, lo.d + cl.depth, cl.first_edge
              FROM closure cl JOIN lo ON cl.ancestor = lo.x
             WHERE cl.relation = ? AND cl.depth > 0
               AND lo.d + cl.depth < ?
        ) SELECT x, d, w FROM lo) AS L
  CROSS JOIN (WITH RECURSIVE hi(y, d) AS (
            SELECT ?, 0
            UNION
            SELECT cl.ancestor, hi.d + cl.depth
              FROM closure cl JOIN hi ON cl.descendant = hi.y
             WHERE cl.relation = ? AND cl.depth > 0
               AND hi.d + cl.depth < ?
        ) SELECT y, d FROM hi) AS H
 WHERE L.x <> H.y
   AND L.d + 1 + H.d <= ?
ON CONFLICT(relation, ancestor, descendant) DO UPDATE SET
    first_edge = CASE WHEN excluded.depth < depth THEN excluded.first_edge ELSE first_edge END,
    depth      = MIN(depth, excluded.depth)
"""


def incremental_update(conn: Any, source_id: int, target_id: int, relation: str) -> IncrementalReport:
    """插入一条边之后，**手工**把该边带来的闭包增量补进 ``closure``。

    :param conn: 数据库连接
    :param source_id: 边的 ``src_id``（**下位/具体** 端）
    :param target_id: 边的 ``dst_id``（**上位/一般** 端）
    :param relation: 关系名
    :returns: :class:`IncrementalReport`

    何时需要它
    ----------
    * ``mode='incremental'`` 时触发器 ``tr_closure_edge_ai`` **已经**做了同样的维护，
      再调用本函数是幂等的（UPSERT 取 ``MIN(depth)``），不会出错；
    * ``mode ∈ {batch, off}``（批量导入、或刻意关闭触发）时，触发器**不会**动作，
      此时由本函数承担增量维护；
    * 传入的边必须已存在于 ``relations``（本函数据其取 ``first_edge`` 见证边），
      否则抛 :class:`ClosureError`。

    数学形态
    --------
    受影响配对 = ``{src 及其下位子树} × {dst 及其上位链}``，
    规模即 ``(1 + |src 的下位子树|) × (1 + |dst 的上位链|)``
    （``docs/CLOSURE_DIRECTION.md`` §7 实测公式）。
    """
    policy = _policy_map(conn)
    max_depth = _max_depth_for(conn, relation, policy)

    edge = conn.execute(
        f"SELECT relation_id FROM relations "
        f"WHERE relation = ? AND src_id = ? AND dst_id = ? AND {_SENSE_EDGE} "
        f"ORDER BY relation_id LIMIT 1",
        (relation, source_id, target_id),
    ).fetchone()
    if edge is None:
        raise ClosureError(
            f"边不存在：({source_id} --{relation}--> {target_id})。"
            f"请先 INSERT INTO relations 再调用 incremental_update。"
        )
    first_edge = int(_as_tuple(edge)[0])

    # 受影响对数上界 = (1+|src 下位子树|) × (1+|dst 上位链|)
    sub = int(conn.execute(
        "SELECT COUNT(*) FROM closure WHERE relation = ? AND ancestor = ?",
        (relation, source_id),
    ).fetchone()[0])
    chain = int(conn.execute(
        "SELECT COUNT(*) FROM closure WHERE relation = ? AND descendant = ?",
        (relation, target_id),
    ).fetchone()[0])
    affected = (1 + sub) * (1 + chain)

    t0 = time.perf_counter()
    with _atomic(conn):
        before = conn.total_changes
        conn.execute(
            _INCREMENTAL_SQL,
            (relation, first_edge, source_id, first_edge, relation, max_depth,
             target_id, relation, max_depth, max_depth),
        )
        written = int(conn.total_changes - before)
    elapsed = (time.perf_counter() - t0) * 1000.0

    report = IncrementalReport(
        relation=relation,
        source_id=source_id,
        target_id=target_id,
        first_edge=first_edge,
        affected_pairs=affected,
        written=written,
        elapsed_ms=elapsed,
    )
    logger.debug("%s", report.render())
    return report


# ===========================================================================
# 7. 上位 / 下位查询
# ===========================================================================
def _walk_sql(
    node_id: int,
    relation: str | None,
    max_depth: int | None,
    *,
    up: bool,
) -> tuple[str, list[Any]]:
    """构造上位/下位查询。

    * 取上位（``up=True``）：约束 ``descendant = node``，返回 ``ancestor``
      → 命中索引 ``idx_closure_relation_desc(relation, descendant, depth)``
    * 取下位（``up=False``）：约束 ``ancestor = node``，返回 ``descendant``
      → 命中主键前缀 ``PK(relation, ancestor, descendant)``
    """
    params: list[Any] = []
    if up:
        where, col = "c.descendant = ?", "c.ancestor"
        params.append(node_id)
    else:
        where, col = "c.ancestor = ?", "c.descendant"
        params.append(node_id)

    conds = [where]
    if relation is not None:
        conds.append("c.relation = ?")
        params.append(relation)
    if max_depth is not None:
        conds.append("c.depth <= ?")
        params.append(int(max_depth))

    sql = (f"SELECT {col} AS node_id, c.depth, c.relation, c.first_edge "
           f"FROM closure c WHERE " + " AND ".join(conds) +
           f" ORDER BY c.depth ASC, {col} ASC")
    return sql, params


def iter_ancestors(
    conn: Any,
    node_id: int,
    relation: str | None = None,
    max_depth: int | None = None,
) -> Iterator[ClosureEntry]:
    """**生成器** 版 :func:`get_ancestors`：大结果集逐行 ``yield``，不全量驻留内存。

    取 ``node_id`` 的全部上位节点（沿 ``relation`` 向上走），按 ``depth`` 升序。
    """
    sql, params = _walk_sql(node_id, relation, max_depth, up=True)
    for row in conn.execute(sql, params):
        t = _as_tuple(row)
        yield ClosureEntry(int(t[0]), int(t[1]), t[2], None if t[3] is None else int(t[3]))


def iter_descendants(
    conn: Any,
    node_id: int,
    relation: str | None = None,
    max_depth: int | None = None,
) -> Iterator[ClosureEntry]:
    """**生成器** 版 :func:`get_descendants`：大结果集逐行 ``yield``。

    取 ``node_id`` 的全部下位节点（含所有子孙），按 ``depth`` 升序。
    """
    sql, params = _walk_sql(node_id, relation, max_depth, up=False)
    for row in conn.execute(sql, params):
        t = _as_tuple(row)
        yield ClosureEntry(int(t[0]), int(t[1]), t[2], None if t[3] is None else int(t[3]))


def get_ancestors(
    conn: Any,
    node_id: int,
    relation: str | None = None,
    max_depth: int | None = None,
) -> list[ClosureEntry]:
    """取 ``node_id`` 的全部**上位**节点，按 ``depth`` 升序返回。

    :param relation: 关系名；``None`` 表示跨全部关系。
    :param max_depth: 只取 ``depth <= max_depth`` 的上位；``None`` 不限。
    :returns: ``list[ClosureEntry]``，每项含 ``node_id / depth / relation / first_edge``。

    .. note::
        结果集可能很大。需要流式处理时用 :func:`iter_ancestors`（生成器）；
        只要 id 列表时用 :func:`get_ancestor_ids`。

    方向自检：本函数查的是「我的上位」，因此约束的是 **``descendant``**。
    """
    return list(iter_ancestors(conn, node_id, relation, max_depth))


def get_descendants(
    conn: Any,
    node_id: int,
    relation: str | None = None,
    max_depth: int | None = None,
) -> list[ClosureEntry]:
    """取 ``node_id`` 的全部**下位**节点（含所有子孙），按 ``depth`` 升序返回。

    :returns: ``list[ClosureEntry]``

    .. note::
        大结果集请用 :func:`iter_descendants`；只要 id 列表时用 :func:`get_descendant_ids`。

    方向自检：本函数查的是「我的下位」，因此约束的是 **``ancestor``**。
    """
    return list(iter_descendants(conn, node_id, relation, max_depth))


def get_ancestor_ids(
    conn: Any,
    node_id: int,
    relation: str | None = None,
    max_depth: int | None = None,
) -> list[int]:
    """只取上位节点的 id 列表（按 depth 升序）。"""
    return [e.node_id for e in iter_ancestors(conn, node_id, relation, max_depth)]


def get_descendant_ids(
    conn: Any,
    node_id: int,
    relation: str | None = None,
    max_depth: int | None = None,
) -> list[int]:
    """只取下位节点的 id 列表（按 depth 升序）。"""
    return [e.node_id for e in iter_descendants(conn, node_id, relation, max_depth)]


# ===========================================================================
# 8. 属性继承（推理引擎核心调用）
# ===========================================================================
def get_property_inheritance(
    conn: Any,
    concept_id: int,
    property_relation: str = DEFAULT_PROPERTY_RELATION,
    *,
    hierarchy_relation: str = HIERARCHY_RELATION,
    max_depth: int | None = None,
) -> list[InheritedProperty]:
    """属性继承查询：取 ``concept_id`` **自身及其全部上位**声明的属性。

    这正是 ``docs/CLOSURE_DIRECTION.md`` §4 的规则——属性挂在上位概念上、下位概念继承::

        lineage = {concept_id(d=0)} ∪ {concept 的上位链, 距离 = closure.depth}
        结果   = lineage 中每个概念沿 property_relation 指向的 dst_id

    :param concept_id: 查询的概念（``senses.sense_id``）
    :param property_relation: 属性关系名，默认 ``has_property``
    :param hierarchy_relation: 继承所依赖的层级关系，默认 ``is_a``
    :param max_depth: 只沿 ``depth <= max_depth`` 的上位继承；``None`` 不限
    :returns: ``list[InheritedProperty]``，按 ``distance`` 升序；同一属性取最近声明者。

    .. warning::
        方向若写反（``r.src_id = c.descendant`` 而非 ``c.ancestor``），会去取**子类**属性：
        语法正确、结果荒谬。本函数的 ``descendant = concept`` 口径即 §4 的正确形式，
        已由 ``validate_closure.py`` STEP 2 的推理链用例锁定。

    示例
    ----
    ::

        # 老虎 → 猫科 → 哺乳动物 → 动物；属性挂在"动物"上
        props = get_property_inheritance(conn, tiger_id)
        for p in props:
            print(p.property_id, "距离", p.distance, "声明者", p.declared_by)
    """
    params: list[Any] = [concept_id, hierarchy_relation, concept_id]
    depth_filter = ""
    if max_depth is not None:
        depth_filter = " AND c.depth <= ?"
        params.append(int(max_depth))
    params.append(property_relation)

    sql = f"""
    WITH RECURSIVE lineage(concept_id, dist) AS (
        SELECT ?, 0
        UNION
        SELECT c.ancestor, c.depth
          FROM closure c
         WHERE c.relation = ? AND c.descendant = ?{depth_filter}
    )
    SELECT r.dst_id AS property_id, MIN(l.dist) AS distance, l.concept_id AS declared_by
      FROM lineage l
      JOIN relations r
        ON r.src_id = l.concept_id
       AND r.relation = ?
       AND r.src_kind = 'sense' AND r.dst_kind = 'sense'
     GROUP BY r.dst_id
     ORDER BY distance ASC, property_id ASC
    """
    out: list[InheritedProperty] = []
    for row in conn.execute(sql, params):
        t = _as_tuple(row)
        out.append(InheritedProperty(int(t[0]), int(t[1]), int(t[2])))
    return out


# ===========================================================================
# 9. 不变量校验
# ===========================================================================
# (编号, 含义, 定位 SQL) —— SQL 应「返回 0 行」；返回非空即为违反。
_INVARIANTS: tuple[tuple[str, str, str], ...] = (
    (
        "I1",
        "闭包不含自环（ancestor = descendant）",
        "SELECT c.relation, c.ancestor, c.descendant, c.depth "
        "FROM closure c WHERE c.ancestor = c.descendant",
    ),
    (
        "I2",
        "每条闭包行都能在 relations 上找到同关系的义项边（无孤儿关系）",
        "SELECT c.relation, c.ancestor, c.descendant, c.depth FROM closure c "
        f"WHERE NOT EXISTS (SELECT 1 FROM relations r WHERE r.relation = c.relation AND {_SENSE_EDGE})",
    ),
    (
        "I3",
        "depth=1 的闭包行与 relations 一一对应（方向断言：dst_id=ancestor ∧ src_id=descendant）",
        "SELECT c.relation, c.ancestor, c.descendant, c.depth FROM closure c "
        "WHERE c.depth = 1 AND NOT EXISTS ("
        "  SELECT 1 FROM relations r WHERE r.relation = c.relation "
        f"    AND {_SENSE_EDGE} AND r.src_id = c.descendant AND r.dst_id = c.ancestor)",
    ),
    (
        "I4",
        "first_edge 非空且指向同关系的边",
        "SELECT c.relation, c.ancestor, c.descendant, c.first_edge FROM closure c "
        "WHERE c.first_edge IS NULL OR c.first_edge NOT IN "
        "(SELECT relation_id FROM relations WHERE relation = c.relation)",
    ),
    (
        "I5",
        "闭包深度不超过策略上限 max_depth",
        "SELECT c.relation, c.ancestor, c.descendant, c.depth, p.max_depth "
        "FROM closure c JOIN closure_policy p ON p.relation = c.relation "
        "WHERE c.depth > p.max_depth",
    ),
)

# 附加检查（默认不参与，需显式开启）
_POLICY_CHECKS: tuple[tuple[str, str, str], ...] = (
    (
        "P1",
        "mode=off 或 is_transitive=0 的关系不应残留闭包行（策略漂移）",
        "SELECT c.relation, c.ancestor, c.descendant, c.depth FROM closure c "
        "LEFT JOIN closure_policy p ON p.relation = c.relation "
        "LEFT JOIN relation_types t ON t.relation = c.relation "
        "WHERE COALESCE(p.mode,'off') = 'off' OR COALESCE(t.is_transitive,0) = 0",
    ),
    (
        "P2",
        "is_transitive=1 的关系必须显式拥有 closure_policy 行（缺行会静默无闭包）",
        "SELECT t.relation, t.max_depth FROM relation_types t "
        "WHERE t.is_transitive = 1 AND NOT EXISTS "
        "(SELECT 1 FROM closure_policy p WHERE p.relation = t.relation)",
    ),
)


def _run_check(conn: Any, inv: str, desc: str, sql: str) -> Violation | None:
    """跑一条检查：先计数，非 0 再取样本行。"""
    count = int(conn.execute(f"SELECT COUNT(*) FROM ({sql})", {}).fetchone()[0])
    if count == 0:
        return None
    sample = [_as_tuple(r) for r in conn.execute(f"SELECT * FROM ({sql}) LIMIT ?", (_DETAIL_LIMIT,))]
    return Violation(inv, desc, count, sample, sql)


def _differential_check(conn: Any, relation: str, max_depth: int) -> Violation | None:
    """I6：同关系的物化闭包 vs 在线递归 CTE oracle 的差分对拍。

    使用**独立实现**（只读 relations 的递归 CTE）当 oracle，不拿闭包验闭包。
    输出两类差异：``MISSING``（oracle 有、闭包无）与 ``EXTRA``（闭包有、oracle 无）。
    """
    sql = (
        _ORACLE_CTE
        + """
, oracle(descendant, ancestor, d) AS (
    SELECT desc_id, anc_id, MIN(d) FROM w WHERE desc_id <> anc_id GROUP BY desc_id, anc_id
)
SELECT 'MISSING' AS kind, o.ancestor, o.descendant, o.d FROM oracle o
 WHERE NOT EXISTS (SELECT 1 FROM closure c
                    WHERE c.relation = ? AND c.ancestor = o.ancestor
                      AND c.descendant = o.descendant AND c.depth = o.d)
UNION ALL
SELECT 'EXTRA', c.ancestor, c.descendant, c.depth FROM closure c
 WHERE c.relation = ?
   AND NOT EXISTS (SELECT 1 FROM oracle o
                    WHERE o.ancestor = c.ancestor AND o.descendant = c.descendant AND o.d = c.depth)
"""
    )
    params = (relation, relation, max_depth, relation, relation)
    rows = [_as_tuple(r) for r in conn.execute(sql, params)]
    if not rows:
        return None
    missing = sum(1 for r in rows if r[0] == "MISSING")
    extra = len(rows) - missing
    return Violation(
        "I6",
        f"增量/物化结果与全量重建（oracle）不一致（{relation}：缺 {missing}、多 {extra}）",
        len(rows),
        rows[:_DETAIL_LIMIT],
        sql,
    )


def inspect_closure_integrals(
    conn: Any,
    *,
    include_differential: bool = False,
    include_policy: bool = False,
    relations: Sequence[str] | None = None,
) -> list[Violation]:
    """返回所有被打破的不变量（结构化），空列表表示全部通过。

    :param include_differential: 额外做 I6 差分对拍（逐关系在线递归，较慢）
    :param include_policy: 额外做 P1/P2 策略一致性检查
    :param relations: 只对指定关系做 I6；``None`` → 全部 active 关系
    """
    violations: list[Violation] = []
    for inv, desc, sql in _INVARIANTS:
        v = _run_check(conn, inv, desc, sql)
        if v is not None:
            violations.append(v)

    if include_policy:
        for inv, desc, sql in _POLICY_CHECKS:
            v = _run_check(conn, inv, desc, sql)
            if v is not None:
                violations.append(v)

    if include_differential:
        policy = _policy_map(conn)
        targets = list(relations) if relations is not None else _active_relations(conn)
        for rel in targets:
            if rel not in policy:
                continue
            mode, mx = policy[rel]
            if mode == "off":
                continue
            v = _differential_check(conn, rel, mx)
            if v is not None:
                violations.append(v)

    return violations


def validate_closure_integrals(
    conn: Any,
    *,
    include_differential: bool = False,
    include_policy: bool = False,
    relations: Sequence[str] | None = None,
) -> bool:
    """校验 I1–I5（可选 I6 / P1–P2）不变量。

    :returns: ``True`` 全部通过；``False`` 存在违反。
        返回 ``False`` 时，会把**每一条被打破的不变量及其具体违反行**打到
        ``logos.db.closure`` 日志的 ERROR 级别（并保留在
        :func:`inspect_closure_integrals` 的结构化结果里）。

    .. note::
        默认只跑 I1–I5（快）。CI 里建议 ``include_differential=True`` ——
        I6「增量 ≡ 重建」是唯一能发现「两条路径结果不一致」的检查。
    """
    violations = inspect_closure_integrals(
        conn,
        include_differential=include_differential,
        include_policy=include_policy,
        relations=relations,
    )
    if not violations:
        logger.debug("闭包不变量校验通过（I1–I5%s）",
                     "+I6" if include_differential else "")
        return True

    for v in violations:
        logger.error("闭包不变量被打破 %s", v.render())
    return False


# ===========================================================================
# 10. 批量导入模式
# ===========================================================================
def _read_trigger_sql(conn: Any, names: Sequence[str]) -> dict[str, str]:
    """从 ``sqlite_master`` 取触发器原始 CREATE 语句。"""
    marks = ",".join("?" * len(names))
    rows = conn.execute(
        f"SELECT name, sql FROM sqlite_master WHERE type='trigger' AND name IN ({marks})",
        tuple(names),
    ).fetchall()
    return {_as_tuple(r)[0]: _as_tuple(r)[1] for r in rows if _as_tuple(r)[1]}


def bulk_import_mode_off(
    conn: Any,
    relations: Sequence[str] | None = None,
    *,
    strategy: str = "policy",
) -> BulkState:
    """进入**批量导入模式**：暂停闭包自动维护，避免每条边触发子树笛卡尔积。

    :param relations: 需要暂停的关系；``None`` → 全部 ``mode='incremental'`` 的关系
    :param strategy:

        ``"policy"``（默认，推荐）
            把目标关系的 ``closure_policy.mode`` 置为 ``off``。此时
            ``tr_closure_edge_ai``（条件 ``mode='incremental'``）与
            ``tr_closure_edge_ad``（条件 ``mode<>'off'``）双双失效。**schema 原生、
            可逆、不破坏任何对象定义**。
        ``"drop"``
            真正 ``DROP TRIGGER`` 两个闭包触发器（先备份其 CREATE 语句）。
            连触发器的 ``WHEN`` 判断开销都省掉，适合超大批量；但期间若发生
            意外，触发器会处于「缺失」状态 —— 务必用 :func:`bulk_import` 上下文
            或 try/finally 保证恢复。

    :returns: :class:`BulkState` 快照，务必交回 :func:`bulk_import_mode_on`。

    .. warning::
        ``strategy='drop'`` 会删除 schema 对象，只在确认流程可控时使用；
        默认的 ``policy`` 策略通过 ``mode`` 开关达到同样效果且零风险。
    """
    strategy_norm = strategy.lower()
    if strategy_norm not in ("policy", "drop"):
        raise ClosureError(f"未知 strategy={strategy!r}（可选 'policy' / 'drop'）")

    if relations is None:
        target = _active_relations(conn)
    else:
        target = list(relations)
        for rel in target:
            _relation_info(conn, rel)  # 触发未知关系报错

    state = BulkState(strategy=strategy_norm, relations=None if relations is None else target)

    if strategy_norm == "policy":
        policy = _policy_map(conn)
        for rel in target:
            if rel not in policy:
                continue
            state.policy[rel] = policy[rel][0]
        with _atomic(conn):
            for rel in state.policy:
                conn.execute(
                    "UPDATE closure_policy SET mode='off', "
                    "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE relation = ?",
                    (rel,),
                )
        logger.info("批量导入模式 ON（policy）：暂停 %d 个关系的闭包维护", len(state.policy))
    else:  # drop
        state.dropped_triggers = _read_trigger_sql(conn, CLOSURE_TRIGGERS)
        with _atomic(conn):
            for name in state.dropped_triggers:
                conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        logger.warning("批量导入模式 ON（drop）：已移除触发器 %s",
                       ", ".join(sorted(state.dropped_triggers)) or "（无）")

    _bulk_state_store(conn, state)
    return state


def bulk_import_mode_on(
    conn: Any,
    state: BulkState | None = None,
    *,
    rebuild: bool = True,
    relations: Sequence[str] | None = None,
) -> int:
    """退出**批量导入模式**：恢复闭包自动维护，并按需全量重建。

    :param state: :func:`bulk_import_mode_off` 的返回值；``None`` 时取连接上暂存的快照
    :param rebuild: 是否在恢复后立即 :func:`rebuild_closure`（推荐 ``True``）
    :param relations: 重建范围；``None`` → 重建 ``state`` 涉及的关系（或全部）
    :returns: 恢复为 ``incremental/batch`` 的关系数

    顺序很重要：**先恢复触发，再重建**。若先重建再恢复，重建期间若有并发写入
    会被漏记；先恢复则最多产生幂等的重复维护（UPSERT 收敛）。
    """
    if state is None:
        state = _bulk_state_pop(conn)
        if state is None:
            raise ClosureError(
                "找不到批量导入状态快照。请把 bulk_import_mode_off() 的返回值显式传入，"
                "或确保在同一连接对象上配对使用。"
            )
    else:
        _bulk_state_pop(conn)  # 清理可能存在的暂存副本

    restored = 0
    if state.strategy == "policy":
        with _atomic(conn):
            for rel, old_mode in state.policy.items():
                conn.execute(
                    "UPDATE closure_policy SET mode=?, "
                    "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE relation = ?",
                    (old_mode, rel),
                )
                restored += 1
        logger.info("批量导入模式 OFF（policy）：恢复 %d 个关系", restored)
    else:  # drop
        with _atomic(conn):
            for name, sql in state.dropped_triggers.items():
                conn.execute(sql)
                restored += 1
        logger.info("批量导入模式 OFF（drop）：重建触发器 %d 个", restored)

    if rebuild:
        if relations is not None:
            targets: Sequence[str] | None = list(relations)
        elif state.relations is not None:
            targets = state.relations
        else:
            targets = None
        report = rebuild_closure(conn, None) if targets is None else _rebuild_many(conn, targets)
        logger.info("批量导入后重建：%s", report.render().replace("\n", " | "))

    return restored


def _rebuild_many(conn: Any, relations: Sequence[str]) -> RebuildReport:
    """对若干关系逐个重建并合并成一份报告。"""
    merged = RebuildReport()
    t0 = time.perf_counter()
    for rel in relations:
        r = rebuild_closure(conn, rel)
        merged.rebuilt.update(r.rebuilt)
        merged.cleared.extend(r.cleared)
        merged.skipped.extend(r.skipped)
    merged.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return merged


@contextmanager
def bulk_import(
    conn: Any,
    relations: Sequence[str] | None = None,
    *,
    strategy: str = "policy",
    rebuild: bool = True,
) -> Iterator[BulkState]:
    """批量导入上下文：进入即暂停维护，退出即恢复（并重建）。

    ::

        with bulk_import(conn):                       # 进入：暂停闭包维护
            for edge in huge_edge_list:
                conn.execute("INSERT INTO relations(...) VALUES(?,?,?)", edge)
        # 退出：恢复维护 + 全量重建 + 不变量自检

    **异常安全**：即使块内抛错，也会先恢复触发器（``rebuild=False``），
    避免把库留在「闭包静默不更新」的状态；异常原样抛出。
    """
    state = bulk_import_mode_off(conn, relations, strategy=strategy)
    try:
        yield state
    except BaseException:
        bulk_import_mode_on(conn, state, rebuild=False)
        logger.error("批量导入异常，已恢复闭包维护（未重建，请人工确认数据）")
        raise
    else:
        bulk_import_mode_on(conn, state, rebuild=rebuild)


# ===========================================================================
# 11. 统计（观测用）
# ===========================================================================
def closure_stats(conn: Any, relation: str | None = None) -> dict[str, Any]:
    """闭包表规模概览：总行数、按关系行数、最大深度、覆盖关系数。"""
    if relation is not None:
        base = " WHERE relation = ?"
        params: tuple[Any, ...] = (relation,)
    else:
        base, params = "", ()

    total = int(conn.execute(f"SELECT COUNT(*) FROM closure{base}", params).fetchone()[0])
    max_depth = conn.execute(f"SELECT MAX(depth) FROM closure{base}", params).fetchone()[0]
    by_rel = {
        _as_tuple(r)[0]: int(_as_tuple(r)[1])
        for r in conn.execute(
            "SELECT relation, COUNT(*) FROM closure"
            + (" WHERE relation = ?" if relation else "")
            + " GROUP BY relation ORDER BY relation",
            params,
        )
    }
    return {
        "total_rows": total,
        "max_depth": int(max_depth) if max_depth is not None else 0,
        "relations": len(by_rel),
        "by_relation": by_rel,
    }
