#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``logos/db/closure.py`` 回归校验。

运行：  python validate_closure.py
退出码：0 全绿；1 有失败。

覆盖：
  S0 建库（走正规入口 init_database）
  S1 方向语义（上位/下位、排序、生成器）
  S2 属性继承（**方向断言**：只能继承上位属性，绝不能取到子类属性）
  S3 全量重建（Python 路径 vs DB 队列路径 vs 触发器，三者一致）
  S4 增量更新（与全量重建差分一致、幂等、缺边报错、代价公式）
  S5 不变量检测（I1–I6 / P1–P2 的真实触发）
  S6 批量导入模式（policy / drop 两种策略 + 异常安全 + 上下文管理器）
  S7 索引与查询计划（上位/下位各自命中索引）
  S8 环图剪枝与批量大图（bulk_import + I6 差分）
  S9 统计函数
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from logos.db.connection import get_connection, init_database  # noqa: E402
from logos.db import closure as C  # noqa: E402

SCHEMA = HERE / "schema_v3.sql"
DB = HERE / "logos_validate_closure.db"


# ===========================================================================
# 断言框架
# ===========================================================================
class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []
        self._idx = 0

    def section(self, name: str) -> None:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        self._idx += 1
        if cond:
            self.passed += 1
            print(f"  PASS  [{self._idx:>3}] {name}")
        else:
            self.failed += 1
            self.failures.append(f"{name} :: {detail}")
            print(f"  FAIL  [{self._idx:>3}] {name}   <-- {detail}")
        return bool(cond)

    def eq(self, name: str, got, want) -> bool:
        return self.check(name, got == want, f"got={got!r} want={want!r}")

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n{'=' * 78}")
        print(f"PASS {self.passed} / FAIL {self.failed} / TOTAL {total}")
        if self.failures:
            print("\n失败明细：")
            for f in self.failures:
                print(f"  - {f}")
        print("=" * 78)
        return 0 if self.failed == 0 else 1


ck = Checker()


# ===========================================================================
# 工具
# ===========================================================================
def _rm(path: pathlib.Path) -> None:
    for suf in ("", "-wal", "-shm", "-journal"):
        p = pathlib.Path(str(path) + suf)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


def add_sense(conn, sid: int, pos: str = "n") -> None:
    """造一个可用的 (word, sense) 对；用高 id 段避免与种子冲突。"""
    conn.execute("INSERT OR IGNORE INTO words(word_id, word) VALUES(?,?)", (sid, f"验证词{sid}"))
    conn.execute(
        "INSERT OR IGNORE INTO senses(sense_id, word_id, pos, sense_no, definition) "
        "VALUES(?,?,?,1,?)",
        (sid, sid, pos, f"验证概念{sid}"),
    )


def edge(conn, src: int, dst: int, relation: str = "is_a") -> int:
    """插一条边，返回 relation_id。"""
    cur = conn.execute(
        "INSERT INTO relations(src_id, dst_id, relation) VALUES(?,?,?)", (src, dst, relation)
    )
    return int(cur.lastrowid)


def closure_pairs(conn, relation: str) -> set[tuple[int, int, int]]:
    rows = conn.execute(
        "SELECT ancestor, descendant, depth FROM closure WHERE relation = ?", (relation,)
    ).fetchall()
    return {(int(r[0]), int(r[1]), int(r[2])) for r in rows}


def oracle_pairs(conn, relation: str, max_depth: int) -> set[tuple[int, int, int]]:
    """独立实现在线闭包（只读 relations），当差分 oracle 用。

    递归里 ``desc_id`` 固定为边的 src 端（下位/具体），``anc_id`` 沿 ``src_id`` 向上走
    （上位/一般）；因此输出 ``(desc_id, anc_id, depth)`` 即
    ``(descendant, ancestor, depth)``，与 closure 表列序**一致**，无需对调。
    """
    sql = (
        "WITH RECURSIVE w(desc_id, anc_id, d, visited) AS ("
        "  SELECT r.src_id, r.dst_id, 1, ','||r.src_id||','||r.dst_id||','"
        "    FROM relations r WHERE r.relation=? AND r.src_kind='sense' AND r.dst_kind='sense'"
        "  UNION ALL"
        "  SELECT w.desc_id, r.dst_id, w.d+1, w.visited||r.dst_id||','"
        "    FROM w JOIN relations r ON r.src_id=w.anc_id"
        "   WHERE r.relation=? AND r.src_kind='sense' AND r.dst_kind='sense'"
        "     AND instr(w.visited, ','||r.dst_id||',')=0 AND w.d < ?"
        ") SELECT desc_id, anc_id, MIN(d) FROM w WHERE desc_id <> anc_id GROUP BY desc_id, anc_id"
    )
    rows = conn.execute(sql, (relation, relation, max_depth)).fetchall()
    # (descendant, ancestor, depth) —— 与 closure 表 (ancestor, descendant, depth) 只差列序
    return {(int(r[1]), int(r[0]), int(r[2])) for r in rows}


def trigger_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('tr_closure_edge_ai','tr_closure_edge_ad')"
    ).fetchall()
    return {r[0] for r in rows}


# ===========================================================================
# S0 建库
# ===========================================================================
ck.section("S0 建库（init_database 正规入口）")
_rm(DB)
rep = init_database(DB, reset=True, validate=True)
ck.check("schema 加载成功", rep.schema_loaded, rep.render())
ck.check("integrity_check 通过", rep.integrity_ok)
ck.check("foreign_key_check 通过", rep.foreign_key_ok)
ck.check("种子已加载", bool(rep.seed_loaded), str(rep.seed_loaded))
conn = get_connection(DB, must_exist=True)
ck.check("closure 表存在", conn.table_exists("closure"))
ck.check("closure_policy 有策略行",
         conn.execute("SELECT COUNT(*) FROM closure_policy").fetchone()[0] > 0)
ck.check("闭包触发器齐全", trigger_names(conn) == set(C.CLOSURE_TRIGGERS), str(trigger_names(conn)))

# 高 id 段实体：老虎 900001 → 动物 900002 → 生物 900003；东北虎 900004 → 老虎
TIGER, ANIMAL, BIO, SIBERIAN = 900001, 900002, 900003, 900004
for s in (TIGER, ANIMAL, BIO, SIBERIAN):
    add_sense(conn, s)

# ===========================================================================
# S1 方向语义
# ===========================================================================
ck.section("S1 方向语义（上位/下位）")
edge(conn, TIGER, ANIMAL)      # 老虎 --is_a--> 动物
edge(conn, ANIMAL, BIO)        # 动物 --is_a--> 生物
edge(conn, SIBERIAN, TIGER)    # 东北虎 --is_a--> 老虎

anc = C.get_ancestors(conn, TIGER, "is_a")
ck.eq("老虎的上位 = [动物, 生物]（按 depth）", [e.node_id for e in anc], [ANIMAL, BIO])
ck.eq("老虎上位 depth 序列", [e.depth for e in anc], [1, 2])

desc = C.get_descendants(conn, BIO, "is_a")
ck.eq("生物的下位 = [动物, 老虎, 东北虎]",
      [e.node_id for e in desc], [ANIMAL, TIGER, SIBERIAN])
ck.eq("生物下位 depth 序列", [e.depth for e in desc], [1, 2, 3])

ck.check("get_ancestors 返回 ClosureEntry 且含见证边",
         all(isinstance(e, C.ClosureEntry) and e.first_edge is not None for e in anc))
ck.check("iter_ancestors 是生成器（惰性）",
         hasattr(C.iter_ancestors(conn, TIGER, "is_a"), "__next__"))
ck.eq("list(iter_ancestors) == get_ancestors",
      [e.node_id for e in C.iter_ancestors(conn, TIGER, "is_a")], [ANIMAL, BIO])
ck.eq("get_ancestor_ids", C.get_ancestor_ids(conn, TIGER, "is_a"), [ANIMAL, BIO])
ck.eq("get_descendant_ids", C.get_descendant_ids(conn, BIO, "is_a"), [ANIMAL, TIGER, SIBERIAN])
ck.eq("max_depth 截断", C.get_ancestor_ids(conn, TIGER, "is_a", max_depth=1), [ANIMAL])
ck.eq("relation=None 也能查（跨关系）", C.get_ancestor_ids(conn, TIGER, None), [ANIMAL, BIO])
ck.check("无上位的节点返回空", C.get_ancestors(conn, BIO, "is_a") == [])
ck.check("I3 方向：depth=1 行与 edge 一一对应（老虎,动物）",
         conn.execute(
             "SELECT COUNT(*) FROM closure c WHERE c.relation='is_a' AND c.depth=1 "
             "AND c.ancestor=? AND c.descendant=? AND EXISTS("
             "  SELECT 1 FROM relations r WHERE r.relation='is_a' AND r.src_id=c.descendant AND r.dst_id=c.ancestor)",
             (ANIMAL, TIGER)).fetchone()[0] == 1)

# ===========================================================================
# S2 属性继承（方向断言）
# ===========================================================================
ck.section("S2 属性继承（方向断言）")
P_LIFE, P_CAT, P_SNOW = 900011, 900012, 900013
for s in (P_LIFE, P_CAT, P_SNOW):
    add_sense(conn, s)
edge(conn, ANIMAL, P_LIFE, "has_property")    # 动物 有 生命
edge(conn, TIGER, P_CAT, "has_property")      # 老虎 有 猫科特征
edge(conn, SIBERIAN, P_SNOW, "has_property")  # 东北虎 有 雪地适应

props_tiger = C.get_property_inheritance(conn, TIGER, "has_property")
ids_tiger = {p.property_id for p in props_tiger}
ck.check("老虎继承到自身属性（distance=0，declared_by=自己）",
         any(p.property_id == P_CAT and p.distance == 0 and p.declared_by == TIGER
             for p in props_tiger), str(props_tiger))
ck.check("老虎继承到上位属性（动物·有生命，distance=1）",
         any(p.property_id == P_LIFE and p.distance == 1 and p.declared_by == ANIMAL
             for p in props_tiger), str(props_tiger))
ck.check("★方向断言：老虎**不得**继承下位（东北虎·雪地适应）",
         P_SNOW not in ids_tiger, f"got={ids_tiger}")

props_sib = {p.property_id: p for p in C.get_property_inheritance(conn, SIBERIAN)}
ck.check("东北虎继承三条（自身/老虎/动物），distance = 0/1/2",
         props_sib.get(P_SNOW, None) and props_sib[P_SNOW].distance == 0
         and props_sib.get(P_CAT, None) and props_sib[P_CAT].distance == 1
         and props_sib.get(P_LIFE, None) and props_sib[P_LIFE].distance == 2,
         str(props_sib))

# ===========================================================================
# S3 全量重建
# ===========================================================================
ck.section("S3 全量重建（Python vs DB 队列 vs 触发器）")
with_trigger = closure_pairs(conn, "is_a")

conn.execute("DELETE FROM closure WHERE relation='is_a'")
rep_rebuild = C.rebuild_closure(conn, "is_a")
ck.eq("rebuild_closure('is_a') 行数 == 触发器结果", closure_pairs(conn, "is_a"), with_trigger)
ck.eq("重建报告行数", rep_rebuild.total_rows, len(with_trigger))

conn.execute("DELETE FROM closure WHERE relation='is_a'")
C.queue_rebuild(conn, "is_a")          # 走 DB 触发器重建
ck.eq("queue_rebuild（DB 触发器）结果一致", closure_pairs(conn, "is_a"), with_trigger)
ck.eq("队列行被触发器消费", conn.execute(
    "SELECT COUNT(*) FROM closure_rebuild_queue WHERE relation='is_a'").fetchone()[0], 0)

rep_all = C.rebuild_closure(conn)      # 全量
ck.check("rebuild_closure(None) 覆盖所有 active 关系",
         set(rep_all.rebuilt) == set(C._active_relations(conn)), str(sorted(rep_all.rebuilt)))
ck.eq("全量重建后 is_a 仍与触发器一致", closure_pairs(conn, "is_a"), with_trigger)
ck.check("mode=off 关系被清空而非重建（synonym 无闭包行）",
         conn.execute("SELECT COUNT(*) FROM closure WHERE relation='synonym'").fetchone()[0] == 0)

# ===========================================================================
# S4 增量更新
# ===========================================================================
ck.section("S4 增量更新（与全量差分一致 / 幂等 / 缺边报错）")
# 关掉 is_a 增量触发，模拟"触发器不管"的场景
conn.execute("UPDATE closure_policy SET mode='off' WHERE relation='is_a'")
LIFE = 900005
add_sense(conn, LIFE)
e_life = edge(conn, BIO, LIFE)          # 生物 --is_a--> 生命（mode=off → 触发器不动作）
ck.eq("off 模式下插入边不产生闭包",
      conn.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=?", (LIFE,)).fetchone()[0], 0)

r_inc = C.incremental_update(conn, BIO, LIFE, "is_a")
ck.eq("增量报告的受影响上界 = (1+|生物下位子树|)×(1+|生命上位链|)", r_inc.affected_pairs, 4)
ck.eq("增量写入行数", r_inc.written, 4)
ck.eq("增量后 == 在线 oracle（差分对拍）", closure_pairs(conn, "is_a"), oracle_pairs(conn, "is_a", 12))

before = closure_pairs(conn, "is_a")
r_inc2 = C.incremental_update(conn, BIO, LIFE, "is_a")
ck.eq("重复调用幂等（结果不变）", closure_pairs(conn, "is_a"), before)

try:
    C.incremental_update(conn, 123456, 654321, "is_a")
    ck.check("缺边应抛 ClosureError", False, "未抛异常")
except C.ClosureError:
    ck.check("缺边抛 ClosureError", True)

# 恢复 incremental：触发器路径与 Python 路径叠加仍收敛
conn.execute("UPDATE closure_policy SET mode='incremental' WHERE relation='is_a'")
NEW_ROOT = 900006
add_sense(conn, NEW_ROOT)
edge(conn, LIFE, NEW_ROOT)               # 触发器会自动维护
ck.eq("触发器路径结果 == oracle", closure_pairs(conn, "is_a"), oracle_pairs(conn, "is_a", 12))
C.incremental_update(conn, LIFE, NEW_ROOT, "is_a")   # 再手工叠一次
ck.eq("触发器 + 手工增量 仍 == oracle", closure_pairs(conn, "is_a"), oracle_pairs(conn, "is_a", 12))

# ===========================================================================
# S5 不变量检测
# ===========================================================================
ck.section("S5 不变量检测（I1–I6 / P1–P2）")
ck.check("干净库 validate 通过（I1–I5）", C.validate_closure_integrals(conn))
ck.check("干净库 validate 通过（含 I6 差分）",
         C.validate_closure_integrals(conn, include_differential=True))
ck.check("干净库 validate 通过（含 P1/P2）",
         C.validate_closure_integrals(conn, include_policy=True))
ck.check("干净库 inspect 返回空", C.inspect_closure_integrals(
    conn, include_differential=True, include_policy=True) == [])

EDGE_IS_A = conn.execute("SELECT relation_id FROM relations WHERE relation='is_a' LIMIT 1").fetchone()[0]

# I1 —— 表级 CHECK 先挡住自环
try:
    conn.execute("INSERT INTO closure(relation,ancestor,descendant,depth,first_edge) VALUES('is_a',1,1,1,?)", (EDGE_IS_A,))
    ck.check("schema CHECK 应挡住自环", False, "竟然插入成功")
except sqlite3.IntegrityError:
    ck.check("schema CHECK(ancestor<>descendant) 挡住自环", True)
ck.check("I1 检测器在正常库上无违反",
         not any(v.invariant == "I1" for v in C.inspect_closure_integrals(conn)))

# I3 —— 掉换一条 depth=1 行的方向
conn.execute("UPDATE closure SET ancestor=?, descendant=? WHERE relation='is_a' "
             "AND ancestor=? AND descendant=? AND depth=1", (TIGER, ANIMAL, ANIMAL, TIGER))
v3 = [v for v in C.inspect_closure_integrals(conn) if v.invariant == "I3"]
ck.check("I3 能抓到『方向被写反』的行", bool(v3) and v3[0].count >= 1, str(v3))
ck.check("validate 返回 False（I3）", C.validate_closure_integrals(conn) is False)
C.rebuild_closure(conn, "is_a")
ck.check("重建后 I3 修复", C.validate_closure_integrals(conn))

# I4 —— 见证边置空
conn.execute("UPDATE closure SET first_edge=NULL WHERE relation='is_a' AND ancestor=? AND descendant=?",
             (LIFE, BIO))
ck.check("I4 能抓到 first_edge IS NULL", not C.validate_closure_integrals(conn))
C.rebuild_closure(conn, "is_a")

# I5 —— 插入一条超深行（depth=64 > max_depth=12）
conn.execute("INSERT INTO closure(relation,ancestor,descendant,depth,first_edge) "
             "VALUES('is_a',999001,999002,64,?)", (EDGE_IS_A,))
v5 = [v for v in C.inspect_closure_integrals(conn) if v.invariant == "I5"]
ck.check("I5 能抓到超过 max_depth 的行", bool(v5) and v5[0].count == 1, str(v5))
conn.execute("DELETE FROM closure WHERE relation='is_a' AND ancestor=999001 AND descendant=999002")

# I6 —— 删掉一行 → MISSING；插一行孤儿 → EXTRA
conn.execute("DELETE FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=? "
             "AND depth=1", (LIFE, BIO))
v6 = [v for v in C.inspect_closure_integrals(conn, include_differential=True, relations=["is_a"])
      if v.invariant == "I6"]
ck.check("I6 能抓到 MISSING（物化缺行）", bool(v6) and "缺 1" in v6[0].description, str(v6))
C.rebuild_closure(conn, "is_a")
conn.execute("INSERT INTO closure(relation,ancestor,descendant,depth,first_edge) "
             "VALUES('is_a',998001,998002,1,?)", (EDGE_IS_A,))
v6b = [v for v in C.inspect_closure_integrals(conn, include_differential=True, relations=["is_a"])
       if v.invariant == "I6"]
ck.check("I6 能抓到 EXTRA（闭包多行）", bool(v6b) and "多 1" in v6b[0].description, str(v6b))
conn.execute("DELETE FROM closure WHERE relation='is_a' AND ancestor=998001 AND descendant=998002")

# P1 —— mode=off 却残留闭包行
conn.execute("UPDATE closure_policy SET mode='off' WHERE relation='is_a'")
vp1 = [v for v in C.inspect_closure_integrals(conn, include_policy=True) if v.invariant == "P1"]
ck.check("P1 能抓到 off 关系残留闭包行", bool(vp1) and vp1[0].count > 0, str(vp1))
conn.execute("UPDATE closure_policy SET mode='incremental' WHERE relation='is_a'")

# P2 —— is_transitive=1 却缺策略行
old = conn.execute("SELECT mode,max_depth FROM closure_policy WHERE relation='entails'").fetchone()
conn.execute("DELETE FROM closure_policy WHERE relation='entails'")
vp2 = [v for v in C.inspect_closure_integrals(conn, include_policy=True) if v.invariant == "P2"]
ck.check("P2 能抓到缺失策略行的传递关系", bool(vp2) and any("entails" in str(r) for r in vp2[0].sample),
         str(vp2))
conn.execute("INSERT INTO closure_policy(relation,mode,max_depth) VALUES('entails',?,?)",
             (old[0], old[1]))
ck.check("P2 修复后全绿",
         C.validate_closure_integrals(conn, include_policy=True, include_differential=True))

# ===========================================================================
# S6 批量导入模式
# ===========================================================================
ck.section("S6 批量导入模式")
conn.execute("UPDATE closure_policy SET mode='incremental' WHERE relation='is_a'")
before_incremental = C._active_relations(conn)
state = C.bulk_import_mode_off(conn)     # policy 策略
ck.eq("policy 策略：全部 active 关系被置 off", C._active_relations(conn), [])
ck.check("快照记录了原模式（全部非 off）",
         bool(state.policy) and all(m in ("incremental", "batch") for m in state.policy.values()),
         str(state.policy))
add_sense(conn, 900030)
add_sense(conn, 900031)
edge(conn, 900030, 900031)
ck.eq("off 期间插入边不产生闭包",
      conn.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=900031").fetchone()[0], 0)
n_restored = C.bulk_import_mode_on(conn, state, rebuild=True)
ck.eq("恢复的关系数", n_restored, len(state.policy))
ck.eq("模式已恢复", sorted(C._active_relations(conn)), sorted(before_incremental))
ck.check("恢复+重建后新边闭包已补上",
         conn.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=900031 "
                      "AND descendant=900030").fetchone()[0] == 1)
ck.check("恢复后校验全绿", C.validate_closure_integrals(conn, include_differential=True))

# 上下文管理器 + 异常安全
try:
    with C.bulk_import(conn, ["is_a"]) as st:
        ck.check("上下文进入后 is_a 为 off",
                 conn.execute("SELECT mode FROM closure_policy WHERE relation='is_a'").fetchone()[0] == "off")
        raise RuntimeError("模拟导入中途失败")
except RuntimeError:
    ck.check("异常被原样抛出", True)
ck.eq("异常后 is_a 模式已恢复", 
      conn.execute("SELECT mode FROM closure_policy WHERE relation='is_a'").fetchone()[0], "incremental")
ck.check("异常后未重建（数据留待人工确认）仍能通过 I1–I5 结构检查",
      C.validate_closure_integrals(conn))

with C.bulk_import(conn, ["is_a"]) as st:
    D1, D2 = 900040, 900041
    add_sense(conn, D1); add_sense(conn, D2)
    edge(conn, D1, D2)
ck.check("上下文正常退出后闭包已重建",
      conn.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
                   (D2, D1)).fetchone()[0] == 1)

# drop 策略
st_drop = C.bulk_import_mode_off(conn, ["is_a"], strategy="drop")
ck.eq("drop 策略：触发器被移除", trigger_names(conn), set())
ck.check("drop 快照备份了 CREATE 语句", set(st_drop.dropped_triggers) == set(C.CLOSURE_TRIGGERS))
C.bulk_import_mode_on(conn, st_drop, rebuild=True)
ck.eq("drop 策略：触发器已复原", trigger_names(conn), set(C.CLOSURE_TRIGGERS))
add_sense(conn, 900050); add_sense(conn, 900051)
edge(conn, 900050, 900051)
ck.check("复原的触发器仍可工作",
      conn.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
                   (900051, 900050)).fetchone()[0] == 1)
ck.check("drop 往返后校验全绿", C.validate_closure_integrals(conn, include_differential=True))

# ===========================================================================
# S7 索引与查询计划
# ===========================================================================
ck.section("S7 索引与查询计划")
plan_up = "\n".join(conn.explain(
    "SELECT c.ancestor, c.depth FROM closure c WHERE c.relation=? AND c.descendant=? ORDER BY c.depth",
    ("is_a", TIGER)))
ck.check("取上位命中 idx_closure_relation_desc", "idx_closure_relation_desc" in plan_up, plan_up)
ck.check("取上位不做全表扫描", "SCAN closure" not in plan_up.upper(), plan_up)

plan_down = "\n".join(conn.explain(
    "SELECT c.descendant, c.depth FROM closure c WHERE c.relation=? AND c.ancestor=?",
    ("is_a", BIO)))
ck.check("取下位命中主键索引", "PRIMARY KEY" in plan_down.upper(), plan_down)
ck.check("取下位不做全表扫描", "SCAN closure" not in plan_down.upper(), plan_down)

# ===========================================================================
# S8 环图剪枝 + 批量大图
# ===========================================================================
ck.section("S8 环图剪枝 + 批量大图（bulk_import + I6）")
# 造一条专用关系，测环
conn.execute("INSERT INTO relation_types(relation,relation_class,direction,is_transitive,"
             "is_symmetric,is_reflexive,is_functional,max_depth,description) "
             "VALUES('cyc_test','层级','directed',1,0,0,0,12,'环测试关系')")
conn.execute("INSERT INTO closure_policy(relation,mode,max_depth) VALUES('cyc_test','incremental',12)")
A, B, D = 910001, 910002, 910003
for s in (A, B, D):
    add_sense(conn, s)
edge(conn, A, B, "cyc_test")
edge(conn, B, D, "cyc_test")
edge(conn, D, A, "cyc_test")   # 成环 A→B→D→A
ck.check("环图不产生自环（i1）", conn.execute(
    "SELECT COUNT(*) FROM closure WHERE relation='cyc_test' AND ancestor=descendant").fetchone()[0] == 0)
ck.eq("环图闭包 == oracle（剪枝正确）", closure_pairs(conn, "cyc_test"), oracle_pairs(conn, "cyc_test", 12))
ck.check("环图 I6 差分通过",
      C.validate_closure_integrals(conn, include_differential=True, relations=["cyc_test"]))

# 批量大图：depth=4 branch=3 → 1+3+9+27+81 = 121 节点
BASE = 800000
nodes = []
edges = []
nid = BASE
root = nid; nid += 1
nodes.append(root)
frontier = [root]
for _ in range(4):
    nxt = []
    for p in frontier:
        for _ in range(3):
            nodes.append(nid); edges.append((nid, p)); nxt.append(nid); nid += 1
    frontier = nxt
t0 = time.perf_counter()
with C.bulk_import(conn):
    for s in nodes:
        add_sense(conn, s)
    for c, p in edges:
        edge(conn, c, p)
t_batch = time.perf_counter() - t0
batch_rows = closure_pairs(conn, "is_a")
ck.check("批量大图：闭包行数 == oracle",
      batch_rows == oracle_pairs(conn, "is_a", 12), f"got {len(batch_rows)}")
ck.check("批量大图 I6 差分通过",
      C.validate_closure_integrals(conn, include_differential=True, relations=["is_a"]))
ck.check("批量导入总耗时合理（<5s）", t_batch < 5.0, f"{t_batch:.3f}s")
print(f"        （121 节点闭包 {len(batch_rows)} 行，批量导入+重建 {t_batch*1000:.0f} ms）")

# ===========================================================================
# S9 统计
# ===========================================================================
ck.section("S9 统计函数")
stats = C.closure_stats(conn)
ck.check("closure_stats 总量 > 0", stats["total_rows"] > 0, str(stats))
ck.eq("by_relation 合计 == total_rows", sum(stats["by_relation"].values()), stats["total_rows"])
ck.check("closure_stats(relation) 单关系过滤", C.closure_stats(conn, "is_a")["total_rows"]
         == stats["by_relation"].get("is_a", 0))

# ===========================================================================
# S10 I1 检测器专测（去 CHECK 的库）
# ===========================================================================
ck.section("S10 I1 检测器专测（用去掉 CHECK 的表结构）")
DB2 = HERE / "logos_validate_i1.db"
_rm(DB2)
patched = SCHEMA.read_text(encoding="utf-8").replace(
    "CHECK (ancestor <> descendant)", "CHECK (ancestor = ancestor)")
raw = sqlite3.connect(str(DB2))
raw.executescript(patched)
raw.commit()
raw.close()
c2 = get_connection(DB2, must_exist=True)
c2.execute("INSERT INTO closure(relation,ancestor,descendant,depth,first_edge) "
           "VALUES('is_a',42,42,1,NULL)")
v1 = [v for v in C.inspect_closure_integrals(c2) if v.invariant == "I1"]
ck.check("I1 检测器能抓到自环行", bool(v1) and v1[0].count == 1, str(v1))
ck.check("含自环时 validate 返回 False", C.validate_closure_integrals(c2) is False)
c2.close()
_rm(DB2)

conn.close()
_rm(DB)
sys.exit(ck.summary())
