#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
schema_v3.sql 验证器：
  1. 在干净 DB 上完整执行 schema（开 FK 约束）
  2. 结构自检：表/索引/视图/触发器计数、integrity_check、foreign_key_check
  3. 功能自检：闭包增量维护、环保护、UPSERT 赋值顺序、FTS 中文检索、
               STRICT 类型拒绝、FK 拒绝、视图逆向展开
运行： python validate_schema.py
"""
import sqlite3, io, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema_v3.sql")
DB = os.path.join(HERE, "logos_test.db")

ok_count = 0
fail_count = 0


def check(name, cond, detail=""):
    global ok_count, fail_count
    if cond:
        ok_count += 1
        print(f"  [PASS] {name}")
    else:
        fail_count += 1
        print(f"  [FAIL] {name}  {detail}")


def expect_error(name, sql, params=()):
    """断言某条 SQL 必须失败（约束/触发器生效）"""
    global ok_count, fail_count
    try:
        conn.execute(sql, params)
        fail_count += 1
        print(f"  [FAIL] {name}  —— 本应被拒绝，却执行成功")
        return None
    except sqlite3.Error as e:
        ok_count += 1
        print(f"  [PASS] {name}  ({type(e).__name__}: {e})")
        return e


if os.path.exists(DB):
    os.remove(DB)

print(f"SQLite runtime = {sqlite3.sqlite_version}")
print("=" * 78)
print("STEP 1  执行 schema_v3.sql")
print("=" * 78)

conn = sqlite3.connect(DB)
conn.execute("PRAGMA foreign_keys = ON")
conn.execute("PRAGMA recursive_triggers = ON")   # 闭包 删除→重建 链式触发所必需

sql = open(SCHEMA, encoding="utf-8").read()
t0 = time.time()
try:
    conn.executescript(sql)
except sqlite3.Error as e:
    print(f"\n!!! SCHEMA 执行失败: {e}")
    sys.exit(1)
print(f"  schema 执行成功，耗时 {time.time()-t0:.3f}s\n")

cur = conn.cursor()

print("=" * 78)
print("STEP 2  结构自检")
print("=" * 78)
tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
).fetchall()]
views = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()]
trigs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()]
idxes = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
).fetchall()]
print(f"  表 {len(tables)} / 视图 {len(views)} / 触发器 {len(trigs)} / 索引 {len(idxes)}")
print(f"  表: {', '.join(sorted(tables))}")

check("无未命名对象残留 (placeholder)", not [t for t in tables if 'placeholder' in t])
check("integrity_check = ok", cur.execute("PRAGMA integrity_check").fetchone()[0] == "ok")
fk_bad = cur.execute("PRAGMA foreign_key_check").fetchall()
check("foreign_key_check 无违规", len(fk_bad) == 0, str(fk_bad[:3]))

strict_tables = 0
for t in tables:
    ddl = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)
    ).fetchone()[0]
    if "STRICT" in ddl.upper():
        strict_tables += 1
strict_expect_non = [t for t in tables if t.startswith("senses_fts")]  # FTS5 影子表
check(f"STRICT 表 = {strict_tables}/{len(tables) - len(strict_expect_non)}",
      strict_tables == len(tables) - len(strict_expect_non),
      f"（{len(strict_expect_non)} 张 senses_fts_* 影子表无法 STRICT，属预期）")

print("\n  —— 种子数据计数 ——")
for t in ["words", "senses", "relations", "semantic_roles", "sentence_patterns",
          "pattern_slots", "relation_types", "classifiers", "ellipsis_rules",
          "complements", "morphemes", "pos_tags", "phonological_rules",
          "characters", "character_readings", "closure"]:
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"     {t:<22} = {n}")

print()
print("=" * 78)
print("STEP 3  功能自检")
print("=" * 78)

# --- 3.1 生成列 ---
cur.execute("SELECT word, char_count, is_separable FROM words WHERE word_id=1")
w, cc, sep = cur.fetchone()
check("生成列 char_count", w == "洗澡" and cc == 2, f"{w}/{cc}")
seps = cur.execute(
    "SELECT word, is_separable FROM words WHERE is_separable=1"
).fetchall()
check("生成列 is_separable 仅对离合词为 1", seps == [("洗澡", 1)], str(seps))

# --- 3.2 STRICT 类型拒绝 ---
expect_error("STRICT 拒绝 TEXT 写入 INTEGER 列",
             "INSERT INTO characters(char, strokes) VALUES('测','很多')")

# --- 3.3 UNIQUE 阻断重复边 ---
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(13,12,'is_a')")
expect_error("UNIQUE 阻断重复关系三元组",
             "INSERT INTO relations(src_id,dst_id,relation) VALUES(13,12,'is_a')")

# --- 3.4 CHECK 阻断自环 ---
expect_error("CHECK 阻断 sense 自环",
             "INSERT INTO relations(src_id,dst_id,relation) VALUES(13,13,'causes')")

# --- 3.5 端点校验触发器 ---
expect_error("触发器校验端点存在性",
             "INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
             "VALUES('sense',99999,'sense',12,'is_a')")

# --- 3.6 闭包增量维护 ---
cur.execute("UPDATE closure_policy SET mode='incremental' WHERE relation='is_a'")
cur.execute("INSERT INTO words(word_id,word) VALUES(100,'生物')")
cur.execute("""INSERT INTO senses(sense_id,word_id,pos,sense_no,definition,pinyin)
               VALUES(100,100,'n',1,'自然界中具有生命现象的物质体系','sheng1wu4')""")
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(12,100,'is_a')")   # 动物 is_a 生物
rows = cur.execute(
    "SELECT ancestor, descendant, depth FROM closure WHERE relation='is_a' "
    "ORDER BY ancestor, descendant"
).fetchall()
# 约定: closure(ancestor=上位, descendant=下位)，边是 老虎->动物->生物
check("闭包增量插入 (老虎/动物/生物，方向为 上位→下位)",
      rows == [(12, 13, 1), (100, 12, 1), (100, 13, 2)], str(rows))
check("闭包 first_edge 见证边非空",
      cur.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND first_edge IS NULL").fetchone()[0] == 0)

# 方向自检：取"老虎(13)的全部上位"必须是 动物/生物
sup = cur.execute("""SELECT c.ancestor, w.word FROM closure c
                     JOIN senses s ON s.sense_id=c.ancestor JOIN words w ON w.word_id=s.word_id
                     WHERE c.relation='is_a' AND c.descendant=13
                     ORDER BY c.depth""").fetchall()
check("方向语义：13(老虎) 的上位 = 动物、生物", [x[1] for x in sup] == ["动物", "生物"], str(sup))

# --- 3.7 环保护 ---
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(12,13,'is_a')")   # 制造环
cyc = cur.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=descendant").fetchone()[0]
check("成环不产生自环闭包行 (CHECK 未被触发)", cyc == 0, f"自环行数={cyc}")
cur.execute("DELETE FROM relations WHERE src_id=12 AND dst_id=13 AND relation='is_a'")
rows2 = cur.execute(
    "SELECT ancestor, descendant, depth FROM closure WHERE relation='is_a' ORDER BY ancestor, depth"
).fetchall()
check("删除边后闭包自动重建回正确状态", rows2 == rows, f"删除前={rows} 删除后={rows2}")

# --- 3.8 UPSERT 赋值顺序（first_edge 写在 depth 之前）---
fe = cur.execute(
    "SELECT first_edge, depth FROM closure WHERE relation='is_a' AND ancestor=100 AND descendant=13"
).fetchone()
check("最短路径的见证边被正确保留", fe is not None and fe[0] is not None and fe[1] == 2, str(fe))

# --- 3.9 视图：逆向关系在线展开 ---
inv = cur.execute("""SELECT src_id, relation, dst_id FROM v_relations_all
                     WHERE relation='part_of' """).fetchall()
before = cur.execute("SELECT COUNT(*) FROM relations WHERE relation='part_of'").fetchone()[0]
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(12,100,'has_part')")
inv = cur.execute("""SELECT src_id, dst_id FROM v_relations_all
                     WHERE relation='part_of' AND src_id=100 AND dst_id=12""").fetchall()
check("v_relations_all 自动展开逆向关系，且未落库",
      inv == [(100, 12)] and
      cur.execute("SELECT COUNT(*) FROM relations WHERE relation='part_of'").fetchone()[0] == before)
cur.execute("DELETE FROM relations WHERE src_id=12 AND dst_id=100 AND relation='has_part'")

# --- 3.10 FTS5 中文双通道 ---
tq = cur.execute("""SELECT s.sense_id FROM senses_fts_trigram f
                    JOIN senses s ON s.sense_id=f.rowid
                    WHERE senses_fts_trigram MATCH '家养动物'""").fetchall()
check("trigram 通道：>=3 字子串命中「家养动物」→ 猫", tq == [(2,)], str(tq))
sq = cur.execute("""SELECT s.sense_id FROM senses_fts_seg f
                    JOIN senses s ON s.sense_id=f.rowid
                    WHERE senses_fts_seg MATCH '动物'""").fetchall()
check("预分词通道：2 字词「动物」有命中", len(sq) > 0, str(sq))
sy = cur.execute("""SELECT COUNT(*) FROM senses_fts_trigram
                    WHERE senses_fts_trigram MATCH '动物'""").fetchone()[0]
check("trigram 通道对 2 字查询确实无命中（故必须双通道）", sy == 0, str(sy))

# --- 3.11 量词推荐：显式搭配 + 类别默认回退 ---
qc = cur.execute("""
 SELECT cw.word, p.preference, 'explicit' FROM noun_classifier_pairs p
   JOIN classifiers cl ON cl.classifier_sense_id=p.classifier_sense_id
   JOIN senses cs ON cs.sense_id=cl.classifier_sense_id
   JOIN words cw ON cw.word_id=cs.word_id WHERE p.noun_sense_id=2
 UNION ALL
 SELECT cw2.word, d.preference, 'class_default' FROM classifier_class_defaults d
   JOIN sense_categories sc ON sc.cat=d.cat
   JOIN classifiers cl2 ON cl2.classifier_sense_id=d.classifier_sense_id
   JOIN senses cs2 ON cs2.sense_id=cl2.classifier_sense_id
   JOIN words cw2 ON cw2.word_id=cs2.word_id WHERE sc.sense_id=2
 ORDER BY 2 DESC""").fetchall()
check("量词推荐：猫 → 只 (显式+类别默认)", qc and qc[0][0] == "只", str(qc))

# --- 3.12 省略回填：同一槽位多策略 ---
n = cur.execute("""SELECT COUNT(*) FROM ellipsis_rules
                   WHERE context_type='dialogue' AND slot_role='施事'""").fetchone()[0]
check("同一语境同一槽位支持多策略+优先级 (v2.0 的 PK 缺陷已修)", n >= 2, f"策略数={n}")

# --- 3.13 在线递归推导祖先（不依赖物化闭包）---
anc = cur.execute("""
WITH RECURSIVE anc(id, depth) AS (
    SELECT 13, 0
    UNION
    SELECT r.dst_id, anc.depth+1 FROM anc JOIN relations r ON r.src_id=anc.id
     WHERE r.relation='is_a' AND r.src_kind='sense' AND r.dst_kind='sense' AND anc.depth<12
)
SELECT w.word, MIN(anc.depth) FROM anc
  JOIN senses s ON s.sense_id=anc.id JOIN words w ON w.word_id=s.word_id
 WHERE anc.id<>13 GROUP BY anc.id ORDER BY 2""").fetchall()
check("递归 CTE 在线推导：老虎→动物→生物",
      [a[0] for a in anc] == ["动物", "生物"], str(anc))

# --- 3.14 变更日志触发 ---
audit = cur.execute("SELECT COUNT(*) FROM changelog WHERE table_name='relations'").fetchone()[0]
check("relations 写入自动留痕 changelog", audit > 0, f"审计行数={audit}")

# --- 3.15 引擎/优化器检查 ---
cur.execute("ANALYZE")
q = cur.execute("""EXPLAIN QUERY PLAN
     SELECT * FROM closure WHERE relation='is_a' AND ancestor=13""").fetchall()
plan = " ".join(str(r) for r in q)
check("闭包前缀查询走索引 SEARCH 而非全表 SCAN",
      "SEARCH" in plan.upper() and "SCAN" not in plan.upper(), plan)

print()
print("=" * 78)
print("STEP 4  定点修复回归（逐项锁定，防止静默回退）")
print("=" * 78)

# --- R1 PRAGMA foreign_keys 真正生效 ---
check("R1 PRAGMA foreign_keys = ON（SQLite 默认 OFF，不设则全部 FK 形同虚设）",
      cur.execute("PRAGMA foreign_keys").fetchone()[0] == 1)
expect_error("R1 FK 违规被拒：relation 必须存在于 relation_types",
             "INSERT INTO relations(src_id,dst_id,relation) VALUES(12,100,'不存在的关系')")

# --- R2 补 UNIQUE / CHECK ---
expect_error("R2 UNIQUE 阻断重复关系三元组（含 dst）",
             "INSERT INTO relations(src_id,dst_id,relation) VALUES(13,12,'is_a')")
expect_error("R2 CHECK 阻断非法枚举值",
             "INSERT INTO words(word_id,word,word_type) VALUES(9001,'测试词','乱写词类')")
expect_error("R2 CHECK 阻断 sense 自环边",
             "INSERT INTO relations(src_id,dst_id,relation) VALUES(13,13,'causes')")


# --- R3 删重复索引：程序化检测"列集是既有 UNIQUE/PK 索引真前缀"的普通索引 ---
def find_redundant_indexes(conn):
    """返回 [(表, 冗余索引, 覆盖它的索引)]。

    判定规则：普通索引的**列签名**若是同表另一索引签名的**真前缀**，
    则后者可完全替代前者（同样的查询都能走索引）→ 前者冗余。

    实现要点（均由实测踩坑得出）：
      · 必须用 PRAGMA index_xinfo 而非 index_info —— 后者**不返回 DESC / collation**，
        会把 `(a DESC)` 误判成 `(a)` 从而误删（实测：tmp_expr2 被误报）。
      · 只取 key=1 的列（WITHOUT ROWID 的索引会把 PK 列以 key=0 追加到尾部）。
      · 表达式列（cid=-2）签名标记为 <expr:seq>，且**不作候选**，避免与同名列冲突。
      · 约束自带索引（origin pk/u）不作被删候选，但可作覆盖者。
      · 带 WHERE 的部分索引更小，属刻意优化，不作候选。
    """
    def signature(nm):
        sig, has_expr = [], False
        for seq, cid, cname, desc, coll, key in conn.execute(
                f"PRAGMA index_xinfo('{nm}')"):
            if key != 1:
                continue
            if cid < 0:
                has_expr = True
                sig.append((f"<expr:{seq}>", desc, coll))
            else:
                sig.append((cname, desc, coll))
        return tuple(sig), has_expr

    out = []
    tabs = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%'")]
    for t in tabs:
        allx = []
        for _s, nm, _u, org, part in conn.execute(f"PRAGMA index_list('{t}')").fetchall():
            cols, has_expr = signature(nm)
            allx.append((nm, cols, org, part, has_expr))
        for nm, cols, org, part, has_expr in allx:
            if org in ("pk", "u") or part or has_expr:
                continue                      # 不可作被删候选
            for nm2, cols2, _org2, _part2, _e2 in allx:
                if nm == nm2 or len(cols) > len(cols2):
                    continue
                if cols2[:len(cols)] != cols:
                    continue
                out.append((t, nm, nm2))
                break
    return out


_red = find_redundant_indexes(conn)
check("R3 无冗余索引（列集均非既有 UNIQUE/PK 索引的真前缀）", not _red, str(_red))

# --- R4 cross_links 主键修复：同一源可指向多个外部目标 ---
cur.execute("INSERT INTO entity_nodes(node_id,node_type,domain,label,entity_class) "
            "VALUES(9001,'domain_entity','medical','TP53','Protein')")
cur.execute("INSERT INTO entity_nodes(node_id,node_type,domain,label,entity_class) "
            "VALUES(9002,'domain_entity','medical','hsa04115','Pathway')")
for nid in (9001, 9002):
    cur.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
                "VALUES('sense',2,'node',?, 'domain_link')", (nid,))
_n = cur.execute("SELECT COUNT(*) FROM relations WHERE src_kind='sense' AND src_id=2 "
                 "AND dst_kind='node'").fetchone()[0]
check("R4 同一源可指向多个外部目标（v2.0 cross_links 主键漏 target_id 已修）",
      _n == 2, f"目标数={_n}")
_cl = cur.execute("SELECT target_id FROM cross_links WHERE source_id=2 "
                  "AND relation='domain_link' ORDER BY target_id").fetchall()
check("R4 兼容视图 cross_links 还原 v2.0 语义且支持多目标",
      [r[0] for r in _cl] == [9001, 9002], str(_cl))
expect_error("R4 完全重复的跨域边仍被 UNIQUE 阻断",
             "INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
             "VALUES('sense',2,'node',9001,'domain_link')")

# --- R5 ellipsis_rules 主键修复 ---
_pk = [r[1] for r in cur.execute("PRAGMA table_info('ellipsis_rules')") if r[5]]
check("R5 ellipsis_rules 主键改为代理键 ellipsis_rule_id",
      _pk == ["ellipsis_rule_id"], str(_pk))
_m = cur.execute("SELECT COUNT(*) FROM ellipsis_rules "
                 "WHERE context_type='dialogue' AND slot_role='施事'").fetchone()[0]
check("R5 同一(语境,槽位)可挂多条策略（v2.0 复合主键只允许一条已修）",
      _m >= 2, f"策略数={_m}")

# --- R6 统一 pattern_slots 子表 + 删 JSON 冗余字段 ---
_cols = [r[1] for r in cur.execute("PRAGMA table_info('sentence_patterns')")]
check("R6 sentence_patterns 已无 required_slots/optional_slots 冗余 JSON 列",
      not any(c.endswith("_slots") for c in _cols), str(_cols))
_orphan = cur.execute("""SELECT sp.name FROM sentence_patterns sp WHERE NOT EXISTS
        (SELECT 1 FROM pattern_slots ps WHERE ps.pattern_id = sp.pattern_id)""").fetchall()
check("R6 每个句式都有实体槽位行（子表不再是死表）", not _orphan, str(_orphan))
_mis = []
for _pid, _name, _tpl in cur.execute(
        "SELECT pattern_id, name, template FROM sentence_patterns"):
    _ph = set(re.findall(r"\{([^}]+)\}", _tpl))
    _sl = {r[0] for r in cur.execute(
        "SELECT slot_name FROM pattern_slots WHERE pattern_id=?", (_pid,))}
    if _ph != _sl:
        _mis.append((_name, sorted(_ph - _sl), sorted(_sl - _ph)))
check("R6 模板占位符与 pattern_slots 一一对应（32/32）", not _mis, str(_mis))
check("R6 可选性只在 is_required 表达（模板不再有 ? 后缀双写）",
      cur.execute("SELECT COUNT(*) FROM sentence_patterns "
                  "WHERE template LIKE '%?%'").fetchone()[0] == 0)

# --- R7 闭包方向约定锁定 ---
_bad = cur.execute("""SELECT c.relation, c.ancestor, c.descendant FROM closure c
     WHERE c.depth = 1 AND NOT EXISTS (
         SELECT 1 FROM relations r WHERE r.relation = c.relation
           AND r.src_id = c.descendant AND r.dst_id = c.ancestor)""").fetchall()
check("R7 方向锁定：depth=1 闭包行满足 dst=ancestor ∧ src=descendant",
      not _bad, str(_bad[:3]))
_ns = cur.execute("SELECT COUNT(*) FROM closure WHERE ancestor = descendant").fetchone()[0]
check("R7 闭包无自环行", _ns == 0, f"自环={_ns}")

# --- R8 触发器写法（FROM 子查询）+ first_edge 弱引用 + UPSERT 顺序 ---
_ok_form, _bad_form = [], []
for _trg in ("tr_closure_edge_ai", "tr_closure_rebuild_ai"):
    _ddl = cur.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                       (_trg,)).fetchone()[0]
    _norm = re.sub(r"\s+", " ", _ddl)
    if "FROM (WITH RECURSIVE" in _norm.upper():
        _ok_form.append(_trg)
    if re.search(r"INSERT\s+INTO\s+\w+\s*\([^;]*?\)\s*WITH\s+RECURSIVE", _norm, re.I):
        _bad_form.append(_trg)
check("R8 两个闭包触发器的递归 CTE 均嵌在 FROM 子查询中", len(_ok_form) == 2, str(_ok_form))
check("R8 触发器体内无 `INSERT ... WITH RECURSIVE` 非法写法", not _bad_form, str(_bad_form))

_ddl = cur.execute("SELECT sql FROM sqlite_master WHERE type='trigger' "
                   "AND name='tr_closure_edge_ai'").fetchone()[0]
_norm = re.sub(r"\s+", " ", _ddl)
_i_fe, _i_dp = _norm.find("first_edge = CASE"), _norm.find("depth = MIN")
check("R8 UPSERT 中 first_edge 写在 depth 之前（否则见证边永不更新）",
      0 <= _i_fe < _i_dp, f"pos(first_edge)={_i_fe} pos(depth)={_i_dp}")

_ddl = cur.execute("SELECT sql FROM sqlite_master WHERE name='closure'").fetchone()[0]
_fe_line = [l for l in _ddl.splitlines() if "first_edge" in l and "INTEGER" in l]
check("R8 first_edge 为弱引用（刻意不加外键，否则删除触发器永远匹配不到失效行）",
      bool(_fe_line) and "REFERENCES" not in _fe_line[0].upper(), str(_fe_line))

# --- R9 闭包维护三档模式齐备 ---
_modes = {r[0] for r in cur.execute("SELECT mode FROM closure_policy")}
check("R9 closure_policy 三档模式齐备（incremental / batch / off）",
      {"incremental", "batch", "off"} <= _modes, str(sorted(_modes)))


print()
print("=" * 78)
print("STEP 5  边界与不变量补齐（差分对拍 / 负向用例 / 空库 / 单节点）")
print("=" * 78)


# --------------------------------------------------------------------------
# 5.0 落地前的不变量契约（必须最先跑，避免被后续构造的测试数据污染）
# --------------------------------------------------------------------------
_nopolicy = cur.execute("""
    SELECT rt.relation FROM relation_types rt
     WHERE rt.is_transitive = 1
       AND NOT EXISTS (SELECT 1 FROM closure_policy cp WHERE cp.relation = rt.relation)
     ORDER BY rt.relation""").fetchall()
check("5.0a 每个 is_transitive=1 的关系都有显式 closure_policy 行（缺行=静默无闭包）",
      not _nopolicy, f"缺失: {[r[0] for r in _nopolicy]}")

_over = cur.execute("""
    SELECT cp.relation, cp.max_depth, rt.max_depth FROM closure_policy cp
      JOIN relation_types rt ON rt.relation = cp.relation
     WHERE cp.max_depth > rt.max_depth""").fetchall()
check("5.0b closure_policy.max_depth 不得超过 relation_types.max_depth（类型级上限）",
      not _over, str(_over))

_syminv = cur.execute("SELECT relation FROM relation_types "
                      "WHERE direction='symmetric' AND inverse_of IS NOT NULL").fetchall()
check("5.0c symmetric 关系不得声明 inverse_of（否则 v_relations_all 三倍展开→重复行）",
      not _syminv, str(_syminv))

_invbad = cur.execute("""
    SELECT rt.relation, rt.inverse_of FROM relation_types rt
      LEFT JOIN relation_types t2 ON t2.relation = rt.inverse_of
     WHERE rt.inverse_of IS NOT NULL
       AND (t2.inverse_of IS NULL OR t2.inverse_of <> rt.relation)""").fetchall()
check("5.0d relation_types.inverse_of 全部互逆成对", not _invbad, str(_invbad))

_reach = cur.execute("""
    WITH RECURSIVE w(role, d) AS (
        SELECT role, 0 FROM semantic_roles WHERE parent_role IS NULL
        UNION
        SELECT r.role, w.d + 1 FROM semantic_roles r JOIN w ON r.parent_role = w.role
         WHERE w.d < 40)
    SELECT COUNT(DISTINCT role) FROM w""").fetchone()[0]
_totroles = cur.execute("SELECT COUNT(*) FROM semantic_roles").fetchone()[0]
check(f"5.0e semantic_roles 自引用链无环且全部可达根（{_reach}/{_totroles}）",
      _reach == _totroles, f"{_reach}/{_totroles}")


# --------------------------------------------------------------------------
# 5.1 pattern_slots 结构不变量（R6 只查"集合相等"，未查位置与必填性）
# --------------------------------------------------------------------------
_gap, _noreq = [], []
for _pid, _nm in cur.execute("SELECT pattern_id, name FROM sentence_patterns"):
    _pos = [r[0] for r in cur.execute(
        "SELECT position FROM pattern_slots WHERE pattern_id=? ORDER BY position", (_pid,))]
    if _pos and _pos != list(range(1, len(_pos) + 1)):
        _gap.append((_nm, _pos))
    if cur.execute("SELECT COUNT(*) FROM pattern_slots WHERE pattern_id=? AND is_required=1",
                   (_pid,)).fetchone()[0] == 0:
        _noreq.append(_nm)
check("5.1a 每个句式的槽位 position 连续为 1..n（无空洞、无重号）", not _gap, str(_gap))
check("5.1b 每个句式至少一个必填槽（否则可匹配空串）", not _noreq, str(_noreq))


# --------------------------------------------------------------------------
# 5.2 pattern_slots 是"活表"：把 R6 的静态断言升级为动态行为断言
# --------------------------------------------------------------------------
cur.execute("INSERT INTO sentence_patterns(pattern_id,name,category,mood,template) "
            "VALUES(9801,'探针单槽句式','特殊句式',"
            "(SELECT mood FROM sentence_moods LIMIT 1),'{X}')")
cur.execute("INSERT INTO pattern_slots(pattern_id,slot_name,position,is_required) "
            "VALUES(9801,'X',1,1)")
_ph = set(re.findall(r"\{([^}]+)\}",
                     cur.execute("SELECT template FROM sentence_patterns WHERE pattern_id=9801")
                     .fetchone()[0]))
_sl = {r[0] for r in cur.execute(
    "SELECT slot_name FROM pattern_slots WHERE pattern_id=9801")}
check("5.2a 单槽位句式（边界）：模板占位符 == 子表槽位（R6 逻辑对动态数据同样成立）",
      _ph == _sl == {"X"}, f"{_ph}/{_sl}")
cur.execute("DELETE FROM sentence_patterns WHERE pattern_id=9801")
_left = cur.execute("SELECT COUNT(*) FROM pattern_slots WHERE pattern_id=9801").fetchone()[0]
check("5.2b 删除句式级联清空其槽位（ON DELETE CASCADE 真实行为）", _left == 0, f"残留={_left}")


# --------------------------------------------------------------------------
# 5.3 闭包差分对拍：物化闭包 vs 独立实现的在线递归 CTE
# --------------------------------------------------------------------------
def online_closure(rel, md):
    """在线递归 CTE 独立计算闭包，作为物化闭包的**差分对拍基准**（不读 closure 表）"""
    rows = cur.execute("""
        WITH RECURSIVE w(anc, cur, d, visited) AS (
            SELECT r.src_id, r.dst_id, 1, ','||r.src_id||','||r.dst_id||','
              FROM relations r
             WHERE r.relation=? AND r.src_kind='sense' AND r.dst_kind='sense'
            UNION ALL
            SELECT w.anc, r.dst_id, w.d+1, w.visited||r.dst_id||','
              FROM w JOIN relations r ON r.src_id = w.cur
             WHERE r.relation=? AND r.src_kind='sense' AND r.dst_kind='sense'
               AND instr(w.visited, ','||r.dst_id||',') = 0
               AND w.d < ?)
        SELECT cur, anc, MIN(d) FROM w WHERE cur <> anc GROUP BY cur, anc""",
        (rel, rel, md)).fetchall()
    # 注意元组顺序：闭包表的约定是 ancestor = dst 侧（上位）、descendant = src 侧（下位），
    # 而本 CTE 的 (cur, anc) = (起点 src 的下游 dst, 原始 src) = (上位, 下位)，
    # 故必须翻转成 (descendant, ancestor, depth) 才能与 closure 表对拍。
    return {(r[1], r[0], r[2]) for r in rows}


def materialized_closure(rel):
    return set(cur.execute("SELECT descendant, ancestor, depth FROM closure WHERE relation=?",
                           (rel,)))


def mk_senses(ids, prefix):
    for i in ids:
        cur.execute("INSERT INTO words(word_id, word) VALUES(?,?)", (i, f"{prefix}{i}"))
        cur.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) "
                    "VALUES(?,?,'n',1,?)", (i, i, f"{prefix}节点{i}"))


def mk_relation(rel, md, mode, cls='层级'):
    cur.execute("""INSERT INTO relation_types(relation,relation_class,direction,is_transitive,
                   is_symmetric,is_reflexive,is_functional,max_depth,description)
                   VALUES(?,?,'directed',1,0,0,0,?,'测试用关系')""", (rel, cls, md))
    cur.execute("INSERT INTO closure_policy(relation,mode,max_depth) VALUES(?,?,?)",
                (rel, mode, md))


# 链 + 菱形：840→841→842，840→844→842（两条长度 2 的路径，用于验证重叠路径与见证边）
DIFF_EDGES = [(840, 841), (841, 842), (842, 843), (840, 844), (844, 842)]
mk_senses([840, 841, 842, 843, 844], "差")
mk_relation("t_diff", 6, "incremental")
for a, b in DIFF_EDGES:
    cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(?,?,'t_diff')", (a, b))

_oracle = online_closure("t_diff", 6)
_mat = materialized_closure("t_diff")
check("5.3a 差分对拍：增量维护的物化闭包 == 独立在线递归 CTE（含菱形重叠路径）",
      _mat == _oracle, f"物化-期望={sorted(_mat - _oracle)} 期望-物化={sorted(_oracle - _mat)}")

cur.execute("UPDATE closure_policy SET mode='batch' WHERE relation='t_diff'")
cur.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('t_diff')")
_mat2 = materialized_closure("t_diff")
check("5.3b 全量重建结果 == 增量维护结果 == 在线期望（max_depth 语义一致性）",
      _mat2 == _oracle == _mat,
      f"重建-期望={sorted(_mat2 - _oracle)} 期望-重建={sorted(_oracle - _mat2)}")
check("5.3c 重建触发器自清空队列（应用层无需手动清理）",
      cur.execute("SELECT COUNT(*) FROM closure_rebuild_queue WHERE relation='t_diff'")
      .fetchone()[0] == 0)
cur.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('t_diff')")
check("5.3d 重建幂等：连续两次重建结果完全相同", materialized_closure("t_diff") == _mat2)


# --------------------------------------------------------------------------
# 5.4 删除路径：见证边失效 + 重叠路径恢复（first_edge 设计的核心风险）
# --------------------------------------------------------------------------
cur.execute("UPDATE closure_policy SET mode='incremental' WHERE relation='t_diff'")
cur.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('t_diff')")
_pv_before = cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_diff' "
                         "AND ancestor=842 AND descendant=840").fetchone()[0]
# 840→841 是 (ancestor=842, descendant=840) 最短路径的见证边；删它后应靠 840→844→842 恢复
cur.execute("DELETE FROM relations WHERE relation='t_diff' AND src_id=840 AND dst_id=841")
_pv_after = cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_diff' "
                        "AND ancestor=842 AND descendant=840").fetchone()[0]
check("5.4a 删除见证边后，重建能通过另一条重叠路径恢复该闭包对（菱形）",
      _pv_before == 1 and _pv_after == 1, f"删前={_pv_before} 删后={_pv_after}")
_dangle = cur.execute("""
    SELECT COUNT(*) FROM closure c
     WHERE c.relation='t_diff' AND c.depth=1
       AND NOT EXISTS (SELECT 1 FROM relations r
                        WHERE r.relation='t_diff' AND r.src_kind='sense' AND r.dst_kind='sense'
                          AND r.src_id=c.descendant AND r.dst_id=c.ancestor)""").fetchone()[0]
check("5.4b 删除后无悬空 depth=1 闭包行（first_edge 失效逻辑生效）", _dangle == 0, f"悬空={_dangle}")
cur.execute("DELETE FROM relations WHERE relation='t_diff'")
check("5.4c 删除全部边后闭包归零（不残留任何行）", not materialized_closure("t_diff"))


# --------------------------------------------------------------------------
# 5.5 max_depth 截断：链长 4 > max_depth 2
# --------------------------------------------------------------------------
mk_senses([850, 851, 852, 853], "截")
mk_relation("t_cap", 2, "incremental")
for a, b in ((850, 851), (851, 852), (852, 853)):
    cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(?,?,'t_cap')", (a, b))
_mx = cur.execute("SELECT MAX(depth) FROM closure WHERE relation='t_cap'").fetchone()[0]
check(f"5.5a 增量路径严格遵守 max_depth（实测最大 depth={_mx}，不得出现 2*max_depth+1）",
      _mx == 2, f"max depth={_mx}")
_totim = cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_cap'").fetchone()[0]
_d2im = cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_cap' AND depth=2"
                    ).fetchone()[0]
_t3im = cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_cap' "
                    "AND ancestor=853 AND descendant=850").fetchone()[0]
check("5.5b 截断而非丢弃整条链：3 条 depth=1 + 2 条 depth=2，被截断的 (850→853) 对不存在",
      _totim == 5 and _d2im == 2 and _t3im == 0, f"总={_totim} d2={_d2im} 超出对={_t3im}")
cur.execute("UPDATE closure_policy SET mode='batch' WHERE relation='t_cap'")
cur.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('t_cap')")
check("5.5c 截断边界上 增量 == 重建（同一份数据的两种闭包必须一致）",
      materialized_closure("t_cap") == online_closure("t_cap", 2))


# --------------------------------------------------------------------------
# 5.6 负向用例：不该产生闭包的关系**确实不产生**（回归最易静默失效）
# --------------------------------------------------------------------------
mk_senses([860, 861], "负")
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(860,861,'agent')")
check("5.6a 非传递关系（is_transitive=0）不写入闭包",
      cur.execute("SELECT COUNT(*) FROM closure WHERE relation='agent'").fetchone()[0] == 0)
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(860,861,'similar')")
check("5.6b 对称 + mode='off' 关系不写入闭包（v2.0 把它塞进 closure 纯属浪费）",
      cur.execute("SELECT COUNT(*) FROM closure WHERE relation='similar'").fetchone()[0] == 0)
cur.execute("INSERT INTO entity_nodes(node_id,node_type,domain,label,entity_class) "
            "VALUES(870,'domain_entity','medical','X','Gene'),"
            "(871,'domain_entity','medical','Y','Gene')")
cur.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
            "VALUES('node',870,'node',871,'domain_link')")
check("5.6c mode='batch' 关系插入时不内联写闭包（必须显式入队重建）",
      cur.execute("SELECT COUNT(*) FROM closure WHERE relation='domain_link'").fetchone()[0] == 0)


# --------------------------------------------------------------------------
# 5.7 UPSERT 真路径（替代 R8 的"DDL 文本位置"断言 —— 那条只证明写法，不证明行为）
# --------------------------------------------------------------------------
for a, b in DIFF_EDGES:
    cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(?,?,'t_diff')", (a, b))
_d0, _f0 = cur.execute("SELECT depth, first_edge FROM closure WHERE relation='t_diff' "
                       "AND ancestor=842 AND descendant=840").fetchone()
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(840,842,'t_diff')")
_newid = cur.execute("SELECT relation_id FROM relations WHERE src_id=840 AND dst_id=842 "
                     "AND relation='t_diff'").fetchone()[0]
_d1, _f1 = cur.execute("SELECT depth, first_edge FROM closure WHERE relation='t_diff' "
                       "AND ancestor=842 AND descendant=840").fetchone()
check("5.7a UPSERT DO UPDATE 真路径：直连边使 depth 2→1 且 first_edge 换为该边",
      _d0 == 2 and _d1 == 1 and _f1 == _newid, f"前=({_d0},{_f0}) 后=({_d1},{_f1}) new={_newid}")
_long = cur.execute("SELECT depth FROM closure WHERE relation='t_diff' "
                    "AND ancestor=843 AND descendant=842").fetchone()
check("5.7b 保护分支：更长路径不覆盖已建立的最短路径（depth 恒为 1）",
      _long is not None and _long[0] == 1, str(_long))


# --------------------------------------------------------------------------
# 5.8 外键与 PRAGMA：连接级生效性 + 事务内 no-op 陷阱 + 触发器兜底
# --------------------------------------------------------------------------
_c0 = sqlite3.connect(":memory:")
check("5.8a 新连接默认 foreign_keys=OFF（证明每次 connect 后必须显式开启）",
      _c0.execute("PRAGMA foreign_keys").fetchone()[0] == 0)
_c1 = sqlite3.connect(":memory:")
_c1.execute("BEGIN")
_c1.execute("PRAGMA foreign_keys = ON")
_v1 = _c1.execute("PRAGMA foreign_keys").fetchone()[0]
_c1.rollback()
check("5.8b 事务内执行 PRAGMA foreign_keys=ON 静默失效（必须在 BEGIN 之前设置）",
      _v1 == 0, f"实际={_v1}")
_c2 = sqlite3.connect(":memory:")
_c2.execute("PRAGMA foreign_keys = ON")
_c2.executescript(sql)
try:
    _c2.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(12,13,'不存在的关系')")
    _r3, _m3 = False, "未拒绝"
except sqlite3.Error as _e3:
    _r3, _m3 = True, str(_e3)
check(f"5.8c FK=ON 时非法 relation 引用被拒（{_m3}）", _r3)
_c3 = sqlite3.connect(":memory:")
_c3.executescript(sql)
_c3.execute("PRAGMA foreign_keys = OFF")
try:
    _c3.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
                "VALUES('sense',999999,'sense',13,'is_a')")
    _r4, _m4 = False, "未拒绝"
except sqlite3.Error as _e4:
    _r4, _m4 = True, str(_e4)
check(f"5.8d FK=OFF 时端点存在性触发器仍拦截（比外键更强，异构节点无 FK 可用）",
      _r4, _m4)
expect_error("5.8e CHECK 拒绝 weight < 0",
             "INSERT INTO relations(src_id,dst_id,relation,weight) VALUES(12,13,'is_a',-1)")
expect_error("5.8f CHECK 拒绝 confidence > 1",
             "INSERT INTO relations(src_id,dst_id,relation,confidence) VALUES(12,13,'is_a',1.5)")
expect_error("5.8g 端点身份不可变：UPDATE src_id 被拒（须删除后重建）",
             "UPDATE relations SET src_id=11 WHERE relation='is_a' AND src_id=13 AND dst_id=12")
expect_error("5.8h FK 拒绝 relations.source_id 指向不存在的来源",
             "INSERT INTO relations(src_id,dst_id,relation,source_id) VALUES(12,13,'is_a',99999)")
expect_error("5.8i FK 拒绝 pattern_slots.syntactic_function 指向不存在的句法功能",
             "INSERT INTO pattern_slots(pattern_id,slot_name,position,syntactic_function) "
             "VALUES(1,'ZZ',99,'不存在的功能')")

# 陷阱固化：把 schema 包进单事务加载能提速约 38x（文件库实测 4.29s → 0.113s），
# 但 schema 第 38 行那句 PRAGMA foreign_keys=ON 在事务内是 **no-op**，
# 结果种子加载期完全无外键保护（实测 FK=0）。此用例把这个坑锁住。
_c6 = sqlite3.connect(":memory:")
_c6.executescript("BEGIN;\n" + sql + "\nCOMMIT;")
_fk6 = _c6.execute("PRAGMA foreign_keys").fetchone()[0]
check("5.8j 陷阱固化：单事务加载让 schema 内置 PRAGMA 静默失效（FK=0，种子加载无外键保护）"
      " → 建库务必在连接级先设 PRAGMA，别依赖脚本内那句",
      _fk6 == 0, f"FK={_fk6}")

_c7 = sqlite3.connect(":memory:")
_c7.executescript(sql)
_c7.execute("PRAGMA foreign_keys = OFF")
_c7.execute("INSERT INTO relations(src_id,dst_id,relation,source_id) VALUES(12,13,'is_a',99999)")
_ck = _c7.execute("PRAGMA foreign_key_check").fetchall()
check("5.8k FK=OFF 时违规写入不被拦，但 PRAGMA foreign_key_check 仍能查出（唯一兜底防线）",
      len(_ck) > 0, f"查出={len(_ck)} 行")


# --------------------------------------------------------------------------
# 5.9 空库（新知识库冷启动）：视图 / 闭包重建 / FTS 是否健壮
# --------------------------------------------------------------------------
_ce = sqlite3.connect(":memory:")
_ce.executescript(sql)
_ce.execute("PRAGMA foreign_keys = OFF")
_etabs = [r[0] for r in _ce.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
for _t in _etabs:
    # 【关键】必须跳过 FTS5 影子表：直接 DELETE 它们会与 _data/_idx/_docsize 失配，
    # 导致 "fts5: corruption found reading blob"（实测踩坑）。
    # 正确做法是只删 senses，由 tr_senses_fts_ad 触发器负责同步清理影子表。
    if _t.startswith("senses_fts"):
        continue
    _ce.execute(f"DELETE FROM {_t}")
_nonempty = {t: _ce.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in _etabs
             if not t.startswith("senses_fts")
             and _ce.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]}
_fc = {t: _ce.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
       for t in ("senses_fts_trigram", "senses_fts_seg")}
check("5.9a 空库前置条件：用户表全部零行 + FTS 内容表零行"
      "（_data/_idx/_config 是 FTS 内部簿记表，空索引也保留段标记，不参与断言）",
      not _nonempty and not any(_fc.values()), f"用户表={_nonempty} FTS内容表={_fc}")
try:
    _vc = {v: _ce.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
           for v in ("v_relations_all", "v_senses_full", "v_closure_2", "cross_links", "entries")}
    _vok = all(n == 0 for n in _vc.values())
except sqlite3.Error as _e5:
    _vok, _vc = False, str(_e5)
check("5.9b 空库下 5 个视图查询均返回 0 行且不报错", _vok, str(_vc))
try:
    _ce.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('is_a')")
    _en = _ce.execute("SELECT COUNT(*) FROM closure").fetchone()[0]
    _eok = _en == 0
except sqlite3.Error as _e6:
    _eok, _en = False, str(_e6)
check("5.9c 空库触发闭包重建不报错且保持 0 行（空集重建幂等）", _eok, str(_en))
try:
    _f1 = _ce.execute("SELECT COUNT(*) FROM senses_fts_trigram "
                      "WHERE senses_fts_trigram MATCH '动物'").fetchone()[0]
    _f2 = _ce.execute("SELECT COUNT(*) FROM senses_fts_seg "
                      "WHERE senses_fts_seg MATCH '动物'").fetchone()[0]
    _fok = (_f1 == 0 and _f2 == 0)
except sqlite3.Error as _e7:
    _fok, _f1, _f2 = False, str(_e7), str(_e7)
check("5.9d 空库 FTS 双通道查询返回 0 行且不报错", _fok, f"trigram={_f1} seg={_f2}")
try:
    _ce.execute("INSERT INTO words(word_id,word) VALUES(880,'空库后新增')")
    _ce.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) "
                "VALUES(880,880,'n',1,'空库后的第一个义项')")
    _n7 = _ce.execute("SELECT COUNT(*) FROM senses_fts_seg").fetchone()[0]
    _eok2 = _n7 >= 1
except sqlite3.Error as _e8:
    _eok2, _n7 = False, str(_e8)
check("5.9e 空库冷启动后新增第一条义项，FTS 触发器正常工作", _eok2, str(_n7))
_ce.close()


# --------------------------------------------------------------------------
# 5.10 单节点 / 循环引用
# --------------------------------------------------------------------------
_cs = sqlite3.connect(":memory:")
_cs.executescript(sql)
_cs.execute("INSERT INTO words(word_id,word) VALUES(890,'孤')")
_cs.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) "
            "VALUES(890,890,'n',1,'孤立节点')")
check("5.10a 单节点（零边）不产生任何闭包行",
      _cs.execute("SELECT COUNT(*) FROM closure WHERE ancestor=890 OR descendant=890")
      .fetchone()[0] == 0)
_cs.execute("UPDATE closure_policy SET mode='batch' WHERE relation='is_a'")
try:
    _cs.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('is_a')")
    _sok = _cs.execute("SELECT COUNT(*) FROM closure").fetchone()[0] == 0
except sqlite3.Error as _e9:
    _sok = False
check("5.10b 单节点库触发重建：在线递归 CTE 在单行数据上不报错", _sok)
_cs.close()

expect_error("5.10c sense 自环被 CHECK 拒绝（不触碰闭包的 ancestor<>descendant 约束）",
             "INSERT INTO relations(src_id,dst_id,relation) VALUES(13,13,'is_a')")
_cap_before = cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_cap'").fetchone()[0]
cur.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(841,840,'t_diff')")  # 成环
check("5.10d 闭包链成环不产生自环行（ancestor = descendant）",
      cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_diff' AND ancestor=descendant")
      .fetchone()[0] == 0)
check("5.10e 某关系成环不影响其他关系的闭包（触发器按 relation 严格隔离）",
      cur.execute("SELECT COUNT(*) FROM closure WHERE relation='t_cap'").fetchone()[0] == _cap_before)
cur.execute("DELETE FROM relations WHERE relation='t_diff' AND src_id=841 AND dst_id=840")


# --------------------------------------------------------------------------
# 5.11 cross_links 无重复（视图层，去重责任在 relations.UNIQUE）
# --------------------------------------------------------------------------
cur.execute("INSERT INTO entity_nodes(node_id,node_type,domain,label,entity_class) "
            "VALUES(895,'domain_entity','medical','P','Protein'),"
            "(896,'domain_entity','medical','Q','Protein')")
cur.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
            "VALUES('node',895,'node',896,'domain_link')")
cur.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
            "VALUES('node',895,'node',896,'similar')")
_dup = cur.execute("""SELECT source_id, target_id, relation, COUNT(*) n FROM cross_links
                      GROUP BY source_id, target_id, relation HAVING n > 1""").fetchall()
check("5.11a cross_links 无同源同靶同关系的重复行（fan-out = 0）", not _dup, str(_dup))
_n895 = cur.execute("SELECT COUNT(*) FROM cross_links WHERE source_id=895 AND target_id=896"
                    ).fetchone()[0]
check("5.11b 两条不同关系 → 恰好 2 行（不放大、不丢失）", _n895 == 2, f"行数={_n895}")
_symback = cur.execute("SELECT COUNT(*) FROM v_relations_all "
                       "WHERE src_id=896 AND dst_id=895 AND relation='similar'").fetchone()[0]
check("5.11c v_relations_all 对对称关系自动反向展开恰好一次（不多不少）",
      _symback == 1, f"行数={_symback}")
cur.execute("DELETE FROM relations WHERE dst_kind='node' AND dst_id=896 AND relation='similar'")
_n2 = cur.execute("SELECT COUNT(*) FROM cross_links WHERE source_id=895 AND target_id=896"
                  ).fetchone()[0]
check("5.11d 删除底层边后视图行数同步下降（视图不落库、无缓存陈旧）", _n2 == 1, f"行数={_n2}")


# --------------------------------------------------------------------------
# 5.12 ellipsis_rules 多策略：运行时插入 + 唯一性边界 + 排序
# --------------------------------------------------------------------------
cur.execute("INSERT INTO ellipsis_rules(context_type,slot_role,strategy,priority,max_distance) "
            "VALUES('dialogue','施事','nearest_np',5,2)")
_cnt = cur.execute("SELECT COUNT(*) FROM ellipsis_rules "
                   "WHERE context_type='dialogue' AND slot_role='施事'").fetchone()[0]
check("5.12a 运行时可为同一(语境,槽位)追加第三条策略（v2.0 复合主键只允许一条）",
      _cnt == 3, f"策略数={_cnt}")
expect_error("5.12b 完全重复的(context_type,slot_role,strategy,priority) 被 UNIQUE 拒绝",
             "INSERT INTO ellipsis_rules(context_type,slot_role,strategy,priority) "
             "VALUES('dialogue','施事','nearest_np',5)")
_ord = [r[0] for r in cur.execute(
    "SELECT strategy FROM ellipsis_rules WHERE context_type='dialogue' AND slot_role='施事' "
    "ORDER BY priority DESC, strategy")]
check(f"5.12c 多策略可按 priority 稳定排序（{_ord}）",
      len(_ord) == 3 and _ord[0] == 'speaker', str(_ord))
cur.execute("DELETE FROM ellipsis_rules WHERE context_type='dialogue' "
            "AND slot_role='施事' AND strategy='nearest_np'")


# --------------------------------------------------------------------------
# 5.13 冗余索引探测器：阳性对照 + DESC/表达式阴性对照（R3 原来只有空断言）
# --------------------------------------------------------------------------
cur.execute("CREATE INDEX tmp_redundant_probe ON senses(word_id, pos)")
_probe = find_redundant_indexes(conn)
check("5.13a 阳性对照：探测器能识别出人为构造的冗余索引（证明断言有判别力）",
      any(x[1] == "tmp_redundant_probe" for x in _probe), str(_probe))
cur.execute("DROP INDEX tmp_redundant_probe")
cur.execute("CREATE INDEX tmp_desc_probe ON senses(word_id DESC, pos DESC)")
_p2 = [x for x in find_redundant_indexes(conn) if x[1] == "tmp_desc_probe"]
check("5.13b 阴性对照：DESC 排序不同的索引不得被判为冗余（index_info 不暴露排序→会误删）",
      not _p2, str(_p2))
cur.execute("DROP INDEX tmp_desc_probe")
check("5.13c 清理后无冗余索引残留", not find_redundant_indexes(conn))

conn.commit()
print()
print("=" * 78)
print(f"结果: PASS {ok_count} / FAIL {fail_count}")
print("=" * 78)
conn.close()
sys.exit(1 if fail_count else 0)
