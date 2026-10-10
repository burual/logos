#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_gaps.py —— 对 validate_schema.py 47 项用例的**盲区探针**（只诊断，不进主套件）

目的：在把新用例写进 validate_schema.py 之前，先实测确认哪些断言"确实成立"、
哪些断言会暴露 schema 的真实缺陷。避免把想当然的断言写进回归套件。

运行： python probe_gaps.py
"""
import sqlite3, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema_v3.sql")
SQL = open(SCHEMA, encoding="utf-8").read()


def fresh(fk=True):
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = " + ("ON" if fk else "OFF"))
    c.execute("PRAGMA recursive_triggers = ON")
    c.executescript(SQL)
    return c


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ===========================================================================
hdr("P1  传递关系是否都有显式 closure_policy 行（缺失 = 静默无闭包）")
c = fresh()
miss = c.execute("""
    SELECT rt.relation, rt.is_transitive, rt.max_depth
      FROM relation_types rt
     WHERE rt.is_transitive = 1
       AND NOT EXISTS (SELECT 1 FROM closure_policy cp WHERE cp.relation = rt.relation)
     ORDER BY rt.relation""").fetchall()
tr = c.execute("SELECT COUNT(*) FROM relation_types WHERE is_transitive=1").fetchone()[0]
pol = c.execute("SELECT COUNT(*) FROM closure_policy").fetchone()[0]
print(f"  is_transitive=1 的关系: {tr} 个 / closure_policy 行: {pol} 行")
print(f"  缺失策略行的传递关系 ({len(miss)} 个):")
for r in miss:
    print(f"     {r[0]:<14} is_transitive={r[1]}  relation_types.max_depth={r[2]}")
# 这些关系是否真的不产生闭包？
if miss:
    c.execute("INSERT INTO words(word_id,word) VALUES(800,'甲'),(801,'乙'),(802,'丙')")
    for i, w in enumerate((800, 801, 802)):
        c.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) VALUES(?,?,'n',1,?)",
                  (w, w, f"探针{w}"))
    rel = miss[0][0]
    c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(800,801,?)", (rel,))
    c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(801,802,?)", (rel,))
    n = c.execute("SELECT COUNT(*) FROM closure WHERE relation=?", (rel,)).fetchone()[0]
    print(f"  → 实测：连插两条 {rel} 边后 closure 行数 = {n}  （期望 3，实际见上）")

# ===========================================================================
hdr("P2  max_depth 语义：增量维护 vs 全量重建 是否一致？")
c = fresh()
c.execute("""INSERT INTO relation_types(relation,relation_class,direction,is_transitive,
             is_symmetric,is_reflexive,is_functional,max_depth,description)
             VALUES('t_probe','层级','directed',1,0,0,0,2,'探针关系')""")
c.execute("INSERT INTO closure_policy(relation,mode,max_depth) VALUES('t_probe','incremental',2)")
for i, w in enumerate((810, 811, 812, 813)):
    c.execute("INSERT INTO words(word_id,word) VALUES(?,?)", (w, f"链{w}"))
    c.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) VALUES(?,?,'n',1,?)",
              (w, w, f"链节点{w}"))
# 链 810 -> 811 -> 812 -> 813   (src=下位, dst=上位)
for a, b in ((810, 811), (811, 812), (812, 813)):
    c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(?,?,'t_probe')", (a, b))
inc = c.execute("SELECT ancestor,descendant,depth FROM closure WHERE relation='t_probe' "
                "ORDER BY depth,ancestor").fetchall()
print(f"  [增量] mode=incremental, max_depth=2 → {len(inc)} 行，最大 depth={max(x[2] for x in inc)}")
for r in inc:
    print(f"        ancestor={r[0]} descendant={r[1]} depth={r[2]}")
c.execute("UPDATE closure_policy SET mode='batch' WHERE relation='t_probe'")
c.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('t_probe')")
reb = c.execute("SELECT ancestor,descendant,depth FROM closure WHERE relation='t_probe' "
                "ORDER BY depth,ancestor").fetchall()
print(f"  [重建] mode=batch,       max_depth=2 → {len(reb)} 行，最大 depth={max(x[2] for x in reb)}")
for r in reb:
    print(f"        ancestor={r[0]} descendant={r[1]} depth={r[2]}")
only_inc = sorted(set(inc) - set(reb))
only_reb = sorted(set(reb) - set(inc))
print(f"  → 增量多出 {len(only_inc)} 行: {only_inc}")
print(f"  → 重建多出 {len(only_reb)} 行: {only_reb}")
print(f"  → 结论：{'两条路径不一致（缺陷）' if inc != reb else '两条路径一致'}")

# ===========================================================================
hdr("P3  mode='off' 时删除边，是否留下陈旧闭包行？")
c = fresh()
c.execute("""INSERT INTO relation_types(relation,relation_class,direction,is_transitive,
             is_symmetric,is_reflexive,is_functional,max_depth,description)
             VALUES('t_off','层级','directed',1,0,0,0,8,'探针关系')""")
c.execute("INSERT INTO closure_policy(relation,mode,max_depth) VALUES('t_off','incremental',8)")
for w in (820, 821, 822):
    c.execute("INSERT INTO words(word_id,word) VALUES(?,?)", (w, f"离{w}"))
    c.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) VALUES(?,?,'n',1,?)",
              (w, w, f"离节点{w}"))
for a, b in ((820, 821), (821, 822)):
    c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(?,?,'t_off')", (a, b))
print(f"  增量维护后 closure 行数 = {c.execute(chr(83)+'ELECT COUNT(*) FROM closure WHERE relation=\'t_off\'').fetchone()[0]}")
c.execute("UPDATE closure_policy SET mode='off' WHERE relation='t_off'")
c.execute("DELETE FROM relations WHERE relation='t_off' AND src_id=820 AND dst_id=821")
n_stale = c.execute("SELECT COUNT(*) FROM closure WHERE relation='t_off'").fetchone()[0]
q = c.execute("SELECT COUNT(*) FROM closure_rebuild_queue WHERE relation='t_off'").fetchone()[0]
# 真实可达对：820->821 边已删，只剩 821->822，故只有 (822,821,1) 可达
truth = c.execute("""SELECT COUNT(*) FROM closure c WHERE c.relation='t_off' AND NOT EXISTS
     (SELECT 1 FROM relations r WHERE r.relation='t_off' AND r.src_id=c.descendant AND r.dst_id=c.ancestor)""").fetchone()[0]
print(f"  → 切到 off 后删除一条边：closure 残留 {n_stale} 行，重建队列 {q} 行")
print(f"  → 其中 first_edge 已指向不存在边的陈旧行 = {truth} 行")
print(f"  → 结论：{'存在陈旧闭包残留（缺陷）' if n_stale > 1 else '无残留'}")

# ===========================================================================
hdr("P4  UPSERT 真路径：先有长路径，再插直连边 → depth/first_edge 是否更新？")
c = fresh()
for w in (830, 831, 832, 833):
    c.execute("INSERT INTO words(word_id,word) VALUES(?,?)", (w, f"菱{w}"))
    c.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) VALUES(?,?,'n',1,?)",
              (w, w, f"菱节点{w}"))
# 830 -is_a-> 831 -is_a-> 832 ；830 -is_a-> 833 -is_a-> 832   （两条长度 2 的路径）
for a, b in ((830, 831), (831, 832), (830, 833), (833, 832)):
    c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(?,?,'is_a')", (a, b))
row = c.execute("SELECT depth, first_edge FROM closure WHERE relation='is_a' "
                "AND ancestor=832 AND descendant=830").fetchone()
print(f"  两条 2 跳路径后：(ancestor=832, descendant=830) depth={row[0]} first_edge={row[1]}")
# 直连边 830 -is_a-> 832，应把 depth 压到 1 并换见证边
c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(830,832,'is_a')")
new_id = c.execute("SELECT relation_id FROM relations WHERE src_id=830 AND dst_id=832 "
                   "AND relation='is_a'").fetchone()[0]
row2 = c.execute("SELECT depth, first_edge FROM closure WHERE relation='is_a' "
                 "AND ancestor=832 AND descendant=830").fetchone()
print(f"  插入直连边(relation_id={new_id})后：depth={row2[0]} first_edge={row2[1]}")
print(f"  → DO UPDATE 生效：{'是' if row2[0] == 1 and row2[1] == new_id else '否（缺陷）'}")
# 反向：长路径不得覆盖已存在的最短路径
keep = c.execute("SELECT depth, first_edge FROM closure WHERE relation='is_a' "
                 "AND ancestor=831 AND descendant=830").fetchone()
print(f"  「更长路径不覆盖最短路径」：(831,830) depth={keep[0]}（应恒为 1）")

# ===========================================================================
hdr("P5  冗余索引检测器是否有阳性对照？（当前 R3 只断言列表为空 → 探测器坏掉也 PASS）")
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("vs", os.path.join(HERE, "validate_schema.py"))
# 不执行 validate_schema.py 主体，直接内联复制其检测函数做对照实验
def find_redundant_indexes(conn):
    out = []
    tabs = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%'")]
    for t in tabs:
        allx = []
        for _s, nm, _u, org, part in conn.execute(f"PRAGMA index_list('{t}')").fetchall():
            cols = tuple(r[2] for r in conn.execute(f"PRAGMA index_info('{nm}')"))
            allx.append((nm, cols, org, part))
        for nm, cols, org, part in allx:
            if org in ("pk", "u"):
                continue
            for nm2, cols2, _org2, part2 in allx:
                if nm == nm2 or len(cols) > len(cols2):
                    continue
                if cols2[:len(cols)] != cols:
                    continue
                if part and not part2:
                    continue
                out.append((t, nm, nm2))
                break
    return out

c = fresh()
print(f"  基线（干净 schema）：{find_redundant_indexes(c)}")
c.execute("CREATE INDEX tmp_dup_idx ON senses(word_id, pos)")   # UNIQUE(word_id,pos,definition) 的真前缀
pos = find_redundant_indexes(c)
print(f"  阳性对照（人为建冗余索引 senses(word_id,pos)）：{pos}")
print(f"  → 探测器{'能' if pos else '**不能**'}识别冗余 → 阳性对照{'通过' if pos else '缺失'}")

# 边界：表达式索引 / DESC 索引 是否误报
c.execute("CREATE INDEX tmp_expr_idx ON senses(word_id, sense_no DESC)")
c.execute("CREATE INDEX tmp_expr2 ON senses(word_id DESC, pos DESC)")
pos2 = find_redundant_indexes(c)
extra = [x for x in pos2 if x[1].startswith("tmp_expr")]
print(f"  DESC/表达式索引是否被误判为冗余：{extra if extra else '否（正确）'}")
info = c.execute("PRAGMA index_info('tmp_expr2')").fetchall()
print(f"  PRAGMA index_info('tmp_expr2') = {info}  ← 注意：不返回 DESC/排序信息")
c.execute("DROP INDEX tmp_dup_idx"); c.execute("DROP INDEX tmp_expr_idx"); c.execute("DROP INDEX tmp_expr2")
print(f"  清理后：{find_redundant_indexes(c)}")

# ===========================================================================
hdr("P6  空库：清空全部数据后，视图/闭包重建/递归 CTE 是否仍然健壮？")
c = fresh(fk=False)
c.execute("PRAGMA foreign_keys = OFF")   # schema 自带 PRAGMA ON，必须在加载后再关
tabs = [r[0] for r in c.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
for t in tabs:
    try:
        c.execute(f"DELETE FROM {t}")
    except sqlite3.Error as e:
        print(f"    清空 {t} 失败: {e}")
left = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tabs}
print(f"  清空后非空表：{ {k: v for k, v in left.items() if v} }")
c.execute("PRAGMA foreign_keys = ON")
for v in ("v_relations_all", "v_senses_full", "v_closure_2", "cross_links", "entries"):
    try:
        n = c.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
        print(f"  {v:<18} 空库查询 → {n} 行  OK")
    except sqlite3.Error as e:
        print(f"  {v:<18} 空库查询失败: {e}")
try:
    c.execute("INSERT INTO closure_rebuild_queue(relation) VALUES('is_a')")
    print(f"  空库触发闭包重建 → OK（closure 行数 {c.execute('SELECT COUNT(*) FROM closure').fetchone()[0]}）")
except sqlite3.Error as e:
    print(f"  空库闭包重建失败: {e}")
# 空 relation_types 下插一条边：应被 FK 拒绝
print(f"  空库下 pattern 占位符一致性（0 个句式）→ trivially OK")

# ===========================================================================
hdr("P7  自引用链是否有环：semantic_roles.parent_role / relation_types.inverse_of")
c = fresh()
cyc = c.execute("""
WITH RECURSIVE w(role, path, d) AS (
    SELECT role, ',' || role || ',', 0 FROM semantic_roles WHERE parent_role IS NULL
    UNION ALL
    SELECT r.role, w.path || r.role || ',', w.d + 1
      FROM semantic_roles r JOIN w ON r.parent_role = w.role
     WHERE w.d < 20 AND instr(w.path, ',' || r.role || ',') = 0
)
SELECT COUNT(*) FROM semantic_roles""").fetchone()[0]
reach = c.execute("""
WITH RECURSIVE w(role, d) AS (
    SELECT role, 0 FROM semantic_roles WHERE parent_role IS NULL
    UNION
    SELECT r.role, w.d + 1 FROM semantic_roles r JOIN w ON r.parent_role = w.role WHERE w.d < 20
)
SELECT COUNT(DISTINCT role) FROM w""").fetchone()[0]
tot = c.execute("SELECT COUNT(*) FROM semantic_roles").fetchone()[0]
print(f"  semantic_roles 总数={tot}，从根可达={reach}")
unreach = c.execute("""
WITH RECURSIVE w(role, d) AS (
    SELECT role, 0 FROM semantic_roles WHERE parent_role IS NULL
    UNION SELECT r.role, w.d+1 FROM semantic_roles r JOIN w ON r.parent_role=w.role WHERE w.d<20)
SELECT role, parent_role FROM semantic_roles
 WHERE role NOT IN (SELECT role FROM w)""").fetchall()
print(f"  从根不可达（潜藏环 / 悬空父级）: {unreach if unreach else '无'}")

inv_pairs = c.execute("""
SELECT rt.relation, rt.inverse_of, t2.inverse_of AS back, rt.direction
  FROM relation_types rt LEFT JOIN relation_types t2 ON t2.relation = rt.inverse_of
 WHERE rt.inverse_of IS NOT NULL AND (t2.inverse_of IS NULL OR t2.inverse_of <> rt.relation)""").fetchall()
print(f"  inverse_of 非互逆的声明: {inv_pairs if inv_pairs else '无（全部成对）'}")
sym_inv = c.execute("SELECT relation, direction, inverse_of FROM relation_types "
                    "WHERE direction='symmetric' AND inverse_of IS NOT NULL").fetchall()
print(f"  symmetric 且 inverse_of 非空（会让 v_relations_all 三倍展开）: {sym_inv if sym_inv else '无'}")

# ===========================================================================
hdr("P8  ON DELETE CASCADE 行为（sentence_patterns → pattern_slots）")
c = fresh()
c.execute("""INSERT INTO sentence_patterns(pattern_id,name,category,mood,template)
             VALUES(9101,'探针句式','特殊句式',(SELECT mood FROM sentence_moods LIMIT 1),'{X} {Y}')""")
c.execute("INSERT INTO pattern_slots(pattern_id,slot_name,position) VALUES(9101,'X',1),(9101,'Y',2)")
before = c.execute("SELECT COUNT(*) FROM pattern_slots WHERE pattern_id=9101").fetchone()[0]
c.execute("DELETE FROM sentence_patterns WHERE pattern_id=9101")
after = c.execute("SELECT COUNT(*) FROM pattern_slots WHERE pattern_id=9101").fetchone()[0]
print(f"  删除句式前槽位={before}，删除后={after} → CASCADE {'生效' if after == 0 else '未生效（缺陷）'}")

# ===========================================================================
hdr("P9  PRAGMA foreign_keys：连接级生效性 与 事务内 no-op 陷阱")
c = sqlite3.connect(":memory:")
print(f"  新连接默认 foreign_keys = {c.execute('PRAGMA foreign_keys').fetchone()[0]}  （期望 0）")
c.executescript(SQL)
c.execute("PRAGMA foreign_keys = OFF")   # 覆盖 schema 自带的 ON，模拟"忘了设 PRAGMA"的连接
try:
    c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(12,13,'不存在的关系')")
    print("  未开 FK 时插入非法关系 → 成功（证明默认 OFF 时 FK 形同虚设）")
except sqlite3.Error as e:
    print(f"  未开 FK 时仍被拒：{type(e).__name__}: {e}  ← 说明有触发器在兜底（比 FK 更强）")
c2 = sqlite3.connect(":memory:")
c2.execute("PRAGMA foreign_keys = ON")
c2.executescript(SQL)
try:
    c2.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(12,13,'不存在的关系')")
    print("  开了 FK 后仍插入成功 → 异常")
except sqlite3.Error as e:
    print(f"  开了 FK 后插入被拒：{type(e).__name__}: {e}")
c3 = sqlite3.connect(":memory:")
c3.execute("BEGIN")
c3.execute("PRAGMA foreign_keys = ON")
print(f"  事务内执行 PRAGMA foreign_keys=ON → 实际值 = {c3.execute('PRAGMA foreign_keys').fetchone()[0]}  （期望 0 = 静默失效）")
# 端点校验触发器是否独立于 FK 生效
c4 = sqlite3.connect(":memory:")
c4.execute("PRAGMA foreign_keys = OFF")
c4.executescript(SQL)
try:
    c4.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
               "VALUES('sense',999999,'sense',13,'is_a')")
    print("  FK=OFF 且端点不存在 → 插入成功（触发器未兜底）")
except sqlite3.Error as e:
    print(f"  FK=OFF 且端点不存在 → 被触发器拒绝：{e}  ← 兜底有效")

# ===========================================================================
hdr("P10 pattern_slots 结构不变量：position 连续性 / 是否存在必填槽")
c = fresh()
bad_gap, bad_dup, no_req = [], [], []
for pid, name in c.execute("SELECT pattern_id,name FROM sentence_patterns"):
    pos = [r[0] for r in c.execute(
        "SELECT position FROM pattern_slots WHERE pattern_id=? ORDER BY position", (pid,))]
    if pos and pos != list(range(1, len(pos) + 1)):
        bad_gap.append((name, pos))
    nreq = c.execute("SELECT COUNT(*) FROM pattern_slots WHERE pattern_id=? AND is_required=1",
                     (pid,)).fetchone()[0]
    if nreq == 0:
        no_req.append(name)
    nopt = c.execute("SELECT COUNT(*) FROM pattern_slots WHERE pattern_id=? AND is_required=0",
                     (pid,)).fetchone()[0]
    if nopt and nreq == 0:
        bad_dup.append(name)
print(f"  position 非连续(1..n)的句式: {bad_gap if bad_gap else '无'}")
print(f"  无任何必填槽的句式: {no_req if no_req else '无'}")
pos_bad = c.execute("SELECT pattern_id, slot_name, position FROM pattern_slots "
                    "WHERE position < 1").fetchall()
print(f"  position<1 的槽位: {pos_bad if pos_bad else '无'}")

# ===========================================================================
hdr("P11 cross_links 视图 fan-out：对称关系 / 重复行")
c = fresh()
c.execute("INSERT INTO entity_nodes(node_id,node_type,domain,label,entity_class) "
          "VALUES(9201,'domain_entity','medical','A','Protein'),"
          "(9202,'domain_entity','medical','B','Protein')")
c.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
          "VALUES('node',9201,'node',9202,'domain_link')")
c.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
          "VALUES('node',9201,'node',9202,'similar')")   # symmetric
cid = c.execute("SELECT COUNT(*) FROM cross_links WHERE source_id=9201 AND target_id=9202").fetchone()[0]
print(f"  cross_links  中 (9201→9202) 行数 = {cid}  （两条边：domain_link + similar，各 1 行）")
allsym = c.execute("SELECT relation, COUNT(*) FROM v_relations_all WHERE src_id=9201 AND dst_id=9202 "
                   "GROUP BY relation").fetchall()
print(f"  v_relations_all 中 (9201→9202) 按关系分组 = {allsym}")
sym_back = c.execute("SELECT COUNT(*) FROM v_relations_all WHERE src_id=9202 AND dst_id=9201").fetchone()[0]
print(f"  v_relations_all 中 (9202→9201) 行数 = {sym_back}  ← 对称关系被自动反向展开")
dup = c.execute("""SELECT source_id,target_id,relation,COUNT(*) n FROM cross_links
                   GROUP BY source_id,target_id,relation HAVING n>1""").fetchall()
print(f"  cross_links 重复组 (同源同靶同关系): {dup if dup else '无'}")

# ===========================================================================
hdr("P12 负向用例：非传递 / 对称 / off 模式关系是否**确实不产生**闭包")
c = fresh()
for w in (930, 931):
    c.execute("INSERT INTO words(word_id,word) VALUES(?,?)", (w, f"负{w}"))
    c.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) VALUES(?,?,'n',1,?)",
              (w, w, f"负节点{w}"))
cases = [("agent", "非传递 is_transitive=0"),
         ("similar", "对称 + mode='off'"),
         ("domain_link", "mode='batch'（不应内联写闭包）")]
for rel, why in cases:
    if rel == "domain_link":
        c.execute("INSERT INTO entity_nodes(node_id,node_type,domain,label,entity_class) "
                  "VALUES(930,'domain_entity','medical','X','Gene'),"
                  "(931,'domain_entity','medical','Y','Gene')")
        c.execute("INSERT INTO relations(src_kind,src_id,dst_kind,dst_id,relation) "
                  "VALUES('node',930,'node',931,'domain_link')")
    else:
        c.execute("INSERT INTO relations(src_id,dst_id,relation) VALUES(930,931,?)", (rel,))
    n = c.execute("SELECT COUNT(*) FROM closure WHERE relation=?", (rel,)).fetchone()[0]
    print(f"  {rel:<14} ({why:<34}) → closure 行数 = {n}  {'OK' if n == 0 else '**有闭包(不符合预期)**'}")

# ===========================================================================
hdr("P13 空库 FTS：清空后重建 FTS 触发器是否报错")
c = fresh(fk=False)
c.execute("PRAGMA foreign_keys = OFF")
c.execute("DELETE FROM senses")
try:
    c.execute("INSERT INTO words(word_id,word) VALUES(940,'测试')")
    c.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) VALUES(940,940,'n',1,'空库后的义项')")
    n = c.execute("SELECT COUNT(*) FROM senses_fts_seg").fetchone()[0]
    print(f"  空库后插入义项 → FTS 影子表行数 = {n}  OK")
except sqlite3.Error as e:
    print(f"  空库后插入义项失败: {e}")

print("\n" + "=" * 78)
print("探针结束")
print("=" * 78)
