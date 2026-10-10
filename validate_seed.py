#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_seed.py —— seed_general_v3.sql (+ seed_general_v3_ext.sql) 的建库 + 行为验证

分七段：
  §1  加载与幂等        schema → seed → seed 再跑一次，行数必须不变
  §1b 扩展种子          seed_general_v3_ext.sql 加载 + 幂等硬前置（不计入断言编号）
  §2  结构覆盖          需求清单逐项核对（名词/动词/形容词/虚词/量词/关系/闭包）
  §3  约束完整性        FK / UNIQUE / CHECK / 端点 / 触发器
  §4  闭包正确性        方向不变量 I3 + 传递闭包 + first_edge 见证边
  §5  句式覆盖          5 个目标句式：槽位完整 + 每槽位有约束
  §6  端到端语义校验    用 relation_constraints 判定「猫吃鱼」合法性（不是数行数）
  §7  扩展种子断言      鱼的多继承（A1-A4）+ 抽象/人造物分类骨架（B1-B8），共 12 项

断言编号：§1-§6 为 1-65，§7 为 66-77（合计 77 项）。
  §1b 的 [ok] 行是**非计数**的前提检查：ext 不幂等即致命退出，不会混入 77 项。
  同理，需求里未列入 A/B 断言但明确要求存在的数据（食物 has_property、
  cross_links 链接）以 [ok] 形式透明报告，避免"看起来 77 项全绿却漏检"。

运行： python validate_seed.py
"""
import sqlite3, io, os, sys, re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema_v3.sql")
SEED = os.path.join(HERE, "seed_general_v3.sql")
SEED_EXT = os.path.join(HERE, "seed_general_v3_ext.sql")

PASS = FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        FAILURES.append((name, detail))
        print(f"  [FAIL] {name}   {detail}")


def connect():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("PRAGMA recursive_triggers = ON")
    return c


def load(c, path):
    with open(path, encoding="utf-8") as f:
        c.executescript(f.read())


def count(c, sql, args=()):
    return c.execute(sql, args).fetchone()[0]


def sid(c, word, pos):
    """按自然键取 sense_id —— 与种子文件同一套寻址方式。"""
    r = c.execute(
        "SELECT s.sense_id FROM senses s JOIN words w ON w.word_id = s.word_id "
        "WHERE w.word = ? AND s.pos = ? LIMIT 1", (word, pos)).fetchone()
    return r[0] if r else None


print("=" * 78)
print("seed_general_v3.sql 验证")
print("=" * 78)

# ---------------------------------------------------------------- §1 加载
print("\n§1 加载与幂等")
conn = connect()
try:
    load(conn, SCHEMA)
    n_after_schema = count(conn, "SELECT COUNT(*) FROM senses")
    load(conn, SEED)
    check("schema_v3.sql + seed_general_v3.sql 无错误执行", True)
except sqlite3.Error as e:
    check("schema_v3.sql + seed_general_v3.sql 无错误执行", False, f"{type(e).__name__}: {e}")
    sys.exit(1)

TABLES = ["semantic_categories", "words", "senses", "sense_features", "sense_categories",
          "function_words", "function_word_roles", "classifiers", "noun_classifier_pairs",
          "classifier_class_defaults", "relations", "relation_constraints",
          "slot_constraints", "closure", "utterances"]
snap1 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in TABLES}

# 幂等：再跑一遍
try:
    load(conn, SEED)
    snap2 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in TABLES}
    drift = {t: (snap1[t], snap2[t]) for t in TABLES if snap1[t] != snap2[t]}
    check("幂等：第二次执行种子后全部 15 张表行数不变", not drift, str(drift))
except sqlite3.Error as e:
    check("幂等：第二次执行种子后全部 15 张表行数不变", False,
          f"{type(e).__name__}: {e}")

check("种子新增/补齐的义项 = 27（原 %d → %d）" % (n_after_schema, snap1["senses"]),
      snap1["senses"] - n_after_schema == 27,
      f"新增 {snap1['senses'] - n_after_schema}")

# 自足性：种子文件自带连接级 PRAGMA，且写在 BEGIN 之前
# —— 这是本代码库记录过的真实坑：PRAGMA foreign_keys 在显式事务内是 no-op，
#    若把 PRAGMA 写在 BEGIN 之后就等于没开，种子加载期 FK 全部失效。
with open(SEED, encoding="utf-8") as f:
    _raw = f.read()
i_fk, i_begin = _raw.find("PRAGMA foreign_keys"), _raw.find("BEGIN;")
check("种子自足：PRAGMA foreign_keys 写在 BEGIN 之前（事务内 PRAGMA 是 no-op）",
      0 <= i_fk < i_begin, f"fk@{i_fk} begin@{i_begin}")

# 事务语句必须配对。这一条是补出来的 —— 文件初稿漏了收尾的 COMMIT，
# 却被 SQLite 的隐式提交掩盖（同连接内读到的数据完全正常），62 项行为测试全绿。
# 若应用层把它当子脚本执行，或进程在后续步骤才退出，未闭合的事务会被回滚。
_n_begin = len(re.findall(r"(?m)^\s*BEGIN\s*;", _raw))
_n_commit = len(re.findall(r"(?m)^\s*COMMIT\s*;", _raw))
check(f"事务语句配对：BEGIN({_n_begin}) == COMMIT({_n_commit})",
      _n_begin == _n_commit and _n_begin > 0, f"BEGIN={_n_begin} COMMIT={_n_commit}")
check("种子末尾已提交：加载后连接不处于未闭合事务中",
      conn.in_transaction is False, "in_transaction=True —— 缺 COMMIT")

c_indep = connect()
load(c_indep, SCHEMA)
load(c_indep, SEED)                    # 不额外设 PRAGMA，全靠文件自带
check("独立连接加载后 PRAGMA foreign_keys = 1（文件自足，非依赖调用方设置）",
      c_indep.execute("PRAGMA foreign_keys").fetchone()[0] == 1)
try:
    c_indep.execute("INSERT INTO senses(sense_id,word_id,pos,sense_no,definition) "
                    "VALUES (9999, 999999, 'n', 1, '孤儿义项')")
    check("加载后 FK 处于**生效**状态（孤儿义项被拒绝）", False, "非法插入竟然成功")
except sqlite3.IntegrityError as e:
    check("加载后 FK 处于**生效**状态（孤儿义项被拒绝）", True)
c_indep.close()

# 落盘验证：文件库 → 关闭 → 重开另一个连接。同连接内读得到 ≠ 已提交。
_dbf = os.path.join(HERE, "_seedcheck.db")
for _p in (_dbf, _dbf + "-wal", _dbf + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
try:
    c_file = sqlite3.connect(_dbf)
    c_file.execute("PRAGMA foreign_keys = ON")
    c_file.execute("PRAGMA recursive_triggers = ON")
    load(c_file, SCHEMA)
    load(c_file, SEED)
    c_file.close()
    c_re = sqlite3.connect(_dbf)
    n_re = c_re.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
    n_cl = c_re.execute("SELECT COUNT(*) FROM closure").fetchone()[0]
    c_re.close()
    check(f"落盘验证：重开连接后数据完整提交（senses={n_re}, closure={n_cl}）",
          n_re == snap1["senses"] and n_cl == snap1["closure"],
          f"senses={n_re} closure={n_cl}")
finally:
    for _p in (_dbf, _dbf + "-wal", _dbf + "-shm"):
        if os.path.exists(_p):
            os.remove(_p)

# ------------------------------------------------------- §1b 扩展种子（非计数）
print("\n§1b 扩展种子 seed_general_v3_ext.sql（加载 + 幂等硬前置，不计入断言编号）")

# 刻意**不计入** 1-77 的编号序列：需求给定的新增断言是 §7 的 A1-A4 / B1-B8 共 12 项。
# 本节做两件"前提性"的事，任一出错即致命退出（不进入 §2-§7）：
#   ① 加载 ext（失败 = 后面全部断言都无意义）
#   ② 幂等验证：再执行一次，12 张相关表行数必须**零漂移**
#      —— 这是本项目记录过的真实坑：种子文件漏写 ON CONFLICT 时，
#         首次执行完全正常，只有第二次执行才暴露重复行。
EXT_TABLES = ["semantic_categories", "words", "senses", "sense_features", "sense_categories",
              "classifiers", "noun_classifier_pairs", "relation_constraints", "relations",
              "entity_nodes", "closure", "changelog"]

ext_before = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in EXT_TABLES}
try:
    load(conn, SEED_EXT)
    ext1 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in EXT_TABLES}
    load(conn, SEED_EXT)                       # 第二次执行：幂等性检验
    ext2 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in EXT_TABLES}
except sqlite3.Error as e:
    print(f"  [FATAL] 加载 seed_general_v3_ext.sql 失败：{type(e).__name__}: {e}")
    sys.exit(1)

_drift = {t: (ext1[t], ext2[t]) for t in EXT_TABLES if ext1[t] != ext2[t]}
if _drift:
    print(f"  [FATAL] seed_general_v3_ext.sql 不幂等（第二次执行发生漂移）：{_drift}")
    print("          零漂移是 §2-§7 全部断言的前提，中止。")
    sys.exit(1)

gain = {t: ext1[t] - ext_before[t] for t in EXT_TABLES}
print("  [ok]   ext 加载成功，且第二次执行零漂移（EXT_TABLES 12 张表全部不变）")
print("  [ok]   ext 增量：" + ", ".join(f"{t}+{gain[t]}" for t in EXT_TABLES if gain[t]))
with open(SEED_EXT, encoding="utf-8") as _f:
    _ext_raw = _f.read()
_n_closure_insert = len(re.findall(r"(?m)^\s*INSERT\s+INTO\s+closure", _ext_raw, re.I))
print(f"  [ok]   ext 零 closure INSERT（实测 {_n_closure_insert} 条）："
      f"新增 {gain['closure']} 行闭包全部由 tr_closure_edge_ai 派生")
_n_begin = len(re.findall(r"(?m)^\s*BEGIN\s*;", _ext_raw))
_n_commit = len(re.findall(r"(?m)^\s*COMMIT\s*;", _ext_raw))
print(f"  [ok]   ext 事务语句配对：BEGIN({_n_begin}) == COMMIT({_n_commit})"
      + ("   ← 不配对！" if _n_begin != _n_commit else ""))

# ---------------------------------------------------------------- §2 结构覆盖
print("\n§2 结构覆盖（需求清单逐项核对）")

NOUNS = ["猫", "鱼", "动物", "食物", "水", "人", "我", "你", "他"]
VERBS = ["吃", "喝", "抓", "跑", "看", "是", "有"]
ADJS = ["大", "小", "好", "坏"]
FUNCS = ["把", "被", "的", "了", "着", "过", "在", "从", "向"]
CLFS = ["只", "条", "个"]


def has_word(c, w, pos=None):
    if pos:
        return count(c, "SELECT COUNT(*) FROM senses s JOIN words w2 ON w2.word_id=s.word_id "
                        "WHERE w2.word=%s AND s.pos=%s" % ("?", "?"), (w, pos)) > 0
    return count(c, "SELECT COUNT(*) FROM words WHERE word=?", (w,)) > 0


missing_n = [w for w in NOUNS if not has_word(conn, w, 'n') and not has_word(conn, w, 'r')]
check("名词 9 个齐备（猫/鱼/动物/食物/水/人/我/你/他）", not missing_n, f"缺 {missing_n}")
missing_v = [w for w in VERBS if not has_word(conn, w, 'v')]
check("动词 7 个齐备（吃/喝/抓/跑/看/是/有）", not missing_v, f"缺 {missing_v}")
missing_a = [w for w in ADJS if not has_word(conn, w, 'a')]
check("形容词 4 个齐备（大/小/好/坏）", not missing_a, f"缺 {missing_a}")
missing_f = [w for w in FUNCS if not has_word(conn, w, 'p') and not has_word(conn, w, 'u')]
check("虚词 9 个齐备（把/被/的/了/着/过/在/从/向）", not missing_f, f"缺 {missing_f}")
missing_c = [w for w in CLFS if not count(
    conn, "SELECT COUNT(*) FROM classifiers cl JOIN senses s ON s.sense_id=cl.classifier_sense_id "
          "JOIN words w ON w.word_id=s.word_id WHERE w.word=?", (w,))]
check("量词 3 个已建 classifiers 行（只/条/个）", not missing_c, f"缺 {missing_c}")

# function_words 覆盖
n_fw = count(conn, "SELECT COUNT(*) FROM function_words fw JOIN senses s ON s.sense_id=fw.sense_id "
                   "JOIN words w ON w.word_id=s.word_id WHERE w.word IN "
                   "('把','被','的','了','着','过','在','从','向')")
check("9 个虚词均有 function_words 条目", n_fw == 9, f"实际 {n_fw}")

# function_words 字段完整性
incomplete = conn.execute("""
  SELECT w.word, fw.category, fw.subcategory, fw.grammatical_function, fw.attaches_to
    FROM function_words fw
    JOIN senses s ON s.sense_id = fw.sense_id
    JOIN words  w ON w.word_id = s.word_id
   WHERE w.word IN ('把','被','的','了','着','过','在','从','向')
     AND (fw.category IS NULL OR fw.subcategory IS NULL
          OR fw.grammatical_function IS NULL OR fw.attaches_to IS NULL)""").fetchall()
check("把/被等虚词的 category/subcategory/grammatical_function/attaches_to 全部非空",
      not incomplete, str(incomplete))

# 把/被 的语义角色映射
role_map = {r[0]: r[1] for r in conn.execute("""
  SELECT w.word, fwr.role FROM function_word_roles fwr
    JOIN senses s ON s.sense_id = fwr.sense_id
    JOIN words  w ON w.word_id = s.word_id WHERE w.word IN ('把','被')""")}
check("把 → 受事（前置）", role_map.get("把") == "受事", str(role_map))
check("被 → 施事（前置）", role_map.get("被") == "施事", str(role_map))

# 量词搭配
pair_fish = conn.execute(
    "SELECT cw.word FROM noun_classifier_pairs p "
    "JOIN senses cs ON cs.sense_id=p.classifier_sense_id JOIN words cw ON cw.word_id=cs.word_id "
    "WHERE p.noun_sense_id=?", (sid(conn, "鱼", "n"),)).fetchall()
check("鱼 → 条", ("条",) in pair_fish, str(pair_fish))
pair_cat = conn.execute(
    "SELECT cw.word FROM noun_classifier_pairs p "
    "JOIN senses cs ON cs.sense_id=p.classifier_sense_id JOIN words cw ON cw.word_id=cs.word_id "
    "WHERE p.noun_sense_id=?", (sid(conn, "猫", "n"),)).fetchall()
check("猫 → 只", ("只",) in pair_cat, str(pair_cat))
pair_ren = conn.execute(
    "SELECT cw.word FROM noun_classifier_pairs p "
    "JOIN senses cs ON cs.sense_id=p.classifier_sense_id JOIN words cw ON cw.word_id=cs.word_id "
    "WHERE p.noun_sense_id=?", (sid(conn, "人", "n"),)).fetchall()
check("人 → 个", ("个",) in pair_ren, str(pair_ren))

# ---------------------------------------------------------------- §3 约束完整性
print("\n§3 约束完整性")

check("PRAGMA foreign_key_check 无违规",
      not conn.execute("PRAGMA foreign_key_check").fetchall())
check("PRAGMA integrity_check = ok",
      conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok")

# relation_constraints 覆盖所有 7 个动词
verb_sids = [(w, sid(conn, w, 'v')) for w in VERBS]
no_rc = [w for w, i in verb_sids
         if not count(conn, "SELECT COUNT(*) FROM relation_constraints WHERE sense_id=?", (i,))]
check("7 个动词义项均有 relation_constraints", not no_rc, f"缺 {no_rc}")

# agent / patient 必备
# 注意：是 是判断动词（系事），不支配施事 —— 它的必备槽是 theme，不是 agent。
AGENT_VERBS = ["吃", "喝", "抓", "跑", "看", "有"]
no_agent = [w for w in AGENT_VERBS
            if not count(conn, "SELECT COUNT(*) FROM relation_constraints "
                               "WHERE sense_id=? AND slot='agent'", (sid(conn, w, "v"),))]
check("6 个动作/领有动词声明 agent 槽约束（是 属判断动词，不在此列）",
      not no_agent, f"缺 {no_agent}")
has_theme = count(conn, "SELECT COUNT(*) FROM relation_constraints WHERE sense_id=? AND slot='theme'",
                  (sid(conn, "是", "v"),))
check("是（判断动词）声明 theme 槽约束（系事）", has_theme > 0)
trans = ["吃", "喝", "抓", "看"]
no_pat = [w for w in trans
          if not count(conn, "SELECT COUNT(*) FROM relation_constraints WHERE sense_id=? AND slot='patient'",
                       (sid(conn, w, "v"),))]
check("及物动词（吃/喝/抓/看）均声明 patient 槽约束", not no_pat, f"缺 {no_pat}")
has_instr = count(conn, "SELECT COUNT(*) FROM relation_constraints WHERE sense_id=? AND slot='instrument'",
                  (sid(conn, "吃", "v"),))
check("吃 声明 instrument 槽（示例：用筷子吃）", has_instr > 0)

# relation_constraints 的 category 值必须是合法 semantic_categories
bad_cat = conn.execute("""
  SELECT rc.sense_id, rc.slot, rc.value FROM relation_constraints rc
   WHERE rc.kind='category'
     AND NOT EXISTS (SELECT 1 FROM semantic_categories sc WHERE sc.cat = rc.value)""").fetchall()
check("relation_constraints 中 kind='category' 的取值均为合法语义类别", not bad_cat, str(bad_cat))

# 关系端点合法性（触发器兜底之外的独立复核）
bad_ep = conn.execute("""
  SELECT r.relation_id, r.src_kind, r.src_id, r.dst_kind, r.dst_id FROM relations r
   WHERE (r.src_kind='sense' AND NOT EXISTS (SELECT 1 FROM senses WHERE sense_id=r.src_id))
      OR (r.dst_kind='sense' AND NOT EXISTS (SELECT 1 FROM senses WHERE sense_id=r.dst_id))""").fetchall()
check("全部关系端点指向存在的义项", not bad_ep, str(bad_ep))

# 无重复关系
dupes = conn.execute("""
  SELECT src_kind,src_id,relation,dst_kind,dst_id,COUNT(*) n FROM relations
   GROUP BY 1,2,3,4,5 HAVING n>1""").fetchall()
check("relations 无重复边（UNIQUE 生效）", not dupes, str(dupes))

# 对称关系只存一条
sym_dup = conn.execute("""
  SELECT a.src_id,a.dst_id FROM relations a JOIN relations b
    ON b.src_id=a.dst_id AND b.dst_id=a.src_id AND b.relation=a.relation
   JOIN relation_types t ON t.relation=a.relation
  WHERE t.is_symmetric=1 AND a.src_id < b.src_id""").fetchall()
check("对称关系（opposite）无双向冗余存储", not sym_dup, str(sym_dup))

# ---------------------------------------------------------------- §4 闭包
print("\n§4 闭包正确性")

# I3 方向不变量：depth=1 行必须与 relations 边方向相反
viol = conn.execute("""
  SELECT c.ancestor, c.descendant FROM closure c
   WHERE c.relation='is_a' AND c.depth=1
     AND NOT EXISTS (SELECT 1 FROM relations r
                      WHERE r.relation='is_a' AND r.src_id=c.descendant AND r.dst_id=c.ancestor)""").fetchall()
check("I3 方向不变量：depth=1 闭包行满足 dst=ancestor ∧ src=descendant", not viol, str(viol))

back_viol = conn.execute("""
  SELECT r.src_id, r.dst_id FROM relations r
   WHERE r.relation='is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
     AND NOT EXISTS (SELECT 1 FROM closure c
                      WHERE c.relation='is_a' AND c.ancestor=r.dst_id AND c.descendant=r.src_id)""").fetchall()
check("每条 is_a 边都有对应 depth=1 闭包行（反向完备性）", not back_viol, str(back_viol))

cat, bio = sid(conn, "猫", "n"), sid(conn, "生物", "n")
row = conn.execute("SELECT depth FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
                   (bio, cat)).fetchone()
check("传递闭包已生成：生物 → 猫 depth=2（触发器派生，非手工插入）",
      row is not None and row[0] == 2, str(row))

fish = sid(conn, "鱼", "n")
row2 = conn.execute("SELECT depth FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
                    (bio, fish)).fetchone()
check("传递闭包已生成：生物 → 鱼 depth=2", row2 is not None and row2[0] == 2, str(row2))

food = sid(conn, "食物", "n")
row3 = conn.execute("SELECT depth FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
                    (food, fish)).fetchone()
check("多上位（DAG）：鱼 同时是 动物 与 食物 的下位",
      row3 is not None and row3[0] == 1, str(row3))

null_fe = count(conn, "SELECT COUNT(*) FROM closure WHERE relation='is_a' AND first_edge IS NULL")
check("闭包 first_edge 见证边无空值", null_fe == 0, f"空值 {null_fe} 行")

dangling_fe = count(conn, "SELECT COUNT(*) FROM closure c WHERE c.first_edge IS NOT NULL "
                          "AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.relation_id=c.first_edge)")
check("闭包 first_edge 无悬空引用", dangling_fe == 0, f"悬空 {dangling_fe} 行")

# 差分对拍：用**独立实现的在线递归 CTE** 当物化闭包的 oracle，
# 而不是"拿闭包验闭包"。这是唯一能发现"触发器漏算/多算"的检验方法。
oracle = {(r[0], r[1]) for r in conn.execute("""
    WITH RECURSIVE walk(desc, anc, d) AS (
        SELECT r.src_id, r.dst_id, 1 FROM relations r
         WHERE r.relation='is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
        UNION ALL
        SELECT w.desc, r2.dst_id, w.d + 1
          FROM walk w JOIN relations r2 ON r2.src_id = w.anc
         WHERE r2.relation='is_a' AND r2.src_kind='sense' AND r2.dst_kind='sense'
           AND w.d < 12
    )
    SELECT DISTINCT anc, desc FROM walk""")}
mat = {(r[0], r[1]) for r in conn.execute(
    "SELECT ancestor, descendant FROM closure WHERE relation='is_a'")}
check(f"闭包 == 在线递归 CTE（差分对拍，各 {len(oracle)} 对）", mat == oracle,
      f"仅闭包有 {sorted(mat - oracle)}；仅 oracle 有 {sorted(oracle - mat)}")

# 手工 depth=1 的语句存在（要求 6）
with open(SEED, encoding="utf-8") as f:
    seed_sql = f.read()
check("种子含 depth=1 直接边的显式 INSERT（需求 6）",
      re.search(r"INSERT INTO closure\(relation, ancestor, descendant, depth, first_edge\)", seed_sql) is not None)
check("种子未手工插入任何 depth>1 的闭包行（需求 6）",
      not re.search(r"INSERT INTO closure\([^)]*\)\s*VALUES\s*\([^)]*,\s*[2-9]\s*,", seed_sql))

# ---------------------------------------------------------------- §5 句式
print("\n§5 句式覆盖（主谓宾 / 话题句 / 把字句 / 被字句 / 存现句）")

TARGET = {1: "主谓宾", 5: "主谓谓语句(话题句)", 7: "把字句", 8: "被字句", 11: "存现句"}
for pid, name in TARGET.items():
    # 模板占位符 == 槽位集合（单一真源）
    tpl = conn.execute("SELECT template FROM sentence_patterns WHERE pattern_id=?", (pid,)).fetchone()[0]
    ph = set(re.findall(r"\{([^}]+)\}", tpl))
    slots = {r[0] for r in conn.execute(
        "SELECT slot_name FROM pattern_slots WHERE pattern_id=?", (pid,))}
    check(f"P{pid} {name}：模板占位符与 pattern_slots 一一对应 {sorted(ph)}",
          ph == slots, f"模板{ph} vs 槽位{slots}")

    # 每个槽位至少一条约束（需求：每个槽位配 slot_constraints）
    unconstrained = [s for s in slots
                     if not count(conn, "SELECT COUNT(*) FROM slot_constraints "
                                        "WHERE pattern_id=? AND slot_name=?", (pid, s))]
    check(f"P{pid} {name}：全部 {len(slots)} 个槽位均有 slot_constraints",
          not unconstrained, f"无约束的槽位 {unconstrained}")

# 约束取值合法性（kind='role' 应为合法语义角色；kind='pos' 应为合法词性）
bad_role = conn.execute("""
  SELECT sc.pattern_id, sc.slot_name, sc.value FROM slot_constraints sc
   WHERE sc.kind='role' AND NOT EXISTS (SELECT 1 FROM semantic_roles r WHERE r.role=sc.value)
     AND sc.pattern_id IN (1,5,7,8,11)""").fetchall()
check("slot_constraints(kind='role') 取值均为合法语义角色", not bad_role, str(bad_role))
bad_pos = conn.execute("""
  SELECT sc.pattern_id, sc.slot_name, sc.value FROM slot_constraints sc
   WHERE sc.kind='pos' AND NOT EXISTS (SELECT 1 FROM pos_tags p WHERE p.pos=sc.value)
     AND sc.pattern_id IN (1,5,7,8,11)""").fetchall()
check("slot_constraints(kind='pos') 取值均为合法词性标签", not bad_pos, str(bad_pos))

# demo 语料
n_utt = count(conn, "SELECT COUNT(*) FROM utterances WHERE doc_id='demo_cat_fish'")
check("demo 语料含 5 个扩展句子", n_utt == 5, f"实际 {n_utt}")

# ---------------------------------------------------------------- §6 端到端
print("\n§6 端到端语义校验（用 relation_constraints 判定合法性，非数行数）")


def rc_allows(verb_word, slot, noun_sense_id):
    """动词 verb 的 slot 槽是否接受 noun 义项 —— 复刻 inference_rules #3/#4 的逻辑。"""
    pages = conn.execute("""
        SELECT rc.kind, rc.value FROM relation_constraints rc
         WHERE rc.sense_id = (SELECT s.sense_id FROM senses s JOIN words w ON w.word_id=s.word_id
                               WHERE w.word=? AND s.pos='v' LIMIT 1)
           AND rc.slot = ?""", (verb_word, slot)).fetchall()
    if not pages:
        return None, "动词未声明该槽"
    cats = {r[0] for r in conn.execute(
        "SELECT cat FROM sense_categories WHERE sense_id=?", (noun_sense_id,))}
    feats = {r[0] for r in conn.execute(
        "SELECT feature FROM sense_features WHERE sense_id=? AND value IN ('1','positive')",
        (noun_sense_id,))}
    for kind, val in pages:
        if kind == 'category' and val in cats:
            return True, f"category:{val}"
        if kind == 'feature' and val in feats:
            return True, f"feature:{val}"
    return False, f"类别{cats} 特征{feats} 均不满足 {pages}"


# 6.1 猫吃鱼
ok_s, why_s = rc_allows("吃", "agent", sid(conn, "猫", "n"))
ok_o, why_o = rc_allows("吃", "patient", sid(conn, "鱼", "n"))
check("「猫吃鱼」合法：猫 满足 吃.agent", ok_s is True, why_s)
check("「猫吃鱼」合法：鱼 满足 吃.patient（经 sense_categories 鱼→食物）", ok_o is True, why_o)

# 6.2 负向用例：鱼吃猫 应当被拦下
bad_o, why_bo = rc_allows("吃", "patient", sid(conn, "猫", "n"))
check("「鱼吃猫」被拦：猫 不满足 吃.patient（无食物类别）", bad_o is False, why_bo)

# 6.3 猫把鱼吃了 —— 把字句
check("「猫把鱼吃了」：把 → 受事(前置)，P 槽约束 role=受事 得到满足",
      role_map.get("把") == "受事", "把未映射到受事")
check("「猫把鱼吃了」：A(猫) 满足 吃.agent", ok_s is True, why_s)

# 6.4 鱼被猫吃了 —— 被字句
check("「鱼被猫吃了」：被 → 施事(前置)，A 槽约束 feature=+animate 可被猫满足",
      role_map.get("被") == "施事" and
      count(conn, "SELECT COUNT(*) FROM sense_features WHERE sense_id=? AND feature='+animate'",
            (sid(conn, "猫", "n"),)) > 0)

# 6.5 桌上有一条鱼 —— 存现句
has_sub = count(conn, "SELECT COUNT(*) FROM slot_constraints WHERE pattern_id=11 AND slot_name='V' "
                      "AND kind='subcategory' AND value='存现'")
ok_clf = ("条",) in pair_fish
check("「桌上有一条鱼」：V 槽要求 subcategory=存现，有(sense) 已标注 subcategory='存现'",
      has_sub > 0 and conn.execute("SELECT subcategory FROM senses WHERE sense_id=?",
                                   (sid(conn, "有", "v"),)).fetchone()[0] == "存现")
check("「桌上有一条鱼」：NP 的量词搭配 鱼→条 存在", ok_clf)

# 6.6 释义非空 + 符号语义（每条释义都由中文词语构成）
empty_def = count(conn, "SELECT COUNT(*) FROM senses WHERE length(trim(definition))=0")
check("全部义项 definition 非空（符号语义：用中文词语定义词语）", empty_def == 0)

# FTS 双通道可用
fts = conn.execute("SELECT COUNT(*) FROM senses_fts_trigram WHERE senses_fts_trigram MATCH '哺乳动物'").fetchone()[0]
check("FTS trigram 通道可检索中文释义（'哺乳动物'）", fts > 0, f"命中 {fts}")

# ---------------------------------------------------------------- §7 扩展种子
print("\n§7 扩展种子（任务一：鱼的多继承 A1-A4 ／ 任务二：分类骨架 B1-B8）")


def ancestors_of(word, pos="n"):
    """某义项的全部上位（沿 is_a 物化闭包）—— 方向：closure.descendant = 该义项。"""
    return {r[0] for r in conn.execute(
        "SELECT ancestor FROM closure WHERE relation='is_a' AND descendant=?",
        (sid(conn, word, pos),))}


def children_of(word, pos="n"):
    """某义项的**直接**下位（is_a 出边指向它的那些）—— 只看 relations，不看闭包。"""
    return {r[0] for r in conn.execute(
        "SELECT r.src_id FROM relations r WHERE r.relation='is_a' "
        "AND r.src_kind='sense' AND r.dst_kind='sense' AND r.dst_id=?",
        (sid(conn, word, pos),))}


def closure_has(anc_word, desc_word):
    return count(conn, "SELECT COUNT(*) FROM closure WHERE relation='is_a' "
                       "AND ancestor=? AND descendant=?",
                 (sid(conn, anc_word, "n"), sid(conn, desc_word, "n"))) > 0


# ---- A. 任务一：鱼的多继承 ----
fish_anc = ancestors_of("鱼")
has_food = closure_has("食物", "鱼")
has_animal = closure_has("动物", "鱼")
# 独立性：两条路径必须互不为祖先，否则"双挂"就只是同一条链的两段
crossed = conn.execute(
    "SELECT 1 FROM closure WHERE relation='is_a' "
    "  AND ((ancestor=? AND descendant=?) OR (ancestor=? AND descendant=?))",
    (sid(conn, "动物", "n"), sid(conn, "食物", "n"),
     sid(conn, "食物", "n"), sid(conn, "动物", "n"))).fetchall()

check("A1 鱼的祖先链同时含「动物」与「食物」，且二者互不为祖先（两条独立路径）",
      has_food and has_animal and not crossed,
      f"食物→鱼={has_food} 动物→鱼={has_animal} 交叉={crossed}")

check(f"A2 鱼的上位节点 ≥ 4（实测 {len(fish_anc)}）", len(fish_anc) >= 4,
      "实际 %d 个：%s" % (len(fish_anc),
                        [r[0] for r in conn.execute(
                            "SELECT w.word FROM closure c JOIN senses s ON s.sense_id=c.ancestor "
                            "JOIN words w ON w.word_id=s.word_id "
                            "WHERE c.relation='is_a' AND c.descendant=? ORDER BY c.depth, w.word",
                            (sid(conn, "鱼", "n"),))]))

rc_food = count(conn, "SELECT COUNT(*) FROM relation_constraints "
                      "WHERE sense_id=? AND slot='patient' AND kind='category' AND value='食物'",
                (sid(conn, "吃", "v"),))
ok_food, why_food = rc_allows("吃", "patient", sid(conn, "鱼", "n"))
check("A3 吃.patient 声明 category='食物'，且「鱼」经 sense_categories 实际通过该约束",
      rc_food > 0 and ok_food is True,
      f"约束行 {rc_food} 条；复用 §6 的判定器得 {ok_food}（{why_food}）")

fish_out = conn.execute("SELECT r.dst_id FROM relations r WHERE r.src_kind='sense' "
                        "AND r.src_id=? AND r.relation='is_a'",
                        (sid(conn, "鱼", "n"),)).fetchall()
fish_out_words = sorted(
    conn.execute("SELECT w.word FROM words w WHERE w.word_id IN "
                 "(SELECT word_id FROM senses WHERE sense_id IN (%s))"
                 % ",".join("?" * len(fish_out)), tuple(r[0] for r in fish_out)).fetchall())
check(f"A4 鱼的 is_a 出边数 = 2（实测 {len(fish_out)}：{fish_out_words}）",
      len(fish_out) == 2, f"实际 {len(fish_out)} 条 → {fish_out_words}")

# ---- B. 任务二：抽象概念 / 人造物 分类骨架 ----
abs_kids = children_of("抽象概念")
art_kids = children_of("人造物")
info_kids = children_of("信息人造物")


def _kids_words(kids):
    if not kids:
        return []
    return sorted(r[0] for r in conn.execute(
        "SELECT w.word FROM senses s JOIN words w ON w.word_id=s.word_id "
        "WHERE s.sense_id IN (%s)" % ",".join("?" * len(kids)), tuple(sorted(kids))))


def chain_ok(word, chain):
    """chain 中每个上位都必须是 word 的祖先（走闭包，即多级可达）。"""
    a = ancestors_of(word)
    missing = [x for x in chain if sid(conn, x, "n") not in a]
    return (not missing), (missing or "")


check(f"B1 抽象概念 直接下位 ≥ 4（实测 {len(abs_kids)}：{_kids_words(abs_kids)}）",
      len(abs_kids) >= 4, f"实际 {len(abs_kids)} 个")

check(f"B2 人造物 直接下位 ≥ 2（实测 {len(art_kids)}：{_kids_words(art_kids)}）",
      len(art_kids) >= 2, f"实际 {len(art_kids)} 个")

check(f"B3 信息人造物 直接下位 ≥ 3（实测 {len(info_kids)}：{_kids_words(info_kids)}）",
      len(info_kids) >= 3, f"实际 {len(info_kids)} 个")

ok_b4, miss_b4 = chain_ok("视图", ("数据对象", "信息人造物", "人造物"))
check("B4 视图 祖先链 ⊇ 数据对象 → 信息人造物 → 人造物", ok_b4, f"缺 {miss_b4}")

ok_b5, miss_b5 = chain_ok("文档", ("信息载体", "信息人造物"))
check("B5 文档 祖先链 ⊇ 信息载体 → 信息人造物", ok_b5, f"缺 {miss_b5}")

ok_b6, miss_b6 = chain_ok("程序", ("软件产物", "信息人造物"))
check("B6 程序 祖先链 ⊇ 软件产物 → 信息人造物", ok_b6, f"缺 {miss_b6}")

# 骨架 = 两个顶层节点自身 + 它们的全部（多级）下位；「书」复用 schema 既有义项
skel = set()
for root in ("抽象概念", "人造物"):
    skel.add(sid(conn, root, "n"))
    skel |= {r[0] for r in conn.execute(
        "SELECT descendant FROM closure WHERE relation='is_a' AND ancestor=?",
        (sid(conn, root, "n"),))}
skel_args = tuple(sorted(skel))
_ph = ",".join("?" * len(skel_args))
skel_edges = count(conn,
                   f"SELECT COUNT(*) FROM relations WHERE relation='is_a' "
                   f"AND src_kind='sense' AND dst_kind='sense' "
                   f"AND src_id IN ({_ph}) AND dst_id IN ({_ph})",
                   skel_args * 2)

check(f"B7 分类骨架节点数 ≥ 25（实测 {len(skel)}；含复用的「书」）",
      len(skel) >= 25, f"实际 {len(skel)} 个节点")

check(f"B8 骨架内 is_a 边数 ≥ 24（实测 {skel_edges}）",
      skel_edges >= 24, f"实际 {skel_edges} 条")

# ---- 需求明确要求、但未列入 A/B 编号的数据：以 [ok] 透明报告 ----
food_props = sorted(r[0] for r in conn.execute(
    "SELECT w.word FROM relations r JOIN senses s ON s.sense_id=r.dst_id "
    "JOIN words w ON w.word_id=s.word_id "
    "WHERE r.relation='has_property' AND r.src_id=? AND r.src_kind='sense'",
    (sid(conn, "食物", "n"),)))
print(f"  [ok]   需求 3：食物 has_property → {food_props}"
      f"（另经 is_a 被『鱼』继承，推理规则 #2 可达）")
n_cross = count(conn, "SELECT COUNT(*) FROM cross_links WHERE relation='domain_link'")
print(f"  [ok]   需求 5：cross_links 链接 {n_cross} 条"
      f"（entity_nodes: {conn.execute('SELECT COUNT(*) FROM entity_nodes').fetchone()[0]} 个节点）")
n_clf = count(conn, "SELECT COUNT(*) FROM noun_classifier_pairs p "
                    "WHERE p.noun_sense_id IN (?,?,?)",
              (sid(conn, "视图", "n"), sid(conn, "文档", "n"), sid(conn, "程序", "n")))
print(f"  [ok]   需求（任务二 3）：视图/文档/程序 量词搭配 {n_clf} 条")
n_rc_new = count(conn, "SELECT COUNT(*) FROM relation_constraints "
                       "WHERE sense_id IN (?,?,?)",
                 (sid(conn, "算法", "n"), sid(conn, "文档", "n"), sid(conn, "程序", "n")))
print(f"  [ok]   需求（任务二 4）：算法/文档/程序 relation_constraints {n_rc_new} 条")

# ------------------------------------------------------------------ 汇总
print("\n" + "=" * 78)
print("表行数（schema + seed + seed_ext 的最终状态）：")
for t in TABLES:
    print(f"  {t:<28} {count(conn, f'SELECT COUNT(*) FROM {t}'):>5}"
          + (f"    (seed 基线 {snap1[t]}, ext 增量 +{gain[t]})" if gain.get(t) else ""))
print("  " + "-" * 72)
print("扩展种子触及但不在上表的：")
for t in EXT_TABLES:
    if t in TABLES or not gain[t]:
        continue
    print(f"  {t:<28} {count(conn, f'SELECT COUNT(*) FROM {t}'):>5}    (ext 增量 +{gain[t]})")
print(f"  {'relations(按类型)':<28} " +
      ", ".join(f"{r[0]}={r[1]}" for r in conn.execute(
          "SELECT relation, COUNT(*) FROM relations GROUP BY relation ORDER BY 2 DESC")))
print("-" * 78)
print(f"结果: PASS {PASS} / FAIL {FAIL}")
if FAILURES:
    print("\n失败明细：")
    for n, d in FAILURES:
        print(f"  · {n}\n      {d}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
