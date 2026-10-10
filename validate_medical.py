#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_medical.py —— seed_medical_v3.sql 的建库 + 行为验证

风格与 validate_seed.py / validate_closure.py 一致：逐项 check + PASS/FAIL 计数 +
非零退出码。**不改动 validate_seed.py**（它的 77 项编号属于通用库，另行独立计数）。

分十三段：
  §M1 加载与提交       schema → seed → ext → medical → imaging →
                        patch_symptom → patch_symptom_layers →
                        patch_symptom_severity 全链路；落盘重开验证
  §M2 词层与义项层     30 个医学义项 + 信号兜底 + 类别挂载合法性
  §M3 关系类型登记     11 个新类型、4 对互逆成对、3 个领域类型、传递性不变量
  §M4 is_a 骨架层级    五条分支 + 多继承 + 深链可达性
  §M5 has_part 与医学关系  has_symptom / treated_by / responsible_for / indicates
  §M6 闭包自动派生     depth>1 实测、差分对拍、反向完备性、零手工 INSERT
  §M7 跨库 cross_links 三条领域边 + 端点合法性
  §M8 幂等与自足性     二次执行零漂移、PRAGMA 位置、事务配对、前置依赖硬检查
  §M9 推理链演示       非计数的 [ok] 输出（展示需求里的关系如何被查询/继承）
  §M10 症状继承补丁     seed_medical_v3_patch_symptom.sql 的 6 症状 + 5 层边 +
                        继承结果集（疾病层 / 脑部疾病层）+ 幂等与依赖闸门
  §M11 症状严重度义素   seed_medical_v3_patch_symptom_severity.sql 的 8 症状
                        严重度标注（+severe/+moderate/+mild）+ 继承检索
                        （closure ⋈ is_a ⋈ has_symptom ⟕ sense_features）
  §M12 症状分层补丁     seed_medical_v3_patch_symptom_layers.sql 的 3 个新义项
                        （炎症性疾病 / 感染性疾病 / 脑炎）+ 4 is_a + 4 has_symptom
                        —— 把 §M11 里"口径收窄"的 2 条需求断言**按字面补齐**
  §M13 影像特征子树     seed_medical_v3_imaging.sql 的 37 个影像义项（4040-4076）
                        + 5 个特征维度 + 结节/恶性/良性结节 + typically_has
                        + 12 个影像义素 + 跨库边（信号 → 强化特征）

运行： python validate_medical.py
"""
import io
import os
import pathlib
import re
import sqlite3
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = HERE / "schema_v3.sql"
SEED = HERE / "seed_general_v3.sql"
SEED_EXT = HERE / "seed_general_v3_ext.sql"
SEED_MED = HERE / "seed_medical_v3.sql"
IMG_PATCH = HERE / "seed_medical_v3_imaging.sql"                  # §M13：影像特征子树
SEED_MED_PATCH = HERE / "seed_medical_v3_patch_symptom.sql"   # §M10：症状继承补丁
LAY_PATCH = HERE / "seed_medical_v3_patch_symptom_layers.sql"     # §M12：症状分层补丁
SEV_PATCH = HERE / "seed_medical_v3_patch_symptom_severity.sql"  # §M11：症状严重度义素

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


def connect(path=":memory:"):
    c = sqlite3.connect(str(path))
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("PRAGMA recursive_triggers = ON")
    return c


def load(c, path):
    with open(str(path), encoding="utf-8") as f:
        c.executescript(f.read())


def count(c, sql, args=()):
    return c.execute(sql, args).fetchone()[0]


def sid(c, word, pos="n"):
    """按自然键取 sense_id —— 与种子文件同一套寻址方式。"""
    r = c.execute("SELECT s.sense_id FROM senses s JOIN words w ON w.word_id = s.word_id "
                  "WHERE w.word = ? AND s.pos = ? LIMIT 1", (word, pos)).fetchone()
    return r[0] if r else None


def ancestors(c, word, pos="n", relation="is_a"):
    """某义项的全部上位（沿物化闭包）。方向：closure.descendant = 该义项。"""
    return {r[0] for r in c.execute(
        "SELECT ancestor FROM closure WHERE relation=? AND descendant=?",
        (relation, sid(c, word, pos)))}


def parts_of(c, word, relation="has_part"):
    """整体 → 部分，返回 {部分 sense_id: depth}。

    ⚠ 方向陷阱：closure 的行方向**恒为边的反向**（不变量 I3：ancestor = 边的 dst）。
    has_part 的边是「整体 → 部分」，dst 是**部分**，故闭包行里
    ancestor 落在**部分**一侧、descendant 落在**整体**一侧。
    要「取 X 的全部部分」必须查 WHERE descendant = X，再取 ancestor。
    """
    return {r[0]: r[1] for r in c.execute(
        "SELECT ancestor, depth FROM closure WHERE relation=? AND descendant=?",
        (relation, sid(c, word, "n")))}


def word_of(c, sense_id):
    r = c.execute("SELECT w.word FROM senses s JOIN words w ON w.word_id = s.word_id "
                  "WHERE s.sense_id = ?", (sense_id,)).fetchone()
    return r[0] if r else None


def words_of(c, ids):
    out = [word_of(c, i) for i in ids]
    return sorted([w for w in out if w])


def reachable(c, anc_word, desc_word, relation="is_a"):
    return count(c, "SELECT COUNT(*) FROM closure WHERE relation=? AND ancestor=? AND descendant=?",
                 (relation, sid(c, anc_word), sid(c, desc_word))) > 0


print("=" * 78)
print("seed_medical_v3.sql 验证（医学库骨架 + 跨库链接）")
print("=" * 78)

# =============================================================== §M1 加载与提交
print("\n§M1 加载与提交"
      "（schema → seed → ext → medical → imaging → patch_symptom"
      " → patch_symptom_layers → severity）")

conn = connect()
try:
    load(conn, SCHEMA)
    load(conn, SEED)
    load(conn, SEED_EXT)
    n_before = count(conn, "SELECT COUNT(*) FROM senses")
    load(conn, SEED_MED)
    n_after = count(conn, "SELECT COUNT(*) FROM senses")
    n_img_before = count(conn, "SELECT COUNT(*) FROM senses")
    n_img_rel_before = count(conn, "SELECT COUNT(*) FROM relations")
    n_img_clo_before = count(conn, "SELECT COUNT(*) FROM closure")
    n_img_sf_before = count(conn, "SELECT COUNT(*) FROM sense_features")
    n_img_mf_before = count(conn, "SELECT COUNT(*) FROM semantic_features")
    n_img_en_before = count(conn, "SELECT COUNT(*) FROM entity_nodes")
    n_img_rt_before = count(conn, "SELECT COUNT(*) FROM relation_types")
    load(conn, IMG_PATCH)
    n_img = count(conn, "SELECT COUNT(*) FROM senses")
    n_img_rel = count(conn, "SELECT COUNT(*) FROM relations")
    n_img_clo = count(conn, "SELECT COUNT(*) FROM closure")
    n_img_sf = count(conn, "SELECT COUNT(*) FROM sense_features")
    n_img_mf = count(conn, "SELECT COUNT(*) FROM semantic_features")
    n_img_en = count(conn, "SELECT COUNT(*) FROM entity_nodes")
    n_img_rt = count(conn, "SELECT COUNT(*) FROM relation_types")
    load(conn, SEED_MED_PATCH)
    n_patch = count(conn, "SELECT COUNT(*) FROM senses")
    n_rel_before_lay = count(conn, "SELECT COUNT(*) FROM relations")
    n_clo_before_lay = count(conn, "SELECT COUNT(*) FROM closure")
    load(conn, LAY_PATCH)
    n_lay = count(conn, "SELECT COUNT(*) FROM senses")
    n_rel_lay = count(conn, "SELECT COUNT(*) FROM relations")
    n_clo_lay = count(conn, "SELECT COUNT(*) FROM closure")
    n_sf_before = count(conn, "SELECT COUNT(*) FROM sense_features")
    n_mf_before = count(conn, "SELECT COUNT(*) FROM semantic_features")
    load(conn, SEV_PATCH)
    n_sf = count(conn, "SELECT COUNT(*) FROM sense_features")
    n_mf = count(conn, "SELECT COUNT(*) FROM semantic_features")
    check("八个 SQL 文件按序无错误执行（schema → seed → ext → medical → imaging"
          " → patch_symptom → patch_symptom_layers → patch_symptom_severity）", True)
except sqlite3.Error as e:
    check("八个 SQL 文件按序无错误执行（schema → seed → ext → medical → imaging"
          " → patch_symptom → patch_symptom_layers → patch_symptom_severity）", False,
          f"{type(e).__name__}: {e}")
    sys.exit(1)

check(f"医学库新增义项 = 31（{n_before} → {n_after}）", n_after - n_before == 31,
      f"新增 {n_after - n_before}")
check(f"影像补丁新增义项 = 37（{n_img_before} → {n_img}）", n_img - n_img_before == 37,
      f"新增 {n_img - n_img_before}")
check(f"影像补丁新增关系 = 46（relations {n_img_rel_before} → {n_img_rel}）："
      f"37 is_a + 2 has_property + 6 typically_has + 1 跨库",
      n_img_rel - n_img_rel_before == 46, f"新增 {n_img_rel - n_img_rel_before}")
check(f"影像补丁经触发器派生闭包 = 148（closure {n_img_clo_before} → {n_img_clo}）："
      f"is_a 146 + has_property 2（补丁本身零 closure INSERT）",
      n_img_clo - n_img_clo_before == 148, f"新增 {n_img_clo - n_img_clo_before}")
check(f"影像补丁新增义素标注 = 12（sense_features {n_img_sf_before} → {n_img_sf}）",
      n_img_sf - n_img_sf_before == 12, f"新增 {n_img_sf - n_img_sf_before}")
check(f"影像补丁新增义素定义 = 12（semantic_features {n_img_mf_before} → {n_img_mf}）",
      n_img_mf - n_img_mf_before == 12, f"新增 {n_img_mf - n_img_mf_before}")
check(f"影像补丁新增领域节点 = 1（entity_nodes {n_img_en_before} → {n_img_en}）",
      n_img_en - n_img_en_before == 1, f"新增 {n_img_en - n_img_en_before}")
check(f"影像补丁新增关系类型 = 2（relation_types {n_img_rt_before} → {n_img_rt}）："
      f"typically_has / typical_of", n_img_rt - n_img_rt_before == 2,
      f"新增 {n_img_rt - n_img_rt_before}")
check(f"症状继承补丁新增义项 = 6（{n_img} → {n_patch}）", n_patch - n_img == 6,
      f"新增 {n_patch - n_img}")
check(f"症状分层补丁新增义项 = 3（{n_patch} → {n_lay}）", n_lay - n_patch == 3,
      f"新增 {n_lay - n_patch}")
check(f"症状分层补丁新增关系 = 8（relations {n_rel_before_lay} → {n_rel_lay}）："
      f"4 is_a + 4 has_symptom", n_rel_lay - n_rel_before_lay == 8,
      f"新增 {n_rel_lay - n_rel_before_lay}")
check(f"症状分层补丁经触发器派生闭包 = 15（closure {n_clo_before_lay} → {n_clo_lay}）"
      f"—— 补丁本身零 closure INSERT", n_clo_lay - n_clo_before_lay == 15,
      f"新增 {n_clo_lay - n_clo_before_lay}")
check(f"严重度补丁新增义素标注 = 8（sense_features {n_sf_before} → {n_sf}）",
      n_sf - n_sf_before == 8, f"新增 {n_sf - n_sf_before}")
check(f"严重度补丁新增义素定义 = 3（semantic_features {n_mf_before} → {n_mf}）",
      n_mf - n_mf_before == 3, f"新增 {n_mf - n_mf_before}")

check("PRAGMA integrity_check = ok",
      conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok")
check("PRAGMA foreign_key_check 无违规",
      not conn.execute("PRAGMA foreign_key_check").fetchall())

# 落盘验证：文件库 → 关闭 → 重开另一个连接。同连接内读得到 ≠ 已提交。
_dbf = HERE / "_medicalcheck.db"
for _p in (str(_dbf), str(_dbf) + "-wal", str(_dbf) + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
try:
    c_file = connect(_dbf)
    load(c_file, SCHEMA)
    load(c_file, SEED)
    load(c_file, SEED_EXT)
    load(c_file, SEED_MED)
    load(c_file, IMG_PATCH)
    load(c_file, SEED_MED_PATCH)
    load(c_file, LAY_PATCH)
    load(c_file, SEV_PATCH)
    c_file.close()
    c_re = sqlite3.connect(str(_dbf))
    n_re = c_re.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
    n_cl = c_re.execute("SELECT COUNT(*) FROM closure").fetchone()[0]
    c_re.close()
    check(f"落盘验证：重开连接后数据完整提交（senses={n_re}, closure={n_cl}）",
          n_re == n_lay and n_cl == count(conn, "SELECT COUNT(*) FROM closure"),
          f"senses={n_re} closure={n_cl}")
    # 临时库文件不得残留
    for _p in (str(_dbf), str(_dbf) + "-wal", str(_dbf) + "-shm"):
        if os.path.exists(_p):
            os.remove(_p)
    check("落盘验证后临时库已清理（不留 _medicalcheck.db）", not _dbf.exists())
finally:
    for _p in (str(_dbf), str(_dbf) + "-wal", str(_dbf) + "-shm"):
        if os.path.exists(_p):
            os.remove(_p)

# ============================================================= §M2 词层与义项
print("\n§M2 词层与义项层")

MED_NODES = [
    # 顶层
    "医学概念",
    # 解剖结构
    "解剖结构", "神经系统", "脑", "脊髓", "大脑", "小脑", "大脑皮层",
    # 生理过程
    "生理过程", "神经传导", "信号处理", "记忆",
    # 病理状态
    "病理状态", "疾病", "脑部疾病", "癫痫", "脑卒中", "神经退行性疾病",
    "阿尔茨海默病", "症状", "抽搐", "记忆障碍",
    # 干预与检测
    "干预手段", "药物治疗", "神经调控", "脑深部电刺激", "医学检测", "脑电图",
    # 营养与信号
    "营养素", "脑电图信号",
]
missing = [w for w in MED_NODES if sid(conn, w) is None]
check(f"需求列出的 30 个医学义项全部存在（实测缺 {len(missing)}）", not missing, f"缺 {missing}")

check("「信号」通用库兜底词条已建立（需求 3 的跨库端点，实测通用库原本无此词）",
      sid(conn, "信号") is not None, "signal sense 缺失")

# sense_id = word_id 的 1:1 约定（v0.1 每词单义项）
mismatch = [w for w in MED_NODES
            if sid(conn, w) is not None
            and count(conn, "SELECT COUNT(*) FROM senses s JOIN words wd ON wd.word_id=s.word_id "
                            "WHERE wd.word=? AND s.pos='n' AND s.is_primary=1", (w,)) != 1]
check("每个医学义项恰有 1 个 is_primary=1 的义项（v0.1 单词义）", not mismatch, str(mismatch))

# 符号语义：释义只能由中文词语组成，且非空 —— 一旦引入数字/编码串就说明混入了非符号语义
bad_def = conn.execute(
    "SELECT w.word, s.definition FROM senses s JOIN words w ON w.word_id=s.word_id "
    "WHERE s.sense_id BETWEEN 4001 AND 4030 "
    "  AND (length(trim(s.definition))=0 "
    "       OR s.definition GLOB '*[0-9a-zA-Z]*')").fetchall()
check("30 条医学释义非空、且不含数字/拉丁字母（纯中文词语定义，无向量化痕迹）",
      not bad_def, str(bad_def))

# 类别挂载：每个医学义项至少 1 个类别，且取值合法
no_cat = [w for w in MED_NODES
          if not count(conn, "SELECT COUNT(*) FROM sense_categories WHERE sense_id=?",
                       (sid(conn, w),))]
check("全部医学义项均已挂载 sense_categories", not no_cat, f"缺 {no_cat}")

bad_cat = conn.execute("""
  SELECT w.word, sc.cat FROM sense_categories sc
    JOIN words w ON w.word_id = (SELECT word_id FROM senses WHERE sense_id=sc.sense_id)
   WHERE (sc.sense_id BETWEEN 4001 AND 4030 OR sc.sense_id = 4101)
     AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat)""").fetchall()
check("医学义项的类别取值均为合法 semantic_categories（未新增类别行）", not bad_cat, str(bad_cat))

# ============================================================ §M3 关系类型登记
print("\n§M3 关系类型登记")

NEW_RELS = ["has_symptom", "symptom_of", "treated_by", "treats",
            "responsible_for", "depends_on", "indicates", "indicated_by",
            "nutritional_relevance", "functional_analogy", "domain_specialization"]
miss_rel = [r for r in NEW_RELS
            if not count(conn, "SELECT COUNT(*) FROM relation_types WHERE relation=?", (r,))]
check(f"11 个新关系类型全部登记（实测缺 {len(miss_rel)}）", not miss_rel, f"缺 {miss_rel}")

bad_inv = conn.execute("""
  SELECT rt.relation, rt.inverse_of FROM relation_types rt
    LEFT JOIN relation_types t2 ON t2.relation = rt.inverse_of
   WHERE rt.relation IN ('has_symptom','symptom_of','treated_by','treats',
                         'responsible_for','depends_on','indicates','indicated_by')
     AND (rt.inverse_of IS NULL OR t2.relation IS NULL OR t2.inverse_of <> rt.relation)""").fetchall()
check("4 对医学关系互逆成对（inverse_of 双向一致，无悬空/无单向）", not bad_inv, str(bad_inv))

domain_ok = conn.execute(
    "SELECT COUNT(*) FROM relation_types WHERE relation IN "
    "('nutritional_relevance','functional_analogy','domain_specialization') "
    "AND relation_class='领域' AND direction='directed'").fetchone()[0]
check("3 个跨库链接类型均为 relation_class='领域' 且 directed", domain_ok == 3, f"实测 {domain_ok}")

trans_new = conn.execute(
    "SELECT relation FROM relation_types WHERE relation IN (%s) AND is_transitive=1"
    % ",".join("?" * len(NEW_RELS)), NEW_RELS).fetchall()
check("11 个新类型一律 is_transitive=0（跨库边一端是 node，物化不可达；"
      "显式声明以消除『is_transitive=1 却缺 closure_policy → 静默不物化』的坑）",
      not trans_new, f"意外标记为可传递：{trans_new}")

# 全库不变量（本项目记录过的真实坑）
no_policy = conn.execute("""
  SELECT rt.relation FROM relation_types rt
   WHERE rt.is_transitive=1
     AND NOT EXISTS (SELECT 1 FROM closure_policy cp WHERE cp.relation = rt.relation)""").fetchall()
check("全库不变量：所有 is_transitive=1 的关系都有 closure_policy 行",
      not no_policy, f"缺策略行 {no_policy}")

# ============================================================== §M4 is_a 骨架
print("\n§M4 is_a 骨架层级（顶层 / 解剖 / 生理 / 病理 / 干预 / 检测）")

check("顶层：医学概念 ⊑ 抽象概念（挂到 ext 建的抽象概念下，非悬空根）",
      reachable(conn, "抽象概念", "医学概念"))
check("顶层：医学概念的 is_a 出边恰为 1 条",
      count(conn, "SELECT COUNT(*) FROM relations WHERE relation='is_a' AND src_kind='sense' "
                  "AND src_id=? AND dst_kind='sense'", (sid(conn, "医学概念"),)) == 1)

med_desc = {r[0] for r in conn.execute(
    "SELECT descendant FROM closure WHERE relation='is_a' AND ancestor=?",
    (sid(conn, "医学概念"),))}
# 24（seed_medical_v3 骨架）+ 6（patch_symptom 新增的通用症状，经 症状 → 病理状态 上溯）
# + 3（patch_symptom_layers 新增的疾病分层：炎症性疾病 / 感染性疾病 / 脑炎）
# + 37（imaging 新增的影像子树：影像特征 33 + 病理结构 4）= 70
check(f"医学概念 的 is_a 下位 = 70 个（实测 {len(med_desc)}；"
      f"24 骨架 + 6 症状 + 3 分层 + 37 影像）",
      len(med_desc) == 70,
      f"实际 {len(med_desc)}：{words_of(conn, med_desc)}")

# 骨架锚定：30 个医学节点都必须能连到「医学概念」或两个显式声明的通用库锚点。
# 「向上」有两种合法走法：is_a 的 dst（上位）、has_part 的 src（所属整体）——
# 因为 大脑/小脑/大脑皮层 只通过 has_part 挂在「脑」下，没有 is_a 上位。
def upper_closure(word):
    """沿『更一般 / 所属整体』方向收集可达节点（含自身）。

    两条上行走法：
      · is_a：   cur --is_a-->   dst      → dst 是 cur 的上位，向上取 dst
      · has_part：src --has_part--> cur   → src 是 cur 所属的整体，向上取 src
    """
    seen, stack = set(), [sid(conn, word)]
    while stack:
        cur = stack.pop()
        if cur is None or cur in seen:
            continue
        seen.add(cur)
        for r in conn.execute(
                "SELECT dst_id FROM relations WHERE relation='is_a' "
                "AND src_kind='sense' AND dst_kind='sense' AND src_id=?", (cur,)):
            stack.append(r[0])
        for r in conn.execute(
                "SELECT src_id FROM relations WHERE relation='has_part' "
                "AND src_kind='sense' AND dst_kind='sense' AND dst_id=?", (cur,)):
            stack.append(r[0])
    return seen


_anchors = {"医学概念": sid(conn, "医学概念"), "营养物": sid(conn, "营养物"),
            "信号": sid(conn, "信号")}
orphans = [w for w in MED_NODES if sid(conn, w) is not None
           and not (upper_closure(w) & set(_anchors.values()))]
check(f"骨架锚定：30 个医学节点个个可达 医学概念 或通用库锚点（营养物/信号），无孤岛"
      f"（is_a 下位 24 + 医学概念自身 1 + 仅 has_part 挂载 3 + 通用库锚点 2 = 30）",
      not orphans, f"孤岛 {orphans}")

# 「非 is_a 锚定」的节点应恰为：仅靠 has_part 挂载的 3 个 + 挂通用库锚点的 2 个
via_part = {w for w in MED_NODES if sid(conn, w) is not None
            and not (ancestors(conn, w) | {sid(conn, w)}) & {sid(conn, "医学概念")}
            and not (ancestors(conn, w) & {sid(conn, "营养物"), sid(conn, "信号")})}
via_core = {w for w in MED_NODES if sid(conn, w) is not None
            and (ancestors(conn, w) & {sid(conn, "营养物"), sid(conn, "信号")})}
check("is_a 之外的挂载方式仅两类：has_part（大脑/小脑/大脑皮层）与通用库锚点（营养素/脑电图信号）",
      via_part == {"大脑", "小脑", "大脑皮层"} and via_core == {"营养素", "脑电图信号"},
      f"has_part 挂载 {sorted(via_part)}；通用库锚点 {sorted(via_core)}")

# 解剖链：脑 ⊑ 神经系统 ⊑ 解剖结构
check("解剖链：脑 ⊑ 神经系统 ⊑ 解剖结构 ⊑ 医学概念",
      reachable(conn, "神经系统", "脑") and reachable(conn, "解剖结构", "脑")
      and reachable(conn, "医学概念", "脑"))
check("解剖链：脊髓 与 脑 同属神经系统（兄弟节点）",
      reachable(conn, "神经系统", "脊髓") and reachable(conn, "神经系统", "脑"))

# 生理链
check("生理链：神经传导 / 信号处理 / 记忆 均 ⊑ 生理过程",
      all(reachable(conn, "生理过程", w) for w in ("神经传导", "信号处理", "记忆")))

# 病理链
check("病理链：脑部疾病 ⊑ 疾病 ⊑ 病理状态 ⊑ 医学概念",
      reachable(conn, "疾病", "脑部疾病") and reachable(conn, "病理状态", "脑部疾病")
      and reachable(conn, "医学概念", "脑部疾病"))
check("病理链：症状 ⊑ 病理状态（与疾病同一层级的另一分支）",
      reachable(conn, "病理状态", "症状") and not reachable(conn, "疾病", "症状"))
check("病理链：抽搐 / 记忆障碍 均 ⊑ 症状",
      all(reachable(conn, "症状", w) for w in ("抽搐", "记忆障碍")))
check("疾病分支：癫痫 / 脑卒中 / 神经退行性疾病 均为脑部疾病下位",
      all(reachable(conn, "脑部疾病", w) for w in ("癫痫", "脑卒中", "神经退行性疾病")))
check("疾病分支：阿尔茨海默病 ⊑ 神经退行性疾病（经 2 跳到脑部疾病）",
      reachable(conn, "神经退行性疾病", "阿尔茨海默病")
      and reachable(conn, "脑部疾病", "阿尔茨海默病"))

# 干预链 / 检测链
check("干预链：药物治疗 / 神经调控 ⊑ 干预手段，脑深部电刺激 ⊑ 神经调控",
      reachable(conn, "干预手段", "药物治疗") and reachable(conn, "干预手段", "神经调控")
      and reachable(conn, "神经调控", "脑深部电刺激"))
check("检测链：脑电图 ⊑ 医学检测 ⊑ 医学概念",
      reachable(conn, "医学检测", "脑电图") and reachable(conn, "医学概念", "脑电图"))
check("跨库层级：脑电图信号 ⊑ 信号（人类可读的层级，与 cross_links 领域边并存）",
      reachable(conn, "信号", "脑电图信号"))
check("跨库层级：营养素 ⊑ 营养物（复用 ext 建的功能分类链）",
      reachable(conn, "营养物", "营养素"))

# 多继承
epi_out = conn.execute(
    "SELECT dst_id FROM relations WHERE relation='is_a' AND src_kind='sense' "
    "AND dst_kind='sense' AND src_id=?", (sid(conn, "癫痫"),)).fetchall()
epi_parents = words_of(conn, [r[0] for r in epi_out])
check(f"多继承：癫痫 的 is_a 出边 = 2 条（实测 {len(epi_out)}：{epi_parents}）",
      len(epi_out) == 2 and set(epi_parents) == {"疾病", "脑部疾病"},
      f"{len(epi_out)} 条 → {epi_parents}")
d_direct = count(conn, "SELECT depth FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
                 (sid(conn, "疾病"), sid(conn, "癫痫")))
check("多继承：癫痫→疾病 同时有直接边(1)与经脑部疾病(2)两条路径，闭包取 MIN=1",
      d_direct == 1, f"depth={d_direct}")

anc_ad = ancestors(conn, "阿尔茨海默病")
max_depth = max(r[0] for r in conn.execute(
    "SELECT depth FROM closure WHERE relation='is_a' AND descendant=?", (sid(conn, "阿尔茨海默病"),)))
check(f"深链可达：阿尔茨海默病 上位 {len(anc_ad)} 个，最深 {max_depth} 跳（含抽象概念）",
      len(anc_ad) == 6 and max_depth == 6,
      f"{len(anc_ad)} 个 / 最深 {max_depth}：{words_of(conn, anc_ad)}")

# ==================================================== §M5 has_part 与医学关系
print("\n§M5 has_part 与医学关系（has_symptom / treated_by / responsible_for / indicates）")

brain_parts = parts_of(conn, "脑")
check(f"has_part：脑 的部分 = 大脑 / 小脑 / 大脑皮层（实测 {words_of(conn, brain_parts)}）",
      set(words_of(conn, brain_parts)) == {"大脑", "小脑", "大脑皮层"},
      str(words_of(conn, brain_parts)))
check("has_part 传递：脑 → 大脑皮层 物化 depth=2（全库唯一非 is_a 的 depth>1 行）",
      brain_parts.get(sid(conn, "大脑皮层")) == 2,
      f"depth={brain_parts.get(sid(conn, '大脑皮层'))}")
check("has_part 传递的方向正确：部分在 ancestor 侧、整体在 descendant 侧（I3：ancestor = 边的 dst）",
      count(conn, "SELECT COUNT(*) FROM closure WHERE relation='has_part' AND ancestor=? "
                  "AND descendant=? AND depth=2",
            (sid(conn, "大脑皮层"), sid(conn, "脑"))) == 1
      and count(conn, "SELECT COUNT(*) FROM closure WHERE relation='has_part' AND ancestor=? "
                      "AND descendant=?",
                (sid(conn, "脑"), sid(conn, "大脑皮层"))) == 0)

_non_isa_deep = conn.execute(
    "SELECT relation, COUNT(*) FROM closure WHERE depth>1 AND relation<>'is_a' "
    "GROUP BY relation ORDER BY relation").fetchall()
check(f"非 is_a 的 depth>1 物化行恰为 1 条（实测 {_non_isa_deep}）：has_part 的传递性真实生效",
      _non_isa_deep == [("has_part", 1)], str(_non_isa_deep))


def rel_has(src, rel, dst):
    return count(conn, "SELECT COUNT(*) FROM relations WHERE relation=? AND src_kind='sense' "
                       "AND dst_kind='sense' AND src_id=? AND dst_id=?",
                 (rel, sid(conn, src), sid(conn, dst))) > 0


check("has_symptom：癫痫 → 抽搐", rel_has("癫痫", "has_symptom", "抽搐"))
check("has_symptom：阿尔茨海默病 → 记忆障碍", rel_has("阿尔茨海默病", "has_symptom", "记忆障碍"))
check("treated_by：癫痫 → 神经调控（需求原文）", rel_has("癫痫", "treated_by", "神经调控"))
check("treated_by：癫痫 → 药物治疗（同病多手段，验证一病可挂多个干预）",
      rel_has("癫痫", "treated_by", "药物治疗"))
check("responsible_for：大脑 → 记忆", rel_has("大脑", "responsible_for", "记忆"))
check("indicates：记忆障碍 → 阿尔茨海默病（与 has_symptom 方向相反、立场不同）",
      rel_has("记忆障碍", "indicates", "阿尔茨海默病"))

check("has_symptom 与 indicates 是**两个不同关系类型**（不是同一关系的互逆）",
      count(conn, "SELECT COUNT(*) FROM relation_types WHERE relation='has_symptom' "
                  "AND inverse_of='symptom_of'") == 1
      and count(conn, "SELECT COUNT(*) FROM relation_types WHERE relation='indicates' "
                      "AND inverse_of='indicated_by'") == 1)

# v_relations_all 的逆向推导可用（不必冗余存反向边）
inv_ok = count(conn, """
  SELECT COUNT(*) FROM v_relations_all
   WHERE relation='treats' AND src_id=? AND dst_id=?""",
               (sid(conn, "神经调控"), sid(conn, "癫痫"))) > 0
check("v_relations_all 可在线推导逆关系：神经调控 --treats--> 癫痫（无冗余反向行）", inv_ok)

NONTRANS = {"has_symptom", "treated_by", "responsible_for", "indicates",
            "nutritional_relevance", "functional_analogy", "domain_specialization"}
leaked = conn.execute("SELECT DISTINCT relation FROM closure WHERE relation IN (%s)"
                      % ",".join("?" * len(NONTRANS)), tuple(sorted(NONTRANS))).fetchall()
check("6 条医学关系 + 3 条跨库边均未产生闭包行（is_transitive=0 的实际行为一致）",
      not leaked, f"意外物化 {leaked}")

# ========================================================= §M6 闭包自动派生
print("\n§M6 闭包自动派生（禁止手工插入深层闭包）")

with open(str(SEED_MED), encoding="utf-8") as f:
    raw = f.read()
check("零 closure 的 INSERT（实测 %d 条）"
      % len(re.findall(r"(?m)^\s*INSERT\s+INTO\s+closure\b", raw, re.I)),
      len(re.findall(r"(?m)^\s*INSERT\s+INTO\s+closure\b", raw, re.I)) == 0)
check("未手工插入任何 depth>1 的闭包行",
      not re.search(r"INSERT INTO closure\([^)]*\)\s*VALUES\s*\([^)]*,\s*[2-9]\s*,", raw))

depth_gt1 = count(conn, "SELECT COUNT(*) FROM closure WHERE depth > 1")
check(f"物化闭包中存在 depth>1 的行（实测 {depth_gt1} 行，全部由触发器派生）", depth_gt1 > 0)

# 差分对拍：用**独立实现的在线递归 CTE** 当 oracle，而不是"拿闭包验闭包"
for rel in ("is_a", "has_part"):
    md = count(conn, "SELECT max_depth FROM closure_policy WHERE relation=?", (rel,))
    oracle = {(r[0], r[1]) for r in conn.execute("""
        WITH RECURSIVE walk(desc, anc, d) AS (
            SELECT r.src_id, r.dst_id, 1 FROM relations r
             WHERE r.relation=? AND r.src_kind='sense' AND r.dst_kind='sense'
            UNION ALL
            SELECT w.desc, r2.dst_id, w.d + 1 FROM walk w
              JOIN relations r2 ON r2.src_id = w.anc
             WHERE r2.relation=? AND r2.src_kind='sense' AND r2.dst_kind='sense'
               AND w.d < ?)
        SELECT DISTINCT anc, desc FROM walk""", (rel, rel, md))}
    mat = {(r[0], r[1]) for r in conn.execute(
        "SELECT ancestor, descendant FROM closure WHERE relation=?", (rel,))}
    check(f"差分对拍 {rel}：物化闭包 == 在线递归 CTE（各 {len(oracle)} 对）",
          mat == oracle,
          f"仅闭包有 {sorted(mat - oracle)[:5]}；仅 oracle 有 {sorted(oracle - mat)[:5]}")

# 反向完备性：每条 sense↔sense 边都必须有对应的 depth=1 闭包行（只对可传递关系）
no_depth1 = conn.execute("""
  SELECT r.relation, r.src_id, r.dst_id FROM relations r
    JOIN closure_policy cp ON cp.relation = r.relation AND cp.mode = 'incremental'
   WHERE r.src_kind='sense' AND r.dst_kind='sense'
     AND NOT EXISTS (SELECT 1 FROM closure c WHERE c.relation=r.relation
                      AND c.ancestor=r.dst_id AND c.descendant=r.src_id AND c.depth=1)""").fetchall()
check("反向完备性：每条可传递的 sense↔sense 边都有 depth=1 闭包行", not no_depth1, str(no_depth1))

null_fe = count(conn, "SELECT COUNT(*) FROM closure WHERE relation IN ('is_a','has_part') "
                      "AND first_edge IS NULL")
check("闭包 first_edge 见证边无空值（is_a / has_part）", null_fe == 0, f"空值 {null_fe} 行")

dupes = conn.execute("SELECT relation, ancestor, descendant, COUNT(*) n FROM closure "
                     "GROUP BY 1,2,3 HAVING n>1").fetchall()
check("闭包无重复行（PK 生效）", not dupes, str(dupes))

# =========================================================== §M7 跨库 cross_links
print("\n§M7 跨库 cross_links（通用库 → 医学库）")

cross = conn.execute("""
  SELECT r.relation, w.word, n.label, n.domain, r.source_kind, r.target_kind
    FROM cross_links r
    JOIN senses s ON s.sense_id = r.source_id AND r.source_kind = 'sense'
    JOIN words  w ON w.word_id  = s.word_id
    JOIN entity_nodes n ON n.node_id = r.target_id AND r.target_kind = 'node'
   WHERE n.domain = 'medical'
   ORDER BY r.relation""").fetchall()
got = {(r[0], r[1], r[2]) for r in cross}
expect = {("nutritional_relevance", "鱼", "营养素"),
          ("functional_analogy", "吃", "药物治疗"),
          ("domain_specialization", "信号", "脑电图信号"),
          ("domain_specialization", "信号", "强化特征")}   # §M13：影像子树新增
check(f"四条跨库链接齐备且类型正确（实测 {len(got)} 条）", got == expect,
      f"实得 {sorted(got)}；期望 {sorted(expect)}")
check("跨库链接端点形态均为 sense → node（因此能被 cross_links 视图读出）",
      all(r[4] == "sense" and r[5] == "node" for r in cross), str([(r[4], r[5]) for r in cross]))

med_nodes = conn.execute("SELECT node_id, label, entity_class FROM entity_nodes "
                         "WHERE domain='medical' ORDER BY node_id").fetchall()
check(f"医学库领域节点 = 4（实测 {len(med_nodes)}：{[r[1] for r in med_nodes]}）"
      f"—— 营养素 / 药物治疗 / 脑电图信号 / 强化特征",
      len(med_nodes) == 4, str(med_nodes))

bad_ep = conn.execute("""
  SELECT r.relation_id FROM relations r
   WHERE (r.src_kind='sense' AND NOT EXISTS (SELECT 1 FROM senses WHERE sense_id=r.src_id))
      OR (r.dst_kind='sense' AND NOT EXISTS (SELECT 1 FROM senses WHERE sense_id=r.dst_id))
      OR (r.src_kind='node'  AND NOT EXISTS (SELECT 1 FROM entity_nodes WHERE node_id=r.src_id))
      OR (r.dst_kind='node'  AND NOT EXISTS (SELECT 1 FROM entity_nodes WHERE node_id=r.dst_id))""").fetchall()
check("全部关系端点（含跨库边）指向存在的义项/节点", not bad_ep, str(bad_ep))

check("跨库边不参与闭包物化：closure 中无这 3 个领域关系的行",
      count(conn, "SELECT COUNT(*) FROM closure WHERE relation IN "
                  "('nutritional_relevance','functional_analogy','domain_specialization')") == 0)

# ======================================================= §M8 幂等与自足性
print("\n§M8 幂等与自足性")

TABLES = ["relation_types", "words", "senses", "sense_categories", "relations",
          "entity_nodes", "closure", "changelog"]
s1 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in TABLES}
try:
    load(conn, SEED_MED)
    s2 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in TABLES}
    drift = {t: (s1[t], s2[t]) for t in TABLES if s1[t] != s2[t]}
    check("幂等：第二次执行医学库种子后 8 张表行数零漂移", not drift, str(drift))
except sqlite3.Error as e:
    check("幂等：第二次执行医学库种子后 8 张表行数零漂移", False, f"{type(e).__name__}: {e}")

i_fk, i_begin = raw.find("PRAGMA foreign_keys"), raw.find("BEGIN;")
check("自足性：PRAGMA foreign_keys 写在 BEGIN 之前（事务内 PRAGMA 是 no-op）",
      0 <= i_fk < i_begin, f"fk@{i_fk} begin@{i_begin}")
check("自足性：PRAGMA recursive_triggers 已声明（闭包触发器依赖触发器级联）",
      raw.find("PRAGMA recursive_triggers") < i_begin)
n_b = len(re.findall(r"(?m)^\s*BEGIN\s*;", raw))
n_c = len(re.findall(r"(?m)^\s*COMMIT\s*;", raw))
check(f"事务配对：BEGIN({n_b}) == COMMIT({n_c})", n_b == n_c and n_b > 0,
      f"BEGIN={n_b} COMMIT={n_c}")
check("末尾已提交：加载后连接不处于未闭合事务中", conn.in_transaction is False)

# 前置依赖硬检查：裸 schema 上加载本文件必须**失败**，且报错可读
c_bare = connect()
load(c_bare, SCHEMA)
try:
    load(c_bare, SEED_MED)
    check("依赖闸门：裸 schema（无 seed/ext）上加载医学库被拒绝", False, "竟然加载成功")
except sqlite3.IntegrityError as e:
    msg = str(e)
    check("依赖闸门：裸 schema（无 seed/ext）上加载医学库被拒绝，且报错文本自带修复提示",
          "prereq" in msg, msg)
except sqlite3.Error as e:
    check("依赖闸门：裸 schema（无 seed/ext）上加载医学库被拒绝，且报错文本自带修复提示",
          False, f"{type(e).__name__}: {e}")
c_bare.close()

# ======================================================= §M9 推理链演示（非计数）
print("\n§M9 推理链演示（[ok] 非计数输出 —— 展示需求里的关系如何被查询/继承）")

epi = sid(conn, "癫痫")
chain = conn.execute("""
  SELECT w.word, c.depth FROM closure c
    JOIN senses s ON s.sense_id = c.ancestor JOIN words w ON w.word_id = s.word_id
   WHERE c.relation='is_a' AND c.descendant=? ORDER BY c.depth, w.word""", (epi,)).fetchall()
print("  [ok]   癫痫 的 is_a 上位链：" +
      " → ".join(f"{w}({d})" for w, d in chain))

# 症状：先看全部直接声明，再看"沿 is_a 继承"这条推理路径（推理规则 #2）
sym_direct = conn.execute("""
  SELECT ws.word, wd.word FROM relations r
    JOIN senses s1 ON s1.sense_id = r.src_id JOIN words ws ON ws.word_id = s1.word_id
    JOIN senses s2 ON s2.sense_id = r.dst_id JOIN words wd ON wd.word_id = s2.word_id
   WHERE r.relation='has_symptom' ORDER BY ws.word""").fetchall()
print("  [ok]   已声明的 has_symptom 边：" + "、".join(f"{d}→{s}" for d, s in sym_direct))

sym_inherit = conn.execute("""
  SELECT DISTINCT ws.word, wd.word FROM closure c
    JOIN relations r ON r.relation='has_symptom' AND r.src_kind='sense' AND r.src_id = c.ancestor
    JOIN senses s1 ON s1.sense_id = r.src_id JOIN words ws ON ws.word_id = s1.word_id
    JOIN senses s2 ON s2.sense_id = r.dst_id JOIN words wd ON wd.word_id = s2.word_id
   WHERE c.relation='is_a' AND c.descendant = ?""", (epi,)).fetchall()
if sym_inherit:
    print("  [ok]   癫痫 沿 is_a 继承到的症状：" + "、".join(f"{d}→{s}" for d, s in sym_inherit))
else:
    print("  [ok]   癫痫 沿 is_a 继承到的症状：（空）")
    print("             说明：未加载 seed_medical_v3_patch_symptom.sql，"
          "或 疾病 / 脑部疾病 层未声明通用症状。")
    print("             补法：在 疾病 层声明 疾病 has_symptom 疼痛/乏力，")
    print("                   在 脑部疾病 层声明 脑部疾病 has_symptom 头痛/眩晕/意识模糊，")
    print("                   全部下位疾病即会经 is_a 自动继承（推理规则 #2）。")

treat = conn.execute("""
  SELECT ws.word, wd.word FROM relations r
    JOIN senses s1 ON s1.sense_id=r.src_id JOIN words ws ON ws.word_id=s1.word_id
    JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
   WHERE r.relation='treated_by' AND r.src_id=? ORDER BY wd.word""", (epi,)).fetchall()
print("  [ok]   癫痫 的干预手段：" + "、".join(f"{d}" for _, d in treat))

_parts = sorted(((word_of(conn, k), v) for k, v in parts_of(conn, "脑").items()),
                key=lambda t: (t[1], t[0] or ""))
print("  [ok]   脑 的部分（含多级）：" + "、".join(f"{w}({d}跳)" for w, d in _parts))

print("  [ok]   记忆障碍 的诊断指向：" +
      "、".join(r[0] for r in conn.execute("""
        SELECT wd.word FROM relations r
          JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
         WHERE r.relation='indicates' AND r.src_id=?""", (sid(conn, "记忆障碍"),))))

print("  [ok]   跨库链接：" + "；".join(
    f"通用库{r[1]} --{r[0]}--> 医学库{r[2]}" for r in
    sorted(cross, key=lambda r: r[0])))

# ================================================= §M10 症状继承（补丁文件）
print("\n§M10 症状继承（seed_medical_v3_patch_symptom.sql）")

with open(str(SEED_MED_PATCH), encoding="utf-8") as f:
    praw = f.read()

# 需求列的 6 个通用症状；前 5 个要落到层上，发热只建义项（见补丁 §(2)）
GEN_SYMPTOMS = ["疼痛", "发热", "头痛", "眩晕", "乏力", "意识模糊"]
LAYERED = {"疼痛", "乏力", "头痛", "眩晕", "意识模糊"}          # 声明到层上的 5 种
LAYER_EDGES = {"疾病": ["疼痛", "乏力"],
               "脑部疾病": ["头痛", "眩晕", "意识模糊"]}

# ---- M10.1 义项层 ----
miss_sym = [w for w in GEN_SYMPTOMS if sid(conn, w) is None]
check(f"M10 六个通用症状义项全部新增（实测缺 {len(miss_sym)}）", not miss_sym, f"缺 {miss_sym}")

new_ids = {sid(conn, w) for w in GEN_SYMPTOMS}
check("M10 新症状 sense_id 恰为补丁声明段 4031-4036（与既有段位隔离）",
      new_ids == set(range(4031, 4037)), str(sorted(new_ids)))

bad_isa2 = [w for w in GEN_SYMPTOMS if not rel_has(w, "is_a", "症状")]
check("M10 六个新症状均 is_a 症状（挂在「症状」类别下）", not bad_isa2, str(bad_isa2))

bad_def2 = conn.execute(
    "SELECT w.word, s.definition FROM senses s JOIN words w ON w.word_id=s.word_id "
    "WHERE s.sense_id BETWEEN 4031 AND 4036 "
    "  AND (length(trim(s.definition))=0 OR s.definition GLOB '*[0-9a-zA-Z]*')").fetchall()
check("M10 六条新增症状释义非空且不含数字/拉丁字母（纯中文词语定义）", not bad_def2,
      str(bad_def2))

bad_cat2 = conn.execute("""
  SELECT sc.sense_id, sc.cat FROM sense_categories sc
   WHERE sc.sense_id BETWEEN 4031 AND 4036
     AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat)""").fetchall()
no_cat2 = [w for w in GEN_SYMPTOMS
           if not count(conn, "SELECT COUNT(*) FROM sense_categories WHERE sense_id=?",
                        (sid(conn, w),))]
check("M10 新增症状均已挂载合法类别（'性状'，未新增类别行）", not bad_cat2 and not no_cat2,
      f"非法 {bad_cat2} / 未挂 {no_cat2}")
check("M10 新增症状类别一律 '性状'（与病理状态同一粗类）",
      {r[0] for r in conn.execute(
          "SELECT DISTINCT cat FROM sense_categories WHERE sense_id BETWEEN 4031 AND 4036")}
      == {"性状"})

# ---- M10.2 层上声明 ----
layer_bad = [f"{layer}→{s}" for layer, syms in LAYER_EDGES.items() for s in syms
             if not rel_has(layer, "has_symptom", s)]
check("M10 层上症状边齐备：疾病→疼痛/乏力 + 脑部疾病→头痛/眩晕/意识模糊",
      not layer_bad, str(layer_bad))
check("M10 疾病 / 脑部疾病 层的 has_symptom 出边数 = 2 / 3",
      count(conn, "SELECT COUNT(*) FROM relations WHERE relation='has_symptom' "
                  "AND src_kind='sense' AND src_id=?", (sid(conn, "疾病"),)) == 2
      and count(conn, "SELECT COUNT(*) FROM relations WHERE relation='has_symptom' "
                      "AND src_kind='sense' AND src_id=?", (sid(conn, "脑部疾病"),)) == 3)


# ---- M10.3 继承（推理：D --is_a-->* X 且 X --has_symptom--> S ⇒ D 有症状 S）----
def sym_direct(disease):
    """直接声明的症状（不含继承）。"""
    return {r[0] for r in conn.execute(
        "SELECT wd.word FROM relations r "
        "  JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id "
        " WHERE r.relation='has_symptom' AND r.src_kind='sense' AND r.src_id=?",
        (sid(conn, disease),))}


def sym_inherit(disease):
    """沿 is_a 上位继承到的症状（只走 ancestor 一侧，不含疾病自身声明）。"""
    return {r[0] for r in conn.execute("""
        SELECT DISTINCT wd.word FROM closure cl
          JOIN relations r ON r.relation='has_symptom' AND r.src_kind='sense'
                          AND r.src_id = cl.ancestor
          JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
         WHERE cl.relation='is_a' AND cl.descendant = ?""", (sid(conn, disease),))}


def sym_total(disease):
    """症状结果集 = 直接声明 ∪ 沿 is_a 继承（去重）。"""
    return {r[0] for r in conn.execute("""
        SELECT DISTINCT wd.word FROM relations r
          JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
         WHERE r.relation='has_symptom' AND r.src_kind='sense'
           AND ( r.src_id = ?
                 OR r.src_id IN (SELECT ancestor FROM closure
                                  WHERE relation='is_a' AND descendant=?) )""",
        (sid(conn, disease), sid(conn, disease)))}


_epi_inh = sym_inherit("癫痫")
check("M10 疾病层继承：癫痫 沿 is_a 继承到「疼痛」「乏力」（经 疾病 层）",
      {"疼痛", "乏力"} <= _epi_inh, str(sorted(_epi_inh)))
check("M10 脑部疾病层继承：癫痫 沿 is_a 继承到「头痛」「眩晕」「意识模糊」（经 脑部疾病 层）",
      {"头痛", "眩晕", "意识模糊"} <= _epi_inh, str(sorted(_epi_inh)))

_ad_inh = sym_inherit("阿尔茨海默病")
check("M10 疾病层继承：阿尔茨海默病 沿 is_a 继承到「疼痛」「乏力」（经 疾病 层）",
      {"疼痛", "乏力"} <= _ad_inh, str(sorted(_ad_inh)))
check("M10 脑部疾病层继承：阿尔茨海默病 沿 is_a 继承到「头痛」「眩晕」「意识模糊」"
      "（经 脑部疾病 层，含 2 跳 神经退行性疾病 → 脑部疾病）",
      {"头痛", "眩晕", "意识模糊"} <= _ad_inh, str(sorted(_ad_inh)))

_stroke_inh = sym_inherit("脑卒中")
check("M10 脑卒中 沿 is_a 继承到全部 5 种通用症状",
      {"疼痛", "乏力", "头痛", "眩晕", "意识模糊"} <= _stroke_inh, str(sorted(_stroke_inh)))

check("M10 继承来源可分辨：癫痫 的「疼痛」来自 疾病 层、「头痛」来自 脑部疾病 层",
      rel_has("疾病", "has_symptom", "疼痛")
      and not rel_has("脑部疾病", "has_symptom", "疼痛")
      and rel_has("脑部疾病", "has_symptom", "头痛")
      and not rel_has("疾病", "has_symptom", "头痛"),
      "层归属错位")

# 结果集 = 直接 ∪ 继承（去重）
_merge_bad = []
for d in ("癫痫", "脑卒中", "神经退行性疾病", "阿尔茨海默病"):
    direct, inherited, total = sym_direct(d), sym_inherit(d), sym_total(d)
    if total != (direct | inherited):
        _merge_bad.append(f"{d}: total={sorted(total)} vs direct∪inh={sorted(direct | inherited)}")
check("M10 症状继承查询结果集 = 直接声明 ∪ 沿 is_a 继承（去重）", not _merge_bad,
      "; ".join(_merge_bad))

check("M10 癫痫 症状结果集 = 直接{抽搐, 记忆障碍, 发热} ∪ 5 种继承 = 8 种"
      "（后 2 条直接边由 patch_symptom_layers 补，见 §M12）",
      sym_total("癫痫") == {"抽搐", "记忆障碍", "发热",
                          "疼痛", "乏力", "头痛", "眩晕", "意识模糊"},
      str(sorted(sym_total("癫痫"))))
check("M10 脑卒中 症状结果集 = 5 种通用症状（无直接声明）",
      sym_total("脑卒中") == {"疼痛", "乏力", "头痛", "眩晕", "意识模糊"},
      str(sorted(sym_total("脑卒中"))))
check("M10 阿尔茨海默病 症状结果集 = {记忆障碍} ∪ 5 种继承 = 6 种",
      sym_total("阿尔茨海默病") == {"记忆障碍", "疼痛", "乏力", "头痛", "眩晕", "意识模糊"},
      str(sorted(sym_total("阿尔茨海默病"))))

# ---- M10.4 闭包自动派生 / 无手工 / 幂等 / 依赖闸门 / 自足性 ----
_clo_new = {w: count(conn, "SELECT COUNT(*) FROM closure WHERE relation='is_a' AND descendant=?",
                    (sid(conn, w),)) for w in GEN_SYMPTOMS}
check(f"M10 六个新症状的 is_a 闭包各 4 行（症状/病理状态/医学概念/抽象概念）（实测 {_clo_new}）",
      set(_clo_new.values()) == {4}, str(_clo_new))
check("M10 新症状经 is_a 可达抽象概念（depth=4，非悬空孤岛）",
      all(reachable(conn, "抽象概念", w) for w in GEN_SYMPTOMS))

check("M10 补丁零 closure 的 INSERT（实测 %d 条，全部由触发器派生）"
      % len(re.findall(r"(?m)^\s*INSERT\s+INTO\s+closure\b", praw, re.I)),
      len(re.findall(r"(?m)^\s*INSERT\s+INTO\s+closure\b", praw, re.I)) == 0)
check("M10 补丁未重定义任何关系类型（不 INSERT relation_types）",
      not re.search(r"(?m)^\s*INSERT\s+INTO\s+relation_types\b", praw, re.I))
check("M10 has_symptom 未产生闭包行（is_transitive=0，继承靠 is_a 传播）",
      count(conn, "SELECT COUNT(*) FROM closure WHERE relation='has_symptom'") == 0)

_T2 = ["words", "senses", "sense_categories", "relations", "closure", "changelog"]
_p1 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _T2}
try:
    load(conn, SEED_MED_PATCH)
    _p2 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _T2}
    _drift = {t: (_p1[t], _p2[t]) for t in _T2 if _p1[t] != _p2[t]}
    check("M10 幂等：第二次执行症状补丁后 6 张表行数零漂移", not _drift, str(_drift))
except sqlite3.Error as e:
    check("M10 幂等：第二次执行症状补丁后 6 张表行数零漂移", False,
          f"{type(e).__name__}: {e}")

_i_fk2, _i_beg2 = praw.find("PRAGMA foreign_keys"), praw.find("BEGIN;")
check("M10 补丁自足性：PRAGMA foreign_keys 写在 BEGIN 之前（事务内是 no-op）",
      0 <= _i_fk2 < _i_beg2, f"fk@{_i_fk2} begin@{_i_beg2}")
check("M10 补丁自足性：PRAGMA recursive_triggers 已声明",
      0 <= praw.find("PRAGMA recursive_triggers") < _i_beg2)
check("M10 补丁事务配对：BEGIN == COMMIT",
      len(re.findall(r"(?m)^\s*BEGIN\s*;", praw)) == len(re.findall(r"(?m)^\s*COMMIT\s*;", praw))
      and len(re.findall(r"(?m)^\s*BEGIN\s*;", praw)) > 0)

# 依赖闸门：裸 schema（无 medical）上加载补丁必须失败，且报错可读
c_bare2 = connect()
load(c_bare2, SCHEMA)
try:
    load(c_bare2, SEED_MED_PATCH)
    check("M10 依赖闸门：裸 schema（无 seed_medical_v3）上加载症状补丁被拒绝", False,
          "竟然加载成功")
except sqlite3.IntegrityError as e:
    check("M10 依赖闸门：裸 schema 上加载症状补丁被拒绝，且报错文本自带修复提示",
          "prereq" in str(e), str(e))
except sqlite3.Error as e:
    check("M10 依赖闸门：裸 schema 上加载症状补丁被拒绝，且报错文本自带修复提示",
          False, f"{type(e).__name__}: {e}")
c_bare2.close()

# 非计数 [ok]：把症状继承链打印出来（供人工核对）
print("  [ok]   通用症状继承链（层 → 症状）：" + "；".join(
    f"{layer} --has_symptom--> {'/'.join(syms)}" for layer, syms in LAYER_EDGES.items()))
for _d in ("癫痫", "脑卒中", "阿尔茨海默病"):
    print(f"  [ok]   {_d} 症状结果集（{len(sym_total(_d))} 种）："
          + "、".join(sorted(sym_total(_d))))

# ========================================== §M11 症状严重度义素（补丁文件）
print("\n§M11 症状严重度义素（seed_medical_v3_patch_symptom_severity.sql）")

with open(str(SEV_PATCH), encoding="utf-8") as f:
    sraw = f.read()

SEV_FEATURES = ["+severe", "+moderate", "+mild"]

# 需求给定的分级清单（注意：需求只列了 7 个症状，漏了第 8 个「意识模糊」；
# 补丁按其在 patch_symptom §2.2 的语义限定「重症 / 脑部疾病共有」归入 +severe）
SEV_REQUIRED = {"+severe": {"疼痛", "头痛", "抽搐"},
                "+moderate": {"眩晕", "乏力", "记忆障碍"},
                "+mild": {"发热"}}
SEV_FULL = {"+severe": {"疼痛", "头痛", "抽搐", "意识模糊"},
            "+moderate": {"眩晕", "乏力", "记忆障碍"},
            "+mild": {"发热"}}
SYM8 = ["疼痛", "发热", "头痛", "眩晕", "乏力", "意识模糊", "抽搐", "记忆障碍"]

# 「症状」子树（根 + 全部 is_a 下位）—— 严重度义素的合法落点
_sym_root = sid(conn, "症状")
_sym_tree = {_sym_root} | {r[0] for r in conn.execute(
    "SELECT descendant FROM closure WHERE relation='is_a' AND ancestor=?", (_sym_root,))}
_sym_leaves = _sym_tree - {_sym_root}


def sev_of(word):
    """某症状义项上挂的严重度义素（集合；空集 = 未标注）。"""
    return {r[0] for r in conn.execute(
        "SELECT feature FROM sense_features WHERE sense_id=? "
        "  AND feature IN ('+severe','+moderate','+mild')", (sid(conn, word),))}


def sev_bucket(feature):
    """全库标了该严重度的症状词形集合（义素清单口径）。"""
    return {r[0] for r in conn.execute(
        "SELECT w.word FROM sense_features sf "
        "  JOIN senses s ON s.sense_id = sf.sense_id JOIN words w ON w.word_id = s.word_id "
        " WHERE sf.feature = ?", (feature,))}


def sev_retrieval(disease):
    """需求要求的查询形态：
        closure(relation='is_a') ⋈ relations('has_symptom') ⟕ sense_features

    返回 {(症状词形, 严重度 or None)}；只走**继承**（上位层声明的边）。
    LEFT JOIN 保证「未标严重度的症状」也不丢行（本任务下 8 个症状全已标注，
    故无 NULL 行 —— 但查询形态本身对未标注数据是安全的）。
    """
    return {(r[0], r[1]) for r in conn.execute("""
        SELECT DISTINCT wd.word, sf.feature
          FROM closure cl
          JOIN relations r ON r.relation = 'has_symptom' AND r.src_kind = 'sense'
                          AND r.src_id = cl.ancestor
          JOIN senses s2 ON s2.sense_id = r.dst_id
          JOIN words  wd ON wd.word_id  = s2.word_id
          LEFT JOIN sense_features sf ON sf.sense_id = r.dst_id
                                     AND sf.feature IN ('+severe','+moderate','+mild')
         WHERE cl.relation = 'is_a' AND cl.descendant = ?""", (sid(conn, disease),))}


def sev_total(disease):
    """「疾病 D 的症状 + 严重度」= 直接声明 ∪ 沿 is_a 继承（与 §M10 sym_total 同口径）。"""
    return {(r[0], r[1]) for r in conn.execute("""
        SELECT DISTINCT wd.word, sf.feature
          FROM relations r
          JOIN senses s2 ON s2.sense_id = r.dst_id
          JOIN words  wd ON wd.word_id  = s2.word_id
          LEFT JOIN sense_features sf ON sf.sense_id = r.dst_id
                                     AND sf.feature IN ('+severe','+moderate','+mild')
         WHERE r.relation = 'has_symptom' AND r.src_kind = 'sense'
           AND ( r.src_id = ?
                 OR r.src_id IN (SELECT ancestor FROM closure
                                  WHERE relation = 'is_a' AND descendant = ?) )""",
        (sid(conn, disease), sid(conn, disease)))}


def bucket_of(pairs, feat):
    return {w for w, f in pairs if f == feat}


# ---- M11.1 覆盖：8 个症状全部标注 ----
check(f"M11 症状子树共 8 个义项（除根「症状」，实测 {len(_sym_leaves)}）",
      len(_sym_leaves) == 8, str(words_of(conn, _sym_leaves)))
_miss_sev = [w for w in SYM8 if not sev_of(w)]
check(f"M11 ★需求项：八个症状义项均有严重度义素标注（实测缺 {len(_miss_sev)}）",
      not _miss_sev, f"缺 {_miss_sev}")
check("M11 症状子树与需求列的 8 词一致（无遗漏、无多计）",
      {word_of(conn, i) for i in _sym_leaves} == set(SYM8),
      str(sorted({word_of(conn, i) for i in _sym_leaves})))

# ---- M11.2 取值域 ----
_registered = {r[0] for r in conn.execute(
    "SELECT feature FROM semantic_features WHERE feature IN ('+severe','+moderate','+mild')")}
check("M11 三个严重度义素已在 semantic_features 登记（+severe / +moderate / +mild）",
      _registered == set(SEV_FEATURES), str(sorted(_registered)))
_used_feat = {r[0] for r in conn.execute(
    "SELECT DISTINCT feature FROM sense_features WHERE feature IN ('+severe','+moderate','+mild')")}
check("M11 ★需求项：严重度取值仅限 '+severe' / '+moderate' / '+mild' 三种",
      bool(_used_feat) and _used_feat <= set(SEV_FEATURES), str(sorted(_used_feat)))
_rogue = conn.execute(
    "SELECT DISTINCT sf.feature FROM sense_features sf WHERE sf.sense_id IN (%s) "
    "  AND sf.feature NOT IN (SELECT feature FROM semantic_features)"
    % ",".join("?" * len(_sym_tree)), tuple(_sym_tree)).fetchall()
check("M11 症状义项上的义素全部已在 semantic_features 登记（无野义素 / FK 未被绕过）",
      not _rogue, str(_rogue))
_sev_cats = {r[0] for r in conn.execute(
    "SELECT DISTINCT sc.cat FROM sense_categories sc WHERE sc.sense_id IN (%s)"
    % ",".join("?" * len(_sym_tree)), tuple(_sym_tree))}
check("M11 被标注症状的类别一致（一律 '性状'；类别粒度粗，故成员资格以 is_a 子树为准）",
      _sev_cats == {"性状"}, str(sorted(_sev_cats)))

# ---- M11.3 每个症状恰 1 个严重度义素 ----
_multi = {w: sorted(sev_of(w)) for w in SYM8 if len(sev_of(w)) != 1}
check("M11 ★需求项：每个症状恰有 1 个严重度义素（不漏标、不重复标注）",
      not _multi, str(_multi))
_badval = conn.execute(
    "SELECT sense_id, feature, value FROM sense_features "
    " WHERE feature IN ('+severe','+moderate','+mild') AND value <> '1'").fetchall()
check("M11 严重度义素 value 一律 '1'（boolean 义素口径，与 schema §14 一致）",
      not _badval, str(_badval))

# ---- M11.4 分级正确性（义素清单口径）----
for _feat in SEV_FEATURES:
    _got = sev_bucket(_feat)
    check(f"M11 分级正确：{_feat} 桶 = {sorted(SEV_FULL[_feat])}（实测 {sorted(_got)}）",
          _got == SEV_FULL[_feat], f"实测 {sorted(_got)}")
    check(f"M11 ★需求项：{_feat} 包含「{'/'.join(sorted(SEV_REQUIRED[_feat]))}」",
          SEV_REQUIRED[_feat] <= _got, f"实测 {sorted(_got)}")

# ---- M11.5 继承检索（需求指定的 SQL 形态）----
_epi_sev = sev_total("癫痫")
check("M11 ★需求项：继承检索 SQL（closure ⋈ is_a ⋈ has_symptom ⟕ sense_features）"
      "对 癫痫 返回 5 种继承症状且严重度齐备",
      sev_retrieval("癫痫") == {("乏力", "+moderate"), ("眩晕", "+moderate"),
                                ("头痛", "+severe"), ("意识模糊", "+severe"),
                                ("疼痛", "+severe")},
      str(sorted(sev_retrieval("癫痫"))))
check("M11 ★需求项：癫痫 的 '+severe' 症状（直接 ∪ 继承）包含「疼痛 / 头痛 / 抽搐」",
      SEV_REQUIRED["+severe"] <= bucket_of(_epi_sev, "+severe"),
      f"实测 severe={sorted(bucket_of(_epi_sev, '+severe'))}")
check("M11 ★需求项：癫痫 的 '+moderate' 症状包含「眩晕 / 乏力 / 记忆障碍」"
      "（记忆障碍 由 patch_symptom_layers 给癫痫补的**直接边**落入，见 §M12）",
      SEV_REQUIRED["+moderate"] <= bucket_of(_epi_sev, "+moderate"),
      f"实测 moderate={sorted(bucket_of(_epi_sev, '+moderate'))}")
check("M11 ★需求项：癫痫 的 '+mild' 症状包含「发热」"
      "（发热 由 patch_symptom_layers 给癫痫补的**直接边**落入，见 §M12）",
      SEV_REQUIRED["+mild"] <= bucket_of(_epi_sev, "+mild"),
      f"实测 mild={sorted(bucket_of(_epi_sev, '+mild'))}")
check("M11 癫痫 的 '+moderate' 症状恰为 {乏力, 眩晕, 记忆障碍}（无多计）",
      bucket_of(_epi_sev, "+moderate") == {"乏力", "眩晕", "记忆障碍"},
      f"实测 {sorted(bucket_of(_epi_sev, '+moderate'))}")
_ad_sev = sev_total("阿尔茨海默病")
check("M11 ★需求项：'+moderate' 的症状集（阿尔茨海默病，含直接声明的 记忆障碍）"
      "包含「眩晕 / 乏力 / 记忆障碍」",
      {"眩晕", "乏力", "记忆障碍"} <= bucket_of(_ad_sev, "+moderate"),
      f"实测 moderate={sorted(bucket_of(_ad_sev, '+moderate'))}")
check("M11 阿尔茨海默病 的 '+severe' 症状 = {疼痛, 头痛, 意识模糊}（跨 2 跳继承）",
      bucket_of(_ad_sev, "+severe") == {"疼痛", "头痛", "意识模糊"},
      f"实测 {sorted(bucket_of(_ad_sev, '+severe'))}")
_mild_reach = {d for d in ("癫痫", "脑卒中", "神经退行性疾病", "阿尔茨海默病",
                           "炎症性疾病", "感染性疾病", "脑炎")
               if bucket_of(sev_total(d), "+mild")}
check("M11 ★需求项：'+mild' 症状集（全库义素清单口径）包含「发热」",
      SEV_REQUIRED["+mild"] <= sev_bucket("+mild"),
      f"清单={sorted(sev_bucket('+mild'))}")
check("M11 '+mild' 已**可被检索**（补丁三补出「炎症性疾病」层 + 癫痫直接边）："
      "炎症性疾病（层边源）→ 感染性疾病 / 脑炎（继承）→ 癫痫（直接边）",
      _mild_reach == {"炎症性疾病", "感染性疾病", "脑炎", "癫痫"},
      f"实测可检索到发热的疾病={sorted(_mild_reach)}")
_all_lj = conn.execute(
    "SELECT s.sense_id, w.word, sf.feature FROM senses s JOIN words w ON w.word_id = s.word_id "
    " LEFT JOIN sense_features sf ON sf.sense_id = s.sense_id "
    "                            AND sf.feature IN ('+severe','+moderate','+mild') "
    " WHERE s.sense_id IN (%s)" % ",".join("?" * len(_sym_leaves)), tuple(_sym_leaves)).fetchall()
check("M11 检索无扇出 / 不丢行：8 个症状（不含根「症状」）在 LEFT JOIN 下恰各 1 行",
      len(_all_lj) == 8 and len({r[0] for r in _all_lj}) == 8
      and all(r[2] for r in _all_lj),
      f"实测 {len(_all_lj)} 行 / {len({r[0] for r in _all_lj})} 个义项")

# ---- M11.6 全库不变量：严重度义素只落在「症状」子树 ----
_off_tree = conn.execute("""
  SELECT sf.sense_id, w.word FROM sense_features sf
    JOIN senses s ON s.sense_id = sf.sense_id JOIN words w ON w.word_id = s.word_id
   WHERE sf.feature IN ('+severe','+moderate','+mild')
     AND sf.sense_id NOT IN (%s)""" % ",".join("?" * len(_sym_tree)),
    tuple(_sym_tree)).fetchall()
check("M11 ★需求项：全库不变量 —— 严重度义素仅指向「症状」义项"
      "（经 closure is_a 子树判定；sense_categories 侧已在上方交叉校验）",
      not _off_tree, str(_off_tree))

# ---- M11.7 纯追加：只写义素/来源表，不碰数据与结构 ----
# 剔除临时表（_prereq_check / _sev_count_check 是 §0 的具名 CHECK 闸门）
_targets = {t for t in re.findall(r"(?mi)^\s*INSERT\s+INTO\s+([A-Za-z_]+)", sraw)
            if not t.startswith("_")}
check(f"M11 补丁只向 {sorted(_targets)} 追加（不写 words/senses/relations/closure/sense_categories）",
      _targets == {"sources", "semantic_features", "sense_features"}, str(sorted(_targets)))
check("M11 补丁零 closure 的 INSERT",
      not re.search(r"(?m)^\s*INSERT\s+INTO\s+closure\b", sraw, re.I))
check("M11 补丁不改 schema：无 ALTER TABLE / 无非 TEMP 的 CREATE TABLE",
      not re.search(r"(?mi)^\s*ALTER\s+TABLE\b", sraw)
      and not re.search(r"(?mi)^\s*CREATE\s+TABLE\s+(?!IF\s+NOT\s+EXISTS\s+_[a-z_]*check)", sraw),
      "出现 ALTER / 非临时建表")
check("M11 补丁未重定义关系类型 / 闭包策略（不 INSERT relation_types / closure_policy）",
      not re.search(r"(?mi)^\s*INSERT\s+INTO\s+(relation_types|closure_policy)\b", sraw))

# ---- M11.8 幂等 / 自足性 / 依赖闸门 ----
_S_TABLES = ["sense_features", "semantic_features", "sources"]
_s1 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _S_TABLES}
try:
    load(conn, SEV_PATCH)
    _s2 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _S_TABLES}
    _sdrift = {t: (_s1[t], _s2[t]) for t in _S_TABLES if _s1[t] != _s2[t]}
    check("M11 幂等：第二次执行严重度补丁后 3 张表行数零漂移", not _sdrift, str(_sdrift))
except sqlite3.Error as e:
    check("M11 幂等：第二次执行严重度补丁后 3 张表行数零漂移", False,
          f"{type(e).__name__}: {e}")

_i_fk3, _i_beg3 = sraw.find("PRAGMA foreign_keys"), sraw.find("BEGIN;")
check("M11 补丁自足性：PRAGMA foreign_keys 写在 BEGIN 之前（事务内是 no-op）",
      0 <= _i_fk3 < _i_beg3, f"fk@{_i_fk3} begin@{_i_beg3}")
check("M11 补丁自足性：PRAGMA recursive_triggers 已声明",
      0 <= sraw.find("PRAGMA recursive_triggers") < _i_beg3)
check("M11 补丁事务配对：BEGIN == COMMIT",
      len(re.findall(r"(?m)^\s*BEGIN\s*;", sraw)) == len(re.findall(r"(?m)^\s*COMMIT\s*;", sraw))
      and len(re.findall(r"(?m)^\s*BEGIN\s*;", sraw)) > 0)

# 闸门 (a)：裸 schema 上加载 → 被拒绝
c_bare3 = connect()
load(c_bare3, SCHEMA)
try:
    load(c_bare3, SEV_PATCH)
    check("M11 依赖闸门：裸 schema 上加载严重度补丁被拒绝", False, "竟然加载成功")
except sqlite3.IntegrityError as e:
    check("M11 依赖闸门：裸 schema 上加载严重度补丁被拒绝，且报错文本自带修复提示",
          "prereq" in str(e), str(e))
except sqlite3.Error as e:
    check("M11 依赖闸门：裸 schema 上加载严重度补丁被拒绝，且报错文本自带修复提示",
          False, f"{type(e).__name__}: {e}")
c_bare3.close()

# 闸门 (b)：有 medical 但**没有** patch_symptom（4031-4036 未建）→ 同样被拒绝
c_nosym = connect()
for _p in (SCHEMA, SEED, SEED_EXT, SEED_MED):
    load(c_nosym, _p)
try:
    load(c_nosym, SEV_PATCH)
    check("M11 依赖闸门：缺少 patch_symptom（4031-4036 症状义项未建）时被拒绝", False,
          "竟然加载成功 —— 说明内连接静默少行，未被 §0 拦下")
except sqlite3.IntegrityError as e:
    check("M11 依赖闸门：缺少 patch_symptom（4031-4036 症状义项未建）时被拒绝",
          "prereq" in str(e), str(e))
except sqlite3.Error as e:
    check("M11 依赖闸门：缺少 patch_symptom 时被拒绝", False, f"{type(e).__name__}: {e}")
c_nosym.close()

# 加载顺序不变量：文件名必须让 init_database 的 sorted(glob) 把本文件排在
# patch_symptom 之后（否则自动发现管线会静默丢 8 行）
_seed_names = sorted(p.name for p in HERE.glob("seed_*.sql"))
check("M11 加载顺序不变量：本补丁在字典序中排在 patch_symptom 之后"
      "（'severity' 的 'e' < 'symptom' 的 'y'，直接叫 patch_severity 会被错误地提前加载）",
      _seed_names.index(SEV_PATCH.name) > _seed_names.index(SEED_MED_PATCH.name),
      str(_seed_names))

# ---- 非计数 [ok]：把严重度标注与检索结果如实打印（供人工核对）----
print("  [ok]   严重度义素清单（全库）：" + "；".join(
    f"{feat} = {'/'.join(sorted(sev_bucket(feat)))}" for feat in SEV_FEATURES))
for _d in ("癫痫", "脑卒中", "神经退行性疾病", "阿尔茨海默病"):
    _pairs = sorted(sev_total(_d), key=lambda t: (t[1] or "~", t[0]))
    print(f"  [ok]   {_d} 的症状（严重度）："
          + "、".join(f"{w}[{(f or '未标注').lstrip('+')}]" for w, f in _pairs))
print("  [ok]   已知妥协：严重度义素的 feature_group 复用「极性」——"
      "schema 的 feature_group 是 CHECK 白名单（生命度/具体性/可数性/自主性/有界性/极性），"
      "无「严重度」分组；新增分组须 ALTER TABLE，与「不改 schema」约束冲突")
print("  [ok]   已知限制：sense_features 表**没有** source_id 列（列仅 sense_id/feature/value），"
      "故 source_id=6 登记在 sources 表 + semantic_features.description 中，未落进义素行")
print("  [ok]   需求项 2 处口径**已按字面补齐**（由 patch_symptom_layers 补丁三完成，见 §M12）：")
print("           · 「癫痫 继承到 记忆障碍」—— 癫痫 补了**直接** has_symptom→记忆障碍 边")
print("             （颞叶病灶累及海马）。按项目统一口径「结果集 = 直接声明 ∪ 沿 is_a 继承」，")
print("             癫痫 的结果集自然含 记忆障碍。注意它**不是层边**——记忆障碍 并非所有")
print("             疾病 / 脑部疾病共有，只能在 癫痫 与 阿尔茨海默病 上各自直接声明。")
print("           · 「癫痫 继承到 发热」—— 补出「炎症性疾病」层并把 发热 挂上去后，")
print("             发热 首次**由继承可达**（感染性疾病 / 脑炎）；癫痫 另补一条")
print("             **低置信度(0.5)直接边**表达「发热诱发发作」的弱关联（见 §M12 取舍 (3)）。")
print("  [ok]   补位说明：需求的分级清单只列了 7 个症状，漏了「意识模糊」；"
      "严重度补丁按其余量（重症 / 脑部疾病共有）归入 +severe，凑满 8 个症状各 1 行")

# ========================================== §M12 症状分层补丁（补丁三，追加）
print("\n§M12 症状分层（seed_medical_v3_patch_symptom_layers.sql）")

with open(str(LAY_PATCH), encoding="utf-8") as f:
    lraw = f.read()

LAY_NEW = ["炎症性疾病", "感染性疾病", "脑炎"]
LAY_ISA = {("炎症性疾病", "疾病"), ("感染性疾病", "炎症性疾病"),
           ("脑炎", "炎症性疾病"), ("脑炎", "脑部疾病")}
# 层边（下沉到「炎症性疾病」层）与直接边（只在特定疾病上成立）混合
LAY_SYM = {("炎症性疾病", "发热"): 0.90,     # 层边：炎症 / 感染性疾病共有
           ("脑炎", "抽搐"): 0.75,           # 直接边：脑炎可致惊厥
           ("癫痫", "记忆障碍"): 0.85,       # 直接边：颞叶病灶累及海马
           ("癫痫", "发热"): 0.50}           # ★ 弱关联（语义妥协，见文件头取舍 (3)）

# ---- M12.1 义项层（4 项）----
miss_lay = [w for w in LAY_NEW if sid(conn, w) is None]
check(f"M12 三个分层义项全部新增（实测缺 {len(miss_lay)}）", not miss_lay, f"缺 {miss_lay}")
lay_ids = {sid(conn, w) for w in LAY_NEW}
check("M12 新义项 sense_id 恰为补丁声明段 4037-4039（与 4031-4036 症状段 / 4101 兜底段隔离）",
      lay_ids == set(range(4037, 4040)), str(sorted(lay_ids)))
bad_def3 = conn.execute(
    "SELECT w.word, s.definition FROM senses s JOIN words w ON w.word_id=s.word_id "
    "WHERE s.sense_id BETWEEN 4037 AND 4039 "
    "  AND (length(trim(s.definition))=0 OR s.definition GLOB '*[0-9a-zA-Z]*')").fetchall()
check("M12 三条新增释义非空且不含数字/拉丁字母（纯中文词语定义，无向量化痕迹）",
      not bad_def3, str(bad_def3))
bad_cat3 = conn.execute("""
  SELECT sc.sense_id, sc.cat FROM sense_categories sc
   WHERE sc.sense_id BETWEEN 4037 AND 4039
     AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat)""").fetchall()
no_cat3 = [w for w in LAY_NEW if not count(
    conn, "SELECT COUNT(*) FROM sense_categories WHERE sense_id=?", (sid(conn, w),))]
check("M12 新增义项均已挂载合法类别（'性状'，未新增类别行）",
      not bad_cat3 and not no_cat3, f"非法 {bad_cat3} / 未挂 {no_cat3}")

# ---- M12.2 is_a 分层与闭包（5 项）----
bad_isa3 = [f"{c}→{p}" for c, p in sorted(LAY_ISA) if not rel_has(c, "is_a", p)]
check("M12 四条 is_a 边齐备：炎症性疾病→疾病 / 感染性疾病→炎症性疾病 / "
      "脑炎→炎症性疾病 / 脑炎→脑部疾病", not bad_isa3, str(bad_isa3))
check("M12 ★多继承：脑炎 的 is_a 出边 = 2（炎症性疾病 与 脑部疾病）",
      count(conn, "SELECT COUNT(*) FROM relations WHERE relation='is_a' AND src_kind='sense' "
                  "AND src_id=?", (sid(conn, "脑炎"),)) == 2)
check("M12 层级：炎症性疾病 ⊑ 疾病 ⊑ 病理状态；感染性疾病 ⊑ 炎症性疾病（更窄的层挂在更宽层下）",
      reachable(conn, "疾病", "炎症性疾病") and reachable(conn, "病理状态", "炎症性疾病")
      and reachable(conn, "炎症性疾病", "感染性疾病"))
check("M12 三个新分层节点均锚定到 医学概念（非悬空孤岛）",
      all(reachable(conn, "医学概念", w) for w in LAY_NEW))
_clo_lay = {w: count(conn, "SELECT COUNT(*) FROM closure WHERE relation='is_a' AND descendant=?",
                     (sid(conn, w),)) for w in LAY_NEW}
check(f"M12 is_a 闭包行数 = 4 / 5 / 6（实测 {_clo_lay}）：炎症性疾病 4，"
      "感染性疾病 多 1 跳，脑炎 两支并集 6 —— 全部自动派生",
      _clo_lay == {"炎症性疾病": 4, "感染性疾病": 5, "脑炎": 6}, str(_clo_lay))

# ---- M12.3 has_symptom 边：层边 vs 直接边（4 项）----
bad_sym3 = [f"{s}→{d}" for (s, d) in LAY_SYM if not rel_has(s, "has_symptom", d)]
check("M12 四条 has_symptom 边齐备", not bad_sym3, str(bad_sym3))
_conf3 = {(s, d): count(
    conn, "SELECT COUNT(*) FROM relations WHERE relation='has_symptom' AND src_kind='sense' "
          "AND dst_kind='sense' AND src_id=? AND dst_id=? AND confidence=?",
    (sid(conn, s), sid(conn, d), c)) for (s, d), c in LAY_SYM.items()}
check(f"M12 四条边的 confidence 与设计一致（实测 {_conf3}）",
      all(v == 1 for v in _conf3.values()), str(_conf3))
check("M12 ★癫痫→发热 是**低置信度弱关联**（confidence=0.5 = 全库 has_symptom 最低），"
      "与层边 0.9 / 直接边 0.75~0.85 在数值上可区分（语义妥协，见文档 §12.3）",
      count(conn, "SELECT MIN(confidence) FROM relations WHERE relation='has_symptom'") == 0.5)
check("M12 ★癫痫→发热 不是层边：发热 的声明源恰为 炎症性疾病（层边 0.9）与 癫痫（弱边 0.5）",
      {r[0] for r in conn.execute("""
        SELECT ws.word FROM relations r
          JOIN senses s1 ON s1.sense_id=r.src_id JOIN words ws ON ws.word_id=s1.word_id
         WHERE r.relation='has_symptom' AND r.src_kind='sense' AND r.dst_id=?""",
          (sid(conn, "发热"),))} == {"炎症性疾病", "癫痫"})

# ---- M12.4 继承结果集：层边 + 直接边混合（5 项）----
_epi_total = sym_total("癫痫")
check("M12 ★癫痫 的症状结果集 = 8 种（层边继承 5 + 直接边 3：抽搐/记忆障碍/发热）",
      _epi_total == {"抽搐", "记忆障碍", "发热",
                     "疼痛", "乏力", "头痛", "眩晕", "意识模糊"},
      str(sorted(_epi_total)))
_ence_total = sym_total("脑炎")
check("M12 ★脑炎 的症状结果集 = 7 种（炎症层 1 + 疾病层 2 + 脑部疾病层 3 + 直接 1）",
      _ence_total == {"发热", "疼痛", "乏力", "头痛", "眩晕", "意识模糊", "抽搐"},
      str(sorted(_ence_total)))
_inf_total = sym_total("感染性疾病")
check("M12 ★发热 首次**由继承可达**：感染性疾病 经 炎症性疾病 层取到 发热/疼痛/乏力",
      _inf_total == {"发热", "疼痛", "乏力"}, str(sorted(_inf_total)))
check("M12 ★癫痫→记忆障碍 不是层边：记忆障碍 只在 癫痫 与 阿尔茨海默病 上各自直接声明",
      {r[0] for r in conn.execute("""
        SELECT ws.word FROM relations r
          JOIN senses s1 ON s1.sense_id=r.src_id JOIN words ws ON ws.word_id=s1.word_id
         WHERE r.relation='has_symptom' AND r.src_kind='sense' AND r.dst_id=?""",
          (sid(conn, "记忆障碍"),))} == {"癫痫", "阿尔茨海默病"})
check("M12 回归：脑卒中 / 神经退行性疾病 的结果集仍为 5 种"
      "（二者不在 炎症性疾病 子树内，不受本补丁影响）",
      sym_total("脑卒中") == {"疼痛", "乏力", "头痛", "眩晕", "意识模糊"}
      and sym_total("神经退行性疾病") == {"疼痛", "乏力", "头痛", "眩晕", "意识模糊"},
      f"脑卒中={sorted(sym_total('脑卒中'))}")

# ---- M12.5 与 §M11 严重度检索的衔接（2 项）：两条断言由"收窄"回到"字面" ----
_epi_sev12 = sev_total("癫痫")
check("M12 ★衔接：癫痫 的 '+moderate' 桶 = {乏力, 眩晕, 记忆障碍}"
      "（需求清单**按字面**成立，不再需要收窄到 阿尔茨海默病）",
      bucket_of(_epi_sev12, "+moderate") == {"乏力", "眩晕", "记忆障碍"},
      str(sorted(bucket_of(_epi_sev12, "+moderate"))))
check("M12 ★衔接：癫痫 的 '+mild' 桶 = {发热}（需求清单**按字面**成立）",
      bucket_of(_epi_sev12, "+mild") == {"发热"},
      str(sorted(bucket_of(_epi_sev12, "+mild"))))

# ---- M12.6 纯追加：只写数据表，不碰结构定义（4 项）----
_l_targets = {t for t in re.findall(r"(?mi)^\s*INSERT\s+INTO\s+([A-Za-z_]+)", lraw)
              if not t.startswith("_")}
check(f"M12 补丁只向 {sorted(_l_targets)} 追加（不写 relation_types / closure_policy / "
      f"sense_features / entity_nodes）",
      _l_targets == {"words", "senses", "sense_categories", "relations"},
      str(sorted(_l_targets)))
check("M12 补丁零 closure 的 INSERT（全部由触发器派生）",
      not re.search(r"(?m)^\s*INSERT\s+INTO\s+closure\b", lraw, re.I))
check("M12 补丁不改 schema：无 ALTER TABLE / 无非 TEMP 的 CREATE TABLE",
      not re.search(r"(?mi)^\s*ALTER\s+TABLE\b", lraw)
      and not re.search(r"(?mi)^\s*CREATE\s+TABLE\s+(?!IF\s+NOT\s+EXISTS\s+_[a-z_]*)", lraw),
      "出现 ALTER / 非临时建表")
check("M12 补丁未重定义关系类型 / 闭包策略（不 INSERT relation_types / closure_policy）",
      not re.search(r"(?mi)^\s*INSERT\s+INTO\s+(relation_types|closure_policy)\b", lraw))
check("M12 补丁不写义素（与严重度补丁职责分离：一个只增数据，一个只增标注）",
      not re.search(r"(?mi)^\s*INSERT\s+INTO\s+(sense_features|semantic_features|sources)\b",
                    lraw))

# ---- M12.7 幂等 / 自足性 / 依赖闸门 / 加载顺序（7 项）----
_L_TABLES = ["words", "senses", "sense_categories", "relations", "closure", "changelog"]
_l1 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _L_TABLES}
try:
    load(conn, LAY_PATCH)
    _l2 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _L_TABLES}
    _ldrift = {t: (_l1[t], _l2[t]) for t in _L_TABLES if _l1[t] != _l2[t]}
    check("M12 幂等：第二次执行分层补丁后 6 张表行数零漂移", not _ldrift, str(_ldrift))
except sqlite3.Error as e:
    check("M12 幂等：第二次执行分层补丁后 6 张表行数零漂移", False, f"{type(e).__name__}: {e}")

_i_fk4, _i_beg4 = lraw.find("PRAGMA foreign_keys"), lraw.find("BEGIN;")
check("M12 补丁自足性：PRAGMA foreign_keys 写在 BEGIN 之前（事务内是 no-op）",
      0 <= _i_fk4 < _i_beg4, f"fk@{_i_fk4} begin@{_i_beg4}")
check("M12 补丁自足性：PRAGMA recursive_triggers 已声明",
      0 <= lraw.find("PRAGMA recursive_triggers") < _i_beg4)
check("M12 补丁事务配对：BEGIN == COMMIT",
      len(re.findall(r"(?m)^\s*BEGIN\s*;", lraw)) == len(re.findall(r"(?m)^\s*COMMIT\s*;", lraw))
      and len(re.findall(r"(?m)^\s*BEGIN\s*;", lraw)) > 0)

# 闸门 (a)：裸 schema 上加载 → 被拒绝
c_bare4 = connect()
load(c_bare4, SCHEMA)
try:
    load(c_bare4, LAY_PATCH)
    check("M12 依赖闸门：裸 schema 上加载分层补丁被拒绝", False, "竟然加载成功")
except sqlite3.IntegrityError as e:
    check("M12 依赖闸门：裸 schema 上加载分层补丁被拒绝，且报错文本自带修复提示",
          "prereq" in str(e), str(e))
except sqlite3.Error as e:
    check("M12 依赖闸门：裸 schema 上加载分层补丁被拒绝", False, f"{type(e).__name__}: {e}")
c_bare4.close()

# 闸门 (b)：有 medical 但**没有** patch_symptom（发热义项未建）→ 同样被拒绝
c_nosym2 = connect()
for _p in (SCHEMA, SEED, SEED_EXT, SEED_MED):
    load(c_nosym2, _p)
try:
    load(c_nosym2, LAY_PATCH)
    check("M12 依赖闸门：缺少 patch_symptom（发热义项未建）时被拒绝", False,
          "竟然加载成功 —— 说明内连接静默少行，未被 §0 拦下")
except sqlite3.IntegrityError as e:
    check("M12 依赖闸门：缺少 patch_symptom（发热义项未建）时被拒绝", "prereq" in str(e), str(e))
except sqlite3.Error as e:
    check("M12 依赖闸门：缺少 patch_symptom 时被拒绝", False, f"{type(e).__name__}: {e}")
c_nosym2.close()

# 加载顺序不变量：本补丁字典序必须落在 patch_symptom 之后
check("M12 加载顺序不变量：本补丁在字典序中排在 patch_symptom 之后"
      "（公共前缀后 '.' U+002E < '_' U+005F）",
      LAY_PATCH.name in _seed_names
      and _seed_names.index(LAY_PATCH.name) > _seed_names.index(SEED_MED_PATCH.name),
      str(_seed_names))

# ---- 非计数 [ok]：分层补丁的形状与结果，供人工核对 ----
print("  [ok]   新增分层：炎症性疾病 ⊑ 疾病；感染性疾病 ⊑ 炎症性疾病；"
      "脑炎 ⊑ 炎症性疾病 ∧ 脑部疾病（多继承）")
print("  [ok]   新增症状边（层边 vs 直接边）：" + "；".join(
    f"{s} --has_symptom({c})--> {d}" for (s, d), c in sorted(LAY_SYM.items())))
print("  [ok]   癫痫 结果集（8 种）：" + "、".join(sorted(_epi_total)))
print("  [ok]   脑炎 结果集（7 种）：" + "、".join(sorted(_ence_total)))
print("  [ok]   感染性疾病 结果集（3 种）：" + "、".join(sorted(_inf_total)))

# ============================================ §M13 影像特征子树（影像管线补丁）
print("\n§M13 影像特征子树（seed_medical_v3_imaging.sql）")

with open(str(IMG_PATCH), encoding="utf-8") as f:
    iraw = f.read()

# 需求列的层级清单（词形 → 上位）
IMG_TOP = "影像特征"
IMG_DIMS = {"形态学特征": IMG_TOP, "大小特征": IMG_TOP,
            "密度特征": IMG_TOP, "强化特征": IMG_TOP, "位置特征": IMG_TOP}
IMG_LEAVES = {
    "形状特征": ["圆形", "椭圆形", "不规则形", "分叶状"],
    "边缘特征": ["光滑", "毛刺状", "模糊", "针刺状"],
    "大小特征": ["微小", "小型", "中等", "大型"],          # ★ 小/大 撞词 → 小型/大型
    "密度特征": ["低密度", "等密度", "高密度", "钙化"],
    "强化特征": ["均匀强化", "不均匀强化", "环形强化", "无强化"],
    "位置特征": ["肺上叶", "肺下叶", "肺门", "外周", "中央"],
}
IMG_SHAPE_EDGE = {"形状特征": "形态学特征", "边缘特征": "形态学特征"}
IMG_ALL = ([IMG_TOP] + list(IMG_DIMS) + list(IMG_SHAPE_EDGE)
           + [w for v in IMG_LEAVES.values() for w in v])
IMG_STRUCT = ["病理结构", "结节", "恶性结节", "良性结节"]
IMG_TYP = {("恶性结节", "不规则形"): 0.8, ("恶性结节", "毛刺状"): 0.8,
           ("恶性结节", "高密度"): 0.8, ("良性结节", "圆形"): 0.8,
           ("良性结节", "光滑"): 0.8, ("良性结节", "低密度"): 0.8}
IMG_FEAT = {"圆形": "+圆形", "椭圆形": "+椭圆形", "不规则形": "+不规则", "分叶状": "+分叶",
            "光滑": "+光滑", "毛刺状": "+毛刺", "模糊": "+模糊", "针刺状": "+针刺",
            "微小": "+微小", "小型": "+小型", "中等": "+中等", "大型": "+大型"}

# ---- M13.1 义项层（7 项）----
miss_img = [w for w in IMG_ALL + IMG_STRUCT if sid(conn, w) is None]
check(f"M13 37 个影像义项全部新增（实测缺 {len(miss_img)}）", not miss_img, f"缺 {miss_img}")
img_ids = {sid(conn, w) for w in IMG_ALL + IMG_STRUCT}
check("M13 新义项 sense_id 恰为 4040-4076（与 4001-4030 / 4031-4036 / 4037-4039 "
      "/ 4101 段完全隔离）",
      img_ids == set(range(4040, 4077)), str(sorted(img_ids)[:3] + ["…"] + sorted(img_ids)[-3:]))
bad_def_img = conn.execute(
    "SELECT w.word, s.definition FROM senses s JOIN words w ON w.word_id=s.word_id "
    "WHERE s.sense_id BETWEEN 4040 AND 4076 "
    "  AND (length(trim(s.definition))=0 OR s.definition GLOB '*[0-9a-zA-Z]*')").fetchall()
check("M13 37 条新增释义非空且不含数字/拉丁字母"
      "（尺寸的毫米口径只写在注释里 —— 数值属 Layer 2 参数，符号库只认符号）",
      not bad_def_img, str(bad_def_img))
bad_cat_img = conn.execute("""
  SELECT sc.sense_id, sc.cat FROM sense_categories sc
   WHERE sc.sense_id BETWEEN 4040 AND 4076
     AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat)""").fetchall()
no_cat_img = [w for w in IMG_ALL + IMG_STRUCT if not count(
    conn, "SELECT COUNT(*) FROM sense_categories WHERE sense_id=?", (sid(conn, w),))]
check("M13 新增义项均已挂载合法类别（未新增类别行）",
      not bad_cat_img and not no_cat_img, f"非法 {bad_cat_img} / 未挂 {no_cat_img}")
_img_cat_dist = {r[0]: r[1] for r in conn.execute(
    "SELECT cat, COUNT(*) FROM sense_categories WHERE sense_id BETWEEN 4040 AND 4076 "
    "GROUP BY cat")}
check(f"M13 类别分布 = 抽象 8（维度）/ 性状 20（取值）/ 实体 4（病灶结构）/ 处所 5（位置），"
      f"实测 {_img_cat_dist}",
      _img_cat_dist == {"抽象": 8, "性状": 20, "实体": 4, "处所": 5}, str(_img_cat_dist))
_img_multisense = [w for w in IMG_ALL + IMG_STRUCT if count(
    conn, "SELECT COUNT(*) FROM senses s JOIN words wd ON wd.word_id=s.word_id "
          "WHERE wd.word=? AND s.pos='n' AND s.is_primary=1", (w,)) != 1]
check("M13 每个影像义项恰有 1 个 is_primary=1 的义项（v0.1 单词义）",
      not _img_multisense, str(_img_multisense))
_img_badtype = conn.execute(
    "SELECT word, word_type FROM words WHERE word_id BETWEEN 4040 AND 4076 "
    "  AND (word_type IS NULL OR word_type NOT IN "
    "       ('单纯词','合成词','复合词','派生词','重叠词','联绵词','叠音词','音译词','缩略语'))"
).fetchall()
check("M13 新增词条 word_type 均为合法取值", not _img_badtype, str(_img_badtype))

# ---- M13.2 is_a 层级（8 项）----
check("M13 ★需求项：影像特征 ⊑ 医学概念（顶层，挂到既有医学树而非悬空根）",
      reachable(conn, "医学概念", "影像特征"))
_bad_dim = [d for d, p in IMG_DIMS.items() if not rel_has(d, "is_a", p)]
check(f"M13 ★需求项：五个特征维度 ⊑ 影像特征（实测缺 {len(_bad_dim)}）", not _bad_dim, str(_bad_dim))
_bad_se = [d for d, p in IMG_SHAPE_EDGE.items() if not rel_has(d, "is_a", p)]
check("M13 ★需求项：形状特征 / 边缘特征 ⊑ 形态学特征", not _bad_se, str(_bad_se))
_bad_leaf = [f"{leaf}→{dim}" for dim, leaves in IMG_LEAVES.items()
             for leaf in leaves if not rel_has(leaf, "is_a", dim)]
check(f"M13 ★需求项：25 个取值分别 ⊑ 其所属维度（实测缺 {len(_bad_leaf)}）",
      not _bad_leaf, str(_bad_leaf))
check("M13 ★需求项：病理结构 ⊑ 医学概念（与 解剖结构 / 生理过程 / 病理状态 并列的顶层分支；"
      "不是 病理状态 —— 结节是**结构**不是状态，见补丁取舍 (2)）",
      rel_has("病理结构", "is_a", "医学概念")
      and not reachable(conn, "病理状态", "病理结构"))
check("M13 ★需求项：结节 ⊑ 病理结构；恶性结节 / 良性结节 ⊑ 结节",
      rel_has("结节", "is_a", "病理结构") and rel_has("恶性结节", "is_a", "结节")
      and rel_has("良性结节", "is_a", "结节"))
check("M13 结节 的 is_a 下位恰为 {恶性结节, 良性结节}",
      {r[0] for r in conn.execute(
          "SELECT descendant FROM closure WHERE relation='is_a' AND ancestor=?",
          (sid(conn, "结节"),))} == {sid(conn, "恶性结节"), sid(conn, "良性结节")})
_img_desc = {r[0] for r in conn.execute(
    "SELECT descendant FROM closure WHERE relation='is_a' AND ancestor=?",
    (sid(conn, IMG_TOP),))}
check(f"M13 影像特征 的 is_a 下位 = 32 个（实测 {len(_img_desc)}；"
      f"5 维度 + 形状 4 + 边缘 4 + 大小 4 + 密度 4 + 强化 4 + 位置 5 + 形状/边缘 2 子维度）",
      len(_img_desc) == 32, f"实际 {len(_img_desc)}：{words_of(conn, _img_desc)}")
check("M13 每个新义项的 is_a 出边恰 1 条（本次无多继承节点 —— 与 癫痫/脑炎 的形态不同）",
      all(count(conn, "SELECT COUNT(*) FROM relations WHERE relation='is_a' "
                      "AND src_kind='sense' AND src_id=?", (sid(conn, w),)) == 1
          for w in IMG_ALL + IMG_STRUCT))
_clo_img = {w: count(conn, "SELECT COUNT(*) FROM closure WHERE relation='is_a' AND descendant=?",
                     (sid(conn, w),)) for w in
            ("影像特征", "形态学特征", "形状特征", "圆形", "大小特征", "微小",
             "病理结构", "结节", "恶性结节")}
check(f"M13 is_a 闭包按层深自动派生（实测 {_clo_img}）：影像特征 2 / 形态学 3 / "
      f"形状特征 4 / 圆形 5 / 大小特征 3 / 微小 4 / 病理结构 2 / 结节 3 / 恶性结节 4",
      _clo_img == {"影像特征": 2, "形态学特征": 3, "形状特征": 4, "圆形": 5,
                   "大小特征": 3, "微小": 4, "病理结构": 2, "结节": 3, "恶性结节": 4},
      str(_clo_img))
check("M13 深链可达：圆形 的 is_a 上位 5 跳（形状特征→形态学特征→影像特征→医学概念→抽象概念）；"
      "恶性结节 4 跳 —— 影像子树锚定到 抽象概念，非悬空孤岛",
      count(conn, "SELECT depth FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
            (sid(conn, "抽象概念"), sid(conn, "圆形"))) == 5
      and count(conn, "SELECT depth FROM closure WHERE relation='is_a' AND ancestor=? AND descendant=?",
                (sid(conn, "抽象概念"), sid(conn, "恶性结节"))) == 4)

# ---- M13.3 关系类型（4 项）----
check("M13 ★需求项：typically_has 已登记，且 is_transitive=0 / inverse_of=typical_of",
      count(conn, "SELECT COUNT(*) FROM relation_types WHERE relation='typically_has' "
                  "AND is_transitive=0 AND inverse_of='typical_of'") == 1)
check("M13 逆关系 typical_of 成对（双向 inverse_of 一致，无悬空/无单向）",
      count(conn, """
        SELECT COUNT(*) FROM relation_types rt
          JOIN relation_types t2 ON t2.relation = rt.inverse_of
         WHERE rt.relation='typically_has' AND rt.inverse_of='typical_of'
           AND t2.inverse_of='typically_has'""") == 1)
check("M13 typically_has / typical_of 的 relation_class='属性'（与 has_property / has_symptom 同类）",
      {r[0] for r in conn.execute(
          "SELECT DISTINCT relation_class FROM relation_types "
          " WHERE relation IN ('typically_has','typical_of')")} == {"属性"})
check("M13 全库不变量仍然成立：所有 is_transitive=1 的关系都有 closure_policy 行"
      "（新类型是 is_transitive=0，故不需要策略行 —— 从源头避开"
      "『is_transitive=1 却缺策略行 → 静默不物化』的坑）",
      not conn.execute("""
        SELECT rt.relation FROM relation_types rt WHERE rt.is_transitive=1
          AND NOT EXISTS (SELECT 1 FROM closure_policy cp WHERE cp.relation=rt.relation)""").fetchall())

# ---- M13.4 has_property / typically_has 边（7 项）----
check("M13 ★需求项：结节 has_property 形态学特征 / 密度特征",
      rel_has("结节", "has_property", "形态学特征") and rel_has("结节", "has_property", "密度特征"))
check("M13 has_property 出边恰 2 条（均在 结节 上）",
      count(conn, "SELECT COUNT(*) FROM relations WHERE relation='has_property' "
                  "AND src_kind='sense' AND src_id=?", (sid(conn, "结节"),)) == 2)
check("M13 ★ has_property 是 is_transitive=1 + incremental ⇒ **会派生闭包**"
      "（本补丁 2 行；注意与医学库 6 条医学关系 is_transitive=0 / 零闭包的行为不同，"
      "见补丁取舍 (3)）",
      count(conn, "SELECT COUNT(*) FROM closure WHERE relation='has_property' "
                  "AND descendant=?", (sid(conn, "结节"),)) == 2
      and count(conn, "SELECT COUNT(*) FROM closure WHERE relation='has_property'") == 4)
_bad_typ = [f"{s}→{d}" for (s, d) in IMG_TYP if not rel_has(s, "typically_has", d)]
check("M13 ★需求项：typically_has 六条边齐备（恶性结节→不规则形/毛刺状/高密度；"
      "良性结节→圆形/光滑/低密度）", not _bad_typ, str(_bad_typ))
_conf_img = {(s, d): count(
    conn, "SELECT COUNT(*) FROM relations WHERE relation='typically_has' AND src_kind='sense' "
          "AND dst_kind='sense' AND src_id=? AND dst_id=? AND confidence=?",
    (sid(conn, s), sid(conn, d), c)) for (s, d), c in IMG_TYP.items()}
check(f"M13 六条 typically_has 的 confidence 均为 0.8（'统计上典型但不必然'）",
      all(v == 1 for v in _conf_img.values()), str(_conf_img))
check("M13 typically_has 未产生任何闭包行（is_transitive=0 的实际行为一致）",
      count(conn, "SELECT COUNT(*) FROM closure WHERE relation='typically_has'") == 0)
_typ_fwd = {r[0] for r in conn.execute("""
    SELECT DISTINCT wf.word FROM relations r
      JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wf ON wf.word_id=s2.word_id
     WHERE r.relation='typically_has' AND r.src_kind='sense'
       AND ( r.src_id = ?
             OR r.src_id IN (SELECT ancestor FROM closure
                              WHERE relation='is_a' AND descendant=?) )""",
    (sid(conn, "恶性结节"), sid(conn, "恶性结节")))}
check("M13 ★功能查询（正向）：恶性结节 的典型影像特征 = {不规则形, 毛刺状, 高密度}"
      "（Layer 3 的诊断倾向依据）",
      _typ_fwd == {"不规则形", "毛刺状", "高密度"}, str(sorted(_typ_fwd)))
_typ_rev = conn.execute("""
    SELECT wn.word, COUNT(*) AS hit FROM relations r
      JOIN senses s1 ON s1.sense_id=r.src_id JOIN words wn ON wn.word_id=s1.word_id
      JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wf ON wf.word_id=s2.word_id
     WHERE r.relation='typically_has' AND r.src_kind='sense'
       AND wf.word IN ('不规则形','毛刺状','高密度')
     GROUP BY wn.word ORDER BY hit DESC""").fetchall()
check(f"M13 ★功能查询（反向）：特征向量 {{不规则形,毛刺状,高密度}} → 唯一命中 恶性结节 hit=3"
      f"（实测 {_typ_rev}）",
      len(_typ_rev) == 1 and _typ_rev[0][0] == "恶性结节" and _typ_rev[0][1] == 3,
      str(_typ_rev))

# ---- M13.5 义素（8 项）----
_reg_img = {r[0] for r in conn.execute(
    "SELECT feature FROM semantic_features WHERE feature IN "
    "('+圆形','+椭圆形','+不规则','+分叶','+光滑','+毛刺','+模糊','+针刺',"
    " '+微小','+小型','+中等','+大型')")}
check(f"M13 ★需求项：12 个影像义素已在 semantic_features 登记（实测 {len(_reg_img)}）",
      _reg_img == set(IMG_FEAT.values()), str(sorted(_reg_img)))
_img_sf = {r[0]: r[1] for r in conn.execute(
    "SELECT sense_id, feature FROM sense_features WHERE sense_id BETWEEN 4040 AND 4076")}
check(f"M13 ★需求项：12 个形状/边缘/大小取值各标 1 行义素（实测 {len(_img_sf)} 行）",
      len(_img_sf) == 12
      and _img_sf == {sid(conn, w): f for w, f in IMG_FEAT.items()},
      str(sorted(_img_sf.items())))
_bad_val_img = conn.execute(
    "SELECT sense_id, feature, value FROM sense_features "
    " WHERE sense_id BETWEEN 4040 AND 4076 AND value <> '1'").fetchall()
check("M13 影像义素 value 一律 '1'（boolean 口径，与 schema §14 及严重度义素一致）",
      not _bad_val_img, str(_bad_val_img))
check("M13 ★需求项：义素只标在「形状 / 边缘 / 大小」三个维度上 —— "
      "密度(4)/强化(4)/位置(5) 的取值**未标义素**（需求只要 3 个维度，"
      "补齐 SQL 见补丁 §7.6）",
      not [w for w in (IMG_LEAVES["密度特征"] + IMG_LEAVES["强化特征"] + IMG_LEAVES["位置特征"])
           if count(conn, "SELECT COUNT(*) FROM sense_features WHERE sense_id=?",
                    (sid(conn, w),))])
check("M13 义素名与词形**刻意不同名**（+毛刺 ↔ 毛刺状、+不规则 ↔ 不规则形）："
      "义素是 Layer 2 分类器的输出标签，不照抄词形",
      IMG_FEAT["毛刺状"] == "+毛刺" and IMG_FEAT["不规则形"] == "+不规则"
      and IMG_FEAT["针刺状"] == "+针刺" and IMG_FEAT["分叶状"] == "+分叶")
_img_off = conn.execute("""
  SELECT sf.sense_id, w.word FROM sense_features sf
    JOIN senses s ON s.sense_id=sf.sense_id JOIN words w ON w.word_id=s.word_id
   WHERE sf.sense_id BETWEEN 4040 AND 4076
     AND sf.feature NOT IN (SELECT feature FROM semantic_features)""").fetchall()
check("M13 影像义项上的义素全部已在 semantic_features 登记（无野义素 / FK 未被绕过）",
      not _img_off, str(_img_off))
check("M13 义素落点不变量：12 个影像义素全部落在「形状/边缘/大小」子树内"
      "（与症状严重度义素互不串场：+severe/+moderate/+mild 仍在症状子树）",
      {r[0] for r in conn.execute(
          "SELECT DISTINCT feature FROM sense_features WHERE feature IN "
          "('+severe','+moderate','+mild')")} == {"+severe", "+moderate", "+mild"}
      and count(conn, "SELECT COUNT(*) FROM sense_features WHERE feature IN "
                      "('+圆形','+椭圆形','+不规则','+分叶','+光滑','+毛刺','+模糊','+针刺',"
                      " '+微小','+小型','+中等','+大型')") == 12)

# ---- M13.6 跨库链接（3 项）----
check("M13 ★需求项：跨库边 信号 --domain_specialization--> 医学库节点「强化特征」",
      count(conn, "SELECT COUNT(*) FROM relations r WHERE r.relation='domain_specialization' "
                  "AND r.src_kind='sense' AND r.src_id=? "
                  "AND r.dst_kind='node' AND r.dst_id=3104",
            (sid(conn, "信号"),)) == 1)
_en_3104 = conn.execute("SELECT label, entity_class, domain FROM entity_nodes "
                        "WHERE node_id=3104").fetchone()
check(f"M13 领域节点 3104 = 强化特征 / ImagingFeature / medical（实测 {_en_3104}）",
      _en_3104 == ("强化特征", "ImagingFeature", "medical"), str(_en_3104))
check("M13 跨库边不参与闭包物化（domain_specialization is_transitive=0，端点为 node）",
      count(conn, "SELECT COUNT(*) FROM closure WHERE relation='domain_specialization'") == 0)

# ---- M13.7 纯追加 / 幂等 / 自足性 / 闸门 / 加载顺序（9 项）----
_i_targets = {t for t in re.findall(r"(?mi)^\s*INSERT\s+INTO\s+([A-Za-z_]+)", iraw)
              if not t.startswith("_")}
check(f"M13 补丁只向 {sorted(_i_targets)} 追加"
      f"（不动 closure / closure_policy / semantic_categories / relation_constraints / "
      f"noun_classifier_pairs）",
      _i_targets == {"words", "senses", "sense_categories", "relation_types", "relations",
                     "entity_nodes", "sources", "semantic_features", "sense_features"},
      str(sorted(_i_targets)))
check("M13 补丁零 closure 的 INSERT（148 行全部由触发器派生）",
      not re.search(r"(?m)^\s*INSERT\s+INTO\s+closure\b", iraw, re.I))
check("M13 补丁不改 schema：无 ALTER TABLE / 无非 TEMP 的 CREATE TABLE",
      not re.search(r"(?mi)^\s*ALTER\s+TABLE\b", iraw)
      and not re.search(r"(?mi)^\s*CREATE\s+TABLE\s+(?!IF\s+NOT\s+EXISTS\s+_[a-z_]*)", iraw),
      "出现 ALTER / 非临时建表")
check("M13 补丁只新增 typically_has / typical_of 两个关系类型，"
      "且不 UPDATE 任何既有类型定义（不重定义 is_a / has_property / domain_specialization）",
      count(conn, "SELECT COUNT(*) FROM relation_types "
                  "WHERE relation IN ('typically_has','typical_of')") == 2
      and not re.search(r"(?mi)^\s*UPDATE\s+relation_types\b", iraw))

_I_TABLES = ["words", "senses", "sense_categories", "relation_types", "relations",
             "entity_nodes", "closure", "changelog", "sense_features",
             "semantic_features", "sources"]
_i1 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _I_TABLES}
try:
    load(conn, IMG_PATCH)
    _i2 = {t: count(conn, f"SELECT COUNT(*) FROM {t}") for t in _I_TABLES}
    _idrift = {t: (_i1[t], _i2[t]) for t in _I_TABLES if _i1[t] != _i2[t]}
    check("M13 幂等：第二次执行影像补丁后 11 张表行数零漂移", not _idrift, str(_idrift))
except sqlite3.Error as e:
    check("M13 幂等：第二次执行影像补丁后 11 张表行数零漂移", False, f"{type(e).__name__}: {e}")

_i_fk5, _i_beg5 = iraw.find("PRAGMA foreign_keys"), iraw.find("BEGIN;")
check("M13 补丁自足性：PRAGMA foreign_keys 写在 BEGIN 之前（事务内是 no-op）",
      0 <= _i_fk5 < _i_beg5, f"fk@{_i_fk5} begin@{_i_beg5}")
check("M13 补丁自足性：PRAGMA recursive_triggers 已声明",
      0 <= iraw.find("PRAGMA recursive_triggers") < _i_beg5)
check("M13 补丁事务配对：BEGIN == COMMIT",
      len(re.findall(r"(?m)^\s*BEGIN\s*;", iraw)) == len(re.findall(r"(?m)^\s*COMMIT\s*;", iraw))
      and len(re.findall(r"(?m)^\s*BEGIN\s*;", iraw)) > 0)

# 闸门 (a)：裸 schema 上加载 → 被拒绝
c_bare5 = connect()
load(c_bare5, SCHEMA)
try:
    load(c_bare5, IMG_PATCH)
    check("M13 依赖闸门：裸 schema（无 seed_medical_v3）上加载影像补丁被拒绝", False,
          "竟然加载成功")
except sqlite3.IntegrityError as e:
    check("M13 依赖闸门：裸 schema 上加载影像补丁被拒绝，且报错文本自带修复提示",
          "prereq" in str(e), str(e))
except sqlite3.Error as e:
    check("M13 依赖闸门：裸 schema 上加载影像补丁被拒绝，且报错文本自带修复提示",
          False, f"{type(e).__name__}: {e}")
c_bare5.close()

# 闸门 (b)：有 schema+seed+ext 但**没有** seed_medical_v3（医学概念/信号未建）→ 同样被拒绝
c_noimg = connect()
for _p in (SCHEMA, SEED, SEED_EXT):
    load(c_noimg, _p)
try:
    load(c_noimg, IMG_PATCH)
    check("M13 依赖闸门：缺少 seed_medical_v3（医学概念 / 信号 未建）时被拒绝", False,
          "竟然加载成功 —— 说明内连接静默少行，未被 §0 拦下")
except sqlite3.IntegrityError as e:
    check("M13 依赖闸门：缺少 seed_medical_v3（医学概念 / 信号 未建）时被拒绝",
          "prereq" in str(e), str(e))
except sqlite3.Error as e:
    check("M13 依赖闸门：缺少 seed_medical_v3 时被拒绝", False, f"{type(e).__name__}: {e}")
c_noimg.close()

# 加载顺序不变量：imaging 字典序必须落在 seed_medical_v3.sql 之后、症状补丁之前
# （'i' 0x69 < 'p' 0x70；影像子树与症状体系无交集，故这个位次既安全又不构成依赖）
check("M13 加载顺序不变量：影像补丁在字典序中排在 seed_medical_v3.sql 之后、"
      "patch_symptom 之前（'i' 0x69 < 'p' 0x70）",
      IMG_PATCH.name in _seed_names
      and _seed_names.index(SEED_MED.name) < _seed_names.index(IMG_PATCH.name)
      < _seed_names.index(SEED_MED_PATCH.name),
      str(_seed_names))

# ---- 非计数 [ok]：形状树 / 映射对照 / 三处取舍，供人工核对 ----
print("  [ok]   影像特征子树（37 个新义项）：")
print("           影像特征 ⊑ 医学概念")
for _d, _leaves in IMG_LEAVES.items():
    _mid = " ─(形状/边缘)─ " if _d in IMG_SHAPE_EDGE else " ─ "
    print(f"             ├─ {_d}{_mid} " + " / ".join(_leaves))
print("           病理结构 ⊑ 医学概念 ─ 结节 ─ 恶性结节 / 良性结节"
      "（typically_has 各 3 条典型特征）")
print("  [ok]   词形 ↔ 义素对照（Layer 2 分类器输出标签）：" + "；".join(
    f"{w}→{f}" for w, f in sorted(IMG_FEAT.items())))
print("  [ok]   ★ 取舍 1：需求的大小分级写「小(5-10mm) / 大(>30mm)」，但「小」(word_id 1011) 与"
      "「大」(word_id 1010) **已存在于通用库**（seed 1000-1027 段的形容词），")
print("          且 words.word 是全局 UNIQUE（不带 pos）⇒ 在「不改 schema、不改已有数据」下"
      "无法再建同名词条。故改用 **小型 / 大型**（微小 / 中等 不撞词，保留）。")
print("          另两条备选路（复用语义词 1010/1011、改 schema 加 pos）的代价见补丁 §7.8 (a)。")
print("  [ok]   ★ 取舍 2：病理结构 挂 医学概念（与 解剖结构/生理过程/病理状态 并列的顶层分支），"
      "不挂 病理状态 —— 结节是结构不是状态。改挂的 SQL 见 §7.8 (b)。")
print("  [ok]   ★ 取舍 3：has_property 在 schema 里是 is_transitive=1 + incremental，"
      "故「结节 has_property 形态学/密度特征」会派生 2 行闭包；")
print("          typically_has 显式 is_transitive=0，零闭包。二者行为不同，勿想当然。")
print("  [ok]   数值→符号映射契约（Layer 2 参数，**不入库**，登记于补丁 §7.1）：")
print("           形状 圆形度>0.90→圆形 / 0.75-0.90→椭圆形 / <0.75→不规则形 / 分叶≥2→分叶状")
print("           大小 直径<5→微小 / 5-10→小型 / 10-30→中等 / ≥30→大型（单位：毫米）")
print("           边缘 梯度方差低→光滑 / 中→模糊 / 短密放射→毛刺状 / 细长放射→针刺状")

# ------------------------------------------------------------------ 汇总
print("\n" + "=" * 78)
print("表行数（schema + seed + ext + medical + imaging + patch_symptom "
      "+ patch_symptom_layers + severity 的最终状态）：")
for t in ["semantic_categories", "relation_types", "words", "senses", "sense_categories",
          "relations", "entity_nodes", "closure", "changelog",
          "semantic_features", "sense_features", "sources"]:
    print(f"  {t:<24} {count(conn, f'SELECT COUNT(*) FROM {t}'):>5}")
print("  " + "-" * 74)
print("  relations(按类型) " +
      ", ".join(f"{r[0]}={r[1]}" for r in conn.execute(
          "SELECT relation, COUNT(*) FROM relations GROUP BY relation ORDER BY 2 DESC")))
print(f"  {'closure(按类型)':<22} " +
      ", ".join(f"{r[0]}={r[1]}" for r in conn.execute(
          "SELECT relation, COUNT(*) FROM closure GROUP BY relation ORDER BY 2 DESC")))
print("  " + "-" * 74)
print(f"  医学概念 下位节点数   {len(med_desc)}")
print(f"  depth>1 闭包行数      {depth_gt1}")
print(f"  医学库跨库链接数      {len(cross)}")
print("-" * 78)
print(f"结果: PASS {PASS} / FAIL {FAIL}")
if FAILURES:
    print("\n失败明细：")
    for n, d in FAILURES:
        print(f"  · {n}\n      {d}")
print("=" * 78)

conn.close()
sys.exit(1 if FAIL else 0)
