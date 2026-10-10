# Logos 项目长期约定

> 跨会话必须遵守的硬约定。日常进展写 `YYYY-MM-DD.md`，本文件只放"下次开工前必须知道"的事。

## 1. ID 段登记表（新增数据前先看这里）

| 段位 | 归属 | 内容 |
|---|---|---|
| 1–27 | `schema_v3.sql` §14 | 猫/鱼/动物/把/只/书/桌子… 26 词 27 义项 |
| 1000–1027 | `seed_general_v3.sql` | 通用领域种子 27 词 / 27 义项 |
| 2001–2005, 2101–2135, 2201 | `seed_general_v3_ext.sql` | 营养物·可食用·有营养 / 分类骨架 35 义项 / 量词「份」 |
| 3001 | `seed_general_v3_ext.sql` | `entity_nodes` 动物学库条目（cross_links 用） |
| 3101–3103 | `seed_medical_v3.sql` | `entity_nodes` 医学库条目（营养素/药物治疗/脑电图信号） |
| 3104 | `seed_medical_v3_imaging.sql` | `entity_nodes` 影像库条目「强化特征 / ImagingFeature」（cross_links 用） |
| 4001–4030, 4101 | `seed_medical_v3.sql` | 医学骨架 30 义项 / 4101 = 通用库兜底词「信号」 |
| 4031–4036 | `seed_medical_v3_patch_symptom.sql` | 6 个通用症状（疼痛/发热/头痛/眩晕/乏力/意识模糊） |
| 4037–4039 | `seed_medical_v3_patch_symptom_layers.sql` | 疾病分层 3 义项（炎症性疾病/感染性疾病/脑炎） |
| 4040–4076 | `seed_medical_v3_imaging.sql` | **影像特征子树 37 义项**：影像特征/形态学·形状·边缘·大小·密度·强化·位置特征 + 25 取值 + 病理结构/结节/恶性结节/良性结节 |
| （不占 ID） | `seed_medical_v3_patch_symptom_severity.sql` | **纯义素标注**，不新增 word/sense；只写 `sources`+1 / `semantic_features`+3 / `sense_features`+8 |
| 7000+ | `tests/*.py` 夹具 | 老鼠/石头 |

**新种子必须自选空段**，并把段位登记进本表 + 对应 SQL 文件头 + 测试夹具注释。
**同时必须让文件名字典序落在其依赖之后**（见 §2）。

## 1b. 加载顺序（八份，靠字典序也靠显式清单）

`schema_v3.sql` → `seed_general_v3.sql` → `seed_general_v3_ext.sql` →
`seed_medical_v3.sql` → `seed_medical_v3_imaging.sql` →
`seed_medical_v3_patch_symptom.sql` →
`seed_medical_v3_patch_symptom_layers.sql` →
`seed_medical_v3_patch_symptom_severity.sql`

- 字典序天然正确：`seed_g` < `seed_m`；`seed_medical_v3.sql` <
  `seed_medical_v3_imaging.sql` < `seed_medical_v3_patch_symptom.sql`
  （`.` U+002E < `_` U+005F；`..._imaging` 的 `'i'`(0x69) < `..._patch` 的
  `'p'`(0x70)，故影像子树天然夹在两者之间 ⇒ 它依赖的「医学概念/信号」已就位）
- 两份症状补丁共用前缀 `..._patch_symptom` + `_`，其后 `'l'`(0x6C, layers) <
  `'s'`(0x73, severity) ⇒ layers 在前（二者互不依赖，位置自由，
  但**都必须晚于** `..._patch_symptom.sql`）
- ⚠ **反例**：`..._patch_severity.sql` 的 `'e'` < `'y'` ⇒ 排在
  `..._patch_symptom.sql` **之前** → 症状义项未建时执行 → 静默丢 8 行。
  故严重度补丁取名 `..._patch_symptom_severity.sql`。实测见
  `docs/MEDICAL_LIB_DESIGN.md` §11
- `apply_seeds.py` 的 `SQL_FILES` 是显式清单，**新增种子要同步加进去**，
  并更新其 `EXPECT`（`senses/relations/closure` + `sense_features/semantic_features` 期望值）
- 全量基线（八份加载完）：senses 170 / relations 166 / closure 371 / words 169 /
  entity_nodes 5 / relation_types 45 / semantic_features 22 / sense_features 98 /
  sources 7 / sense_categories 148；`apply_seeds.py` 自检 38 项

## 1c. `closure.ancestor` 恒等于**边的 dst**（不变量 I3）—— 方向陷阱

`tr_closure_edge_ai` 写入的闭包行恒为「边的反向」：`ancestor = relations.dst_id`。

- 对 `is_a`（下位→上位）这正好是「上位在 ancestor 侧」，符合直觉
- 对 **`has_part`（整体→部分）则相反**：dst 是**部分**，所以
  **部分在 ancestor 侧、整体在 descendant 侧**
  - 查「整体 X 有哪些部分」→ `WHERE descendant = X`，取 `ancestor`
  - `closure.py` 的 `get_ancestors(脑, 'has_part')` 实际返回「脑的组成部分」
  - 若希望 ancestor 恰是「整体」，应改用逆关系 `part_of` 存储

## 2. `init_database` 自动发现种子（最高频的坑）

`logos/db/connection.py`：`SEED_GLOB = "seed_*.sql"`，`_resolve_seed_paths` 对项目根的
`seed_*.sql` **按文件名排序**加载。

- 新种子文件**无需注册**即进入所有走 `init_database` 的管线（e2e、解析器测试等）
- 因此它也会**改变下游测试的 ID 空间** —— 只有 7000 段是安全的
- 依赖顺序靠字典序：`seed_general_v3.sql` < `seed_general_v3_ext.sql`
  （`.` U+002E < `_` U+005F），扩名必须落在基础种子之后
- **排序看的是整个文件名，不只是"扩展名"**：`..._patch_severity.sql` vs
  `..._patch_symptom.sql` 在第二个字符就分出大小（`'e'` < `'y'`），前者反而在前。
  起名时务必用 `sorted(glob("seed_*.sql"))` 实跑一遍确认顺序
- 反向保险：给新补丁加**前置依赖硬检查**（§4b 手法 1）。若顺序被弄反，
  `init_database` 会**响亮报错**而不是静默少行（实测故意改名后抛
  `SeedError: CHECK constraint failed: _prereq_missing_...`）

## 3. 种子 SQL 的三条铁律

1. 全部 INSERT 必须幂等：`ON CONFLICT DO NOTHING` / `SELECT ... WHERE NOT EXISTS`
2. `INSERT ... SELECT ... ON CONFLICT` **必须带 WHERE**（哪怕 `WHERE 1`），
   否则被解析成 `SELECT ... ON` 连接语法
3. `PRAGMA foreign_keys` 必须写在 `BEGIN` 之前（事务内是 no-op）；
   对既有实体按 `(word, pos)` 自然键子查询，不硬编码既有 ID

## 4. 校验脚本约定

| 脚本 | 项数 | 说明 |
|---|---:|---|
| `validate_schema.py` | 102 | 只加载 schema |
| `validate_seed.py` | 77 | schema + seed + seed_ext（§1-§6 = 65，§7 = A/B 12） |
| `validate_connection.py` | 159 | |
| `validate_closure.py` | 87 | 只加载 schema |
| `validate_syntax_parser.py` | 214 | 只加载 schema |
| `validate_medical.py` | 230 | schema + seed + ext + medical + imaging + 3 份补丁（§M1-§M13；独立计数，不动 validate_seed 的 77） |
| `tests/test_e2e_cat_eats_fish.py` | 64 (+4 XFAIL) | 走 `init_database`，会加载全部 seed |
| `apply_seeds.py` 自检 | 38 | 建库后结构性自检（exit 0 = 全绿） |

- 断言编号是需求给定项，**不擅自增删**；额外检查以 `[ok]` 非计数形式透明报告
- 已知偏差用 `XFAIL` 档记录，使「FAIL=0」与「诚实记录缺陷」兼容
- **新增种子会改变下游计数**：`init_database` 自动发现 + 自检脚本里的硬编码期望值。
  加 `seed_medical_v3.sql` 时踩到两处：e2e 夹具 ID 撞车（已迁 7000 段）、
  `apply_seeds.py` 自检「分类骨架节点数 = 36」变成 61（已按 `sense_id < 4000` 过滤）。
  加 `seed_medical_v3_patch_symptom.sql`（4031–4036）时未撞车，仅需更新
  `EXPECT` 与 §M4「医学概念 is_a 下位 24 → 30」。
  加 `seed_medical_v3_patch_symptom_severity.sql` 时不新增 word/sense，
  对 ID 空间零影响，仅需更新 `EXPECT`（+`sense_features`/`semantic_features`）。
  加 `seed_medical_v3_patch_symptom_layers.sql`（4037–4039）时未撞车，需更新
  `EXPECT`（130→133 / 112→120 / 208→223）与 §M4「医学概念 is_a 下位 30 → 33」、
  §M10「癫痫 结果集 6 → 8」。

## 4c. 症状继承的查询口径（易写错）

`has_symptom` 一律 `is_transitive=0`（**不产生闭包行**），继承发生在 `is_a` 维度上：

```sql
-- 「疾病 D 的症状」= 直接声明 ∪ 沿 is_a 继承（去重）
SELECT DISTINCT wd.word FROM relations r
  JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
 WHERE r.relation='has_symptom' AND r.src_kind='sense'
   AND ( r.src_id = :D
         OR r.src_id IN (SELECT ancestor FROM closure
                          WHERE relation='is_a' AND descendant=:D) );
```

- 通用症状声明在**层**上：`疾病→疼痛/乏力`、`脑部疾病→头痛/眩晕/意识模糊`、
  `炎症性疾病→发热`（补丁三新增的第三层）
- **层边 vs 直接边**都要会用：只在特定疾病上成立的症状走直接边 ——
  `癫痫→抽搐/记忆障碍/发热`、`脑炎→抽搐`、`阿尔茨海默病→记忆障碍`。
  于是 `癫痫` 结果集 = 层边继承 5 + 直接边 3 = 8 种（勿按"只有层边"想当然）
- 别忘了 `r.src_kind='sense'`（关系两端可能带 node，见跨库边）
- 完整设计与实测：`docs/MEDICAL_LIB_DESIGN.md`（§12 记补丁三）

## 4d. 症状严重度义素（`sense_features`）口径

严重度是**义项固有属性**（落在 `sense_features`），不是边可信度 ——
`relations.confidence` 保持"边有多可信"语义，两者正交，勿混用。

- 义素：`+severe` / `+moderate` / `+mild`，`value='1'`（boolean 口径）
- 分级：`+severe` 疼痛/头痛/抽搐/意识模糊；`+moderate` 眩晕/乏力/记忆障碍；`+mild` 发热
- 检索 = §4c 的继承口径 + `LEFT JOIN sense_features sf ON sf.sense_id=r.dst_id
  AND sf.feature IN ('+severe','+moderate','+mild')`
- ⚠ 已知妥协（详见 `docs/MEDICAL_LIB_DESIGN.md` §10.5 / §12.3）：
  1. `semantic_features.feature_group` 是 CHECK 白名单且**无「严重度」分组** →
     暂复用「极性」；加分组须 ALTER TABLE（与"不改 schema"冲突）
  2. `sense_features` **没有 source_id 列** → `source_id=6` 只能登记在 `sources`
     表 + `semantic_features.description`，落不进义素行
  3. ~~记忆障碍 不在 癫痫 的继承锥内 / 发热 无 `has_symptom` 边~~ →
     **已由补丁三解除**（癫痫 补直接边、发热 挂到新的 `炎症性疾病` 层）。
     ⚠ 但 `癫痫→发热` 是 `confidence=0.5` 的**弱关联**（发热是诱发因素而非症状），
     严格口径下应迁到 `has_precipitant`；撤回 SQL 在补丁三 §4.5

## 4e. 影像特征子树（`seed_medical_v3_imaging.sql`，ID 4040–4076）

`影像特征 ⊑ 医学概念`；五维度：形态学（→形状/边缘 子维度）·大小·密度·强化·位置。
层级：影像特征 2 / 形态学特征 3 / 形状特征 4 / 圆形 5 / 病理结构 2 / 结节 3 / 恶性结节 4。

- **`has_property` 是 `is_transitive=1`（incremental/12）** ⇒ 与医学库其余关系不同：
  `结节 has_property 形态学特征 / 密度特征` 这 **2 条边会派生 2 行闭包**（全库 has_property 闭包仅此 4 行）
- **`typically_has`（新增，is_transitive=0, inverse_of=typical_of）零闭包**；
  6 条边 confidence=0.8：恶性结节→不规则形/毛刺状/高密度，良性结节→圆形/光滑/低密度
- ⚠ **「小」「大」撞词**（已占 1010/1011，`pos='a'`；`words.word` 全局 UNIQUE 不带 pos）
  ⇒ 大小取值改 **小型 / 大型**（「中等」「微小」不撞，保留）
- 义素 12 行（形状/边缘/大小各 4）：`+圆形 +椭圆形 +不规则 +分叶 +光滑 +毛刺 +模糊
  +针刺 +微小 +小型 +中等 +大型`，`feature_group='极性'`（无「影像」分组，同 §4d 妥协）
- 跨库边：`信号 --domain_specialization--> entity_nodes 3104「强化特征」`，confidence=0.75
- 依赖闸门：具名 CHECK `_prereq_missing_run_seed_medical_v3_first_for_imaging`
  （裸 schema 或仅 schema+seed+ext 都会被拒）
- 设计全貌：`docs/MEDICAL_LIB_DESIGN.md` §13（架构定位/取舍/新增量/正反查询/六条已知限制）

## 4b. 种子文件里的两个可复用手法

1. **前置依赖硬检查**（比 ext 的 `NOT EXISTS` 兜底更强）：SQLite 的 `RAISE()` 只能
   出现在触发器体内，脚本里无法直接断言。改为建临时表 + **具名 CHECK 约束**：
   ```sql
   CREATE TEMP TABLE IF NOT EXISTS t(ok INTEGER NOT NULL,
       CONSTRAINT _prereq_missing_run_seed_xxx_first CHECK (ok = 1));
   DELETE FROM t;  INSERT INTO t(ok) SELECT CASE WHEN <条件> THEN 1 ELSE 0 END;  DROP TABLE t;
   ```
   失败时报错为 `CHECK constraint failed: <约束名>` —— **把修复提示写进约束名**，
   报错即自解释。（实测 SQLite 3.53.1 报的是约束名，不是表达式，故必须用具名约束。）
2. **自然键 CTE 映射词形→sense_id** 时，CTE 要**同时带 pos**：
   ext §1.9 只写 `e.pos='n'` 是因为源词恰好是名词；医学库的跨库边源词「吃」是**动词**，
   照抄会**静默丢边**（内连接不报错、只是少一行）。

## 5. 层文档

`docs/` 下每层一份：`SEED_GENERAL.md`、`SEED_EXT_NOTES.md`、`CONNECTION_LAYER.md`、
`CLOSURE_LAYER.md`、`CLOSURE_DIRECTION.md`、`SYNTAX_PARSER.md`、`BUILD_PRECHECK.md`、
`TEST_AUDIT.md`、`E2E_TEST_REPORT.md`、`MEDICAL_LIB_DESIGN.md`（医学库 + 症状继承
+ 症状严重度义素 + 炎症性疾病分层 + 影像特征子树 + 加载顺序实证）。
改数据/代码时同步改文档里的实测数字。

**计数字典（改数据前先看）**：`semantic_features` 22（原 10 + 症状严重度 3 + 影像 12）/
`sources` 7 / `relation_types` 45（原 43 + typically_has/typical_of）/
（义素种类）`+animate +human +countable +concrete +volitional +bounded polarity
+severe +moderate +mild +圆形 +椭圆形 +不规则 +分叶 +光滑 +毛刺 +模糊 +针刺
+微小 +小型 +中等 +大型`。

## 6. 未决事项（挂起，等用户决策）

1. `relation_types.max_depth` 是死列，与 `closure_policy.max_depth` 双写且取值冲突
2. `closure_policy.mode='off'` 时既有闭包行不清理（建议加 `tr_closure_policy_au`）
3. 可选索引 `idx_relations_src_id ON relations(src_id) WHERE src_kind='sense'`
4. `_CLOSED_CLASS_FALLBACK`（解析器里 90 个 OOV 虚词兜底）是临时桥梁，词典补齐后逐项删除
5. e2e 报告的 5 个「缝隙类」缺陷 K1-K5（倒装句竞争、连动句吸收、约束冲突不回传置信度、
   自相矛盾输出、受事合法性 rule 4 只认 category）
6. 影像子树三处留白（`docs/MEDICAL_LIB_DESIGN.md` §13.8）：密度/强化/位置取值未打义素；
   影像特征 → 疾病诊断的推理链未建（当前只到「结节 has_property …」）；`relation_constraints`
   未接入影像槽位校验
