# seed_general_v3_ext.sql —— 扩展种子数据说明

> 叠加在 `seed_general_v3.sql` 之上的第二批数据。两个目标：
> **① 让「鱼」在两个维度上各自成立（is_a 双挂 = DAG 多继承）；
> ② 补一层「抽象概念 / 人造物」分类骨架**，使科技与信息类词条有上位可挂。
>
> 验证状态：**`validate_seed.py` → PASS 77 / FAIL 0**（原 65 项 + 新增 12 项；全量回归见文末）

---

## 1. 运行方式

```bash
# 一次性建库：schema → seed → seed_ext，并做结构自检
python apply_seeds.py                    # 从零重建 ROOT/logos.db
python apply_seeds.py --keep             # 保留已有库做增量叠加（种子本身幂等）
python apply_seeds.py path/to/other.db   # 自定义目标库

python validate_seed.py                  # 77 项自检（会自动加载本文件）
```

`apply_seeds.py` 的四个要点：

1. **逐条语句执行，不是 `executescript`**。用 `sqlite3.complete_statement()` 切分语句
   后逐条 `execute`，报错才能精确到「文件 + 第 N 条语句 + 语句文本 + 冲突的键/端点」。
   `complete_statement` 能正确识别 `CREATE TRIGGER ... BEGIN ... END;` 的触发器体
   （体内的分号不误切）—— 本项目的 `docs/CONNECTION_LAYER.md` 已实测确认。
2. **连接用 `isolation_level=None`**。三个 SQL 文件都自带 `BEGIN; ... COMMIT;`；
   若沿用 python sqlite3 的隐式事务管理，显式 `BEGIN;` 会与驱动自动开启的事务撞车。
   设为 `None` 后 `conn.in_transaction` 也成为一个可信的收尾校验（漏写 COMMIT 立刻暴露）。
3. **失败绝不静默跳过**：外键冲突会从语句字面量反推出**缺失的父行值**
   （如 `senses.word_id = 999999 → words.word_id 中不存在`），UNIQUE 冲突会查出
   **库中已存在的那一行**，都附可执行的修复 SQL；编码/路径错误给出文件、字节偏移、
   搜索过的目录。退出码：`0` 成功 / `1` 自检未全绿 / `2` 路径错误 / `3` SQL 出错 / `4` 编码错误。
4. **`init_database` 会自动发现种子**：`logos/db/connection.py` 的 `SEED_GLOB = "seed_*.sql"`
   按文件名排序加载项目根的全部 `seed_*.sql`。因此本文件无需注册就进入所有走
   `init_database` 的管线 —— 这带来一个必须遵守的约束，见 [§8](#8-一个必须知道的联动init_database-自动发现-seed_sql)。

---

## 2. ID 段约定（先看这个，再看数据）

| 段位 | 归属 | 内容 |
|---|---|---|
| 1–27 | `schema_v3.sql` §14 | 猫/鱼/动物/把/只/书/桌子… 26 词 27 义项 |
| 1000–1027 | `seed_general_v3.sql` | 通用领域种子 27 词 |
| **2001–2005** | **本文件** | 营养物(2001)、食物(2002)/生物(2003) 兜底、可食用(2004)、有营养(2005) |
| **2101–2135** | **本文件** | 抽象概念 / 人造物 分类骨架（35 个义项） |
| **2201** | **本文件** | 量词「份」 |
| **3001** | **本文件** | `entity_nodes` 动物学库条目（cross_links 用） |
| 7000+ | `tests/` | 测试夹具（老鼠/石头） |

对**既有**实体一律按 `(word, pos)` 自然键子查询引用（临时视图 `ext_sid`），
因此本文件对 schema/seed 内部的 ID 取值不敏感。临时视图另起名 `ext_sid` 而非复用
`seed_sid`：两个文件可能在同一连接先后加载，重名会让后者的 `CREATE TEMP VIEW IF NOT EXISTS`
命中旧定义。

---

## 3. 任务一：鱼的多继承

### 3.1 两条独立路径

```
路径 A（生物分类维度）   鱼 → 动物 → 生物
路径 B（功能分类维度）   鱼 → 食物 → 营养物
```

`鱼` 因此是 **DAG 节点而非树节点**：同时是「动物」与「食物」的下位。
这不是"冗余"，而是两个正交视角 —— 前者回答"鱼属于哪类生物"，后者回答"鱼能拿来做什么"。

### 3.2 四条链的原子性与"独立性"

`validate_seed.py` 的 A1 不只检查"两条链都在"，还检查**两者互不为祖先**：

```sql
SELECT 1 FROM closure WHERE relation='is_a'
  AND ((ancestor=动物 AND descendant=食物) OR (ancestor=食物 AND descendant=动物));
-- 必须返回 0 行
```

少了这一条，"双挂"可能退化成一个假的 DAG —— 例如误写 `动物 is_a 食物` 时，
`鱼 → 动物 → 食物` 会让 A1 与 A2 同时通过，但语义已经错了（那是在说"动物是食物"）。
**断言要能分辨"两条路"和"一条链掰成两段"**，这是本任务里最容易被漏掉的一点。

实测：鱼的上位 = `动物(1跳) / 食物(1跳) / 生物(2跳) / 营养物(2跳)`，共 4 个（A2）。
鱼的 is_a **出边数 = 2**（A4）—— 注意是出边数为 2，不是上位数为 2：多继承的判据是
"从鱼出发有几条边"，闭包行数会因传递而更多。

### 3.3 为什么 `has_property` 要建属性节点

需求原文：`食物 has_property 可食用 / 有营养来源`。

schema 里 `has_property` 的**两端都是 sense**（`relations.src_kind/dst_kind ∈ {sense, node}`），
不建节点就落不了边。于是本文件新建两个形容词性义项：

| 节点 | pos | 释义 |
|---|---|---|
| 可食用 | `a` | 可以供人食用，无毒无害 |
| 有营养 | `a` | 含有生物所需的养分 |

两点说明：

1. **`有营养来源` → `有营养`**。"来源"是释义成分而不是属性名本身（"食物**是**营养的来源"
   是命题，不是属性）。若需严格保留字面，只改本文件 §1.5 的 `word` 值即可，其余无需改动。
2. **这与 `seed_general_v3.sql` §10 的相反原则并不矛盾**。seed 用大段注释说明
   "论元需求不要用 has_property 承载"（会把"需要施事"实体化成伪名词）。两者分工不同：

   | 语义 | 承载机制 | 理由 |
   |---|---|---|
   | 论元需求（吃需要受事） | `relation_constraints` + `relation_types.agent/patient` | 可索引、可被规则引擎消费，且天然是"对动词的约束"而非"动词拥有的名词" |
   | 事物属性（食物可食用） | `has_property` | 属性本身就是一个可命名的概念，实体化是正当的 |

### 3.4 收益是端到端可见的

`has_property` 的闭包策略是 `incremental`，插边即由 `tr_closure_edge_ai` 派生闭包；
推理规则 #2「属性继承」= 沿 `is_a` 上位再取 `has_property`。于是
`鱼 is_a 食物` + `食物 has_property 可食用` ⇒ **鱼 可食用**。
实测 `tests/test_e2e_cat_eats_fish.py` 的 Phase 5 输出：

```
◇ 属性继承
    鱼 --[has_property]--> 可食用  (conf=0.90)
    鱼 --[has_property]--> 有营养  (conf=0.90)
◇ is_a传递
    鱼 --[is_a]--> 生物   (conf=1.00)
    鱼 --[is_a]--> 营养物 (conf=1.00)
```

这四条在本次扩展**之前完全不存在** —— 新种子不是"多存了几行"，而是真的让解析器
多推出四条事实。

---

## 4. 任务二：抽象概念 / 人造物分类骨架

### 4.1 形状（36 节点 / 34 条 is_a 边）

```
抽象概念(1)                                  人造物(1)
├── 方法(1) ── 算法 / 策略 / 模式              ├── 物理人造物(1) ── 工具 / 设备 / 建筑
├── 属性(1) ── 物理属性 / 抽象属性             └── 信息人造物(1)
├── 关系(1) ── 因果 / 部分整体                     ├── 数据对象(1) ── 数据库/表/视图/索引/查询
└── 规则(1) ── 语法规则 / 推理规则                 ├── 信息载体(1) ── 文档/书/网页/文件
                                                   └── 软件产物(1) ── 程序/库/API/模型
```

实测：抽象概念下 4 个直接子节点（B1）、人造物下 2 个（B2）、信息人造物下 3 个（B3）；
骨架共 **36 个节点**（B7，≥25）、**34 条 is_a 边**（B8，≥24）。

> ⚠ **口径提醒（后加入的医学库 / 症状补丁会改变全量计数）**
> 上表的 36/34 是 **B7/B8 的统计口径**，即「schema + seed + ext」三份加载后的状态。
> 之后新增的 `seed_medical_v3.sql` 把「医学概念」挂在**抽象概念**下（这是设计内的扩展），
> 于是**四份** SQL 全量加载时（实测）：
> * 抽象概念子树（含自身）**39 节点 / 39 条 is_a 边**，直接子节点从 4 个增至 5 个
>   （方法 / 属性 / 关系 / 规则 / **医学概念**）；
> * 人造物子树 **22 节点 / 21 条边**；
> * 两子树合计（即 `apply_seeds.py` skel 口径）**61 节点 / 60 条边**。
> 再叠加症状继承补丁 `seed_medical_v3_patch_symptom.sql`（**第五份**）后：
> 抽象概念子树增至 **45 节点 / 45 条边**，合计 **67 节点 / 66 条边**。
> 两处计数都对，只是**口径不同**：
> * `validate_seed.py` §7 只加载 schema+seed+ext → 恒为 36/34（B1 的「≥4」在两种口径下都成立）；
> * `apply_seeds.py` 自检已按 `sense_id < 4000` 过滤，报「ext 时代节点 36（全量 67）」；
> * 医学分支由 `validate_medical.py` §M4 单独断言（医学概念 的 is_a 下位 = 30
>   = 主文件骨架 24 + 症状补丁 6）。
> 三者互不矛盾，读数时先看口径。症状继承的完整设计与实测见
> `docs/MEDICAL_LIB_DESIGN.md`。

### 4.2 为什么骨架落在**义项层**而不是 `semantic_categories`

schema §1 注释已明示：类别层（`semantic_categories`，供量词默认值与约束取值共用）
与义项层的 `is_a` 关系是**两套正交体系**。骨架要能被 `is_a` 遍历、被 closure 物化、
被 `v_closure_2` 与推理规则消费，就**必须**落在义项层。

因此本文件对类别层只做**映射**（不新建类别）：

| 骨架分支 | 映射到的已有类别 |
|---|---|
| 抽象概念及其 13 个下位 | `抽象` |
| 人造物及其 17 个下位 | `人工物` |
| 工具 / 设备 / 建筑 | `器物` |

### 4.3 为什么「信息人造物」要三分支

`数据对象 / 信息载体 / 软件产物` 的分野对应**三种被使用的方式**：

* 数据对象 —— 被**查询与操作**（数据库、表、视图、索引、查询）
* 信息载体 —— 被**阅读与传播**（文档、书、网页、文件）
* 软件产物 —— 被**执行与复用**（程序、库、API、模型）

这三个动词（查询 / 阅读 / 执行）在中文里不共享受事类别，所以不能合并成一个分支。
而三条祖先链正是 B4/B5/B6 要验证的：视图 3 跳、文档 2 跳、程序 2 跳。

### 4.4 为什么复用「书」而不新建

schema §14 已有 `书`（word 20 / sense 21，类别 `人工物`）。骨架里它天然是
「文档」的同位概念（`信息载体 → 书`）—— 所以本文件**只补一条 is_a 边**，不建重复词条。
`validate_seed.py` 的 B7 统计骨架节点数时会把它算进去（35 新建 + 1 复用 = 36），
`+countable` / `+concrete` 两个义素也因 `ON CONFLICT DO NOTHING` 自动跳过。

---

## 5. 写入规模（实测）

`schema + seed → +ext` 的**增量**：

| 表 | 增量 | 说明 |
|---|---|---|
| `semantic_categories` | **+0** | 「食物」已由 seed §1 建好 → 本节是 DO NOTHING 空操作（纯 schema 流程才会 +1） |
| `words` | **+39** | 营养物 1 + 可食用/有营养 2 + 骨架 35 + 量词「份」1 |
| `senses` | **+39** | 与 `words` 1:1（v0.1 不做多义） |
| `sense_features` | **+42** | `+countable` 35（骨架全部）+ `+concrete` 7（实体载体） |
| `sense_categories` | **+38** | 营养物 1 + 属性节点 2 + 抽象 14 / 人工物 18 / 器物 3 |
| `classifiers` | **+1** | 份（名量词 / 部分） |
| `noun_classifier_pairs` | **+4** | 视图→个 / 文档→份 / 程序→个 / 文件→份 |
| `relation_constraints` | **+6** | 算法 / 文档 / 程序 各 2（`吃.patient=食物` 已由 seed §10 建好 → DO NOTHING） |
| `relations` | **+38** | 34 `is_a`（骨架）+ 1 `is_a`（食物→营养物）+ 2 `has_property` + 1 `domain_link` |
| `entity_nodes` | **+1** | 动物学库「Fish」条目 |
| `closure` | **+79** | 全部**自动派生**（本文件不含任何 `closure` 的 INSERT） |
| `changelog` | **+38** | 由 `tr_relations_audit_ai` 自动记录 |

最终态：`words` 53→92、`senses` 54→93、`relations` 23→61、`closure` 16→95。

**为什么要登记 `closure +79`**：它不是手写的，而是 38 条边的副作用。把它写进文档是为了
让"闭包行数与边数的比例"可被审计 —— 36 节点的树形 DAG 平均每节点 ~2.2 行闭包，
这个量级在千节点级会变成噪声，届时应把 `is_a` 切到 `batch` 模式（见 §9）。

---

## 6. 幂等的三种写法与为什么必须这样

本文件的所有 `INSERT` 都落在下面三种形态之一：

```sql
-- ① 纯新增、可能重复执行 → ON CONFLICT DO NOTHING
INSERT INTO words(...) VALUES (2004,'可食用',...) ON CONFLICT DO NOTHING;

-- ② 需要"若不存在才建"（兜底路径）→ SELECT + WHERE NOT EXISTS
INSERT INTO words(...) SELECT 2001,'营养物',...
 WHERE NOT EXISTS (SELECT 1 FROM words WHERE word='营养物');

-- ③ INSERT ... SELECT 且带 ON CONFLICT → 必须写 WHERE，哪怕恒真
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT s.sid, e.cat, e.is_primary FROM e JOIN ext_sid s ON ...
 WHERE 1                        -- ← 不是废话：消除 INSERT...SELECT...ON CONFLICT 的解析歧义
ON CONFLICT DO NOTHING;
```

③ 里的 `WHERE 1` 是 `seed_general_v3.sql` §9 记录过的真实坑：`INSERT ... SELECT ...`
紧接 `ON CONFLICT` 时 SQLite 会把它解析成 `SELECT ... ON` 连接语法而报错；
补一个 `WHERE` 子句即可消除歧义（此处它还兼做过滤）。

**幂等性是被实测的、不是被声称的**：`validate_seed.py` §1b 会**连续执行本文件两次**，
比对 12 张相关表的行数，出现任何漂移即致命退出（不计入断言编号，见 §7）。

---

## 7. 验证覆盖（77 项，全部为行为测试）

```
§1  加载与幂等        9 项   （原样保留，含落盘重开验证）
§1b 扩展种子          0 项   [ok] 非计数：加载 + 二次执行零漂移 + 零 closure INSERT + 事务配对
§2  结构覆盖         12 项
§3  约束完整性       11 项
§4  闭包正确性       10 项   （含"闭包 == 在线递归 CTE"差分对拍，现在覆盖 40 条 is_a 边）
§5  句式覆盖         13 项
§6  端到端语义校验   10 项
§7  扩展种子断言     12 项   ← 本次新增：A1-A4（多继承）+ B1-B8（分类骨架），编号 66-77
──────────────────────────────
合计 PASS 77 / FAIL 0
```

### 需求明确要求、但**未**列入 A/B 编号的数据（以 `[ok]` 透明报告）

| 需求 | 实测 |
|---|---|
| 任务一 3：食物 has_property 可食用 / 有营养 | `['可食用','有营养']`，且被「鱼」经 is_a 继承 |
| 任务一 5：cross_links | 1 条 `sense 鱼 --domain_link--> node Fish` |
| 任务二 3：视图/文档/程序 量词搭配 | 3 条（视图→个、文档→份、程序→个） |
| 任务二 4：算法/文档/程序 relation_constraints | 6 条 |

**为什么是 `[ok]` 而不是第 78-81 项**：需求把验收清单显式写成"原有 65 + 新增 12 = 77"，
擅自把额外检查塞进编号会让这个数字失真。但"不编号"不等于"不检查" ——
它们仍然每次运行都打印实测值，只是不参与 PASS/FAIL 计数。若希望把它们升级为
正式断言，把它们从 §7 末尾的打印块移入 `check()` 即可，总数变为 81。

### 值得单说的三项

**B4/B5/B6 用闭包而不是直接边。** 三条祖先链都要求**多级可达**
（视图 3 跳、文档/程序 2 跳）。如果只用 `relations` 查直接边，把
`视图 is_a 人造物` 直接写上也能通过 —— 那就把"链"验成了"边"。所以断言走
`closure`，并复用 §4 的差分对拍保证闭包本身可信。

**B8 的两端都必须落在骨架内。** `src_id IN (骨架) AND dst_id IN (骨架)` 双端限定，
避免把骨架节点指向外部（如 `书 is_a 人工物类别`）的边也算进来。

**A3 复用 §6 的判定器。** A3 不是"表里有那一行就算过"，而是调用 §6 定义的 `rc_allows()`，
即**真的按推理规则 #3/#4 的逻辑**判定「鱼」是否通过 `吃.patient`。这样"声明了约束"
与"约束真的能用"被分开检查 —— 前者查表，后者走一遍规则。

---

## 8. 一个必须知道的联动：`init_database` 自动发现 `seed_*.sql`

`logos/db/connection.py`：

```python
SEED_GLOB: str = "seed_*.sql"
def _resolve_seed_paths(...): return sorted(search_root.glob(SEED_GLOB))
```

两个后果：

1. **本文件不必注册**，只要命名为 `seed_*.sql` 放在项目根，所有走 `init_database`
   的管线（`tests/test_e2e_cat_eats_fish.py`、`validate_syntax_parser.py` 等）都会自动加载它。
2. **依赖顺序靠文件名的字典序**：`seed_general_v3.sql` < `seed_general_v3_ext.sql`
   （`'.'` U+002E < `'_'` U+005F），恰好是正确顺序。**扩名要落在基础种子之后**，
   否则 ext 会在 base 之前执行，兜底路径虽能跑通但语义会与预期不同。

**本次因此触发了一次真实回归并已修复**：e2e 测试的夹具使用了 2001/2002 作为
「老鼠/石头」的 ID，与本文件新增的 2001-2005 段相撞 →
`sqlite3.IntegrityError: UNIQUE constraint failed: words.word_id`。
修法是把夹具迁到 **7000 段**，并在夹具注释里写明各段位归属。
教训：**新增种子会改变所有自动加载它的测试的 ID 空间** —— 所以本文件把段位登记
写进了 §2 表格，`tests/test_e2e_cat_eats_fish.py` 的 `FIXTURE_SQL` 也同步登记。

---

## 9. 后续扩展建议

1. **给两条 is_a 路径加维度标注。** `relations` 已有 `weight` / `confidence` / `context_id`
   三个字段，目前 `is_a` 边全部为默认值。建议：生物分类路径挂 `context_id=4`（动物学），
   功能路径留空或挂新的「饮食」域 —— 这样 `鱼` 的多归属在查询期可被区分，而不只是并列。
2. **加厚 `has_property` 链。** 当前只有「食物」有 2 条属性边。生物/动物/人工物/器物
   都该有属性（动物 `has_property` 能运动 / 需要呼吸，工具 `has_property` 可握持…），
   属性继承推理才真正有东西可继承。
3. **补 `has_part`：`数据库 has_part 表`。** 现在 表/视图/索引 都挂 `数据对象`（is_a 关系成立），
   但「数据库由表构成」是 `has_part` 而非 `is_a`。`relation_types.has_part` 已有定义、
   `closure_policy` 已配 `incremental`，只差数据。这是骨架从"分类树"走向"结构图"的关键一步。
4. **量词层补齐。** 「份」的 `subtype` 现取 `部分`；建议为 `数据对象` 补一行
   `classifier_class_defaults(cat='人工物' …)`，让未显式搭配的信息类名词也有默认量词。
5. **`API` 的读音。** 该词无汉字读音，`pinyin` 留 `NULL`，只填了 `pinyin_plain` /
   `pinyin_initial`（均为 `api`）。若要做全库拼音检索，需要补一套罗马化约定
   （是把 `API` 记为 `ei1pi1ai2` 还是直接按字母串，属于词典政策而非 schema 问题）。
6. **闭包规模预警。** 36 节点 34 边已产出 79 行闭包。`is_a` 的
   `closure_policy.max_depth = 12`，深链会让行数按 O(N×D) 增长。骨架扩到千节点级时，
   建议把 `is_a` 切成 `batch` 模式 + 用 `closure_rebuild_queue` 全量重建
   （`schema_v3.sql` §13.5 已给出停用/重建的标准流程）。
7. **类别层与义项层的"接口"目前靠人肉对照。** §4.2 的映射表只写在文档里。
   若骨架继续长大，建议加一张显式映射表（如 `sense_categories` 已有，只是它的语义是
   "成员关系"而非"骨架投影"）或在 `semantic_categories` 里补 `抽象概念` / `人造物`
   两个与义项层同名的类别，让两层可被程序化对齐。

---

## 10. 全量回归

```
validate_schema.py          PASS 102 / FAIL 0
validate_seed.py            PASS  77 / FAIL 0     ← 原 65 → 77
validate_connection.py      PASS 159 / FAIL 0
validate_closure.py         PASS  87 / FAIL 0
validate_syntax_parser.py   PASS 214 / FAIL 0
tests/test_e2e_cat_eats_fish.py   PASS 64 / FAIL 0 / XFAIL 4
apply_seeds.py              自检 15 项 全绿，exit 0
────────────────────────────────────────
合计 703 项断言，FAIL 0
```

新增种子**没有**改动任何 schema 对象，因此 `validate_schema.py` / `precheck_schema.py`
不受影响；`validate_closure.py` 与 `validate_syntax_parser.py` 自建库时只加载
`schema_v3.sql`（不自动发现种子），故数字与扩展前一致。

`validate_seed.py` 的 §4「闭包 == 在线递归 CTE」差分对拍现在覆盖 40 条 `is_a` 边
（原 5 条），是本次扩展中最有分量的回归保证 —— 它用**独立实现的在线递归**当物化闭包的
oracle，而不是"拿闭包验闭包"。
