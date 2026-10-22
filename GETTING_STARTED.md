# Logos 使用说明书

> 面向**克隆本仓库的人**：从零跑起来、看懂数据、查得动、改得对。
>
> 阅读顺序建议：**§1 跑起来 → §4 看懂数据 → §6 查询配方**，其余按需跳读。

---

## 目录

- [1. 五分钟跑起来](#1-五分钟跑起来)
- [2. 环境要求](#2-环境要求)
- [3. 目录导览](#3-目录导览)
- [4. 核心概念](#4-核心概念)
- [5. 数据规模](#5-数据规模)
- [6. 查询配方（可直接复制）](#6-查询配方可直接复制)
- [7. Python API](#7-python-api)
- [8. 校验体系](#8-校验体系)
- [9. 如何扩展（新增词 / 义项 / 关系）](#9-如何扩展新增词--义项--关系)
- [10. 常见坑（TOP 8）](#10-常见坑top-8)
- [11. 文档地图](#11-文档地图)
- [12. 许可](#12-许可)

---

## 1. 五分钟跑起来

```bash
git clone <你的仓库地址>
cd 数据库logos

# ① 建库：把 8 份 SQL 铺进 logos.db，并做一次结构性自检
python apply_seeds.py

# ② 校验：医学库全量断言（含 schema + 全部种子）
python validate_medical.py
```

两条命令的预期结果：

- `apply_seeds.py` 结尾打印 **`结果: 全部自检通过（FAIL 0）`**，并在项目根生成 `logos.db`。
- `validate_medical.py` 结尾打印 **`结果: PASS 230 / FAIL 0`**。

> **建库产物不入版本控制**（`.gitignore` 里 `*.db*`）。克隆后 `logos.db` 不存在是正常的，
> 跑一次 `apply_seeds.py` 即可重建 —— 数据库是**纯派生物**，源码才是唯一真相。

想一次跑完全部校验：

```bash
python validate_schema.py        && python validate_seed.py     && \
python validate_connection.py    && python validate_closure.py  && \
python validate_syntax_parser.py && python validate_medical.py  && \
python tests/test_e2e_cat_eats_fish.py
```

---

## 2. 环境要求

| 项目 | 要求 | 本仓库验证环境 |
|---|---|---|
| Python | **3.10+**（推荐 3.11+） | CPython **3.13.14** |
| SQLite | **≥ 3.34.0**（FTS5 需 `trigram` 分词器） | **3.53.1** |
| 第三方库 | **无。零依赖。** | — |
| 操作系统 | Windows / macOS / Linux 均可 | Windows 11 |

**没有 `requirements.txt`，也不需要有。** 全部 import 都是标准库
（`sqlite3` / `pathlib` / `dataclasses` / `re` / `json` …），`pip install` 这一步可以整个跳过。

`sqlite3` 是 Python 自带的，但**版本随 Python 发行版走**。若你的 SQLite < 3.34，
`CREATE VIRTUAL TABLE ... USING fts5(..., tokenize='trigram')` 会失败。
先自检环境，一条命令看清所有关键信息：

```bash
python -m logos.db.connection
```

输出示例（节选）：

```
python        : 3.13.14
sqlite        : 3.53.1
threadsafety  : 3
项目根        : .../数据库logos
schema 存在   : True
```

> 开头的 `RuntimeWarning: 'logos.db.connection' found in sys.modules ...` 是
> `runpy` 的正常提示，**不影响结果**。

---

## 3. 目录导览

```
数据库logos/
├── README.md                     # 项目门面
├── GETTING_STARTED.md            # ★ 本文件
│
├── schema_v3.sql                 # 表结构：52 STRICT 表 / 5 视图 / 10 触发器 / 34 索引
│
├── seed_general_v3.sql           # 通用领域种子
├── seed_general_v3_ext.sql       # 通用扩展（营养物 / 分类骨架 / 量词）
├── seed_medical_v3.sql           # 医学骨架 + 跨库领域链接
├── seed_medical_v3_imaging.sql   # 影像特征子树（形态/大小/密度/强化/位置 + 结节）
├── seed_medical_v3_patch_symptom.sql           # 症状继承（通用症状下沉到层）
├── seed_medical_v3_patch_symptom_layers.sql    # 疾病分层（炎症性疾病层）
├── seed_medical_v3_patch_symptom_severity.sql  # 症状严重度义素（纯标注）
│
├── apply_seeds.py                # ★ 建库 + 结构性自检
├── validate_schema.py            # ↓ 六个校验脚本（见 §8）
├── validate_seed.py
├── validate_connection.py
├── validate_closure.py
├── validate_syntax_parser.py
├── validate_medical.py           #   医学库专项（§M1–§M13）
│
├── logos/                        # 库代码
│   ├── db/
│   │   ├── connection.py         # 连接层 / init_database / 种子自动发现
│   │   └── closure.py            # 闭包读写 + 不变量巡检
│   └── parser/
│       └── syntax_parser.py      # 词典驱动的中文句法解析
│
├── tests/
│   └── test_e2e_cat_eats_fish.py # 端到端（走 init_database，加载全部种子）
│
└── docs/                         # 每层一份设计文档（见 §11）
```

> `audit_indexes.py` / `bench_closure.py` / `precheck_schema.py` / `probe_gaps.py` /
> `schema_review.md` 是开发期的**诊断工装**，不影响建库与校验，可以无视。

---

## 4. 核心概念

理解下面 6 个词，这个库就能读了。

### 4.1 五层词汇模型

```
characters（字） → morphemes（语素） → words（词） → senses（义项） → 关系/义素
```

**关系与义素挂在「义项」上，不是挂在「词」上。** 这是全库最重要的一条约定：
一个词可以有多个义项（如「把」既是量词又是介词），语义标注落在义项粒度。

- `words.word` 是**全局唯一**（`COLLATE BINARY`，**不带 pos**）—— 同一个字符串不能
  既是名词词条又是形容词词条。想区分词性，用**不同的词形**。
- `senses` 用 `(word_id, pos, sense_no)` 定位一个义项。

### 4.2 关系（relations）

一条关系 = `(src_kind, src_id, relation, dst_kind, dst_id)`，端点可以是 `sense`
也可以是 `node`（跨库节点）。常见关系：

| 关系 | 含义 | 方向 | 传递？ |
|---|---|---|---|
| `is_a` | 上下位 | 下位 → 上位 | ✅ 产生闭包 |
| `has_part` | 部分整体 | 整体 → 部分 | ✅ 产生闭包 |
| `has_property` | 属性 | 主体 → 属性 | ✅ 产生闭包 |
| `has_symptom` | 疾病 → 症状 | — | ❌ 不产生闭包 |
| `typically_has` | 类别 → 典型特征（统计倾向） | — | ❌ 不产生闭包 |

全部关系类型见 `relation_types`（当前 **45** 种），每种都带
`relation_class / direction / inverse_of / is_transitive / max_depth` 等元数据。

### 4.3 闭包（closure）

传递闭包由**触发器** `tr_closure_edge_ai` 在关系边插入时**自动物化** —— 你不需要手写闭包行。

物化条件（缺一不可）：

1. `src_kind='sense'` **且** `dst_kind='sense'`
2. 关系的 `is_transitive = 1`
3. `closure_policy` 里该关系 `mode='incremental'`

> ⚠️ **方向陷阱（不变量 I3）**：闭包行恒为**边的反向** —— `closure.ancestor = relations.dst_id`。
>
> - 对 `is_a`（下位→上位）：`ancestor` 是**上位**，符合直觉。
> - 对 `has_part`（整体→部分）：`dst` 是**部分**，所以 `ancestor` 落在**部分**一侧！
>   要查「整体 X 有哪些部分」，应查 `WHERE descendant = X`，再取 `ancestor`。

### 4.4 义素（sense_features / semantic_features）

义素是**义项的固有属性**，不是边的可信度 —— 两者正交，别混用：

| 维度 | 载体 | 语义 |
|---|---|---|
| 义项有多「尖」 | `sense_features` | `+severe` / `+圆形` / `+毛刺` … |
| 边有多可信 | `relations.confidence` | 0.0 – 1.0 |

### 4.5 跨库节点（entity_nodes）

`node` 类型的端点必须先在 `entity_nodes` 注册（触发器会校验）。
跨库链接不是独立的表，而是**一个视图**：

```sql
SELECT * FROM cross_links;   -- = relations 里任一端点为 node 的行
```

### 4.6 种子自动发现

`logos/db/connection.py` 用 `SEED_GLOB = "seed_*.sql"` **按文件名字典序**加载。

**文件名顺序 = 依赖顺序。** 这不是巧合，是设计（详见 §9.2）。

---

## 5. 数据规模

跑完 `apply_seeds.py` 后的实测值（本仓库基线）：

| 表 | 行数 | 说明 |
|---|---:|---|
| `words` | 169 | 词条 |
| `senses` | 170 | 义项 |
| `relations` | 166 | 语义关系（含跨库边） |
| `closure` | 371 | 闭包（全部由触发器派生） |
| `entity_nodes` | 5 | 跨库节点（动物学 1 + 医学 4） |
| `relation_types` | 45 | 关系类型 |
| `sense_features` | 98 | 义素标注 |
| `semantic_features` | 22 | 义素定义 |

表结构：**52 张表 / 5 个视图 / 10 个触发器 / 34 个索引**。

---

## 6. 查询配方（可直接复制）

建好库后，用任意 SQLite 客户端打开 `logos.db`，或走 Python（见 §7）。以下每条都已实测。

### 6.1 词 → 义项

```sql
SELECT w.word, s.pos, s.sense_no, s.definition
  FROM words w JOIN senses s ON s.word_id = w.word_id
 WHERE w.word = '结节';
```

### 6.2 上位链（`is_a` 闭包）

```sql
SELECT w.word, cl.depth
  FROM closure cl
  JOIN senses s ON s.sense_id = cl.ancestor
  JOIN words  w ON w.word_id = s.word_id
 WHERE cl.relation = 'is_a'
   AND cl.descendant = (SELECT s2.sense_id FROM senses s2
                          JOIN words w2 ON w2.word_id = s2.word_id
                         WHERE w2.word = '恶性结节')
 ORDER BY cl.depth;
```

结果：`结节(1) → 病理结构(2) → 医学概念(3) → 抽象概念(4)`。

### 6.3 典型特征（正向：类别 → 特征）

```sql
SELECT wd.word, r.confidence
  FROM relations r
  JOIN senses s ON s.sense_id = r.dst_id
  JOIN words  wd ON wd.word_id = s.word_id
 WHERE r.relation = 'typically_has' AND r.src_kind = 'sense'
   AND r.src_id = (SELECT s2.sense_id FROM senses s2
                     JOIN words w2 ON w2.word_id = s2.word_id
                    WHERE w2.word = '恶性结节');
```

结果：`不规则形 / 毛刺状 / 高密度`，confidence 均 `0.8`。

### 6.4 反向：由特征推类别

```sql
SELECT wd.word, r.confidence
  FROM relations r
  JOIN senses s ON s.sense_id = r.src_id
  JOIN words  wd ON wd.word_id = s.word_id
  JOIN senses s3 ON s3.sense_id = r.dst_id
  JOIN words  w3 ON w3.word_id = s3.word_id
 WHERE r.relation = 'typically_has' AND w3.word = '高密度';
```

### 6.5 症状继承（层边 ∪ 直接边）

`has_symptom` 不产生闭包，继承发生在 **`is_a`** 维度上：

```sql
SELECT DISTINCT wd.word
  FROM relations r
  JOIN senses s ON s.sense_id = r.dst_id
  JOIN words  wd ON wd.word_id = s.word_id
 WHERE r.relation = 'has_symptom' AND r.src_kind = 'sense'
   AND ( r.src_id = (SELECT s2.sense_id FROM senses s2
                       JOIN words w2 ON w2.word_id = s2.word_id
                      WHERE w2.word = '癫痫')
         OR r.src_id IN (SELECT ancestor FROM closure
                          WHERE relation = 'is_a'
                            AND descendant = (SELECT s3.sense_id FROM senses s3
                                                JOIN words w3 ON w3.word_id = s3.word_id
                                               WHERE w3.word = '癫痫')) );
```

结果 8 种：`疼痛 乏力 头痛 眩晕 意识模糊 抽搐 记忆障碍 发热`。

> **层边 vs 直接边**：通用症状声明在抽象层（疾病层 / 脑部疾病层 / 炎症性疾病层），
> 只在特定疾病成立的症状走直接边。癫痫的 8 种 = 层边继承 5 + 直接边 3。

### 6.6 叠加严重度

```sql
-- 在 6.5 的基础上加：
LEFT JOIN sense_features sf
       ON sf.sense_id = r.dst_id
      AND sf.feature IN ('+severe','+moderate','+mild')
-- … WHERE … AND sf.feature IS NOT NULL
--   SELECT DISTINCT wd.word, sf.feature
```

`+severe` 疼痛/头痛/抽搐/意识模糊 · `+moderate` 眩晕/乏力/记忆障碍 · `+mild` 发热。

### 6.7 跨库链接

```sql
SELECT * FROM cross_links;
```

### 6.8 一致性巡检（改完数据必跑）

```sql
SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor = descendant;  -- 期望 0（自环）
SELECT COUNT(*) FROM (SELECT relation,ancestor,descendant FROM closure
                       GROUP BY 1,2,3 HAVING COUNT(*) > 1);                     -- 期望 0（重复）
PRAGMA foreign_key_check;                                                       -- 期望空
PRAGMA integrity_check;                                                         -- 期望 ok
```

---

## 7. Python API

```python
from logos.db import init_database, connect

# 建库（等价于 python apply_seeds.py），返回 InitReport
report = init_database("logos.db")
print(report.tables, report.rows.get("senses"))

# 只读连接（with 退出时自动关闭）
with connect("logos.db", read_only=True) as conn:
    n = conn.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    print("integrity:", conn.integrity_ok())
```

句法解析（词典驱动，不接模型）：

```python
from logos.parser import build_parser

parser, conn = build_parser("logos.db")     # 返回 (解析器, 连接)
print(parser.parse("猫吃鱼。").pattern_name)  # -> 主谓宾
conn.close()
```

`logos.parser` 的完整导出见 `logos/parser/__init__.py` 的 `__all__`
（`SyntaxParser` / `parse` / `tokenize` / `build_parser` / `explain` …）。

---

## 8. 校验体系

这个项目**没有单元测试框架**，取而代之的是 6 个「断言脚本」+ 1 个 e2e + 1 个建库自检。

| 脚本 | 断言数 | 作用范围 |
|---|---:|---|
| `validate_schema.py` | 102 | 只加载 schema：表 / 约束 / 触发器 / 索引 |
| `validate_seed.py` | 77 | schema + 通用种子 |
| `validate_connection.py` | 159 | 连接层 / PRAGMA / 事务 / 缓存 |
| `validate_closure.py` | 87 | 闭包不变量（只加载 schema） |
| `validate_syntax_parser.py` | 214 | 句法解析器 |
| `validate_medical.py` | 230 | 医学库专项（§M1–§M13） |
| `tests/test_e2e_cat_eats_fish.py` | 68（PASS 64 + XFAIL 4） | 端到端，走 `init_database` |
| `apply_seeds.py` 自检 | 39 | 建库后结构自检 |
| **合计** | **976** | **FAIL 0** |

约定：

- **断言编号是需求给定项，不擅自增删。** 额外检查以 `[ok]` 非计数形式透明报告。
- **已知缺陷用 `XFAIL` 档记录** —— 这样「FAIL=0」与「诚实记录缺陷」可以并存。
- 全部脚本 **exit code = 0 表示通过**，可直接接 CI。

```bash
# 单跑
python validate_medical.py

# 只看结论行
python validate_closure.py | tail -3
```

---

## 9. 如何扩展（新增词 / 义项 / 关系）

> ⚠️ **改数据前先读这一节。** 这里的每条规则都是踩过坑总结出来的。

### 9.1 先领一个空 ID 段

ID 段是**手工登记**的，不自动分配。当前占用：

| 段位 | 归属 |
|---|---|
| 1–27 | schema 内置示例（猫/鱼/动物/把/只…） |
| 1000–1027 | 通用领域种子 |
| 2001–2005, 2101–2135, 2201 | 通用扩展 |
| 3001, 3101–3104 | `entity_nodes` 跨库节点 |
| 4001–4030, 4101 | 医学骨架 |
| 4031–4036 | 通用症状 |
| 4037–4039 | 疾病分层 |
| 4040–4076 | 影像特征子树 |
| 7000+ | 测试夹具 |

**新增时必须自选一个空段**，登记到本表 + 对应 SQL 文件头 + 测试夹具注释。

### 9.2 文件名必须让字典序落在依赖之后

`init_database` 按 `sorted(glob("seed_*.sql"))` 加载，**顺序就是依赖顺序**。

排序看的是**整个文件名**，不只是「扩展名」：

- `.`(U+002E) < `_`(U+005F) ⇒ `seed_medical_v3.sql` 排在 `seed_medical_v3_imaging.sql` 之前。
- `'i'`(0x69) < `'p'`(0x70) ⇒ `..._imaging` 夹在 `..._medical_v3` 与 `..._patch_symptom` 之间。
- ⚠️ **反例**：若把严重度补丁命名为 `..._patch_severity.sql`，`'e'`(0x65) 会排在
  `..._patch_symptom.sql` 的 `'y'`(0x79) **之前** ⇒ 在被依赖的症状义项还没建时就执行 ⇒
  **静默丢 8 行**。所以它叫 `..._patch_symptom_severity.sql`。

**改名前务必实跑一遍确认：**

```bash
python -c "import glob; print('\n'.join(sorted(glob.glob('seed_*.sql'))))"
```

### 9.3 种子 SQL 的三条铁律

1. **幂等**：所有 `INSERT` 必须 `ON CONFLICT DO NOTHING` 或 `SELECT ... WHERE NOT EXISTS` ——
   同一个脚本重复跑不能改变数据库状态。
2. **`INSERT ... SELECT ... ON CONFLICT` 必须带 `WHERE`**（哪怕 `WHERE 1`），
   否则会被解析成 `SELECT ... ON` 连接语法。
3. **`PRAGMA foreign_keys` 必须写在 `BEGIN` 之前**（事务内是 no-op）；
   对既有实体用 `(word, pos)` 自然键子查询，**不要硬编码既有 ID**。

### 9.4 加一个「前置依赖硬检查」

比「靠顺序碰运气」更强的做法：在种子开头建临时表 + **具名 CHECK 约束**，
把修复提示写进**约束名**。失败时报错文本即修复提示：

```sql
CREATE TEMP TABLE IF NOT EXISTS _prereq_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _prereq_missing_run_seed_medical_v3_first_for_imaging CHECK (ok = 1)
);
DELETE FROM _prereq_check;
INSERT INTO _prereq_check(ok)
SELECT CASE WHEN EXISTS(SELECT 1 FROM words WHERE word = '医学概念') THEN 1 ELSE 0 END;
DROP TABLE _prereq_check;
```

### 9.5 四件同步不能忘

1. **新增种子** → 加进 `apply_seeds.py` 的 `SQL_FILES` 显式清单；
2. → 更新其 `EXPECT`（`senses` / `relations` / `closure` / `sense_features` / `semantic_features` 期望值）；
3. → 在 `validate_*.py` 里补对应断言（医学类走 `validate_medical.py` 新开一节）；
4. → 同步 `docs/` 里的实测数字。

> **新增种子会改变下游计数** —— 因为 `init_database` 会自动发现它，
> e2e、解析器测试等都跟着变。这也是测试夹具独占 **7000 段** 的原因（避免撞 ID）。

---

## 10. 常见坑（TOP 8）

| # | 现象 | 原因 | 对策 |
|---|---|---|---|
| 1 | 关系边插了，但 `closure` 里查不到 | 端点不是 `sense`，或关系 `is_transitive=0`，或缺 `closure_policy` 行 | 检查三个物化条件（§4.3） |
| 2 | 查「整体有哪些部分」返回空 | **方向陷阱 I3**：`has_part` 的闭包行方向是反的 | 改查 `WHERE descendant = 整体`（§4.3） |
| 3 | 想建「小」这个词但冲突 | `words.word` 全局唯一且**不带 pos** | 换词形（如「小型」），别试图改 schema |
| 4 | 新种子静默少了几行 | 文件名字典序把依赖排到了前面 | 加 §9.4 的硬检查；实跑 `sorted(glob(...))` |
| 5 | `INSERT ... SELECT ... ON CONFLICT` 语法报错 | 少了 `WHERE` | 补 `WHERE 1` |
| 6 | `cannot start a transaction within a transaction` | 驱动隐式事务撞了 SQL 里的显式 `BEGIN` | `apply_seeds.py` 已用 `isolation_level=None` 处理；自写脚本注意同样问题 |
| 7 | `unable to open database file` | 目录不可写 / 被占用 / 沙箱限制 | 换可写目录；确认没有别的进程占着 `logos.db` |
| 8 | 改了数据但校验还是老数字 | 用的是旧的 `logos.db` | 重跑 `apply_seeds.py`（默认从零重建） |

---

## 11. 文档地图

`docs/` 下每层一份，改哪层看哪份：

| 文档 | 内容 |
|---|---|
| `SEED_GENERAL.md` | 通用领域种子设计 |
| `SEED_EXT_NOTES.md` | 通用扩展（多继承 / 分类骨架 / 量词） |
| `CONNECTION_LAYER.md` | 连接层 / PRAGMA / 事务 / 种子发现 |
| `CLOSURE_LAYER.md` | 闭包层实现 |
| `CLOSURE_DIRECTION.md` | ★ 闭包方向不变量 I3（必读） |
| `SYNTAX_PARSER.md` | 句法解析器五段流水线 |
| `BUILD_PRECHECK.md` | 建库前置检查 / 环境要求 |
| `TEST_AUDIT.md` | 测试与断言审计 |
| `E2E_TEST_REPORT.md` | 端到端报告 |
| `MEDICAL_LIB_DESIGN.md` | 医学库 + 症状继承 + 严重度 + 影像特征子树 |

---

## 12. 许可

[MIT](LICENSE) © 2026 Logos Contributors

---

> **设计哲学**：用中文词定义中文词，用 `is_a` / 闭包 / 约束表代替向量相似度。
>
> 语义是一张**可查询的图**，不是一个**不可解释的向量**。每一条关系都被 SQL 断言钉住 ——
> 任何一处语义错误，都应当能被一条查询捕获。
