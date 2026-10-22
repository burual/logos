# Logos — 中文符号语义 NLP 系统

> **用中文词定义中文词，用 `is_a` / 闭包 / 约束表代替向量相似度。**
>
> 语义正确性不靠模型评测，而由 **SQL 断言**守护 —— 每一条语义关系都可被查询、被验证、被推翻。

📖 **第一次接触本项目？请先读 [使用说明书（GETTING_STARTED.md）](GETTING_STARTED.md)** ——
从克隆到跑通、从看懂数据到改数据，全部命令都实测过。

---

## 核心理念

- **用中文词语定义词语** —— 释义本身也是词典中的词，不引入外部向量空间。
- **用数据库关系替代向量** —— 上下位（`is_a`）、部分整体（`has_part`）、属性（`has_property`）
  等显式关系构成可解释的语义骨架。
- **语义正确性由 SQL 断言守护** —— 传递闭包、约束表与校验脚本共同构成"可证伪"的语义层。

## 架构概述

```
┌──────────────────────────────────────────────────────────────┐
│  schema_v3.sql                                               │
│  SQLite STRICT 表（52 张）+ 10 触发器 + 34 索引 + 5 视图      │
│                                                              │
│   words ──< senses ──< relations >── senses                  │
│                │            │                                │
│                │            └──► tr_closure_edge_ai          │
│                │                  （触发器维护传递闭包）      │
│                ▼                                             │
│           sense_features        closure（物化闭包表）         │
│                                      │                       │
│                                      ▼                       │
│                             logos/db/closure.py              │
│                             logos/parser/syntax_parser.py    │
└──────────────────────────────────────────────────────────────┘
```

- **SQLite `STRICT` 表** —— 列类型强校验，杜绝隐式类型漂移。
- **触发器维护传递闭包** —— 关系边一经插入，闭包行由 `tr_closure_edge_ai` 自动派生；
  `is_transitive=1` 且 `closure_policy.mode='incremental'` 的关系才会物化。
- **方向不变量 I3** —— `closure.ancestor` 恒等于边的 `dst`（见 `docs/CLOSURE_DIRECTION.md`）。
- **自动发现种子** —— `init_database` 以 `SEED_GLOB = "seed_*.sql"` 按文件名排序加载，
  顺序即依赖顺序。

## 快速开始

```bash
# 1) 从零建库 + 结构性自检（exit 0 表示全绿）
python apply_seeds.py

# 2) 往既有库增量叠加种子（跳过 schema，只重放 seed_*.sql；目标库须已存在且 schema 兼容）
python apply_seeds.py logos.db --keep

# 3) 医学库全量校验（schema + 全部种子 + 医学子树）
python validate_medical.py
```

> `--keep` 只补数据、不动结构：种子文件全部幂等（`ON CONFLICT DO NOTHING` / `WHERE NOT EXISTS`），
> 可反复执行；但若改动过 `schema_v3.sql`，仍需用默认模式（不加 `--keep`）从零重建。

建库产物（`logos.db` 等）不入版本控制 —— `clone` 后跑一次 `apply_seeds.py` 即可重建。

## 验证状态

| 脚本 | 断言数 | 状态 |
|------|--------|------|
| validate_schema | 102 | ✅ |
| validate_seed | 77 | ✅ |
| validate_connection | 159 | ✅ |
| validate_closure | 87 | ✅ |
| validate_syntax_parser | 214 | ✅ |
| validate_medical | 230 | ✅ |
| e2e | 64 PASS / 4 XFAIL | ✅ |
| apply_seeds 自检 | 39 | ✅ |
| **合计** | **976** | **FAIL 0** |

> 已知偏差以 `XFAIL` 档记录，使「FAIL=0」与「诚实记录缺陷」兼容。

## 项目结构

```
数据库logos/
├── README.md                                  # 项目门面
├── GETTING_STARTED.md                         # ★ 使用说明书（克隆后先读这个）
├── schema_v3.sql                              # STRICT 表结构（52 表 / 10 触发器 / 34 索引 / 5 视图）
│
├── seed_general_v3.sql                        # 通用领域种子
├── seed_general_v3_ext.sql                    # 通用扩展（营养物 / 分类骨架 / 量词）
├── seed_medical_v3.sql                        # 医学骨架
├── seed_medical_v3_imaging.sql                # 影像特征子树（6 维度 / 25 取值 / 12 义素）
├── seed_medical_v3_patch_symptom.sql          # 通用症状补丁
├── seed_medical_v3_patch_symptom_layers.sql   # 疾病分层补丁
├── seed_medical_v3_patch_symptom_severity.sql # 症状严重度义素（纯标注，不新增词）
│
├── apply_seeds.py                             # 建库 + 结构自检（--keep 增量叠加）
├── validate_schema.py                         # 表结构 / 约束 / 触发器
├── validate_seed.py                           # 通用种子
├── validate_connection.py                     # 连接层 / PRAGMA / 事务
├── validate_closure.py                        # 传递闭包不变量
├── validate_syntax_parser.py                  # 句法解析器
├── validate_medical.py                        # 医学库（§M1–§M13）
│
├── logos/
│   ├── db/
│   │   ├── connection.py                      # init_database / get_connection / 种子自动发现
│   │   └── closure.py                         # 闭包读写与不变量巡检
│   └── parser/
│       └── syntax_parser.py                   # 词典驱动的中文句法解析
│
├── tests/
│   └── test_e2e_cat_eats_fish.py              # 端到端（走 init_database，加载全部种子）
│
└── docs/                                      # 每层一份设计文档
    ├── BUILD_PRECHECK.md
    ├── CLOSURE_DIRECTION.md
    ├── CLOSURE_LAYER.md
    ├── CONNECTION_LAYER.md
    ├── E2E_TEST_REPORT.md
    ├── MEDICAL_LIB_DESIGN.md                  # 医学库 + 症状继承 + 严重度 + 影像特征子树
    ├── SEED_EXT_NOTES.md
    ├── SEED_GENERAL.md
    ├── SYNTAX_PARSER.md
    └── TEST_AUDIT.md
```

## 领域库状态

| 领域库 | 状态 | 内容 |
|--------|------|------|
| 通用库 | ✅ | 基础词表 + 扩展分类骨架 + 量词 |
| 医学库 | ✅ | 医学骨架 + **症状继承**（沿 `is_a` 闭包）+ **症状严重度**（`sense_features`）+ **影像特征子树** |
| 动物库 | 🚧 | 规划中（当前仅有跨库链接节点） |

### 两条推理链的现状

- **症状继承（✅ 已实现）** —— 疾病的症状 = 直接声明 ∪ 沿 `is_a` 继承（`has_symptom` 不产生闭包，
  继承发生在 `is_a` 维度）。如 `癫痫` ⇒ 8 种症状 = 层边继承 5 + 直接边 3。
- **影像特征映射（🚧 设计完成，实现待补）** —— 设计文档与**符号词表**已就位：
  `seed_medical_v3_imaging.sql`（37 个义项 / 12 个义素 / 6 条 `typically_has`），
  符号侧的查询可用，如 `恶性结节` 典型特征 = {不规则形, 毛刺状, 高密度}（`typically_has`，confidence 0.8）。
  **CNN 检测（Layer 1）与 ML 映射层（Layer 2，SVM/RF → 符号特征）尚未实现** ——
  仓库内没有 `imaging/` 模块，当前只覆盖「数值 → 符号」的目标词表端。

## 设计哲学

> **"用中文词定义中文词，用 `is_a` / 闭包 / 约束表代替向量相似度。"**

语义是一张**可查询的图**，不是一个**不可解释的向量**。
每一条关系都被 SQL 断言钉住，任何一处语义错误都应当能被一条查询捕获。

## 路线图（Roadmap / Planned）

以下能力**尚未完整实现**（目前只有设计文档、符号词表或推理原型）：

- **影像管线 Layer 2 ML 映射器** —— SVM / 随机森林 → 符号特征；Layer 1 的 CNN 检测同样待补。
  已有成果为符号词表与设计文档（`seed_medical_v3_imaging.sql`、`docs/MEDICAL_LIB_DESIGN.md` §13）。
- **动物库领域扩展** —— 目前只预留了跨库链接节点，领域子树尚未建立。
- **Logos Transformer** —— 基于关系图的注意力推理（relational graph attention）。设计见
  [docs/LOGOS_TRANSFORMER_DESIGN.md](docs/LOGOS_TRANSFORMER_DESIGN.md)；`logos_transformer/` 下已有
  可运行原型（随机权重、推理-only，训练与符号层回写尚未实现）。
- **自然语言接口** —— 面向终端用户的自然语言问答 / 写作入口。

## 许可证

[MIT](LICENSE) © 2026 Logos Contributors
