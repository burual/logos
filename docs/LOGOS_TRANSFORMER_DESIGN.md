# Logos Transformer —— 关系图注意力推理模块设计

> **状态**：设计完成 + 最小可运行原型 + 闭包预测训练与源分离验证（见 §9）
> **代码**：`logos_transformer/prototype_gat.py`（推理）、`logos_transformer/train_closure.py`（训练）
> **依赖**：`torch`（纯 PyTorch 实现，**不依赖** PyTorch Geometric）
> **文档版本**：v0.1（2026-10）
> **适用数据库**：`schema_v3.sql` 建出的 `logos.db`（当前全量 250 义项 / 477 条义项级关系 / 46 种关系类型 / 1090 条闭包；§9 的 W_r 实验基于动物库扩展前的 170 / 161 / 45 / 371 快照）

---

## 0. 目标与非目标

Logos 现有的语义推理是**符号的、精确的**：`relations` 存事实，触发器把 `is_transitive=1`
的关系物化进 `closure`，查询即证明。这条链路的优点是零幻觉、可解释；
缺点是**只能回答已经写进库里的关系**，并且对"最相似的下一个关系"这类软问题无能为力。

Logos Transformer 的定位是**符号层的候选生成器**，不是替代品：

| | 目标 | 说明 |
|---|---|---|
| ✅ | 注意力加权遍历 | 在多跳邻居上分配可解释的权重，而不是均匀聚合 |
| ✅ | 软推理（候选生成） | 生成 `relations` / `closure` 里没有、但结构上"像"的关系候选 |
| ✅ | 置信度回写 | 候选带 `confidence`，写入独立表 `inferred_relations`，供人工/规则复核 |
| ✅ | 与闭包共存 | 推断结果**不触发**闭包计算，不污染事实层 |
| ❌ | 替代闭包 | 精确的传递闭包仍然由 SQL 触发器负责 |
| ❌ | 生产级训练 | 本版只做**闭包预测的小规模监督实验**（§9），不做部署级训练 |
| ❌ | 端到端问答 | 自然语言接口属于后续里程碑 |

一句话：**闭包回答"确定是什么"，Transformer 提示"可能还有什么"。**

---

## 1. 图构建（Graph Construction）

### 1.1 数据来源

图从 SQLite **只读**导出（原型里用 `PRAGMA query_only=ON` 保证不改库）：

```sql
-- 节点：全部义项 + 词形 + 词性 + 释义
SELECT s.sense_id, w.word, s.pos, s.definition, s.frequency
FROM senses s JOIN words w ON w.word_id = s.word_id
ORDER BY s.sense_id;

-- 语义边：只取义项→义项的断言
SELECT relation_id, src_id, dst_id, relation, weight, confidence
FROM relations
WHERE src_kind = 'sense' AND dst_kind = 'sense' AND is_negated = 0
ORDER BY relation_id;

-- 关系类型词表（决定 edge type 的编号空间）
SELECT relation, relation_class, direction, inverse_of,
       is_transitive, is_symmetric, is_functional
FROM relation_types
ORDER BY relation;
```

实测口径（`logos.db`，2026-10）：

| 项目 | 数量 | 说明 |
|---|---:|---|
| `senses` | 170 | 图节点 |
| 义项级关系 `relations` | 161 | 其中 `is_a` 115 条（约 71%），长尾明显 |
| 指向 `entity_nodes` 的边 | 5 | **排除**：实体节点不在 `senses` 域内 |
| `closure` | 371 | 130 条 depth=1 + 241 条 depth≥2 |
| `relation_types` | 45 | 编号空间；实际出现 16 种 |
| 有边的义项 | 133 | 其余 37 个是孤立节点（只有自环） |

> **为什么排除 `src_kind='node'` 的边**：`entity_nodes` 是跨库链接的注册表（当前 5 行），
> 与 `senses` 不是同一个 id 空间。原型只建"义项同构图"；异构图的处理见 §7。

### 1.2 节点与节点特征

- **节点集合**：全部 170 个 `sense_id`，映射为连续下标 `0..169`（保留 `sense_id` 与原词形用于报告）。
- **节点特征**：原型用 `nn.Embedding(170, hidden_dim)`，即**随机初始化**的义项向量。
  - 选择理由：本版不训练，特征只需要可复现（固定 seed）。
  - 未来替换为符号特征：`pos` one-hot、`sense_features` 义素集合、释义中包含的义项（definition-as-bag-of-senses）。
  - 关键约束：**不引入外部预训练向量**，保持"用库内符号定义符号"的项目原则。

### 1.3 边、方向与闭包

**语义边是有向的**（`is_a` 只能从下位指向上位）。但消息传递必须是双向的，
否则像 `猫` 这种只有出边的节点收不到任何邻居消息（它的入边数 = 0），GAT 退化为自环。

因此采用**方向位编码**：

```
edge_type = 2 * relation_index + dir_bit
    dir_bit = 0  正向：src --R--> dst
    dir_bit = 1  反向：dst --R--> src   （语义不变，只是消息方向相反）
自环：edge_type = 2 * num_relation_types
```

正反向共用同一个关系类型的参数吗？**不共用**——`2r` 与 `2r+1` 是两种独立边类型，
模型可以学到"上位→下位"和"下位→上位"的不对称权重。这是把方向信息显式喂给注意力，
而不是靠节点位置隐式推断。

**闭包（`closure`）的方向约定**（容易踩坑，必须写清楚）：

对直接边 `s --R--> t`，`closure` 的 depth=1 行是 **`(relation=R, ancestor=t, descendant=s, depth=1, first_edge=relation_id)`**，
即 `ancestor` 存的是关系的**目标端**，`descendant` 存的是**源端**，
语义读作 **`descendant --R--> ancestor`**。例如：

```
relations: (2, 'is_a', 12)          -- 猫 is_a 哺乳动物
closure  : ('is_a', 12, 2, 1, 1)    -- ancestor=12, descendant=2, first_edge=1
closure  : ('is_a', 1026, 2, 2, 1)  -- 猫 is_a 1026，经第一跳 relation_id=1（2→12）
```

`first_edge` 是**从 `descendant` 出发的第一跳**的 `relation_id`，可用于还原见证路径。

**闭包边如何用**：

| 用法 | 是否默认 | 理由 |
|---|---|---|
| 消息传递边 | ❌ 默认关闭 | depth≥2 的闭包边等于把多跳答案直接连成一条边，模型无需推理即可看见，会高估效果 |
| 对照 / 评估标签 | ✅ | 报告中列出目标节点的 depth≥2 隐含关系，作为"理想输出"参照 |
| 消融实验边 | 可选（`--include-closure`） | 加号为独立关系类型 `closure:is_a`，可对比"给了捷径"与"没给捷径" |

> 结论：**训练/推理主图只用 `relations` 的直接边；`closure` 只做评估与可选消融。**

### 1.4 原型图规模

| 量 | 值 |
|---|---:|
| 节点 | 170 |
| 语义边（直接） | 161 |
| 双向消息边 | 322 |
| 自环 | 170 |
| **消息边合计** | **492** |
| 关系类型 | 45 → 边类型 `45×2+1 = 91` |

图极小（492 条消息边），单卡/CPU 都能秒级跑完，适合做管线验证，不适合报告有统计意义的指标。

---

## 2. 模型架构（Model Architecture）

### 2.1 总体结构

```
        node embedding (170 × 32, 随机初始化)
                     │
        ┌────────────▼─────────────┐
        │ RelationAwareGATLayer ×2 │   每层：关系专属 W_r、a_r
        │  heads=4, head_dim=8     │   边类型 = 关系 × 方向
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ 节点表示 h ∈ R^(170×32)   │
        │ 每边注意力 α ∈ R^(492×4)  │  ← 可解释性来源
        └──────────────────────────┘
```

### 2.2 关系感知图注意力层

对每个头 `k`、每条消息边 `j → i`（边类型 `r`，含方向位）：

```
e_{ij,k} = LeakyReLU( a_{r,k}^{tgt⊤} W_{r,k} h_i  +  a_{r,k}^{src⊤} W_{r,k} h_j )

α_{ij,k} = softmax_{j ∈ N(i)} ( e_{ij,k} )                       ← 按目标节点归一化

h_i' = ‖_{k=1..H}  σ( Σ_{j ∈ N(i)} α_{ij,k} · W_{r(i,j),k} h_j )
```

要点：

1. **每个关系类型有独立的 `W_r` 与 `a_r`**（不是把关系 one-hot 拼到特征上再共享权重）。
   参数量 `R × H × (F×O + 2O)`，在 91×4×(32×8+16) 量级，完全可接受。
   好处是 `is_a` 与 `has_symptom` 可以学到完全不同的传播算子。
2. **注意力按目标节点归一化**，因此 `Σ_j α_{ij,k} = 1`，天然可读作"i 把多少注意力分给了 j"。
3. **自环参与归一化**：孤立节点不会得到全零表示，同时自环权重本身也是可观察量
   （自环占比过高说明该节点主要在"自言自语"，是数据稀疏的信号）。
4. `σ` 为 ELU；原型最后一层不再激活，直接输出嵌入。
4. 原型中 `hidden_dim=32`、`heads=4`、`head_dim=8`，两层都做 head concat（32 = 4×8）。
   第 1 层 attention 反映**直接邻居**，第 2 层 attention 反映**邻居的邻居**混入后的重新分配。

### 2.3 输出与可解释性

| 输出 | 形状 | 用途 |
|---|---|---|
| 节点嵌入 `H` | `(N, hidden_dim)` | 下游关系打分 / 检索 |
| 每边注意力 `α` | `(E, heads)` | 解释"为什么连到它"；报告取 heads 均值 |
| 关系打分 `score(s,R,t)` | 标量 | §3 的候选排序（原型未训练，仅接口占位） |

**诚实声明**：随机权重下的注意力**没有语义含义**。本原型只证明
"SQLite 图 → 张量 → 关系感知注意力 → 可读报告"这条管线成立。
训练之后注意力才可以作为置信度先验；即便如此，**注意力 ≠ 因果解释**，
只能当"模型看了哪里"的证据，不能当"语义为什么成立"的证明。

### 2.4 超参数

| 参数 | 原型默认 | 未来建议 | 说明 |
|---|---:|---:|---|
| `hidden_dim` | 32 | 128–256 | 当前图太小，32 足够；再大必然过拟合 |
| `num_heads` | 4 | 4–8 | 头数 × head_dim = hidden_dim |
| `num_layers` | 2 | 2–4 | 2 层已覆盖 is_a 链的 2 跳；更深的收益受数据量限制 |
| `dropout` | 0.1 | 0.1–0.3 | 原型 `eval()` 下不生效 |
| `negative_slope` | 0.2 | 0.2 | LeakyReLU |
| `seed` | 42 | — | 固定随机权重，保证报告可复现 |
| 训练轮数 | — | 200–500 | 见 §5 |

---

## 3. 推理协议（Inference Protocol）

### 3.1 查询形式

```
(s, R, ?)   —— 给定源义项与关系类型，找目标
(?, R, t)   —— 给定目标与关系类型，找源（把边反向即可）
```

CLI 形态（规划）：

```bash
python logos_transformer/query.py --db logos.db --src 猫 --rel is_a --top-k 10
```

### 3.2 打分：注意力 + 关系打分

对候选 `t`，定义三档置信度：

**(a) 1 跳（直接邻居）**

```
conf_1hop(s, R, t) = mean_k α_{s→t, k}          （若 s --R--> t 存在）
```

**(b) 多跳（注意力路径组合）**

沿路径 `s = v0 → v1 → ... → vL = t`：

```
conf_path = Π_{l=1..L} mean_k α_{v_{l-1} → v_l, k} · γ^(L-1)
```

`γ ∈ (0,1)` 是长度衰减（建议 0.7），避免长路径靠数量堆积。
同一 `(s,R,t)` 有多条路径时取 **max**（最短/最强见证优先），不做 sum。
实现上只展开 `L ≤ 3`，且中间节点必须存在（不生成虚构节点）。

**(c) 关系打分（训练后启用）**

```
score(s, R, t) = σ( MLP_R( [ h_s ‖ h_t ‖ h_s ⊙ h_t ] ) )
```

**(d) 融合**

```
confidence = λ · conf_path + (1 − λ) · score        λ 默认 0.5
```

原型只输出 (a) 与 (b) 所需的第一层注意力，`MLP_R` 留空（不训练就没有意义）。

### 3.3 过滤与阈值

返回结果前依次过滤：

1. **已知事实**：`(s,R,t)` 已在 `relations` 或 `closure` 中 → 不返回（也可用 `--include-known` 调试）。
2. **自身**：`s == t` 排除（自环注意力只用于诊断）。
3. **关系约束**：`relation_constraints` 的 `slot/kind/value` 若与候选的 `pos` / `sense_features` 冲突 → 降权或剔除。
4. **阈值**：`confidence ≥ τ`（默认 0.60，可配置）。
5. **Top-k**：默认 10，按 `confidence` 降序。
6. **去重**：同一 `(s,R,t)` 只保留最高分。

输出结构：

```json
{"src_id": 2, "relation": "is_a", "dst_id": 1026,
 "confidence": 0.71, "path": [2, 12, 1026], "model_version": "logos-gat-v0.1"}
```

### 3.4 冷启动限制

原型权重随机，**排序不可用于真实推断**。§3 的协议是为训练后版本定义的接口；
原型的职责是证明"注意力矩阵可取出、可归一化、可对照闭包"。

---

## 4. 与符号层的集成（Integration with Symbolic Layer）

### 4.1 建议的新表

```sql
CREATE TABLE IF NOT EXISTS inferred_relations (
    inferred_id    INTEGER PRIMARY KEY,
    src_id         INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    relation       TEXT    NOT NULL REFERENCES relation_types(relation),
    dst_id         INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    confidence     REAL    NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    model_version  TEXT    NOT NULL,
    path_json      TEXT,            -- 见证路径，如 "[2,12,1026]"，便于复核
    reviewed       INTEGER NOT NULL DEFAULT 0 CHECK (reviewed IN (0,1)),
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (src_id, relation, dst_id, model_version)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_inferred_conf
    ON inferred_relations(confidence DESC)
    WHERE reviewed = 0;
```

> 该 DDL **只是设计**，本次不修改 `schema_v3.sql`（约束：不动 .sql）。
> 落地时需要一次独立的 schema 迁移，并补进 `validate_schema.py` 的表计数。

### 4.2 为什么不直接写进 `relations`

| 方案 | 问题 |
|---|---|
| 直接插 `relations` | `tr_closure_edge_ai` 会立刻把它当事实物化进 `closure`，**建议变成事实**，且无法回滚 |
| 插 `relations` 但 `confidence` 很低 | `relations.confidence` 语义是"来源置信度"（人工/词典），与模型分数混在一起会污染审计；`views`/触发器也不区分来源 |
| **独立表 `inferred_relations`** ✅ | 命名空间隔离：不被任何视图引用、不进闭包、不参与 `validate_*` 计数；可按 `model_version` 整批回滚 |

### 4.3 晋升路径（promotion）

推断结果只有经过显式动作才能进入事实层：

```
inferred_relations (reviewed=0)
        │  人工/规则复核（reviewed=1）
        ▼
INSERT INTO relations (..., source_id = <新增的 'Logos Transformer' source>)
        │
        ▼
tr_closure_edge_ai 自动触发 → closure 物化（此时才成为"事实"）
```

建议在 `sources` 表新增一行 `('Logos Transformer', '模型推断', ...)`，
让每条晋升后的关系都可追溯到模型版本。

### 4.4 API

**Python 模块**（原型已可 import）：

```python
from logos_transformer.prototype_gat import load_graph, build_model, run_inference

graph = load_graph("logos.db")
model = build_model(graph, hidden_dim=32, num_heads=4, num_layers=2, seed=42)
embeddings, attentions = run_inference(model, graph)
```

**CLI**（原型）：

```bash
python logos_transformer/prototype_gat.py                  # 默认：猫，top-5
python logos_transformer/prototype_gat.py --sense 脑炎 --top-k 8
python logos_transformer/prototype_gat.py --include-closure   # 消融：把闭包捷径也当边
```

未来查询 CLI（未实现）：

```bash
python logos_transformer/query.py --src 猫 --rel is_a --top-k 10 --threshold 0.6 --write
```

`--write` 是唯一会写库的入口，且只写 `inferred_relations`，**永不**写 `relations`。

### 4.5 不变量（实现时必须守住）

1. 推理过程对 `logos.db` **只读**（`PRAGMA query_only=ON`）。
2. 推断边**不触发**闭包；不修改 `relations` / `closure` / `senses`。
3. 每条推断边必须带 `model_version`；无版本的记录视为非法。
4. `validate_schema.py` / `validate_seed.py` 的计数**不含** `inferred_relations`。
5. 关闭外键时也能安全读取（只读路径不依赖 FK）。

---

## 5. 训练策略（future，仅记录）

### 5.1 自监督：掩码关系恢复

1. 从 161 条直接边中按 `relation_id` 划分 train/val/test（**按边划分，不按节点**，避免同一子图泄漏）。
2. 训练时随机 mask 10% 的边（从消息图中删除），让模型为 `(s,R,?)` 排序。
3. 负样本：随机替换 `dst`（过滤掉已存在的事实与闭包），比例 1:5。
4. 损失：`BCEWithLogits` 或 `CrossEntropy`（1 正 + K 负）。

### 5.2 监督信号与闭包的正确用法

- `closure` 的 depth≥2 行可以作为 `is_a` 的**额外正样本**，但必须**按深度划分**：
  - 训练只用 depth ∈ {1,2}
  - 测试用 depth ≥ 3（模型没见过完整路径）
- **禁止**把 depth≥2 闭包边同时放进消息图和标签（标签泄漏）。
- 对 `is_a`，闭包本身在 depth=1 上 100% 正确，因此**必须报告"无闭包基线"**：
  只给 1 跳边，模型能否恢复 2–3 跳？否则指标没有信息量。

### 5.3 训练配置（建议）

| 项 | 建议值 |
|---|---|
| 优化器 | AdamW, lr=1e-3, weight_decay=1e-4 |
| batch | 全图（170 节点，无需采样） |
| epochs | 200–500，早停 patience=30 |
| 损失 | BCEWithLogits（负采样后） |
| 评估 | MRR、Hits@1/3/10，按 relation 分组（`is_a` vs 其他） |
| 基线 | (a) 随机；(b) 度数启发式；(c) 仅闭包 |

### 5.4 实现状态

- 训练循环、数据集划分、checkpoint 保存已在 `logos_transformer/train_closure.py` 实现，训练目标为**闭包预测**（见 §9）。
- **关系条件化打分头 `W_r` 已实现**：`logos_transformer/train_closure_relaware.py` 为 45 种关系各建一个 32×32 矩阵，稀有/未见关系回退到共享矩阵，见 §9.10。注意当前闭包标签中 240/241 是 `is_a`，因此该实现尚无法在数据上证明关系多样性的收益。
- 本节其余建议（掩码关系恢复、AdamW + weight_decay、按深度划分、查询级负采样）**尚未实现**，仍作为后续方向。

---

## 6. 原型实现（Prototype）

| 项 | 内容 |
|---|---|
| 文件 | `logos_transformer/prototype_gat.py` |
| 依赖 | `torch >= 2.0`（自实现 GAT，**不用 PyG**，避免 `torch-scatter` 编译） |
| 运行 | `python logos_transformer/prototype_gat.py` |
| 输入 | `logos.db`（缺失则报错退出 1） |
| 输出 | 节点数 / 边数 / 嵌入形状 / 目标义项的逐边注意力 / Top-k 邻居 / 闭包对照 |
| 参数 | `--db --sense --top-k --hidden --heads --layers --dropout --seed --include-closure` |
| 失败模式 | 无 torch → `Requires torch. Install with: pip install torch`，退出 1；无 DB → 明确提示，退出 1 |

实现要点（与本文档一致）：

- `load_graph()` 只读导出，返回 `node_ids / labels / edge_index / edge_type / relation_names`。
- `RelationAwareGATLayer` 用 `scatter_reduce(amax)` + `index_add` 实现按目标节点的分段 softmax，
  不依赖任何扩展包。
- 报告里的注意力 = 第 1 层 4 个头均值；同时打印第 2 层均值作为"两跳混入后"的对照。
- **不写数据库**，不生成 `inferred_relations`（该表尚无迁移）。

---

## 7. 风险与边界

| 风险 | 影响 | 缓解 |
|---|---|---|
| 图太小（170 节点 / 161 边） | 训练必然过拟合，指标不可信 | 定位为管线验证；训练前先扩库 |
| `is_a` 占 71% | 模型倾向只学好 `is_a` | 按关系分组报告；对长尾关系单独采样 |
| 注意力被误读为解释 | 错误归因 | 文档与输出中固定声明"注意力≠因果"；报告附随机基线 |
| 闭包方向约定易错 | 训练标签反向，指标虚高 | §1.3 固定 `descendant --R--> ancestor`，代码与文档共用该约定 |
| 异构节点（`entity_nodes`） | 跨库链接无法参与 | 本版排除；未来用 `node_type` 做异构 GAT |
| 中文编码（Windows cp936） | 报告乱码 | 脚本 `sys.stdout.reconfigure(encoding="utf-8")` |
| 模型版本漂移 | 推断结果不可复现 | `model_version` + 固定 seed + 记录超参数 |

---

## 8. 里程碑

| 阶段 | 内容 | 状态 |
|---|---|---|
| **M0** | 设计文档 + 随机权重 GAT 原型 + 注意力报告 | ✅ 本次交付 |
| M1 | 掩码关系恢复训练；MRR/Hits@k；与闭包基线对比 | 🚧 闭包预测已完成（§9）；掩码关系恢复待做 |
| M2 | `inferred_relations` 迁移 + `query.py --write` + 复核流程 | ⬜ |
| M3 | 与查询规划器集成：先查闭包，未命中再问 Transformer | ⬜ |
| M4 | 异构扩展（`entity_nodes`）与释义级节点特征 | ⬜ |
| M5 | 自然语言接口（见 README Roadmap） | ⬜ |

---

## 9. 训练实验：闭包预测（closure prediction）

首个监督实验，验证"只看 1 跳消息，能否预测 depth≥2 的隐式多跳关系"。
实现：`logos_transformer/train_closure.py`（**不修改** `prototype_gat.py`）。

### 9.1 任务与数据

- **标签**：`closure` 中 `depth >= 2` 的行，口径为 `descendant --R--> ancestor`（§1.3），
  正样本即 `(descendant, ancestor, relation)`。
- **输入**：仅 1 跳语义边（`include_closure=False`），闭包**不进消息图**，避免标签泄漏。
- **负样本**：对每个正样本，从"与源节点既无直接边、也无任何闭包路径（双向）"的节点中
  随机采样 3 个；按 `(src, dst)` 去重。负样本继承其正样本的关系类型，便于分层与分组指标。

| 项目 | 数量 |
|---|---|
| 正样本 | 241（`is_a` 240 + `has_part` 1） |
| 深度分布 | depth2 103 / depth3 76 / depth4 48 / depth5 13 / depth6 1 |
| 负样本（3× 采样，去重后） | 712 |
| 样本总数 | 953 |
| 划分（按关系分层，70/15/15） | train 667 / val 143 / test 143 |
| 训练集类别 | 169 正 / 498 负（`pos_weight=2.95`） |

> `has_part` 只有 1 个正样本，全部落在 train；因此 **val/test 的正样本全部是 `is_a`**。

### 9.2 模型与超参数

在原型 GAT 之后接双线性打分头 `σ(h_s^T W h_t)`（`W` 无 bias），整体端到端训练。

| 项 | 值 |
|---|---|
| 编码器 | 2 层关系感知 GAT，4 头，hidden=32，dropout=0.1（203K 参数） |
| 预测头 | `h_s^T W h_t`，`W` 为 32×32，Xavier 初始化 |
| 损失 | `BCEWithLogitsLoss(pos_weight=2.95)` |
| 优化器 | Adam lr=1e-3 + `ReduceLROnPlateau(factor=0.5, patience=10)` |
| batch / epochs | 64 / 最多 200，早停 patience=20，seed=42 |

打分头**与关系类型无关**（共享一个 `W`）；关系仅用于分层划分、分组指标与负采样。
这是本实验最明显的简化。关系条件化的 `W_r` 已在 §9.10 实现并复测；当前数据
（240/241 为 `is_a`）不足以证明其收益。

### 9.3 实测结果（随机划分）

> 下表是**按 pair 随机划分**的结果，存在同 source 跨 split 的泄漏；源分离复测见 §9.8。

运行：`python logos_transformer/train_closure.py`（best epoch 40，第 60 epoch 早停，val loss 0.0811）。

| 指标 | 值 |
|---|---|
| Test BCE / accuracy | 0.1444 / 0.9720 |
| ROC-AUC | **0.9969** |
| Precision@10（过滤全库，按 query） | 0.1167 |
| Precision@20（过滤全库，按 query） | 0.0600 |
| MRR（逐正样本，过滤全库 170 候选） | **0.7656** |
| MRR（按 query 取首个命中） | 0.8583 |
| Hits@1 / @3 / @10 | 0.6389 / 0.9167 / 0.9722 |
| 查询数（source×relation） | 30 |

按闭包深度拆分（过滤全库排序）：

| depth | n | MRR | Hits@1 | Hits@10 |
|---|---|---|---|---|
| 2 | 17 | 0.7353 | 0.5882 | 1.0000 |
| 3 | 12 | 0.8247 | 0.7500 | 0.9167 |
| 4 | 4 | 0.7083 | 0.5000 | 1.0000 |
| 5 | 3 | 0.7778 | 0.6667 | 1.0000 |

depth=6 的唯一正样本落在 train，test 无该深度样本。**深度 4/5 仍有命中**，说明指标并非只靠
1–2 跳；但因 2 层 GAT 的感受野只有 2 跳，这部分更多来自结构与节点嵌入，而非消息传递。

### 9.4 注意力是否发生位移

第 1 层注意力（4 头均值）的平均归一化熵：

| | 熵（1.000 = 均匀，越低越集中） |
|---|---|
| 训练前（随机权重） | 0.9997 |
| 训练后 | **0.9930** |

「猫」的入边注意力 Top-3：

| 训练前 | 训练后 |
|---|---|
| (self) 0.2061 / 动物 is_a 0.2021 / 跑 can_do 0.1987 | 吃 can_do 0.2427 / 抓 can_do 0.2368 / 跑 can_do 0.2130 |

训练后自环与 `is_a 动物` 让位给 `can_do` 出边，注意力**确实发生了位移**且更集中。
注意：注意力 ≠ 因果解释，这里只报告分布变化。

### 9.5 定性样例（「猫」= sense 2）

Top-5 多跳目标（隐藏了 1 跳邻居）：

| 排名 | 目标 | 分数 | 闭包匹配 |
|---|---|---|---|
| 1 | 生物(1026) | 1.0000 | ✅ `is_a`, depth 2 |
| 2 | 抽象概念(2101) | 1.0000 | — |
| 3 | 医学概念(4001) | 1.0000 | — |
| 4 | 营养物(2001) | 0.9999 | — |
| 5 | 脑(4004) | 0.9997 | — |

猫的唯一闭包目标「生物」被排到 **#1**；第 2–5 名是结构上相似但库中未断言关系的概念，
正符合"候选生成器"的定位（低精度、高召回，交给复核）。

### 9.6 结论与限制

- **可行**：只看 1 跳消息，模型能把多跳闭包目标排到很前（随机划分 MRR 0.766、Hits@3 0.917；源分离后 MRR 0.569，见 §9.8）。
- **高 AUC 有水份**：划分按关系分层，同一 source 常同时出现在 train 与 test（**pair completion，
  而非冷启动 source**）；再加上 transductive 节点嵌入，AUC 偏高可以预期。诚实的压力测试是
  **按 source 划分**（留出整个词），已在 §9.8 补做；MRR 下降 0.197，但 train ≈ test，
  说明随机划分的虚高很可能主要来自 source 泄漏。
- **头与关系无关**，`has_part` 仅 1 个正样本，长尾关系完全没有评估；§9.10 已实现
  关系条件化 `W_r`，但受同样的数据多样性限制。
- **Precision@K 低**：每个 query 在 170 候选里通常只有 1–2 个真目标，`P@10` 上限本就 ≈0.1；
  排序类指标（MRR/Hits）更有信息量。
- 模型是 **transductive** 的：节点嵌入按 `sense_id` 学习，checkpoint 只能在**同一节点顺序**下加载。

**数据扩充建议**（当前仅 241 个正样本、953 个样本对，规模偏小）：

1. 为可逆/对称关系增加逆向关系类型，把反向边也作为正样本；
2. 扩充动物库（同域节点与 1 跳边），提升 depth≥2 闭包的绝对数量；
3. 用 **source-disjoint** 划分替代当前分层划分，单独报告冷启动能力；
4. 只保留 depth=2 做公平的 2 跳任务，或改用 k-fold 交叉验证降低单次划分的方差。

### 9.7 产物与复现

```bash
python logos_transformer/train_closure.py --epochs 5      # 冒烟测试
python logos_transformer/train_closure.py                 # 完整训练
# CLI: --db --epochs --lr --hidden --heads --layers --batch-size \
#      --output-dir --neg-ratio --patience --seed
```

| 产物 | 路径 |
|---|---|
| 最优权重（`state_dict`） | `logos_transformer/checkpoints/closure_gat_best.pt` |
| 训练报告（指标 / 注意力 / 定性 / 限制） | `logos_transformer/checkpoints/training_report.txt` |

失败模式：无 torch → `Requires torch. Install with: pip install torch`，退出 1；
无 DB → 明确提示，退出 1；无闭包数据 → `No closure data found, run apply_seeds.py first`，退出 1。

### 9.8 源分离验证（source-disjoint, 2026-10）

为排除同一 source 跨 split 的 transductive 泄漏，新增 `train_closure_disjoint.py`：
按 **source sense** 做 70/15/15 划分（101 → 71 / 15 / 15），闭包边跟随其 source，
负样本也只由本 split 的 source 生成；test source 不参与训练或早停。

| 指标 | 随机划分（§9.3） | 源分离（§9.8） |
|---|---|---|
| Test ROC-AUC | 0.9969 | 1.0000 |
| Test MRR | 0.7656 | **0.5690** |
| Hits@1 | 0.6389 | **0.3571** |
| Hits@10 | 0.9722 | 1.0000 |

- train / val / test 的 MRR 分别为 0.5555 / 0.5389 / 0.5690：**train ≈ test**，
  说明模型没有依赖 source 记忆；随机划分的 0.766 与 source 泄漏造成的虚高一致。
- AUC 对"随机闭包无关负样本"已经饱和（全候选 hard AUC 0.9944），不能单独作为
  "学会关系结构"的证据。更关键的是：一个**与 source 无关的目标先验**（按目标平均
  得分排序）即可达到 test MRR 0.5259，约为逐 source MRR 的 92%。当前模型主要学到
  "哪些节点像祖先"，源条件化的关系信号只占很小一部分。
- ~~深度衰减明显：depth2 MRR 0.7667 / Hits@1 0.6667；depth3 0.4000 / 0.0；
  depth4 0.3000 / 0.0；depth5 0.1667 / 0.0。~~ **2026-10-22 更正**：该 by-depth
  视图受 `filtered_metrics` 的 rank/depth 对齐 bug 影响（总体 MRR 不受影响）。
  按逐正样本重算：depth2 0.5022 / 0.2667；depth3 0.5500 / 0.3750；
  depth4 0.6667 / 0.3333；depth5 1.0000 / 1.0000（Hits@10 均为 1.0）。
  因此**没有**简单的深度衰减，depth≥3 仍有 top-1 命中；depth4/5 的 n=3/2，
  不应过度解读。

**结论**：源分离消除了泄漏，模型对未见 source 不崩溃（Hits@10 = 1.0），但
"学会了多跳关系推理"目前**不能**被证实——大部分排序来自目标先验（0.5259 /
0.5690 ≈ 92%）。更正后的 per-depth 并没有简单的深度衰减，depth≥3 仍有 top-1
命中，但 depth4/5 的样本量只有 3/2。下一步应使用**硬负样本**（其他 source 的
同关系、同深度闭包目标）重跑，用"模型 vs 目标先验"的差距衡量真正的源条件化
能力。完整对比与建议见
[`SOURCE_DISJOINT_COMPARISON.md`](../SOURCE_DISJOINT_COMPARISON.md)。

产物：`logos_transformer/checkpoints_disjoint/closure_gat_disjoint_best.pt` +
`training_report.txt`。

### 9.9 硬负样本训练（hard negatives, 2026-10）

§9.8 的结论是：随机负样本太容易，目标先验已解释约 92% 的 MRR。为验证模型能否
真正利用 source，新增 `train_closure_hardneg.py`：保持同一 source-disjoint 划分、
模型与超参数，将负样本分为三档：

- **hard**：其他 source 的同 relation、同深度带（2 / 3 / 4+）闭包目标，且从
  query source 不可达（direct/closure 双向排除）；
- **medium**：同 relation、其他深度带，同样排除可达目标；
- **easy**：随机节点。

默认每正样本 6 个负样本 = 3 hard / 2 medium / 1 easy。hard 池不足时按
hard → medium → easy 回退，并在报告中记录 shortfall。每个 epoch 在 **val**
positive 上计算 hard-neg MRR、目标先验 MRR 与 margin，并与 loss 同列打印。

| 指标 | 随机划分 | 源分离 | 硬负样本 |
|---|---:|---:|---:|
| Test MRR | 0.7656 | 0.5690 | **0.6404** |
| Hits@1 | 0.6389 | 0.3571 | **0.4643** |
| Hits@10 | 0.9722 | 1.0000 | 0.9643 |
| Hard-neg MRR | — | 0.7411 | **0.9714** |
| Hard-neg margin | — | +0.0407 | **+0.2292** |
| Full-catalogue margin | — | +0.0431 | **+0.1619** |

- 两个要求阈值（hard-neg MRR > 0.3、margin > 0.1）在点估计上通过；配对
  source-cluster bootstrap（15 test sources, 10k）显示 hardMRR 提升
  +0.2294 [+0.1068, +0.3694]、hard-pool margin 提升 +0.1904 [+0.0599, +0.3582]、
  full-catalogue margin 提升 +0.1211 [+0.0330, +0.2444]，均显著。
- 但绝对 full-catalogue MRR 提升 +0.0737 [−0.0025, +0.1824]，不显著；
  Hits@10 从 1.0 降到 0.9643（`人→生物` full rank #22）；`猫→生物` 从 #3 降到
  #13。绝对 hard-pool margin 的 95% CI 为 [+0.0814, +0.4285]，下界略低于 0.1。
- hard 池天然很小：`is_a` 在 depth2/3/4+ 只有 14/6/3 个不同目标；22/180 train、
  1/33 val、0/28 test 的 hard 池为空。hard 负样本主要是**跨分支 hub**
  （动物 source 的池里是 `人造物` / `医学概念` / `病理状态`，而不是另一个动物
  祖先），所以模型学到的是较粗的"分支归属"，不是同分支内的细粒度祖先排序。
- 结论：硬负样本训练**显著增强 source 条件化信号**并降低对目标先验的依赖，
  但不是全面的多跳排序提升；细粒度推理仍受数据多样性限制。下一步：扩充动物
  库 / 同分支祖先、target-disjoint 划分、top-K full-catalogue 硬负样本、以及
  pairwise ranking loss。relation-aware `W_r` 已在 §9.10 实现并复测；完整分析见
  [`HARD_NEGATIVE_COMPARISON.md`](../HARD_NEGATIVE_COMPARISON.md)。

产物：`logos_transformer/checkpoints_hardneg/closure_gat_hardneg_best.pt`、
`closure_gat_hardneg_best_margin.pt`（诊断用）、`training_report.txt`。

### 9.10 关系感知打分头（relation-aware `W_r`, 2026-10）

§9.9 的后续建议之一是实现关系条件化的打分头。新增
`logos_transformer/train_closure_relaware.py`，保持与 hard-negative 实验**完全相同**的
source-disjoint 划分、硬/中/易负采样、优化器与超参数，只替换打分头：

```text
旧: score(s,t)   = h_s^T W h_t
新: score(s,t,r) = h_s^T W_r h_t
```

实现细节：

- 为图中 **45 种关系**各建一个 32×32 的 `W_r`（46,080 参数），另建一个
  `W_shared`（1,024 参数）与 45 个门控标量（45 参数）；关系头合计 **47,149** 参数，
  模型总计 **250,605** 参数。
- 训练正样本数 `count >= 5`：直接用 `W_r`；`0 < count < 5`：使用
  `sigmoid(g_r) * W_r + (1 - sigmoid(g_r)) * W_shared`；`count == 0`：回退到
  `W_shared`。`count` 以 buffer 保存，checkpoint 可复现同样的 dedicated/gated/unseen 判定。
- 可选 `--freeze-gat-epochs 10`（默认）：前 10 个 epoch 冻结 GAT、只训练关系头，
  第 11 个 epoch 解冻；冻结期间关闭 GAT dropout。
- 新增 per-relation 全库 MRR / Hits、per-relation hard-pool MRR / margin，以及按关系
  分组的第 1 层注意力 top-3 边诊断。
- 与 `checkpoints_hardneg/closure_gat_hardneg_best.pt` 在**同一批 test pairs 与
  hard pools** 上逐项对比。

数据事实（这是本次结论的关键）：

| relation | depth>=2 正样本 | train / val / test |
|---|---:|---|
| `is_a` | 240 | 179 / 33 / 28 |
| `has_part` | 1 | 1 / 0 / 0 |
| `has_property` | **0** | 0 / 0 / 0 |
| 其余 42 种 | 0 | 0 / 0 / 0 |

因此 43 个 `W_r` 从未收到闭包正样本；`has_part` 只有 1 个训练样本；test 的 28 个
正样本**全部是 `is_a`**。per-relation test MRR 与 pooled MRR 在数学上必然相同。

实测（50 epoch 早停；best val loss 0.1086 @ epoch 30；best val margin +0.1862 @ epoch 20）：

| 指标 | 随机划分 | 源分离 | 硬负样本 | 关系感知（本次） |
|---|---:|---:|---:|---:|
| Test MRR | 0.7656 | 0.5690 | 0.6404 | **0.6889** |
| Hits@1 | 0.6389 | 0.3571 | 0.4643 | **0.5000** |
| Hits@3 | 0.9167 | 0.6429 | 0.7143 | **0.8571** |
| Hits@10 | 0.9722 | 1.0000 | 0.9643 | 0.9643 |
| Hard-neg MRR | — | 0.7411 | 0.9714 | **0.9688** |
| Hard-pool margin | — | +0.0407 | +0.2292 | **+0.2106** |
| Full-catalogue margin | — | +0.0431 | +0.1619 | **+0.2032** |

- **Hard-neg MRR 0.9688**，略低于 0.97 阈值，也略低于 hard-negative 的 0.9714；
  paired source-cluster bootstrap（15 test sources, 10k）差值 −0.0028，
  95% CI [−0.0098, +0.0000]。
- **Hard-pool margin +0.2106**，低于 0.2292；paired 差值 **−0.0191**
  [−0.0400, −0.0041]，即 hard-pool 上的退步是统计支持的。
- Full-catalogue MRR 提升 +0.0486（CI [−0.0002, +0.1408]），full-catalogue margin
  提升 +0.0412（CI [−0.0039, +0.1319]），两者都不显著。
- `猫 -> 生物` 的 full rank 从 hard-negative 的 #13 变为 **#12**，没有回到
  source-disjoint 的 #3；hard-pool rank 从 #4 退到 #6。
- 逐层注意力诊断按关系输出了 top-3 边；`is_a` 的注意力仍集中在少数结构边上，
  但注意力模式不能解释 full-catalogue 的误排（如 `影像特征`）。

**结论**：`W_r` 已正确实现，且是低成本的扩展；但当前数据无法验证关系多样性的收益。
test 只有一个关系，43 个头未被训练，`has_property` 没有 depth>=2 正样本。观察到的
full-catalogue 小幅提升更可能来自初始化、10 epoch GAT 冻结及优化轨迹，而不是
关系条件化本身。当前瓶颈仍是**闭包标签的关系多样性 + 同分支负样本多样性**，不是
打分头。下一步应优先扩充同分支祖先（如 `猫科`、`哺乳动物`）与 `has_property`
多跳标签；若要隔离冻结的影响，应补跑 `--freeze-gat-epochs 0` 消融。

产物：`logos_transformer/checkpoints_relaware/closure_gat_relaware_best.pt`、
`closure_gat_relaware_best_margin.pt`（诊断用）、`training_report.txt`。
完整分析、逐关系表、逐深度表与 bootstrap 见
[`RELATION_AWARE_COMPARISON.md`](../RELATION_AWARE_COMPARISON.md)。

---

## 附录 A：数据核对 SQL

```sql
-- 图规模
SELECT COUNT(*) FROM senses;                                   -- 170
SELECT COUNT(*) FROM relations
 WHERE src_kind='sense' AND dst_kind='sense' AND is_negated=0; -- 161
SELECT COUNT(*) FROM closure;                                  -- 371
SELECT SUM(depth=1), SUM(depth>=2) FROM closure;               -- 130 / 241

-- 孤立节点数
SELECT COUNT(*) FROM senses s
 WHERE NOT EXISTS (SELECT 1 FROM relations r
                    WHERE (r.src_kind='sense' AND r.src_id=s.sense_id)
                       OR (r.dst_kind='sense' AND r.dst_id=s.sense_id));  -- 37

-- 目标节点的闭包对照（猫 = 2）
SELECT relation, ancestor, depth, first_edge
  FROM closure WHERE descendant = 2 AND depth >= 2;
```

## 附录 B：术语

| 术语 | 含义 |
|---|---|
| 义项（sense） | `senses` 的一行，一个词的一个语义 |
| 语义边 | `relations` 中 `src_kind='sense'` 且 `dst_kind='sense'` 的行 |
| 消息边 | 双向展开 + 自环后的边，供 GAT 聚合 |
| 边类型 | `2 × 关系编号 + 方向位`，自环单独一种 |
| 闭包 | `closure` 中由触发器物化的传递关系；本项目里 `ancestor` 是语义目标端 |
