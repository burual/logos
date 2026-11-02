# Logos Transformer —— 关系图注意力推理模块设计

> **状态**：设计完成 + 最小可运行原型 + 闭包预测训练与源分离验证（见 §9）
> **代码**：`logos_transformer/prototype_gat.py`（推理）、`logos_transformer/train_closure.py`（训练）
> **依赖**：`torch`（纯 PyTorch 实现，**不依赖** PyTorch Geometric）
> **文档版本**：v0.1（2026-10）
> **适用数据库**：`schema_v3.sql` 建出的 `logos.db`（当前全量 250 义项 / 477 条义项级关系 / 46 种关系类型 / 1090 条闭包；§9 的 W_r 实验已在动物库扩展后的 250 / 477 / 46 / 1090 快照上复测，见 §9.10.1；§9.10 正文保留扩展前的 170 / 161 / 45 / 371 结果作为基线）

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

### 4.1 `inferred_relations`：设计草案 → 已实现

> **已实现**（2026-11-02，`logos_transformer/infer_and_write.py`）。实际列名与本节
> 最初的草案不同（草案是自拟的，实现按集成任务书给定的列名），差异见下方对照表。
> 表由推理脚本 `CREATE TABLE IF NOT EXISTS` 建立，**未写进 `schema_v3.sql`**
> （约束：不动 .sql 种子/模式文件），也不参与任何 `validate_*` 计数。

```sql
CREATE TABLE IF NOT EXISTS inferred_relations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id      INTEGER NOT NULL,
    relation_type  TEXT    NOT NULL,
    target_id      INTEGER NOT NULL,
    confidence     REAL    NOT NULL,
    model_version  TEXT    NOT NULL,
    epoch          INTEGER,
    depth_estimate INTEGER,          -- 估计跳数：1 / 2 / 3（3 表示 3+），NULL=无路径
    verified       BOOLEAN DEFAULT FALSE,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (source_id <> target_id),
    FOREIGN KEY (source_id) REFERENCES senses(sense_id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES senses(sense_id) ON DELETE CASCADE,
    UNIQUE (source_id, relation_type, target_id)
);
```

草案 vs 实际：

| 草案 | 实际 | 原因 |
|---|---|---|
| `inferred_id` | `id` | 与任务书一致 |
| `src_id` / `dst_id` | `source_id` / `target_id` | 同上 |
| `relation` | `relation_type` | 同上 |
| `reviewed INTEGER` | `verified BOOLEAN` | 同上；语义相同（0=未复核） |
| `path_json` 见证路径 | `depth_estimate INTEGER` | 见 §10.5：没有可靠见证路径，硬造会骗人 |
| `REFERENCES relation_types(relation)` | 无此外键 | 关系名可能在 `closure` 存在但 `relation_types` 无对应行时静默失败；改由脚本侧校验 |
| `UNIQUE (..., model_version)` | `UNIQUE (source_id, relation_type, target_id)` | 同一条假设只留最新一版，避免多版本堆叠；重跑时 `INSERT OR IGNORE` 幂等 |
| `STRICT` 表 | 普通表 | SQLite `STRICT` 与 `BOOLEAN`/`TIMESTAMP` 亲和名 + `AUTOINCREMENT` 组合在该库版本上易踩坑，且本表非权威数据 |

配套索引（脚本建立）：

```sql
CREATE INDEX IF NOT EXISTS idx_inferred_rel_conf
    ON inferred_relations(relation_type, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_inferred_source
    ON inferred_relations(source_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_inferred_unverified
    ON inferred_relations(verified) WHERE verified = 0;
```

### 4.2 为什么不直接写进 `relations`

| 方案 | 问题 |
|---|---|
| 直接插 `relations` | `tr_closure_edge_ai` 会立刻把它当事实物化进 `closure`，**建议变成事实**，且无法回滚 |
| 插 `relations` 但 `confidence` 很低 | `relations.confidence` 语义是"来源置信度"（人工/词典），与模型分数混在一起会污染审计；`views`/触发器也不区分来源 |
| **独立表 `inferred_relations`** ✅ | 命名空间隔离：不被任何视图引用、不进闭包、不参与 `validate_*` 计数；可按 `model_version` 整批回滚 |

### 4.3 晋升路径（promotion）

推断结果只有经过显式动作才能进入事实层：

```
inferred_relations (verified=0)
        │  人工/规则复核（verified=1）
        ▼
INSERT INTO relations (..., source_id = <新增的 'Logos Transformer' source>)
        │
        ▼
tr_closure_edge_ai 自动触发 → closure 物化（此时才成为"事实"）
```

建议在 `sources` 表新增一行 `('Logos Transformer', '模型推断', ...)`，
让每条晋升后的关系都可追溯到模型版本。

> **状态**：晋升这一段**尚未实现**，而且刻意不做。写入端（`infer_and_write.py`）
> 只到 `inferred_relations` 为止；`verified` 目前恒为 0，没有任何自动置 1 的路径。

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

> **已实现替代品**（2026-11-02）：写入端 `infer_and_write.py`、只读查询端
> `query_inferences.py`，接口见 §10。二者都不写 `relations`。

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
- **关系条件化打分头 `W_r` 已实现**：`logos_transformer/train_closure_relaware.py` 为 46 种关系各建一个 32×32 矩阵，稀有/未见关系回退到共享矩阵，见 §9.10。扩展后的数据集有 6 个关系获得专属训练（`is_a` / `can_do` / `has_property` / `typically_has` / `part_of` / `has_part`），test 覆盖 6 个关系（§9.11）。
- **单关系隔离与梯度掩码已实现**（§9.11）：`--relation-subset` 把切分/负样本池/头/指标收窄到指定关系；`--grad-mask` 按批次清零无正样本的关系头梯度；`--early-stop rel_mrr` 支持按主关系 val MRR 早停；`--grad-report-every` 输出逐头梯度幅度诊断。四个开关默认关闭，不带开关时与旧版本逐位一致。
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
这是本实验最明显的简化。关系条件化的 `W_r` 已在 §9.10 实现；扩展数据集复测（§9.10.1）
中 5 个关系获得专属训练、test 覆盖 4 个关系，`is_a` / `has_property` 显示出明确高于
target prior 的排序能力，`can_do` 仍缺 test 证据。扩展前的单关系数据（240/241 为
`is_a`）无法区分关系条件化与单一矩阵。

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
- **头与关系无关**（§9.2–§9.9 的 `W`）；§9.10 已实现关系条件化 `W_r`。在扩展数据集
  （§9.10.1）上 5 个关系有专属头、test 覆盖 4 个关系；扩展前的数据只有 `is_a`
  闭包正样本，无法评估关系多样性。
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

### 9.10.1 扩展数据集复测（250 义项 / 1090 闭包，2026-10-26）

动物库扩展后在同一脚本上重跑（`--epochs 200`，66 epoch 早停；best val loss
**0.2981** @ epoch 46，best val margin **+0.3360** @ epoch 46）。图规模 250 义项 /
477 条义项级关系 / 1204 条消息边 / 1090 条闭包（depth=1 452，depth>=2 **638**），
关系类型 **46** 种。这次复测的关键变化是**闭包标签终于覆盖多种关系**：

| relation | train 正样本 | depth>=2 行数 | 有效头 |
|---|---:|---:|---|
| `is_a` | 228 | 317 | 专属 `W_r` |
| `can_do` | 124 | 178 | 专属 `W_r` |
| `has_property` | 55 | 75 | 专属 `W_r` |
| `typically_has` | 44 | 58 | 专属 `W_r` |
| `part_of` | 5 | 9 | 专属 `W_r` |
| `has_part` | 1 | 1 | gated `W_r` + `W_shared` |
| 其余 40 种（含 `opposite_of`） | 0 | 0 | `W_shared` |

`opposite_of` 有 7 条闭包但全部是 depth=1，因此不是闭包预测目标，在逐关系
test 表中被跳过（不记为 0）。关系头参数 48,174（46 个 `W_r` + `W_shared` + 门控），
模型总计 258,542 参数。

逐关系 test 结果（full = 全库过滤排序；hard = 全部合格硬负样本）：

| relation | test n | full MRR | hard MRR | hard margin | 判读 |
|---|---:|---:|---:|---:|---|
| `is_a` | 44 | **0.5634** | 0.9470 | +0.3713 | 稳定，仍是最强头 |
| `has_property` | 3 | **0.6111** | 1.0000 | +0.3889 | 非随机，但 n 很小 |
| `typically_has` | 1 | **1.0000** | 1.0000 | +0.7500 | 主要证据来自 train/val |
| `can_do` | 7 | **0.3563** | 0.8571 | −0.1429 | 硬池强，但 test 上没有超过 prior |
| `part_of` | 0 | — | — | — | 仅 5 个 train 正样本 |
| `has_part` | 0 | — | — | — | 1 个 gated 正样本 |
| `opposite_of` | 0 | — | — | — | 0 条 depth>=2 |

train/val 的同口径结果说明这些头确实在学习，而不是 test 噪声：`has_property`
train/val full MRR 为 0.5163 / 0.5502，`typically_has` 为 0.6386 / 0.6930，
`is_a` 为 0.5831 / 0.4970，`can_do` 为 0.3797 / 0.3734（train/val 硬池 MRR 均为
1.000，但 test 的 7 个正样本没有体现出独立于 prior 的增益）。

整体对比（**注意：test 组成已变，不是配对比较**）：

| 指标 | 扩展前（170 / 45 / 371） | 扩展后（250 / 46 / 1090） |
|---|---:|---:|
| test 正样本 / source 数 | 28 / 15 | 55 / 20 |
| test full MRR | **0.6889** | **0.5476** |
| Hits@1 / @3 / @10 | 0.5000 / 0.8571 / 0.9643 | 0.3455 / 0.7273 / 0.9455 |
| hard-neg MRR | **0.9688** | **0.9394** |
| hard-pool margin | **+0.2106** | **+0.3137** |
| full-catalogue margin | **+0.2032** | **+0.3530** |
| `猫 -> 生物` full rank | #12 | **#2** |

pooled MRR 的下降主要来自 test 组成（新增 `can_do` 等低 MRR 关系、候选节点
170→250、hard K 7.5→11.0），而不是关系头失效；隔离目标频率后的 **margin 在两种
协议上都明显上升**。因此“关系感知头只在 `is_a` 上可测”这一限制已经解除：
5 个头获得专属训练，test 覆盖 4 种关系，`is_a` 与 `has_property` 有清晰的
prior-relative 收益，`typically_has` 在 train/val 上有效，`can_do` 尚缺 test 证据。

限制：扩展前的 hard-negative checkpoint 与 250 节点图不兼容
（`node_embedding` 170×32 vs 250×32），本次报告的自动基线对比为 `n/a`；上表旧值
来自旧 `training_report.txt`，仅作方向性参照。严格的 head 消融需要在扩展图上重跑
`train_closure_hardneg.py`。

完整逐关系表、逐深度表与预期核对见
[`RELATION_AWARE_COMPARISON.md`](../RELATION_AWARE_COMPARISON.md) §1b。

### 9.11 弱关系扩容 + 训练动力学（335 义项 / 1794 闭包，2026-11-02）

§9.10.1 的结论是「弱关系 test n 太小，既不能证实也不能证伪」。本轮两件事同时做：
**数据侧**补 `seed_animal_kingdom_extended.sql`，**训练侧**给脚本加单关系隔离与
梯度掩码。

**数据侧。** 新种子 ID 段 5081–5165（85 义项，纯追加），原则是「造新源」而不是
把边堆在少数源上 —— source-disjoint 切分下 `test n ≈ 0.15 × 该关系独立源数`。
depth>=2 闭包：`can_do` 178→**529**、`has_property` 75→**129**、
`typically_has` 58→**77**、`part_of` 9→**38**、`has_part` 1→**9**，
五个目标全部达标。全库变为 335 义项 / 694 关系 / 1794 闭包（d1 664 / d2 1130）/
1713 消息边。test 逐关系正样本：`can_do` 7→**59**、`has_property` 3→**16**、
`typically_has` 1→**13**、`part_of` 0→**2**、`has_part` 0→**1**。

**训练侧新增四个开关**（全部默认关闭，不带开关时与旧版本逐位一致）：

| 开关 | 语义 |
|---|---|
| `--relation-subset REL...` | 正样本 / 切分 / 负样本池 / `W_r` 头 / 指标全部收窄到子集；未知关系名报错并列出可用关系 |
| `--grad-mask` | backward 之后、step 之前清零「本批无正样本」的关系头梯度；`W_shared` 仅当整批没有共享路径关系时清零 |
| `--early-stop rel_mrr` + `--primary-relation` | 用主关系 val 硬池 MRR 早停，单独存 `..._best_<rel>.pt` 并优先用它做最终模型 |
| `--grad-report-every N` | 每 N epoch 打印各头平均 `|grad|`（默认 10） |

**实测要点**（完整表格见 `RELATION_AWARE_COMPARISON.md` §1c）：

| 运行 | epochs | test full MRR | test hard MRR | hard margin |
|---|---:|---:|---:|---:|
| hardneg 基线（扩展图重训，配对基线） | 24 | 0.2793 | 0.6883 | +0.0906 |
| C1 全量 relaware | 33 | **0.3609** | 0.6938 | **+0.1623** |
| D1 全量 + `--grad-mask` | 33 | 0.3605 | **0.7061** | **+0.1911** |
| C2a 隔离 `can_do` | 44 | 0.2445 | **0.8923** | **+0.1560** |
| C2b 隔离 + mask + rel_mrr 早停 | 41 | 0.2446 | **0.9000** | **+0.1641** |

三条与直觉相反的发现：

1. **`W_shared` 从未收到梯度。** 6 个有正样本的关系全部持有专属 `W_r`，没有任何
   批次走共享路径。所以「未训练关系经共享头注入噪声、拖累 `can_do`」的假说在本图
   上不成立；干扰来自**共享 GAT 编码器 + 联合优化**，这正是隔离后 `can_do`
   hard margin 从 +0.0003 涨到 +0.1560 的原因。
2. **弱关系的梯度没有被 `is_a` 淹没。** 解冻后各头量级同为 1e-3，epoch 30 时
   `can_do`（1.52e-03）已超过 `is_a`（1.19e-03）。瓶颈是样本量与负样本池，不是
   梯度尺度。
3. **梯度掩码在单关系子集下是严格 no-op**（两次运行逐位一致、`zeroed=0`）；它的
   收益只在联合训练里出现，且集中在弱关系（`has_property` margin +0.1979→+0.3492、
   `typically_has` +0.0017→+0.1730）。

三条未达标 / 需注意的口径：

- `part_of` test n 仍只有 2（目标 ≥5）。38 条 depth>=2 分散在 16 个源上，test 只
  分到 2 个源 —— 这是切分口径的硬约束，只能靠继续增加 part_of 的**独立源**解决。
- `can_do` 的 **full-catalogue** MRR 为 0.2445，未达「> 0.4」；达到 >0.4 的是
  hard-pool MRR（0.8923）。且隔离跑的硬负池对频次先验已饱和（基线 hard MRR
  1.0000、prior 1.0000），该池上绝对差值不可读，只能读 prior-relative margin。
- `has_property` / `typically_has` 的 test n 虽从 1–3 涨到 13–16，仍偏小；
  `typically_has` 在 §9.10.1 里的 1.0000 是小样本幻觉，本轮回落到 0.2433。

复现命令见 `RELATION_AWARE_COMPARISON.md` §1c.7。训练产物目录
`logos_transformer/checkpoints*/` 已加入 `.gitignore`。

---

## 10. 端到端推理流水线（End-to-End Inference Pipeline，2026-11-02）

在此之前，整条链路是「训练 → 评估 → 报表」，**模型从不产出任何可被数据库看见的东西**。
§10 补上最后一段：从 checkpoint 到 `inferred_relations` 的写回，以及只读复核。

### 10.1 数据流

```
logos.db ─┬─ senses / relations / closure ─→ 图导出（prototype_gat.load_graph）
          │                                            │
          ├─ closure（真值）                           ▼
          │     │                          GAT 前向（2 层 4 头，hidden=32）
          │     │ 排除「已存在」                        │
          │     ▼                                      ▼
          └─→ 候选生成 A/B ──→ 词性守卫 ──→ σ(h_sᵀW_r h_t) ──→ 阈值 + 每组 top-K
                                                              │
                                                              ▼
                                                   inferred_relations（假设）
                                                              │
                                            query_inferences.py（只读复核）
```

其中 **`closure` 是事实，`inferred_relations` 是假设**，两者物理隔离：

| | `closure` | `inferred_relations` |
|---|---|---|
| 来源 | `tr_closure_edge_ai` 触发器由真实 `relations` 边物化 | 模型打分 |
| 可复现性 | 重跑种子即可完全重建 | 依赖 checkpoint，重建数据库即丢失 |
| 是否参与视图 | 是（多处视图/校验引用） | 否（不被任何视图或 `validate_*` 引用） |
| 可信度语义 | 事实 | `verified=0` 即「未复核的假设」 |

设计理由见 §4.2：若直接插 `relations`，触发器会立刻把假设变成事实且无法回滚。

### 10.2 候选生成：绕开 O(N²×R)

朴素做法是把 335 × 335 × 46 ≈ 5.16M 个三元组全部打分。实际只对 **closure 真正派生过的
6 个关系** 打分（`closure` 里存在 depth≥2 行的关系：`is_a` / `can_do` /
`has_property` / `typically_has` / `part_of` / `has_part`），再加两条结构策略：

| 策略 | 做法 | 规模（实测） |
|---|---|---|
| **A 邻域二跳** | 对每个有直接边的义项，取无向 1 跳邻居再扩一跳，剔除自身与 1 跳节点，对「该源已参与的每个关系」打分 | 扫描 212 个源 → 10,604 条 |
| **B 直接边补全** | 对 `relations` 里存在、但该源在该关系上**没有任何 depth≥2 闭包行**的 (源, 关系) 对，对全部 335 节点打分 | 87 对 × 335 → 28,863 条 |
| 去重后 | A ∪ B | **37,611 条**（vs 673,350 = 335²×6，压缩 18×） |

A 覆盖「多跳可达但闭包没算出来」的目标；B 覆盖「整个关系的继承都缺失」的源——后者是
闭包引擎结构上算不出来、但模型仍可能给出假设的部分。`--max-neighbourhood`
（默认 60）对 A 的二跳集合按节点度数截断，避免枢纽节点（「生物」「抽象概念」）淹没候选池。

### 10.3 打分

```
h  = GAT(edge_index, edge_type)              # 全图一次前向，(N, 32)
σ  = sigmoid( h_sᵀ · W_eff(r) · h_t )        # 与训练时 score_pairs 完全同一算式
```

* `W_eff(r)` 由 checkpoint 自带的 `relation_count` 决定：`≥5` 个训练正样本 → 专属
  `W_r`；`0 < n < 5` → 门控混合；`n = 0` → `W_shared`。
* 实现上用 `score_all(h, r) = h W hᵀ` 一次算出 (N, N) 矩阵再按候选索引取值，
  6 个关系共 6 × 112,225 个浮点数，比分对 einsum 快一个量级。
* **排除规则**：`(源, 关系, 目标)` 只要在 `closure` 里出现过（任何 depth）就跳过。
  因此「猫 is_a 生物」这类已成立的事实**不会**出现在 `inferred_relations` 里。

### 10.4 质量闸门（本节的实测教训）

只按 σ ≥ 0.8 过滤是不够的。首次运行（无闸门）写出 **1,220 行**，其中
`蛇 is_a 潜水`（0.9997）、`犬科 has_property 潜水`（1.0000）这类明显荒谬的假设占了主流。

**根因**：训练用的负样本来自 hard/medium/easy 池，池内负样本与正样本**同支同词性**
（动物 vs 动物、动词 vs 动词）。模型从没见过「(动物, 动词) 在 `is_a` 下是负样本」，
于是这个方向从未被压制，σ 在 OOD 目标上直接饱和到 1.0。训练报表里的 MRR / margin
只度量**池内排序**，完全看不见这类错误——这正是「从不写回」掩盖掉的问题。

前两道闸门默认开启、判据全部来自数据库自身，不含硬编码：

**闸门 1：目标词性守卫（默认开启）**

`closure` 里该关系实际收过的目标词性分布（`load_relation_target_pos`），
占比 ≥ `--pos-min-share`（默认 0.05）者进入白名单：

| relation | 闭包目标词性分布 | 白名单 |
|---|---|---|
| `is_a` | n:100% | n |
| `can_do` | v:88% n:12% | v, n |
| `has_property` | a:82% n:18% | a, n |
| `typically_has` | n:92% a:8% | a, n |
| `part_of` / `has_part` | n:100% | n |

> `can_do` 的 n:12% 是 `fix_pos_annotations.sql` 把「防御」改标成名词后出现的
> （它是 18 条 `can_do` 边的目标）。改标前是 v:100%，剔除 13,702 条
> （37,611 → 23,909）；改标后 13,615 条（→ 23,996）。**写入行数不变（154 → 154）**，
> 变的只是白名单里多了一个 n。详见 §10.10.5。

剔除 `is_a → 潜水` 这一整类错误。

**闸门 2：每个 (源, 关系) 只留 top-K（默认 3）**

模型是按**排名**（MRR / margin）验证的，从未做概率校准；σ 在 0.99 附近饱和，
第 1 名与第 7 名的数值差不可读。可行的兜底是限制列表长度而非相信绝对值：
267 行 → **154 行**（79 个 (源, 关系) 组）。

**闸门 3：符号守卫（`--source-pos-guard` / `--hierarchical-guard` /
`--leaf-target-guard`，全部 opt-in，默认关）**

前两道闸门之后剩下的 154 行里，仍有 `皮肤系统 part_of 猴子`、
`医学检测 is_a 影像特征`、`猫 has_property 生物` 这类错误。它们不再落在词性轴上，
而是落在**层级与方向**上。第三道闸门用 `closure` / `relation_types` 里的结构信息
把它们拦掉：154 → **118 行**。判据、实测与漏网分析见 §10.10。

最终实测（三道闸门全开）：**118 行写入**，分布 `typically_has` 54 / `can_do` 27 /
`has_property` 15 / `part_of` 12 / `is_a` 10，置信度 mean 0.8839、min 0.8028、
max 0.9999。闸门只开前两道时是 154 行（`typically_has` 57 / `has_property` 30 /
`is_a` 28 / `can_do` 27 / `part_of` 12，mean 0.8869）。

### 10.5 `depth_estimate` 的口径

估计跳数，**先找同关系的路径**（闭包正是同关系组合），找不到再退回无向全关系路径
（GAT 的消息图是全关系连通的，跨关系路径也是分数的一种可能解释），仍找不到则写
`NULL` 而不是硬填：

```sql
depth_estimate = min(同关系有向 BFS 距离, 3)      -- 3 表示 3+
              or min(无向全关系 BFS 距离, 3)
              or NULL
```

16 行是 `NULL`，且**全部来自策略 B**——它们的源（`营养素`、`脑电图信号`、`恶性结节`、
`信息人造物`、`物理人造物`、`医学概念`）在该关系上既没有同关系路径，也没有通往目标的
无向路径，属于「纯嵌入假设」，是最需要人工复核的一批。
草案里的 `path_json`（见证路径）被弃用：模型不会产出路径，硬造一条会误导复核者。

### 10.6 实测：本流水线的输出质量

**通过的**：`猫 --is_a--> 生物` 在全量 335 节点上 **rank 1/335，p=1.000**（训练报表的
验收项，`--dry-run` 的「抽查」块会直接打印）；`人 can_do 潜水`（0.9999）、
`鸟类 typically_has 尾巴`（0.896）等是合理的。

**不通过的（必须说清）**：

| 现象 | 例子 | 说明 |
|---|---|---|
| 关系方向混淆 | `皮肤系统 part_of 猴子` | `part_of`/`has_part` 互为逆关系，模型分不清主体与部件 |
| 常识错误 | `蚯蚓 has_property 有脊柱`（0.966）、`蛇 has_property 温血`（0.986） | 结构合法（a 类形容词）但语义错 |
| 源侧词性未守卫 | `防御 can_do 潜水`（0.999） | 闸门 1 只管目标；`防御`被标成 `v`，属数据标注问题 |
| σ 不可当概率 | 置信度前 20 行全部 ≥ 0.988，其中 16 行 > 0.99 | 见 §10.4 根因；数值无法区分对错 |

结论：**`inferred_relations` 是待复核假设池，不是知识**。`verified` 恒为 0，
没有任何自动晋升路径（§4.3）。要让精度可用，正确方向是补「跨词性负样本」重训
（§5.2 的负样本池设计），而不是继续调阈值。

> **后续（已做）**：跨词性负样本与温度标定见 §11。结论是 cross-POS 负样本确实能把
> OOD 拒绝率从 88% 提到 100%（剂量足够时）且 MRR 不退化，但**写入量不降反升**；
> 温度标定在这个模型上几乎无事可做。上表四类错误里，只有「关系方向混淆」这一类
> 与词性轴相关，其余仍留在原处 —— 它们不是校准问题。

### 10.7 入口与命令

```bash
# 只打分，不改库（不建表）
python logos_transformer/infer_and_write.py --dry-run --min-confidence 0.8

# 写库 + SQL 复核
python logos_transformer/infer_and_write.py --min-confidence 0.8 --verify

# 只推断指定关系；关掉闸门看裸分布
python logos_transformer/infer_and_write.py --relation-subset can_do is_a --dry-run
python logos_transformer/infer_and_write.py --dry-run --no-pos-guard --max-per-source-relation 0

# 符号守卫（§10.10，全部 opt-in）
python logos_transformer/infer_and_write.py --dry-run \
    --source-pos-guard --hierarchical-guard --leaf-target-guard
python logos_transformer/infer_and_write.py \
    --source-pos-guard --hierarchical-guard --leaf-target-guard --verify

# 词性标注修正（幂等 UPDATE，改 senses.pos）
python logos_transformer/infer_and_write.py --fix-pos --dry-run

# 只读复核
python logos_transformer/query_inferences.py --stats
python logos_transformer/query_inferences.py --source 猫 --min-confidence 0.5
python logos_transformer/query_inferences.py --relation is_a --unverified-only
```

关键默认值：`--min-confidence 0.8`、`--max-per-source-relation 3`、
`--pos-min-share 0.05`、`--leaf-min-share 0.95`、`--strategies A B`、
`--max-candidates 200000`（安全阀）。三个守卫开关默认全关 ——
不传任何新参数时，行为与 §10.4 闸门 1/2 的版本逐位一致（实测同为 154 行）。

### 10.8 checkpoint 解析（一个必须写下来的坑）

任务书指定的路径 `checkpoints_relaware/closure_gat_relaware_best.pt` 已被
C2a「隔离 `can_do`」实验**覆盖**：该目录现在只有 **1 个 `W_r` 头（can_do）**，
215,139 参数。用它做多关系推理时，`is_a`/`has_property` 等全部落到 `W_shared`，
打分恒为 **0.5000**（`猫 is_a 生物` rank 162/335）——看起来像「模型没学会」，
实际是**装错了权重**。

因此脚本 `--checkpoint` 缺省为 **自动解析**：按「`W_r` 头对目标关系的覆盖率」
在两个候选里挑，并打印选择依据：

| checkpoint | 参数 | `W_r` 头 | 用途 |
|---|---:|---:|---|
| `checkpoints_relaware_ext/closure_gat_relaware_best.pt` | 261,354 | 46 | ✅ 多关系推理默认选它（C1 全量重训） |
| `checkpoints_relaware/closure_gat_relaware_best.pt` | 215,139 | 1（can_do） | 隔离实验产物，只适合作单关系对照 |

架构信息（层数/头数/hidden/节点数/头名/`relation_count`）全部从 `state_dict` 的形状
与键名反推，不读任何外部配置；节点数或边类型数与当前图不符时**报错退出 1**，
而不是静默跑出一堆无意义分数。

### 10.9 守卫的不变量

1. 推理对 `logos.db` 的读取走 `PRAGMA query_only = ON`；只有写库那一步开独立连接。
2. `relations` / `closure` **一个字节都不改**（实测重跑前后 694 / 1794 不变）；
   `senses` 只在显式给 `--fix-pos` 时被改一行 `pos`（§10.10.5），其余路径不动。
3. 写库全部 `INSERT OR IGNORE`；重跑第二次 `written = 0`、`跳过重复 = 118`（幂等）。
4. `--dry-run` 连表都不建；但**不抑制 `--fix-pos`**（词性修正是显式动作，会写 `senses.pos`）。
5. `model_version` = `logos-gat-closure-relaware-v0.2@<checkpoint 目录名>`，可按版本整批回滚。
6. `epoch` 只有在显式给 `--epoch` 时才有值；checkpoint 是纯 `state_dict`，不含训练元数据
   （写 NULL 而非猜一个数字）。
7. **`inferred_relations` 不参与数据库可重建性**：`python apply_seeds.py`（从零重建）会
   删掉 `logos.db`，假设池随之消失；`--keep` 只重放 seed、不动结构，表与数据保留
   （实测 `--keep` 后表内行数与自检 39 项均不受影响）。重建模型 → 重新推理是预期流程，
   而不是缺陷：假设本来就不该被当成可复现的种子数据。
   `fix_pos_annotations.sql` 同理 —— 重新播种会把「防御」写回 `v`，需再跑一次 `--fix-pos`。
8. 符号守卫（§10.10）**只读**：`senses.pos`、`closure`、`relations`、`relation_types`
   全部按原样读，守卫本身不写任何表。

### 10.10 符号守卫：把「层级 / 方向 / 词性」交回数据库判断

#### 10.10.1 残差的结构

三道闸门的前两道之后，154 行残差按**错误类型**可以分成三类，而这三类都不是
「模型没学好」：

| 类型 | 例子 | 为什么 GAT + `W_r` 学不到 |
|---|---|---|
| 层级/分支错 | `医学检测 is_a 影像特征` 0.9074 | 两者都是 `医学概念` 的直接下位，距离 2 跳、词性同为 n。图上**没有任何一条边**表示「它们是兄弟、不能互为上位」 |
| 方向/逆关系错 | `鳍 typically_has 硬骨鱼` | 库里已有 `鳍 typical_of 硬骨鱼`（鳍是硬骨鱼的典型特征）。方向信息在 `relation_types.inverse_of` 里，不在消息传递里 |
| 语义错（同词性、同层级） | `蛇 has_property 温血`、`蚯蚓 has_property 有脊柱` | 词性对、方向对、目标也是原子值。只有世界知识能否决 |

前两类是**符号层本来就有、但模型看不见**的信息。于是新增四个 opt-in 守卫，
判据全部在运行时从数据库读出，没有一个关系名或词性写死在代码里。

#### 10.10.2 四个守卫

| 守卫 | CLI | 判据来源 | 否决条件 | 候选层剔除 | 写入层砍掉 |
|---|---|---|---:|---:|---:|
| 源侧词性 | `--source-pos-guard` | `relations`+`senses` | 源词性占该关系源侧分布 < `--pos-min-share` | **0** | 0 |
| 层级深度 | `--hierarchical-guard` | `closure`（is_a） | 目标不比源**更特殊**，或源的祖先集不被目标祖先集包含 | 3,162（13.2%） | **18** |
| 方向 | `--hierarchical-guard` | `relation_types.inverse_of` / `is_symmetric` | 逆关系三元组或同关系反向边已存在 | 139（0.6%） | 0 |
| 原子目标 | `--leaf-target-guard` | `closure` + `relations` | 该关系的既有目标 ≥95% 是 is_a 叶子，而候选目标不是叶子 | 3,665（15.3%） | **18** |

（候选层基数为 23,996 条；写入层基数为 154 行。两个守卫的写入层效果**精确可加**：
154 − 18 − 18 = 118。）

#### 10.10.3 层级守卫：先用 152 条真边标定判据，再决定用什么

任务书建议用 `closure` 里的 `MIN(depth)` 作为「分类学深度」。**这个口径在本库不可用**：
一个义项只要有一条直接 `is_a`，它的 `MIN(depth)` 就是 1，于是「深度 1 → 深度 1」
会被判成反向。用 152 条既有直接 `is_a` 边标定，四种判据的通过率是：

| 判据 | 在 152 条真边上的通过率 | 结论 |
|---|---:|---|
| `MIN(depth)[src] > MIN(depth)[tgt]` | **11/152** | 不可用：会把 `猫 is_a 猫科`、`人 is_a 动物` 全判成反向 |
| `|ancestors(src)| > |ancestors(tgt)|` | **152/152** | 采纳（条件 1） |
| `ancestors(tgt) ⊆ ancestors(src)` | **152/152** | 采纳（条件 2） |
| 条件 1 ∧ 条件 2 | **152/152** | **实际使用** |

所以守卫用的是**祖先集大小**而不是最短跳数：`closure` 里一个义项的祖先集合
天然是「它的全部上位」，而「更特殊」的单调度量就是这个集合的基数（加一条
`src is_a tgt` 会让 `tgt` 及其全部祖先成为 `src` 的祖先，严格增）。

这段标定**在运行时每次都会跑并打印**（`guard_calibration_lines`）：如果将来的种子
改变了分类树形状、让判据开始否决真边，输出里会直接标 `<- 不可用（会误杀真边）`，
而不是静默少写几行。

`MIN(depth)` 仍然保留在 `compute_depth_map` 里，作为 `depth_estimate` 列的口径记录
与控制台诊断，但不参与判定。

两条条件各自都在干活，缺一不可：

| 假设 | 祖先集大小 | 祖先集被包含 | 被哪条拦下 |
|---|---|---|---|
| `医学检测 is_a 影像特征` | 2 vs 2（相等） | ✔（两者祖先集完全相同 `{医学概念, 抽象概念}`） | 条件 1 |
| `无脊椎动物 is_a 脊椎动物` | 2 vs 2 | ✔ | 条件 1 |
| `恶性结节 is_a 形态学特征` | 4 vs 3（通过） | ✘（`形态学特征` 的祖先含 `影像特征`，不在 `恶性结节` 的祖先 `{医学概念, 抽象概念, 病理结构, 结节}` 里） | 条件 2 |

`医学检测` 与 `影像特征` 的祖先集**完全相同**，所以「祖先集被包含」是成立的 ——
只留条件 2 会把这条放过；反过来 `恶性结节 → 形态学特征` 的大小条件通过，
只留条件 1 也会放过。两条同时要求才拦得住。

任一方不在 `is_a` 分类树里时守卫**弃权**（放行）—— 树外没有祖先关系可比，
硬判只会制造误杀。当前库里这类节点有 `防御`、`皮肤系统`、`翼膜` 等。

#### 10.10.4 方向守卫：读 `relation_types`，不要从数据里猜

任务书建议「从 `closure` 里统计哪些关系对同时双向出现，把单向的当有向」。
这条路在本库走不通：`part_of` 59 条边、`has_part` 17 条，**逆关系在库里存不存在
是「这条边有没有被断言过」的属性，不是关系的属性**。按观测频率推断方向，
会把「尚未断言的逆关系」误判成「不可能」。

`relation_types` 已经把方向声明好了（`inverse_of` / `is_symmetric` / `relation_class`），
直接读即可。守卫做两件事：

- 声明的逆关系已存在 ⇒ 否决（`鹰 has_part 心脏` 被否，因为 `心脏 part_of 鹰`
  已由 `心脏 part_of 循环系统` + `循环系统 part_of 鹰` 派生进闭包）；
- 同关系反向边已存在 ⇒ 否决（`A is_a B` 而 `B is_a A` 已隐含，会成环）。

方向守卫在候选层拦下 139 条，其中 53 条 `typically_has`、46 条 `has_property`、
21 条 `part_of`、18 条 `has_part`、1 条 `can_do`。**但它一行写入都没砍到** ——
这些错误假设的置信度都排在同组 top-3 之外。它是「防止将来变坏」的守卫，
不是「修当前输出」的守卫。

#### 10.10.5 原子目标守卫与词性修正

`has_property` / `typically_has` 的目标是**原子值**（`有脊柱`、`胡须`、`潜水`），
`is_a` 的目标必然是**类别**。这个差别可以量化：该关系的既有边里，目标为
`is_a` 叶子的占比。实测 `typically_has` 100%（149/149）、`can_do` 100%（146/146）、
`has_part` 100%（17/17）、`has_property` 98.5%（132/134）、`part_of` 93.2%（55/59）、
`is_a` **0%**（0/152）。阈值取 0.95 ⇒ 前四个生效，`part_of` 与 `is_a` 不生效
（`part_of` 的 4 条非叶子目标全是真值：`循环系统/呼吸系统/骨骼系统 part_of 脊椎动物`、
`翅膀 part_of 鸟类`）。

它在写入层砍掉 18 行，全部是 `X has_property 生物` / `X typically_has 生物`
（`生物` 有 39 个下位，是类别不是属性值）。

`fix_pos_annotations.sql`（`--fix-pos`）把「防御」的 `v` 改成 `n`：它在库里是
行为类别名词（`防御 can_do 伪装/变色/喷射推进`），与「运动/捕猎/通讯」同类。
**但这条修正会松动闸门 1**：`防御` 同时是 18 条 `can_do` 边的目标，改标后
`can_do` 的目标词性从 `v:100%` 变成 `v:88% n:12%`，n 越过 5% 门槛进入白名单，
候选池多放行 87 条。**写入行数不变（154 → 154）**。

这个副作用本身就是结论：**`can_do` 目标 100% 是动词这件事，有一部分是那个错误
标注撑起来的**。真正的缺陷是 `can_do` 同时承载了两个方向
（动物→行为类别 / 行为类别→具体动作），单个词性维度承载不了 —— 该拆关系，
而不是继续调阈值。

#### 10.10.6 实测：四配置

| # | 配置 | 候选 | 过阈值 | 写入 | 平均置信度 | `is_a` | `has_property` | `typically_has` | `can_do` | `part_of` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 无守卫 | 37,611 | 1,220 | 477 | 0.9563 | 140 | 142 | 144 | 30 | 12 |
| 2 | 仅目标词性（默认） | 23,996 | 267 | **154** | 0.8869 | 28 | 30 | 57 | 27 | 12 |
| 3 | + 源侧词性 | 23,996 | 267 | 154 | 0.8869 | 28 | 30 | 57 | 27 | 12 |
| 4 | 全部守卫 | 23,996 | 211 | **118** | 0.8839 | 10 | 15 | 54 | 27 | 12 |

（配置 4 的「候选」列仍写 23,996，是因为守卫在打分之后应用 —— 被否决的候选
也一起打分了，这样「不加守卫本会写多少」才是精确对照而不是估算；
守卫放行 17,030 条。）

配置 4 砍掉的 36 行平均置信度 **0.8969**，留下的 118 行是 **0.8839** ——
被砍的比留下的更自信，说明错的就是最自信的那批，不是尾部噪声。

`猫 is_a 生物`（真值，已在闭包中，不会被写入）在四个配置下都是
**rank 1/335，σ = 1.000**，守卫不影响它：守卫只删候选，不改分数。

#### 10.10.7 结构性守卫拦不住什么（边界）

这几条必须写下来，否则「加了守卫」会被误读成「精度问题解决了」。以下假设在配置 4 里
**全部存活**：

1. **`猫 is_a 人`（0.9915）、`鱼 is_a 人`（0.9886）。** `ancestors(人) = {动物, 生物}`
   是 `ancestors(猫) = {动物, 哺乳动物, 猫科, 生物, 脊椎动物}` 的真子集，且 5 > 2。
   在只看闭包的前提下，它和一条真的 `is_a` 边**完全不可区分** —— 数据库里没有
   「猫不是人」这条信息。否决它需要世界知识或互斥公理（disjointness），两者本库都没有。
2. **`鱼 / 青蛙 / 鲤科 is_a 无脊椎动物`（0.8327 / 0.8319 / 0.8036）。** 这三条是典型的
   「同词性、错分支」：祖先集包含成立（`{动物, 生物}` ⊂ 鱼的祖先集）、祖先集大小递增成立
   （7 > 2）。要拦它们必须知道「脊椎动物与非脊椎动物互斥」。
3. **`医学概念 is_a 生物`（0.8545）。** 目标是根：`ancestors(生物) = ∅` 恒被包含，
   祖先集大小 0 < 1 恒成立，所以**任何根目标都能通过**。守卫无法区分
   `生物` 与 `抽象概念` 哪个才是该节点的根。
4. **`皮肤系统 part_of 猴子`（0.8944）。** 任务书猜它是「逆关系方向反了」，实测不是：
   `猴子` 在 `part_of` / `has_part` 上一个出边都没有，没有逆边可撞。而
   `循环系统 part_of 脊椎动物` 是**种子里的事实** —— 同一个模式，差别只在粒度
   （系统属于「类」而不是「个体」）。这是粒度问题，不是方向问题。
5. **`蛇 has_property 温血`、`蚯蚓 has_property 有脊柱`、`防御 can_do 潜水` 全部保留。**
   词性对、方向对、目标是原子值，四个守卫的判据**全部通过**。

因此：守卫清掉的是「目标不是原子值」（18 行）和「祖先关系不成立」（18 行），
`is_a` 从 28 行降到 10 行（其中 4 行是空真 `→ 抽象概念`、6 行仍错），
而 `can_do` 27 行、`has_property` 15 行、`typically_has` 54 行基本未动。
154 → 118 这个量级说明：**精度提升有限，`verified` 依然全 0，仍然没有任何自动
晋升到 `relations` 的路径。** 逐条判定见 `GUARDS_EVALUATION.md` §4。

> 为什么 `is_a` 从 28 掉到 10，而 `can_do` 一行不掉：`is_a` 的目标是**类别**，
> 类别之间有偏序，闭包能表达；`can_do` 的目标是**动作**，动作之间没有偏序边
> （`潜水 ⊑ 运动` 这种关系库里没有），所以没有可用的结构判据。

---

## 11. 校准与 OOD 拒绝（Calibration and OOD Rejection，2026-11-02）

第 10 节的写回流水线暴露了一个排序指标看不见的缺陷：模型在**训练分布之外**的输入上
σ 饱和 —— `蛇 is_a 潜水`（潜水是动词，`is_a` 目标应为名词）拿到 0.9997。
本节记录针对它的两步处方、实测剂量-反应曲线，以及两条**没有做到、且在数学上做不到**
的目标。完整数字见 `CALIBRATION_COMPARISON.md`。

### 11.1 根因：负样本池与正样本同词性

`sample_hardneg` 的 hard/medium 两个池子都只从**同一关系的正样本目标**里取，
于是「负样本目标」永远和「正样本目标」同词性。模型从未见过
「(动物, 动词) 在 `is_a` 下是负样本」，也就没有任何理由把那个方向压下去。

而 MRR / margin 只在这个池子内部排序，所以它们一直健康 ——
**这正是「从不写回」所掩盖的问题**：指标度量的是池内排序，缺陷出现在池外。

### 11.2 处方一：cross-POS 负样本

对每条正样本 `(src, dst, r)` 增加第 4 档负样本：目标词性是该关系在 `closure` 里
**从未接收过**的（`is_a` → 动/形容词，`can_do` → 名/形容词，`has_property` → 动词）。

- 词性白名单**从 `closure` 读出**（占比 ≥ `--pos-min-share`，默认 0.05），
  与第 10 节推理期的类型守卫**共用同一个函数**（`train_calibrated` 直接 import
  `infer_and_write.build_pos_guard`），避免训练/推理两套类型期望。
- **cross-POS 永不用于补位**：hard/medium/easy 互相吸收缺口，cross 缺了就记
  `short_cross`。拿故意违反类型的样本去填 hard 缺口，会让两档语义混在一起。
- 没有词性记录的关系统统跳过（空池 + `zero_cross`），不拿全节点当候选灌噪声。
- 默认配比 `40/30/20/10`，6 条负样本时按最大余数法分到 `2/2/1/1`；
  `--cross-pos-ratio 0` 时比例重归一化成 `3/2/1`，**逐项复现第 9 节基线的全部指标**。

### 11.3 处方二：温度标定

在 val 上用一个标量 T 最小化 NLL，`calibrated = σ(logit / T)`。T 作为 buffer 存进
checkpoint（`CalibratedClosurePredictor` 继承原模型类，**不改**既有训练脚本），
并镜像到同目录 `calibration.json`。

**一个必须写下来的数学事实**：σ 严格单调 ⇒

```
σ(l / T) < 0.5  ⟺  l < 0     （任意 T > 0）
```

所以温度标定**不能**改变 OOD 拒绝率、**不能**改变任何排序（MRR/margin 逐位不变）、
也**不能**把一条正 logit 变成负 logit。它只能改概率数值本身。

### 11.4 指标口径

| 指标 | 定义 | 口径陷阱 |
|---|---|---|
| ECE | 按 `max(p,1−p)` 等宽分箱，Σ(n_b/N)·\|acc−conf\| | 分母是**该 split 的样本表长度**，`--neg-per-pos` 不同则不可比 |
| reliability | 分箱 → (n, 平均置信度, 准确率) | 空箱丢弃 |
| OOD rejection | 探针里 σ < 0.5 的比例 | 探针由 **held-out（test）源** 构造，与被测模型训练源不相交 |

因为 in-dist ECE 的分母会随负样本数变化，**跨配置只比 `ECE shifted`**
（恒为 141 条 test 正样本 + 144 条探针 = 285）。

### 11.5 实测（四组，除标注外全部同图同切分同探针）

| # | 每正样本 hard/med/easy/cross | test MRR | hardMRR | margin | ECE test | ECE shifted | OOD 拒绝 | T | 写入@0.8 |
|---|---|---|---|---|---|---|---|---|---|
| A 原模型 | 3/2/1/0 | 0.3609 | 0.6938 | +0.1623 | 0.0411 | 0.0728 | 0.8819 | — | 154 |
| A′ `--cross-pos-ratio 0` | 3/2/1/0 | 0.3609 | 0.6938 | +0.1623 | 0.0411 | 0.0728 | 0.8819 | 1.000 | 154 |
| B 默认（cross 1/6） | 2/2/1/1 | 0.3694 | 0.6865 | +0.1534 | 0.0252 | 0.0546 | 0.9653 | 1.1088 | 207 |
| C cross 3/6 | 1/1/1/3 | 0.3648 | 0.6935 | +0.1634 | 0.0271 | 0.0303 | **1.0000** | 1.1281 | 268 |
| D `--neg-per-pos 9` | 3/3/2/1 | 0.3529 | **0.7136** | **+0.1690** | 0.0393※ | 0.0517 | 0.9722 | 1.000 | 182 |

※ 分母 1410（非 987），不与上三行直接比。

**A′ 是同口径对照**：关掉 cross-POS 后六项指标逐位等于原模型，证明 A→B/C 的差异
全部来自 cross-POS，而非「换了脚本」或「配比变了」。

剂量-反应（只改 cross 配额）：

| cross 占比 | `蛇 is_a 潜水` | `犬科 has_property 潜水` |
|---|---|---|
| 0/6 | +8.291（0.9997） | +8.513（0.9998） |
| 1/6 | +6.806（0.9989） | +5.884（0.9972） |
| 3/6 | **+3.055（0.9550）** | **−1.172（0.2365）** |

方向正确、单调，但斜率低：把 `蛇 is_a 潜水` 压到 <0.3 需要比 3/6 更强的剂量，
而 3/6 已把 hard 负样本从 3 条压到 1 条。

### 11.6 三条与预期相反的结论

1. **温度标定在这里几乎无事可做。** raw 最优 T 在 0.87–1.13 之间（A′/D 甚至 < 1，被
   `--min-temperature 1.0` 截断成 no-op）。加上 T 后 ECE **反而略升**
   （B 0.0252→0.0260，C 的 shifted 0.0303→0.0471）：NLL 最优 ≠ ECE 最优，
   在样本量小且模型本已接近校准时分箱噪声压倒位移收益。
2. **写入量不降反升（154 → 182/207/268）。** 两个机制叠加：
   (a) 固定预算 `--neg-per-pos 6` 下 cross 挤掉 hard（D 保住 hard=3 后回落到 182）；
   (b) bilinear 打分没有 bias 项，压低「错词性」一侧必然**整体抬高同词性一侧**，
   而阈值 0.8 附近的候选恰恰是同词性的 hub 名词。
3. **残留错误不在词性轴上。** `医学检测 is_a 影像特征`、`防御 can_do 潜水`
   （`防御` 被标成 `v`，数据问题）、`人 can_do 跃出水面` —— 全是同词性错语义、
   源侧标注错或跨域借用。继续加 cross-POS 剂量只会继续抬高同词性一侧。

### 11.7 入口

```bash
python logos_transformer/train_calibrated.py                    # 默认 cross 1/6
python logos_transformer/train_calibrated.py --cross-pos-ratio 0 # 复现基线（= A′）
python logos_transformer/train_calibrated.py \
    --cross-pos-ratio 0.5 --hard-ratio 0.25 --medium-ratio 0.15 --easy-ratio 0.1
python logos_transformer/infer_and_write.py --calibrated --dry-run
python logos_transformer/infer_and_write.py --calibrated --temperature 1.5 --dry-run
```

产物（`checkpoints*/` 已在 `.gitignore` 内）：

```
<output-dir>/closure_gat_calibrated_best.pt      权重 + temperature 缓冲
<output-dir>/closure_gat_calibrated_untuned.pt   同权重 T=1，用于 A/B
<output-dir>/calibration.json                    T / ECE / OOD / 指标机读记录
<output-dir>/training_report.txt                 文本报告
```

`infer_and_write.py --calibrated` 的温度解析顺序：`--temperature` > checkpoint 的
`temperature` 缓冲 > 同目录 `calibration.json` > 退回 1.0 **并打印警告**。
装载时按模型自身的键表过滤 state_dict（标定 checkpoint 多一个 `temperature` 键），
被丢弃的键会打印出来，不静默吞掉。

### 11.8 已知边界

1. `--calibrated` 是**单调变换**，写入量随 T 上升而下降，但下降的「对错比例」不变 ——
   报告里同时打印未标定/已标定的行数，并注明「条数下降不代表错误变少」。
2. OOD 拒绝率只对**同一构造方式**的 cross-POS 探针有意义，不能外推成「通用 OOD 检测」。
3. 探针只有 144 条（23 个 held-out 源 × 各关系 4 条），单个百分点的分辨率约 ±3%。
4. 温度在 val 的 NLL 上拟合，但 ECE 在 test 上度量；两者最优 T 不同，故 ECE 可能不降。

### 11.9 与第 10 节符号守卫的分工

两者解决的是**正交的两类**问题，不能互相替代：

| | §11 校准 / cross-POS | §10.10 符号守卫 |
|---|---|---|
| 作用对象 | 分数（`σ` 的数值） | 候选（三元组的去留） |
| 能否改变排序 | 不能（T 单调） | 能（直接删候选） |
| 能否改变写入行数 | 能（把行推到阈值另一侧） | 能（否决该行） |
| 修的是哪种错 | 跨词性（`蛇 is_a 潜水`） | 跨层级 / 反向 / 类型（`医学检测 is_a 影像特征`） |
| 判据是否来自数据库 | 否（来自训练分布） | 是（`closure` / `relation_types` / `senses.pos`） |
| 实测写入量（阈值 0.8） | 154 → 182/207/268（**上升**） | 154 → **118**（下降） |

校准那一轮没能降低写入量，根因是它在**分数**这一层做单调变换；守卫是直接在
**候选集合**上做删减，所以能真正减少行数。反过来，守卫对 `蛇 is_a 潜水`
这类跨词性错误完全无能为力 —— 那需要重新训练（§11.2）。

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
