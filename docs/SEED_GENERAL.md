# seed_general_v3.sql —— 通用领域种子数据说明

> v0.1 demo 的最小可用数据集。目标：跑通「猫吃鱼」及其 4 个扩展句式
> （把字句 / 被字句 / 话题句 / 存现句）。
>
> 验证状态：**`validate_seed.py` → PASS 77 / FAIL 0**（本文件对应的 §1-§6 共 65 项 +
> 扩展种子 `seed_general_v3_ext.sql` 的 §7 共 12 项；全量回归见文末）

---

## 1. 运行方式

```bash
# 必须按顺序：结构 → 数据
python - <<'PY'
import sqlite3
c = sqlite3.connect("logos.db")
c.execute("PRAGMA foreign_keys = ON")
c.execute("PRAGMA recursive_triggers = ON")
c.executescript(open("schema_v3.sql",       encoding="utf-8").read())
c.executescript(open("seed_general_v3.sql", encoding="utf-8").read())
PY

python validate_seed.py     # 77 项自检（会自动加载同目录的 seed_*.sql）
```

两点必须注意：

1. **`schema_v3.sql` 必须先跑**。种子文件是叠加式的，它不建任何表；且它引用
   schema 已种入的既有词条（猫/鱼/动物/把/只/看/他/桌子）。
2. **`PRAGMA foreign_keys` 必须在每个连接上重设**（SQLite 默认 OFF，PRAGMA 不随
   schema 持久化），且**必须写在 `BEGIN` 之前** —— 事务内执行是 no-op。本文件已
   自带这两条 PRAGMA，通过 `validate_seed.py` 的独立连接用例验证过自足性。

---

## 2. 为什么不能硬编码 ID

`schema_v3.sql` 的 §14 已经种入 26 个词 / 27 个义项，其中就有本 demo 需要的
猫、鱼、动物、把、只、看、他。如果种子文件按固定 ID 插入，会撞主键或产生重复词条。

因此本文件采取**双轨寻址**：

| 情况 | 做法 |
|---|---|
| schema 中**已存在**的词条 | 走临时视图 `seed_sid` 按 `(word, pos)` 取 `sense_id`，零硬编码 |
| 本文件**新增**的词条 | 统一使用 `1000` 段（words/senses 均为 1001–1027），与 schema 的 1–27 完全隔离 |

临时视图（仅本会话有效，不落盘）：

```sql
CREATE TEMP VIEW IF NOT EXISTS seed_sid AS
    SELECT w.word, s.pos, s.sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;
```

推论：**改 schema 里任何一个 ID，本种子文件都不会错位。** 这也是它能反复执行而不
出错、不产生重复行的原因。

---

## 3. 写入规模（`validate_seed.py` 实测）

触及 15 张表，合计 **+195 行**。

| # | 表 | 新增 | 内容 |
|---|---|---:|---|
| 1 | `semantic_categories` | +1 | 食物（本体骨架补一个节点，供 `relation_constraints` 取值） |
| 2 | `words` | +27 | 人/我/你/吃/喝/抓/跑/是/有/大/小/好/坏/被/的/了/着/过/在/从/向/条/个/水/食物/生物/游泳 |
| 3 | `senses` | +27 | 与上表 1:1，每词单义项（v0.1 不做多义） |
| 4 | `sense_features` | +22 | 义素：生命度 / 具体性 / 可数性 / 自主性 / 极性 |
| 5 | `sense_categories` | +20 | 义项 ↔ 语义类别（含为既有词条补类别） |
| 6 | `function_words` | +8 | 被/的/了/着/过/在/从/向（把 已存在，UPSERT 补全） |
| 7 | `function_word_roles` | +5 | 被→施事、在→处所、从→起点+时点、向→方向 |
| 8 | `classifiers` | +2 | 条 / 个 |
| 9 | `noun_classifier_pairs` | +4 | 鱼→条、人→个、我→个、你→个 |
| 10 | `classifier_class_defaults` | +1 | 人 → 个 |
| 11 | `relations` | +23 | 5×`is_a` + 8×`can_do` + 6×`agent` + 2×`patient` + 2×`opposite` |
| 12 | `relation_constraints` | +17 | 各动词的 agent / patient / theme / instrument 选择限制 |
| 13 | `slot_constraints` | +17 | 5 个目标句式的槽位约束 |
| 14 | `closure` | +16 | 5 行 `is_a` depth=1 + 11 行**触发器派生** |
| 15 | `utterances` | +5 | demo 语料 |

### 关系网络（实测输出）

```
is_a（闭包，ancestor=上位 ← descendant=下位）
  动物 ← 猫     depth=1        生物 ← 猫   depth=2  ← 触发器派生
  动物 ← 鱼     depth=1        生物 ← 鱼   depth=2  ← 触发器派生
  动物 ← 人     depth=1        生物 ← 人   depth=2  ← 触发器派生
  食物 ← 鱼     depth=1        （鱼 有两个上位 → DAG，不是树）
  生物 ← 动物   depth=1

can_do
  人→吃  人→喝  人→跑  猫→吃  猫→抓  猫→跑  鱼→吃  鱼→游泳

agent / patient （动词 → 典型论元类别）
  吃 -agent-> 动物   吃 -agent-> 人      抓 -agent-> 动物   抓 -agent-> 人
  喝 -agent-> 人     看 -agent-> 人
  吃 -patient-> 食物                 喝 -patient-> 水

opposite（对称，只存一条，视图自动双向展开）
  大 ↔ 小      好 ↔ 坏
```

---

## 4. 一处刻意的设计取舍

需求原文写的是「吃 **has_property** 需要_施事(猫/人) + 需要_受事(食物)」。

本文件**没有**用 `has_property` 承载论元需求，而是拆成两条正交路径：

```sql
-- 路径 A：关系层（供图遍历 —— "谁知道吃什么"）
INSERT INTO relations(...) VALUES (吃, 'agent',   动物), (吃, 'agent', 人);
INSERT INTO relations(...) VALUES (吃, 'patient', 食物);

-- 路径 B：约束层（供解析期合法性校验 —— 驱动 inference_rules #3 / #4）
INSERT INTO relation_constraints(sense_id, slot, kind, value)
VALUES (吃, 'agent',   'category', '动物'),
       (吃, 'agent',   'category', '人'),
       (吃, 'patient', 'category', '食物');
```

理由：

- `relation_types` 中已有专用关系 `agent`（注释：*动词→典型施事类别*）与
  `patient`（*动词→典型受事类别*），语义就是需求想表达的东西；
- 用 `has_property` 表达「需要施事」会造出 `需要_施事` / `需要_受事` 这类**并不
  存在的伪名词**，污染词表；
- 路径 B 是 `inference_rules` #3「施事合法性」/ #4「受事合法性」实际读取的表，
  可被索引、可被规则引擎消费。**语义等价，且真正可推理。**

---

## 5. 数据怎么支撑「跑通猫吃鱼」

合法性判定不是靠字符串匹配，而是查 `relation_constraints` + `sense_categories`：

```
「猫吃鱼」
  吃.agent   = category:动物,人     猫 ∈ {动物}          → ✅ 合法
  吃.patient = category:食物        鱼 ∈ {动物, 食物}    → ✅ 合法（靠 鱼→食物 这次类别）

「鱼吃猫」（负向用例）
  吃.patient = category:食物        猫 ∈ {动物, 人工物}  → ❌ 拦下
```

四个扩展句子各有其数据落点：

| 句子 | 句式 | 依赖的种子数据 |
|---|---|---|
| 猫吃鱼。 | P1 主谓宾 | `吃` 的 agent/patient 约束 + `sense_categories` |
| 猫把鱼吃了。 | P7 把字句 | `function_word_roles(把) → 受事/前置` + `slot_constraints(7,'P','definiteness','definite')` |
| 鱼被猫吃了。 | P8 被字句 | `function_word_roles(被) → 施事/前置` + `slot_constraints(8,'A','feature','+animate')` |
| 鱼，猫吃了。 | P5 主谓谓语句 | `slot_constraints(5,'Topic','pos','n')` + 话题槽 `default_role=话题` |
| 桌上有一条鱼。 | P11 存现句 | `有.subcategory='存现'` + `slot_constraints(11,'V','subcategory','存现')` + 量词搭配 `鱼→条` |

---

## 6. 闭包表：只插 depth=1，depth>1 交给触发器

需求要求「手动插入 depth=1 的直接边，触发器自动维护 depth>1」。实现如下：

```sql
INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
SELECT r.relation, r.dst_id, r.src_id, 1, r.relation_id
  FROM relations r
 WHERE r.relation = 'is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
ON CONFLICT(relation, ancestor, descendant) DO NOTHING;
```

三个设计点：

1. **从 `relations` 反推、并回填 `first_edge`（见证边）**，而不是写死 ID —— 保证闭包行
   与关系边永不脱钩。
2. **`depth>1` 一行都不手写**。`is_a` 在 `closure_policy` 中默认为 `incremental`，
   §9 插入 `is_a` 边时 `tr_closure_edge_ai` 已经自动写出了 depth=1 与 depth=2。
   因此这条语句在默认配置下**命中冲突、是 DO NOTHING 空操作**。
3. 它的价值在于**兜底**：如果运维把 `mode` 切成了 `off`，或在触发器被禁用的导入
   流水线里运行本文件，这条语句保证 demo 的 depth=1 闭包必定存在。
   `validate_seed.py` 同时断言了「存在这条语句」与「没有手工插入任何 depth>1 行」。

> 方向约定（详见 `docs/CLOSURE_DIRECTION.md`）：
> 关系边是 `下位 → 上位`（猫→动物），闭包行是 `(ancestor=上位, descendant=下位)`。
> 一行 `(动物, 猫)` 读作「动物是猫的上位，猫 is_a 动物」。

---

## 7. 验证覆盖（§1-§6 共 65 项，全部为行为测试）

> 另有 `seed_general_v3_ext.sql` 带来的 §1b（非计数前提检查）与 §7（12 项 A/B 断言），
> 见 `docs/SEED_EXT_NOTES.md`。

| 段 | 项数 | 内容 |
|---|---:|---|
| §1 加载与幂等 | 9 | 无错建库 / 二次执行行数不变 / 新增义项数 / 种子自足性 / FK 生效 / **事务配对与落盘** |
| §2 结构覆盖 | 12 | 需求清单逐项核对（名词 9、动词 7、形容词 4、虚词 9、量词 3）+ 量词搭配 + 虚词角色映射 |
| §3 约束完整性 | 11 | `foreign_key_check` / `integrity_check` / 动词约束齐备 / category 值合法 / 端点存在 / 无重复边 / 对称关系无冗余 |
| §4 闭包正确性 | 10 | 方向不变量 I3 / 反向完备性 / depth=2 派生 / DAG 多上位 / 见证边有效 / **与在线递归 CTE 差分对拍** |
| §5 句式覆盖 | 13 | 5 个句式 × （模板占位符 ≡ 槽位集合）+ 每槽位有约束 + 约束值合法 |
| §6 端到端语义 | 10 | **用 `relation_constraints` 判定「猫吃鱼」合法、「鱼吃猫」被拦** + 4 个扩展句式 + FTS 双通道 |

### 四项值得单说的断言

**① 闭包差分对拍。** 用**独立实现的在线递归 CTE** 当物化闭包的 oracle，
而不是"拿闭包验闭包"。这是唯一能发现"触发器漏算/多算"的检验方法：

```
[PASS] 闭包 == 在线递归 CTE（差分对拍，各 8 对）
```

**② 幂等性。** 连跑两次种子，逐表比对行数：

```
[PASS] 幂等：第二次执行种子后全部 15 张表行数不变
```

**③ 种子自足性 + 那个事务陷阱。** 在**不预设任何 PRAGMA** 的独立连接上加载本文件，
断言 `foreign_keys` 最终为 1，且孤儿义项插入被拒绝：

```
[PASS] 种子自足：PRAGMA foreign_keys 写在 BEGIN 之前（事务内 PRAGMA 是 no-op）
[PASS] 独立连接加载后 PRAGMA foreign_keys = 1
[PASS] 加载后 FK 处于生效状态（孤儿义项被拒绝）
```

这条针对的是本代码库记录过的真实坑：为了提速把整个文件包进 `BEGIN...COMMIT`，
却把 `PRAGMA foreign_keys=ON` 写在了 `BEGIN` 之后 —— 事务内该 PRAGMA 是 no-op，
于是 FK 完全失效，种子加载期写入脏数据。本文件把 PRAGMA 放在 `BEGIN` 之前。

**④ 事务配对与落盘 —— 写这个文件时真实踩到的 bug。**

初稿分块写入时漏掉了收尾的 `COMMIT;`，`BEGIN` 一直悬着。**当时 62 项行为测试全绿**，
因为 SQLite 在连接关闭前不会回滚，同连接内读到的数据完全正常 —— 隐式提交把问题
盖住了。只有当这段脚本被当作子步骤嵌入更长的流程、或进程在后续步骤才退出时，
未闭合的事务才会被回滚，数据全部消失。

补的断言：

```
[PASS] 事务语句配对：BEGIN(1) == COMMIT(1)
[PASS] 种子末尾已提交：加载后连接不处于未闭合事务中
[PASS] 落盘验证：重开连接后数据完整提交（senses=54, closure=16）
```

第三条是关键：**换一个连接重新打开文件库再读**。同连接内读得到 ≠ 已提交 ——
前者只能证明语句执行过，后者才能证明事务真的结束了。

---

## 8. 已知边界（v0.1 有意不做）

| 项 | 说明 |
|---|---|
| 一词多义 | 每个新词只建 1 个义项。`senses` 支持多义（`UNIQUE(word_id,pos,sense_no)`），加数据即可，无需改表 |
| 多音字 | 吃/喝/抓 等均单音。多音字需补 `character_readings` 行 |
| 离合词扩展 | 「洗了个澡」已在 schema 种子里；本文件未新增离合词 |
| 体标记的助词二分 | `了` 只建了「完成体」义项；句末语气词「了2」未建（会与动态助词争同一 `sense_id`，需按 `sense_no=2` 另立） |
| 领域库 | 本文件不建 `entity_nodes`，未演示跨域链接（KEGG/Reactome）。加新领域库只需 INSERT 一行 `domains` + `entity_nodes`，不改表结构 |
| 篇章层 | `topic_chains` / `anaphora` 未新增。schema 种子已用 `demo` 文档演示过，本文件聚焦词-义项-关系三层 |

---

## 9. 全量回归

```
validate_schema.py     结果: PASS 102 / FAIL 0
audit_indexes.py       结论: 可删 0 个，保留 1 个近似索引
precheck_schema.py     ✅ 可以建库 —— FAIL 0 项 / WARNING 2 项 / PASS 12 项
validate_seed.py       结果: PASS  77 / FAIL 0     ← 本文件 + seed_general_v3_ext.sql（65 + 12）
```
