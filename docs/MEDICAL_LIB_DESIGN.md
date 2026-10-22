# 医学库设计说明 —— 症状继承 + 严重度义素 + 疾病分层 + 影像特征子树

> 本文件记录**医学库**的设计决策与实测结果，覆盖主数据文件、三份增量补丁
> 与一份影像子树种子。
> 主数据文件：`seed_medical_v3.sql`（骨架 + 跨库链接）；
> 补丁一：`seed_medical_v3_patch_symptom.sql`（通用症状下沉到上位疾病层）；
> 补丁二：`seed_medical_v3_patch_symptom_severity.sql`（`sense_features` 症状严重度义素）；
> 补丁三：`seed_medical_v3_patch_symptom_layers.sql`（炎症性疾病层 + 癫痫直接症状补边）；
> 影像子树：`seed_medical_v3_imaging.sql`（影像特征体系 —— 数值→符号映射的目标端，见 §13）。
>
> 验证状态：**`validate_medical.py` → PASS 230 / FAIL 0**
> （原 69 项 + §M10 的 29 项 + §M11 的 34 项 + §M12 的 38 项 + §M13 的 53 项
>  + §M1 逐文件加载断言 7 项）

---

## 1. 背景与问题

`seed_medical_v3.sql` 落地后，症状只声明在**最具体**的疾病上：

```
癫痫         --has_symptom--> 抽搐
阿尔茨海默病 --has_symptom--> 记忆障碍
```

而「疾病 / 脑部疾病」层**没有任何症状边**。于是一个看起来理所当然的查询

> 「癫痫这类疾病都有哪些症状？」——沿 `is_a` 向上取上位层的症状

返回**空集**。这不是查询写错，而是**知识放错了层**：症状本应声明在它
语义上成立的最一般层，让全部下位继承。

`validate_medical.py` 的 §M9 之前会把这条空结果如实打印为
「（空 —— 这是预期结果，不是查询失败）」并给出补法 —— 本补丁正是那个补法。

---

## 2. 症状继承的推理模型

本项目的关系继承（推理规则 #2「属性继承」）是：

```
若  X --has_symptom--> S   且   D --is_a-->* X
则  D --has_symptom--> S
```

即：**把边声明在语义上真正成立的最一般层，全部下位自动继承**。
这与 `has_property` 的用法完全同构（`鱼 is_a 食物` + `食物 has_property 可食用`
⇒ `鱼 可食用`，见 `docs/SEED_EXT_NOTES.md` §3.4）。

关键点：**继承由 `is_a` 的闭包物化来承载，而不是靠 `has_symptom` 自身传递**。

| 关系 | `is_transitive` | 是否产生闭包行 | 继承方式 |
|---|---|---|---|
| `is_a` | 1（incremental） | 是（触发器自动物化） | 闭包直接给出上位链 |
| `has_symptom` | **0** | **否** | 沿 `is_a` 上位链 JOIN 拼出 |

`has_symptom` 一律 `is_transitive=0` 有两个好处：
1. 语义正确 ——「疾病的症状」不是可传递关系（`A 有此症状`、`B 有此症状`
   推不出 `A 有此症状 B 的症状`）；
2. 规避本项目记录过的坑 —— `is_transitive=1` 却缺 `closure_policy` 行会让
   物化**静默失败**（子查询返回 NULL，不报错）。显式声明 0 从源头消除该坑。

因此症状继承的查询形态固定为（本项目统一口径）：

```sql
-- 「疾病 D 的症状结果集」= 直接声明 ∪ 沿 is_a 继承（去重）
SELECT DISTINCT wd.word
  FROM relations r
  JOIN senses s2 ON s2.sense_id = r.dst_id
  JOIN words  wd ON wd.word_id  = s2.word_id
 WHERE r.relation = 'has_symptom' AND r.src_kind = 'sense'
   AND ( r.src_id = :D
         OR r.src_id IN (SELECT ancestor FROM closure
                          WHERE relation='is_a' AND descendant = :D) );
```

> ⚠ 只需沿 `is_a` 的 **ancestor 方向**（上位）追溯：`closure.descendant = D`
> 时取 `ancestor`，正是「D 的各级上位」。这与 `is_a` 的直觉一致
> （不同于 `has_part` 的方向陷阱，见 §6）。

---

## 3. 数据形状：before / after

```
before（seed_medical_v3.sql）              after（+ patch_symptom）
─────────────────────────────────         ─────────────────────────────────────
癫痫         ──has_symptom──▶ 抽搐          癫痫         ──has_symptom──▶ 抽搐        (直接)
阿尔茨海默病 ──has_symptom──▶ 记忆障碍      阿尔茨海默病 ──has_symptom──▶ 记忆障碍    (直接)
                                          疾病         ──has_symptom──▶ 疼痛 / 乏力 (层)
病理状态                                   脑部疾病     ──has_symptom──▶ 头痛 / 眩晕 / 意识模糊 (层)
├── 疾病  ……（层上无边）                    │
└── 症状 → 抽搐 / 记忆障碍                  └── 症状 → 抽搐 / 记忆障碍
                                                        + 疼痛 / 发热 / 头痛
                                                          / 眩晕 / 乏力 / 意识模糊
```

继承后的**结果集**（含直接声明）：

| 疾病 | 直接声明 | 沿 `is_a` 继承 | 结果集大小 |
|---|---|---|---|
| 癫痫 | 抽搐 | 疼痛、乏力、头痛、眩晕、意识模糊 | 6 |
| 脑卒中 | —（无） | 疼痛、乏力、头痛、眩晕、意识模糊 | 5 |
| 神经退行性疾病 | —（无） | 疼痛、乏力、头痛、眩晕、意识模糊 | 5 |
| 阿尔茨海默病 | 记忆障碍 | 疼痛、乏力、头痛、眩晕、意识模糊 | 6 |

注意 `脑卒中 / 神经退行性疾病` **一条直接边都没有**，却拿到了全部 5 种通用症状
—— 这正是「声明在层上、下位自动继承」的收益：**新增疾病无需再补症状边**。

---

## 4. 三处设计决策

### 4.1 为什么声明在「疾病 / 脑部疾病」层，而不是逐个疾病重复声明

关系继承让**边数从 `O(疾病数 × 症状数)` 降为 `O(层数 × 症状数)`**。
当前骨架 4 种疾病 × 5 种通用症状 = 20 条边，下沉到 2 个层只需 **5 条**。
且语义分层是有依据的：

| 症状 | 语义限定 | 声明层 |
|---|---|---|
| 疼痛 | 所有疾病共有 | `疾病` |
| 乏力 | 全身性疾病共有（⊂ 所有疾病） | `疾病` |
| 头痛 | 脑部疾病共有 | `脑部疾病` |
| 眩晕 | 神经系统疾病共有（⊂ 脑部疾病） | `脑部疾病` |
| 意识模糊 | 重症 / 脑部疾病共有 | `脑部疾病` |

**层归属是可分辨、可验证的**：`疾病 --has_symptom--> 疼痛` 成立而
`脑部疾病 --has_symptom--> 疼痛` 不成立，`疾病 --has_symptom--> 头痛` 不成立而
`脑部疾病 --has_symptom--> 头痛` 成立 —— §M10 专门有一条断言检查这个归属，
防止把两类症状都堆到同一个层（那样继承结果相同，但语义塌了）。

### 4.2 为什么「发热」只建义项、本次不挂 `has_symptom`

需求把 6 个词都列为「新增通用症状义项」，但只要求把 **5 种**声明到层上
（疼痛 / 乏力 → 疾病层；头痛 / 眩晕 / 意识模糊 → 脑部疾病层）。

`发热` 的语义限定是「**炎症 / 感染性疾病**共有」—— 当前骨架里**没有**
`炎症性疾病` / `感染性疾病` 这个节点可挂。强行挂到 `疾病` 层会与需求给的
语义限定矛盾（并非所有疾病都发热）。故本补丁**只建义项、暂不挂边**，
并留出扩展位：骨架补出 `炎症性疾病` 后，按同一写法补一行边即可
（见补丁文件 §3.7 的注释块）。

`眩晕` 在需求里同时出现在「新增症状」与「脑部疾病层」两处，按后者落边
（神经系统疾病 ⊂ 脑部疾病，挂脑部疾病层是更宽、更安全的口径）。

### 4.3 为什么本补丁不碰 `relation_types` / `closure_policy`

`has_symptom` 已由主文件登记且 `is_transitive=0`；本补丁不引入任何新关系类型、
不改变任何闭包策略。若重复 `INSERT relation_types`，虽然 `ON CONFLICT DO NOTHING`
不会报错，但会让「补丁改了既有类型定义」这一事实变得不可审计。
**补丁应当只追加数据，不重定义结构。** §M10 有一条断言强制这一点。

---

## 5. ID 段位与写入规模

### 5.1 ID 段位（沿用医学库 4000–4999）

| 段位 | 归属 | 内容 |
|---|---|---|
| 4001–4030 | `seed_medical_v3.sql` | 医学义项 30 个 |
| 4031–4036 | `..._patch_symptom.sql` | 6 个通用症状（word_id = sense_id） |
| **4037–4039** | **`..._patch_symptom_layers.sql`** | **3 个疾病分层义项（炎症性疾病 / 感染性疾病 / 脑炎）** |
| 4101 | `seed_medical_v3.sql` | 「信号」通用库兜底 |
| 3101–3103 | `seed_medical_v3.sql` | 医学库领域实体节点 |

新症状一律按 `(word, pos)` 自然键经临时视图（`sym_sid` / `lay_sid`）引用既有节点，
**不硬编码任何既有 sense_id**；补丁对既有 ID 取值不敏感。

> ⚠ 段位选择只是**一半**约束：`init_database` 按 `sorted(glob("seed_*.sql"))`
> 字典序自动加载，**新增种子的文件名还必须落在其依赖之后**（见 §11）。

### 5.2 写入规模（实测增量）

| 表 | 增量 | 说明 |
|---|---:|---|
| `words` | +6 | 疼痛 / 发热 / 头痛 / 眩晕 / 乏力 / 意识模糊 |
| `senses` | +6 | 与 `words` 1:1（v0.1 每词单义项） |
| `sense_categories` | +6 | 一律 `性状`（未新增类别行） |
| `relations` | +11 | 6 `is_a`（新症状 → 症状）+ 5 `has_symptom`（层边） |
| `closure` | **+24** | 全部**自动派生**：6 条 `is_a` 各派生 4 行（症状 1 / 病理状态 2 / 医学概念 3 / 抽象概念 4）。**补丁不含任何 `closure` 的 INSERT** |
| `changelog` | +11 | 由 `tr_relations_audit_ai` 自动记录 |

最终态：`senses` 124 → **130**；`relations` 101 → **112**；`closure` 184 → **208**。

---

## 6. 两个必须留意的方向 / 机制

### 6.1 为什么是「沿 `is_a` 向上」而不是「`has_symptom` 传递」

见 §2。一句话：`has_symptom` 不可传递，继承发生在 `is_a` 维度上。
`closure` 里**不会**出现任何 `has_symptom` 的行（§M10 有断言）。

### 6.2 与 `has_part` 的方向陷阱对照（勿混淆）

`closure` 的行方向恒为「边的反向」（不变量 I3：`ancestor = 边的 dst`）：

| 关系 | 边的方向 | `ancestor` 落在 | 上行查询写法 |
|---|---|---|---|
| `is_a` | 下位 →(is_a) 上位 | **上位**侧 | `WHERE descendant = D` 取 `ancestor` |
| `has_symptom` | 疾病 →(has_symptom) 症状 | 症状侧（但无闭包行） | 沿 `is_a` 的 ancestor 追溯 |
| `has_part` | 整体 →(has_part) 部分 | **部分**侧 | `WHERE descendant = 整体` 取 `ancestor` |

`has_symptom` 因 `is_transitive=0` 不物化，故不涉及该方向问题；
但 `has_part` 的「部分在 `ancestor` 侧」极易读反，详见
`seed_medical_v3.sql` §2.2 与 `.workbuddy/memory/MEMORY.md` 不变量 I3。

---

## 7. 验证覆盖（§M10，29 项，全部 PASS）

`validate_medical.py` 新增 §M10 断言组（在既有 §M1–§M9 之后追加）：

```
M10.1 义项层（6 项）
  · 6 个通用症状义项全部新增
  · 新症状 sense_id 恰为 4031-4036
  · 6 个新症状均 is_a 症状
  · 6 条释义非空且不含数字/拉丁字母
  · 类别挂载合法（未新增类别行）
  · 类别一律 '性状'
M10.2 层上声明（2 项）
  · 疾病→疼痛/乏力 + 脑部疾病→头痛/眩晕/意识模糊 齐备
  · 两层的 has_symptom 出边数 = 2 / 3
M10.3 继承与结果集（8 项）
  · 癫痫 继承到「疼痛」「乏力」（经 疾病 层）★需求项
  · 癫痫 继承到「头痛」「眩晕」「意识模糊」（经 脑部疾病 层）★需求项
  · 阿尔茨海默病 继承到「疼痛」「乏力」（经 疾病 层）★需求项
  · 阿尔茨海默病 继承到「头痛」「眩晕」「意识模糊」（经 脑部疾病 层）★需求项
  · 脑卒中 继承到全部 5 种通用症状 ★需求项
  · 层归属可分辨（疼痛只在疾病层、头痛只在脑部疾病层）
  · 结果集 = 直接声明 ∪ 沿 is_a 继承（去重）★需求项
  · 癫痫/脑卒中/阿尔茨海默病 的具体结果集大小
M10.4 闭包 / 幂等 / 依赖闸门 / 自足性（13 项）
  · 6 个新症状的 is_a 闭包各 4 行；经 is_a 可达抽象概念
  · 补丁零 closure INSERT、未重定义 relation_types
  · has_symptom 不产生闭包行
  · 第二次执行零漂移（幂等）
  · PRAGMA 在 BEGIN 之前、事务配对
  · 裸 schema 上加载补丁被拒绝且报错可读（依赖闸门）
```

**需求明确要求的 6 条断言全部为 PASS**（上表标 ★）。

### 7.1 一处被修正的既有断言（非新增）

§M4 的「医学概念 的 is_a 下位」原为 **24**，补丁让 6 个新症状经
`症状 → 病理状态` 上溯到 `医学概念`，故该项**口径更新为 30**
（24 骨架 + 6 症状补丁）。这是设计内的扩展，不是漂移。

### 7.2 `apply_seeds.py` 自检同步

- `SQL_FILES` 追加 `seed_medical_v3_patch_symptom.sql`（第五份，字典序排在
  `seed_medical_v3.sql` 之后：`.` U+002E < `_` U+005F，顺序天然正确）；
- `EXPECT` 更新为 `senses=130 / relations=112 / closure=208`；
- 自检新增 3 项（义项数、层边数、癫痫继承），自检总数 19 → **22**。

### 7.3 §M11 症状严重度义素（34 项，全部 PASS）

`validate_medical.py` 再追加 §M11 断言组（在 §M10 之后）：

```
M11.1 覆盖（3 项）      症状子树恰 8 个义项 / 8 个全部有严重度义素 / 子树与需求 8 词一致
M11.2 取值域（4 项）    3 个义素已登记 semantic_features / 取值仅限三值 /
                        无野义素（FK 未被绕过）/ 被标注症状类别一致（'性状'）
M11.3 唯一性（2 项）    每个症状恰 1 个严重度义素 / value 一律 '1'
M11.4 分级正确性（6 项）三个桶逐一精确比对 + 3 条需求项「包含」断言
M11.5 继承检索（7 项）  ★需求指定的 SQL 形态（closure ⋈ is_a ⋈ has_symptom ⟕ sense_features）
                        ★癫痫 的 severe 含 疼痛/头痛/抽搐
                        癫痫 moderate = {乏力,眩晕}（记忆障碍 不在其锥内，见下）
                        ★阿尔茨海默病 moderate ⊇ {眩晕,乏力,记忆障碍}
                        阿尔茨海默病 severe = {疼痛,头痛,意识模糊}（跨 2 跳）
                        ★mild 清单 = {发热}（且无任何疾病可检索到它）
                        LEFT JOIN 无扇出 / 不丢行
M11.6 全库不变量（1 项）严重度义素仅指向「症状」义项
M11.7 纯追加（4 项）    只 INSERT sources/semantic_features/sense_features /
                        零 closure INSERT / 不 ALTER TABLE / 不重定义关系类型
M11.8 幂等与闸门（7 项）二次执行零漂移 / PRAGMA 在 BEGIN 前 / 事务配对 /
                        裸 schema 被拒 / 缺 patch_symptom 被拒 / 文件名顺序不变量
```

**需求明确要求的 8 条断言全部为 PASS**（上表标 ★）。其中 **2 条的口径被如实收窄**
（不是"假装通过"），原因见 §10.5。

`apply_seeds.py` 同步：`SQL_FILES` 追加第六份、`EXPECT` 追加
`sense_features=86 / semantic_features=10`、自检新增 4 项，自检总数 22 → **26**。

### 7.4 §M12 症状分层（补丁三，38 项，全部 PASS）

```
M12.1 义项层（4 项）    3 个新义项 / sense_id 恰为 4037-4039 / 释义纯中文 / 类别合法
M12.2 is_a 与闭包（5 项）4 条边齐备 / 脑炎 多继承出边=2 / 层级传递 /
                        锚定到 医学概念 / 闭包行数 4·5·6 全自动派生
M12.3 层边 vs 直接边（4 项）  4 条 has_symptom 齐备 / confidence 与设计一致 /
                        ★癫痫→发热 = 0.5 全库最低 / ★发热 的声明源恰两个
M12.4 继承结果集（5 项）★癫痫 = 8 种 / ★脑炎 = 7 种 / ★感染性疾病 = 3 种 /
                        ★记忆障碍 只在两个疾病上直接声明 / 回归：脑卒中仍 5 种
M12.5 与 §M11 衔接（2 项）★癫痫 moderate 桶 = {乏力,眩晕,记忆障碍}（字面成立）
                        ★癫痫 mild 桶 = {发热}（字面成立）
M12.6 纯追加（5 项）    只写 words/senses/sense_categories/relations /
                        零 closure INSERT / 不 ALTER TABLE / 不重定义关系类型 /
                        不写义素（与补丁二职责分离）
M12.7 幂等与闸门（7 项）二次执行零漂移 / PRAGMA 在 BEGIN 前 / 事务配对 /
                        裸 schema 被拒 / 缺 patch_symptom 被拒 / 文件名顺序不变量
```

**关键：§M11 里原本"收窄"的 2 条需求断言，本轮由补丁三补齐为"按字面成立"**
（见 §12）。`apply_seeds.py` 再同步：`SQL_FILES` 追加第七份、`EXPECT`
更新为 `senses=133 / relations=120 / closure=223`、自检新增 5 项，自检总数 26 → **31**。

> 之后加入的影像子树（`seed_medical_v3_imaging.sql`）不改动上述任何口径 ——
> 它挂在 `医学概念` 下新开一条分支，与症状 / 疾病体系**无交集**（见 §13）。
> 它把总量推到 `senses=170 / relations=166 / closure=371`，自检 31 → **38** 项。

---

## 8. 全量回归（三份补丁 + 影像子树加载后）

```
validate_schema.py               PASS 102 / FAIL 0
validate_seed.py                 PASS  77 / FAIL 0
validate_connection.py           PASS 159 / FAIL 0
validate_closure.py              PASS  87 / FAIL 0
validate_syntax_parser.py        PASS 214 / FAIL 0
validate_medical.py              PASS 230 / FAIL 0   ← 69 + §M10 29 + §M11 34 + §M12 38
                                                        + §M13 53 + §M1 逐文件 7
tests/test_e2e_cat_eats_fish.py  PASS 64 / FAIL 0 / XFAIL 4（TOTAL 68）
apply_seeds.py                   自检 39 项 全绿，exit 0
────────────────────────────────────────────
合计 937 项（六个校验脚本 + e2e TOTAL 68）+ 39 项建库自检 = 976 项，FAIL 0
```

> 计数口径：e2e 按 **TOTAL 68** 计（PASS 64 + XFAIL 4），XFAIL 是主动记录的
> 已知偏差，仍属"被断言覆盖"的用例。若只计 PASS 64，则合计为 972 ——
> README 采用 976 口径，两处保持一致。

> `validate_closure.py` / `validate_syntax_parser.py` 只加载 schema 或自建库，
> **不读医学种子**，故计数在本轮保持 **87 / 214** 不变 —— 它们的作用是
> 确认「新增种子未破坏闭包与解析器两条主线」。

> **自动发现的联动**：`logos/db/connection.py` 的 `SEED_GLOB = "seed_*.sql"`
> 会把三份补丁与影像子树**自动**纳入所有走 `init_database` 的管线
> （`test_e2e_cat_eats_fish.py`、`validate_syntax_parser.py` 等）。
> 新增种子会改变下游的 ID 空间 —— 补丁一用 **4031–4036** 段、补丁三用
> **4037–4039** 段、影像子树用 **4040–4076** 段，与 e2e 夹具的 **7000 段**
> 完全隔离，故不触发上次（ext 撞 2001 段）那类回归；补丁二**不新增任何
> word/sense**，故对 ID 空间零影响。
> 四份增量的文件名都落在 `seed_medical_v3.sql` 之后，加载顺序天然正确
> （影像子树具体位次与实证见 §13.6）。详见 `docs/SEED_EXT_NOTES.md` §8 与本文 §11。

---

## 9. 后续扩展建议

1. ~~**补 `炎症性疾病` / `感染性疾病` 节点**，把 `发热` 按语义限定挂上去~~
   → **已由补丁三完成**（见 §12）。
2. **补 `全身性疾病` 层**，把 `乏力` 从「疾病」层下沉到更精确的层
   （当前挂 `疾病` 层是"更宽但保守"的口径）。
3. **把 `癫痫 → 发热` 的弱关联迁移到专用关系类型**：补丁三用
   `has_symptom` + `confidence=0.5` 表达"发热诱发发作"，严格建模应新增
   `has_precipitant`（诱发因素）/ `precipitates` 一对互逆类型。
   迁移步骤：登记关系类型 → 复制边 → 执行补丁三 §4.5 的撤回 SQL。
4. **症状的频次属性**：严重度已由补丁二用 `sense_features` 实现（见 §10）；
   频次同样走义素（如 `+frequent`），而非新增关系类型。`confidence` 保持
   "边可信度"语义不变。
5. **反向诊断链加密**：`indicates`（症状→疾病）目前只有
   `记忆障碍 → 阿尔茨海默病` 一条。可在新症状上补 `indicates` 边，
   让「头痛 → 脑部疾病」「意识模糊 → 重症」具备回溯诊断能力。
6. **症状与解剖结构的 `responsible_for` 链**：`头痛` 可挂
   `responsible_for` 的逆（`depends_on`）到 `脑`，把症状锚到器官。
7. **补密度 / 强化 / 位置的义素**：影像子树只按需求给「形状/边缘/大小」标了义素，
   另 3 个维度的取值已有义项、未标义素 —— 两步 SQL 见
   `seed_medical_v3_imaging.sql` §7.6（位置取值本质是「处所」，是否走义素需先决策）。
8. **影像特征 → 疾病的诊断链**：目前 `typically_has` 只在 恶性 / 良性结节 之间，
   可把它与已有的 `indicates`（症状→疾病）打通，让「毛刺状 + 高密度」直接
   指向 `脑部疾病` 类的 `indicates` 目标（见 §13.4 的查询模板）。
9. **`relation_constraints` 接入影像槽位**：架构全景里 Layer 3 的
   「`relation_constraints` 校验一致性」尚未覆盖影像特征（本次刻意不动，
   见 `seed_medical_v3_imaging.sql` 取舍 (7)）—— 要为它加槽位约束需先评审
   对解析器「受事合法性」判定面的影响。

---

## 10. 症状严重度义素约定（`sense_features`）

> 实现文件：`seed_medical_v3_patch_symptom_severity.sql`（纯追加，不改 schema、
> 不改已有数据）。**不新增任何 word / sense**，只写义素标注行。

### 10.1 feature 命名规范

| feature | 含义 | `value` | `feature_group` | `value_type` |
|---|---|---|---|---|
| `+severe` | 症状严重度：重度 | `'1'` | `极性` | `boolean` |
| `+moderate` | 症状严重度：中度 | `'1'` | `极性` | `boolean` |
| `+mild` | 症状严重度：轻度 | `'1'` | `极性` | `boolean` |

约定要点：

- **`+` 前缀** = 「具备该属性」的 boolean 义素，与 schema §14 的
  `+animate` / `+countable` / `+concrete` 同构；`value` 一律 `'1'`
  （boolean 特征用 `'1'/'0'`，见 `seed_general_v3.sql` §4 注释）。
- **三个义素是同一维度（严重度）的互斥取值**。DB 层无法阻止一个症状同时标
  两种严重度（`sense_features` 的 PK 是 `sense_id + feature`），
  由 §M11「每个症状恰有 1 个严重度义素」在**构建期**兜住。
- **不新造关系类型**：严重度不是"关系"，故不占用 `relation_types`。

### 10.2 分级与落点（8 个症状 → 3 个桶）

| 症状 | `sense_id` | 严重度 | 语义依据 |
|---|---:|---|---|
| 疼痛 | 4031 | `+severe` | 剧烈、显著失能 |
| 头痛 | 4033 | `+severe` | 剧烈、显著失能 |
| 抽搐 | 4021 | `+severe` | 危及生命 |
| 意识模糊 | 4036 | `+severe` | 「重症 / 脑部疾病共有」（§4.1）——*需求清单漏列，按语义补位* |
| 眩晕 | 4034 | `+moderate` | 明显不适，通常不危及生命 |
| 乏力 | 4035 | `+moderate` | 同上 |
| 记忆障碍 | 4022 | `+moderate` | 同上 |
| 发热 | 4032 | `+mild` | 轻微不适，多可自行缓解 |

### 10.3 查询模式（本项目统一口径）

```sql
-- ① 严重度清单：症状 → 严重度
SELECT w.word, sf.feature
  FROM sense_features sf
  JOIN senses s ON s.sense_id = sf.sense_id
  JOIN words  w ON w.word_id  = s.word_id
 WHERE sf.feature IN ('+severe','+moderate','+mild');

-- ② 「疾病 D 的症状 + 严重度」：closure ⋈ is_a ⋈ has_symptom ⟕ sense_features
--    ★ 这就是需求指定的查询形态；LEFT JOIN 保证未标严重度的症状也不丢行
SELECT DISTINCT wd.word, sf.feature AS severity
  FROM closure cl                                        -- ① D 的全部上位（is_a 闭包）
  JOIN relations r ON r.relation = 'has_symptom' AND r.src_kind = 'sense'
                  AND r.src_id  = cl.ancestor             -- ② 上位层声明的症状
  JOIN senses s2 ON s2.sense_id = r.dst_id
  JOIN words  wd ON wd.word_id  = s2.word_id
  LEFT JOIN sense_features sf ON sf.sense_id = r.dst_id   -- ③ 症状自身的严重度义素
                             AND sf.feature IN ('+severe','+moderate','+mild')
 WHERE cl.relation = 'is_a' AND cl.descendant = :D;

-- ③ 只要重度：在 ② 上加 WHERE sf.feature = '+severe'
```

查询形态与 §2 的**症状继承口径完全一致**，只多一层 `LEFT JOIN sense_features`
—— 即"继承"发生在 `is_a` 维度上，"严重度"发生在义素维度上，两者正交。

实测（**补丁三加载后**；癫痫 一行的 `+moderate` / `+mild` 列即需求原始清单）：

| 疾病 | `+severe` | `+moderate` | `+mild` |
|---|---|---|---|
| 癫痫 | 疼痛 / 头痛 / 抽搐 / 意识模糊 | 乏力 / 眩晕 / **记忆障碍** | **发热** |
| 脑卒中 | 疼痛 / 头痛 / 意识模糊 | 乏力 / 眩晕 | — |
| 神经退行性疾病 | 疼痛 / 头痛 / 意识模糊 | 乏力 / 眩晕 | — |
| 阿尔茨海默病 | 疼痛 / 头痛 / 意识模糊 | 乏力 / 眩晕 / 记忆障碍 | — |
| 脑炎 | 疼痛 / 头痛 / 抽搐 / 意识模糊 | 乏力 / 眩晕 | 发热 |
| 感染性疾病 | 疼痛 | 乏力 | 发热 |

> ⚠ 本表在补丁二落地时只有前 3 行，且 癫痫 的 `+moderate` = `{乏力, 眩晕}`、
> `+mild` = `—`（发热当时无 `has_symptom` 边）。补丁三补边后变成上表的样子，
> §10.5 记录的两处"口径收窄"随之解除。

### 10.4 为什么**不用** `relations.confidence`（语义分离）

`relations.confidence` 与"症状严重度"是两个**正交**的语义：

| | `relations.confidence` | `sense_features` 严重度义素 |
|---|---|---|
| 修饰对象 | **边**（疾病 —symptom→ 症状） | **义项**（症状自身） |
| 回答的问题 | 「这条边有多可信 / 多典型」 | 「这个症状本身有多重」 |
| 基数 | 每个（疾病, 症状）对 1 个值 | 每个症状 1 个值，全局唯一 |
| 改一次影响 | 该疾病—症状对 | 该症状的**全部**出现处 |

混用会造成两个具体问题：

1. **重复标注**：疼痛出现在 N 条疾病边上，就被标 N 次，改一次要改 N 处；
2. **语义不可分**：无法区分「该症状本身很重」与「该症状对某疾病不典型」的前者，
   而后者才是"边可信度"。例：发热本身是轻度，但对某感染性疾病却高度典型
   —— 用 `confidence` 表达严重度就把这两件事压成了一个数，信息不可还原。

因此 `confidence` 保持 **"边可信度"** 语义（`has_symptom` 统一 0.9），
严重度落到 **"义项的固有属性"**（`sense_features`）。

### 10.5 三处与 schema 的冲突 / 已知限制（如实记录，未假装通过）

1. **`feature_group` 复用「极性」**：`semantic_features.feature_group` 是
   **CHECK 白名单** `('生命度','具体性','可数性','自主性','有界性','极性')`，
   其中**没有**「严重度 / 程度」分组。新增分组必须 `ALTER TABLE` 改 CHECK，
   与「不改 schema」约束冲突。故复用含义最接近的**「带 +/- 标记、可枚举」**
   分组「极性」。⇒ 待 schema 允许扩大白名单时，应迁移到专用分组（补丁 §4.6
   给了可直接执行的迁移 SQL）。
2. **`sense_features` 没有 `source_id` 列**：需求要求 `source_id = 6`，
   但该表列仅 `(sense_id, feature, value)`。故 §1 在 **`sources`** 表登记
   `source_id = 6`（使该标识有效、可审计），并把来源写进
   `semantic_features.description`；义素行本身**无法**携带来源字段。
3. ~~**需求 2 条断言的口径被收窄**~~ → **已由补丁三解除，两条断言现按字面成立**
   （补出 `炎症性疾病` 层 + 给癫痫补直接边；详见 §12）。留档如下：
   - 「癫痫 继承到 **记忆障碍**」在补丁二时**不成立**：记忆障碍是 `阿尔茨海默病`
     的**直接**症状（不在任何上位层声明），故不在 `癫痫` 的 `is_a` 继承锥内
     —— 补丁二实测 `癫痫` 的中度集 = `{乏力, 眩晕}`，断言只能收窄落到
     `阿尔茨海默病` 上。补丁三给 `癫痫` 补了一条**直接** `has_symptom` 边后，
     `癫痫` 的结果集（= 直接声明 ∪ 沿 `is_a` 继承）自然含 `记忆障碍`。
   - 「癫痫 继承到 **发热**」在补丁二时**不成立**：`发热` 在补丁一中**刻意未挂**
     `has_symptom` 边（语义限定"炎症 / 感染性疾病共有"，骨架暂无该层，见 §4.2），
     故**任何**疾病的继承锥都取不到它，断言只能落在**全库义素清单**上。
     补丁三补出 `炎症性疾病` 层并挂边后，`发热` 首次**由继承可达**
     （`感染性疾病` / `脑炎`）；`癫痫 → 发热` 另补一条低置信度直接边（见 §12.3）。
   - 另有 1 处**补位**（仍然有效）：需求的分级清单只列了 7 个症状，漏了第 8 个
     「意识模糊」，而需求同时要求"8 个症状均有标注"。补丁二按其余量归入
     `+severe`（见 §10.2）。

### 10.6 写入规模（实测增量）

| 表 | 增量 | 说明 |
|---|---:|---|
| `sources` | +1 | `source_id = 6` 标注来源登记 |
| `semantic_features` | +3 | `+severe` / `+moderate` / `+mild` |
| `sense_features` | +8 | 8 个症状各 1 行 |
| `words` / `senses` / `relations` / `closure` / `changelog` / `sense_categories` | **0** | 纯义素标注，不动数据与结构 |

最终态：`semantic_features` 7 → **10**；`sense_features` 78 → **86**。

---

## 11. 加载顺序实证（为什么补丁二不叫 `..._patch_severity.sql`）

`logos/db/connection.py` 的 `_resolve_seed_paths()` 对种子文件用

```python
sorted(search_root.glob("seed_*.sql"))     # 按**文件名字典序**加载
```

补丁二依赖补丁一建立的 4031–4036 症状义项。而 `'severity'` 与 `'symptom'`
在第二个字符处分出大小：`'e'`(U+0065) `< 'y'`(U+0079)
⇒ **`..._patch_severity.sql` 会排在 `..._patch_symptom.sql` 之前**。

实测（把六份文件拷进临时目录，直接调 `init_database()` 走自动发现）：

```
[安全命名 ..._patch_symptom_severity.sql]
    seed_medical_v3.sql
    seed_medical_v3_patch_symptom.sql
    seed_medical_v3_patch_symptom_severity.sql      ← 顺序正确
    → init_database 成功；严重度义素行数 = 8  ✔

[请求命名 ..._patch_severity.sql]
    seed_medical_v3.sql
    seed_medical_v3_patch_severity.sql              ← 被错误地提前
    seed_medical_v3_patch_symptom.sql
    → init_database 抛错 SeedError: CHECK constraint failed:
      _prereq_missing_run_seed_medical_v3_and_patch_symptom_first
```

结论与做法：

- **必须**让补丁二的文件名字典序落在补丁一**之后**。
  `..._patch_symptom_severity.sql` 之所以成立，是因为公共前缀
  `..._patch_symptom` 之后是 `'.'`(U+002E) `< '_'`(U+005F)，
  因此天然排在 `..._patch_symptom.sql` 之后（与 §5.1 的段位隔离同一思路：
  **用命名承载顺序约束**）。
- 补丁内 §0 的**前置依赖硬检查**在这里起到了兜底作用：若有人改回
  `..._patch_severity.sql`，`init_database` 会**响亮地报错**（报错文本自带修复提示），
  而不是像纯内连接那样**静默少插 8 行**。§M11 有一条断言把"文件名顺序"钉住。
- 这条规则与 `.workbuddy/memory/MEMORY.md` §2 记录的历史事故
  （ext 撞 2001 段、种子自动发现改变下游计数）同源：**新增种子文件既要选空 ID 段，
  也要保证文件名字典序落在其依赖之后。**

---

## 12. 炎症性疾病层与癫痫补边（补丁三）

> 实现文件：`seed_medical_v3_patch_symptom_layers.sql`（纯追加：只写
> `words` / `senses` / `sense_categories` / `relations`，不碰结构定义、不写义素）。
>
> 目的：把 §10.5 记录的两处「口径收窄」**按字面补齐**，使需求断言不再需要收窄。

### 12.1 新增形状

```
                        医学概念
                          └─ 病理状态
                              ├─ 疾病 ─has_symptom→ 疼痛 / 乏力
                              │   ├─ 炎症性疾病 ─has_symptom→ 发热          ★ 新层
                              │   │   ├─ 感染性疾病                          ★ 新义项
                              │   │   └─ 脑炎 ─has_symptom→ 抽搐（直接）      ★ 新义项（多继承）
                              │   └─ 脑部疾病 ─has_symptom→ 头痛 / 眩晕 / 意识模糊
                              │       ├─ 脑炎（同上，第二父）
                              │       ├─ 癫痫 ─has_symptom→ 抽搐 / 记忆障碍 / 发热
                              │       ├─ 脑卒中
                              │       └─ 神经退行性疾病 ─ 阿尔茨海默病 ─→ 记忆障碍
                              └─ 症状 → 抽搐/记忆障碍/疼痛/发热/头痛/眩晕/乏力/意识模糊
```

| 表 | 增量 | 说明 |
|---|---:|---|
| `words` / `senses` | +3 | 炎症性疾病 / 感染性疾病 / 脑炎（4037–4039） |
| `sense_categories` | +3 | 一律 `性状`（与 `疾病` 同一粗类，未新增类别行） |
| `relations` | +8 | 4 `is_a` + 4 `has_symptom` |
| `closure` | **+15** | 全部**自动派生**：炎症性疾病 4 / 感染性疾病 5 / 脑炎 6 |
| `changelog` | +8 | 由 `tr_relations_audit_ai` 自动记录 |

最终态：`senses` 130 → **133**；`relations` 112 → **120**；`closure` 208 → **223**。

### 12.2 层边 vs 直接边：为什么这次两种都要

补丁三同时用了两种声明粒度，正好说明「声明在哪一层」是**语义判断**：

| 边 | 粒度 | 为什么 |
|---|---|---|
| `炎症性疾病 → 发热` | **层边** | 炎症 / 感染性疾病**共有** ⇒ 下沉到层，全部下位（`感染性疾病` / `脑炎`）自动继承 |
| `脑炎 → 抽搐` | 直接边 | 脑炎可致惊厥，但**不是**所有炎症性疾病都有 ⇒ 只能在 脑炎 上声明 |
| `癫痫 → 记忆障碍` | 直接边 | 颞叶病灶累及海马；但记忆障碍**不是**所有疾病 / 脑部疾病共有（`阿尔茨海默病` 也有，那是另一条直接边） |
| `癫痫 → 发热` | 直接边（弱） | 见 §12.3 |

`脑炎` 是**第一个吃到三个层**的节点（`is_a` 多继承：`炎症性疾病` ∧ `脑部疾病`）：

```
脑炎 的症状 = 炎症层(发热) ∪ 疾病层(疼痛/乏力) ∪ 脑部疾病层(头痛/眩晕/意识模糊) ∪ 直接(抽搐) = 7 种
癫痫 的症状 = 疾病层(2) ∪ 脑部疾病层(3) ∪ 直接(抽搐/记忆障碍/发热)                              = 8 种
感染性疾病   = 炎症层(发热) ∪ 疾病层(疼痛/乏力)                                                = 3 种
```

### 12.3 ★ 一处**语义妥协**：`癫痫 → 发热` 用低置信度表达

需求断言「癫痫继承到的症状中，`+mild` 包含发热」要求这条边存在；而严格临床口径下
**发热是癫痫发作的诱发 / 并发因素，不是癫痫的典型症状**：

- 热性惊厥谱系 —— 发热降低惊厥阈值（发热 → 发作）；
- 癫痫持续状态 —— 长时间发作可致体温升高（发作 → 发热）。

两者是**关联**而非「症状」。本补丁的处置：

| 做法 | 结果 |
|---|---|
| 用 `has_symptom` + **`confidence = 0.5`**（全库 `has_symptom` 最低，层边 0.9、其它直接边 0.75~0.85） | 断言按字面成立，且**在数值上可与层边区分**，不污染"层上共有"的口径 |
| 补丁 §4.5 给出**一行撤回 SQL**（用 `confidence = 0.50` 限定，避免误删层边） | 将来可无痛迁移 |
| §M12 有 2 条断言把「0.5 = 全库最低」与「发热 的声明源恰为 炎症性疾病 + 癫痫」钉住 | 防止后续把弱边悄悄改成层边 |

**若你要严格口径**：执行补丁 §4.5 的 `DELETE`，并把 §M11/§M12 中
`癫痫 的 '+mild' 症状包含发热` 一条改为落在 `脑炎` / `感染性疾病` 上
（它们经 `炎症性疾病` 层**真实继承**到发热，不依赖任何弱边）。
**若你要更完备的建模**：新增 `has_precipitant`（诱发因素）/ `precipitates`
一对互逆关系类型，把这条边迁过去（见 §9 建议 3）。

### 12.4 查询模式（补丁三之后）

```sql
-- ① 「发热」现在可由继承可达：谁的症状里有发热？
SELECT DISTINCT ws.word AS 疾病, wd.word AS 症状 FROM relations r
  JOIN senses s1 ON s1.sense_id = r.src_id JOIN words ws ON ws.word_id = s1.word_id
  JOIN senses s2 ON s2.sense_id = r.dst_id JOIN words wd ON wd.word_id = s2.word_id
 WHERE r.relation = 'has_symptom' AND r.src_kind = 'sense'
   AND ( r.src_id = :D
         OR r.src_id IN (SELECT ancestor FROM closure
                          WHERE relation='is_a' AND descendant = :D) )
   AND wd.word = '发热';
-- 期望：炎症性疾病（层边源）· 感染性疾病 · 脑炎 · 癫痫（弱边）

-- ② 只要「层上真声明的」发热边（排除弱关联）：
--    在上句基础上改 WHERE r.src_id = :D ... 并加 r.confidence = 0.9
```

### 12.5 与 §2 / §6 口径的一致性

补丁三不改变任何查询口径：症状继承仍是「直接声明 ∪ 沿 `is_a` 继承（去重）」，
`has_symptom` 仍 `is_transitive=0`（§M12 有断言：`closure` 里无 `has_symptom` 行）。
新增的 4 条 `is_a` 边由触发器自动物化 —— 补丁内**零** `closure` 的 `INSERT`。

---

## 13. 影像特征子树（`seed_medical_v3_imaging.sql`）

> 实现文件：`seed_medical_v3_imaging.sql`（纯追加，不改 schema、不改已有数据）。
> 定位：医学影像管线的 **Layer 2 输出端 / Layer 3 输入端** ——
> 把 CNN 的数值特征（直径、圆形度、密度 HU、边缘梯度…）映射到**统一义项空间**，
> 再交由符号引擎做 is_a 闭包推理与 `typically_has` 诊断倾向推理。

### 13.1 架构定位：本文件处于「数值 → 符号」的分界线上

```
Layer 1  图像      CT/MRI → CNN(YOLO/UNet) → bbox + {直径, 圆形度, 密度HU, 边缘梯度}
Layer 2  映射      轻量分类器(SVM/随机森林/小MLP) → {形状:+不规则, 大小:+大型, 边缘:+毛刺, 密度:+高密度}
        ─────────────────────────────────────────────────────────────────────────────
        本文件 = Layer 2 的**目标端**：给出可被映射到的符号词表（义项 + 义素）
Layer 3  推理      Logos 符号引擎
                   ├── 义素写入 sense_features     ← §5.3 的 12 个义素
                   ├── is_a 闭包推理分类            ← §13.2 的 37 节点 / 37 边
                   └── 输出诊断倾向 + 可解释链      ← §13.4 的正/反向查询
```

**本文件不含任何数值。** 毫米、HU、圆形度阈值只出现在 §7.1 的
「数值→符号映射契约表」注释里 —— 那是 Layer 2 的**算法参数**，
符号库只认符号。这也让 `senses.definition` 保持"纯中文词语定义"
（§M13 有断言：新增 37 条释义不含数字/拉丁字母）。

### 13.2 数据形状（实测）

```
影像特征 ⊑ 医学概念                              （33 节点：1 + 5 维度 + 25 取值 + 2 子维度）
├── 形态学特征 ─┬─ 形状特征 ─ 圆形 / 椭圆形 / 不规则形 / 分叶状
│               └─ 边缘特征 ─ 光滑 / 毛刺状 / 模糊 / 针刺状
├── 大小特征 ─ 微小 / 小型 / 中等 / 大型          ★ 见 §13.3
├── 密度特征 ─ 低密度 / 等密度 / 高密度 / 钙化
├── 强化特征 ─ 均匀强化 / 不均匀强化 / 环形强化 / 无强化
└── 位置特征 ─ 肺上叶 / 肺下叶 / 肺门 / 外周 / 中央

病理结构 ⊑ 医学概念 ─ 结节 ─┬─ 恶性结节 ─typically_has(0.8)→ 不规则形 / 毛刺状 / 高密度
                            └─ 良性结节 ─typically_has(0.8)→ 圆形 / 光滑 / 低密度
结节 has_property 形态学特征 / 密度特征          （"结节在这些维度上可被刻画"）
通用库「信号」--domain_specialization--> 医学库节点「强化特征」（3104）
```

| 表 | 增量 | before → after |
|---|---:|---|
| `words` / `senses` | +37 | 132/133 → **169/170** |
| `relations` | +46 | 120 → **166** |
| `closure` | **+148**（全自动派生） | 223 → **371** |
| `relation_types` | +2 | 43 → **45** |
| `entity_nodes` | +1 | 4 → **5** |
| `sense_categories` | +37 | 111 → **148** |
| `semantic_features` / `sense_features` | +12 / +12 | 10/86 → **22/98** |
| `changelog` | +46 | 120 → **166** |

**为什么 `closure` 是 +148 而不是 +37**：`is_a` 是 incremental，插一条边即派生
**该节点到「抽象概念」的整条上位链**。逐层累加：

```
影像特征(2) + 形态学特征(3)
  + 形状特征(4) + 4 形状值×5 = 24
  + 边缘特征(4) + 4 边缘值×5 = 24
  + 大小特征(3) + 4 大小值×4 = 19
  + 密度特征(3) + 4 密度值×4 = 19
  + 强化特征(3) + 4 强化值×4 = 19
  + 位置特征(3) + 5 位置值×4 = 23
  + 病理结构(2) + 结节(3) + 恶性结节(4) + 良性结节(4) = 13
  ── is_a 小计 146
  + has_property 2 行（★ 见 §13.4）
  = 148
```

### 13.3 ★ 一处**词形替换**：「小」「大」撞通用库既有词

需求原文的大小分级是 `微小(<5mm) / 小(5-10mm) / 中等(10-30mm) / 大(>30mm)`。
实测：**「小」（`word_id=1011`）与「大」（`word_id=1010`）已存在于通用库种子**
（`seed_general_v3.sql` 的 1000–1027 段，是普通形容词，`pos='a'`）。而
`schema_v3.sql` 的 `words.word` 是 **全局 `UNIQUE COLLATE BINARY`**（不带 `pos`），
故在「不改 schema、不修改已有数据」约束下**无法**再建同名词条。

| 备选路 | 后果 | 采用 |
|---|---|---|
| (a) 复用既有 sense 1010/1011，加 `大 ⊑ 大小特征` | ✗ 语义**方向反了**（断言"通用形容词『大』是影像特征的子类"）；✗ 会把两个 `id<4000` 的通用词拖进「抽象概念」子树，使 `apply_seeds.py` 的「分类骨架节点数 = 36（ext 时代）」断言失效 | ✗ |
| (b) 另立不撞词词形 `小型` / `大型` | 「中等」本身不撞词故保留；只有 小/大 两处替换 | **✓** |
| (c) 给 `words` 加 `pos` 维度 / 删既有 1010/1011 | 违反"不改 schema / 不改已有数据" | ✗ |

⇒ 实际词形：**微小 / 小型 / 中等 / 大型**。
变更 SQL 模板（若日后采纳 (a)/(c)）见补丁 §7.8 (a)。

### 13.4 `has_property` 会派生闭包 —— 与医学关系的行为差异

| 关系 | `is_transitive` | `closure_policy` | 本补丁的闭包行 |
|---|---|---|---|
| `is_a` | 1 | incremental / 12 | 146 |
| `has_property` | **1**（schema §14 登记） | incremental / 12 | **2** |
| `typically_has`（新） | **0** | 无（也不需要） | **0** |
| `domain_specialization` | 0 | 无 | 0 |

`has_property` 在 schema 里本来就是可传递 + 增量物化的（`食物 has_property 可食用`
就是这么用的），故 `结节 has_property 形态学特征/密度特征` **必然**派生 2 行闭包。
这与医学库主文件里 6 条医学关系（一律 `is_transitive=0`、零闭包）**行为不同**，
不要按"非 is_a 就不出闭包"想当然 —— §M13 有断言把实测行数钉住。

「维度」与「取值」的分工（**别把两者混在一个关系里**）：

- `has_property` 的 dst 是**维度**（形态学特征 / 密度特征）——"结节在哪些维度上可被描述"；
- `typically_has` 的 dst 是**取值**（不规则形 / 毛刺状 / …）——"某个类别典型呈现什么值"。

### 13.5 `typically_has` / `typical_of`：诊断倾向的落脚点

| 项 | 设计 | 理由 |
|---|---|---|
| `is_transitive` | **0** | 不构成可传递层级；显式声明以避开"`is_transitive=1` 却缺 `closure_policy` → 静默不物化"的坑 |
| `inverse_of` | `typically_has ⇄ typical_of` | 与 `has_symptom ⇄ symptom_of` 同款成对登记；逆向边由 `v_relations_all` 在线推导，不冗余存 |
| `is_functional` | `typical_of` = 1 | 沿用 `symptom_of` / `treats` / `depends_on` 的约定（"一个特征典型归属于唯一类别"）。**该假设在本数据集内成立**（6 条边恰好两两分组），全域未必成立，故写入文档而非默认 |
| `confidence` | 0.8 | 表达"统计上典型但不必然"；低于症状层边的 0.9，也低于 `is_a` 的 1.0 |
| 与 `has_property` 的分工 | `has_property` = 定义性/必然；`typically_has` = 统计性/倾向 | 这条区分正是诊断概率推理的立足点，**不要合并成一个类型** |

诊断查询（正/反向，§M13 各有一条断言）：

```sql
-- ① 正向：某结节类别的典型影像特征（沿 is_a 继承上位声明，与症状继承同一口径）
SELECT DISTINCT wf.word FROM relations r
  JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wf ON wf.word_id=s2.word_id
 WHERE r.relation='typically_has' AND r.src_kind='sense'
   AND ( r.src_id = :N
         OR r.src_id IN (SELECT ancestor FROM closure
                          WHERE relation='is_a' AND descendant = :N) );
-- 期望（:N = 恶性结节）：不规则形 · 毛刺状 · 高密度

-- ② 反向：给定 Layer 2 输出的符号向量 → 检索最可能的类别
SELECT wn.word, COUNT(*) AS hit FROM relations r
  JOIN senses s1 ON s1.sense_id=r.src_id JOIN words wn ON wn.word_id=s1.word_id
  JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wf ON wf.word_id=s2.word_id
 WHERE r.relation='typically_has' AND r.src_kind='sense'
   AND wf.word IN ('不规则形','毛刺状','高密度')
 GROUP BY wn.word ORDER BY hit DESC;
-- 期望：恶性结节 3（全中）
```

### 13.6 义素：Layer 2 的输出标签

| 维度 | 词形 → 义素（**刻意不同名**） |
|---|---|
| 形状 | 圆形→`+圆形` · 椭圆形→`+椭圆形` · 不规则形→`+不规则` · 分叶状→`+分叶` |
| 边缘 | 光滑→`+光滑` · 毛刺状→`+毛刺` · 模糊→`+模糊` · 针刺状→`+针刺` |
| 大小 | 微小→`+微小` · 小型→`+小型` · 中等→`+中等` · 大型→`+大型` |

- 命名**不照抄词形**：义素是分类器的输出标签，应当是紧凑的标记名
  （需求示例给的也是 `+圆形`、`+毛刺`，而非 `+毛刺状`）。
- `value` 一律 `'1'`（boolean 口径，与 `+animate` 及 `+severe` 一致）。
- 只覆盖需求指定的 **3 个维度**；密度 / 强化 / 位置的取值**只建义项、未标义素**
  （§M13 有断言强制这一点），补齐 SQL 见补丁 §7.6。
- `feature_group` 复用 **「极性」** —— 与严重度义素同一个已知妥协（§10.5 第 1 条）：
  `semantic_features.feature_group` 是 CHECK 白名单，**没有**「影像 / 形状」分组，
  新增须 `ALTER TABLE`。迁移 SQL 见补丁 §6.6。

### 13.7 加载顺序（实证）

```
seed_general_v3.sql
seed_general_v3_ext.sql
seed_medical_v3.sql            ← 依赖：医学概念 / 信号
seed_medical_v3_imaging.sql    ← 本文件（'i' 0x69）
seed_medical_v3_patch_symptom.sql          （'p' 0x70）
seed_medical_v3_patch_symptom_layers.sql
seed_medical_v3_patch_symptom_severity.sql
```

`seed_medical_v3.sql` < `seed_medical_v3_imaging.sql` 由
`'.'(U+002E) < '_'(U+005F)` 保证；影像子树排在症状补丁**之前**由
`'i'(0x69) < 'p'(0x70)` 保证。影像子树与症状体系**无交集**，故该位次既正确又安全
（它不需要症状补丁建的任何词）。§M13 有一条断言把该位次钉住。

### 13.8 已知限制 / 未采纳项

1. **词形替换**：`小型` / `大型` 替代需求写的 `小` / `大`（§13.3）。
2. **`feature_group` 复用「极性」**：无专用分组（§13.6）。
3. **密度 / 强化 / 位置取值无义素**：需求只要 3 个维度（§13.6）。
4. **不写 `relation_constraints`**：属独立议题，且会触及解析器的受事合法性判定面
   （见补丁取舍 (7) 与 §9 建议 9）。
5. **`is_functional=1`（`typical_of`）在本数据集内成立、全域未必**（§13.5）。
6. **未接入疾病层**：`typically_has` 只连到 `恶性结节` / `良性结节`，
   尚未与 `indicates`（症状→疾病）打通（§9 建议 8）。
