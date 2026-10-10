# `validate_schema.py` 用例审查报告

> 审查对象：`validate_schema.py`（原 47 项自检）
> 运行环境：SQLite **3.53.1** / Python 3.13.12
> 结论：**47 项全部保留，新增 55 项 → 102 项，`PASS 102 / FAIL 0`**
> 审查方式：不读代码下结论，而是**为怀疑的盲区先写探针实测**（`probe_gaps.py`），
> 确认是"测试没写"还是"schema 真有问题"，再决定如何补用例。

---

## 0. 先回答最尖锐的那个问题：47 项**不全是**行为测试

**29 项是行为测试，18 项不是。** 明细：

| 类型 | 数量 | 编号（按 STEP 顺序） | 风险 |
|---|---|---|---|
| ✅ 行为测试（执行 DML/查询，断言运行时结果） | **29** | 生成列×2、STRICT 拒绝、UNIQUE/CHECK×4、端点触发器、闭包×6、视图展开、FTS×3、量词、递归 CTE、changelog、PRAGMA+FK 拒绝、R2×3、R4×3、R7×2 | — |
| ⚠️ 结构元数据 | 6 | placeholder 残留、STRICT 表计数、R3 冗余索引、R5 主键列、R6 无 JSON 列 | 低 |
| 🔴 **DDL 文本匹配** | **4** | R8×4（CTE 在 FROM / 无非法写法 / UPSERT 位置 / 弱引用） | **高：改格式即误报，且完全不证明运行时行为** |
| ⚠️ 种子数据静态一致性 | 6 | 3.12 多策略计数、R5 多策略计数、R6 有槽位 / 占位符==槽位 / 无 `?` 后缀、R9 三档模式 | 中：种子一改就失效，且不覆盖运行时新建的数据 |
| ℹ️ 库级完整性 | 2 | `integrity_check` / `foreign_key_check` | 低（但价值高，属外部真值源） |
| ℹ️ 优化器计划 | 1 | 3.15 `EXPLAIN QUERY PLAN` | 低：真跑优化器，但断言的是计划形态 |

### 两个具体问题

**问题 A（🔴）：R8 的四条断言全是"对 DDL 文本做位置/子串匹配"。**

```python
_i_fe, _i_dp = _norm.find("first_edge = CASE"), _norm.find("depth = MIN")
check("R8 UPSERT 中 first_edge 写在 depth 之前", 0 <= _i_fe < _i_dp, ...)
```

这条断言的问题不是"错"，而是**弱**：

1. 它只证明**写法顺序**，不证明 DO UPDATE 真的生效。哪怕 `CASE WHEN excluded.depth < depth`
   的条件被写反成 `>`，只要顺序不变，这条永远 PASS。
2. 把 `first_edge = CASE` 写成 `first_edge=CASE`（去掉空格）即误报 FAIL。
3. 同一个 `schema_v3.sql` 被格式化/换行后立刻失效。

**已用行为测试替代**（R8 四条**保留**作为"写法锁定"，另加真行为断言）：

| 原文本断言 | 等价的行为断言 | 实测结果 |
|---|---|---|
| UPSERT 顺序 | 5.7a 先造两条 2 跳路径，再插直连边 → `depth` 2→1 且 `first_edge` 换为该边 | PASS |
| 同上 | 5.7b 更长路径不得覆盖已建立的最短路径（`depth` 恒为 1） | PASS |
| 弱引用 | 5.4a 删掉见证边后，重建靠**另一条重叠路径**恢复该闭包对（菱形） | PASS |
| 弱引用 | 5.4b 删除后无悬空 `depth=1` 行 | PASS |

**问题 B（🟡）：多条断言"无阳性对照"——探测器坏掉也 PASS。**

典型是 R3：`check("无冗余索引", not _red)`。若 `find_redundant_indexes()` 因任何原因恒返回 `[]`，
这条**永远 PASS**。实测确认了探测器可被喂入构造数据检出（5.13a），但原套件里没有这一步。

---

## 1. 七维度覆盖矩阵

| # | 维度 | 原有覆盖 | 覆盖强度 | 缺口 | 本轮新增 |
|---|---|---|---|---|---|
| 1 | 外键完整性 / `PRAGMA foreign_keys` | R1×2 + `foreign_key_check` | 🟡 中 | 不测 `ON DELETE` 级联、不测**连接级生效性**、不测**事务内 no-op 陷阱**、不测其他 FK 列 | 5.2b、5.8a–5.8k（13 项） |
| 2 | 闭包方向不变量 I3 | R7×2 + 3.6 方向语义 | 🟡 中 | **只验 `depth=1`**；无独立实现的差分对拍；不验 `max_depth` 截断；不验「增量 == 重建」；**无负向用例** | 5.3a–d、5.4a–c、5.5a–c、5.6a–c（13 项） |
| 3 | 冗余索引检测 | R3 | 🔴 弱 | 无阳性对照；`PRAGMA index_info` **不返回 DESC/collation → 会把 DESC 索引误判为冗余** | 改写探测器（`index_xinfo`）+ 5.13a–c（3 项） |
| 4 | `pattern_slots` 覆盖 | R6×4 | 🟡 中 | 只验「集合相等」；不验 `position` 连续性；全部基于**种子静态数据**，未在运行时新建句式上验证 | 5.1a–b、5.2a–b（4 项） |
| 5 | UPSERT 顺序 | R8（文本位置） | 🔴 弱 | 见 §0 问题 A | 5.7a–b（2 项） |
| 6 | `ellipsis_rules` 多策略 | 3.12 + R5（均种子 COUNT） | 🟡 中 | 未运行时插入；未测**组合唯一性边界**；未测 priority 排序 | 5.12a–c（3 项） |
| 7 | `cross_links` 无重复 | R4×3 | 🟡 中 | 无 fan-out 断言（对称关系反向展开是否重复）；无「视图随底层同步」断言；无「symmetric + inverse_of 会导致三倍展开」的守卫 | 5.11a–d、5.0c（5 项） |
| ＋ | **空库**（新知识库冷启动） | **无** | 🔴 缺失 | 视图 / 闭包重建 / FTS 在零数据下是否健壮 | 5.9a–e（5 项） |
| ＋ | **单节点** | **无** | 🔴 缺失 | 零边、单行递归 CTE | 5.10a–b（2 项） |
| ＋ | **循环引用** | 3.7（仅 1 条链） | 🟡 中 | 跨关系类型隔离；自引用链（`semantic_roles.parent_role`）是否有环 | 5.10c–e、5.0e（4 项） |
| ＋ | **单一真源契约** | **无** | 🔴 缺失 | 传递关系缺策略行＝静默无闭包；`max_depth` 双写；`inverse_of` 互逆性 | 5.0a–b、5.0d（3 项） |

---

## 2. 实测发现的 5 个缺陷（这才是补用例的真正收获）

### D1 🔴 7 个传递关系缺 `closure_policy` 行 → **静默不产生任何闭包**（已修）

触发器条件是 `(SELECT mode FROM closure_policy WHERE relation = NEW.relation) = 'incremental'`。
**缺行时子查询返回 NULL，NULL = 'incremental' 为假 → 不报错、不插入、无任何提示。**

```
is_transitive=1 的关系: 15 个 / closure_policy 行: 10 行
缺失策略行的传递关系 (7 个):
   can_do / caused_by / entails / has_property / precedes / presupposes / topic_of
→ 实测：连插两条 can_do 边后 closure 行数 = 0  （期望 3）
```

**已修**：补齐 7 行（属性类 `incremental/12` 因"属性继承"推理依赖闭包；事件类与 `causes` 保持一致用 `batch/8`）。
**已加回归锁**：5.0a「每个 `is_transitive=1` 的关系都必须有显式策略行」。

### D2 🔴 `max_depth` 在「增量」与「重建」两条路径上语义不一致（已修）

增量触发器的 `lo`/`hi` 递归项卡的是 `d ≤ max_depth`，但最终深度是 `L.d + 1 + H.d`，
**实际上限是 `2*max_depth+1`**；而重建路径严格 `≤ max_depth`。

```
max_depth=2，链 810→811→812→813：
  [增量] 6 行，最大 depth = 3   ← 多出 (813,810,3)
  [重建] 5 行，最大 depth = 2
→ 同一份数据，两种闭包
```

**已修**：在最终 `WHERE` 统一卡口 `L.d + 1 + H.d <= max_depth`，`lo`/`hi` 只做 `≤ max_depth-1` 剪枝。
修后实测两条路径均为 5 行、最大 depth=2。
**已加回归锁**：5.3b / 5.5c「增量 == 重建 == 独立在线 CTE」。

### D3 🟡 `relation_types.max_depth` 是**死列**，与 `closure_policy.max_depth` 双写同一事实（待决策）

全文只有 `closure_policy.max_depth` 被触发器读取（891/903/947 行），
`relation_types.max_depth` **无任何读取点**。且两处取值实际冲突：`has_part` 是 12 vs 8。

**暂处理**：不改结构，改为在 5.0b 断言 `closure_policy.max_depth <= relation_types.max_depth`，
把该列的角色固化为"类型级上限"。
**待你决策**：(a) 删掉 `relation_types.max_depth`；或 (b) 删掉 `closure_policy.max_depth`、触发器统一读前者（真单一真源）。

### D4 🟡 从 `incremental` 切到 `off` 时，既有闭包行不会被清理（待决策）

`tr_closure_edge_ad` 的 `WHEN ... mode <> 'off'` 意味着 `off` 模式下删除边**不会**触发失效与重建：

```
增量维护后 closure 行数 = 3
→ 切到 off 后删除一条边：closure 残留 3 行，重建队列 0 行
→ 其中 2 行的 first_edge 已指向不存在的边（陈旧数据，会导致错误推理）
```

**建议修复**（6 行，未擅自落地，因涉及"模式切换语义"的设计决定）：

```sql
CREATE TRIGGER tr_closure_policy_au AFTER UPDATE OF mode ON closure_policy
WHEN OLD.mode <> NEW.mode
BEGIN
    DELETE FROM closure WHERE relation = NEW.relation;
    INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at)
        VALUES(NEW.relation, strftime('%Y-%m-%dT%H:%M:%fZ','now'));
END;
```

### D5 🟡 冗余索引探测器把 **DESC 索引误判为冗余** → 会误删（已修工具）

`PRAGMA index_info` 只返回列名，**不返回排序方向与 collation**：

```
CREATE INDEX tmp_expr2 ON senses(word_id DESC, pos DESC);
→ 探测器误报：('senses', 'tmp_expr2', 'tmp_dup_idx')  ← 判定为冗余，会导致误删
PRAGMA index_info('tmp_expr2') = [(0,1,'word_id'), (1,2,'pos')]   ← 排序信息丢失
```

**已修**：`find_redundant_indexes()` 改用 `PRAGMA index_xinfo`，签名由
`(列名)` 升级为 `(列名, desc, coll)`，并只取 `key=1` 的列（`WITHOUT ROWID` 索引会把 PK 列以 `key=0` 追加在尾部）。
表达式索引（`cid=-2`）标记为 `<expr:seq>` 且不作被删候选。
**已加回归锁**：5.13a 阳性对照、5.13b DESC 阴性对照、5.13c 清理后归零。

---

## 3. 补的 55 项用例（按组）

| 组 | 项数 | 内容 | 关键设计 |
|---|---|---|---|
| 5.0 单一真源契约 | 5 | 策略行齐备、`max_depth` 上限、`symmetric` 不得有 `inverse_of`、`inverse_of` 互逆成对、`semantic_roles` 自引用链无环 | **必须最先跑**，否则被后续构造数据污染 |
| 5.1 `pattern_slots` 不变量 | 2 | `position` 连续 1..n、每句式至少一个必填槽 | 补 R6 只查集合相等的空白 |
| 5.2 活表动态验证 | 2 | 运行时新建单槽位句式 → 占位符==槽位；删句式级联清槽位 | 把静态断言升级为行为断言 |
| 5.3 **差分对拍** | 4 | 物化闭包 == **独立实现的在线递归 CTE**；重建 == 增量 == 期望；队列自清空；重建幂等 | 用"独立实现 + 重叠路径菱形"做 oracle，而非拿闭包验闭包 |
| 5.4 删除路径 | 3 | 删见证边后靠重叠路径恢复；无悬空 `depth=1`；删空后归零 | 直击 `first_edge` 设计的核心风险 |
| 5.5 `max_depth` 截断 | 3 | 严格 `≤ max_depth`；截断而非丢链；边界上增量==重建 | D2 的回归锁 |
| 5.6 负向用例 | 3 | 非传递不写闭包；对称+`off` 不写闭包；`batch` 不内联写闭包 | 负向用例最易静默失效 |
| 5.7 UPSERT 真路径 | 2 | DO UPDATE 使 `depth` 2→1 且换见证边；更长路径不覆盖最短 | 行为化替代 R8 文本断言 |
| 5.8 外键/PRAGMA | 11 | 新连接默认 OFF、事务内 PRAGMA 静默失效、FK 拒绝、**触发器在 FK=OFF 时仍兜底**、CHECK×2、端点不可变、FK×2、**单事务加载陷阱**、`foreign_key_check` 兜底 | 覆盖维度 1 的全部空白 |
| 5.9 空库 | 5 | 5 个视图零行不报错、空集重建幂等、FTS 双通道零行、冷启动后 FTS 触发器正常 | 新知识库冷启动路径 |
| 5.10 单节点/循环 | 5 | 零边不产生闭包、单行库重建不报错、自环被拒、成环不产生自环行、**成环不影响其他关系** | 覆盖"单节点""循环引用" |
| 5.11 `cross_links` 无重复 | 4 | 无同源同靶同关系重复行、两条边恰好 2 行、对称反向展开恰好 1 次、删底层边视图同步 | 覆盖维度 7 |
| 5.12 `ellipsis_rules` 多策略 | 3 | 运行时追加第 3 条策略、组合 `UNIQUE` 边界、`priority` 稳定排序 | 覆盖维度 6 |
| 5.13 索引探测器对照 | 3 | 阳性对照、DESC 阴性对照、清理归零 | 补 R3 无判别力的问题 |

---

## 4. 写用例过程中踩到的坑（会影响你的其他测试）

| 坑 | 现象 | 正确做法 |
|---|---|---|
| **直接 DELETE FTS5 影子表** | `fts5: corruption found reading blob 274877906945` | 只删 `senses`，由 `tr_senses_fts_ad` 同步清理；**跳过** `senses_fts_*` |
| `senses_fts_*_data/_idx/_config` 永不空 | 空索引也保留段标记 | 只对**内容表**（`senses_fts_trigram` / `senses_fts_seg`）断言零行 |
| **闭包元组方向** | 差分对拍时两个集合完全镜像 | `closure` 的 `ancestor` = dst 侧、`descendant` = src 侧；在线 CTE 输出 `(cur, anc)` 需**翻转** |
| `PRAGMA index_info` | 丢失 DESC / collation | 用 `PRAGMA index_xinfo` |
| 事务内 `PRAGMA foreign_keys` | 静默 no-op | 必须在 `BEGIN` **之前**于连接级设置 |

---

## 5. 顺手量化的一个建库性能问题

`validate_schema.py` 输出 `schema 执行成功，耗时 4.5s`。定位后是**文件库逐句 autocommit**（150 次刷盘）：

| 场景 | 耗时 | 倍率 | FK |
|---|---|---|---|
| 文件库，逐句 autocommit（现状） | 4.29 s | 1× | ON |
| 文件库，**整体包一个事务** | **0.113 s** | **38×** | ON（连接级先设） |
| 文件库，单事务 + `journal_mode=MEMORY` | 0.026 s | 165× | ON |
| 内存库 | 0.007 s | 613× | ON |

**但有个陷阱**：`schema_v3.sql` 第 38 行自带的 `PRAGMA foreign_keys=ON` 在事务内是 no-op。
若为了提速把整个 schema 包进 `BEGIN...COMMIT` 而**没有在连接级先设 PRAGMA**，则**种子加载期完全无外键保护**（实测 FK=0）。
两个用例已把它锁住：**5.8j**（陷阱存在）、**5.8k**（`PRAGMA foreign_key_check` 不受开关影响，是唯一兜底）。

推荐建库姿势：

```python
conn = sqlite3.connect("logos.db")
conn.execute("PRAGMA foreign_keys = ON")   # 必须在 BEGIN 之前
conn.execute("PRAGMA recursive_triggers = ON")
conn.executescript("BEGIN;\n" + open("schema_v3.sql", encoding="utf-8").read() + "\nCOMMIT;")
assert conn.execute("PRAGMA foreign_key_check").fetchall() == []   # 兜底自检
```

---

## 6. 运行方式

```bash
python validate_schema.py     # 102 项自检（结构 + 功能 + 定点回归 + 边界）
python probe_gaps.py          # 盲区探针（只诊断，不进套件；用于复核本报告结论）
python audit_indexes.py       # 冗余索引体检
python precheck_schema.py     # 建库前静态检查
python bench_closure.py       # 闭包维护成本基准
```

```
结果: PASS 102 / FAIL 0
表 62 / 视图 5 / 触发器 10 / 索引 34
```

---

## 7. 待你决策的三项

| 编号 | 事项 | 选项 |
|---|---|---|
| D3 | `relation_types.max_depth` 死列 | (a) 删除该列；(b) 删除 `closure_policy.max_depth`，触发器统一读前者（真单一真源） |
| D4 | `off` 模式切换留下陈旧闭包 | 采纳 §2 的 `tr_closure_policy_au` 触发器（6 行） |
| 🟢 | 遗留项：闭包重建触发器按 `r.src_id = w.cur` 关联，而索引无 `src_id` 打头项 | 补 `CREATE INDEX idx_relations_src_id ON relations(src_id) WHERE src_kind='sense';` |
