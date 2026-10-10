# 闭包维护层（`logos/db/closure.py`）

> 适用对象：`logos/db/closure.py`、`validate_closure.py`、所有需要「上位/下位/属性继承」的应用代码
> 依赖权威口径：`docs/CLOSURE_DIRECTION.md`（方向与不变量）、`docs/CONNECTION_LAYER.md`（连接层）
> 实测环境：Python 3.13.12（托管）+ SQLite 3.53.1

---

## 1. 定位

`closure` 表是**物化**的传递闭包。它由三条路径维护，本模块是其中两条的 Python 实现，并与第三条（触发器）保持**结果等价**：

| 路径 | 触发者 | 何时用 | 入口 |
|---|---|---|---|
| 增量·触发器 | `tr_closure_edge_ai` / `tr_closure_edge_ad` | 交互式单条增删（`mode='incremental'`） | 无（插 `relations` 即自动） |
| 增量·Python | — | 触发器被关（`batch`/`off`）时手工补 | `incremental_update()` |
| 全量·Python | — | 批量导入后一次性重建 | `rebuild_closure()` |
| 全量·DB 队列 | `tr_closure_rebuild_ai` | 想在数据库侧完成重建 | `queue_rebuild()` |

三条路径的结果一致性由不变量 **I6**（差分对拍）锁定。

---

## 2. 公开 API

| 函数 | 作用 | 返回 |
|---|---|---|
| `rebuild_closure(conn, relation=None)` | 全量/按关系重建；`off` 关系清空、缺策略行跳过 | `RebuildReport` |
| `incremental_update(conn, source_id, target_id, relation)` | 插边后手工补闭包增量 | `IncrementalReport` |
| `get_ancestors(conn, node_id, relation=None, max_depth=None)` | 全部**上位**，`depth` 升序 | `list[ClosureEntry]` |
| `get_descendants(...)` | 全部**下位**，`depth` 升序 | `list[ClosureEntry]` |
| `get_property_inheritance(conn, concept_id, property_relation="has_property")` | 属性继承（推理引擎核心） | `list[InheritedProperty]` |
| `validate_closure_integrals(conn, ...)` | 校验 I1–I5（可选 I6 / P1–P2） | `bool` |
| `bulk_import_mode_off(conn, relations=None, strategy="policy")` | 进入批量模式（暂停维护） | `BulkState` |
| `bulk_import_mode_on(conn, state=None, rebuild=True)` | 退出批量模式（恢复 + 重建） | `int` |
| `iter_ancestors` / `iter_descendants` | **生成器**版，大结果集流式读取 | `Iterator[ClosureEntry]` |
| `get_ancestor_ids` / `get_descendant_ids` | 只要 id 列表 | `list[int]` |
| `inspect_closure_integrals` | 结构化不一致明细（含具体违反行） | `list[Violation]` |
| `queue_rebuild` / `closure_stats` / `bulk_import`（上下文管理器） | 便利函数 | — |

结果类型：`ClosureEntry(node_id, depth, relation, first_edge)`、`InheritedProperty(property_id, distance, declared_by)`。

---

## 3. 方向速查（照抄，勿自行改列名）

```
relations 边：   src(下位/具体)  --relation-->  dst(上位/一般)
closure  行：   (ancestor=上位/一般, descendant=下位/具体, depth)
```

> **自检口诀**：问"谁的上位"→ 约束 `descendant`；问"谁的下位"→ 约束 `ancestor`。

```python
get_ancestors(conn, 老虎, "is_a")     # → [动物(d1), 生物(d2)]   WHERE descendant=老虎
get_descendants(conn, 生物, "is_a")   # → [动物(d1), 老虎(d2), 东北虎(d3)]
```

**属性继承方向**（`CLOSURE_DIRECTION.md` §4）：

```python
# 属性挂在上位概念上、下位概念继承
get_property_inheritance(conn, 老虎)   # → 老虎自身属性(d0) + 动物属性(d1)
                                       #    绝不会出现"东北虎"的属性（那是下位）
```

代码里的正确形式是 `JOIN relations r ON r.src_id = c.ancestor`（**上位**概念持有属性）；
写成 `r.src_id = c.descendant` 会取到**子类**属性 —— 语法正确、结果荒谬。

---

## 4. 全量重建：两条路径，同一结果

`rebuild_closure` 用**前导式递归 CTE**（Python 层合法）：

```sql
WITH RECURSIVE closure_build(down_id, up_id, depth, first_edge, visited) AS (
    SELECT r.src_id, r.dst_id, 1, r.relation_id, ','||r.src_id||','||r.dst_id||','  -- 播种
      FROM relations r WHERE r.relation = ? AND r.src_kind='sense' AND r.dst_kind='sense'
    UNION ALL
    SELECT cb.down_id, r.dst_id, cb.depth + 1, cb.first_edge, cb.visited||r.dst_id||','  -- 向上延伸
      FROM closure_build cb JOIN relations r ON r.src_id = cb.up_id
     WHERE r.relation = ? AND ... AND cb.depth < ? AND instr(cb.visited, ','||r.dst_id||',') = 0
)
INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
SELECT ?, cb.up_id, cb.down_id, MIN(cb.depth), cb.first_edge     -- 裸列+MIN 取回见证边
  FROM closure_build cb WHERE cb.up_id <> cb.down_id
 GROUP BY cb.down_id, cb.up_id
```

* `down_id` = 边的 src 端（下位），`up_id` = 向上走到的（上位）→ 落表时 `ancestor=up_id`。
* `MIN(depth)` + 裸列 `first_edge`：SQLite 的裸列取自**取到最小值的那一行**，一次拿到「最短深度 + 见证边」。
* `visited` 字符串防环；`cb.depth < max_depth` 剪枝；最终 `GROUP BY` 去重。
* **不使用触发器**，与 `queue_rebuild`（DB 侧）结果逐行相同 —— 由 `validate_closure.py` S3 用集合相等断言。

### 4.1 ⚠️ `rowcount` 陷阱（本次实测踩坑）

CPython 的 `sqlite3` 按语句**首个关键字**判定类型。前导式为 `WITH` 的 `WITH ... INSERT`
**不被识别为 INSERT**，`cursor.rowcount` 恒为 `-1` —— 重建报告会显示"写入 0 行"，而实际写入正常。

**修法**：一律用 `Connection.total_changes` 的前后差值计数。

```python
before = conn.total_changes
conn.execute(_REBUILD_SQL, (...))
written = conn.total_changes - before        # ✅ 而非 cursor.rowcount（会是 -1）
```

---

## 5. 增量更新

`incremental_update` 是触发器 `tr_closure_edge_ai` 的**手工对应物**，两者同构：

* `lo`：从 `src` 出发沿「上位→下位」反向收集 **src 的下位子树**（含 src 自身）。
* `hi`：从 `dst` 出发沿「下位→上位」收集 **dst 的上位链**（含 dst 自身）。
* 新配对 `(ancestor=H.y, descendant=L.x)`，`depth = L.d + 1 + H.d`。
* UPSERT：`first_edge = CASE WHEN excluded.depth < depth THEN ... END` **必须写在** `depth = MIN(...)` **之前**
  （SET 从左到右求值）。

**受影响对数**（`CLOSURE_DIRECTION.md` §7 实测公式）：

```
受影响配对 = (1 + |src 的下位子树|) × (1 + |dst 的上位链|)
```

`IncrementalReport.affected_pairs` 即此上界，`written` 是实际写入行数。

退出条件：传入的边必须已在 `relations` 里（本函数据其取 `first_edge`），否则抛 `ClosureError`。
`mode='incremental'` 时触发器已做过同样维护，再调用是**幂等**的（UPSERT 收敛）。

---

## 6. 批量导入

批量导入时**绝不能**保持 `incremental`：每条边都触发一次子树笛卡尔积，复杂度瞬间劣化。
单边代价实测（`bench_closure.py`）：叶子 ≈ 1.15 ms，根节点 ≈ 561.3 ms。

```python
with bulk_import(conn):                      # 进入：暂停维护
    for edge in huge_edge_list:
        conn.execute("INSERT INTO relations(...) VALUES(?,?,?)", edge)
# 退出：恢复维护 → 全量重建 → 可继续校验
```

### 6.1 两种策略

| 策略 | 做法 | 特点 |
|---|---|---|
| `policy`（默认） | 把目标关系 `closure_policy.mode` 置 `off` | schema 原生、可逆、无风险；两触发器（条件依赖 `mode`）双双失效 |
| `drop` | 真正 `DROP TRIGGER` 两个闭包触发器（先备份 CREATE 语句） | 连 `WHEN` 判断开销都省掉；期间触发器缺失，必须确保恢复 |

> SQLite 没有 `SET TRIGGERS DISABLED`。`policy` 策略通过 `mode='off'` 达到等效效果。

### 6.2 异常安全

`bulk_import` 上下文即使块内抛错，也会先 `bulk_import_mode_on(..., rebuild=False)`
把触发器恢复，**绝不把库留在"闭包静默不更新"的状态**，然后原样抛出异常。
恢复顺序：**先恢复触发，再重建**。

---

## 7. 不变量

`validate_closure_integrals` 默认校验 I1–I5；`include_differential=True` 追加 I6；
`include_policy=True` 追加 P1/P2。返回 `False` 时，逐条违规（含**具体违反行**）打到
`logos.db.closure` 日志的 ERROR 级，结构化明细见 `inspect_closure_integrals`。

| # | 含义 | 可被什么触发 |
|---|---|---|
| I1 | 闭包无自环 | 表级 `CHECK(ancestor<>descendant)` 先挡；检测器专测见 S10 |
| I2 | 闭包行不指向"无义项边的关系" | 关系被删空后残留闭包 |
| I3 | `depth=1` 行与 `relations` 一一对应 | **方向断言**：方向写反立即命中 |
| I4 | `first_edge` 非空且指向同关系边 | 见证边置空 / 指向他关系 |
| I5 | 深度 ≤ `closure_policy.max_depth` | 深度上限卡口缺失 |
| I6 | 物化 ≡ 在线重建 | 增量路径与重建路径不一致（**唯一能发现"同一份数据两种闭包"的检查**） |
| P1 | `off`/非传递关系不残留闭包行 | 策略漂移 |
| P2 | `is_transitive=1` 必有策略行 | 缺行会**静默无闭包**（`CLOSURE_DIRECTION.md` §6.5） |

I6 用**独立实现**（只读 `relations` 的在线递归 CTE）当 oracle，绝不拿闭包验闭包。

---

## 8. 实测踩坑清单

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | 重建报告"写入 0 行" | 前导式 `WITH` 不被 CPython 判为 DML，`rowcount=-1` | 用 `total_changes` 差值 |
| 2 | I6 报告"缺 N / 多 N"对称 | oracle 的 `cur` 实为**上位**、`anc` 实为**下位**，比较列对调了 | CTE 列改名 `desc_id/anc_id`，按语义比较 |
| 3 | 触发器内写 `WITH ... INSERT` 报语法错 | 前导式 `WITH` 在触发器体内非法 | 嵌 `FROM` 子查询；或改 `INSERT ... WITH ... SELECT`（合法） |
| 4 | UPSERT 后见证边永不修正 | `SET` 从左到右求值，`depth` 若写在 `first_edge` 前则条件恒假 | `first_edge` 必须在前 |
| 5 | `first_edge` 加 `ON DELETE SET NULL` 后失效行删不掉 | 外键动作先于 AFTER DELETE 触发器 | 弱引用，不加外键 |
| 6 | `is_transitive=1` 却静默无闭包 | 缺 `closure_policy` 行 → 子查询 NULL → 比较为假 | 显式补策略行；P2 锁定 |

---

## 9. 索引与查询计划（实测）

| 查询 | 约束列 | 命中索引 |
|---|---|---|
| `get_ancestors` | `relation, descendant` | `idx_closure_relation_desc(relation, descendant, depth)`，无排序 |
| `get_descendants` | `relation, ancestor` | `PK(relation, ancestor, descendant)`（`SEARCH c USING PRIMARY KEY`） |

两者均**不做全表扫描**（`validate_closure.py` S7 用 `EXPLAIN QUERY PLAN` 断言）。
`relation=None`（跨关系查询）无法走上述索引，属低频路径，需要时请显式传关系名。

---

## 10. 校验与基准

```bash
python validate_closure.py        # 87 项，覆盖 S0–S10
python bench_closure.py           # 闭包维护成本基准
```

`validate_closure.py` 关键实测（SQLite 3.53.1）：

* **PASS 87 / FAIL 0**
* 批量导入 121 节点树（闭包 452 行）：**53 ms**（含暂停→导入→恢复→全量重建）
* 环图 `A→B→D→A` 闭包与 oracle 逐行一致（剪枝正确、无自环）
* `drop` 策略往返：触发器移除 → 复原 → 仍可工作

---

## 11. 已知遗留

* `relation=None` 的上位/下位查询不走索引（无 `closure(descendant)` 单列索引）。
  若跨关系查询成为热路径，需评估新增索引（当前不为它付写入成本）。
* `relation_types.max_depth` 与 `closure_policy.max_depth` 双列并存，本层统一以
  **`closure_policy.max_depth`** 为准（缺策略行才退回前者并告警）。
