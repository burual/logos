# 闭包方向约定 (Closure Direction Convention)

> 适用对象：`closure` 表、`relations` 表、§13.2/§13.3 两个闭包触发器、所有依赖
> "上位/下位"推理的应用代码与规则。
>
> 本文是**唯一权威口径**。任何与该文冲突的代码注释、旧文档、口头约定一律作废。

---

## 1. 为什么需要一纸约定

v2.0 的闭包表只有两个列名，没有方向语义：

```sql
CREATE TABLE closure (
    ancestor   INTEGER NOT NULL,
    descendant INTEGER NOT NULL,
    ...
);
```

列名 `ancestor` / `descendant` 在**中文语境下极易反向理解**：说到"祖先"多数人
先想到"更一般、更上位"的概念（动物是老虎的祖先），但 `is_a` 边存储方向是
`老虎 → 动物`（从具体指向一般）。两套直觉直接冲突，导致：

- 取"某概念的全部上位"时写成 `WHERE ancestor = :x`，实际拿到全部下位；
- "属性继承"推理规则取到的是**子类**属性而非父类属性，整条推理链取反；
- 由于闭包是物化的，错误不会报错，只会**静默给出语义相反的结果**。

结论：方向必须由文档 + 约束 + 测试三重锁定。

---

## 2. 权威定义

### 2.1 边的方向（`relations` 表）

```
关系边的语义：src  is_a  dst
              ↑ 下位(具体)   ↑ 上位(一般)

示例：  老虎(13) --is_a--> 动物(12) --is_a--> 生物(100)
```

即 **边总是从"具体"指向"一般"**。

### 2.2 闭包行的方向（`closure` 表）

```
一行 (ancestor = X, descendant = Y)  读作：  X 是 Y 的上位，Y is_a X
                                            X = 更一般，Y = 更具体

示例：  (12, 13)   动物 是 老虎 的上位
        (100, 13)  生物 是 老虎 的上位（depth = 2）
        (100, 12)  生物 是 动物 的上位
```

**记忆口诀**：行的方向 = 边的**反向**。

```
边:    老虎 --is_a--> 动物
行:    (ancestor=动物, descendant=老虎)      ← 注意左右一颠倒
```

### 2.3 因果方向对照表

| 存储位置 | 列名 | 语义 | 位置 |
|---|---|---|---|
| `relations` | `src_id` | 下位 / 具体 | 左 |
| `relations` | `dst_id` | 上位 / 一般 | 右 |
| `closure` | `ancestor` | **上位 / 一般** | 左 |
| `closure` | `descendant` | **下位 / 具体** | 右 |

> `ancestor` 的命名取自"**上位祖先**"义（植物学分类习惯：属是种的祖先概念），
> 而非"谁先出现"。这是本约定的官方解释，不要再做别的联想。

---

## 3. 标准查询配方（照抄即可，不要自己改写列名）

```sql
-- ① 取概念 Y 的全部上位（沿 is_a 向上走：老虎 → 动物 → 生物）
SELECT c.ancestor AS super_concept, c.depth
  FROM closure c
 WHERE c.relation = 'is_a' AND c.descendant = :y      -- 过滤 descendant！
 ORDER BY c.depth;

-- ② 取概念 X 的全部下位（含所有子孙，如 动物 → 老虎/猫/狗…）
SELECT c.descendant AS sub_concept, c.depth
  FROM closure c
 WHERE c.relation = 'is_a' AND c.ancestor = :x        -- 过滤 ancestor！
 ORDER BY c.depth;

-- ③ 判定 A 是否为 B 的上位（A is_a 的祖先链里有没有 B）
SELECT EXISTS (SELECT 1 FROM closure
                WHERE relation = 'is_a' AND ancestor = :A AND descendant = :B);

-- ④ 最短路径长度
SELECT MIN(depth) FROM closure
 WHERE relation = 'is_a' AND ancestor = :A AND descendant = :B;

-- ⑤ 见证边（哪条边撑起了这条闭包，删除时用于定位失效行）
SELECT r.* FROM closure c JOIN relations r ON r.relation_id = c.first_edge
 WHERE c.relation = 'is_a' AND c.ancestor = :A AND c.descendant = :B;
```

**自检口决**：问"谁的上位"就约束 `descendant`；问"谁的下位"就约束 `ancestor`。

---

## 4. 属性继承为什么依赖方向

`has_property` 的继承方向是：**属性挂在上位概念上，下位概念继承**。

```sql
-- 正确：取 concept 的全部上位，再沿它们的 has_property 拿属性
SELECT r.dst_id AS property
  FROM closure c
  JOIN relations r ON r.src_id = c.ancestor        -- 上位概念持有属性
 WHERE c.relation = 'is_a'
   AND c.descendant = :concept                     -- 我是谁的下位
   AND r.relation = 'has_property'
   AND r.src_kind = 'sense' AND r.dst_kind = 'sense';

-- 错误（若把方向写反，等价于"向下继承"）：r.src_id = c.descendant
```

若 §2 的方向被写反，这条规则会去取**子类**的属性——语法正确、结果荒谬。

---

## 5. 不变量（Invariants）

应用层与 CI 应保证以下恒真，任一被打破即为方向或填充逻辑出错：

| # | 不变量 | 检查 SQL（应返回 0 行） |
|---|---|---|
| I1 | 闭包不含自环 | `SELECT * FROM closure WHERE ancestor = descendant;` |
| I2 | 每条闭包行都能在 `relations` 上找到一条同关系的边（无孤儿） | `SELECT c.* FROM closure c WHERE NOT EXISTS (SELECT 1 FROM relations r WHERE r.relation = c.relation AND r.src_kind='sense' AND r.dst_kind='sense');` |
| I3 | `depth = 1` 的闭包行与 `relations` 一一对应 | `SELECT c.* FROM closure c WHERE c.depth = 1 AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.relation = c.relation AND r.src_id = c.descendant AND r.dst_id = c.ancestor);` |
| I4 | `first_edge` 非空且指向同关系的边 | `SELECT * FROM closure WHERE first_edge IS NULL OR first_edge NOT IN (SELECT relation_id FROM relations WHERE relation = closure.relation);` |
| I5 | 深度不超过策略上限 | `SELECT c.* FROM closure c JOIN closure_policy p ON p.relation = c.relation WHERE c.depth > p.max_depth;` |
| I6 | **同一份 `relations`，增量维护的结果与全量重建的结果必须完全相同** | 见下方"差分对拍"配方 |

> I3 是**方向断言的核心**：它直接把 `closure` 行翻译回 `relations` 的边。
> 只要满足 `dst_id = ancestor AND src_id = descendant`，方向就没错。已纳入
> `validate_schema.py` 的回归用例。
>
> **I5 曾经不成立（v3.0.0 缺陷，v3.0.1 已修）**：增量触发器的 `lo`/`hi` 递归项卡的是
> `d ≤ max_depth`，但最终深度是 `L.d + 1 + H.d`，因此**实际上限是 `2*max_depth + 1`**；
> 而重建路径严格 `≤ max_depth`。实测 `max_depth=2` 时：增量产出 6 行（最大 depth=3），
> 重建产出 5 行（最大 depth=2）—— 同一份数据两种闭包。
> 修法：`lo`/`hi` 只做 `≤ max_depth - 1` 的剪枝，最终在 `WHERE` 统一卡口
> `L.d + 1 + H.d <= max_depth`。
>
> **I6 的差分对拍配方**（`validate_schema.py` 的 5.3/5.5 即按此实现）。用**独立实现**在线计算，
> 不要拿闭包去验闭包：
>
> ```sql
> -- oracle：在线递归 CTE。
> -- ⚠️ 列名 anc/cur 极易读反：anc 固定为边的 src 端（下位/具体），
> --    cur 沿 src_id 向上走（上位/一般）。故 `SELECT cur, anc` 得到的是
> --    (ancestor, descendant)，与 closure 表列序**一致**，**不需要对调**。
> WITH RECURSIVE w(anc, cur, d, visited) AS (
>     SELECT r.src_id, r.dst_id, 1, ','||r.src_id||','||r.dst_id||','
>       FROM relations r
>      WHERE r.relation = :rel AND r.src_kind='sense' AND r.dst_kind='sense'
>     UNION ALL
>     SELECT w.anc, r.dst_id, w.d+1, w.visited||r.dst_id||','
>       FROM w JOIN relations r ON r.src_id = w.cur
>      WHERE r.relation = :rel AND r.src_kind='sense' AND r.dst_kind='sense'
>        AND instr(w.visited, ','||r.dst_id||',') = 0
>        AND w.d < :max_depth)
> SELECT cur, anc, MIN(d) FROM w WHERE cur <> anc GROUP BY cur, anc;
> ```
>
> 再与 `SELECT descendant, ancestor, depth FROM closure WHERE relation = :rel` 做集合比较。

---

## 6. 触发器实现的五个硬约束（均已实测）

### 6.1 触发器体内不能用**前导式** `WITH ... INSERT`

**失败的是「`WITH` 打头」这一种写法**（实测 SQLite 3.53.1）：

```sql
-- ✗ 语法错误：near "INSERT": syntax error
CREATE TRIGGER t AFTER INSERT ON x BEGIN
    WITH RECURSIVE w AS (...) INSERT INTO y(...) SELECT * FROM w;
END;
```

**必须把递归 CTE 嵌进 `FROM` 子查询**：

```sql
-- ✓ 正确写法
INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
SELECT NEW.relation, H.y, L.x, L.d + 1 + H.d, L.w
  FROM (WITH RECURSIVE lo(x, d, w) AS ( ... ) SELECT x, d, w FROM lo) AS L
 CROSS JOIN (WITH RECURSIVE hi(y, d) AS ( ... ) SELECT y, d FROM hi) AS H
 WHERE L.x <> H.y;
```

**三种写法的合法性矩阵**（本机 SQLite 3.53.1 实测）：

| 写法 | 触发器体内 | Python 层 `execute()` |
|---|---|---|
| `WITH ... INSERT INTO ... SELECT`（前导式） | ✗ `near "INSERT"` | ✓ |
| `INSERT INTO ... WITH ... SELECT` | ✓ | ✓ |
| `INSERT ... SELECT ... FROM (WITH ...) AS L` | ✓ | ✓ |

即：**限制只针对「`WITH` 作为语句首关键字」，且只存在于触发器体内**；
``INSERT ... WITH ... SELECT`` 在触发器里其实合法，Python 层更是三种都行。
`logos/db/closure.py` 的重建走前导式（Python 层），因此不被此约束影响。

> **⚠️ `rowcount` 陷阱**：CPython 的 `sqlite3` 按语句**首个关键字**判定类型，
> 前导式为 `WITH` 的 DML **不被识别为 INSERT/DELETE**，`cursor.rowcount` 恒返回 `-1`。
> 统计前导式 CTE 写入了多少行，必须用 `Connection.total_changes` 的前后差值。

### 6.2 UPSERT 的 `SET` 从左到右求值，后者看得见前者

```sql
-- ✓ first_edge 必须写在 depth 之前
ON CONFLICT(relation, ancestor, descendant) DO UPDATE SET
    first_edge = CASE WHEN excluded.depth < depth THEN excluded.first_edge ELSE first_edge END,
    depth      = MIN(depth, excluded.depth);
```

若颠倒顺序，`excluded.depth < depth` 恒为假（`depth` 已被更新为更小值），
见证边永远不会被修正，删除时的失效定位随之失效。

### 6.3 `first_edge` 刻意不加外键

```sql
first_edge INTEGER,          -- 弱引用，故意不加 REFERENCES
```

若加 `ON DELETE SET NULL`，外键动作会在 **AFTER DELETE 触发器之前**把值清空，
触发器里的 `DELETE FROM closure WHERE first_edge = OLD.relation_id` 就再也匹配
不到任何行，失效行永远删不掉。这是 v3.0 初稿踩过的真实坑。

### 6.4 深度上限必须在**最终 WHERE** 卡口，不能只卡递归项

见 §5 的 I5 说明。`lo`/`hi` 的递归项只负责剪枝（`≤ max_depth - 1`），
真正决定"这一行该不该写入"的是 `L.d + 1 + H.d <= max_depth`。
漏掉这一步会让增量路径的实际上限变成 `2*max_depth + 1`，与重建路径不一致。

### 6.5 `is_transitive = 1` 的关系**必须**有显式 `closure_policy` 行

触发器条件是 `(SELECT mode FROM closure_policy WHERE relation = NEW.relation) = 'incremental'`。
**缺行时子查询返回 NULL，比较为假 → 不报错、不插入、不产生任何闭包。**

v3.0.0 实测有 7 个关系中招（`can_do` / `caused_by` / `entails` / `has_property` /
`precedes` / `presupposes` / `topic_of`）—— 连插两条 `can_do` 边后 `closure` 行数仍为 0。
v3.0.1 已补齐，并由 `validate_schema.py` 的 5.0a 永久锁定：
**不要依赖任何隐式默认，`off` 也要显式写出来。**

---

## 7. 维护模式与代价（实测数据）

`closure_policy.mode` 三档：`incremental` / `batch` / `off`。

插入单条边的代价 ≈ **(1 + |src 的下位子树|) × (1 + |dst 的上位链|)**：

| src 挂载位置 | src 下位子树规模 | 单边插入延迟中位数 |
|---|---|---|
| 叶子节点 | 0 | ≈ 1.15 ms |
| 第 2 层 | 120 | ≈ 62.5 ms |
| 根节点 | 1092 | ≈ 561.3 ms |

**运维节奏**：

```sql
-- 交互式单条增删（用户手工加一条关系）
UPDATE closure_policy SET mode = 'incremental' WHERE relation = 'is_a';

-- 批量导入：先关增量触发
UPDATE closure_policy SET mode = 'off' WHERE relation = 'is_a';
--    … 大批量 INSERT relations …

-- 导入完成：一行触发全量重建（幂等，可重复调用）
UPDATE closure_policy SET mode = 'batch' WHERE relation = 'is_a';
INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at)
     VALUES ('is_a', strftime('%Y-%m-%dT%H:%M:%fZ','now'));
```

**绝不要**在批量导入时保持 `incremental`：每条边都触发一次子树笛卡尔积，
复杂度瞬间劣化到不可接受。

---

## 8. 相关对象索引

| 对象 | 位置 |
|---|---|
| `closure` 表定义 + 方向注释 | `schema_v3.sql` §7（`CREATE TABLE closure`） |
| 增量维护触发器 | `schema_v3.sql` §13.2 `tr_closure_edge_ai` / `tr_closure_edge_ad` |
| 全量重建触发器 | `schema_v3.sql` §13.3 `tr_closure_rebuild_ai` |
| 在线兜底递归（不依赖物化） | `schema_v3.sql` §15.3 |
| 方向回归用例 | `validate_schema.py` STEP 2「方向语义」与 STEP 4「R8/R9」 |
| 成本基准 | `bench_closure.py` |

---

## 9. 变更流程

任何触及 `relations` 语义方向或 `closure` 填充逻辑的改动，必须同时：

1. 更新本文 §2 的对照表；
2. 保证 §5 的 I1–I5 全部返回 0 行；
3. 跑通 `validate_schema.py` 的方向回归用例（R8 取上位、R9 触发器写法）；
4. 若改了触发器写法，跑一遍 `bench_closure.py` 确认维护代价没有数量级回退。

> 落款：本约定随 schema v3.0 生效（本机 SQLite 3.53.1 实测）。
