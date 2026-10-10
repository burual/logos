# Logos 数据库连接层 `logos/db/connection.py`

> 目标运行时：**Python 3.13 / SQLite 3.53.1**（STRICT 表要求 ≥ 3.37）
> 零第三方依赖，只用标准库 `sqlite3`。

---

## 1. 为什么需要一个连接层

直接 `sqlite3.connect()` 有三个"看起来能跑、实际出事"的问题，本项目全都踩过：

| 问题 | 后果 |
|---|---|
| `PRAGMA foreign_keys` **默认 OFF** | 52 张表的全部 64 条外键形同虚设，写入脏数据不报错 |
| 调优 PRAGMA 是**连接级**的 | 换个连接就全部失效，性能与安全性静默回退 |
| `with sqlite3.connect(...) as conn` **不关闭连接** | 只提交事务；连接泄漏，WAL 下长期占用锁 |

连接层把这三件事收敛成一条默认正确的路径。

---

## 2. 快速上手

```python
from logos.db import init_database, connect, get_connection, transaction

# ① 建库 + 填种子（幂等：可反复执行）
report = init_database()                      # 默认 logos.db + schema_v3.sql + seed_*.sql
print(report.render())

# ② 上下文管理器：退出时保证关闭
with connect(read_only=True) as conn:
    conn.execute("SELECT word FROM words LIMIT 5").fetchall()

# ③ 显式事务：成功提交，异常回滚
with get_connection() as conn:
    with transaction(conn):
        conn.execute("INSERT INTO contexts(name) VALUES('新场景')")
```

---

## 3. 接口清单

### 3.1 题目要求的四个接口

| 接口 | 说明 |
|---|---|
| `get_connection(db_path)` | 建连并完成 PRAGMA 调优；返回 `LogosConnection`（`sqlite3.Connection` 子类） |
| `get_memory_cache()` | 进程内哈希缓存，返回 `MemoryCache`（**是 `dict` 子类**） |
| `init_database(db_path, schema_sql_path)` | 建库 + 填种子 + 校验，返回 `InitReport` |
| `close_connection(conn)` | 收尾优化 + 关闭；未闭合事务默认回滚并告警 |

### 3.2 附加接口

| 接口 | 用途 |
|---|---|
| `connect(**kw)` | 上下文管理器（负责 close，补上标准库的缺口） |
| `transaction(conn, mode)` / `savepoint(conn)` | 显式事务 / 正确嵌套 |
| `configure_connection(conn, profile)` | 对既有连接重新套用调优 |
| `close_all()` / `live_connections()` | 泄漏检测与批量收尾 |
| `timed` / `timed_block` / `execute_timed` / `count_sql` | 查询计时 |
| `get_timing_stats()` / `dump_timing_stats()` | 统计输出 |
| `hash_key()` / `file_digest()` | 稳定哈希键 / 文件指纹（缓存失效用） |
| `split_statements(sql)` | 触发器体感知的 SQL 切分（带行号） |
| `PragmaProfile` | 调优档案（不可变、可共享、可覆盖） |

### 3.3 异常体系

```
DatabaseError                       # 基类，携带 db_path / sql / params / cause
├── DatabaseConnectionError         # 打不开（不存在 / 无权限）
├── SchemaError                     # 建表脚本失败（消息含行号）
├── SeedError                       # 种子脚本失败（消息含行号）
├── TransactionError                # 事务用法非法（嵌套、模式非法、事务内配置）
└── IntegrityCheckError             # integrity_check / foreign_key_check 未通过
```

`str(e)` 会拼成 `说明 [db=...] [cause=...] [sql=...]`，日志里可直接定位。

---

## 4. PRAGMA 档案

`DEFAULT_PROFILE` 的取值与**实测读回值**：

| PRAGMA | 设定 | 实测读回 | 作用 |
|---|---|---|---|
| `journal_mode` | `WAL` | `wal` | 读写并发 |
| `synchronous` | `NORMAL` | `1` | WAL 下安全且快 |
| `cache_size` | `-64000` | `-64000` | ≈64MB 页缓存 |
| `temp_store` | `MEMORY` | `2` | 临时表不落盘 |
| `mmap_size` | `268435456` | `268435456` | 256MB 内存映射 |
| `busy_timeout` | `5000` | `5000` | 锁等待 5s |
| `foreign_keys` | `ON` | `1` | **全库引用完整性的唯一保障** |
| `recursive_triggers` | `ON` | `1` | 闭包级联维护 |
| `wal_autocheckpoint` | `1000` | `1000` | 防 `-wal` 无限增长 |
| `defer_foreign_keys` | `OFF` | `0` | 默认立即校验（错误能报到具体行） |

三个实现细节：

1. **顺序有讲究**：`journal_mode` 必须在 `synchronous` 之前设置（改变日志模式可能重置同步级别）；
   `foreign_keys` 放在最后并**强制读回校验**，不为 1 直接抛 `DatabaseError`。
2. **只读/内存库自动跳过** `journal_mode`（只读库上会被拒、内存库无意义），
   并把跳过原因写进 `pragmas_notes`。
3. **值走白名单校验**：PRAGMA 无法参数化，只能拼字符串，因此
   `_PRAGMA_VALUE_RE` 只放行 `-?\d+` 或 `[A-Za-z_]\w*`，`-1; DROP TABLE t` 会被拒。

---

## 5. 四个陷阱与规避方式（全部有实测用例锁定）

### 5.1 `foreign_keys` 默认 OFF，且**事务内是 no-op**

```
事务外：PRAGMA foreign_keys=ON   → 读回 1
事务内：PRAGMA foreign_keys=OFF  → 读回 1（静默忽略！）
```

规避：连接建立时立即设置（必然无事务），随后读回校验；
`configure_connection` 一旦发现 `conn.in_transaction` 就抛
`TransactionError`，**而不是悄悄跳过**。

`validate_connection.py` 里有阳性对照：把 FK 显式关成 0 后，同一笔悬空插入**成功** ——
证明前面的"插入被拒"确实来自 PRAGMA，不是巧合。

### 5.2 `PRAGMA journal_mode=WAL` 的报错**延迟到 fetch**

只 `execute()` 不取行会把错误吞掉。规避：`_set_pragma()` 一律 `.fetchone()`。

### 5.3 `executescript()` 的两个副作用（故本模块不使用它）

| 副作用 | 实测证据 |
|---|---|
| 出错后**留下未闭合事务** | `BEGIN; CREATE...; 坏语句;` → 抛错后 `in_transaction=True` |
| 执行前**隐式提交**调用方的事务 | `BEGIN IMMEDIATE; INSERT...;` 后调 `executescript("SELECT 1;")` → 事务被提交，`ROLLBACK` 报 `no transaction is active` |

规避：用 `split_statements()` 逐句执行，出错时**自行回滚**并报出行号。

### 5.4 `with sqlite3.connect(...)` **不关闭连接**

只提交（异常时回滚）。规避：`connect()` 上下文管理器在 `finally` 里 `close_connection()`。

---

## 6. `split_statements()` —— 一个需要澄清的认知

先纠正一个**容易搞错的结论**：标准库 `sqlite3.complete_statement()` 底层是 SQLite
自带的 `sqlite3_complete()`，它**能够**正确识别 `CREATE TRIGGER ... BEGIN ... END;`
的块结构。实测：

```
简单触发器（体内 2 条语句）  → complete_statement 切为 1 段（完整）
schema_v3.sql 的 10 个触发器 → 全部完整，逐句执行 0 失败
```

所以"官方切分器会被触发器体内的分号切断"是**不成立**的。

那为什么还要自己写？因为 `complete_statement` 有**三条能力缺失**：

| 能力差异 | 自研 | 标准库 | 影响 |
|---|---|---|---|
| **行号** | 返回 `(语句, 行号)` | 只返回文本 | 错误无法报到 `schema.sql:123` |
| **同一行多条语句** | 正确拆开 | 按行累积，会合并 | `schema_v3.sql` 第 1642、1648 行正是这种 |
| **纯注释块** | 跳过 | 当成 1 条"语句"返回 | 两个脚本各多出 1 条不可执行项 |

实测数量：`schema_v3.sql` 自研切 **150** 条、官方 **151** 条，差值恰好是 1 个尾部纯注释块。

自研实现是字符级状态机，正确跳过：单引号字符串（含 `''` 转义）、双引号/反引号/方括号标识符、
`--` 行注释、`/* */` 块注释（其中的分号不切分）、以及 `BEGIN ... END` 与 `CASE ... END`。

`BEGIN` 的歧义（事务 vs 触发器体）按上下文区分：**语句最开头的 `BEGIN` 是事务**（`first_code` 标志），
出现在代码之后的 `BEGIN` 才开启块。

---

## 7. 事务模型

**`isolation_level=None`**：Python 层不再隐式开启事务，`isolation_level` 由本模块固定，
每条语句在无事务时即时生效。这消除了"忘记 commit"和"PRAGMA 落进事务变 no-op"两类问题。

```python
with transaction(conn):          # BEGIN IMMEDIATE → COMMIT / ROLLBACK
    ...
with transaction(conn, allow_nested=True):   # 自动改用 SAVEPOINT
    ...
with savepoint(conn, "sp1"):     # 显式 SAVEPOINT
    ...
```

* 默认 `IMMEDIATE`：立刻取写锁，把"升级锁失败"提前到事务开头，
  避免 WAL 下两个写事务互相饿死（`SQLITE_BUSY_SNAPSHOT`）。
* **嵌套会被提前拦截**并抛 `TransactionError`，而不是等 SQLite 报
  `cannot start a transaction within a transaction`。
* 事务块内若出现 `COMMIT`/`ROLLBACK`，退出时会抛 `TransactionError`（属设计外用法）。

---

## 8. 连接生命周期

`close_connection()` 的三段收尾：

1. **未闭合事务** → 默认 `rollback()` **并打 WARNING**，不静默提交。
   看到这条日志说明某处漏了 `commit()`，那是 bug，不该被自动提交掩盖；
   确实要提交就显式传 `commit=True`。
2. **`PRAGMA optimize`** → SQLite 官方建议的关闭前动作，让统计信息写回；
   仅对可写、非内存、无事务的连接执行，失败静默忽略。
3. 从 `_LIVE`（`WeakSet`）登记表移除。

> 实现上有个小前提：**原生 `sqlite3.Connection` 既不支持属性赋值也不支持 weakref**
> （`AttributeError` / `TypeError`）。所以本模块用 `LogosConnection` 子类承载元数据 ——
> 子类实例是普通 Python 对象，可挂属性、可 weakref，
> 而 `isinstance(conn, sqlite3.Connection)` 依旧为真，对调用方完全透明。

`live_connections()` / `close_all()` 用于泄漏排查：测试收尾时若 `close_all()` 返回值 > 0，
说明调用方漏关了连接。

---

## 9. 内存哈希缓存

```python
cache = get_memory_cache("senses", max_entries=4096)     # isinstance(cache, dict) == True
key = hash_key("sense_detail", word, pos)                # 跨进程稳定（blake2b）
detail = cache.get_or_compute(key, lambda: query(conn, word, pos))
```

* `get_memory_cache()` 按名字返回进程内单例，返回类型 `MemoryCache` 是 `dict` 子类。
* `put()` 有容量上限与 **FIFO 淘汰**，并统计 `hits / misses / evictions / hit_rate`。
* `hash_key()` 用 blake2b —— 内置 `hash()` 对 `str` 会被 `PYTHONHASHSEED` 随机化，
  **不能**当持久缓存键；实测两个不同 `PYTHONHASHSEED` 的子进程得到同一结果。
* `file_digest(path)` 给脚本做 sha256 指纹，可把 schema/seed 版本绑进缓存键，脚本一改缓存自动失效。

> 注意：`cache[k] = v` 会走 `put()` 触发淘汰，但 `dict.update()` / `setdefault()` 是 C 层实现，
> **不会**经过本类的钩子。需要淘汰语义时请显式调用 `put()`。

---

## 10. 计时与性能分析

```python
@timed("search_words")            # 也可写成 @timed（用函数限定名当标签）
def search(conn, kw): ...

with timed_block("batch_enrich"):
    ...

rows, ms = execute_timed(conn, "SELECT ...", label="closure_query")

with count_sql(conn) as info:      # 统计块内执行了几条 SQL（基于 trace callback）
    ...
print(info["count"], info["sql"])

dump_timing_stats()                # 打印 n / total / avg / p50 / p95 / max 表格
```

标签级统计保留最近 1024 个样本算分位数。**异常路径同样计时**（异常耗时可反映超时/锁等待），
异常随后原样抛出。

---

## 11. 实测性能数据

### 11.1 schema 加载：单事务 vs 逐句提交

`schema_v3.sql` 150 条语句，文件库（Windows，含 62 表 / 34 索引 / 10 触发器）：

| 方式 | 耗时 | 库文件 |
|---|---|---|
| 每条语句独立提交（autocommit） | **4302.4 ms** | 572 KB |
| 整包一个事务 | **37.5 ms** | 572 KB |

→ **加速 114.8×**。原因是 autocommit 下 150 次事务各触发一次 fsync；
包进单事务只需一次。

`_load_script` 会**自动判断**：脚本自带顶层事务控制（如 `seed_general_v3.sql` 的
`BEGIN;`/`COMMIT;`）就原样执行、由脚本自己保证原子性；没有则由连接层包一个事务。

### 11.2 `init_database` 端到端

| 场景 | 耗时 |
|---|---|
| 首次建库（185 条语句，含校验） | **169.4 ms** |
| 二次执行（跳过 schema，仅重跑种子） | **74.5 ms** |

产出：表 62 / 视图 5 / 触发器 10 / 索引 34，`integrity_check` 与 `foreign_key_check` 均通过；
二次执行后**逐表行数完全不变**（种子幂等）。

### 11.3 WAL 并发

只读连接持有事务期间，另一写连接仍可同时读（两侧均读到 53 行）——这是 WAL 相对
`journal_mode=DELETE` 的核心收益。

---

## 12. `init_database` 的行为矩阵

| 目标库状态 | `reset=False`（默认） | `reset=True` |
|---|---|---|
| 文件不存在 | 建库 + 加载 schema + 种子 | 同左 |
| 已存在且已建表 | **跳过 schema**，仅重跑种子 | 删除文件后完整重建 |

之所以要区分：`schema_v3.sql` 的 52 个 `CREATE TABLE` **没有** `IF NOT EXISTS`，
重复执行必然报错；而 `seed_general_v3.sql` 通篇 `ON CONFLICT DO NOTHING`，可安全反复执行。

`reset=True` 有**安全护栏**：仅当文件是 `.db` 后缀**或**文件头是 `SQLite format 3` 魔数时才删除，
绝不误删无关文件。

**路径处理**：只读连接用 `Path.as_uri()` 构造 `file:...?mode=ro`，它对非 ASCII 字符做百分号编码。
本项目路径含中文（`数据库logos`），直接拼 `file:` 前缀会打不开库 —— 这一点由
独立子进程的用例锁定。

---

## 13. 验证方式

```bash
python validate_connection.py     # 158 项行为测试，全绿
python -m logos.db.connection     # 打印当前环境的连接层自检
```

`validate_connection.py` 全部是**行为测试**（读回值 / 抛出的异常 / 行数变化 / 与独立实现对拍），
不测"表或函数存不存在"。几处关键设计：

* **阳性对照**：证明 FK 拒绝对应来自 PRAGMA（关掉后插入成功）。
* **差分对拍**：拆句执行 vs `executescript`，产出的表/视图/触发器/索引数量必须完全一致。
* **原子性**：坏脚本加载失败后，先前已执行的 DDL 也必须回滚（无半成品 schema）。
* **幂等性**：二次 `init_database` 后逐表行数比对。
* **跨进程稳定性**：两个不同 `PYTHONHASHSEED` 的子进程得到同一 `hash_key`。

项目全量回归：

```
validate_schema.py       PASS 102 / FAIL 0
validate_seed.py         PASS  65 / FAIL 0
precheck_schema.py       FAIL 0 项 / WARNING 2 项 / PASS 12 项
audit_indexes.py         可删 0 个
validate_connection.py   PASS 158 / FAIL 0
```

---

## 14. 已知边界

1. **`mmap_size` 在内存库上读回 `None`**（mmap 不适用于内存库），属预期，
   因此断言写成 `>0 且 ≤ 请求值`，文件库上实测为 268435456。
2. **`journal_mode` 可能被文件系统拒绝**（网络盘/部分挂载不支持 WAL）。
   此时不抛错，但会在 `pragmas_notes` 与日志里给出 WARNING —— 调用方不应假设它一定是 WAL。
3. **`get_connection` 在写模式下会静默创建空库**（继承 `sqlite3` 语义）。
   读场景建议 `must_exist=True`，或先调 `init_database`。
4. **未提供连接池**。当前每处 `connect()` 独立建连；WAL 下多连接并发读没有问题。
   若后续出现高频短连接场景，可在此基础上加一个以 `(db_path, read_only)` 为键的池 ——
   但要注意 PRAGMA 必须逐连接重放（这正是本模块已封装的逻辑）。
5. `count_sql()` 依赖 `set_trace_callback`。由于 `sqlite3` **没有读取当前 trace
   callback 的接口**，本函数无法自动保留调用方原有的 tracer ——
   若你先前装过，请通过 `count_sql(conn, restore=my_tracer)` 传回来，
   块结束时会重新装上；未传则清空为 `None`。
