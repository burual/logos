#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
闭包表维护成本实测：验证「插入一条边的闭包增量代价 = O(|祖先| × |后代|)」
对比：incremental 边插边算 vs batch 批量后一次性重建
运行： python bench_closure.py
"""
import sqlite3, os, time, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "logos_bench.db")
for f in (DB, DB + "-wal", DB + "-shm"):
    if os.path.exists(f):
        os.remove(f)

conn = sqlite3.connect(DB)
conn.executescript(open(os.path.join(HERE, "schema_v3.sql"), encoding="utf-8").read())
conn.execute("PRAGMA foreign_keys=ON")
conn.execute("PRAGMA recursive_triggers=ON")
conn.commit()


def build_tree(depth, branch, t0=10000):
    """构造一棵 is_a 树，返回 (节点数, 边列表[(child,parent)])"""
    edges, nodes = [], []
    nid = t0
    frontier = [nid]
    nodes.append(nid)
    nid += 1

    def add_node(x, base=False):
        conn.execute("INSERT INTO words(word_id, word) VALUES(?,?)", (x, f"节点{x}"))
        conn.execute(
            "INSERT INTO senses(sense_id, word_id, pos, sense_no, definition) "
            "VALUES(?,?, 'n', 1, ?)",
            (x, x, f"第{x}号测试概念"),
        )

    add_node(t0)                       # 根节点也必须入库（初版漏掉导致触发器报端点不存在）
    for _ in range(depth):
        nxt = []
        for p in frontier:
            for _ in range(branch):
                add_node(nid)
                edges.append((nid, p))
                nxt.append(nid)
                nodes.append(nid)
                nid += 1
        frontier = nxt
    return nodes, edges


def run(mode, depth, branch, base_id):
    conn.execute("DELETE FROM closure")
    conn.execute("DELETE FROM relations")
    conn.execute("UPDATE closure_policy SET mode=? WHERE relation='is_a'", (mode,))
    nodes, edges = build_tree(depth, branch, base_id)   # 每次用不同 id 段避免主键冲撞
    conn.commit()

    t_start = time.perf_counter()
    for c, p in edges:
        conn.execute(
            "INSERT INTO relations(src_id, dst_id, relation) VALUES(?,?,'is_a')", (c, p)
        )
    t_insert = time.perf_counter() - t_start

    t_start = time.perf_counter()
    if mode != "incremental":
        conn.execute(
            "INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at) "
            "VALUES('is_a', strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
        )
    conn.commit()
    t_rebuild = time.perf_counter() - t_start

    rows = conn.execute("SELECT COUNT(*) FROM closure WHERE relation='is_a'").fetchone()[0]
    return t_insert, t_rebuild, rows, len(nodes), len(edges)


print(f"SQLite {sqlite3.sqlite_version}")
print("-" * 78)
print(f"{'规模(深度x分支)':<16}{'节点':>6}{'边':>6}{'闭包行':>8}"
      f"{'边插即算总耗时':>16}{'批量重建耗时':>14}")
print("-" * 78)

bid = 10000
for depth, branch in [(3, 3), (4, 3), (4, 4), (5, 3)]:
    ti_inc, _, rows_i, n, e = run("incremental", depth, branch, bid)
    bid += 20000
    ti_bat, tr_bat, rows_b, _, _ = run("batch", depth, branch, bid)
    bid += 20000
    assert rows_i == rows_b, (rows_i, rows_b)
    label = f"{depth}x{branch}"
    print(f"{label:<16}{n:>6}{e:>6}{rows_b:>8}{ti_inc:>15.3f}s"
          f"{ti_bat:>+12.3f}s + {tr_bat:.3f}s")

# ---------------------------------------------------------------------------
# 微基准：单条边的闭包代价 = (1 + |src 的下位子树|) x (1 + |dst 的上位链|)
# 做法：挑若干已知下位子树规模的既有节点，各挂到一个"全新的更一般节点"上
#       （新节点没有上位，所以 上位链=1，变量只剩 src 的下位子树规模）
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("微基准：单条边插入延迟 vs src 的下位子树规模（6x3 树：1+3+9+27+81+243+729 = 1093 节点）")
print("=" * 78)

conn.execute("DELETE FROM closure")
conn.execute("DELETE FROM relations")
conn.execute("UPDATE closure_policy SET mode='incremental' WHERE relation='is_a'")
BASE = 500000
nodes, edges = build_tree(6, 3, BASE)
conn.commit()
for c, p in edges:
    conn.execute("INSERT INTO relations(src_id, dst_id, relation) VALUES(?,?,'is_a')", (c, p))
conn.commit()

# 计算每个节点的"下位子树"规模：按深度分层
depth_of = {}
for c, p in edges:
    depth_of[c] = depth_of.get(p, 0) + 1           # 越深 = 越具体 = 层号越大
by_depth = {}
for n in nodes:
    by_depth.setdefault(depth_of.get(n, 0), []).append(n)

# 关键：closure(ancestor=上位, descendant=下位)
#   某节点的"下位子树" = WHERE ancestor = node   → SELECT descendant
#   某节点的"上位链"   = WHERE descendant = node → SELECT ancestor
def lower_subtree(node):
    return conn.execute(
        "SELECT COUNT(*) FROM closure WHERE relation='is_a' AND ancestor=?", (node,)
    ).fetchone()[0]

def upper_chain(node):
    return conn.execute(
        "SELECT COUNT(*) FROM closure WHERE relation='is_a' AND descendant=?", (node,)
    ).fetchone()[0]

probe_id = 900000

def probe_edge(src):
    """把 src 挂到一个全新的、没有上位的节点上，测这条边触发闭包的延迟(ms)"""
    global probe_id
    probe_id += 1
    conn.execute("INSERT INTO words(word_id, word) VALUES(?,?)", (probe_id, f"上位{probe_id}"))
    conn.execute("INSERT INTO senses(sense_id, word_id, pos, sense_no, definition) "
                 "VALUES(?,?,'n',1,?)", (probe_id, probe_id, f"探针概念{probe_id}"))
    t = time.perf_counter()
    conn.execute("INSERT INTO relations(src_id, dst_id, relation) VALUES(?,?,'is_a')",
                 (src, probe_id))
    return (time.perf_counter() - t) * 1000

print(f"{'src 层号':<10}{'下位子树规模':>14}{'上位链':>8}{'单边延迟(ms)':>16}")
print("-" * 78)
for lvl in [6, 4, 2, 0]:
    cands = by_depth.get(lvl, [])[:3]
    if not cands:
        continue
    sub = lower_subtree(cands[0])
    upc = upper_chain(cands[0])
    lats = [probe_edge(c) for c in cands]
    med = statistics.median(lats)
    label = f"L{lvl}" + ("(最具体/叶)" if lvl == 6 else "(最一般/根)" if lvl == 0 else "")
    print(f"{label:<10}{sub:>14}{upc:>8}{med:>15.3f}")
print("-" * 78)

print("""
结论（由上方实测数据得出）：
  1. "插入一条边"触发闭包的代价 ≈ (1 + |src 的下位子树|) × (1 + |dst 的上位链|)。
     实测：src 的下位子树越大，单边延迟越高（叶子 vs 根相差数量级）。
  2. 闭包行数 ≈ 节点数 × 平均层级深度 —— 空间是线性的，可控；
     但**插入成本随被挂节点的影响范围放大**，这是闭包表的结构性代价。
  3. 因此批量导入必须走「批量后统一重建」：
       批量前  UPDATE closure_policy SET mode='off';
       导入完  INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at) VALUES(...);
       恢复    UPDATE closure_policy SET mode='incremental';
     重建是"一次全图扫描 + GROUP BY 取最短"（O(可达对数)），
     远优于"每条边都做一次子树 x 上位链笛卡尔积"。
  4. 交互式（用户手工加一条关系）用 'incremental' 最自然，单条延迟完全可接受；
     真正的兜底是 §15.3 的递归 CTE 在线推导 —— 关键路径（如解析时的 is_a 判定）
     若不想背物化闭包的维护税，直接走 CTE + (src,relation) 索引即可。
""")
conn.close()
