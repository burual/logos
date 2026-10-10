#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冗余索引探测器 —— 对 schema.sql 做静态体检。

原理（SQLite 索引可替代性规则）：
  索引 B(a,b,c) 能被用来服务任何只用 (a)、(a,b)、(a,b,c) 做等值/范围查找的查询，
  因此若 A.cols 是 B.cols 的**真前缀**，则 A 完全冗余，可以安全删除。
  例外：
    · A 是**部分索引**（带 WHERE）而 B 是完整索引 —— 不可删，
      因为部分索引更小，是刻意的优化手段（比 B 快）。
    · B 同样是部分索引但 WHERE 不同 —— A 仍有独立价值，不可删。
    · 唯一索引/主键隐含索引可以"支撑"一个普通索引，但不能反向被支撑，
      因为唯一性语义不同（不过若 A 是普通索引、B 是唯一索引且 A 是 B 的前缀，A 仍冗余）。

用法： python audit_indexes.py [schema文件]   默认 schema_v3.sql
"""
import sqlite3
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "schema_v3.sql")


def where_clause(sql):
    """从 CREATE INDEX 语句里抠出 WHERE 子句（部分索引判定），失败返回 ''"""
    if not sql:
        return ""
    m = re.search(r"\)\s*WHERE\s+(.*)$", sql, re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def main():
    conn = sqlite3.connect(":memory:")
    conn.executescript(open(SCHEMA, encoding="utf-8").read())

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' ORDER BY name"
    )]

    redundant, keep_partial, total_explicit = [], [], 0

    for t in tables:
        idxlist = conn.execute(f"PRAGMA index_list('{t}')").fetchall()
        # (seq, name, unique, origin, partial)
        items = []
        for _seq, name, uniq, origin, partial in idxlist:
            cols = [r[2] for r in conn.execute(f"PRAGMA index_info('{name}')")]
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (name,)
            ).fetchone()
            ddl = row[0] if row else None
            if name.startswith("sqlite_autoindex") or origin in ("pk", "u"):
                continue  # 约束自带索引，不是"我们建的"
            total_explicit += 1
            items.append(dict(name=name, cols=cols, uniq=uniq, origin=origin,
                              partial=partial, ddl=ddl, where=where_clause(ddl)))

        # 把约束隐含索引也纳入"可替代者"集合
        all_idx = []
        for name, uniq, origin, partial in ((r[1], r[2], r[3], r[4]) for r in idxlist):
            cols = [r[2] for r in conn.execute(f"PRAGMA index_info('{name}')")]
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (name,)
            ).fetchone()
            ddl = row[0] if row else None
            all_idx.append(dict(name=name, cols=cols, uniq=uniq, origin=origin,
                                partial=partial, where=where_clause(ddl)))

        for a in items:
            for b in all_idx:
                if a["name"] == b["name"]:
                    continue
                if a["cols"] != b["cols"][: len(a["cols"])]:
                    continue                      # a 不是 b 的前缀
                if len(a["cols"]) > len(b["cols"]):
                    continue                      # 真前缀才算被覆盖
                if a["cols"] == b["cols"] and b["origin"] in ("pk", "u"):
                    pass                              # 约束索引可覆盖同名等价普通索引
                # 部分索引的特殊处理
                if a["partial"] and not b["partial"]:
                    keep_partial.append((t, a, b))
                    continue
                if a["partial"] and b["partial"] and a["where"] != b["where"]:
                    keep_partial.append((t, a, b))
                    continue
                redundant.append((t, a, b))
                break

    print(f"schema = {os.path.basename(SCHEMA)}")
    print(f"用户显式创建的索引总数 = {total_explicit}")
    print("=" * 78)
    if redundant:
        print("【冗余索引 —— 建议删除】")
        for t, a, b in redundant:
            print(f"  ✗ {a['name']}")
            print(f"      表 {t}  列 ({', '.join(a['cols'])})")
            print(f"      被 {b['name']} ({', '.join(b['cols'])}) "
                  f"[origin={b['origin']}] 完全覆盖 → 纯浪费写入成本与空间")
    else:
        print("【冗余索引】无")
    print()
    if keep_partial:
        print("【看似重复但应保留 —— 部分索引更小，是刻意优化】")
        for t, a, b in keep_partial:
            print(f"  · {a['name']} (WHERE {a['where'] or '—'})"
                  f"  vs  {b['name']} ({', '.join(b['cols'])})  [表 {t}]")
    else:
        print("【应保留的近似索引】无")
    print("=" * 78)
    print(f"结论: 可删 {len(redundant)} 个，保留 {len(keep_partial)} 个近似索引")
    return len(redundant)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 0)
