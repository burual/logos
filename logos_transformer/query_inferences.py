#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only query tool for ``inferred_relations``.

``inferred_relations`` holds model *hypotheses*; ``closure`` holds ground truth
derived by the triggers.  This tool never writes: it inspects what the inference
pipeline produced so a human can decide whether a hypothesis deserves to be
promoted to a real ``relations`` edge (a separate, deliberate step).

Usage
-----
    python logos_transformer/query_inferences.py --stats
    python logos_transformer/query_inferences.py --source 猫
    python logos_transformer/query_inferences.py --relation is_a --min-confidence 0.9
    python logos_transformer/query_inferences.py --unverified-only --limit 50

Exit codes
----------
    0  ok (including "table exists but empty")
    1  database missing / not a Logos database
    2  invalid arguments
"""

from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys
from typing import List, Optional, Sequence

DEFAULT_DB = pathlib.Path(__file__).resolve().parent.parent / "logos.db"

BASE_SQL = """
SELECT ir.id,
       ir.source_id, w1.word, s1.pos,
       ir.relation_type,
       ir.target_id, w2.word, s2.pos,
       ir.confidence, ir.depth_estimate, ir.verified, ir.model_version
FROM inferred_relations ir
JOIN senses s1 ON ir.source_id = s1.sense_id
JOIN words  w1 ON s1.word_id  = w1.word_id
JOIN senses s2 ON ir.target_id = s2.sense_id
JOIN words  w2 ON s2.word_id  = w2.word_id
"""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _bar(value: float, width: int = 30) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def _fmt_bool(value) -> str:
    return "yes" if value else "no"


def print_stats(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM inferred_relations").fetchone()[0]
    verified = conn.execute(
        "SELECT COUNT(*) FROM inferred_relations WHERE verified = 1"
    ).fetchone()[0]
    print("=" * 92)
    print("  inferred_relations — 汇总统计")
    print("=" * 92)
    print(f"  总行数     : {total:,}")
    print(f"  已人工确认 : {verified:,}  ({verified / total * 100 if total else 0:.1f}%)")
    if total == 0:
        print("\n  [空] 先运行：python logos_transformer/infer_and_write.py --verify")
        return

    avg_conf, min_conf, max_conf = conn.execute(
        "SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM inferred_relations"
    ).fetchone()
    print(f"  置信度     : mean {avg_conf:.4f}  min {min_conf:.4f}  max {max_conf:.4f}")

    print()
    print("  按 relation_type 分布：")
    print(f"    {'relation_type':<18} {'rows':>6} {'avg':>7} {'max':>7} "
          f"{'depth>=2':>9} {'sources':>8}")
    print("    " + "-" * 68)
    for relation, rows, avg, mx, deep, srcs in conn.execute(
        """
        SELECT relation_type, COUNT(*), AVG(confidence), MAX(confidence),
               SUM(CASE WHEN depth_estimate >= 2 THEN 1 ELSE 0 END),
               COUNT(DISTINCT source_id)
        FROM inferred_relations GROUP BY relation_type ORDER BY COUNT(*) DESC
        """
    ):
        print(f"    {relation:<18} {rows:>6} {avg:>7.4f} {mx:>7.4f} {deep:>9} {srcs:>8}")

    print()
    print("  按 model_version 分布：")
    for version, rows in conn.execute(
        "SELECT model_version, COUNT(*) FROM inferred_relations "
        "GROUP BY model_version ORDER BY COUNT(*) DESC"
    ):
        print(f"    {version:<48} {rows:>6}")

    print()
    print("  置信度直方图（0.05 一档；表中只存通过写入阈值的行为，故低档通常为空）：")
    buckets = {
        int(row[0]): int(row[1])
        for row in conn.execute(
            """
            SELECT CAST(MIN(confidence * 20, 19.999) AS INTEGER) AS b, COUNT(*)
            FROM inferred_relations GROUP BY b
            """
        )
    }
    peak = max(buckets.values()) if buckets else 1
    # 只画有数据的区间（外扩一档），否则 20 行里有 17 行是空的。
    lo = max(0, min(buckets) - 1) if buckets else 0
    hi = min(19, max(buckets) + 1) if buckets else 19
    for b in range(lo, hi + 1):
        low = b / 20.0
        high = low + 0.05
        count = buckets.get(b, 0)
        scaled = count / peak
        print(f"    [{low:.2f}, {high:.2f})  {count:>5}  {_bar(scaled, 34)}")

    print()
    print("  top-5 最高置信度：")
    print(f"    {'源':<14} {'关系':<14} {'目标':<16} {'置信度':>7} {'深度':>4} {'已确认':>6}")
    print("    " + "-" * 76)
    for row in conn.execute(
        BASE_SQL + " ORDER BY ir.confidence DESC LIMIT 5"
    ):
        _id, _sid, src, _spos, relation, _tid, target, _tpos, conf, depth, ver, _m = row
        print(f"    {src:<14} {relation:<14} {target:<16} {conf:>7.4f} "
              f"{'-' if depth is None else depth:>4} {_fmt_bool(ver):>6}")


def print_rows(
    conn: sqlite3.Connection,
    where: List[str],
    params: List[object],
    min_confidence: float,
    limit: int,
    header_note: str,
) -> None:
    where = list(where) + ["ir.confidence >= ?"]
    params = list(params) + [min_confidence]
    sql = (
        BASE_SQL
        + " WHERE " + " AND ".join(where)
        + " ORDER BY ir.confidence DESC, ir.source_id, ir.relation_type"
        + f" LIMIT {int(limit)}"
    )
    rows = conn.execute(sql, params).fetchall()

    count_sql = (
        "SELECT COUNT(*) FROM inferred_relations ir"
        " JOIN senses s1 ON ir.source_id = s1.sense_id"
        " JOIN words  w1 ON s1.word_id  = w1.word_id"
        " JOIN senses s2 ON ir.target_id = s2.sense_id"
        " JOIN words  w2 ON s2.word_id  = w2.word_id"
        " WHERE " + " AND ".join(where)
    )
    total = conn.execute(count_sql, params).fetchone()[0]

    print("=" * 92)
    print(f"  {header_note}")
    print("=" * 92)
    print(f"  命中 {total:,} 行，显示前 {min(len(rows), limit)} 行")
    if not rows:
        print("\n  [空] 没有满足条件的推断。")
        return
    print()
    print(f"    {'源':<14} {'关系':<16} {'目标':<16} {'置信度':>7} {'深度':>4} {'已确认':>6}")
    print("    " + "-" * 78)
    for row in rows:
        _id, _sid, src, _spos, relation, _tid, target, _tpos, conf, depth, ver, _m = row
        print(f"    {src:<14} {relation:<16} {target:<16} {conf:>7.4f} "
              f"{'-' if depth is None else depth:>4} {_fmt_bool(ver):>6}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the model hypotheses stored in inferred_relations (read-only)."
    )
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    parser.add_argument("--source", default=None,
                        help="按源词模糊匹配（SQL LIKE %name%）")
    parser.add_argument("--relation", default=None, help="按关系类型精确过滤")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="只显示置信度 >= 该值的行，默认 0.0")
    parser.add_argument("--unverified-only", action="store_true",
                        help="只显示 verified = 0 的行")
    parser.add_argument("--stats", action="store_true",
                        help="打印汇总统计（关系分布 / 置信度直方图 / top-5）")
    parser.add_argument("--limit", type=int, default=50, help="最多显示多少行，默认 50")
    args = parser.parse_args(argv)

    if not (0.0 <= args.min_confidence <= 1.0):
        parser.error("--min-confidence 必须在 [0, 1]")
    if args.limit < 1:
        parser.error("--limit 必须 >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):  # pragma: no cover
                pass

    args = parse_args(argv)
    db_path: pathlib.Path = args.db.expanduser()
    if not db_path.exists():
        print(f"[错误] 找不到数据库：{db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    try:
        if not _table_exists(conn, "senses"):
            print(f"[错误] {db_path} 不是 Logos 数据库（缺 senses 表）", file=sys.stderr)
            return 1

        print(f"数据库 : {db_path}")
        if not _table_exists(conn, "inferred_relations"):
            print()
            print("  [空] inferred_relations 表尚不存在。")
            print("       先运行：python logos_transformer/infer_and_write.py --min-confidence 0.8")
            return 0

        total = conn.execute("SELECT COUNT(*) FROM inferred_relations").fetchone()[0]
        if total == 0 and not args.stats:
            print()
            print("  [空] 表存在但没有任何行。")
            print("       先运行：python logos_transformer/infer_and_write.py --min-confidence 0.8")
            return 0

        if args.stats:
            print()
            print_stats(conn)
            return 0

        where: List[str] = []
        params: List[object] = []
        notes: List[str] = []
        if args.source:
            where.append("w1.word LIKE ?")
            params.append(f"%{args.source}%")
            notes.append(f"源 ≈ {args.source}")
        if args.relation:
            where.append("ir.relation_type = ?")
            params.append(args.relation)
            notes.append(f"关系 = {args.relation}")
        if args.unverified_only:
            where.append("ir.verified = 0")
            notes.append("仅未确认")
        notes.append(f"confidence >= {args.min_confidence:.2f}")

        if not where:  # 没有任何过滤 -> 给全表
            where.append("1 = 1")

        print()
        print_rows(
            conn,
            where=where,
            params=params,
            min_confidence=args.min_confidence,
            limit=args.limit,
            header_note="inferred_relations — " + " / ".join(notes),
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
