#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对任意 checkpoint 重算「118 条金标按 0.8 阈值会写回什么」。

为什么单独做这一步
------------------
``infer_and_write.py --dry-run`` 报的「将写入 N 行」只是**条数**。条数多不代表
错得多、少不代表错得少 —— 实测已经出现「写 95 行的配置质量比写 183 行的差」。
真正要回答的是：**同一个阈值下，写进库的东西里正确率变了吗**。

做法：拿基线写回的那 118 条（已逐条人工判定，见 ``residual_gold.json``）当固定
探针集，对每个 checkpoint 重算 prob。若某条的 prob 掉到 0.8 以下，它就不会被
写回 —— 相当于模型自己把它过滤掉了。于是：

    correct 保留率 = 仍 >0.8 的 correct 条数 / 13
    wrong   保留率 = 仍 >0.8 的错误条数 / 96
    保留集合的错误率 = wrong / (correct + wrong)

correct 保留率高、wrong 保留率低 ⇒ 过滤是对的。两者同升同降 ⇒ 只是整体抬/压
分数，没有变聪明。

用法::

    python logos_transformer/gold_probe.py
    python logos_transformer/gold_probe.py --checkpoint logos_transformer/checkpoints_pretrained/mini_frozen/closure_gat_relaware_best.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
from typing import Dict, List, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import infer_and_write as iw  # noqa: E402
from prototype_gat import load_graph  # noqa: E402
from train_pretrained_init import gold_report, load_gold  # noqa: E402

DEFAULT_DIR = _ROOT / "logos_transformer" / "checkpoints_pretrained"
DEFAULT_GOLD = _ROOT / "residual_gold.json"
THRESHOLD = 0.8

# 目录名 -> checkpoint 文件名（dense 用折叠存档，写回管线只认 table）
CANDIDATES = [
    ("random", "closure_gat_relaware_best.pt", "基线（随机初始化）"),
    ("mini_tuned", "closure_gat_relaware_best.pt", "mini · table · 微调"),
    ("mini_frozen", "closure_gat_relaware_best.pt", "mini · table · 冻结"),
    ("mini_early3", "closure_gat_relaware_best.pt", "mini · table · 只冻 3 epoch"),
    ("mini_dense", "closure_gat_relaware_best_folded.pt", "mini · dense · 微调（折叠）"),
    ("charngram", "closure_gat_relaware_best.pt", "字形哈希 · table · 微调"),
]


def probe_one(
    ckpt: pathlib.Path,
    graph,
    conn: sqlite3.Connection,
    gold: List[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    info = iw.inspect_checkpoint(state)
    model = iw.build_and_load(ckpt, info, graph, seed=42)
    model.eval()
    report = gold_report(model, graph, conn, gold)
    rows = [r for r in report["rows"] if not r["control"]]

    kept = [r for r in rows if float(r["prob"]) >= threshold]
    good = [r for r in kept if r["verdict"] == "correct"]
    bad = [r for r in kept if r["verdict"] in ("wrong", "wrong_relation")]
    total_good = sum(1 for r in rows if r["verdict"] == "correct")
    total_bad = sum(1 for r in rows if r["verdict"] in ("wrong", "wrong_relation"))

    sep = report["writeback"]["separation"]  # type: ignore[index]
    return {
        "n_rows": len(rows),
        "kept": len(kept),
        "kept_good": len(good),
        "kept_bad": len(bad),
        "total_good": total_good,
        "total_bad": total_bad,
        "good_rate": len(good) / total_good if total_good else float("nan"),
        "bad_rate": len(bad) / total_bad if total_bad else float("nan"),
        # 分母只算有明确判定的条（uncertain 两边都不计），与金标口径一致
        "error_share": len(bad) / (len(good) + len(bad))
        if (good or bad) else float("nan"),
        "separation": sep["auc_like"],
        "mean_good": sep["mean_correct"],
        "mean_bad": sep["mean_bad"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="按阈值重算金标探针集的写回质量")
    parser.add_argument("--db", type=pathlib.Path, default=_ROOT / "logos.db")
    parser.add_argument("--gold", type=pathlib.Path, default=DEFAULT_GOLD)
    parser.add_argument("--dir", type=pathlib.Path, default=DEFAULT_DIR)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args(argv)

    if not args.gold.exists():
        print(f"金标不存在：{args.gold}（先跑 make_residual_gold.py）", file=sys.stderr)
        return 1
    gold = load_gold(args.gold)
    graph = load_graph(args.db, include_closure=False, verbose=False)
    conn = sqlite3.connect(str(args.db))
    conn.execute("PRAGMA query_only = ON")

    try:
        if args.checkpoint is not None:
            targets = [("explicit", args.checkpoint, args.checkpoint.parent.name)]
        else:
            targets = []
            for name, fname, label in CANDIDATES:
                path = args.dir / name / fname
                if path.exists():
                    targets.append((name, path, label))
        if not targets:
            print(f"没有找到任何 checkpoint：{args.dir}", file=sys.stderr)
            return 1

        print(f"阈值 {args.threshold:.2f} | 探针集 = residual_gold.json 的 118 条写回\n")
        print("| 配置 | 仍 >阈值 | correct 保留 | wrong 保留 | 保留集合错误率 "
              "| 分离度 |")
        print("|---|---:|---:|---:|---:|---:|")
        payload = {}
        for name, path, label in targets:
            stats = probe_one(path, graph, conn, gold, args.threshold)
            payload[name] = stats
            print("| {} | {}/{} | {}/{} ({:.0%}) | {}/{} ({:.0%}) | {:.1%} | {:.4f} |".format(
                label, stats["kept"], stats["n_rows"],
                stats["kept_good"], stats["total_good"], stats["good_rate"],
                stats["kept_bad"], stats["total_bad"], stats["bad_rate"],
                stats["error_share"], stats["separation"],
            ))
        out = _ROOT / "logos_transformer" / "checkpoints_pretrained" / "gold_probe.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果 json：{out}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
