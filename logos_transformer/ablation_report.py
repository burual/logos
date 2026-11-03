#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 ``checkpoints_pretrained/*/result.json`` 汇总成 markdown 消融表。

数字全部从 json 读，不手抄 —— 文档里的表就是这个脚本的输出。这样重跑实验
之后表格只需要重新生成，不会出现「文档写着 0.3556、json 里是 0.3402」这类
不一致（本项目已经踩过一次：`MIN(depth)` 口径错导致 113/118 满篇改）。

用法::

    python logos_transformer/ablation_report.py
    python logos_transformer/ablation_report.py --dir logos_transformer/checkpoints_pretrained
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DIR = _ROOT / "logos_transformer" / "checkpoints_pretrained"

# 输出顺序（与实验编号一致）；不在表里的目录会被排到末尾
ORDER = ["random", "mini_tuned", "mini_frozen", "mini_early3", "mini_dense",
         "charngram", "mini_freeze10"]

LABELS = {
    "random": "基线（随机初始化）",
    "mini_tuned": "mini · table · 微调",
    "mini_frozen": "mini · table · 冻结",
    "mini_early3": "mini · table · 只冻 3 epoch",
    "mini_dense": "mini · dense · 微调",
    "charngram": "字形哈希 · table · 微调",
    "mini_freeze10": "mini · table · 只冻 10 epoch",
}


def load_rows(root: pathlib.Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        payload["_name"] = path.parent.name
        rows.append(payload)
    rows.sort(key=lambda r: (
        ORDER.index(str(r["_name"])) if str(r["_name"]) in ORDER else 99,
        str(r["_name"]),
    ))
    return rows


def fmt(value, pattern: str = "{:.4f}", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        f = float(value)
    except (TypeError, ValueError):
        return dash
    if f != f:            # nan
        return dash
    return pattern.format(f)


def probe_prob(row: Dict[str, object], src: str, rel: str, tgt: str) -> Optional[float]:
    for probe in row.get("probes") or []:
        if (probe.get("src") == src and probe.get("relation") == rel
                and probe.get("tgt") == tgt):
            prob = probe.get("prob")
            return float(prob) if prob is not None else None
    return None


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="汇总预训练初始化消融结果")
    parser.add_argument("--dir", type=pathlib.Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)

    rows = load_rows(args.dir)
    if not rows:
        print(f"没有找到 result.json：{args.dir}", file=sys.stderr)
        return 1

    # ---- 表 1：排序类指标 ---- #
    print("### 排序类指标\n")
    print("| 配置 | 覆盖率 | test MRR | test hits@10 | hard-test MRR | hard margin "
          "| OOD 拒绝率 | 参数量 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        feats = row.get("features") or {}
        metric = (row.get("metrics") or {}).get("test") or {}
        hard = row.get("hard_test") or {}
        print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            LABELS.get(str(row["_name"]), str(row["_name"])),
            fmt(feats.get("coverage"), "{:.1%}"),
            fmt(metric.get("mrr")),
            fmt(metric.get("hits@10"), "{:.3f}"),
            fmt(hard.get("model_mrr")),
            fmt(hard.get("margin"), "{:+.4f}"),
            fmt(row.get("ood_rejection")),
            f"{int(row.get('params') or 0):,}",
        ))

    # ---- 表 2：写回候选的探针分数 ---- #
    probes = [
        ("鱼/n", "is_a", "无脊椎动物/n", "错·应降"),
        ("猫/n", "is_a", "人/n", "错·任务书预期降"),
        ("猫/n", "is_a", "猫科/n", "对·真边"),
        ("蛇/n", "has_property", "温血/a", "错·事实类"),
    ]
    print("\n### 打分探针（sigmoid 后）\n")
    print("| 配置 | " + " | ".join(
        f"{a.split('/')[0]} {r} {b.split('/')[0]}<br><sub>{note}</sub>"
        for a, r, b, note in probes) + " |")
    print("|---|" + "---:|" * len(probes))
    for row in rows:
        cells = [fmt(probe_prob(row, a, r, b)) for a, r, b, _ in probes]
        print("| {} | {} |".format(
            LABELS.get(str(row["_name"]), str(row["_name"])), " | ".join(cells)))

    # ---- 表 3：金标 ---- #
    print("\n### 118 条写回的金标一致度（写回组内比较）\n")
    print("| 配置 | correct n | correct 均分 | bad 均分 | 分离度 | 对照组均分 |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        gold = row.get("gold") or {}
        wb = (gold.get("writeback") or {})
        sep = wb.get("separation") or {}
        ctl = gold.get("controls") or {}
        ctl_mean = ((ctl.get("by_verdict") or {}).get("correct") or {}).get("mean_prob")
        print("| {} | {} | {} | {} | {} | {} |".format(
            LABELS.get(str(row["_name"]), str(row["_name"])),
            sep.get("n_correct", "—"),
            fmt(sep.get("mean_correct")),
            fmt(sep.get("mean_bad")),
            fmt(sep.get("auc_like")),
            fmt(ctl_mean),
        ))

    # ---- 附：训练元信息 ---- #
    print("\n### 训练元信息\n")
    print("| 配置 | init | arch | epochs run | best epoch | 向量源 |")
    print("|---|---|---|---:|---:|---|")
    for row in rows:
        feats = row.get("features") or {}
        print("| {} | {} | {} | {} | {} | {} |".format(
            LABELS.get(str(row["_name"]), str(row["_name"])),
            row.get("init"), row.get("arch"),
            row.get("epochs_run"), row.get("best_epoch"),
            feats.get("source", "—"),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
