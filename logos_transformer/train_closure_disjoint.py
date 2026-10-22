#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-disjoint closure-prediction experiment (honesty check).

Motivation
----------
``train_closure.py`` splits *pairs* randomly, so the same source sense can be
supervised in train and evaluated in test.  With transductive node embeddings
this inflates test AUC.  This script keeps the same model and task but splits
**sources** (sense ids), so every closure source appears in exactly one of
train / val / test.  Negatives inherit their positive's source, therefore no
negative pair can leak across splits.

Evaluation reports train-source, val-source and test-source metrics side by
side: train is a memorisation sanity check, test is the honest number.

Usage
-----
    python logos_transformer/train_closure_disjoint.py --epochs 5   # smoke
    python logos_transformer/train_closure_disjoint.py              # full run

Outputs
-------
    <output-dir>/closure_gat_disjoint_best.pt
    <output-dir>/training_report.txt

Exit codes
----------
    1  missing torch / database / closure data
    2  invalid command-line arguments
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment dependent
    print("Requires torch. Install with: pip install torch", file=sys.stderr)
    raise SystemExit(1)

# Reuse the encoder, data containers and metrics from the pair-split experiment.
# 直接运行本脚本时 sys.path[0] 是 logos_transformer/；作为包导入时走 fallback。
try:
    from prototype_gat import RelationGraph, build_model, load_graph  # type: ignore
    from train_closure import (  # type: ignore
        ClosurePredictor,
        Facts,
        Sample,
        _load_pair_sets,
        attention_stats,
        closure_target_ranks,
        evaluate,
        filtered_metrics,
        labels_tensor,
        pairs_tensor,
        qualitative_examples,
        roc_auc,
        _fmt,
    )
except ImportError:  # pragma: no cover
    from logos_transformer.prototype_gat import RelationGraph, build_model, load_graph  # type: ignore
    from logos_transformer.train_closure import (  # type: ignore
        ClosurePredictor,
        Facts,
        Sample,
        _load_pair_sets,
        attention_stats,
        closure_target_ranks,
        evaluate,
        filtered_metrics,
        labels_tensor,
        pairs_tensor,
        qualitative_examples,
        roc_auc,
        _fmt,
    )

DISJOINT_VERSION = "logos-gat-closure-disjoint-v0.1"
DEFAULT_OUTPUT_DIR = "./logos_transformer/checkpoints_disjoint"
SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_positives(conn: sqlite3.Connection, graph: RelationGraph) -> Tuple[List[Sample], Facts]:
    """Load depth>=2 closure positives plus the pair sets used for filtering."""
    index = graph.id_to_index
    direct, blocked, closure_map = _load_pair_sets(conn, graph)
    rows = conn.execute(
        "SELECT relation, descendant, ancestor, depth FROM closure "
        "WHERE depth >= 2 ORDER BY relation, depth, descendant, ancestor"
    ).fetchall()
    if not rows:
        print("No closure data found, run apply_seeds.py first", file=sys.stderr)
        raise SystemExit(1)

    positives: List[Sample] = []
    for relation, descendant, ancestor, depth in rows:
        s = index.get(int(descendant))
        t = index.get(int(ancestor))
        if s is None or t is None:
            continue
        positives.append(
            Sample(
                src=s, dst=t,
                src_sid=int(descendant), dst_sid=int(ancestor),
                relation=str(relation), depth=int(depth), label=1,
            )
        )
    if not positives:
        print("No closure data found, run apply_seeds.py first", file=sys.stderr)
        raise SystemExit(1)
    return positives, Facts(direct, blocked, closure_map)


def split_by_source(
    positives: Sequence[Sample],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[Dict[str, List[Sample]], Dict[str, Set[int]], str]:
    """Split sources (sense ids) into train/val/test; edges follow their source.

    Falls back to a random sample-level split when there are fewer than three
    unique sources (the source-disjoint premise is meaningless then).

    Returns:
        (positives_by_split, sources_by_split, mode)
    """
    by_source: Dict[int, List[Sample]] = defaultdict(list)
    for sample in positives:
        by_source[sample.src].append(sample)
    sources = sorted(by_source)
    n_sources = len(sources)

    if n_sources < 3:
        print(
            f"[warning] only {n_sources} unique source(s); falling back to a "
            "random sample-level split (source-disjoint split needs >= 3).",
            file=sys.stderr,
        )
        rng = random.Random(seed)
        order = list(positives)
        rng.shuffle(order)
        n = len(order)
        n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
        n_test = max(1, int(round(n * ratios[2]))) if n >= 3 else 0
        n_train = max(1, n - n_val - n_test)
        buckets = {
            "train": order[:n_train],
            "val": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:],
        }
        sets = {
            name: {sample.src for sample in items}
            for name, items in buckets.items()
        }
        return buckets, sets, "random-fallback"

    rng = random.Random(seed)
    rng.shuffle(sources)
    n_val = max(1, int(round(n_sources * ratios[1])))
    n_test = max(1, int(round(n_sources * ratios[2])))
    n_train = n_sources - n_val - n_test
    if n_train < 1:                      # 防御：极小图
        n_train, n_val, n_test = 1, 1, n_sources - 2

    train_src = set(sources[:n_train])
    val_src = set(sources[n_train:n_train + n_val])
    test_src = set(sources[n_train + n_val:])

    buckets = {
        "train": [s for src in train_src for s in by_source[src]],
        "val": [s for src in val_src for s in by_source[src]],
        "test": [s for src in test_src for s in by_source[src]],
    }
    sets = {"train": train_src, "val": val_src, "test": test_src}
    return buckets, sets, "source-disjoint"


def sample_negatives(
    positives: Sequence[Sample],
    graph: RelationGraph,
    facts: Facts,
    neg_ratio: int,
    rng: random.Random,
    seen: Set[Tuple[int, int]],
) -> List[Sample]:
    """Sample closure-free negatives for the given positives only.

    Negative sources are exactly the positives' sources, so a negative can
    never belong to another split.  Targets may come from any node, but a pair
    is rejected if it is a direct edge or closure fact in either direction.
    """
    num_nodes = len(graph.node_ids)
    all_nodes = list(range(num_nodes))
    forbidden_by_src: Dict[int, Set[int]] = defaultdict(set)
    for src, dst in facts.blocked_pairs:
        forbidden_by_src[src].add(dst)

    negatives: List[Sample] = []
    for pos in positives:
        forbidden = {pos.src} | forbidden_by_src[pos.src]
        candidates = [j for j in all_nodes if j not in forbidden]
        if not candidates:
            continue
        for dst in rng.sample(candidates, min(neg_ratio, len(candidates))):
            key = (pos.src, dst)
            if key in seen:
                continue
            seen.add(key)
            negatives.append(
                Sample(
                    src=pos.src, dst=dst,
                    src_sid=pos.src_sid, dst_sid=graph.node_ids[dst],
                    relation=pos.relation, depth=0, label=0,
                )
            )
    return negatives


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_bucket(
    model: ClosurePredictor,
    graph: RelationGraph,
    facts: Facts,
    samples: Sequence[Sample],
    criterion: nn.Module,
) -> Dict[str, object]:
    """Loss/accuracy/AUC on sampled pairs + filtered ranking on positives."""
    positives = [s for s in samples if s.label == 1]
    loss, acc, logits = evaluate(model, graph, samples, criterion)
    labels = [s.label for s in samples]
    auc = roc_auc(labels, logits)
    ranking = filtered_metrics(model, graph, facts, positives)
    return {
        "n_pos": len(positives),
        "n_neg": len(samples) - len(positives),
        "loss": loss,
        "acc": acc,
        "auc": auc,
        "p@10": ranking["p@10"],
        "p@20": ranking["p@20"],
        "mrr": ranking["mrr"],
        "mrr_per_query": ranking["mrr_per_query"],
        "hits@1": ranking["hits@1"],
        "hits@3": ranking["hits@3"],
        "hits@10": ranking["hits@10"],
        "queries": ranking["queries"],
        "by_depth": ranking["by_depth"],
    }


def target_prior_mrr(
    model: ClosurePredictor,
    graph: RelationGraph,
    facts: Facts,
    positives: Sequence[Sample],
) -> float:
    """MRR of a source-independent baseline: rank targets by mean score.

    If this baseline is close to the per-source MRR, the model's ranking is
    driven mainly by a global "target looks like an ancestor" prior rather than
    source-conditioned relational structure.
    """
    model.eval()
    with torch.no_grad():
        h, _ = model.encode(graph.edge_index, graph.edge_type)
        score_matrix = model.score_all(h)
    bias = score_matrix.mean(dim=0)          # (N,) target-only score
    num_nodes = len(graph.node_ids)

    keep_by_query: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    for sample in positives:
        keep_by_query[(sample.src, sample.relation)].add(sample.dst)

    rr: List[float] = []
    for sample in positives:
        keep = keep_by_query[(sample.src, sample.relation)]
        row = bias.clone()
        row[sample.src] = -1e9
        for j in range(num_nodes):
            if j in keep:
                continue
            if (sample.src, j) in facts.blocked_pairs:
                row[j] = -1e9
        order = torch.argsort(row, descending=True)
        rank = int((order == sample.dst).nonzero(as_tuple=True)[0].item()) + 1
        rr.append(1.0 / rank)
    return sum(rr) / len(rr) if rr else float("nan")


def build_report(
    *,
    args: argparse.Namespace,
    graph: RelationGraph,
    total_sources: int,
    mode: str,
    sources_by_split: Dict[str, Set[int]],
    data: Dict[str, Dict[str, object]],
    best_epoch: int,
    epochs_run: int,
    stopped_early: bool,
    metrics: Dict[str, Dict[str, object]],
    prior_mrr: Dict[str, float],
    init_attention: Dict[str, object],
    final_attention: Dict[str, object],
    cat_split: str,
    qualitative: Sequence[Dict[str, object]],
    target_ranks: Sequence[Tuple[str, str, int, int]],
    checkpoint: pathlib.Path,
) -> str:
    def depth_counter(items: Sequence[Sample]) -> Dict[int, int]:
        return dict(sorted(Counter(s.depth for s in items).items()))

    train = data["train"]
    train_pos = len([s for s in train["pos"]])                 # type: ignore[arg-type]
    train_neg = len(train["samples"]) - train_pos               # type: ignore[arg-type]
    pos_weight = train_neg / max(train_pos, 1)

    lines: List[str] = []
    lines.append("=" * 86)
    lines.append("Logos Transformer — Source-Disjoint Closure Training Report")
    lines.append("=" * 86)
    lines.append(f"generated_at        : {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"model_version       : {DISJOINT_VERSION}")
    lines.append(f"database            : {args.db}")
    lines.append(f"checkpoint          : {checkpoint}")
    lines.append(f"split_mode          : {mode}")
    lines.append("")
    lines.append("--- Graph ---")
    lines.append(f"nodes               : {graph.stats['nodes']} senses")
    lines.append(f"direct semantic edges: {graph.stats['semantic_edges']}")
    lines.append(f"message edges       : {graph.stats['message_edges']} (bidirectional + self-loops)")
    lines.append(f"closure rows        : {graph.stats['closure_rows']} "
                 f"(depth=1: {graph.stats['closure_depth1']}, depth>=2: {graph.stats['closure_deep']})")
    lines.append("")
    lines.append("--- Source-disjoint split ---")
    lines.append(f"unique sources      : {total_sources}")
    for name in SPLIT_NAMES:
        srcs = sources_by_split[name]
        items = data[name]
        pos_list: List[Sample] = items["pos"]                  # type: ignore[assignment]
        lines.append(
            f"{name:<6} sources={len(srcs):>3}  edges={len(pos_list):>3}  "
            f"depth={depth_counter(pos_list)}"
        )
    lines.append(f"test source ids     : {sorted(graph.node_ids[i] for i in sources_by_split['test'])}")
    lines.append(f"val source ids      : {sorted(graph.node_ids[i] for i in sources_by_split['val'])}")
    lines.append("")
    lines.append("--- Dataset per split (positives + closure-free negatives) ---")
    for name in SPLIT_NAMES:
        items = data[name]
        samples: List[Sample] = items["samples"]               # type: ignore[assignment]
        n_pos = len(items["pos"])                              # type: ignore[arg-type]
        lines.append(
            f"{name:<6} total={len(samples):>3}  pos={n_pos:>3}  "
            f"neg={len(samples) - n_pos:>3}"
        )
    lines.append(f"train class balance : {train_pos} pos / {train_neg} neg "
                 f"(pos_weight={pos_weight:.2f})")
    lines.append("")
    lines.append("--- Hyperparameters ---")
    lines.append(f"hidden_dim          : {args.hidden}")
    lines.append(f"num_heads           : {args.heads}")
    lines.append(f"num_layers          : {args.layers}")
    lines.append(f"batch_size          : {args.batch_size}")
    lines.append(f"lr                  : {args.lr}")
    lines.append(f"max_epochs          : {args.epochs}")
    lines.append(f"patience            : {args.patience}")
    lines.append(f"seed                : {args.seed}")
    lines.append("optimizer           : Adam, ReduceLROnPlateau(factor=0.5, patience=10)")
    lines.append(f"loss                : BCEWithLogitsLoss(pos_weight={pos_weight:.2f})")
    lines.append("")
    lines.append("--- Training ---")
    lines.append(f"epochs_run          : {epochs_run}")
    lines.append(f"best_epoch          : {best_epoch}")
    lines.append(f"early_stopped       : {stopped_early}")
    lines.append("")
    lines.append("--- Side-by-side evaluation (sampled pairs + filtered full-catalogue ranking) ---")
    header = (
        f"{'split':<6} {'sources':>7} {'pos':>4} {'neg':>4} {'AUC':>7} {'MRR':>7} "
        f"{'H@1':>7} {'H@3':>7} {'H@10':>7} {'P@10':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name in SPLIT_NAMES:
        m = metrics[name]
        lines.append(
            f"{name:<6} {len(sources_by_split[name]):>7} {m['n_pos']:>4} {m['n_neg']:>4} "
            f"{_fmt(m['auc']):>7} {_fmt(m['mrr']):>7} {_fmt(m['hits@1']):>7} "
            f"{_fmt(m['hits@3']):>7} {_fmt(m['hits@10']):>7} {_fmt(m['p@10']):>7}"
        )
    lines.append("")
    lines.append("Target-prior baseline (source-independent mean score, same filtering):")
    lines.append(f"  train MRR={_fmt(prior_mrr['train'])}  val MRR={_fmt(prior_mrr['val'])}  "
                 f"test MRR={_fmt(prior_mrr['test'])}")
    lines.append("  If this is close to the per-source MRR above, ranking is driven mainly")
    lines.append("  by a global target prior, not by source-conditioned relational structure.")
    lines.append("")
    lines.append("Test-source metrics by closure depth:")
    for depth, dm in metrics["test"]["by_depth"].items():       # type: ignore[union-attr]
        lines.append(
            f"  depth={depth} (n={dm['n']:>2})  MRR={_fmt(dm['mrr'])}  "
            f"Hits@1={_fmt(dm['hits@1'])}  Hits@10={_fmt(dm['hits@10'])}"
        )
    lines.append("")
    lines.append("--- Attention shift (layer-1, head-average) ---")
    lines.append(f"mean normalised entropy (init)    : {_fmt(init_attention['mean_entropy'])}")
    lines.append(f"mean normalised entropy (trained) : {_fmt(final_attention['mean_entropy'])}")
    lines.append("  (1.000 = uniform, lower = more selective)")
    lines.append(f"attention @ 猫 (init):")
    for alpha, name, relation in init_attention["target_top"]:
        lines.append(f"  {name:<16} {relation:<16} {alpha:.4f}")
    lines.append(f"attention @ 猫 (trained):")
    for alpha, name, relation in final_attention["target_top"]:
        lines.append(f"  {name:<16} {relation:<16} {alpha:.4f}")
    lines.append("")
    lines.append(f"--- Qualitative: top-5 targets for「猫」(sense 2, split={cat_split}) ---")
    for i, item in enumerate(qualitative, 1):
        extra = ""
        if item["closure"]:
            rel, depth = item["closure"]
            extra = f"  [closure: {rel}, depth {depth}]"
        lines.append(f"  {i}. {item['label']}({item['sid']})  score={item['score']:.4f}{extra}")
    if target_ranks:
        lines.append("  Known closure targets and their ranks:")
        for label, relation, depth, rank in target_ranks:
            lines.append(f"    {label:<14} {relation} depth={depth}  rank=#{rank}")
    lines.append("")
    lines.append("--- Caveats ---")
    lines.append("* Source-disjoint means no source's closure edges are supervised in two")
    lines.append("  splits. It does NOT make the model inductive: test-source nodes still")
    lines.append("  appear in the message graph and their node embeddings are updated")
    lines.append("  during training (semi-transductive).")
    lines.append("* Test targets may have been seen as targets/context in train; the")
    lines.append("  measured generalisation is to unseen *query sources*, not unseen nodes.")
    lines.append("* Val sources are used for early stopping, so val is not an unbiased")
    lines.append(f"  test set. The test split has {len(sources_by_split['test'])} sources; with so few")
    lines.append("  queries the confidence interval is wide.")
    lines.append("* The head is relation-agnostic and has_part has a single positive.")
    lines.append("* Train MRR is not ~1.0: training uses 3 sampled negatives per positive,")
    lines.append("  while MRR ranks all 170 nodes; unsampled ancestor-like confounders can")
    lines.append("  outrank the true target even for trained sources.")
    lines.append("* AUC uses random closure-free negatives and saturates at 1.0; the")
    lines.append("  filtered full-catalogue MRR/Hits are the discriminative metrics here.")
    lines.append("* Expand the animal library / add sources before reading small metric")
    lines.append("  differences as architectural signals.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train/evaluate the Logos GAT with a source-disjoint split."
    )
    parser.add_argument("--db", type=pathlib.Path, default=root / "logos.db")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs 必须 >= 1")
    if args.hidden < 1 or args.heads < 1 or args.hidden % args.heads != 0:
        parser.error("--hidden 必须能被 --heads 整除")
    if args.layers < 1:
        parser.error("--layers 必须 >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size 必须 >= 1")
    if args.neg_ratio < 1:
        parser.error("--neg-ratio 必须 >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    args = parse_args(argv)
    db_path = pathlib.Path(args.db).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()

    graph = load_graph(db_path, include_closure=False, verbose=False)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    try:
        has_closure = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='closure'"
        ).fetchone()
        if not has_closure:
            print("No closure data found, run apply_seeds.py first", file=sys.stderr)
            return 1
        closure_total = conn.execute("SELECT COUNT(*) FROM closure").fetchone()[0]
        deep_total = conn.execute("SELECT COUNT(*) FROM closure WHERE depth >= 2").fetchone()[0]
        if closure_total == 0 or deep_total == 0:
            print("No closure data found, run apply_seeds.py first", file=sys.stderr)
            return 1
        positives, facts = load_positives(conn, graph)
    finally:
        conn.close()

    total_sources = len({s.src for s in positives})
    if total_sources < 10:
        print(
            f"[warning] only {total_sources} unique source senses; a source-disjoint "
            "split is statistically unreliable. Expand the animal library before "
            "drawing conclusions.",
            file=sys.stderr,
        )

    positives_by_split, sources_by_split, mode = split_by_source(positives, seed=args.seed)

    # Negatives are sampled per split with a shared RNG and dedupe set.  In the
    # normal path sources are disjoint, so no pair can cross splits.
    rng = random.Random(args.seed + 7)
    seen_negatives: Set[Tuple[int, int]] = set()
    data: Dict[str, Dict[str, object]] = {}
    for name in SPLIT_NAMES:
        pos_list = positives_by_split[name]
        neg_list = sample_negatives(
            pos_list, graph, facts, args.neg_ratio, rng, seen_negatives
        )
        data[name] = {
            "pos": pos_list,
            "neg": neg_list,
            "samples": list(pos_list) + neg_list,
        }

    train_samples: List[Sample] = data["train"]["samples"]        # type: ignore[assignment]
    val_samples: List[Sample] = data["val"]["samples"]            # type: ignore[assignment]
    train_pos = sum(1 for s in train_samples if s.label == 1)
    train_neg = len(train_samples) - train_pos
    pos_weight = train_neg / max(train_pos, 1)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    base_model = build_model(
        graph,
        hidden_dim=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=0.1,
        seed=args.seed,
    )
    model = ClosurePredictor(base_model, args.hidden)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    print("=" * 86)
    print("Logos Transformer — source-disjoint closure prediction")
    print("=" * 86)
    print(f"db={db_path}")
    print(f"graph: {graph.stats['nodes']} nodes / {graph.stats['semantic_edges']} direct edges"
          f" / {graph.stats['message_edges']} message edges")
    print(f"split mode: {mode} | unique sources: {total_sources}")
    for name in SPLIT_NAMES:
        pos_list: List[Sample] = data[name]["pos"]                # type: ignore[assignment]
        samples: List[Sample] = data[name]["samples"]             # type: ignore[assignment]
        depths = dict(sorted(Counter(s.depth for s in pos_list).items()))
        print(
            f"  {name:<5} sources={len(sources_by_split[name]):>3}  edges={len(pos_list):>3}"
            f"  depth={depths}  samples={len(samples):>3} "
            f"({len(pos_list)} pos / {len(samples) - len(pos_list)} neg)"
        )
    print(f"train pos_weight={pos_weight:.2f} | "
          f"model={args.layers}x{args.heads} heads, hidden={args.hidden}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print("-" * 86)

    cat_idx = graph.id_to_index.get(2, 0)
    cat_split = next(
        (name for name in SPLIT_NAMES if cat_idx in sources_by_split[name]), "—"
    )
    init_attention = attention_stats(model, graph, cat_idx)
    init_loss, init_acc, _ = evaluate(model, graph, train_samples, criterion)
    print(f"init: train loss={init_loss:.4f} acc={init_acc:.4f} | "
          f"attention entropy={init_attention['mean_entropy']:.4f}")

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epochs_run = 0
    batch_rng = random.Random(args.seed + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "closure_gat_disjoint_best.pt"

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        model.train()
        order = list(train_samples)
        batch_rng.shuffle(order)
        total_loss = 0.0
        total = 0
        correct = 0
        for i in range(0, len(order), args.batch_size):
            batch = order[i:i + args.batch_size]
            optimizer.zero_grad()
            embeddings, _ = model.encode(graph.edge_index, graph.edge_type)
            logits = model.score_pairs(embeddings, pairs_tensor(batch))
            labels = labels_tensor(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total += len(batch)
            correct += int(((logits > 0).float() == labels).detach().sum())

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        if val_samples:
            val_loss, val_acc, _ = evaluate(model, graph, val_samples, criterion)
        else:
            val_loss, val_acc = train_loss, train_acc
        if math.isnan(val_loss):
            val_loss = train_loss
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0 or args.epochs < 10:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:>3}/{args.epochs} | train loss={train_loss:.4f} "
                  f"acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} | "
                  f"lr={lr:.2e}")

        if stale >= args.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, "
                  f"val loss {best_val:.4f})")
            break

    stopped_early = stale >= args.patience and epochs_run < args.epochs
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    metrics: Dict[str, Dict[str, object]] = {}
    for name in SPLIT_NAMES:
        metrics[name] = eval_bucket(
            model, graph, facts, data[name]["samples"], criterion   # type: ignore[arg-type]
        )
    prior_mrr = {
        name: target_prior_mrr(model, graph, facts, data[name]["pos"])  # type: ignore[arg-type]
        for name in SPLIT_NAMES
    }

    final_attention = attention_stats(model, graph, cat_idx)
    qualitative = qualitative_examples(model, graph, facts, source_sid=2, top_k=5)
    target_ranks = closure_target_ranks(model, graph, facts, source_sid=2)

    header = (
        f"{'split':<6} {'sources':>7} {'pos':>4} {'neg':>4} {'AUC':>7} {'MRR':>7} "
        f"{'H@1':>7} {'H@3':>7} {'H@10':>7} {'P@10':>7}"
    )
    print("-" * 86)
    print(header)
    print("-" * len(header))
    for name in SPLIT_NAMES:
        m = metrics[name]
        print(
            f"{name:<6} {len(sources_by_split[name]):>7} {m['n_pos']:>4} {m['n_neg']:>4} "
            f"{_fmt(m['auc']):>7} {_fmt(m['mrr']):>7} {_fmt(m['hits@1']):>7} "
            f"{_fmt(m['hits@3']):>7} {_fmt(m['hits@10']):>7} {_fmt(m['p@10']):>7}"
        )
    print("-" * 86)
    print(f"target-prior MRR (source-independent): train={_fmt(prior_mrr['train'])} "
          f"val={_fmt(prior_mrr['val'])} test={_fmt(prior_mrr['test'])}")
    print(f"attention entropy: init={init_attention['mean_entropy']:.4f} "
          f"-> trained={final_attention['mean_entropy']:.4f} | 猫 split={cat_split}")

    report = build_report(
        args=args,
        graph=graph,
        total_sources=total_sources,
        mode=mode,
        sources_by_split=sources_by_split,
        data=data,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
        metrics=metrics,
        prior_mrr=prior_mrr,
        init_attention=init_attention,
        final_attention=final_attention,
        cat_split=cat_split,
        qualitative=qualitative,
        target_ranks=target_ranks,
        checkpoint=checkpoint_path,
    )
    report_path = output_dir / "training_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"checkpoint : {checkpoint_path}")
    print(f"report     : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
