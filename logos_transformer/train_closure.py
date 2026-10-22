#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the Logos Transformer GAT to predict multi-hop closure edges.

Task
----
The GAT only sees **direct** semantic edges (1-hop).  The supervision signal is
the materialised transitive closure at ``depth >= 2``: given a source sense,
can the model rank its indirect closure targets above unrelated senses?

The prediction head is the bilinear scorer from the design doc:

    score(s, t) = h_s^T W h_t          (W learnable, no bias)

trained with BCEWithLogitsLoss on positive/negative pairs.  This is a
*transductive* experiment: node embeddings are learned per sense, so the
checkpoint is tied to the node ordering of the database it was trained on.

Usage
-----
    python logos_transformer/train_closure.py --epochs 5       # smoke test
    python logos_transformer/train_closure.py                  # full run

Outputs
-------
    <output-dir>/closure_gat_best.pt    best state_dict (val loss)
    <output-dir>/training_report.txt    metrics, attention shift, caveats

Exit codes
----------
    1  missing torch / database / closure data
    2  invalid command-line arguments
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import random
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

# torch 会在 import 时尝试初始化 numpy；本实验不依赖 numpy。
warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment dependent
    print("Requires torch. Install with: pip install torch", file=sys.stderr)
    raise SystemExit(1)

# 复用 prototype 的图导出与 GAT，**不修改** prototype_gat.py。
# 直接运行本脚本时 sys.path[0] 是 logos_transformer/，可裸名导入；
# 作为包导入时走 fallback。
try:
    from prototype_gat import (  # type: ignore
        LogosGAT,
        RelationGraph,
        build_model,
        load_graph,
        MODEL_VERSION as PROTOTYPE_VERSION,
    )
except ImportError:  # pragma: no cover
    from logos_transformer.prototype_gat import (  # type: ignore
        LogosGAT,
        RelationGraph,
        build_model,
        load_graph,
        MODEL_VERSION as PROTOTYPE_VERSION,
    )

TRAIN_VERSION = "logos-gat-closure-v0.1"
DEFAULT_OUTPUT_DIR = "./logos_transformer/checkpoints"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Sample:
    """One labelled (source, target) pair."""

    src: int            # node index
    dst: int            # node index
    src_sid: int        # original sense_id (reporting)
    dst_sid: int
    relation: str
    depth: int          # 0 for sampled negatives
    label: int          # 1 positive, 0 negative


@dataclasses.dataclass
class Facts:
    """Pair sets used for negative sampling and filtered ranking."""

    direct_pairs: Set[Tuple[int, int]]     # both directions, node indices
    blocked_pairs: Set[Tuple[int, int]]    # direct + closure, both directions
    closure_map: Dict[Tuple[int, int], Tuple[str, int]]  # (src_sid, dst_sid) -> (relation, depth)


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _load_pair_sets(conn: sqlite3.Connection, graph: RelationGraph) -> Tuple[Set, Set, Dict]:
    """Build direct-edge and closure pair sets (both directions, node indices)."""
    index = graph.id_to_index
    direct: Set[Tuple[int, int]] = set()
    for src_sid, dst_sid in conn.execute(
        "SELECT src_id, dst_id FROM relations "
        "WHERE src_kind = 'sense' AND dst_kind = 'sense' AND is_negated = 0"
    ):
        a = index.get(int(src_sid))
        b = index.get(int(dst_sid))
        if a is None or b is None:
            continue
        direct.add((a, b))
        direct.add((b, a))

    blocked = set(direct)
    closure_map: Dict[Tuple[int, int], Tuple[str, int]] = {}
    for relation, descendant, ancestor, depth in conn.execute(
        "SELECT relation, descendant, ancestor, depth FROM closure"
    ):
        a = index.get(int(descendant))
        b = index.get(int(ancestor))
        if a is None or b is None:
            continue
        blocked.add((a, b))
        blocked.add((b, a))
        if int(depth) >= 2:
            closure_map[(int(descendant), int(ancestor))] = (str(relation), int(depth))
    return direct, blocked, closure_map


def load_samples(
    conn: sqlite3.Connection,
    graph: RelationGraph,
    neg_ratio: int = 3,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], Facts, Dict[str, object]]:
    """Load closure positives and sample negatives not connected in closure.

    Positives are closure rows with depth >= 2, read as
    ``descendant --relation--> ancestor`` (see design doc §1.3).

    Negatives are uniformly sampled targets that are neither direct neighbours
    nor part of any closure path to/from the source (in either direction).
    """
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

    rng = random.Random(seed)
    num_nodes = len(graph.node_ids)
    all_nodes = list(range(num_nodes))
    negatives: List[Sample] = []
    seen_negative_pairs: Set[Tuple[int, int]] = set()

    for pos in positives:
        forbidden = {pos.src}
        forbidden.update(t for (s, t) in blocked if s == pos.src)
        candidates = [j for j in all_nodes if j not in forbidden]
        if not candidates:
            continue
        take = min(neg_ratio, len(candidates))
        for dst in rng.sample(candidates, take):
            key = (pos.src, dst)
            if key in seen_negative_pairs:
                continue
            seen_negative_pairs.add(key)
            negatives.append(
                Sample(
                    src=pos.src, dst=dst,
                    src_sid=pos.src_sid, dst_sid=graph.node_ids[dst],
                    # 负样本继承正样本的关系类型，便于分层划分与分组指标。
                    relation=pos.relation, depth=0, label=0,
                )
            )

    stats: Dict[str, object] = {
        "positives": len(positives),
        "negatives": len(negatives),
        "total": len(positives) + len(negatives),
        "depth_counts": dict(sorted(Counter(p.depth for p in positives).items())),
        "positive_relations": dict(Counter(p.relation for p in positives)),
        "direct_pairs": len(direct),
        "blocked_pairs": len(blocked),
    }
    return positives, negatives, Facts(direct, blocked, closure_map), stats


def stratified_split(
    positives: Sequence[Sample],
    negatives: Sequence[Sample],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split positives and negatives separately, stratified by relation type.

    Tiny strata are handled defensively: a single sample stays in train, and
    strata with 3+ samples always contribute at least one val and one test item.
    """
    rng = random.Random(seed)

    def split_group(items: Sequence[Sample]) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        buckets: Dict[str, List[Sample]] = defaultdict(list)
        for item in items:
            buckets[item.relation].append(item)

        train: List[Sample] = []
        val: List[Sample] = []
        test: List[Sample] = []
        for relation in sorted(buckets):
            group = list(buckets[relation])
            rng.shuffle(group)
            n = len(group)
            if n == 1:
                train.extend(group)
                continue
            if n == 2:
                train.extend(group)
                continue
            n_val = max(1, int(round(n * ratios[1])))
            n_test = max(1, int(round(n * ratios[2])))
            if n - n_val - n_test < 1:
                n_val, n_test = 1, 1
            n_train = n - n_val - n_test
            train.extend(group[:n_train])
            val.extend(group[n_train:n_train + n_val])
            test.extend(group[n_train + n_val:])
        return train, val, test

    tr_p, va_p, te_p = split_group(positives)
    tr_n, va_n, te_n = split_group(negatives)
    return tr_p + tr_n, va_p + va_n, te_p + te_n


# --------------------------------------------------------------------------- #
# Model: GAT encoder + bilinear closure head
# --------------------------------------------------------------------------- #
class ClosurePredictor(nn.Module):
    """GAT encoder plus the bilinear head ``h_s^T W h_t``."""

    def __init__(self, gat: LogosGAT, hidden_dim: int) -> None:
        super().__init__()
        self.gat = gat
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """Return node embeddings and per-layer attention."""
        return self.gat(edge_index, edge_type)

    def score_pairs(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Score (source, target) pairs; returns raw logits, shape (B,)."""
        hi = h[pairs[:, 0]]
        hj = h[pairs[:, 1]]
        return torch.einsum("nd,de,ne->n", hi, self.W.weight, hj)

    def score_all(self, h: torch.Tensor) -> torch.Tensor:
        """Full (N, N) score matrix for filtered ranking."""
        return h @ self.W.weight @ h.t()


def pairs_tensor(samples: Sequence[Sample]) -> torch.Tensor:
    return torch.tensor([[s.src, s.dst] for s in samples], dtype=torch.long)


def labels_tensor(samples: Sequence[Sample]) -> torch.Tensor:
    return torch.tensor([float(s.label) for s in samples], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Training / evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate(
    model: ClosurePredictor,
    graph: RelationGraph,
    samples: Sequence[Sample],
    criterion: nn.Module,
) -> Tuple[float, float, List[float]]:
    """Return (loss, accuracy, logits) for a sample list."""
    if not samples:
        return float("nan"), float("nan"), []
    model.eval()
    with torch.no_grad():
        embeddings, _ = model.encode(graph.edge_index, graph.edge_type)
        pair_logits = model.score_pairs(embeddings, pairs_tensor(samples))
        labels = labels_tensor(samples)
        loss = float(criterion(pair_logits, labels))
        preds = (pair_logits > 0).float()
        acc = float((preds == labels).float().mean())
    return loss, acc, pair_logits.tolist()


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based ROC-AUC with tie handling (no sklearn dependency)."""
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0        # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def filtered_metrics(
    model: ClosurePredictor,
    graph: RelationGraph,
    facts: Facts,
    test_positives: Sequence[Sample],
) -> Dict[str, object]:
    """Filtered full-catalogue ranking (170 candidates).

    Every test query ``(source, relation)`` ranks all senses.  Known direct
    edges and known closure facts are masked out, except test positives of the
    same query — the standard filtered protocol.

    Returns MRR/Hits@K over test positives (and per query), plus per-query
    Precision@10/20 over the filtered candidate list.
    """
    model.eval()
    with torch.no_grad():
        h, _ = model.encode(graph.edge_index, graph.edge_type)
        score_matrix = model.score_all(h)

    num_nodes = len(graph.node_ids)
    keep_by_query: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    for sample in test_positives:
        keep_by_query[(sample.src, sample.relation)].add(sample.dst)

    ranks: List[int] = []
    rank_records: List[Tuple[int, int, str]] = []   # (src_sid, rank, relation)
    precision: Dict[int, List[float]] = {10: [], 20: []}
    for (src, relation), keep in keep_by_query.items():
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[src] = False
        for j in range(num_nodes):
            if j in keep:
                mask[j] = True
            elif (src, j) in facts.blocked_pairs:
                mask[j] = False
        row = score_matrix[src].masked_fill(~mask, -1e9)
        order = torch.argsort(row, descending=True)
        position = torch.empty(num_nodes, dtype=torch.long)
        position[order] = torch.arange(1, num_nodes + 1)

        for dst in keep:
            ranks.append(int(position[dst]))
            rank_records.append((graph.node_ids[src], int(position[dst]), relation))
        for k in precision:
            top = order[:k].tolist()
            precision[k].append(sum(1 for idx in top if idx in keep) / k)

    def hits_at(k: int) -> float:
        return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else float("nan")

    # 每个查询（source, relation）取最优命中，按 query 计 MRR。
    per_query: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for src_sid, rank, relation in rank_records:
        per_query[(src_sid, relation)].append(rank)
    query_rr = [1.0 / min(rs) for rs in per_query.values()]

    by_depth: Dict[int, List[int]] = defaultdict(list)
    for sample, rank in zip(test_positives, ranks):
        by_depth[sample.depth].append(rank)

    return {
        "mrr": sum(1.0 / r for r in ranks) / len(ranks) if ranks else float("nan"),
        "mrr_per_query": sum(query_rr) / len(query_rr) if query_rr else float("nan"),
        "hits@1": hits_at(1),
        "hits@3": hits_at(3),
        "hits@10": hits_at(10),
        "p@10": sum(precision[10]) / len(precision[10]) if precision[10] else float("nan"),
        "p@20": sum(precision[20]) / len(precision[20]) if precision[20] else float("nan"),
        "queries": len(per_query),
        "ranks": ranks,
        "by_depth": {
            depth: {
                "n": len(rs),
                "mrr": sum(1.0 / r for r in rs) / len(rs),
                "hits@1": sum(1 for r in rs if r <= 1) / len(rs),
                "hits@10": sum(1 for r in rs if r <= 10) / len(rs),
            }
            for depth, rs in sorted(by_depth.items())
        },
    }


def attention_stats(model: ClosurePredictor, graph: RelationGraph, target_idx: int) -> Dict[str, object]:
    """Mean normalised attention entropy and the target's top in-edges."""
    model.eval()
    with torch.no_grad():
        _, attentions = model.encode(graph.edge_index, graph.edge_type)
        alpha = attentions[0].mean(dim=1)                # (E,) head-average
        dst = graph.edge_index[1]

        ent = -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log())
        deg = torch.zeros(len(graph.node_ids))
        deg.index_add_(0, dst, torch.ones_like(alpha))
        node_ent = torch.zeros(len(graph.node_ids))
        node_ent.index_add_(0, dst, ent)
        valid = deg > 1
        normalised = node_ent[valid] / deg[valid].log()
        mean_entropy = float(normalised.mean()) if valid.any() else float("nan")

        # 目标节点的入边注意力（layer-1，头均值）
        incident = (dst == target_idx).nonzero(as_tuple=True)[0]
        rows = []
        for e in incident.tolist():
            src_idx = int(graph.edge_index[0, e])
            direction = int(graph.edge_dir[e])
            rel_idx = int(graph.edge_rel[e])
            if direction == 2:      # self-loop
                name, relation = "(self)", "—"
            else:
                name = graph.labels[src_idx]
                full = graph.relation_names[rel_idx]
                relation = (full[len("closure:"):] + " [closure]") if full.startswith("closure:") else full
            rows.append((float(alpha[e]), name, relation))
        rows.sort(reverse=True)
    return {"mean_entropy": mean_entropy, "target_top": rows[:3]}


def qualitative_examples(
    model: ClosurePredictor,
    graph: RelationGraph,
    facts: Facts,
    source_sid: int = 2,
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """Top-k predicted multi-hop targets for one sense, with closure flags."""
    if source_sid not in graph.id_to_index:
        return []
    s = graph.id_to_index[source_sid]
    model.eval()
    with torch.no_grad():
        h, _ = model.encode(graph.edge_index, graph.edge_type)
        scores = model.score_all(h)[s].clone()

    mask = torch.ones(len(graph.node_ids), dtype=torch.bool)
    mask[s] = False
    for j in range(len(graph.node_ids)):
        if (s, j) in facts.direct_pairs:       # hide trivial 1-hop neighbours
            mask[j] = False
    scores = scores.masked_fill(~mask, -1e9)
    top = torch.topk(scores, k=min(top_k, int(mask.sum())))

    out = []
    for value, idx in zip(top.values.tolist(), top.indices.tolist()):
        dst_sid = graph.node_ids[idx]
        closure_hit = facts.closure_map.get((source_sid, dst_sid))
        out.append(
            {
                "label": graph.labels[idx],
                "sid": dst_sid,
                "score": float(torch.sigmoid(torch.tensor(value))),
                "closure": closure_hit,
            }
        )
    return out


def closure_target_ranks(
    model: ClosurePredictor,
    graph: RelationGraph,
    facts: Facts,
    source_sid: int,
) -> List[Tuple[str, str, int, int]]:
    """Rank of every known closure target of a source (for the qualitative block)."""
    if source_sid not in graph.id_to_index:
        return []
    s = graph.id_to_index[source_sid]
    model.eval()
    with torch.no_grad():
        h, _ = model.encode(graph.edge_index, graph.edge_type)
        scores = model.score_all(h)[s].clone()
    mask = torch.ones(len(graph.node_ids), dtype=torch.bool)
    mask[s] = False
    for j in range(len(graph.node_ids)):
        if (s, j) in facts.direct_pairs:
            mask[j] = False
    scores = scores.masked_fill(~mask, -1e9)
    order = torch.argsort(scores, descending=True).tolist()
    rank_of = {idx: i + 1 for i, idx in enumerate(order)}

    out = []
    for (src_sid, dst_sid), (relation, depth) in facts.closure_map.items():
        if src_sid != source_sid:
            continue
        idx = graph.id_to_index.get(dst_sid)
        if idx is None:
            continue
        out.append((graph.labels[idx], relation, depth, rank_of.get(idx, -1)))
    out.sort(key=lambda x: x[3])
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def build_report(
    *,
    args: argparse.Namespace,
    graph: RelationGraph,
    stats: Dict[str, object],
    split_sizes: Tuple[int, int, int],
    train_pos: int,
    train_neg: int,
    best_epoch: int,
    epochs_run: int,
    stopped_early: bool,
    test_metrics: Dict[str, float],
    depth_metrics: Dict[int, Dict[str, float]],
    init_attention: Dict[str, object],
    final_attention: Dict[str, object],
    qualitative: Sequence[Dict[str, object]],
    target_ranks: Sequence[Tuple[str, str, int, int]],
    checkpoint: pathlib.Path,
) -> str:
    n_tr, n_va, n_te = split_sizes
    pos_weight = train_neg / max(train_pos, 1)
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Logos Transformer — Closure Training Report")
    lines.append("=" * 78)
    lines.append(f"generated_at        : {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"model_version       : {TRAIN_VERSION} (encoder: {PROTOTYPE_VERSION})")
    lines.append(f"database            : {args.db}")
    lines.append(f"checkpoint          : {checkpoint}")
    lines.append("")
    lines.append("--- Graph ---")
    lines.append(f"nodes               : {graph.stats['nodes']} senses")
    lines.append(f"direct semantic edges: {graph.stats['semantic_edges']}")
    lines.append(f"message edges       : {graph.stats['message_edges']} (bidirectional + self-loops)")
    lines.append(f"closure rows        : {graph.stats['closure_rows']} "
                 f"(depth=1: {graph.stats['closure_depth1']}, depth>=2: {graph.stats['closure_deep']})")
    lines.append("")
    lines.append("--- Dataset (closure depth >= 2) ---")
    lines.append(f"positives           : {stats['positives']}  {stats['positive_relations']}")
    lines.append(f"negatives           : {stats['negatives']} (sampled {args.neg_ratio}x, closure-free)")
    lines.append(f"total pairs         : {stats['total']}")
    lines.append(f"depth distribution  : {stats['depth_counts']}")
    lines.append(f"split (train/val/test): {n_tr} / {n_va} / {n_te}")
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
    lines.append(f"optimizer           : Adam, ReduceLROnPlateau(factor=0.5, patience=10)")
    lines.append(f"loss                : BCEWithLogitsLoss(pos_weight={pos_weight:.2f})")
    lines.append("")
    lines.append("--- Training ---")
    lines.append(f"epochs_run          : {epochs_run}")
    lines.append(f"best_epoch          : {best_epoch}")
    lines.append(f"early_stopped       : {stopped_early}")
    lines.append("")
    lines.append("--- Test metrics (sampled split) ---")
    lines.append(f"BCE loss            : {_fmt(test_metrics['bce'])}")
    lines.append(f"accuracy (logit>0)  : {_fmt(test_metrics['accuracy'])}")
    lines.append(f"ROC-AUC             : {_fmt(test_metrics['auc'])}")
    lines.append("")
    lines.append("--- Filtered full-catalogue ranking (170 candidates) ---")
    lines.append(f"Precision@10        : {_fmt(test_metrics['p@10'])}")
    lines.append(f"Precision@20        : {_fmt(test_metrics['p@20'])}")
    lines.append(f"MRR (per positive)  : {_fmt(test_metrics['mrr'])}")
    lines.append(f"MRR (per query)     : {_fmt(test_metrics['mrr_per_query'])}")
    lines.append(f"Hits@1 / @3 / @10   : {_fmt(test_metrics['hits@1'])} / "
                 f"{_fmt(test_metrics['hits@3'])} / {_fmt(test_metrics['hits@10'])}")
    lines.append(f"queries             : {test_metrics['queries']}")
    lines.append("by closure depth    :")
    for depth, m in depth_metrics.items():
        lines.append(f"  depth={depth} (n={m['n']:>3})  MRR={_fmt(m['mrr'])}  "
                     f"Hits@1={_fmt(m['hits@1'])}  Hits@10={_fmt(m['hits@10'])}")
    lines.append("")
    lines.append("--- Attention shift (layer-1, head-average) ---")
    lines.append(f"mean normalised entropy (init)    : {_fmt(init_attention['mean_entropy'])}")
    lines.append(f"mean normalised entropy (trained) : {_fmt(final_attention['mean_entropy'])}")
    lines.append("  (1.000 = uniform, lower = more selective)")
    lines.append(f"attention @ {graph.labels[graph.id_to_index.get(2, 0)]} (init):")
    for alpha, name, relation in init_attention["target_top"]:
        lines.append(f"  {name:<16} {relation:<16} {alpha:.4f}")
    lines.append(f"attention @ {graph.labels[graph.id_to_index.get(2, 0)]} (trained):")
    for alpha, name, relation in final_attention["target_top"]:
        lines.append(f"  {name:<16} {relation:<16} {alpha:.4f}")
    lines.append("")
    lines.append("--- Qualitative: top-5 multi-hop targets for「猫」(sense 2) ---")
    if qualitative:
        for i, item in enumerate(qualitative, 1):
            tag = "closure" if item["closure"] else "no closure"
            extra = ""
            if item["closure"]:
                rel, depth = item["closure"]
                extra = f"  [{rel}, depth {depth}]"
            lines.append(f"  {i}. {item['label']}({item['sid']})  score={item['score']:.4f}  "
                         f"{tag}{extra}")
    else:
        lines.append("  (source sense not found)")
    if target_ranks:
        lines.append("  Known closure targets and their ranks:")
        for label, relation, depth, rank in target_ranks:
            lines.append(f"    {label:<14} {relation} depth={depth}  rank=#{rank}")
    lines.append("")
    lines.append("--- Caveats & data augmentation ideas ---")
    lines.append(f"* Only {stats['positives']} positives; the task is heavily is_a-dominated.")
    lines.append("* The head h_s^T W h_t is relation-agnostic. has_part has a single positive;")
    lines.append("  relation-conditioned heads (W_r) are the natural next step.")
    lines.append("* A 2-layer GAT sees 2 hops, but positives reach depth 6; depth>=4 is")
    lines.append("  largely structural extrapolation, not message passing.")
    lines.append("* The model is transductive: node embeddings are per sense_id and the")
    lines.append("  checkpoint only loads against the same node ordering.")
    lines.append("* Splits are stratified by relation, so a source often appears in both")
    lines.append("  train and test (pair completion, not cold-start sources). The high AUC")
    lines.append("  partly reflects this; a source-disjoint split is the honest stress test.")
    lines.append("* Augmentation options: (a) add inverse relation types and use reverse")
    lines.append("  edges as extra positives; (b) expand the animal kingdom library;")
    lines.append("  (c) restrict to depth-2 positives for a fair 2-hop task; (d) k-fold")
    lines.append("  CV instead of a single split.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train the Logos GAT to predict depth>=2 closure edges."
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

    # ---- dependency / data guards -----------------------------------------
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

        positives, negatives, facts, data_stats = load_samples(
            conn, graph, neg_ratio=args.neg_ratio, seed=args.seed
        )
    finally:
        conn.close()

    train, val, test = stratified_split(positives, negatives, seed=args.seed)
    train_pos = sum(1 for s in train if s.label == 1)
    train_neg = len(train) - train_pos
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

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    print("=" * 78)
    print("Logos Transformer — closure prediction training")
    print("=" * 78)
    print(f"db={db_path}")
    print(f"graph: {graph.stats['nodes']} nodes / {graph.stats['semantic_edges']} direct edges"
          f" / {graph.stats['message_edges']} message edges")
    print(f"positives={data_stats['positives']}  negatives={data_stats['negatives']}"
          f"  depth={data_stats['depth_counts']}")
    print(f"split: train={len(train)} val={len(val)} test={len(test)}"
          f"  (pos_weight={pos_weight:.2f})")
    print(f"model: {args.layers} layers x {args.heads} heads, hidden={args.hidden},"
          f" params={sum(p.numel() for p in model.parameters()):,}")
    print("-" * 78)

    init_attention = attention_stats(model, graph, graph.id_to_index.get(2, 0))
    init_loss, init_acc, _ = evaluate(model, graph, train, criterion)
    print(f"init: train loss={init_loss:.4f} acc={init_acc:.4f} | "
          f"attention entropy={init_attention['mean_entropy']:.4f}")

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epochs_run = 0
    batch_rng = random.Random(args.seed + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "closure_gat_best.pt"

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        model.train()
        order = list(train)
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
        val_loss, val_acc, _ = evaluate(model, graph, val, criterion)
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

    test_loss, test_acc, test_logits = evaluate(model, graph, test, criterion)
    test_labels = [s.label for s in test]
    auc = roc_auc(test_labels, test_logits)
    ranking = filtered_metrics(model, graph, facts, [s for s in test if s.label == 1])
    final_attention = attention_stats(model, graph, graph.id_to_index.get(2, 0))
    qualitative = qualitative_examples(model, graph, facts, source_sid=2, top_k=5)
    target_ranks = closure_target_ranks(model, graph, facts, source_sid=2)

    test_metrics = {
        "bce": test_loss,
        "accuracy": test_acc,
        "auc": auc,
        "p@10": ranking["p@10"],
        "p@20": ranking["p@20"],
        "mrr": ranking["mrr"],
        "mrr_per_query": ranking["mrr_per_query"],
        "hits@1": ranking["hits@1"],
        "hits@3": ranking["hits@3"],
        "hits@10": ranking["hits@10"],
        "queries": ranking["queries"],
    }

    print("-" * 78)
    print(f"test: BCE={test_loss:.4f} acc={test_acc:.4f} AUC={_fmt(auc)}")
    print(f"filtered ranking: P@10={_fmt(ranking['p@10'])} P@20={_fmt(ranking['p@20'])} "
          f"MRR={_fmt(ranking['mrr'])} "
          f"(per-query {_fmt(ranking['mrr_per_query'])}) "
          f"Hits@1={_fmt(ranking['hits@1'])} @3={_fmt(ranking['hits@3'])} "
          f"@10={_fmt(ranking['hits@10'])}")
    print(f"attention entropy: init={init_attention['mean_entropy']:.4f} "
          f"-> trained={final_attention['mean_entropy']:.4f}")
    if qualitative:
        print("top-5 multi-hop targets for 猫:")
        for i, item in enumerate(qualitative, 1):
            flag = " [closure]" if item["closure"] else ""
            print(f"  {i}. {item['label']}({item['sid']}) score={item['score']:.4f}{flag}")

    report = build_report(
        args=args,
        graph=graph,
        stats=data_stats,
        split_sizes=(len(train), len(val), len(test)),
        train_pos=train_pos,
        train_neg=train_neg,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
        test_metrics=test_metrics,
        depth_metrics=ranking["by_depth"],
        init_attention=init_attention,
        final_attention=final_attention,
        qualitative=qualitative,
        target_ranks=target_ranks,
        checkpoint=checkpoint_path,
    )
    report_path = output_dir / "training_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print("-" * 78)
    print(f"checkpoint : {checkpoint_path}")
    print(f"report     : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
