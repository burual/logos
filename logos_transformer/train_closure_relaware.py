#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relation-aware closure prediction (per-relation scoring head).

Motivation
----------
``train_closure_hardneg.py`` showed that hard negatives give a real but *coarse*
source-conditioned signal: the model learns cross-branch membership (animal ->
生物, tool -> 人造物) but still fails fine-grained same-branch ranking.  One
reason is that the scoring head is relation-agnostic:

    score(s, t) = h_s^T W h_t

The same ``W`` is used for every relation, so the model cannot represent
"``is_a`` ancestor" differently from "``has_property`` attribute" or
"``has_part`` component".  This script replaces the single ``W`` with:

    score(s, t, r) = h_s^T W_r h_t

Design
------
* One ``W_r`` (32 x 32) for every relation in the graph vocabulary (46 types in
  the expanded animal-kingdom database).
* A single ``W_shared`` for relations with fewer than ``MIN_DEDICATED`` (5)
  training positives.  Those relations use a learned gate:

      W_eff(r) = sigmoid(g_r) * W_r + (1 - sigmoid(g_r)) * W_shared

* Relations that never occur in a training positive fall back to ``W_shared``.
* The GAT encoder is unchanged.  Optionally freeze it for the first
  ``--freeze-gat-epochs`` epochs so the relation heads can adapt to the initial
  embeddings before the encoder moves.

The rest of the protocol is deliberately identical to the hard-negative run:
same source-disjoint split, same hard/medium/easy sampling, same optimizer,
same checkpoint selection (best val loss), and the same hard-negative and
target-prior diagnostics.

Expanded-data caveat
--------------------
The current graph has 46 relation types and 638 depth>=2 closure positives.
Five relations have >=5 train positives (``is_a``, ``can_do``, ``has_property``,
``typically_has``, ``part_of``) and therefore receive dedicated heads;
``has_part`` has one positive and is gated-shared; the remaining 40 relation
types have no depth>=2 positive and fall back to the shared head.  The test
split exercises ``is_a``, ``can_do``, ``has_property`` and ``typically_has``.
``opposite_of`` has only depth=1 closure and is not a closure-prediction target.
The report prints per-relation metrics so each trained head is evaluated on its
own relation type rather than only on ``is_a``.

Usage
-----
    python logos_transformer/train_closure_relaware.py --epochs 5   # smoke
    python logos_transformer/train_closure_relaware.py              # full run

Outputs
-------
    <output-dir>/closure_gat_relaware_best.pt          best val loss (primary)
    <output-dir>/closure_gat_relaware_best_margin.pt   best val margin (diagnostic)
    <output-dir>/training_report.txt

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

warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment dependent
    print("Requires torch. Install with: pip install torch", file=sys.stderr)
    raise SystemExit(1)

# 复用图导出、数据容器、采样器与基线指标。既有训练脚本保持只读。
try:
    from prototype_gat import RelationGraph, build_model, load_graph  # type: ignore
    from train_closure import (  # type: ignore
        Facts,
        Sample,
        _fmt,
        attention_stats,
        labels_tensor,
        pairs_tensor,
        roc_auc,
    )
    from train_closure_disjoint import (  # type: ignore
        eval_bucket,
        load_positives,
        split_by_source,
        target_prior_mrr,
    )
    from train_closure_hardneg import (  # type: ignore
        NegativePools,
        SamplingStats,
        _hard_row,
        _pool_line,
        _split_ratios,
        build_negative_pools,
        compute_scores,
        hard_negative_metrics,
        load_closure_checkpoint,
        pool_key,
        relation_band_summary,
        sample_hardneg,
        summarize_pools,
    )
except ImportError:  # pragma: no cover
    from logos_transformer.prototype_gat import (  # type: ignore
        RelationGraph,
        build_model,
        load_graph,
    )
    from logos_transformer.train_closure import (  # type: ignore
        Facts,
        Sample,
        _fmt,
        attention_stats,
        labels_tensor,
        pairs_tensor,
        roc_auc,
    )
    from logos_transformer.train_closure_disjoint import (  # type: ignore
        eval_bucket,
        load_positives,
        split_by_source,
        target_prior_mrr,
    )
    from logos_transformer.train_closure_hardneg import (  # type: ignore
        NegativePools,
        SamplingStats,
        _hard_row,
        _pool_line,
        _split_ratios,
        build_negative_pools,
        compute_scores,
        hard_negative_metrics,
        load_closure_checkpoint,
        pool_key,
        relation_band_summary,
        sample_hardneg,
        summarize_pools,
    )

RELAWARE_VERSION = "logos-gat-closure-relaware-v0.1"
DEFAULT_OUTPUT_DIR = "./logos_transformer/checkpoints_relaware"
DEFAULT_COMPARE_CHECKPOINT = (
    "./logos_transformer/checkpoints_hardneg/closure_gat_hardneg_best.pt"
)
SPLIT_NAMES = ("train", "val", "test")
MIN_DEDICATED = 5


# --------------------------------------------------------------------------- #
# Relation-aware scoring head
# --------------------------------------------------------------------------- #
class RelationAwareClosurePredictor(nn.Module):
    """GAT encoder plus per-relation bilinear heads ``h_s^T W_r h_t``.

    ``W_r`` is instantiated for every relation in the graph vocabulary.  The
    effective matrix depends on the number of *training* positives for that
    relation:

    * ``count >= 5``  -> dedicated ``W_r``;
    * ``0 < count < 5`` -> ``sigmoid(g_r) * W_r + (1 - sigmoid(g_r)) * W_shared``;
    * ``count == 0``  -> ``W_shared`` (the head was never trained).

    The counts are stored as a buffer, so a checkpoint carries the same
    dedicated/gated/unseen decision it was trained with.
    """

    def __init__(
        self,
        gat: nn.Module,
        hidden_dim: int,
        relation_names: Sequence[str],
        relation_counts: Dict[str, int],
    ) -> None:
        super().__init__()
        self.gat = gat
        self.hidden_dim = hidden_dim
        self.relation_names = list(relation_names)
        self.rel_to_idx = {name: i for i, name in enumerate(self.relation_names)}

        # 每种关系一个 32x32 矩阵；额外一个共享矩阵与一个门控标量。
        self.W_r = nn.ParameterDict()
        for name in self.relation_names:
            param = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(param)
            self.W_r[name] = param

        self.W_shared = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W_shared)

        # 稀有关系的门控：sigmoid(g) 越大越偏向自己的 W_r。
        self.share_logit = nn.ParameterDict()
        for name in self.relation_names:
            self.share_logit[name] = nn.Parameter(torch.zeros(1))

        counts = [int(relation_counts.get(name, 0)) for name in self.relation_names]
        self.register_buffer("relation_count", torch.tensor(counts, dtype=torch.long))
        self.register_buffer(
            "relation_seen", (torch.tensor(counts, dtype=torch.long) >= MIN_DEDICATED).long()
        )

    # -- scoring ---------------------------------------------------------- #
    def _effective_matrix(self, relation: str) -> torch.Tensor:
        idx = self.rel_to_idx.get(relation)
        if idx is None:
            return self.W_shared
        count = int(self.relation_count[idx].item())
        if count >= MIN_DEDICATED:
            return self.W_r[relation]
        if count == 0:
            return self.W_shared
        gate = torch.sigmoid(self.share_logit[relation])
        return gate * self.W_r[relation] + (1.0 - gate) * self.W_shared

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """Return node embeddings and per-layer attention (GAT unchanged)."""
        return self.gat(edge_index, edge_type)

    def score_pairs(
        self,
        h: torch.Tensor,
        pairs: torch.Tensor,
        relations: Sequence[str],
    ) -> torch.Tensor:
        """Score (source, target, relation) triples; raw logits, shape (B,)."""
        if pairs.numel() == 0:
            return h.new_zeros(0)
        out = h.new_empty(len(relations))
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, relation in enumerate(relations):
            buckets[relation].append(i)
        for relation, indices in buckets.items():
            idx = torch.tensor(indices, dtype=torch.long, device=pairs.device)
            hi = h[pairs[idx, 0]]
            hj = h[pairs[idx, 1]]
            matrix = self._effective_matrix(relation)
            out[idx] = torch.einsum("nd,de,ne->n", hi, matrix, hj)
        return out

    def score_all(self, h: torch.Tensor, relation: str) -> torch.Tensor:
        """Full (N, N) score matrix for one relation."""
        return h @ self._effective_matrix(relation) @ h.t()

    # -- reporting -------------------------------------------------------- #
    def head_summary(self) -> Dict[str, object]:
        counts = {
            name: int(self.relation_count[i].item())
            for i, name in enumerate(self.relation_names)
        }
        dedicated = [n for n, c in counts.items() if c >= MIN_DEDICATED]
        gated = [n for n, c in counts.items() if 0 < c < MIN_DEDICATED]
        unseen = [n for n, c in counts.items() if c == 0]
        return {
            "counts": counts,
            "dedicated": dedicated,
            "gated": gated,
            "unseen": unseen,
            "n_relation_types": len(self.relation_names),
        }


# --------------------------------------------------------------------------- #
# Metrics (relation-aware variants of the hard-negative protocol)
# --------------------------------------------------------------------------- #
def encode_and_score(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    relations: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
    """Encode once and build one (N, N) score matrix per relation."""
    model.eval()
    with torch.no_grad():
        h, attentions = model.encode(graph.edge_index, graph.edge_type)
        scores = {rel: model.score_all(h, rel) for rel in sorted(set(relations))}
    return h, scores, attentions


def pair_logits(
    model: RelationAwareClosurePredictor,
    h: torch.Tensor,
    samples: Sequence[Sample],
) -> torch.Tensor:
    if not samples:
        return h.new_zeros(0)
    return model.score_pairs(
        h,
        pairs_tensor(samples),
        [s.relation for s in samples],
    )


def _agg_rank_records(records: Sequence[Tuple[Sample, int]]) -> Dict[str, float]:
    if not records:
        return {}
    n = len(records)
    return {
        "n": n,
        "mrr": sum(1.0 / r for _, r in records) / n,
        "hits@1": sum(1 for _, r in records if r <= 1) / n,
        "hits@3": sum(1 for _, r in records if r <= 3) / n,
        "hits@10": sum(1 for _, r in records if r <= 10) / n,
    }


def _agg_rr_records(records: Sequence[Tuple[Sample, float]]) -> Dict[str, float]:
    if not records:
        return {}
    n = len(records)
    return {
        "n": n,
        "mrr": sum(rr for _, rr in records) / n,
        "hits@1": sum(1 for _, rr in records if rr >= 1.0) / n,
        "hits@10": sum(1 for _, rr in records if rr >= 0.1) / n,
    }


def filtered_metrics_relaware(
    score_by_rel: Dict[str, torch.Tensor],
    graph: RelationGraph,
    facts: Facts,
    positives: Sequence[Sample],
) -> Dict[str, object]:
    """Filtered full-catalogue ranking, with the query's own relation head.

    This mirrors ``train_closure.filtered_metrics`` but selects ``S_r`` for the
    query relation.  Per-depth and per-relation breakdowns are computed from
    explicit per-positive ranks, avoiding the set-order zip bug in the old
    ``by_depth`` view.
    """
    if not positives:
        return {
            "mrr": float("nan"), "mrr_per_query": float("nan"),
            "hits@1": float("nan"), "hits@3": float("nan"),
            "hits@10": float("nan"), "p@10": float("nan"),
            "queries": 0, "ranks": [], "by_depth": {}, "by_relation": {},
        }

    num_nodes = len(graph.node_ids)
    keep_by_query: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    sample_by: Dict[Tuple[int, str, int], Sample] = {}
    for sample in positives:
        keep_by_query[(sample.src, sample.relation)].add(sample.dst)
        sample_by[(sample.src, sample.relation, sample.dst)] = sample

    ranks: List[int] = []
    records: List[Tuple[Sample, int]] = []
    precision: Dict[int, List[float]] = {10: [], 20: []}
    query_ranks: Dict[Tuple[int, str], List[int]] = defaultdict(list)

    for (src, relation), keep in keep_by_query.items():
        matrix = score_by_rel.get(relation)
        if matrix is None:
            continue
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[src] = False
        for j in range(num_nodes):
            if j in keep:
                mask[j] = True
            elif (src, j) in facts.blocked_pairs:
                mask[j] = False
        row = matrix[src].masked_fill(~mask, -1e9)
        order = torch.argsort(row, descending=True)
        position = torch.empty(num_nodes, dtype=torch.long)
        position[order] = torch.arange(1, num_nodes + 1)

        for dst in keep:
            sample = sample_by[(src, relation, dst)]
            rank = int(position[dst])
            ranks.append(rank)
            records.append((sample, rank))
            query_ranks[(src, relation)].append(rank)
        for k in precision:
            top = order[:k].tolist()
            precision[k].append(sum(1 for idx in top if idx in keep) / k)

    def hits_at(k: int) -> float:
        return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else float("nan")

    by_depth: Dict[int, List[Tuple[Sample, int]]] = defaultdict(list)
    by_relation: Dict[str, List[Tuple[Sample, int]]] = defaultdict(list)
    for sample, rank in records:
        by_depth[sample.depth].append((sample, rank))
        by_relation[sample.relation].append((sample, rank))

    query_rr = [1.0 / min(rs) for rs in query_ranks.values()]
    return {
        "mrr": sum(1.0 / r for r in ranks) / len(ranks) if ranks else float("nan"),
        "mrr_per_query": sum(query_rr) / len(query_rr) if query_rr else float("nan"),
        "hits@1": hits_at(1),
        "hits@3": hits_at(3),
        "hits@10": hits_at(10),
        "p@10": sum(precision[10]) / len(precision[10]) if precision[10] else float("nan"),
        "p@20": sum(precision[20]) / len(precision[20]) if precision[20] else float("nan"),
        "queries": len(query_ranks),
        "ranks": ranks,
        "by_depth": {
            depth: _agg_rank_records(items) for depth, items in sorted(by_depth.items())
        },
        "by_relation": {
            rel: _agg_rank_records(items) for rel, items in sorted(by_relation.items())
        },
    }


def target_prior_relaware(
    score_by_rel: Dict[str, torch.Tensor],
    graph: RelationGraph,
    facts: Facts,
    positives: Sequence[Sample],
) -> Dict[str, object]:
    """Source-independent target prior, computed per relation.

    For relation ``r`` the prior is ``S_r.mean(dim=0)``: the target-only average
    score over all sources.  This is the same diagnostic as §9.8, but now each
    relation has its own score surface.
    """
    if not positives:
        return {"overall": float("nan"), "by_relation": {}}
    num_nodes = len(graph.node_ids)
    keep_by_query: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    for sample in positives:
        keep_by_query[(sample.src, sample.relation)].add(sample.dst)

    bias_cache: Dict[str, torch.Tensor] = {}
    rr_records: List[Tuple[Sample, float]] = []
    by_relation: Dict[str, List[Tuple[Sample, float]]] = defaultdict(list)
    for sample in positives:
        matrix = score_by_rel.get(sample.relation)
        if matrix is None:
            continue
        if sample.relation not in bias_cache:
            bias_cache[sample.relation] = matrix.mean(dim=0)
        bias = bias_cache[sample.relation]
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
        rr = 1.0 / rank
        rr_records.append((sample, rr))
        by_relation[sample.relation].append((sample, rr))

    return {
        "overall": sum(rr for _, rr in rr_records) / len(rr_records)
        if rr_records else float("nan"),
        "by_relation": {
            rel: _agg_rr_records(items) for rel, items in sorted(by_relation.items())
        },
    }


@dataclasses.dataclass
class HardMetricsRel:
    model_mrr: float
    prior_mrr: float
    margin: float
    mean_k: float
    min_k: int
    max_k: int
    random_mrr: float
    beats_random_frac: float
    n_queries: int
    n_skipped: int
    by_depth: Dict[int, Dict[str, float]]
    by_relation: Dict[str, Dict[str, float]]


def _agg_hard_records(
    records: Sequence[Tuple[float, float, int]]
) -> Dict[str, float]:
    if not records:
        return {}
    n = len(records)
    return {
        "n": n,
        "model_mrr": sum(r[0] for r in records) / n,
        "prior_mrr": sum(r[1] for r in records) / n,
        "margin": sum(r[0] - r[1] for r in records) / n,
        "mean_k": sum(r[2] for r in records) / n,
        "random_mrr": sum(1.0 / (r[2] + 1) for r in records) / n,
    }


def hard_metrics(
    score_by_rel: Dict[str, torch.Tensor],
    pools: NegativePools,
    positives: Sequence[Sample],
    min_k: int = 0,
    depths: Optional[Set[int]] = None,
) -> HardMetricsRel:
    """Rank each positive against all eligible hard negatives, per relation."""
    bias_cache: Dict[str, torch.Tensor] = {}
    model_rr: List[float] = []
    prior_rr: List[float] = []
    ks: List[int] = []
    skipped = 0
    depth_records: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    relation_records: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)

    for pos in positives:
        pool = pools.hard[pool_key(pos)]
        if not pool or len(pool) < min_k:
            skipped += 1
            continue
        if depths is not None and pos.depth not in depths:
            skipped += 1
            continue
        matrix = score_by_rel.get(pos.relation)
        if matrix is None:
            skipped += 1
            continue
        if pos.relation not in bias_cache:
            bias_cache[pos.relation] = matrix.mean(dim=0)
        bias = bias_cache[pos.relation]

        cand = pool + [pos.dst]
        cand_t = torch.tensor(cand, dtype=torch.long)
        row = matrix[pos.src].index_select(0, cand_t)
        order = torch.argsort(row, descending=True).tolist()
        rank = order.index(len(pool)) + 1

        prior_row = bias.index_select(0, cand_t)
        prior_order = torch.argsort(prior_row, descending=True).tolist()
        prior_rank = prior_order.index(len(pool)) + 1

        rec = (1.0 / rank, 1.0 / prior_rank, len(pool))
        model_rr.append(rec[0])
        prior_rr.append(rec[1])
        ks.append(len(pool))
        depth_records[pos.depth].append(rec)
        relation_records[pos.relation].append(rec)

    n = len(model_rr)
    if n == 0:
        return HardMetricsRel(
            model_mrr=float("nan"), prior_mrr=float("nan"), margin=float("nan"),
            mean_k=0.0, min_k=0, max_k=0, random_mrr=float("nan"),
            beats_random_frac=float("nan"), n_queries=0, n_skipped=skipped,
            by_depth={}, by_relation={},
        )
    return HardMetricsRel(
        model_mrr=sum(model_rr) / n,
        prior_mrr=sum(prior_rr) / n,
        margin=sum(model_rr) / n - sum(prior_rr) / n,
        mean_k=sum(ks) / n,
        min_k=min(ks),
        max_k=max(ks),
        random_mrr=sum(1.0 / (k + 1) for k in ks) / n,
        beats_random_frac=sum(
            1 for rr, k in zip(model_rr, ks) if rr > 1.0 / (k + 1)
        ) / n,
        n_queries=n,
        n_skipped=skipped,
        by_depth={
            depth: _agg_hard_records(records)
            for depth, records in sorted(depth_records.items())
        },
        by_relation={
            rel: _agg_hard_records(records)
            for rel, records in sorted(relation_records.items())
        },
    )


def hard_rank_for_positive(
    score_by_rel: Dict[str, torch.Tensor],
    pools: NegativePools,
    positive: Sample,
) -> Optional[Dict[str, object]]:
    """Hard-pool rank and prior rank for one positive (for 猫 diagnostics)."""
    pool = pools.hard[pool_key(positive)]
    matrix = score_by_rel.get(positive.relation)
    if not pool or matrix is None:
        return None
    bias = matrix.mean(dim=0)
    cand_t = torch.tensor(pool + [positive.dst], dtype=torch.long)
    order = torch.argsort(
        matrix[positive.src].index_select(0, cand_t), descending=True
    ).tolist()
    prior_order = torch.argsort(
        bias.index_select(0, cand_t), descending=True
    ).tolist()
    return {
        "hard_rank": order.index(len(pool)) + 1,
        "prior_rank": prior_order.index(len(pool)) + 1,
        "k": len(pool),
    }


def attention_by_relation(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """Layer-1 head-average attention, top-k edges for each base relation.

    Self-loops are excluded (they have ``edge_rel = -1``).  For a reverse
    message edge we print the original semantic direction and mark the message
    as reverse.
    """
    model.eval()
    with torch.no_grad():
        _, attentions = model.encode(graph.edge_index, graph.edge_type)
    alpha = attentions[0].mean(dim=1)
    groups: Dict[str, List[Tuple[float, int, int, str]]] = defaultdict(list)
    for e in range(graph.edge_index.size(1)):
        rel_idx = int(graph.edge_rel[e])
        if rel_idx < 0:
            continue
        relation = graph.relation_names[rel_idx]
        direction = int(graph.edge_dir[e])
        if direction == 0:
            sem_src = int(graph.edge_index[0, e])
            sem_dst = int(graph.edge_index[1, e])
            marker = "forward"
        elif direction == 1:
            sem_src = int(graph.edge_index[1, e])
            sem_dst = int(graph.edge_index[0, e])
            marker = "reverse"
        else:
            continue
        groups[relation].append((float(alpha[e]), sem_src, sem_dst, marker))

    out: List[Dict[str, object]] = []
    for relation in sorted(groups):
        rows = sorted(groups[relation], key=lambda x: x[0], reverse=True)[:top_k]
        out.append({
            "relation": relation,
            "n_edges": len(groups[relation]),
            "top": rows,
        })
    return out


def qualitative_examples_relaware(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    facts: Facts,
    h: torch.Tensor,
    score_by_rel: Dict[str, torch.Tensor],
    source_sid: int = 2,
    relation: str = "is_a",
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """Top-k multi-hop targets for one source, using the relation's head."""
    if source_sid not in graph.id_to_index:
        return []
    matrix = score_by_rel.get(relation)
    if matrix is None:
        return []
    s = graph.id_to_index[source_sid]
    scores = matrix[s].clone()
    mask = torch.ones(len(graph.node_ids), dtype=torch.bool)
    mask[s] = False
    for j in range(len(graph.node_ids)):
        if (s, j) in facts.direct_pairs:
            mask[j] = False
    scores = scores.masked_fill(~mask, -1e9)
    top = torch.topk(scores, k=min(top_k, int(mask.sum())))

    out: List[Dict[str, object]] = []
    for value, idx in zip(top.values.tolist(), top.indices.tolist()):
        dst_sid = graph.node_ids[idx]
        closure_hit = facts.closure_map.get((source_sid, dst_sid))
        out.append({
            "label": graph.labels[idx],
            "sid": dst_sid,
            "score": float(torch.sigmoid(torch.tensor(value))),
            "closure": closure_hit,
        })
    return out


def closure_target_ranks_relaware(
    graph: RelationGraph,
    facts: Facts,
    score_by_rel: Dict[str, torch.Tensor],
    source_sid: int = 2,
) -> List[Tuple[str, str, int, int]]:
    """Rank every known closure target of a source under its relation head."""
    if source_sid not in graph.id_to_index:
        return []
    s = graph.id_to_index[source_sid]
    num_nodes = len(graph.node_ids)
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[s] = False
    for j in range(num_nodes):
        if (s, j) in facts.direct_pairs:
            mask[j] = False

    out: List[Tuple[str, str, int, int]] = []
    rank_cache: Dict[str, Dict[int, int]] = {}
    for (src_sid, dst_sid), (relation, depth) in facts.closure_map.items():
        if src_sid != source_sid:
            continue
        idx = graph.id_to_index.get(dst_sid)
        matrix = score_by_rel.get(relation)
        if idx is None or matrix is None:
            continue
        if relation not in rank_cache:
            row = matrix[s].masked_fill(~mask, -1e9)
            order = torch.argsort(row, descending=True).tolist()
            rank_cache[relation] = {node: i + 1 for i, node in enumerate(order)}
        out.append((graph.labels[idx], relation, depth, rank_cache[relation][idx]))
    out.sort(key=lambda x: x[3])
    return out


@torch.no_grad()
def eval_bucket_relaware(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    facts: Facts,
    samples: Sequence[Sample],
    criterion: nn.Module,
    h: torch.Tensor,
    score_by_rel: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    """Loss/accuracy/AUC on sampled pairs + filtered ranking on positives."""
    positives = [s for s in samples if s.label == 1]
    logits = pair_logits(model, h, samples)
    labels = labels_tensor(samples)
    loss = float(criterion(logits, labels))
    acc = float(((logits > 0).float() == labels).float().mean())
    auc = roc_auc([s.label for s in samples], logits.tolist())
    ranking = filtered_metrics_relaware(score_by_rel, graph, facts, positives)
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
        "by_relation": ranking["by_relation"],
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _relation_count_rows(
    graph: RelationGraph, train_positives: Sequence[Sample]
) -> List[Dict[str, object]]:
    counts = Counter(p.relation for p in train_positives)
    rows: List[Dict[str, object]] = []
    for rel in graph.relation_names:
        c = counts.get(rel, 0)
        if c >= MIN_DEDICATED:
            kind = "dedicated W_r"
        elif c > 0:
            kind = "gated W_r + W_shared"
        else:
            kind = "unseen -> W_shared"
        rows.append({"relation": rel, "train_pos": c, "kind": kind})
    return rows


def build_report(
    *,
    args: argparse.Namespace,
    graph: RelationGraph,
    total_sources: int,
    mode: str,
    sources_by_split: Dict[str, Set[int]],
    data: Dict[str, Dict[str, object]],
    pool_summary: Dict[str, Dict[str, object]],
    sampling_stats: Dict[str, SamplingStats],
    relation_bands: Sequence[Dict[str, object]],
    relation_rows: Sequence[Dict[str, object]],
    best_epoch: int,
    best_val_loss: float,
    best_margin_epoch: int,
    best_val_margin: float,
    epochs_run: int,
    stopped_early: bool,
    margin_history: Sequence[Tuple[int, float, float, float, float]],
    metrics: Dict[str, Dict[str, object]],
    prior: Dict[str, Dict[str, object]],
    hard_by_split: Dict[str, HardMetricsRel],
    hard_test_k2: HardMetricsRel,
    hard_test_d2: HardMetricsRel,
    baseline_name: str,
    baseline_metrics: Optional[Dict[str, object]],
    baseline_hard: Optional[HardMetricsRel],
    baseline_hard_k2: Optional[HardMetricsRel],
    baseline_hard_d2: Optional[HardMetricsRel],
    baseline_prior_test: Optional[float],
    baseline_rel_metrics: Optional[Dict[str, object]],
    baseline_rel_hard: Optional[HardMetricsRel],
    init_attention: Dict[str, object],
    final_attention: Dict[str, object],
    attention_rel: Sequence[Dict[str, object]],
    cat_split: str,
    qualitative: Sequence[Dict[str, object]],
    target_ranks: Sequence[Tuple[str, str, int, int]],
    cat_hard: Optional[Dict[str, object]],
    checkpoint: pathlib.Path,
    margin_checkpoint: pathlib.Path,
) -> str:
    def depth_counter(items: Sequence[Sample]) -> Dict[int, int]:
        return dict(sorted(Counter(s.depth for s in items).items()))

    train = data["train"]
    train_pos = len(train["pos"])                                   # type: ignore[arg-type]
    train_neg = len(train["samples"]) - train_pos                    # type: ignore[arg-type]
    pos_weight = train_neg / max(train_pos, 1)

    lines: List[str] = []
    lines.append("=" * 108)
    lines.append("Logos Transformer — Relation-Aware Closure Training Report")
    lines.append("=" * 108)
    lines.append(f"generated_at        : {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"model_version       : {RELAWARE_VERSION}")
    lines.append(f"database            : {args.db}")
    lines.append(f"checkpoint (val loss): {checkpoint}")
    lines.append(f"checkpoint (margin)  : {margin_checkpoint}  [diagnostic only]")
    lines.append(f"split_mode          : {mode}")
    lines.append("")
    lines.append("--- Graph ---")
    lines.append(f"nodes                : {graph.stats['nodes']} senses")
    lines.append(f"direct semantic edges: {graph.stats['semantic_edges']}")
    lines.append(f"message edges        : {graph.stats['message_edges']} "
                 "(bidirectional + self-loops)")
    lines.append(f"closure rows         : {graph.stats['closure_rows']} "
                 f"(depth=1: {graph.stats['closure_depth1']}, "
                 f"depth>=2: {graph.stats['closure_deep']})")
    lines.append(f"relation vocabulary  : {len(graph.relation_names)} types")
    lines.append("")
    lines.append("--- Relation-aware scoring head ---")
    lines.append(f"score                : h_s^T W_r h_t   (per relation)")
    lines.append(f"W_r matrices         : {len(graph.relation_names)} x "
                 f"{args.hidden}x{args.hidden} = "
                 f"{len(graph.relation_names) * args.hidden * args.hidden:,} params")
    lines.append(f"W_shared             : {args.hidden}x{args.hidden} = "
                 f"{args.hidden * args.hidden:,} params")
    lines.append(f"dedicated threshold  : >= {MIN_DEDICATED} train positives -> W_r only")
    lines.append("rare relations       : sigmoid(g_r)*W_r + (1-sigmoid(g_r))*W_shared")
    lines.append("unseen relations     : W_shared")
    lines.append("")
    lines.append(f"{'relation':<22} {'train_pos':>9} {'effective head':<32}")
    lines.append("-" * 66)
    for row in relation_rows:
        if row["train_pos"] or row["kind"] != "unseen -> W_shared":
            lines.append(f"{str(row['relation']):<22} {row['train_pos']:>9} "
                         f"{str(row['kind']):<32}")
    lines.append("  (only active / rare rows shown; the remaining vocabulary uses W_shared)")
    lines.append("")
    lines.append("--- Source-disjoint split (identical seed/split to the earlier runs) ---")
    lines.append(f"unique sources       : {total_sources}")
    for name in SPLIT_NAMES:
        items = data[name]
        pos_list: List[Sample] = items["pos"]                        # type: ignore[assignment]
        lines.append(
            f"{name:<6} sources={len(sources_by_split[name]):>3}  "
            f"edges={len(pos_list):>3}  depth={depth_counter(pos_list)}"
        )
    lines.append(f"test source ids      : "
                 f"{sorted(graph.node_ids[i] for i in sources_by_split['test'])}")
    lines.append(f"val source ids       : "
                 f"{sorted(graph.node_ids[i] for i in sources_by_split['val'])}")
    lines.append("")
    lines.append("--- Closure target concentration ---")
    header = f"{'relation':<12} {'band':>5} {'rows':>5} {'distinct_targets':>17} {'distinct_sources':>17}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in relation_bands:
        lines.append(
            f"{str(row['relation']):<12} {str(row['band']):>5} {row['rows']:>5} "
            f"{row['distinct_targets']:>17} {row['distinct_sources']:>17}"
        )
    lines.append("  When distinct_targets is much smaller than rows, many sources share a")
    lines.append("  target; those shared targets are blocked as negatives for other sources.")
    lines.append("")
    lines.append("--- Hard-negative pools per positive ---")
    lines.append("hard = same relation + same depth band (2 / 3 / 4+), other source,")
    lines.append("       not direct/closure-related to the query source;")
    lines.append("medium = same relation, different band, same exclusion; easy = random.")
    for name in SPLIT_NAMES:
        lines.append(_pool_line(name, pool_summary[name]))
    lines.append("")
    lines.append("--- Dataset per split (positives + mixed negatives) ---")
    header = (f"{'split':<6} {'pos':>4} {'neg':>4} {'hard':>5} {'med':>4} "
              f"{'easy':>5} {'short_h':>7} {'short_m':>7} {'short_e':>7} "
              f"{'missing':>7} {'unique':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for name in SPLIT_NAMES:
        items = data[name]
        st = sampling_stats[name]
        lines.append(
            f"{name:<6} {len(items['pos']):>4} "
            f"{len(items['samples']) - len(items['pos']):>4} "
            f"{st.used_hard:>5} {st.used_medium:>4} {st.used_easy:>5} "
            f"{st.short_hard:>7} {st.short_medium:>7} {st.short_easy:>7} "
            f"{st.shortfall:>7} {st.unique_pairs:>6}"
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
    lines.append(f"freeze_gat_epochs   : {args.freeze_gat_epochs}")
    lines.append(f"neg_per_pos         : {args.neg_per_pos} "
                 f"({args.hard_ratio:.0%} hard / {args.medium_ratio:.0%} medium / "
                 f"{1.0 - args.hard_ratio - args.medium_ratio:.0%} easy requested)")
    lines.append("depth bands         : 2 / 3 / 4+")
    lines.append("optimizer           : Adam, ReduceLROnPlateau(factor=0.5, patience=10)")
    lines.append(f"loss                : BCEWithLogitsLoss(pos_weight={pos_weight:.2f})")
    lines.append("")
    lines.append("--- Training ---")
    lines.append(f"epochs_run          : {epochs_run}")
    lines.append(f"best_epoch          : {best_epoch} (val loss {_fmt(best_val_loss)})")
    lines.append(f"best_margin_epoch   : {best_margin_epoch} (val margin {best_val_margin:+.4f})")
    lines.append(f"early_stopped       : {stopped_early}")
    if margin_history:
        first = margin_history[0]
        last = margin_history[-1]
        lines.append(
            f"val margin curve    : first epoch {first[3]:+.4f} "
            f"(model {_fmt(first[1])} / prior {_fmt(first[2])}) -> "
            f"last epoch {last[3]:+.4f} "
            f"(model {_fmt(last[1])} / prior {_fmt(last[2])})"
        )
    lines.append("  Checkpoint selection is best *val loss*; the margin checkpoint is")
    lines.append("  saved separately as an oracle-selected diagnostic, not the primary model.")
    lines.append("")
    lines.append("--- Full-catalogue filtered ranking ---")
    header = (
        f"{'split':<6} {'sources':>7} {'pos':>4} {'neg':>4} {'AUC':>7} {'MRR':>7} "
        f"{'H@1':>7} {'H@3':>7} {'H@10':>7} {'P@10':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name in SPLIT_NAMES:
        m = metrics[name]
        lines.append(
            f"{name:<6} {len(sources_by_split[name]):>7} {m['n_pos']:>4} "
            f"{m['n_neg']:>4} {_fmt(m['auc']):>7} {_fmt(m['mrr']):>7} "
            f"{_fmt(m['hits@1']):>7} {_fmt(m['hits@3']):>7} "
            f"{_fmt(m['hits@10']):>7} {_fmt(m['p@10']):>7}"
        )
    lines.append("")
    lines.append("Target-prior baseline (source-independent mean score, per relation):")
    lines.append(f"  train MRR={_fmt(prior['train']['overall'])}  "
                 f"val MRR={_fmt(prior['val']['overall'])}  "
                 f"test MRR={_fmt(prior['test']['overall'])}")
    lines.append("Full-catalogue margin (per-source model MRR - target-prior MRR):")
    lines.append(
        f"  train {metrics['train']['mrr'] - prior['train']['overall']:+.4f}  "
        f"val {metrics['val']['mrr'] - prior['val']['overall']:+.4f}  "
        f"test {metrics['test']['mrr'] - prior['test']['overall']:+.4f}"
    )
    lines.append("  AUC here is computed on the sampled hard/medium/easy mix, so it is")
    lines.append("  not comparable to the random-negative AUC in earlier reports.")
    lines.append("")
    lines.append("--- Per-relation full-catalogue metrics ---")
    header = (f"{'split':<6} {'relation':<14} {'n':>4} {'MRR':>7} {'H@1':>7} "
              f"{'H@10':>7} {'prior':>7} {'margin':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for name in SPLIT_NAMES:
        for rel in sorted(metrics[name]["by_relation"]):          # type: ignore[union-attr]
            m = metrics[name]["by_relation"][rel]                 # type: ignore[index]
            p = prior[name]["by_relation"].get(rel, {"mrr": float("nan")})  # type: ignore[union-attr]
            lines.append(
                f"{name:<6} {rel:<14} {int(m['n']):>4} {_fmt(m['mrr']):>7} "
                f"{_fmt(m['hits@1']):>7} {_fmt(m['hits@10']):>7} "
                f"{_fmt(p.get('mrr', float('nan'))):>7} "
                f"{m['mrr'] - p.get('mrr', float('nan')):+.4f}"
            )
    tested_rels = sorted(metrics["test"]["by_relation"])           # type: ignore[union-attr]
    lines.append("  Test relations: " + (", ".join(tested_rels) if tested_rels else "none"))
    lines.append("  Relations with no depth>=2 test positive are omitted from the test")
    lines.append("  rows above; they are not scored as zero.")
    lines.append("")
    lines.append("--- Hard-negative protocol (rank against ALL eligible hard negatives) ---")
    header = (f"{'split':<6} {'pos':>4} {'hardMRR':>8} {'prior':>8} {'margin':>8} "
              f"{'Kmean':>6} {'n':>4} {'skip':>4} {'random':>7} {'>rand':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for name in SPLIT_NAMES:
        lines.append(_hard_row(name, hard_by_split[name], len(data[name]["pos"])))  # type: ignore[arg-type]
    lines.append("  random = mean 1/(K+1) under a uniformly random ordering; 'skip' counts")
    lines.append("  empty pools or filtered-out queries.")
    lines.append("")
    lines.append("--- Per-relation hard-negative metrics ---")
    lines.append(f"{'split':<6} {'relation':<14} {'n':>4} {'hardMRR':>8} "
                 f"{'prior':>8} {'margin':>8} {'K':>5}")
    lines.append("-" * 58)
    for name in SPLIT_NAMES:
        for rel, hm in sorted(hard_by_split[name].by_relation.items()):
            lines.append(
                f"{name:<6} {rel:<14} {int(hm['n']):>4} "
                f"{_fmt(hm['model_mrr']):>8} {_fmt(hm['prior_mrr']):>8} "
                f"{hm['margin']:+.4f} {hm['mean_k']:>5.1f}"
            )
    lines.append("")
    lines.append("--- Nontrivial hard-neg subsets (test) ---")
    lines.append(f"{'model':<22} {'subset':<10} {'n':>4} {'hardMRR':>8} "
                 f"{'prior':>8} {'margin':>8} {'K':>5}")
    lines.append("-" * 70)
    for label, hm in (("all", hard_by_split["test"]),
                      ("K>=2", hard_test_k2),
                      ("depth=2", hard_test_d2)):
        lines.append(
            f"{'relaware':<22} {label:<10} {hm.n_queries:>4} "
            f"{_fmt(hm.model_mrr):>8} {_fmt(hm.prior_mrr):>8} "
            f"{hm.margin:+.4f} {hm.mean_k:>5.1f}"
        )
    if baseline_hard_k2 is not None and baseline_hard_d2 is not None:
        for label, hm in (("K>=2", baseline_hard_k2), ("depth=2", baseline_hard_d2)):
            lines.append(
                f"{baseline_name:<22} {label:<10} {hm.n_queries:>4} "
                f"{_fmt(hm.model_mrr):>8} {_fmt(hm.prior_mrr):>8} "
                f"{hm.margin:+.4f} {hm.mean_k:>5.1f}"
            )
    lines.append("")
    lines.append("--- Test comparison on identical hard pools / test pairs ---")
    header = (f"{'model':<24} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} "
              f"{'AUC*':>7} {'hardMRR':>8} {'prior':>8} {'margin':>8} {'K':>5}")
    lines.append(header)
    lines.append("-" * len(header))
    if baseline_metrics is not None and baseline_hard is not None:
        lines.append(
            f"{baseline_name:<24} {_fmt(baseline_metrics['mrr']):>7} "
            f"{_fmt(baseline_metrics['hits@1']):>7} "
            f"{_fmt(baseline_metrics['hits@3']):>7} "
            f"{_fmt(baseline_metrics['hits@10']):>7} "
            f"{_fmt(baseline_metrics['auc']):>7} "
            f"{_fmt(baseline_hard.model_mrr):>8} "
            f"{_fmt(baseline_hard.prior_mrr):>8} "
            f"{baseline_hard.margin:+.4f} {baseline_hard.mean_k:>5.1f}"
        )
    else:
        lines.append(f"{baseline_name:<24} n/a (checkpoint missing or incompatible)")
    test_m = metrics["test"]
    test_h = hard_by_split["test"]
    lines.append(
        f"{'relaware (best val loss)':<24} {_fmt(test_m['mrr']):>7} "
        f"{_fmt(test_m['hits@1']):>7} {_fmt(test_m['hits@3']):>7} "
        f"{_fmt(test_m['hits@10']):>7} {_fmt(test_m['auc']):>7} "
        f"{_fmt(test_h.model_mrr):>8} {_fmt(test_h.prior_mrr):>8} "
        f"{test_h.margin:+.4f} {test_h.mean_k:>5.1f}"
    )
    if baseline_hard is not None:
        lines.append(f"  margin delta vs {baseline_name}: "
                     f"{test_h.margin - baseline_hard.margin:+.4f}")
        lines.append(f"  hardMRR delta vs {baseline_name}: "
                     f"{test_h.model_mrr - baseline_hard.model_mrr:+.4f}")
        lines.append(f"  full MRR delta vs {baseline_name}: "
                     f"{test_m['mrr'] - baseline_metrics['mrr']:+.4f}")  # type: ignore[index]
    lines.append("")
    lines.append("--- Test full-catalogue metrics by depth (per-positive ranks) ---")
    for depth, dm in sorted(test_m["by_depth"].items()):          # type: ignore[union-attr]
        lines.append(
            f"  depth={depth} (n={int(dm['n']):>2})  MRR={_fmt(dm['mrr'])}  "
            f"Hits@1={_fmt(dm['hits@1'])}  Hits@10={_fmt(dm['hits@10'])}"
        )
    lines.append("")
    lines.append("--- Test hard-negative metrics by depth ---")
    lines.append(f"{'depth':>5} {'n':>4} {'hardMRR':>8} {'prior':>8} "
                 f"{'margin':>8} {'Kmean':>6} {'random':>7}")
    for depth, dm in test_h.by_depth.items():
        lines.append(
            f"{depth:>5} {int(dm['n']):>4} {_fmt(dm['model_mrr']):>8} "
            f"{_fmt(dm['prior_mrr']):>8} {dm['margin']:+.4f} "
            f"{dm['mean_k']:>6.1f} {_fmt(dm['random_mrr']):>7}"
        )
    lines.append("")
    lines.append("--- Attention shift (layer-1, head-average) ---")
    lines.append(f"mean normalised entropy (init)    : {_fmt(init_attention['mean_entropy'])}")
    lines.append(f"mean normalised entropy (trained) : {_fmt(final_attention['mean_entropy'])}")
    lines.append("  (1.000 = uniform, lower = more selective)")
    lines.append("attention @ 猫 (init):")
    for alpha, name, relation in init_attention["target_top"]:
        lines.append(f"  {name:<16} {relation:<16} {alpha:.4f}")
    lines.append("attention @ 猫 (trained):")
    for alpha, name, relation in final_attention["target_top"]:
        lines.append(f"  {name:<16} {relation:<16} {alpha:.4f}")
    lines.append("")
    lines.append("--- Attention weights by relation (layer-1, head-average, top-3) ---")
    for group in attention_rel:
        lines.append(f"{group['relation']} ({group['n_edges']} message edges)")
        for alpha, sem_src, sem_dst, marker in group["top"]:
            lines.append(
                f"  {graph.labels[sem_src]}({graph.node_ids[sem_src]}) "
                f"--{group['relation']}--> "
                f"{graph.labels[sem_dst]}({graph.node_ids[sem_dst]})  "
                f"alpha={alpha:.4f}  [{marker} message]"
            )
    lines.append("")
    lines.append(f"--- Qualitative: top-5 targets for「猫」(sense 2, split={cat_split}) ---")
    for i, item in enumerate(qualitative, 1):
        extra = ""
        if item["closure"]:
            rel, depth = item["closure"]
            extra = f"  [closure: {rel}, depth {depth}]"
        lines.append(f"  {i}. {item['label']}({item['sid']})  "
                     f"score={item['score']:.4f}{extra}")
    if target_ranks:
        lines.append("  Known closure targets and their ranks (per-relation head):")
        for label, relation, depth, rank in target_ranks:
            lines.append(f"    {label:<14} {relation} depth={depth}  rank=#{rank}")
    if cat_hard is not None:
        lines.append(f"  猫->生物 hard-pool rank=#{cat_hard['hard_rank']} "
                     f"(prior #{cat_hard['prior_rank']}, K={cat_hard['k']})")
    lines.append("")
    lines.append("--- Expected-outcome check ---")
    test_rel = metrics["test"]["by_relation"]                    # type: ignore[index]
    lines.append(f"Pooled test MRR                     : {_fmt(test_m['mrr'])}")
    for rel in sorted(test_rel):
        fm = test_rel[rel]
        lines.append(f"  {rel:<14} n={int(fm['n']):>2} MRR={_fmt(fm['mrr'])} "
                     f"H@1={_fmt(fm['hits@1'])}")
    lines.append("  Each row uses the query relation's own effective head, so a trained")
    lines.append("  W_r can be read separately from the shared fallback. Relations with")
    lines.append("  no depth>=2 test positive are omitted, not reported as zero.")
    cat_full = next((r for _, rel, _, r in target_ranks if rel == "is_a"), None)
    lines.append(f"猫->生物 full-catalogue rank       : "
                 f"#{cat_full if cat_full is not None else 'n/a'}")
    lines.append(f"Hard-neg MRR >= 0.97               : {_fmt(test_h.model_mrr)} -> "
                 f"{'PASS' if test_h.model_mrr >= 0.97 else 'FAIL'}")
    if baseline_hard is not None:
        d_margin = test_h.margin - baseline_hard.margin
        lines.append(f"Margin vs {baseline_name:<22}: {d_margin:+.4f} "
                     f"({'improved' if d_margin > 0 else 'not improved'})")
    else:
        lines.append("Margin vs baseline                 : n/a (checkpoint incompatible "
                     "with this graph)")
        lines.append("  The comparison checkpoint was trained on a different node set, so")
        lines.append("  absolute per-relation MRR is the primary read for this run.")
    lines.append("")
    lines.append("--- Verdict ---")
    n_test_rel = len(test_rel)
    if baseline_hard is not None:
        if test_h.model_mrr >= 0.97 and test_h.margin > baseline_hard.margin:
            lines.append("The relation-aware head preserves hard-neg retrieval and improves")
            lines.append("the margin over the hard-negative baseline on these test pairs.")
        elif test_h.model_mrr >= 0.97:
            lines.append("Hard-neg retrieval is preserved, but the margin does not improve.")
            if n_test_rel > 1:
                lines.append(f"The test split covers {n_test_rel} relations; see the")
                lines.append("per-relation table for which trained heads carry the signal.")
            else:
                lines.append("Only one test relation is present, so relation diversity")
                lines.append("cannot be evaluated on this split.")
        else:
            lines.append("Hard-neg retrieval regressed below 0.97. The per-relation table")
            lines.append("shows whether the regression is concentrated in specific heads.")
    else:
        lines.append("No compatible baseline checkpoint for this graph; this run reports")
        lines.append("absolute metrics only. With a source-disjoint split and "
                     f"{n_test_rel} test relation(s),")
        lines.append("per-relation MRR is the primary read.")
    lines.append("")
    lines.append("--- Caveats ---")
    train_rel_counts = Counter(p.relation for p in train["pos"])   # type: ignore[arg-type]
    dedicated_rels = sorted(r for r, c in train_rel_counts.items() if c >= MIN_DEDICATED)
    gated_rels = sorted(r for r, c in train_rel_counts.items() if 0 < c < MIN_DEDICATED)
    n_unseen = len(graph.relation_names) - len(dedicated_rels) - len(gated_rels)
    lines.append(f"* {len(graph.relation_names)} relation types: {len(dedicated_rels)} "
                 f"dedicated ({', '.join(dedicated_rels) or 'none'}); "
                 f"{len(gated_rels)} gated ({', '.join(gated_rels) or 'none'}); "
                 f"{n_unseen} use W_shared.")
    lines.append("* A relation needs >= "
                 f"{MIN_DEDICATED} train positives for a dedicated W_r. Relations")
    lines.append("  with no depth>=2 positive fall back to W_shared and are omitted from")
    lines.append("  the test per-relation rows rather than scored as zero.")
    if gated_rels:
        lines.append(f"* Gated relation(s) {', '.join(gated_rels)} have fewer than")
        lines.append(f"  {MIN_DEDICATED} train positives, so their W_r/W_shared blend is a")
        lines.append("  low-sample fit and may not be stable.")
    lines.append("* Freezing the GAT for the first epochs trains the heads on fixed random")
    lines.append("  embeddings; this can overfit and is reported, not silently tuned.")
    depth_k = ", ".join(
        f"depth {depth} K={dm['mean_k']:.1f}"
        for depth, dm in sorted(test_h.by_depth.items())
    )
    if depth_k:
        lines.append(f"* Test hard-pool sizes: {depth_k}. Small pools make the")
        lines.append("  hard-pool MRR coarse, and most hard negatives are cross-branch hubs.")
    lines.append("* Source-disjoint is still transductive: test-source nodes appear in the")
    lines.append("  message graph and their embeddings are updated during training.")
    lines.append(f"* Test has {len(sources_by_split['test'])} sources / "
                 f"{len(data['test']['pos'])} positives; small deltas are not decisive.")
    lines.append("* AUC is on the sampled hard/medium/easy mix, not the random-negative AUC")
    lines.append("  used in the earlier random-split report.")
    lines.append("")
    lines.append("--- Reproduction ---")
    lines.append("  python logos_transformer/train_closure_relaware.py --epochs 5   # smoke")
    lines.append("  python logos_transformer/train_closure_relaware.py              # full")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def _load_state(path: pathlib.Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def load_relaware_checkpoint(
    path: pathlib.Path,
    graph: RelationGraph,
    args: argparse.Namespace,
    relation_counts: Dict[str, int],
) -> Tuple[Optional[RelationAwareClosurePredictor], Optional[str]]:
    if not path.exists():
        return None, f"not found: {path}"
    try:
        base = build_model(
            graph,
            hidden_dim=args.hidden,
            num_heads=args.heads,
            num_layers=args.layers,
            dropout=0.1,
            seed=args.seed,
        )
        model = RelationAwareClosurePredictor(
            base, args.hidden, graph.relation_names, relation_counts
        )
        model.load_state_dict(_load_state(path))
        model.eval()
        return model, None
    except Exception as exc:  # pragma: no cover - corrupt/incompatible ckpt
        return None, f"{type(exc).__name__}: {exc}"


@torch.no_grad()
def loss_acc(
    model: RelationAwareClosurePredictor,
    h: torch.Tensor,
    samples: Sequence[Sample],
    criterion: nn.Module,
) -> Tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    logits = pair_logits(model, h, samples)
    labels = labels_tensor(samples)
    loss = float(criterion(logits, labels))
    acc = float(((logits > 0).float() == labels).float().mean())
    return loss, acc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train/evaluate the Logos GAT with a per-relation scoring head "
                    "and hard negatives (source-disjoint split)."
    )
    parser.add_argument("--db", type=pathlib.Path, default=root / "logos.db")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--neg-per-pos", type=int, default=6,
                        help="total negatives per positive (default 6: 3 hard / 2 medium / 1 easy)")
    parser.add_argument("--hard-ratio", type=float, default=0.5)
    parser.add_argument("--medium-ratio", type=float, default=0.3)
    parser.add_argument("--freeze-gat-epochs", type=int, default=10,
                        help="freeze the GAT encoder for the first N epochs (0 = never)")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-checkpoint", type=pathlib.Path,
                        default=root / DEFAULT_COMPARE_CHECKPOINT,
                        help="hard-negative checkpoint evaluated on the same test pairs/pools")
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs 必须 >= 1")
    if args.hidden < 1 or args.heads < 1 or args.hidden % args.heads != 0:
        parser.error("--hidden 必须能被 --heads 整除")
    if args.layers < 1:
        parser.error("--layers 必须 >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size 必须 >= 1")
    if args.neg_per_pos < 1:
        parser.error("--neg-per-pos 必须 >= 1")
    if not (0.0 <= args.hard_ratio <= 1.0):
        parser.error("--hard-ratio 必须在 [0, 1]")
    if not (0.0 <= args.medium_ratio <= 1.0):
        parser.error("--medium-ratio 必须在 [0, 1]")
    if args.hard_ratio + args.medium_ratio > 1.0 + 1e-9:
        parser.error("--hard-ratio + --medium-ratio 不能超过 1")
    if args.freeze_gat_epochs < 0:
        parser.error("--freeze-gat-epochs 必须 >= 0")
    if args.patience < 1:
        parser.error("--patience 必须 >= 1")
    if args.seed < 0:
        parser.error("--seed 必须 >= 0")
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
    compare_path = pathlib.Path(args.compare_checkpoint).expanduser().resolve()

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1
    try:
        graph = load_graph(db_path, include_closure=False, verbose=False)
    except Exception as exc:  # pragma: no cover - malformed DB
        print(f"Failed to load graph: {exc}", file=sys.stderr)
        return 1

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
        deep_total = conn.execute(
            "SELECT COUNT(*) FROM closure WHERE depth >= 2"
        ).fetchone()[0]
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
            "drawing conclusions.", file=sys.stderr,
        )
    if args.seed != 42 and compare_path.exists():
        print(
            "[warning] --seed differs from 42; the comparison checkpoint was trained "
            "on the seed-42 split and is not directly comparable.", file=sys.stderr,
        )

    positives_by_split, sources_by_split, mode = split_by_source(
        positives, seed=args.seed
    )
    pools = build_negative_pools(positives, facts)
    pool_summary = {
        name: summarize_pools(pools, positives_by_split[name])
        for name in SPLIT_NAMES
    }
    relation_bands = relation_band_summary(positives)

    n_hard, n_medium, n_easy = _split_ratios(
        args.neg_per_pos, args.hard_ratio, args.medium_ratio
    )
    rng = random.Random(args.seed + 7)
    data: Dict[str, Dict[str, object]] = {}
    sampling_stats: Dict[str, SamplingStats] = {}
    for name in SPLIT_NAMES:
        pos_list = positives_by_split[name]
        neg_list, st = sample_hardneg(
            pos_list, graph, facts, pools, n_hard, n_medium, n_easy, rng
        )
        data[name] = {
            "pos": pos_list,
            "neg": neg_list,
            "samples": list(pos_list) + neg_list,
        }
        sampling_stats[name] = st

    train_samples: List[Sample] = data["train"]["samples"]            # type: ignore[assignment]
    val_samples: List[Sample] = data["val"]["samples"]                # type: ignore[assignment]
    train_pos = sum(1 for s in train_samples if s.label == 1)
    train_neg = len(train_samples) - train_pos
    pos_weight = train_neg / max(train_pos, 1)
    train_positives = [s for s in train_samples if s.label == 1]
    relation_counts = dict(Counter(s.relation for s in train_positives))
    active_relations = sorted({p.relation for p in positives})

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
    model = RelationAwareClosurePredictor(
        base_model, args.hidden, graph.relation_names, relation_counts
    )
    head_summary = model.head_summary()

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 冻结 GAT：只训练关系头，避免编码器过早被关系头带偏。
    freeze_active = args.freeze_gat_epochs > 0
    if freeze_active:
        for param in model.gat.parameters():
            param.requires_grad = False

    print("=" * 108)
    print("Logos Transformer — relation-aware closure prediction (source-disjoint)")
    print("=" * 108)
    print(f"db={db_path}")
    print(f"graph: {graph.stats['nodes']} nodes / {graph.stats['semantic_edges']} "
          f"direct edges / {graph.stats['message_edges']} message edges")
    print(f"split mode: {mode} | unique sources: {total_sources}")
    for name in SPLIT_NAMES:
        pos_list: List[Sample] = data[name]["pos"]                   # type: ignore[assignment]
        samples: List[Sample] = data[name]["samples"]                # type: ignore[assignment]
        depths = dict(sorted(Counter(s.depth for s in pos_list).items()))
        st = sampling_stats[name]
        print(
            f"  {name:<5} sources={len(sources_by_split[name]):>3}  "
            f"edges={len(pos_list):>3}  depth={depths}  samples={len(samples):>3} "
            f"({len(pos_list)} pos / {len(samples) - len(pos_list)} neg)"
        )
        print(f"        neg mix: hard={st.used_hard} medium={st.used_medium} "
              f"easy={st.used_easy} | requested hard={st.requested_hard} "
              f"| short_h={st.short_hard} zero_h={st.zero_hard} "
              f"missing={st.shortfall}")
    n_params = sum(p.numel() for p in model.parameters())
    head_params = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("W_r") or n.startswith("W_shared") or n.startswith("share_logit")
    )
    print(f"train pos_weight={pos_weight:.2f} | "
          f"model={args.layers}x{args.heads} heads, hidden={args.hidden}, "
          f"params={n_params:,} (relation heads {head_params:,})")
    print(f"relation heads: {head_summary['n_relation_types']} types | "
          f"dedicated={head_summary['dedicated']} | "
          f"gated={head_summary['gated']} | "
          f"unseen->shared={len(head_summary['unseen'])}")
    print(f"freeze_gat_epochs={args.freeze_gat_epochs} | "
          f"hard_ratio={args.hard_ratio:.2f} medium_ratio={args.medium_ratio:.2f} "
          f"-> per positive: {n_hard} hard / {n_medium} medium / {n_easy} easy")
    print("-" * 108)
    for name in SPLIT_NAMES:
        print(_pool_line(name, pool_summary[name]))
    if sampling_stats["train"].short_hard:
        st = sampling_stats["train"]
        print(
            f"[warning] hard pool exhausted for {st.short_hard}/{st.n_pos} train "
            f"positives ({st.zero_hard} empty); fallback used: "
            f"medium={st.used_medium} easy={st.used_easy}, "
            f"{st.shortfall} requested negatives could not be filled.",
            file=sys.stderr,
        )
    print("-" * 108)

    cat_idx = graph.id_to_index.get(2, 0)
    cat_split = next(
        (name for name in SPLIT_NAMES if cat_idx in sources_by_split[name]), "—"
    )
    init_attention = attention_stats(model, graph, cat_idx)
    init_h, _, _ = encode_and_score(model, graph, active_relations)
    init_loss, init_acc = loss_acc(model, init_h, train_samples, criterion)
    print(f"init: train loss={init_loss:.4f} acc={init_acc:.4f} | "
          f"attention entropy={init_attention['mean_entropy']:.4f}")

    best_val = float("inf")
    best_epoch = 0
    best_margin = float("-inf")
    best_margin_epoch = 0
    stale = 0
    epochs_run = 0
    margin_history: List[Tuple[int, float, float, float, float]] = []

    batch_rng = random.Random(args.seed + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "closure_gat_relaware_best.pt"
    margin_checkpoint_path = output_dir / "closure_gat_relaware_best_margin.pt"

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        if freeze_active and epoch == args.freeze_gat_epochs + 1:
            for param in model.gat.parameters():
                param.requires_grad = True
            freeze_active = False
            print(f"[unfreeze] GAT encoder unfrozen at epoch {epoch}")

        model.train()
        if freeze_active:
            model.gat.eval()      # 冻结权重之外，也关闭 GAT dropout

        order = list(train_samples)
        batch_rng.shuffle(order)
        total_loss = 0.0
        total = 0
        correct = 0
        for i in range(0, len(order), args.batch_size):
            batch = order[i:i + args.batch_size]
            optimizer.zero_grad()
            embeddings, _ = model.encode(graph.edge_index, graph.edge_type)
            logits = model.score_pairs(
                embeddings,
                pairs_tensor(batch),
                [s.relation for s in batch],
            )
            labels = labels_tensor(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total += len(batch)
            correct += int(((logits > 0).float() == labels).detach().sum())

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        with torch.no_grad():
            val_h, _ = model.encode(graph.edge_index, graph.edge_type)
            val_loss, val_acc = loss_acc(model, val_h, val_samples, criterion)
            val_scores = {
                rel: model.score_all(val_h, rel) for rel in active_relations
            }
        if math.isnan(val_loss):
            val_loss = train_loss
        scheduler.step(val_loss)

        val_hard = hard_metrics(
            val_scores, pools, data["val"]["pos"]                   # type: ignore[arg-type]
        )
        if val_hard.n_queries:
            margin_history.append(
                (epoch, val_hard.model_mrr, val_hard.prior_mrr,
                 val_hard.margin, val_loss)
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1

        if val_hard.n_queries and val_hard.margin > best_margin + 1e-6:
            best_margin = val_hard.margin
            best_margin_epoch = epoch
            torch.save(model.state_dict(), margin_checkpoint_path)

        if val_hard.n_queries:
            hard_txt = (
                f"hard-val MRR={_fmt(val_hard.model_mrr)} "
                f"prior={_fmt(val_hard.prior_mrr)} "
                f"margin={val_hard.margin:+.4f} "
                f"(n={val_hard.n_queries}, K={val_hard.mean_k:.1f})"
            )
        else:
            hard_txt = "hard-val MRR=n/a (no eligible queries)"
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:>3}/{args.epochs} | train loss={train_loss:.4f} "
              f"acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"{hard_txt} | lr={lr:.2e}")

        if stale >= args.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, "
                  f"val loss {best_val:.4f}; best val margin "
                  f"{best_margin:+.4f} at epoch {best_margin_epoch})")
            break

    stopped_early = stale >= args.patience and epochs_run < args.epochs
    if checkpoint_path.exists():
        model.load_state_dict(_load_state(checkpoint_path))
    model.eval()

    h, scores_by_rel, _ = encode_and_score(model, graph, active_relations)
    metrics: Dict[str, Dict[str, object]] = {}
    for name in SPLIT_NAMES:
        metrics[name] = eval_bucket_relaware(
            model, graph, facts, data[name]["samples"], criterion,  # type: ignore[arg-type]
            h, scores_by_rel,
        )
    prior = {
        name: target_prior_relaware(
            scores_by_rel, graph, facts, data[name]["pos"]         # type: ignore[arg-type]
        )
        for name in SPLIT_NAMES
    }
    hard_by_split = {
        name: hard_metrics(
            scores_by_rel, pools, data[name]["pos"]                # type: ignore[arg-type]
        )
        for name in SPLIT_NAMES
    }
    hard_test_k2 = hard_metrics(
        scores_by_rel, pools, data["test"]["pos"], min_k=2         # type: ignore[arg-type]
    )
    hard_test_d2 = hard_metrics(
        scores_by_rel, pools, data["test"]["pos"], depths={2}      # type: ignore[arg-type]
    )

    # Baseline: hard-negative checkpoint on the same pairs and pools.
    baseline_name = "hardneg ckpt"
    baseline_metrics: Optional[Dict[str, object]] = None
    baseline_hard: Optional[HardMetricsRel] = None
    baseline_hard_k2: Optional[HardMetricsRel] = None
    baseline_hard_d2: Optional[HardMetricsRel] = None
    baseline_prior_test: Optional[float] = None
    baseline_rel_metrics: Optional[Dict[str, object]] = None
    baseline_rel_hard: Optional[HardMetricsRel] = None
    baseline_model, baseline_err = load_closure_checkpoint(compare_path, graph, args)
    if baseline_model is not None:
        baseline_model.eval()
        baseline_metrics = eval_bucket(
            baseline_model, graph, facts, data["test"]["samples"], criterion  # type: ignore[arg-type]
        )
        baseline_prior_test = target_prior_mrr(
            baseline_model, graph, facts, data["test"]["pos"]              # type: ignore[arg-type]
        )
        baseline_scores = compute_scores(baseline_model, graph)
        baseline_by_rel = {rel: baseline_scores for rel in active_relations}
        baseline_hard = hard_metrics(
            baseline_by_rel, pools, data["test"]["pos"]                 # type: ignore[arg-type]
        )
        baseline_hard_k2 = hard_metrics(
            baseline_by_rel, pools, data["test"]["pos"], min_k=2        # type: ignore[arg-type]
        )
        baseline_hard_d2 = hard_metrics(
            baseline_by_rel, pools, data["test"]["pos"], depths={2}     # type: ignore[arg-type]
        )
        baseline_rel_metrics = filtered_metrics_relaware(
            baseline_by_rel, graph, facts, data["test"]["pos"]          # type: ignore[arg-type]
        )
        baseline_rel_hard = baseline_hard
    else:
        print(f"[warning] comparison checkpoint skipped: {baseline_err}", file=sys.stderr)

    final_attention = attention_stats(model, graph, cat_idx)
    attention_rel = attention_by_relation(model, graph, top_k=3)
    qualitative = qualitative_examples_relaware(
        model, graph, facts, h, scores_by_rel, source_sid=2, relation="is_a", top_k=5
    )
    target_ranks = closure_target_ranks_relaware(
        graph, facts, scores_by_rel, source_sid=2
    )
    cat_pos = next(
        (p for p in data["val"]["pos"]                         # type: ignore[union-attr]
         if p.src_sid == 2 and p.relation == "is_a"),
        None,
    )
    cat_hard = hard_rank_for_positive(scores_by_rel, pools, cat_pos) if cat_pos else None

    # Console summary.
    print("-" * 108)
    print("full-catalogue test metrics + hard-negative protocol")
    print(f"{'model':<24} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} "
          f"{'AUC*':>7} {'hardMRR':>8} {'prior':>8} {'margin':>8} {'K':>5}")
    if baseline_metrics is not None and baseline_hard is not None:
        print(
            f"{baseline_name:<24} {_fmt(baseline_metrics['mrr']):>7} "
            f"{_fmt(baseline_metrics['hits@1']):>7} "
            f"{_fmt(baseline_metrics['hits@3']):>7} "
            f"{_fmt(baseline_metrics['hits@10']):>7} "
            f"{_fmt(baseline_metrics['auc']):>7} "
            f"{_fmt(baseline_hard.model_mrr):>8} "
            f"{_fmt(baseline_hard.prior_mrr):>8} "
            f"{baseline_hard.margin:+.4f} {baseline_hard.mean_k:>5.1f}"
        )
    test_m = metrics["test"]
    test_h = hard_by_split["test"]
    print(
        f"{'relaware (best val loss)':<24} {_fmt(test_m['mrr']):>7} "
        f"{_fmt(test_m['hits@1']):>7} {_fmt(test_m['hits@3']):>7} "
        f"{_fmt(test_m['hits@10']):>7} {_fmt(test_m['auc']):>7} "
        f"{_fmt(test_h.model_mrr):>8} {_fmt(test_h.prior_mrr):>8} "
        f"{test_h.margin:+.4f} {test_h.mean_k:>5.1f}"
    )
    print(f"best epoch (val loss)={best_epoch} | best val margin="
          f"{best_margin:+.4f} @ epoch {best_margin_epoch}")
    print(f"test hard-neg K>=2: n={hard_test_k2.n_queries} "
          f"hardMRR={_fmt(hard_test_k2.model_mrr)} "
          f"prior={_fmt(hard_test_k2.prior_mrr)} "
          f"margin={hard_test_k2.margin:+.4f} | "
          f"depth2: n={hard_test_d2.n_queries} "
          f"hardMRR={_fmt(hard_test_d2.model_mrr)} "
          f"margin={hard_test_d2.margin:+.4f}")
    print("-" * 108)
    print("per-relation test metrics (full catalogue / hard pool):")
    for rel in sorted(test_m["by_relation"]):                    # type: ignore[union-attr]
        fm = test_m["by_relation"][rel]                           # type: ignore[index]
        hm = test_h.by_relation.get(rel, {})
        print(f"  {rel:<14} n={int(fm['n']):>2} fullMRR={_fmt(fm['mrr'])} "
              f"H@1={_fmt(fm['hits@1'])} | hardMRR={_fmt(hm.get('model_mrr'))} "
              f"margin={hm.get('margin', float('nan')):+.4f}")
    print("-" * 108)
    print("attention by relation (layer-1, head-average, top-3):")
    for group in attention_rel:
        print(f"  {group['relation']} ({group['n_edges']} message edges)")
        for alpha, sem_src, sem_dst, marker in group["top"]:
            print(f"    {graph.labels[sem_src]} --{group['relation']}--> "
                  f"{graph.labels[sem_dst]}  alpha={alpha:.4f} [{marker}]")
    print("-" * 108)
    print("top-5 multi-hop targets for 猫 (is_a head):")
    for i, item in enumerate(qualitative, 1):
        flag = " [closure]" if item["closure"] else ""
        print(f"  {i}. {item['label']}({item['sid']}) score={item['score']:.4f}{flag}")
    if cat_hard is not None:
        print(f"猫->生物 hard rank=#{cat_hard['hard_rank']} "
              f"(prior #{cat_hard['prior_rank']}, K={cat_hard['k']})")

    report = build_report(
        args=args,
        graph=graph,
        total_sources=total_sources,
        mode=mode,
        sources_by_split=sources_by_split,
        data=data,
        pool_summary=pool_summary,
        sampling_stats=sampling_stats,
        relation_bands=relation_bands,
        relation_rows=_relation_count_rows(graph, train_positives),
        best_epoch=best_epoch,
        best_val_loss=best_val,
        best_margin_epoch=best_margin_epoch,
        best_val_margin=best_margin,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
        margin_history=margin_history,
        metrics=metrics,
        prior=prior,
        hard_by_split=hard_by_split,
        hard_test_k2=hard_test_k2,
        hard_test_d2=hard_test_d2,
        baseline_name=baseline_name,
        baseline_metrics=baseline_metrics,
        baseline_hard=baseline_hard,
        baseline_hard_k2=baseline_hard_k2,
        baseline_hard_d2=baseline_hard_d2,
        baseline_prior_test=baseline_prior_test,
        baseline_rel_metrics=baseline_rel_metrics,
        baseline_rel_hard=baseline_rel_hard,
        init_attention=init_attention,
        final_attention=final_attention,
        attention_rel=attention_rel,
        cat_split=cat_split,
        qualitative=qualitative,
        target_ranks=target_ranks,
        cat_hard=cat_hard,
        checkpoint=checkpoint_path,
        margin_checkpoint=margin_checkpoint_path,
    )
    report_path = output_dir / "training_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print("-" * 108)
    print(f"checkpoint : {checkpoint_path}")
    print(f"report     : {report_path}")
    print(f"margin ckpt: {margin_checkpoint_path}  [diagnostic only]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



