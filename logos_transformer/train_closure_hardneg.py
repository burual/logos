#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard-negative closure prediction (source-disjoint split).

Motivation
----------
``train_closure_disjoint.py`` showed that random closure-free negatives are too
easy: a source-independent **target prior** already reproduces ~92% of the
filtered MRR.  The model never has to decide *which* ancestor-like hub belongs
to *this* source, because the negatives (table, run, ...) are unrelated to the
query.

This script keeps the same graph, model, source-disjoint split and
hyperparameters, but builds three difficulty tiers of negatives per positive:

* **hard**   — targets of *other sources* with the same relation and same depth
  band (2 / 3 / 4+), excluding anything reachable from the query source;
* **medium** — targets of other sources with the same relation but a different
  depth band, still unreachable from the query source;
* **easy**   — uniformly random nodes that are neither direct neighbours nor
  closure-related (the old protocol).

Default mix is 50% hard / 30% medium / 20% easy.  If a pool is too small the
sampler falls back hard -> medium -> easy and prints an aggregate warning
(the closure targets of this 170-node graph are concentrated hubs, so this
happens often; see the pool statistics in the report).

Diagnostics
-----------
After every epoch the script ranks each **validation** positive against *all*
of its eligible hard negatives (not just the 3 sampled ones) and computes:

    model MRR, target-prior MRR, margin = model MRR - prior MRR

The target prior ranks targets by their mean score over all sources, i.e. the
source-independent baseline from §9.8.  A positive margin means the model adds
source-conditioned signal beyond "which node looks like an ancestor".

Evaluation reports the usual filtered full-catalogue MRR / Hits / AUC **and**
the hard-negative MRR (each positive ranked only against its hard pool), plus
the same protocol applied to the source-disjoint checkpoint for comparison.

Usage
-----
    python logos_transformer/train_closure_hardneg.py --epochs 5   # smoke
    python logos_transformer/train_closure_hardneg.py              # full run

Outputs
-------
    <output-dir>/closure_gat_hardneg_best.pt          best val loss (primary)
    <output-dir>/closure_gat_hardneg_best_margin.pt   best val margin (diagnostic)
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
import statistics
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

# Reuse the encoder, data containers and metrics from the existing experiments.
# 直接运行本脚本时 sys.path[0] 是 logos_transformer/；作为包导入时走 fallback。
try:
    from prototype_gat import RelationGraph, build_model, load_graph  # type: ignore
    from train_closure import (  # type: ignore
        ClosurePredictor,
        Facts,
        Sample,
        _fmt,
        attention_stats,
        closure_target_ranks,
        evaluate,
        filtered_metrics,
        labels_tensor,
        pairs_tensor,
        qualitative_examples,
        roc_auc,
    )
    from train_closure_disjoint import (  # type: ignore
        eval_bucket,
        load_positives,
        split_by_source,
        target_prior_mrr,
    )
except ImportError:  # pragma: no cover
    from logos_transformer.prototype_gat import (  # type: ignore
        RelationGraph,
        build_model,
        load_graph,
    )
    from logos_transformer.train_closure import (  # type: ignore
        ClosurePredictor,
        Facts,
        Sample,
        _fmt,
        attention_stats,
        closure_target_ranks,
        evaluate,
        filtered_metrics,
        labels_tensor,
        pairs_tensor,
        qualitative_examples,
        roc_auc,
    )
    from logos_transformer.train_closure_disjoint import (  # type: ignore
        eval_bucket,
        load_positives,
        split_by_source,
        target_prior_mrr,
    )

HARDNEG_VERSION = "logos-gat-closure-hardneg-v0.1"
DEFAULT_OUTPUT_DIR = "./logos_transformer/checkpoints_hardneg"
DEFAULT_COMPARE_CHECKPOINT = (
    "./logos_transformer/checkpoints_disjoint/closure_gat_disjoint_best.pt"
)
SPLIT_NAMES = ("train", "val", "test")
PoolKey = Tuple[int, int, str, int]


# --------------------------------------------------------------------------- #
# Negative pools
# --------------------------------------------------------------------------- #
def depth_band(depth: int) -> str:
    """Depth band used to match hard negatives.

    Exact depth gives 31/241 empty pools (depth 6 has a single positive); the
    bands 2 / 3 / 4+ reduce this while keeping the structural meaning.
    """
    return str(depth) if depth <= 3 else "4+"


def pool_key(sample: Sample) -> PoolKey:
    return (sample.src, sample.dst, sample.relation, sample.depth)


@dataclasses.dataclass
class NegativePools:
    """Per-positive candidate lists, keyed by (src, dst, relation, depth)."""

    hard: Dict[PoolKey, List[int]]
    medium: Dict[PoolKey, List[int]]


def build_negative_pools(
    positives: Sequence[Sample],
    facts: Facts,
) -> NegativePools:
    """Build hard/medium candidate pools for every positive.

    A candidate is a *target* of another source with the same relation.  It is
    rejected when it is the positive's own target, the query source itself, or
    any direct/closure neighbour of the query source (``facts.blocked_pairs``
    already contains both directions, so "reachable" is checked symmetrically).

    hard   = same depth band
    medium = same relation, any depth, minus the hard pool (tiers stay disjoint)
    """
    by_rel_band: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    by_rel: Dict[str, List[Sample]] = defaultdict(list)
    for sample in positives:
        by_rel_band[(sample.relation, depth_band(sample.depth))].append(sample)
        by_rel[sample.relation].append(sample)

    hard: Dict[PoolKey, List[int]] = {}
    medium: Dict[PoolKey, List[int]] = {}
    for sample in positives:
        key = pool_key(sample)
        hard_targets = {
            other.dst
            for other in by_rel_band[(sample.relation, depth_band(sample.depth))]
            if other.src != sample.src
            and other.dst != sample.dst
            and other.dst != sample.src
            and (sample.src, other.dst) not in facts.blocked_pairs
        }
        medium_targets = {
            other.dst
            for other in by_rel[sample.relation]
            if other.src != sample.src
            and other.dst != sample.dst
            and other.dst != sample.src
            and other.dst not in hard_targets
            and (sample.src, other.dst) not in facts.blocked_pairs
        }
        hard[key] = sorted(hard_targets)
        medium[key] = sorted(medium_targets)
    return NegativePools(hard=hard, medium=medium)


def summarize_pools(
    pools: NegativePools,
    positives: Sequence[Sample],
) -> Dict[str, object]:
    hard_ks = [len(pools.hard[pool_key(p)]) for p in positives]
    medium_ks = [len(pools.medium[pool_key(p)]) for p in positives]

    def _stat(values: Sequence[int]) -> Dict[str, float]:
        if not values:
            return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
        return {
            "min": min(values),
            "median": float(statistics.median(values)),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    return {
        "n": len(positives),
        "hard": {**_stat(hard_ks), "zero": sum(1 for k in hard_ks if k == 0)},
        "medium": {**_stat(medium_ks), "zero": sum(1 for k in medium_ks if k == 0)},
        "hard_k_dist": dict(sorted(Counter(hard_ks).items())),
    }


def relation_band_summary(positives: Sequence[Sample]) -> List[Dict[str, object]]:
    """Rows / distinct targets / distinct sources per (relation, band).

    This table explains why hard pools are tiny: ``is_a`` depth-2 has 102 rows
    but only 14 distinct targets, so most other sources share the same hubs.
    """
    groups: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for sample in positives:
        groups[(sample.relation, depth_band(sample.depth))].append(sample)
    rows: List[Dict[str, object]] = []
    for (relation, band), items in sorted(groups.items()):
        rows.append(
            {
                "relation": relation,
                "band": band,
                "rows": len(items),
                "distinct_targets": len({s.dst for s in items}),
                "distinct_sources": len({s.src for s in items}),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Negative sampling
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SamplingStats:
    n_pos: int = 0
    requested_hard: int = 0
    requested_medium: int = 0
    requested_easy: int = 0
    used_hard: int = 0
    used_medium: int = 0
    used_easy: int = 0
    short_hard: int = 0          # positives where the hard pool was too small
    short_medium: int = 0        # ... after the hard fallback
    short_easy: int = 0          # ... even easy candidates ran out
    zero_hard: int = 0           # positives with an empty hard pool
    shortfall: int = 0           # requested negatives that could not be filled
    unique_pairs: int = 0

    @property
    def used_total(self) -> int:
        return self.used_hard + self.used_medium + self.used_easy


def sample_hardneg(
    positives: Sequence[Sample],
    graph: RelationGraph,
    facts: Facts,
    pools: NegativePools,
    n_hard: int,
    n_medium: int,
    n_easy: int,
    rng: random.Random,
) -> Tuple[List[Sample], SamplingStats]:
    """Sample hard / medium / easy negatives for each positive.

    Fallback order: hard -> medium -> easy.  A positive never gets the same
    target twice.  The same ``(source, target)`` pair may appear for two
    different positives of one source; that is intentional (each closure query
    is a separate training example) and is reported via ``unique_pairs``.
    """
    all_nodes = list(range(len(graph.node_ids)))
    forbidden_by_src: Dict[int, Set[int]] = defaultdict(set)
    for src, dst in facts.blocked_pairs:
        forbidden_by_src[src].add(dst)

    stats = SamplingStats(n_pos=len(positives))
    requested_total = n_hard + n_medium + n_easy
    stats.requested_hard = n_hard * len(positives)
    stats.requested_medium = n_medium * len(positives)
    stats.requested_easy = n_easy * len(positives)

    negatives: List[Sample] = []
    for pos in positives:
        key = pool_key(pos)
        hard_pool = [j for j in pools.hard[key] if j != pos.src and j != pos.dst]
        medium_pool = [
            j for j in pools.medium[key]
            if j != pos.src and j != pos.dst and j not in hard_pool
        ]
        forbidden = {pos.src, pos.dst} | forbidden_by_src[pos.src]
        chosen: List[int] = []

        # 1) hard
        take_h = min(n_hard, len(hard_pool))
        if take_h:
            chosen.extend(rng.sample(hard_pool, take_h))
        stats.used_hard += take_h
        if take_h < n_hard:
            stats.short_hard += 1
            if not hard_pool:
                stats.zero_hard += 1

        # 2) medium (plus the hard shortfall)
        deficit_h = n_hard - take_h
        want_m = n_medium + deficit_h
        available_m = [j for j in medium_pool if j not in chosen]
        take_m = min(want_m, len(available_m))
        if take_m:
            chosen.extend(rng.sample(available_m, take_m))
        stats.used_medium += take_m
        if take_m < want_m:
            stats.short_medium += 1

        # 3) easy (plus any remaining shortfall)
        deficit_m = want_m - take_m
        want_e = n_easy + deficit_m
        available_e = [
            j for j in all_nodes if j not in forbidden and j not in chosen
        ]
        take_e = min(want_e, len(available_e))
        if take_e:
            chosen.extend(rng.sample(available_e, take_e))
        stats.used_easy += take_e
        if take_e < want_e:
            stats.short_easy += 1

        stats.shortfall += requested_total - len(chosen)
        for dst in chosen:
            negatives.append(
                Sample(
                    src=pos.src,
                    dst=dst,
                    src_sid=pos.src_sid,
                    dst_sid=graph.node_ids[dst],
                    relation=pos.relation,
                    depth=0,
                    label=0,
                )
            )

    stats.unique_pairs = len({(s.src, s.dst) for s in negatives})
    return negatives, stats


# --------------------------------------------------------------------------- #
# Scoring / hard-negative metrics
# --------------------------------------------------------------------------- #
def compute_scores(model: ClosurePredictor, graph: RelationGraph) -> torch.Tensor:
    """Full (N, N) score matrix for the current model, in eval mode."""
    model.eval()
    with torch.no_grad():
        embeddings, _ = model.encode(graph.edge_index, graph.edge_type)
        return model.score_all(embeddings)


def eval_with_scores(
    model: ClosurePredictor,
    graph: RelationGraph,
    samples: Sequence[Sample],
    criterion: nn.Module,
) -> Tuple[float, float, Optional[torch.Tensor]]:
    """Loss/accuracy on a sample list plus the score matrix (one forward)."""
    if not samples:
        return float("nan"), float("nan"), None
    scores = compute_scores(model, graph)
    pairs = pairs_tensor(samples)
    logits = scores[pairs[:, 0], pairs[:, 1]]
    labels = labels_tensor(samples)
    loss = float(criterion(logits, labels))
    acc = float(((logits > 0).float() == labels).float().mean())
    return loss, acc, scores


def full_catalogue_by_depth(
    scores: torch.Tensor,
    graph: RelationGraph,
    facts: Facts,
    positives: Sequence[Sample],
) -> Dict[int, Dict[str, float]]:
    """Per-depth filtered ranks, ranked per positive.

    ``filtered_metrics`` computes the overall MRR/Hits correctly, but its
    ``by_depth`` view zips a set-ordered rank list with ``test_positives``.
    When one source has several closure targets, those two orders can differ,
    so ranks are assigned to the wrong depth.  This helper ranks every positive
    explicitly with the same mask and then groups by depth.
    """
    num_nodes = len(graph.node_ids)
    keep_by_query: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    for sample in positives:
        keep_by_query[(sample.src, sample.relation)].add(sample.dst)

    by_depth: Dict[int, List[int]] = defaultdict(list)
    for sample in positives:
        keep = keep_by_query[(sample.src, sample.relation)]
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[sample.src] = False
        for j in range(num_nodes):
            if j in keep:
                mask[j] = True
            elif (sample.src, j) in facts.blocked_pairs:
                mask[j] = False
        row = scores[sample.src].masked_fill(~mask, -1e9)
        order = torch.argsort(row, descending=True).tolist()
        by_depth[sample.depth].append(order.index(sample.dst) + 1)

    return {
        depth: {
            "n": len(ranks),
            "mrr": sum(1.0 / r for r in ranks) / len(ranks),
            "hits@1": sum(1 for r in ranks if r <= 1) / len(ranks),
            "hits@10": sum(1 for r in ranks if r <= 10) / len(ranks),
        }
        for depth, ranks in sorted(by_depth.items())
    }


@dataclasses.dataclass
class HardMetrics:
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


def hard_negative_metrics(
    scores: torch.Tensor,
    pools: NegativePools,
    positives: Sequence[Sample],
    min_k: int = 0,
    depths: Optional[Set[int]] = None,
) -> HardMetrics:
    """Rank each positive against *all* eligible hard negatives.

    Candidate set = ``{target} U hard_pool``.  This is deliberately larger than
    the 3 hard negatives seen during training: it avoids measuring a sampled
    subset and makes K (and the random baseline) explicit.

    ``min_k`` / ``depths`` restrict the report to nontrivial subsets (e.g.
    K>=2, depth=2) without changing the pools.

    The target prior is the model's own source-independent mean score,
    ``scores.mean(0)``, exactly as in the §9.8 baseline.
    """
    bias = scores.mean(dim=0)
    model_rr: List[float] = []
    prior_rr: List[float] = []
    ks: List[int] = []
    skipped = 0
    depth_records: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)

    for pos in positives:
        pool = pools.hard[pool_key(pos)]
        if not pool or len(pool) < min_k:
            skipped += 1
            continue
        if depths is not None and pos.depth not in depths:
            skipped += 1
            continue
        cand = pool + [pos.dst]
        cand_t = torch.tensor(cand, dtype=torch.long)
        # 正样本固定放在候选列表末尾，rank = 1 + 比它得分高的候选数。
        row = scores[pos.src].index_select(0, cand_t)
        order = torch.argsort(row, descending=True).tolist()
        rank = order.index(len(pool)) + 1

        prior_row = bias.index_select(0, cand_t)
        prior_order = torch.argsort(prior_row, descending=True).tolist()
        prior_rank = prior_order.index(len(pool)) + 1

        model_rr.append(1.0 / rank)
        prior_rr.append(1.0 / prior_rank)
        ks.append(len(pool))
        depth_records[pos.depth].append(
            (1.0 / rank, 1.0 / prior_rank, len(pool))
        )

    n = len(model_rr)
    if n == 0:
        return HardMetrics(
            model_mrr=float("nan"),
            prior_mrr=float("nan"),
            margin=float("nan"),
            mean_k=0.0,
            min_k=0,
            max_k=0,
            random_mrr=float("nan"),
            beats_random_frac=float("nan"),
            n_queries=0,
            n_skipped=skipped,
            by_depth={},
        )

    by_depth: Dict[int, Dict[str, float]] = {}
    for depth, records in sorted(depth_records.items()):
        m = len(records)
        by_depth[depth] = {
            "n": m,
            "model_mrr": sum(r[0] for r in records) / m,
            "prior_mrr": sum(r[1] for r in records) / m,
            "margin": sum(r[0] - r[1] for r in records) / m,
            "mean_k": sum(r[2] for r in records) / m,
            "random_mrr": sum(1.0 / (r[2] + 1) for r in records) / m,
        }

    return HardMetrics(
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
        by_depth=by_depth,
    )


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def load_closure_checkpoint(
    path: pathlib.Path,
    graph: RelationGraph,
    args: argparse.Namespace,
) -> Tuple[Optional[ClosurePredictor], Optional[str]]:
    """Load a plain ``state_dict`` checkpoint; return (model, error)."""
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
        model = ClosurePredictor(base, args.hidden)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as exc:  # pragma: no cover - corrupt/incompatible ckpt
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pool_line(name: str, summary: Dict[str, object]) -> str:
    hard = summary["hard"]          # type: ignore[index]
    medium = summary["medium"]      # type: ignore[index]
    return (
        f"{name:<6} n={summary['n']:>3}  hard K: min={hard['min']} "
        f"median={hard['median']:.1f} mean={hard['mean']:.1f} max={hard['max']} "
        f"zero={hard['zero']:>3} | medium median={medium['median']:.1f} "
        f"zero={medium['zero']:>3}"
    )


def _hard_row(name: str, hm: HardMetrics, n_pos: int) -> str:
    return (
        f"{name:<6} {n_pos:>4} {_fmt(hm.model_mrr):>8} {_fmt(hm.prior_mrr):>8} "
        f"{_fmt(hm.margin):>8} {hm.mean_k:>6.1f} {hm.n_queries:>4} "
        f"{hm.n_skipped:>3} {_fmt(hm.random_mrr):>7} "
        f"{_fmt(hm.beats_random_frac):>6}"
    )


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
    best_epoch: int,
    best_val_loss: float,
    best_margin_epoch: int,
    best_val_margin: float,
    epochs_run: int,
    stopped_early: bool,
    margin_history: Sequence[Tuple[int, float, float, float, float]],
    metrics: Dict[str, Dict[str, object]],
    full_by_depth: Dict[int, Dict[str, float]],
    prior_mrr: Dict[str, float],
    hard_metrics: Dict[str, HardMetrics],
    hard_test_k2: HardMetrics,
    hard_test_d2: HardMetrics,
    baseline_name: str,
    baseline_metrics: Optional[Dict[str, object]],
    baseline_hard: Optional[HardMetrics],
    baseline_hard_k2: Optional[HardMetrics],
    baseline_hard_d2: Optional[HardMetrics],
    baseline_prior_test: Optional[float],
    margin_ckpt_metrics: Optional[Dict[str, Dict[str, object]]],
    margin_ckpt_hard: Optional[Dict[str, HardMetrics]],
    init_attention: Dict[str, object],
    final_attention: Dict[str, object],
    cat_split: str,
    qualitative: Sequence[Dict[str, object]],
    target_ranks: Sequence[Tuple[str, str, int, int]],
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
    lines.append("=" * 100)
    lines.append("Logos Transformer — Hard-Negative Closure Training Report")
    lines.append("=" * 100)
    lines.append(f"generated_at        : {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"model_version       : {HARDNEG_VERSION}")
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
    lines.append("")
    lines.append("--- Source-disjoint split (identical seed/split to train_closure_disjoint.py) ---")
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
    lines.append("--- Closure target concentration (why hard pools are small) ---")
    header = f"{'relation':<12} {'band':>5} {'rows':>5} {'distinct_targets':>17} {'distinct_sources':>17}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in relation_bands:
        lines.append(
            f"{str(row['relation']):<12} {str(row['band']):>5} {row['rows']:>5} "
            f"{row['distinct_targets']:>17} {row['distinct_sources']:>17}"
        )
    lines.append("  is_a depth-2 has 102 rows but only 14 distinct targets; other sources'")
    lines.append("  targets are usually shared ancestors and therefore blocked as negatives.")
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
            f"{name:<6} {len(items['pos']):>4} {len(items['samples']) - len(items['pos']):>4} "
            f"{st.used_hard:>5} {st.used_medium:>4} {st.used_easy:>5} "
            f"{st.short_hard:>7} {st.short_medium:>7} {st.short_easy:>7} "
            f"{st.shortfall:>7} {st.unique_pairs:>6}"
        )
    lines.append(f"train class balance : {train_pos} pos / {train_neg} neg "
                 f"(pos_weight={pos_weight:.2f})")
    lines.append("  short_h/m/e count positives whose pool was too small; 'missing' counts")
    lines.append("  requested negatives that could not be filled even from the easy pool.")
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
    lines.append(f"neg_per_pos         : {args.neg_per_pos} "
                 f"({args.hard_ratio:.0%} hard / {args.medium_ratio:.0%} medium / "
                 f"{1.0 - args.hard_ratio - args.medium_ratio:.0%} easy requested)")
    lines.append(f"sampled per positive: {args.neg_per_pos} "
                 f"(hard≈{sampling_stats['train'].requested_hard // max(train_pos, 1)}, "
                 f"medium≈{sampling_stats['train'].requested_medium // max(train_pos, 1)}, "
                 f"easy≈{sampling_stats['train'].requested_easy // max(train_pos, 1)})")
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
    lines.append("  Checkpoint selection is best *val loss*, matching the earlier runs.")
    lines.append("  The margin checkpoint is saved separately and reported only as an")
    lines.append("  oracle-selected upper bound; it is not the primary model.")
    lines.append("")
    lines.append("--- Full-catalogue filtered ranking (170 candidates) ---")
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
    lines.append("Target-prior baseline (source-independent mean score, full catalogue):")
    lines.append(f"  train MRR={_fmt(prior_mrr['train'])}  "
                 f"val MRR={_fmt(prior_mrr['val'])}  "
                 f"test MRR={_fmt(prior_mrr['test'])}")
    lines.append("Full-catalogue margin (per-source model MRR - target-prior MRR):")
    lines.append(
        f"  train {metrics['train']['mrr'] - prior_mrr['train']:+.4f}  "
        f"val {metrics['val']['mrr'] - prior_mrr['val']:+.4f}  "
        f"test {metrics['test']['mrr'] - prior_mrr['test']:+.4f}"
    )
    lines.append("  AUC here is computed on the sampled hard/medium/easy mix, so it is")
    lines.append("  not comparable to the random-negative AUC in earlier reports.")
    lines.append("")
    lines.append("--- Hard-negative protocol (rank against ALL eligible hard negatives) ---")
    header = (f"{'split':<6} {'pos':>4} {'hardMRR':>8} {'prior':>8} {'margin':>8} "
              f"{'Kmean':>6} {'n':>4} {'skip':>4} {'random':>7} {'>rand':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for name in SPLIT_NAMES:
        lines.append(_hard_row(name, hard_metrics[name], len(data[name]["pos"])))  # type: ignore[arg-type]
    lines.append("  random = mean 1/(K+1) under a uniformly random ordering; '>rand' is the")
    lines.append("  fraction of positives whose model RR beats that per-query baseline.")
    lines.append("  'skip' = positives with an empty hard pool (excluded from MRR).")
    lines.append("")
    lines.append("--- Nontrivial hard-neg subsets (test) ---")
    lines.append("K=1 / depth 4+ queries are prior-trivial (one candidate or a shared hub);")
    lines.append("reporting them separately avoids inflating the headline hardMRR.")
    lines.append(f"{'model':<22} {'subset':<10} {'n':>4} {'hardMRR':>8} "
                 f"{'prior':>8} {'margin':>8} {'K':>5}")
    lines.append("-" * 70)
    for label, hm in (("all", hard_metrics["test"]),
                      ("K>=2", hard_test_k2),
                      ("depth=2", hard_test_d2)):
        lines.append(
            f"{'hardneg':<22} {label:<10} {hm.n_queries:>4} "
            f"{_fmt(hm.model_mrr):>8} {_fmt(hm.prior_mrr):>8} "
            f"{_fmt(hm.margin):>8} {hm.mean_k:>5.1f}"
        )
    if baseline_hard_k2 is not None and baseline_hard_d2 is not None:
        for label, hm in (("K>=2", baseline_hard_k2), ("depth=2", baseline_hard_d2)):
            lines.append(
                f"{baseline_name:<22} {label:<10} {hm.n_queries:>4} "
                f"{_fmt(hm.model_mrr):>8} {_fmt(hm.prior_mrr):>8} "
                f"{_fmt(hm.margin):>8} {hm.mean_k:>5.1f}"
            )
    lines.append("")
    lines.append("--- Test comparison on identical hard pools / test pairs ---")
    header = (f"{'model':<30} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} "
              f"{'AUC*':>7} {'hardMRR':>8} {'prior':>8} {'margin':>8} {'K':>5}")
    lines.append(header)
    lines.append("-" * len(header))
    if baseline_metrics is not None and baseline_hard is not None:
        lines.append(
            f"{baseline_name:<30} {_fmt(baseline_metrics['mrr']):>7} "
            f"{_fmt(baseline_metrics['hits@1']):>7} "
            f"{_fmt(baseline_metrics['hits@3']):>7} "
            f"{_fmt(baseline_metrics['hits@10']):>7} "
            f"{_fmt(baseline_metrics['auc']):>7} "
            f"{_fmt(baseline_hard.model_mrr):>8} "
            f"{_fmt(baseline_hard.prior_mrr):>8} "
            f"{_fmt(baseline_hard.margin):>8} {baseline_hard.mean_k:>5.1f}"
        )
    else:
        lines.append(f"{baseline_name:<30} n/a (checkpoint missing or incompatible)")
    test_m = metrics["test"]
    test_h = hard_metrics["test"]
    lines.append(
        f"{'hardneg (best val loss)':<30} {_fmt(test_m['mrr']):>7} "
        f"{_fmt(test_m['hits@1']):>7} {_fmt(test_m['hits@3']):>7} "
        f"{_fmt(test_m['hits@10']):>7} {_fmt(test_m['auc']):>7} "
        f"{_fmt(test_h.model_mrr):>8} {_fmt(test_h.prior_mrr):>8} "
        f"{_fmt(test_h.margin):>8} {test_h.mean_k:>5.1f}"
    )
    if margin_ckpt_metrics is not None and margin_ckpt_hard is not None:
        mm = margin_ckpt_metrics["test"]
        mh = margin_ckpt_hard["test"]
        lines.append(
            f"{'hardneg (best val margin)':<30} {_fmt(mm['mrr']):>7} "
            f"{_fmt(mm['hits@1']):>7} {_fmt(mm['hits@3']):>7} "
            f"{_fmt(mm['hits@10']):>7} {_fmt(mm['auc']):>7} "
            f"{_fmt(mh.model_mrr):>8} {_fmt(mh.prior_mrr):>8} "
            f"{_fmt(mh.margin):>8} {mh.mean_k:>5.1f}"
        )
    lines.append("  MRR/Hits are filtered full-catalogue and identical in protocol to the")
    lines.append("  earlier reports; AUC* is on the shared hard-mix test pairs, not the")
    lines.append("  random negatives used in earlier AUCs. hardMRR/prior/margin use the")
    lines.append("  same hard pools for every model, so these columns are paired.")
    if baseline_hard is not None:
        lines.append(f"  margin delta vs {baseline_name}: "
                     f"{test_h.margin - baseline_hard.margin:+.4f}")
    lines.append("")
    lines.append("--- Test hard-negative metrics by depth (best val loss) ---")
    lines.append(f"{'depth':>5} {'n':>4} {'hardMRR':>8} {'prior':>8} {'margin':>8} "
                 f"{'Kmean':>6} {'random':>7}")
    for depth, dm in hard_metrics["test"].by_depth.items():
        lines.append(
            f"{depth:>5} {int(dm['n']):>4} {_fmt(dm['model_mrr']):>8} "
            f"{_fmt(dm['prior_mrr']):>8} {_fmt(dm['margin']):>8} "
            f"{dm['mean_k']:>6.1f} {_fmt(dm['random_mrr']):>7}"
        )
    lines.append("  depth bands 4+ merge depths 4-6; several cells have n<=3, so treat")
    lines.append("  per-depth hard numbers as descriptive, not inferential.")
    lines.append("")
    lines.append("--- Test full-catalogue metrics by depth (best val loss) ---")
    for depth, dm in full_by_depth.items():
        lines.append(
            f"  depth={depth} (n={int(dm['n']):>2})  MRR={_fmt(dm['mrr'])}  "
            f"Hits@1={_fmt(dm['hits@1'])}  Hits@10={_fmt(dm['hits@10'])}"
        )
    lines.append("  (Per-positive ranks; the by_depth view in train_closure.py's")
    lines.append("   filtered_metrics can misalign when a source has multiple targets.)")
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
    lines.append(f"--- Qualitative: top-5 targets for「猫」(sense 2, split={cat_split}) ---")
    for i, item in enumerate(qualitative, 1):
        extra = ""
        if item["closure"]:
            rel, depth = item["closure"]
            extra = f"  [closure: {rel}, depth {depth}]"
        lines.append(f"  {i}. {item['label']}({item['sid']})  "
                     f"score={item['score']:.4f}{extra}")
    if target_ranks:
        lines.append("  Known closure targets and their ranks:")
        for label, relation, depth, rank in target_ranks:
            lines.append(f"    {label:<14} {relation} depth={depth}  rank=#{rank}")
    lines.append("")
    lines.append("--- Verdict vs requested thresholds ---")
    mrr_ok = test_h.model_mrr > 0.3
    margin_ok = test_h.margin > 0.1
    full_margin = test_m["mrr"] - prior_mrr["test"]
    lines.append(f"Hard-neg MRR > 0.3        : {_fmt(test_h.model_mrr)} -> "
                 f"{'PASS' if mrr_ok else 'FAIL'}")
    lines.append(f"Model-vs-prior margin > 0.1: {test_h.margin:+.4f} -> "
                 f"{'PASS' if margin_ok else 'FAIL'}")
    lines.append(f"Full-catalogue margin      : {full_margin:+.4f} "
                 f"({_fmt(test_m['mrr'])} - {_fmt(prior_mrr['test'])})")
    if baseline_hard is not None:
        baseline_full_margin = (
            baseline_metrics["mrr"] - baseline_prior_test  # type: ignore[index]
            if baseline_metrics is not None and baseline_prior_test is not None
            else float("nan")
        )
        lines.append(
            f"Baseline (same pools)     : "
            f"hardMRR={_fmt(baseline_hard.model_mrr)} "
            f"margin={baseline_hard.margin:+.4f} "
            f"full-margin≈{baseline_full_margin:+.4f}"
        )
    else:
        lines.append("Baseline (same pools)     : n/a")
    if mrr_ok and margin_ok:
        lines.append("VERDICT: both requested thresholds pass on the point estimates.")
        lines.append("The hard-neg model adds source-conditioned signal beyond the target")
        lines.append("prior.  Two caveats: (1) the test set is small, so the absolute")
        lines.append("margin CI is wide (see HARD_NEGATIVE_COMPARISON.md for paired")
        lines.append("bootstrap CIs); (2) the full-catalogue MRR gain is positive but not")
        lines.append("statistically decisive, and 猫 is a counterexample (生物 moves from")
        lines.append("rank #3 in the disjoint model to #13 here).  The supported claim is")
        lines.append("coarse source-conditioned branch discrimination, not uniform")
        lines.append("fine-grained multi-hop improvement.")
    elif mrr_ok and not margin_ok:
        lines.append("VERDICT: hard-neg retrieval is above 0.3, but the margin over the")
        lines.append("target prior is below 0.1. The ranking is still largely explained by")
        lines.append("a source-independent prior; absolute retrieval alone is not enough.")
    elif margin_ok and not mrr_ok:
        lines.append("VERDICT: the model beats the prior by >0.1 but absolute hard-neg MRR")
        lines.append("is below 0.3. Signal exists, but the candidate set remains hard.")
    else:
        lines.append("VERDICT: FAIL on both. At this graph size the architecture/data do")
        lines.append("not support source-conditioned discrimination beyond the target prior.")
        lines.append("Next steps: relation-aware head (W_r per relation) and/or expand the")
        lines.append("animal library so hard pools contain more than a handful of targets.")
    if margin_ckpt_hard is not None:
        mh = margin_ckpt_hard["test"]
        lines.append(f"Best-margin checkpoint upper bound: hardMRR={_fmt(mh.model_mrr)} "
                     f"margin={mh.margin:+.4f} (oracle model selection; not primary).")
    lines.append("")
    lines.append("--- Caveats ---")
    lines.append("* Hard pools are intrinsically tiny.  is_a has only 14 / 6 / 3 distinct")
    lines.append("  targets at depth 2 / 3 / 4+, because closure targets are shared hubs.")
    lines.append("  Many other-source targets are also ancestors of the query source and")
    lines.append("  are therefore *not* valid negatives (they are blocked, not sampled).")
    lines.append("* 22/180 train, 1/33 val and 0/28 test positives have an empty hard pool")
    lines.append("  (exact counts are in the pool table).  Empty-pool test positives are")
    lines.append("  skipped in hard-neg MRR; the report prints n and skipped.")
    lines.append("* Hard negatives are mostly *cross-branch* hubs (e.g. an animal source's")
    lines.append("  pool contains 人造物 / 医学概念 / 病理状态, not another animal ancestor).")
    lines.append("  The learned signal is therefore coarse source-conditioned branch")
    lines.append("  membership, not fine-grained same-branch ancestor ranking.  The")
    lines.append("  K>=2 / depth-2 subset table keeps the prior-trivial queries separate.")
    lines.append("* Full-catalogue ranking is not uniformly better: 猫's 生物 moves from")
    lines.append("  rank #3 (disjoint model) to #13 here, and 人->生物 is rank #22.  The")
    lines.append("  aggregate full-catalogue MRR gain is positive but not statistically")
    lines.append("  decisive with 15 test sources (see HARD_NEGATIVE_COMPARISON.md).")
    lines.append("* Because K is often 1-13, hard-neg MRR is coarse: with K=1 the only")
    lines.append("  possible values are 1.0 and 0.5.  The random baseline is reported")
    lines.append("  per query as 1/(K+1), not assumed to be 1/K.")
    lines.append("* The target prior is model-derived (mean score over all sources, test")
    lines.append("  sources included) and source-independent.  It is a diagnostic upper")
    lines.append("  bound on 'target looks like an ancestor', not a deployable baseline.")
    lines.append("* Source-disjoint is still transductive: test-source nodes appear in the")
    lines.append("  message graph and their embeddings are updated during training.")
    lines.append("* Test has 15 sources / 28 positives; paired-source bootstrap CIs are")
    lines.append("  reported in HARD_NEGATIVE_COMPARISON.md.  Small deltas are not decisive.")
    lines.append("* AUC on the hard-mix test pairs is not comparable to the random-negative")
    lines.append("  AUC (0.9969 / 1.0000) in the earlier reports.")
    lines.append("* has_part has a single positive and no other has_part source, so it")
    lines.append("  always falls back to medium/easy; it cannot be hard-negative trained.")
    lines.append("* Duplicate (source, target) negatives can appear for different")
    lines.append("  positives of the same source; 'unique' in the dataset table shows how")
    lines.append("  many distinct pairs the sampled negatives contain.")
    lines.append("")
    lines.append("--- Reproduction ---")
    lines.append("  python logos_transformer/train_closure_hardneg.py --epochs 5   # smoke")
    lines.append("  python logos_transformer/train_closure_hardneg.py              # full")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train/evaluate the Logos GAT with hard negatives "
                    "(source-disjoint split)."
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
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-checkpoint", type=pathlib.Path,
                        default=root / DEFAULT_COMPARE_CHECKPOINT,
                        help="source-disjoint checkpoint evaluated on the same hard pools")
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
    if args.patience < 1:
        parser.error("--patience 必须 >= 1")
    if args.seed < 0:
        parser.error("--seed 必须 >= 0")
    return args


def _split_ratios(n_total: int, hard_ratio: float, medium_ratio: float) -> Tuple[int, int, int]:
    n_hard = int(round(n_total * hard_ratio))
    n_medium = int(round(n_total * medium_ratio))
    if n_hard + n_medium > n_total:      # 舍入防御
        n_medium = max(0, n_total - n_hard)
    n_easy = n_total - n_hard - n_medium
    return n_hard, n_medium, n_easy


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
            "drawing conclusions.",
            file=sys.stderr,
        )
    if args.seed != 42 and compare_path.exists():
        print(
            "[warning] --seed differs from 42; the comparison checkpoint was trained "
            "on the seed-42 split and is not directly comparable.",
            file=sys.stderr,
        )

    # Same split as train_closure_disjoint.py (seed 42 by default).
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

    train_samples: List[Sample] = data["train"]["samples"]           # type: ignore[assignment]
    val_samples: List[Sample] = data["val"]["samples"]               # type: ignore[assignment]
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

    print("=" * 100)
    print("Logos Transformer — hard-negative closure prediction (source-disjoint)")
    print("=" * 100)
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
    print(f"train pos_weight={pos_weight:.2f} | "
          f"model={args.layers}x{args.heads} heads, hidden={args.hidden}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"hard_ratio={args.hard_ratio:.2f} medium_ratio={args.medium_ratio:.2f} "
          f"-> per positive: {n_hard} hard / {n_medium} medium / {n_easy} easy")
    print("-" * 100)
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
    print("-" * 100)

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
    best_margin = float("-inf")
    best_margin_epoch = 0
    stale = 0
    epochs_run = 0
    margin_history: List[Tuple[int, float, float, float, float]] = []

    batch_rng = random.Random(args.seed + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "closure_gat_hardneg_best.pt"
    margin_checkpoint_path = output_dir / "closure_gat_hardneg_best_margin.pt"

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
            val_loss, val_acc, val_scores = eval_with_scores(
                model, graph, val_samples, criterion
            )
        else:
            val_loss, val_acc, val_scores = train_loss, train_acc, compute_scores(model, graph)
        if math.isnan(val_loss):
            val_loss = train_loss
        scheduler.step(val_loss)

        val_hard = hard_negative_metrics(
            val_scores if val_scores is not None else compute_scores(model, graph),
            pools,
            data["val"]["pos"],                                      # type: ignore[arg-type]
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

        # 诊断用 checkpoint：只按 val hard margin 选，不作为主结果。
        if val_hard.n_queries and val_hard.margin > best_margin + 1e-6:
            best_margin = val_hard.margin
            best_margin_epoch = epoch
            torch.save(model.state_dict(), margin_checkpoint_path)

        # 每个 epoch 都打印 margin（需求要求与 loss 同列；不做抽样）。
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
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()

    metrics: Dict[str, Dict[str, object]] = {}
    for name in SPLIT_NAMES:
        metrics[name] = eval_bucket(
            model, graph, facts, data[name]["samples"], criterion   # type: ignore[arg-type]
        )
    prior_mrr = {
        name: target_prior_mrr(model, graph, facts, data[name]["pos"])  # type: ignore[arg-type]
        for name in SPLIT_NAMES
    }
    test_scores = compute_scores(model, graph)
    full_by_depth = full_catalogue_by_depth(
        test_scores, graph, facts, data["test"]["pos"]          # type: ignore[arg-type]
    )
    hard_metrics = {
        name: hard_negative_metrics(test_scores, pools, data[name]["pos"])  # type: ignore[arg-type]
        for name in SPLIT_NAMES
    }
    # Nontrivial subsets: K=1 / depth 4+ queries are trivially prior-consistent.
    hard_test_k2 = hard_negative_metrics(
        test_scores, pools, data["test"]["pos"], min_k=2        # type: ignore[arg-type]
    )
    hard_test_d2 = hard_negative_metrics(
        test_scores, pools, data["test"]["pos"], depths={2}     # type: ignore[arg-type]
    )

    # Diagnostic best-margin checkpoint (oracle selection).
    margin_ckpt_metrics: Optional[Dict[str, Dict[str, object]]] = None
    margin_ckpt_hard: Optional[Dict[str, HardMetrics]] = None
    if margin_checkpoint_path.exists():
        margin_model, err = load_closure_checkpoint(margin_checkpoint_path, graph, args)
        if margin_model is not None:
            margin_model.eval()
            margin_ckpt_metrics = {
                name: eval_bucket(
                    margin_model, graph, facts, data[name]["samples"], criterion  # type: ignore[arg-type]
                )
                for name in SPLIT_NAMES
            }
            margin_scores = compute_scores(margin_model, graph)
            margin_ckpt_hard = {
                name: hard_negative_metrics(
                    margin_scores, pools, data[name]["pos"]                 # type: ignore[arg-type]
                )
                for name in SPLIT_NAMES
            }
        else:
            print(f"[warning] could not load margin checkpoint: {err}", file=sys.stderr)

    # Source-disjoint baseline on the *same* test pairs and hard pools.
    baseline_name = "source-disjoint ckpt"
    baseline_metrics: Optional[Dict[str, object]] = None
    baseline_hard: Optional[HardMetrics] = None
    baseline_hard_k2: Optional[HardMetrics] = None
    baseline_hard_d2: Optional[HardMetrics] = None
    baseline_prior_test: Optional[float] = None
    baseline_model, baseline_err = load_closure_checkpoint(compare_path, graph, args)
    if baseline_model is not None:
        baseline_metrics = eval_bucket(
            baseline_model, graph, facts, data["test"]["samples"], criterion  # type: ignore[arg-type]
        )
        baseline_prior_test = target_prior_mrr(
            baseline_model, graph, facts, data["test"]["pos"]               # type: ignore[arg-type]
        )
        baseline_scores = compute_scores(baseline_model, graph)
        baseline_hard = hard_negative_metrics(
            baseline_scores, pools, data["test"]["pos"]                  # type: ignore[arg-type]
        )
        baseline_hard_k2 = hard_negative_metrics(
            baseline_scores, pools, data["test"]["pos"], min_k=2          # type: ignore[arg-type]
        )
        baseline_hard_d2 = hard_negative_metrics(
            baseline_scores, pools, data["test"]["pos"], depths={2}       # type: ignore[arg-type]
        )
    else:
        print(f"[warning] comparison checkpoint skipped: {baseline_err}", file=sys.stderr)

    final_attention = attention_stats(model, graph, cat_idx)
    qualitative = qualitative_examples(model, graph, facts, source_sid=2, top_k=5)
    target_ranks = closure_target_ranks(model, graph, facts, source_sid=2)

    # Console summary.
    print("-" * 100)
    print("full-catalogue test metrics + hard-negative protocol")
    print(f"{'model':<30} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} "
          f"{'AUC*':>7} {'hardMRR':>8} {'prior':>8} {'margin':>8} {'K':>5}")
    if baseline_metrics is not None and baseline_hard is not None:
        print(
            f"{baseline_name:<30} {_fmt(baseline_metrics['mrr']):>7} "
            f"{_fmt(baseline_metrics['hits@1']):>7} "
            f"{_fmt(baseline_metrics['hits@3']):>7} "
            f"{_fmt(baseline_metrics['hits@10']):>7} "
            f"{_fmt(baseline_metrics['auc']):>7} "
            f"{_fmt(baseline_hard.model_mrr):>8} "
            f"{_fmt(baseline_hard.prior_mrr):>8} "
            f"{_fmt(baseline_hard.margin):>8} {baseline_hard.mean_k:>5.1f}"
        )
    test_m = metrics["test"]
    test_h = hard_metrics["test"]
    print(
        f"{'hardneg (best val loss)':<30} {_fmt(test_m['mrr']):>7} "
        f"{_fmt(test_m['hits@1']):>7} {_fmt(test_m['hits@3']):>7} "
        f"{_fmt(test_m['hits@10']):>7} {_fmt(test_m['auc']):>7} "
        f"{_fmt(test_h.model_mrr):>8} {_fmt(test_h.prior_mrr):>8} "
        f"{_fmt(test_h.margin):>8} {test_h.mean_k:>5.1f}"
    )
    if margin_ckpt_metrics is not None and margin_ckpt_hard is not None:
        mm = margin_ckpt_metrics["test"]
        mh = margin_ckpt_hard["test"]
        print(
            f"{'hardneg (best val margin)':<30} {_fmt(mm['mrr']):>7} "
            f"{_fmt(mm['hits@1']):>7} {_fmt(mm['hits@3']):>7} "
            f"{_fmt(mm['hits@10']):>7} {_fmt(mm['auc']):>7} "
            f"{_fmt(mh.model_mrr):>8} {_fmt(mh.prior_mrr):>8} "
            f"{_fmt(mh.margin):>8} {mh.mean_k:>5.1f}"
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
    print(f"attention entropy: init={init_attention['mean_entropy']:.4f} "
          f"-> trained={final_attention['mean_entropy']:.4f} | 猫 split={cat_split}")

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
        best_epoch=best_epoch,
        best_val_loss=best_val,
        best_margin_epoch=best_margin_epoch,
        best_val_margin=best_margin,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
        margin_history=margin_history,
        metrics=metrics,
        full_by_depth=full_by_depth,
        prior_mrr=prior_mrr,
        hard_metrics=hard_metrics,
        hard_test_k2=hard_test_k2,
        hard_test_d2=hard_test_d2,
        baseline_name=baseline_name,
        baseline_metrics=baseline_metrics,
        baseline_hard=baseline_hard,
        baseline_hard_k2=baseline_hard_k2,
        baseline_hard_d2=baseline_hard_d2,
        baseline_prior_test=baseline_prior_test,
        margin_ckpt_metrics=margin_ckpt_metrics,
        margin_ckpt_hard=margin_ckpt_hard,
        init_attention=init_attention,
        final_attention=final_attention,
        cat_split=cat_split,
        qualitative=qualitative,
        target_ranks=target_ranks,
        checkpoint=checkpoint_path,
        margin_checkpoint=margin_checkpoint_path,
    )
    report_path = output_dir / "training_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"checkpoint : {checkpoint_path}")
    print(f"report     : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
