#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrated closure prediction: cross-POS negatives + temperature scaling.

Why this script exists
----------------------
Round 12 (``infer_and_write.py``) exposed a failure that every ranking metric in
this project is structurally blind to: the model saturates on inputs the
training distribution never contained::

    蛇   --is_a-->         潜水   sigma = 0.9997   (潜水 is a verb, is_a targets are nouns)
    犬科 --has_property--> 潜水   sigma = 1.0000

Root cause: ``train_closure_hardneg.sample_hardneg`` draws every negative from
the *same relation's* positive targets, so a negative target always shares the
positive's part of speech.  The model therefore never sees
"(animal, verb) is negative under ``is_a``" and has no reason to push that
direction down.  MRR and margin only rank inside that pool, so they stay healthy
while OOD behaviour is broken.

This script fixes the *scoring surface*, not the threshold:

1. **Cross-POS negatives.**  For each positive we add a fourth negative tier
   whose target has a part of speech the closure engine has never attached to
   that relation.  The POS expectation is read off ``closure`` itself (exactly
   the same reading ``infer_and_write.build_pos_guard`` uses at inference time),
   so there is no hand-written type table anywhere.
2. **Temperature scaling.**  A single scalar ``T`` minimises NLL on the
   validation split, giving ``calibrated = sigmoid(logit / T)``.  ``T`` is stored
   in the checkpoint as a buffer and mirrored into ``calibration.json``.
3. **Calibration diagnostics.**  ECE, reliability table, and an OOD rejection
   rate measured on held-out (test-source) cross-POS probes.

A mathematical caveat worth stating up front
--------------------------------------------
Temperature scaling is a **strictly monotone** transform, so it cannot change
any ranking (MRR / margin are identical before and after) and it cannot change
the sign of any logit.  Hence:

    sigma(logit / T) < 0.5   <=>   logit < 0        (for every T > 0)

The OOD *rejection rate* at a 0.5 threshold can therefore only be improved by
cross-POS training, never by ``T``.  What ``T`` buys is a usable *probability*
(ECE) and a defensible write count at a fixed threshold.  The report prints the
two effects separately instead of pretending one number covers both.

Usage
-----
    python logos_transformer/train_calibrated.py --epochs 5              # smoke
    python logos_transformer/train_calibrated.py                         # full run
    python logos_transformer/train_calibrated.py \\
        --cross-pos-ratio 0 --output-dir ./logos_transformer/checkpoints_calibrated_ctl

Outputs
-------
    <output-dir>/closure_gat_calibrated_best.pt      weights + fitted temperature
    <output-dir>/closure_gat_calibrated_untuned.pt   same weights, T = 1 (A/B control)
    <output-dir>/calibration.json                    machine-readable calibration record
    <output-dir>/training_report.txt                 full text report

Exit codes
----------
    1  missing torch / database / closure data / checkpoint
    2  invalid command-line arguments
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import pathlib
import random
import sqlite3
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment dependent
    print("Requires torch. Install with: pip install torch", file=sys.stderr)
    raise SystemExit(1)

# 复用既有模块：模型类、切分、负样本池、指标。**不改动**任何既有训练脚本。
try:
    from prototype_gat import RelationGraph, build_model, load_graph  # type: ignore
    from train_closure import Facts, Sample, _fmt, labels_tensor, pairs_tensor  # type: ignore
    from train_closure_disjoint import load_positives, split_by_source  # type: ignore
    from train_closure_hardneg import (  # type: ignore
        NegativePools,
        build_negative_pools,
        pool_key,
    )
    from train_closure_relaware import (  # type: ignore
        MIN_DEDICATED,
        RELAWARE_VERSION,
        RelationAwareClosurePredictor,
        encode_and_score,
        eval_bucket_relaware,
        hard_metrics,
        loss_acc,
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
        labels_tensor,
        pairs_tensor,
    )
    from logos_transformer.train_closure_disjoint import (  # type: ignore
        load_positives,
        split_by_source,
    )
    from logos_transformer.train_closure_hardneg import (  # type: ignore
        NegativePools,
        build_negative_pools,
        pool_key,
    )
    from logos_transformer.train_closure_relaware import (  # type: ignore
        MIN_DEDICATED,
        RELAWARE_VERSION,
        RelationAwareClosurePredictor,
        encode_and_score,
        eval_bucket_relaware,
        hard_metrics,
        loss_acc,
    )

# 类型期望与推理期闸门必须是**同一份定义**，否则「用训练改的东西」和
# 「用推理卡的东西」会悄悄分叉 —— 所以直接复用 infer_and_write 的两个读取函数。
try:
    from infer_and_write import build_pos_guard, load_relation_target_pos  # type: ignore
except ImportError:  # pragma: no cover
    from logos_transformer.infer_and_write import (  # type: ignore
        build_pos_guard,
        load_relation_target_pos,
    )

CALIB_VERSION = "logos-gat-closure-calibrated-v1.0"
DEFAULT_OUTPUT_DIR = "./logos_transformer/checkpoints_calibrated"
DEFAULT_COMPARE_CHECKPOINT = (
    "./logos_transformer/checkpoints_relaware_ext/closure_gat_relaware_best.pt"
)
SPLIT_NAMES = ("train", "val", "test")
NEG_TIERS = ("hard", "medium", "easy", "cross")

# 语义探针：前两条**必须**保持高置信度，后三条是 round-12 实测出的饱和错误。
SANITY_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("猫", "is_a", "生物"),
    ("人", "can_do", "潜水"),
    ("蛇", "is_a", "潜水"),
    ("犬科", "has_property", "潜水"),
    ("鸟类", "typically_has", "尾巴"),
)


# --------------------------------------------------------------------------- #
# Model: the relaware predictor + a temperature buffer
# --------------------------------------------------------------------------- #
class CalibratedClosurePredictor(RelationAwareClosurePredictor):
    """``RelationAwareClosurePredictor`` with one extra scalar: ``temperature``.

    The buffer is part of ``state_dict``, so a checkpoint carries its own
    calibration; nothing has to be kept in a side channel.  The parent class is
    left untouched (constraint: existing training scripts stay read-only), which
    is why this is a subclass instead of a field added in place.
    """

    def __init__(self, *args, temperature: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "temperature", torch.tensor(float(temperature), dtype=torch.float32)
        )


def calibrated_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """sigmoid(logit / T); ``T = 1`` reproduces the raw model."""
    t = max(float(temperature), 1e-6)
    return torch.sigmoid(logits / t)


# --------------------------------------------------------------------------- #
# Cross-POS machinery
# --------------------------------------------------------------------------- #
def wrong_pos_index(
    graph: RelationGraph,
    allowed: Dict[str, Set[str]],
    relations: Sequence[str],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Split node indices into (allowed target POS, wrong target POS), per relation.

    A relation with no observed POS distribution in ``closure`` gets two empty
    lists: we refuse to invent a type expectation, and the cross-POS tier is
    simply skipped for it (reported as a shortfall, never as a fake negative).
    """
    by_pos: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, pos in enumerate(graph.pos_tags):
        by_pos[str(pos)].append(idx)

    allowed_nodes: Dict[str, List[int]] = {}
    wrong_nodes: Dict[str, List[int]] = {}
    for relation in relations:
        ok = allowed.get(relation)
        if not ok:
            allowed_nodes[relation] = []
            wrong_nodes[relation] = []
            continue
        allowed_nodes[relation] = [i for p in sorted(ok) for i in by_pos.get(p, [])]
        wrong_nodes[relation] = [
            i for p, items in sorted(by_pos.items()) if p not in ok for i in items
        ]
    return allowed_nodes, wrong_nodes


def allocate_tiers(
    total: int,
    ratios: Dict[str, float],
) -> Dict[str, int]:
    """Largest-remainder allocation of ``total`` negatives over the tiers.

    Renormalises the given ratios, so ``{"hard":0.4,"medium":0.3,"easy":0.2}``
    (cross-POS disabled) yields the historical 3/2/1 split on 6 negatives, while
    the full 4-tier default ``0.4/0.3/0.2/0.1`` yields 2/2/1/1.
    """
    weights = {k: float(v) for k, v in ratios.items() if float(v) > 0.0}
    out = {tier: 0 for tier in ratios}
    if not weights or total <= 0:
        return out
    scale = sum(weights.values())
    raw = {k: total * v / scale for k, v in weights.items()}
    base = {k: int(math.floor(x)) for k, x in raw.items()}
    remainder = total - sum(base.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))
    for key in order[:remainder]:
        base[key] += 1
    out.update(base)
    return out


@dataclasses.dataclass
class CalibSamplingStats:
    """Per-split negative sampling accounting, four tiers."""

    n_pos: int = 0
    requested: Dict[str, int] = dataclasses.field(default_factory=dict)
    used: Dict[str, int] = dataclasses.field(default_factory=dict)
    short_hard: int = 0
    short_medium: int = 0
    short_easy: int = 0
    zero_hard: int = 0
    short_cross: int = 0          # cross-POS 候选不足的正样本数
    zero_cross: int = 0           # 该关系没有 cross-POS 候选（无允许词性记录）
    shortfall: int = 0            # 请求但填不满的负样本总数（cross 缺口也计入）
    cross_by_relation: Dict[str, int] = dataclasses.field(default_factory=dict)
    unique_pairs: int = 0

    @property
    def used_total(self) -> int:
        return sum(self.used.values())


def sample_negatives_crosspos(
    positives: Sequence[Sample],
    graph: RelationGraph,
    facts: Facts,
    pools: NegativePools,
    quota: Dict[str, int],
    wrong_nodes: Dict[str, List[int]],
    rng: random.Random,
) -> Tuple[List[Sample], CalibSamplingStats]:
    """Sample four negative tiers per positive: hard / medium / easy / cross-POS.

    Tier semantics and fallback rules:

    * ``hard``   target of another source, same relation *and* same depth band;
    * ``medium`` same relation, any depth, minus the hard tier;
    * ``easy``   any non-reachable node;
    * ``cross``  target whose part of speech the closure engine has **never**
      accepted for this relation (``is_a`` -> verb/adjective, ``can_do`` ->
      noun/adjective, ``has_property`` -> verb, ...).

    Unlike the first three, the cross tier is **never** used as filler: filling
    a hard-negative shortfall with a deliberately type-violating pair would make
    the tier indistinguishable from noise.  A cross shortfall is reported and
    the positive simply gets fewer negatives.
    """
    all_nodes = list(range(len(graph.node_ids)))
    forbidden_by_src: Dict[int, Set[int]] = collections.defaultdict(set)
    for src, dst in facts.blocked_pairs:
        forbidden_by_src[src].add(dst)

    stats = CalibSamplingStats(n_pos=len(positives))
    stats.requested = {tier: int(quota.get(tier, 0)) * len(positives) for tier in NEG_TIERS}
    stats.used = {tier: 0 for tier in NEG_TIERS}
    cross_counter: collections.Counter = collections.Counter()
    requested_total = sum(quota.get(tier, 0) for tier in NEG_TIERS)

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

        def _take(pool: Sequence[int], want: int) -> int:
            available = [j for j in pool if j not in forbidden and j not in chosen]
            take = min(max(want, 0), len(available))
            if take:
                chosen.extend(rng.sample(available, take))
            return take

        # 1) hard
        want_h = quota.get("hard", 0)
        take_h = _take(hard_pool, want_h)
        stats.used["hard"] += take_h
        if take_h < want_h:
            stats.short_hard += 1
            if not hard_pool:
                stats.zero_hard += 1

        # 2) medium（吸收 hard 的缺口）
        want_m = quota.get("medium", 0) + (want_h - take_h)
        take_m = _take(medium_pool, want_m)
        stats.used["medium"] += take_m
        if take_m < want_m:
            stats.short_medium += 1

        # 3) easy（吸收 medium 的缺口）
        want_e = quota.get("easy", 0) + (want_m - take_m)
        take_e = _take(all_nodes, want_e)
        stats.used["easy"] += take_e
        if take_e < want_e:
            stats.short_easy += 1

        # 4) cross-POS：独立配额，缺了就是缺了，不用别的档位冒充
        want_c = quota.get("cross", 0)
        take_c = 0
        if want_c > 0:
            cross_pool = wrong_nodes.get(pos.relation) or []
            take_c = _take(cross_pool, want_c)
            stats.used["cross"] += take_c
            cross_counter[pos.relation] += take_c
            if take_c < want_c:
                stats.short_cross += 1
                if not cross_pool:
                    stats.zero_cross += 1

        stats.shortfall += max(0, requested_total - (take_h + take_m + take_e + take_c))

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

    stats.cross_by_relation = dict(sorted(cross_counter.items()))
    stats.unique_pairs = len({(s.src, s.dst) for s in negatives})
    return negatives, stats


def build_ood_probe(
    heldout_positives: Sequence[Sample],
    graph: RelationGraph,
    facts: Facts,
    wrong_nodes: Dict[str, List[int]],
    per_source: int,
    rng: random.Random,
) -> List[Sample]:
    """Cross-POS probes whose *sources* are held out from training.

    These are the same kind of pair as the training cross-POS tier (so the probe
    measures exactly the defect the tier is meant to fix) but they come from
    test-split sources, which the model never saw as a query source.
    """
    by_src: Dict[int, Set[str]] = collections.defaultdict(set)
    for sample in heldout_positives:
        by_src[sample.src].add(sample.relation)

    forbidden_by_src: Dict[int, Set[int]] = collections.defaultdict(set)
    for src, dst in facts.blocked_pairs:
        forbidden_by_src[src].add(dst)

    probes: List[Sample] = []
    seen: Set[Tuple[int, str, int]] = set()
    for src in sorted(by_src):
        forbidden = {src} | forbidden_by_src[src]
        for relation in sorted(by_src[src]):
            pool = [
                j for j in (wrong_nodes.get(relation) or []) if j not in forbidden
            ]
            if not pool:
                continue
            take = min(per_source, len(pool))
            for dst in rng.sample(pool, take):
                key = (src, relation, dst)
                if key in seen:
                    continue
                seen.add(key)
                probes.append(
                    Sample(
                        src=src,
                        dst=dst,
                        src_sid=graph.node_ids[src],
                        dst_sid=graph.node_ids[dst],
                        relation=relation,
                        depth=0,
                        label=0,
                    )
                )
    return probes


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TemperatureFit:
    """Result of the one-parameter NLL fit on the validation split."""

    raw_optimum: float      # NLL 最优的 T（不设下限）
    applied: float          # 实际写入 checkpoint 的 T（受 --min-temperature 约束）
    nll_before: float
    nll_after: float
    nll_at_raw: float
    n_val: int
    n_pos: int
    searched: Tuple[float, float]


def _nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    with torch.no_grad():
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits / max(temperature, 1e-6), labels
        )
    return float(loss)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    min_temperature: float = 1.0,
    grid: int = 400,
    span: Tuple[float, float] = (0.05, 50.0),
) -> TemperatureFit:
    """Grid-search the scalar T that minimises NLL, then clamp from below.

    A 400-point log-spaced grid on one parameter is deterministic, has no
    optimiser state, and resolves T to ~1% — LBFGS on a single scalar buys
    nothing here except a failure mode.  ``T`` is then clamped to
    ``--min-temperature`` (default 1.0): the specification asks for *softening*,
    and a T below 1 would mean the model is under-confident, which the report
    flags explicitly rather than silently sharpening.
    """
    if logits.numel() == 0:
        return TemperatureFit(1.0, 1.0, float("nan"), float("nan"), float("nan"),
                              0, 0, span)
    lo, hi = span
    candidates = [
        math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (grid - 1))
        for i in range(grid)
    ]
    nlls = [_nll(logits, labels, t) for t in candidates]
    best_i = min(range(len(candidates)), key=lambda i: (nlls[i], candidates[i]))
    raw = candidates[best_i]
    applied = max(raw, float(min_temperature))
    return TemperatureFit(
        raw_optimum=raw,
        applied=applied,
        nll_before=_nll(logits, labels, 1.0),
        nll_after=_nll(logits, labels, applied),
        nll_at_raw=nlls[best_i],
        n_val=int(labels.numel()),
        n_pos=int(labels.sum().item()),
        searched=span,
    )


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ReliabilityBin:
    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return self.accuracy - self.mean_confidence


def ece(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 15,
) -> Tuple[float, List[ReliabilityBin]]:
    """Expected Calibration Error with equal-width confidence bins.

    Confidence is ``max(p, 1-p)`` and correctness is ``(p >= 0.5) == label``, the
    standard multiclass reduction to binary.  Empty bins are dropped from both
    the ECE sum and the returned table.
    """
    if not probs:
        return float("nan"), []
    edges = [i / n_bins for i in range(n_bins + 1)]
    sums = [[0, 0.0, 0] for _ in range(n_bins)]   # [count, conf_sum, correct]
    for p, y in zip(probs, labels):
        conf = max(float(p), 1.0 - float(p))
        correct = 1 if (float(p) >= 0.5) == bool(y) else 0
        idx = min(int(conf * n_bins), n_bins - 1)
        sums[idx][0] += 1
        sums[idx][1] += conf
        sums[idx][2] += correct

    total = len(probs)
    value = 0.0
    rows: List[ReliabilityBin] = []
    for i, (count, conf_sum, correct) in enumerate(sums):
        if count == 0:
            continue
        mean_conf = conf_sum / count
        acc = correct / count
        value += count / total * abs(acc - mean_conf)
        rows.append(
            ReliabilityBin(
                low=edges[i], high=edges[i + 1], count=count,
                mean_confidence=mean_conf, accuracy=acc,
            )
        )
    return value, rows


def rejection_rate(probs: Sequence[float], threshold: float = 0.5) -> float:
    """Fraction of (all-negative) probes correctly scored below ``threshold``."""
    if not probs:
        return float("nan")
    return sum(1 for p in probs if p < threshold) / len(probs)


# --------------------------------------------------------------------------- #
# Logit extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_logits(
    model: RelationAwareClosurePredictor,
    h: torch.Tensor,
    score_by_rel: Dict[str, torch.Tensor],
    samples: Sequence[Sample],
) -> torch.Tensor:
    """Raw logits for a labelled sample list, using the full (N, N) matrices."""
    if not samples:
        return torch.zeros(0)
    out = torch.empty(len(samples), dtype=torch.float32)
    buckets: Dict[str, List[int]] = collections.defaultdict(list)
    for i, sample in enumerate(samples):
        buckets[sample.relation].append(i)
    for relation, indices in buckets.items():
        matrix = score_by_rel.get(relation)
        if matrix is None:
            out[torch.tensor(indices, dtype=torch.long)] = 0.0
            continue
        pos = torch.tensor(indices, dtype=torch.long)
        rows = torch.tensor([samples[i].src for i in indices], dtype=torch.long)
        cols = torch.tensor([samples[i].dst for i in indices], dtype=torch.long)
        out[pos] = matrix[rows, cols]
    return out


@torch.no_grad()
def triple_logits(
    model: RelationAwareClosurePredictor,
    h: torch.Tensor,
    triples: Sequence[Tuple[int, int, str]],
) -> Dict[str, float]:
    """logit for a handful of (src_idx, dst_idx, relation) probes."""
    out: Dict[str, float] = {}
    cache: Dict[str, torch.Tensor] = {}
    for src, dst, relation in triples:
        if relation not in cache:
            cache[relation] = model.score_all(h, relation)
        out[f"{src}|{relation}|{dst}"] = float(cache[relation][src, dst])
    return out


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def _bar(value: float, width: int = 26) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def reliability_block(
    title: str,
    before: Sequence[float],
    after: Sequence[float],
    labels: Sequence[int],
    n_bins: int,
) -> List[str]:
    """Two-column reliability table (raw vs calibrated) for one evaluation set."""
    ece_before, rows_before = ece(before, labels, n_bins)
    ece_after, rows_after = ece(after, labels, n_bins)
    lines = [
        "",
        title,
        f"  ECE: raw={ece_before:.4f}   calibrated={ece_after:.4f}   "
        f"(n={len(labels)}, pos={sum(labels)})",
        f"  {'bucket':<14} {'n':>5} {'conf(raw)':>10} {'acc(raw)':>9} "
        f"{'conf(cal)':>10} {'acc(cal)':>9}",
    ]
    by_low_after = {round(r.low, 6): r for r in rows_after}
    for row in rows_before:
        other = by_low_after.get(round(row.low, 6))
        lines.append(
            f"  [{row.low:.2f},{row.high:.2f})  {row.count:>5} "
            f"{row.mean_confidence:>10.3f} {row.accuracy:>9.3f} "
            f"{(other.mean_confidence if other else float('nan')):>10.3f} "
            f"{(other.accuracy if other else float('nan')):>9.3f}"
        )
    return lines


def sanity_probe_report(
    graph: RelationGraph,
    labels: Dict[str, int],
    logits: Dict[str, float],
    temperature: float,
    specs: Sequence[Tuple[str, str, str]],
) -> List[str]:
    """Score a handful of hand-picked triples that must / must not fire."""
    lines = []
    for src_word, relation, dst_word in specs:
        src = labels.get(src_word)
        dst = labels.get(dst_word)
        if src is None or dst is None:
            lines.append(f"  {src_word:<8} --{relation:<14}--> {dst_word:<10}  (节点缺失，跳过)")
            continue
        src_idx = graph.id_to_index.get(src)
        dst_idx = graph.id_to_index.get(dst)
        if src_idx is None or dst_idx is None:
            lines.append(f"  {src_word:<8} --{relation:<14}--> {dst_word:<10}  (不在图中)")
            continue
        key = f"{src_idx}|{relation}|{dst_idx}"
        logit = logits.get(key)
        if logit is None:
            lines.append(f"  {src_word:<8} --{relation:<14}--> {dst_word:<10}  (该关系无评分)")
            continue
        raw = float(torch.sigmoid(torch.tensor(logit)))
        cal = float(calibrated_probs(torch.tensor([logit]), temperature)[0])
        flag = "" if raw < 0.5 else "  <-- 饱和"
        lines.append(
            f"  {src_word:<8} --{relation:<14}--> {dst_word:<10} "
            f"logit={logit:+7.3f}  raw={raw:.4f}  calibrated={cal:.4f}{flag}"
        )
    return lines


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train the relation-aware GAT with cross-POS negatives and "
                    "apply temperature scaling for calibrated inference."
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
                        help="每个正样本的负样本总数（默认 6）")
    parser.add_argument("--hard-ratio", type=float, default=0.4)
    parser.add_argument("--medium-ratio", type=float, default=0.3)
    parser.add_argument("--easy-ratio", type=float, default=0.2)
    parser.add_argument("--cross-pos-ratio", type=float, default=0.1,
                        help="cross-POS 负样本占比（默认 0.1；0 = 关闭，退化为旧协议）")
    parser.add_argument("--pos-min-share", type=float, default=0.05,
                        help="某词性在 closure 中占比达到该值才算「合法目标词性」")
    parser.add_argument("--temperature-tuning", dest="temperature_tuning",
                        action="store_true", default=True,
                        help="训练后在 val 集上拟合温度（默认开启）")
    parser.add_argument("--no-temperature-tuning", dest="temperature_tuning",
                        action="store_false",
                        help="关闭温度标定，仅训练 cross-POS 负样本")
    parser.add_argument("--min-temperature", type=float, default=1.0,
                        help="温度下限；默认 1.0（只允许软化，不允许锐化）")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--probe-per-source", type=int, default=4,
                        help="每个 (held-out 源, 关系) 生成的 cross-POS 探针数量")
    parser.add_argument("--freeze-gat-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pos-weight", action="store_true",
                        help="关闭 pos_weight（默认 = neg/pos）。关闭会改变正负样本的"
                             "先验，仅用于标定消融。")
    parser.add_argument("--compare-checkpoint", type=pathlib.Path,
                        default=root / DEFAULT_COMPARE_CHECKPOINT,
                        help="对照用的旧 checkpoint（无 cross-POS）")
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
    for name in ("hard_ratio", "medium_ratio", "easy_ratio", "cross_pos_ratio"):
        value = getattr(args, name)
        if not (0.0 <= value <= 1.0):
            parser.error(f"--{name.replace('_', '-')} 必须在 [0, 1]")
    if args.hard_ratio + args.medium_ratio + args.easy_ratio + args.cross_pos_ratio <= 0:
        parser.error("四个负样本比例不能全为 0")
    if not (0.0 <= args.pos_min_share <= 1.0):
        parser.error("--pos-min-share 必须在 [0, 1]")
    if args.min_temperature <= 0:
        parser.error("--min-temperature 必须 > 0")
    if args.ece_bins < 1:
        parser.error("--ece-bins 必须 >= 1")
    if args.probe_per_source < 1:
        parser.error("--probe-per-source 必须 >= 1")
    if args.freeze_gat_epochs < 0:
        parser.error("--freeze-gat-epochs 必须 >= 0")
    if args.patience < 1:
        parser.error("--patience 必须 >= 1")
    if args.seed < 0:
        parser.error("--seed 必须 >= 0")
    return args


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass

    args = parse_args(argv)
    t0 = time.time()
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
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='closure'"
        ).fetchone():
            print("No closure data found, run apply_seeds.py first", file=sys.stderr)
            return 1
        deep_total = conn.execute(
            "SELECT COUNT(*) FROM closure WHERE depth >= 2"
        ).fetchone()[0]
        if deep_total == 0:
            print("No closure data found, run apply_seeds.py first", file=sys.stderr)
            return 1
        positives, facts = load_positives(conn, graph)
        target_pos = load_relation_target_pos(conn)
    finally:
        conn.close()

    allowed = build_pos_guard(target_pos, args.pos_min_share)
    active_relations = sorted({p.relation for p in positives})
    allowed_nodes, wrong_nodes = wrong_pos_index(graph, allowed, active_relations)

    # ---- 切分与负样本 ----------------------------------------------------- #
    positives_by_split, sources_by_split, mode = split_by_source(
        positives, seed=args.seed
    )
    pools = build_negative_pools(positives, facts)

    ratios = {
        "hard": args.hard_ratio,
        "medium": args.medium_ratio,
        "easy": args.easy_ratio,
        "cross": args.cross_pos_ratio,
    }
    quota = allocate_tiers(args.neg_per_pos, ratios)

    rng = random.Random(args.seed + 7)
    data: Dict[str, Dict[str, object]] = {}
    sampling: Dict[str, CalibSamplingStats] = {}
    for name in SPLIT_NAMES:
        pos_list = positives_by_split[name]
        neg_list, st = sample_negatives_crosspos(
            pos_list, graph, facts, pools, quota, wrong_nodes, rng
        )
        data[name] = {
            "pos": pos_list,
            "neg": neg_list,
            "samples": list(pos_list) + neg_list,
        }
        sampling[name] = st

    probe_rng = random.Random(args.seed + 11)
    ood_probe = build_ood_probe(
        positives_by_split["test"], graph, facts, wrong_nodes,
        args.probe_per_source, probe_rng,
    )

    train_samples: List[Sample] = data["train"]["samples"]        # type: ignore[assignment]
    val_samples: List[Sample] = data["val"]["samples"]            # type: ignore[assignment]
    test_samples: List[Sample] = data["test"]["samples"]          # type: ignore[assignment]
    test_pos: List[Sample] = data["test"]["pos"]                  # type: ignore[assignment]

    train_pos = sum(1 for s in train_samples if s.label == 1)
    train_neg = len(train_samples) - train_pos
    pos_weight = 1.0 if args.no_pos_weight else train_neg / max(train_pos, 1)
    relation_counts = dict(collections.Counter(s.relation for s in train_samples if s.label == 1))

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
    model = CalibratedClosurePredictor(
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

    # ---- 训练前摘要 ------------------------------------------------------- #
    print("=" * 104)
    print("Logos Transformer — calibrated closure prediction (cross-POS negatives)")
    print("=" * 104)
    print(f"version={CALIB_VERSION} (架构来自 {RELAWARE_VERSION})")
    print(f"db={db_path}")
    print(f"graph: {graph.stats['nodes']} nodes / {graph.stats['semantic_edges']} "
          f"direct edges / {graph.stats['message_edges']} message edges")
    print(f"split: {mode} | sources={len({s.src for s in positives})} | "
          f"positives={len(positives)}")
    print()
    print("  [关系 -> 合法目标词性（从 closure 读出，非硬编码）]")
    for relation in active_relations:
        counts = target_pos.get(relation, {})
        total = sum(counts.values()) or 1
        dist = "  ".join(
            f"{pos}:{count / total * 100:.0f}%"
            for pos, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        n_wrong = len(wrong_nodes.get(relation, []))
        print(f"    {relation:<16} {dist:<24} -> 允许 {','.join(sorted(allowed.get(relation, ()))) or '(无)'}"
              f" | cross-POS 候选 {n_wrong:>3}")
    print()
    for name in SPLIT_NAMES:
        pos_list: List[Sample] = data[name]["pos"]                 # type: ignore[assignment]
        samples: List[Sample] = data[name]["samples"]              # type: ignore[assignment]
        st = sampling[name]
        depths = dict(sorted(collections.Counter(s.depth for s in pos_list).items()))
        mix = " ".join(f"{tier}={st.used[tier]}" for tier in NEG_TIERS)
        print(f"  {name:<5} sources={len(sources_by_split[name]):>3}  "
              f"pos={len(pos_list):>3}  depth={depths}  samples={len(samples):>4}  "
              f"neg mix: {mix}")
        print(f"        short_h={st.short_hard} short_cross={st.short_cross} "
              f"(zero_cross={st.zero_cross}) missing={st.shortfall} "
              f"unique_pairs={st.unique_pairs}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train pos_weight={pos_weight:.2f} | {args.layers}x{args.heads} heads, "
          f"hidden={args.hidden}, params={n_params:,}")
    print(f"heads: {head_summary['n_relation_types']} types | "
          f"dedicated={len(head_summary['dedicated'])} | "
          f"gated={len(head_summary['gated'])} | "
          f"unseen->shared={len(head_summary['unseen'])}")
    print(f"per positive: " + " / ".join(f"{quota[t]} {t}" for t in NEG_TIERS))
    print(f"cross-POS by relation (train): "
          f"{sampling['train'].cross_by_relation or '(none)'}")
    print(f"OOD probe: {len(ood_probe)} cross-POS pairs from "
          f"{len({p.src for p in ood_probe})} held-out sources")
    print("-" * 104)

    # ---- 训练 ------------------------------------------------------------- #
    output_dir.mkdir(parents=True, exist_ok=True)
    untuned_path = output_dir / "closure_gat_calibrated_untuned.pt"
    tuned_path = output_dir / "closure_gat_calibrated_best.pt"

    freeze_active = args.freeze_gat_epochs > 0
    if freeze_active:
        for param in model.gat.parameters():
            param.requires_grad = False

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epochs_run = 0
    batch_rng = random.Random(args.seed + 1)
    train_curve: List[Tuple[int, float, float, float, float]] = []

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        if freeze_active and epoch == args.freeze_gat_epochs + 1:
            for param in model.gat.parameters():
                param.requires_grad = True
            freeze_active = False
            print(f"[unfreeze] GAT encoder unfrozen at epoch {epoch}")

        model.train()
        if freeze_active:
            model.gat.eval()

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
                embeddings, pairs_tensor(batch), [s.relation for s in batch]
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
        if math.isnan(val_loss):
            val_loss = train_loss
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), untuned_path)
        else:
            stale += 1

        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:>3}/{args.epochs} | train loss={train_loss:.4f} "
              f"acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"lr={lr:.2e}")
        train_curve.append((epoch, train_loss, val_loss, train_acc, val_acc))

        if stale >= args.patience:
            print(f"early stopping at epoch {epoch} "
                  f"(val loss stale for {stale} epoch(s))")
            break

    stopped_early = epochs_run < args.epochs
    if not untuned_path.exists():
        print("训练没有产出 checkpoint（可能一次都没改进）", file=sys.stderr)
        return 1
    state = torch.load(untuned_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ---- 温度标定 --------------------------------------------------------- #
    h, scores_by_rel, _attentions = encode_and_score(model, graph, active_relations)
    val_logits = sample_logits(model, h, scores_by_rel, val_samples)
    val_labels = torch.tensor([float(s.label) for s in val_samples])
    if args.temperature_tuning:
        fit = fit_temperature(val_logits, val_labels, args.min_temperature)
    else:
        fit = TemperatureFit(1.0, 1.0, _nll(val_logits, val_labels, 1.0),
                             _nll(val_logits, val_labels, 1.0),
                             _nll(val_logits, val_labels, 1.0),
                             int(val_labels.numel()), int(val_labels.sum().item()),
                             (1.0, 1.0))
    T = fit.applied

    # ---- 评测：分布内 ----------------------------------------------------- #
    metrics: Dict[str, Dict[str, object]] = {}
    for name in SPLIT_NAMES:
        metrics[name] = eval_bucket_relaware(
            model, graph, facts, data[name]["samples"], criterion,  # type: ignore[arg-type]
            h, scores_by_rel,
        )
    hard_by_split = {
        name: hard_metrics(scores_by_rel, pools, data[name]["pos"])  # type: ignore[arg-type]
        for name in SPLIT_NAMES
    }
    test_logits = sample_logits(model, h, scores_by_rel, test_samples)
    test_labels = [int(s.label) for s in test_samples]
    raw_test_probs = torch.sigmoid(test_logits).tolist()
    cal_test_probs = calibrated_probs(test_logits, T).tolist()

    # ---- 评测：OOD（cross-POS 探针）-------------------------------------- #
    probe_logits = sample_logits(model, h, scores_by_rel, ood_probe)
    raw_probe_probs = torch.sigmoid(probe_logits).tolist()
    cal_probe_probs = calibrated_probs(probe_logits, T).tolist()

    probe_by_rel: Dict[str, List[float]] = collections.defaultdict(list)
    probe_by_rel_cal: Dict[str, List[float]] = collections.defaultdict(list)
    for sample, raw, cal in zip(ood_probe, raw_probe_probs, cal_probe_probs):
        probe_by_rel[sample.relation].append(raw)
        probe_by_rel_cal[sample.relation].append(cal)

    # 混合集：test 正样本 + 全部 cross-POS 探针（模仿「写回时遇到的真分布」）
    shift_labels = [1] * len(test_pos) + [0] * len(ood_probe)
    shift_logits = torch.cat([
        sample_logits(model, h, scores_by_rel, test_pos), probe_logits
    ])
    raw_shift = torch.sigmoid(shift_logits).tolist()
    cal_shift = calibrated_probs(shift_logits, T).tolist()

    # ---- 对照：旧 checkpoint（无 cross-POS）-------------------------------- #
    compare: Optional[Dict[str, object]] = None
    if compare_path.exists():
        try:
            compare = _evaluate_compare(
                compare_path, graph, facts, data, ood_probe, pools, args,
            )
        except Exception as exc:  # pragma: no cover - 结构不匹配
            print(f"[warning] 对照 checkpoint 评测失败：{exc}", file=sys.stderr)

    # ---- 控制台报表 ------------------------------------------------------- #
    print("-" * 104)
    print("in-distribution test metrics (ranking is unchanged by temperature)")
    print(f"{'model':<26} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7} "
          f"{'hardMRR':>8} {'prior':>8} {'margin':>8} {'K':>5}")
    test_m = metrics["test"]
    test_h = hard_by_split["test"]
    if compare is not None:
        cm, ch = compare["metrics"], compare["hard"]
        print(f"{'relaware (no cross-POS)':<26} {_fmt(cm['mrr']):>7} "
              f"{_fmt(cm['hits@1']):>7} {_fmt(cm['hits@3']):>7} "
              f"{_fmt(cm['hits@10']):>7} {_fmt(ch.model_mrr):>8} "
              f"{_fmt(ch.prior_mrr):>8} {ch.margin:+.4f} {ch.mean_k:>5.1f}")
    print(f"{'calibrated (cross-POS)':<26} {_fmt(test_m['mrr']):>7} "
          f"{_fmt(test_m['hits@1']):>7} {_fmt(test_m['hits@3']):>7} "
          f"{_fmt(test_m['hits@10']):>7} {_fmt(test_h.model_mrr):>8} "
          f"{_fmt(test_h.prior_mrr):>8} {test_h.margin:+.4f} {test_h.mean_k:>5.1f}")
    print(f"best epoch (val loss)={best_epoch} | val loss={best_val:.4f} | "
          f"epochs_run={epochs_run} | early={stopped_early}")
    print("-" * 104)
    print("per-relation test metrics:")
    for rel in sorted(test_m["by_relation"]):                       # type: ignore[union-attr]
        fm = test_m["by_relation"][rel]                             # type: ignore[index]
        hm = test_h.by_relation.get(rel, {})
        print(f"  {rel:<14} n={int(fm['n']):>2} fullMRR={_fmt(fm['mrr'])} "
              f"H@1={_fmt(fm['hits@1'])} | hardMRR={_fmt(hm.get('model_mrr'))} "
              f"margin={hm.get('margin', float('nan')):+.4f}")
    print("-" * 104)
    print(f"temperature: raw optimum={fit.raw_optimum:.4f} -> applied T={T:.4f} "
          f"({'clamped' if T > fit.raw_optimum + 1e-9 else 'as fitted'})")
    print(f"  val NLL: before={fit.nll_before:.4f} after={fit.nll_after:.4f} "
          f"(n={fit.n_val}, pos={fit.n_pos})")
    print(f"  [注] T>1 只压缩概率，不改任何排序：logit 的正负号决定 0.5 阈值，"
          f"故 OOD 拒绝率只能由 cross-POS 训练改善。")

    for title, probs_before, probs_after, labels in (
        ("reliability — test split (in-distribution)",
         raw_test_probs, cal_test_probs, test_labels),
        ("reliability — shifted set (test positives + cross-POS probes)",
         raw_shift, cal_shift, shift_labels),
    ):
        for line in reliability_block(title, probs_before, probs_after, labels,
                                     args.ece_bins):
            print(line)
        ece_raw, _ = ece(probs_before, labels, args.ece_bins)
        ece_cal, _ = ece(probs_after, labels, args.ece_bins)
        print(f"  {'':<2}ECE {ece_raw:.4f} -> {ece_cal:.4f}")
    if compare is not None:
        print(f"\n  [旧模型（无 cross-POS）同一口径] "
              f"ECE test={compare['ece_test']:.4f} "
              f"shifted={compare['ece_shift']:.4f}")

    print()
    print("OOD rejection (cross-POS probes scored < 0.5; higher is better)")
    overall_raw = rejection_rate(raw_probe_probs)
    overall_cal = rejection_rate(cal_probe_probs)
    print(f"  {'all relations':<16} n={len(ood_probe):>4} "
          f"raw={overall_raw:.4f} calibrated={overall_cal:.4f} "
          f"(identical by construction; T is monotone)")
    if compare is not None:
        cmp_probe = compare.get("probe_probs") or []
        if cmp_probe:
            print(f"  {'no cross-POS':<16} n={len(cmp_probe):>4} "
                  f"raw={rejection_rate(cmp_probe):.4f}  <- 旧模型（同一批探针）")
    for relation in sorted(probe_by_rel):
        raw = rejection_rate(probe_by_rel[relation])
        cal = rejection_rate(probe_by_rel_cal[relation])
        old = ""
        if compare is not None:
            cmp_by_rel = compare.get("probe_by_rel") or {}
            if relation in cmp_by_rel:
                old = f" old={rejection_rate(cmp_by_rel[relation]):.4f}"
        print(f"  {relation:<16} n={len(probe_by_rel[relation]):>4} "
              f"raw={raw:.4f} calibrated={cal:.4f}{old}")

    print()
    print("sanity probes (must stay high: 猫 is_a 生物 / must drop: 蛇 is_a 潜水)")
    word_to_sid: Dict[str, int] = {}
    for idx, sid in enumerate(graph.node_ids):
        word_to_sid.setdefault(graph.labels[idx], sid)
    probe_triples = []
    for src_word, relation, dst_word in SANITY_SPECS:
        src_sid = word_to_sid.get(src_word)
        dst_sid = word_to_sid.get(dst_word)
        if src_sid is None or dst_sid is None:
            continue
        if src_sid not in graph.id_to_index or dst_sid not in graph.id_to_index:
            continue
        probe_triples.append(
            (graph.id_to_index[src_sid], graph.id_to_index[dst_sid], relation)
        )
    sanity_logits = triple_logits(model, h, probe_triples)
    sanity_lines = sanity_probe_report(
        graph, word_to_sid, sanity_logits, T, SANITY_SPECS
    )
    for line in sanity_lines:
        print(line)
    old_sanity_lines: List[str] = []
    if compare is not None and compare.get("sanity"):
        old_sanity_lines = sanity_probe_report(
            graph, word_to_sid, compare["sanity"], 1.0, SANITY_SPECS  # type: ignore[arg-type]
        )
        print("  [旧模型（无 cross-POS）同一批探针]")
        for line in old_sanity_lines:
            print(line)

    # ---- 写 checkpoint 与报告 -------------------------------------------- #
    with torch.no_grad():
        model.temperature.fill_(float(T))
    torch.save(model.state_dict(), tuned_path)

    ece_test_raw, _rows_test_raw = ece(raw_test_probs, test_labels, args.ece_bins)
    ece_test_cal, _ = ece(cal_test_probs, test_labels, args.ece_bins)
    ece_shift_raw, _rows_shift_raw = ece(raw_shift, shift_labels, args.ece_bins)
    ece_shift_cal, _ = ece(cal_shift, shift_labels, args.ece_bins)

    eval_sets = {
        "test split (in-distribution)": (raw_test_probs, cal_test_probs, test_labels),
        "shifted set (test positives + cross-POS probes)": (
            raw_shift, cal_shift, shift_labels,
        ),
    }
    sanity_by_key = {
        f"{w1} {rel} {w2}": sanity_logits.get(
            f"{graph.id_to_index[word_to_sid[w1]]}|{rel}|"
            f"{graph.id_to_index[word_to_sid[w2]]}"
        ) if (w1 in word_to_sid and w2 in word_to_sid) else None
        for w1, rel, w2 in SANITY_SPECS
    }

    calibration_record: Dict[str, object] = {
        "version": CALIB_VERSION,
        "architecture": RELAWARE_VERSION,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "db": str(db_path),
        "seed": args.seed,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_params": n_params,
        "negatives_per_positive": quota,
        "cross_pos_ratio": args.cross_pos_ratio,
        "pos_weight": pos_weight,
        "n_cross_pos_train": sampling["train"].used.get("cross", 0),
        "cross_pos_by_relation": sampling["train"].cross_by_relation,
        "temperature": {
            "raw_optimum": fit.raw_optimum,
            "applied": T,
            "min_temperature": args.min_temperature,
            "nll_before": fit.nll_before,
            "nll_after": fit.nll_after,
            "fit_split": "val",
            "n_val": fit.n_val,
            "n_pos_val": fit.n_pos,
        },
        "ece": {
            "test_in_distribution": {"raw": ece_test_raw, "calibrated": ece_test_cal},
            "shifted_test_positives_plus_probes": {
                "raw": ece_shift_raw, "calibrated": ece_shift_cal,
            },
            "n_bins": args.ece_bins,
        },
        "ood_rejection_at_0.5": {
            "n_probes": len(ood_probe),
            "raw": overall_raw,
            "calibrated": overall_cal,
            "by_relation": {
                rel: rejection_rate(probs) for rel, probs in sorted(probe_by_rel.items())
            },
        },
        "test_metrics": {
            "mrr": float(test_m["mrr"]),                 # type: ignore[arg-type]
            "hits@1": float(test_m["hits@1"]),           # type: ignore[arg-type]
            "hits@3": float(test_m["hits@3"]),           # type: ignore[arg-type]
            "hard_mrr": float(test_h.model_mrr),
            "hard_prior_mrr": float(test_h.prior_mrr),
            "hard_margin": float(test_h.margin),
            "hard_n_queries": int(test_h.n_queries),
        },
        "sanity": sanity_by_key,
        "checkpoints": {
            "tuned": str(tuned_path),
            "untuned": str(untuned_path),
        },
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = _build_report(
        args=args, graph=graph, mode=mode, data=data, sampling=sampling,
        quota=quota, fit=fit, T=T,
        metrics=metrics, hard_by_split=hard_by_split,
        eval_sets=eval_sets,
        ece_scalars={
            "test_in": (ece_test_raw, ece_test_cal),
            "shifted": (ece_shift_raw, ece_shift_cal),
        },
        ood=(len(ood_probe), overall_raw, overall_cal, probe_by_rel),
        compare=compare, sanity_lines=sanity_lines,
        old_sanity_lines=old_sanity_lines, curve=train_curve,
        epochs_run=epochs_run, best_epoch=best_epoch, best_val=best_val,
        stopped_early=stopped_early, tuned_path=tuned_path,
        untuned_path=untuned_path, elapsed=time.time() - t0,
        target_pos=target_pos, allowed=allowed, wrong_nodes=wrong_nodes,
    )
    report_path = output_dir / "training_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    print("-" * 104)
    print(f"checkpoint (T included) : {tuned_path}")
    print(f"checkpoint (T=1 control) : {untuned_path}")
    print(f"calibration record       : {output_dir / 'calibration.json'}")
    print(f"report                   : {report_path}")
    print(f"用时 {time.time() - t0:.1f}s")
    return 0


def _evaluate_compare(
    path: pathlib.Path,
    graph: RelationGraph,
    facts: Facts,
    data: Dict[str, Dict[str, object]],
    ood_probe: Sequence[Sample],
    pools: NegativePools,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Evaluate an older (non-calibrated) checkpoint on the very same sets.

    Same split, same hard pools, same probes — so the only variable between the
    two rows of the comparison table is the training objective.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    emb = state.get("gat.node_embedding.weight")
    if emb is None or int(emb.shape[0]) != len(graph.node_ids):
        raise ValueError("checkpoint 与当前图不匹配")
    head_names = sorted(k.split("W_r.", 1)[1] for k in state if k.startswith("W_r."))
    counts = state.get("relation_count")
    count_list = [int(c) for c in counts.tolist()] if counts is not None else []
    if len(count_list) != len(head_names):
        count_list = [0] * len(head_names)
    hidden = int(emb.shape[1])
    base = build_model(
        graph,
        hidden_dim=hidden,
        num_heads=int(state["gat.layers.0.weight"].shape[1]),
        num_layers=len(
            {int(k.split(".")[2]) for k in state if k.startswith("gat.layers.")}
        ),
        dropout=0.1,
        seed=args.seed,
    )
    model = RelationAwareClosurePredictor(
        base, hidden, graph.relation_names,
        {name: count_list[i] for i, name in enumerate(head_names)},
        head_relations=head_names,
    )
    model.load_state_dict(state)
    model.eval()

    relations = sorted({s.relation for s in data["test"]["samples"]})  # type: ignore[union-attr]
    h, scores_by_rel, _ = encode_and_score(model, graph, relations)
    criterion = nn.BCEWithLogitsLoss()
    metrics = eval_bucket_relaware(
        model, graph, facts, data["test"]["samples"], criterion, h, scores_by_rel
    )
    hard = hard_metrics(scores_by_rel, pools, data["test"]["pos"])  # type: ignore[arg-type]

    logits = sample_logits(model, h, scores_by_rel, ood_probe)
    probs = torch.sigmoid(logits).tolist()
    by_rel: Dict[str, List[float]] = collections.defaultdict(list)
    for sample, prob in zip(ood_probe, probs):
        by_rel[sample.relation].append(prob)

    # 同一套 ECE 口径，才能和本脚本的新模型逐格对比。
    test_logits_cmp = sample_logits(model, h, scores_by_rel, data["test"]["samples"])  # type: ignore[arg-type]
    ece_test, _ = ece(
        torch.sigmoid(test_logits_cmp).tolist(),
        [int(s.label) for s in data["test"]["samples"]],  # type: ignore[union-attr]
        args.ece_bins,
    )
    pos_logits_cmp = sample_logits(model, h, scores_by_rel, data["test"]["pos"])  # type: ignore[arg-type]
    shift_labels = [1] * len(data["test"]["pos"]) + [0] * len(ood_probe)  # type: ignore[arg-type]
    ece_shift, _ = ece(
        torch.sigmoid(torch.cat([pos_logits_cmp, logits])).tolist(),
        shift_labels, args.ece_bins,
    )

    sid_of: Dict[str, int] = {}
    for idx, sid in enumerate(graph.node_ids):
        sid_of.setdefault(graph.labels[idx], sid)
    triples = []
    for src_word, relation, dst_word in SANITY_SPECS:
        if src_word in sid_of and dst_word in sid_of:
            triples.append((
                graph.id_to_index[sid_of[src_word]],
                graph.id_to_index[sid_of[dst_word]],
                relation,
            ))
    return {
        "metrics": metrics,
        "hard": hard,
        "probe_probs": probs,
        "probe_by_rel": {k: v for k, v in by_rel.items()},
        "sanity": triple_logits(model, h, triples),
        "ece_test": ece_test,
        "ece_shift": ece_shift,
        "path": str(path),
    }


def _build_report(**kw) -> str:
    """Plain-text report; the console already showed the same numbers."""
    args: argparse.Namespace = kw["args"]
    graph: RelationGraph = kw["graph"]
    data = kw["data"]
    sampling = kw["sampling"]
    quota = kw["quota"]
    fit: TemperatureFit = kw["fit"]
    T: float = kw["T"]
    metrics = kw["metrics"]
    hard_by_split = kw["hard_by_split"]
    ood = kw["ood"]
    compare = kw["compare"]
    curve = kw["curve"]
    ece_scalars = kw["ece_scalars"]

    lines: List[str] = []
    add = lines.append
    add(f"Logos Transformer — 校准化推理训练报告 ({CALIB_VERSION})")
    add("=" * 100)
    add(f"db                : {args.db}")
    add(f"输出目录          : {args.output_dir}")
    add(f"split mode        : {kw['mode']}")
    add(f"nodes / direct edges : {graph.stats['nodes']} / {graph.stats['semantic_edges']}")
    add(f"seed / epochs     : {args.seed} / {kw['epochs_run']} "
        f"(best epoch {kw['best_epoch']}, early={kw['stopped_early']})")
    add(f"best val loss     : {kw['best_val']:.6f}")
    add(f"用时              : {kw['elapsed']:.1f}s")
    add("")
    add("## 1. 负样本配比")
    add("每正样本: " + " / ".join(f"{quota[t]} {t}" for t in NEG_TIERS)
        + f"  (total {sum(quota.values())})")
    add(f"比例    : hard={args.hard_ratio} medium={args.medium_ratio} "
        f"easy={args.easy_ratio} cross-POS={args.cross_pos_ratio}")
    add(f"{'split':<7}{'pos':>5}{'hard':>7}{'medium':>8}{'easy':>7}{'cross':>7}"
        f"{'short_h':>9}{'short_x':>9}{'missing':>9}")
    for name in SPLIT_NAMES:
        st = sampling[name]
        add(f"{name:<7}{len(data[name]['pos']):>5}"          # type: ignore[arg-type]
            f"{st.used.get('hard', 0):>7}{st.used.get('medium', 0):>8}"
            f"{st.used.get('easy', 0):>7}{st.used.get('cross', 0):>7}"
            f"{st.short_hard:>9}{st.short_cross:>9}{st.shortfall:>9}")
    add(f"train cross-POS by relation: {sampling['train'].cross_by_relation}")
    add("")
    add("## 2. 类型期望（从 closure 读出，非硬编码）")
    add(f"{'relation':<16}{'allowed POS':<14}{'closure 分布':<28}{'cross-POS 候选':>14}")
    relations_seen = {s.relation for name in SPLIT_NAMES for s in data[name]['pos']}  # type: ignore[union-attr]
    for relation, counts in sorted(kw["target_pos"].items()):
        if relation not in relations_seen:
            continue
        total = sum(counts.values()) or 1
        dist = " ".join(f"{p}:{c / total * 100:.0f}%" for p, c in
                        sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        allowed_txt = ",".join(sorted(kw["allowed"].get(relation, ()))) or "(无)"
        add(f"{relation:<16}{allowed_txt:<14}{dist:<28}"
            f"{len(kw['wrong_nodes'].get(relation, [])):>14}")
    add("")
    add("## 3. 温度标定")
    add(f"拟合集            : val (n={fit.n_val}, pos={fit.n_pos})")
    add(f"搜索区间          : [{fit.searched[0]}, {fit.searched[1]}]，400 点对数网格")
    add(f"raw NLL 最优 T    : {fit.raw_optimum:.6f}")
    add(f"实际采用 T        : {T:.6f}（下限 {args.min_temperature}）")
    add(f"val NLL           : {fit.nll_before:.6f} -> {fit.nll_after:.6f}")
    add("注意              : T 是严格单调变换 -> MRR / margin / 0.5 阈值两侧的归属"
        "全部不变；它只改概率数值。")
    add("")
    add("## 4. 校准质量")
    add(f"{'metric':<40}{'raw':>10}{'calibrated':>12}")
    add(f"{'ECE test (in-distribution)':<40}"
        f"{ece_scalars['test_in'][0]:>10.4f}{ece_scalars['test_in'][1]:>12.4f}")
    add(f"{'ECE shifted (positives + cross-POS probes)':<40}"
        f"{ece_scalars['shifted'][0]:>10.4f}{ece_scalars['shifted'][1]:>12.4f}")
    add(f"{'OOD rejection @0.5 (n=' + str(ood[0]) + ')':<40}"
        f"{ood[1]:>10.4f}{ood[2]:>12.4f}")
    if compare is not None and compare.get("probe_probs"):
        add(f"{'OOD rejection @0.5 — 旧模型（同一批探针）':<40}"
            f"{rejection_rate(compare['probe_probs']):>10.4f}        n/a")
        add(f"{'ECE test — 旧模型（无 cross-POS）':<40}"
            f"{compare['ece_test']:>10.4f}        n/a")
        add(f"{'ECE shifted — 旧模型（无 cross-POS）':<40}"
            f"{compare['ece_shift']:>10.4f}        n/a")
    add("")
    for title, (before, after, labels) in kw["eval_sets"].items():
        for line in reliability_block(
            f"reliability — {title}", before, after, labels, args.ece_bins
        ):
            add(line)
    add("")
    add("## 5. 排序质量（不受 T 影响）")
    test_m = metrics["test"]
    test_h = hard_by_split["test"]
    add(f"{'model':<28}{'MRR':>8}{'H@1':>8}{'H@3':>8}{'H@10':>8}"
        f"{'hardMRR':>9}{'prior':>8}{'margin':>9}{'K':>6}")
    if compare is not None:
        cm, ch = compare["metrics"], compare["hard"]
        add(f"{'relaware (no cross-POS)':<28}{_fmt(cm['mrr']):>8}"
            f"{_fmt(cm['hits@1']):>8}{_fmt(cm['hits@3']):>8}{_fmt(cm['hits@10']):>8}"
            f"{_fmt(ch.model_mrr):>9}{_fmt(ch.prior_mrr):>8}{ch.margin:>+9.4f}"
            f"{ch.mean_k:>6.1f}")
    add(f"{'calibrated (cross-POS)':<28}{_fmt(test_m['mrr']):>8}"
        f"{_fmt(test_m['hits@1']):>8}{_fmt(test_m['hits@3']):>8}"
        f"{_fmt(test_m['hits@10']):>8}{_fmt(test_h.model_mrr):>9}"
        f"{_fmt(test_h.prior_mrr):>8}{test_h.margin:>+9.4f}{test_h.mean_k:>6.1f}")
    add("")
    add(f"{'relation':<16}{'n':>4}{'fullMRR':>10}{'H@1':>8}{'hardMRR':>10}"
        f"{'margin':>9}{'probes':>8}{'reject':>8}")
    for rel in sorted(test_m["by_relation"]):                    # type: ignore[union-attr]
        fm = test_m["by_relation"][rel]                          # type: ignore[index]
        hm = test_h.by_relation.get(rel, {})
        probes = ood[3].get(rel) or []
        add(f"{rel:<16}{int(fm['n']):>4}{_fmt(fm['mrr']):>10}"
            f"{_fmt(fm['hits@1']):>8}{_fmt(hm.get('model_mrr')):>10}"
            f"{hm.get('margin', float('nan')):>+9.4f}{len(probes):>8}"
            f"{rejection_rate(probes):>8.4f}")
    add("")
    add("## 6. 语义探针（raw / 校准后）")
    lines.extend(kw["sanity_lines"])
    if kw["old_sanity_lines"]:
        add("[旧模型（无 cross-POS）]")
        lines.extend(kw["old_sanity_lines"])
    add("")
    add("## 7. 训练曲线（每 5 个 epoch 抽样）")
    add(f"{'epoch':>6}{'train_loss':>12}{'val_loss':>11}{'train_acc':>11}{'val_acc':>9}")
    for row in curve:
        if row[0] % 5 == 0 or row[0] == curve[-1][0]:
            add(f"{row[0]:>6}{row[1]:>12.4f}{row[2]:>11.4f}"
                f"{row[3]:>11.4f}{row[4]:>9.4f}")
    add("")
    add("## 8. 产物")
    add(f"tuned checkpoint   : {kw['tuned_path']}   (含 temperature 缓冲)")
    add(f"untuned checkpoint : {kw['untuned_path']}   (T=1，A/B 对照)")
    add("calibration.json   : 与 checkpoint 同目录")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
