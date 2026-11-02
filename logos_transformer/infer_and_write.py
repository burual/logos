#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end inference write-back for the Logos Transformer.

Pipeline
--------
    trained checkpoint -> candidate generation -> GAT forward + W_r scoring
        -> ``inferred_relations`` -> query verification (query_inferences.py)

Why a separate table
--------------------
``closure`` is **ground truth**: every row there is produced by the closure
triggers from a real ``relations`` edge (``closure_rebuild_queue`` ->
``closure_policy``).  It must stay reproducible from the seeds.  Model output is
a *hypothesis* and lives in ``inferred_relations``, where it can be scored,
filtered, marked ``verified`` by a human, and later promoted to a real edge by a
separate (deliberate) step.  Mixing the two would destroy the ability to re-derive
the database from seed files alone.

Candidate generation (avoiding O(N^2 * R))
-----------------------------------------
Scoring every (source, relation, target) triple of the 335-node graph would be
335 x 335 x 46 ~= 5.16M pairs.  Two structural strategies cut that to O(10^4):

* **A -- neighbour-of-neighbour**: a multi-hop closure result is, by
  construction, reachable in 2..k directed steps.  For each sense that declares
  at least one direct relation, take its undirected 1-hop neighbourhood, expand
  once more, drop the 1-hop nodes and the node itself, and score the remainder
  for every relation the source already participates in.
* **B -- direct-edge completion**: for each ``(source, relation)`` pair that
  exists in ``relations`` but has **no** depth>=2 ``closure`` row, score *all*
  nodes.  This catches sources whose inheritance is completely missing (the
  model is the only source of a hypothesis for them).

Both strategies only propose relations that the closure engine actually tracks
(relations with at least one depth>=2 ``closure`` row).  Scoring a non-transitive
relation such as ``synonym`` would be meaningless -- the closure engine never
derives it.

Schema deviations from the task description
-------------------------------------------
* ``senses`` has no ``name`` column and its primary key is ``sense_id`` (not
  ``id``).  Names live in ``words.word``; the foreign keys therefore reference
  ``senses(sense_id)``.
* ``epoch`` is written as NULL unless ``--epoch`` is given: the checkpoints are
  plain ``state_dict`` files, they carry no training metadata.

Usage
-----
    python logos_transformer/infer_and_write.py --dry-run          # score only
    python logos_transformer/infer_and_write.py --verify           # write + check
    python logos_transformer/infer_and_write.py --relation-subset can_do is_a
    python logos_transformer/infer_and_write.py --top-k 5 --dry-run

Exit codes
----------
    1  missing torch / database / checkpoint, or schema mismatch
    2  invalid command-line arguments
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sqlite3
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- #
# 依赖检查：torch 缺失时必须「先说人话再退出 1」，而不是抛 traceback。
# --------------------------------------------------------------------------- #
try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    print("[错误] 需要 torch。安装：pip install torch", file=sys.stderr)
    print("       本项目在受限环境下可用：把 CPU wheel 解压到 ._pylibs/ 并设置 PYTHONPATH。",
          file=sys.stderr)
    raise SystemExit(1)

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
# 同时放项目根与 logos_transformer/，这样两种导入风格都能命中。
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prototype_gat import RelationGraph, build_model, load_graph  # noqa: E402
from train_closure_relaware import (  # noqa: E402
    RELAWARE_VERSION,
    RelationAwareClosurePredictor,
)

DEFAULT_DB = _ROOT / "logos.db"

# Checkpoint 解析顺序：**覆盖率优先**。
# 1) checkpoints_relaware_ext —— 46 个 W_r 头（C1 全量重训），是本图的「多关系」模型；
# 2) checkpoints_relaware     —— 任务书给的名字，但该目录已被 C2a「隔离 can_do」跑覆盖，
#                                只剩 1 个 can_do 头，其余关系全部落到 W_shared。
# 自动解析会打印实际选中哪一个以及原因；显式传 --checkpoint 时永不替换。
DEFAULT_CHECKPOINT_CANDIDATES: Tuple[str, ...] = (
    "./logos_transformer/checkpoints_relaware_ext/closure_gat_relaware_best.pt",
    "./logos_transformer/checkpoints_relaware/closure_gat_relaware_best.pt",
)

# inferred_relations：模型假设的落库位置。刻意与 closure 分离。
# 偏差说明：senses 的主键列名是 sense_id；senses 没有 name 列（名称在 words.word）。
INFERRED_DDL = """
CREATE TABLE IF NOT EXISTS inferred_relations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id      INTEGER NOT NULL,
    relation_type  TEXT    NOT NULL,
    target_id      INTEGER NOT NULL,
    confidence     REAL    NOT NULL,
    model_version  TEXT    NOT NULL,
    epoch          INTEGER,
    depth_estimate INTEGER,
    verified       BOOLEAN DEFAULT FALSE,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (source_id <> target_id),
    FOREIGN KEY (source_id) REFERENCES senses(sense_id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES senses(sense_id) ON DELETE CASCADE,
    UNIQUE (source_id, relation_type, target_id)
);
"""

INFERRED_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_inferred_rel_conf "
    "ON inferred_relations(relation_type, confidence DESC);",
    "CREATE INDEX IF NOT EXISTS idx_inferred_source "
    "ON inferred_relations(source_id, confidence DESC);",
    "CREATE INDEX IF NOT EXISTS idx_inferred_unverified "
    "ON inferred_relations(verified) WHERE verified = 0;",
)

# 名字 -> 中文标签，仅用于报表可读性。
RELATION_LABEL = {
    "is_a": "上位（is_a）",
    "can_do": "能做什么（can_do）",
    "has_property": "有何属性（has_property）",
    "typically_has": "典型具备（typically_has）",
    "part_of": "属于（part_of）",
    "has_part": "组成部分（has_part）",
}


# --------------------------------------------------------------------------- #
# 数据库读取
# --------------------------------------------------------------------------- #
def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def load_sense_labels(conn: sqlite3.Connection) -> Dict[int, Tuple[str, str]]:
    """sense_id -> (word, pos)."""
    rows = conn.execute(
        """
        SELECT s.sense_id, w.word, s.pos
        FROM senses s JOIN words w ON w.word_id = s.word_id
        """
    ).fetchall()
    return {int(r[0]): (str(r[1]), str(r[2])) for r in rows}


def load_direct_edges(conn: sqlite3.Connection) -> List[Tuple[int, str, int]]:
    """Direct sense->sense edges that the graph export also uses."""
    rows = conn.execute(
        """
        SELECT src_id, relation, dst_id
        FROM relations
        WHERE src_kind = 'sense' AND dst_kind = 'sense' AND is_negated = 0
        ORDER BY relation_id
        """
    ).fetchall()
    return [(int(r[0]), str(r[1]), int(r[2])) for r in rows]


def load_closure_index(
    conn: sqlite3.Connection,
) -> Tuple[Set[Tuple[int, str, int]], Dict[Tuple[int, str], int]]:
    """Return (existing triples, max depth per (descendant, relation)).

    ``closure`` stores ``ancestor`` as the *destination* of the last edge, so a
    row reads "descendant --relation--> ancestor at depth d".
    """
    triples: Set[Tuple[int, str, int]] = set()
    max_depth: Dict[Tuple[int, str], int] = {}
    for relation, ancestor, descendant, depth in conn.execute(
        "SELECT relation, ancestor, descendant, depth FROM closure"
    ):
        key = (int(descendant), str(relation), int(ancestor))
        triples.add(key)
        d = int(depth)
        slot = (int(descendant), str(relation))
        if d > max_depth.get(slot, 0):
            max_depth[slot] = d
    return triples, max_depth


def closure_tracked_relations(conn: sqlite3.Connection) -> List[str]:
    """Relations the closure engine actually derives (>=1 depth>=2 row)."""
    if not _table_exists(conn, "closure"):
        return []
    return [
        str(r[0])
        for r in conn.execute(
            "SELECT DISTINCT relation FROM closure WHERE depth >= 2 ORDER BY relation"
        )
    ]


def load_relation_target_pos(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    """Observed (relation -> {target POS: count}) distribution in ``closure``.

    The whole point is that the type expectation is **read off the data**, not
    hand-written: whatever the seeds actually attach to ``can_do`` defines what a
    ``can_do`` target may look like.  On the current database the distribution is
    nearly deterministic (``is_a`` 100% noun, ``can_do`` 100% verb,
    ``part_of``/``has_part`` 100% noun, ``has_property`` 82% adjective).
    """
    out: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    for relation, pos in conn.execute(
        """
        SELECT cl.relation, s.pos
        FROM closure cl JOIN senses s ON s.sense_id = cl.ancestor
        """
    ):
        out[str(relation)][str(pos)] += 1
    return {relation: dict(counter) for relation, counter in out.items()}


def build_pos_guard(
    target_pos: Dict[str, Dict[str, int]],
    min_share: float,
) -> Dict[str, Set[str]]:
    """relation -> allowed target POS classes, keeping classes above ``min_share``."""
    guard: Dict[str, Set[str]] = {}
    for relation, counts in target_pos.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        guard[relation] = {
            pos for pos, count in counts.items() if count / total >= min_share
        }
    return guard


def apply_pos_guard(
    triples: Sequence[Tuple[int, str, int]],
    pos_of: Dict[int, str],
    guard: Dict[str, Set[str]],
) -> Tuple[List[Tuple[int, str, int]], List[Tuple[int, str, int]]]:
    """Split candidates into (kept, dropped) by the learned type expectation."""
    kept: List[Tuple[int, str, int]] = []
    dropped: List[Tuple[int, str, int]] = []
    for src, relation, target in triples:
        allowed = guard.get(relation)
        if allowed is None or pos_of.get(target) in allowed:
            kept.append((src, relation, target))
        else:
            dropped.append((src, relation, target))
    return kept, dropped


def limit_per_group(
    scored: Sequence[Tuple[int, str, int, float]],
    max_per_group: int,
) -> List[Tuple[int, str, int, float]]:
    """Keep at most ``max_per_group`` rows per (source, relation), highest first.

    This is the quality gate that matters: the model was trained and validated on
    **ranking** (MRR / margin), never calibrated as a probability, and its
    sigmoid saturates near 1.0 on many out-of-distribution targets.  Absolute
    confidence therefore cannot separate hypothesis #1 from hypothesis #7, but the
    ordering within one (source, relation) list is exactly what the protocol
    measured -- so we cap the list instead of trusting the number.
    """
    if max_per_group <= 0:
        return list(scored)
    grouped: Dict[Tuple[int, str], List[Tuple[int, str, int, float]]] = collections.defaultdict(list)
    for item in scored:
        grouped[(item[0], item[1])].append(item)
    out: List[Tuple[int, str, int, float]] = []
    for items in grouped.values():
        items.sort(key=lambda x: (-x[3], x[2]))
        out.extend(items[:max_per_group])
    out.sort(key=lambda x: (-x[3], x[1], x[0], x[2]))
    return out


# --------------------------------------------------------------------------- #
# Checkpoint 解析与装载
# --------------------------------------------------------------------------- #
def _torch_load(path: pathlib.Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # 旧版 torch 没有 weights_only
        return torch.load(path, map_location="cpu")


def inspect_checkpoint(state: Dict[str, "torch.Tensor"]) -> Dict[str, object]:
    """Recover the architecture from a plain ``state_dict`` (fully self-describing)."""
    if "gat.node_embedding.weight" not in state:
        raise SystemExit("[错误] checkpoint 里没有 gat.node_embedding.weight，不是本项目的模型。")
    emb = state["gat.node_embedding.weight"]
    num_nodes, hidden = int(emb.shape[0]), int(emb.shape[1])

    layer_ids = set()
    for key in state:
        if key.startswith("gat.layers."):
            layer_ids.add(int(key.split(".")[2]))
    num_layers = len(layer_ids)

    w0 = state["gat.layers.0.weight"]              # (E_types, heads, in_dim, head_dim)
    num_edge_types, num_heads = int(w0.shape[0]), int(w0.shape[1])

    # ParameterDict 的键形如 "W_r.is_a"；training 时顺序 = graph.relation_names 的字典序。
    head_names = sorted(
        key.split("W_r.", 1)[1] for key in state if key.startswith("W_r.")
    )
    counts = state.get("relation_count")
    count_list = [int(c) for c in counts.tolist()] if counts is not None else []

    return {
        "num_nodes": num_nodes,
        "hidden": hidden,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "num_edge_types": num_edge_types,
        "head_names": head_names,
        "counts": count_list,
    }


def resolve_checkpoint(
    explicit: Optional[pathlib.Path],
    wanted_relations: Sequence[str],
) -> Tuple[pathlib.Path, Dict[str, object], str]:
    """Pick a checkpoint and return (path, info, reason).

    Explicit ``--checkpoint`` always wins (the caller may know better); the
    automatic path prefers the checkpoint whose ``W_r`` heads cover the most of
    the requested relations.
    """
    candidates: List[pathlib.Path] = (
        [explicit] if explicit is not None
        else [_ROOT / c for c in DEFAULT_CHECKPOINT_CANDIDATES]
    )

    existing = [p for p in candidates if p.exists()]
    if not existing:
        tried = "\n".join(f"  - {p}" for p in candidates)
        print("[错误] 找不到任何可用的 checkpoint，试过：", file=sys.stderr)
        print(tried, file=sys.stderr)
        print("       先训练一次：python logos_transformer/train_closure_relaware.py",
              file=sys.stderr)
        raise SystemExit(1)
    if explicit is not None and len(existing) == 0:  # pragma: no cover - 上面已覆盖
        print(f"[错误] checkpoint 不存在：{explicit}", file=sys.stderr)
        raise SystemExit(1)

    wanted = set(wanted_relations)
    best: Optional[Tuple[int, pathlib.Path, Dict[str, object]]] = None
    for path in existing:
        info = inspect_checkpoint(_torch_load(path))
        coverage = len(wanted & set(info["head_names"])) if wanted else len(info["head_names"])
        if best is None or coverage > best[0]:
            best = (coverage, path, info)

    assert best is not None
    coverage, path, info = best
    if explicit is not None:
        reason = "由 --checkpoint 显式指定"
    else:
        reason = "自动解析：W_r 头对目标关系的覆盖率最高"
    missing = sorted(wanted - set(info["head_names"]))
    extra_note = ""
    if missing:
        extra_note = (
            f"；{len(missing)} 个目标关系无专属头，回落到 W_shared："
            + ", ".join(missing[:6])
            + (" ..." if len(missing) > 6 else "")
        )
    return path, info, reason + extra_note


def build_and_load(
    path: pathlib.Path,
    info: Dict[str, object],
    graph: RelationGraph,
    seed: int,
) -> RelationAwareClosurePredictor:
    """Rebuild ``RelationAwareClosurePredictor`` from the checkpoint's own shapes."""
    if int(info["num_nodes"]) != len(graph.node_ids):
        print(
            f"[错误] checkpoint 的节点数 {info['num_nodes']} 与数据库图的 "
            f"{len(graph.node_ids)} 不一致。用同一份 logos.db 重新训练后再推理。",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if int(info["num_edge_types"]) != graph.num_edge_types:
        print(
            f"[错误] checkpoint 的边类型数 {info['num_edge_types']} 与图的 "
            f"{graph.num_edge_types} 不一致（relation_types 表变了？）。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    head_names: List[str] = list(info["head_names"])  # type: ignore[arg-type]
    counts: List[int] = list(info["counts"])          # type: ignore[arg-type]
    if len(counts) != len(head_names):
        print(
            f"[错误] relation_count 长度 {len(counts)} 与 W_r 头数 {len(head_names)} 不一致，"
            "checkpoint 可能来自 --relation-subset 顺序与字典序不同的运行。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    base = build_model(
        graph,
        hidden_dim=int(info["hidden"]),
        num_heads=int(info["num_heads"]),
        num_layers=int(info["num_layers"]),
        dropout=0.1,
        seed=seed,
    )
    # 用 checkpoint 自己的头列表（而不是全词表）构造，才能 1:1 对齐权重。
    model = RelationAwareClosurePredictor(
        base,
        int(info["hidden"]),
        graph.relation_names,
        {name: counts[i] for i, name in enumerate(head_names)},
        head_relations=head_names,
    )
    try:
        model.load_state_dict(_torch_load(path))
    except RuntimeError as exc:
        print(f"[错误] 权重装载失败（结构不匹配）：{exc}", file=sys.stderr)
        raise SystemExit(1)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# 候选生成
# --------------------------------------------------------------------------- #
def _one_hop(
    edges: Iterable[Tuple[int, str, int]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[str]]]:
    """Undirected 1-hop neighbourhood + the relation set each sense declares."""
    neighbours: Dict[int, Set[int]] = collections.defaultdict(set)
    declared: Dict[int, Set[str]] = collections.defaultdict(set)
    for src, relation, dst in edges:
        neighbours[src].add(dst)
        neighbours[dst].add(src)
        declared[src].add(relation)
    return neighbours, declared


def generate_candidates_a(
    edges: Sequence[Tuple[int, str, int]],
    closure_triples: Set[Tuple[int, str, int]],
    relations: Sequence[str],
    max_neighbourhood: int = 60,
) -> Tuple[List[Tuple[int, str, int]], int]:
    """Strategy A: score (source, relation, 2-hop target) triples.

    Returns ``(candidates, n_sources_scanned)``.
    """
    neighbours, declared = _one_hop(edges)
    allowed = set(relations)
    out: List[Tuple[int, str, int]] = []
    scanned = 0
    for src in sorted(neighbours):
        rels = declared[src] & allowed
        if not rels:
            continue
        scanned += 1
        two_hop: Set[int] = set()
        for mid in neighbours[src]:
            two_hop |= neighbours.get(mid, set())
        # 二跳集合里剔除自身与直接邻居：那些已经在 closure 里（或本来就直连）。
        two_hop.discard(src)
        two_hop -= neighbours[src]
        if max_neighbourhood > 0 and len(two_hop) > max_neighbourhood:
            # 只保留度数最高的若干目标——枢纽节点（如「生物」）几乎对所有源成立，
            # 截断可避免单个枢纽把候选池淹没。按目标度数降序排序保证可复现。
            two_hop = set(
                sorted(two_hop, key=lambda t: (-len(neighbours.get(t, ())), t))[
                    :max_neighbourhood
                ]
            )
        for relation in sorted(rels):
            for target in two_hop:
                if (src, relation, target) in closure_triples:
                    continue
                out.append((src, relation, target))
    return out, scanned


def generate_candidates_b(
    edges: Sequence[Tuple[int, str, int]],
    closure_triples: Set[Tuple[int, str, int]],
    closure_max_depth: Dict[Tuple[int, str], int],
    relations: Sequence[str],
    all_nodes: Sequence[int],
) -> Tuple[List[Tuple[int, str, int]], int]:
    """Strategy B: for (source, relation) with no depth>=2 row, score every node."""
    allowed = set(relations)
    pairs: Set[Tuple[int, str]] = {(src, rel) for src, rel, _dst in edges if rel in allowed}
    missing = [
        (src, rel) for src, rel in sorted(pairs)
        if closure_max_depth.get((src, rel), 0) < 2
    ]
    out: List[Tuple[int, str, int]] = []
    for src, relation in missing:
        for target in all_nodes:
            if target == src:
                continue
            if (src, relation, target) in closure_triples:
                continue
            out.append((src, relation, target))
    return out, len(missing)


# --------------------------------------------------------------------------- #
# 深度估计
# --------------------------------------------------------------------------- #
def _bfs(adj: Dict[int, Set[int]], start: int, max_depth: int = 8) -> Dict[int, int]:
    """Breadth-first shortest hop counts from ``start``, capped at ``max_depth``."""
    dist = {start: 0}
    frontier = [start]
    for depth in range(1, max_depth + 1):
        nxt: List[int] = []
        for node in frontier:
            for nb in adj.get(node, ()):
                if nb not in dist:
                    dist[nb] = depth
                    nxt.append(nb)
        if not nxt:
            break
        frontier = nxt
    return dist


def compute_depth_estimates(
    edges: Sequence[Tuple[int, str, int]],
    triples: Sequence[Tuple[int, str, int]],
    cap: int = 3,
) -> List[Optional[int]]:
    """Estimated hop count for each triple (1/2/3+), same relation first.

    The closure engine composes edges of **the same relation**, so the honest
    estimate is the directed same-relation path length.  When no such path
    exists we fall back to the undirected all-relation distance (the model sees
    a typed but fully connected message graph, so a cross-relation path is a
    plausible explanation for the score).  ``None`` means "no path found" --
    stored as SQL NULL rather than faked.
    """
    rel_adj: Dict[str, Dict[int, Set[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    undirected: Dict[int, Set[int]] = collections.defaultdict(set)
    for src, relation, dst in edges:
        rel_adj[relation][src].add(dst)
        undirected[src].add(dst)
        undirected[dst].add(src)

    needed: Dict[Tuple[str, int], Dict[int, int]] = {}
    for src, relation, _target in triples:
        key = (relation, src)
        if key not in needed:
            needed[key] = _bfs(rel_adj.get(relation, {}), src)

    undirected_cache: Dict[int, Dict[int, int]] = {}
    out: List[Optional[int]] = []
    for src, relation, target in triples:
        dist = needed[(relation, src)].get(target)
        if dist is None:
            if src not in undirected_cache:
                undirected_cache[src] = _bfs(undirected, src)
            dist = undirected_cache[src].get(target)
        if dist is None or dist <= 0:
            out.append(None)
        else:
            out.append(min(int(dist), cap))
    return out


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_graph(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
) -> torch.Tensor:
    """One forward pass over the whole graph; returns (N, hidden) embeddings."""
    h, _attentions = model.encode(graph.edge_index, graph.edge_type)
    return h.detach()


@torch.no_grad()
def score_triples(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    triples: Sequence[Tuple[int, str, int]],
    h: Optional[torch.Tensor] = None,
) -> List[float]:
    """sigmoid(h_s^T W_r h_t) for every triple; encodes the graph at most once."""
    if not triples:
        return []
    if h is None:
        h = encode_graph(model, graph)

    by_relation: Dict[str, List[Tuple[int, int, int]]] = collections.defaultdict(list)
    for pos, (src, relation, target) in enumerate(triples):
        by_relation[relation].append(
            (pos, graph.id_to_index[src], graph.id_to_index[target])
        )

    scores = [0.0] * len(triples)
    for relation, items in by_relation.items():
        matrix = model.score_all(h, relation)  # (N, N) 全矩阵，一次 matmul
        positions = torch.tensor([i[0] for i in items], dtype=torch.long)
        rows = torch.tensor([i[1] for i in items], dtype=torch.long)
        cols = torch.tensor([i[2] for i in items], dtype=torch.long)
        logits = matrix[rows, cols]
        probs = torch.sigmoid(logits)
        for pos, value in zip(positions.tolist(), probs.tolist()):
            scores[pos] = float(value)
    return scores


# --------------------------------------------------------------------------- #
# 落库
# --------------------------------------------------------------------------- #
def ensure_table(conn: sqlite3.Connection) -> bool:
    """Create ``inferred_relations`` if absent. Returns True when created."""
    existed = _table_exists(conn, "inferred_relations")
    conn.executescript(INFERRED_DDL)
    for ddl in INFERRED_INDEXES:
        conn.execute(ddl)
    conn.commit()
    return not existed


def write_rows(
    conn: sqlite3.Connection,
    rows: Sequence[Tuple[int, str, int, float, str, Optional[int], Optional[int]]],
) -> int:
    """INSERT OR IGNORE every row; returns the number actually inserted."""
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO inferred_relations
            (source_id, relation_type, target_id, confidence,
             model_version, epoch, depth_estimate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #
def _bar(value: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def _fmt_name(labels: Dict[int, Tuple[str, str]], sense_id: int) -> str:
    word, pos = labels.get(sense_id, (f"<{sense_id}>", "?"))
    return f"{word}/{pos}"


def print_summary(
    labels: Dict[int, Tuple[str, str]],
    above: Sequence[Tuple[int, str, int, float]],
    gated: Sequence[Tuple[int, str, int, float]],
    min_confidence: float,
    dry_run: bool,
    top_k: int,
    max_per_group: int,
) -> None:
    print()
    print("-" * 92)
    print("  打分结果")
    print("-" * 92)
    print(f"  达到阈值 >= {min_confidence:.2f}     : {len(above):,}")
    if max_per_group > 0:
        print(f"  每个 (源,关系) 保留前 {max_per_group} : {len(gated):,}"
              f"（共 {len({(s, r) for s, r, _t, _c in above}):,} 个 (源,关系) 组）")
    print(f"  待写入                  : {len(gated):,}"
          + ("   （--dry-run：未写库）" if dry_run else "（写库结果见下方）"))
    if not gated:
        print("  [提示] 没有候选通过筛选。可降低 --min-confidence 或关闭 --no-pos-guard 观察。")
        return

    per_relation = collections.Counter(t[1] for t in gated)
    print()
    print("  按关系分布（待写入）：")
    for relation, count in per_relation.most_common():
        print(f"    {relation:<16} {count:>5}")

    print()
    print("  全局 top-10 最高置信度：")
    print(f"    {'源':<14} {'关系':<14} {'目标':<16} {'置信度':>7}  条")
    print("    " + "-" * 88)
    for src, relation, target, confidence in gated[:10]:
        print(
            f"    {_fmt_name(labels, src):<14} {relation:<14} "
            f"{_fmt_name(labels, target):<16} {confidence:>7.4f}  {_bar(confidence, 16)}"
        )

    if top_k > 0:
        print()
        print(f"  每个关系的前 {top_k} 条：")
        grouped: Dict[str, List[Tuple[int, str, int, float]]] = collections.defaultdict(list)
        for triple in gated:
            grouped[triple[1]].append(triple)
        for relation in sorted(grouped):
            print()
            print(f"    [{RELATION_LABEL.get(relation, relation)}]")
            for src, _rel, target, confidence in grouped[relation][:top_k]:
                print(
                    f"      {_fmt_name(labels, src):<14} -> {_fmt_name(labels, target):<16} "
                    f"{confidence:.4f}"
                )


def print_sanity(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    h: torch.Tensor,
    labels: Dict[int, Tuple[str, str]],
    closure_triples: Set[Tuple[int, str, int]],
    relations: Sequence[str],
    anchor_word: str = "猫",
    anchor_target: str = "生物",
    top_k: int = 4,
) -> None:
    """Show the model's raw ranking over *all* nodes for one anchor sense.

    The trained protocol was validated on ranking (``猫 --is_a--> 生物`` rank #1),
    so the honest sanity check is exactly that: where does the known-true target
    land among all 335 nodes, and what else does the model prefer?  Pairs already
    present in ``closure`` never reach ``inferred_relations`` (they are skipped by
    design), so without this block the headline result would be invisible.
    """
    word_to_id = {word: sid for sid, (word, _pos) in labels.items()}
    src_id = word_to_id.get(anchor_word)
    if src_id is None:
        print(f"  [跳过] 数据库里没有「{anchor_word}」，无法做抽查。")
        return
    src_idx = graph.id_to_index[src_id]
    target_id = word_to_id.get(anchor_target)

    print()
    print(f"  [抽查] 「{anchor_word}」在全量 {len(labels)} 节点上的偏好（已存在的闭包边标注 *）")
    for relation in relations:
        matrix = model.score_all(h, relation)[src_idx]
        probs = torch.sigmoid(matrix)
        # 排除自身，取前 top_k
        order = torch.argsort(probs, descending=True).tolist()
        printed = 0
        parts: List[str] = []
        for idx in order:
            node_id = graph.node_ids[idx]
            if node_id == src_id:
                continue
            mark = "*" if (src_id, relation, node_id) in closure_triples else " "
            parts.append(f"{graph.labels[idx]}{mark}({probs[idx].item():.3f})")
            printed += 1
            if printed >= top_k:
                break
        line = f"    {relation:<16} " + "  ".join(parts)
        if relation == "is_a" and target_id is not None:
            tgt_idx = graph.id_to_index[target_id]
            rank = int((matrix > matrix[tgt_idx]).sum()) + 1
            line += f"   |  {anchor_target} rank {rank}/{len(labels)} " \
                    f"p={probs[tgt_idx].item():.3f}"
        print(line)


VERIFY_SQL = """
SELECT ir.source_id, w1.word, ir.relation_type, ir.target_id, w2.word,
       ir.confidence, ir.depth_estimate, ir.verified
FROM inferred_relations ir
JOIN senses s1 ON ir.source_id = s1.sense_id
JOIN words  w1 ON s1.word_id  = w1.word_id
JOIN senses s2 ON ir.target_id = s2.sense_id
JOIN words  w2 ON s2.word_id  = w2.word_id
ORDER BY ir.confidence DESC
LIMIT 20
"""


def run_verify(conn: sqlite3.Connection, min_confidence: float) -> None:
    print()
    print("=" * 92)
    print("  验证：inferred_relations 落库结果")
    print("=" * 92)
    if not _table_exists(conn, "inferred_relations"):
        print("  [空] 表不存在（本次是 --dry-run？）")
        return

    total = conn.execute("SELECT COUNT(*) FROM inferred_relations").fetchone()[0]
    if total == 0:
        print("  [空] 表存在但没有任何行。")
        return

    print(f"  总行数 : {total:,}")
    print()
    print("  按关系统计：")
    print(f"    {'relation_type':<18} {'rows':>6} {'avg_conf':>9} {'max_conf':>9} {'d>=2':>6}")
    print("    " + "-" * 60)
    for relation, rows, avg_conf, max_conf, deep in conn.execute(
        """
        SELECT relation_type, COUNT(*), AVG(confidence), MAX(confidence),
               SUM(CASE WHEN depth_estimate >= 2 THEN 1 ELSE 0 END)
        FROM inferred_relations GROUP BY relation_type ORDER BY COUNT(*) DESC
        """
    ):
        print(f"    {relation:<18} {rows:>6} {avg_conf:>9.4f} {max_conf:>9.4f} {deep:>6}")

    print()
    print("  top-20（置信度降序）：")
    print(f"    {'源':<14} {'关系':<14} {'目标':<16} {'置信度':>7} {'深度':>4}")
    print("    " + "-" * 76)
    for _sid, src, relation, _tid, target, confidence, depth, _verified in conn.execute(
        VERIFY_SQL
    ):
        print(
            f"    {src:<14} {relation:<14} {target:<16} {confidence:>7.4f} "
            f"{'-' if depth is None else depth:>4}"
        )

    # 抽查：任务书点名的高置信度合理性检查。
    print()
    print("  抽查（猫）：")
    rows = conn.execute(
        """
        SELECT w2.word, ir.relation_type, ir.confidence
        FROM inferred_relations ir
        JOIN senses s1 ON ir.source_id = s1.sense_id
        JOIN words  w1 ON s1.word_id = w1.word_id
        JOIN senses s2 ON ir.target_id = s2.sense_id
        JOIN words  w2 ON s2.word_id = w2.word_id
        WHERE w1.word = '猫' AND ir.confidence >= ?
        ORDER BY ir.confidence DESC
        """,
        (min_confidence,),
    ).fetchall()
    if not rows:
        print(f"    （无 >= {min_confidence:.2f} 的推断）")
    for word, relation, confidence in rows[:12]:
        print(f"    猫 -> {word:<14} {relation:<14} {confidence:.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained Logos Transformer and write high-confidence "
                    "closure hypotheses into inferred_relations."
    )
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None,
                        help="checkpoint 路径；缺省按「W_r 头覆盖率」自动解析："
                             + " -> ".join(DEFAULT_CHECKPOINT_CANDIDATES))
    parser.add_argument("--min-confidence", type=float, default=0.8,
                        help="写入阈值（sigmoid 后的置信度），默认 0.8")
    parser.add_argument("--top-k", type=int, default=5,
                        help="按关系打印前 k 条（0=不打印），默认 5")
    parser.add_argument("--relation-subset", nargs="*", default=None, metavar="REL",
                        help="只推断这些关系；缺省 = closure 里出现过 depth>=2 的全部关系")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打分不写库（不建表、不改库）")
    parser.add_argument("--verify", action="store_true",
                        help="写库后跑 SQL 复核并打印结果")
    parser.add_argument("--strategies", nargs="*", default=["A", "B"],
                        choices=["A", "B"], help="候选生成策略，默认两者都用")
    parser.add_argument("--max-neighbourhood", type=int, default=60,
                        help="策略 A 每个源的二跳目标上限（0=不截断），默认 60")
    parser.add_argument("--max-candidates", type=int, default=200000,
                        help="安全阀：候选超过该数则报错退出，默认 200000")
    parser.add_argument("--no-pos-guard", action="store_true",
                        help="关闭「目标词性守卫」。默认开启：只推断与既有闭包中该关系实际"
                             "出现过的目标词性一致的候选（分布由数据库自行读出）。")
    parser.add_argument("--pos-min-share", type=float, default=0.05,
                        help="目标词性成为白名单的最低占比，默认 0.05")
    parser.add_argument("--max-per-source-relation", type=int, default=3,
                        help="每个 (源,关系) 最多保留几条假设（0=不限制），默认 3。"
                             "模型按排名而非概率验证，σ 会饱和，故用列表长度而非绝对值兜底。")
    parser.add_argument("--epoch", type=int, default=None,
                        help="写入 epoch 列；checkpoint 本身不含训练元数据，缺省写 NULL")
    parser.add_argument("--seed", type=int, default=42,
                        help="重建模型时的初始化种子（只为可复现，权重随后被覆盖）")
    args = parser.parse_args(argv)

    if not (0.0 <= args.min_confidence <= 1.0):
        parser.error("--min-confidence 必须在 [0, 1]")
    if args.top_k < 0:
        parser.error("--top-k 必须 >= 0")
    if args.max_neighbourhood < 0:
        parser.error("--max-neighbourhood 必须 >= 0")
    if args.max_candidates < 1:
        parser.error("--max-candidates 必须 >= 1")
    if not (0.0 <= args.pos_min_share <= 1.0):
        parser.error("--pos-min-share 必须在 [0, 1]")
    if args.max_per_source_relation < 0:
        parser.error("--max-per-source-relation 必须 >= 0")
    if args.relation_subset is not None and not args.relation_subset:
        parser.error("--relation-subset 至少要给一个关系名")
    if not args.strategies:
        parser.error("--strategies 至少要给一个策略")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):  # pragma: no cover
                pass

    args = parse_args(argv)
    t0 = time.time()

    db_path: pathlib.Path = args.db.expanduser()
    if not db_path.exists():
        print(f"[错误] 找不到数据库：{db_path}", file=sys.stderr)
        print("       先运行：python apply_seeds.py", file=sys.stderr)
        return 1

    # ---- 只读阶段：先把要推断的关系与候选算清楚 --------------------------- #
    ro = sqlite3.connect(str(db_path))
    ro.execute("PRAGMA query_only = ON")
    try:
        if not _table_exists(ro, "senses") or not _table_exists(ro, "relations"):
            print(f"[错误] {db_path} 不是 Logos 数据库（缺 senses/relations）", file=sys.stderr)
            return 1

        labels = load_sense_labels(ro)
        edges = load_direct_edges(ro)
        closure_triples, closure_max_depth = load_closure_index(ro)
        tracked = closure_tracked_relations(ro)

        if args.relation_subset is not None:
            requested = [r for r in args.relation_subset if r in tracked]
            unknown = sorted(set(args.relation_subset) - set(tracked))
            if unknown:
                print(f"[提示] 这些关系不在 closure 跟踪范围内，已跳过：{', '.join(unknown)}")
            if not requested:
                print("[错误] --relation-subset 里没有任何 closure 跟踪的关系。", file=sys.stderr)
                print(f"       closure 跟踪的关系：{', '.join(tracked)}", file=sys.stderr)
                return 2
        else:
            requested = list(tracked)

        if not requested:
            print("[错误] closure 表里没有 depth>=2 的行，无法确定要推断哪些关系。",
                  file=sys.stderr)
            return 1

        all_nodes = sorted(labels)
        candidates: List[Tuple[int, str, int]] = []
        stats: Dict[str, int] = {}
        if "A" in args.strategies:
            a_list, a_sources = generate_candidates_a(
                edges, closure_triples, requested, args.max_neighbourhood
            )
            candidates.extend(a_list)
            stats["A"] = len(a_list)
            stats["A_sources"] = a_sources
        if "B" in args.strategies:
            b_list, b_pairs = generate_candidates_b(
                edges, closure_triples, closure_max_depth, requested, all_nodes
            )
            candidates.extend(b_list)
            stats["B"] = len(b_list)
            stats["B_pairs"] = b_pairs

        # 去重（A/B 会重叠），保持稳定顺序便于复现。
        seen: Set[Tuple[int, str, int]] = set()
        unique: List[Tuple[int, str, int]] = []
        for triple in candidates:
            if triple in seen:
                continue
            seen.add(triple)
            unique.append(triple)
        candidates = unique
        target_pos = load_relation_target_pos(ro)
    finally:
        ro.close()

    # ---- 词性守卫：只保留「该关系在既有闭包里真的收过这种词性」的目标 ------ #
    pos_of: Dict[int, str] = {sid: pos for sid, (_word, pos) in labels.items()}
    guard = (
        {} if args.no_pos_guard
        else build_pos_guard(target_pos, args.pos_min_share)
    )
    dropped_by_pos: List[Tuple[int, str, int]] = []
    n_raw = len(candidates)
    if guard:
        candidates, dropped_by_pos = apply_pos_guard(candidates, pos_of, guard)

    naive = len(labels) * len(labels) * max(len(tracked), 1)
    print("=" * 92)
    print("  Logos Transformer — 端到端推理与写回")
    print("=" * 92)
    print(f"  数据库       : {db_path}")
    print(f"  节点 / 直接边 / 闭包行 : {len(labels)} / {len(edges)} / {len(closure_triples)}")
    print(f"  推断关系     : {', '.join(requested)}")
    print()
    print("  [候选生成]")
    if "A" in stats:
        print(f"    策略 A（邻域二跳）      : 扫描 {stats['A_sources']} 个源 -> "
              f"{stats['A']:,} 条候选")
    if "B" in stats:
        print(f"    策略 B（直接边补全）    : {stats['B_pairs']} 个 (源,关系) 对 × "
              f"{len(labels)} 节点 -> {stats['B']:,} 条候选")
    print(f"    去重后                  : {n_raw:,} 条")
    print(f"    朴素全枚举（对照）      : {naive:,} 条（{len(labels)}² × {len(tracked)}）"
          f" -> 压缩 {naive / max(n_raw, 1):.0f}x")

    if guard:
        print(f"    词性守卫剔除            : {len(dropped_by_pos):,} 条"
              f"（目标词性不属于该关系在闭包中的既有分布）")
        print(f"    进入打分                : {len(candidates):,} 条")
    else:
        print("    词性守卫                : 已关闭（--no-pos-guard）")

    if not candidates:
        print("\n  [提示] 没有生成任何候选（闭包可能已被填满）。")
        return 0
    if len(candidates) > args.max_candidates:
        print(f"\n[错误] 候选数 {len(candidates):,} 超过 --max-candidates "
              f"{args.max_candidates:,}。调小 --max-neighbourhood 或收窄 --relation-subset。",
              file=sys.stderr)
        return 1

    if guard:
        print()
        print("  [词性守卫] 期望从 closure 读出，不是硬编码：")
        for relation in requested:
            counts = target_pos.get(relation, {})
            total = sum(counts.values()) or 1
            dist = "  ".join(
                f"{pos}:{count / total * 100:.0f}%"
                for pos, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            allowed = sorted(guard.get(relation, ()))
            print(f"    {relation:<16} {dist:<26} -> 允许 {','.join(allowed) or '(不限)'}")

    # ---- checkpoint + 图 --------------------------------------------------- #
    checkpoint, info, reason = resolve_checkpoint(args.checkpoint, requested)
    if not checkpoint.exists():  # pragma: no cover - resolve 已保证
        print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
        return 1
    print()
    print("  [模型]")
    print(f"    checkpoint             : {checkpoint}")
    print(f"    选择依据               : {reason}")
    print(f"    结构                   : {info['num_layers']} 层 / {info['num_heads']} 头 / "
          f"hidden={info['hidden']} / {info['num_nodes']} 节点")
    print(f"    专属 W_r 头            : {len(info['head_names'])} 个 -> "
          f"{', '.join(info['head_names'][:8])}"
          + (" ..." if len(info['head_names']) > 8 else ""))

    print()
    print("  [构建图]")
    graph = load_graph(db_path, include_closure=False, verbose=False)
    model = build_and_load(checkpoint, info, graph, args.seed)

    # 头覆盖率提示：没有专属头的关系会走 W_shared，置信度不可与专属头同等解读。
    uncovered = [r for r in requested if r not in set(info["head_names"])]
    if uncovered:
        print(f"    [警告] 这些关系回落到 W_shared（无专属训练）：{', '.join(uncovered)}")

    # ---- 打分 ------------------------------------------------------------- #
    print()
    print("  [打分]")
    h = encode_graph(model, graph)
    scores = score_triples(model, graph, candidates, h=h)
    scored = [
        (src, relation, target, confidence)
        for (src, relation, target), confidence in zip(candidates, scores)
    ]
    scored.sort(key=lambda item: (-item[3], item[1], item[0], item[2]))

    epochs_note = args.epoch if args.epoch is not None else "NULL"
    model_version = f"{RELAWARE_VERSION}@{checkpoint.parent.name}"

    above = [t for t in scored if t[3] >= args.min_confidence]
    gated = limit_per_group(above, args.max_per_source_relation)

    print_sanity(
        model=model, graph=graph, h=h, labels=labels,
        closure_triples=closure_triples, relations=requested,
    )

    print_summary(
        labels=labels,
        above=above,
        gated=gated,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
        top_k=args.top_k,
        max_per_group=args.max_per_source_relation,
    )

    # ---- 写库 ------------------------------------------------------------- #
    written = 0
    if args.dry_run:
        print()
        print(f"  --dry-run：将写入 {len(gated):,} 行（epoch={epochs_note}，"
              f"model_version={model_version}），实际未改动数据库。")
    else:
        depths = compute_depth_estimates(edges, [(s, r, t) for s, r, t, _c in gated])
        rows = [
            (src, relation, target, round(confidence, 6), model_version,
             args.epoch, depth)
            for (src, relation, target, confidence), depth in zip(gated, depths)
        ]
        rw = sqlite3.connect(str(db_path))
        try:
            rw.execute("PRAGMA foreign_keys = ON")
            created = ensure_table(rw)
            written = write_rows(rw, rows)
            total = rw.execute("SELECT COUNT(*) FROM inferred_relations").fetchone()[0]
        finally:
            rw.close()
        print()
        print(f"  表 inferred_relations {'已新建' if created else '已存在'}")
        print(f"  本次写入 : {written:,} 行（INSERT OR IGNORE；重复项被跳过）")
        print(f"  表内总行 : {total:,}")
        if len(gated) - written > 0:
            print(f"  跳过重复 : {len(gated) - written:,} 行（UNIQUE 约束命中）")
        print("  注意     : verified 默认 0 —— 这些是模型假设，不是已确认事实。")

    if args.verify:
        vc = sqlite3.connect(str(db_path))
        vc.execute("PRAGMA query_only = ON")
        try:
            run_verify(vc, args.min_confidence)
        finally:
            vc.close()

    print()
    print(f"  用时 {time.time() - t0:.2f}s")
    if not args.dry_run:
        print("  复核：python logos_transformer/query_inferences.py --stats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
