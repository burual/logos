#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logos Transformer -- minimal relational graph-attention prototype.

This is an *inference-only* proof of concept.  It exports the sense-level
relation graph from ``logos.db``, runs a two-layer relation-aware GAT with
random (fixed-seed) weights, and prints the attention distribution around one
target sense.  Nothing is written back to the database.

Design notes live in ``docs/LOGOS_TRANSFORMER_DESIGN.md``.

Usage:
    python logos_transformer/prototype_gat.py
    python logos_transformer/prototype_gat.py --sense 2 --top-k 5
    python logos_transformer/prototype_gat.py --sense 脑炎 --include-closure

Exit codes:
    0  success
    1  missing dependency / database / usable graph
    2  bad command-line argument
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import sqlite3
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Dependency guard: fail with an actionable message instead of a traceback.
# 依赖缺失时给出可执行提示，而不是抛一堆栈。
#
# torch 在 import 时会尝试初始化 numpy 并在缺失时发 UserWarning；numpy 不是本
# 原型的依赖（requirements 只有 torch），因此提前过滤这一条特定告警。
# --------------------------------------------------------------------------- #
warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - environment dependent
    print("Requires torch. Install with: pip install torch", file=sys.stderr)
    raise SystemExit(1)

MODEL_VERSION = "logos-gat-v0.1-prototype"
DEFAULT_TARGET = "猫"          # 猫 / sense_id=2：有一个 is_a 和三个 can_do 出边
SELF_DIR = 2                   # edge_dir 取值：0=正向 1=反向 2=自环


# --------------------------------------------------------------------------- #
# Graph export
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RelationGraph:
    """A sense-only homogeneous graph with typed, direction-aware edges."""

    node_ids: List[int]
    labels: List[str]
    definitions: List[str]
    pos_tags: List[str]
    id_to_index: Dict[int, int]

    edge_index: torch.Tensor      # shape (2, E): [src_idx, dst_idx]
    edge_type: torch.Tensor       # shape (E,): 2*rel + dir_bit, self = 2*K
    edge_rel: torch.Tensor        # shape (E,): base relation index (self = -1)
    edge_dir: torch.Tensor        # shape (E,): 0 forward, 1 reverse, 2 self

    relation_names: List[str]     # base vocabulary (45 types + optional closure:*)
    num_edge_types: int
    stats: Dict[str, int]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _load_senses(conn: sqlite3.Connection):
    rows = conn.execute(
        """
        SELECT s.sense_id, w.word, s.pos, s.definition
        FROM senses s
        JOIN words w ON w.word_id = s.word_id
        ORDER BY s.sense_id
        """
    ).fetchall()
    node_ids = [int(r[0]) for r in rows]
    labels = [str(r[1]) for r in rows]
    pos_tags = [str(r[2]) for r in rows]
    definitions = [str(r[3] or "") for r in rows]
    return node_ids, labels, pos_tags, definitions


def load_graph(
    db_path: pathlib.Path,
    include_closure: bool = False,
    verbose: bool = True,
) -> RelationGraph:
    """Export the sense-level relation graph from a Logos SQLite database.

    Args:
        db_path: Path to ``logos.db`` (``str`` or ``pathlib.Path``).
        include_closure: If True, add closure rows with depth >= 2 as extra
            edges of type ``closure:<relation>`` (ablation only; off by default).
        verbose: Print progress notes.

    Returns:
        RelationGraph with bidirectional message edges plus self-loops.
    """
    db_path = pathlib.Path(db_path).expanduser()
    if not db_path.exists():
        print(f"[错误] 找不到数据库：{db_path}", file=sys.stderr)
        print("       请先在项目根目录运行：python apply_seeds.py", file=sys.stderr)
        raise SystemExit(1)

    # 只读连接：原型绝不允许改库。
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")

    try:
        for required in ("senses", "relations", "relation_types"):
            if not _table_exists(conn, required):
                print(f"[错误] 数据库缺少表 `{required}`，不是 Logos 数据库：{db_path}",
                      file=sys.stderr)
                raise SystemExit(1)

        node_ids, labels, pos_tags, definitions = _load_senses(conn)
        if not node_ids:
            print("[错误] senses 表为空，无法构图。", file=sys.stderr)
            raise SystemExit(1)

        id_to_index = {sid: i for i, sid in enumerate(node_ids)}

        # 关系词表：全部 45 种，保证 edge type 编号在不同库上稳定。
        relation_names = [
            str(r[0])
            for r in conn.execute("SELECT relation FROM relation_types ORDER BY relation")
        ]

        # 语义边：只取义项→义项，跳过否定边。
        rel_rows = conn.execute(
            """
            SELECT relation_id, src_id, dst_id, relation, weight, confidence
            FROM relations
            WHERE src_kind = 'sense' AND dst_kind = 'sense' AND is_negated = 0
            ORDER BY relation_id
            """
        ).fetchall()

        non_sense_edges = conn.execute(
            """
            SELECT COUNT(*) FROM relations
            WHERE NOT (src_kind = 'sense' AND dst_kind = 'sense')
            """
        ).fetchone()[0]

        semantic_edges: List[Tuple[int, int, int, int]] = []  # src, dst, rel_idx, relation_id
        skipped_missing = 0
        rel_index = {name: i for i, name in enumerate(relation_names)}
        for relation_id, src_id, dst_id, relation, _weight, _conf in rel_rows:
            si = id_to_index.get(int(src_id))
            di = id_to_index.get(int(dst_id))
            ri = rel_index.get(str(relation))
            if si is None or di is None or ri is None:
                skipped_missing += 1
                continue
            semantic_edges.append((si, di, ri, int(relation_id)))

        closure_rows = closure_depth1 = closure_deep = 0
        if include_closure:
            if _table_exists(conn, "closure"):
                closure_edges = conn.execute(
                    """
                    SELECT relation, descendant, ancestor, depth
                    FROM closure
                    WHERE depth >= 2
                    ORDER BY relation, depth, descendant, ancestor
                    """
                ).fetchall()
                for relation, descendant, ancestor, _depth in closure_edges:
                    si = id_to_index.get(int(descendant))
                    di = id_to_index.get(int(ancestor))
                    if si is None or di is None:
                        continue
                    name = f"closure:{relation}"
                    if name not in rel_index:
                        rel_index[name] = len(relation_names)
                        relation_names.append(name)
                    semantic_edges.append((si, di, rel_index[name], -1))
                    closure_deep += 1
        if _table_exists(conn, "closure"):
            closure_rows = conn.execute("SELECT COUNT(*) FROM closure").fetchone()[0]
            closure_depth1, closure_deep_db = conn.execute(
                "SELECT SUM(depth = 1), SUM(depth >= 2) FROM closure"
            ).fetchone()
            closure_depth1 = int(closure_depth1 or 0)
            closure_deep_db = int(closure_deep_db or 0)
        else:
            closure_deep_db = 0

        num_base = len(relation_names)

        # ------------------------------------------------------------------ #
        # 双向展开 + 自环。语义边有向，但消息传递必须双向，否则只有出边的
        # 节点（如“猫”）收不到邻居消息。方向位进入 edge type，模型可学不对称。
        # ------------------------------------------------------------------ #
        srcs: List[int] = []
        dsts: List[int] = []
        etypes: List[int] = []
        erels: List[int] = []
        edirs: List[int] = []

        for si, di, ri, _rid in semantic_edges:
            # forward: si --R--> di
            srcs.append(si); dsts.append(di); etypes.append(2 * ri); erels.append(ri); edirs.append(0)
            # reverse: di --R--> si （语义不变，仅消息方向相反）
            srcs.append(di); dsts.append(si); etypes.append(2 * ri + 1); erels.append(ri); edirs.append(1)

        self_type = 2 * num_base
        for i in range(len(node_ids)):
            srcs.append(i); dsts.append(i); etypes.append(self_type); erels.append(-1); edirs.append(SELF_DIR)

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_type = torch.tensor(etypes, dtype=torch.long)
        edge_rel = torch.tensor(erels, dtype=torch.long)
        edge_dir = torch.tensor(edirs, dtype=torch.long)

        # 度数（只看正向语义边）
        outs = [0] * len(node_ids)
        ins = [0] * len(node_ids)
        for si, di, _ri, _rid in semantic_edges:
            outs[si] += 1
            ins[di] += 1
        isolated = sum(1 for i in range(len(node_ids)) if outs[i] + ins[i] == 0)

        stats = {
            "nodes": len(node_ids),
            "semantic_edges": len(semantic_edges) - closure_deep,
            "closure_edges_added": closure_deep,
            "message_edges": len(srcs),
            "dropped_non_sense": int(non_sense_edges),
            "skipped_missing_endpoint": skipped_missing,
            "isolated": isolated,
            "closure_rows": int(closure_rows),
            "closure_depth1": closure_depth1,
            "closure_deep": int(closure_deep_db),
            "relation_types": num_base,
        }
    finally:
        conn.close()

    if verbose:
        print(f"图导出完成：{stats['nodes']} 节点 / {stats['semantic_edges']} 条语义边 / "
              f"{stats['message_edges']} 条消息边")

    if stats["semantic_edges"] == 0:
        print("[错误] relations 表中没有义项→义项的边，无法运行 GAT。", file=sys.stderr)
        raise SystemExit(1)

    return RelationGraph(
        node_ids=node_ids,
        labels=labels,
        definitions=definitions,
        pos_tags=pos_tags,
        id_to_index=id_to_index,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_rel=edge_rel,
        edge_dir=edge_dir,
        relation_names=relation_names,
        num_edge_types=2 * num_base + 1,
        stats=stats,
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over edges grouped by their destination node.

    Args:
        scores: (E, H) unnormalised attention logits.
        index:  (E,) destination node index for each edge.
        num_nodes: number of destinations.

    Returns:
        (E, H) attention weights with ``sum_j alpha[i, j] == 1`` per node/head.
    """
    num_heads = scores.size(1)
    expanded = index.unsqueeze(1).expand(-1, num_heads)

    # 数值稳定：先减去每个目标节点的最大值（scatter amax），再做 exp。
    max_per_node = torch.full(
        (num_nodes, num_heads), float("-inf"), dtype=scores.dtype, device=scores.device
    )
    max_per_node.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
    max_per_node = torch.where(
        torch.isfinite(max_per_node), max_per_node, torch.zeros_like(max_per_node)
    )

    exp_scores = (scores - max_per_node[index]).exp()
    denom = torch.zeros(num_nodes, num_heads, dtype=scores.dtype, device=scores.device)
    denom.index_add_(0, index, exp_scores)
    return exp_scores / denom[index].clamp_min(1e-16)


class RelationAwareGATLayer(nn.Module):
    """One multi-head GAT layer with a separate projection per edge type.

    Edge type encodes ``relation x direction``; every type owns its own
    ``W_r`` and attention vectors, so ``is_a`` and ``has_symptom`` never share
    a propagation operator.
    """

    def __init__(
        self,
        in_dim: int,
        head_dim: int,
        num_heads: int,
        num_edge_types: int,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat = concat

        # (R, H, F, O) —— 每种边类型独立的线性投影
        bound = math.sqrt(6.0 / (in_dim + head_dim))
        self.weight = nn.Parameter(torch.empty(num_edge_types, num_heads, in_dim, head_dim))
        self.att_src = nn.Parameter(torch.empty(num_edge_types, num_heads, head_dim))
        self.att_dst = nn.Parameter(torch.empty(num_edge_types, num_heads, head_dim))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.att_src, -bound, bound)
        nn.init.uniform_(self.att_dst, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        num_nodes = x.size(0)
        heads, head_dim = self.num_heads, self.head_dim

        # (R, N, H, O)：一次算出所有边类型的节点投影，再按边 gather。
        xw = torch.einsum("nf,rhfo->rnho", x, self.weight)
        h_src = xw[edge_type, src]          # (E, H, O)
        h_dst = xw[edge_type, dst]          # (E, H, O)

        logits = (h_src * self.att_src[edge_type]).sum(-1) + \
                 (h_dst * self.att_dst[edge_type]).sum(-1)   # (E, H)
        logits = F.leaky_relu(logits, self.negative_slope)

        alpha = _segment_softmax(logits, dst, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 消息从 src 流向 dst，按注意力加权后散射求和。
        messages = h_src * alpha.unsqueeze(-1)
        out = x.new_zeros((num_nodes, heads, head_dim))
        out.index_add_(0, dst, messages)

        if self.concat:
            out = out.reshape(num_nodes, heads * head_dim)
        else:
            out = out.mean(dim=1)
        return out, alpha


class LogosGAT(nn.Module):
    """Two-layer relation-aware GAT used by the prototype."""

    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.normal_(self.node_embedding.weight, std=0.1)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                RelationAwareGATLayer(
                    in_dim=hidden_dim,
                    head_dim=hidden_dim // num_heads,
                    num_heads=num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                    concat=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = F.dropout(self.node_embedding.weight, p=self.dropout, training=self.training)
        attentions: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            h, alpha = layer(h, edge_index, edge_type)
            attentions.append(alpha)
            if i < len(self.layers) - 1:
                h = F.elu(h)          # 层间激活
        return h, attentions


def build_model(
    graph: RelationGraph,
    hidden_dim: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    seed: int = 42,
) -> LogosGAT:
    """Create the prototype model with reproducible random weights."""
    torch.manual_seed(seed)
    model = LogosGAT(
        num_nodes=len(graph.node_ids),
        num_edge_types=graph.num_edge_types,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: LogosGAT, graph: RelationGraph
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run one forward pass; returns node embeddings and per-layer attention."""
    return model(graph.edge_index, graph.edge_type)


# --------------------------------------------------------------------------- #
# Target selection and reporting
# --------------------------------------------------------------------------- #
def resolve_target(conn: sqlite3.Connection, spec: str) -> Optional[int]:
    """Resolve a sense_id (digits) or an exact surface form to a sense_id."""
    if spec.isdigit():
        row = conn.execute("SELECT sense_id FROM senses WHERE sense_id = ?", (int(spec),)).fetchone()
        return int(row[0]) if row else None
    row = conn.execute(
        """
        SELECT s.sense_id
        FROM senses s JOIN words w ON w.word_id = s.word_id
        WHERE w.word = ?
        ORDER BY s.is_primary DESC, s.sense_id
        LIMIT 1
        """,
        (spec,),
    ).fetchone()
    return int(row[0]) if row else None


def _closure_reference(conn: sqlite3.Connection, target_sid: int, limit: int = 6):
    """Return closure-implied (not direct) relations starting at the target."""
    if not _table_exists(conn, "closure"):
        return []
    rows = conn.execute(
        """
        SELECT c.relation, c.ancestor, c.depth, r.dst_id
        FROM closure c
        LEFT JOIN relations r ON r.relation_id = c.first_edge
        WHERE c.descendant = ? AND c.depth >= 2
        ORDER BY c.depth, c.relation, c.ancestor
        LIMIT ?
        """,
        (target_sid, limit),
    ).fetchall()
    out = []
    for relation, ancestor, depth, via in rows:
        out.append(
            {
                "relation": str(relation),
                "dst_id": int(ancestor),
                "depth": int(depth),
                "via_id": int(via) if via is not None else None,
            }
        )
    return out


def _bar(value: float, max_value: float, width: int = 22) -> str:
    if max_value <= 0:
        return ""
    n = int(round(width * value / max_value))
    return "█" * max(n, 0)


def print_report(
    graph: RelationGraph,
    model: LogosGAT,
    embeddings: torch.Tensor,
    attentions: Sequence[torch.Tensor],
    target_idx: int,
    db_path: pathlib.Path,
    top_k: int,
    closure_ref,
) -> None:
    """Print the human-readable attention report."""
    stats = graph.stats
    layer1 = attentions[0].mean(dim=1)      # (E,) 头均值
    layer2 = attentions[-1].mean(dim=1)

    incident_mask = graph.edge_index[1] == target_idx
    incident_idx = torch.nonzero(incident_mask, as_tuple=False).flatten()

    rows = []
    self_row = None
    for e in incident_idx.tolist():
        direction = int(graph.edge_dir[e])
        a1 = float(layer1[e])
        a2 = float(layer2[e])
        if direction == SELF_DIR:
            self_row = {"a1": a1, "a2": a2}
            continue
        neighbor_idx = int(graph.edge_index[0, e])
        rel_idx = int(graph.edge_rel[e])
        full_rel = graph.relation_names[rel_idx]
        is_closure = full_rel.startswith("closure:")
        rows.append(
            {
                "neighbor": graph.labels[neighbor_idx],
                "neighbor_id": graph.node_ids[neighbor_idx],
                "relation": (full_rel[len("closure:"):] + " [closure]") if is_closure else full_rel,
                "is_closure": is_closure,
                # 语义方向相对目标：out = 目标→邻居；in = 邻居→目标
                "direction": "out" if direction == 1 else "in",
                "a1": a1,
                "a2": a2,
            }
        )
    rows.sort(key=lambda r: r["a1"], reverse=True)
    shown = rows[:top_k]

    target_sid = graph.node_ids[target_idx]
    target_label = graph.labels[target_idx]
    target_pos = graph.pos_tags[target_idx]
    target_def = graph.definitions[target_idx]

    print()
    print("=" * 78)
    print("Logos Transformer — relational GAT prototype (inference only)")
    print("=" * 78)
    print(f"数据库          : {db_path}")
    print(f"模型版本        : {MODEL_VERSION}")
    print(f"图               : {stats['nodes']} 节点 / {stats['semantic_edges']} 条语义边"
          f" / {stats['message_edges']} 条消息边")
    print(f"关系类型        : {stats['relation_types']} 种 → {graph.num_edge_types} 种边类型"
          f"（含方向位与自环）")
    print(f"闭包统计        : {stats['closure_rows']} 行"
          f"（depth=1: {stats['closure_depth1']}，depth≥2: {stats['closure_deep']}）")
    if stats["dropped_non_sense"]:
        print(f"已排除          : {stats['dropped_non_sense']} 条指向 entity_nodes 的边")
    if stats["isolated"]:
        print(f"孤立义项        : {stats['isolated']} 个（仅有自环）")
    if graph.stats["closure_edges_added"]:
        print(f"消融开关        : 已把 {graph.stats['closure_edges_added']} 条 depth≥2 闭包边加入消息图")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型            : {len(model.layers)} 层 × {model.layers[0].num_heads} 头"
          f"，hidden={embeddings.size(1)}，参数量={num_params:,}")
    print(f"节点嵌入 shape   : {tuple(embeddings.shape)}")
    norms = embeddings.norm(dim=1)
    print(f"嵌入 L2 范数     : mean={float(norms.mean()):.3f}  "
          f"min={float(norms.min()):.3f}  max={float(norms.max()):.3f}")

    print()
    print(f"目标义项        : sense_id={target_sid}  「{target_label}」  [{target_pos}]")
    if target_def:
        print(f"释义            : {target_def}")
    direct_count = sum(1 for row in rows if not row["is_closure"])
    closure_count = len(rows) - direct_count
    out_deg = sum(1 for row in rows if row["direction"] == "out" and not row["is_closure"])
    in_deg = sum(1 for row in rows if row["direction"] == "in" and not row["is_closure"])
    print(f"目标关联边      : {len(rows)}（直接语义边 {direct_count} + 闭包边 {closure_count}；"
          f"直接出度 {out_deg} / 入度 {in_deg}）")
    if in_deg == 0:
        print("                  消息传递已双向展开，因此语义入度为 0 也能收到邻居消息。")

    print()
    print(f"Layer-1 注意力（{model.layers[0].num_heads} 头均值，按目标节点归一化）")
    print(f"  {'#':<3}{'邻居':<20}{'关系':<24}{'方向':<6}{'注意力':<8}分布")
    print("  " + "-" * 74)
    max_a1 = max([r["a1"] for r in shown] + ([self_row["a1"]] if self_row else [1e-9]))
    for rank, row in enumerate(shown, 1):
        neighbor = f"{row['neighbor']}({row['neighbor_id']})"
        print(f"  {rank:<3}{neighbor:<20}{row['relation']:<24}{row['direction']:<6}"
              f"{row['a1']:.4f}  {_bar(row['a1'], max_a1)}")
    if self_row is not None:
        print(f"  {'·':<3}{'(self)':<20}{'—':<24}{'self':<6}"
              f"{self_row['a1']:.4f}  {_bar(self_row['a1'], max_a1)}")

    print()
    print(f"Top-{top_k} 最受关注邻居（实际 {len(shown)} 个；方向 out=目标→邻居，in=邻居→目标）")
    if shown:
        for rank, row in enumerate(shown, 1):
            if row["direction"] == "in":
                triple = (f"{row['neighbor']} --{row['relation']}--> {target_label}")
            else:
                triple = (f"{target_label} --{row['relation']}--> {row['neighbor']}")
            print(f"  {rank}. {triple}  conf={row['a1']:.4f}")
    else:
        print("  （该节点没有语义邻居）")

    # 注意力分布统计：熵越接近 1，注意力越均匀。
    probs = [r["a1"] for r in rows] + ([self_row["a1"]] if self_row else [])
    if len(probs) > 1:
        total = sum(probs)
        p = [x / total for x in probs]
        entropy = -sum(x * math.log(x + 1e-12) for x in p)
        norm_entropy = entropy / math.log(len(p))
        print(f"注意力分布      : max={max(probs):.4f}  "
              f"归一化熵={norm_entropy:.3f}（1.000=完全均匀，0=全部集中于一个邻居）")
        if norm_entropy > 0.98:
            print("                  ↳ 随机权重下接近均匀是预期结果；训练后才会出现选择性。")

    # 第 2 层注意力：两跳信息混入后的重新分配，用作对照。
    if len(attentions) > 1 and shown:
        top2 = sorted(rows, key=lambda r: r["a2"], reverse=True)[:min(3, len(rows))]
        print()
        print("Layer-2 注意力 Top-3（两跳混入后的重新分配，对照用）")
        for row in top2:
            print(f"  {row['neighbor']}({row['neighbor_id']})  "
                  f"{row['relation']}  [{row['direction']}]  {row['a2']:.4f}")

    if closure_ref:
        print()
        if graph.stats["closure_edges_added"]:
            print("闭包对照（这些 depth≥2 边本次已作为独立边类型加入消息图，见上方消融开关）")
        else:
            print("闭包对照（已知的 depth≥2 隐含关系，不参与消息传递）")
        for ref in closure_ref:
            dst_label = graph.labels[graph.id_to_index[ref["dst_id"]]] \
                if ref["dst_id"] in graph.id_to_index else str(ref["dst_id"])
            via = ""
            if ref["via_id"] is not None and ref["via_id"] in graph.id_to_index:
                via = f"，首跳经 {graph.labels[graph.id_to_index[ref['via_id']]]}"
            print(f"  {target_label} --{ref['relation']}--> {dst_label}"
                  f"（depth {ref['depth']}{via}）")
    print()
    print("注意：权重为随机初始化且未训练，注意力数值只证明管线可跑通，不具备语义含义。")
    print("设计文档：docs/LOGOS_TRANSFORMER_DESIGN.md")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pick_default_target(graph: RelationGraph, conn: sqlite3.Connection, spec: str) -> int:
    sid = resolve_target(conn, spec)
    if sid is not None and sid in graph.id_to_index:
        return graph.id_to_index[sid]

    print(f"[警告] 找不到义项「{spec}」，改用度数最高的义项。", file=sys.stderr)
    degree = {}
    for e in range(graph.edge_index.size(1)):
        if int(graph.edge_dir[e]) != 0:
            continue
        s = int(graph.edge_index[0, e])
        d = int(graph.edge_index[1, e])
        degree[s] = degree.get(s, 0) + 1
        degree[d] = degree.get(d, 0) + 1
    if not degree:
        print("[错误] 图中没有可用的语义边。", file=sys.stderr)
        raise SystemExit(1)
    best = max(degree, key=lambda k: degree[k])
    print(f"        使用 sense_id={graph.node_ids[best]}「{graph.labels[best]}」"
          f"（度={degree[best]}）", file=sys.stderr)
    return best


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Logos Transformer prototype: relation-aware GAT over the Logos sense graph."
    )
    parser.add_argument("--db", type=pathlib.Path, default=root / "logos.db",
                        help="Path to logos.db (default: <repo>/logos.db)")
    parser.add_argument("--sense", default=DEFAULT_TARGET,
                        help="Target sense: surface form or sense_id (default: 猫)")
    parser.add_argument("--top-k", type=int, default=5, help="Neighbours to report (default: 5)")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden dimension (default: 32)")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--layers", type=int, default=2, help="GAT layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--include-closure", action="store_true",
                        help="Ablation: add depth>=2 closure edges as extra edge types")
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k 必须 >= 1")
    if args.layers < 1:
        parser.error("--layers 必须 >= 1")
    if args.heads < 1 or args.hidden % args.heads != 0:
        parser.error("--hidden 必须能被 --heads 整除")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows 控制台常用 cp936；显式 UTF-8 避免中文标签乱码/异常。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    args = parse_args(argv)
    db_path = args.db.expanduser().resolve()

    graph = load_graph(db_path, include_closure=args.include_closure, verbose=False)
    model = build_model(
        graph,
        hidden_dim=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        seed=args.seed,
    )

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    try:
        target_idx = _pick_default_target(graph, conn, args.sense)
        target_sid = graph.node_ids[target_idx]
        closure_ref = _closure_reference(conn, target_sid)
    finally:
        conn.close()

    embeddings, attentions = run_inference(model, graph)

    print_report(
        graph=graph,
        model=model,
        embeddings=embeddings,
        attentions=attentions,
        target_idx=target_idx,
        db_path=db_path,
        top_k=args.top_k,
        closure_ref=closure_ref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
