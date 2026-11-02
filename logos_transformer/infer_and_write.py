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
import json
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

# --calibrated 时的优先候选：cross-POS 重训 + 温度标定的产物（见 train_calibrated.py）。
# 只在显式要求标定时才优先，以免默认行为被悄悄改掉（round-12 的写回数字靠默认路径复现）。
CALIBRATED_CHECKPOINT_CANDIDATES: Tuple[str, ...] = (
    "./logos_transformer/checkpoints_calibrated/closure_gat_calibrated_best.pt",
    "./logos_transformer/checkpoints_calibrated/closure_gat_calibrated_untuned.pt",
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

# 词性修正脚本：刻意不叫 seed_*.sql —— init_database 会 glob 该模式并把命中文件
# 当作种子加载，那会连带改变下游所有测试的 ID 空间。这里必须由 --fix-pos 显式触发。
POS_FIX_SQL = _ROOT / "fix_pos_annotations.sql"


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


# --------------------------------------------------------------------------- #
# 符号守卫：源侧词性 / is_a 层级 / 关系方向 / 原子目标
#
# 四个守卫的判据全部从数据库自身读出（relations / closure / relation_types），
# 没有一条关系名、词性或深度阈值是硬编码的。判据缺失时一律放行（abstain），
# 所以「不误杀」是可审计的：每个守卫只在数据真的给出了矛盾证据时才否决。
# --------------------------------------------------------------------------- #
RelationMeta = Dict[str, Dict[str, object]]


def load_relation_source_pos(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    """Observed (relation -> {source POS: count}) over the ``relations`` table.

    Mirror image of :func:`load_relation_target_pos`: the target side was read off
    ``closure`` (that is where the *expected type of the answer* lives), while the
    source side can only come from the asserted edges.
    """
    out: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: collections.Counter()
    )
    for relation, pos in conn.execute(
        """
        SELECT r.relation, s.pos
        FROM relations r JOIN senses s ON s.sense_id = r.src_id
        WHERE r.src_kind = 'sense' AND r.dst_kind = 'sense' AND r.is_negated = 0
        """
    ):
        out[str(relation)][str(pos)] += 1
    return {relation: dict(counter) for relation, counter in out.items()}


def compute_depth_map(conn: sqlite3.Connection) -> Dict[int, int]:
    """sense_id -> MIN(is_a closure depth).  Senses outside the taxonomy are absent.

    ``closure.depth`` is the hop count from ``descendant`` to ``ancestor``, so the
    minimum over a sense's ancestors is its shortest distance to a root.

    **This map must not be used as a specificity measure.** Because almost every
    sense has a direct parent, ``MIN(depth)`` collapses to 1 for anything that is
    not a root: on this database 141 of the 152 existing direct ``is_a`` edges have
    ``MIN(depth)[src] == MIN(depth)[tgt]``, so a guard built on it would reject
    them.  It is kept for diagnostics (see :func:`print_guard_calibration`) and as
    the documented 口径 of the ``depth_estimate`` column, not as a guard input.
    """
    depths: Dict[int, int] = {}
    for descendant, depth in conn.execute(
        "SELECT descendant, MIN(depth) FROM closure "
        "WHERE relation = 'is_a' GROUP BY descendant"
    ):
        depths[int(descendant)] = int(depth)
    return depths


def compute_specificity(
    ancestors: Dict[int, Set[int]],
    in_taxonomy: Set[int],
) -> Dict[int, int]:
    """sense_id -> number of is_a ancestors ("how specific is this sense").

    This is the monotone counterpart of ``MIN(depth)``: adding ``src is_a tgt``
    makes ``tgt`` (and everything above it) an ancestor of ``src``, so a genuine
    edge always satisfies ``specificity[src] > specificity[tgt]`` -- strictly,
    because ``tgt`` itself is not yet among ``src``'s ancestors.  Measured on
    **152/152** existing direct ``is_a`` edges; ``MIN(depth)`` only satisfies
    **11/152**.
    """
    return {sid: len(ancestors.get(sid, ())) for sid in in_taxonomy}


def compute_ancestor_map(
    conn: sqlite3.Connection,
) -> Tuple[Dict[int, Set[int]], Set[int]]:
    """(sense_id -> is_a ancestors, nodes participating in the is_a taxonomy).

    The second element includes roots (which have no ancestors but are ancestors
    of others), so that ``猫 is_a 生物`` is not silently exempted from the guard.
    """
    ancestors: Dict[int, Set[int]] = collections.defaultdict(set)
    in_taxonomy: Set[int] = set()
    for descendant, ancestor in conn.execute(
        "SELECT descendant, ancestor FROM closure WHERE relation = 'is_a'"
    ):
        ancestors[int(descendant)].add(int(ancestor))
        in_taxonomy.add(int(descendant))
        in_taxonomy.add(int(ancestor))
    return ancestors, in_taxonomy


def load_relation_meta(conn: sqlite3.Connection) -> RelationMeta:
    """relation -> {inverse, symmetric, taxonomic, transitive} from ``relation_types``.

    Directionality is *declared by the schema*, not inferred from the data: every
    relation carries ``inverse_of`` / ``is_symmetric``.  Deriving it from observed
    edge pairs (the task's suggestion) would silently mis-classify relations whose
    inverse simply has not been asserted yet -- ``part_of`` has 59 edges and
    ``has_part`` 17, so "does the inverse exist somewhere" is not a property of the
    relation at all.
    """
    meta: RelationMeta = {}
    for relation, relation_class, inverse, symmetric, transitive in conn.execute(
        "SELECT relation, relation_class, inverse_of, is_symmetric, is_transitive "
        "FROM relation_types"
    ):
        meta[str(relation)] = {
            "class": str(relation_class),
            # 分类学关系：目标是「上位」，因此目标必须严格更浅（见 is_a_depth_guard）
            "taxonomic": str(relation_class) == "层级" and inverse is None,
            "inverse": str(inverse) if inverse else None,
            "symmetric": bool(symmetric),
            "transitive": bool(transitive),
        }
    return meta


def load_taxonomy_children(conn: sqlite3.Connection) -> Set[int]:
    """Senses that are the parent (ancestor) of at least one other sense in is_a.

    Membership == "this is a category, not an atom".  Used by the atomic-target
    guard, which asks the reverse question: is the *predicted* target a leaf?
    """
    return {
        int(row[0])
        for row in conn.execute("SELECT DISTINCT ancestor FROM closure WHERE relation = 'is_a'")
    }


def load_leaf_target_share(
    conn: sqlite3.Connection,
    children: Optional[Set[int]] = None,
) -> Dict[str, float]:
    """relation -> share of existing edges whose target is a leaf of the is_a tree.

    A "leaf" is a sense that is nobody's ancestor.  Property-like relations point
    at atoms (``有脊柱``, ``胡须``, ``潜水``); taxonomic relations point at
    categories by definition.  Measuring the share per relation is what lets the
    guard decide *which* relations demand an atomic target -- ``is_a`` scores
    0.00 and is therefore never constrained, while ``has_property`` scores 0.99.
    """
    if children is None:
        children = load_taxonomy_children(conn)

    totals: Dict[str, int] = collections.Counter()
    leaves: Dict[str, int] = collections.Counter()
    for relation, target in conn.execute(
        "SELECT relation, dst_id FROM relations "
        "WHERE src_kind = 'sense' AND dst_kind = 'sense' AND is_negated = 0"
    ):
        totals[str(relation)] += 1
        if int(target) not in children:
            leaves[str(relation)] += 1
    return {
        relation: leaves[relation] / total
        for relation, total in totals.items()
        if total > 0
    }


def build_leaf_guard(shares: Dict[str, float], min_share: float) -> Set[str]:
    """Relations whose observed targets are atomic enough to require a leaf target."""
    return {relation for relation, share in shares.items() if share >= min_share}


def is_a_depth_guard(
    source_id: int,
    target_id: int,
    specificity: Dict[int, int],
    ancestor_map: Dict[int, Set[int]],
    in_taxonomy: Set[int],
) -> bool:
    """For is_a: reject unless the target is *strictly* less specific than the source.

    Two necessary conditions for ``src is_a tgt``, both satisfied by **152/152**
    existing direct is_a edges (measured, see GUARDS_EVALUATION.md §2):

    1. ``spec[src] > spec[tgt]`` -- the parent sits strictly above the child in the
       ancestor partial order;
    2. ``ancestors(tgt) ⊆ ancestors(src)`` -- the parent's whole ancestry is also
       the child's ancestry.

    Condition 2 alone lets ``医学检测 is_a 影像特征`` through (both carry the
    identical ancestor set ``{医学概念, 抽象概念}``); condition 1 catches it.
    Condition 1 alone is weaker (any cross-branch edge with a depth gap passes),
    hence both.

    Abstains when either node is absent from the is_a taxonomy: outside the tree
    there is no ordering to compare, and guessing would create false positives.
    """
    if source_id not in in_taxonomy or target_id not in in_taxonomy:
        return True
    if specificity.get(target_id, 0) >= specificity.get(source_id, 0):
        return False
    return ancestor_map.get(target_id, set()) <= ancestor_map.get(source_id, set())


def direction_guard(
    source_id: int,
    relation: str,
    target_id: int,
    meta: RelationMeta,
    direct_edges: Set[Tuple[int, str, int]],
    closure_triples: Set[Tuple[int, str, int]],
) -> Tuple[bool, str]:
    """Reject a triple that contradicts a declared direction or closes a cycle.

    Returns ``(ok, reason)`` where ``reason`` names the rule that fired, so the
    report can attribute every rejection.

    * symmetric relations are exempt (``synonym``, ``opposite`` …);
    * ``inverse_of`` declared -> the same fact written backwards must not already
      exist (``A part_of B`` vs ``B has_part A``);
    * a reversed edge of the *same* relation -> the triple closes a cycle
      (``A is_a B`` when ``B is_a A`` is already implied).
    """
    info = meta.get(relation, {})
    if info.get("symmetric"):
        return True, ""

    inverse = info.get("inverse")
    if inverse:
        backwards = (target_id, str(inverse), source_id)
        if backwards in direct_edges or backwards in closure_triples:
            return False, f"inverse:{inverse}"

    reversed_same = (target_id, relation, source_id)
    if reversed_same in direct_edges or reversed_same in closure_triples:
        return False, "reverse"
    return True, ""


def guard_calibration_lines(
    is_a_edges: Sequence[Tuple[int, int]],
    depth_map: Dict[int, int],
    specificity: Dict[int, int],
    ancestor_map: Dict[int, Set[int]],
) -> List[str]:
    """Calibrate the is_a criteria against every existing direct is_a edge.

    The point is to make the choice of criterion falsifiable at runtime: if a
    future seed changes the taxonomy so that ``MIN(depth)`` becomes usable, or so
    that the adopted conditions stop holding, this block says so loudly instead of
    letting the guard silently reject true edges.
    """
    total = len(is_a_edges)
    if not total:
        return []

    def gt_depth(src: int, tgt: int) -> bool:
        return depth_map.get(src, 0) > depth_map.get(tgt, 0)

    def gt_spec(src: int, tgt: int) -> bool:
        return specificity.get(src, 0) > specificity.get(tgt, 0)

    def subset(src: int, tgt: int) -> bool:
        return ancestor_map.get(tgt, set()) <= ancestor_map.get(src, set())

    n_depth = sum(1 for s, t in is_a_edges if gt_depth(s, t))
    n_spec = sum(1 for s, t in is_a_edges if gt_spec(s, t))
    n_sub = sum(1 for s, t in is_a_edges if subset(s, t))
    n_both = sum(1 for s, t in is_a_edges if gt_spec(s, t) and subset(s, t))

    def mark(hits: int) -> str:
        return "  <- 采纳" if hits == total else (
            "  <- 不可用（会误杀真边）" if hits < total * 0.9 else "  <- 有反例"
        )

    return [
        f"用 {total} 条既有 is_a 直接边标定判据（不通过 = 该条件会否决真边）：",
        f"      MIN(depth) 严格递减      : {n_depth}/{total}{mark(n_depth)}",
        f"      祖先集大小严格递增        : {n_spec}/{total}{mark(n_spec)}",
        f"      祖先集被包含             : {n_sub}/{total}{mark(n_sub)}",
        f"      两条同时成立             : {n_both}/{total}{mark(n_both)}",
    ]


def apply_symbolic_guards(
    triples: Sequence[Tuple[int, str, int]],
    *,
    pos_of: Dict[int, str],
    source_guard: Optional[Dict[str, Set[str]]],
    hierarchical: bool,
    relation_meta: RelationMeta,
    specificity: Dict[int, int],
    ancestor_map: Dict[int, Set[int]],
    in_taxonomy: Set[int],
    leaf_relations: Set[str],
    children: Set[int],
    direct_edges: Set[Tuple[int, str, int]],
    closure_triples: Set[Tuple[int, str, int]],
) -> Tuple[List[Tuple[int, str, int]], Dict[str, List[Tuple[int, str, int]]]]:
    """Split candidates into (kept, {guard_name: dropped}).

    Guards are independent and short-circuit in a fixed order so that the
    attribution sums exactly to the number of candidates removed.  Order is chosen
    by "cheapest / most obviously wrong first": source POS, then is_a depth, then
    direction, then target atomicity.
    """
    kept: List[Tuple[int, str, int]] = []
    dropped: Dict[str, List[Tuple[int, str, int]]] = {
        "source_pos": [],
        "is_a_depth": [],
        "direction": [],
        "leaf_target": [],
    }

    for source_id, relation, target_id in triples:
        if source_guard is not None:
            allowed = source_guard.get(relation)
            if allowed is not None and pos_of.get(source_id) not in allowed:
                dropped["source_pos"].append((source_id, relation, target_id))
                continue

        if hierarchical and relation_meta.get(relation, {}).get("taxonomic"):
            if not is_a_depth_guard(
                source_id, target_id, specificity, ancestor_map, in_taxonomy
            ):
                dropped["is_a_depth"].append((source_id, relation, target_id))
                continue

        if hierarchical:
            ok, _reason = direction_guard(
                source_id, relation, target_id, relation_meta,
                direct_edges, closure_triples,
            )
            if not ok:
                dropped["direction"].append((source_id, relation, target_id))
                continue

        if relation in leaf_relations and target_id in children:
            # children 的成员 == 「还是别人的上位」== 是类别而非原子值
            dropped["leaf_target"].append((source_id, relation, target_id))
            continue

        kept.append((source_id, relation, target_id))
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


def checkpoint_temperature(
    state: Dict[str, "torch.Tensor"],
    path: Optional[pathlib.Path] = None,
) -> Optional[float]:
    """Recover the fitted temperature, if the checkpoint carries one.

    Two sources, in order: the ``temperature`` buffer written by
    ``train_calibrated.CalibratedClosurePredictor``, then a ``calibration.json``
    sidecar in the same directory.  Old (uncalibrated) checkpoints have neither
    and return ``None`` -- the caller must not silently treat that as ``T = 1``
    when the user explicitly asked for calibration.
    """
    tensor = state.get("temperature")
    if tensor is not None:
        try:
            return float(tensor.reshape(-1)[0].item())
        except (AttributeError, IndexError):  # pragma: no cover - 形状异常
            pass
    if path is not None:
        sidecar = path.parent / "calibration.json"
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                value = payload.get("temperature", {}).get("applied")
                if value is not None:
                    return float(value)
            except (json.JSONDecodeError, OSError, AttributeError, TypeError):
                pass
    return None


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
        "temperature": checkpoint_temperature(state),
    }


def resolve_checkpoint(
    explicit: Optional[pathlib.Path],
    wanted_relations: Sequence[str],
    prefer_calibrated: bool = False,
) -> Tuple[pathlib.Path, Dict[str, object], str]:
    """Pick a checkpoint and return (path, info, reason).

    Explicit ``--checkpoint`` always wins (the caller may know better); the
    automatic path prefers the checkpoint whose ``W_r`` heads cover the most of
    the requested relations.  ``prefer_calibrated`` only changes *which* list is
    tried first.
    """
    if explicit is not None:
        candidates: List[pathlib.Path] = [explicit]
    elif prefer_calibrated:
        candidates = [_ROOT / c for c in CALIBRATED_CHECKPOINT_CANDIDATES] + [
            _ROOT / c for c in DEFAULT_CHECKPOINT_CANDIDATES
        ]
    else:
        candidates = [_ROOT / c for c in DEFAULT_CHECKPOINT_CANDIDATES]

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
        # 标定过的 checkpoint 多一个 temperature 缓冲；基类没有这个键，
        # strict 装载会报「unexpected key」，所以先按模型自身的键表过滤。
        raw_state = _torch_load(path)
        own_keys = set(model.state_dict())
        filtered = {k: v for k, v in raw_state.items() if k in own_keys}
        dropped = sorted(set(raw_state) - own_keys)
        model.load_state_dict(filtered)
        if dropped:
            print(f"    [注] checkpoint 里有 {len(dropped)} 个非模型键被忽略："
                  f"{', '.join(dropped)}")
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
def score_triples_logits(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    triples: Sequence[Tuple[int, str, int]],
    h: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Raw logits ``h_s^T W_r h_t`` for every triple; encodes the graph at most once.

    Logits (not sigmoid probabilities) are returned so temperature scaling can be
    applied afterwards without recomputing anything: both the raw and the
    calibrated view are monotone transforms of the same tensor.
    """
    if not triples:
        return torch.zeros(0)
    if h is None:
        h = encode_graph(model, graph)

    by_relation: Dict[str, List[Tuple[int, int, int]]] = collections.defaultdict(list)
    for pos, (src, relation, target) in enumerate(triples):
        by_relation[relation].append(
            (pos, graph.id_to_index[src], graph.id_to_index[target])
        )

    out = torch.zeros(len(triples), dtype=torch.float32)
    for relation, items in by_relation.items():
        matrix = model.score_all(h, relation)  # (N, N) 全矩阵，一次 matmul
        positions = torch.tensor([i[0] for i in items], dtype=torch.long)
        rows = torch.tensor([i[1] for i in items], dtype=torch.long)
        cols = torch.tensor([i[2] for i in items], dtype=torch.long)
        out[positions] = matrix[rows, cols]
    return out


@torch.no_grad()
def score_triples(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    triples: Sequence[Tuple[int, str, int]],
    h: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> List[float]:
    """sigmoid(logit / T) for every triple; ``T = 1`` is the raw model."""
    logits = score_triples_logits(model, graph, triples, h=h)
    if logits.numel() == 0:
        return []
    probs = torch.sigmoid(logits / max(float(temperature), 1e-6))
    return [float(v) for v in probs.tolist()]


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


def apply_pos_fixes(
    db_path: pathlib.Path,
    sql_path: pathlib.Path = POS_FIX_SQL,
) -> Tuple[int, str]:
    """Run ``fix_pos_annotations.sql``; returns ``(rows_changed, message)``.

    The statement is a plain ``UPDATE`` so it is idempotent by construction: a
    second run changes nothing.  Note that re-seeding reverts it -- the seeds are
    the source of truth for ``pos``, and this file is an explicit override, applied
    by ``--fix-pos`` at inference time rather than baked into the schema.
    """
    if not sql_path.exists():
        return 0, f"找不到词性修正脚本：{sql_path}"
    script = sql_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(str(db_path))
    try:
        # 不能用 conn.total_changes：senses 上有 FTS 同步触发器
        # （tr_senses_fts_au），一次 UPDATE 会连带写入 FTS 影子表，
        # total_changes 会虚报成 9~14 行。改为直接对比 pos 快照。
        before = dict(conn.execute("SELECT sense_id, pos FROM senses"))
        conn.executescript(script)
        conn.commit()
        after = dict(conn.execute("SELECT sense_id, pos FROM senses"))
    finally:
        conn.close()
    changed = sum(1 for sid, pos in after.items() if before.get(sid) != pos)
    return changed, f"已应用 {sql_path.name}（改动 {changed} 个义项）"


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #
def _bar(value: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def _pct(part: int, whole: int) -> str:
    """``part`` as a percentage of ``whole``, ``n/a`` instead of dividing by zero."""
    if whole <= 0:
        return "n/a"
    return f"{part / whole * 100:.1f}%"


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
    temperature: float = 1.0,
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
        probs = torch.sigmoid(matrix / max(float(temperature), 1e-6))
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
    parser.add_argument("--calibrated", action="store_true",
                        help="对打分做温度标定：sigmoid(logit / T)。T 优先取 --temperature，"
                             "其次取 checkpoint 的 temperature 缓冲 / 同目录 calibration.json；"
                             "两者都没有时退回 T=1 并给出警告。同时打印同一阈值下"
                             "「未标定 vs 已标定」的写入量对比。")
    parser.add_argument("--temperature", type=float, default=None,
                        help="手动指定温度，覆盖 checkpoint 里的值（隐含 --calibrated）")
    parser.add_argument("--seed", type=int, default=42,
                        help="重建模型时的初始化种子（只为可复现，权重随后被覆盖）")
    parser.add_argument("--source-pos-guard", action="store_true",
                        help="源侧词性守卫：源词性在该关系的既有源侧分布中占比低于 "
                             "--pos-min-share 时丢弃该候选（分布由 relations+senses 读出）")
    parser.add_argument("--hierarchical-guard", action="store_true",
                        help="层级/方向守卫：分类学关系要求目标严格更浅（深度递减且祖先集被包含），"
                             "并禁止与已声明的逆关系或反向边冲突的三元组（依据 relation_types）")
    parser.add_argument("--leaf-target-guard", action="store_true",
                        help="原子目标守卫：目标必须是 is_a 叶子；只对「既有目标几乎全是叶子」"
                             "的关系生效（阈值 --leaf-min-share）")
    parser.add_argument("--leaf-min-share", type=float, default=0.95,
                        help="判定「该关系要求原子目标」的最低叶子占比，默认 0.95")
    parser.add_argument("--fix-pos", action="store_true",
                        help="推理前应用 fix_pos_annotations.sql（幂等 UPDATE，写 senses.pos）")
    args = parser.parse_args(argv)

    if args.temperature is not None:
        if args.temperature <= 0:
            parser.error("--temperature 必须 > 0")
        args.calibrated = True

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
    if not (0.0 <= args.leaf_min_share <= 1.0):
        parser.error("--leaf-min-share 必须在 [0, 1]")
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

    # ---- 词性修正（显式动作，先于任何读取）-------------------------------- #
    if args.fix_pos:
        changed, message = apply_pos_fixes(db_path)
        print(f"[词性修正] {message}")
        if args.dry_run:
            print("          注意：--dry-run 不抑制 --fix-pos —— 词性修正是显式动作，"
                  "会写 senses.pos。")

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
        source_pos = load_relation_source_pos(ro)
        depth_map = compute_depth_map(ro)
        ancestor_map, in_taxonomy = compute_ancestor_map(ro)
        specificity = compute_specificity(ancestor_map, in_taxonomy)
        is_a_edges = [
            (int(r[0]), int(r[1]))
            for r in ro.execute(
                "SELECT src_id, dst_id FROM relations WHERE relation = 'is_a' "
                "AND src_kind = 'sense' AND dst_kind = 'sense'"
            )
        ]
        children = load_taxonomy_children(ro)
        leaf_shares = load_leaf_target_share(ro, children)
        relation_meta = load_relation_meta(ro)
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

    # ---- 符号守卫（全部 opt-in）------------------------------------------- #
    # 注意：这里只做「判定」，不立刻从 candidates 里删 —— 打分函数
    # score_triples_logits 内部对每个关系算一次 N×N 全矩阵，候选条数几乎不影响
    # 计算量。保留被否决的候选一起打分，才能在报表里给出精确的
    # 「不加守卫本会写多少」对照，而不是拍一个估算值。
    source_guard = (
        build_pos_guard(source_pos, args.pos_min_share)
        if args.source_pos_guard
        else None
    )
    leaf_relations = (
        build_leaf_guard(leaf_shares, args.leaf_min_share)
        if args.leaf_target_guard
        else set()
    )
    guarded_triples, dropped_by_guard = apply_symbolic_guards(
        candidates,
        pos_of=pos_of,
        source_guard=source_guard,
        hierarchical=args.hierarchical_guard,
        relation_meta=relation_meta,
        specificity=specificity,
        ancestor_map=ancestor_map,
        in_taxonomy=in_taxonomy,
        leaf_relations=leaf_relations,
        children=children,
        direct_edges=set(edges),
        closure_triples=closure_triples,
    )
    kept_set: Set[Tuple[int, str, int]] = set(guarded_triples)

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

    # ---- 符号守卫报表 ------------------------------------------------------ #
    n_guarded = len(guarded_triples)
    if source_guard is not None or args.hierarchical_guard or leaf_relations:
        print()
        print("  [符号守卫] 判据全部来自数据库（relations / closure / relation_types）：")
        if source_guard is not None:
            n_src_drop = len(dropped_by_guard["source_pos"])
            print(f"    源侧词性   : 剔除 {n_src_drop:,} 条"
                  f"（{_pct(n_src_drop, len(candidates))}）"
                  f"  阈值 pos 占比 >= {args.pos_min_share:.2f}")
            for relation in requested:
                counts = source_pos.get(relation, {})
                total = sum(counts.values()) or 1
                dist = "  ".join(
                    f"{pos}:{count / total * 100:.0f}%"
                    for pos, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                )
                allowed = sorted(source_guard.get(relation, ()))
                print(f"      {relation:<16} {dist:<24} -> 允许 {','.join(allowed) or '(不限)'}")
        if args.hierarchical_guard:
            n_depth = len(dropped_by_guard["is_a_depth"])
            n_dir = len(dropped_by_guard["direction"])
            print(f"    层级深度   : 剔除 {n_depth:,} 条（{_pct(n_depth, len(candidates))}）"
                  f"  分类学关系要求「祖先集大小严格递增 + 祖先集被包含」")
            for line in guard_calibration_lines(
                is_a_edges, depth_map, specificity, ancestor_map
            ):
                print(f"      {line}")
            print(f"    方向       : 剔除 {n_dir:,} 条（{_pct(n_dir, len(candidates))}）"
                  f"  禁止与已声明逆关系 / 反向边冲突")
        if leaf_relations:
            n_leaf = len(dropped_by_guard["leaf_target"])
            print(f"    原子目标   : 剔除 {n_leaf:,} 条（{_pct(n_leaf, len(candidates))}）"
                  f"  生效关系 {', '.join(sorted(leaf_relations))}")
        print(f"    合计放行   : {n_guarded:,} / {len(candidates):,} 条"
              f"（{_pct(n_guarded, len(candidates))}）")
    else:
        print()
        print("  [符号守卫] 全部关闭（--source-pos-guard / --hierarchical-guard /"
              " --leaf-target-guard）")

    # ---- checkpoint + 图 --------------------------------------------------- #
    checkpoint, info, reason = resolve_checkpoint(
        args.checkpoint, requested, prefer_calibrated=args.calibrated
    )
    if not checkpoint.exists():  # pragma: no cover - resolve 已保证
        print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
        return 1

    # 温度解析：显式 > checkpoint 缓冲 > 同目录 calibration.json > 1.0
    temperature = 1.0
    temperature_source = "无标定（T = 1）"
    if args.calibrated:
        if args.temperature is not None:
            temperature = float(args.temperature)
            temperature_source = "--temperature 显式指定"
        elif info.get("temperature") is not None:
            temperature = float(info["temperature"])  # type: ignore[arg-type]
            temperature_source = "checkpoint 的 temperature 缓冲"
        else:
            temperature_source = "该 checkpoint 不含温度，退回 T = 1"
            print("[警告] --calibrated 但该 checkpoint 不含温度信息，T 退回 1.0。"
                  "\n       标定模型见 train_calibrated.py 的 checkpoints_calibrated/ 产物。",
                  file=sys.stderr)

    print()
    print("  [模型]")
    print(f"    checkpoint             : {checkpoint}")
    print(f"    选择依据               : {reason}")
    print(f"    结构                   : {info['num_layers']} 层 / {info['num_heads']} 头 / "
          f"hidden={info['hidden']} / {info['num_nodes']} 节点")
    print(f"    专属 W_r 头            : {len(info['head_names'])} 个 -> "
          f"{', '.join(info['head_names'][:8])}"
          + (" ..." if len(info['head_names']) > 8 else ""))
    print(f"    温度 T                 : {temperature:.4f}（{temperature_source}）"
          + ("  [标定已启用]" if args.calibrated else "  [未标定]"))

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
    logits = score_triples_logits(model, graph, candidates, h=h)
    raw_probs = torch.sigmoid(logits).tolist()
    cal_probs = torch.sigmoid(logits / max(temperature, 1e-6)).tolist()
    probs = cal_probs if args.calibrated else raw_probs
    scored = [
        (src, relation, target, confidence)
        for (src, relation, target), confidence in zip(candidates, probs)
    ]
    scored.sort(key=lambda item: (-item[3], item[1], item[0], item[2]))

    epochs_note = args.epoch if args.epoch is not None else "NULL"
    model_version = f"{RELAWARE_VERSION}@{checkpoint.parent.name}"
    if args.calibrated:
        model_version += f"+T{temperature:.4f}"

    above = [t for t in scored if t[3] >= args.min_confidence]
    # 「不加符号守卫本会写多少」：刻意在同一次打分结果上取，不是估算 ——
    # 因为被否决的候选也已打分，两个计数只差一个集合判定。
    gated_unfiltered = limit_per_group(above, args.max_per_source_relation)
    above_guarded = [t for t in above if (t[0], t[1], t[2]) in kept_set]
    gated = limit_per_group(above_guarded, args.max_per_source_relation)

    if len(guarded_triples) != len(candidates):
        kept_rows = {(s, r, t) for s, r, t, _c in gated}
        base_rows = {(s, r, t) for s, r, t, _c in gated_unfiltered}
        promoted = len(kept_rows - base_rows)
        reason_of = {
            triple: name
            for name, items in dropped_by_guard.items()
            for triple in items
        }
        killed_by: Dict[str, List[float]] = collections.defaultdict(list)
        for src, relation, target, confidence in gated_unfiltered:
            reason = reason_of.get((src, relation, target))
            if reason:
                killed_by[reason].append(confidence)
        print()
        print("  [守卫效果]")
        print(f"    阈值 {args.min_confidence:.2f} 以上的候选 : "
              f"不加守卫 {len(above):,} 条 / 加守卫 {len(above_guarded):,} 条")
        print(f"    每组 top-{args.max_per_source_relation} 后写入 : "
              f"{len(gated_unfiltered):,} 行 -> {len(gated):,} 行"
              f"（{_pct(len(gated), len(gated_unfiltered))}）")
        label = {"source_pos": "源侧词性", "is_a_depth": "层级深度",
                 "direction": "方向", "leaf_target": "原子目标"}
        detail = "  ".join(
            f"{label.get(name, name)} {len(values)} 行"
            for name, values in sorted(killed_by.items(), key=lambda kv: -len(kv[1]))
        )
        print(f"    被砍掉的既有假设 : {detail or '无'}")
        if killed_by:
            killed_conf = [c for values in killed_by.values() for c in values]
            kept_conf = [c for _s, _r, _t, c in gated]
            print(f"    平均置信度       : 留下 {sum(kept_conf) / max(len(kept_conf), 1):.4f}"
                  f" / 砍掉 {sum(killed_conf) / len(killed_conf):.4f}"
                  "（砍掉的更高，说明错的就是最自信的那批）")
        print(f"    其中 {promoted} 行是「补位」进来的：守卫砍掉该组最优假设后，"
              f"原本排第 {args.max_per_source_relation + 1} 位的假设顶上。"
              "故行数降幅小于否决条数。")

    # 同一阈值下的「未标定 vs 已标定」对比：温度是单调变换，所以能动的只有
    # 「跨过阈值的那一部分」，条数下降不代表错误减少 —— 这一点必须打出来。
    if args.calibrated and abs(temperature - 1.0) > 1e-9:
        raw_above = [t for (s, r, tg), t in zip(candidates, raw_probs)
                     if t >= args.min_confidence]
        raw_gated = limit_per_group(
            [(s, r, tg, t) for (s, r, tg), t in zip(candidates, raw_probs)
             if t >= args.min_confidence],
            args.max_per_source_relation,
        )
        print(f"    温度标定 T={temperature:.4f}：阈值 {args.min_confidence:.2f} 下 "
              f"未标定 {len(raw_above):,} 行 / 已标定 {len(above):,} 行；"
              f"每组 top-{args.max_per_source_relation} 后 "
              f"{len(raw_gated):,} -> {len(gated):,} 行")
        print("    （T 严格单调：排序不变，改的只是「落在阈值哪一侧」，"
              "不等于错误变少）")

    print_sanity(
        model=model, graph=graph, h=h, labels=labels,
        closure_triples=closure_triples, relations=requested,
        temperature=temperature,
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
