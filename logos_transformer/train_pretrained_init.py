#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用外部预训练词向量初始化 GAT 的节点特征 —— 冷启动实验。

背景
----
335 个义项的图只有 694 条语义边，平均每节点 2 条。校准 + 符号守卫之后仍剩
118 条写回，错误都是「需要外部语义知识」的类型（``鱼 is_a 无脊椎动物``、
``蛇 has_property 温血``、``猫 is_a 人``）。这些判据不在库里，靠 1~2 条邻边
学不出来。这个脚本给每个节点一个**稠密的语义初值**，看能不能补上这个缺口。

两种架构
--------
``--arch table``（默认）
    保持 ``LogosGAT`` 原结构（``nn.Embedding(N, hidden)``），只把 ``weight``
    的初值换成 ``proj(词向量)``。参数字典与基线**逐键一致**，因此
    ``infer_and_write.py`` 可以直接加载（它按 ``gat.node_embedding.weight``
    的形状反推架构）。训练时权重照常更新 = fine-tune 每个节点。

``--arch dense``
    任务书原方案：``input_proj = Linear(D, hidden)`` + ELU，节点表示由稠密
    向量算出。参数量更少（D×hidden 而不是 N×hidden），但 checkpoint 多了
    ``gat.input_proj.*`` 键，写回管线认不出来。所以训练结束后会**折叠**成
    等价的 table 存档（``*_folded.pt``）：eval 模式下 dropout 关闭，
    ``E = ELU(proj(V))`` 与 ``forward`` 的第一步严格等价，折叠是精确的。

初始化尺度
----------
``LogosGAT`` 原本 ``normal_(std=0.1)``，每行范数 ≈ ``0.1*sqrt(hidden)``。
投影后的向量按同一范数重标（``rescale_rows``），否则 56 维属性表投影出来的
范数是随机初始化的 3~4 倍，logits 一开始就饱和、训练不动。

口径
----
数据切分 / 负样本 / 全部指标函数都从 ``train_closure_relaware`` 直接 import，
不重写。``--init random`` 必须逐位复现基线（test MRR 0.3609 / hard MRR
0.6938 / 写回 154 行），否则同口径不成立、后面所有对比都无效。

用法
----
    # 同口径对照（必须先跑这个）
    python logos_transformer/train_pretrained_init.py --init random --epochs 5

    # mini 向量（仓库自带，手写属性表 —— 只是上界探针）
    python logos_transformer/train_pretrained_init.py --init mini --epochs 200

    # 零外部知识的字形哈希对照
    python logos_transformer/train_pretrained_init.py --init charngram

    # 真实预训练向量（腾讯 / fastText），流式过滤只留 335 个词
    python logos_transformer/train_pretrained_init.py \
        --init file --vectors /path/to/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt \
        --embedding-dim 200
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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent

# --------------------------------------------------------------------------- #
# 复用基线脚本：数据切分、负样本、全部指标。刻意不重写，保证同口径。
# --------------------------------------------------------------------------- #
try:  # 作为包导入
    from .embeddings import (  # type: ignore
        ChineseWordVectorLoader,
        build_node_features,
        charngram_loader,
        collect_sense_words,
        keep_set_for,
        load_mini,
    )
    from .prototype_gat import LogosGAT, RelationGraph, build_model, load_graph  # type: ignore
    from .train_closure import Facts, Sample, _fmt, labels_tensor, pairs_tensor  # type: ignore
    from .train_closure_disjoint import load_positives, split_by_source  # type: ignore
    from .train_closure_hardneg import (  # type: ignore
        _split_ratios,
        build_negative_pools,
        sample_hardneg,
    )
    from .train_closure_relaware import (  # type: ignore
        SPLIT_NAMES,
        RelationAwareClosurePredictor,
        encode_and_score,
        eval_bucket_relaware,
        hard_metrics,
        loss_acc,
        target_prior_relaware,
    )
    from .train_calibrated import (  # type: ignore
        allocate_tiers,
        build_ood_probe,
        rejection_rate,
        sample_negatives_crosspos,
        wrong_pos_index,
    )
    from . import infer_and_write as iw  # type: ignore
    from . import train_calibrated as calib  # type: ignore
except ImportError:  # pragma: no cover - 直接脚本运行
    from embeddings import (  # type: ignore
        ChineseWordVectorLoader,
        build_node_features,
        charngram_loader,
        collect_sense_words,
        keep_set_for,
        load_mini,
    )
    from prototype_gat import LogosGAT, RelationGraph, build_model, load_graph  # type: ignore
    from train_closure import Facts, Sample, _fmt, labels_tensor, pairs_tensor  # type: ignore
    from train_closure_disjoint import load_positives, split_by_source  # type: ignore
    from train_closure_hardneg import (  # type: ignore
        _split_ratios,
        build_negative_pools,
        sample_hardneg,
    )
    from train_closure_relaware import (  # type: ignore
        SPLIT_NAMES,
        RelationAwareClosurePredictor,
        encode_and_score,
        eval_bucket_relaware,
        hard_metrics,
        loss_acc,
        target_prior_relaware,
    )
    from train_calibrated import (  # type: ignore
        allocate_tiers,
        build_ood_probe,
        rejection_rate,
        sample_negatives_crosspos,
        wrong_pos_index,
    )
    import infer_and_write as iw  # type: ignore
    import train_calibrated as calib  # type: ignore


VERSION = "logos-gat-pretrained-init-v0.1"
DEFAULT_OUTPUT_DIR = str(_ROOT / "logos_transformer" / "checkpoints_pretrained")

# 与基线脚本保持一致的默认值（改这里就等于改对照口径，务必同步）。
DEFAULT_HIDDEN = 32
DEFAULT_HEADS = 4
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.1


# --------------------------------------------------------------------------- #
# 特征投影
# --------------------------------------------------------------------------- #
def rescale_rows(mat: torch.Tensor, target: float) -> torch.Tensor:
    """把每行缩放到同一个 L2 范数；全零行保持全零（OOV 留给图结构去补）。"""
    norms = mat.norm(dim=1, keepdim=True)
    scale = target / norms.clamp_min(1e-8)
    out = mat * scale
    out = torch.where(norms > 1e-8, out, torch.zeros_like(out))
    return out


def project_features(
    features: torch.Tensor,
    hidden: int,
    *,
    method: str = "orth",
    seed: int = 42,
    target_norm: Optional[float] = None,
) -> torch.Tensor:
    """``[N, D]`` → ``[N, hidden]``，并重标到与随机初始化相同的范数。

    ``method``:
        * ``orth``  —— 固定种子的随机正交投影（保范数，默认）
        * ``pca``   —— 取协方差前 ``hidden`` 个特征向量（确定性、无随机）
        * ``slice`` —— 直接截断 / 右侧补零（最笨，用于排除「投影本身有用」）
    """
    if features.dim() != 2:
        raise SystemExit(f"features 必须是二维，得到 {tuple(features.shape)}")
    n, d = features.shape
    x = features.float()
    if d == hidden:
        out = x
    elif d < hidden:
        out = torch.cat([x, torch.zeros(n, hidden - d)], dim=1)
    elif method == "slice":
        out = x[:, :hidden]
    elif method == "pca":
        xc = x - x.mean(dim=0, keepdim=True)
        cov = xc.t() @ xc / max(n - 1, 1)
        _vals, vecs = torch.linalg.eigh(cov)          # 升序
        out = xc @ vecs[:, -hidden:]
    elif method == "orth":
        g = torch.Generator().manual_seed(int(seed))
        w = torch.randn(d, hidden, generator=g)
        q, _ = torch.linalg.qr(w)                     # [D, hidden] 正交列
        out = x @ q
    else:
        raise SystemExit(f"未知投影方式：{method}")

    norm = target_norm if target_norm is not None else 0.1 * math.sqrt(hidden)
    return rescale_rows(out, norm)


def feature_coverage_stats(features: torch.Tensor, hidden: int) -> Dict[str, object]:
    nonzero = (features.abs().sum(dim=1) > 1e-12).sum().item()
    norms = features.norm(dim=1)
    live = norms[norms > 1e-12]
    return {
        "num_nodes": int(features.size(0)),
        "dim": int(hidden),
        "nonzero_rows": int(nonzero),
        "zero_rows": int(features.size(0) - nonzero),
        "row_norm_mean": float(live.mean()) if live.numel() else 0.0,
        "row_norm_min": float(live.min()) if live.numel() else 0.0,
        "row_norm_max": float(live.max()) if live.numel() else 0.0,
    }


# --------------------------------------------------------------------------- #
# 稠密输入架构（--arch dense）
# --------------------------------------------------------------------------- #
class DenseInputGAT(nn.Module):
    """``input_proj: D→hidden`` + ELU，替代 ``node_embedding`` 查表。

    其余部分（关系感知注意力层、层间 ELU、dropout）与 ``LogosGAT`` 完全相同，
    这样两边的差异只有「节点初值从哪来」，别的都不动。

    ``input_features`` 注册成 buffer，所以 checkpoint 自包含：加载时不用再给
    向量文件。
    """

    def __init__(
        self,
        input_features: torch.Tensor,
        num_edge_types: int,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        from prototype_gat import RelationAwareGATLayer  # 局部导入，避免顶层循环

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.register_buffer("input_features", input_features.float())
        in_dim = int(input_features.size(1))
        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        g = torch.Generator().manual_seed(int(seed))
        w = torch.randn(hidden_dim, in_dim, generator=g)   # [out, in]
        q, _ = torch.linalg.qr(w.t())                     # [in, hidden]
        with torch.no_grad():
            self.input_proj.weight.copy_(q.t().contiguous())
            self.input_proj.bias.zero_()
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

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        h = F.elu(self.input_proj(self.input_features))
        h = F.dropout(h, p=self.dropout, training=self.training)
        attentions: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            h, alpha = layer(h, edge_index, edge_type)
            attentions.append(alpha)
            if i < len(self.layers) - 1:
                h = F.elu(h)
        return h, attentions


# --------------------------------------------------------------------------- #
# 模型构造
# --------------------------------------------------------------------------- #
def build_predictor(
    graph: RelationGraph,
    *,
    arch: str,
    hidden: int,
    heads: int,
    layers: int,
    dropout: float,
    seed: int,
    features: Optional[torch.Tensor],
    relation_counts: Dict[str, int],
    head_relations: Optional[Sequence[str]] = None,
) -> Tuple[RelationAwareClosurePredictor, Dict[str, object]]:
    """建模型；``features is None`` 或不给 arch 变换 ⇒ 与基线完全相同。"""
    info: Dict[str, object] = {"arch": arch, "init": "random", "init_stats": {}}
    if arch == "table":
        base = build_model(
            graph,
            hidden_dim=hidden,
            num_heads=heads,
            num_layers=layers,
            dropout=dropout,
            seed=seed,
        )
        if features is not None:
            projected = project_features(features, hidden, seed=seed)
            with torch.no_grad():
                base.node_embedding.weight.copy_(projected)
            info["init"] = "pretrained"
            info["init_stats"] = feature_coverage_stats(projected, hidden)
    elif arch == "dense":
        if features is None:
            raise SystemExit("--arch dense 需要节点特征；--init random 请用 --arch table")
        in_dim = int(features.size(1))
        # 输入也按 table 分支的尺度重标：正交投影保范数，所以 proj(V) 的范数
        # 就等于 ||V||。不重标的话原始 mini 向量范数约 2.4，是随机初始化的
        # 4 倍，logits 一上来就饱和 —— 那样 dense 与 table 的差异里混进了尺度
        # 效应，不是「架构差异」。
        scaled = rescale_rows(features.float(), 0.1 * math.sqrt(hidden))
        # 与 table 分支保持同一随机起点，便于对照注意力层初始化
        torch.manual_seed(seed)
        base = DenseInputGAT(
            scaled,
            graph.num_edge_types,
            hidden_dim=hidden,
            num_heads=heads,
            num_layers=layers,
            dropout=dropout,
            seed=seed,
        )
        info["init"] = "pretrained"
        with torch.no_grad():
            h0 = F.elu(base.input_proj(base.input_features))
        info["init_stats"] = feature_coverage_stats(h0, hidden)
        info["input_dim"] = in_dim
    else:
        raise SystemExit(f"未知架构：{arch}")

    model = RelationAwareClosurePredictor(
        base, hidden, graph.relation_names, relation_counts, head_relations=head_relations
    )
    return model, info


def fold_dense_checkpoint(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    *,
    hidden: int,
    heads: int,
    layers: int,
    dropout: float,
) -> Dict[str, "torch.Tensor"]:
    """把 ``DenseInputGAT`` 折叠成等价的 table 参数表，供写回管线加载。

    eval 模式下 dropout 关闭，``LogosGAT.forward`` 的第一步就是
    ``F.dropout(weight)``，所以 ``weight := ELU(proj(V))`` 与稠密架构严格等价。
    """
    state = model.state_dict()
    gat = model.gat
    with torch.no_grad():
        folded = F.elu(gat.input_proj(gat.input_features))
    out: Dict[str, "torch.Tensor"] = {}
    for key, value in state.items():
        if key.startswith("gat.input_proj.") or key == "gat.input_features":
            continue
        out[key] = value
    out["gat.node_embedding.weight"] = folded.detach().clone()
    return out


# --------------------------------------------------------------------------- #
# 节点特征载入
# --------------------------------------------------------------------------- #
def load_node_features(
    db_path: pathlib.Path,
    graph: RelationGraph,
    *,
    init: str,
    vectors: Optional[pathlib.Path],
    embedding_dim: Optional[int],
    oov_mode: str,
    seed: int,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    """按 ``--init`` 取节点特征；对齐到 ``graph.node_ids`` 的行序。"""
    if init == "random":
        return None, {"init": "random", "coverage": 0.0}

    sense_words = collect_sense_words(db_path)
    sid_order = [sid for sid, _w, _p in sense_words]

    if init in ("mini", "charngram"):
        if init == "mini":
            loader = load_mini(verbose=False)
            source = "mini_vectors.txt（手写属性表）"
        else:
            loader = charngram_loader(dim=embedding_dim or 64)
            source = "字形三元组哈希（零外部数据）"
            # charngram 加载器本身是空的（没有任何真词），必须把 OOV 回退切到
            # charngram 才会产出向量。忘了这一步的话每个义项都拿到零向量，
            # 模型退化成「只有 GAT」，指标会掉到随机水平 —— 静默失败，不报错。
            oov_mode = "charngram"
        dim = embedding_dim or loader.dim
    elif init == "file":
        if vectors is None:
            raise SystemExit("--init file 需要 --vectors <path>")
        keep = keep_set_for(sense_words)
        loader = ChineseWordVectorLoader(
            vectors, dim=embedding_dim, keep=keep, verbose=True
        )
        source = f"{pathlib.Path(vectors).name}（流式过滤到 {len(keep)} 个键）"
        dim = loader.dim or (embedding_dim or 200)
    else:
        raise SystemExit(f"未知 --init：{init}")

    report = loader.coverage_report([(w, p) for _s, w, p in sense_words], oov_mode=oov_mode)
    print("-" * 100)
    for line in report.lines():
        print(line)

    features, stats = build_node_features(
        db_path, loader, dim, oov_mode=oov_mode, verbose=False
    )

    # 行序对齐：build_node_features 按 sense_id 升序，图节点顺序未必相同
    if sid_order == list(graph.node_ids):
        perm_note = "sense_id 升序（与图节点顺序一致）"
    else:
        index_of = {sid: i for i, sid in enumerate(sid_order)}
        perm = torch.tensor([index_of[sid] for sid in graph.node_ids], dtype=torch.long)
        features = features[perm]
        perm_note = "已按 graph.node_ids 重排（两者顺序不同）"
    print(f"行序对齐      : {perm_note}")

    info = {
        "init": init,
        "source": source,
        "dim": int(features.size(1)),
        "coverage": float(stats["coverage"]),
        "covered": int(stats["covered"]),
        "total": int(stats["num_nodes"]),
        "by_level": stats["by_level"],
        "perm": perm_note,
    }
    return features, info


# --------------------------------------------------------------------------- #
# 打分探针 / 余弦报告
# --------------------------------------------------------------------------- #
def resolve_sense(graph: RelationGraph, conn: sqlite3.Connection, spec: str) -> Optional[int]:
    """``猫`` 或 ``猫/n`` 或 ``猫:12`` → 图节点下标。"""
    if ":" in spec and spec.rsplit(":", 1)[-1].isdigit():
        sid = int(spec.rsplit(":", 1)[-1])
        return graph.id_to_index.get(sid)
    word, _, pos = spec.partition("/")
    sql = (
        "SELECT s.sense_id FROM senses s JOIN words w ON w.word_id = s.word_id "
        "WHERE w.word = ?"
    )
    rows = conn.execute(sql + (" AND s.pos = ?" if pos else ""),
                        (word, pos) if pos else (word,)).fetchall()
    if not rows:
        return None
    if len(rows) > 1 and not pos:
        return None
    return graph.id_to_index.get(int(rows[0][0]))


def score_triples(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    conn: sqlite3.Connection,
    triples: Sequence[Tuple[str, str, str]],
) -> List[Dict[str, object]]:
    """对 ``(src, relation, tgt)`` 逐条打分（raw logit + sigmoid）。"""
    model.eval()
    out: List[Dict[str, object]] = []
    with torch.no_grad():
        h, _ = model.encode(graph.edge_index, graph.edge_type)
        for src_spec, relation, tgt_spec in triples:
            si = resolve_sense(graph, conn, src_spec)
            ti = resolve_sense(graph, conn, tgt_spec)
            if si is None or ti is None:
                out.append({
                    "src": src_spec, "relation": relation, "tgt": tgt_spec,
                    "error": "未解析到节点", "logit": float("nan"), "prob": float("nan"),
                })
                continue
            pairs = torch.tensor([[si, ti]], dtype=torch.long)
            logit = float(model.score_pairs(h, pairs, [relation])[0])
            out.append({
                "src": src_spec, "relation": relation, "tgt": tgt_spec,
                "logit": logit, "prob": 1.0 / (1.0 + math.exp(-logit)),
            })
    return out


def cosine_between(vectors: Dict[str, torch.Tensor], a: str, b: str) -> Optional[float]:
    va, vb = vectors.get(a), vectors.get(b)
    if va is None or vb is None:
        return None
    na, nb = va.norm(), vb.norm()
    if float(na) < 1e-9 or float(nb) < 1e-9:
        return None
    return float(torch.dot(va, vb) / (na * nb))


# --------------------------------------------------------------------------- #
# 金标评估
# --------------------------------------------------------------------------- #
def load_gold(path: pathlib.Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise SystemExit(f"金标格式不对：{path}（期望 list 或 {{'items': [...]}}）")
    return items


def gold_report(
    model: RelationAwareClosurePredictor,
    graph: RelationGraph,
    conn: sqlite3.Connection,
    gold: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """对金标每条打分，分两组报告。

    **写回组**（``control`` 缺省为假）才是「模型实际会写进库的东西」，它按
    置信度 top-3 挑出来，天然是高分条 —— 所以只能在**组内**比较正确与错误，
    跨组平均没有意义。

    **对照组**取自 ``relations`` 表的既有断言（模型没在训练里见过的关系组合），
    单列出来表示「模型对未见真边的分数」，是泛化健康度的参照，不参与分离度。

    分离度口径：``correct`` 的分数 > 错误条（``wrong`` ∪ ``wrong_relation``）的
    分数，占所有配对数之比。0.5 = 分不开，1.0 = 完全分开。这个数比平均分更能
    反映「模型能否把正确与错误排开」，因为它对组内分数尺度不敏感。
    """
    triples = [
        (str(g["source"]), str(g["relation"]), str(g["target"])) for g in gold
    ]
    scored = score_triples(model, graph, conn, triples)
    rows: List[Dict[str, object]] = []
    for item, sc in zip(gold, scored):
        rows.append({
            **item,
            "prob": float(sc.get("prob", float("nan"))),
            "logit": sc.get("logit"),
            "control": bool(item.get("control", False)),
        })

    def _agg(subset: Sequence[Dict[str, object]]) -> Dict[str, object]:
        by_verdict: Dict[str, List[float]] = collections.defaultdict(list)
        for row in subset:
            prob = float(row["prob"])  # type: ignore[arg-type]
            if not math.isnan(prob):
                by_verdict[str(row["verdict"])].append(prob)
        good = by_verdict.get("correct", [])
        bad = by_verdict.get("wrong", []) + by_verdict.get("wrong_relation", [])
        pairs = [(c, w) for c in good for w in bad]
        wins = sum(1 for c, w in pairs if c > w)
        ties = sum(1 for c, w in pairs if c == w)
        return {
            "n": len(subset),
            "by_verdict": {
                k: {"n": len(v), "mean_prob": sum(v) / len(v) if v else 0.0}
                for k, v in sorted(by_verdict.items())
            },
            "separation": {
                "n_correct": len(good),
                "n_bad": len(bad),
                "auc_like": (wins + 0.5 * ties) / len(pairs) if pairs else float("nan"),
                "mean_correct": sum(good) / len(good) if good else float("nan"),
                "mean_bad": sum(bad) / len(bad) if bad else float("nan"),
            },
        }

    writeback = _agg([r for r in rows if not r["control"]])
    controls = _agg([r for r in rows if r["control"]])
    return {"rows": rows, "writeback": writeback, "controls": controls}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用外部预训练词向量初始化 Logos GAT 的节点特征。"
                    "数据切分 / 负样本 / 指标全部复用 train_closure_relaware。"
    )
    # --- 本轮新增 --- #
    g = parser.add_argument_group("词向量初始化")
    g.add_argument("--init", choices=("random", "mini", "charngram", "file"),
                   default="random",
                   help="random=同口径基线；mini=仓库自带手写属性表；"
                        "charngram=字形哈希（零外部数据对照）；file=真实向量文件")
    g.add_argument("--vectors", type=pathlib.Path, default=None,
                   help="--init file 的向量文件（.txt/.gz/.bz2），流式过滤")
    g.add_argument("--embedding-dim", type=int, default=None,
                   help="向量维度；缺省从文件/内置源推断")
    g.add_argument("--oov-mode", choices=("zero", "charngram"), default="zero",
                   help="OOV 回退：零向量（默认）或字形哈希")
    g.add_argument("--arch", choices=("table", "dense"), default="table",
                   help="table=保持 LogosGAT（写回兼容）；dense=Linear(D,hidden)+ELU")
    g.add_argument("--freeze-embed", action="store_true",
                   help="整个训练期冻结节点初值（table: node_embedding.weight）")
    g.add_argument("--freeze-epochs", type=int, default=0,
                   help="前 N 个 epoch 冻结嵌入派生参数，之后解冻（0=不冻）")
    g.add_argument("--probe-triples", default=None,
                   help='打分探针："猫/n|is_a|人/n,鱼/n|is_a|无脊椎动物/n"')
    g.add_argument("--gold", type=pathlib.Path, default=None,
                   help="金标 json（residual_gold.json），逐条打分并报告分离度")
    g.add_argument("--compare-mode", action="store_true",
                   help="与基线 checkpoint 并排比较（test MRR / hard MRR / 探针分数）")
    g.add_argument("--result-json", type=pathlib.Path, default=None,
                   help="把本次结果写到 json，便于汇总消融表")

    # --- 与基线一致的训练参数（默认值必须逐项相同）--- #
    t = parser.add_argument_group("训练（与 train_closure_relaware 同口径）")
    t.add_argument("--db", type=pathlib.Path, default=_ROOT / "logos.db")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    t.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    t.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    t.add_argument("--neg-per-pos", type=int, default=6)
    t.add_argument("--hard-ratio", type=float, default=0.5)
    t.add_argument("--medium-ratio", type=float, default=0.3)
    t.add_argument("--freeze-gat-epochs", type=int, default=10)
    t.add_argument("--patience", type=int, default=20)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--relation-subset", nargs="*", default=None, metavar="REL")
    t.add_argument("--cross-pos-ratio", type=float, default=0.0,
                   help="cross-POS 负样本占比（默认 0，保证与基线可比）")
    t.add_argument("--pos-min-share", type=float, default=0.05,
                   help="cross-POS 抽样用的目标词性门槛（同 train_calibrated）")
    t.add_argument("--compare-checkpoint", type=pathlib.Path,
                   default=_ROOT / "logos_transformer/checkpoints_hardneg/closure_gat_hardneg_best.pt")
    args = parser.parse_args(argv)

    # 校验（只留会真正改变语义的几条）
    if args.hidden < 1 or args.heads < 1 or args.hidden % args.heads != 0:
        parser.error("--hidden 必须能被 --heads 整除")
    if args.layers < 1:
        parser.error("--layers 必须 >= 1")
    if args.neg_per_pos < 1:
        parser.error("--neg-per-pos 必须 >= 1")
    if not (0.0 <= args.hard_ratio <= 1.0) or not (0.0 <= args.medium_ratio <= 1.0):
        parser.error("--hard-ratio / --medium-ratio 必须在 [0,1]")
    if args.hard_ratio + args.medium_ratio > 1.0 + 1e-9:
        parser.error("--hard-ratio + --medium-ratio 不能超过 1")
    if args.freeze_epochs < 0 or args.freeze_gat_epochs < 0:
        parser.error("--freeze-epochs / --freeze-gat-epochs 必须 >= 0")
    if not (0.0 <= args.cross_pos_ratio <= 1.0):
        parser.error("--cross-pos-ratio 必须在 [0,1]")
    if args.init == "file" and args.vectors is None:
        parser.error("--init file 需要 --vectors")
    if args.init == "random" and args.arch == "dense":
        parser.error("--init random 与 --arch dense 一起没有意义（稠密架构必须给特征）")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    args = parse_args(argv)
    t_start = time.time()
    db_path = pathlib.Path(args.db).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    compare_path = pathlib.Path(args.compare_checkpoint).expanduser().resolve()
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    graph = load_graph(db_path, include_closure=False, verbose=False)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    try:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='closure'"
        ).fetchone():
            print("No closure data found, run apply_seeds.py first", file=sys.stderr)
            return 1
        if conn.execute("SELECT COUNT(*) FROM closure WHERE depth >= 2").fetchone()[0] == 0:
            print("No closure data found, run apply_seeds.py first", file=sys.stderr)
            return 1
        positives, facts = load_positives(conn, graph)
        target_pos = iw.load_relation_target_pos(conn)
    finally:
        conn.close()

    subset: Optional[List[str]] = None
    if args.relation_subset is not None:
        wanted = [str(r) for r in args.relation_subset]
        unknown = [r for r in wanted if r not in set(graph.relation_names)]
        if unknown:
            print(f"[error] --relation-subset 含未知关系: {', '.join(unknown)}", file=sys.stderr)
            return 1
        subset = list(dict.fromkeys(wanted))
        positives = [p for p in positives if p.relation in set(subset)]
        if not positives:
            print("[error] 关系子集在 closure 里没有 depth>=2 正样本", file=sys.stderr)
            return 1

    # ---- 节点特征（在任何 RNG 消耗之前算完）---- #
    features, feat_info = load_node_features(
        db_path, graph,
        init=args.init, vectors=args.vectors,
        embedding_dim=args.embedding_dim, oov_mode=args.oov_mode, seed=args.seed,
    )

    # ---- 切分与负样本（与基线同一调用序列）---- #
    positives_by_split, sources_by_split, mode = split_by_source(positives, seed=args.seed)
    pools = build_negative_pools(positives, facts)

    allowed = iw.build_pos_guard(target_pos, args.pos_min_share)
    active_relations = sorted({p.relation for p in positives})
    _allowed_nodes, wrong_nodes = wrong_pos_index(graph, allowed, active_relations)

    rng = random.Random(args.seed + 7)
    data: Dict[str, Dict[str, object]] = {}
    for name in SPLIT_NAMES:
        pos_list = positives_by_split[name]
        if args.cross_pos_ratio > 0:
            ratios = {"hard": args.hard_ratio, "medium": args.medium_ratio,
                      "easy": 1.0 - args.hard_ratio - args.medium_ratio,
                      "cross": args.cross_pos_ratio}
            quota = allocate_tiers(args.neg_per_pos, ratios)
            neg_list, _st = sample_negatives_crosspos(
                pos_list, graph, facts, pools, quota, wrong_nodes, rng
            )
        else:
            n_hard, n_medium, n_easy = _split_ratios(
                args.neg_per_pos, args.hard_ratio, args.medium_ratio
            )
            neg_list, _st = sample_hardneg(
                pos_list, graph, facts, pools, n_hard, n_medium, n_easy, rng
            )
        data[name] = {"pos": pos_list, "neg": neg_list,
                      "samples": list(pos_list) + neg_list}

    train_samples: List[Sample] = data["train"]["samples"]        # type: ignore[assignment]
    val_samples: List[Sample] = data["val"]["samples"]            # type: ignore[assignment]
    train_pos = sum(1 for s in train_samples if s.label == 1)
    pos_weight = (len(train_samples) - train_pos) / max(train_pos, 1)
    relation_counts = dict(Counter_(s.relation for s in train_samples if s.label == 1))

    # ---- 建模型（RNG 起点与基线一致）---- #
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, model_info = build_predictor(
        graph,
        arch=args.arch,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        dropout=DEFAULT_DROPOUT,
        seed=args.seed,
        features=features,
        relation_counts=relation_counts,
        head_relations=subset,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 冻结分两层，每 epoch 开头统一重算（而不是靠事件，否则 --freeze-epochs
    # 会被 --freeze-gat-epochs 完全遮蔽：node_embedding 本来就在 model.gat 里，
    # GAT 解冻的那一刻会顺带把嵌入也解冻，两个开关就变成同一个开关）。
    #   GAT 整体：epoch <= --freeze-gat-epochs 时冻（基线语义）
    #   嵌入      ：--freeze-embed 永久冻；--freeze-epochs N 只冻前 N 个
    embed_params = _embed_params(model)

    # ---- 抬头 ---- #
    print("=" * 100)
    print(f"Logos Transformer — pretrained node features ({VERSION})")
    print("=" * 100)
    print(f"db={db_path}")
    print(f"graph: {graph.stats['nodes']} nodes / {graph.stats['semantic_edges']} direct "
          f"edges / {graph.stats['message_edges']} message edges")
    print(f"split mode: {mode}")
    for name in SPLIT_NAMES:
        pos_list: List[Sample] = data[name]["pos"]                # type: ignore[assignment]
        samples: List[Sample] = data[name]["samples"]              # type: ignore[assignment]
        print(f"  {name:<5} sources={len(sources_by_split[name]):>3}  "
              f"edges={len(pos_list):>3}  depth="
              f"{dict(sorted(Counter_(s.depth for s in pos_list).items()))}  "
              f"samples={len(samples):>3} ({len(pos_list)} pos / "
              f"{len(samples) - len(pos_list)} neg)")
    print("-" * 100)
    print(f"init={args.init} | arch={args.arch} | "
          f"vector coverage={feat_info.get('coverage', 0.0):.1%} "
          f"({feat_info.get('covered', 0)}/{feat_info.get('total', 0)}) | "
          f"freeze_embed={args.freeze_embed} freeze_epochs={args.freeze_epochs} "
          f"freeze_gat_epochs={args.freeze_gat_epochs}")
    if model_info.get("init_stats"):
        st = model_info["init_stats"]
        print(f"node feature rows : 非零 {st['nonzero_rows']}/{st['num_nodes']} | "
              f"行范数 mean={st['row_norm_mean']:.4f} "
              f"min={st['row_norm_min']:.4f} max={st['row_norm_max']:.4f} "
              f"(随机初始化参照 0.1*sqrt({args.hidden})={0.1 * math.sqrt(args.hidden):.4f})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train pos_weight={pos_weight:.2f} | {args.layers}x{args.heads} heads, "
          f"hidden={args.hidden}, params={n_params:,}")
    print("-" * 100)

    # ---- init 诊断（必须保留：会消耗 dropout 随机数，删了就复现不了基线）---- #
    cat_idx = graph.id_to_index.get(2, 0)
    init_h, _init_scores, _ = encode_and_score(model, graph, active_relations)
    init_loss, init_acc = loss_acc(model, init_h, train_samples, criterion)
    print(f"init: train loss={init_loss:.4f} acc={init_acc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "closure_gat_relaware_best.pt"
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epochs_run = 0
    batch_rng = random.Random(args.seed + 1)

    prev_gat_frozen: Optional[bool] = None
    prev_embed_frozen: Optional[bool] = None
    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        gat_frozen = epoch <= args.freeze_gat_epochs
        for param in model.gat.parameters():
            param.requires_grad = not gat_frozen
        # 嵌入的独立窗口只在显式给了 --freeze-embed / --freeze-epochs 时生效；
        # 否则跟随 GAT（= 基线的 freeze_gat_epochs 语义）。必须写在 GAT 那次
        # 赋值之后，否则 GAT 的冻结会把它盖回去。
        if args.freeze_embed:
            for p in embed_params:
                p.requires_grad = False
        elif args.freeze_epochs > 0:
            for p in embed_params:
                p.requires_grad = not (epoch <= args.freeze_epochs)
        embed_frozen = not embed_params[0].requires_grad if embed_params else False
        if prev_gat_frozen is True and not gat_frozen:
            print(f"[unfreeze] GAT 解冻于 epoch {epoch}")
        if prev_embed_frozen is True and not embed_frozen:
            print(f"[unfreeze] 节点初值解冻于 epoch {epoch}")
        prev_gat_frozen, prev_embed_frozen = gat_frozen, embed_frozen

        model.train()
        if gat_frozen:
            model.gat.eval()

        order = list(train_samples)
        batch_rng.shuffle(order)
        total_loss = 0.0
        total = 0
        correct = 0
        for i in range(0, len(order), args.batch_size):
            batch = order[i:i + args.batch_size]
            optimizer.zero_grad()
            emb, _ = model.encode(graph.edge_index, graph.edge_type)
            logits = model.score_pairs(emb, pairs_tensor(batch), [s.relation for s in batch])
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
            val_scores = {rel: model.score_all(val_h, rel) for rel in active_relations}
        if math.isnan(val_loss):
            val_loss = train_loss
        scheduler.step(val_loss)
        val_hard = hard_metrics(val_scores, pools, data["val"]["pos"])  # type: ignore[arg-type]

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1

        hard_txt = (
            f"hard-val MRR={_fmt(val_hard.model_mrr)} prior={_fmt(val_hard.prior_mrr)} "
            f"margin={val_hard.margin:+.4f} (n={val_hard.n_queries})"
            if val_hard.n_queries else "hard-val MRR=n/a"
        )
        print(f"epoch {epoch:>3}/{args.epochs} | train loss={train_loss:.4f} "
              f"acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"{hard_txt} | lr={optimizer.param_groups[0]['lr']:.2e}")
        if stale >= args.patience:
            print(f"early stopping at epoch {epoch} (val loss stale for {stale} epoch(s))")
            break

    # ---- 收尾：加载最优、算全部指标 ---- #
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warning] state_dict 偏差 missing={list(missing)[:4]} "
                  f"unexpected={list(unexpected)[:4]}", file=sys.stderr)
    model.eval()

    h, scores_by_rel, _ = encode_and_score(model, graph, active_relations)
    metrics: Dict[str, Dict[str, object]] = {}
    for name in SPLIT_NAMES:
        metrics[name] = eval_bucket_relaware(
            model, graph, facts, data[name]["samples"], criterion,  # type: ignore[arg-type]
            h, scores_by_rel,
        )
    prior = {
        name: target_prior_relaware(scores_by_rel, graph, facts, data[name]["pos"])  # type: ignore[arg-type]
        for name in SPLIT_NAMES
    }
    hard_by_split = {
        name: hard_metrics(scores_by_rel, pools, data[name]["pos"])  # type: ignore[arg-type]
        for name in SPLIT_NAMES
    }

    # ---- OOD 探针（cross-POS 负样本的拒绝率）---- #
    ood_reject = float("nan")
    ood_n = 0
    if wrong_nodes:
        probe_rng = random.Random(args.seed + 101)
        probes = build_ood_probe(
            data["test"]["pos"], graph, facts, wrong_nodes,  # type: ignore[arg-type]
            per_source=4, rng=probe_rng,
        )
        if probes:
            with torch.no_grad():
                logits = pair_logits_safe(model, h, probes)
                probs = torch.sigmoid(logits).tolist()
            ood_reject = rejection_rate(probs, threshold=0.5)
            ood_n = len(probes)

    # ---- 打分探针 ---- #
    probes_out: List[Dict[str, object]] = []
    if args.probe_triples:
        triples = []
        for chunk in args.probe_triples.split(","):
            parts = [p.strip() for p in chunk.split("|")]
            if len(parts) != 3:
                print(f"[warning] 探针格式不对，跳过：{chunk}", file=sys.stderr)
                continue
            triples.append((parts[0], parts[1], parts[2]))
        if triples:
            wconn = sqlite3.connect(str(db_path))
            wconn.execute("PRAGMA query_only = ON")
            try:
                probes_out = score_triples(model, graph, wconn, triples)
            finally:
                wconn.close()

    # ---- 金标 ---- #
    gold_out: Optional[Dict[str, object]] = None
    if args.gold is not None:
        gold_path = pathlib.Path(args.gold).expanduser().resolve()
        if gold_path.exists():
            gconn = sqlite3.connect(str(db_path))
            gconn.execute("PRAGMA query_only = ON")
            try:
                gold_out = gold_report(model, graph, gconn, load_gold(gold_path))
            finally:
                gconn.close()

    # ---- 与基线 checkpoint 并排 ---- #
    baseline: Optional[Dict[str, object]] = None
    if args.compare_mode and compare_path.exists():
        try:
            bstate = torch.load(compare_path, map_location="cpu", weights_only=False)
            binfo = iw.inspect_checkpoint(bstate)
            bmodel, berr = build_predictor(
                graph,
                arch="table",
                hidden=int(binfo["hidden"]),
                heads=int(binfo["num_heads"]),
                layers=int(binfo["num_layers"]),
                dropout=DEFAULT_DROPOUT,
                seed=args.seed,
                features=None,
                relation_counts=dict(zip(binfo["head_names"], binfo["counts"])),
                head_relations=subset,
            )
            missing, unexpected = bmodel.load_state_dict(bstate, strict=False)
            bmodel.eval()
            bh, bscores, _ = encode_and_score(bmodel, graph, active_relations)
            baseline = {
                "path": str(compare_path),
                "test_mrr": float(
                    eval_bucket_relaware(bmodel, graph, facts, data["test"]["samples"],
                                         criterion, bh, bscores)["mrr"]  # type: ignore[arg-type]
                ),
                "hard_test_mrr": float(
                    hard_metrics(bscores, pools, data["test"]["pos"]).model_mrr  # type: ignore[arg-type]
                ),
            }
            if args.probe_triples:
                wconn = sqlite3.connect(str(db_path))
                wconn.execute("PRAGMA query_only = ON")
                try:
                    baseline["probes"] = score_triples(
                        bmodel, graph, wconn,
                        [(p["src"], p["relation"], p["tgt"]) for p in probes_out],
                    )
                finally:
                    wconn.close()
        except Exception as exc:  # noqa: BLE001 - 基线只是参照，失败不该中断主流程
            print(f"[warning] 基线 checkpoint 对比失败：{exc}", file=sys.stderr)

    # ---- dense 折叠存档 ---- #
    folded_path: Optional[pathlib.Path] = None
    if args.arch == "dense":
        folded_path = output_dir / "closure_gat_relaware_best_folded.pt"
        torch.save(
            fold_dense_checkpoint(
                model, graph,
                hidden=args.hidden, heads=args.heads,
                layers=args.layers, dropout=DEFAULT_DROPOUT,
            ),
            folded_path,
        )

    # ---- 报告 ---- #
    print()
    print("=" * 100)
    print("结果")
    print("=" * 100)
    for name in SPLIT_NAMES:
        m = metrics[name]
        print(f"{name:<5} loss={m['loss']:.4f} acc={m['acc']:.4f} auc={m['auc']:.4f} "
              f"MRR={m['mrr']:.4f} hits@1={m['hits@1']:.3f} hits@10={m['hits@10']:.3f} "
              f"queries={m['queries']} | prior MRR={prior[name]['overall']:.4f}")
    hm = hard_by_split["test"]
    print(f"\nhard-test MRR={hm.model_mrr:.4f} prior={hm.prior_mrr:.4f} "
          f"margin={hm.margin:+.4f} (n={hm.n_queries}, mean K={hm.mean_k:.1f})")
    if not math.isnan(ood_reject):
        print(f"OOD 拒绝率（cross-POS 测试探针）: {ood_reject:.4f} (n={ood_n})")

    if probes_out:
        print("\n打分探针：")
        for row in probes_out:
            if "error" in row:
                print(f"  {row['src']} {row['relation']} {row['tgt']}: {row['error']}")
                continue
            base_txt = ""
            if baseline and "probes" in baseline:
                for b in baseline["probes"]:  # type: ignore[union-attr]
                    if (b.get("src"), b.get("relation"), b.get("tgt")) == (
                        row["src"], row["relation"], row["tgt"]
                    ):
                        base_txt = f" (基线 {b.get('prob', float('nan')):.4f})"
                        break
            print(f"  {row['src']:<14} {row['relation']:<14} {row['tgt']:<14} "
                  f"logit={row['logit']:+.4f} prob={row['prob']:.4f}{base_txt}")

    if gold_out:
        wb = gold_out["writeback"]                                  # type: ignore[index]
        sep = wb["separation"]                                     # type: ignore[index]
        print(f"\n金标 · 写回组 n={wb['n']} | correct={sep['n_correct']} "
              f"bad={sep['n_bad']} | 分离度={sep['auc_like']:.4f} "
              f"(correct 均分 {sep['mean_correct']:.4f} vs bad {sep['mean_bad']:.4f})")
        for verdict, agg in sorted(wb["by_verdict"].items()):      # type: ignore[union-attr]
            print(f"  {verdict:<16} n={agg['n']:>3} mean_prob={agg['mean_prob']:.4f}")
        ctl = gold_out["controls"]                                 # type: ignore[index]
        ctl_mean = (ctl["by_verdict"].get("correct") or {}).get("mean_prob", float("nan"))
        print(f"金标 · 对照组 n={ctl['n']}（模型未见过的库内真边）"
              f" mean_prob={ctl_mean:.4f}")

    if baseline:
        print(f"\n与基线并排（{pathlib.Path(baseline['path']).name}）：")
        print(f"  {'指标':<16} {'本次':>10} {'基线':>10} {'差':>10}")
        for label, key in (("test MRR", "test_mrr"), ("hard-test MRR", "hard_test_mrr")):
            cur = float(metrics["test"]["mrr"]) if key == "test_mrr" else float(hm.model_mrr)
            base_v = float(baseline[key])  # type: ignore[arg-type]
            print(f"  {label:<16} {cur:>10.4f} {base_v:>10.4f} {cur - base_v:>+10.4f}")

    print(f"\ncheckpoint : {ckpt}")
    if folded_path:
        print(f"折叠存档   : {folded_path}（写回管线用这个）")
    print(f"耗时       : {time.time() - t_start:.1f}s | epochs={epochs_run} "
          f"best_epoch={best_epoch}")

    # ---- 落盘 json（供消融表汇总）---- #
    result_path = pathlib.Path(args.result_json).expanduser().resolve() if args.result_json \
        else output_dir / "result.json"
    payload = {
        "version": VERSION,
        "init": args.init,
        "arch": args.arch,
        "seed": args.seed,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "features": feat_info,
        "params": n_params,
        "metrics": {k: {kk: vv for kk, vv in v.items()
                        if isinstance(vv, (int, float, str))} for k, v in metrics.items()},
        "prior": prior,
        "hard_test": {
            "model_mrr": hm.model_mrr, "prior_mrr": hm.prior_mrr,
            "margin": hm.margin, "n_queries": hm.n_queries,
        },
        "ood_rejection": ood_reject,
        "probes": probes_out,
        "baseline": baseline,
        "gold": (
            {"writeback": gold_out["writeback"], "controls": gold_out["controls"],
             "rows": gold_out["rows"]}
            if gold_out else None
        ),
        "checkpoint": str(ckpt),
        "folded": str(folded_path) if folded_path else None,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(_json_clean(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果 json  : {result_path}")
    return 0


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _json_clean(value):
    """nan / inf → None，让 json 保持标准（Python 默认会写 NaN）。"""
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    return value


def Counter_(iterable) -> collections.Counter:
    return collections.Counter(iterable)


def _embed_params(model: RelationAwareClosurePredictor) -> List[nn.Parameter]:
    """「嵌入派生参数」：table 架构的查表权重 / dense 架构的输入投影。"""
    gat = model.gat
    if hasattr(gat, "input_proj"):
        return [gat.input_proj.weight, gat.input_proj.bias]
    if hasattr(gat, "node_embedding"):
        return [gat.node_embedding.weight]
    return []


def pair_logits_safe(
    model: RelationAwareClosurePredictor,
    h: torch.Tensor,
    samples: Sequence[Sample],
) -> torch.Tensor:
    pairs = pairs_tensor(samples)
    return model.score_pairs(h, pairs, [s.relation for s in samples])


if __name__ == "__main__":
    raise SystemExit(main())
