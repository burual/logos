#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""外部预训练词向量 → Logos GAT 的节点初始特征。

为什么需要这个模块
------------------
335 个义项 / 689 条直接边 ⇒ 平均每节点 2 条边。校准（cross-POS 负样本 + 温度）
把 OOD 拒绝率推到 1.0000，符号守卫再把写入量从 154 压到 118，但残留错误全是
**图里没有的常识**：

    fish is_a 无脊椎动物        （缺「有脊柱」这条互斥知识）
    蛇 has_property 温血        （缺动物学常识）
    猫 is_a 人                  （缺「人不是猫的上位」这类层级常识）

1–2 条边给不了这种语义。预训练词向量给每个义项一个稠密初始表示，把「稀疏」
从「学不到」变成「起点已经带了一部分知识」。

三种可用向量源
--------------
* ``file``      真实预训练文件（腾讯 AI Lab / fastText sgns.merge.word 等）。
                支持 ``.txt`` / ``.gz`` / ``.bz2``，**流式按需过滤**：给一个 keep
                集合就只保留需要的词，6GB 文件不需要全量进内存。
* ``mini``      内置 ``mini_vectors.txt``：手工编码的动物学属性矩阵（见 §mini）。
                仓库自带，供 CI / 无网环境跑通管线。⚠ **它是 oracle 性质的上界探针**，
                不是「预训练向量有效」的证据。
* ``charngram`` 字形三元组哈希特征：完全不依赖外部数据，用来回答「任何稠密初始化
                都有用吗」这个对照问题。

OOV 回退链
----------
``word_pos`` → ``word`` → ``compound``（复合词拆成子词取平均）→ ``charngram`` / ``zero``。
真实向量文件下 OOV 默认给零向量（模型只能靠图结构），也可以在图层面用
``fill_oov_by_neighbours`` 用邻居均值补。

用法
----
    python logos_transformer/embeddings.py --report                 # 覆盖率报告
    python logos_transformer/embeddings.py --build-mini            # 重建 mini_vectors.txt
    python logos_transformer/embeddings.py \\
        --vectors /path/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt --report
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import io
import math
import pathlib
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # torch 只用于最后张量化；纯统计可以不要
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MINI_PATH = pathlib.Path(__file__).resolve().parent / "mini_vectors.txt"
MINI_DIM = 56


# --------------------------------------------------------------------------- #
# 向量文件读取
# --------------------------------------------------------------------------- #
def _open_text(path: pathlib.Path) -> io.TextIOBase:
    """按扩展名打开（顺带处理腾讯/word2vec 常见的 .gz / .bz2）。"""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_vector_line(line: str) -> Optional[Tuple[str, List[float]]]:
    """``词 v1 v2 ...`` → (词, 向量)；不是向量行则返回 None。"""
    parts = line.split()
    if len(parts) < 3:                      # 至少 词 + 2 维；表头 "400000 200" 只有 2 段
        return None
    word = parts[0]
    if not word:
        return None
    try:
        vec = [float(x) for x in parts[1:]]
    except ValueError:
        return None                          # 表头或坏行
    return word, vec


class ChineseWordVectorLoader:
    """中文词向量加载器（词 → 向量），支持大文件流式过滤。

    Args:
        vector_path: 向量文件路径；``None`` 表示空加载器（只用内置特征）。
        dim: 期望维度；``None`` 从文件首行推断。给错维度会**报错**而不是静默截断。
        keep: 只保留这些词（含 ``word_pos`` 形式）。6GB 文件的正确用法：
            先用 :func:`collect_sense_words` 拿到需要的词，再流式过滤一遍。
        max_vocab: 最多收多少个词（0 = 不设限）。调试时用来把大文件当小文件跑。
    """

    def __init__(
        self,
        vector_path: Optional[pathlib.Path] = None,
        dim: Optional[int] = None,
        keep: Optional[Set[str]] = None,
        max_vocab: int = 0,
        verbose: bool = False,
    ) -> None:
        self.dim = int(dim) if dim else 0
        self.vectors: Dict[str, "torch.Tensor"] = {}
        self.source_path = pathlib.Path(vector_path) if vector_path else None
        self.keep = keep
        self.max_vocab = int(max_vocab)
        self.verbose = verbose
        self.stats: Dict[str, object] = {
            "lines_scanned": 0, "lines_kept": 0, "malformed": 0, "dim_mismatch": 0,
        }
        self._charngram_cache: Dict[str, "torch.Tensor"] = {}
        if self.source_path is not None:
            self.load(self.source_path)

    # -- 载入 -------------------------------------------------------------- #
    def load(self, path: pathlib.Path) -> None:
        """流式读入；有 ``keep`` 时只解析需要的行（内存 O(|keep|·dim)）。"""
        if torch is None:
            raise SystemExit("需要 torch 来存放向量：pip install torch")
        path = pathlib.Path(path)
        if not path.exists():
            raise SystemExit(f"[错误] 向量文件不存在：{path}")

        keep = self.keep
        n_kept = 0
        with _open_text(path) as fh:
            for lineno, line in enumerate(fh, 1):
                if not line or line[0] in "#\n":
                    continue
                # word2vec 文本格式首行是 "词数 维度"，_parse_vector_line 会返回 None。
                parsed = _parse_vector_line(line)
                self.stats["lines_scanned"] = lineno
                if parsed is None:
                    if lineno == 1:
                        continue                    # 表头，正常跳过
                    self.stats["malformed"] = int(self.stats["malformed"]) + 1
                    continue
                word, vec = parsed
                if keep is not None and word not in keep:
                    continue
                if self.dim and len(vec) != self.dim:
                    self.stats["dim_mismatch"] = int(self.stats["dim_mismatch"]) + 1
                    continue
                if not self.dim:
                    self.dim = len(vec)
                self.vectors[word] = torch.tensor(vec, dtype=torch.float32)
                n_kept += 1
                if self.max_vocab and n_kept >= self.max_vocab:
                    break
        self.stats["lines_kept"] = len(self.vectors)
        if not self.dim:
            raise SystemExit(f"[错误] 没能从 {path} 解析出任何向量行（格式不对？）")
        if self.verbose:
            print(f"  [vectors] {path.name}: 扫描 {self.stats['lines_scanned']} 行 / "
                  f"保留 {self.stats['lines_kept']} 词 / dim={self.dim}")

    # -- 取向量 ------------------------------------------------------------ #
    def _lookup(self, key: str) -> Optional["torch.Tensor"]:
        return self.vectors.get(key)

    def get_vector(
        self,
        word: str,
        sense_pos: Optional[str] = None,
        *,
        oov_mode: str = "zero",
        report_source: bool = False,
    ):
        """按四级回退取向量。

        回退顺序（命中即返回）：``词_pos`` → ``词`` → 复合词子词平均 → ``charngram``/``zero``。

        Args:
            word: 词形，例如 ``猫``。
            sense_pos: 词性（``n``/``v``/``a``…），用于第一级 ``猫_n`` 精确匹配。
            oov_mode: ``zero``（零向量）或 ``charngram``（字形哈希，永不 OOV）。
            report_source: 为 True 时返回 ``(向量, 命中级别)``。

        Returns:
            ``[dim]`` 张量；``report_source=True`` 时返回 ``(张量, 级别字符串)``。
        """
        if torch is None:
            raise SystemExit("需要 torch：pip install torch")
        hit: Optional["torch.Tensor"] = None
        level = "oov"

        if sense_pos:
            hit = self._lookup(f"{word}_{sense_pos}")
            if hit is not None:
                level = "word_pos"
        if hit is None:
            hit = self._lookup(word)
            if hit is not None:
                level = "word"
        if hit is None:
            hit = self._compound(word)
            if hit is not None:
                level = "compound"
        if hit is None:
            if oov_mode == "charngram":
                hit, level = self.charngram(word, self.dim), "charngram"
            else:
                hit, level = torch.zeros(self.dim), "zero"

        if level in ("charngram",) and self.dim and hit.numel() != self.dim:
            # charngram 与外部向量维度不同时，投影到同维（确定性，可复现）
            hit = self._resize(hit, self.dim)
        return (hit, level) if report_source else hit

    def _compound(self, word: str) -> Optional["torch.Tensor"]:
        """复合词回退：``鲸类头部`` → ``鲸类`` + ``头部`` 的平均。

        只做二分为两段（前缀 + 后缀），从中间向外找第一个两段都命中的切分。
        三字以上仍不命中就退到「最长的已知前缀」——避免把 ``脑电图信号`` 拆成
        ``脑`` + ``电图信号`` 这种无意义的切分。
        """
        n = len(word)
        if n < 3:
            return None
        for cut in range(1, n):
            head, tail = word[:cut], word[cut:]
            vh, vt = self._lookup(head), self._lookup(tail)
            if vh is not None and vt is not None:
                return (vh + vt) / 2.0
        # 单段回退：优先最长前缀，其次最长后缀
        for cut in range(n - 1, 0, -1):
            hit = self._lookup(word[:cut]) or self._lookup(word[cut:])
            if hit is not None:
                return hit
        return None

    # -- 内置特征 ---------------------------------------------------------- #
    def charngram(self, word: str, dim: int) -> "torch.Tensor":
        """字形三元组特征哈希（不依赖任何外部数据，永不 OOV）。

        用字符 bigram/trigram 的 md5 做确定性哈希投影，再 L2 归一化。它只编码
        「长得像」——``鲤鱼``/``鲫鱼`` 会靠近，但 ``鱼`` 与 ``有脊柱`` 不会。
        因此它适合作为**无外部知识的对照**：如果它能提升指标，说明收益来自
        「不再共享一个随机向量」，而不是来自语义知识。
        """
        if torch is None:
            raise SystemExit("需要 torch：pip install torch")
        key = f"{word}|{dim}"
        cached = self._charngram_cache.get(key)
        if cached is not None:
            return cached
        vec = torch.zeros(dim)
        padded = f"<{word}>"
        grams = [padded[i:i + n] for n in (2, 3) for i in range(len(padded) - n + 1)]
        if not grams:
            grams = [padded]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(vec.norm())
        if norm > 0:
            vec = vec / norm
        self._charngram_cache[key] = vec
        return vec

    @staticmethod
    def _resize(vec: "torch.Tensor", dim: int) -> "torch.Tensor":
        """把任意维度确定性投影到 ``dim``（补零 / 截断）。"""
        out = vec.new_zeros(dim)
        n = min(dim, vec.numel())
        out[:n] = vec[:n]
        return out

    # -- 覆盖率 ------------------------------------------------------------ #
    def coverage_report(self, sense_words: Sequence[Tuple[str, str]], oov_mode: str = "zero"):
        """统计覆盖率。``sense_words`` 是 ``[(word, pos), ...]``。

        Returns:
            ``CoverageReport``（可打印、可直接进 JSON）。
        """
        by_level: Dict[str, int] = {}
        missing: List[Tuple[str, str]] = []
        for word, pos in sense_words:
            _vec, level = self.get_vector(word, pos, oov_mode=oov_mode, report_source=True)
            by_level[level] = by_level.get(level, 0) + 1
            if level in ("zero", "charngram"):
                missing.append((word, pos))
        total = len(sense_words)
        covered = total - by_level.get("zero", 0)
        return CoverageReport(
            total=total,
            covered=covered,
            by_level=by_level,
            missing=missing,
            dim=self.dim,
            source=str(self.source_path) if self.source_path else "builtin",
        )


@dataclass
class CoverageReport:
    total: int
    covered: int
    by_level: Dict[str, int]
    missing: List[Tuple[str, str]] = field(default_factory=list)
    dim: int = 0
    source: str = ""

    @property
    def ratio(self) -> float:
        return self.covered / self.total if self.total else 0.0

    def lines(self, limit: int = 12) -> List[str]:
        out = [
            f"向量源        : {self.source} (dim={self.dim})",
            f"义项覆盖      : {self.covered}/{self.total} = {self.ratio:.1%}",
        ]
        order = ["word_pos", "word", "compound", "charngram", "zero"]
        pretty = {"word_pos": "词_pos 精确", "word": "裸词", "compound": "复合词拆分",
                  "charngram": "字形哈希", "zero": "零向量(OOV)"}
        for level in order:
            if level in self.by_level:
                out.append(f"  {pretty[level]:<12}: {self.by_level[level]:>4}")
        if self.missing:
            sample = ", ".join(f"{w}/{p}" for w, p in self.missing[:limit])
            more = " ..." if len(self.missing) > limit else ""
            out.append(f"  OOV 样例    : {sample}{more}")
        return out


# --------------------------------------------------------------------------- #
# 从数据库取词表 / 构建节点特征
# --------------------------------------------------------------------------- #
def collect_sense_words(db_path: pathlib.Path) -> List[Tuple[int, str, str]]:
    """``[(sense_id, word, pos), ...]``，按 sense_id 升序（= 图节点顺序）。"""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    try:
        rows = conn.execute(
            """
            SELECT s.sense_id, w.word, s.pos
            FROM senses s JOIN words w ON w.word_id = s.word_id
            ORDER BY s.sense_id
            """
        ).fetchall()
    finally:
        conn.close()
    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


def keep_set_for(sense_words: Sequence[Tuple[int, str, str]]) -> Set[str]:
    """给 ``ChineseWordVectorLoader(keep=...)`` 用的过滤集合。

    同时收 ``词``、``词_pos`` 和复合词的所有子串前缀/后缀，这样真实向量文件里
    ``猫_n`` 与 ``鲸类`` 这类条目都能命中，而 6GB 文件只留几十个词。
    """
    keep: Set[str] = set()
    for _sid, word, pos in sense_words:
        keep.add(word)
        keep.add(f"{word}_{pos}")
        n = len(word)
        for cut in range(1, n):
            keep.add(word[:cut])
            keep.add(word[cut:])
    return keep


def build_node_features(
    senses_db_path: pathlib.Path,
    vector_loader: ChineseWordVectorLoader,
    embedding_dim: int = 200,
    *,
    oov_mode: str = "zero",
    verbose: bool = True,
):
    """每个义项一个向量，堆成 ``[num_nodes, embedding_dim]``。

    Args:
        senses_db_path: ``logos.db`` 路径。
        vector_loader: 已载入的向量源。
        embedding_dim: 输出维度；与 loader 维度不一致时按 ``_resize`` 补零/截断。
        oov_mode: ``zero`` 或 ``charngram``。

    Returns:
        ``(features, stats)``；``features`` 是 ``float32`` 张量（行序 = sense_id 升序），
        ``stats`` 含覆盖率与逐级命中数。
    """
    if torch is None:
        raise SystemExit("需要 torch：pip install torch")
    sense_words = collect_sense_words(senses_db_path)
    # loader 的维度可能来自文件（200），也可能来自 mini（54）
    src_dim = vector_loader.dim or embedding_dim
    mat = torch.zeros(len(sense_words), src_dim, dtype=torch.float32)
    by_level: Dict[str, int] = {}
    oov: List[Tuple[str, str]] = []
    for row, (_sid, word, pos) in enumerate(sense_words):
        vec, level = vector_loader.get_vector(
            word, pos, oov_mode=oov_mode, report_source=True
        )
        mat[row] = vector_loader._resize(vec, src_dim)
        by_level[level] = by_level.get(level, 0) + 1
        if level in ("zero",):
            oov.append((word, pos))

    if src_dim != embedding_dim:
        mat = (
            mat[:, :embedding_dim]
            if src_dim > embedding_dim
            else torch.cat(
                [mat, torch.zeros(len(sense_words), embedding_dim - src_dim)], dim=1
            )
        )

    covered = len(sense_words) - by_level.get("zero", 0)
    stats = {
        "num_nodes": len(sense_words),
        "dim": embedding_dim,
        "source_dim": src_dim,
        "covered": covered,
        "coverage": covered / max(len(sense_words), 1),
        "by_level": by_level,
        "oov": oov,
        "sources": dict(by_level),
    }
    if verbose:
        print(f"  [node features] {len(sense_words)} 节点 × {embedding_dim}d | "
              f"覆盖 {stats['coverage']:.1%} ({covered}/{len(sense_words)}) | "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_level.items())))
    return mat, stats


def fill_oov_by_neighbours(
    features: "torch.Tensor",
    edge_index: "torch.Tensor",
    oov_mask: "torch.Tensor",
    iters: int = 1,
) -> Tuple["torch.Tensor", int]:
    """用一度邻居的均值填补零向量节点（图传播版 OOV 回退）。

    零向量节点在 GAT 里等价于「只有偏置」，会浪费一层传播；用邻居均值做初值
    等于把「它在图里的位置」编码进特征。``iters`` 次迭代 = ``iters`` 跳。

    Returns:
        ``(新特征, 被填补的节点数)``。
    """
    if torch is None:
        raise SystemExit("需要 torch：pip install torch")
    out = features.clone()
    filled_total = 0
    with torch.no_grad():
        for _ in range(max(iters, 0)):
            src, dst = edge_index
            acc = torch.zeros_like(out)
            cnt = torch.zeros(out.size(0))
            acc.index_add_(0, dst, out[src])
            cnt.index_add_(0, dst, torch.ones(src.numel()))
            has = cnt > 0
            zero = out.abs().sum(dim=1) < 1e-12          # 仍是零向量的节点
            target = zero & has
            n = int(target.sum())
            if n == 0:
                break
            out[target] = acc[target] / cnt[target].clamp_min(1e-9).unsqueeze(1)
            filled_total += n
    # 迭代结束仍是零向量的，用全局均值兜底，避免下游除零
    zero = out.abs().sum(dim=1) < 1e-12
    if int(zero.sum()) and out.numel():
        out[zero] = out[~zero].mean(dim=0) if int((~zero).sum()) else out[zero]
    return out, filled_total


# --------------------------------------------------------------------------- #
# 内置 mini 向量：手工编码的动物学属性矩阵
# --------------------------------------------------------------------------- #
# 54 维，分 6 个语义块。取值 0/1（少数 0.5 表示「部分/可变」）。
# ⚠ 这些属性是**从教科书动物学手写的**，不是从图上读的，也不是学出来的：
#    - 图里已经断言的（例如 猫 has_property 有毛）这里也写了，因为 mini 向量的
#      定位是「把该领域常识一次性补齐」，重复不损失信息；
#    - 因此 用 mini 跑出来的任何提升都是**上界**，不能当作「预训练向量有效」的证据。
MINI_DIMS: List[str] = [
    # 0-13 体型与生理属性
    "有毛", "温血", "胎生", "有羽毛", "卵生", "有鳞", "冷血", "无四肢",
    "有脊柱", "无脊柱", "水生", "陆生", "有外骨骼", "六足",
    # 14-23 类群
    "哺乳类", "鸟类", "鱼类", "爬行类", "两栖类", "无脊椎", "昆虫类", "节肢类",
    "软体类", "头足类",
    # 24-31 部位
    "头部部位", "躯干部位", "肢体部位", "尾部部位", "鳍部位", "鳃部位",
    "翅部位", "壳部位",
    # 32-39 行为
    "水生行为", "陆行行为", "飞行行为", "捕食行为", "繁殖行为", "防御行为",
    "社会行为", "感知行为",
    # 40-47 医学
    "解剖", "疾病", "症状", "信号", "干预", "影像", "形态", "大小",
    # 48-53 抽象与人造物
    "抽象", "物理", "信息", "工具", "关系", "属性",
    # 54-55 功能词（让虚词不再共用零向量）
    "虚词", "量词",
]

# 词 → {维名: 值}。只写真值；未列出的维默认 0。
MINI_TABLE: Dict[str, Dict[str, float]] = {
    # --- 属性词本身（让 has_property 的目标有向量） ---
    "有毛": {"有毛": 1}, "温血": {"温血": 1}, "胎生": {"胎生": 1},
    "有羽毛": {"有羽毛": 1}, "卵生": {"卵生": 1}, "有鳞": {"有鳞": 1},
    "冷血": {"冷血": 1}, "无四肢": {"无四肢": 1}, "有脊柱": {"有脊柱": 1},
    "无脊柱": {"无脊柱": 1}, "有外骨骼": {"有外骨骼": 1}, "六足": {"六足": 1},
    "水生": {"水生": 1}, "陆生": {"陆生": 1}, "有内骨骼": {"有脊柱": 1},
    # --- 动物（属性取自动物学常识；与图上的 has_property 独立） ---
    "猫": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
           "捕食行为": 1, "社会行为": 0.5},
    "老虎": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
             "捕食行为": 1},
    "猫科": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
             "捕食行为": 1},
    "犬科": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
             "捕食行为": 1, "社会行为": 1},
    "狗": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
           "社会行为": 1, "捕食行为": 0.5},
    "哺乳动物": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1,
                 "哺乳类": 1},
    "脊椎动物": {"有脊柱": 1, "温血": 0.5, "水生": 0.5, "陆生": 0.5},
    "无脊椎动物": {"无脊柱": 1, "无脊椎": 1, "冷血": 1, "有外骨骼": 0.5},
    "动物": {"有脊柱": 0.5, "捕食行为": 0.5, "陆生": 0.5},
    "鱼": {"有鳞": 1, "冷血": 1, "有脊柱": 1, "水生": 1, "鱼类": 1, "卵生": 1,
           "鳃部位": 1, "鳍部位": 1, "水生行为": 1},
    "硬骨鱼": {"有鳞": 1, "冷血": 1, "有脊柱": 1, "水生": 1, "鱼类": 1, "卵生": 1,
               "鳃部位": 1, "鳍部位": 1},
    "鲤科": {"有鳞": 1, "冷血": 1, "有脊柱": 1, "水生": 1, "鱼类": 1, "卵生": 1,
             "鳃部位": 1, "鳍部位": 1},
    "鸟类": {"有羽毛": 1, "温血": 1, "卵生": 1, "有脊柱": 1, "鸟类": 1,
             "翅部位": 1, "飞行行为": 1},
    "鸟": {"有羽毛": 1, "温血": 1, "卵生": 1, "有脊柱": 1, "鸟类": 1, "翅部位": 1,
           "飞行行为": 1},
    "雀形目": {"有羽毛": 1, "温血": 1, "卵生": 1, "有脊柱": 1, "鸟类": 1,
               "翅部位": 1, "飞行行为": 1},
    "鹰": {"有羽毛": 1, "温血": 1, "卵生": 1, "有脊柱": 1, "鸟类": 1, "翅部位": 1,
           "飞行行为": 1, "捕食行为": 1},
    "企鹅": {"有羽毛": 1, "温血": 1, "卵生": 1, "有脊柱": 1, "鸟类": 1,
             "飞行行为": 0, "水生行为": 1, "水生": 0.5, "社会行为": 1},
    "爬行纲": {"有鳞": 1, "冷血": 1, "卵生": 1, "有脊柱": 1, "爬行类": 1,
               "陆生": 1},
    "蛇": {"有鳞": 1, "冷血": 1, "卵生": 0.5, "有脊柱": 1, "爬行类": 1, "陆生": 1,
           "无四肢": 1, "捕食行为": 1},
    "两栖纲": {"冷血": 1, "卵生": 1, "有脊柱": 1, "两栖类": 1, "水生": 0.5,
               "陆生": 0.5},
    "青蛙": {"冷血": 1, "卵生": 1, "有脊柱": 1, "两栖类": 1, "水生": 0.5,
             "陆生": 0.5, "捕食行为": 1},
    "昆虫纲": {"无脊柱": 1, "无脊椎": 1, "昆虫类": 1, "六足": 1, "有外骨骼": 1,
               "卵生": 1, "陆生": 1, "飞行行为": 0.5, "冷血": 1},
    "昆虫": {"无脊柱": 1, "无脊椎": 1, "昆虫类": 1, "六足": 1, "有外骨骼": 1,
             "卵生": 1, "陆生": 1, "冷血": 1},
    "节肢动物": {"无脊柱": 1, "无脊椎": 1, "节肢类": 1, "有外骨骼": 1, "卵生": 1,
                 "冷血": 1, "六足": 0.5},
    "甲壳纲": {"无脊柱": 1, "无脊椎": 1, "节肢类": 1, "有外骨骼": 1, "水生": 1,
               "卵生": 1, "冷血": 1},
    "虾": {"无脊柱": 1, "无脊椎": 1, "节肢类": 1, "有外骨骼": 1, "水生": 1,
           "卵生": 1, "冷血": 1, "水生行为": 1},
    "蝴蝶": {"无脊柱": 1, "无脊椎": 1, "昆虫类": 1, "六足": 1, "有外骨骼": 1,
             "卵生": 1, "飞行行为": 1, "冷血": 1},
    "蚯蚓": {"无脊柱": 1, "无脊椎": 1, "陆生": 1, "卵生": 1,
             "冷血": 1, "无四肢": 1},
    "软体动物": {"无脊柱": 1, "无脊椎": 1, "软体类": 1, "卵生": 1, "冷血": 1,
                 "水生": 0.5, "壳部位": 1},
    "头足纲": {"无脊柱": 1, "无脊椎": 1, "软体类": 1, "头足类": 1, "水生": 1,
               "冷血": 1, "卵生": 1, "捕食行为": 1},
    "章鱼": {"无脊柱": 1, "无脊椎": 1, "软体类": 1, "头足类": 1, "水生": 1,
             "冷血": 1, "卵生": 1, "捕食行为": 1, "感知行为": 1},
    "环节动物": {"无脊柱": 1, "无脊椎": 1, "陆生": 0.5, "水生": 0.5, "卵生": 1,
                 "冷血": 1},
    "海豚": {"有毛": 0, "温血": 1, "胎生": 1, "有脊柱": 1, "水生": 1, "哺乳类": 1,
             "水生行为": 1, "社会行为": 1, "感知行为": 1},
    "蝙蝠": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "哺乳类": 1, "陆生": 1,
             "飞行行为": 1, "翅部位": 1, "感知行为": 1},
    "大象": {"有毛": 0.5, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
             "社会行为": 1},
    "猴子": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
             "社会行为": 1, "感知行为": 1},
    "人": {"有毛": 0.5, "温血": 1, "胎生": 1, "有脊柱": 1, "陆生": 1, "哺乳类": 1,
           "社会行为": 1, "感知行为": 1},
    "河狸": {"有毛": 1, "温血": 1, "胎生": 1, "有脊柱": 1, "哺乳类": 1,
             "水生": 0.5, "陆生": 1, "社会行为": 1},
    "蚂蚁": {"无脊柱": 1, "无脊椎": 1, "昆虫类": 1, "六足": 1, "有外骨骼": 1,
             "卵生": 1, "社会行为": 1, "冷血": 1},
    "蜜蜂": {"无脊柱": 1, "无脊椎": 1, "昆虫类": 1, "六足": 1, "有外骨骼": 1,
             "卵生": 1, "社会行为": 1, "飞行行为": 1, "冷血": 1},
    # --- 部位 ---
    "头部": {"头部部位": 1},
    "躯干": {"躯干部位": 1}, "尾部": {"尾部部位": 1}, "尾巴": {"尾部部位": 1},
    "前肢": {"肢体部位": 1}, "爪子": {"肢体部位": 1}, "利爪": {"肢体部位": 1},
    "蹼足": {"肢体部位": 1, "水生": 0.5},
    "翅膀": {"翅部位": 1}, "飞羽": {"翅部位": 1, "有羽毛": 1},
    "尾羽": {"尾部部位": 1, "有羽毛": 1}, "羽毛": {"有羽毛": 1},
    "喙": {"头部部位": 1}, "钩喙": {"头部部位": 1},
    "鳍": {"鳍部位": 1}, "背鳍": {"鳍部位": 1}, "胸鳍": {"鳍部位": 1},
    "鳍条": {"鳍部位": 1}, "鳃": {"鳃部位": 1}, "鳞片": {"有鳞": 1},
    "外壳": {"壳部位": 1, "有外骨骼": 1}, "墨囊": {"壳部位": 0.5},
    "触手": {"肢体部位": 1, "头足类": 1}, "八腕": {"肢体部位": 1, "头足类": 1},
    "触角": {"头部部位": 1, "昆虫类": 1}, "复眼": {"头部部位": 1, "昆虫类": 1},
    "长鼻": {"头部部位": 1}, "象牙": {"头部部位": 1}, "门齿": {"头部部位": 1},
    "翼膜": {"翅部位": 1}, "胡须": {"头部部位": 1},
    "脊柱": {"有脊柱": 1}, "骨骼系统": {"有脊柱": 1},
    "心脏": {"躯干部位": 1}, "肺": {"躯干部位": 1},
    "循环系统": {"躯干部位": 1}, "呼吸系统": {"躯干部位": 1},
    "皮肤系统": {"躯干部位": 1},
    # --- 行为 ---
    "游泳": {"水生行为": 1}, "潜水": {"水生行为": 1}, "跃出水面": {"水生行为": 1},
    "喷水": {"水生行为": 1}, "喷射推进": {"水生行为": 1, "头足类": 1},
    "奔跑": {"陆行行为": 1}, "跳跃": {"陆行行为": 1}, "攀爬": {"陆行行为": 1},
    "挖掘": {"陆行行为": 1}, "搬运": {"陆行行为": 1}, "啃树": {"陆行行为": 1},
    "筑坝": {"陆行行为": 1, "水生": 0.5},
    "飞行": {"飞行行为": 1}, "翱翔": {"飞行行为": 1}, "俯冲": {"飞行行为": 1},
    "滑行": {"飞行行为": 0.5}, "空中捕食": {"飞行行为": 1, "捕食行为": 1},
    "捕猎": {"捕食行为": 1}, "扑击": {"捕食行为": 1}, "追踪": {"捕食行为": 1},
    "取食": {"捕食行为": 1}, "采蜜": {"捕食行为": 1, "社会行为": 1},
    "产卵": {"繁殖行为": 1, "卵生": 1}, "筑巢": {"繁殖行为": 1},
    "育幼": {"繁殖行为": 1, "社会行为": 1}, "群居育幼": {"繁殖行为": 1, "社会行为": 1},
    "繁殖": {"繁殖行为": 1}, "群体协作": {"社会行为": 1},
    "通讯": {"社会行为": 1}, "信息素通讯": {"社会行为": 1, "感知行为": 1},
    "舞蹈通讯": {"社会行为": 1, "感知行为": 1},
    "防御": {"防御行为": 1}, "伪装": {"防御行为": 1}, "变色": {"防御行为": 1},
    "感知": {"感知行为": 1}, "回声定位": {"感知行为": 1}, "声呐导航": {"感知行为": 1},
    "记忆路径": {"感知行为": 1}, "使用工具": {"感知行为": 1, "工具": 1},
    "开瓶": {"感知行为": 1}, "面部表情": {"社会行为": 1},
    "倒挂": {"陆行行为": 0.5}, "运动": {"陆行行为": 0.5, "水生行为": 0.5},
    "用鼻取物": {"感知行为": 1}, "视力敏锐": {"感知行为": 1},
    # --- 属性（行为侧） ---
    "高智商": {"感知行为": 1, "社会行为": 0.5}, "社会性": {"社会行为": 1},
    "夜行性": {"陆行行为": 0.5}, "长寿命": {"陆行行为": 0.5},
    "半水生": {"水生": 0.5, "陆生": 0.5}, "体型庞大": {"陆行行为": 0.5},
    "耐寒": {"陆生": 0.5}, "对生拇指": {"肢体部位": 1},
    "门齿发达": {"头部部位": 1}, "骨骼中空": {"有脊柱": 1, "鸟类": 1},
    "三心脏": {"躯干部位": 1},
    # --- 属性组（占位，用所属类群的属性） ---
    "哺乳动物属性组": {"哺乳类": 1, "有毛": 1, "胎生": 1, "温血": 1, "有脊柱": 1},
    "鸟类属性组": {"鸟类": 1, "有羽毛": 1, "卵生": 1, "温血": 1, "有脊柱": 1},
    "鱼类属性组": {"鱼类": 1, "有鳞": 1, "鳃部位": 1, "水生": 1, "冷血": 1},
    "爬行类属性组": {"爬行类": 1, "有鳞": 1, "冷血": 1, "卵生": 1, "有脊柱": 1},
    "两栖类属性组": {"两栖类": 1, "冷血": 1, "卵生": 1, "水生": 0.5, "陆生": 0.5},
    "无脊椎动物属性组": {"无脊椎": 1, "无脊柱": 1, "有外骨骼": 0.5, "冷血": 1},
    "猛禽属性组": {"鸟类": 1, "捕食行为": 1, "飞行行为": 1, "有羽毛": 1},
    "鲸类属性组": {"哺乳类": 1, "水生": 1, "温血": 1, "胎生": 1, "有脊柱": 1},
    "社会性昆虫属性组": {"昆虫类": 1, "社会行为": 1, "六足": 1, "无脊椎": 1},
    "长鼻类属性组": {"哺乳类": 1, "陆生": 1, "社会行为": 1, "头部部位": 1},
    "头足类属性组": {"头足类": 1, "软体类": 1, "水生": 1, "无脊椎": 1},
    "翼手类属性组": {"哺乳类": 1, "飞行行为": 1, "翅部位": 1, "有脊柱": 1},
    "灵长类属性组": {"哺乳类": 1, "社会行为": 1, "感知行为": 1, "陆生": 1},
    "企鹅属性组": {"鸟类": 1, "水生行为": 1, "社会行为": 1, "卵生": 1},
    "河狸属性组": {"哺乳类": 1, "水生": 0.5, "陆生": 1, "社会行为": 1},
    # --- 区部（鲸类头部 等） ---
    "鲸类头部": {"头部部位": 1, "水生": 1, "哺乳类": 1},
    "象面部": {"头部部位": 1, "哺乳类": 1},
    "昆虫头部": {"头部部位": 1, "昆虫类": 1, "无脊椎": 1},
    # --- 医学 ---
    "医学概念": {"抽象": 1}, "解剖结构": {"解剖": 1},
    "神经系统": {"解剖": 1}, "脑": {"解剖": 1, "头部部位": 1},
    "脊髓": {"解剖": 1}, "大脑": {"解剖": 1}, "小脑": {"解剖": 1},
    "大脑皮层": {"解剖": 1},
    "生理过程": {"解剖": 0.5}, "神经传导": {"信号": 1}, "信号处理": {"信号": 1},
    "记忆": {"信号": 1}, "病理状态": {"疾病": 1}, "疾病": {"疾病": 1},
    "脑部疾病": {"疾病": 1}, "癫痫": {"疾病": 1, "症状": 0.5},
    "脑卒中": {"疾病": 1}, "神经退行性疾病": {"疾病": 1},
    "阿尔茨海默病": {"疾病": 1},
    "症状": {"症状": 1}, "抽搐": {"症状": 1}, "记忆障碍": {"症状": 1},
    "疼痛": {"症状": 1}, "发热": {"症状": 1}, "头痛": {"症状": 1},
    "眩晕": {"症状": 1}, "乏力": {"症状": 1}, "意识模糊": {"症状": 1},
    "炎症性疾病": {"疾病": 1}, "感染性疾病": {"疾病": 1}, "脑炎": {"疾病": 1},
    "干预手段": {"干预": 1}, "药物治疗": {"干预": 1}, "神经调控": {"干预": 1},
    "脑深部电刺激": {"干预": 1},
    "医学检测": {"干预": 0.5}, "脑电图": {"干预": 0.5, "信号": 1},
    "脑电图信号": {"信号": 1}, "信号": {"信号": 1}, "营养素": {"解剖": 0.5},
    # --- 影像 ---
    "影像特征": {"影像": 1}, "形态学特征": {"影像": 1, "形态": 1},
    "形状特征": {"影像": 1, "形态": 1}, "圆形": {"形态": 1},
    "椭圆形": {"形态": 1}, "不规则形": {"形态": 1}, "分叶状": {"形态": 1},
    "边缘特征": {"影像": 1, "形态": 1}, "光滑": {"形态": 1}, "毛刺状": {"形态": 1},
    "模糊": {"形态": 1}, "针刺状": {"形态": 1},
    "大小特征": {"影像": 1, "大小": 1}, "微小": {"大小": 1}, "小型": {"大小": 1},
    "中等": {"大小": 1}, "大型": {"大小": 1},
    "密度特征": {"影像": 1}, "低密度": {"影像": 1}, "等密度": {"影像": 1},
    "高密度": {"影像": 1}, "钙化": {"影像": 1},
    "强化特征": {"影像": 1}, "均匀强化": {"影像": 1}, "不均匀强化": {"影像": 1},
    "环形强化": {"影像": 1}, "无强化": {"影像": 1},
    "位置特征": {"影像": 1}, "肺上叶": {"解剖": 1}, "肺下叶": {"解剖": 1},
    "肺门": {"解剖": 1}, "外周": {"影像": 1}, "中央": {"影像": 1},
    "病理结构": {"疾病": 1}, "结节": {"疾病": 1}, "恶性结节": {"疾病": 1},
    "良性结节": {"疾病": 1},
    # --- 通用抽象层 ---
    "生物": {"有脊柱": 0.5}, "抽象概念": {"抽象": 1}, "方法": {"抽象": 1},
    "算法": {"抽象": 1}, "策略": {"抽象": 1}, "模式": {"抽象": 1},
    "属性": {"属性": 1}, "物理属性": {"属性": 1}, "抽象属性": {"属性": 1},
    "关系": {"关系": 1}, "因果": {"关系": 1}, "部分整体": {"关系": 1},
    "规则": {"抽象": 1}, "语法规则": {"抽象": 1, "关系": 0.5},
    "推理规则": {"抽象": 1, "关系": 0.5},
    "人造物": {"工具": 0.5}, "物理人造物": {"物理": 1}, "工具": {"工具": 1},
    "设备": {"工具": 1}, "建筑": {"物理": 1}, "信息人造物": {"信息": 1},
    "数据对象": {"信息": 1}, "数据库": {"信息": 1, "工具": 0.5},
    "表": {"信息": 1}, "视图": {"信息": 1}, "索引": {"信息": 1},
    "查询": {"信息": 1}, "信息载体": {"信息": 1}, "文档": {"信息": 1},
    "网页": {"信息": 1}, "文件": {"信息": 1}, "软件产物": {"信息": 1},
    "程序": {"信息": 1, "工具": 0.5}, "库": {"信息": 1}, "API": {"信息": 1},
    "模型": {"抽象": 1},
    # --- 功能词 / 量词 / 身体词（给个可区分的稠密向量即可） ---
    "洗澡": {"防御行为": 0.5}, "吃": {"捕食行为": 1}, "喝": {"捕食行为": 1},
    "抓": {"捕食行为": 1}, "跑": {"陆行行为": 1}, "看": {"感知行为": 1},
    "说": {"社会行为": 1}, "研究": {"感知行为": 1}, "完成": {"抽象": 0.5},
    "给": {"社会行为": 0.5}, "是": {"关系": 1}, "有": {"关系": 1},
    "高": {"大小": 0.5}, "大": {"大小": 1}, "小": {"大小": 1},
    "好": {"属性": 1}, "坏": {"属性": 1},
    "高兴": {"感知行为": 1}, "花": {"形态": 0.5}, "花儿": {"形态": 0.5},
    "葡萄": {"陆生": 0.5}, "桌子": {"物理": 1}, "书": {"信息": 1},
    "水": {"水生": 1}, "食物": {"捕食行为": 0.5}, "营养物": {"解剖": 0.5},
    "可食用": {"属性": 1}, "有营养": {"属性": 1},
    "作业": {"抽象": 1}, "澡": {"防御行为": 0.5}, "洗": {"防御行为": 0.5},
    # --- 虚词 / 量词 / 代词：给可区分的向量，避免它们全落在零向量上 ---
    "把": {"虚词": 1, "关系": 0.5}, "被": {"虚词": 1}, "在": {"虚词": 1},
    "从": {"虚词": 1}, "向": {"虚词": 1},
    "的": {"虚词": 1}, "了": {"虚词": 1}, "着": {"虚词": 1}, "过": {"虚词": 1},
    "只": {"量词": 1}, "个": {"量词": 1}, "条": {"量词": 1}, "份": {"量词": 1},
    "他": {"虚词": 0.5}, "我": {"虚词": 0.5}, "你": {"虚词": 0.5},
    "完": {"抽象": 0.5}, "打算": {"抽象": 0.5},
}


def build_mini_vectors(path: pathlib.Path = DEFAULT_MINI_PATH) -> pathlib.Path:
    """把 :data:`MINI_TABLE` 写成 ``word v1 v2 ...`` 文本（可复现，值四舍五入到 3 位）。"""
    order = {name: i for i, name in enumerate(MINI_DIMS)}
    rows: Dict[str, List[float]] = {}
    for word, props in MINI_TABLE.items():
        vec = [0.0] * MINI_DIM
        for name, value in props.items():
            if name not in order:
                raise SystemExit(f"[错误] 未知维名 {name}（词 {word}）")
            vec[order[name]] = float(value)
        rows[word] = vec

    lines = [
        f"# Logos mini word vectors — 手写动物学属性矩阵 v1（{MINI_DIM} 维，非学习得到）",
        f"# 生成：python logos_transformer/embeddings.py --build-mini",
        "# 维序：" + " ".join(f"{i}:{n}" for i, n in enumerate(MINI_DIMS)),
        f"# 覆盖 {len(rows)} 词；仅供 CI / 冒烟 / 上界探针，见 PRETRAINED_EMBEDDINGS.md",
    ]
    for word in sorted(rows):
        vec = rows[word]
        body = " ".join(f"{v:g}" for v in vec)
        lines.append(f"{word} {body}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_mini(verbose: bool = False) -> ChineseWordVectorLoader:
    """载入内置 mini 向量（缺失时自动重建）。"""
    if not DEFAULT_MINI_PATH.exists():
        build_mini_vectors(DEFAULT_MINI_PATH)
    return ChineseWordVectorLoader(DEFAULT_MINI_PATH, dim=MINI_DIM, verbose=verbose)


def charngram_loader(dim: int = 64) -> ChineseWordVectorLoader:
    """空加载器 + 全量字形哈希：不读任何文件，永不 OOV。"""
    loader = ChineseWordVectorLoader(None, dim=dim)
    loader.vectors = {}          # 保证不会命中「真词」
    return loader


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="词向量覆盖率报告 / 重建 mini 向量")
    parser.add_argument("--db", type=pathlib.Path, default=_ROOT / "logos.db")
    parser.add_argument("--vectors", type=pathlib.Path, default=None,
                        help="预训练向量文件（.txt/.gz/.bz2）；缺省用内置 mini")
    parser.add_argument("--dim", type=int, default=None, help="向量维度（缺省自动推断）")
    parser.add_argument("--oov-mode", choices=("zero", "charngram"), default="zero")
    parser.add_argument("--report", action="store_true", help="打印覆盖率报告")
    parser.add_argument("--build-mini", action="store_true", help="重建 mini_vectors.txt")
    parser.add_argument("--neighbour-fill", type=int, default=0,
                        help="用 N 跳邻居均值填补 OOV（0=不填）")
    args = parser.parse_args(argv)

    if args.build_mini:
        path = build_mini_vectors(DEFAULT_MINI_PATH)
        print(f"已重建 {path}（{len(MINI_TABLE)} 词 × {MINI_DIM} 维）")
        if not args.report:
            return 0

    db = pathlib.Path(args.db).expanduser().resolve()
    if not db.exists():
        print(f"数据库不存在：{db}", file=sys.stderr)
        return 1
    sense_words = collect_sense_words(db)

    if args.vectors is not None:
        keep = keep_set_for(sense_words)
        loader = ChineseWordVectorLoader(
            args.vectors, dim=args.dim, keep=keep, verbose=True
        )
    else:
        loader = load_mini(verbose=True)

    report = loader.coverage_report([(w, p) for _s, w, p in sense_words],
                                   oov_mode=args.oov_mode)
    print("-" * 66)
    for line in report.lines():
        print(line)

    features, stats = build_node_features(db, loader, loader.dim or MINI_DIM,
                                         oov_mode=args.oov_mode)
    if args.neighbour_fill and torch is not None:
        from prototype_gat import load_graph  # 延迟导入，避免 --report 也要建图
        graph = load_graph(db, include_closure=False, verbose=False)
        mask = features.abs().sum(dim=1) < 1e-12
        features, filled = fill_oov_by_neighbours(
            features, graph.edge_index, mask, iters=args.neighbour_fill
        )
        print(f"邻居填充      : {filled} 个 OOV 节点由 {args.neighbour_fill} 跳邻居均值初始化")
    print("-" * 66)
    print(f"特征矩阵      : {tuple(features.shape)}  "
          f"非零行 {int((features.abs().sum(dim=1) > 0).sum())}/{features.size(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
