#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logos 基于模板的中文句法-语义解析器。

设计立场
--------
Logos 用**数据库里的中文关系**替代向量做语义解析，因此本模块**不训练、不加载
任何模型**，全部知识来自 schema 中已落地的那几张表：

===================  ==================================================
表                    在本模块中的角色
===================  ==================================================
``words``/``senses``  分词词典 + 词性 + 次类 + 频次 + 离合点
``separable_patterns`` / ``bound_objects``  离合词切分与重组
``sentence_patterns`` / ``pattern_slots``   句式模板（槽位真源）
``slot_constraints``  槽位的合法填充条件（部位、词性、义素、有定、体…）
``function_words`` / ``function_word_roles``  虚词 → 语义角色的**直接映射**
``semantic_roles``   角色的 ``case_marker`` 与分组（角色合法性判据）
``sense_features`` / ``sense_categories``  选择限制的细粒度依据
``relations`` / ``v_relations_all``  谓词→典型施事/受事的词汇语义约束
``inference_rules`` / ``rule_premises`` / ``rule_conclusions``  推理规则
``ellipsis_rules``   省略回填策略（按 context_type × slot_role × priority）
``topic_chains`` / ``utterances``  篇章层话题链
``complements``      述补结构实例（动结/动趋/可能补语）
===================  ==================================================

解析流程（``parse`` 一条龙）::

    tokenize → match_pattern → fill_slots → annotate_semantic_roles → query_relations

方向约定（与 ``logos.db.closure`` 一致，务必先读 ``docs/CLOSURE_DIRECTION.md``）
---------------------------------------------------------------------------
``relations`` 边是 ``src(下位/具体) --relation--> dst(上位/一般)``，
``closure`` 行是 ``(ancestor=上位, descendant=下位)`` —— **行的方向 = 边的反向**。
本模块只读闭包，全部通过 :func:`logos.db.closure.get_property_inheritance` /
``get_ancestor_ids`` 访问，不自己写 SQL 碰 ``closure`` 表。

与 spec 的接口差异（有意为之，均向后兼容）
------------------------------------------
* ``fill_slots`` 返回 ``dict[str, SlotFill]``。``SlotFill`` 实现了 ``__len__`` /
  ``__iter__`` / ``__getitem__``，因此可直接当作 spec 要求的 ``Token | list[Token]``
  使用（单 token 槽位迭代出 1 项，多 token 槽位迭代出 N 项）。
* ``annotate_semantic_roles`` / ``query_relations`` 的 spec 签名只有一个参数，
  本模块追加了 ``match=None`` / ``tokens=None`` 两个**带默认值**的可选参数，
  便于在传入自定义 ``filled_slots`` 时仍能拿到虚词上下文。

评分模型（``_score`` 的唯一真源，便于回归锁定）
-----------------------------------------------
::

    score = spec_weight × (字面量分 + 槽位分 + 约束分) × (0.35 + 0.65×覆盖率)
            + 述补命中加分 + 话题逗号加分 － 标记跳过罚 － 残余罚

* **字面量分**用 IDF 加权：``6 + 4×ln(句式总数 / 含该字面量的句式数)``。
  「把/被/难道/越来越」这类只出现在 1–2 个模板里的标记自然权重高，
  这正是「检测关键虚词标记」所需要的。
* **覆盖率** = （字面量命中 token + 槽位填充 token）/ 句式内全部 token。
  跳过的助词**不计入**分子 —— 因此「跳过一堆助词硬凑基本句型」会被自然压低。
* 约束是**软约束**（违反扣权重分并在结果里点名），唯一硬约束是 ``kind='marker'``。

已知边界（v0.1，明确记录而非假装没有）
--------------------------------------
1. 分词是**正向最大匹配 + 词典**，不吃 OOV 的歧义切分；OOV 走
   :data:`_CLOSED_CLASS_FALLBACK`（约 60 个虚词/数词/方位后缀）兜底，
   兜底的兜底记 ``pos='?'``。词典长大后该表应收缩到 0。
2. 句式模板要求**整句吃干**（残余 ≤ ``max_trailing_residual`` 个 token），
   因此 `请你吃鱼` 这类"模板只写了 V、实际还带宾语"的句子会带残余罚分命中，
   而不是被硬拒。
3. 省略回填只实现 ``ellipsis_rules`` 里 10 个 strategy 中可在单机内存态完成的那些；
   ``world_knowledge`` / ``indefinite_default`` 返回**未解析占位符** ``∅``，
   并在 ``confidence`` 上打折，不伪造实体。
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..db.closure import (
    HIERARCHY_RELATION,
    get_ancestor_ids,
    get_ancestors,
    get_descendant_ids,
    get_descendants,
    get_property_inheritance,
)
from ..db.connection import DatabaseError

__all__ = [
    # 异常
    "ParserError",
    "TokenizeError",
    "TemplateCompileError",
    "MatchError",
    # 数据类型
    "Token",
    "SlotFill",
    "SemanticRole",
    "InferenceResult",
    "ParseResult",
    "PatternMatch",
    "PatternAttempt",
    "ConstraintCheck",
    "ParserContext",
    # 主类与便捷函数
    "SyntaxParser",
    "parse",
    "tokenize",
    "sense_text",
    "build_parser",
    "compile_template",
    # 常量
    "PUNCT_POS",
    "SENTENCE_PUNCT",
    "CLAUSE_PUNCT",
    "CONTENT_POS",
    "SKIPPABLE_POS",
    "SKIPPABLE_FW_CATEGORIES",
    "ROLE_TO_CONSTRAINT_SLOT",
    "DEFAULT_MAX_SLOT_TOKENS",
    "SPEC_WEIGHTS",
    "CONSTRAINT_WEIGHTS",
    "HARD_CONSTRAINT_KINDS",
]

logger = logging.getLogger("logos.parser.syntax_parser")


# ===========================================================================
# 1. 常量
# ===========================================================================
#: 标点 token 的词性（与 ``pos_tags`` 无关，是本模块内部约定的"非词"标记）
PUNCT_POS = "w"

#: 句末标点：切句依据。`；` 归入句末（分句语义上各自独立）
SENTENCE_PUNCT = frozenset("。！？!?；;…")

#: 句中标点：**允许跨过**（带罚分），是「逗号前为 Topic」的承载者
CLAUSE_PUNCT = frozenset("，,、：:·～—")

#: 全部被识别为标点的字符（含成对符号与省略号）
_PUNCT_CHARS = frozenset(
    "。，、；：？！…—～·「」『』（）〔〕【】《》〈〉“”‘’"
    ",.!?;:'\"()[]{}<>-—/\\|~`@#$%^&*_+= \t\n\r"
)

#: 可自由跳过的词性：助词・语气词・副词・叹词
SKIPPABLE_POS = frozenset({"u", "y", "d", "e"})

#: 可自由跳过的虚词大类（来自 ``function_words.category``）
SKIPPABLE_FW_CATEGORIES = frozenset(
    {"动态助词", "结构助词", "语气助词", "比况助词", "数助词", "副词", "叹词", "拟声词"}
)

#: 实词词性白名单（用于 ``is_content`` 与残余合法性判断）
CONTENT_POS = frozenset(
    {"n", "nt", "nl", "nd", "v", "a", "b", "m", "q", "r", "i", "l", "s", "p", "c", "o", "?"}
)

#: 语义角色 → ``relation_constraints.slot`` 的名称映射（两套命名各自独立，需显式对齐）
ROLE_TO_CONSTRAINT_SLOT: dict[str, str | None] = {
    "施事": "agent",
    "致事": "agent",
    "感事": "agent",
    "受事": "patient",
    "结果": "result",
    "与事": "dative",
    "受益者": "dative",
    "受损者": "patient",
    "对象": "target",
    "比较对象": "target",
    "伴随": "target",
    "喻体": "target",
    "工具": "instrument",
    "材料": "instrument",
    "方式": "instrument",
    "处所": "location",
    "起点": "location",
    "终点": "location",
    "经由": "location",
    "方向": "location",
    "时点": "time",
    "时段": "time",
    "频次": "time",
    "数量": "theme",
    "程度": "theme",
    "幅度": "theme",
    "系事": "theme",
    "主题": "theme",
    "话题": "theme",
    "焦点": "theme",
    "述题": None,
    "命题": None,
}

#: 句式类别权重：特殊句式/话题结构优先于基本句型（避免"最泛的模板通吃"）
SPEC_WEIGHTS: dict[str, float] = {
    "特殊句式": 1.15,
    "话题结构": 1.15,
    "复句": 1.12,
    "紧缩句": 1.12,
    "句类": 1.10,
    "短语结构": 1.05,
    "基本句型": 1.00,
}

#: 各类约束的权重（命中加分 / 违反扣 ``1.5×``）
CONSTRAINT_WEIGHTS: dict[str, float] = {
    "lexeme": 4.0,
    "marker": 4.0,
    "subcategory": 3.0,
    "pos": 2.0,
    "feature": 2.0,
    "animacy": 2.0,
    "countability": 2.0,
    "classifier": 2.0,
    "role": 1.5,
    "definiteness": 1.5,
    "aspect": 1.5,
}

#: 唯一硬约束：标注「某虚词标记必现」的，缺失即整条模板判定为不匹配
HARD_CONSTRAINT_KINDS = frozenset({"marker"})

#: 槽位默认最多吃几个 token（防止一个槽位吞掉整句）
DEFAULT_MAX_SLOT_TOKENS = 6

_CON_VIOLATION_MULT = 1.5
_COVERAGE_FLOOR = 0.35
#: 跳过助词/语气词的代价（轻 —— 它们本就不承载论元信息）
_SKIP_MARKER_PENALTY = 0.8
#: 跳过副词的代价（重 —— 副词承载语义，`猫吃_不_吃鱼` 的 `不` 不能当噪声）
_SKIP_ADVERB_PENALTY = 3.0
#: 槽位区间**内部**夹着可跳过成分的代价（`{V}=吃不吃` 这种吞并判为可疑）
_INTERNAL_MARKER_PENALTY = 4.0
#: 句首/句尾残余成分的代价
_RESIDUAL_PENALTY = 1.5
#: 每个 OOV token 被纳入槽位的代价（词典覆盖不足时，结构匹配的可信度下降）
_OOV_TOKEN_PENALTY = 1.2
#: 省略一个**必填**槽位的代价（省略必须比"填上"更好时才该赢）
_OMITTED_SLOT_PENALTY = 3.0
#: 一条句式最多允许省略几个必填槽位（≥2 说明模板基本没被实例化 → 拒绝）
_MAX_OMITTED_SLOTS = 1
_FUNC_POS_PRIOR = 1.5
_COMPLEMENT_BONUS = 3.0
_TOPIC_BREAK_BONUS = 2.0
_LITERAL_BASE = 6.0
_LITERAL_IDF_SCALE = 4.0
_SLOT_REQUIRED = 4.0
_SLOT_OPTIONAL = 2.0

#: 短语结构类模板（动结/动趋/可能补语）**不接受句级残余成分** ——
#: `猫吃鱼` 不该被 `{V} {R}` 解释成 V=吃 R=鱼。
_PHRASE_CATEGORIES = frozenset({"短语结构"})

#: 谓语槽偏好的词性 / 名词性槽位偏好的词性（结构先验，扣分而非否决）
_VERBAL_POS = frozenset({"v", "a", "b", "d", "?"})
_NOMINAL_POS = frozenset({"n", "nt", "nl", "nd", "r", "m", "q", "i", "l", "?"})

#: 汉语数词（OOV 时判为 ``m``）—— 数词是开放的，不该逐个建词条
_CN_NUMERALS = frozenset("零一二三四五六七八九十百千万亿两几数多少半兆")
#: 指示代词（OOV 时判为 ``r``）
_DEMONSTRATIVES = frozenset("这那哪")
_DETERMINERS = frozenset({"每", "某", "各", "任何", "所有", "有的", "整个"})
#: 方位后缀：OOV 且以此结尾 → 处所词 ``nl``（"桌上/屋里"）
_LOC_SUFFIX = (
    "上", "下", "里", "外", "中", "内", "前", "后", "间", "旁",
    "左", "右", "东", "南", "西", "北", "边", "面", "处", "底", "顶", "侧",
)

#: OOV 兜底词表：**分词词典尚未覆盖的封闭类虚词**。
#: 这张表是临时桥梁 —— 词典补齐对应词条后应逐项删除，``cache_info()['fallback_used']``
#: 会统计它的实际使用量，用于驱动收缩。
_CLOSED_CLASS_FALLBACK: dict[str, str] = {
    # 副词
    "不": "d", "没": "d", "没有": "d", "别": "d", "很": "d", "都": "d", "也": "d",
    "就": "d", "才": "d", "还": "d", "再": "d", "又": "d", "越": "d", "太": "d",
    "更": "d", "最": "d", "十分": "d", "非常": "d", "已经": "d", "将要": "d",
    "曾经": "d", "马上": "d", "赶快": "d", "仍然": "d", "当然": "d", "难道": "d",
    "渐渐": "d", "终于": "d", "十分地": "d", "必须": "d", "一定": "d",
    # 介词 / 连词
    "比": "p", "跟": "p", "和": "p", "与": "p", "对": "p", "为": "p", "把": "p",
    "由": "p", "至于": "p", "关于": "p", "按照": "p", "根据": "p", "除了": "p",
    "因为": "p", "由于": "p", "为了": "p", "朝着": "p", "沿着": "p",
    "而": "c", "并": "c", "而且": "c", "但是": "c", "虽然": "c", "如果": "c",
    "只要": "c", "即使": "c", "还是": "c", "或者": "c", "以及": "c", "不过": "c",
    # 助词 / 语气词
    "得": "u", "地": "u", "之": "u", "所": "u", "一样": "u", "似的": "u",
    "吗": "y", "呢": "y", "吧": "y", "啊": "y", "呀": "y", "哇": "y",
    "嘛": "y", "哦": "y", "罢了": "y",
    # 动词性标记
    "请": "v", "让": "v", "使": "v", "叫": "v", "要": "v", "能": "v", "会": "v",
    "可以": "v", "应该": "v", "愿意": "v", "想": "v", "来": "v", "去": "v",
    # 其他
    "所_": "u",
}

#: ``sentence_moods`` 的语气 → 句末标点/语气词的判定线索
_MOOD_MARKERS: dict[str, tuple[str, ...]] = {
    "疑问": ("吗", "呢", "吧", "么", "?", "？"),
    "祈使": ("吧", "啊", "!", "！"),
    "感叹": ("啊", "呀", "哇", "多么", "!", "！"),
}


# ===========================================================================
# 2. 异常
# ===========================================================================
class ParserError(DatabaseError):
    """解析器基类异常（沿用连接层的 ``DatabaseError`` 体系）。"""


class TokenizeError(ParserError):
    """分词阶段失败（输入非法、词典为空等）。"""


class TemplateCompileError(ParserError):
    """``sentence_patterns.template`` 无法编译成可执行的槽位序列。"""


class MatchError(ParserError):
    """匹配阶段失败（无可用句式、约束全部冲突等）。"""


# ===========================================================================
# 3. 基础数据类型
# ===========================================================================
@dataclass(frozen=True)
class Token:
    """一个词元。

    :param text: 表层形式
    :param pos: 词性（``pos_tags.pos``）
    :param entry_id: 词条 id（``words.word_id``；OOV 为 ``None``）
    :param sense_id: 义项 id（``senses.sense_id``；OOV 为 ``None``）
    :param start: 在**原文本**中的起始字符偏移
    :param end: 结束偏移（不含）
    :param is_separable: 是否属于离合结构（整体或切分后的任一段）
    :param separator_pos: 离合点的字符位置（来自 ``words.separator_pos``）
    :param is_bound_root: 是否为黏着词根（``澡/忙/架``）
    :param is_oov: 词典未收录，词性来自兜底推断
    :param is_marker_literal: 该 token 是句式模板里的语法标记词（自带 pos，非实词）
    :param separable_group: 离合结构分组号（同一离合结构共享）
    :param sep_role: 在离合结构中的角色，``host``（动词）或 ``object``（黏着宾语）
    :param sep_group_sense_id: 离合词整体义项（如 `洗澡` 的 v 义项），供回溯
    :param subcategory: ``senses.subcategory``（动词次类：判断/存现/给予/使令…）
    :param fw_category: ``function_words.category``（介词/结构助词/动态助词…）
    :param frequency: ``senses.frequency``
    :param is_primary: 是否主义项
    :param inferred_pos: 词性推断来源（``''`` = 词典命中，其余为兜底依据）
    """

    text: str
    pos: str
    entry_id: int | None = None
    sense_id: int | None = None
    start: int = 0
    end: int = 0
    is_separable: bool = False
    separator_pos: int | None = None
    is_bound_root: bool = False
    is_oov: bool = False
    is_marker_literal: bool = False
    separable_group: int | None = None
    sep_role: str | None = None
    sep_group_sense_id: int | None = None
    subcategory: str | None = None
    fw_category: str | None = None
    frequency: int = 0
    is_primary: bool = False
    inferred_pos: str = ""

    # ---- 便捷判定 ----
    @property
    def is_punct(self) -> bool:
        return self.pos == PUNCT_POS

    @property
    def is_skippable(self) -> bool:
        return self.pos in SKIPPABLE_POS or (self.fw_category in SKIPPABLE_FW_CATEGORIES)

    @property
    def is_content(self) -> bool:
        return not self.is_punct and not self.is_skippable

    @property
    def is_clause_punct(self) -> bool:
        return self.is_punct and self.text in CLAUSE_PUNCT

    @property
    def is_sentence_punct(self) -> bool:
        return self.is_punct and self.text in SENTENCE_PUNCT

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "pos": self.pos,
            "entry_id": self.entry_id,
            "sense_id": self.sense_id,
            "start": self.start,
            "end": self.end,
            "is_separable": self.is_separable,
            "is_bound_root": self.is_bound_root,
            "is_oov": self.is_oov,
            "sep_role": self.sep_role,
            "separable_group": self.separable_group,
            "subcategory": self.subcategory,
            "fw_category": self.fw_category,
        }

    def __str__(self) -> str:  # pragma: no cover - 调试用
        tag = self.pos
        if self.is_separable:
            tag += "*"
        if self.is_oov:
            tag += "?"
        return f"{self.text}/{tag}"


@dataclass
class ConstraintCheck:
    """一条槽位约束的检查结果。

    :param pattern_id/slot_name/kind/value: 与 ``slot_constraints`` 一一对应
    :param ok: 是否通过
    :param severity: ``hard``（否决整条模板）或 ``soft``（扣分并点名）
    :param detail: 人类可读的判定依据
    """

    pattern_id: int
    slot_name: str
    kind: str
    value: str
    ok: bool
    severity: str = "soft"
    detail: str = ""

    def render(self) -> str:
        flag = "✓" if self.ok else "✗"
        return f"{flag} [{self.slot_name}] {self.kind}={self.value} —— {self.detail}"


@dataclass
class SlotFill:
    """一个已填充的槽位。

    实现 ``__len__`` / ``__iter__`` / ``__getitem__``，可直接当
    ``list[Token]``（或单 token）使用，兼容 spec 里
    ``dict[str, Token | list[Token]]`` 的形态。
    """

    slot_name: str
    tokens: list[Token] = field(default_factory=list)
    #: ``matched`` | ``ellipsis`` | ``absent``
    source: str = "matched"
    ellipsis_rule_id: int | None = None
    strategy: str | None = None
    confidence: float = 1.0
    note: str = ""

    @property
    def text(self) -> str:
        return "".join(t.text for t in self.tokens)

    @property
    def head(self) -> Token | None:
        """中心语：谓语槽取第一个动词性 token，其余取最后一个实词。"""
        if not self.tokens:
            return None
        for t in self.tokens:
            if t.pos in ("v", "a", "b"):
                return t
        return self.tokens[-1]

    @property
    def head_sense_id(self) -> int | None:
        h = self.head
        return h.sense_id if h else None

    def as_value(self) -> Any:
        """spec 形态：单 token 返回 ``Token``，多 token 返回 ``list[Token]``。"""
        if len(self.tokens) == 1:
            return self.tokens[0]
        return list(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, i: Any) -> Any:
        return self.tokens[i]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot_name,
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "source": self.source,
            "strategy": self.strategy,
            "ellipsis_rule_id": self.ellipsis_rule_id,
            "confidence": round(self.confidence, 4),
            "note": self.note,
        }


@dataclass
class SemanticRole:
    """一个语义角色指派。

    :param role: ``semantic_roles.role``
    :param slot_name: 来源槽位
    :param tokens: 承载该角色的 token 序列
    :param marker: 引出该角色的虚词（如 ``把`` / ``被``），无则为空
    :param source: ``default``（槽位默认角色）/ ``marker``（虚词改写）/ ``ellipsis``
    :param confidence: 角色判定置信度
    :param notes: 选择限制等附注
    """

    role: str
    slot_name: str
    tokens: list[Token] = field(default_factory=list)
    marker: str = ""
    source: str = "default"
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)
    #: 「把字句 / 被字句」等句式整体义（取自 ``sentence_patterns.semantics``）
    pattern_semantics: str = ""

    @property
    def text(self) -> str:
        return "".join(t.text for t in self.tokens)

    @property
    def head(self) -> Token | None:
        return self.tokens[-1] if self.tokens else None

    @property
    def head_sense_id(self) -> int | None:
        h = self.head
        return h.sense_id if h else None

    @property
    def entity(self) -> str:
        """实体名（中心语的词形）。

        与 :attr:`text` 的区别：``受事="一条鱼"`` 时 ``entity="鱼"``。
        推理链里的主语必须用 ``entity`` —— 否则会打印
        ``一条鱼 --can_do--> 吃``，看起来像"一条鱼"是知识库里的条目。
        """
        h = self.head
        if h is not None and h.entry_id is not None:
            return h.text
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "slot": self.slot_name,
            "text": self.text,
            "entity": self.entity,
            "marker": self.marker,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "head_sense_id": self.head_sense_id,
            "notes": list(self.notes),
        }


@dataclass
class InferenceResult:
    """一条推理结果（含推理链，可解释）。"""

    kind: str                      # relation | closure | inherit | rule | constraint | default
    subject: str = ""
    relation: str = ""
    obj: str = ""
    confidence: float = 1.0
    chain: list[str] = field(default_factory=list)
    rule_id: int | None = None
    rule_name: str = ""
    rule_type: str = ""
    note: str = ""
    #: 命中的行（如 ``relations.relation_id``），便于回溯到证据
    evidence: list[Any] = field(default_factory=list)

    def render(self) -> str:
        head = f"{self.subject} --{self.relation}--> {self.obj}"
        tail = f"  ({self.rule_name or self.kind}, conf={self.confidence:.2f})"
        return head + tail

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "confidence": round(self.confidence, 4),
            "chain": list(self.chain),
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "note": self.note,
            "evidence": list(self.evidence),
        }


@dataclass
class PatternMatch:
    """一次成功的句式匹配。"""

    pattern_id: int
    name: str
    category: str
    mood: str | None
    template: str
    semantics: str | None
    variant: str
    score: float
    coverage: float
    #: 槽位名 → 该槽位占用的 token 区间列表（同名槽位多次出现时长度 > 1）
    slot_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    literal_hits: list[tuple[str, int]] = field(default_factory=list)
    skipped_markers: list[int] = field(default_factory=list)
    leading_residual: tuple[int, int] = (0, 0)
    trailing_residual: tuple[int, int] = (0, 0)
    clause_breaks: list[int] = field(default_factory=list)
    constraint_checks: list[ConstraintCheck] = field(default_factory=list)
    #: 被省略的必填槽位（由 ``ellipsis_rules`` 授权）
    omitted_slots: list[str] = field(default_factory=list)
    #: 句子在 token 列表中的区间
    sentence_span: tuple[int, int] = (0, 0)
    sentence_terminator: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def constraint_errors(self) -> list[str]:
        return [c.render() for c in self.constraint_checks if not c.ok]

    @property
    def occurrences(self) -> int:
        return sum(len(v) for v in self.slot_spans.values())

    def spans_of(self, slot_name: str) -> list[tuple[int, int]]:
        return list(self.slot_spans.get(slot_name, ()))

    def render(self, tokens: Sequence[Token] | None = None) -> str:
        lines = [f"句式 {self.name}(#{self.pattern_id}) [{self.category}] "
                 f"variant={self.variant!r} score={self.score:.2f} coverage={self.coverage:.0%}"]
        if tokens:
            for slot, spans in self.slot_spans.items():
                for (s, e) in spans:
                    lines.append(f"  · {slot:<7} = {''.join(t.text for t in tokens[s:e])}")
        for c in self.constraint_checks:
            if not c.ok:
                lines.append("  ! " + c.render())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "category": self.category,
            "mood": self.mood,
            "template": self.template,
            "semantics": self.semantics,
            "variant": self.variant,
            "score": round(self.score, 4),
            "coverage": round(self.coverage, 4),
            "slot_spans": {k: [list(x) for x in v] for k, v in self.slot_spans.items()},
            "literal_hits": [list(x) for x in self.literal_hits],
            "constraint_errors": self.constraint_errors,
            "notes": list(self.notes),
        }


@dataclass
class PatternAttempt:
    """一次句式尝试（无论成功失败），用于「top-3 候选 + 失败原因」。"""

    pattern_id: int
    name: str
    category: str
    matched: bool
    score: float = 0.0
    coverage: float = 0.0
    reason: str = ""
    constraint_errors: list[str] = field(default_factory=list)
    slot_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def render(self) -> str:
        flag = "OK " if self.matched else "REJ"
        return f"[{flag}] {self.name}(#{self.pattern_id}) score={self.score:.2f} —— {self.reason}"


@dataclass
class ParserContext:
    """解析上下文（篇章层）。

    :param doc_id: 文档 id，与 ``utterances.doc_id`` 对齐
    :param speaker/listener: 说话人与听话人（省略回填的 ``speaker``/``listener`` 策略用）
    :param topic: 当前话题的表层形式
    :param context_type: 覆盖自动推断的 ``ellipsis_rules.context_type``
    :param turn_index: 当前轮次（与 ``max_distance`` 比较）
    :param speaker_entity/listener_entity: 说话人/听话人的实体（词条形式）
    """

    doc_id: str = "default"
    speaker: str | None = None
    listener: str | None = None
    topic: str | None = None
    context_type: str | None = None
    turn_index: int = 0
    speaker_entity: str | None = None
    listener_entity: str | None = None

    @classmethod
    def from_any(cls, obj: Any) -> "ParserContext":
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Mapping):
            allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
            return cls(**{k: v for k, v in obj.items() if k in allowed})
        raise ParserError(f"无法把 {type(obj).__name__} 当作 ParserContext")


@dataclass
class ParseResult:
    """``SyntaxParser.parse`` 的完整结果。"""

    text: str
    tokens: list[Token] = field(default_factory=list)
    pattern_name: str | None = None
    pattern_id: int | None = None
    category: str | None = None
    semantics: str | None = None
    mood: str | None = None
    filled_slots: dict[str, SlotFill] = field(default_factory=dict)
    roles: dict[str, SemanticRole] = field(default_factory=dict)
    inferences: list[InferenceResult] = field(default_factory=list)
    confidence: float = 0.0
    match: PatternMatch | None = None
    #: 失败时的 top-3 候选与原因；成功时附低分候选供对比
    diagnostics: list[PatternAttempt] = field(default_factory=list)
    context_used: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.pattern_id is not None

    @property
    def failed_reasons(self) -> list[str]:
        return [a.render() for a in self.diagnostics if not a.matched]

    def role_of(self, role: str) -> SemanticRole | None:
        return self.roles.get(role)

    def text_of(self, slot_name: str) -> str:
        f = self.filled_slots.get(slot_name)
        return f.text if f else ""

    def render(self) -> str:
        lines = [f"原文：{self.text}",
                 "词元：" + "  ".join(str(t) for t in self.tokens)]
        if self.match:
            lines.append(self.match.render(self.tokens))
        else:
            lines.append("句式：<未匹配>")
        if self.roles:
            lines.append("语义角色：")
            for name, r in self.roles.items():
                mk = f"（{r.marker}）" if r.marker else ""
                lines.append(f"  · {name:<8} {r.text}{mk}  [{r.source}]")
        if self.inferences:
            lines.append("推理链：")
            for inf in self.inferences:
                lines.append("  · " + inf.render())
                for c in inf.chain:
                    lines.append(f"      {c}")
        lines.append(f"置信度：{self.confidence:.2%}  耗时 {self.elapsed_ms:.2f} ms")
        if not self.ok and self.diagnostics:
            lines.append("候选（top-3，含失败原因）：")
            for a in self.diagnostics[:3]:
                lines.append("  · " + a.render())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "pattern_name": self.pattern_name,
            "pattern_id": self.pattern_id,
            "category": self.category,
            "semantics": self.semantics,
            "mood": self.mood,
            "match": self.match.to_dict() if self.match else None,
            "filled_slots": {k: v.to_dict() for k, v in self.filled_slots.items()},
            "roles": {k: v.to_dict() for k, v in self.roles.items()},
            "inferences": [i.to_dict() for i in self.inferences],
            "confidence": round(self.confidence, 4),
            "diagnostics": [
                {"pattern_id": a.pattern_id, "name": a.name, "matched": a.matched,
                 "score": round(a.score, 4), "reason": a.reason,
                 "constraint_errors": a.constraint_errors}
                for a in self.diagnostics
            ],
            "context_used": dict(self.context_used),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


# ===========================================================================
# 4. 内部数据结构
# ===========================================================================
@dataclass(frozen=True)
class _Lex:
    """词典条目（一个 (word, pos) 组合即一个条目）。"""

    word: str
    word_id: int
    sense_id: int
    pos: str
    sense_no: int
    is_primary: bool
    frequency: int
    subcategory: str | None
    separator_pos: int | None
    is_bound_root: bool


@dataclass(frozen=True)
class _Item:
    """模板里的一项：字面量（可多选）或槽位。"""

    kind: str                                   # 'lit' | 'slot'
    value: str                                  # 槽位名，或字面量的展示形式
    alternatives: tuple[str, ...] = ()           # kind='lit' 时的候选表层

    def describe(self) -> str:
        if self.kind == "slot":
            return "{" + self.value + "}"
        return "/".join(self.alternatives)


@dataclass
class _SlotDef:
    """``pattern_slots`` 的一行。"""

    slot_name: str
    position: int
    is_required: bool
    syntactic_function: str | None
    default_role: str | None
    note: str | None = None

    @property
    def is_predicate(self) -> bool:
        return self.syntactic_function == "谓语"


@dataclass
class _Pattern:
    """一条句式及其全部从属数据（一次性载入内存）。"""

    pattern_id: int
    name: str
    category: str
    mood: str | None
    template: str
    semantics: str | None
    is_productive: bool
    constraints: dict[str, list[tuple[str, str, str]]]   # slot → [(kind, value, sev)]
    slots: list[_SlotDef]
    slot_by_name: dict[str, _SlotDef]
    variants: list[list[_Item]]
    literals: list[str]
    literal_count: int
    required_slots: list[str]

    def render(self) -> str:
        return f"{self.name}(#{self.pattern_id}) {self.template}"


@dataclass
class _EllipsisRule:
    rule_id: int
    context_type: str
    slot_role: str
    strategy: str
    priority: int
    max_distance: int | None
    description: str | None


@dataclass
class _Rule:
    rule_id: int
    name: str
    rule_type: str
    description: str
    condition_sql: str | None
    priority: int
    premises: list[tuple[int, str, str, str | None, str | None, bool]]
    conclusions: list[tuple[int, str, str | None, str | None, str | None, float]]


@dataclass
class _Parse:
    """枚举出的一种槽位切分方案。"""

    spans: dict[str, list[tuple[int, int]]]
    literal_hits: list[tuple[str, int]]
    skipped: list[int]
    clause_breaks: list[int]
    leading: tuple[int, int]
    trailing: tuple[int, int]
    #: 被省略的**必填**槽位名（仅当 ``ellipsis_rules`` 允许该角色省略时才可能出现）
    omitted: list[str] = field(default_factory=list)
    #: 被纳入槽位/字面量的 token 里，词典命中数 与 OOV 数
    known: int = 0
    oov: int = 0
    score: float = 0.0
    coverage: float = 0.0
    checks: list[ConstraintCheck] = field(default_factory=list)
    reason: str = ""


@dataclass
class _Mention:
    """篇章状态里的一个实体提及。"""

    text: str
    pos: str
    sense_id: int | None
    role: str | None
    utterance_index: int
    source: str = "parse"


@dataclass
class _Discourse:
    """单个文档的篇章状态（内存态，可选与 ``utterances`` 表对齐）。"""

    doc_id: str
    mentions: list[_Mention] = field(default_factory=list)
    subject: _Mention | None = None
    object: _Mention | None = None
    topic: _Mention | None = None
    count: int = 0

    def push(self, m: _Mention) -> None:
        self.mentions.append(m)
        if m.role in ("施事", "致事", "感事"):
            self.subject = m
        elif m.role in ("受事", "结果", "与事"):
            self.object = m
        elif m.role in ("话题", "主题"):
            self.topic = m


# ===========================================================================
# 5. 模板编译
# ===========================================================================
_SLOT_RE = re.compile(r"^\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}$")

#: 从 ``condition_sql`` 里读出它约束的是 ``relation_constraints.slot`` 的哪个槽位
_CONSTRAINT_SLOT_RE = re.compile(r"slot\s*=\s*'(?P<slot>[a-z_]+)'", re.IGNORECASE)


def compile_template(template: str) -> list[list[_Item]]:
    """把 ``sentence_patterns.template`` 编译成「变体 × 项序列」。

    语法（**实测自 schema 里 32 条模板的全部形态**）::

        {S} {V} {O}                 → 一个变体，3 个槽位
        {A} 把 {P} {V} {C}          → 字面量 `把` 作为锚点
        {V} 得 {C} / {V} 不 {C}     → **独立**的 `/`（两侧有空格）切分为 2 个变体
        请/别 {V}                   → **紧贴**的 `请/别` 是字面量的同义候选，非变体分隔

    判定规则：先按空白切项；切出的项若**整体等于** ``/`` 则是变体分隔符，
    否则项内部的 ``/`` 是字面量候选（如 ``请/别`` → ``('请','别')``）。
    """
    if not template or not template.strip():
        raise TemplateCompileError("句式模板为空")

    raw = template.split()
    variants: list[list[str]] = [[]]
    for it in raw:
        if it == "/":
            variants.append([])
            continue
        variants[-1].append(it)

    out: list[list[_Item]] = []
    for chunk in variants:
        if not chunk:
            continue
        items: list[_Item] = []
        for it in chunk:
            m = _SLOT_RE.match(it)
            if m:
                items.append(_Item("slot", m.group("name")))
                continue
            if "{" in it or "}" in it:
                raise TemplateCompileError(f"模板项无法解析：{it!r}（模板 {template!r}）")
            alts = tuple(a for a in it.split("/") if a)
            if not alts:
                raise TemplateCompileError(f"模板项为空：{it!r}（模板 {template!r}）")
            items.append(_Item("lit", alts[0], alts))
        if items:
            out.append(items)

    if not out:
        raise TemplateCompileError(f"模板编译后没有任何变体：{template!r}")
    return out


def _variant_text(items: Sequence[_Item]) -> str:
    return " ".join(i.describe() for i in items)


# ===========================================================================
# 6. 主类
# ===========================================================================
class SyntaxParser:
    """基于数据库模板的中文句法-语义解析器。

    :param conn: 已打开的 SQLite 连接（建议 ``read_only=True``）
    :param context: 默认解析上下文（:class:`ParserContext` 或 dict）
    :param max_slot_tokens: 单个槽位最多吃几个 token
    :param max_parses: 每条 (句式, 变体) 最多枚举多少种切分，防组合爆炸
    :param max_leading_residual: 允许的句首残余 token 数
    :param max_trailing_residual: 允许的句尾残余 token 数
    :param min_coverage: 判定"匹配成功"的最低覆盖率
    :param func_pos_prior: 是否启用「句法成分 ↔ 词性」结构先验扣分
    :param use_db_context: 是否把 ``utterances`` 表载入篇章状态作为省略回填依据
    :param persist: ``parse`` 时是否把结果写回 ``utterances``/``anaphora``

    线程安全：本类持有连接与内存缓存，**不保证多线程并发写**；只读解析可各线程各建实例。
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        context: ParserContext | Mapping[str, Any] | None = None,
        max_slot_tokens: int = DEFAULT_MAX_SLOT_TOKENS,
        max_parses: int = 256,
        max_leading_residual: int = 2,
        max_trailing_residual: int = 3,
        min_coverage: float = 0.5,
        func_pos_prior: bool = True,
        use_db_context: bool = False,
        persist: bool = False,
    ) -> None:
        if not isinstance(conn, sqlite3.Connection):
            raise ParserError(f"conn 必须是 sqlite3.Connection，收到 {type(conn).__name__}")
        self.conn = conn
        self.max_slot_tokens = max(1, int(max_slot_tokens))
        self.max_parses = max(1, int(max_parses))
        self.max_leading_residual = max(0, int(max_leading_residual))
        self.max_trailing_residual = max(0, int(max_trailing_residual))
        self.min_coverage = float(min_coverage)
        self.func_pos_prior = bool(func_pos_prior)
        self.use_db_context = bool(use_db_context)
        self.persist = bool(persist)
        self.context = ParserContext.from_any(context)

        # ---- 缓存容器（``reload`` 统一填充） ----
        self._lexicon: dict[str, _Lex] = {}
        self._grammar_literals: dict[str, _Lex] = {}
        self._max_word_len: int = 1
        self._sense_index: dict[int, dict[str, Any]] = {}
        self._patterns: list[_Pattern] = []
        self._pattern_by_id: dict[int, _Pattern] = {}
        self._pattern_by_name: dict[str, _Pattern] = {}
        self._literal_freq: dict[str, int] = {}
        self._ellipsis: dict[tuple[str, str], list[_EllipsisRule]] = {}
        self._ellipsis_roles: set[str] = set()
        self._fw_info: dict[int, dict[str, Any]] = {}
        self._fw_roles: dict[int, list[tuple[str, str]]] = {}
        self._role_meta: dict[str, dict[str, Any]] = {}
        self._relation_types: dict[str, dict[str, Any]] = {}
        self._rules: list[_Rule] = []
        self._complements: dict[tuple[int, int, str], str] = {}
        self._comp_pairs: dict[str, set[tuple[int, int]]] = {}
        self._separable_words: dict[int, list[dict[str, Any]]] = {}
        self._bound_objects: dict[int, list[int]] = {}     # object_word_id → [host_verb_id]
        self._classifier_defaults: list[tuple[str, int, float]] = []
        self._noun_classifiers: dict[int, list[tuple[int, float, bool]]] = {}
        self._cat_parent: dict[str, str | None] = {}
        # ---- 运行期缓存 ----
        self._feature_cache: dict[tuple[int, str], str | None] = {}
        self._ancestor_cache: dict[tuple[int, str, int], list[int]] = {}
        self._hdepth_cache: dict[tuple[int, str, int], list[tuple[int, int]]] = {}
        self._discourse: dict[str, _Discourse] = {}
        #: 已尝试从库里重建过篇章状态的 doc_id（``use_db_context`` 专用，
        #: 避免每次 fill_slots 都打一次库；重建失败也记入，防反复重试）
        self._db_context_loaded: set[str] = set()
        self._stats: dict[str, Any] = {
            "fallback_used": 0, "oov_tokens": 0, "patterns_loaded": 0,
            "match_calls": 0, "parses_enumerated": 0, "db_queries": 0,
        }
        self.reload()

    # =====================================================================
    # 6.1 载入缓存
    # =====================================================================
    def reload(self) -> dict[str, int]:
        """（重新）从数据库载入全部模板与词典缓存。

        :returns: 各缓存项的条目数，便于自检
        """
        self._load_lexicon()
        self._load_patterns()
        self._load_function_words()
        self._load_semantic_roles()
        self._load_ellipsis_rules()
        self._load_relation_meta()
        self._load_rules()
        self._load_misc()
        self._feature_cache.clear()
        self._ancestor_cache.clear()
        self._stats["patterns_loaded"] = len(self._patterns)
        counts = self.cache_info()
        logger.debug("解析器缓存已载入：%s", counts)
        return counts

    # ------------------------------------------------------------------
    def _q(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        """统一的只读查询入口（统计 + 元组化，屏蔽 row_factory 差异）。"""
        self._stats["db_queries"] += 1
        try:
            cur = self.conn.execute(sql, tuple(params))
        except sqlite3.Error as exc:
            raise ParserError(f"查询失败：{sql.strip()[:120]}… → {exc}") from exc
        out: list[tuple[Any, ...]] = []
        for row in cur:
            out.append(tuple(row))
        return out

    def _load_lexicon(self) -> None:
        rows = self._q(
            """
            SELECT w.word, w.word_id, s.sense_id, s.pos, s.sense_no, s.is_primary,
                   s.frequency, s.subcategory, w.separator_pos, w.is_bound_root
              FROM words w JOIN senses s ON s.word_id = w.word_id
             ORDER BY w.char_count DESC, s.is_primary DESC, s.frequency DESC
            """
        )
        lex: dict[str, _Lex] = {}
        sense_index: dict[int, dict[str, Any]] = {}
        maxlen = 1
        for (word, wid, sid, pos, sno, primary, freq, subcat, seppos, bound) in rows:
            e = _Lex(str(word), int(wid), int(sid), str(pos), int(sno or 1),
                     bool(primary), int(freq or 0), subcat,
                     None if seppos is None else int(seppos), bool(bound))
            prev = lex.get(e.word)
            # 同形多义项：主义项优先，其次高频，最后 sense_no 小的
            if prev is None or self._lex_better(e, prev):
                lex[e.word] = e
            sense_index[int(sid)] = {
                "word": str(word), "word_id": int(wid), "pos": str(pos),
                "sense_no": int(sno or 1), "subcategory": subcat,
                "frequency": int(freq or 0), "is_primary": bool(primary),
            }
            maxlen = max(maxlen, len(str(word)))
        self._lexicon = lex
        self._sense_index = sense_index
        self._max_word_len = maxlen

    @staticmethod
    def _lex_better(a: _Lex, b: _Lex) -> bool:
        if a.is_primary != b.is_primary:
            return a.is_primary
        if a.frequency != b.frequency:
            return a.frequency > b.frequency
        return a.sense_no < b.sense_no

    # ------------------------------------------------------------------
    def _load_patterns(self) -> None:
        prows = self._q(
            """
            SELECT pattern_id, name, category, mood, template, semantics,
                   is_productive, constraints_json
              FROM sentence_patterns
             WHERE is_productive = 1
             ORDER BY pattern_id
            """
        )
        srows = self._q(
            """
            SELECT pattern_id, slot_name, position, is_required,
                   syntactic_function, default_role, note
              FROM pattern_slots ORDER BY pattern_id, position
            """
        )
        crows = self._q(
            "SELECT pattern_id, slot_name, kind, value, is_exclusive FROM slot_constraints"
        )

        slots_by: dict[int, list[_SlotDef]] = {}
        for (pid, name, pos_, req, fn, role, note) in srows:
            slots_by.setdefault(int(pid), []).append(
                _SlotDef(str(name), int(pos_), bool(req), fn, role, note)
            )

        cons_by: dict[tuple[int, str], list[tuple[str, str, str]]] = {}
        for (pid, slot, kind, value, excl) in crows:
            sev = "hard" if str(kind) in HARD_CONSTRAINT_KINDS else "soft"
            if excl:
                sev = "hard"
            cons_by.setdefault((int(pid), str(slot)), []).append((str(kind), str(value), sev))

        literal_freq: dict[str, int] = {}
        patterns: list[_Pattern] = []
        for (pid, name, cat, mood, tmpl, sem, productive, _cj) in prows:
            pid = int(pid)
            try:
                variants = compile_template(str(tmpl))
            except TemplateCompileError as exc:
                # 单条模板坏掉不该拖垮整个解析器：跳过并留痕
                logger.warning("跳过无法编译的句式 %s(#%s)：%s", name, pid, exc)
                continue
            lits = sorted({alt for v in variants for i in v if i.kind == "lit"
                           for alt in i.alternatives})
            for lit in lits:
                literal_freq[lit] = literal_freq.get(lit, 0) + 1
            slots = slots_by.get(pid, [])
            slot_by_name = {s.slot_name: s for s in slots}
            constraints: dict[str, list[tuple[str, str, str]]] = {}
            for (_, sname), lst in cons_by.items():
                if _ == pid:
                    constraints[sname] = lst
            patterns.append(
                _Pattern(
                    pattern_id=pid, name=str(name), category=str(cat), mood=mood,
                    template=str(tmpl), semantics=sem, is_productive=bool(productive),
                    constraints=constraints, slots=slots, slot_by_name=slot_by_name,
                    variants=variants, literals=lits, literal_count=len(lits),
                    required_slots=[s.slot_name for s in slots if s.is_required],
                )
            )
        self._patterns = patterns
        self._pattern_by_id = {p.pattern_id: p for p in patterns}
        self._pattern_by_name = {p.name: p for p in patterns}
        self._literal_freq = literal_freq

        # 把多字字面量注入词典，避免 `越来越/一样/不如/难道` 被逐字切碎。
        # 只注入词典里没有的形态，绝不覆盖真实词条的 sense_id。
        gram: dict[str, _Lex] = {}
        for lit in literal_freq:
            if len(lit) < 2 or lit in self._lexicon or lit in gram:
                continue
            gram[lit] = _Lex(lit, -1, -1, "u", 1, False, 0, None, None, False)
        self._grammar_literals = gram

    # ------------------------------------------------------------------
    def _load_function_words(self) -> None:
        fw: dict[int, dict[str, Any]] = {}
        for (sid, cat, sub, fn, attaches, sel, final) in self._q(
            """SELECT sense_id, category, subcategory, grammatical_function,
                      attaches_to, selects_verb_subcat, is_sentence_final
                 FROM function_words"""
        ):
            fw[int(sid)] = {
                "category": str(cat), "subcategory": str(sub),
                "grammatical_function": fn, "attaches_to": attaches,
                "selects_verb_subcat": sel, "is_sentence_final": bool(final),
                "word": self._sense_index.get(int(sid), {}).get("word", ""),
            }
        self._fw_info = fw

        roles: dict[int, list[tuple[str, str]]] = {}
        for (sid, role, pos_) in self._q(
            "SELECT sense_id, role, position FROM function_word_roles"
        ):
            roles.setdefault(int(sid), []).append((str(role), str(pos_)))
        self._fw_roles = roles

    # ------------------------------------------------------------------
    def _load_semantic_roles(self) -> None:
        meta: dict[str, dict[str, Any]] = {}
        for (role, parent, group, desc, typ, marker) in self._q(
            """SELECT role, parent_role, role_group, description, typical_pos, case_marker
                 FROM semantic_roles"""
        ):
            meta[str(role)] = {
                "parent": parent, "group": str(group), "description": str(desc),
                "typical_pos": typ, "case_marker": marker or "",
            }
        self._role_meta = meta

    # ------------------------------------------------------------------
    def _load_ellipsis_rules(self) -> None:
        idx: dict[tuple[str, str], list[_EllipsisRule]] = {}
        for (rid, ctx, role, strat, prio, dist, desc) in self._q(
            """SELECT ellipsis_rule_id, context_type, slot_role, strategy,
                      priority, max_distance, description
                 FROM ellipsis_rules
                ORDER BY context_type, slot_role, priority DESC"""
        ):
            r = _EllipsisRule(int(rid), str(ctx), str(role), str(strat), int(prio),
                              None if dist is None else int(dist), desc)
            idx.setdefault((r.context_type, r.slot_role), []).append(r)
        self._ellipsis = idx
        #: 有省略规则的语义角色 —— 决定一个**必填**槽位是否允许被合法省略。
        #: 这是「汉语允许主语脱落」这件事在数据库里的唯一声明处。
        self._ellipsis_roles = {role for (_ctx, role) in idx}

    # ------------------------------------------------------------------
    def _load_relation_meta(self) -> None:
        rt: dict[str, dict[str, Any]] = {}
        for (rel, cls, direction, inv, trans, sym, refl, func, depth, desc, color) in self._q(
            """SELECT relation, relation_class, direction, inverse_of, is_transitive,
                      is_symmetric, is_reflexive, is_functional, max_depth,
                      description, color_code
                 FROM relation_types"""
        ):
            rt[str(rel)] = {
                "class": str(cls), "direction": str(direction), "inverse_of": inv,
                "is_transitive": bool(trans), "is_symmetric": bool(sym),
                "is_reflexive": bool(refl), "is_functional": bool(func),
                "max_depth": int(depth), "description": str(desc), "color": color,
            }
        self._relation_types = rt

    # ------------------------------------------------------------------
    def _load_rules(self) -> None:
        base: dict[int, dict[str, Any]] = {}
        for (rid, name, rtype, desc, cond, action, prio, enabled, ver) in self._q(
            """SELECT rule_id, name, rule_type, description, condition_sql, action_sql,
                      priority, is_enabled, version
                 FROM inference_rules ORDER BY priority DESC, rule_id"""
        ):
            base[int(rid)] = {
                "name": str(name), "type": str(rtype), "description": str(desc),
                "condition_sql": cond, "priority": int(prio), "enabled": bool(enabled),
                "premises": [], "conclusions": [],
            }
        for (rid, seq, kind, subj, pred, obj, neg) in self._q(
            """SELECT rule_id, seq, kind, subject, predicate, object, is_negated
                 FROM rule_premises ORDER BY rule_id, seq"""
        ):
            if int(rid) in base:
                base[int(rid)]["premises"].append(
                    (int(seq), str(kind), str(subj), pred, obj, bool(neg))
                )
        for (rid, seq, kind, subj, rel, obj, conf) in self._q(
            """SELECT rule_id, seq, kind, subject, relation, object, confidence
                 FROM rule_conclusions ORDER BY rule_id, seq"""
        ):
            if int(rid) in base:
                base[int(rid)]["conclusions"].append(
                    (int(seq), str(kind), subj, rel, obj, float(conf))
                )
        self._rules = [
            _Rule(rid, v["name"], v["type"], v["description"], v["condition_sql"],
                  v["priority"], v["premises"], v["conclusions"])
            for rid, v in sorted(base.items()) if v["enabled"]
        ]

    # ------------------------------------------------------------------
    def _load_misc(self) -> None:
        comp: dict[tuple[int, int, str], str] = {}
        pairs: dict[str, set[tuple[int, int]]] = {}
        for (ctype, head, csense, state, order, ex) in self._q(
            """SELECT comp_type, head_sense_id, comp_sense_id, result_state,
                      surface_order, example
                 FROM complements"""
        ):
            comp[(int(head), int(csense), str(order))] = str(ctype)
            comp[(int(head), int(csense), "*")] = str(ctype)
            pairs.setdefault(str(ctype), set()).add((int(head), int(csense)))
        self._complements = comp
        #: comp_type → 已登记的 (述语义项, 补语义项) 集合。
        #: 短语结构模板（动结/动趋/可能补语）用它做**硬校验**，防止 `{V} {R}`
        #: 把 `猫吃鱼` 解释成「猫吃 + 鱼」。
        self._comp_pairs = pairs

        sep: dict[int, list[dict[str, Any]]] = {}
        for (wid, kind, infix, surface, note) in self._q(
            """SELECT word_id, kind, infixes_json, surface_example, note
                 FROM separable_patterns ORDER BY word_id, sep_pattern_id"""
        ):
            sep.setdefault(int(wid), []).append(
                {"kind": str(kind), "infixes": infix, "example": str(surface), "note": note}
            )
        self._separable_words = sep

        bound: dict[int, list[int]] = {}
        for (obj, host) in self._q(
            "SELECT object_word_id, host_verb_id FROM bound_objects"
        ):
            bound.setdefault(int(obj), []).append(int(host))
        self._bound_objects = bound

        cls_defaults: list[tuple[str, int, float]] = []
        for (cat, csid, pref) in self._q(
            """SELECT cat, classifier_sense_id, preference
                 FROM classifier_class_defaults ORDER BY preference DESC"""
        ):
            cls_defaults.append((str(cat), int(csid), float(pref)))
        self._classifier_defaults = cls_defaults

        nc: dict[int, list[tuple[int, float, bool]]] = {}
        for (noun, cls_, pref, is_def) in self._q(
            "SELECT noun_sense_id, classifier_sense_id, preference, is_default "
            "FROM noun_classifier_pairs ORDER BY preference DESC"
        ):
            nc.setdefault(int(noun), []).append((int(cls_), float(pref), bool(is_def)))
        self._noun_classifiers = nc

        parent: dict[str, str | None] = {}
        for (cat, pcat) in self._q("SELECT cat, parent_cat FROM semantic_categories"):
            parent[str(cat)] = None if pcat is None else str(pcat)
        self._cat_parent = parent

    # ------------------------------------------------------------------
    def cache_info(self) -> dict[str, int]:
        """各缓存条目数（用于自检与文档）。"""
        return {
            "lexicon": len(self._lexicon),
            "grammar_literals": len(self._grammar_literals),
            "senses": len(self._sense_index),
            "patterns": len(self._patterns),
            "pattern_slots": sum(len(p.slots) for p in self._patterns),
            "slot_constraints": sum(len(c) for p in self._patterns
                                    for c in p.constraints.values()),
            "literal_freq": len(self._literal_freq),
            "function_words": len(self._fw_info),
            "function_word_roles": sum(len(v) for v in self._fw_roles.values()),
            "semantic_roles": len(self._role_meta),
            "ellipsis_rules": sum(len(v) for v in self._ellipsis.values()),
            "relation_types": len(self._relation_types),
            "inference_rules": len(self._rules),
            "complements": len(self._complements),
            "separables": len(self._separable_words),
            "bound_objects": sum(len(v) for v in self._bound_objects.values()),
            "max_word_len": self._max_word_len,
        }

    # =====================================================================
    # 6.2 分词
    # =====================================================================
    def tokenize(self, text: str) -> list[Token]:
        """正向最大匹配分词（v0.1：纯词典，不接模型）。

        :param text: 待切分文本（UTF-8，可含中英混排）
        :returns: ``list[Token]``，标点也产出 token（``pos='w'``）

        :raises TokenizeError: 输入非法或词典为空

        流程::

            1. 全角→半角规范化（仅数字/字母，标点保持原形以保留句读信息）
            2. 正向最大匹配（词典 + 句式模板里的多字标记词）
            3. 离合词切分：`洗澡`（separator_pos=1）→ `洗` | `澡`
            4. 离合结构重组：`洗 了 个 澡` → 依 `bound_objects` 标记为同一组
            5. OOV 兜底：数词/指示代词/方位后缀/封闭类虚词表 → 否则 pos='?'

        .. note::
            「离合词」两条路径都覆盖：整体出现时**切分**（步骤 3），
            拆开出现时**重组**（步骤 4）——否则 `洗了个澡` 会退化成 4 个无关 token。
        """
        if not isinstance(text, str):
            raise TokenizeError(f"text 必须是 str，收到 {type(text).__name__}")
        if not self._lexicon and not self._grammar_literals:
            raise TokenizeError("词典为空：请先确认 words/senses 表已填充")

        norm = self._normalize(text)
        tokens = self._fmm(norm)
        tokens = self._detect_separable_groups(tokens)
        return tokens

    # ------------------------------------------------------------------
    _FW2HW = {ord("０") + i: ord("0") + i for i in range(10)}
    _FW2HW.update({ord("Ａ") + i: ord("A") + i for i in range(26)})
    _FW2HW.update({ord("ａ") + i: ord("a") + i for i in range(26)})
    _FW2HW[ord("\u3000")] = ord(" ")

    @classmethod
    def _normalize(cls, text: str) -> str:
        """只把全角数字/字母/空格折成半角，**不动标点**（句读信息必须保留）。"""
        return text.translate(cls._FW2HW)

    # ------------------------------------------------------------------
    def _fmm(self, text: str) -> list[Token]:
        """正向最大匹配主循环。"""
        n = len(text)
        out: list[Token] = []
        i = 0
        group_seq = 0
        while i < n:
            ch = text[i]
            # ① 空白直接跳过（不产生 token）
            if ch.isspace():
                i += 1
                continue
            # ② 标点
            if ch in _PUNCT_CHARS:
                out.append(Token(ch, PUNCT_POS, start=i, end=i + 1))
                i += 1
                continue
            # ③ 正向最大匹配（词典优先，其次模板标记词）
            hit: _Lex | None = None
            span = 1
            for L in range(min(self._max_word_len, n - i), 0, -1):
                sub = text[i:i + L]
                lex = self._lexicon.get(sub)
                if lex is None:
                    lex = self._grammar_literals.get(sub)
                if lex is not None:
                    hit, span = lex, L
                    break
            if hit is None:
                # 先把连续同类字符并成一个 token（数字串 `2026`、方位串 `上边`），
                # 否则每个字符各自走单字兜底，`2026` 会被切成 4 个 `?`，
                # `_guess_pos` 里的"数词"规则对阿拉伯数字也永远不生效。
                run = self._oov_run(text, i)
                if run is not None:
                    tok, nxt = run
                    out.append(tok)
                    i = nxt
                    continue
                out.append(self._oov_token(ch, i))
                i += 1
                continue
            sub = text[i:i + span]
            # ④ 离合词切分
            if hit.separator_pos is not None and 1 <= hit.separator_pos < len(sub):
                k = hit.separator_pos
                group_seq += 1
                parts = [(sub[:k], "host"), (sub[k:], "object")]
                # 多于两段时（分离点靠后）继续按字符补足
                for ptext, prole in parts:
                    out.append(self._make_part_token(
                        ptext, i, hit, prole, k, group_seq))
                    i += len(ptext)
                continue
            out.append(Token(
                text=sub, pos=hit.pos, entry_id=hit.word_id, sense_id=hit.sense_id,
                start=i, end=i + span, is_separable=False,
                separator_pos=hit.separator_pos, is_bound_root=hit.is_bound_root,
                is_oov=False, is_marker_literal=hit.word_id < 0,
                subcategory=hit.subcategory,
                fw_category=self._fw_info.get(hit.sense_id, {}).get("category"),
                frequency=hit.frequency, is_primary=hit.is_primary,
            ))
            i += span
        return out

    def _make_part_token(
        self, ptext: str, start: int, whole: _Lex, role: str, sep: int, group: int
    ) -> Token:
        """构造离合词切分后的单个成分 token（尽量解析出成分自身的义项）。"""
        own = self._lexicon.get(ptext)
        if own is not None:
            sense_id, entry_id, pos = own.sense_id, own.word_id, own.pos
            subcat = own.subcategory
        else:
            sense_id, entry_id = whole.sense_id, whole.word_id
            pos = whole.pos
            subcat = whole.subcategory
        return Token(
            text=ptext, pos=pos, entry_id=entry_id, sense_id=sense_id,
            start=start, end=start + len(ptext), is_separable=True,
            separator_pos=sep, is_bound_root=bool(own.is_bound_root) if own else False,
            is_oov=own is None,
            separable_group=group, sep_role=role,
            sep_group_sense_id=whole.sense_id,
            subcategory=subcat,
            fw_category=self._fw_info.get(sense_id, {}).get("category"),
            frequency=own.frequency if own else whole.frequency,
            is_primary=own.is_primary if own else whole.is_primary,
        )

    # ------------------------------------------------------------------
    def _oov_token(self, ch: str, start: int) -> Token:
        """词典未收录 → 兜底推断词性。"""
        pos, how = self._guess_pos(ch)
        self._stats["oov_tokens"] += 1
        if how != "unknown":
            self._stats["fallback_used"] += 1
        return Token(
            text=ch, pos=pos, entry_id=None, sense_id=None,
            start=start, end=start + len(ch), is_oov=True, inferred_pos=how,
        )

    def _oov_run(self, text: str, i: int) -> tuple[Token, int] | None:
        """把位置 ``i`` 起的一段**连续同质**字符并成一个 OOV token。

        只处理有明确词性归属的两类连续字符：阿拉伯数字串（数词）、方位后缀字串
        （方位词）。:returns: ``(token, 下一个位置)``；不适用时返回 ``None``，
        由调用方退回 :meth:`_oov_token` 的单字兜底。
        """
        ch = text[i]
        if ch.isdigit():
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
        elif ch in _LOC_SUFFIX:
            j = i + 1
            while j < len(text) and text[j] in _LOC_SUFFIX:
                j += 1
            if j == i + 1:
                return None     # 单字方位：单字兜底同样得到 'nl'，不必特殊处理
        else:
            return None
        surface = text[i:j]
        pos, how = self._guess_pos(surface)
        self._stats["oov_tokens"] += 1
        self._stats["fallback_used"] += 1
        return Token(
            text=surface, pos=pos, entry_id=None, sense_id=None,
            start=i, end=j, is_oov=True, inferred_pos=how,
        ), j

    @staticmethod
    def _guess_pos(word: str) -> tuple[str, str]:
        """OOV 词性兜底：封闭类词表 → 数词 → 指示代词 → 方位后缀 → ``'?'``。"""
        if word in _CLOSED_CLASS_FALLBACK:
            return _CLOSED_CLASS_FALLBACK[word], "closed_class"
        # 全部字符都是数词字（含阿拉伯数字）→ 数词。`三个` 这类混合串不算。
        if word and all(c in _CN_NUMERALS or c.isdigit() for c in word):
            return "m", "numeral"
        if word in _DEMONSTRATIVES or word in _DETERMINERS:
            return "r", "demonstrative"
        if word and word[-1] in _LOC_SUFFIX and len(word) <= 3:
            return "nl", "locative_suffix"
        return "?", "unknown"

    # ------------------------------------------------------------------
    def _detect_separable_groups(self, tokens: list[Token]) -> list[Token]:
        """把「拆开出现的离合结构」重新标记成一组（``洗 了 个 澡``）。"""
        if not self._bound_objects:
            return tokens
        n = len(tokens)
        out = list(tokens)
        group_seq = max((t.separable_group or 0) for t in out) if out else 0
        window = 6
        for i, t in enumerate(out):
            if t.separable_group is not None:
                continue
            hosts = self._bound_objects.get(t.entry_id) if t.entry_id is not None else None
            if not hosts:
                continue
            # 向后找施与动词（澡洗了 / 洗了澡）
            cand: tuple[int, int, str] | None = None
            for j in range(i + 1, min(n, i + window + 1)):
                if out[j].pos == PUNCT_POS:
                    break
                if out[j].entry_id in hosts:
                    cand = (i, j, "backward")
                    break
            if cand is None:
                for j in range(i - 1, max(-1, i - window - 1), -1):
                    if out[j].pos == PUNCT_POS:
                        break
                    if out[j].entry_id in hosts:
                        cand = (i, j, "forward")
                        break
            if cand is None:
                continue
            obj_i, host_i, _dir = cand
            group_seq += 1
            lo, hi = min(obj_i, host_i), max(obj_i, host_i)
            kinds = self._separable_words.get(out[host_i].entry_id or -1, [])
            note = kinds[0]["kind"] if kinds else None
            for k in range(lo, hi + 1):
                tk = out[k]
                out[k] = Token(
                    text=tk.text, pos=tk.pos, entry_id=tk.entry_id, sense_id=tk.sense_id,
                    start=tk.start, end=tk.end, is_separable=True,
                    separator_pos=tk.separator_pos, is_bound_root=tk.is_bound_root,
                    is_oov=tk.is_oov, is_marker_literal=tk.is_marker_literal,
                    separable_group=group_seq,
                    sep_role="host" if k == host_i else ("object" if k == obj_i else "infix"),
                    sep_group_sense_id=tk.sep_group_sense_id,
                    subcategory=tk.subcategory, fw_category=tk.fw_category,
                    frequency=tk.frequency, is_primary=tk.is_primary,
                    inferred_pos=tk.inferred_pos,
                )
            if note:
                logger.debug("离合结构 %s：%s", "".join(
                    out[k].text for k in range(lo, hi + 1)), note)
        return out

    # =====================================================================
    # 6.3 句式匹配
    # =====================================================================
    def match_pattern(self, tokens: Sequence[Token]) -> PatternMatch | None:
        """在全部活跃句式中挑出**得分最高**的那个匹配。

        :param tokens: :meth:`tokenize` 的输出（可含句末标点）
        :returns: :class:`PatternMatch`；无任何模板可匹配时返回 ``None``

        匹配策略（对应任务里的 a/b/c 三步）::

            a. 检测关键虚词标记 —— 模板字面量（把/被/比/跟/难道/越来越…）
               按 IDF 加权计分，字面量一条都没命中的模板直接判负
            b. 按槽位顺序尝试填充 —— 游标式回溯枚举，槽位区间必须首尾皆为实词、
               长度 ≤ max_slot_tokens、不跨句末标点
            c. 用 slot_constraints 验证每个填充 —— 软约束扣分并点名，
               kind='marker' 为硬约束（缺失即拒绝）

        每次调用都会刷新 :attr:`last_attempts`（全部句式的尝试与失败原因）。
        """
        sentences = self._split_sentences(list(tokens))
        self._stats["match_calls"] += 1
        self._stats["parses_enumerated"] = 0

        best: PatternMatch | None = None
        attempts: list[PatternAttempt] = []

        for s_idx, (body, term) in enumerate(sentences):
            if not body:
                continue
            for pat in self._patterns:
                cand, reason, errs = self._try_pattern(pat, body, s_idx, term or "")
                if cand is not None and (best is None or cand.score > best.score):
                    best = cand
                attempts.append(PatternAttempt(
                    pattern_id=pat.pattern_id, name=pat.name, category=pat.category,
                    matched=cand is not None,
                    score=cand.score if cand else 0.0,
                    coverage=cand.coverage if cand else 0.0,
                    reason=reason,
                    constraint_errors=errs,
                    slot_spans=cand.slot_spans if cand else {},
                ))

        attempts.sort(key=lambda a: (not a.matched, -a.score))
        self.last_attempts = attempts
        if best is not None:
            logger.debug("匹配 %s：score=%.2f coverage=%.0f%%",
                         best.name, best.score, best.coverage * 100)
        return best

    # ------------------------------------------------------------------
    def match_candidates(
        self, tokens: Sequence[Token], limit: int = 3, *, include_failed: bool = True
    ) -> list[PatternAttempt]:
        """取 top-N 候选（含失败原因），用于「匹配失败 → top-3 + 原因」。

        :param include_failed: 为 ``False`` 时只返回成功命中的候选
        """
        self.match_pattern(tokens)
        items = getattr(self, "last_attempts", [])
        if not include_failed:
            items = [a for a in items if a.matched]
        return items[:max(1, limit)]

    # ------------------------------------------------------------------
    def _split_sentences(
        self, tokens: list[Token]
    ) -> list[tuple[list[Token], str]]:
        """按句末标点切句，返回 ``[(句体 token, 终止符), …]``（句体保留句中标点）。"""
        out: list[tuple[list[Token], str]] = []
        cur: list[Token] = []
        term = ""
        for t in tokens:
            if t.is_sentence_punct:
                body = [x for x in cur if not x.is_sentence_punct]
                out.append((body, term or t.text))
                cur, term = [], ""
                continue
            cur.append(t)
        if cur:
            body = [x for x in cur if not x.is_sentence_punct]
            out.append((body, term))
        if not out:
            out.append(([], ""))
        return out

    # ------------------------------------------------------------------
    def _try_pattern(
        self, pat: _Pattern, body: list[Token], s_idx: int, term: str
    ) -> tuple[PatternMatch | None, str, list[str]]:
        """尝试单个句式的全部变体，返回 (最佳匹配, 原因, 约束冲突)。"""
        best_parse: _Parse | None = None
        fail_reason = "无可填充的槽位切分"
        hard_errs: list[str] = []

        for items in pat.variants:
            for parse in self._enumerate_variant(pat, items, body):
                self._stats["parses_enumerated"] += 1
                ok, reason, errs = self._validate_parse(pat, parse, body)
                if not ok:
                    fail_reason = reason
                    if errs:
                        hard_errs = errs
                    continue
                if best_parse is None or parse.score > best_parse.score:
                    best_parse = parse
                    fail_reason = ""

        if best_parse is None:
            return None, fail_reason, hard_errs

        items_best = None
        for items in pat.variants:
            if _variant_text(items) == best_parse.reason:
                items_best = items
                break

        match = PatternMatch(
            pattern_id=pat.pattern_id, name=pat.name, category=pat.category,
            mood=pat.mood, template=pat.template, semantics=pat.semantics,
            variant=best_parse.reason, score=best_parse.score,
            coverage=best_parse.coverage, slot_spans=best_parse.spans,
            literal_hits=best_parse.literal_hits,
            skipped_markers=best_parse.skipped,
            leading_residual=best_parse.leading,
            trailing_residual=best_parse.trailing,
            clause_breaks=best_parse.clause_breaks,
            constraint_checks=best_parse.checks,
            omitted_slots=list(best_parse.omitted),
            sentence_span=(0, len(body)), sentence_terminator=term,
        )
        if match.omitted_slots:
            match.notes.append(
                "以下必填槽位按 ellipsis_rules 授权省略（待省略回填）："
                + "、".join(match.omitted_slots)
            )
        if match.trailing_residual[1] > match.trailing_residual[0]:
            match.notes.append(
                "句尾存在未纳入槽位的成分："
                + "".join(t.text for t in body[match.trailing_residual[0]:match.trailing_residual[1]])
            )
        if match.leading_residual[1] > match.leading_residual[0]:
            match.notes.append(
                "句首存在未纳入槽位的成分："
                + "".join(t.text for t in body[match.leading_residual[0]:match.leading_residual[1]])
            )
        # 透明化"槽位吞并"：模板写得比句子粗时，某个槽位会连吃多个 token。
        # 这不算错，但必须让调用方看得见，而不是悄悄把整句塞进一个槽位。
        for name, spans in match.slot_spans.items():
            for (s, e) in spans:
                if e - s >= 3:
                    match.notes.append(
                        f"槽位 {{{name}}} 吞并了 {e - s} 个 token："
                        + "".join(t.text for t in body[s:e]) + "（模板未单独建模该成分）"
                    )
        return match, "", [c.render() for c in match.constraint_checks if not c.ok]

    # =====================================================================
    # 6.4 枚举与校验（匹配策略 b / c 的实现）
    # =====================================================================
    def _enumerate_variant(
        self, pat: _Pattern, items: list[_Item], tokens: list[Token]
    ) -> list[_Parse]:
        """回溯枚举一个变体下的全部合法槽位切分（上限 :attr:`max_parses`）。

        枚举顺序 = **优先最短槽位区间**（``for end in range(j+1, …)``），
        因此先出现的方案是"最保守"的读法；再由 :meth:`_score` 择优。
        """
        n = len(tokens)
        results: list[_Parse] = []
        if n == 0:
            return results

        starts: list[int] = []
        for lead in range(0, min(self.max_leading_residual, n) + 1):
            if lead > 0:
                tk = tokens[lead - 1]
                if not (tk.is_content or tk.is_clause_punct):
                    break
            starts.append(lead)

        variant_label = _variant_text(items)

        def rec(
            idx: int, pos: int,
            spans: dict[str, list[tuple[int, int]]],
            hits: list[tuple[str, int]],
            skipped: list[int], breaks: list[int], lead: int,
            omitted: list[str],
        ) -> None:
            if len(results) >= self.max_parses:
                return
            if idx == len(items):
                tail = self._consume_tail(tokens, pos)
                if tail is None:
                    return
                t_skip, t_res, t_br = tail
                results.append(_Parse(
                    spans={k: list(v) for k, v in spans.items()},
                    literal_hits=list(hits),
                    skipped=sorted(set(skipped) | set(t_skip)),
                    clause_breaks=sorted(set(breaks) | set(t_br)),
                    leading=(0, lead),
                    trailing=t_res,
                    omitted=list(omitted),
                    reason=variant_label,
                ))
                return

            item = items[idx]
            if item.kind == "lit":
                for alt in item.alternatives:
                    found = self._find_literal(tokens, pos, alt)
                    if found is None:
                        continue
                    k, sk, br = found
                    rec(idx + 1, k + 1, spans, hits + [(alt, k)],
                        skipped + sk, breaks + br, lead, omitted)
                return

            # ---- 槽位：先吃掉前导可跳过成分，再枚举区间 ----
            j = pos
            lsk: list[int] = []
            lbr: list[int] = []
            while j < n and self._skippable(tokens[j]):
                lsk.append(j)
                if tokens[j].is_clause_punct:
                    lbr.append(j)
                j += 1

            # ① 先说"留空"这条分支，再枚举"填上"的分支。
            #    可选槽位（如把字句的 C）留空是常态；
            #    必填槽位只有在 ellipsis_rules 声明了该角色可省略时才允许留空
            #    —— 这是「汉语允许主语脱落」在数据库里的唯一声明处。
            if self._may_omit(pat, item.value):
                extra = [item.value] if self._is_required(pat, item.value) else []
                rec(idx + 1, pos, spans, hits, skipped, breaks, lead, omitted + extra)

            if j >= n:
                return
            maxend = min(n, j + self.max_slot_tokens)
            for end in range(j + 1, maxend + 1):
                span = tokens[j:end]
                if not self._span_ok(span):
                    if tokens[j].is_punct:
                        break
                    continue
                nxt = {k: list(v) for k, v in spans.items()}
                nxt.setdefault(item.value, []).append((j, end))
                rec(idx + 1, end, nxt, hits, skipped + lsk, breaks + lbr, lead, omitted)

        for lead in starts:
            rec(0, lead, {}, [], [], [], lead, [])
        return results

    def _is_required(self, pat: _Pattern, slot_name: str) -> bool:
        sd = pat.slot_by_name.get(slot_name)
        return True if sd is None else sd.is_required

    def _may_omit(self, pat: _Pattern, slot_name: str) -> bool:
        """该槽位是否允许留空。

        * 可选槽位（``pattern_slots.is_required = 0``）→ 允许
        * 必填槽位 → 仅当其语义角色在 ``ellipsis_rules`` 里有省略规则时允许
        """
        sd = pat.slot_by_name.get(slot_name)
        if sd is None:
            # 模板引用了 pattern_slots 未登记的槽位 → 不猜，直接要求填满
            return False
        if not sd.is_required:
            return True
        role = sd.default_role or self._fallback_role(sd)
        return role in self._ellipsis_roles

    # ------------------------------------------------------------------
    def _skippable(self, t: Token) -> bool:
        """该 token 在匹配时是否可被自由跳过（助词/语气词/副词/句中标点）。"""
        return t.is_skippable or t.is_clause_punct

    def _span_ok(self, span: list[Token]) -> bool:
        """槽位区间合法性：首尾皆实词、不含标点、长度受限。"""
        if not span or len(span) > self.max_slot_tokens:
            return False
        if not span[0].is_content or not span[-1].is_content:
            return False
        return all(not t.is_punct for t in span)

    def _find_literal(
        self, tokens: list[Token], pos: int, lit: str, max_skip: int = 3
    ) -> tuple[int, list[int], list[int]] | None:
        """从 ``pos`` 起找字面量 ``lit``：**先试直接命中，再逐个跳过可跳过成分**。

        之所以先试直接命中，是为了让「就/都/越」这类既是标记又是副词的字符
        优先被模板消费，而不是先被跳掉导致模板空转。
        """
        n = len(tokens)
        k = pos
        sk: list[int] = []
        br: list[int] = []
        while k < n:
            if tokens[k].text == lit:
                return k, sk, br
            t = tokens[k]
            if not self._skippable(t):
                return None
            sk.append(k)
            if t.is_clause_punct:
                br.append(k)
            k += 1
            if len(sk) > max_skip:
                return None
        return None

    def _consume_tail(
        self, tokens: list[Token], pos: int
    ) -> tuple[list[int], tuple[int, int], list[int]] | None:
        """吃完句尾：可跳过成分照单全收，残余成分数量受 ``max_trailing_residual`` 限制。"""
        n = len(tokens)
        j = pos
        sk: list[int] = []
        br: list[int] = []
        res: list[int] = []
        while j < n:
            t = tokens[j]
            if self._skippable(t):
                sk.append(j)
                if t.is_clause_punct:
                    br.append(j)
                j += 1
                continue
            if t.is_punct:
                return None
            if len(res) < self.max_trailing_residual:
                res.append(j)
                j += 1
                continue
            return None
        span = (res[0], res[-1] + 1) if res else (pos, pos)
        return sk, span, br

    # =====================================================================
    # 6.5 校验与评分
    # =====================================================================
    def _validate_parse(
        self, pat: _Pattern, parse: _Parse, tokens: list[Token]
    ) -> tuple[bool, str, list[str]]:
        """七道闸门 + 打分。

        闸门顺序：必填槽位 → 同形约束 → 关键虚词标记 → 短语结构残余/登记
        → 词汇证据 → 硬约束 → 覆盖率。
        """
        # ① 必填槽位（省略必须被显式登记在 parse.omitted 里才算数）
        for name in pat.required_slots:
            if not parse.spans.get(name) and name not in parse.omitted:
                return False, f"必填槽位 {{{name}}} 未填充", []
        if len(parse.omitted) > _MAX_OMITTED_SLOTS:
            return (False,
                    f"一次省略了 {len(parse.omitted)} 个必填槽位（{'/'.join(parse.omitted)}）"
                    f"→ 模板未被有效实例化", [])
        # ② 同名槽位多次出现 → 必须同形（如 `{V} 不 {V}` 的"吃不吃"）
        for name, spans in parse.spans.items():
            if len(spans) > 1:
                texts = {"".join(t.text for t in tokens[s:e]) for (s, e) in spans}
                if len(texts) > 1:
                    return False, f"槽位 {{{name}}} 多次出现但形式不同：{'/'.join(sorted(texts))}", []
        # ③ 关键虚词标记闸门：模板写了字面量，就至少要命中一个
        if pat.literal_count > 0 and not parse.literal_hits:
            return False, "未命中任何关键虚词标记（模板要求至少一个）", []
        # ④ 短语结构模板（动结/动趋/可能补语）两道闸门：
        #    不接受句级残余成分；且 V+C 组合必须在 complements 表里登记过。
        if pat.category in _PHRASE_CATEGORIES:
            lead = parse.leading[1] - parse.leading[0]
            trail = parse.trailing[1] - parse.trailing[0]
            if lead or trail:
                return (False,
                        f"短语结构模板不接受句级残余成分（句首 {lead} / 句尾 {trail} 个）", [])
            ctype = self._comp_type_of(pat)
            registered = self._comp_pairs.get(ctype or "") if ctype else None
            if registered:
                got = self._comp_pair(pat, parse, tokens)
                if got is None:
                    return False, f"{ctype}模板：V/C 槽位中心语缺义项，无法与 complements 比对", []
                if got not in registered:
                    names = "/".join(
                        f"{self._sense_text(a)}+{self._sense_text(b)}"
                        for a, b in sorted(registered)[:4]
                    )
                    return (False,
                            f"{ctype}未登记该组合（{self._sense_text(got[0])}+"
                            f"{self._sense_text(got[1])}；已登记：{names}）", [])
        # ⑤ 词汇证据闸门：整句全是 OOV 时不能"靠结构空转"匹配上模板。
        #    没有这一条，`苹果香蕉橘子` 会被 双宾句/兼语句 各自认领一次。
        #    注意：证据必须**在此刻现算**——`_score` 要到闸门 ⑦ 才执行，
        #    若直接读 `parse.known` 会拿到 dataclass 默认值 0，把 `猫吃鱼。`
        #    这类正常句子一并否决（历史回归）。
        if self._lexical_evidence(parse, tokens)[0] == 0:
            return False, ("句中没有任何词典命中的实词（全部 OOV），"
                           "拒绝靠结构空转匹配"), []
        # ⑥ 约束
        parse.checks = self._check_constraints(pat, parse, tokens)
        hard = [c.render() for c in parse.checks if not c.ok and c.severity == "hard"]
        if hard:
            return False, "硬约束未通过：" + "；".join(hard), hard
        # ⑦ 覆盖率 + 打分
        parse.score, parse.coverage = self._score(pat, parse, tokens)
        if parse.coverage < self.min_coverage:
            return False, f"覆盖率 {parse.coverage:.0%} < 阈值 {self.min_coverage:.0%}", []
        return True, "", [c.render() for c in parse.checks if not c.ok]

    # ------------------------------------------------------------------
    def _check_constraints(
        self, pat: _Pattern, parse: _Parse, tokens: list[Token]
    ) -> list[ConstraintCheck]:
        """用 ``slot_constraints`` 逐条验证槽位填充（匹配策略 c）。"""
        checks: list[ConstraintCheck] = []
        for slot_name, cons in pat.constraints.items():
            spans = parse.spans.get(slot_name) or []
            span_tokens = [tokens[i] for (s, e) in spans for i in range(s, e)]
            for (kind, value, sev) in cons:
                if not span_tokens:
                    # 可选槽位未填充 → 约束不适用，不计分也不报错
                    if kind == "marker":
                        ok = any(t.text == value for t in tokens)
                        checks.append(ConstraintCheck(
                            pat.pattern_id, slot_name, kind, value, ok, sev,
                            "标记出现在句中" if ok else "标记未出现",
                        ))
                    continue
                ok, partial, detail = self._eval_constraint(
                    kind, value, span_tokens, pat, slot_name, tokens
                )
                if partial and ok:
                    detail = detail + "（OOV 部分满足）"
                checks.append(ConstraintCheck(
                    pat.pattern_id, slot_name, kind, value, ok, sev, detail
                ))
        return checks

    def _eval_constraint(
        self, kind: str, value: str, span: list[Token],
        pat: _Pattern, slot_name: str, tokens: list[Token],
    ) -> tuple[bool, bool, str]:
        """:returns: ``(通过, 是否部分满足, 说明)``"""
        sd = pat.slot_by_name.get(slot_name)
        head = self._slot_head(sd, span) or (span[-1] if span else None)
        if head is None:
            return False, False, "槽位为空"

        if kind == "pos":
            if head.pos == value:
                return True, False, f"{head.text}/{head.pos} 命中"
            if head.pos == "?":
                if sd is not None and sd.is_predicate:
                    # 谓语是句子的骨架，不接受"词性未知"顶岗：
                    # 否则 `给我一本书` 会被读成 S=给我一 / V=本 / O=书
                    return False, False, f"{head.text} 词性未知（OOV），不能充当谓语槽"
                return True, True, f"{head.text} 为 OOV，词性按可容通过"
            return False, False, f"{head.text}/{head.pos} ≠ {value}"

        if kind == "role":
            if sd is not None and sd.default_role == value:
                return True, False, "槽位默认角色一致"
            meta = self._role_meta.get(value, {})
            marker = meta.get("case_marker") or ""
            for m in [x for x in str(marker).split("/") if x]:
                if any(t.text == m for t in tokens):
                    return True, False, f"由虚词「{m}」引出"
            got = sd.default_role if sd else None
            return False, False, f"槽位默认角色为 {got!r}，约束要求 {value}"

        if kind in ("feature", "animacy", "countability"):
            feat, want = self._normalize_feature(kind, value)
            gotv = self._has_feature(head.sense_id, feat)
            if gotv is None:
                if head.is_oov or head.sense_id is None:
                    return True, True, f"{head.text} 无义项记录，{feat} 按可容通过"
                return False, False, f"{head.text} 未标注 {feat}"
            if feat == "+animate" and value == "inanimate":
                return gotv != "1", False, f"{feat}={gotv}"
            if feat == "+animate" and value == "animate":
                return gotv == "1", False, f"{feat}={gotv}"
            return gotv == want, False, f"{feat}={gotv}（期望 {want}）"

        if kind == "definiteness":
            got = self._definiteness(span, tokens)
            return got == value, False, f"判定为 {got}"

        if kind == "classifier":
            if any(t.text == value for t in span):
                return True, False, f"量词「{value}」出现在槽位内"
            pairs = self._noun_classifiers.get(head.sense_id or -1, [])
            if pairs:
                names = [self._sense_text(c) for (c, _p, _d) in pairs]
                return False, False, f"{head.text} 的可用量词为 {'/'.join(names)}"
            return False, False, f"{head.text} 无量词配对记录"

        if kind == "aspect":
            want = self._normalize_aspect(value)
            for t in tokens:
                info = self._fw_info.get(t.sense_id or -1)
                if info and info["category"] == "动态助词":
                    if info["subcategory"] == want:
                        return True, False, f"体标记「{t.text}」= {want}"
            return False, False, f"句中无 {want} 体标记"

        if kind == "subcategory":
            if head.subcategory == value:
                return True, False, f"{head.text} 次类={value}"
            if head.subcategory is None and head.is_oov:
                # 次类是**词汇属性**，词典没有就不能假定它有。
                # （若按"可容通过"处理，`{A}{V}{IO}{DO}` 会把 `给我一本书`
                #   读成 A=给我 / V=一，因为 OOV 的 `一` 被免检了。）
                return False, False, f"{head.text} 为 OOV，次类无法核验 → 判为不满足"
            return False, False, f"{head.text} 次类={head.subcategory!r} ≠ {value}"

        if kind == "lexeme":
            if any(t.text == value for t in span):
                return True, False, f"词形「{value}」命中"
            return False, False, f"槽位内无词形「{value}」"

        if kind == "marker":
            if any(t.text == value for t in tokens):
                return True, False, f"标记「{value}」出现"
            return False, False, f"缺少标记「{value}」"

        return True, True, f"未知约束类型 {kind}，跳过"

    @staticmethod
    def _normalize_feature(kind: str, value: str) -> tuple[str, str]:
        if kind == "animacy":
            return "+animate", "1"
        if kind == "countability":
            return "+countable", "1"
        if value.startswith("+"):
            return value, "1"
        if value.startswith("-"):
            return "+" + value[1:], "0"
        return value, "1"

    @staticmethod
    def _normalize_aspect(value: str) -> str:
        return {
            "perfective": "完成体", "complete": "完成体", "完成": "完成体",
            "progressive": "持续体", "continuous": "持续体", "持续": "持续体",
            "experiential": "经历体", "经历": "经历体",
        }.get(value, value)

    def _definiteness(self, span: list[Token], tokens: list[Token]) -> str:
        """有定性判定（v0.1 的保守口径）。

        汉语没有冠词，本模块用三条可验证线索，**其余一律判"有定"**：

        * 槽位内出现指示代词/人称代词（``pos='r'``）→ 有定
        * 槽位起始为 ``一/两/几 + 量词`` 或含数量短语 → 无定
        * 槽位内含「的」字定语且定语为专名 → 有定

        .. note::
            把「数量短语 → 无定」单列，是为了让 ``把字句`` 的
            ``slot_constraints(P,'definiteness','definite')`` 真的能咬人：
            `猫把鱼吃了` 通过，`猫把一条鱼吃了` 会被点名。
        """
        if not span:
            return "indefinite"
        if any(t.pos == "r" for t in span):
            return "definite"
        for i, t in enumerate(span):
            if t.pos == "m":
                nxt = span[i + 1] if i + 1 < len(span) else None
                if nxt is not None and nxt.pos in ("q", "?"):
                    return "indefinite"
                return "indefinite"
            if t.pos == "q":
                return "indefinite"
        return "definite"

    def _has_feature(
        self, sense_id: int | None, feature: str, max_hops: int = 3
    ) -> str | None:
        """查义素值：先查本义项，未标注则沿 ``is_a`` 上位链继承（≤ ``max_hops`` 层）。

        「继承」不是锦上添花 —— ``猫`` 的 ``+animate`` 就同时写在自身与
        ``动物``/``生物`` 上，一旦某天只写在上位，这一层继承必须仍然工作。
        """
        if sense_id is None:
            return None
        key = (int(sense_id), feature)
        if key in self._feature_cache:
            return self._feature_cache[key]
        value = self._feature_direct(int(sense_id), feature)
        if value is None:
            try:
                for anc in self._ancestor_ids(int(sense_id), HIERARCHY_RELATION, max_hops):
                    value = self._feature_direct(anc, feature)
                    if value is not None:
                        break
            except ParserError:
                value = None
        self._feature_cache[key] = value
        return value

    def _feature_direct(self, sense_id: int, feature: str) -> str | None:
        rows = self._q(
            "SELECT value FROM sense_features WHERE sense_id = ? AND feature = ?",
            (sense_id, feature),
        )
        return str(rows[0][0]) if rows else None

    def _ancestor_ids(self, sense_id: int, relation: str, max_depth: int) -> list[int]:
        key = (int(sense_id), relation, int(max_depth))
        cached = self._ancestor_cache.get(key)
        if cached is not None:
            return cached
        try:
            ids = get_ancestor_ids(self.conn, int(sense_id), relation, int(max_depth))
        except DatabaseError as exc:
            raise ParserError(f"闭包查询失败：{exc}") from exc
        self._ancestor_cache[key] = ids
        return ids

    def _ancestor_entries(
        self, sense_id: int, relation: str, max_depth: int
    ) -> list[tuple[int, int]]:
        """上位列表 ``[(ancestor_id, depth), …]``（带深度，按 depth 升序）。

        之所以要深度：``鱼`` 的上位是 ``{动物, 食物}`` 两个**并列**父类，
        ``生物`` 在第二层 —— 把它们拼成 "鱼 ⊑ 动物 ⊑ 食物 ⊑ 生物" 是错的。
        渲染时必须显式带距离，调用方才能看出这是多条路径而非一条链。
        """
        key = (int(sense_id), relation, int(max_depth))
        cached = self._hdepth_cache.get(key)
        if cached is not None:
            return cached
        try:
            entries = [(e.node_id, e.depth)
                       for e in get_ancestors(self.conn, int(sense_id), relation, max_depth)]
        except DatabaseError as exc:
            raise ParserError(f"闭包查询失败：{exc}") from exc
        self._hdepth_cache[key] = entries
        return entries

    # ------------------------------------------------------------------
    def _slot_head(self, sd: _SlotDef | None, span: list[Token]) -> Token | None:
        """槽位中心语：谓语槽取首个动词性 token，其余取最后一个实词。"""
        if not span:
            return None
        if sd is not None and sd.is_predicate:
            for t in span:
                if t.pos in ("v", "a", "b"):
                    return t
            for t in span:
                if t.pos == "?":
                    return t
            return span[-1]
        content = [t for t in span if t.is_content]
        return content[-1] if content else span[-1]

    def _literal_weight(self, lit: str) -> float:
        """标记字面量的 IDF 权重：越"稀有"的标记信号越强。"""
        freq = max(1, int(self._literal_freq.get(lit, 1)))
        total = max(1, len(self._patterns))
        return _LITERAL_BASE + _LITERAL_IDF_SCALE * math.log(total / freq)

    @staticmethod
    def _consumed_indices(parse: _Parse) -> set[int]:
        """被该切分方案实际消费的 token 下标（字面量命中 ∪ 全部槽位区间）。"""
        consumed: set[int] = set()
        for _alt, idx in parse.literal_hits:
            consumed.add(idx)
        for spans in parse.spans.values():
            for (s, e) in spans:
                consumed.update(range(s, e))
        return consumed

    def _lexical_evidence(self, parse: _Parse, tokens: list[Token]) -> tuple[int, int]:
        """统计「词汇证据」：被消费的 token 里有多少词典命中（known）/ OOV。

        这是词汇证据闸门（``_validate_parse`` 闸门 ⑤）与 OOV 惩罚
        （``_score``）的**唯一真源**。必须在闸门判定之前调用，否则读到的是
        ``_Parse.known`` 的默认值 0，会把正常句子一票否决。

        返回 ``(known, consumed_total)``，并顺带回写 ``parse.known`` /
        ``parse.oov``。

        .. note::
            ``_CLOSED_CLASS_FALLBACK`` 兜底出来的虚词（不/没/就/越…）虽然词性
            可信，但 ``entry_id is None``，因此**计入 OOV**。这是刻意的信号：
            兜底词表只是临时桥梁，按 OOV 轻罚（1.2）能持续暴露"词典缺项"，
            且惩罚很轻、不会否决句子（否决由覆盖率闸门独立负责）。
        """
        consumed = self._consumed_indices(parse)
        known = 0
        for idx in consumed:
            tk = tokens[idx]
            # 字面量命中的封闭类虚词（把/被/比…）没有 entry_id，
            # 但它们是词典登记过的标记，同样算词汇证据。
            if tk.entry_id is not None or tk.is_marker_literal:
                known += 1
        parse.known = known
        parse.oov = len(consumed) - known
        return known, len(consumed)

    def _score(
        self, pat: _Pattern, parse: _Parse, tokens: list[Token]
    ) -> tuple[float, float]:
        """唯一真源的评分函数（公式见模块 docstring）。"""
        n = len(tokens)
        consumed = self._consumed_indices(parse)

        coverage = (len(consumed) / n) if n else 0.0

        self._lexical_evidence(parse, tokens)

        lit_score = sum(self._literal_weight(alt) for alt, _ in parse.literal_hits)

        slot_score = 0.0
        for name, spans in parse.spans.items():
            sd = pat.slot_by_name.get(name)
            unit = _SLOT_REQUIRED if (sd is None or sd.is_required) else _SLOT_OPTIONAL
            slot_score += unit * len(spans)

        con_score = 0.0
        for c in parse.checks:
            w = CONSTRAINT_WEIGHTS.get(c.kind, 1.5)
            if c.ok:
                con_score += w * (0.5 if "OOV" in c.detail else 1.0)
            else:
                # 因 OOV 而无法核验的违反，只算半价 —— 不能因为词典缺词就重罚
                mult = _CON_VIOLATION_MULT * (0.5 if "OOV" in c.detail else 1.0)
                con_score -= w * mult

        prior = 0.0
        internal = 0
        if self.func_pos_prior:
            for name, spans in parse.spans.items():
                sd = pat.slot_by_name.get(name)
                for (s, e) in spans:
                    if not self._pos_compatible(sd, tokens[s:e]):
                        prior -= _FUNC_POS_PRIOR
        # 槽位区间**内部**夹可跳过成分（`{V}=吃不吃`）→ 判为吞并，扣分
        for spans in parse.spans.values():
            for (s, e) in spans:
                internal += sum(1 for k in range(s + 1, e - 1) if not tokens[k].is_content)

        bonus = 0.0
        if self._complement_hit(pat, parse, tokens):
            bonus += _COMPLEMENT_BONUS
        if pat.category == "话题结构" and self._topic_break(parse):
            bonus += _TOPIC_BREAK_BONUS

        # 跳过代价按被跳成分的词性分级：助词便宜、副词昂贵
        skip_penalty = 0.0
        for k in parse.skipped:
            if k >= n:
                continue
            tk = tokens[k]
            if tk.pos in ("u", "y"):
                skip_penalty += _SKIP_MARKER_PENALTY
            elif tk.is_clause_punct:
                skip_penalty += _SKIP_MARKER_PENALTY
            else:
                skip_penalty += _SKIP_ADVERB_PENALTY

        residual = max(0, n - len(consumed) - len(parse.skipped))
        penalty = (skip_penalty
                   + _RESIDUAL_PENALTY * residual
                   + _INTERNAL_MARKER_PENALTY * internal
                   + _OOV_TOKEN_PENALTY * parse.oov
                   + _OMITTED_SLOT_PENALTY * len(parse.omitted))

        spec = SPEC_WEIGHTS.get(pat.category, 1.0)
        total = lit_score + slot_score + con_score + prior
        score = (spec * total * (_COVERAGE_FLOOR + (1.0 - _COVERAGE_FLOOR) * coverage)
                 + bonus - penalty)
        return score, coverage

    @staticmethod
    def _pos_compatible(sd: _SlotDef | None, span: list[Token]) -> bool:
        """「句法成分 ↔ 词性」结构先验（不满足只扣分，不否决）。"""
        if sd is None or not sd.syntactic_function or not span:
            return True
        pref = _POS_PREF.get(sd.syntactic_function)
        if pref is None:
            return True
        content = [t for t in span if t.is_content]
        if not content:
            return True
        head = content[0] if sd.is_predicate else content[-1]
        return head.pos in pref

    def _complement_hit(self, pat: _Pattern, parse: _Parse, tokens: list[Token]) -> bool:
        """V + (R|C|D) 命中 ``complements`` 表 → 述补结构加分。"""
        if not self._complements:
            return False
        vs = parse.spans.get("V") or []
        for alt in ("R", "C", "D"):
            rs = parse.spans.get(alt) or []
            if not vs or not rs:
                continue
            vh = self._slot_head(pat.slot_by_name.get("V"),
                                 [tokens[i] for i in range(*vs[0])])
            rh = self._slot_head(pat.slot_by_name.get(alt),
                                 [tokens[i] for i in range(*rs[0])])
            if vh is None or rh is None or vh.sense_id is None or rh.sense_id is None:
                continue
            if (vh.sense_id, rh.sense_id, "*") in self._complements:
                return True
        return False

    @staticmethod
    def _comp_type_of(pat: _Pattern) -> str | None:
        """从 ``sentence_patterns.semantics`` 推出 ``complement_types.comp_type``。

        ``短语结构`` 类的三条模板语义分别是「结果 / 趋向 / 可能」，
        与 ``complement_types`` 的「结果补语 / 趋向补语 / 可能补语」一一对应 ——
        与其在代码里写死 pattern_id，不如按语义列推导。
        """
        sem = (pat.semantics or "").strip()
        if not sem:
            return None
        for ct in ("结果补语", "趋向补语", "可能补语", "程度补语", "数量补语", "状态补语"):
            if ct.startswith(sem) or sem.startswith(ct[:2]):
                return ct
        return None

    def _comp_pair(
        self, pat: _Pattern, parse: _Parse, tokens: list[Token]
    ) -> tuple[int, int] | None:
        """取出 (述语义项, 补语义项)，供与 ``complements`` 比对。"""
        vs = parse.spans.get("V") or []
        if not vs:
            return None
        vh = self._slot_head(pat.slot_by_name.get("V"), [tokens[i] for i in range(*vs[0])])
        if vh is None or vh.sense_id is None:
            return None
        for alt in ("R", "C", "D"):
            rs = parse.spans.get(alt) or []
            if not rs:
                continue
            rh = self._slot_head(pat.slot_by_name.get(alt),
                                 [tokens[i] for i in range(*rs[0])])
            if rh is None or rh.sense_id is None:
                continue
            return (int(vh.sense_id), int(rh.sense_id))
        return None

    @staticmethod
    def _topic_break(parse: _Parse) -> bool:
        """「逗号前为 Topic」：句中标点出现在 Topic 槽位之后。"""
        spans = parse.spans.get("Topic") or []
        if not spans:
            return False
        end = spans[0][1]
        return any(br >= end for br in parse.clause_breaks)

    # =====================================================================
    # 6.6 槽位填充（含省略回填与话题链）
    # =====================================================================
    def fill_slots(
        self,
        match: PatternMatch,
        tokens: Sequence[Token],
        context: ParserContext | Mapping[str, Any] | None = None,
    ) -> dict[str, SlotFill]:
        """把 token 填进模板槽位，并处理省略回填与话题链。

        :param match: :meth:`match_pattern` 的结果
        :param tokens: 同一份 token 序列
        :param context: 覆盖 :attr:`context`
        :returns: ``dict[slot_name, SlotFill]``（``SlotFill`` 可直接当 ``list[Token]`` 用）

        省略回填：查 ``ellipsis_rules`` —— 按 ``(context_type, slot_role)`` 取出全部策略，
        按 ``priority`` 降序逐个尝试，命中即停并记录 ``ellipsis_rule_id`` 与 ``strategy``；
        ``max_distance`` 与当前 ``turn_index`` 比较，超出跨句距离的策略直接跳过。

        话题链：句首 ``Topic`` 槽位（及省略的主语）优先取 ``topic_chains.topic_node_id``，
        没有篇章记录时回退到内存态篇章状态。
        """
        ctx = ParserContext.from_any(context) if context is not None else self.context
        body = list(tokens)
        pattern = self._pattern_by_id.get(match.pattern_id)
        fills: dict[str, SlotFill] = {}
        order = [s for s in pattern.slots] if pattern else []
        # 模板里出现但 schema 未登记的槽位（防御性处理）也一并纳入
        known = {s.slot_name for s in order}
        for name in match.slot_spans:
            if name not in known and pattern is not None:
                order.append(_SlotDef(name, 999, True, None, None))

        context_type = self._context_type(match, ctx)

        for sd in order:
            name = sd.slot_name
            spans = match.spans_of(name)
            if spans:
                toks = [body[i] for (s, e) in spans for i in range(s, e)]
                fills[name] = SlotFill(name, toks, source="matched", confidence=1.0)
                continue
            role = sd.default_role or self._fallback_role(sd)
            got = self._apply_ellipsis(role, context_type, ctx, match)
            if got is not None:
                fill, rule = got
                fills[name] = SlotFill(
                    name, fill, source="ellipsis", ellipsis_rule_id=rule.rule_id,
                    strategy=rule.strategy,
                    confidence=0.75 if fill[0].entry_id is not None else 0.4,
                    note=rule.description or "",
                )
            # 可选槽位未填充 → 不进入 filled_slots（与"省略"区分开）

        return fills

    def _fallback_role(self, sd: _SlotDef) -> str:
        if sd.syntactic_function == "谓语":
            return "述题"
        if sd.syntactic_function == "话题":
            return "话题"
        return f"未定:{sd.slot_name}"

    def _context_type(self, match: PatternMatch, ctx: ParserContext) -> str:
        """推断 ``ellipsis_rules.context_type``：显式覆盖 > 句类 > 篇章状态。"""
        if ctx.context_type:
            return ctx.context_type
        if match.mood == "祈使":
            return "imperative"
        if ctx.speaker and ctx.listener:
            return "dialogue"
        state = self._discourse_state(ctx.doc_id)
        if state and state.count > 1:
            return "narrative"
        return "topic_chain"

    # ------------------------------------------------------------------
    def _apply_ellipsis(
        self, role: str, context_type: str, ctx: ParserContext, match: PatternMatch
    ) -> tuple[list[Token], _EllipsisRule] | None:
        """按 ``ellipsis_rules`` 的优先级尝试回填一个槽位。

        ``max_distance`` 的语义是**最大跨句数**：某策略取到的实体若来自 3 句之前，
        而规则写着 ``max_distance=3``，仍算命中；超出即换下一条策略。
        ``speaker``/``listener``（``max_distance=0``）取自上下文而非篇章历史，
        距离记为 0，因此永远满足。
        """
        rules = self._ellipsis.get((context_type, role), [])
        if not rules:
            return None
        state = self._discourse_state(ctx.doc_id)
        for rule in sorted(rules, key=lambda r: -r.priority):
            got = self._run_strategy(rule.strategy, ctx, state, match)
            if got is None:
                continue
            fill, distance = got
            if (rule.max_distance is not None and distance is not None
                    and distance > rule.max_distance):
                logger.debug("策略 %s 距离 %d > max_distance %d，跳过",
                             rule.strategy, distance, rule.max_distance)
                continue
            return fill, rule
        return None

    def _run_strategy(
        self, strategy: str, ctx: ParserContext, state: _Discourse | None,
        match: PatternMatch,
    ) -> tuple[list[Token], int | None] | None:
        """执行一种省略回填策略。

        :returns: ``(填回的 token 列表, 跨句距离)``；距离为 ``None`` 表示不来自篇章历史

        ``world_knowledge`` / ``indefinite_default`` 返回**未解析占位符** ``∅``
        （``pos='?'``、``sense_id=None``），不伪造实体 —— 调用方据
        ``SlotFill.confidence`` 区分"回填到了"与"只能标个空位"。
        """
        m: _Mention | None = None
        if strategy == "speaker":
            surface = ctx.speaker_entity or ctx.speaker
            if not surface:
                return None
            return ([self._token_from_surface(
                str(surface), "r" if str(surface) in _HUMAN_PRONOUNS else "n")], 0)
        if strategy == "listener":
            surface = ctx.listener_entity or ctx.listener
            if not surface:
                return None
            return ([self._token_from_surface(
                str(surface), "r" if str(surface) in _HUMAN_PRONOUNS else "n")], 0)
        if strategy == "previous_subject":
            m = state.subject if state else None
        elif strategy == "previous_object":
            m = state.object if state else None
        elif strategy == "topic":
            m = (state.topic if state else None)
            if m is None and ctx.topic:
                return ([self._token_from_surface(str(ctx.topic), "n")], None)
            if m is None and ctx.doc_id:
                m = self._topic_from_db(ctx.doc_id)
        elif strategy in ("previous_mentioned", "salient_entity", "nearest_np"):
            m = state.object if state and state.object else (state.subject if state else None)
        elif strategy in ("world_knowledge", "indefinite_default"):
            return ([Token("∅", "?", None, None, -1, -1, is_oov=True,
                           inferred_pos="ellipsis")], None)
        if m is None:
            return None
        # 当前解析第 N 句时 state.count 已是 N-1（本句尚未 _remember）
        distance = None
        if state is not None:
            distance = max(0, state.count - m.utterance_index + 1)
        return ([self._token_from_surface(m.text, m.pos)], distance)

    def _token_from_surface(self, surface: str, prefer_pos: str = "n") -> Token:
        """把表层形式还原成 token（回填出来的 token ``start/end = -1``）。"""
        lex = self._lexicon.get(surface)
        if lex is not None:
            return Token(
                text=surface, pos=lex.pos, entry_id=lex.word_id, sense_id=lex.sense_id,
                start=-1, end=-1, subcategory=lex.subcategory,
                fw_category=self._fw_info.get(lex.sense_id, {}).get("category"),
                frequency=lex.frequency, is_primary=lex.is_primary,
            )
        pos, how = self._guess_pos(surface)
        if pos == "?":
            pos = prefer_pos
        return Token(text=surface, pos=pos, start=-1, end=-1, is_oov=True,
                     inferred_pos=how or "ellipsis")

    # ------------------------------------------------------------------
    def _discourse_state(self, doc_id: str) -> _Discourse | None:
        """取某文档的篇章状态；必要时按需从库里重建（``use_db_context``）。

        :returns: ``_Discourse``；内存里没有且未开启 ``use_db_context`` 时返回 ``None``

        这是 `_context_type` / `_apply_ellipsis` 读取篇章状态的**唯一入口** ——
        省略回填需要"上一次运行留下的上下文"时，就由这里从
        ``utterances``（⨯ ``anaphora`` ⨯ ``topic_chains``）重建。
        """
        state = self._discourse.get(doc_id)
        if state is not None:
            return state
        if not self.use_db_context or doc_id in self._db_context_loaded:
            return None
        self._db_context_loaded.add(doc_id)     # 先标记，失败也不重试
        state = self._load_discourse_from_db(doc_id)
        if state is not None:
            self._discourse[doc_id] = state
        return state

    def _load_discourse_from_db(self, doc_id: str) -> _Discourse | None:
        """从 ``utterances`` ⨯ ``anaphora`` ⨯ ``topic_chains`` 重建篇章状态。

        仅由 :meth:`_discourse_state` 在 ``use_db_context=True`` 时调用。
        设计意图是让"省略回填"能跨进程/跨运行工作：上一次运行 ``persist``
        落盘的施事/受事与同指记录，本次解析 `∅ 吃鱼。` 这类零形句时可直接复用。

        ``anaphora`` 哪怕一条都没有，只要该文档有 ``utterances`` 记录，
        就返回一个"计数正确、但槽位为空"的状态 —— 让 ``_context_type`` 能据此
        判定为 ``narrative``（历史不止一句），从而选对省略规则。
        """
        rows = self._q(
            """SELECT u.seq, a.slot_role, a.form, a.antecedent_node_id
                 FROM utterances u
                 JOIN anaphora a ON a.utterance_id = u.utterance_id
                WHERE u.doc_id = ?
                ORDER BY u.seq ASC, a.anaphora_id ASC""",
            (doc_id,),
        )
        agg = self._q(
            "SELECT COALESCE(MAX(seq), 0) FROM utterances WHERE doc_id = ?",
            (doc_id,),
        )
        max_seq = int(agg[0][0]) if agg else 0
        if not rows and max_seq == 0:
            return None

        state = _Discourse(doc_id)
        for (seq, slot_role, form, node) in rows:
            text, pos, sense_id = self._resolve_mention_surface(form, node)
            if not text:
                continue        # 零形且无先行节点 → 无法还原实体，跳过
            state.push(_Mention(text, pos, sense_id,
                                str(slot_role) if slot_role else None,
                                int(seq), "anaphora"))
        state.count = max_seq
        # ``anaphora`` 为空或全为不可还原的零形 → 退一步用 topic_chains 定话题
        if state.topic is None:
            state.topic = self._topic_from_db(doc_id)
        return state

    def _resolve_mention_surface(
        self, form: Any, node: Any
    ) -> tuple[str, str, int | None]:
        """把 ``anaphora`` 的 (表层形式, 先行节点) 还原成 ``(text, pos, sense_id)``。

        优先级：先行节点对应的词典词形 > 表层形式本身。两者都拿不到（零形回指
        且未登记先行节点）时返回空串，由调用方跳过。
        """
        if node is not None:
            info = self._sense_index.get(int(node))
            if info is not None:
                return str(info["word"]), str(info["pos"]), int(node)
        text = "" if form is None else str(form).strip()
        if not text or text == "∅":
            return "", "", None
        pos, _how = self._guess_pos(text)
        if pos == "?":
            pos = "r" if text in _HUMAN_PRONOUNS else "n"
        return text, pos, None

    def _topic_from_db(self, doc_id: str) -> _Mention | None:
        rows = self._q(
            """SELECT topic_node_id, start_seq, end_seq FROM topic_chains
                WHERE doc_id = ? AND topic_node_id IS NOT NULL
                ORDER BY start_seq DESC LIMIT 1""",
            (doc_id,),
        )
        if not rows:
            return None
        node = rows[0][0]
        if node is None:
            return None
        info = self._sense_index.get(int(node))
        if info is None:
            return None
        return _Mention(str(info["word"]), str(info["pos"]), int(node), "话题", -1, "topic_chains")

    # =====================================================================
    # 6.7 语义角色标注
    # =====================================================================
    def annotate_semantic_roles(
        self,
        filled_slots: Mapping[str, Any],
        match: PatternMatch | None = None,
        tokens: Sequence[Token] | None = None,
    ) -> dict[str, SemanticRole]:
        """把槽位映射成语义角色，并用虚词表做角色改写。

        两条数据来源，优先级 **虚词改写 > 槽位默认角色**：

        1. ``pattern_slots.default_role``（+ 谓语槽回退为 ``述题``）
        2. ``function_words`` ⨯ ``function_word_roles``：如
           ``把 → 受事（前置）``、``被 → 施事（前置）``、``在 → 处所``、``从 → 起点``。
           改写规则：若某虚词紧邻某槽位区间之前，则该槽位角色改为该虚词声明的角色 ——
           这就是「把字句提 P（受事）」「被字句提 A（施事）」的实现方式，
           并且**完全由数据库驱动**，不在代码里硬编码"把"字。

        :raises MatchError: ``match`` 为 ``None`` 且 ``filled_slots`` 里没有任何可用信息
        """
        toks = list(tokens) if tokens is not None else None
        pattern = self._pattern_by_id.get(match.pattern_id) if match else None
        if pattern is None and not filled_slots:
            raise MatchError("annotate_semantic_roles 需要 match 或非空 filled_slots")

        # ---- 虚词 → 角色（数据库驱动，不硬编码） ----
        marker_roles: dict[int, tuple[str, str]] = {}   # token index → (role, surface)
        if toks:
            for i, t in enumerate(toks):
                if not t.is_content or t.pos != "p":
                    continue
                for (role, position) in self._fw_roles.get(t.sense_id or -1, []):
                    marker_roles[i] = (role, t.text)
                    if position == "框式":
                        marker_roles[i] = (role, t.text)

        roles: dict[str, SemanticRole] = {}
        for slot_name, fill in filled_slots.items():
            sd = pattern.slot_by_name.get(slot_name) if pattern else None
            f = fill if isinstance(fill, SlotFill) else SlotFill(slot_name, _as_token_list(fill))
            default_role = (sd.default_role if sd else None) or self._fallback_role(
                sd or _SlotDef(slot_name, 999, True, None, None)
            )
            role = default_role
            source = "ellipsis" if f.source == "ellipsis" else "default"
            marker_text = ""
            # 虚词改写：该槽位在 token 流里的起点前是否紧邻一个介词
            if toks and f.tokens and f.source == "matched":
                start = f.tokens[0].start
                idx = next((i for i, t in enumerate(toks) if t.start == start), None)
                if idx is not None:
                    # 只看**紧邻**的介词：中间不允许夹实词。
                    # 若允许回溯 2~3 个 token，`鱼 被 猫 吃` 里的「被」会连
                    # 谓语槽「吃」也改写成施事 —— 因为「猫」被当成了透明。
                    k = idx - 1
                    while k >= 0 and toks[k].is_skippable:
                        k -= 1
                    if k >= 0 and toks[k].is_content and toks[k].pos == "p":
                        hit = marker_roles.get(k)
                        if hit is not None:
                            role, marker_text = hit[0], hit[1]
                            source = "marker"
            key = role
            if key in roles:
                key = f"{role}#{slot_name}"
            roles[key] = SemanticRole(
                role=role, slot_name=slot_name, tokens=list(f.tokens),
                marker=marker_text, source=source,
                confidence=f.confidence if source != "ellipsis" else f.confidence * 0.8,
                pattern_semantics=(match.semantics if match else "") or "",
            )

        self._infer_roles_from_semantics(roles, match, toks)
        for r in roles.values():
            if r.pattern_semantics and "被动" in r.pattern_semantics and r.role == "受事":
                r.notes.append("被动句式：受事处于主语位置")
        return roles

    def _infer_roles_from_semantics(
        self, roles: dict[str, SemanticRole], match: PatternMatch | None,
        toks: list[Token] | None,
    ) -> None:
        """句式整体义补强角色判定（处置/被动/给予/存现/比较）。"""
        if match is None or not match.semantics:
            return
        sem = match.semantics
        if "处置" in sem:
            for r in roles.values():
                if r.slot_name == "P" and r.role == "受事":
                    r.notes.append("处置式：P 为被处置的受事")
                if r.slot_name == "A" and r.role == "施事":
                    r.notes.append("处置式：A 为施事")
        if "给予" in sem:
            for r in roles.values():
                if r.slot_name == "IO":
                    r.notes.append("双宾句式：IO 为与事（接受者）")
                if r.slot_name == "DO":
                    r.notes.append("双宾句式：DO 为受事（给予物）")
        if "存在" in sem or "出现" in sem:
            for r in roles.values():
                if r.slot_name == "L":
                    r.notes.append("存现句：L 为处所")
                if r.slot_name == "NP":
                    r.notes.append("存现句：NP 为存在物")

    # =====================================================================
    # 6.8 关系查询与推理
    # =====================================================================
    def query_relations(
        self,
        semantic_roles: Mapping[str, Any],
        match: PatternMatch | None = None,
        tokens: Sequence[Token] | None = None,
    ) -> list[InferenceResult]:
        """用语义角色去查询知识库，并触发 ``inference_rules`` 里的规则。

        依次执行四类查询，全部结果带**可解释的推理链**：

        1. **词汇语义合法性**：谓词 → 典型施事/受事的 ``relations`` 行，
           用义项类别（``sense_categories`` ＋ ``sense_category`` 层级）判断同类性。
        2. **属性/能力继承**：对每个名词性角色调用
           :func:`logos.db.closure.get_property_inheritance`，取 ``can_do`` 与
           ``has_property``，并给出「自身及上位链上的声明者」。
        3. **闭包推理**：物化闭包里的 ``depth ≥ 2`` 行 → 传递性推导（``is_a`` 传递）。
        4. **规则引擎**：遍历 ``inference_rules``：

           * 带 ``rule_premises``/``rule_conclusions`` 的（transitive / symmetric / inherit）
             走**声明式**匹配；
           * 只有 ``condition_sql`` 的（constraint / default）走**逃生通道**执行 ——
             仅允许单条 ``SELECT``，参数以 ``:name`` 命名绑定，缺参即跳过，
             绝不拼字符串，因此不存在注入面。

        :returns: ``list[InferenceResult]``，按置信度降序
        """
        roles: dict[str, SemanticRole] = {}
        for k, v in semantic_roles.items():
            roles[k] = v if isinstance(v, SemanticRole) else _coerce_role(k, v)

        out: list[InferenceResult] = []
        predicate = self._find_predicate(roles)
        verb_sense = predicate.head_sense_id if predicate else None
        verb_text = predicate.text if predicate else ""

        # ---- ① 词汇语义合法性 ----
        if verb_sense is not None:
            out.extend(self._check_role_legality(verb_sense, verb_text, roles))

        # ---- ② 属性/能力继承 ----
        for name, r in roles.items():
            if r.head_sense_id is None or r.role in ("述题", "命题"):
                continue
            out.extend(self._inherited_properties(r))

        # ---- ③ 闭包推理（depth ≥ 2） ----
        out.extend(self._closure_inference(roles))

        # ---- ④ 规则引擎 ----
        out.extend(self._run_rules(roles, verb_sense, verb_text, match, tokens))

        return self._dedupe(out)

    @staticmethod
    def _dedupe(items: list[InferenceResult]) -> list[InferenceResult]:
        """按 ``(主体, 关系, 客体)`` 去重并保留置信度最高的那条。

        同一事实可能被两条路径得到（物化闭包 vs ``is_a传递`` 规则），
        两条都留着只会让推理链变得噪音化 —— 保留高置信度的，其余丢弃。
        """
        best: dict[tuple[str, str, str], InferenceResult] = {}
        for it in items:
            key = (it.subject, it.relation, it.obj)
            cur = best.get(key)
            if cur is None or it.confidence > cur.confidence:
                best[key] = it
        out = list(best.values())
        out.sort(key=lambda x: -x.confidence)
        return out

    @staticmethod
    def _find_predicate(roles: Mapping[str, SemanticRole]) -> SemanticRole | None:
        for r in roles.values():
            if r.role in ("述题", "命题") and r.tokens:
                return r
        for r in roles.values():
            if r.slot_name in ("V", "V1", "V2"):
                return r
        return None

    # ------------------------------------------------------------------
    def _check_role_legality(
        self, verb_sense: int, verb_text: str, roles: Mapping[str, SemanticRole]
    ) -> list[InferenceResult]:
        """谓词 → 典型论元的词汇语义校验（读 ``relations`` + ``relation_constraints``）。"""
        out: list[InferenceResult] = []
        rows = self._q(
            """SELECT relation_id, relation, dst_id, confidence
                 FROM relations
                WHERE src_kind='sense' AND src_id = ? AND dst_kind='sense'
                  AND relation IN ('agent','patient','dative','instrument','location','result')
                ORDER BY relation, relation_id""",
            (verb_sense,),
        )
        typical: dict[str, list[tuple[int, int, Any]]] = {}
        for (rid, rel, dst, conf) in rows:
            typical.setdefault(str(rel), []).append((int(dst), int(rid), conf))

        for name, r in roles.items():
            if r.role in ("述题", "命题") or r.head_sense_id is None:
                continue
            cslot = ROLE_TO_CONSTRAINT_SLOT.get(r.role)
            if not cslot:
                continue
            cands = typical.get(cslot, [])
            if not cands:
                continue
            hit = None
            for (dst, rid, conf) in cands:
                how = self._same_category(r.head_sense_id, dst)
                if how:
                    hit = (dst, rid, conf, how)
                    break
            target = self._sense_text(cands[0][0])
            if hit is None:
                out.append(InferenceResult(
                    kind="constraint", subject=verb_text, relation=cslot,
                    obj=r.entity, confidence=0.35,
                    chain=[f"{verb_text} 的典型{cslot}为「{target}」",
                           f"句中 {r.entity} 的义项类别与其不重合 → 选择限制存疑"],
                    note="选择限制未通过（软警告，不阻断解析）",
                    evidence=[c[1] for c in cands],
                ))
                r.notes.append(f"选择限制：{cslot} 期望「{target}」，实际 {r.entity}")
            else:
                dst, rid, conf, how = hit
                out.append(InferenceResult(
                    kind="relation", rule_id=None, rule_name="典型论元",
                    subject=verb_text, relation=cslot, obj=r.entity,
                    confidence=float(conf) if conf is not None else 0.9,
                    chain=[f"{verb_text} --{cslot}--> 「{self._sense_text(dst)}」",
                           f"{r.entity} 与「{self._sense_text(dst)}」{how}"],
                    evidence=[rid],
                ))
        return out

    def _same_category(self, sense_a: int, sense_b: int) -> str:
        """两个义项是否同类（类别闭包相交）或存在 ``is_a`` 关系。

        :returns: 人类可读的判定依据；不同类时返回 ``''``
        """
        if sense_a == sense_b:
            return "同一义项"
        ca = self._category_closure(sense_a)
        cb = self._category_closure(sense_b)
        inter = ca & cb
        if inter:
            return f"同属类别 {'/'.join(sorted(inter))}"
        try:
            anc = set(self._ancestor_ids(sense_a, HIERARCHY_RELATION, 4))
        except ParserError:
            anc = set()
        if sense_b in anc:
            return f"{self._sense_text(sense_a)} is_a {self._sense_text(sense_b)}"
        try:
            anc_b = set(self._ancestor_ids(sense_b, HIERARCHY_RELATION, 4))
        except ParserError:
            anc_b = set()
        if sense_a in anc_b:
            return f"{self._sense_text(sense_b)} is_a {self._sense_text(sense_a)}"
        return ""

    def _category_closure(self, sense_id: int) -> set[str]:
        """义项类别 ＋ 类别上位闭包（``semantic_categories.parent_cat``）。"""
        rows = self._q(
            "SELECT cat FROM sense_categories WHERE sense_id = ?", (sense_id,)
        )
        out: set[str] = set()
        for (cat,) in rows:
            c = str(cat)
            guard = 0
            while c and c not in out and guard < 12:
                out.add(c)
                c = self._cat_parent.get(c) or ""
                guard += 1
        return out

    # ------------------------------------------------------------------
    def _inherited_properties(self, role: SemanticRole) -> list[InferenceResult]:
        """属性/能力继承：``get_property_inheritance`` 的两种关系各查一遍。"""
        out: list[InferenceResult] = []
        sense_id = role.head_sense_id
        if sense_id is None:
            return out
        try:
            ancs = self._ancestor_entries(sense_id, HIERARCHY_RELATION, 3)
        except ParserError:
            ancs = []
        # 多父类是常态（鱼 is_a 动物 且 is_a 食物），因此渲染成"带距离的并列"，
        # 绝不拼成单链：`鱼 ⊑ 动物 ⊑ 食物 ⊑ 生物` 是错的。
        lineage = (f"{role.entity} 的上位："
                   + "、".join(f"{self._sense_text(a)}({d}跳)" for a, d in ancs)
                   ) if ancs else f"{role.entity} 无 is_a 上位"

        for prop_rel in ("can_do", "has_property"):
            try:
                props = get_property_inheritance(self.conn, sense_id, prop_rel)
            except DatabaseError:
                continue
            for p in props:
                pname = self._sense_text(p.property_id)
                hops = f"（距离 {p.distance}）" if p.distance else "（直接声明）"
                detail = [lineage,
                          f"{self._sense_text(p.declared_by)} --{prop_rel}--> {pname}{hops}"]
                out.append(InferenceResult(
                    kind="inherit", subject=role.entity, relation=prop_rel, obj=pname,
                    confidence=1.0 if p.distance == 0 else max(0.5, 0.9 ** p.distance),
                    chain=detail,
                    rule_name="属性继承", rule_type="inherit",
                    note=f"由「{self._sense_text(p.declared_by)}」声明",
                    evidence=[p.property_id],
                ))
        return out

    # ------------------------------------------------------------------
    def _closure_inference(self, roles: Mapping[str, SemanticRole]) -> list[InferenceResult]:
        """物化闭包中 ``depth ≥ 2`` 的行 → 传递性推导（``is_a`` 传递）。"""
        out: list[InferenceResult] = []
        seen: set[int] = set()
        for r in roles.values():
            sid = r.head_sense_id
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            try:
                rows = self._q(
                    """SELECT c.depth, c.ancestor, c.first_edge
                         FROM closure c
                        WHERE c.relation = ? AND c.descendant = ? AND c.depth >= 2
                        ORDER BY c.depth, c.ancestor LIMIT 8""",
                    (HIERARCHY_RELATION, sid),
                )
            except ParserError:
                continue
            for (depth, anc, fist_edge) in rows:
                d = int(depth)
                chain = self._chain_text(sid, int(anc), d)
                out.append(InferenceResult(
                    kind="closure", subject=r.entity, relation=HIERARCHY_RELATION,
                    obj=self._sense_text(int(anc)),
                    confidence=max(0.5, 0.9 ** d),
                    chain=chain, rule_name="is_a传递", rule_type="transitive",
                    note=f"闭包深度 {d}（见证边 #{fist_edge}）"if fist_edge else f"闭包深度 {d}",
                ))
        return out

    def _chain_text(self, start: int, end: int, depth: int) -> list[str]:
        """重建一条 ``start → … → end`` 的层级链（逐跳从 ``relations`` 取边）。"""
        out = [f"{self._sense_text(start)} --is_a--> …（{depth} 跳）--> {self._sense_text(end)}"]
        rows = self._q(
            """SELECT src_id, dst_id, relation_id FROM relations
                WHERE relation = ? AND src_kind='sense' AND dst_kind='sense'
                  AND (src_id = ? OR dst_id = ? OR src_id = ? OR dst_id = ?)""",
            (HIERARCHY_RELATION, start, start, end, end),
        )
        for (s, d, rid) in rows:
            out.append(f"  边 #{rid}: {self._sense_text(int(s))} is_a {self._sense_text(int(d))}")
        return out

    # ------------------------------------------------------------------
    def _run_rules(
        self,
        roles: Mapping[str, SemanticRole],
        verb_sense: int | None,
        verb_text: str,
        match: PatternMatch | None,
        tokens: Sequence[Token] | None,
    ) -> list[InferenceResult]:
        """遍历 ``inference_rules``：声明式前提优先，``condition_sql`` 兜底。"""
        out: list[InferenceResult] = []
        bindings = self._rule_bindings(roles, verb_sense, verb_text)
        for rule in self._rules:
            if rule.conclusions:
                out.extend(self._run_declarative_rule(rule, roles, bindings))
            elif rule.condition_sql:
                out.extend(self._run_condition_rule(rule, bindings))

        # default 类规则（量词默认）单独兜一遍：它不属于任何角色的直接推理
        out.extend(self._classifier_suggestions(roles))
        return out

    def _rule_bindings(
        self, roles: Mapping[str, SemanticRole], verb_sense: int | None, verb_text: str
    ) -> dict[str, Any]:
        """给 ``condition_sql`` / 声明式规则准备变量绑定。"""
        b: dict[str, Any] = {}
        if verb_sense is not None:
            b["verb"] = verb_sense
        for r in roles.values():
            if r.role in ("施事", "致事", "感事") and r.head_sense_id is not None:
                b.setdefault("a", r.head_sense_id)
                b.setdefault("subject_type", r.head_sense_id)
                b["agent_word"] = r.text
            elif r.role in ("受事", "结果") and r.head_sense_id is not None:
                b.setdefault("b", r.head_sense_id)
                b.setdefault("object_type", r.head_sense_id)
                b["patient_word"] = r.text
            elif r.head_sense_id is not None:
                b.setdefault("noun", r.head_sense_id)
                b.setdefault("c", r.head_sense_id)
        b.setdefault("verb_word", verb_text)
        return b

    # ------------------------------------------------------------------
    def _run_declarative_rule(
        self, rule: _Rule, roles: Mapping[str, SemanticRole], bindings: dict[str, Any]
    ) -> list[InferenceResult]:
        """声明式规则：``rule_premises`` 全部成立则输出 ``rule_conclusions``。

        v0.1 支持三类可判定的前提形状（覆盖 schema 里全部 3 条有前提的规则）：

        * ``edge(A, rel, B)`` —— 在 ``v_relations_all`` 里查边
        * ``closure(A, rel, Ancestor)`` —— 用闭包查上位
        * ``edge(Ancestor, rel, P)`` —— 上位概念上的属性边
        """
        edges = list(self._enumerate_edges(roles))
        if not edges:
            return []
        # 只保留"关于**本句概念**"的结论：邻域扩张会把 `人 is_a 生物`
        # 这类跟句子无关的推导也拉进来（人只是 动物 的另一个下位）。
        focus = {r.head_sense_id for r in roles.values() if r.head_sense_id is not None}
        premise_txt = "；".join(
            f"{kind}({s}, {p}, {o})" for (_q, kind, s, p, o, _n) in rule.premises
        )
        out: list[InferenceResult] = []
        seen: set[tuple[int, str, int]] = set()
        for var_map in self._solve_premises(rule, edges, roles):
            for (_seq, kind, csubj, crel, cobj, conf) in rule.conclusions:
                if kind != "edge" or not crel:
                    continue
                s = var_map.get(str(csubj), csubj) if csubj else None
                o = var_map.get(str(cobj), cobj) if cobj else None
                if not isinstance(s, int) or not isinstance(o, int):
                    continue
                if s not in focus:
                    continue
                key = (s, str(crel), o)
                if key in seen:
                    continue
                # 「是否为新知识」以**基表 relations** 为准：
                # 逆向/对称边由 v_relations_all 在线展开，不算物化新边，但要在链里说明。
                if self._edge_exists_base(s, str(crel), o):
                    continue
                seen.add(key)
                in_view = self._edge_exists(s, str(crel), o)
                out.append(InferenceResult(
                    kind="rule", rule_id=rule.rule_id, rule_name=rule.name,
                    rule_type=rule.rule_type,
                    subject=self._sense_text_any(s),
                    relation=str(crel), obj=self._sense_text_any(o),
                    confidence=float(conf),
                    chain=[
                        f"前提：{premise_txt}",
                        "绑定：" + ", ".join(
                            f"{k}={self._sense_text_any(v)}" for k, v in var_map.items()),
                        f"结论（{rule.name}）：{self._sense_text_any(s)} "
                        f"--{crel}--> {self._sense_text_any(o)}",
                        ("基表 relations 无此边，但 v_relations_all 视图已可在线展开 → "
                         "规则的作用是把结论显式化" if in_view
                         else "基表 relations 与视图均无此边 → 属新增推导知识"),
                    ],
                    note="由声明式前提推导（基表无显式边）",
                ))
        return out

    def _solve_premises(
        self, rule: _Rule, edges: Sequence[tuple[int, str, int]],
        roles: Mapping[str, SemanticRole],
    ) -> list[dict[str, Any]]:
        """**跨前提回溯合一**，返回若干变量绑定解。

        这是本模块唯一需要"真连接"的地方 —— 单边匹配做不到
        ``A is_a B ∧ B is_a C``：两条前提必须落在**不同的边**上，
        且共享变量 ``B`` 的取值要相等。实现为经典的回溯连接：

        * ``edge(S, rel, O)``：在该关系的候选边集合里枚举，逐条合一
        * ``closure(S, rel, O)``：``S`` 已绑定则查闭包填 ``O``；
          ``S`` 未绑定（如 ``inherit`` 规则的 ``A``）则以句中概念为种子逐一尝试

        前提形状不认识时整条规则放弃（返回空解），而不是猜。
        """
        by_rel: dict[str, list[tuple[int, int]]] = {}
        for (s, rel, d) in edges:
            by_rel.setdefault(rel, []).append((s, d))
        seeds = sorted({r.head_sense_id for r in roles.values()
                        if r.head_sense_id is not None})
        sols: list[dict[str, Any]] = []

        def rec(i: int, var: dict[str, Any]) -> None:
            if len(sols) >= 20:
                return
            if i == len(rule.premises):
                sols.append(dict(var))
                return
            _seq, kind, s, p, o, neg = rule.premises[i]
            if kind == "edge" and p:
                for (cs, cd) in by_rel.get(p, ()):
                    v2 = dict(var)
                    if not self._bind_var(v2, s, cs):
                        continue
                    if not self._bind_var(v2, o, cd):
                        continue
                    if neg:
                        continue
                    rec(i + 1, v2)
                return
            if kind == "closure" and p:
                bound = var.get(str(s))
                cands_ = [bound] if isinstance(bound, int) else list(seeds)
                for subj_val in cands_:
                    try:
                        ancs = self._ancestor_ids(int(subj_val), str(p), 6)
                    except ParserError:
                        ancs = []
                    for anc in ancs:
                        v2 = dict(var)
                        if not self._bind_var(v2, s, int(subj_val)):
                            continue
                        if not self._bind_var(v2, o, anc):
                            continue
                        if neg:
                            continue
                        rec(i + 1, v2)
                return
            # 未支持的前提形状 → 放弃整条规则
            return

        rec(0, {})
        return sols

    @staticmethod
    def _bind_var(var: dict[str, Any], name: str | None, value: Any) -> bool:
        """变量合一：首字母大写视为变量，已绑定则要求取值相等。"""
        if name is None:
            return True
        if name[:1].isupper():
            if name in var:
                return var[name] == value
            var[name] = value
            return True
        return True

    def _enumerate_edges(
        self, roles: Mapping[str, SemanticRole], max_hops: int = 2
    ) -> Iterator[tuple[int, str, int]]:
        """列出参与推理的边（含 ``v_relations_all`` 的逆向/对称展开）。

        **必须做一跳邻域扩张**：句子里出现 ``猫``，但传递规则需要的是
        ``动物 is_a 生物`` 这条边 —— 它的两端都不在句面上。
        只查 ``src_id/dst_id ∈ 句面概念`` 会让 ``A is_a B ∧ B is_a C``
        永远配不上（本模块第一版就踩了这个坑）。
        """
        ids = {r.head_sense_id for r in roles.values() if r.head_sense_id is not None}
        if not ids:
            return
        expanded = set(ids)
        for sid in list(ids):
            for rel in (HIERARCHY_RELATION, "instance_of"):
                try:
                    expanded.update(self._ancestor_ids(sid, rel, max_hops))
                    expanded.update(get_descendant_ids(self.conn, sid, rel, max_hops))
                except (ParserError, DatabaseError):
                    continue
        expanded.discard(None)
        if not expanded:
            return
        marks = ",".join("?" for _ in expanded)
        args = list(expanded) + list(expanded)
        rows = self._q(
            f"""SELECT src_id, relation, dst_id FROM v_relations_all
                 WHERE src_kind='sense' AND dst_kind='sense'
                   AND (src_id IN ({marks}) OR dst_id IN ({marks}))""",
            args,
        )
        if not rows:  # pragma: no cover - v_relations_all 缺失时的兜底
            rows = self._q(
                f"""SELECT src_id, relation, dst_id FROM relations
                     WHERE (src_id IN ({marks}) OR dst_id IN ({marks}))""",
                args,
            )
        for (s, rel, d) in rows:
            yield (int(s), str(rel), int(d))

    def _edge_exists(self, src: Any, relation: str, dst: Any) -> bool:
        """边是否可**在线得到**（含 ``v_relations_all`` 的逆向/对称展开）。"""
        if not isinstance(src, int) or not isinstance(dst, int):
            return False
        rows = self._q(
            """SELECT 1 FROM v_relations_all
                WHERE src_kind='sense' AND dst_kind='sense'
                  AND src_id = ? AND relation = ? AND dst_id = ? LIMIT 1""",
            (src, relation, dst),
        )
        return bool(rows)

    def _edge_exists_base(self, src: Any, relation: str, dst: Any) -> bool:
        """边是否**显式存于基表** ``relations``（判断"是否新知识"的口径）。"""
        if not isinstance(src, int) or not isinstance(dst, int):
            return False
        rows = self._q(
            """SELECT 1 FROM relations
                WHERE src_kind='sense' AND dst_kind='sense'
                  AND src_id = ? AND relation = ? AND dst_id = ? LIMIT 1""",
            (src, relation, dst),
        )
        return bool(rows)

    # ------------------------------------------------------------------
    def _sense_text(self, sense_id: Any) -> str:
        """义项 id → 表层词形（渲染推理链用）。"""
        if sense_id is None:
            return ""
        if isinstance(sense_id, bool):
            return str(sense_id)
        if isinstance(sense_id, int):
            info = self._sense_index.get(sense_id)
            return str(info["word"]) if info else str(sense_id)
        return str(sense_id)

    def _sense_text_any(self, v: Any) -> str:
        """任意值 → 可读文本（是义项 id 则换成词形）。"""
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int) and int(v) in self._sense_index:
            return str(self._sense_index[int(v)]["word"])
        return str(v)

    # ------------------------------------------------------------------
    _PARAM_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")

    def _run_condition_rule(
        self, rule: _Rule, bindings: dict[str, Any]
    ) -> list[InferenceResult]:
        """执行 ``inference_rules.condition_sql``（schema 明确设计的"逃生通道"）。

        安全护栏（三条，缺一不可）：

        1. 只接受**单条** ``SELECT``（正则拒绝 ``;`` 多语句与非 SELECT 起始）
        2. 全部 ``:name`` 参数必须能在 ``bindings`` 里找到，否则**跳过**而非报错
        3. 参数一律走 ``sqlite3`` 的命名绑定，绝不字符串拼接
        """
        raw = (rule.condition_sql or "").strip()
        body_sql = raw.rstrip().rstrip(";")          # 允许尾随分号
        if not body_sql:
            return []
        if ";" in body_sql:
            logger.warning("规则 %s 的 condition_sql 含多条语句，已跳过", rule.name)
            return []
        head = body_sql.lstrip("( \n\t").split(None, 1)[0].upper()
        if head not in ("SELECT", "WITH"):
            logger.warning("规则 %s 的 condition_sql 非只读语句（%s），已跳过",
                           rule.name, head)
            return []
        names = set(self._PARAM_RE.findall(body_sql))
        params = {n: bindings[n] for n in names if n in bindings}
        missing = names - set(params)
        if missing:
            logger.debug("规则 %s 缺少参数 %s，跳过", rule.name, sorted(missing))
            return []
        try:
            cur = self.conn.execute(body_sql, params)
            rows = [tuple(r) for r in cur]
            self._stats["db_queries"] += 1
        except sqlite3.Error as exc:
            logger.debug("规则 %s 执行失败：%s", rule.name, exc)
            return []
        if not rows:
            if rule.rule_type != "constraint":
                return []
            # 查不到行有两种成因：真的违例，或**知识库缺该槽位的约束行**。
            # 二者对调用方意义不同，因此把成因写进 note，而不是含糊地说"未通过"。
            # 约束查的是哪个槽位，直接从 SQL 里读，别在代码里另立一张对照表
            slot_m = _CONSTRAINT_SLOT_RE.search(body_sql)
            slot_checked = slot_m.group(1) if slot_m else None
            has_rows = bool(
                slot_checked and self._q(
                    "SELECT 1 FROM relation_constraints WHERE sense_id = ? AND slot = ? LIMIT 1",
                    (bindings.get("verb"), slot_checked),
                )
            )
            return [InferenceResult(
                kind="constraint", rule_id=rule.rule_id, rule_name=rule.name,
                rule_type=rule.rule_type,
                subject=bindings.get("verb_word") or "",
                relation="选择限制", obj="未通过",
                confidence=0.3 if has_rows else 0.2,
                chain=[f"规则「{rule.name}」：{rule.description}",
                       (f"绑定 {self._bind_hint(params)}；relation_constraints 无匹配行 → 违例"
                        if has_rows else
                        f"绑定 {self._bind_hint(params)}；知识库中该槽位没有约束行 → 无法核验")],
                note="选择限制未通过（软警告，不阻断解析）" if has_rows
                     else "知识库缺该槽位的选择限制数据",
            )]
        out: list[InferenceResult] = []
        for row in rows[:5]:
            # constraint 类的 condition_sql 是 `SELECT 1 ...`（存在即通过），
            # 那个 `1` **不是**义项 id —— 不能拿去查 senses 表当词形渲染。
            # 只有 default 类（如量词默认）返回的才是真正的义项外键。
            if rule.rule_type == "constraint":
                obj_txt = "通过"
                chain_tail = (f"绑定 {self._bind_hint(params)}；"
                              f"relation_constraints 命中 → 选择限制满足")
            else:
                obj_txt = self._row_text(row)
                chain_tail = f"绑定 {self._bind_hint(params)} → {obj_txt}"
            out.append(InferenceResult(
                kind="default" if rule.rule_type == "default" else "constraint",
                rule_id=rule.rule_id, rule_name=rule.name, rule_type=rule.rule_type,
                subject=bindings.get("verb_word") or self._sense_text(bindings.get("noun")),
                relation="选择限制" if rule.rule_type == "constraint" else rule.rule_type,
                obj=obj_txt,
                confidence=0.8,
                chain=[f"规则「{rule.name}」：{rule.description}", chain_tail],
                note="condition_sql 逃生通道命中（只读 SELECT + 命名参数绑定）",
                evidence=[list(row)],
            ))
        return out

    def _bind_hint(self, params: Mapping[str, Any]) -> str:
        """把参数字典渲染成人话（义项 id 换成词形）。"""
        parts = []
        for k, v in params.items():
            parts.append(f"{k}={self._sense_text(v) if isinstance(v, int) else v}")
        return ", ".join(parts)

    def _row_text(self, row: Sequence[Any]) -> str:
        parts: list[str] = []
        for v in row:
            if isinstance(v, int) and v in self._sense_index:
                parts.append(self._sense_text(v))
            else:
                parts.append(str(v))
        return "/".join(parts)

    # ------------------------------------------------------------------
    def _classifier_suggestions(
        self, roles: Mapping[str, SemanticRole]
    ) -> list[InferenceResult]:
        """量词默认值推理（``classifier_class_defaults`` / ``noun_classifier_pairs``）。"""
        out: list[InferenceResult] = []
        for r in roles.values():
            if r.head_sense_id is None or r.role in ("述题", "命题"):
                continue
            pairs = self._noun_classifiers.get(r.head_sense_id, [])
            if pairs:
                cls_, pref, is_def = pairs[0]
                out.append(InferenceResult(
                    kind="default", subject=r.entity, relation="classifier",
                    obj=self._sense_text(cls_), confidence=float(pref),
                    chain=[f"{r.entity} 的显式量词搭配：{self._sense_text(cls_)}"
                           f"（preference={pref:.2f}{'，默认' if is_def else ''}）"],
                    rule_name="量词默认", rule_type="default",
                ))
                continue
            cats = self._category_closure(r.head_sense_id)
            for (cat, csid, pref) in self._classifier_defaults:
                if cat in cats:
                    out.append(InferenceResult(
                        kind="default", subject=r.entity, relation="classifier",
                        obj=self._sense_text(csid), confidence=float(pref) * 0.8,
                        chain=[f"{r.entity} 属于类别「{cat}」",
                               f"类别「{cat}」的默认量词：{self._sense_text(csid)}"],
                        rule_name="量词默认", rule_type="default",
                    ))
                    break
        return out

    # =====================================================================
    # 6.9 端到端解析
    # =====================================================================
    def parse(
        self,
        text: str,
        context: ParserContext | Mapping[str, Any] | None = None,
        *,
        persist: bool | None = None,
    ) -> ParseResult:
        """端到端解析：``tokenize → match_pattern → fill_slots → annotate → query``。

        :param text: 待解析文本（可含多个分句）
        :param context: 篇章上下文；``None`` 时沿用 :attr:`context`
        :param persist: 是否把结果写入 ``utterances``/``anaphora``（默认取构造参数）
        :returns: :class:`ParseResult`；**匹配失败不抛异常**，而是返回
            ``pattern_id is None`` 的结果，且 ``diagnostics`` 里带 top-3 候选与失败原因
        """
        t0 = time.perf_counter()
        ctx = ParserContext.from_any(context) if context is not None else self.context
        self.context = ctx
        do_persist = self.persist if persist is None else bool(persist)

        result = ParseResult(text=text)
        result.context_used = {
            "doc_id": ctx.doc_id, "speaker": ctx.speaker, "listener": ctx.listener,
            "topic": ctx.topic, "turn_index": ctx.turn_index,
        }

        try:
            tokens = self.tokenize(text)
        except ParserError as exc:
            logger.error("分词失败：%s", exc)
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.diagnostics = [PatternAttempt(0, "<tokenize>", "", False, reason=str(exc))]
            return result
        result.tokens = tokens

        match = self.match_pattern(tokens)
        attempts = list(getattr(self, "last_attempts", []))
        if match is None:
            result.diagnostics = attempts[:3]
            result.confidence = 0.0
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("未匹配任何句式：%r（top-3 原因见 diagnostics）", text)
            return result
        result.diagnostics = [a for a in attempts if a.pattern_id != match.pattern_id][:3]

        result.match = match
        result.pattern_id = match.pattern_id
        result.pattern_name = match.name
        result.category = match.category
        result.semantics = match.semantics
        result.mood = self._detect_mood(match, tokens)

        filled = self.fill_slots(match, tokens, ctx)
        result.filled_slots = filled
        roles = self.annotate_semantic_roles(filled, match, tokens)
        result.roles = roles
        result.inferences = self.query_relations(roles, match, tokens)
        result.confidence = self._confidence(match, filled, result.inferences)

        # 顺序要紧：先并入篇章状态（``_remember`` 会把 ``state.count`` 增到
        # 本句序号），再落盘。反过来的话首次落盘会写出 ``seq=0``，撞上
        # ``utterances`` 的 ``CHECK (seq >= 1)`` 而静默降级为内存态。
        self._remember(ctx, result, match)
        if do_persist:
            self._persist(ctx, result)

        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def parse_document(
        self, texts: Sequence[str], *, doc_id: str = "default",
        context: ParserContext | Mapping[str, Any] | None = None,
        persist: bool | None = None,
    ) -> list[ParseResult]:
        """逐句解析并维护话题链（零形回指消解依赖句序）。"""
        base = ParserContext.from_any(context) if context is not None else self.context
        ctx = ParserContext(
            doc_id=doc_id, speaker=base.speaker, listener=base.listener,
            topic=base.topic, context_type=base.context_type,
            speaker_entity=base.speaker_entity, listener_entity=base.listener_entity,
        )
        out: list[ParseResult] = []
        for i, txt in enumerate(texts, start=1):
            ctx.turn_index = i
            out.append(self.parse(txt, ctx, persist=persist))
        self.context = ctx
        return out

    # ------------------------------------------------------------------
    def _detect_mood(self, match: PatternMatch, tokens: Sequence[Token]) -> str:
        """判定句类：模板 ``mood`` 优先，其次句末标点与语气词。"""
        if match.sentence_terminator in ("？", "?"):
            return "疑问"
        if match.sentence_terminator in ("！", "!"):
            for m, cues in _MOOD_MARKERS.items():
                if any(t.text in cues for t in tokens):
                    return m
            return "感叹"
        for t in tokens:
            for m, cues in _MOOD_MARKERS.items():
                if t.text in cues and t.pos in ("y", "d"):
                    return m
        return match.mood or "陈述"

    def _confidence(
        self, match: PatternMatch, filled: Mapping[str, SlotFill],
        inferences: Sequence[InferenceResult],
    ) -> float:
        """综合置信度：句式分 / 覆盖率 / 约束通过率 / 槽位补齐率 / 推理可得性。"""
        c_match = min(1.0, max(0.0, match.score) / 30.0)
        c_cov = match.coverage
        sat = sum(1 for c in match.constraint_checks if c.ok)
        vio = sum(1 for c in match.constraint_checks if not c.ok)
        c_con = sat / (sat + vio) if (sat + vio) else 1.0
        required = [n for n in match.slot_spans] or [f.slot_name for f in filled.values() if f.source == "matched"]
        if required:
            filled_req = len([n for n in required if n in filled])
            c_role = filled_req / len(required)
        else:
            c_role = 0.5
        c_inf = 0.6 + 0.4 * min(1.0, len(inferences) / 3.0)
        conf = (0.35 * c_match + 0.15 * c_cov + 0.20 * c_con
                + 0.20 * c_role + 0.10 * c_inf)
        for f in filled.values():
            if f.source == "ellipsis":
                conf *= 0.9 if f.tokens and f.tokens[0].entry_id is not None else 0.6
        return round(max(0.0, min(1.0, conf)), 4)

    # ------------------------------------------------------------------
    def _remember(
        self, ctx: ParserContext, result: ParseResult, match: PatternMatch | None
    ) -> None:
        """把本次结果并入内存篇章状态（供 ``previous_subject`` 等策略使用）。"""
        state = self._discourse.setdefault(ctx.doc_id, _Discourse(ctx.doc_id))
        state.count += 1
        for name, r in result.roles.items():
            if not r.tokens or r.head_sense_id is None:
                if not r.tokens:
                    continue
            state.push(_Mention(r.entity, (r.head.pos if r.head else "n"),
                                r.head_sense_id, r.role, state.count))
        if result.semantics and "处置" in (result.semantics or "") and result.filled_slots.get("P"):
            pf = result.filled_slots["P"]
            state.topic = _Mention(pf.text, (pf.head.pos if pf.head else "n"),
                                   pf.head_sense_id, "受事", state.count)

    def _persist(self, ctx: ParserContext, result: ParseResult) -> None:
        """可选落盘：写 ``utterances`` 与省略回填的 ``anaphora`` 记录。

        注意：本模块**默认不写库**。开启 ``persist`` 时若连接为只读，会捕获异常并降级为
        仅内存态，不影响解析结果。
        """
        if getattr(self.conn, "logos_meta", {}).get("read_only"):
            logger.warning("连接为只读，persist 已自动降级为内存态")
            return
        from ..db.connection import transaction  # 延迟导入，避免层间硬依赖

        try:
            with transaction(self.conn):
                state = self._discourse_state(ctx.doc_id)
                seq = (state.count if state else 0)
                cur = self.conn.execute(
                    """INSERT INTO utterances(doc_id, seq, text, speaker, mood, pattern_id)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(doc_id, seq) DO UPDATE SET
                           text=excluded.text, speaker=excluded.speaker,
                           mood=excluded.mood, pattern_id=excluded.pattern_id""",
                    (ctx.doc_id, seq, result.text, ctx.speaker, result.mood,
                     result.pattern_id),
                )
                uid = self._last_utterance_id(ctx.doc_id, seq)
                if uid is None:
                    del cur
                    return
                for fill in result.filled_slots.values():
                    if fill.source != "ellipsis":
                        continue
                    # 记下**填回的表层形式与先行节点**，否则落盘的数据无法
                    # 被 _load_discourse_from_db 还原（use_db_context 的往返依赖它）。
                    self.conn.execute(
                        """INSERT INTO anaphora(utterance_id, slot_role, form, form_type,
                                                antecedent_node_id, chain_id, confidence)
                           VALUES(?,?,?,?,?,?,?)""",
                        (uid, self._role_of_slot(result, fill.slot_name),
                         fill.text or "∅", self._anaphora_form_type(fill),
                         fill.head_sense_id, None, fill.confidence),
                    )
                del cur
        except (sqlite3.Error, DatabaseError) as exc:
            logger.warning("persist 写入失败，已降级为内存态：%s", exc)

    @staticmethod
    def _anaphora_form_type(fill: SlotFill) -> str:
        """把回填结果映射到 ``anaphora.form_type`` 的取值域。

        该列有 CHECK 约束（pronoun/zero/definite_np/demonstrative/proper_noun/
        collective）。原实现一律写 ``'zero'`` —— 哪怕实际回填的是"他"这样的人称
        代词，导致落盘数据与事实不符、下游也无法区分零形与代词回指。
        """
        head = fill.head
        if head is None or not head.text or head.text == "∅":
            return "zero"
        if head.pos == "r":
            return "pronoun"
        if head.text[0:1] in _DEMONSTRATIVES:
            return "demonstrative"
        return "definite_np"

    def _last_utterance_id(self, doc_id: str, seq: int) -> int | None:
        rows = self._q(
            "SELECT utterance_id FROM utterances WHERE doc_id=? AND seq=?", (doc_id, seq)
        )
        return int(rows[0][0]) if rows else None

    @staticmethod
    def _role_of_slot(result: ParseResult, slot_name: str) -> str | None:
        for r in result.roles.values():
            if r.slot_name == slot_name:
                return r.role
        return None

    # ------------------------------------------------------------------
    def explain(self, text: str, context: Any = None) -> str:
        """解析并返回人类可读的完整报告（调试与文档示例用）。"""
        return self.parse(text, context).render()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"<SyntaxParser patterns={len(self._patterns)} "
                f"lexicon={len(self._lexicon)} discourse={len(self._discourse)}>")


# ===========================================================================
# 7. 内部小工具
# ===========================================================================
_HUMAN_PRONOUNS = frozenset({"我", "你", "他", "她", "它", "我们", "你们", "他们"})

_POS_PREF: dict[str, frozenset[str]] = {
    "谓语": _VERBAL_POS,
    "主语": _NOMINAL_POS,
    "宾语": _NOMINAL_POS,
    "话题": _NOMINAL_POS,
    "中心语": frozenset(_NOMINAL_POS | {"v"}),
    "定语": frozenset(_NOMINAL_POS | {"a"}),
    "补语": frozenset(_NOMINAL_POS | {"v", "a"}),
    "状语": frozenset(_NOMINAL_POS | _VERBAL_POS | {"p", "c"}),
    "独立语": frozenset(_NOMINAL_POS | _VERBAL_POS | {"p", "c", "e", "o"}),
}


def _as_token_list(v: Any) -> list[Token]:
    """把 ``Token | list[Token] | SlotFill`` 统一成 ``list[Token]``。"""
    if v is None:
        return []
    if isinstance(v, Token):
        return [v]
    if isinstance(v, SlotFill):
        return list(v.tokens)
    if isinstance(v, (list, tuple)):
        return [x for x in v if isinstance(x, Token)]
    return []


def _coerce_role(key: str, v: Any) -> SemanticRole:
    """把外部传入的任意"角色值"提升成 :class:`SemanticRole`。"""
    if isinstance(v, SemanticRole):
        return v
    toks = _as_token_list(v)
    return SemanticRole(role=key, slot_name=key, tokens=toks)


def _default_db_path() -> Any:
    from ..db.connection import DEFAULT_DB_PATH

    return DEFAULT_DB_PATH


# ===========================================================================
# 8. 模块级便捷函数
# ===========================================================================
def parse(
    text: str,
    conn: sqlite3.Connection | None = None,
    *,
    db_path: Any = None,
    context: Any = None,
    persist: bool | None = None,
    **parser_kwargs: Any,
) -> ParseResult:
    """一次性解析（自建连接与解析器，用完即关）。

    :param conn: 已有连接；``None`` 时按 ``db_path`` 新建**只读**连接
    :param db_path: 库路径（``None`` → :data:`logos.db.connection.DEFAULT_DB_PATH`）
    :param parser_kwargs: 透传给 :class:`SyntaxParser`

    ::

        from logos.parser import parse
        r = parse("猫把鱼吃了。", db_path="logos.db")
        print(r.pattern_name, r.roles["受事"].text)   # 把字句 鱼
    """
    if conn is not None:
        return SyntaxParser(conn, context=context, **parser_kwargs).parse(
            text, persist=persist
        )
    from ..db.connection import connect

    path = db_path if db_path is not None else _default_db_path()
    with connect(path, read_only=True, must_exist=True) as c:
        return SyntaxParser(c, context=context, **parser_kwargs).parse(text, persist=persist)


def tokenize(
    text: str,
    conn: sqlite3.Connection | None = None,
    *,
    db_path: Any = None,
    **parser_kwargs: Any,
) -> list[Token]:
    """只做分词（不匹配句式）。参数含义同 :func:`parse`。"""
    if conn is not None:
        return SyntaxParser(conn, **parser_kwargs).tokenize(text)
    from ..db.connection import connect

    path = db_path if db_path is not None else _default_db_path()
    with connect(path, read_only=True, must_exist=True) as c:
        return SyntaxParser(c, **parser_kwargs).tokenize(text)


def sense_text(parser: SyntaxParser, sense_id: int) -> str:
    """取义项的表层词形（供外部渲染推理链）。"""
    return parser._sense_text(sense_id)


def build_parser(
    db_path: Any = None,
    *,
    read_only: bool = True,
    must_exist: bool = True,
    **parser_kwargs: Any,
) -> tuple["SyntaxParser", Any]:
    """建连接 + 建解析器，返回 ``(parser, conn)`` 以便调用方自行关闭。

    ::

        parser, conn = build_parser("logos.db")
        try:
            print(parser.explain("鱼被猫吃了。"))
        finally:
            conn.close()
    """
    from ..db.connection import get_connection

    path = db_path if db_path is not None else _default_db_path()
    conn = get_connection(path, read_only=read_only, must_exist=must_exist)
    return SyntaxParser(conn, **parser_kwargs), conn