"""Logos 句法-语义解析层。

对外只暴露 :mod:`logos.parser.syntax_parser` 里经过验证的公共接口::

    from logos.parser import SyntaxParser, tokenize, parse

    with connect("logos.db", read_only=True) as conn:
        parser = SyntaxParser(conn)
        result = parser.parse("猫把鱼吃了。")
        for role, r in result.roles.items():
            print(role, r.text)

需要自己管连接生命周期时，用 :func:`build_parser` 一次拿到 ``(parser, conn)``::

    parser, conn = build_parser("logos.db")   # read_only=True 且库必须已存在
    try:
        print(parser.explain("鱼被猫吃了。"))
    finally:
        conn.close()

分层约定（与本项目其余各层一致）
--------------------------------
* ``logos.db.connection`` —— 连接层（PRAGMA / 事务 / 缓存）
* ``logos.db.closure``    —— 传递闭包维护层（唯一可写 ``closure`` 表的地方）
* ``logos.parser``        —— **只读** 消费层：读句式模板、槽位约束、义素、关系与闭包，
  不做任何写入（``persist=True`` 时写入 ``utterances``/``anaphora`` 属显式可选行为）。
"""

from __future__ import annotations

from .syntax_parser import (
    CLAUSE_PUNCT,
    CONSTRAINT_WEIGHTS,
    CONTENT_POS,
    DEFAULT_MAX_SLOT_TOKENS,
    HARD_CONSTRAINT_KINDS,
    PUNCT_POS,
    ROLE_TO_CONSTRAINT_SLOT,
    SENTENCE_PUNCT,
    SKIPPABLE_FW_CATEGORIES,
    SKIPPABLE_POS,
    SPEC_WEIGHTS,
    ConstraintCheck,
    InferenceResult,
    MatchError,
    ParserContext,
    ParserError,
    ParseResult,
    PatternAttempt,
    PatternMatch,
    SemanticRole,
    SlotFill,
    SyntaxParser,
    TemplateCompileError,
    Token,
    TokenizeError,
    build_parser,
    compile_template,
    parse,
    sense_text,
    tokenize,
)

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
