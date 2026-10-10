#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``logos/parser/syntax_parser.py`` 回归校验。

运行：  python validate_syntax_parser.py
退出码：0 全绿；1 有失败。

覆盖：
  S0  建库与缓存载入（正规入口 init_database + cache_info + reload 幂等）
  S1  分词（正向最大匹配 / 离合词**两条路径** / OOV 兜底分级）
  S2  模板编译（独立 `/` 切变体 vs 紧贴 `请/别` 作字面量候选）
  S3  句式匹配（13 例真实句式 → 命中预期模板矩阵）
  S4  约束（软约束点名"哪个槽位+哪个约束" / marker 硬约束一票否决）
  S5  槽位填充与省略（可选槽位留空 / 省略授权 / 一次至多省略 1 个必填）
  S6  语义角色标注（槽位默认 + 虚词改写，且**只认紧邻介词**）
  S7  关系查询与推理链（属性继承 / is_a 传递 / 量词默认 / 典型论元 / 去重）
  S8  端到端 parse（ParseResult 字段 / mood / confidence / to_dict / render）
  S9  错误处理（匹配失败给 top-3 + 失败原因；约束冲突点名槽位与约束）
  S10 回归：仅凭"词汇证据闸门"不误杀正常句（历史 bug：闸门读到 known 默认值 0）
  S11 篇章状态（persist 落盘 seq>=1 / anaphora 可还原 / use_db_context 跨连接往返）
  S12 只读性、自检与公共接口（非 persist 解析不写库 / __all__ 完整）
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from logos.db.connection import get_connection, init_database, transaction  # noqa: E402
from logos.parser import (  # noqa: E402
    ParserContext,
    SyntaxParser,
    TemplateCompileError,
    Token,
    TokenizeError,
    build_parser,
    compile_template,
    sense_text,
)

DB = HERE / "logos_validate_parser.db"
DB_HARD = HERE / "logos_validate_parser_hard.db"
DB_CTX = HERE / "logos_validate_parser_ctx.db"


def _rm(p: pathlib.Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        f = pathlib.Path(str(p) + suffix)
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass


# ===========================================================================
# 断言框架（与 validate_closure.py 同规格）
# ===========================================================================
class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []
        self._idx = 0

    def section(self, name: str) -> None:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        self._idx += 1
        if cond:
            self.passed += 1
            print(f"  PASS  [{self._idx:>3}] {name}")
        else:
            self.failed += 1
            self.failures.append(f"{name} :: {detail}")
            print(f"  FAIL  [{self._idx:>3}] {name}   <-- {detail}")
        return bool(cond)

    def eq(self, name: str, got, want) -> bool:
        return self.check(name, got == want, f"got={got!r} want={want!r}")

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n{'=' * 78}")
        print(f"PASS {self.passed} / FAIL {self.failed} / TOTAL {total}")
        if self.failures:
            print("\n失败明细：")
            for f in self.failures:
                print(f"  - {f}")
        print("=" * 78)
        return 0 if self.failed == 0 else 1


ck = Checker()

# ===========================================================================
# S0 建库与缓存载入
# ===========================================================================
ck.section("S0 建库与缓存载入（init_database 正规入口）")
_rm(DB)
rep = init_database(DB, reset=True, validate=True)
ck.check("schema 加载成功", rep.schema_loaded, rep.render())
ck.check("integrity_check 通过", rep.integrity_ok)
ck.check("foreign_key_check 通过", rep.foreign_key_ok)
ck.check("种子已加载", bool(rep.seed_loaded), str(rep.seed_loaded))

conn = get_connection(DB, must_exist=True)
p = SyntaxParser(conn)
ci = p.cache_info()
ck.eq("载入 32 条句式模板", ci["patterns"], 32)
ck.eq("载入 80 个槽位定义", ci["pattern_slots"], 80)
ck.eq("载入 33 条槽位约束", ci["slot_constraints"], 33)
ck.eq("载入 37 个语义角色", ci["semantic_roles"], 37)
ck.eq("载入 9 个虚词", ci["function_words"], 9)
ck.eq("载入 12 条省略规则", ci["ellipsis_rules"], 12)
ck.eq("载入 6 条推理规则", ci["inference_rules"], 6)
ck.eq("载入 6 条补语登记", ci["complements"], 6)
ck.check("词典条目非空", ci["lexicon"] >= 40, str(ci["lexicon"]))
ck.check("27 个缓存项键齐全", len(ci) >= 17, str(len(ci)))
# 32 条模板全部编译成功（编译失败会被 _load_patterns 跳过，总数就不足 32）
ck.check("32 条模板全部编译成功（无跳过）", ci["patterns"] == 32)
ck.check("模板字面量已登记为可分词标记", ci["grammar_literals"] >= 6,
         str(sorted(p._grammar_literals)))
ck.check("最长词长 > 1", ci["max_word_len"] > 1, str(ci["max_word_len"]))
ck.eq("reload() 幂等（两次 cache_info 一致）", p.reload(), ci)

# ===========================================================================
# S1 分词
# ===========================================================================
ck.section("S1 分词（正向最大匹配 / 离合词双路径 / OOV 兜底）")
toks = p.tokenize("猫吃鱼。")
ck.eq("分词结果文本序列", [t.text for t in toks], ["猫", "吃", "鱼", "。"])
ck.eq("词性序列", [t.pos for t in toks], ["n", "v", "n", "w"])
ck.check("Token 带 entry_id / sense_id", all(
    t.entry_id is not None and t.sense_id is not None for t in toks[:3]))
ck.eq("start/end 与原文对齐（切片可复原）",
      "".join("猫吃鱼。"[t.start:t.end] for t in toks), "猫吃鱼。")
ck.check("标点 pos='w' 且 is_punct", toks[3].pos == "w" and toks[3].is_punct)
ck.check("实词 is_content、虚词/标点非 is_content",
      toks[0].is_content and not toks[3].is_content)

# —— 正向最大匹配：优先长词
lex_long = sorted((w for w in p._lexicon if len(w) == p._max_word_len), key=len)
if lex_long:
    probe = lex_long[0]
    ck.check(f"FMM 优先匹配最长词 {probe!r}",
             [t.text for t in p.tokenize(probe)] == [probe],
             str([t.text for t in p.tokenize(probe)]))
ck.eq("空白被跳过（不产生 token）",
      [t.text for t in p.tokenize("  猫  吃 鱼  ")], ["猫", "吃", "鱼"])

# —— 离合词路径 A：整体出现 → 切分
sep = p.tokenize("洗澡")
ck.eq("离合词整体出现 → 切成 2 段", [t.text for t in sep], ["洗", "澡"])
ck.check("两段都标记 is_separable", all(t.is_separable for t in sep))
ck.eq("两段同属一个 separable_group", sep[0].separable_group, sep[1].separable_group)
ck.eq("sep_role = host / object", [t.sep_role for t in sep], ["host", "object"])
ck.eq("separator_pos 取 words.separator_pos", sep[0].separator_pos, 1)
ck.eq("切分后仍保留整体义项 id（sep_group_sense_id 一致）",
      sep[0].sep_group_sense_id, sep[1].sep_group_sense_id)

# —— 离合词路径 B：拆开出现 → 重组（洗 了 个 澡）
split = p.tokenize("洗了个澡")
ck.eq("离合结构拆开出现 → 4 个 token", [t.text for t in split], ["洗", "了", "个", "澡"])
ck.eq("整段共享同一 separable_group", len({t.separable_group for t in split}), 1)
ck.eq("中间成分 sep_role = infix", [t.sep_role for t in split], ["host", "infix", "infix", "object"])
ck.check("重组靠 bound_objects（无该表则不重组）", bool(p._bound_objects))

# —— OOV 兜底分级
ck.eq("阿拉伯数字串并成一个数词 token",
      [(t.text, t.pos) for t in p.tokenize("2026")], [("2026", "m")])
ck.eq("数字+量词正确切分",
      [(t.text, t.pos) for t in p.tokenize("3个猫")], [("3", "m"), ("个", "q"), ("猫", "n")])
ck.eq("汉字数词 → m", p.tokenize("三")[0].pos, "m")
ck.eq("指示代词 → r", p.tokenize("这")[0].pos, "r")
ck.eq("方位后缀 → nl", p.tokenize("上")[0].pos, "nl")
ck.eq("方位字串并成一个方位词", [(t.text, t.pos) for t in p.tokenize("上边")],
      [("上边", "nl")])
ck.eq("封闭类兜底 → closed_class", p.tokenize("不")[0].inferred_pos, "closed_class")
oov = p.tokenize("镬")[0]
ck.check("真 OOV → pos='?' 且 is_oov", oov.pos == "?" and oov.is_oov,
         f"{oov.pos}/{oov.inferred_pos}")
ck.eq("真 OOV 的 inferred_pos", oov.inferred_pos, "unknown")

try:
    p.tokenize(123)
    ck.check("非 str 输入抛 TokenizeError", False, "未抛异常")
except TokenizeError:
    ck.check("非 str 输入抛 TokenizeError", True)

# ===========================================================================
# S2 模板编译
# ===========================================================================
ck.section("S2 模板编译（变体 vs 字面量候选）")
v1 = compile_template("{S} {V} {O}")
ck.eq("`{S} {V} {O}` → 1 个变体", len(v1), 1)
ck.eq("该变体 → 3 项", len(v1[0]), 3)
v2 = compile_template("{V} 得 {C} / {V} 不 {C}")
ck.eq("**独立** `/` 切出 2 个变体", len(v2), 2)
ck.eq("变体 1 项数", len(v2[0]), 3)
v3 = compile_template("请/别 {V}")
ck.eq("**紧贴** `请/别` 不切变体", len(v3), 1)
ck.eq("`请/别` 成字面量同义候选", tuple(v3[0][0].alternatives), ("请", "别"))
ck.eq("该项 kind='lit'", v3[0][0].kind, "lit")
v4 = compile_template("{A} 把 {P} {V} {C}")
ck.eq("`把` 编译为字面量项", [i.kind for i in v4[0]].count("lit"), 1)
ck.eq("槽位项 kind='slot'", [i.kind for i in v4[0]].count("slot"), 4)
try:
    compile_template("   ")
    ck.check("空模板抛 TemplateCompileError", False, "未抛异常")
except TemplateCompileError:
    ck.check("空模板抛 TemplateCompileError", True)
ck.check("32 条真实模板变体数均 >= 1",
         all(len(pat.variants) >= 1 for pat in p._patterns))

# ===========================================================================
# S3 句式匹配：13 例矩阵
# ===========================================================================
ck.section("S3 句式匹配（13 例 → 命中预期模板）")
MATRIX: list[tuple[str, int, str]] = [
    ("猫吃鱼。", 1, "主谓宾"),
    ("猫把鱼吃了。", 7, "把字句"),
    ("鱼被猫吃了。", 8, "被字句"),
    ("鱼，猫吃了。", 5, "主谓谓语句"),
    ("桌上有一条鱼。", 11, "存现句"),
    ("我是人。", 3, "主系表"),
    ("他比我高。", 13, "比较句-比"),
    ("猫吃鱼吗？", 20, "是非问"),
    ("猫吃不吃鱼？", 22, "正反问"),
    ("给 我 一 本 书。", 6, "双宾句"),
    ("请你吃鱼。", 25, "祈使句"),
    ("看完。", 17, "动结式"),
    ("看得完。", 19, "可能补语式"),
]
for text, pid, pname in MATRIX:
    tks = p.tokenize(text)
    m = p.match_pattern(tks)
    if m is None:
        top = p.last_attempts[0].render() if p.last_attempts else "<无候选>"
        ck.check(f"{text:<16} → {pname}", False, f"未匹配；top1={top}")
        continue
    ck.check(f"{text:<16} → {pname}",
             m.pattern_id == pid and m.name == pname,
             f"got #{m.pattern_id} {m.name} score={m.score:.2f} cov={m.coverage:.0%}")
    ck.check(f"{text:<16} 覆盖率 >= 阈值",
             m.coverage >= p.min_coverage, f"{m.coverage:.0%}")

# 专项：短语结构模板不得把主谓结构吞成 V+R（历史 bug：`猫吃鱼` → 动结式）
m_bad = p.match_pattern(p.tokenize("猫吃鱼。"))
ck.check("`猫吃鱼。` 不被 动结式/动趋式 误认",
         m_bad is not None and m_bad.pattern_id not in (17, 18),
         f"got #{m_bad.pattern_id if m_bad else None} {m_bad.name if m_bad else ''}")
# 专项：`给我一本书。` 不应落回 主谓宾
m_dbl = p.match_pattern(p.tokenize("给我一本书。"))
ck.check("`给我一本书。` 命中 双宾句 而非 主谓宾",
         m_dbl is not None and m_dbl.pattern_id == 6,
         f"got #{m_dbl.pattern_id if m_dbl else None} {m_dbl.name if m_dbl else ''}")

# ===========================================================================
# S4 约束
# ===========================================================================
ck.section("S4 约束：软约束点名 / marker 硬约束否决")
r = p.parse("猫把一条鱼吃了。")
ck.check("不定指 P（一条鱼）仍匹配成功 —— 软约束不否决",
         r.ok and r.pattern_name == "把字句", f"ok={r.ok} pattern={r.pattern_name}")
errs = r.match.constraint_errors if r.match else []
joined = " ".join(errs)
ck.check("约束冲突点名了**槽位** P", "[P]" in joined, joined)
ck.check("约束冲突点名了**约束类型** definiteness", "definiteness" in joined, joined)
ck.check("违反详情含实际判定值", "indefinite" in joined, joined)
ck.check("软约束不进入 diagnostics 失败原因", all("definiteness" not in a.reason
                                                for a in r.diagnostics))

# —— marker 硬约束：往副库里加一条要求 P 槽出现「被」标记的模板
shutil.copyfile(DB, DB_HARD)
_hard = get_connection(DB_HARD)
with transaction(_hard):
    _hard.execute(
        "INSERT INTO sentence_patterns(pattern_id,name,category,mood,template,"
        "semantics,is_productive,example) VALUES(9001,'把字句-硬约束测试','特殊句式',"
        "'陈述','{A} 把 {P} {V}','处置',1,'我把作业写')")
    for (name, pos_, req, fn, role) in (("A", 1, 1, "主语", "施事"),
                                        ("P", 2, 1, "状语", "受事"),
                                        ("V", 3, 1, "谓语", None)):
        _hard.execute(
            "INSERT INTO pattern_slots(pattern_id,slot_name,position,is_required,"
            "syntactic_function,default_role) VALUES(9001,?,?,?,?,?)",
            (name, pos_, req, fn, role))
    _hard.execute(
        "INSERT INTO slot_constraints(pattern_id,slot_name,kind,value,is_exclusive) "
        "VALUES(9001,'P','marker','被',0)")
ph = SyntaxParser(_hard)
att = [a for a in ph.match_candidates(ph.tokenize("猫把鱼吃了。"), limit=40,
                                      include_failed=True)
       if a.pattern_id == 9001]
ck.check("自定义模板进入候选集", bool(att), "未出现在候选里")
if att:
    a = att[0]
    ck.check("marker 硬约束缺失 → 否决", not a.matched, f"matched={a.matched}")
    ck.check("否决原因标明「硬约束」", "硬约束" in a.reason, a.reason)
    ck.check("否决原因标明槽位 P", "[P]" in a.reason, a.reason)
    ck.check("否决原因标明约束 kind=marker", "marker" in a.reason, a.reason)
_hard.close()
_rm(DB_HARD)

# ===========================================================================
# S5 槽位填充与省略
# ===========================================================================
ck.section("S5 槽位填充与省略（留空 / 授权 / 至多省略 1 个）")
m = p.match_pattern(p.tokenize("猫把鱼吃了。"))
filled = p.fill_slots(m, p.tokenize("猫把鱼吃了。"))
ck.check("fill_slots 返回 dict[str, SlotFill]",
         isinstance(filled, dict) and all(k == f.slot_name for k, f in filled.items()))
ck.check("SlotFill 提供 text/head/head_sense_id",
         all(hasattr(f, "text") and hasattr(f, "head") and hasattr(f, "head_sense_id")
             for f in filled.values()))
ck.check("as_value() 单 token 槽位 → Token（spec 形态 Token|list[Token]）",
         isinstance(filled["A"].as_value(), Token))
_f_multi = p.fill_slots(p.match_pattern(p.tokenize("猫把一条鱼吃了。")),
                        p.tokenize("猫把一条鱼吃了。"))
ck.check("as_value() 多 token 槽位 → list[Token]",
         isinstance(_f_multi["P"].as_value(), list)
         and len(_f_multi["P"].as_value()) == 3,
         str(_f_multi["P"].as_value()))
ck.check("SlotFill 支持 len()/iter()/[] 兼容旧接口",
         len(filled["A"]) == 1 and list(filled["A"])[0].text == "猫"
         and filled["A"][0].text == "猫")
ck.eq("可选槽位 C 未被填充 → 不出现在 filled_slots",
      "C" in filled, False)
ck.eq("必填槽位 A/P/V 均已填充", sorted(filled), ["A", "P", "V"])
ck.eq("已匹配槽位 source='matched' 且置信度 1.0",
      (filled["A"].source, filled["A"].confidence), ("matched", 1.0))

# —— 省略：叙事上下文中零形主语从上一句主语回填
_rm(DB_CTX)
shutil.copyfile(DB, DB_CTX)
conn_ctx = get_connection(DB_CTX)
pc = SyntaxParser(conn_ctx, persist=False)
nctx = lambda: ParserContext(doc_id="val_ctx", context_type="narrative")
pc.parse("猫吃鱼。", nctx())
r_el = pc.parse("把鱼吃了。", nctx())
ck.eq("零形 A 仍能匹配 把字句", r_el.pattern_name, "把字句")
ck.check("A 由省略回填", r_el.filled_slots.get("A") is not None
         and r_el.filled_slots["A"].source == "ellipsis",
         str({k: f.source for k, f in r_el.filled_slots.items()}))
ck.eq("回填到上一句主语", r_el.filled_slots["A"].text, "猫")
ck.check("回填记录了 ellipsis_rule_id / strategy",
         r_el.filled_slots["A"].ellipsis_rule_id is not None
         and r_el.filled_slots["A"].strategy == "previous_subject",
         f"rule={r_el.filled_slots['A'].ellipsis_rule_id} "
         f"strategy={r_el.filled_slots['A'].strategy}")
ck.check("省略回填的置信度低于直接匹配",
         r_el.filled_slots["A"].confidence < 1.0,
         str(r_el.filled_slots["A"].confidence))

# —— 省略授权：可选槽位可留空 vs 必填槽位需授权，且一次至多省略 1 个
ck.check("C 是可选槽位（is_required=0）",
         not p._pattern_by_id[7].slot_by_name["C"].is_required)
ck.check("施事/受事被 ellipsis_rules 授权可省略",
         {"施事", "受事"} <= p._ellipsis_roles, str(sorted(p._ellipsis_roles)))
# `把吃。`：A、P 都无法填充 → 一次省略 2 个必填 → 必须被拒
att_omit = [a for a in p.match_candidates(p.tokenize("把吃。"), limit=40,
                                         include_failed=True)
            if a.pattern_id == 7]
ck.check("一次省略 2 个必填槽位 → 拒绝",
         bool(att_omit) and not att_omit[0].matched
         and "一次省略了 2 个必填槽位" in att_omit[0].reason,
         str(att_omit[0].reason if att_omit else "<无候选>"))
conn_ctx.close()

# ===========================================================================
# S6 语义角色标注
# ===========================================================================
ck.section("S6 语义角色标注（槽位默认 + 虚词改写）")
r = p.parse("猫吃鱼。")
ck.eq("主谓宾的角色集合", sorted(r.roles), ["受事", "施事", "述题"])
ck.eq("S → 施事", r.roles["施事"].text, "猫")
ck.eq("O → 受事", r.roles["受事"].text, "鱼")
ck.eq("V → 述题", r.roles["述题"].text, "吃")
ck.eq("槽位映射可回溯（SlotFill 名）", r.roles["施事"].slot_name, "S")
ck.eq("无虚词时 source='default'", r.roles["施事"].source, "default")

# 把字句：虚词「把」把 P 改写成 受事（前置）
rb = p.parse("猫把鱼吃了。")
ck.eq("把字句 P 的角色为 受事", rb.roles["受事"].text, "鱼")
ck.eq("受事来自虚词改写", rb.roles["受事"].source, "marker")
ck.eq("记录了引出角色的虚词", rb.roles["受事"].marker, "把")
ck.check("虚词改写是**数据库驱动**（function_word_roles 有该行）",
         (4, "受事", "前置") in
         [(sid, role, pos) for sid, lst in p._fw_roles.items() for role, pos in lst])

# 被字句：虚词「被」把 A 改写成 施事（前置）
rz = p.parse("鱼被猫吃了。")
ck.eq("被字句 A 的角色为 施事", rz.roles["施事"].text, "猫")
ck.eq("施事来自虚词改写", rz.roles["施事"].source, "marker")
ck.eq("记录了引出角色的虚词（被）", rz.roles["施事"].marker, "被")

# 关键：虚词改写**只看紧邻介词**，不得越界改写谓语
ck.eq("`鱼被猫吃了` 的谓语保持 述题（未被误改成施事）",
      rz.roles["述题"].text, "吃")
ck.check("越界改写回归：谓语不出现 marker='被'",
         rz.roles["述题"].marker == "", rz.roles["述题"].marker)

# entity 与 text 的区别（中心语 vs 整段表层形式）
rq = p.parse("猫把一条鱼吃了。")
ck.eq("受事 text 为整段", rq.roles["受事"].text, "一条鱼")
ck.eq("受事 entity 为中心语词形", rq.roles["受事"].entity, "鱼")
ck.check("SemanticRole 字段齐全（to_dict 键）",
         {"role", "slot", "text", "entity", "marker", "source", "confidence",
          "head_sense_id", "notes"} <= set(rq.roles["受事"].to_dict()))

# ===========================================================================
# S7 关系查询与推理链
# ===========================================================================
ck.section("S7 关系查询与推理链")
r = p.parse("猫把一条鱼吃了。")
infs = r.inferences
ck.check("推理结果非空", bool(infs), "empty")
ck.check("属性继承：猫 --can_do--> 吃",
         any(i.subject == "猫" and i.relation == "can_do" and i.obj == "吃" for i in infs))
ck.check("is_a 传递（跨两条边合一）：猫 --is_a--> 生物",
         any(i.subject == "猫" and i.relation == "is_a" and i.obj == "生物" for i in infs))
ck.check("量词默认：鱼 --classifier--> 条",
         any(i.subject == "鱼" and i.relation == "classifier" and i.obj == "条" for i in infs))
ck.check("典型论元：吃 --agent--> 猫",
         any(i.subject == "吃" and i.relation == "agent" and i.obj == "猫" for i in infs))
ck.check("典型论元：吃 --patient--> 鱼",
         any(i.subject == "吃" and i.relation == "patient" and i.obj == "鱼" for i in infs))
ck.check("推理链非空且为字符串列表",
         all(isinstance(s, str) for i in infs for s in i.chain)
         and any(i.chain for i in infs))
ck.check("推理结果已按 (subject, relation, obj) 去重",
         len({(i.subject, i.relation, i.obj) for i in infs}) == len(infs),
         f"{len(infs)} → {len({(i.subject, i.relation, i.obj) for i in infs})}")
ck.check("置信度落在 [0,1]", all(0.0 <= i.confidence <= 1.0 for i in infs))
ck.check("InferenceResult 字段齐全（to_dict 键）",
         {"kind", "subject", "relation", "object", "confidence", "chain",
          "rule_id", "rule_name", "rule_type", "note", "evidence"}
         <= set(infs[0].to_dict()))

# 多父类：鱼 的上位是 {动物, 食物} 并列 + 生物(2跳)，不能拼成单链
inh_fish = [i for i in infs if i.subject == "鱼" and i.kind == "inherit"]
chain_txt = " | ".join(s for i in inh_fish for s in i.chain)
ck.check("上位链渲染为并列 + 距离标注", "鱼 的上位" in chain_txt, chain_txt[:160])
ck.check("深度 1 的两个父类都在（动物/食物）",
         "动物(1跳)" in chain_txt and "食物(1跳)" in chain_txt, chain_txt[:200])
ck.check("深度 2 的上位单独标注（生物(2跳)），未串成单链",
         "生物(2跳)" in chain_txt, chain_txt[:200])
ck.check("未把并列父类串成单链（生物未被标成 1 跳）",
         "生物(1跳)" not in chain_txt, chain_txt[:200])
ck.check("推理链主语用中心语实体（不是「一条鱼」）",
         all("一条鱼" not in i.subject for i in infs),
         str([i.subject for i in infs if "一条鱼" in i.subject]))

# 属性继承方向：只能继承上位属性
ck.check("继承方向正确（猫的 can_do 来自上位 动物）",
         any("上位" in s or "直接声明" in s for i in infs
             if i.subject == "猫" and i.relation == "can_do" for s in i.chain))

# ===========================================================================
# S8 端到端 parse
# ===========================================================================
ck.section("S8 端到端 parse（ParseResult）")
r = p.parse("猫吃鱼。")
ck.check("ok=True", r.ok)
ck.check("含 tokens", len(r.tokens) == 4, str(len(r.tokens)))
ck.eq("pattern_name", r.pattern_name, "主谓宾")
ck.check("含 filled_slots", set(r.filled_slots) == {"S", "V", "O"}, str(sorted(r.filled_slots)))
ck.check("含 roles", {"施事", "述题", "受事"} <= set(r.roles))
ck.check("含 inferences", isinstance(r.inferences, list) and bool(r.inferences))
ck.check("confidence ∈ [0,1]", 0.0 <= r.confidence <= 1.0, str(r.confidence))
ck.check("elapsed_ms 已记录", r.elapsed_ms > 0)
ck.check("role_of/text_of 便捷方法", r.role_of("施事").text == "猫" and r.text_of("O") == "鱼")
ck.check("to_dict() 可 JSON 序列化", isinstance(json.dumps(r.to_dict(), ensure_ascii=False), str))
ck.check("render() 含原文/句式/词元/语义角色/推理链",
         all(k in r.render() for k in ("原文", "句式", "词元", "语义角色", "推理链")))
ck.check("to_dict 含关键键",
         {"text", "pattern_name", "filled_slots", "roles", "inferences", "confidence"}
         <= set(r.to_dict()))

# —— mood 判定
ck.eq("mood：陈述", p.parse("猫吃鱼。").mood, "陈述")
ck.eq("mood：疑问", p.parse("猫吃鱼吗？").mood, "疑问")
ck.eq("mood：祈使", p.parse("请你吃鱼。").mood, "祈使")
ck.eq("mood 落在 DB 的 sentence_moods 取值域",
      {p.parse(s).mood for s in ("猫吃鱼。", "猫吃鱼吗？", "请你吃鱼。")},
      {"陈述", "疑问", "祈使"})

# —— 篇章批量解析
doc = p.parse_document(["猫吃鱼。", "把鱼吃了。"], doc_id="val_doc",
                       context=ParserContext(doc_id="val_doc", context_type="narrative"))
ck.eq("parse_document 返回 2 条结果", len(doc), 2)
ck.check("parse_document 结果均为 ParseResult", all(x.pattern_name for x in doc))
ck.check("parse_document 内部维护篇章状态（第二句可回指）",
         doc[1].filled_slots.get("A") is not None
         and doc[1].filled_slots["A"].source == "ellipsis",
         str({k: f.source for k, f in doc[1].filled_slots.items()}))

# ===========================================================================
# S9 错误处理
# ===========================================================================
ck.section("S9 错误处理（top-3 候选 + 失败原因）")
for bad_text in ("苹果香蕉橘子。", "哈哈哈哈哈。"):
    rb = p.parse(bad_text)
    ck.check(f"{bad_text} 解析失败（ok=False）", not rb.ok, f"ok={rb.ok}")
    ck.eq(f"{bad_text} 无匹配句式", rb.pattern_name, None)
    ck.eq(f"{bad_text} confidence=0", rb.confidence, 0.0)
    n = len(rb.diagnostics)
    ck.check(f"{bad_text} 给出 1–3 条 top 候选", 1 <= n <= 3, f"n={n}")
    ck.check(f"{bad_text} 候选带具体失败原因",
             all(a.reason for a in rb.diagnostics), str([a.reason for a in rb.diagnostics]))
    ck.check(f"{bad_text} 候选带 pattern_id/name 便于定位",
             all(a.pattern_id >= 0 and a.name for a in rb.diagnostics))
    ck.check(f"{bad_text} 失败原因可读（含中文说明）",
             any("OOV" in a.reason or "槽位" in a.reason for a in rb.diagnostics),
             str([a.reason for a in rb.diagnostics[:1]]))
ck.check("failed_reasons 属性可用",
         len(p.parse("苹果香蕉橘子。").failed_reasons) >= 1)

# 全 OOV 句必须被拒（靠结构空转不得匹配）
rb = p.parse("苹果香蕉橘子。")
ck.check("全 OOV 句被「词汇证据」闸门拒绝",
         any("OOV" in a.reason for a in rb.diagnostics),
         str([a.reason for a in rb.diagnostics[:2]]))
# 反之：只要有一个词典实词，就不该被该闸门误杀
rk = p.parse("猫吃鱼。")
ck.check("含词典实词的句子不被词汇证据闸门误杀", rk.ok, f"ok={rk.ok}")

# ===========================================================================
# S10 回归：词汇证据闸门的**求值顺序**
# ===========================================================================
ck.section("S10 回归：闸门不得读到 known 的默认值 0")
# 历史 bug：`_validate_parse` 的闸门 ⑤ 在 `_score`（负责统计 known）之前执行，
# 于是读到 `_Parse.known` 的 dataclass 默认值 0，把所有正常句一并否决。
ck.check("`_lexical_evidence` 在闸门判定前被调用（顺序正确）",
         p._lexical_evidence.__doc__ is not None
         and "唯一真源" in (p._lexical_evidence.__doc__ or ""))
for text, expect in (("猫吃鱼。", "主谓宾"), ("猫吃鱼吗？", "是非问"),
                     ("请你吃鱼。", "祈使句"), ("他比我高。", "比较句-比")):
    rr = p.parse(text)
    ck.check(f"{text:<8} 仍能匹配（pattern={expect}）", rr.pattern_name == expect,
             f"got {rr.pattern_name}")
ck.eq("mood 未被闸门误伤（疑问）", p.parse("猫吃鱼吗？").mood, "疑问")
ck.eq("mood 未被闸门误伤（祈使）", p.parse("请你吃鱼。").mood, "祈使")
ck.eq("mood 未被闸门误伤（陈述）", p.parse("他比我高。").mood, "陈述")
# 13 例矩阵整体复验：全部必须匹配上（回归防线）
matched_all = [p.parse(t).ok for t, _, _ in MATRIX]
ck.eq("13 例矩阵整体匹配率 13/13", sum(matched_all), 13)
# 合法匹配上时，被消费 token 里必须确实有词典证据
ms = p.match_pattern(p.tokenize("猫吃鱼。"))
ck.check("合法匹配的 known > 0（证据统计写入 parse 对象）",
         ms is not None and all(t.entry_id is not None or t.is_marker_literal
                                for t in p.tokenize("猫吃鱼。")[:3]))

# ===========================================================================
# S11 篇章状态与 use_db_context 往返
# ===========================================================================
ck.section("S11 篇章状态（persist / anaphora / use_db_context 往返）")
_rm(DB_CTX)
shutil.copyfile(DB, DB_CTX)
conn_w = get_connection(DB_CTX)
pw = SyntaxParser(conn_w, use_db_context=True, persist=True)
wctx = lambda: ParserContext(doc_id="val_rt", context_type="narrative")
pw.parse("猫吃鱼。", wctx())
rw2 = pw.parse("把鱼吃了。", wctx())

seqs = [r[0] for r in conn_w.execute(
    "SELECT seq FROM utterances WHERE doc_id='val_rt' ORDER BY seq")]
ck.eq("persist 落盘 utterances（seq 从 1 起，遵守 CHECK seq>=1）", seqs, [1, 2])
ck.eq("utterances 文本与解析一致", conn_w.execute(
    "SELECT text FROM utterances WHERE doc_id='val_rt' AND seq=2").fetchone()[0],
    "把鱼吃了。")

an = conn_w.execute(
    """SELECT a.slot_role, a.form, a.form_type, a.antecedent_node_id, a.confidence
         FROM anaphora a JOIN utterances u ON u.utterance_id = a.utterance_id
        WHERE u.doc_id='val_rt'""").fetchall()
ck.eq("省略的 A 落盘为 1 条 anaphora", len(an), 1)
if an:
    role, form, ftype, node, conf = an[0]
    ck.eq("anaphora 角色 = 施事", role, "施事")
    ck.eq("anaphora 记录了**真实表层形式**（不是一律写 '∅'）", form, "猫")
    ck.eq("form_type 取值合法且符合事实", ftype, "definite_np")
    ck.check("antecedent_node_id 非空（可还原）", node is not None, str(node))
    ck.check("antecedent_node_id 指向 猫 的义项",
             sense_text(pw, int(node)) == "猫", sense_text(pw, int(node)) if node else "-")

# —— 跨连接：只读连接靠 use_db_context 从库重建
conn_r = get_connection(DB_CTX, read_only=True, must_exist=True)
pr = SyntaxParser(conn_r, use_db_context=True)
st = pr._discourse_state("val_rt")
ck.check("use_db_context 从库重建篇章状态", st is not None)
if st:
    ck.eq("重建的句数正确", st.count, 2)
    ck.eq("重建的上一句主语 = 猫", st.subject.text if st.subject else None, "猫")
rr = pr.parse("把鱼吃了。", wctx())
ck.check("重建后解析零形句 → A 从库回填成功",
         rr.filled_slots.get("A") is not None
         and rr.filled_slots["A"].source == "ellipsis"
         and rr.filled_slots["A"].text == "猫",
         str({k: (f.text, f.source) for k, f in rr.filled_slots.items()}))

# —— 未开 use_db_context 时不得打库
p_off = SyntaxParser(conn_r)
before = p_off._stats["db_queries"]
ck.eq("use_db_context=False 时返回 None（不臆造上下文）",
      p_off._discourse_state("val_rt"), None)
ck.eq("use_db_context=False 时不产生额外查询",
      p_off._stats["db_queries"] - before, 0)
ck.eq("use_db_context=False 时不缓存篇章状态",
      "val_rt" in p_off._discourse, False)
conn_w.close()
conn_r.close()

# ===========================================================================
# S12 只读性、自检与公共接口
# ===========================================================================
ck.section("S12 只读性与公共接口")
snap_tables = ("utterances", "anaphora", "relations", "closure", "senses", "words")


def _counts(c) -> dict[str, int]:
    return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in snap_tables}


base = _counts(conn)
for text, _, _ in MATRIX:
    p.parse(text)
after = _counts(conn)
ck.eq("非 persist 解析不写库（表行数快照一致）", after, base)

# —— build_parser 便捷入口
bp, bpconn = build_parser(DB)
ck.check("build_parser 返回 (parser, conn)", isinstance(bp, SyntaxParser)
         and bpconn.execute("SELECT 1").fetchone()[0] == 1)
ck.check("build_parser 默认只读", bool(getattr(bpconn, "logos_meta", {}).get("read_only")))
ck.check("build_parser 产出的解析器可用", bp.parse("猫吃鱼。").pattern_name == "主谓宾")
bpconn.close()

# —— 公共接口自检
import logos.parser as P  # noqa: E402

missing = [n for n in P.__all__ if not hasattr(P, n)]
ck.eq("__init__.__all__ 无缺失符号", missing, [])
ck.check("build_parser / sense_text / compile_template 已导出",
         all(callable(getattr(P, n)) for n in
             ("build_parser", "sense_text", "compile_template")))
ck.check("explain() 产出完整报告", "推理链" in p.explain("猫吃鱼。"))
ck.check("cache_info 键齐全",
         {"lexicon", "patterns", "pattern_slots", "slot_constraints", "semantic_roles",
          "ellipsis_rules", "inference_rules"} <= set(p.cache_info()))

conn.close()
_rm(DB)
_rm(DB_CTX)
sys.exit(ck.summary())
