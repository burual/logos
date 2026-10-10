# -*- coding: utf-8 -*-
"""tests/test_e2e_cat_eats_fish.py —— Logos 五层端到端集成测试（「猫吃鱼」场景）。

管线：L1 schema_v3.sql → L2 seed_general_v3.sql → L3 connection.py
      → L4 closure.py → L5 syntax_parser.py

与单元测试（validate_*.py，共 627 项）的区别：
  * 单元测试各自只测一层，且大多跑在已建好的库上；
  * 本脚本用 ``init_database()`` 从**零**建库（schema + seed），然后对完整
    ``parse()`` 管线做端到端验证，并输出 6 个阶段的中间结果与耗时占比。

测试夹具（FIXTURE_SQL）
----------------------
种子数据没有「老鼠 / 石头」两个词。为了触发任务要求的
``猫抓老鼠``（can_do + 闭包）与 ``猫吃石头``（受事约束冲突）两条路径，
本脚本在建库后补插 2 词 2 义项 + 类别 + 一条 is_a 边（源标注 source_id=5，
与种子同源）。夹具内容在 ``FIXTURE_SQL`` 中逐行注释，可一眼审阅。

运行::

    python tests/test_e2e_cat_eats_fish.py            # 直接运行，输出 6 阶段报告
    python tests/test_e2e_cat_eats_fish.py --save out # 同时把输出落盘到 out.txt

退出码：0 = 全部断言通过（含已知偏差 XFAIL）；1 = 存在 FAIL。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from logos.db.connection import get_connection, init_database  # noqa: E402
from logos.parser import ParserContext, SyntaxParser  # noqa: E402

# ===========================================================================
# 0. 极简断言台（与 validate_*.py 同风格：PASS/FAIL 计数 + 明细）
# ===========================================================================


class Checks:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.xfail_hit: list[str] = []
        self.xfail_miss: list[tuple[str, str]] = []

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        if cond:
            self.passed.append(name)
            print(f"    PASS  {name}")
        else:
            self.failed.append((name, detail))
            print(f"    FAIL  {name}" + (f"  —— {detail}" if detail else ""))
        return cond

    def xfail(self, name: str, cond_known_deviation: bool, detail: str = "") -> None:
        """已知偏差：``cond_known_deviation`` 为 True 表示「偏差确实存在」。

        存在 → 记 XFAIL（计入发现的问题）；不存在 → 记 XMISS（问题已被修复，
        应把这行从 xfail 升级为 check）。两者都不算 FAIL。
        """
        if cond_known_deviation:
            self.xfail_hit.append(name)
            print(f"    XFAIL {name}  —— {detail}")
        else:
            self.xfail_miss.append((name, detail))
            print(f"    XMISS {name}（已知问题似乎已修复，请升级断言）")

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed) + len(self.xfail_hit) + len(self.xfail_miss)

    def summary(self) -> str:
        return (f"PASS {len(self.passed)} / FAIL {len(self.failed)} / "
                f"XFAIL {len(self.xfail_hit)} / TOTAL {self.total}")


# ===========================================================================
# 1. 测试夹具：种子缺「老鼠/石头」，补 2 词以触发 can_do 闭包与受事约束冲突
# ===========================================================================

FIXTURE_SQL = """
-- 老鼠：is_a 动物(12)，触发器自动派生 老鼠→生物(1026) 的 2 跳闭包
-- ⚠ ID 段约定：本夹具独占 7000 段。不要用 1-27（schema）、1000-1027（seed_general_v3）、
--   2000-2999（seed_general_v3_ext：营养物/属性/分类骨架/量词 份）、3000+（entity_nodes）。
--   connection.py 的 init_database 会按文件名排序自动加载项目根的 seed_*.sql，
--   因此新增种子文件会**自动**进入本测试的库，ID 段必须彼此隔离。
INSERT INTO words(word_id, word, word_type, is_bound_root, register) VALUES
 (7001,'老鼠','单纯词',0,'通用'),
 (7002,'石头','单纯词',0,'通用');
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency, literalness, subcategory, is_primary) VALUES
 (7001,7001,'n',1,'鼠的通称','鼠 的 通称','lao3 shu3','laoshu','ls',900,'本义',NULL,1),
 (7002,7002,'n',1,'石的通称','石 的 通称','shi2 tou5','shitou','st',700,'本义',NULL,1);
INSERT INTO sense_categories(sense_id, cat, is_primary) VALUES
 (7001,'动物',1),      -- 老鼠 ∈ 动物
 (7002,'无机物',1);    -- 石头 ∈ 无机物（食物的对照组）
-- source_id=5 与种子数据同源（sources 表的通用常识源）
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id) VALUES
 ('sense',7001,'sense',12,'is_a',5);
"""

NORMAL: list[tuple[str, dict]] = [
    ("猫吃鱼", {"pattern": "主谓宾", "agent": "猫", "patient": "鱼",
                "must_inf": [("猫", "can_do", "吃")]}),
    ("猫把鱼吃了", {"pattern": "把字句", "agent": "猫", "patient": "鱼",
                    "must_inf": [("吃", "agent", "猫")]}),
    ("鱼被猫吃了", {"pattern": "被字句", "agent": "猫", "patient": "鱼",
                    "must_inf": [("吃", "patient", "鱼")]}),
    ("鱼，猫吃了", {"pattern": "主谓谓语句", "agent": None, "patient": None,
                    "role_contains": ("话题", "鱼"),
                    "must_inf": [("鱼", "is_a", "生物")]}),
    ("猫抓老鼠", {"pattern": "主谓宾", "agent": "猫", "patient": "老鼠",
                  "must_inf": [("猫", "can_do", "抓"), ("老鼠", "is_a", "生物")]}),
]


# ===========================================================================
# 2. 六阶段插桩：手动串联五个层，逐段计时（parse() 另行官方计时）
# ===========================================================================


def run_phases(parser: SyntaxParser, text: str, context: ParserContext) -> dict:
    """逐段调用 L5 的五个方法并计时。返回各阶段产物 + 耗时（ms）。"""
    ph: dict = {"timings": {}}
    pc = time.perf_counter

    t0 = pc()
    tokens = parser.tokenize(text)
    ph["timings"]["1 分词"] = (pc() - t0) * 1000
    ph["tokens"] = tokens

    t0 = pc()
    match = parser.match_pattern(tokens) if tokens else None
    ph["timings"]["2 句式匹配"] = (pc() - t0) * 1000
    ph["match"] = match
    att = sorted(getattr(parser, "last_attempts", []),
                 key=lambda a: (not a.matched, -a.score))
    ph["top3"] = att[:3]

    if match is None:
        return ph

    t0 = pc()
    filled = parser.fill_slots(match, tokens, context)
    ph["timings"]["3 槽位填充"] = (pc() - t0) * 1000
    ph["filled"] = filled

    t0 = pc()
    roles = parser.annotate_semantic_roles(filled, match, tokens)
    ph["timings"]["4 语义角色"] = (pc() - t0) * 1000
    ph["roles"] = roles

    t0 = pc()
    infs = parser.query_relations(roles, match, tokens)
    ph["timings"]["5 关系查询与推理"] = (pc() - t0) * 1000
    ph["inferences"] = infs
    return ph


def print_phases(text: str, ph: dict, result) -> None:
    bar = "═" * 55
    print(f"\n{bar}")
    print(f"句子：{text}")
    print(bar)

    print("【Phase 1: 分词】")
    for i, t in enumerate(ph.get("tokens", [])):
        oov = " OOV" if t.is_oov else ""
        print(f'{i}: "{t.text}" pos={t.pos} entry_id={t.entry_id} '
              f'sense_id={t.sense_id} start={t.start} end={t.end}{oov}')

    print("【Phase 2: 句式匹配】")
    m = ph.get("match")
    if m is None:
        print("匹配到的句式：<无>（应输出 top-3 候选 + 失败原因）")
    else:
        print(f"匹配到的句式：{m.name} (pattern_id={m.pattern_id})")
        print(f"模板：{m.template}   variant={m.variant!r}")
        print(f"匹配得分：{m.score:.2f}   覆盖率：{m.coverage:.0%}")
    print("候选句式排名（top-3）：")
    for a in ph.get("top3", []):
        print(f"    {'OK ' if a.matched else 'REJ'} {a.name}({a.pattern_id}) "
              f"score={a.score:.2f}" + (f" —— {a.reason}" if a.reason else ""))

    print("【Phase 3: 槽位填充】")
    for k, f in (ph.get("filled") or {}).items():
        src = "" if f.source == "matched" else f"  ← {f.source}" + (
            f"（rule={f.ellipsis_rule_id}, {f.strategy}）" if f.ellipsis_rule_id else "")
        print(f'{k} → "{f.text}"{src}' if f.tokens else f"{k} → None（留空）")

    print("【Phase 4: 语义角色标注】")
    for k, r in (ph.get("roles") or {}).items():
        mk = f'   虚词贡献："{r.marker}" → {k}（{r.source} 改写）' if r.marker else ""
        head = r.head
        sid = f" (sense_id={head.sense_id})" if head is not None and head.sense_id else ""
        print(f'{k} → "{r.text}"[{r.source}]{sid}{mk}')

    print("【Phase 5: 关系查询与推理】")
    groups: dict[str, list] = {}
    for inf in ph.get("inferences", []):
        groups.setdefault(inf.rule_name or inf.kind, []).append(inf)
    for g, items in groups.items():
        print(f"  ◇ {g}")
        for inf in items:
            print(f"      {inf.subject} --[{inf.relation}]--> {inf.obj}"
                  f"  (conf={inf.confidence:.2f})")

    print("【Phase 6: 最终输出】")
    r = result
    print(f"ParseResult:")
    print(f"  pattern_name : {r.pattern_name}")
    print(f"  confidence   : {r.confidence:.2f}")
    print("  filled_slots : " + ", ".join(
        f'{k}: "{f.text}"' if f.tokens else f"{k}: None"
        for k, f in r.filled_slots.items()))
    print("  semantic_roles: " + ", ".join(
        f'{k}: "{v.text}"' for k, v in r.roles.items()))
    print(f"  inferences   : {len(r.inferences)} 条")
    print(f"  parse_time_ms: {r.elapsed_ms:.1f}")


# ===========================================================================
# 3. 主流程
# =========================================================================--


def main(argv: list[str]) -> int:
    save_path = None
    if "--save" in argv:
        save_path = argv[argv.index("--save") + 1]

    ck = Checks()
    tmp = tempfile.mkdtemp(prefix="logos_e2e_")
    db = os.path.join(tmp, "e2e.db")
    conn = None
    try:
        # ---- L1+L2：从零建库（schema + seed 自动发现）----
        rep = init_database(db, reset=True, validate=True)
        n_tables = getattr(rep, "tables", "?")
        print("=" * 72)
        print("Logos 端到端集成测试：猫吃鱼场景")
        print("=" * 72)
        print(f"[建库] init_database({db}, reset=True, validate=True)")
        print(f"       表 {n_tables} 个，integrity_check + foreign_key_check 通过")

        # ---- 测试夹具 ----
        conn = get_connection(db)
        conn.executescript(FIXTURE_SQL)
        rows = [(r[0], r[1]) for r in conn.execute(
            "SELECT ancestor, depth FROM closure "
            "WHERE descendant=7001 AND relation='is_a' ORDER BY depth")]
        anc = [r[0] for r in rows]
        print(f"[夹具] 老鼠(7001) 闭包：{rows}（动物=12，生物=1026）")
        ck.check("夹具：触发器为 老鼠 派生 2 跳闭包", 12 in anc and 1026 in anc, str(rows))
        ck.check("夹具：外键与完整性仍成立", conn.fk_ok() and conn.integrity_ok())

        probe = SyntaxParser(conn)   # 插桩用（独立篇章状态）
        main_p = SyntaxParser(conn)  # 官方 parse() 用

        # ================= 正常句子：5 句 =================
        phase_ms: dict[str, list[float]] = {}
        parse_ms: list[float] = []
        print("\n" + "#" * 72)
        print("# 一、正常句子（4 种句式变体 + 属性继承扩展）")
        print("#" * 72)
        for text, exp in NORMAL:
            ph = run_phases(probe, text, ParserContext(doc_id="p-" + text))
            result = main_p.parse(text, ParserContext(doc_id="d-" + text))
            print_phases(text, ph, result)
            for k, v in ph["timings"].items():
                phase_ms.setdefault(k, []).append(v)
            parse_ms.append(result.elapsed_ms)

            print(f"  —— 断言（{text}）")
            ck.check("句式非空", bool(result.pattern_name))
            ck.check(f"命中 {exp['pattern']}", result.pattern_name == exp["pattern"],
                     f"实为 {result.pattern_name}")
            ck.check("confidence > 0.5", result.confidence > 0.5,
                     f"实为 {result.confidence:.3f}")
            slots = {k: f.text for k, f in result.filled_slots.items() if f.tokens}
            has_subj = any(slots.get(s) for s in ("S", "A", "P", "Topic"))
            has_verb = any(f.head and f.head.pos == "v"
                           for f in result.filled_slots.values())
            ck.check("至少填了主语槽与动词槽", has_subj and has_verb, str(slots))
            roles_txt = {k: v.text for k, v in result.roles.items()}
            ck.check("语义角色含 施事/受事/话题 之一",
                     any(k in roles_txt for k in ("施事", "受事", "话题")), str(roles_txt))
            if "role_contains" in exp:
                rk, rv = exp["role_contains"]
                ck.check(f"角色 {rk} = {rv}", roles_txt.get(rk) == rv, str(roles_txt))
            if exp["agent"]:
                ag = result.role_of("施事")
                ck.check(f"施事 = {exp['agent']}", bool(ag) and ag.entity == exp["agent"],
                         str(roles_txt))
            if exp["patient"]:
                pa = result.role_of("受事")
                ck.check(f"受事 = {exp['patient']}", bool(pa) and pa.entity == exp["patient"],
                         str(roles_txt))
            ck.check("推理链非空", len(result.inferences) > 0)
            for (s, rel, o) in exp["must_inf"]:
                ck.check(f"推理含 {s} --{rel}--> {o}",
                         any(i.subject == s and i.relation == rel and i.obj == o
                             for i in result.inferences))
            ck.check("token 覆盖整句",
                     bool(result.tokens) and result.tokens[0].start == 0
                     and result.tokens[-1].end == len(text),
                     f"{result.tokens[0].start}..{result.tokens[-1].end} vs len={len(text)}")

        # 句式特定断言
        print("  —— 断言（句式特定语法行为）")
        r_ba = main_p.parse("猫把鱼吃了", ParserContext(doc_id="spec-ba"))
        pa_role = r_ba.role_of("受事")
        ck.check("把字句：受事由「把」前置标记",
                 bool(pa_role) and pa_role.marker == "把" and pa_role.source == "marker",
                 str((pa_role.marker, pa_role.source)) if pa_role else "无受事角色")
        r_bei = main_p.parse("鱼被猫吃了", ParserContext(doc_id="spec-bei"))
        ag_role = r_bei.role_of("施事")
        ck.check("被字句：施事由「被」前置标记",
                 bool(ag_role) and ag_role.marker == "被" and ag_role.source == "marker",
                 str((ag_role.marker, ag_role.source)) if ag_role else "无施事角色")
        r_top = main_p.parse("鱼，猫吃了", ParserContext(doc_id="spec-topic"))
        ck.check("话题句：逗号前进入 Topic 槽", r_top.text_of("Topic") == "鱼",
                 f"Topic={r_top.text_of('Topic')!r}")

        # ================= 错误处理：3+1 句 =================
        print("\n" + "#" * 72)
        print("# 二、错误处理与省略（3 个边界句 + 1 个省略阳性对照）")
        print("#" * 72)

        # (1) 吃鱼：缺主语
        print("\n───── 边界句 1：吃鱼（缺主语，期望省略回填）─────")
        r_no = main_p.parse("吃鱼")
        print_phases("吃鱼", run_phases(probe, "吃鱼", None), r_no)
        print("  —— 断言（吃鱼）")
        ck.check("吃鱼：不崩溃且返回可用结果", r_no is not None)
        ck.check("吃鱼：有候选诊断信息", len(r_no.diagnostics) > 0)
        ck.xfail("吃鱼：应触发 ellipsis_rules 回填主语（实为倒装句 S=鱼 竞争胜出）",
                 r_no.pattern_name == "倒装句",
                 "主谓宾(省略 S) 得分低于 倒装句，省略读法被压制")

        # (2) 猫吃：缺宾语
        print("\n───── 边界句 2：猫吃（缺宾语，期望 top-3 失败原因）─────")
        r_vo = main_p.parse("猫吃")
        print_phases("猫吃", run_phases(probe, "猫吃", None), r_vo)
        print("  —— 断言（猫吃）")
        ck.check("猫吃：不崩溃且返回结果", r_vo is not None)
        ck.xfail("猫吃：应匹配失败并返回 top-3 失败原因（实为连动句吸收，且省略规则把 鱼 同时回填 O1/O2 两槽）",
                 r_vo.ok and r_vo.pattern_name == "连动句",
                 "slots=" + str({k: (f.text, f.source, f.strategy)
                                 for k, f in r_vo.filled_slots.items()}))

        # (3) 猫吃石头：宾语不合法
        print("\n───── 边界句 3：猫吃石头（石头 ∉ 食物，期望约束冲突）─────")
        r_bad = main_p.parse("猫吃石头")
        print_phases("猫吃石头", run_phases(probe, "猫吃石头", None), r_bad)
        print("  —— 断言（猫吃石头）")
        conflict = any(i.relation == "选择限制" and i.obj == "未通过"
                       and "受事" in i.rule_name for i in r_bad.inferences)
        ck.check("猫吃石头：受事合法性冲突被显式输出（未通过，conf=0.30）", conflict,
                 str([i.render() for i in r_bad.inferences if "选择限制" in i.relation]))
        ck.xfail("猫吃石头：冲突应降低整体置信度（实为与 猫吃鱼 同分 0.854）",
                 abs(r_bad.confidence - main_p.parse("猫吃鱼",
                     ParserContext(doc_id="cmp")).confidence) < 1e-9,
                 "约束冲突只在推理层记 0.30，不回传总分")
        ck.xfail("猫吃石头：矛盾的「吃 --patient--> 石头(0.90)」应被抑制",
                 any(i.subject == "吃" and i.relation == "patient" and i.obj == "石头"
                     and i.confidence >= 0.9 for i in r_bad.inferences),
                 "同一受事既判 未通过 又给出 0.90 论元边，自相矛盾")

        # (4) 阳性对照：把鱼吃了 + 叙事上下文 → 省略回填
        print("\n───── 阳性对照：把鱼吃了（叙事上下文，A 槽省略回填）─────")
        ctx = lambda: ParserContext(doc_id="ell", context_type="narrative")  # noqa: E731
        main_p.parse("猫吃鱼", ctx())
        r_ell = main_p.parse("把鱼吃了", ctx())
        print_phases("把鱼吃了", run_phases(probe, "把鱼吃了", ctx()), r_ell)
        print("  —— 断言（把鱼吃了·省略）")
        a_fill = r_ell.filled_slots.get("A")
        ck.check("省略回填：A = 猫（来自上一句主语）",
                 bool(a_fill) and a_fill.text == "猫",
                 str(a_fill.text if a_fill else None))
        ck.check("省略回填：source=ellipsis 且策略=previous_subject",
                 bool(a_fill) and a_fill.source == "ellipsis"
                 and a_fill.strategy == "previous_subject",
                 str((a_fill.source, a_fill.strategy) if a_fill else None))
        ck.check("省略回填：被省略槽位记录在 omitted_slots",
                 bool(r_ell.match) and "A" in (r_ell.match.omitted_slots or []),
                 str(r_ell.match.omitted_slots if r_ell.match else None))

        # ================= 性能统计 =================
        print("\n" + "#" * 72)
        print("# 三、性能统计")
        print("#" * 72)
        print(f"{'阶段':<14}{'总耗时 ms':>12}{'占比':>10}")
        total_phase = sum(sum(v) for v in phase_ms.values())
        for k in sorted(phase_ms):
            s = sum(phase_ms[k])
            print(f"{k:<14}{s:>12.2f}{s / total_phase:>9.1%}")
        print(f"{'合计(插桩)':<14}{total_phase:>12.2f}")
        print()
        print(f"{'句子':<12}{'parse_time_ms':>14}")
        for (text, _), ms in zip(NORMAL, parse_ms):
            print(f"{text:<12}{ms:>14.2f}")
        avg = sum(parse_ms) / len(parse_ms)
        print(f"{'平均':<12}{avg:>14.2f}   （5 句总耗时 {sum(parse_ms):.2f} ms）")
        ck.check("单句平均 parse_time < 100ms（冷启动后）", avg < 100, f"{avg:.2f} ms")
        ck.check("单句最慢 parse_time < 100ms", max(parse_ms) < 100,
                 f"{max(parse_ms):.2f} ms")

        # ================= 汇总 =================
        print("\n" + "=" * 72)
        print("结果：" + ck.summary())
        print("=" * 72)
        if ck.xfail_hit:
            print("已知偏差（详见 docs/E2E_TEST_REPORT.md「发现的问题」）：")
            for n in ck.xfail_hit:
                print("  · " + n)
        ok = not ck.failed
        return 0 if ok else 1

    except Exception:
        print("\n!!! 端到端流程抛出异常，以下是详细上下文 !!!")
        traceback.print_exc()
        try:
            print("last_attempts =")
            for a in getattr(probe, "last_attempts", [])[:6]:
                print("   ", a.render())
            print("db =", db)
            if conn is not None:
                print("integrity:", conn.integrity_ok(), "fk:", conn.fk_ok())
                print("counts:", {k: v for k, v in conn.counts().items()
                                  if k in ("words", "senses", "relations", "closure")})
        except Exception:
            pass
        return 2
    finally:
        if conn is not None:
            conn.close()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n[清理] 连接已关闭，临时库 {tmp} 已删除")


if __name__ == "__main__":
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        buf = open(out, "w", encoding="utf-8")
        class _Tee:
            def write(self, s):
                sys.__stdout__.write(s)
                buf.write(s)
            def flush(self):
                sys.__stdout__.flush()
                buf.flush()
        sys.stdout = _Tee()  # type: ignore[assignment]
        code = main(sys.argv)
        buf.close()
        sys.stdout = sys.__stdout__
    else:
        code = main(sys.argv)
    sys.exit(code)
