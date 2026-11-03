#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 ``residual_gold.json``：给 ``inferred_relations`` 的每一行打人工判定。

为什么要有这个文件
------------------
上一轮验收只人工看了 top-20，「16/20 是错的」这种结论没法跟不同配置比。
把 118 行逐条判定固化成 json 之后，「嵌入初始化有没有减少错误」才是一个
可复现的数字，而不是「看起来更合理」。

判定四分类
----------
``correct``        三元组成立。
``wrong``          事实或类型错误（``鱼 is_a 无脊椎动物``）。
``wrong_relation`` src 与 tgt 之间确实有语义关系，但**关系名用错了**。
                   本库最典型的是 ``狗 typically_has 有脊柱``：正确说法是
                   ``has_property``。``typically_has`` 的 149 条 asserted 边里
                   146 条目标是部件名词，属性词一律走 ``has_property``。
``uncertain``      母语直觉也有争议，不计入正确也不计入错误。

判定规则（写死在 ``RULES`` 里，与 json 一起交付）
--------------------------------------------------
1. ``can_do(A,B)`` 在库里混了两种读法：动作类→子动作（``运动 can_do 攀爬``）
   与施事→动作（``猫 can_do 捕猎``）。判 ``correct`` 当且仅当 (a) A 是施事且
   确实能执行 B，或 (b) A 是动作类且 B 属于该类。A 是动作类而 B 属于另一个
   动作类（``防御 can_do 潜水``：潜水∈运动，∉防御）判 ``wrong``；只是「可能
   同时发生 / 可能是手段」而无分类关系判 ``uncertain``。
2. ``has_property`` 目标是形容词，按动物学事实判。
3. ``is_a`` 判是否真的上位；根目标（``生物``/``抽象概念``）选错也算 ``wrong``。
4. ``part_of`` 判部分是否真的属于该整体；「类的部分 part_of 某个体」
   （``皮肤系统 part_of 猴子``）是粒度错，判 ``wrong``。
5. ``typically_has`` 目标必须是部件名词；目标是形容词时判 ``wrong_relation``
   （语义对）或 ``wrong``（语义也错）。

用法::

    python logos_transformer/make_residual_gold.py            # 写 residual_gold.json
    python logos_transformer/make_residual_gold.py --check    # 只校验覆盖率
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sqlite3
import sys
from typing import Dict, List, Tuple

_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = _ROOT / "residual_gold.json"

RULES: Dict[str, str] = {
    "correct": "三元组成立（动物学事实 / 库内既有用法一致）",
    "wrong": "事实或类型错误",
    "wrong_relation": "src 与 tgt 之间确有语义关系，但关系名用错（本库多为 typically_has 承接了属性词）",
    "uncertain": "判定本身有争议，不计入正确也不计入错误",
}

VERDICTS: Dict[str, Tuple[str, str]] = {}


def add(verdict: str, reason: str, *keys: str) -> None:
    for key in keys:
        if key in VERDICTS:
            raise SystemExit(f"[错误] 重复判定：{key}")
        VERDICTS[key] = (verdict, reason)


# --------------------------------------------------------------------------- #
# can_do：27 条
# --------------------------------------------------------------------------- #
add("correct", "人具备潜水能力（施事→动作，与 猫 can_do 捕猎 同型）", "人|can_do|潜水")
add("correct", "人具备攀爬能力（施事→动作）", "人|can_do|攀爬")
add("wrong", "「跃出水面」是水生动物特化动作，人无此能力", "人|can_do|跃出水面")

add("wrong", "防御∈行为类，潜水∈运动类，无分类关系", "防御|can_do|潜水")
add("wrong", "防御与「跃出水面」无分类关系", "防御|can_do|跃出水面")
add("wrong", "防御与攀爬无分类关系（攀爬∈运动）", "防御|can_do|攀爬")

add("wrong", "筑造是建造行为，潜水不在其下位", "筑造|can_do|潜水")
add("wrong", "筑造与「跃出水面」无分类关系", "筑造|can_do|跃出水面")
add("wrong", "筑造与攀爬无分类关系", "筑造|can_do|攀爬")

add("wrong", "繁殖与潜水无分类关系", "繁殖|can_do|潜水")
add("wrong", "繁殖与「跃出水面」无分类关系", "繁殖|can_do|跃出水面")
add("wrong", "繁殖与攀爬无分类关系", "繁殖|can_do|攀爬")

add("uncertain", "潜水可作为捕猎手段（水獭、鸬鹚），但两者无分类关系", "捕猎|can_do|潜水")
add("wrong", "「跃出水面」多为交流/清理行为，不属捕猎", "捕猎|can_do|跃出水面")
add("uncertain", "攀爬可作为猫科捕猎手段，但非捕猎的下位动作", "捕猎|can_do|攀爬")

add("wrong", "通讯与潜水无分类关系", "通讯|can_do|潜水")
add("uncertain", "鲸类跃身击浪（breaching）被认为有通讯功能，但归属有争议",
    "通讯|can_do|跃出水面")
add("wrong", "通讯与攀爬无分类关系", "通讯|can_do|攀爬")

add("wrong", "育幼与潜水无分类关系", "育幼|can_do|潜水")
add("wrong", "育幼与「跃出水面」无分类关系", "育幼|can_do|跃出水面")
add("wrong", "育幼与攀爬无分类关系", "育幼|can_do|攀爬")

add("uncertain", "潜水觅食常见（水獭、企鹅），但潜水∈运动而非取食", "取食|can_do|潜水")
add("wrong", "「跃出水面」不是取食方式", "取食|can_do|跃出水面")
add("uncertain", "攀爬取食常见（松鼠、猴），但攀爬∈运动而非取食", "取食|can_do|攀爬")

add("wrong", "感知与潜水无分类关系", "感知|can_do|潜水")
add("wrong", "感知与「跃出水面」无分类关系", "感知|can_do|跃出水面")
add("wrong", "奔跑∈运动类，不属感知", "感知|can_do|奔跑")

# --------------------------------------------------------------------------- #
# has_property：15 条（目标是形容词）
# --------------------------------------------------------------------------- #
add("wrong", "蛇是变温动物", "蛇|has_property|温血")
add("wrong", "蚯蚓是无脊椎动物", "蚯蚓|has_property|有脊柱")
add("wrong", "昆虫是无脊椎动物", "昆虫|has_property|有脊柱")
add("wrong", "青蛙是变温动物", "青蛙|has_property|温血")
add("wrong", "猫是胎生哺乳动物", "猫|has_property|卵生")
add("wrong", "鱼没有羽毛", "鱼|has_property|有羽毛")
add("wrong", "绝大多数鱼是变温动物", "鱼|has_property|温血")
add("wrong", "章鱼是软体动物，无脊柱", "章鱼|has_property|有脊柱")

add("correct", "灵长类是脊椎动物", "灵长类属性组|has_property|有脊柱")
add("correct", "长鼻类（象）是脊椎动物", "长鼻类属性组|has_property|有脊柱")
add("correct", "河狸是脊椎动物", "河狸属性组|has_property|有脊柱")
add("correct", "灵长类是哺乳类，恒温", "灵长类属性组|has_property|温血")
add("correct", "长鼻类是哺乳类，恒温", "长鼻类属性组|has_property|温血")
add("correct", "河狸是哺乳类，恒温", "河狸属性组|has_property|温血")
add("correct", "翼手类（蝙蝠）是脊椎动物", "翼手类属性组|has_property|有脊柱")

# --------------------------------------------------------------------------- #
# is_a：10 条
# --------------------------------------------------------------------------- #
add("wrong", "营养素是物质/化学成分，不是抽象概念", "营养素|is_a|抽象概念")
add("wrong", "猫与人是并列的哺乳动物分支，不构成上位", "猫|is_a|人")
add("wrong", "脑电图信号是物理测量信号，不是抽象概念", "脑电图信号|is_a|抽象概念")
add("wrong", "鱼与人是并列的脊椎动物分支", "鱼|is_a|人")
add("uncertain", "信息人造物（数据集、软件）是否算抽象物有争议",
    "信息人造物|is_a|抽象概念")
add("wrong", "医学概念的上位应是抽象概念，不是生物（根节点选错）",
    "医学概念|is_a|生物")
add("wrong", "物理人造物是具体器物，不是抽象概念", "物理人造物|is_a|抽象概念")
add("wrong", "鱼是脊椎动物，不是无脊椎动物（需互斥公理）", "鱼|is_a|无脊椎动物")
add("wrong", "青蛙是脊椎动物", "青蛙|is_a|无脊椎动物")
add("wrong", "鲤科是硬骨鱼，属脊椎动物", "鲤科|is_a|无脊椎动物")

# --------------------------------------------------------------------------- #
# part_of：12 条（全是「类的部分 part_of 某个具体物种」的粒度错）
# --------------------------------------------------------------------------- #
for _target in ("猴子", "蝙蝠", "大象"):
    add("wrong", f"皮肤系统是器官系统层级，不属于 {_target} 这个具体物种；"
                 "正确是 part_of 脊椎动物（库内既有边）", f"皮肤系统|part_of|{_target}")
    add("wrong", f"鲸类头部是鲸类的部分，不属于 {_target}", f"鲸类头部|part_of|{_target}")
for _target in ("海豚", "猴子", "大象"):
    add("wrong", f"翼膜是蝙蝠的皮膜，{_target} 没有翼膜", f"翼膜|part_of|{_target}")
for _target in ("海豚", "猴子", "蝙蝠"):
    add("wrong", f"昆虫头部不属于 {_target}", f"昆虫头部|part_of|{_target}")

# --------------------------------------------------------------------------- #
# typically_has：54 条
#   目标是部件名词 → 按部件是否真属于该整体判
#   目标是形容词   → 关系名用错（应 has_property），语义对判 wrong_relation
# --------------------------------------------------------------------------- #
_HAS_PART_OK = "typically_has 目标应为部件名词"
add("correct", "鸟类有尾巴", "鸟类|typically_has|尾巴")
add("correct", "鲤科多有口须", "鲤科|typically_has|胡须")
add("correct", "硬骨鱼多有口须/触须", "硬骨鱼|typically_has|胡须")
add("correct", "虾有长触须（口语称胡须）", "虾|typically_has|胡须")
add("uncertain", "蝴蝶有触角，口语是否称「胡须」有争议", "蝴蝶|typically_has|胡须")
add("uncertain", "昆虫纲有触角，术语与「胡须」不完全对应", "昆虫纲|typically_has|胡须")
add("uncertain", "节肢动物有触角，术语与「胡须」不完全对应",
    "节肢动物|typically_has|胡须")

add("wrong", "结节没有心脏", "恶性结节|typically_has|心脏", "良性结节|typically_has|心脏")
add("wrong", "结节没有肺", "恶性结节|typically_has|肺", "良性结节|typically_has|肺")
add("wrong", "语法规则与结节无关", "恶性结节|typically_has|语法规则")
add("wrong", "结节没有胸鳍", "良性结节|typically_has|胸鳍")
add("wrong", "爬行纲无口须", "爬行纲|typically_has|胡须")
add("wrong", "甲壳纲有触角，不是口须", "甲壳纲|typically_has|胡须")
add("wrong", "环节动物无口须", "环节动物|typically_has|胡须")
add("wrong", "头足纲有触腕，不是口须", "头足纲|typically_has|胡须")
add("wrong", "软体动物无口须", "软体动物|typically_has|胡须")
add("wrong", "蛇无口须（吐信是舌，不是须）", "蛇|typically_has|胡须")
add("wrong", "蚯蚓无口须", "蚯蚓|typically_has|胡须")
add("wrong", "雀形目无口须", "雀形目|typically_has|胡须")
add("wrong", "两栖纲无口须", "两栖纲|typically_has|胡须")
add("wrong", "青蛙无口须", "青蛙|typically_has|胡须")

# 属性词误用：语义对，关系名错
_WR = "「有脊柱」是属性（has_property），不是部件；typically_has 的 asserted 边里目标是部件名词"
add("wrong_relation", _WR,
    "狗|typically_has|有脊柱", "鸟|typically_has|有脊柱", "鸟类|typically_has|有脊柱",
    "鱼|typically_has|有脊柱", "蛇|typically_has|有脊柱", "哺乳动物|typically_has|有脊柱",
    "猫科|typically_has|有脊柱", "犬科|typically_has|有脊柱", "鲤科|typically_has|有脊柱",
    "硬骨鱼|typically_has|有脊柱", "爬行纲|typically_has|有脊柱",
    "雀形目|typically_has|有脊柱")
_WR2 = "「卵生」是属性（has_property），不是部件"
add("wrong_relation", _WR2,
    "硬骨鱼|typically_has|卵生", "鲤科|typically_has|卵生", "爬行纲|typically_has|卵生",
    "环节动物|typically_has|卵生", "软体动物|typically_has|卵生",
    "甲壳纲|typically_has|卵生", "头足纲|typically_has|卵生", "蚯蚓|typically_has|卵生",
    "节肢动物|typically_has|卵生", "无脊椎动物|typically_has|卵生",
    "虾|typically_has|卵生", "蝴蝶|typically_has|卵生", "昆虫纲|typically_has|卵生",
    "昆虫|typically_has|卵生", "两栖纲|typically_has|卵生", "雀形目|typically_has|卵生",
    "鸟类|typically_has|卵生")
# 语义也错
add("wrong", "蚯蚓无脊柱（关系名与事实同时错）", "蚯蚓|typically_has|有脊柱")
add("wrong", "昆虫无脊柱（关系名与事实同时错）", "昆虫|typically_has|有脊柱")


# --------------------------------------------------------------------------- #
# 对照组：库里 asserted 的真边，防止金标只有错条、算不出分离度
# --------------------------------------------------------------------------- #
CONTROLS: List[Tuple[str, str, str]] = [
    ("猫", "is_a", "猫科"),
    ("猫科", "is_a", "哺乳动物"),
    ("哺乳动物", "is_a", "脊椎动物"),
    ("运动", "can_do", "攀爬"),
    ("繁殖", "can_do", "产卵"),
    ("哺乳动物属性组", "has_property", "温血"),
    ("鱼类属性组", "has_property", "冷血"),
    ("猫", "typically_has", "头部"),
    ("狗", "typically_has", "前肢"),
    ("心脏", "part_of", "循环系统"),
    ("循环系统", "part_of", "脊椎动物"),
    ("鹰", "has_part", "翅膀"),
]


def fetch_rows(db: pathlib.Path) -> List[Dict[str, object]]:
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            """
            SELECT ws.word, ss.pos, r.relation_type, wt.word, st.pos, r.confidence
            FROM inferred_relations r
            JOIN senses ss ON ss.sense_id = r.source_id
            JOIN words ws ON ws.word_id = ss.word_id
            JOIN senses st ON st.sense_id = r.target_id
            JOIN words wt ON wt.word_id = st.word_id
            ORDER BY r.relation_type, r.confidence DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [
        {"source": a, "source_pos": b, "relation": c, "target": d,
         "target_pos": e, "confidence": f}
        for a, b, c, d, e, f in rows
    ]


def main(argv: List[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="生成 residual_gold.json")
    parser.add_argument("--db", type=pathlib.Path, default=_ROOT / "logos.db")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="只校验覆盖率，不写文件")
    args = parser.parse_args(argv)

    rows = fetch_rows(args.db)
    keys = [f"{r['source']}|{r['relation']}|{r['target']}" for r in rows]

    missing = [k for k in keys if k not in VERDICTS]
    extra = [k for k in VERDICTS if k not in set(keys)]
    if missing:
        print(f"[错误] 有 {len(missing)} 条写回没有判定：", file=sys.stderr)
        for key in missing:
            print("  " + key, file=sys.stderr)
        return 1
    if extra:
        print(f"[警告] 判定表里有 {len(extra)} 条不在写回表中（已忽略）：", file=sys.stderr)
        for key in extra:
            print("  " + key, file=sys.stderr)

    items: List[Dict[str, object]] = []
    for row, key in zip(rows, keys):
        verdict, reason = VERDICTS[key]
        items.append({**row, "verdict": verdict, "reason": reason})

    for src, rel, tgt in CONTROLS:
        items.append({
            "source": src, "source_pos": "", "relation": rel, "target": tgt,
            "target_pos": "", "confidence": None, "verdict": "correct",
            "reason": "库内 asserted 真边（对照组，用于计算分离度）",
            "control": True,
        })

    counts = collections.Counter(i["verdict"] for i in items if not i.get("control"))
    print(f"写回条数      : {len(rows)}")
    print("判定分布      : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    by_rel: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in items:
        if item.get("control"):
            continue
        by_rel[str(item["relation"])][str(item["verdict"])] += 1
    for rel in sorted(by_rel):
        print(f"  {rel:<14} " + ", ".join(f"{k}={v}" for k, v in sorted(by_rel[rel].items())))
    print(f"对照组        : {len(CONTROLS)} 条 asserted 真边")

    if args.check:
        return 0

    payload = {
        "version": 1,
        "generated_by": "logos_transformer/make_residual_gold.py",
        "db": str(args.db),
        "note": "verdict 为人工判定；correct vs (wrong + wrong_relation) 用于算分离度。"
                "uncertain 两边都不计。control=true 的条目取自 relations 表的既有断言。",
        "rules": RULES,
        "counts": dict(counts),
        "items": items,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入        : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
