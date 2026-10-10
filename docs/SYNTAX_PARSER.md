# 句法-语义解析层（`logos/parser/syntax_parser.py`）

> 适用对象：`logos/parser/syntax_parser.py`、`logos/parser/__init__.py`、`validate_syntax_parser.py`
> 依赖权威口径：`docs/CLOSURE_LAYER.md`（闭包/属性继承方向）、`docs/CLOSURE_DIRECTION.md`、`docs/CONNECTION_LAYER.md`
> 实测环境：Python 3.13.12（托管）+ SQLite（STRICT 表）
> 回归校验：`python validate_syntax_parser.py` → **PASS 214 / FAIL 0**（退出码 0）

---

## 1. 定位

本层是 Logos 的**唯一解析入口**：把一句中文串成一条可回溯的推理链，
全程**由数据库里的模板与约束驱动** —— 代码里不硬编码"把"、"被"这类词。

它是**只读消费层**：读句式模板、槽位约束、义素、关系与闭包，
唯一可选的写入是 `persist=True` 时的 `utterances` / `anaphora` 落盘。

五段流水线（每段都对应一个公开方法，可单独调用/单独测）：

```
tokenize → match_pattern → fill_slots → annotate_semantic_roles → query_relations
   ↓            ↓              ↓                ↓                      ↓
 list[Token] PatternMatch  dict[str,SlotFill] dict[str,SemanticRole] list[InferenceResult]
                              └───────────── parse() = 五段串联 ─────────────┘
```

---

## 2. 公开 API

| 方法 | 作用 | 返回 |
|---|---|---|
| `SyntaxParser(conn, **kw)` | 载入活跃句式/槽位/约束/虚词/角色/规则到内存缓存 | — |
| `tokenize(text)` | 正向最大匹配分词（纯词典，不接模型） | `list[Token]` |
| `match_pattern(tokens)` | 枚举所有 (句式 × 变体 × 切分)，返回**得分最高**者 | `PatternMatch \| None` |
| `match_candidates(tokens, limit=3, include_failed=True)` | 候选排名（含失败原因），失败诊断用 | `list[PatternAttempt]` |
| `fill_slots(match, tokens, context=None)` | 填槽 + 省略回填 + 话题链 | `dict[str, SlotFill]` |
| `annotate_semantic_roles(filled, match, tokens)` | 槽位 → 语义角色（含虚词改写） | `dict[str, SemanticRole]` |
| `query_relations(roles, match, tokens)` | 属性继承 + relations + 推理规则 | `list[InferenceResult]` |
| `parse(text, context=None)` | 端到端五段串联 | `ParseResult` |
| `parse_document(texts, doc_id=...)` | 批量 + 维护篇章状态 | `list[ParseResult]` |
| `explain(text)` | 人类可读完整报告（含推理链） | `str` |
| `cache_info()` / `reload()` | 缓存自检 / 重新载入 | `dict[str, int]` |

模块级便捷函数：`parse()`、`tokenize()`、`sense_text()`、`build_parser(db_path) -> (parser, conn)`、`compile_template(template)`。

构造参数要点：`use_db_context`（是否从 `utterances` 重建篇章状态）、`persist`（是否落盘）、
`min_coverage`（判定匹配成功的最低覆盖率，默认 0.5）、`max_slot_tokens`、`max_parses`、
`max_leading_residual` / `max_trailing_residual`（句首/句尾允许的残余 token 数）。

---

## 3. 数据依赖（读什么表、干什么用）

| 表 | 解析器用途 |
|---|---|
| `words` / `senses` | 词典：FMM 分词、`pos`、`subcategory`、`separator_pos`（离合词） |
| `sentence_patterns` | 句式模板（`WHERE is_productive = 1`），共 32 条 |
| `pattern_slots` | 槽位定义：`position` / `is_required` / `default_role` / `syntactic_function` |
| `slot_constraints` | 每条槽位的约束（`kind` × `value`），唯一的硬约束是 `kind='marker'` |
| `function_words` ⨯ `function_word_roles` | 虚词 → 角色改写（把→受事前置、被→施事前置…） |
| `semantic_roles` | 角色元数据：`case_marker` 用于角色合法性判据 |
| `ellipsis_rules` | 省略回填授权（`context_type` ⨯ `slot_role` → `strategy`） |
| `complements` | 动结/动趋/可能补语的 V+C 合法组合登记 |
| `bound_objects` / `separable_patterns` | 离合结构重组（`洗 了 个 澡`） |
| `relations` / `closure` | 关系查询与属性继承（走 `logos.db.closure`） |
| `inference_rules`（+ `rule_premises` / `rule_conclusions`） | 声明式规则推理 |
| `utterances` / `anaphora` / `topic_chains` | 篇章状态（`persist` 写入 / `use_db_context` 读出） |

---

## 4. 分词

正向最大匹配（FMM），词典优先，其次句式模板里的语法标记词（`_grammar_literals`，
如 `越来越` / `难道` / `还是`）—— 这样 `难道` 不会被切成 `难`+`道`。

### 4.1 离合词：**两条路径**都要覆盖

| 出现形式 | 处理 | 依据 |
|---|---|---|
| 整体（`洗澡`） | **切分** → `洗` + `澡`，两段同 `separable_group`，`sep_role = host/object` | `words.separator_pos` |
| 拆开（`洗 了 个 澡`） | **重组** → 整段共享同一 `separable_group`，中间成分 `sep_role = infix` | `bound_objects` |

只做切分不做重组，`洗了个澡` 就会退化成 4 个互不相干的 token。

### 4.2 OOV 兜底分级（`_guess_pos`）

```
封闭类词表 _CLOSED_CLASS_FALLBACK → 数词 → 指示代词 → 方位后缀 → pos='?'
```

**连续同类字符先合并成一个 token**（`_oov_run`）：`2026` → 一个 `m`；`上边` → 一个 `nl`。
若不合并，单字兜底只会看到单个字符 —— `2026` 会被切成 4 个 `?`，
且"数词"规则对阿拉伯数字**永远不生效**（见 §11 踩坑）。

---

## 5. 模板编译：`/` 的两种语义

模板里 `/` 的含义取决于**两侧有没有空格**（实测自 32 条真实模板的全部形态）：

| 写法 | 语义 | 例子 |
|---|---|---|
| `/` **独立**（两侧有空格） | **变体分隔** | `{V} 得 {C} / {V} 不 {C}` → 2 个变体 |
| `请/别` **紧贴** | **字面量同义候选** | `请/别 {V}` → 1 个变体，首项 alternatives=`('请','别')` |

判定规则：先按空白切项；切出的项若**整体等于** `/` 则是变体分隔符，
否则项内部的 `/` 是字面量候选。

---

## 6. 匹配与打分

`match_pattern` 遍历全部 (句式 × 变体 × 槽位切分)，逐个过 **7 道闸门**，再打分取最高。

### 6.1 闸门顺序（顺序即语义，改动前先看 §11）

| # | 闸门 | 拒绝条件 |
|---|---|---|
| ① | 必填槽位 | 必填槽位未填充且不在 `omitted` 里；或一次省略 > 1 个必填槽位 |
| ② | 同形约束 | 同名槽位多次出现但形式不同（`{V} 不 {V}` 的"吃不吃"必须同形） |
| ③ | 关键虚词标记 | 模板有字面量，却一个都没命中 |
| ④ | 短语结构专闸 | 动结/动趋/可能补语：不接受句级残余，且 V+C 必须在 `complements` 登记过 |
| ⑤ | **词汇证据** | 被消费的 token 全是 OOV（拒绝"靠结构空转"匹配） |
| ⑥ | 硬约束 | `kind='marker'` 未通过（**唯一**能一票否决的约束类型） |
| ⑦ | 覆盖率 | `coverage < min_coverage`（默认 0.5） |

**⑤ 与 ⑥ 的顺序是刻意的**：`known` 必须在此刻**现算**（`_lexical_evidence`），
而 `_score` 要到闸门 ⑦ 才执行。若闸门直接读 `_Parse.known`，会拿到 dataclass 默认值 `0`，
把所有正常句子一并否决 —— 见 §11 踩坑 #1。

### 6.2 评分公式

```
coverage = |字面量命中 ∪ 全部槽位区间| / |句内 token|

score = Σ IDF 加权字面量
        + Σ 槽位权重(必填 4.0 / 可选 2.0)
        + Σ 约束分(通过 +w / 违反 −1.5w；OOV 导致的无法核验按半价)
        + 结构先验(句法成分 ↔ 词性)
        × 句式类别权重 × (0.35 + 0.65 × coverage)
        + 补语命中 3.0 + 话题断句 2.0
        − 跳过代价 − 残余 1.5/个 − 槽位内夹虚词 4.0/个 − OOV 1.2/个 − 省略槽位 3.0/个
```

字面量权重带 IDF：`6 + 4 × ln(模板总数 / 含该字面量的模板数)` ——
`把`/`被`/`难道`/`越来越` 这类稀有标记权重大，`是` 这类高频字权重小。

句式类别权重：特殊句式/话题结构 1.15 > 复句/紧缩句 1.12 > 句类 1.10 > 短语结构 1.05 > 基本句型 1.00。

### 6.3 约束体系

| `kind` | 含义 | 严重度 |
|---|---|---|
| `marker` | 句中必须出现该标记 | **hard**（缺失即拒绝） |
| `lexeme` / `subcategory` / `pos` / `feature` / `animacy` / `countability` / `classifier` / `role` / `definiteness` / `aspect` | 各类软约束 | soft（违反扣分 + 点名） |

软约束违反会写进 `PatternMatch.constraint_errors`，格式为 `✗ [P] definiteness=definite —— 判定为 indefinite`
（**槽位 + 约束类型 + 实际判定**），并**不**否决匹配。

---

## 7. 槽位填充与省略

* **可选槽位**（`is_required=0`）可以留空，且**不进入** `filled_slots` —— 与"省略"是两件事。
* **必填槽位**只有在它的语义角色出现在 `ellipsis_rules.slot_role` 里时才允许留空
  （`_ellipsis_roles` 授权集），并且**一次至多省略 1 个**（`_MAX_OMITTED_SLOTS`）。
  例：`把吃。` 的 A、P 都无法填充 → `一次省略了 2 个必填槽位 → 模板未被有效实例化` → 拒绝。

回填策略（`_run_strategy`）全部由 `ellipsis_rules` 驱动：

| `strategy` | 取值来源 |
|---|---|
| `speaker` / `listener` | 上下文（`max_distance=0`，永远满足） |
| `previous_subject` / `previous_object` | 篇章状态里上文的施事/受事 |
| `topic` | 篇章状态 → `ctx.topic` → `topic_chains` 表（三级回退） |
| `previous_mentioned` / `salient_entity` / `nearest_np` | 上文的显著实体 |
| `world_knowledge` / `indefinite_default` | 返回**未解析占位符** `∅`（不伪造实体） |

`max_distance` 的语义是**最大跨句数**；超出即换下一条策略。

---

## 8. 语义角色与虚词改写

两条来源，优先级 **虚词改写 > 槽位默认角色**：

1. `pattern_slots.default_role`（谓语槽回退为 `述题`）
2. `function_words` ⨯ `function_word_roles`：`把 → 受事（前置）`、`被 → 施事（前置）`、
   `在 → 处所`、`从 → 起点/时点`、`向 → 方向`

这就是"把字句提 P（受事）、被字句提 A（施事）"的实现方式，**完全由数据库驱动**。

⚠️ **只认紧邻的介词**：从标记位向左跳过可跳过成分后**只看一个位置**。
若写成回溯 `idx-1 … idx-3`，`鱼被猫吃了` 里的 `被` 会越过 `猫` 把谓语 `吃` 也改写成"施事"
（见 §11 踩坑 #6）。

`SemanticRole` 区分 `text`（整段表层形式）与 `entity`（中心语词形）：
`受事="一条鱼"` 时 `entity="鱼"`。**推理链的主语必须用 `entity`**，
否则会打印成 `一条鱼 --can_do--> 吃`，看起来像"一条鱼"是知识库条目。

---

## 9. 关系查询与推理链

三路合流，输出统一为 `InferenceResult(kind, subject, relation, obj, confidence, chain, ...)`：

| 来源 | 依据 |
|---|---|
| 属性继承 | `closure.get_property_inheritance`（**只能继承上位属性**） |
| 关系直查 | `relations` 表 |
| 声明式规则 | `inference_rules` + `rule_premises` + `rule_conclusions` |
| 量词默认 | `noun_classifier_pairs` / `classifier_class_defaults` |
| 典型论元 | `relations` 里动词的 `agent` / `patient` + 类别核对 |

三个必须守住的点：

1. **链式前提要跨边合一**：`is_a` 传递规则是 `edge(A,is_a,B) ∧ edge(B,is_a,C)`。
   单边匹配做不到 —— 必须把同一变量在两条边间合一，否则该规则永不触发。
2. **邻域要扩张**：穷举边时把句面概念的 `is_a` 上/下位各扩 `2` 跳，
   否则 `猫`（句面）够不到 `动物 is_a 生物`。扩张的副作用是冒出无关推导，
   故再用 `focus` 集合（结论主语 ∈ 本句概念）过滤。
3. **多父类不能串成单链**：`鱼` 的上位是 `{动物, 食物}` **并列** + `生物(2跳)`，
   渲染为 `鱼 的上位：动物(1跳)、食物(1跳)、生物(2跳)`。
   拼成 `鱼 ⊑ 动物 ⊑ 食物 ⊑ 生物` 是错的。

结果按 `(subject, relation, obj)` **去重**，保留最高置信度。

---

## 10. 篇章状态

* `_remember` 把本句的角色并入内存 `_Discourse`（subject/object/topic + 计数）。
* `_persist`（`persist=True`）落盘 `utterances`（`seq` 从 1 起）与省略回填的 `anaphora`。
  ⚠️ **`_remember` 必须先于 `_persist`**：反过来的话首次落盘写出 `seq=0`，
  撞上 `utterances.CHECK (seq >= 1)` 而**静默降级**（见 §11 踩坑 #8）。
* `use_db_context=True` 时，`_discourse_state()` 会按需从
  `utterances ⨯ anaphora ⨯ topic_chains` **重建**篇章状态 —— 这是"跨进程省略回填"的落点：
  上一次运行 `persist` 落盘的主语，本次解析 `把鱼吃了。` 时能直接补回 `A=猫`。
  未开启该参数时**一个额外查询都不打**（由 `validate_syntax_parser.py` S11 断言）。

---

## 11. 实测踩坑清单

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | **闸门读到默认值**：`猫吃鱼。`/`猫吃鱼吗？`/`请你吃鱼。`/`他比我高。` 的 `pattern` 与 `mood` 全变 `None` | 闸门 ⑤ 在 `_score` 之前执行，而 `known` 由 `_score` 统计 → 读到 `_Parse.known` 默认值 `0`，一票否决所有句 | 抽出 `_lexical_evidence()` 作唯一真源，在闸门处**现算** |
| 2 | `2026` 切成 4 个 `?`；`3` 的 `pos='?'` | `_oov_token()` 只拿到**单字**，"数词"规则对阿拉伯数字永不生效 | `_oov_run()` 合并连续同类字符（数字串/方位串） |
| 3 | `洗了个澡` 退化成 4 个无关 token | 只做了离合词切分，没做重组 | 依 `bound_objects` 按 `separable_group` 重组，中间成分标 `infix` |
| 4 | `猫吃鱼。` 被 `可能补语式` 认领（V=猫吃 C=吃鱼） | 短语结构模板的 `{V} 得 {C}` 变体把残余吃干净 | 短语模板加两道闸：不收句级残余 + V+C 必须在 `complements` 登记 |
| 5 | `给我一本书。` 被 `主谓宾` 认领（S=给我一 V=本） | OOV 被当成合法填充 | `subcategory`/`pos` 约束对 OOV **不通过**（次类是词汇属性，词典没有就不能假定） |
| 6 | `鱼被猫吃了` 的谓语 `吃` 被改写成"施事" | 虚词改写回溯 `idx-1 … idx-3` 越过了 `猫` | 只看**紧邻**介词：跳过可跳过成分后只查一个位置 |
| 7 | 推理链把并列父类串成单链 `鱼 ⊑ 动物 ⊑ 食物 ⊑ 生物` | `_ancestor_entries` 丢失深度信息 | 返回 `[(id, depth)]`，渲染成 `动物(1跳)、食物(1跳)、生物(2跳)` |
| 8 | `persist=True` 从不生效（静默降级） | `_persist` 先于 `_remember`，`seq=0` 违反 `CHECK (seq >= 1)` | 调整顺序：先 `_remember` 再 `_persist` |
| 9 | `anaphora` 一律写 `form='∅', form_type='zero', node=NULL` | 落盘时没记填回的表层形式与先行节点 | 写入真实的 `form` / `form_type` / `antecedent_node_id`，使 `use_db_context` 可往返 |

---

## 12. 校验与基准

```bash
python validate_syntax_parser.py     # PASS 214 / FAIL 0，退出码 0
```

覆盖 S0–S12：建库与缓存、分词（含离合词双路径）、模板编译、**13 例句式矩阵**、
软/硬约束、省略授权、语义角色与虚词改写、推理链（含去重与多父类渲染）、
端到端 `ParseResult`、错误处理、**闸门顺序回归**、`use_db_context` 往返、只读性与公共接口。

13 例句式矩阵实测（`coverage` 为覆盖率，分数随词典/约束增补而变，模板命中**稳定**）：

| 输入 | 命中模板 | score | cov |
|---|---|---|---|
| 猫吃鱼。 | 主谓宾 | 17.50 | 100% |
| 猫把鱼吃了。 | 把字句 | 34.81 | 80% |
| 鱼被猫吃了。 | 被字句 | 36.08 | 80% |
| 鱼，猫吃了。 | 主谓谓语句 | 15.72 | 60% |
| 桌上有一条鱼。 | 存现句 | 21.70 | 100% |
| 我是人。 | 主系表 | 25.09 | 100% |
| 他比我高。 | 比较句-比 | 37.74 | 100% |
| 猫吃鱼吗？ | 是非问 | 26.40 | 100% |
| 猫吃不吃鱼？ | 正反问 | 14.90 | 60% |
| 给我一本书。 | 双宾句 | 11.85 | 100% |
| 请你吃鱼。 | 祈使句 | 23.40 | 100% |
| 看完。 | 动结式 | 11.40 | 100% |
| 看得完。 | 可能补语式 | 32.34 | 100% |

---

## 13. 已知边界（v0.1）

1. **纯词典分词，不接模型**：未见词只能靠单字兜底。`_CLOSED_CLASS_FALLBACK`
   是**临时桥梁**（90 个日常虚词），词典补齐后应逐项删除 —— 它兜出来的词
   `entry_id is None`，会计入 OOV 并轻罚（1.2），以此持续暴露"词典缺项"。
2. **组合爆炸防线**：每 (句式, 变体) 最多枚举 `max_parses=256` 种切分；
   超长句建议先按标点分句（`parse_document`）再逐句解析。
3. **`annotate_semantic_roles` 只改角色，不做同指消解**：代词/零形的**先行语判定**
   在 `fill_slots` 阶段完成（据 `ellipsis_rules`），本方法只负责"槽位 → 角色"映射与虚词改写。
4. **`pos='?'` 的 token 不会进入内容槽**：它在评分里按 OOV 计罚，
   并触发"词汇证据"闸门的边界判断。
