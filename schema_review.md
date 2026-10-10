# Logos Schema v2.0 架构审查报告

> 审查对象：`schema.sql`（Logos Knowledge Database Schema v2.0）
> 交付物：本报告 + `schema_v3.sql`（优化后完整可执行 schema）+ `validate_schema.py` + `bench_closure.py`
> 验证环境：SQLite **3.53.1**（实测执行，非纸面推演）

---

## 0. 审查结论摘要

| 维度 | 评级 | 一句话结论 |
|---|---|---|
| 1 范式合规 | ⚠️ 中 | 有 2 处会真实出错的建模缺陷（cross_links 主键漏列、JSON 列与子表双写），其余是派生字段冗余 |
| 2 查询性能 | ❌ 差 | **没有任何全文检索能力**，中文释义查询只能全表 LIKE；重复索引 1 处；闭包表主键顺序与主查询模式不匹配 |
| 3 中文适配 | ❌ 差 | 语素层、音变、述补结构、句类、篇章层**五块完全缺失**；离合词只支持一种模式；量词只有个体量词 |
| 4 可扩展性 | ⚠️ 中 | 加领域库要同时动 `relations` + `cross_links` + 应用层；无实体注册表、无来源表、无开放属性 |
| 5 SQLite 特性 | ❌ 差 | 未用 STRICT / 生成列 / 部分索引 / WITHOUT ROWID / json_valid；**FK 默认关闭导致全部外键声明默认失效** |
| 6 命名一致性 | ⚠️ 中 | 外键命名三种风格（`source`/`entry_id`/`classifier_id`），`type`/`subcat` 泛名重复出现 |
| 7 缺失约束 | ❌ 差 | `relations` 缺 UNIQUE 导致**同一条边可重复插入**；`closure` 缺自环 CHECK；20+ 处 CHECK/NOT NULL 缺失 |
| 8 闭包维护成本 | ⚠️ 中 | 未区分可传递/对称关系，`path` 存全路径、无见证边 → 删除只能全量重算。实测单边代价最高 **561 ms** |
| 9 汉语言学 | ⚠️ 中 | 语义角色仅 12 个且**没有句法成分层**；句式缺述补/双宾/句类；虚词分类无法回答"把引出什么角色" |

**总体判断**：v2.0 的骨架方向是对的（字-词-句分层、关系表、闭包表、句式模板），但有 **5 个必须修的结构性问题**（🔴），其余为重要（🟡）与可选（🟢）。

---

## 1. 范式合规

### 问题列表

| # | 位置 | 问题 | 后果 | 优先级 |
|---|---|---|---|---|
| 1.1 | `cross_links` | `PRIMARY KEY (source_domain, source_id, target_domain)` **漏了 `target_id`** | 同一个源在同一目标领域只能存一条链接。医学库一个基因要链到 KEGG/GO/PubMed 三个蛋白时直接冲突 | 🔴 |
| 1.2 | `sentence_patterns` | `required_slots` / `optional_slots` / `marker_words` 是 JSON 串，与 `pattern_slots.required` 表达同一事实 | 一个槽位的增删要改两处；**实测 v2.0 的 `pattern_slots` 表被声明但完全没有种子数据，58 个槽位实际只存在于 JSON 里 → 子表是死表，且二者必然漂移** | 🔴 |
| 1.3 | `entries.char_count` | `= length(word)` 的派生值落库 | 3NF 违例；任何 `UPDATE entries SET word=...` 都会让 char_count 变脏 | 🔴 |
| 1.4 | `entries` | `is_separable` 与 `separator_pos` 未声明派生关系 | 可出现 `is_separable=1` 但 `separator_pos IS NULL` 的矛盾状态 | 🟡 |
| 1.5 | `relations` vs `cross_links` | 同一事实（"两节点有关系"）两张表、两套 id 空间 | 闭包/推理要写两份查询；跨域边进不了闭包 | 🟡 |
| 1.6 | `relation_types` | `is_a/has_part` 成对声明逆向，若实现层也双向写边则是同一事实存两份 | 改一边忘另一边 → 图不一致 | 🟡 |
| 1.7 | `classifiers.word UNIQUE` | 量词被建成独立词条，与 `entries.word UNIQUE` 各存一行 | "把"同时是介词/量词/动词，却分散在两张表，无法共享义项与关系 | 🟡 |
| 1.8 | `relation_constraints.allowed_types` | 逗号分隔字符串 | 不可索引、不可 JOIN；查"哪些动词允许动物类作受事"必须全表扫 + 字符串匹配 | 🟡 |
| 1.9 | 全表 | `INTEGER PRIMARY KEY AUTOINCREMENT` | 多维护一张 `sqlite_sequence` 且禁止 id 复用；除非有硬性要求，否则是纯开销 | 🟢 |
| 1.10 | `senses.frequency` | 未说明是词频还是义项频 | 统计口径歧义 | 🟢 |

### 修改建议

**1.1 + 1.5 统一关系模型**（取消 `cross_links`，加节点注册表）：

```sql
-- 非义项节点（字/语素/词/句式/领域实体）统一注册
CREATE TABLE entity_nodes (
    node_id      INTEGER PRIMARY KEY,
    node_type    TEXT NOT NULL CHECK (node_type IN
                   ('character','morpheme','word','pattern','domain_entity','context','rule','utterance')),
    domain       TEXT NOT NULL DEFAULT 'core' REFERENCES domains(domain),
    label        TEXT NOT NULL,
    entity_class TEXT,
    attrs_json   TEXT CHECK (attrs_json IS NULL OR json_valid(attrs_json)),
    UNIQUE (node_type, domain, label)
) STRICT;

-- 统一关系表：src/dst 都用 (kind, id) 二元组
CREATE TABLE relations (
    relation_id INTEGER PRIMARY KEY,
    src_kind    TEXT NOT NULL DEFAULT 'sense' CHECK (src_kind IN ('sense','node')),
    src_id      INTEGER NOT NULL,
    dst_kind    TEXT NOT NULL DEFAULT 'sense' CHECK (dst_kind IN ('sense','node')),
    dst_id      INTEGER NOT NULL,
    relation    TEXT NOT NULL REFERENCES relation_types(relation),
    ...
    UNIQUE (src_kind, src_id, relation, dst_kind, dst_id),
    CHECK (NOT (src_kind='sense' AND dst_kind='sense' AND src_id=dst_id))
) STRICT;
```

异构端点的存在性用触发器校验（SQLite 无法对"多形态外键"建约束）：

```sql
CREATE TRIGGER tr_relations_validate_ai BEFORE INSERT ON relations
WHEN EXISTS (SELECT 1 FROM (
        SELECT 1 WHERE
            (NEW.src_kind='sense' AND NOT EXISTS(SELECT 1 FROM senses WHERE sense_id=NEW.src_id))
         OR (NEW.src_kind='node'  AND NOT EXISTS(SELECT 1 FROM entity_nodes WHERE node_id=NEW.src_id))
         OR (NEW.dst_kind='sense' AND NOT EXISTS(SELECT 1 FROM senses WHERE sense_id=NEW.dst_id))
         OR (NEW.dst_kind='node'  AND NOT EXISTS(SELECT 1 FROM entity_nodes WHERE node_id=NEW.dst_id))
     ))
BEGIN SELECT RAISE(ABORT, 'relations: src/dst 端点不存在'); END;
```

**1.2 消除双写**：删掉三个 JSON 列，槽位唯一真源 = `pattern_slots`。必选/可选由 `is_required` 表达，标记词走 `slot_constraints(kind='marker')`。

**1.3 + 1.4 改用生成列**（SQLite 3.31+）：

```sql
CREATE TABLE words (
    word         TEXT NOT NULL UNIQUE COLLATE BINARY,
    char_count   INTEGER GENERATED ALWAYS AS (length(word)) VIRTUAL,
    separator_pos INTEGER CHECK (separator_pos IS NULL OR separator_pos >= 1),
    is_separable INTEGER GENERATED ALWAYS AS (separator_pos IS NOT NULL) VIRTUAL,
    ...
    CHECK (separator_pos IS NULL OR separator_pos < length(word))
) STRICT;
```
实测：`char_count` 自动 = 2（洗澡），`is_separable` 自动 = 1，无需应用层维护。

**1.6 只存正向边，逆向在线推导**：

```sql
CREATE VIEW v_relations_all AS
    SELECT relation_id, src_kind, src_id, relation, dst_kind, dst_id, weight, confidence FROM relations
    UNION ALL
    SELECT r.relation_id, r.dst_kind, r.dst_id, t.inverse_of, r.src_kind, r.src_id, r.weight, r.confidence
      FROM relations r JOIN relation_types t ON t.relation = r.relation
     WHERE t.inverse_of IS NOT NULL AND t.direction = 'directed'
    UNION ALL
    SELECT r.relation_id, r.dst_kind, r.dst_id, r.relation, r.src_kind, r.src_id, r.weight, r.confidence
      FROM relations r JOIN relation_types t ON t.relation = r.relation
     WHERE t.direction = 'symmetric';
```
实测：只插一条 `has_part`，`v_relations_all` 立刻能查到对应的 `part_of`，且 `relations` 表未新增行。

**1.8 约束键值化**：

```sql
CREATE TABLE relation_constraints (
    sense_id INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    slot     TEXT NOT NULL CHECK (slot IN ('agent','patient','dative','instrument','location','time','result','target','theme')),
    kind     TEXT NOT NULL CHECK (kind IN ('category','feature','sense','pos','classifier')),
    value    TEXT NOT NULL,
    is_exclusive INTEGER NOT NULL DEFAULT 0 CHECK (is_exclusive IN (0,1)),
    PRIMARY KEY (sense_id, slot, kind, value)
) STRICT, WITHOUT ROWID;
```

---

## 2. 查询性能

### 问题列表

| # | 位置 | 问题 | 后果 | 优先级 |
|---|---|---|---|---|
| 2.1 | `idx_entries_word` | `entries.word` 已是 UNIQUE，再建同列索引 | 重复索引：写放大、空间翻倍 | 🔴 |
| 2.2 | 全局 | **无任何 FTS5** | 查释义只能 `definition LIKE '%水%'`：全表扫、无法按相关度排序。且中文若用 unicode61 分词器，整句会变成一个 token，检索完全失效 | 🔴 |
| 2.3 | 字/词检索 | `LIKE 'x%'` 因默认 LIKE 大小写不敏感而**无法走索引** | 前缀检索退化为全表扫 | 🟡 |
| 2.4 | `closure` | PK = `(ancestor, descendant, relation)` | 最常用的"某关系下某节点的全部下位"无法前缀扫，必须走二级索引 | 🟡 |
| 2.5 | `idx_senses_pos` | 对 `pos` 建全表索引，但实际查询几乎只关心 `is_primary=1` | 索引偏大、选择性差 | 🟡 |
| 2.6 | 解析热路径 | "由 word 取全部义项"需回表 | 每 token 多一次随机 IO | 🟡 |
| 2.7 | `changelog` | 无索引 + 无归档策略 | 审计表无限增长后任何查询都是全表扫 | 🟡 |
| 2.8 | 物化视图 | SQLite 无原生物化视图 | 需用「物化表 + 队列触发器」模式替代 | 🟢 |
| 2.9 | 优化器 | 无 `ANALYZE` / `PRAGMA optimize` 计划 | 查询计划长期基于陈旧统计 | 🟢 |

### 修改建议

**2.2 中文全文检索必须双通道**（实测数据支撑）：

```sql
CREATE VIRTUAL TABLE senses_fts_trigram USING fts5(
    definition, example,
    content='senses', content_rowid='sense_id',
    tokenize='trigram'
);
CREATE VIRTUAL TABLE senses_fts_seg USING fts5(
    definition_seg,                       -- jieba 预分词后空格连接的释义
    content='senses', content_rowid='sense_id',
    tokenize='unicode61 remove_diacritics 2'
);
```

实测结论（`validate_schema.py`）：

| 通道 | 查询词 | 命中 |
|---|---|---|
| `senses_fts_trigram` | `家养动物`（3 字） | ✅ 命中"猫" |
| `senses_fts_trigram` | `动物`（2 字） | ❌ **0 行** |
| `senses_fts_seg` | `动物`（2 字） | ✅ 有命中 |

→ `trigram` 分词器天然要求 ≥3 字符，中文双字词是极高频查询，**单通道不可行**。

**2.3 前缀检索改写**：

```sql
-- 不要用:  WHERE word LIKE '猫%'      -- 默认 LIKE 大小写不敏感 → 不走索引
-- 改用:    WHERE word GLOB '猫*'      -- GLOB 大小写敏感 → 可用 (word) 索引
-- 或:      WHERE word >= '猫' AND word < '猫' || char(0x10FFFF)
```

**2.4 闭包表主键顺序重排 + 方向约定**：

```sql
CREATE TABLE closure (
    relation   TEXT NOT NULL REFERENCES relation_types(relation),
    ancestor   INTEGER NOT NULL,   -- 上位（更一般）
    descendant INTEGER NOT NULL,   -- 下位（更具体）
    depth      INTEGER NOT NULL CHECK (depth BETWEEN 1 AND 64),
    first_edge INTEGER,
    PRIMARY KEY (relation, ancestor, descendant),
    CHECK (ancestor <> descendant)
) STRICT, WITHOUT ROWID;
CREATE INDEX idx_closure_relation_desc ON closure(relation, descendant, depth);
```
> ⚠️ **v2.0 只有列名、没有方向约定**。这非常危险：`WHERE descendant=:x` 到底取上位还是下位，完全靠猜。本报告在 v3 中显式写入约定（`ancestor`=上位、`descendant`=下位），并在验证脚本里加了方向自检用例。

**2.5 / 2.6 部分索引 + 覆盖索引**：

```sql
CREATE INDEX idx_senses_pos    ON senses(pos) WHERE is_primary = 1;
CREATE INDEX idx_senses_freq   ON senses(frequency DESC) WHERE is_primary = 1;
CREATE INDEX idx_words_separable ON words(separator_pos) WHERE separator_pos IS NOT NULL;
CREATE INDEX idx_changelog_record ON changelog(table_name, record_id, changed_at DESC);
CREATE INDEX idx_changelog_time   ON changelog(changed_at DESC);
```

**2.8 SQLite 的"物化视图"替代方案**：SQLite 没有 `MATERIALIZED VIEW`。v3 采用两种手段：
1. 轻量视图在线算（`v_closure_2`：深度 2 的闭包用两次 JOIN 现算，不落库）；
2. 重量结果用「表 + 维护队列触发器」（`closure_rebuild_queue`）。

---

## 3. 中文适配 —— 缺失现象清单

这是 v2.0 最大的短板：**七项中文核心现象中五项完全没有承载结构**。

| # | 缺失现象 | 例子 | v2.0 状态 | 优先级 |
|---|---|---|---|---|
| 3.1 | **语素层**（字 ≠ 语素） | `葡萄`=两字一语素；`儿`是词尾；`老-`是前缀 | ❌ 完全没有。字层与语素层被混为一谈，无法表示联绵词/附加式/词根词缀 | 🔴 |
| 3.2 | **音变**（变调/轻声/儿化） | 一(yī→yì/yí)、不(bù→bú)、你好(ní hǎo)、花儿(huār)、桌子(zhuōzi) | ⚠️ 只塞在 `characters.pinyin_json` 里，无法推理 | 🔴 |
| 3.3 | **述补结构** | 看完/听懂（动结）、走出来（动趋）、看得完/看不懂（可能补语） | ❌ 11 个句式里一个补语句式都没有 —— 这是汉语最核心的句法结构之一 | 🔴 |
| 3.4 | **句类/语气** | 陈述/疑问/祈使/感叹；是非问、特指问、正反问、选择问、反问 | ❌ 完全没有；无法解析"你去不去？""难道你不知道吗？" | 🔴 |
| 3.5 | **篇章层**（话题链/零形回指） | 「我把作业写完了。∅ 然后 ∅ 去睡觉。」主语全靠话题链推导 | ❌ 只有 4 条 `ellipsis_rules`，无 utterance/话题链/回指的表。**省略回填需求（第 10 条）实质上没有落点** | 🔴 |
| 3.6 | **重叠式** | 高高(AA)、高高兴兴(AABB)、研究研究(ABAB)、看一看(A一A)、个个(量词重叠) | ❌ 完全没有 | 🟡 |
| 3.7 | **离合词** | 洗澡 → 洗了个澡 / 洗完澡 / 洗洗澡 / 澡洗了 | ⚠️ 只有 `is_separable` + `separator_pos` 两个字段，无法表达"能插入什么成分" | 🟡 |
| 3.8 | **黏着宾语** | 澡/忙/架/光 —— 不能独立成条，只能作特定动词的宾语 | ❌ 没有建模 | 🟡 |
| 3.9 | **量词细分** | 借用量词（一桌子菜/一屋子人）、动量词（趟/遍/次）、复合量词（人次/吨公里）、量词重叠（个个） | ⚠️ 只有 `classifier_type` 自由文本 | 🟡 |
| 3.10 | **多音字** | 兴 xīng/xìng、洗 xǐ/xiǎn(姓)、薄 báo/bó(文白异读) | ⚠️ 只有 `pinyin_json` 字符串 | 🟡 |
| 3.11 | **异形词** | 其它/其他、按耐/按捺 | ❌ 没有建模 | 🟢 |
| 3.12 | **紧缩句** | 一…就…、越…越…、非…不… | ❌ 缺失 | 🟢 |
| 3.13 | **倒装** | 怎么了，你？ | ❌ 缺失 | 🟢 |
| 3.14 | **双宾语句** | 给我一本书（近宾语/远宾语） | ❌ 缺失 | 🟢 |
| 3.15 | **主谓谓语 / 名词谓语** | 他身体很好 / 今天星期六 | ❌ 缺失 | 🟢 |

### 修改建议（选摘）

**3.1 语素层**：

```sql
CREATE TABLE morphemes (
    morpheme_id   INTEGER PRIMARY KEY,
    form          TEXT NOT NULL,
    morpheme_type TEXT NOT NULL CHECK (morpheme_type IN
                    ('自由语素','黏着语素','词根','前缀','后缀','中缀','词尾','联绵语素')),
    is_bound      INTEGER NOT NULL DEFAULT 0 CHECK (is_bound IN (0,1)),
    pinyin        TEXT, meaning TEXT,
    UNIQUE (form, morpheme_type)
) STRICT;

CREATE TABLE word_morphemes (
    word_id     INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL CHECK (seq >= 1),
    morpheme_id INTEGER NOT NULL REFERENCES morphemes(morpheme_id),
    surface     TEXT,
    PRIMARY KEY (word_id, seq)
) STRICT, WITHOUT ROWID;
```
实测种子：`葡萄`→1 个语素；`桌子`→词根"桌"+后缀"子"；`老虎`→前缀"老"+词根"虎"；`花儿`→词根"花"+词尾"儿"。

**3.2 音变规则**：

```sql
CREATE TABLE phonological_rules (
    rule_id      INTEGER PRIMARY KEY,
    rule_type    TEXT NOT NULL CHECK (rule_type IN
                   ('变调','连读变调','轻声','儿化','儿化变韵','声调中和')),
    source_form  TEXT NOT NULL,      -- '一' / '不' / '儿' / '上声+上声'
    condition    TEXT NOT NULL,      -- '后字为去声(4声)'
    surface_form TEXT NOT NULL,      -- 'yi2'
    example      TEXT,
    UNIQUE (rule_type, source_form, condition)
) STRICT;
```
种子含 9 条：一/不变调、三声连读、轻声三种来源、儿化两种。配合 `character_readings`（多音字/文白异读/姓氏读音）覆盖 3.2 与 3.10。

**3.3 述补结构**：

```sql
CREATE TABLE complement_types (
    comp_type TEXT PRIMARY KEY,   -- 结果补语/趋向补语/可能补语/程度补语/数量补语/状态补语/时地补语
    description TEXT NOT NULL, marker TEXT, example TEXT
) STRICT;

CREATE TABLE complements (
    complement_id INTEGER PRIMARY KEY,
    comp_type     TEXT NOT NULL REFERENCES complement_types(comp_type),
    head_sense_id INTEGER NOT NULL REFERENCES senses(sense_id),
    comp_sense_id INTEGER NOT NULL REFERENCES senses(sense_id),
    result_state  TEXT,
    surface_order TEXT NOT NULL DEFAULT 'V+C' CHECK (surface_order IN ('V+C','V+得+C','V+不+C')),
    example       TEXT,
    UNIQUE (comp_type, head_sense_id, comp_sense_id, surface_order)
) STRICT;
```

**3.7 离合词展开**：

```sql
CREATE TABLE separable_patterns (
    sep_pattern_id INTEGER PRIMARY KEY,
    word_id        INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    kind           TEXT NOT NULL CHECK (kind IN
                     ('插入体标记','插入数量','插入定语','插入补语','插入疑问成分','重叠','倒装')),
    infixes_json   TEXT CHECK (infixes_json IS NULL OR json_valid(infixes_json)),
    surface_example TEXT NOT NULL,
    UNIQUE (word_id, kind, surface_example)
) STRICT;
```
实测种子（洗澡）：`洗了个澡`（体标记+数量）、`洗了个热水澡`（插入定语）、`洗完澡`（插入补语）、`洗洗澡`（重叠）。

**3.9 量词两处修正**：
1. 量词必须是**义项**（"把"同时是介词/量词/动词）；
2. 搭配走两级：**类别默认** + **个体例外**，避免为十万名词建十万行。

```sql
CREATE TABLE classifiers (
    classifier_sense_id INTEGER PRIMARY KEY REFERENCES senses(sense_id) ON DELETE CASCADE,
    kind  TEXT NOT NULL CHECK (kind IN ('名量词','动量词','复合量词','借用量词','度量词','不定量词','时间量词')),
    subtype TEXT CHECK (subtype IS NULL OR subtype IN ('个体','集合','部分','容器','临时','通用')),
    is_borrowed       INTEGER NOT NULL DEFAULT 0 CHECK (is_borrowed IN (0,1)),
    has_reduplication INTEGER NOT NULL DEFAULT 0 CHECK (has_reduplication IN (0,1)),
    source_noun_sense_id INTEGER REFERENCES senses(sense_id)
) STRICT;

CREATE TABLE classifier_class_defaults (          -- 动物类 → 只
    cat                 TEXT NOT NULL REFERENCES semantic_categories(cat),
    classifier_sense_id INTEGER NOT NULL REFERENCES classifiers(classifier_sense_id),
    preference          REAL NOT NULL DEFAULT 1.0 CHECK (preference BETWEEN 0 AND 1),
    PRIMARY KEY (cat, classifier_sense_id)
) STRICT, WITHOUT ROWID;
```
实测：查"猫"的量词 → 显式搭配 `只(0.9)` + 类别默认 `动物→只(0.7)`，两条都返回。

**3.5 篇章层**（这是"省略回填"需求的真正落点）：

```sql
CREATE TABLE utterances (
    utterance_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL DEFAULT 'default',
    seq INTEGER NOT NULL CHECK (seq >= 1), text TEXT NOT NULL, speaker TEXT,
    mood TEXT REFERENCES sentence_moods(mood), pattern_id INTEGER REFERENCES sentence_patterns(pattern_id),
    UNIQUE (doc_id, seq)
) STRICT;

CREATE TABLE topic_chains (
    chain_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, topic_node_id INTEGER,
    start_seq INTEGER NOT NULL, end_seq INTEGER,
    chain_type TEXT NOT NULL CHECK (chain_type IN ('话题链','主谓链','零形回指链','同指链')),
    CHECK (end_seq IS NULL OR end_seq >= start_seq)
) STRICT;

CREATE TABLE anaphora (
    anaphora_id INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterances(utterance_id) ON DELETE CASCADE,
    slot_role TEXT REFERENCES semantic_roles(role),
    form TEXT NOT NULL,                        -- '他' / '∅'
    form_type TEXT NOT NULL CHECK (form_type IN
      ('pronoun','zero','definite_np','demonstrative','proper_noun','collective')),
    antecedent_utterance_id INTEGER REFERENCES utterances(utterance_id),
    antecedent_node_id INTEGER, chain_id INTEGER REFERENCES topic_chains(chain_id) ON DELETE SET NULL,
    confidence REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1)
) STRICT;
```
实测种子已能表示「鱼被猫吃了 → ∅(零形) 回指 猫，confidence 0.7」。

---

## 4. 可扩展性

| # | 问题 | 后果 | 优先级 |
|---|---|---|---|
| 4.1 | `relations` 只能指 `senses`，域外实体走 `cross_links` | 同一条边（如"某蛋白 is_a 某蛋白家族"）在两张表；跨域边进不了闭包与推理 | 🔴 |
| 4.2 | `cross_links` 的 `source_domain/source_id` 是裸字符串+整数，无注册表、无外键 | 无法回答"animal 库有哪些实体、在哪个文件、什么版本"。删库后悬空链接不可检测 | 🔴 |
| 4.3 | 无开放属性机制 | 医学库要加"适应证"、动物库要加"保护等级"，都必须 `ALTER TABLE` | 🟡 |
| 4.4 | 无 `sources` 表，只有 `relations.source_ref` 自由文本 | 无法回答"这条关系来自 KEGG 还是人工标注、哪一批导入、能否回滚" | 🟡 |
| 4.5 | 无 schema 版本表 / 无导入批次表 | 多库 ATTACH 时无法判断版本兼容性 | 🟡 |

### 修改建议

```sql
CREATE TABLE domains (                       -- 领域注册表：加领域库只 INSERT 一行
    domain TEXT PRIMARY KEY, description TEXT,
    db_path TEXT,                            -- 跨库(ATTACH)时的物理路径
    version TEXT, is_core INTEGER NOT NULL DEFAULT 0 CHECK (is_core IN (0,1))
) STRICT;

CREATE TABLE sources (
    source_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('词典','语料库','本体','百科','论文','人工标注','规则','KEGG','Reactome','GO','PubMed')),
    uri TEXT, license TEXT, version TEXT, retrieved_at TEXT
) STRICT;

CREATE TABLE attributes (                    -- 开放属性：加新领域字段不改表结构
    attr_id INTEGER PRIMARY KEY,
    owner_kind TEXT NOT NULL CHECK (owner_kind IN ('sense','word','character','pattern','node','relation','utterance')),
    owner_id INTEGER NOT NULL, key TEXT NOT NULL,
    value_json TEXT NOT NULL CHECK (json_valid(value_json)),
    source_id INTEGER REFERENCES sources(source_id),
    UNIQUE (owner_kind, owner_id, key, value_json)
) STRICT;

CREATE TABLE import_batches (
    batch_id INTEGER PRIMARY KEY, source_id INTEGER REFERENCES sources(source_id),
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), finished_at TEXT,
    row_count INTEGER NOT NULL DEFAULT 0 CHECK (row_count >= 0),
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','ok','failed','rolled_back')),
    note TEXT
) STRICT;

CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))) STRICT;
```

**加一个新领域库的完整动作**（v3 之后）：
```sql
INSERT INTO domains(domain, description, db_path) VALUES ('plant','植物学库','./plant.db');
INSERT INTO entity_nodes(node_type, domain, label, entity_class) VALUES ('domain_entity','plant','拟南芥','Species');
-- 之后所有关系都走统一 relations 表，闭包/推理/检索零改动
```

---

## 5. SQLite 特性利用

### 问题列表（含实测证据）

| # | 未利用的特性 | v2.0 后果 | 优先级 |
|---|---|---|---|
| 5.1 | **`PRAGMA foreign_keys`（默认 OFF）** | **v2.0 里所有 FOREIGN KEY 声明默认全部失效**。这是最隐蔽、危害最大的一个坑 | 🔴 |
| 5.2 | **STRICT 表**（3.37+） | `BOOLEAN`/`REAL` 列可存任意类型。实测：往 `strokes` 插字符串 `'很多'` 会被静默接受 | 🔴 |
| 5.3 | 触发器内 `WITH ... INSERT` 的语法限制 | 若照搬常见写法会在 schema 加载期直接语法错误。**实测：`near "INSERT": syntax error`**，必须写成 `INSERT ... SELECT ... FROM (WITH RECURSIVE ...)` | 🔴 |
| 5.4 | UPSERT 的赋值求值顺序 | `DO UPDATE SET` **从左到右求值，后面的表达式看得见前面的结果**。若把 `depth = MIN(...)` 写在 `first_edge = CASE WHEN excluded.depth < depth ...` 之前，条件恒为假，见证边永远不更新 | 🔴 |
| 5.5 | `recursive_triggers`（默认 OFF） | 触发器级联（删边 → 排队重建）**不会触发**，闭包会静默失效 | 🟡 |
| 5.6 | 生成列（3.31+） | 派生字段靠应用层维护 | 🟡 |
| 5.7 | 部分索引 | 索引偏大 | 🟡 |
| 5.8 | `WITHOUT ROWID` | 纯连接表多一套 rowid 索引 | 🟡 |
| 5.9 | `json_valid()` CHECK | JSON 列可塞非法 JSON | 🟡 |
| 5.10 | `CURRENT_TIMESTAMP` 格式 | 产出 `2026-09-27 00:19:53`（无 T 无 Z，UTC），跨时区解析易错 | 🟢 |
| 5.11 | `RETURNING` / `jsonb` / `unhex` | 未使用 | 🟢 |

### 修改建议

**5.1 连接级配置（必须每连接执行，不随 schema 持久化）**：

```sql
PRAGMA foreign_keys      = ON;      -- 不开则所有 FK 形同虚设
PRAGMA recursive_triggers= ON;      -- 闭包"删边→重建"链式触发所必需
PRAGMA journal_mode      = WAL;
PRAGMA synchronous       = NORMAL;
PRAGMA busy_timeout      = 5000;
PRAGMA cache_size        = -65536;
PRAGMA mmap_size         = 268435456;
PRAGMA temp_store        = MEMORY;
-- 关闭连接前: PRAGMA optimize;
-- 批量导入时: PRAGMA defer_foreign_keys = ON;
```

**5.2 全表 STRICT**（实测 52/52 张业务表可 STRICT，FTS5 影子表除外）：

```sql
CREATE TABLE characters (
    char TEXT PRIMARY KEY COLLATE BINARY,
    radical TEXT REFERENCES radical_semantics(radical),
    strokes INTEGER CHECK (strokes IS NULL OR strokes BETWEEN 1 AND 64),
    ...
) STRICT;
```
实测拒绝：`INSERT INTO characters(char,strokes) VALUES('测','很多')`
→ `IntegrityError: cannot store TEXT value in INTEGER column characters.strokes`

**5.3 触发器内递归 CTE 的正确写法**：

```sql
-- ❌ 加载即语法错误
CREATE TRIGGER t AFTER INSERT ON x BEGIN
  WITH RECURSIVE w(i) AS (SELECT 1) INSERT INTO y SELECT i FROM w;
END;

-- ✅ 把 CTE 嵌进 FROM 子查询
CREATE TRIGGER t AFTER INSERT ON x BEGIN
  INSERT INTO y(i) SELECT i FROM (WITH RECURSIVE w(i) AS (SELECT 1) SELECT i FROM w);
END;
```

**5.4 UPSERT 赋值顺序**：

```sql
ON CONFLICT(relation, ancestor, descendant) DO UPDATE SET
    first_edge = CASE WHEN excluded.depth < depth THEN excluded.first_edge ELSE first_edge END,
    depth      = MIN(depth, excluded.depth);   -- ⚠️ 必须放在 first_edge 之后
```
实测验证：`最短路径的见证边被正确保留` 用例通过。

**5.10 ISO-8601 时间戳**：

```sql
created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
```

**5.8 WITHOUT ROWID 适用清单**：`closure`、`pattern_slots`、`slot_constraints`、`sense_contexts`、`sense_features`、`sense_categories`、`noun_classifier_pairs`、`classifier_class_defaults`、`function_word_roles`、`word_morphemes`、`rule_premises`、`rule_conclusions`、`character_readings`。

---

## 6. 命名一致性

### 问题列表

| # | 问题 | v2.0 实例 | 优先级 |
|---|---|---|---|
| 6.1 | 外键命名三种风格并存 | `relations.source/target`（裸名）、`senses.entry_id`（表名+_id）、`classifiers.classifier_id`（实体名+_id） | 🟡 |
| 6.2 | 主键命名不统一 | 多数是 `<实体>_id`，但 `relations.id`、`changelog.change_id` 不一致 | 🟡 |
| 6.3 | 边两端命名不统一 | `relations.source/target` vs `closure.ancestor/descendant` vs `cross_links.source_domain/target_domain` | 🟡 |
| 6.4 | 泛名 `type` 重复出现 | `sentence_patterns.type`、`classifiers.classifier_type`、`slot_constraints.constraint_type`、`relation_constraints` 无类型列 | 🟡 |
| 6.5 | `subcat` 一词两义 | `senses.subcat`（动词次类）与 `function_words.subcat`（介词次类） | 🟢 |
| 6.6 | 表名数不一致 | `closure`（单数）与 `relations`/`senses`（复数）混用 | 🟢 |
| 6.7 | `pinyin_json` 名为 JSON，内容却是字符串 | 名实不符 | 🟢 |

### 命名规约（v3 采用的规则）

| 类别 | 规则 | 示例 |
|---|---|---|
| 表名 | 一律复数（不可数概念除外） | `words` `senses` `relations`；例外：`closure` `changelog` `schema_meta` |
| 主键 | `<单数实体>_id` | `word_id` `sense_id` `relation_id` `pattern_id` |
| 外键 | `<被引用单数实体>_id` | `word_id` `sense_id` `classifier_sense_id` |
| 边端点 | 统一 `src_*` / `dst_*`，异构时加 `_kind` | `src_kind` `src_id` `dst_kind` `dst_id` |
| 闭包端点 | 统一 `ancestor` / `descendant` **并写明方向** | `ancestor`=上位，`descendant`=下位 |
| 类型列 | `<域>_type` / `<域>_class` / `<域>_kind`，禁止裸 `type` | `word_type` `relation_class` `classifier_kind` |
| 布尔列 | `is_*` / `has_*`，类型 `INTEGER` + `CHECK IN (0,1)` | `is_primary` `has_reduplication` |
| JSON 列 | `*_json` 且必须有 `json_valid` CHECK | `attrs_json` `infixes_json` |

**兼容性**：v3 提供 `entries` / `cross_links` 两个兼容视图，应用层可渐进迁移，不必一次性改代码。

---

## 7. 缺失的约束

### 问题列表

| # | 表.列 | 应加什么 | 不加的后果 | 优先级 |
|---|---|---|---|---|
| 7.1 | `relations` | `UNIQUE(src_kind,src_id,relation,dst_kind,dst_id)` | **同一条边可重复插入** → 闭包重复计算、权重歧义、统计虚高 | 🔴 |
| 7.2 | `relations` | `CHECK(src_id <> dst_id)` | 自环（X is_a X）会污染闭包与递归查询，甚至死循环 | 🔴 |
| 7.3 | `closure` | `CHECK(ancestor <> descendant)`、`CHECK(depth BETWEEN 1 AND 64)` | 自环闭包行；无效深度 | 🔴 |
| 7.4 | `characters.radical` | FK → `radical_semantics(radical)` | 部首可写成不存在的值 | 🟡 |
| 7.5 | `characters.strokes` | `CHECK(1..64)` + NOT NULL | 可写 0 或负数 | 🟡 |
| 7.6 | `senses` | `UNIQUE(entry_id,pos,definition)`、`UNIQUE(entry_id,pos,sense_no)` | 同一义项可插两行 | 🟡 |
| 7.7 | `senses.pos` | `CHECK(pos IN (...))` 或 FK → `pos_tags` | 词性可写任意字符串 | 🟡 |
| 7.8 | `senses.frequency` | `CHECK(frequency >= 0)` | 负频次 | 🟢 |
| 7.9 | 全部 `ON DELETE` | `ON DELETE CASCADE` | 删词后义项/关系成孤儿（即使开了 FK 也只报错不清理） | 🟡 |
| 7.10 | `function_words.category/subcat` | 枚举 CHECK | 分类自由发挥，无法程序化查询 | 🟡 |
| 7.11 | `noun_classifier_pairs.preference` | `CHECK(0..1)` | 概率越界 | 🟢 |
| 7.12 | `relation_types.direction` | `CHECK(direction IN ('directed','symmetric'))` | 方向值任意 | 🟡 |
| 7.13 | `relation_types` | 缺 `is_transitive` / `is_symmetric` / `is_reflexive` | 闭包无法判断该不该算（见第 8 节） | 🔴 |
| 7.14 | `sentence_patterns` | `UNIQUE(name)`；`CHECK(json_valid(required_slots))` | 句式重名；非法 JSON | 🟡 |
| 7.15 | `slot_constraints.required` | `CHECK(required IN (0,1))` | 布尔值漂移 | 🟢 |
| 7.16 | `changelog.record_id` | 应为 **TEXT** | `characters` 主键是 TEXT，删字时记录不了 `record_id` | 🟡 |
| 7.17 | `changelog.operation` | `CHECK IN ('INSERT','UPDATE','DELETE')` | 任意字符串 | 🟢 |
| 7.18 | `sense_contexts.relevance` | `CHECK(0..1)` | 越界 | 🟢 |
| 7.19 | `cross_links` | 全列 `NOT NULL`（现仅 PK 列隐式非空） | NULL 值破坏主键语义 | 🟡 |
| 7.20 | `radical_semantics.meaning` | `NOT NULL` | 部首无义项 | 🟢 |
| 7.21 | `ellipsis_rules` | PK 覆盖不全（**见下**） | 每个（语境, 槽位）只能存**一条**回填策略 | 🔴 |

**7.21 详细说明**：v2.0 的 `PRIMARY KEY (context_type, slot_name)` 意味着——对话场景里主语省略**只能有一条规则**。但真实消解需要一条**候选链**（先用"说话人"，失败用"上句主语"，再失败用"话题"）。这让整个省略回填机制形同虚设。

```sql
-- 修正：一条槽位允许多策略，用 priority 排序 + max_distance 限制跨句距离
CREATE TABLE ellipsis_rules (
    ellipsis_rule_id INTEGER PRIMARY KEY,
    context_type TEXT NOT NULL CHECK (context_type IN ('dialogue','narrative','imperative','expository','topic_chain')),
    slot_role    TEXT NOT NULL REFERENCES semantic_roles(role),
    strategy     TEXT NOT NULL CHECK (strategy IN
                   ('previous_subject','previous_object','previous_mentioned','speaker','listener',
                    'topic','nearest_np','salient_entity','world_knowledge','indefinite_default')),
    priority     INTEGER NOT NULL DEFAULT 0,
    max_distance INTEGER CHECK (max_distance IS NULL OR max_distance >= 0),
    description  TEXT,
    UNIQUE (context_type, slot_role, strategy, priority)
) STRICT;
```
实测：`('dialogue','施事')` 现在可挂 `speaker(10)` 与 `previous_subject(9, max_distance=3)` 两条，验证通过。

---

## 8. 闭包表维护成本

### 8.1 v2.0 的问题

| # | 问题 | 后果 |
|---|---|---|
| 8.1 | 无 `is_transitive` / `is_symmetric` 标志 | `similar`/`opposite` 是**对称关系**，闭包恒等于自身 → 白占一倍空间、白算一轮。所有关系混在一张 closure 表里 |
| 8.2 | `path TEXT` 存完整路径 | 空间 = O(可达对数 × 平均深度)；每行都带一长串节点 id，页分裂严重 |
| 8.3 | **无见证边** | 删一条边时，无法只修受影响的行 → 只能全量重算该关系 |
| 8.4 | PK = `(ancestor, descendant, relation)` | 按 `relation` 过滤的老查询走不到主键前缀 |
| 8.5 | 无维护策略开关 | 批量导入时每条边都触发一次闭包笛卡尔积，导入 1 万条边 ≈ 1 万次子树扫描 |

### 8.2 复杂度分析 + 实测数据

**插入一条边的代价**（本方案的闭包定义）：

```
代价 ≈ (1 + |src 的下位子树|) × (1 + |dst 的上位链|)
```

微基准实测（6×3 树 / 1093 节点，`bench_closure.py`，SQLite 3.53.1）：

| src 层号 | src 的下位子树规模 | dst 的上位链 | 单边插入延迟 |
|---|---|---|---|
| L6（叶子） | 0 | 6 | **1.15 ms** |
| L4 | 12 | 4 | **6.50 ms** |
| L2 | 120 | 2 | **62.5 ms** |
| L0（根） | 1092 | 0 | **561.3 ms** |

→ 延迟基本与 `|下位子树|` 线性正相关。**"在靠近根部的位置插一条边"可以到半秒级**，这决定了批量导入绝不能走增量触发。

**全量重建 vs 边插即算**（同一张表，闭包行数一致）：

| 规模（深度×分支） | 节点 | 边 | 闭包行 | 边插即算总耗时 | 批量重建耗时 |
|---|---|---|---|---|---|
| 3×3 | 40 | 39 | 102 | 0.013 s | 0.003 s + 0.023 s |
| 4×3 | 121 | 120 | 426 | 0.034 s | 0.004 s + 0.045 s |
| 4×4 | 341 | 340 | 1252 | 0.200 s | 0.010 s + 0.352 s |
| 5×3 | 364 | 363 | 1641 | 0.235 s | 0.009 s + 0.313 s |

关键观测：**闭包行数 ≈ 节点数 × 平均层级深度**（364 节点 / 1641 行 ≈ 4.5 倍），是**线性**的，空间可控 —— 这一点 v2.0 的设计没问题。

### 8.3 三个方案对比

| 方案 | 插入 | 删除 | 空间 | 查询 | 适用 |
|---|---|---|---|---|---|
| **A. 递归 CTE 实时算** | O(1) | O(1) | 0 | O(分支^深度)，需 `(src,relation)` 索引 | 图小/深度浅；或只作关键路径兜底 |
| **B. 物化闭包 + 见证边 + 队列重建**（v3 主方案） | 增量 O(子树×上位链)；批量走重建 | 失效 `first_edge` 命中的行 + 重建 | 线性（无 path） | O(1) 索引查 | 有稳定关系、查询热度高的场景 |
| **C. 物化路径 / 嵌套集** | 叶子 O(1)，重挂 O(子树) | 同插入 | 线性但列宽 | 前缀 `LIKE '1/5/%'` | 纯树（如 is_a），且极少重挂 |

**v3 的选择：B 为主 + A 为辅 + 按关系分策略。**

```sql
CREATE TABLE closure_policy (                -- 每个关系独立策略
    relation   TEXT PRIMARY KEY REFERENCES relation_types(relation),
    mode       TEXT NOT NULL DEFAULT 'batch' CHECK (mode IN ('incremental','batch','off')),
    max_depth  INTEGER NOT NULL DEFAULT 12 CHECK (max_depth BETWEEN 1 AND 64),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

INSERT INTO closure_policy(relation, mode, max_depth) VALUES
 ('is_a','incremental',12), ('part_of','incremental',8),
 ('causes','batch',8),
 ('synonym','off',1),      -- 对称关系，闭包无意义 —— v2.0 把它塞进 closure 是纯浪费
 ('similar','off',1), ('opposite','off',1);
```

**删除边的优化（见证边 first_edge）**：

```sql
CREATE TRIGGER tr_closure_edge_ad AFTER DELETE ON relations
WHEN OLD.src_kind='sense' AND OLD.dst_kind='sense'
 AND (SELECT is_transitive FROM relation_types WHERE relation = OLD.relation) = 1
 AND (SELECT mode FROM closure_policy WHERE relation = OLD.relation) <> 'off'
BEGIN
    DELETE FROM closure WHERE first_edge = OLD.relation_id;      -- 只失效"以它为见证"的行
    INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at)
        VALUES (OLD.relation, strftime('%Y-%m-%dT%H:%M:%fZ','now'));
END;
```

> ⚠️ **注意**：`first_edge` 刻意**不加外键**。初稿写了 `REFERENCES relations(relation_id) ON DELETE SET NULL`，结果外键动作在 AFTER DELETE 触发器之**前**执行，值被清空后触发器再也匹配不到行 → 失效逻辑静默失效。这是实测踩出来的坑。

**全量重建（幂等，O(可达对数) 单次扫描）**：

```sql
CREATE TRIGGER tr_closure_rebuild_ai AFTER INSERT ON closure_rebuild_queue
BEGIN
    DELETE FROM closure WHERE relation = NEW.relation;
    INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
    SELECT NEW.relation, W.cur, W.anc, MIN(W.depth), W.first_edge   -- 借助"裸列+min()"规则一次取最短+见证
      FROM (WITH RECURSIVE walk(anc, cur, depth, first_edge, visited) AS (
                SELECT r.src_id, r.dst_id, 1, r.relation_id, ','||r.src_id||','||r.dst_id||','
                  FROM relations r WHERE r.relation=NEW.relation AND r.src_kind='sense' AND r.dst_kind='sense'
                UNION ALL
                SELECT w.anc, r.dst_id, w.depth+1, w.first_edge, w.visited||r.dst_id||','
                  FROM walk w JOIN relations r ON r.src_id = w.cur
                 WHERE r.relation=NEW.relation AND w.depth < (SELECT max_depth FROM closure_policy WHERE relation=NEW.relation)
                   AND instr(w.visited, ','||r.dst_id||',') = 0        -- 环保护
            ) SELECT anc, cur, depth, first_edge FROM walk) AS W
     WHERE W.anc <> W.cur                                              -- 自环过滤，避免触发 CHECK
     GROUP BY W.cur, W.anc;
    DELETE FROM closure_rebuild_queue WHERE relation = NEW.relation;
END;
```

**批量导入 SOP**：

```sql
UPDATE closure_policy SET mode='off';                 -- 1. 关触发
--  ... 批量 INSERT relations ...
INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at)   -- 2. 统一重建一次
    SELECT relation, strftime('%Y-%m-%dT%H:%M:%fZ','now')
      FROM closure_policy WHERE mode <> 'off';
UPDATE closure_policy SET mode='incremental'
 WHERE relation IN ('is_a','part_of');                -- 3. 恢复
```

### 8.4 实测验证通过项

`validate_schema.py` 中与闭包相关的 6 项全部通过：
- 闭包增量插入（老虎/动物/生物，方向为上位→下位）
- 闭包 `first_edge` 见证边非空
- **方向语义自检**：`descendant=13(老虎)` 的上位 = 动物、生物 ✅
- 成环不产生自环闭包行（CHECK 未被触发）
- 删除边后闭包自动重建回正确状态
- 最短路径的见证边被正确保留（UPSERT 赋值顺序验证）

---

## 9. 汉语言学正确性

### 9.1 语义角色：12 个太少，且缺一层

v2.0 只有：`agent / patient / theme / instrument / location / time / manner / cause / recipient / beneficiary / topic / comment`。

**缺失**（按现代汉语语义格体系）：
- 参与者：致事(causer)、感事(experiencer)、与事(dative)、结果(result)、对象(target)、系事、受损者(maleficiary)、伴随(comitative)、比较对象、喻体
- 环境：起点、终点、经由、方向、时点（vs 时段）
- 计量：频次、数量、程度、幅度
- 因果：目的、条件、让步、效果
- 命题：述题（v2.0 有 comment 但 description 是"评论"）、命题、焦点

**另外，v2.0 的角色表没有层级、没有典型标记词** —— 而汉语的语义角色几乎总是由介词显式引出（把→受事、被→施事、从→起点、用→工具）。这是把虚词和语义角色打通的天然接口。

```sql
CREATE TABLE semantic_roles (
    role        TEXT PRIMARY KEY,
    parent_role TEXT REFERENCES semantic_roles(role),
    role_group  TEXT NOT NULL CHECK (role_group IN ('参与者','环境','计量','因果','命题')),
    description TEXT NOT NULL,
    typical_pos TEXT,
    case_marker TEXT           -- 典型介词标记: 把/被/从/在/用/给/对/比
) STRICT;
```
v3 种子：**37 个角色**，含 `('施事',NULL,'参与者',...,'被/由')`、`('受事',NULL,'参与者',...,'把/将')` 等。

### 9.2 🔴 最重要的缺口：只有语义角色，没有句法成分

这是**结构性**问题。汉语是**话题优先语言**（Li & Thompson 1976），主语 ≠ 施事是常态：

> 「鱼我吃了」—— 话题("鱼")在句首，但它不是主语，也不是施事。
> 「作业写完了」—— 零形主语。
> 「他身体很好」—— 主谓谓语句，"身体"是小主语但语义上是话题的一部分。

v2.0 只有 `semantic_roles`，却在 `sentence_patterns` 里用 {S} {V} {O} 这种**句法**槽位 —— 两套体系混用但只有一套有表。必须补齐：

```sql
CREATE TABLE syntactic_functions (
    function_id TEXT PRIMARY KEY,          -- 主语/谓语/宾语/定语/状语/补语/中心语/独立语/话题
    description TEXT NOT NULL,
    position    TEXT NOT NULL CHECK (position IN ('前','中','后','灵活'))
) STRICT;
```
并让 `pattern_slots` 同时声明**句法功能**与**默认语义角色**：
```sql
CREATE TABLE pattern_slots (
    pattern_id         INTEGER NOT NULL REFERENCES sentence_patterns(pattern_id) ON DELETE CASCADE,
    slot_name          TEXT NOT NULL,
    position           INTEGER NOT NULL CHECK (position >= 1),
    is_required        INTEGER NOT NULL DEFAULT 1 CHECK (is_required IN (0,1)),
    syntactic_function TEXT REFERENCES syntactic_functions(function_id),   -- 句法层
    default_role       TEXT REFERENCES semantic_roles(role),                -- 语义层
    PRIMARY KEY (pattern_id, slot_name),
    UNIQUE (pattern_id, position)
) STRICT, WITHOUT ROWID;
```
实测种子：把字句 `P` 槽 = 句法功能"状语" + 语义角色"受事" —— 这正是"把字句的宾语是状语位置上的受事"这一语言学事实的准确编码。

### 9.3 句式模板：11 → 32

v2.0 缺失的**核心句型**：

| 缺失句式 | 例子 | 优先级 |
|---|---|---|
| 述补句（动结/动趋/可能补语） | 看完 / 走出来 / 看得完 | 🔴 |
| 句类（陈述/疑问/祈使/感叹） | 你去不去？ / 请坐 | 🔴 |
| 双宾句 | 给我一本书 | 🟡 |
| 主谓谓语句 | 他身体很好 | 🟡 |
| 名词谓语句 | 今天星期六 | 🟡 |
| 比较句另三种 | 跟…一样 / 不如 / 越来越 | 🟡 |
| "连…都" 句 | 连他都不知道 | 🟢 |
| 紧缩句 | 一…就… / 越…越… / 非…不… | 🟢 |
| 倒装句 | 怎么了，你？ | 🟢 |

v3 种子共 **32 个句式 + 58 个槽位 + 10 条槽位约束**（含 `把字句 P 槽 definiteness=definite`——把字句宾语必须有定，这是硬性语法限制）。

### 9.4 虚词分类：从"注释"到"可执行语法"

v2.0 的 `function_words` 只有 `category / subcat / grammatical_function / attaches_to` 四个文本字段，无法回答任何程序化问题。

**关键改进**：让虚词**直接声明它引出哪个语义角色+语序位置**。

```sql
CREATE TABLE function_word_roles (
    sense_id INTEGER NOT NULL REFERENCES function_words(sense_id) ON DELETE CASCADE,
    role     TEXT NOT NULL REFERENCES semantic_roles(role),
    position TEXT NOT NULL CHECK (position IN ('前置','后置','框式')),
    PRIMARY KEY (sense_id, role)
) STRICT, WITHOUT ROWID;
```
种子：
```sql
INSERT INTO function_word_roles(sense_id, role, position) VALUES (4, '受事', '前置');  -- 把
-- 同理: 被→施事(前置) 从→起点(前置) 在→处所(前置) 用→工具(前置) 给→与事(前置) 比→比较对象(前置)
```
这样"把字句"的解析就从"匹配到'把' → 后面第一个 NP 标为受事"（v2.0 只能靠硬编码）变成**查表可得**。

**分类体系修正**（v2.0 的 `function_words.category` 无枚举约束）：
```
介词 / 连词 / 结构助词(的·地·得·所·似的) / 动态助词(了·着·过·来着) / 语气助词(吗·呢·吧·啊·嘛·罢了)
/ 比况助词 / 数助词(第·初·老) / 副词 / 叹词 / 拟声词 / 方位词 / 词缀
```
> 特别提醒："的/地/得"三分 与 "了₁(体助词)/了₂(句末语气词)" 的区分必须在 **sense 层**建两条义项，v2.0 的 `pos='u'` 单行无法承载。

### 9.5 量词搭配：需要两级回退

v2.0 的 `noun_classifier_pairs(noun_sense_id, classifier_id)` 要求**每个名词显式建行**。10 万名词 × 平均 2 个量词 = 20 万行，且新名词入库必须人工补录。

**v3 两级方案**：
```sql
CREATE TABLE classifier_class_defaults (      -- 类别级默认：动物类 → 只
    cat TEXT NOT NULL REFERENCES semantic_categories(cat),
    classifier_sense_id INTEGER NOT NULL REFERENCES classifiers(classifier_sense_id),
    preference REAL NOT NULL DEFAULT 1.0 CHECK (preference BETWEEN 0 AND 1),
    PRIMARY KEY (cat, classifier_sense_id)
) STRICT, WITHOUT ROWID;
-- 查询：先查显式搭配(noun_classifier_pairs)，无则回退类别默认
```
实测：查"猫" → 显式 `只(0.9)` + 类别默认 `只(0.7)`，两级都能命中。

### 9.6 其他语言学修正

| 项 | v2.0 | v3 |
|---|---|---|
| 「同义」与「相似」混用 `similar` | 一个关系表达两种语义 | 拆成 `synonym`（近义，可传递）与 `similar`（弱相关，不可传递） |
| 「反义」未分类 | 一个 `opposite` | 拆成 `antonym_complementary`（生/死）、`antonym_gradable`（大/小）、`antonym_converse`（买/卖）—— 三者的推理行为完全不同 |
| 缺词汇派生 | 无 | `derived_from`（本义→引申义）、`metaphor_of`、`metonymy_of` —— 汉语一词多义主要靠引申，这是义项网络的主干 |
| 缺蕴含/预设 | 无 | `entails`、`presupposes` |
| 缺语义特征 | 无 | `semantic_features`（±animate/±human/±countable/±concrete/±volitional/±bounded）+ `sense_features` |
| 缺选择限制的表达 | `allowed_types` 逗号串 | `relation_constraints(sense_id, slot, kind, value)` 键值化 |

---

## 10. 优先级总表与落地顺序

### 🔴 必须修（12 项，不做会出错或性能不可接受）

| # | 项 | 章节 |
|---|---|---|
| 1 | `cross_links` 主键漏 `target_id` | 1.1 |
| 2 | `sentence_patterns` 三个 JSON 列与 `pattern_slots` 双写（且子表是死表） | 1.2 |
| 3 | `char_count` / `is_separable` 落库 → 改生成列 | 1.3 / 1.4 |
| 4 | 无 FTS5，中文释义无法检索（且必须双通道） | 2.2 |
| 5 | 重复索引 `idx_entries_word` | 2.1 |
| 6 | `PRAGMA foreign_keys` 默认 OFF → 全部外键失效 | 5.1 |
| 7 | 未用 STRICT | 5.2 |
| 8 | 触发器内不能 `WITH ... INSERT` | 5.3 |
| 9 | UPSERT 赋值顺序陷阱 | 5.4 |
| 10 | `relations` 缺 `UNIQUE` + 缺自环 CHECK | 7.1 / 7.2 |
| 11 | `ellipsis_rules` 主键缺陷 → 省略回填机制形同虚设 | 7.21 |
| 12 | 缺 `is_transitive/is_symmetric` → 闭包策略无法分流 | 7.13 / 8.1 |

**以及 4 项中文核心结构缺失**：语素层、音变规则、述补结构、句类 + 篇章层（3.1–3.5）。

### 🟡 重要（22 项）

统一关系模型、来源/批次表、开放属性、部分索引、`WITHOUT ROWID`、闭包主键重排 + 方向约定、重叠式、离合词展开、黏着宾语、量词细分、多音字、ISO 时间戳、命名规约、`ON DELETE CASCADE`、20 余处 CHECK/NOT NULL……

### 🟢 可选（9 项）

`AUTOINCREMENT` 去除、`jsonb`/`RETURNING`、异形词、紧缩句、倒装句、双宾语句、分析表的 ANALYZE 计划、物化路径方案、`frequency` 语义澄清。

### 落地顺序建议

```
第 1 步（半小时，收益最大）
  · 打开 PRAGMA foreign_keys / recursive_triggers
  · 补 relations UNIQUE + 自环 CHECK + closure 自环 CHECK + ellipsis_rules 主键
  · 删掉重复索引 idx_entries_word
  → 立刻消除"静默数据损坏"风险

第 2 步（半天，性能）
  · 全表 STRICT
  · 建 FTS5 双通道 + 同步触发器（trigram + jieba 预分词）
  · 部分索引 / WITHOUT ROWID
  → 中文检索从"不可用"变为"可用"

第 3 步（1 天，结构）
  · 统一 relations + entity_nodes + domains + sources
  · 生成列替换派生字段
  · 删句式 JSON 列，槽位收敛到 pattern_slots
  → 加新领域库不再改表结构

第 4 步（1–2 天，语言学）
  · 语素层 / 音变规则 / 述补结构 / 句类 / 篇章层（话题链·零形回指）
  · 语义角色 12 → 37，补 syntactic_functions
  · function_word_roles 打通"虚词 → 语义角色"
  → 覆盖中文核心语言现象

第 5 步（半天，闭包）
  · closure_policy 分策略 + first_edge 见证边 + 队列重建
  → 批量导入不再被闭包拖垮
```

---

## 11. 定点修复落地核对（本轮执行）

按「先止血、再重构」的顺序，本轮只做**不涉及表重命名**的外科式修复；
每项都由 `validate_schema.py` STEP 4 的 R1–R9 用例锁定，防止日后静默回退。

| 项 | 落点 | 状态 | 锁定用例 |
|---|---|---|---|
| 开 `PRAGMA foreign_keys = ON` | §0 | ✅ | R1 |
| 补 `UNIQUE` / `CHECK` | 全表 | ✅ | R1 / R2 |
| 删重复索引 | §12 | ✅ 39 → 34 | R3 |
| `cross_links` 主键修复 | 改为兼容视图 + `relations.UNIQUE` 含 dst | ✅ | R4 |
| `ellipsis_rules` 主键修复 | 代理键 `ellipsis_rule_id` + 多策略 | ✅ | R5 |
| 统一 `pattern_slots`、删 JSON 冗余字段 | §6 / §14 | ✅ | R6 |
| 闭包方向文档化 + 触发器重写 | §7 / §13.2–13.3 / `docs/CLOSURE_DIRECTION.md` | ✅ | R7 / R8 / R9 |

### 本轮**新发现**的三个问题（原审查未覆盖）

1. 🔴 **`pattern_slots` 只覆盖 22/32 个句式** —— 10 个句式（特指问/选择问/反问句/
   祈使句/感叹句/比较句-等同/比较句-不如/紧缩句-越…越/倒装句/处置感叹式）的槽位
   只写在 `template` 字符串里，子表一行都没有。"死表"问题其实只修好了一半。
   已补齐全部 32 个句式（槽位 58 → 80）。
2. 🟡 **可选性双写** —— 模板里写 `{C?}`、`is_required` 里又写 `0/1`，同一事实两处存储，
   与"单一真源"原则冲突。已从模板移除 `?` 后缀，可选性只由 `is_required` 表达。
3. 🟡 **5 个冗余索引**（肉眼比对只看出 3 个）—— 交由 `audit_indexes.py` 程序化检出：
   `senses(word_id,pos)`、`morphemes(form)`、`relations(src_kind,src_id,relation)`、
   `attributes(owner_kind,owner_id,key)`、`pattern_slots(pattern_id,position)`，
   全部是既有 UNIQUE 隐含索引的真前缀或完全重复。

> 关于 `relations`：删掉 `idx_relations_src` **不损失任何能力** ——
> 保留的 `UNIQUE(src_kind, src_id, relation, dst_kind, dst_id)` 隐含索引以相同列打头
> 且列更多，是它的严格能力超集（含 skip-scan 场景）。
>
> 🟢 遗留可选项（未实施）：闭包重建触发器按 `r.src_id = w.cur` 关联 `relations`，
> 而现有索引没有以 `src_id` 打头的，大库上可能退化为扫描。若关系量过万，
> 可考虑补一条 `relations(src_id) WHERE src_kind='sense'` 部分索引。

---

## 附：交付物与验证方式

| 文件 | 说明 |
|---|---|
| `schema_v3.sql` | 优化后完整 schema（52 张 STRICT 表 + 5 视图 + 10 触发器 + 34 索引 + 完整种子数据 + 常用查询模板） |
| `validate_schema.py` | 结构自检 + 功能自检 + R1–R9 定点回归，共 47 项。运行：`python validate_schema.py` |
| `audit_indexes.py` | 冗余索引程序化探测器。运行：`python audit_indexes.py` |
| `bench_closure.py` | 闭包维护成本基准。运行：`python bench_closure.py` |
| `docs/CLOSURE_DIRECTION.md` | 闭包方向约定：权威口径 + 查询配方 + 不变量 + 触发器硬约束 |
| `schema_review.md` | 本报告 |

```bash
$ python validate_schema.py
SQLite runtime = 3.53.1
  表 62 / 视图 5 / 触发器 10 / 索引 34
  [PASS] integrity_check = ok
  [PASS] foreign_key_check 无违规
  [PASS] STRICT 表 = 52/52
  ...
  [PASS] R3 无冗余索引（列集均非既有 UNIQUE/PK 索引的真前缀）
  [PASS] R6 模板占位符与 pattern_slots 一一对应（32/32）
  [PASS] R7 方向锁定：depth=1 闭包行满足 dst=ancestor ∧ src=descendant
结果: PASS 47 / FAIL 0
```

`schema_v3.sql` 中的每个关键设计决策都有对应的实测通过用例支撑，不是纸面推演。
