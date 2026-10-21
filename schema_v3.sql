/*****************************************************************************
 * Logos Knowledge Database Schema v3.0
 * 中文原生符号语义数据库 —— 基于 v2.0 审查结果重写
 *
 * 目标 SQLite 版本: >= 3.37.0 (STRICT 表)
 *                   建议 >= 3.45.0 (trigram 分词器 / jsonb / 优化器增强)
 * 本机实测环境:      SQLite 3.53.1
 * 依赖:             JSON1 (内建)、FTS5 (内建)
 *
 * 相对 v2.0 的九项结构性变更（详见 schema_review.md）:
 *   [范式]  全部表改为 STRICT；派生字段一律用生成列；取消冗余 JSON 列
 *   [性能]  FTS5 trigram + 预分词双通道；WITHOUT ROWID 压缩大表；部分索引
 *   [中文]  新增 语素层 / 音变规则(变调·轻声·儿化) / 重叠式 / 离合式 /
 *           补语类型 / 句类 / 语义特征 / 篇章层(话题链·零形回指)
 *   [扩展]  cross_links 取消 → 统一关系表 + 节点注册表，加领域库不改表
 *   [SQLite] STRICT / 生成列 / UPSERT / 部分索引 / 递归 CTE / json_valid CHECK
 *   [命名]  统一 <实体>_id、src/dst、snake_case、表名复数
 *   [约束]  补齐 CHECK / NOT NULL / UNIQUE / FK / ON DELETE
 *   [闭包]  仅对 is_transitive 关系物化；first_edge 见证边 + 队列式重建
 *   [语言]  语义角色 12 → 34；句式 11 → 30+；虚词直接映射到语义角色
 *****************************************************************************/

/* =========================================================================
 * 0. 连接级配置（必须在每个连接上执行，不随 schema 持久化）
 * =========================================================================
 * PRAGMA foreign_keys   = ON;          -- SQLite 默认关闭！不设则所有 FK 形同虚设
 * PRAGMA journal_mode   = WAL;         -- 读写并发（返回值必须检查）
 * PRAGMA synchronous    = NORMAL;      -- WAL 下安全且快
 * PRAGMA busy_timeout   = 5000;
 * PRAGMA cache_size     = -65536;      -- 64MB 页缓存
 * PRAGMA mmap_size      = 268435456;   -- 256MB 内存映射
 * PRAGMA temp_store     = MEMORY;
 * PRAGMA recursive_triggers = ON;      -- 见 §13 闭包触发器：需要触发器级联
 * -- 关闭连接前:  PRAGMA optimize;
 * -- 批量导入时:  PRAGMA defer_foreign_keys = ON;  + 见 §13.3 批量模式
 * =========================================================================*/

PRAGMA foreign_keys = ON;


/* =========================================================================
 * 1. 元数据层
 * ========================================================================= */

CREATE TABLE IF NOT EXISTS schema_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

CREATE TABLE IF NOT EXISTS sources (
    source_id    INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL CHECK (kind IN
                   ('词典','语料库','本体','百科','论文','人工标注','规则','KEGG','Reactome','GO','PubMed')),
    uri          TEXT,
    license      TEXT,
    version      TEXT,
    retrieved_at TEXT,
    note         TEXT
) STRICT;

-- 领域注册表：新增领域库只 INSERT 一行，不改任何表结构
CREATE TABLE IF NOT EXISTS domains (
    domain      TEXT PRIMARY KEY,
    description TEXT,
    db_path     TEXT,                    -- 跨库(ATTACH)时的物理路径，NULL 表示本库
    version     TEXT,
    is_core     INTEGER NOT NULL DEFAULT 0 CHECK (is_core IN (0,1))
) STRICT;

-- 语义类别（本体骨架）：供字/词/量词类别默认值共用
CREATE TABLE IF NOT EXISTS semantic_categories (
    cat         TEXT PRIMARY KEY,
    parent_cat  TEXT REFERENCES semantic_categories(cat),
    description TEXT
) STRICT;

-- 语义特征（义素）：支撑选择限制，比纯类别层级更细粒度
CREATE TABLE IF NOT EXISTS semantic_features (
    feature       TEXT PRIMARY KEY,       -- +animate / +human / +countable ...
    feature_group TEXT NOT NULL CHECK (feature_group IN ('生命度','具体性','可数性','自主性','有界性','极性')),
    value_type    TEXT NOT NULL CHECK (value_type IN ('boolean','enum','number')),
    description   TEXT NOT NULL
) STRICT;

-- 词性标签表（北大计算语言学所标记集 + 现代汉语词类体系）
CREATE TABLE IF NOT EXISTS pos_tags (
    pos         TEXT PRIMARY KEY,          -- n / v / a / p / u ...
    name        TEXT NOT NULL UNIQUE,      -- 名词 / 动词 / 介词 / 助词
    pos_group   TEXT NOT NULL CHECK (pos_group IN ('实词','虚词','特殊')),
    description TEXT
) STRICT;

-- 句法成分（与语义角色解耦：汉语是"话题优先"语言，二者不等同）
CREATE TABLE IF NOT EXISTS syntactic_functions (
    function_id TEXT PRIMARY KEY,          -- 主语/谓语/宾语/定语/状语/补语/中心语/独立语
    description TEXT NOT NULL,
    position    TEXT NOT NULL CHECK (position IN ('前','中','后','灵活'))
) STRICT;

-- 句类（语气）
CREATE TABLE IF NOT EXISTS sentence_moods (
    mood       TEXT PRIMARY KEY,           -- 陈述/疑问/祈使/感叹
    description TEXT NOT NULL,
    typical_markers TEXT
) STRICT;

-- 补语类型：汉语述补结构是核心句型，v2.0 完全缺失
CREATE TABLE IF NOT EXISTS complement_types (
    comp_type   TEXT PRIMARY KEY,          -- 结果补语/趋向补语/可能补语/程度补语/数量补语/状态补语/时地补语
    description TEXT NOT NULL,
    marker      TEXT,                      -- 得 / 不 / NULL
    example     TEXT
) STRICT;


/* =========================================================================
 * 2. 字层
 * ========================================================================= */

CREATE TABLE IF NOT EXISTS radical_semantics (
    radical  TEXT PRIMARY KEY,
    meaning  TEXT NOT NULL,
    examples TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS characters (
    char          TEXT PRIMARY KEY COLLATE BINARY,
    radical       TEXT REFERENCES radical_semantics(radical),
    strokes       INTEGER CHECK (strokes IS NULL OR strokes BETWEEN 1 AND 64),
    structure     TEXT CHECK (structure IS NULL OR structure IN
                    ('独体','左右','上下','半包围','全包围','品字','镶嵌','特殊')),
    semantic_cat  TEXT REFERENCES semantic_categories(cat),
    is_common     INTEGER NOT NULL DEFAULT 1 CHECK (is_common IN (0,1)),
    trad          TEXT,                                    -- 繁体对应
    simp_of       TEXT REFERENCES characters(char),        -- 该字是某繁体字的简化字
    hint          TEXT                                     -- 字理/造字法说明
) STRICT;

-- 多音字：一字多音，文白异读、姓氏地名特殊读音的正式承载表
CREATE TABLE IF NOT EXISTS character_readings (
    char         TEXT NOT NULL REFERENCES characters(char) ON DELETE CASCADE,
    pinyin       TEXT NOT NULL,            -- 规范拼音（数字标调）: 'xing2'
    tone         INTEGER NOT NULL CHECK (tone BETWEEN 0 AND 5),   -- 0 = 轻声
    reading_type TEXT NOT NULL CHECK (reading_type IN
                   ('常读','文读','白读','姓氏','地名','古音','异读')),
    meaning_hint TEXT,
    is_default   INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0,1)),
    PRIMARY KEY (char, pinyin, reading_type)
) STRICT, WITHOUT ROWID;

-- 音变规则：变调 / 轻声 / 儿化。v2.0 只在 pinyin_json 里塞字符串，无法推理
CREATE TABLE IF NOT EXISTS phonological_rules (
    rule_id      INTEGER PRIMARY KEY,
    rule_type    TEXT NOT NULL CHECK (rule_type IN
                   ('变调','连读变调','轻声','儿化','儿化变韵','声调中和')),
    source_form  TEXT NOT NULL,            -- 触发形式: '一' / '不' / '儿' / '3+3'
    condition    TEXT NOT NULL,            -- 触发条件: '后字为去声(4)' / '前字为阴平'
    surface_form TEXT NOT NULL,            -- 表层读音: 'yi2' / 'hua1r'
    example      TEXT,
    UNIQUE (rule_type, source_form, condition)
) STRICT;


/* =========================================================================
 * 3. 语素层（新增）—— 字 ≠ 语素
 *    '葡萄' 两字一语素；'儿' 是词尾；'老-' 是前缀。v2.0 把三者压进字层
 * ========================================================================= */

CREATE TABLE IF NOT EXISTS morphemes (
    morpheme_id   INTEGER PRIMARY KEY,
    form          TEXT NOT NULL,
    morpheme_type TEXT NOT NULL CHECK (morpheme_type IN
                    ('自由语素','黏着语素','词根','前缀','后缀','中缀','词尾','联绵语素')),
    is_bound      INTEGER NOT NULL DEFAULT 0 CHECK (is_bound IN (0,1)),
    pinyin        TEXT,
    meaning       TEXT,
    UNIQUE (form, morpheme_type)
) STRICT;


/* =========================================================================
 * 4. 词层
 * ========================================================================= */

CREATE TABLE IF NOT EXISTS words (
    word_id      INTEGER PRIMARY KEY,
    word         TEXT NOT NULL UNIQUE COLLATE BINARY,
    -- 生成列：不再手工维护 char_count
    char_count   INTEGER GENERATED ALWAYS AS (length(word)) VIRTUAL,
    -- 离合点：第 N 个字之后可插入成分；NULL = 不可离合
    separator_pos INTEGER CHECK (separator_pos IS NULL OR separator_pos >= 1),
    -- 生成列：is_separable 由 separator_pos 派生，杜绝二者不一致
    is_separable INTEGER GENERATED ALWAYS AS (separator_pos IS NOT NULL) VIRTUAL,
    word_type    TEXT CHECK (word_type IS NULL OR word_type IN
                    ('单纯词','合成词','复合词','派生词','重叠词','联绵词','叠音词','音译词','缩略语','成语','惯用语')),
    is_bound_root INTEGER NOT NULL DEFAULT 0 CHECK (is_bound_root IN (0,1)),  -- 黏着词根(澡/忙/架)
    register     TEXT NOT NULL DEFAULT '通用' CHECK (register IN ('通用','书面','口语','方言','古语','网络')),
    variant_of   INTEGER REFERENCES words(word_id),      -- 异形词: 其它→其他
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    CHECK (separator_pos IS NULL OR separator_pos < length(word))
) STRICT;

-- 词的语素切分
CREATE TABLE IF NOT EXISTS word_morphemes (
    word_id     INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL CHECK (seq >= 1),
    morpheme_id INTEGER NOT NULL REFERENCES morphemes(morpheme_id),
    surface     TEXT,                       -- 该语素在词中的书写形式（儿化/变体）
    PRIMARY KEY (word_id, seq)
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS senses (
    sense_id          INTEGER PRIMARY KEY,
    word_id           INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    pos               TEXT NOT NULL REFERENCES pos_tags(pos),
    sense_no          INTEGER NOT NULL DEFAULT 1 CHECK (sense_no >= 1),
    definition        TEXT NOT NULL CHECK (length(trim(definition)) > 0),
    definition_seg    TEXT,                 -- jieba 分词后的释义（空格分隔），供 unicode61 FTS
    pinyin            TEXT,                 -- 义项级读音（多音词/轻声分化）
    pinyin_plain      TEXT,                 -- 去调拼音: 'xizao'，供容错检索
    pinyin_initial    TEXT,                 -- 声母串: 'xz'，供简拼检索
    tone_code         TEXT,                 -- 声调串: '3;0'
    example           TEXT,
    frequency         INTEGER NOT NULL DEFAULT 0 CHECK (frequency >= 0),
    literalness       TEXT CHECK (literalness IS NULL OR literalness IN
                        ('本义','引申义','比喻义','借代义','通假义')),
    derived_from      INTEGER REFERENCES senses(sense_id),   -- 引申链：由哪一义项派生
    subcategory       TEXT,                 -- 动词次类: 使令/心理/给予/存现/趋向/能愿/判断
    is_primary        INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0,1)),
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    defined_by        TEXT NOT NULL DEFAULT 'senses',        -- 符号语义: 该义项定义在哪个表
    UNIQUE (word_id, pos, sense_no),
    UNIQUE (word_id, pos, definition)                        -- 阻断重复义项
) STRICT;

-- 义项的语义特征（义素分析）—— 选择限制的细粒度依据
CREATE TABLE IF NOT EXISTS sense_features (
    sense_id INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    feature  TEXT NOT NULL REFERENCES semantic_features(feature),
    value    TEXT NOT NULL,
    PRIMARY KEY (sense_id, feature)
) STRICT, WITHOUT ROWID;

-- 义项 ↔ 语义类别
CREATE TABLE IF NOT EXISTS sense_categories (
    sense_id INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    cat      TEXT NOT NULL REFERENCES semantic_categories(cat),
    is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0,1)),
    PRIMARY KEY (sense_id, cat)
) STRICT, WITHOUT ROWID;

/* ---- 4.1 虚词：独立建模 + 直接映射语义角色 ---- */

CREATE TABLE IF NOT EXISTS function_words (
    sense_id     INTEGER PRIMARY KEY REFERENCES senses(sense_id) ON DELETE CASCADE,
    category     TEXT NOT NULL CHECK (category IN
                   ('介词','连词','结构助词','动态助词','语气助词','比况助词','数助词',
                    '副词','叹词','拟声词','方位词','词缀')),
    subcategory  TEXT NOT NULL,          -- 介词: 处置/被动/处所/时间/工具/关涉/比较/排除
    grammatical_function TEXT,
    attaches_to  TEXT,                   -- 前置成分 / 后置成分 / 句末 / 谓语前
    -- 体标记(了/着/过)专用：约束其附着的谓词次类
    selects_verb_subcat TEXT,
    is_sentence_final INTEGER NOT NULL DEFAULT 0 CHECK (is_sentence_final IN (0,1))
) STRICT;

-- 关键设计：介词/助词直接声明它引出哪个语义角色
--   把→受事(前置)  被→施事(前置)  从→起点(前置)  在→处所(前置)  用→工具(前置)
--   给→与事(前置)  对→对象(前置)  比→比较对象(前置)  了→体(后置)
CREATE TABLE IF NOT EXISTS function_word_roles (
    sense_id  INTEGER NOT NULL REFERENCES function_words(sense_id) ON DELETE CASCADE,
    role      TEXT NOT NULL REFERENCES semantic_roles(role),
    position  TEXT NOT NULL CHECK (position IN ('前置','后置','框式')),
    PRIMARY KEY (sense_id, role)
) STRICT, WITHOUT ROWID;

/* ---- 4.2 量词 ---- */

-- 量词 = 义项的特化（'把' 既是动词/介词/量词），不再另起 word 唯一约束
CREATE TABLE IF NOT EXISTS classifiers (
    classifier_sense_id INTEGER PRIMARY KEY REFERENCES senses(sense_id) ON DELETE CASCADE,
    kind       TEXT NOT NULL CHECK (kind IN
                 ('名量词','动量词','复合量词','借用量词','度量词','不定量词','时间量词')),
    subtype    TEXT CHECK (subtype IS NULL OR subtype IN
                 ('个体','集合','部分','容器','临时','通用')),
    is_borrowed INTEGER NOT NULL DEFAULT 0 CHECK (is_borrowed IN (0,1)),  -- 一桌子菜 / 一屋子人
    has_reduplication INTEGER NOT NULL DEFAULT 0 CHECK (has_reduplication IN (0,1)), -- 个个 / 一个个
    source_noun_sense_id INTEGER REFERENCES senses(sense_id)              -- 借用量词的源名词
) STRICT;

CREATE TABLE IF NOT EXISTS noun_classifier_pairs (
    noun_sense_id       INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    classifier_sense_id INTEGER NOT NULL REFERENCES classifiers(classifier_sense_id) ON DELETE CASCADE,
    preference          REAL NOT NULL DEFAULT 1.0 CHECK (preference BETWEEN 0 AND 1),
    is_default          INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0,1)),
    source_id           INTEGER REFERENCES sources(source_id),
    PRIMARY KEY (noun_sense_id, classifier_sense_id)
) STRICT, WITHOUT ROWID;

-- 类别级默认量词：'动物类 → 只'，避免为 10 万个名词逐一建行（可扩展性关键）
CREATE TABLE IF NOT EXISTS classifier_class_defaults (
    cat                 TEXT NOT NULL REFERENCES semantic_categories(cat),
    classifier_sense_id INTEGER NOT NULL REFERENCES classifiers(classifier_sense_id),
    preference          REAL NOT NULL DEFAULT 1.0 CHECK (preference BETWEEN 0 AND 1),
    PRIMARY KEY (cat, classifier_sense_id)
) STRICT, WITHOUT ROWID;

/* ---- 4.3 重叠式 ---- */

CREATE TABLE IF NOT EXISTS reduplication_patterns (
    pattern_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,      -- AA / AABB / ABAB / A一A / ABB / AA(量词) / 一AA
    pos         TEXT REFERENCES pos_tags(pos),
    form_schema TEXT NOT NULL,             -- 'V' -> 'V一V' / 'A' -> 'AA'
    semantics   TEXT NOT NULL,             -- 短时/尝试/遍指/程度加深/亲昵/随意/反复
    example     TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS word_reduplications (
    word_id         INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    pattern_id      INTEGER NOT NULL REFERENCES reduplication_patterns(pattern_id),
    surface         TEXT NOT NULL,                 -- 高高兴兴 / 看一看
    derived_word_id INTEGER REFERENCES words(word_id),  -- 若重叠后成为独立词条
    PRIMARY KEY (word_id, pattern_id)
) STRICT, WITHOUT ROWID;

/* ---- 4.4 离合词 ---- */

CREATE TABLE IF NOT EXISTS separable_patterns (
    sep_pattern_id INTEGER PRIMARY KEY,
    word_id        INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    kind           TEXT NOT NULL CHECK (kind IN
                     ('插入体标记','插入数量','插入定语','插入补语','插入疑问成分','重叠','倒装')),
    infixes_json   TEXT CHECK (infixes_json IS NULL OR json_valid(infixes_json)), -- ["了","过","着","个"]
    surface_example TEXT NOT NULL,         -- 洗了个热水澡 / 洗完澡 / 澡洗了
    note           TEXT,
    UNIQUE (word_id, kind, surface_example)
) STRICT;

-- 黏着宾语：'澡/忙/架' 不能独立作名词条目，只能依附特定动词
CREATE TABLE IF NOT EXISTS bound_objects (
    object_word_id INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    host_verb_id   INTEGER NOT NULL REFERENCES words(word_id),
    PRIMARY KEY (object_word_id, host_verb_id)
) STRICT, WITHOUT ROWID;


/* =========================================================================
 * 5. 关系层
 * ========================================================================= */

/* 节点注册表：把"义项"之外的所有图节点纳入同一张图
 *   → 加新领域库、新词条、新句式都不需要新增关系表（可扩展性） */
CREATE TABLE IF NOT EXISTS entity_nodes (
    node_id      INTEGER PRIMARY KEY,
    node_type    TEXT NOT NULL CHECK (node_type IN
                   ('character','morpheme','word','pattern','domain_entity','context','rule','utterance')),
    domain       TEXT NOT NULL DEFAULT 'core' REFERENCES domains(domain),
    label        TEXT NOT NULL,
    entity_class TEXT,                     -- 领域内的类: 'Protein' / 'Disease'
    attrs_json   TEXT CHECK (attrs_json IS NULL OR json_valid(attrs_json)),
    UNIQUE (node_type, domain, label)
) STRICT;

CREATE TABLE IF NOT EXISTS relation_types (
    relation       TEXT PRIMARY KEY,
    relation_class TEXT NOT NULL CHECK (relation_class IN
                     ('层级','属性','事件','词汇语义','论元','篇章','领域')),
    direction      TEXT NOT NULL CHECK (direction IN ('directed','symmetric')),
    inverse_of     TEXT REFERENCES relation_types(relation),
    -- 闭包计算的决定性标志：v2.0 缺失这三个 flag，导致闭包语义不清
    is_transitive INTEGER NOT NULL DEFAULT 0 CHECK (is_transitive IN (0,1)),
    is_symmetric  INTEGER NOT NULL DEFAULT 0 CHECK (is_symmetric IN (0,1)),
    is_reflexive  INTEGER NOT NULL DEFAULT 0 CHECK (is_reflexive IN (0,1)),
    is_functional INTEGER NOT NULL DEFAULT 0 CHECK (is_functional IN (0,1)), -- 一对多上限
    max_depth     INTEGER NOT NULL DEFAULT 12 CHECK (max_depth BETWEEN 1 AND 64),
    description   TEXT NOT NULL,
    color_code    TEXT CHECK (color_code IS NULL OR
                     color_code GLOB '#[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]')
) STRICT;

/* 统一关系表（取代 relations + cross_links）
 * 范式：只存"正向"边，逆向关系由 v_relations_all 视图在线推导 → 无冗余、无更新异常 */
CREATE TABLE IF NOT EXISTS relations (
    relation_id INTEGER PRIMARY KEY,
    src_kind    TEXT NOT NULL DEFAULT 'sense' CHECK (src_kind IN ('sense','node')),
    src_id      INTEGER NOT NULL,
    dst_kind    TEXT NOT NULL DEFAULT 'sense' CHECK (dst_kind IN ('sense','node')),
    dst_id      INTEGER NOT NULL,
    relation    TEXT NOT NULL REFERENCES relation_types(relation),
    weight      REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0),
    confidence  REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    is_negated  INTEGER NOT NULL DEFAULT 0 CHECK (is_negated IN (0,1)),
    source_id   INTEGER REFERENCES sources(source_id),
    context_id  INTEGER REFERENCES contexts(context_id),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (src_kind, src_id, relation, dst_kind, dst_id),
    CHECK (NOT (src_kind = 'sense' AND dst_kind = 'sense' AND src_id = dst_id))
) STRICT;

-- 论元槽位的选择限制（键值化，可被索引与查询，不再埋在一段逗号串里）
CREATE TABLE IF NOT EXISTS relation_constraints (
    sense_id   INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    slot       TEXT NOT NULL CHECK (slot IN
                 ('agent','patient','dative','instrument','location','time','result','target','theme')),
    kind       TEXT NOT NULL CHECK (kind IN ('category','feature','sense','pos','classifier')),
    value      TEXT NOT NULL,
    is_exclusive INTEGER NOT NULL DEFAULT 0 CHECK (is_exclusive IN (0,1)),
    PRIMARY KEY (sense_id, slot, kind, value)
) STRICT, WITHOUT ROWID;


/* =========================================================================
 * 6. 句法层
 * ========================================================================= */

-- 语义角色：12 → 34，按"参与者/环境/性状/命题"分组，可层级继承
CREATE TABLE IF NOT EXISTS semantic_roles (
    role        TEXT PRIMARY KEY,
    parent_role TEXT REFERENCES semantic_roles(role),
    role_group  TEXT NOT NULL CHECK (role_group IN ('参与者','环境','计量','因果','命题')),
    description TEXT NOT NULL,
    typical_pos TEXT,
    case_marker TEXT                       -- 典型介词标记: 把/被/从/在/用/给/对/比
) STRICT;

CREATE TABLE IF NOT EXISTS sentence_patterns (
    pattern_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    category    TEXT NOT NULL CHECK (category IN
                  ('基本句型','特殊句式','句类','复句','紧缩句','话题结构','短语结构')),
    mood        TEXT REFERENCES sentence_moods(mood),
    template    TEXT NOT NULL CHECK (length(trim(template)) > 0),
    semantics   TEXT,                      -- 句式整体义: 处置/被动/存现/给予/致使/比较/遍指
    is_productive INTEGER NOT NULL DEFAULT 1 CHECK (is_productive IN (0,1)),
    -- 句式级约束（如"把字句的 P 必须有定"），单一 JSON 列，附 json_valid
    constraints_json TEXT CHECK (constraints_json IS NULL OR json_valid(constraints_json)),
    example     TEXT,
    note        TEXT
) STRICT;

-- 唯一的槽位真源：取消 v2.0 的 required_slots / optional_slots 冗余 JSON 列
CREATE TABLE IF NOT EXISTS pattern_slots (
    pattern_id         INTEGER NOT NULL REFERENCES sentence_patterns(pattern_id) ON DELETE CASCADE,
    slot_name          TEXT NOT NULL,
    position           INTEGER NOT NULL CHECK (position >= 1),
    is_required        INTEGER NOT NULL DEFAULT 1 CHECK (is_required IN (0,1)),
    syntactic_function TEXT REFERENCES syntactic_functions(function_id),
    default_role       TEXT REFERENCES semantic_roles(role),
    note               TEXT,
    PRIMARY KEY (pattern_id, slot_name),
    UNIQUE (pattern_id, position)
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS slot_constraints (
    pattern_id  INTEGER NOT NULL,
    slot_name   TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN
                  ('pos','role','feature','definiteness','animacy','countability',
                   'classifier','aspect','subcategory','lexeme','marker')),
    value       TEXT NOT NULL,
    is_exclusive INTEGER NOT NULL DEFAULT 0 CHECK (is_exclusive IN (0,1)),
    PRIMARY KEY (pattern_id, slot_name, kind, value),
    FOREIGN KEY (pattern_id, slot_name)
        REFERENCES pattern_slots(pattern_id, slot_name) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

-- 述补结构的补语实例（动结/动趋/可能补语）
CREATE TABLE IF NOT EXISTS complements (
    complement_id  INTEGER PRIMARY KEY,
    comp_type      TEXT NOT NULL REFERENCES complement_types(comp_type),
    head_sense_id  INTEGER NOT NULL REFERENCES senses(sense_id),   -- 述语
    comp_sense_id  INTEGER NOT NULL REFERENCES senses(sense_id),   -- 补语
    result_state   TEXT,                    -- 语义结果: '完成'/'达成'/'位移'/'可能'
    surface_order  TEXT NOT NULL DEFAULT 'V+C' CHECK (surface_order IN ('V+C','V+得+C','V+不+C')),
    example        TEXT,
    UNIQUE (comp_type, head_sense_id, comp_sense_id, surface_order)
) STRICT;


/* =========================================================================
 * 7. 推理层
 * ========================================================================= */

/* 闭包策略表：只有 is_transitive=1 的关系才可物化；默认 batch 模式 */
CREATE TABLE IF NOT EXISTS closure_policy (
    relation   TEXT PRIMARY KEY REFERENCES relation_types(relation),
    mode       TEXT NOT NULL DEFAULT 'batch' CHECK (mode IN ('incremental','batch','off')),
    max_depth  INTEGER NOT NULL DEFAULT 12 CHECK (max_depth BETWEEN 1 AND 64),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

/* 闭包表（物化）
 *
 * 【方向约定 —— 必须先读，否则查询会取反】
 *   closure.ancestor   = 上位节点（更一般）  e.g. 动物
 *   closure.descendant = 下位节点（更具体）  e.g. 老虎
 *   一行 (ancestor=X, descendant=Y) 读作「X 是 Y 的上位，Y is_a X」
 *   即方向就是 is_a 边的**反向**：边是 老虎 -> 动物，行是 (动物, 老虎)
 *   典型查询：
 *     · 取 Y 的全部上位：  WHERE descendant = :y  → SELECT ancestor
 *     · 取 X 的全部下位：  WHERE ancestor   = :x  → SELECT descendant
 *   v2.0 只有列名没有方向约定，极易写反。
 *
 * PK 顺序 (relation, ancestor, descendant) —— 支持"某关系的某上位节点的所有下位"前缀扫描
 * depth      : 该配对的**最短**路径长度
 * first_edge : 见证边(witness)，指向 relations.relation_id
 *              → 删除一条边时，只需失效 first_edge = 该边 的行，
 *                再排队重建，删除成本从 O(N) 降到 O(受影响对数)
 *              → 且不必存整条 path 字符串，大幅省空间
 * 注意：first_edge 是**弱引用**，刻意不加外键。
 *       若加 ON DELETE SET NULL，外键动作会在 AFTER DELETE 触发器之前把值清空，
 *       触发器就再也匹配不到失效行（这是 v3.0 初稿踩到的真实坑）。 */
CREATE TABLE IF NOT EXISTS closure (
    relation   TEXT NOT NULL REFERENCES relation_types(relation),
    ancestor   INTEGER NOT NULL,
    descendant INTEGER NOT NULL,
    depth      INTEGER NOT NULL CHECK (depth BETWEEN 1 AND 64),
    first_edge INTEGER,
    PRIMARY KEY (relation, ancestor, descendant),
    CHECK (ancestor <> descendant)
) STRICT, WITHOUT ROWID;

-- 重建队列：INSERT 一行即触发该关系的全量重建（应用层唯一需要知道的维护接口）
CREATE TABLE IF NOT EXISTS closure_rebuild_queue (
    relation     TEXT PRIMARY KEY REFERENCES relation_types(relation),
    requested_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

CREATE TABLE IF NOT EXISTS inference_rules (
    rule_id      INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    rule_type    TEXT NOT NULL CHECK (rule_type IN
                   ('transitive','inverse','symmetric','inherit','constraint','default','reject')),
    -- 声明式前提/结论（推荐）：可被索引、可解释、无注入风险
    description  TEXT NOT NULL,
    -- 逃生通道：需要复杂推理时仍可落 SQL 模板，但必须显式标记
    condition_sql TEXT CHECK (condition_sql IS NULL OR length(trim(condition_sql)) > 0),
    action_sql    TEXT,
    priority     INTEGER NOT NULL DEFAULT 0 CHECK (priority BETWEEN -100 AND 100),
    is_enabled   INTEGER NOT NULL DEFAULT 1 CHECK (is_enabled IN (0,1)),
    version      TEXT NOT NULL DEFAULT '1.0',
    CHECK (rule_type <> 'inherit' OR condition_sql IS NOT NULL)
) STRICT;

CREATE TABLE IF NOT EXISTS rule_premises (
    rule_id   INTEGER NOT NULL REFERENCES inference_rules(rule_id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL CHECK (seq >= 1),
    kind      TEXT NOT NULL CHECK (kind IN ('edge','closure','feature','pattern','slot','category')),
    subject   TEXT NOT NULL,
    predicate TEXT,
    object    TEXT,
    is_negated INTEGER NOT NULL DEFAULT 0 CHECK (is_negated IN (0,1)),
    PRIMARY KEY (rule_id, seq)
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS rule_conclusions (
    rule_id    INTEGER NOT NULL REFERENCES inference_rules(rule_id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL CHECK (seq >= 1),
    kind       TEXT NOT NULL CHECK (kind IN ('edge','feature','tag','role','reject','bind')),
    subject    TEXT,
    relation   TEXT,
    object     TEXT,
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    PRIMARY KEY (rule_id, seq)
) STRICT, WITHOUT ROWID;

/* 省略回填：同一槽位允许多个候选策略 + 优先级（v2.0 的 PK 只允许一条，是硬伤） */
CREATE TABLE IF NOT EXISTS ellipsis_rules (
    ellipsis_rule_id INTEGER PRIMARY KEY,
    context_type     TEXT NOT NULL CHECK (context_type IN
                       ('dialogue','narrative','imperative','expository','topic_chain')),
    slot_role        TEXT NOT NULL REFERENCES semantic_roles(role),
    strategy         TEXT NOT NULL CHECK (strategy IN
                       ('previous_subject','previous_object','previous_mentioned','speaker','listener',
                        'topic','nearest_np','salient_entity','world_knowledge','indefinite_default')),
    priority         INTEGER NOT NULL DEFAULT 0,
    max_distance     INTEGER CHECK (max_distance IS NULL OR max_distance >= 0),  -- 最大跨句数
    description      TEXT,
    UNIQUE (context_type, slot_role, strategy, priority)
) STRICT;


/* =========================================================================
 * 8. 篇章层（新增）—— 汉语靠话题链与零形回指消解省略，v2.0 无承载表
 * ========================================================================= */

CREATE TABLE IF NOT EXISTS utterances (
    utterance_id INTEGER PRIMARY KEY,
    doc_id       TEXT NOT NULL DEFAULT 'default',
    seq          INTEGER NOT NULL CHECK (seq >= 1),
    text         TEXT NOT NULL,
    speaker      TEXT,
    mood         TEXT REFERENCES sentence_moods(mood),
    pattern_id   INTEGER REFERENCES sentence_patterns(pattern_id),
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (doc_id, seq)
) STRICT;

CREATE TABLE IF NOT EXISTS topic_chains (
    chain_id   INTEGER PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    topic_node_id INTEGER,                 -- 话题所指（sense 或 entity_node）
    start_seq  INTEGER NOT NULL,
    end_seq    INTEGER,
    chain_type TEXT NOT NULL CHECK (chain_type IN ('话题链','主谓链','零形回指链','同指链')),
    CHECK (end_seq IS NULL OR end_seq >= start_seq)
) STRICT;

CREATE TABLE IF NOT EXISTS anaphora (
    anaphora_id  INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterances(utterance_id) ON DELETE CASCADE,
    slot_role    TEXT REFERENCES semantic_roles(role),
    form         TEXT NOT NULL,            -- 表层形式: '他' / '∅'
    form_type    TEXT NOT NULL CHECK (form_type IN
                   ('pronoun','zero','definite_np','demonstrative','proper_noun','collective')),
    antecedent_utterance_id INTEGER REFERENCES utterances(utterance_id),
    antecedent_node_id      INTEGER,       -- sense_id 或 entity_nodes.node_id
    chain_id     INTEGER REFERENCES topic_chains(chain_id) ON DELETE SET NULL,
    confidence   REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1)
) STRICT;


/* =========================================================================
 * 9. 管理层
 * ========================================================================= */

CREATE TABLE IF NOT EXISTS contexts (
    context_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    domain      TEXT REFERENCES domains(domain)
) STRICT;

CREATE TABLE IF NOT EXISTS sense_contexts (
    sense_id   INTEGER NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    context_id INTEGER NOT NULL REFERENCES contexts(context_id) ON DELETE CASCADE,
    relevance  REAL NOT NULL DEFAULT 1.0 CHECK (relevance BETWEEN 0 AND 1),
    is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0,1)),
    PRIMARY KEY (sense_id, context_id)
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS import_batches (
    batch_id    INTEGER PRIMARY KEY,
    source_id   INTEGER REFERENCES sources(source_id),
    started_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    finished_at TEXT,
    row_count   INTEGER NOT NULL DEFAULT 0 CHECK (row_count >= 0),
    status      TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','ok','failed','rolled_back')),
    note        TEXT
) STRICT;

/* 开放属性（JSON1）：加新领域字段不改表结构 */
CREATE TABLE IF NOT EXISTS attributes (
    attr_id    INTEGER PRIMARY KEY,
    owner_kind TEXT NOT NULL CHECK (owner_kind IN ('sense','word','character','pattern','node','relation','utterance')),
    owner_id   INTEGER NOT NULL,
    key        TEXT NOT NULL,
    value_json TEXT NOT NULL CHECK (json_valid(value_json)),
    source_id  INTEGER REFERENCES sources(source_id),
    UNIQUE (owner_kind, owner_id, key, value_json)
) STRICT;

CREATE TABLE IF NOT EXISTS changelog (
    change_id  INTEGER PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id  TEXT NOT NULL,              -- TEXT：兼容 characters.char 这类文本主键
    operation  TEXT NOT NULL CHECK (operation IN ('INSERT','UPDATE','DELETE')),
    old_value  TEXT CHECK (old_value IS NULL OR json_valid(old_value)),
    new_value  TEXT CHECK (new_value IS NULL OR json_valid(new_value)),
    changed_by TEXT NOT NULL DEFAULT 'system',
    changed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    batch_id   INTEGER REFERENCES import_batches(batch_id)
) STRICT;


/* =========================================================================
 * 10. 全文检索（FTS5）—— 中文必须双通道
 *    trigram : 支持中文子串匹配，但要求查询 >= 3 字符
 *    unicode61: 作用于 jieba 预分词后的 definition_seg，支持 1-2 字词
 * ========================================================================= */

CREATE VIRTUAL TABLE IF NOT EXISTS senses_fts_trigram USING fts5(
    definition,
    example,
    content='senses',
    content_rowid='sense_id',
    tokenize='trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS senses_fts_seg USING fts5(
    definition_seg,
    content='senses',
    content_rowid='sense_id',
    tokenize='unicode61 remove_diacritics 2'
);

-- 外部内容表的同步触发器（保持 senses_fts_* 与 senses 一致）
CREATE TRIGGER IF NOT EXISTS tr_senses_fts_ai AFTER INSERT ON senses BEGIN
    INSERT INTO senses_fts_trigram(rowid, definition, example)
        VALUES (new.sense_id, new.definition, new.example);
    INSERT INTO senses_fts_seg(rowid, definition_seg)
        VALUES (new.sense_id, new.definition_seg);
END;

CREATE TRIGGER IF NOT EXISTS tr_senses_fts_ad AFTER DELETE ON senses BEGIN
    INSERT INTO senses_fts_trigram(senses_fts_trigram, rowid, definition, example)
        VALUES ('delete', old.sense_id, old.definition, old.example);
    INSERT INTO senses_fts_seg(senses_fts_seg, rowid, definition_seg)
        VALUES ('delete', old.sense_id, old.definition_seg);
END;

CREATE TRIGGER IF NOT EXISTS tr_senses_fts_au AFTER UPDATE ON senses BEGIN
    INSERT INTO senses_fts_trigram(senses_fts_trigram, rowid, definition, example)
        VALUES ('delete', old.sense_id, old.definition, old.example);
    INSERT INTO senses_fts_seg(senses_fts_seg, rowid, definition_seg)
        VALUES ('delete', old.sense_id, old.definition_seg);
    INSERT INTO senses_fts_trigram(rowid, definition, example)
        VALUES (new.sense_id, new.definition, new.example);
    INSERT INTO senses_fts_seg(rowid, definition_seg)
        VALUES (new.sense_id, new.definition_seg);
END;


/* =========================================================================
 * 11. 视图
 * ========================================================================= */

-- 正向 + 逆向 + 对称展开：杜绝反向边冗余存储
CREATE VIEW IF NOT EXISTS v_relations_all AS
    SELECT relation_id, src_kind, src_id, relation, dst_kind, dst_id, weight, confidence, 0 AS is_inferred
      FROM relations
    UNION ALL
    SELECT r.relation_id, r.dst_kind, r.dst_id, t.inverse_of, r.src_kind, r.src_id, r.weight, r.confidence, 0
      FROM relations r JOIN relation_types t ON t.relation = r.relation
     WHERE t.inverse_of IS NOT NULL AND t.direction = 'directed'
    UNION ALL
    SELECT r.relation_id, r.dst_kind, r.dst_id, r.relation, r.src_kind, r.src_id, r.weight, r.confidence, 0
      FROM relations r JOIN relation_types t ON t.relation = r.relation
     WHERE t.direction = 'symmetric';

-- 词 → 义项 → 虚词/量词 的合并视图（应用层常用）
CREATE VIEW IF NOT EXISTS v_senses_full AS
SELECT s.sense_id, w.word, w.word_id, w.char_count, s.pos, s.sense_no, s.definition,
       s.pinyin, s.frequency,
       CASE WHEN fw.sense_id IS NOT NULL THEN 1 ELSE 0 END AS is_function_word,
       fw.category AS fw_category, fw.subcategory AS fw_subcategory,
       CASE WHEN cl.classifier_sense_id IS NOT NULL THEN 1 ELSE 0 END AS is_classifier,
       cl.kind AS classifier_kind
  FROM senses s
  JOIN words w ON w.word_id = s.word_id
  LEFT JOIN function_words fw ON fw.sense_id = s.sense_id
  LEFT JOIN classifiers cl    ON cl.classifier_sense_id = s.sense_id;

-- 深度 1-2 的轻量闭包视图：常见推理无需触达物化闭包表
CREATE VIEW IF NOT EXISTS v_closure_2 AS
SELECT r1.relation, r1.src_id AS ancestor, r2.dst_id AS descendant, 2 AS depth
  FROM relations r1
  JOIN relations r2 ON r2.src_id = r1.dst_id AND r2.relation = r1.relation
 WHERE r1.src_kind='sense' AND r1.dst_kind='sense'
   AND r2.src_kind='sense' AND r2.dst_kind='sense'
   AND r1.src_id <> r2.dst_id;

-- 兼容 v2.0 表名，避免应用层立刻改代码
CREATE VIEW IF NOT EXISTS entries AS
SELECT word_id AS entry_id, word, char_count, is_separable, separator_pos
  FROM words;

CREATE VIEW IF NOT EXISTS cross_links AS
SELECT r.src_kind AS source_kind, r.src_id AS source_id,
       r.dst_kind AS target_kind, r.dst_id AS target_id, r.relation
  FROM relations r
 WHERE r.src_kind = 'node' OR r.dst_kind = 'node';


/* =========================================================================
 * 12. 索引
 * =========================================================================
 * 【去冗余原则】
 *   任何"列集是既有 UNIQUE / PK 隐含索引**真前缀**"的普通索引一律不建——
 *   二者可服务完全相同的查询，多建只会白付写入成本、页缓存与磁盘。
 *   本机由 audit_indexes.py **程序化**体检（不靠肉眼比对）确认归零 5 个冗余：
 *     senses(word_id,pos)                  ⊂ UNIQUE(word_id,pos,definition)
 *     morphemes(form)                      ⊂ UNIQUE(form,morpheme_type)
 *     relations(src_kind,src_id,relation)  ⊂ UNIQUE(src_kind,src_id,relation,dst_kind,dst_id)
 *     attributes(owner_kind,owner_id,key)  ⊂ UNIQUE(owner_kind,owner_id,key,value_json)
 *     pattern_slots(pattern_id,position)   ≡ UNIQUE(pattern_id,position)  （完全重复）
 *   反向不成立：relations(dst_kind,dst_id,relation) 以 dst 打头，
 *   UNIQUE 索引（src 打头）无法替代，故必须保留。
 * 【部分索引不算冗余】带 WHERE 的索引天然比完整索引小，是刻意优化，即使列集是
 *   某唯一索引的前缀也保留（例：closure(relation,ancestor) WHERE depth = 1）。
 * ========================================================================= */

CREATE INDEX IF NOT EXISTS idx_words_separable      ON words(separator_pos) WHERE separator_pos IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_words_type           ON words(word_type);
CREATE INDEX IF NOT EXISTS idx_senses_pos           ON senses(pos) WHERE is_primary = 1;
CREATE INDEX IF NOT EXISTS idx_senses_subcat        ON senses(subcategory) WHERE subcategory IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_senses_derived       ON senses(derived_from) WHERE derived_from IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_senses_plain         ON senses(pinyin_plain) WHERE pinyin_plain IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_senses_initial       ON senses(pinyin_initial) WHERE pinyin_initial IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_senses_freq          ON senses(frequency DESC) WHERE is_primary = 1;
CREATE INDEX IF NOT EXISTS idx_function_words_cat   ON function_words(category, subcategory);
CREATE INDEX IF NOT EXISTS idx_fw_roles_role        ON function_word_roles(role);
CREATE INDEX IF NOT EXISTS idx_classifiers_kind     ON classifiers(kind, subtype);
CREATE INDEX IF NOT EXISTS idx_nc_pairs_classifier  ON noun_classifier_pairs(classifier_sense_id, preference DESC);
CREATE INDEX IF NOT EXISTS idx_char_readings_char   ON character_readings(char, is_default);

-- 只留 dst 方向：src 方向 (src_kind,src_id,relation) 已被 UNIQUE 的隐含索引覆盖
CREATE INDEX IF NOT EXISTS idx_relations_dst        ON relations(dst_kind, dst_id, relation);
CREATE INDEX IF NOT EXISTS idx_relations_relation   ON relations(relation) WHERE src_kind = 'sense';
CREATE INDEX IF NOT EXISTS idx_relations_ctx        ON relations(context_id) WHERE context_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_relations_source     ON relations(source_id) WHERE source_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rel_constraints      ON relation_constraints(slot, kind, value);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_lookup  ON entity_nodes(node_type, domain, entity_class);

CREATE INDEX IF NOT EXISTS idx_closure_relation_desc ON closure(relation, descendant, depth);
CREATE INDEX IF NOT EXISTS idx_closure_first_edge    ON closure(first_edge) WHERE first_edge IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_closure_depth1        ON closure(relation, ancestor) WHERE depth = 1;

CREATE INDEX IF NOT EXISTS idx_slot_constraints     ON slot_constraints(kind, value);
CREATE INDEX IF NOT EXISTS idx_complements_head     ON complements(head_sense_id, comp_type);

CREATE INDEX IF NOT EXISTS idx_ellipsis_ctx         ON ellipsis_rules(context_type, slot_role, priority DESC);
CREATE INDEX IF NOT EXISTS idx_rule_premises        ON rule_premises(kind, subject, predicate);
CREATE INDEX IF NOT EXISTS idx_rule_conclusions     ON rule_conclusions(kind, subject, relation);

CREATE INDEX IF NOT EXISTS idx_anaphora_utt         ON anaphora(utterance_id, slot_role);
CREATE INDEX IF NOT EXISTS idx_anaphora_chain       ON anaphora(chain_id) WHERE chain_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_topic_chains_doc     ON topic_chains(doc_id, start_seq);
CREATE INDEX IF NOT EXISTS idx_sense_contexts_ctx   ON sense_contexts(context_id, relevance DESC);
CREATE INDEX IF NOT EXISTS idx_changelog_record     ON changelog(table_name, record_id, changed_at DESC);
CREATE INDEX IF NOT EXISTS idx_changelog_time       ON changelog(changed_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_id_lookup     ON sources(kind, name);


/* =========================================================================
 * 13. 完整性触发与维护例程
 * ========================================================================= */

/* ---- 13.1 关系端点存在性校验（异构节点的 FK 替代方案） ---- */
CREATE TRIGGER IF NOT EXISTS tr_relations_validate_ai BEFORE INSERT ON relations
WHEN EXISTS (SELECT 1 FROM (
        SELECT 1 WHERE
            (NEW.src_kind='sense' AND NOT EXISTS(SELECT 1 FROM senses WHERE sense_id=NEW.src_id))
         OR (NEW.src_kind='node'  AND NOT EXISTS(SELECT 1 FROM entity_nodes WHERE node_id=NEW.src_id))
         OR (NEW.dst_kind='sense' AND NOT EXISTS(SELECT 1 FROM senses WHERE sense_id=NEW.dst_id))
         OR (NEW.dst_kind='node'  AND NOT EXISTS(SELECT 1 FROM entity_nodes WHERE node_id=NEW.dst_id))
     ))
BEGIN
    SELECT RAISE(ABORT, 'relations: src/dst 端点不存在');
END;

-- 端点身份不可变（异构 ID 换来源没有意义）
CREATE TRIGGER IF NOT EXISTS tr_relations_freeze_kind_bu BEFORE UPDATE OF src_kind, src_id, dst_kind, dst_id ON relations
BEGIN
    SELECT RAISE(ABORT, 'relations: 端点不可修改，请删除后重建');
END;

/* relation_types 逆向声明的成对性校验
 * 注意：**不能用 BEFORE INSERT 触发器**校验——互为逆向的两个关系
 * （has_part ⇄ part_of）通常在同一条 INSERT 语句里写入，
 * 此时"另一个"还不存在，触发器会误杀合法数据（实测已验证）。
 * 改为离线一致性查询，见 §15.8。 */

/* ---- 13.2 闭包增量维护 ----
 * 关键实现要点（均已在本机 SQLite 3.53.1 实测）：
 *   (a) 仅在 is_transitive=1 且 closure_policy.mode='incremental' 时生效
 *   (b) 触发器体内**不允许** `WITH ... INSERT`（语法错误 near "INSERT"），
 *       必须把递归 CTE 嵌进 FROM 子查询：INSERT ... SELECT ... FROM (WITH RECURSIVE ...)
 *   (c) UPSERT 的 SET 子句在 SQLite 中**从左到右求值，后面的表达式看得见前面的结果**，
 *       因此 first_edge 必须写在 depth 之前，否则 `excluded.depth < depth` 恒为假
 *   (d) WHERE anc <> des 排除自环，避免触碰 CHECK(ancestor <> descendant)
 *   (e) 【v3.0.1】深度上限统一由最终 WHERE 的 `L.d + 1 + H.d <= max_depth` 卡口，
 *       lo/hi 递归项只做 `<= max_depth-1` 的剪枝。否则增量路径的实际上限是
 *       2*max_depth+1，与重建路径的 max_depth 不一致（同一份数据两种闭包）。
 *   (f) 关系必须有 closure_policy 行：触发器条件写的是 `mode = 'incremental'`，
 *       缺行时子查询返回 NULL、比较为假 → 静默不产生闭包，不报任何错。
 */
CREATE TRIGGER IF NOT EXISTS tr_closure_edge_ai AFTER INSERT ON relations
WHEN NEW.src_kind='sense' AND NEW.dst_kind='sense'
 AND (SELECT is_transitive FROM relation_types WHERE relation = NEW.relation) = 1
 AND (SELECT mode FROM closure_policy WHERE relation = NEW.relation) = 'incremental'
BEGIN
    INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
    SELECT NEW.relation, H.y, L.x, L.d + 1 + H.d, L.w
      FROM (WITH RECURSIVE
              -- lo: 从 src 出发，沿 is_a 反向收集"比 src 更具体"的节点（含 src 自身）
              lo(x, d, w) AS (
                  SELECT NEW.src_id, 0, NEW.relation_id
                  UNION
                  SELECT cl.descendant, lo.d + cl.depth, cl.first_edge
                    FROM closure cl JOIN lo ON cl.ancestor = lo.x
                   WHERE cl.relation = NEW.relation
                     -- 收紧到 < max_depth（即 lo.d ≤ max_depth-1），
                     -- 最终深度由下方 WHERE 的 `L.d + 1 + H.d <= max_depth` 统一卡口
                     AND lo.d + cl.depth < (SELECT max_depth FROM closure_policy
                                            WHERE relation = NEW.relation)
              )
            SELECT x, d, w FROM lo) AS L
      CROSS JOIN (WITH RECURSIVE
              -- hi: 从 dst 出发，沿 is_a 正向收集"比 dst 更一般"的节点（含 dst 自身）
              hi(y, d) AS (
                  SELECT NEW.dst_id, 0
                  UNION
                  SELECT cl.ancestor, hi.d + cl.depth
                    FROM closure cl JOIN hi ON cl.descendant = hi.y
                   WHERE cl.relation = NEW.relation
                     AND hi.d + cl.depth < (SELECT max_depth FROM closure_policy
                                            WHERE relation = NEW.relation)
              )
            SELECT y, d FROM hi) AS H
     WHERE L.x <> H.y
       -- 【v3.0.1 修正】深度上限必须在此统一卡口。
       --   旧写法只在 lo/hi 的递归项上卡 max_depth，而最终深度是 L.d + 1 + H.d，
       --   于是增量路径能产出 depth ≤ 2*max_depth+1 的行（实测 max_depth=2 产出 depth=3），
       --   而重建路径严格 ≤ max_depth → 两条路径结果不一致（同一份数据两种闭包）。
       AND L.d + 1 + H.d <= (SELECT max_depth FROM closure_policy
                             WHERE relation = NEW.relation)
    ON CONFLICT(relation, ancestor, descendant) DO UPDATE SET
        first_edge = CASE WHEN excluded.depth < depth THEN excluded.first_edge ELSE first_edge END,
        depth      = MIN(depth, excluded.depth);
END;

-- 删除边：先失效以其为见证的闭包行，再排队重建（删除是低频操作，重建保证正确）
CREATE TRIGGER IF NOT EXISTS tr_closure_edge_ad AFTER DELETE ON relations
WHEN OLD.src_kind='sense' AND OLD.dst_kind='sense'
 AND (SELECT is_transitive FROM relation_types WHERE relation = OLD.relation) = 1
 AND (SELECT mode FROM closure_policy WHERE relation = OLD.relation) <> 'off'
BEGIN
    DELETE FROM closure WHERE first_edge = OLD.relation_id;
    INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at)
        VALUES (OLD.relation, strftime('%Y-%m-%dT%H:%M:%fZ','now'));
END;

/* ---- 13.3 闭包全量重建（幂等）：应用层只需 INSERT 一行队列 ----
 * 同样需要把递归 CTE 嵌进 FROM 子查询（触发器内不允许 WITH ... INSERT）。
 * 借助 SQLite 的"裸列 + min()"规则：GROUP BY 时裸列取自取到最小值的行，
 * 从而一次取出「最短深度」与其对应见证边。
 */
CREATE TRIGGER IF NOT EXISTS tr_closure_rebuild_ai AFTER INSERT ON closure_rebuild_queue
BEGIN
    DELETE FROM closure WHERE relation = NEW.relation;

    INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
    SELECT NEW.relation, W.cur, W.anc, MIN(W.depth), W.first_edge
      FROM (WITH RECURSIVE walk(anc, cur, depth, first_edge, visited) AS (
                SELECT r.src_id, r.dst_id, 1, r.relation_id,
                       ',' || r.src_id || ',' || r.dst_id || ','
                  FROM relations r
                 WHERE r.relation = NEW.relation
                   AND r.src_kind='sense' AND r.dst_kind='sense'
                UNION ALL
                SELECT w.anc, r.dst_id, w.depth + 1, w.first_edge,
                       w.visited || r.dst_id || ','
                  FROM walk w JOIN relations r ON r.src_id = w.cur
                 WHERE r.relation = NEW.relation
                   AND r.src_kind='sense' AND r.dst_kind='sense'
                   AND w.depth < (SELECT max_depth FROM closure_policy WHERE relation = NEW.relation)
                   AND instr(w.visited, ',' || r.dst_id || ',') = 0
            )
            SELECT anc, cur, depth, first_edge FROM walk) AS W
     WHERE W.anc <> W.cur
     GROUP BY W.cur, W.anc;

    DELETE FROM closure_rebuild_queue WHERE relation = NEW.relation;
END;

/* ---- 13.4 审计：写入变更日志 ---- */
CREATE TRIGGER IF NOT EXISTS tr_relations_audit_ai AFTER INSERT ON relations BEGIN
    INSERT INTO changelog(table_name, record_id, operation, new_value)
    VALUES ('relations', NEW.relation_id, 'INSERT',
            json_object('src', NEW.src_id, 'relation', NEW.relation, 'dst', NEW.dst_id));
END;

CREATE TRIGGER IF NOT EXISTS tr_relations_audit_ad AFTER DELETE ON relations BEGIN
    INSERT INTO changelog(table_name, record_id, operation, old_value)
    VALUES ('relations', OLD.relation_id, 'DELETE',
            json_object('src', OLD.src_id, 'relation', OLD.relation, 'dst', OLD.dst_id));
END;

/* ---- 13.5 批量导入提速：临时停用闭包增量触发 ---- */
-- UPDATE closure_policy SET mode='batch' WHERE relation IN (...);
-- ... 导入 ...
-- INSERT INTO closure_rebuild_queue(relation) SELECT relation FROM closure_policy WHERE mode='batch';
-- UPDATE closure_policy SET mode='incremental' WHERE relation IN (...);


/* =========================================================================
 * 14. 基础数据
 * ========================================================================= */

INSERT INTO schema_meta(key, value) VALUES
 ('schema_version', '3.0.0'),
 ('sqlite_feature_level', 'STRICT + generated columns + upsert + trigram'),
 ('created_at', strftime('%Y-%m-%dT%H:%M:%fZ','now'));

INSERT INTO domains(domain, description, is_core) VALUES
 ('core','通用汉语核心库',1),
 ('animal','动物学库',0),
 ('medical','医学库',0),
 ('tech','科技/互联网库',0);

INSERT INTO sources(source_id, name, kind, license, note) VALUES
 (1,'现代汉语词典(人工整理)','词典',NULL,'符号语义释义基准'),
 (2,'KEGG','KEGG',NULL,'通路/基因/疾病'),
 (3,'Reactome','Reactome',NULL,'通路层级'),
 (4,'GO','GO',NULL,'基因本体'),
 (5,'人工标注','人工标注',NULL,'义项/句式/量词搭配');

INSERT INTO semantic_categories(cat, parent_cat, description) VALUES
 ('实体',NULL,'泛指一切实体'),
 ('生物','实体','生命体'),
 ('人','生物','人类'),
 ('动物','生物','动物'),
 ('植物','生物','植物'),
 ('无机物','实体','非生命自然物'),
 ('人工物','实体','人造物'),
 ('器物','人工物','工具器皿'),
 ('抽象','实体','抽象概念'),
 ('时间','抽象','时间概念'),
 ('处所','抽象','空间/处所'),
 ('事件','抽象','动作事件'),
 ('性状','抽象','性质状态');

INSERT INTO semantic_features(feature, feature_group, value_type, description) VALUES
 ('+animate','生命度','boolean','有生命'),
 ('+human','生命度','boolean','人类'),
 ('+countable','可数性','boolean','可个体计数'),
 ('+concrete','具体性','boolean','具体可感'),
 ('+volitional','自主性','boolean','动作有意志控制'),
 ('+bounded','有界性','boolean','内部有终止点'),
 ('polarity','极性','enum','正/负/中性');

INSERT INTO pos_tags(pos, name, pos_group, description) VALUES
 ('n','名词','实词','表人/事物'),
 ('nt','时间词','实词','表时间'),
 ('nl','处所词','实词','表处所'),
 ('nd','方位词','实词','表方向位置'),
 ('v','动词','实词','表动作行为'),
 ('a','形容词','实词','表性质状态'),
 ('b','区别词','实词','非谓形容词'),
 ('m','数词','实词','表数目'),
 ('q','量词','实词','表单位'),
 ('d','副词','实词','修饰谓词'),
 ('r','代词','实词','替代指称'),
 ('o','拟声词','实词','摹拟声音'),
 ('e','叹词','实词','表感叹'),
 ('p','介词','虚词','引介论元'),
 ('c','连词','虚词','连接'),
 ('u','助词','虚词','附着'),
 ('y','语气词','虚词','表语气'),
 ('i','成语','特殊','固定短语'),
 ('l','惯用语','特殊','习惯表达'),
 ('j','缩略语','特殊','简称');

INSERT INTO syntactic_functions(function_id, description, position) VALUES
 ('主语','句子的陈述对象','前'),
 ('谓语','对主语加以陈述','后'),
 ('宾语','谓语的支配对象','后'),
 ('定语','修饰名词性中心语','前'),
 ('状语','修饰谓词性中心语','前'),
 ('补语','补充说明谓词','后'),
 ('中心语','被修饰的成分','灵活'),
 ('独立语','不与其他成分发生结构关系','灵活'),
 ('话题','句子谈论的对象（不等于主语）','前');

INSERT INTO sentence_moods(mood, description, typical_markers) VALUES
 ('陈述','叙述或说明','了/的/呢'),
 ('疑问','提出问题','吗/呢/吧/难道/什么/几'),
 ('祈使','提出请求或命令','吧/请/别/要'),
 ('感叹','抒发情感','啊/多么/太...了');

INSERT INTO complement_types(comp_type, description, marker, example) VALUES
 ('结果补语','表示动作结果','','看完/听懂'),
 ('趋向补语','表示动作趋向','','走出来/跑上去'),
 ('可能补语','表示可能性','得/不','看得完/看不懂'),
 ('程度补语','表示程度','得','好得很/热死了'),
 ('数量补语','表示数量/时量','','去过三次/等了半天'),
 ('状态补语','描写动作状态','得','跑得很快'),
 ('时地补语','表示时间处所','','生于1990年');

INSERT INTO radical_semantics(radical, meaning, examples) VALUES
 ('氵','与水有关','江/河/湖'),
 ('木','与树木有关','林/树/枝'),
 ('扌','与手的动作有关','打/拿/提'),
 ('口','与口或言语有关','吃/叫/喊'),
 ('心','与心理活动有关','想/思/念'),
 ('亻','与人有关','他/们/住'),
 ('犭','与兽类有关','猫/狗/狼'),
 ('艹','与草木有关','花/草/苗'),
 ('亠','与头部或覆盖有关','高/京'),
 ('八','与分张有关','兴/分');

INSERT INTO characters(char, radical, strokes, structure, semantic_cat, is_common, hint) VALUES
 ('猫','犭',11,'左右','动物',1,'形声字：犭表义，苗表音'),
 ('狗','犭',8,'左右','动物',1,'形声字'),
 ('洗','氵',9,'左右','事件',1,'形声字：氵表义，先表音'),
 ('澡','氵',16,'左右','事件',0,'形声字，多用于"洗澡"'),
 ('花','艹',7,'上下','植物',1,'形声字：艹表义，化表音'),
 ('高','亠',10,'上下','性状',1,'象形字'),
 ('他','亻',5,'左右','人',1,'形声字，代词'),
 ('兴','八',6,'上下','性状',0,'多音字：xīng / xìng');

-- 多音字：一字多音，文白异读与姓氏读音
INSERT INTO character_readings(char, pinyin, tone, reading_type, meaning_hint, is_default) VALUES
 ('兴','xing1',1,'常读','兴起、兴盛',0),
 ('兴','xing4',4,'常读','兴致、高兴',1),
 ('花','hua1',1,'常读','花朵',1),
 ('洗','xi3',3,'常读','洗涤',1),
 ('洗','xian3',3,'姓氏','姓氏"洗"',0);

-- 音变：变调 / 轻声 / 儿化（v2.0 只在 pinyin_json 里塞字符串，无法推理）
INSERT INTO phonological_rules(rule_type, source_form, condition, surface_form, example) VALUES
 ('变调','一','后字为去声(4声)','yi2','一个 yí ge'),
 ('变调','一','后字为阴平/阳平/上声','yi4','一天 yì tiān / 一年 yì nián'),
 ('变调','不','后字为去声(4声)','bu2','不是 bú shì'),
 ('连读变调','上声+上声','前字为3声且后字为3声','3->2','你好 ní hǎo'),
 ('轻声','叠音亲属称谓','第二音节','0','妈妈 māma / 爸爸 bàba'),
 ('轻声','子','作后缀时','0','桌子 zhuōzi'),
 ('轻声','头','作后缀时','0','石头 shítou'),
 ('儿化','儿','作词尾时','r','花儿 huār'),
 ('儿化变韵','儿','前字韵母为a/o/e/u时','r + 卷舌','花儿 huār / 歌儿 gēr');

INSERT INTO semantic_roles(role, parent_role, role_group, description, typical_pos, case_marker) VALUES
 ('施事',NULL,'参与者','动作的发出者（有意志）','n','被/由'),
 ('致事',NULL,'参与者','引起状态变化的事物（无意志）','n','使/让'),
 ('感事',NULL,'参与者','心理/感知经验的主体','n',''),
 ('受事',NULL,'参与者','动作的直接承受者','n','把/将'),
 ('结果','受事','参与者','动作新产生的对象','n','把'),
 ('与事',NULL,'参与者','动作的接受/交与对象','n','给/向/对'),
 ('对象',NULL,'参与者','动作所关涉的对象','n','对/跟/和'),
 ('受益者',NULL,'参与者','因动作而受益的人','n','为/替/给'),
 ('受损者',NULL,'参与者','因动作而受损的人','n','被/让'),
 ('系事',NULL,'参与者','判断句中被说明的对象','n','是'),
 ('工具',NULL,'参与者','动作所凭借的器具','n','用/拿'),
 ('材料',NULL,'参与者','动作所消耗/使用的材料','n','用'),
 ('方式',NULL,'参与者','动作进行的方式','d',''),
 ('伴随',NULL,'参与者','动作的陪同对象','n','跟/和/与'),
 ('比较对象',NULL,'参与者','比较句中的参照项','n','比/跟/和'),
 ('喻体',NULL,'参与者','比况所参照的事物','n','像'),
 ('处所',NULL,'环境','动作发生的地点','nl','在'),
 ('起点',NULL,'环境','动作或位移的起点','nl','从/自'),
 ('终点',NULL,'环境','动作或位移的终点','nl','到/往'),
 ('经由',NULL,'环境','位移经过的路径','nl','经过/沿着'),
 ('方向',NULL,'环境','动作朝向的方向','nd','往/朝'),
 ('时点',NULL,'环境','动作发生的时间点','nt','在/当'),
 ('时段',NULL,'环境','动作持续的时间长度','nt',''),
 ('频次',NULL,'计量','动作重复的次数','m+q',''),
 ('数量',NULL,'计量','动作所涉的数量','m+q',''),
 ('程度',NULL,'计量','性质或动作的程度','d',''),
 ('幅度',NULL,'计量','动作在量上的跨度','m+q',''),
 ('原因',NULL,'因果','动作或状态的原因','n','因为/由于'),
 ('目的',NULL,'因果','动作的目的','n','为了'),
 ('条件',NULL,'因果','动作发生的条件','n','只要/如果'),
 ('让步',NULL,'因果','退让承认的前提','n','虽然/即使'),
 ('效果',NULL,'因果','动作造成的结果','n',''),
 ('主题',NULL,'命题','述题所关涉的事物（不等于受事）','n',''),
 ('话题',NULL,'命题','句子谈论的对象（可跨句维持）','n',''),
 ('述题',NULL,'命题','对话题的评述','v',''),
 ('命题',NULL,'命题','被判断/被陈述的完整命题','v',''),
 ('焦点',NULL,'命题','句子信息重心','','是/连');

INSERT INTO relation_types
 (relation, relation_class, direction, inverse_of, is_transitive, is_symmetric, is_reflexive, is_functional, max_depth, description, color_code) VALUES
 ('is_a','层级','directed',NULL,1,0,1,0,12,'下位→上位（成员/子类）','#2563EB'),
 ('has_part','层级','directed','part_of',1,0,0,0,12,'整体→部分','#7C3AED'),
 ('part_of','层级','directed','has_part',1,0,0,1,12,'部分→整体','#7C3AED'),
 ('instance_of','层级','directed',NULL,1,0,1,0,12,'个体→类','#2563EB'),
 ('synonym','词汇语义','symmetric',NULL,1,1,1,0,12,'同义/近义（≠相似）','#059669'),
 ('similar','词汇语义','symmetric',NULL,0,1,1,0,4,'相关/相似（弱关联）','#10B981'),
 ('opposite','词汇语义','symmetric',NULL,0,1,0,0,4,'反义（含互补/相对/极性）','#DC2626'),
 ('antonym_complementary','词汇语义','symmetric',NULL,0,1,0,0,4,'互补反义（生/死）','#EF4444'),
 ('antonym_gradable','词汇语义','symmetric',NULL,0,1,0,0,4,'相对反义（大/小）','#EF4444'),
 ('antonym_converse','词汇语义','symmetric',NULL,0,1,0,0,4,'反向反义（买/卖）','#EF4444'),
 ('derived_from','词汇语义','directed',NULL,1,0,0,0,12,'本义→引申义派生','#8B5CF6'),
 ('metaphor_of','词汇语义','directed',NULL,0,0,0,0,4,'隐喻来源','#A78BFA'),
 ('metonymy_of','词汇语义','directed',NULL,0,0,0,0,4,'借代来源','#A78BFA'),
 ('has_property','属性','directed',NULL,1,0,0,0,12,'具有属性','#0EA5E9'),
 ('can_do','属性','directed',NULL,1,0,0,0,12,'能力/典型行为','#38BDF8'),
 ('causes','事件','directed','caused_by',1,0,0,0,12,'致使','#F59E0B'),
 ('caused_by','事件','directed','causes',1,0,0,1,12,'被导致','#F59E0B'),
 ('entails','事件','directed',NULL,1,0,0,0,12,'蕴含','#FB923C'),
 ('presupposes','事件','directed',NULL,1,0,0,0,12,'预设','#FB923C'),
 ('precedes','事件','directed',NULL,1,0,0,0,12,'时间先于','#FBBF24'),
 ('agent','论元','directed','agent_of',0,0,0,0,4,'动词→典型施事类别','#64748B'),
 ('agent_of','论元','directed','agent',0,0,0,1,4,'名词→可选动词','#64748B'),
 ('patient','论元','directed','patient_of',0,0,0,0,4,'动词→典型受事类别','#94A3B8'),
 ('patient_of','论元','directed','patient',0,0,0,1,4,'名词→可选动词','#94A3B8'),
 ('instrument','论元','directed',NULL,0,0,0,0,4,'工具','#CBD5E1'),
 ('location_of','论元','directed','located_at',0,0,0,0,4,'位于','#22D3EE'),
 ('located_at','论元','directed','location_of',0,0,0,1,4,'地点','#22D3EE'),
 ('classifier_of','论元','directed',NULL,0,0,0,0,4,'量词→适用名词类别','#14B8A6'),
 ('collocation','论元','directed',NULL,0,0,0,0,4,'搭配','#84CC16'),
 ('topic_of','篇章','directed',NULL,1,0,0,0,12,'话题链成员','#EC4899'),
 ('anaphoric_to','篇章','directed',NULL,0,0,0,0,4,'指代','#F472B6'),
 ('domain_link','领域','directed',NULL,1,0,0,0,12,'跨领域知识链接','#6B7280');

INSERT INTO closure_policy(relation, mode, max_depth) VALUES
 ('is_a','incremental',12),
 ('instance_of','incremental',12),
 ('has_part','incremental',8),
 ('part_of','incremental',8),
 ('derived_from','incremental',12),
 ('causes','batch',8),
 ('synonym','off',1),        -- 对称关系，闭包无意义（v2.0 把它塞进 closure 是浪费）
 ('similar','off',1),
 ('opposite','off',1),
 ('domain_link','batch',8),
 -- ↓ v3.0.1 补齐：这 7 个关系 is_transitive=1 却**没有策略行**，
 --   而触发器条件写的是 `(SELECT mode ...) = 'incremental'`，
 --   缺失行 → 子查询返回 NULL → 比较为假 → **静默不产生任何闭包**。
 --   实测：连插两条 can_do 边后 closure 行数 = 0（应为 3）。
 --   原则：is_transitive=1 的关系**必须**有显式策略行（无隐式默认）。
 ('has_property','incremental',12),   -- 属性继承推理依赖闭包
 ('can_do','incremental',12),         -- 同上
 ('caused_by','batch',8),             -- 与 causes 对称，同为高扇出事件关系
 ('entails','batch',8),
 ('presupposes','batch',8),
 ('precedes','batch',8),
 ('topic_of','batch',8);

INSERT INTO contexts(context_id, name, description, domain) VALUES
 (1,'日常对话','日常生活场景','core'),
 (2,'科技','科技/电子/互联网','tech'),
 (3,'医学','医疗健康场景','medical'),
 (4,'动物学','动物分类与行为研究','animal');

-- 【模板占位符约定】
--   (a) template 里每一处 {X} 都必须在 pattern_slots 有同名行；反过来每个槽位也必须
--       出现在模板中。二者一一对应 —— 槽位只有一个真源，模板只是它的排版视图。
--   (b) 可选性**只用 pattern_slots.is_required 表达**，模板里不再写 `{X?}` 后缀
--       （v3.0 初稿同时在模板写 `?`、又在 is_required 写 0/1，属于双写冗余）。
--   (c) 不占槽位的固定功能词（把/被/吗/难道/越/非/连…）以**字面量**写在模板里，
--       其搭配限制落到 slot_constraints(kind='marker')。
INSERT INTO sentence_patterns(pattern_id, name, category, mood, template, semantics, example) VALUES
 (1,'主谓宾','基本句型','陈述','{S} {V} {O}','动作-受事','猫吃鱼'),
 (2,'主谓','基本句型','陈述','{S} {V}','不及物','他笑了'),
 (3,'主系表','基本句型','陈述','{S} 是 {C}','判断','他是学生'),
 (4,'名词谓语句','基本句型','陈述','{S} {NP}','数量/时间/天气','今天星期六'),
 (5,'主谓谓语句','话题结构','陈述','{Topic} {S} {V}','整体-部分','他身体很好'),
 (6,'双宾句','特殊句式','陈述','{A} {V} {IO} {DO}','给予','给我一本书'),
 (7,'把字句','特殊句式','陈述','{A} 把 {P} {V} {C}','处置/致使','我把作业写完了'),
 (8,'被字句','特殊句式','陈述','{P} 被 {A} {V} {C}','被动/受损','鱼被猫吃了'),
 (9,'连动句','特殊句式','陈述','{S} {V1} {O1} {V2} {O2}','先后动作','他去图书馆看书'),
 (10,'兼语句','特殊句式','陈述','{A} {V1} {B} {V2}','使令/称谓','老师让他回答问题'),
 (11,'存现句','特殊句式','陈述','{L} {V} {NP}','存在/出现/消失','桌上放着一本书'),
 (12,'是...的句','特殊句式','陈述','{A} 是 {B} 的','强调','他是昨天来的'),
 (13,'比较句-比','特殊句式','陈述','{A} 比 {B} {Adj}','差比','他比我高'),
 (14,'比较句-等同','特殊句式','陈述','{A} 跟 {B} 一样 {Adj}','等同','他跟我一样高'),
 (15,'比较句-不如','特殊句式','陈述','{A} 不如 {B} {Adj}','不及','我不如他'),
 (16,'比较句-越来越','特殊句式','陈述','{A} 越来越 {Adj}','递进变化','天气越来越冷'),
 (17,'动结式','短语结构','陈述','{V} {R}','结果','看完/听懂'),
 (18,'动趋式','短语结构','陈述','{V} {D}','趋向','走出来'),
 (19,'可能补语式','短语结构','陈述','{V} 得 {C} / {V} 不 {C}','可能','看得完/看不懂'),
 (20,'是非问','句类','疑问','{S} {V} 吗','疑问-是非','你去吗'),
 (21,'特指问','句类','疑问','{Wh} {V}','疑问-特指','谁来了'),
 (22,'正反问','句类','疑问','{V} 不 {V}','疑问-正反','你去不去'),
 (23,'选择问','句类','疑问','{A} 还是 {B}','疑问-选择','你去还是他去'),
 (24,'反问句','句类','疑问','难道 {S} {V} 吗','否定断言','难道你不知道吗'),
 (25,'祈使句','句类','祈使','请/别 {V}','命令/请求','请坐下'),
 (26,'感叹句','句类','感叹','多么 {Adj} 啊','感叹','多么美啊'),
 (27,'紧缩句-一...就','紧缩句','陈述','一 {V1} 就 {V2}','紧接','他一说我就懂'),
 (28,'紧缩句-越...越','紧缩句','陈述','越 {A1} 越 {A2}','共变','越冷越想吃'),
 (29,'紧缩句-非...不','紧缩句','陈述','非 {X} 不 {V}','必要条件','非他不可'),
 (30,'倒装句','特殊句式','陈述','{P} {S}','强调','怎么了，你'),
 (31,'连...都句','特殊句式','陈述','连 {X} 都 {V}','极端举例','连他都不知道'),
 (32,'处置感叹式','特殊句式','感叹','{S} 把 {P} {V} 得 {C}','程度','把他气得不行');

/* 槽位真源：唯一的槽位定义处（v2.0 的 required_slots / optional_slots JSON 已删）。
 *   · required / optional 只由 is_required 派生，模板里不再写 `?` 后缀
 *   · 32 个句式在此**全部**有槽位行：
 *     v2.0 槽位只活在 JSON 字符串里，子表是死表；
 *     v3.0 初稿也只覆盖 22 个，缺 10 个（句类 / 比较句变体 / 紧缩句 / 倒装）——已补齐。 */
INSERT INTO pattern_slots(pattern_id, slot_name, position, is_required, syntactic_function, default_role) VALUES
 (1,'S',1,1,'主语','施事'), (1,'V',2,1,'谓语',NULL), (1,'O',3,1,'宾语','受事'),
 (2,'S',1,1,'主语','施事'), (2,'V',2,1,'谓语',NULL),
 (3,'S',1,1,'主语','系事'), (3,'C',2,1,'宾语','系事'),
 (4,'S',1,1,'主语','话题'), (4,'NP',2,1,'谓语',NULL),
 (5,'Topic',1,1,'话题','话题'), (5,'S',2,1,'主语','系事'), (5,'V',3,1,'谓语',NULL),
 (6,'A',1,1,'主语','施事'), (6,'V',2,1,'谓语',NULL), (6,'IO',3,1,'宾语','与事'), (6,'DO',4,1,'宾语','受事'),
 (7,'A',1,1,'主语','施事'), (7,'P',2,1,'状语','受事'), (7,'V',3,1,'谓语',NULL), (7,'C',4,0,'补语','效果'),
 (8,'P',1,1,'主语','受事'), (8,'A',2,0,'状语','施事'), (8,'V',3,1,'谓语',NULL), (8,'C',4,0,'补语','效果'),
 (9,'S',1,1,'主语','施事'), (9,'V1',2,1,'谓语',NULL), (9,'O1',3,0,'宾语','受事'), (9,'V2',4,0,'谓语',NULL), (9,'O2',5,0,'宾语','受事'),
 (10,'A',1,1,'主语','施事'), (10,'V1',2,1,'谓语',NULL), (10,'B',3,1,'宾语','受事'), (10,'V2',4,1,'谓语',NULL),
 (11,'L',1,1,'状语','处所'), (11,'V',2,1,'谓语',NULL), (11,'NP',3,1,'宾语','结果'),
 (12,'A',1,1,'主语','话题'), (12,'B',2,1,'宾语','焦点'),
 (13,'A',1,1,'主语','话题'), (13,'B',2,1,'状语','比较对象'), (13,'Adj',3,1,'谓语',NULL),
 (14,'A',1,1,'主语','话题'), (14,'B',2,1,'状语','比较对象'), (14,'Adj',3,1,'谓语',NULL),
 (15,'A',1,1,'主语','话题'), (15,'B',2,1,'宾语','比较对象'), (15,'Adj',3,1,'谓语',NULL),
 (16,'A',1,1,'主语','话题'), (16,'Adj',2,1,'谓语',NULL),
 (17,'V',1,1,'谓语',NULL), (17,'R',2,1,'补语','效果'),
 (18,'V',1,1,'谓语',NULL), (18,'D',2,1,'补语','方向'),
 (19,'V',1,1,'谓语',NULL), (19,'C',2,1,'补语','效果'),
 (20,'S',1,1,'主语','话题'), (20,'V',2,1,'谓语',NULL),
 (21,'Wh',1,1,'主语',NULL), (21,'V',2,1,'谓语',NULL),
 (22,'V',1,1,'谓语',NULL),
 (23,'A',1,1,'话题',NULL), (23,'B',2,1,'谓语',NULL),
 (24,'S',1,1,'主语','施事'), (24,'V',2,1,'谓语',NULL),
 (25,'V',1,1,'谓语',NULL),
 (26,'Adj',1,1,'谓语',NULL),
 (27,'V1',1,1,'谓语',NULL), (27,'V2',2,1,'谓语',NULL),
 (28,'A1',1,1,'谓语',NULL), (28,'A2',2,1,'谓语',NULL),
 (29,'X',1,1,'宾语',NULL), (29,'V',2,1,'谓语',NULL),
 (30,'P',1,1,'谓语','焦点'), (30,'S',2,1,'主语','施事'),
 (31,'X',1,1,'宾语','焦点'), (31,'V',2,1,'谓语',NULL),
 (32,'S',1,1,'主语','施事'), (32,'P',2,1,'状语','受事'), (32,'V',3,1,'谓语',NULL), (32,'C',4,1,'补语','程度');

INSERT INTO slot_constraints(pattern_id, slot_name, kind, value) VALUES
 (7,'P','definiteness','definite'),          -- 把字句宾语必须有定
 (8,'A','feature','+animate'),               -- 被字句施事通常有生
 (11,'L','pos','nl'),
 (11,'V','subcategory','存现'),
 (13,'Adj','pos','a'),
 (16,'A','pos','n'),
 (6,'V','subcategory','给予'),
 (10,'V1','subcategory','使令'),
 (19,'C','lexeme','完'),
 (29,'X','marker','非'),          -- 非...不 的"非"标记必现
 (14,'Adj','pos','a'),
 (15,'Adj','pos','a'),
 (21,'Wh','pos','r'),                       -- 疑问代词词性
 (26,'Adj','pos','a'),
 (32,'P','definiteness','definite'),        -- 处置式/处置感叹式的 P 必须有定
 (32,'C','pos','a');                        -- "得"后补语常为形容词性

INSERT INTO reduplication_patterns(pattern_id, name, pos, form_schema, semantics, example) VALUES
 (1,'AA','a','A -> AA','程度加强/亲昵','高高(的)/红红(的)'),
 (2,'AABB','a','AB -> AABB','程度加强/生动','高高兴兴/干干净净'),
 (3,'ABAB','v','AB -> ABAB','短时/尝试','研究研究/商量商量'),
 (4,'A一A','v','A -> A一A','短时/轻微','看一看/想一想'),
 (5,'A了A','v','A -> A了A','短时(已发生)','看了看'),
 (6,'ABB','a','AB -> ABB(A+BB)','状态描写','红彤彤/绿油油'),
 (7,'量词重叠','q','M -> MM','遍指/每一','个个/条条'),
 (8,'一MM','q','M -> 一MM','逐一','一个个/一张张'),
 (9,'AA(量词)','q','M -> MM','动量遍指','次次/回回'),
 (10,'A是A','v','A -> A是A','让步(形容词性)','好是好');

INSERT INTO inference_rules(rule_id, name, rule_type, description, condition_sql, priority) VALUES
 (1,'is_a传递','transitive','若 A is_a B 且 B is_a C，则 A is_a C',
  'SELECT 1 FROM relations WHERE src_id=:a AND relation=''is_a'' AND dst_id=:b
   UNION ALL SELECT 1 FROM relations WHERE src_id=:b AND relation=''is_a'' AND dst_id=:c',10),
 (2,'属性继承','inherit','子类继承上位类的 has_property',
  'SELECT r.dst_id FROM relations r
    WHERE r.relation=''has_property''
      AND r.src_id IN (SELECT ancestor FROM closure WHERE relation=''is_a'' AND descendant=:a)',9),
 (3,'施事合法性','constraint','动词的施事槽允许主体类型则通过',
  'SELECT 1 FROM relation_constraints rc
    WHERE rc.sense_id=:verb AND rc.slot=''agent''
      AND rc.value IN (SELECT cat FROM sense_categories WHERE sense_id=:subject_type)',8),
 (4,'受事合法性','constraint','动词的受事槽允许客体类型则通过',
  'SELECT 1 FROM relation_constraints rc
    WHERE rc.sense_id=:verb AND rc.slot=''patient''
      AND rc.value IN (SELECT cat FROM sense_categories WHERE sense_id=:object_type)',8),
 (5,'反义对称','symmetric','A opposite B ⇒ B opposite A',
  'SELECT 1 FROM relations WHERE src_id=:a AND relation=''opposite'' AND dst_id=:b',7),
 (6,'量词默认','default','名词无显式量词搭配时按语义类别取默认量词',
  'SELECT classifier_sense_id FROM classifier_class_defaults
    WHERE cat IN (SELECT cat FROM sense_categories WHERE sense_id=:noun) ORDER BY preference DESC',6);

INSERT INTO rule_premises(rule_id, seq, kind, subject, predicate, object) VALUES
 (1,1,'edge','A','is_a','B'),
 (1,2,'edge','B','is_a','C'),
 (2,1,'closure','A','is_a','Ancestor'),
 (2,2,'edge','Ancestor','has_property','P'),
 (5,1,'edge','A','opposite','B');

INSERT INTO rule_conclusions(rule_id, seq, kind, subject, relation, object, confidence) VALUES
 (1,1,'edge','A','is_a','C',1.0),
 (2,1,'edge','A','has_property','P',0.9),
 (5,1,'edge','B','opposite','A',1.0);

INSERT INTO ellipsis_rules(context_type, slot_role, strategy, priority, max_distance, description) VALUES
 ('dialogue','施事','speaker',10,0,'对话中主语默认是说话人'),
 ('dialogue','施事','previous_subject',9,3,'沿用上一句主语'),
 ('dialogue','受事','previous_mentioned',8,3,'宾语省略回指上文刚提到的实体'),
 ('dialogue','受事','topic',7,5,'话题链中的话题'),
 ('narrative','施事','previous_subject',10,5,'叙事中主语链维持'),
 ('narrative','施事','topic',9,8,'话题链延续'),
 ('narrative','受事','previous_mentioned',6,3,'回指上文实体'),
 ('imperative','施事','listener',10,0,'祈使句主语是听话人'),
 ('topic_chain','施事','topic',10,10,'话题链中零形主语回指话题'),
 ('topic_chain','受事','salient_entity',5,5,'显著实体'),
 ('expository','施事','indefinite_default',1,0,'说明文中默认泛指'),
 ('narrative','时点','world_knowledge',1,0,'默认理解为叙事当前时点');

INSERT INTO morphemes(morpheme_id, form, morpheme_type, is_bound, meaning) VALUES
 (1,'葡萄','联绵语素',1,'一种水果(整体记音，不可拆分)'),
 (2,'儿','词尾',1,'名词化/小称标记'),
 (3,'老','前缀',1,'称谓前加成分'),
 (4,'子','后缀',1,'名词化后缀'),
 (5,'们','后缀',1,'复数标记'),
 (6,'第','前缀',1,'序数标记'),
 (7,'洗','自由语素',0,'用水去污'),
 (8,'澡','黏着语素',1,'身体清洁活动(不单用)'),
 (9,'高','自由语素',0,'从下到上距离大'),
 (10,'虎','自由语素',0,'大型猫科猛兽'),
 (11,'花','自由语素',0,'种子植物的繁殖器官'),
 (12,'桌','黏着语素',1,'家具(不单用，需带"子")'),
 (13,'兴','黏着语素',1,'兴致(不单用)');

INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 (1,'洗澡',1,'合成词',0,'通用'),
 (2,'猫',NULL,'单纯词',0,'通用'),
 (3,'只',NULL,'单纯词',0,'通用'),
 (4,'把',NULL,'单纯词',0,'通用'),
 (5,'高',NULL,'单纯词',0,'通用'),
 (6,'兴',NULL,'单纯词',1,'通用'),
 (7,'高兴',NULL,'合成词',0,'通用'),
 (8,'葡萄',NULL,'联绵词',0,'通用'),
 (9,'桌子',NULL,'派生词',0,'通用'),
 (10,'打算',NULL,'合成词',0,'通用'),
 (11,'动物',NULL,'合成词',0,'通用'),
 (12,'老虎',NULL,'派生词',0,'通用'),
 (13,'花',NULL,'单纯词',0,'通用'),
 (14,'花儿',NULL,'派生词',0,'口语'),
 (15,'研究',NULL,'合成词',0,'通用'),
 (16,'完成',NULL,'合成词',0,'通用'),
 (17,'看',NULL,'单纯词',0,'通用'),
 (18,'说',NULL,'单纯词',0,'通用'),
 (19,'给',NULL,'单纯词',0,'通用'),
 (20,'书',NULL,'单纯词',0,'通用'),
 (21,'他',NULL,'单纯词',0,'通用'),
 (22,'澡',NULL,'单纯词',1,'通用'),
 (23,'洗',NULL,'单纯词',0,'通用'),
 (24,'完',NULL,'单纯词',0,'通用'),
 (25,'作业',NULL,'合成词',0,'通用'),
 (26,'鱼',NULL,'单纯词',0,'通用');

INSERT INTO word_morphemes(word_id, seq, morpheme_id, surface) VALUES
 (1,1,7,'洗'), (1,2,8,'澡'),          -- 复合式合成词
 (8,1,1,'葡萄'),                       -- 联绵词：两字一语素
 (9,1,12,'桌'), (9,2,4,'子'),          -- 附加式：词根+后缀
 (12,1,3,'老'), (12,2,10,'虎'),        -- 附加式：前缀+词根
 (14,1,11,'花'), (14,2,2,'儿'),        -- 儿化：词根+词尾
 (7,1,9,'高'), (7,2,13,'兴');          -- 复合式：高+兴

INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg, pinyin, pinyin_plain, pinyin_initial, frequency, literalness, subcategory, is_primary) VALUES
 (1,1,'v',1,'用水洗身体或其一部分','用水 洗 身体 或 其 一部分','xi3zao3','xizao','xz',980,'本义',NULL,1),
 (2,2,'n',1,'哺乳纲猫科小型家养动物，善捕鼠','哺乳纲 猫科 小型 家养 动物 善 捕鼠','mao1','mao','m',1500,'本义',NULL,1),
 (3,3,'q',1,'用于计量某些动物的量词','用 于 计量 某些 动物 的 量词','zhi1','zhi','zh',1400,NULL,NULL,1),
 (4,4,'p',1,'介词，引出被处置的受事，与后面的动词短语构成处置式','介词 引出 被 处置 的 受事 与 后面 的 动词 短语 构成 处置式','ba3','ba','b',2100,NULL,NULL,1),
 (5,4,'q',1,'用于有柄或有把手的器物','用 于 有柄 或 有 把手 的 器物','ba3','ba','b',900,NULL,NULL,0),
 (6,4,'v',1,'握住、抓住','握住 抓住','ba3','ba','b',700,'本义',NULL,0),
 (7,5,'a',1,'从下向上的距离大','从下 向上 的 距离 大','gao1','gao','g',1300,'本义',NULL,1),
 (8,7,'a',1,'心情愉快','心情 愉快','gao1xing4','gaoxing','gx',1200,'引申义',NULL,1),
 (9,8,'n',1,'一种藤本植物，果实成串，可鲜食或制干','一种 藤本 植物 果实 成串 可 鲜食 或 制干','pu2tao2','putao','pt',800,'本义',NULL,1),
 (10,9,'n',1,'有腿有面的家具，用于放置物品','有 腿 有 面 的 家具 用于 放置 物品','zhuo1zi0','zhuozi','zz',1100,'本义',NULL,1),
 (11,10,'v',1,'计划、考虑要做某事','计划 考虑 要 做 某事','da3suan4','dasuan','ds',1000,'本义',NULL,1),
 (12,11,'n',1,'生物的一大类，能自由运动、摄取有机物为食','生物 的 一大类 能 自由 运动 摄取 有机物 为食','dong4wu4','dongwu','dw',1600,'本义',NULL,1),
 (13,12,'n',1,'哺乳纲猫科大型猛兽，虎的通称','哺乳纲 猫科 大型 猛兽 虎 的 通称','lao3hu3','laohu','lh',700,'本义',NULL,1),
 (14,13,'n',1,'种子植物的繁殖器官，多有艳丽颜色和香味','种子 植物 的 繁殖 器官 多 有 艳丽 颜色 和 香味','hua1','hua','h',1500,'本义',NULL,1),
 (15,14,'n',1,'花的儿化形式，口语中常指小的花或花朵的统称','花 的 儿化 形式 口语 中 常 指 小 的 花 或 花朵 的 统称','hua1r','huar','hr',400,'本义',NULL,1),
 (16,15,'v',1,'深入探求事物的真相、性质、规律','深入 探求 事物 的 真相 性质 规律','yan2jiu1','yanjiu','yj',1400,'本义',NULL,1),
 (17,16,'v',1,'事情做完、达到预期结果','事情 做完 达到 预期 结果','wan2cheng2','wancheng','wc',1500,'本义',NULL,1),
 (18,17,'v',1,'使视线接触人或物','使 视线 接触 人 或 物','kan4','kan','k',3000,'本义',NULL,1),
 (19,18,'v',1,'用语言表达意思','用 语言 表达 意思','shuo1','shuo','sh',2900,'本义',NULL,1),
 (20,19,'v',1,'使对方得到某物','使 对方 得到 某物','gei3','gei','g',2800,'本义','给予',1),
 (21,20,'n',1,'装订成册的著作','装订 成册 的 著作','shu1','shu','sh',2600,'本义',NULL,1),
 (22,21,'r',1,'第三人称代词，指称男性或泛指第三人','第三人称 代词 指称 男性 或 泛指 第三人','ta1','ta','t',5000,'本义',NULL,1),
 (23,22,'n',1,'身体上需要清洗的部分，不单用，只能作"洗"的黏着宾语','身体 上 需要 清洗 的 部分 不 单用 只能 作 洗 的 黏着 宾语','zao3','zao','z',600,'本义',NULL,1),
 (24,23,'v',1,'用水或溶剂去掉物体上的污垢','用水 或 溶剂 去掉 物体 上 的 污垢','xi3','xi','x',1900,'本义',NULL,1),
 (25,24,'v',1,'事情结束、达到终了','事情 结束 达到 终了','wan2','wan','w',2400,'本义',NULL,1),
 (26,25,'n',1,'教师布置的学习任务','教师 布置 的 学习 任务','zuo4ye4','zuoye','zy',900,'本义',NULL,1),
 (27,26,'n',1,'生活在水中的脊椎动物，用鳃呼吸','生活 在 水中 的 脊椎 动物 用 鳃 呼吸','yu2','yu','y',1200,'本义',NULL,1);

INSERT INTO sense_features(sense_id, feature, value) VALUES
 (1,'+animate','1'), (1,'+volitional','1'),
 (2,'+animate','1'), (2,'+countable','1'), (2,'+concrete','1'),
 (7,'polarity','positive'),
 (12,'+animate','1'), (12,'+countable','0'),
 (13,'+animate','1'), (13,'+countable','1'),
 (14,'+concrete','1'), (14,'+countable','1'),
 (21,'+concrete','1'), (21,'+countable','1');

INSERT INTO sense_categories(sense_id, cat, is_primary) VALUES
 (2,'动物',1), (2,'人工物',0),
 (12,'动物',1), (13,'动物',1), (14,'植物',1), (21,'人工物',1),
 (23,'抽象',1), (25,'人工物',1), (27,'动物',1),
 (1,'事件',1), (16,'事件',1), (17,'事件',1), (24,'事件',1);

INSERT INTO function_words(sense_id, category, subcategory, grammatical_function, attaches_to, is_sentence_final) VALUES
 (4,'介词','处置','引出受事，构成把字句','介词短语作状语，位于动词前',0);

INSERT INTO function_word_roles(sense_id, role, position) VALUES
 (4,'受事','前置');

INSERT INTO classifiers(classifier_sense_id, kind, subtype, is_borrowed, has_reduplication) VALUES
 (3,'名量词','个体',0,1),
 (5,'名量词','个体',0,1);

INSERT INTO noun_classifier_pairs(noun_sense_id, classifier_sense_id, preference, is_default, source_id) VALUES
 (2,3,0.9,1,5),
 (13,3,0.85,0,5);

INSERT INTO classifier_class_defaults(cat, classifier_sense_id, preference) VALUES
 ('动物',3,0.7);

INSERT INTO word_reduplications(word_id, pattern_id, surface, derived_word_id) VALUES
 (5,1,'高高',NULL),                   -- 形容词 AA 式：程度加强
 (7,2,'高高兴兴',NULL),               -- AABB 式：生动化
 (15,3,'研究研究',NULL),              -- ABAB 式：短时/尝试
 (18,4,'看一看',NULL);                -- A一A 式：短时/轻微

INSERT INTO separable_patterns(word_id, kind, infixes_json, surface_example, note) VALUES
 (1,'插入体标记','["了","过","着"]','洗了个澡','体标记插入离合点之后'),
 (1,'插入数量','["个","次"]','洗了个澡','数量成分插入'),
 (1,'插入定语','["热水","冷水"]','洗了个热水澡','定语修饰黏着宾语'),
 (1,'插入补语','["完","好"]','洗完澡','补语紧随动词'),
 (1,'重叠','["AAB"]','洗洗澡','重叠式表短时');

INSERT INTO bound_objects(object_word_id, host_verb_id) VALUES
 (22,23);   -- '澡'(word 22) 只能作 '洗'(word 23) 的黏着宾语 → 洗+澡 = 离合词

INSERT INTO complements(comp_type, head_sense_id, comp_sense_id, result_state, surface_order, example) VALUES
 ('结果补语',18,25,'达成','V+C','看完'),
 ('结果补语',24,25,'完成','V+C','洗完'),
 ('可能补语',18,25,'可能','V+得+C','看得完'),
 ('可能补语',18,25,'不可能','V+不+C','看不懂');

INSERT INTO relation_constraints(sense_id, slot, kind, value) VALUES
 (1,'agent','category','人'),
 (1,'patient','category','人'),
 (16,'agent','category','人'),
 (16,'patient','category','事件'),
 (19,'agent','category','人'),
 (19,'patient','feature','+animate'),
 (18,'agent','category','人'),
 (18,'patient','feature','+concrete'),
 (19,'patient','feature','+concrete');

INSERT INTO utterances(utterance_id, doc_id, seq, text, speaker, mood, pattern_id) VALUES
 (1,'demo',1,'我把作业写完了。','A','陈述',7),
 (2,'demo',2,'鱼被猫吃了。','A','陈述',8),
 (3,'demo',3,'？你去不去？','B','疑问',22);

INSERT INTO topic_chains(chain_id, doc_id, topic_node_id, start_seq, end_seq, chain_type) VALUES
 (1,'demo',22,1,3,'话题链');

INSERT INTO anaphora(utterance_id, slot_role, form, form_type, antecedent_utterance_id, antecedent_node_id, chain_id, confidence) VALUES
 (2,'施事','猫','definite_np',1,2,1,0.9),
 (3,'施事','∅','zero',2,22,1,0.7);

INSERT INTO sense_contexts(sense_id, context_id, relevance, is_primary) VALUES
 (1,1,0.9,1),
 (2,4,0.95,1),
 (12,4,1.0,1),
 (16,2,0.6,0);

INSERT INTO import_batches(batch_id, source_id, status, row_count, finished_at) VALUES
 (1,5,'ok',128,'2026-09-27T00:00:00.000Z');


/* =========================================================================
 * 15. 闭包演示与常用查询（非 schema 的一部分，供应用层/测试调用）
 * ========================================================================= */

-- 15.1 启用增量模式（默认 batch）
-- UPDATE closure_policy SET mode = 'incremental' WHERE relation IN ('is_a','part_of');

-- 15.2 手工触发某关系的全量重建
-- INSERT OR IGNORE INTO closure_rebuild_queue(relation) VALUES ('is_a');

-- 15.3 在线推导祖先（不依赖物化闭包的兜底路径，带环保护与深度上限）
-- WITH RECURSIVE anc(id, depth) AS (
--     SELECT :start, 0
--     UNION
--     SELECT r.dst_id, anc.depth + 1
--       FROM anc JOIN relations r ON r.src_id = anc.id
--      WHERE r.relation = 'is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
--        AND anc.depth < 12
-- )
-- SELECT s.sense_id, w.word, MIN(anc.depth) AS depth
--   FROM anc JOIN senses s ON s.sense_id = anc.id JOIN words w ON w.word_id = s.word_id
--  WHERE anc.id <> :start GROUP BY anc.id ORDER BY depth;

-- 15.4 中文释义全文检索（>=3 字用 trigram）
-- SELECT s.sense_id, w.word, s.definition, bm25(senses_fts_trigram) AS score
--   FROM senses_fts_trigram f JOIN senses s ON s.sense_id = f.rowid
--   JOIN words w ON w.word_id = s.word_id
--  WHERE senses_fts_trigram MATCH :q ORDER BY score LIMIT 20;

-- 15.5 1-2 字词用预分词通道
-- SELECT s.sense_id, w.word, s.definition
--   FROM senses_fts_seg f JOIN senses s ON s.sense_id = f.rowid
--   JOIN words w ON w.word_id = s.word_id
--  WHERE senses_fts_seg MATCH :seg_token LIMIT 20;

-- 15.6 检字：按部首与笔画
-- SELECT char, strokes FROM characters WHERE radical = :r ORDER BY strokes;

-- 15.7 量词推荐：先查显式搭配，再回退类别默认
-- SELECT cl.classifier_sense_id, cw.word AS classifier, p.preference, 'explicit' AS src
--   FROM noun_classifier_pairs p
--   JOIN classifiers cl ON cl.classifier_sense_id = p.classifier_sense_id
--   JOIN senses cs ON cs.sense_id = cl.classifier_sense_id
--   JOIN words cw ON cw.word_id = cs.word_id
--  WHERE p.noun_sense_id = :noun
-- UNION ALL
-- SELECT d.classifier_sense_id, cw2.word, d.preference, 'class_default'
--   FROM classifier_class_defaults d
--   JOIN sense_categories sc ON sc.cat = d.cat
--   JOIN classifiers cl2 ON cl2.classifier_sense_id = d.classifier_sense_id
--   JOIN senses cs2 ON cs2.sense_id = cl2.classifier_sense_id
--   JOIN words cw2 ON cw2.word_id = cs2.word_id
--  WHERE sc.sense_id = :noun
--  ORDER BY 3 DESC;

-- 15.8 离线一致性巡检（建议纳入 CI；以下每句正常应返回 0 行）
-- (1) 悬空的语义角色父级
-- SELECT role, parent_role FROM semantic_roles
--  WHERE parent_role IS NOT NULL AND parent_role NOT IN (SELECT role FROM semantic_roles);
-- (2) 逆向声明不成对
-- SELECT a.relation, a.inverse_of FROM relation_types a
--  WHERE a.inverse_of IS NOT NULL
--    AND NOT EXISTS (SELECT 1 FROM relation_types b
--                     WHERE b.relation = a.inverse_of AND b.inverse_of = a.relation);
-- (3) is_a 环（会造成推理链永不收敛）
-- WITH RECURSIVE cyc(id, path, closed) AS (
--     SELECT src_id, ',' || src_id || ',', 0 FROM relations WHERE relation='is_a'
--     UNION ALL
--     SELECT r.dst_id, cyc.path || r.dst_id || ',',
--            instr(cyc.path, ',' || r.dst_id || ',') > 0
--       FROM cyc JOIN relations r ON r.src_id = (SELECT CAST(substr(cyc.path,
--                 length(cyc.path) - length(CAST(cyc.id AS TEXT)), length(CAST(cyc.id AS TEXT))) AS INTEGER))
--      WHERE r.relation='is_a' AND cyc.closed = 0
-- ) SELECT * FROM cyc WHERE closed = 1;
-- (4) 闭包与实时可达性不一致（抽样）
-- SELECT c.ancestor, c.descendant FROM closure c
--  WHERE c.relation='is_a' AND NOT EXISTS (
--        SELECT 1 FROM relations r WHERE r.src_id=c.ancestor AND r.relation='is_a' AND r.dst_id=c.descendant)
--    AND c.depth > 1;
-- (5) 量词搭配指向不存在的量词义项（若曾手工改过 id）
-- SELECT * FROM noun_classifier_pairs p
--  WHERE NOT EXISTS (SELECT 1 FROM classifiers c WHERE c.classifier_sense_id = p.classifier_sense_id);
-- (6) 离合词的 separator_pos 是否落在合理范围
-- SELECT word, separator_pos, char_count FROM words
--  WHERE is_separable=1 AND (separator_pos < 1 OR separator_pos >= char_count);

-- 15.9 建议的维护节奏
-- 每次连接:      PRAGMA foreign_keys=ON; PRAGMA recursive_triggers=ON;
-- 批量写入前:    UPDATE closure_policy SET mode='off' WHERE 1;   -- 或 'batch'
-- 批量写入后:    INSERT OR REPLACE INTO closure_rebuild_queue(relation, requested_at)
--                    SELECT relation, strftime('%Y-%m-%dT%H:%M:%fZ','now')
--                      FROM closure_policy WHERE mode <> 'off';
--                UPDATE closure_policy SET mode='incremental' WHERE relation IN ('is_a','part_of');
-- 每日:          PRAGMA optimize;  ANALYZE;
-- 每周:          INSERT INTO senses_fts_trigram(senses_fts_trigram) VALUES('rebuild');
--                INSERT INTO senses_fts_seg(senses_fts_seg) VALUES('rebuild');
/*****************************************************************************
 * End of schema v3.0
 *****************************************************************************/
