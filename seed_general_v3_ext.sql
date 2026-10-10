/*****************************************************************************
 * Logos 通用领域种子数据（扩展）  seed_general_v3_ext.sql   (v0.1 demo)
 * 中文原生符号语义数据库 —— 多继承补全 + 抽象/人造物分类骨架
 *
 * 目标一：让「鱼」在**生物分类**与**功能分类**两个维度上各自成立（DAG 多继承）
 *         鱼 → 动物 → 生物
 *         鱼 → 食物 → 营养物
 * 目标二：补一层「抽象概念 / 人造物」分类骨架，使科技与信息类词条有上位可挂
 *
 * ---------------------------------------------------------------------------
 * 【前置依赖】必须按序执行：schema_v3.sql → seed_general_v3.sql → 本文件。
 *   理由：本文件是**叠加式增量种子**，不重建任何表。
 *     · 复用 schema §14 建好的「动物(sense 12) / 鱼(sense 27) / 书(sense 21)」与
 *       seed §2 建好的「食物(sense 1025) / 生物(sense 1026)」，一律按 (word, pos)
 *       自然键子查询引用，因此对既有 ID 取值不敏感；
 *     · 同时用 NOT EXISTS 兜底：若在**纯 schema** 上单独执行本文件（无 seed），
 *       「食物 / 生物 / 食物类别」会被就地补建（取 2002/2003 段），不会报错。
 *
 * 【幂等性】全部 INSERT 均带 ON CONFLICT DO NOTHING 或 WHERE NOT EXISTS，
 *   可重复执行任意次，结果一致（不产生重复行、不重复写 FTS 与 changelog）。
 *
 * 【ID 分配策略】本文件独占 2000-2999 段，与 schema 的 1-27、seed 的 1000-1027
 *   完全隔离，杜绝主键冲撞：
 *     2001 营养物  2002/2003 食物·生物（仅无 seed 时的兜底）
 *     2004 可食用  2005 有营养
 *     2101-2135 分类骨架（35 个义项）      2201 量词「份」
 *     3001 动物学库领域实体节点（cross_links 用）
 *
 * ---------------------------------------------------------------------------
 * 【写入规模】（下表为实测值：schema → seed → 本文件 的**增量**）
 *   semantic_categories    +0   食物已由 seed §1 建好 → 本节为 DO NOTHING 空操作
 *                               （纯 schema 无 seed 时才 +1，见 1.1 注释）
 *   words                 +39   营养物 1 + 可食用/有营养 2 + 骨架 35 + 量词 份 1
 *   senses                +39   与上表 1:1（每词单义项，v0.1 不做多义）
 *   sense_features        +42   +countable 35（骨架全部）+ +concrete 7（实体载体）
 *   sense_categories      +38   营养物 1 + 属性节点 2 + 抽象 14 / 人工物 18 / 器物 3
 *   classifiers            +1   份（名量词 / 部分）
 *   noun_classifier_pairs  +4   视图→个 / 文档→份 / 程序→个 / 文件→份
 *   relation_constraints   +6   算法 / 文档 / 程序 各 2
 *                               （吃.patient=食物 已由 seed §10 建好 → DO NOTHING）
 *   relations             +38   34 is_a（骨架）+ 1 is_a（食物→营养物，任务一新增）
 *                                   + 2 has_property（食物属性）+ 1 domain_link
 *                               （鱼→动物 / 动物→生物 / 鱼→食物 已由 seed §9.1 建好）
 *   entity_nodes           +1   动物学库「Fish」条目（供 cross_links 视图读取）
 *   closure               +79   全部**自动派生**：is_a 与 has_property 均为
 *                               incremental 策略，由 tr_closure_edge_ai 写出，
 *                               本文件不含任何 closure 的 INSERT
 *   changelog             +38   由 tr_relations_audit_ai 记录，同样自动产生
 *   合计：seed 后 words/senses = 53/54 → 92/93；relations = 23 → 61；closure = 16 → 95
 *
 * 【依赖顺序】（严格按此顺序，不可调换）
 *   semantic_categories ─────────────────────────┐
 *   words → senses ─┬─→ sense_categories ───────┴─→ sense_features
 *                   ├─→ classifiers → noun_classifier_pairs
 *                   ├─→ relations ─┬─→ 触发 tr_closure_edge_ai（is_a / has_property）
 *                   │              └─→ entity_nodes → relations(node 端点)
 *                   └─→ relation_constraints
 *
 * 【两处刻意的设计取舍 —— 请留意】
 *   (1) 需求原文写「食物 has_property 可食用 / 有营养来源」。本文件把这两条属性
 *       **实体化为义项节点**「可食用(a) / 有营养(a)」，因为 schema 里 has_property
 *       两端都是 sense，不建节点就落不了边。
 *       这与 seed §10 的相反原则并不矛盾，二者分工不同：
 *         · 论元需求（吃需要受事）→ relation_constraints + relation_types.agent/patient
 *           （seed §10 明示：把"需要施事"实体化成伪名词是坏设计）
 *         · 事物属性（食物可食用）→ has_property（本条，属性本身就是可命名的概念）
 *       「有营养来源」按汉语习惯实现为属性「有营养」——"来源"是释义成分而非属性名；
 *       若需严格保留字面，只需把本文件 §1.5 的 word 值改成 '有营养来源'，其余无需改动。
 *   (2) 「抽象概念 / 人造物」在本文件里是**义项层节点**，不是 semantic_categories
 *       的新增行。理由：类别层（ontology skeleton，供量词默认值共用）与义项层
 *       is_a 关系是两套正交体系 —— schema §1 注释已明示。骨架要能被 is_a 遍历与
 *       closure 物化，就必须落在义项层；类别层只挂一个已有的「抽象 / 人工物」。
 *
 * 【连接级 PRAGMA】必须在每个连接上执行，不随 schema 持久化。
 *   注意：PRAGMA foreign_keys 在**显式事务内是 no-op**，故写在 BEGIN 之前。
 * =========================================================================*/

PRAGMA foreign_keys     = ON;   -- SQLite 默认 OFF；不开则所有 FK 形同虚设
PRAGMA recursive_triggers = ON; -- 闭包触发器依赖触发器级联语义

BEGIN;

/* =========================================================================
 * 0. 自然键查找视图（仅本会话有效，不落盘）
 *    与 seed_general_v3.sql 的 seed_sid 同构，但**另起名字**：
 *    两个文件可能被同一连接先后加载，重名会让后者的 CREATE 失效
 *    （IF NOT EXISTS 命中旧定义），故此处用 ext_sid 与 seed_sid 并存。
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS ext_sid AS
    SELECT w.word AS word, s.pos AS pos, s.sense_no AS sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;

/* ###########################################################################
 * 任务一：鱼的多继承（is_a 双挂）
 * ######################################################################### */

/* =========================================================================
 * 1.1 本体骨架兜底：确保「食物」类别存在
 *     用途：作为 relation_constraints(kind='category', value='食物') 的取值依据。
 *     说明：seed_general_v3.sql §1 已建此行，常规流程下本语句是 DO NOTHING
 *           空操作；保留它是为了「纯 schema + 本文件」这条无 seed 路径也能成立。
 * ========================================================================= */
INSERT INTO semantic_categories(cat, parent_cat, description) VALUES
 ('食物','实体','可供人或动物食用的东西')
ON CONFLICT(cat) DO NOTHING;

/* =========================================================================
 * 1.2 词层兜底 —— 先查现有 senses 里有没有「食物 / 生物 / 营养物」
 *     实测（schema + seed 之后）：
 *       鱼   → word 26 / sense 27   (pos n)   ✅ 已存在
 *       动物 → word 11 / sense 12   (pos n)   ✅ 已存在
 *       食物 → word 1025 / sense 1025 (pos n) ✅ 由 seed §2 建好
 *       生物 → word 1026 / sense 1026 (pos n) ✅ 由 seed §2 建好
 *       营养物 → ✗ 不存在 → 本文件新建（2001 段）
 *     NOT EXISTS 保证：有 seed 时前两条不执行，无 seed 时就地补建。
 * ========================================================================= */
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register)
SELECT 2001,'营养物',NULL,'合成词',0,'通用'
 WHERE NOT EXISTS (SELECT 1 FROM words WHERE word='营养物');

INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register)
SELECT 2002,'食物',NULL,'合成词',0,'通用'
 WHERE NOT EXISTS (SELECT 1 FROM words WHERE word='食物');

INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register)
SELECT 2003,'生物',NULL,'合成词',0,'通用'
 WHERE NOT EXISTS (SELECT 1 FROM words WHERE word='生物');

/* =========================================================================
 * 1.3 义项层兜底 —— 与 1.2 一一对应
 *     注意：食物 / 生物的 definition 与 seed_general_v3.sql 完全一致，
 *           即使两条路径都执行，UNIQUE(word_id, pos, definition) 也不会冲突。
 * ========================================================================= */
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary)
SELECT 2001,2001,'n',1,'能供给生物养分、维持其生命活动的物质',
       '能 供给 生物 养分 维持 其 生命 活动 的 物质',
       'ying2yang3wu4','yingyangwu','yyw',1500,'本义',NULL,1
 WHERE NOT EXISTS (SELECT 1 FROM senses s JOIN words w ON w.word_id = s.word_id
                    WHERE w.word = '营养物' AND s.pos = 'n');

INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary)
SELECT 2002,2002,'n',1,'可以充饥、供人或动物食用的东西',
       '可以 充饥 供 人 或 动物 食用 的 东西',
       'shi2wu4','shiwu','shw',3400,'本义',NULL,1
 WHERE NOT EXISTS (SELECT 1 FROM senses s JOIN words w ON w.word_id = s.word_id
                    WHERE w.word = '食物' AND s.pos = 'n');

INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary)
SELECT 2003,2003,'n',1,'自然界中具有生命现象、能进行新陈代谢的物体',
       '自然界 中 具有 生命 现象 能 进行 新陈代谢 的 物体',
       'sheng1wu4','shengwu','shw',2800,'本义',NULL,1
 WHERE NOT EXISTS (SELECT 1 FROM senses s JOIN words w ON w.word_id = s.word_id
                    WHERE w.word = '生物' AND s.pos = 'n');

/* 1.4 义项 ↔ 类别（仅当该义项尚无任何类别行时才写，避免与 seed 的定义冲突） */
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT e.sid,'食物',1 FROM ext_sid e
 WHERE e.word='食物' AND e.pos='n'
   AND NOT EXISTS (SELECT 1 FROM sense_categories sc WHERE sc.sense_id = e.sid)
ON CONFLICT DO NOTHING;

INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT e.sid,'生物',1 FROM ext_sid e
 WHERE e.word='生物' AND e.pos='n'
   AND NOT EXISTS (SELECT 1 FROM sense_categories sc WHERE sc.sense_id = e.sid)
ON CONFLICT DO NOTHING;

/* 营养物：功能分类链的顶端。语义上它是"物质"，故挂「实体」而非「食物」
 *   —— 若挂「食物」，会让「吃」的受事约束把营养物也当成可直接吃的东西。 */
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT e.sid,'实体',1 FROM ext_sid e
 WHERE e.word='营养物' AND e.pos='n'
   AND NOT EXISTS (SELECT 1 FROM sense_categories sc WHERE sc.sense_id = e.sid)
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 1.5 「食物」的属性节点（has_property 的两端都必须是 sense，故先立节点）
 *     可食用 (a)  —— 可以供人食用，无毒无害
 *     有营养 (a)  —— 含有生物所需的养分   （见文件头设计取舍 (1)）
 * ========================================================================= */
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 (2004,'可食用',NULL,'合成词',0,'通用'),
 (2005,'有营养',NULL,'合成词',0,'通用')
ON CONFLICT DO NOTHING;

INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary) VALUES
 (2004,2004,'a',1,'可以供人食用，无毒无害',
   '可以 供 人 食用 无毒 无害','ke3shi2yong4','keshiyong','ksy',700,'本义',NULL,1),
 (2005,2005,'a',1,'含有生物所需的养分',
   '含有 生物 所需 的 养分','you3ying2yang3','youyingyang','yyy',600,'本义',NULL,1)
ON CONFLICT DO NOTHING;

INSERT INTO sense_categories(sense_id, cat, is_primary) VALUES
 (2004,'性状',1),
 (2005,'性状',1)
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 1.6 is_a 关系链 —— 两条**独立**路径（这是本任务的核心）
 *     路径 A（生物分类）：鱼 → 动物 → 生物
 *     路径 B（功能分类）：鱼 → 食物 → 营养物
 *     效果：鱼同时是 动物 与 食物 的下位，是 DAG 而非树；
 *           closure 将自动派生 4 条上位记录（动物/食物 depth=1，生物/营养物 depth=2）。
 *     方向约定：边是 下位 → 上位（dst = 上位），与 closure(ancestor=上位, descendant=下位) 相反。
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    ('鱼',  '动物'),    -- 路径 A 第 1 段（seed §9.1 已建，此处兜底幂等）
    ('动物','生物'),    -- 路径 A 第 2 段（seed §9.1 已建，此处兜底幂等）
    ('鱼',  '食物'),    -- 路径 B 第 1 段（seed §9.1 已建，此处兜底幂等）
    ('食物','营养物')   -- 路径 B 第 2 段（★ 新增）
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN ext_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN ext_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 1.7 属性关系：食物 has_property 可食用 / 有营养
 *     has_property 的闭包策略是 incremental（schema §14 closure_policy），
 *     故插边后由 tr_closure_edge_ai 自动写出 depth=1 闭包行，无需手工插入。
 *     依据：inference_rules #2「属性继承」= 沿 is_a 上位再取 has_property，
 *           因此「鱼 is_a 食物」+「食物 has_property 可食用」⇒ 鱼 可食用。
 * ========================================================================= */
WITH e(owner, prop) AS (VALUES
    ('食物','可食用'),
    ('食物','有营养')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', f.sid, 'sense', p.sid, 'has_property', 5
  FROM e
  JOIN ext_sid f ON f.word = e.owner AND f.pos = 'n'
  JOIN ext_sid p ON p.word = e.prop  AND p.pos = 'a'
 WHERE f.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 1.8 确认「吃」的 patient 约束允许「食物」类型
 *     seed §10 已写入该行，此处为幂等兜底（与 seed 的双路径设计保持一致：
 *       关系层 relations(吃, patient, 食物) —— 供图遍历「什么能吃」
 *       约束层 relation_constraints ……      —— 供解析期合法性校验）
 * ========================================================================= */
INSERT INTO relation_constraints(sense_id, slot, kind, value)
SELECT v.sid,'patient','category','食物'
  FROM ext_sid v
 WHERE v.word = '吃' AND v.pos = 'v'
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 1.9 cross_links —— 与动物学库的跨领域链接
 *     背景：schema 的 cross_links 是**视图**（WHERE src_kind='node' OR dst_kind='node'），
 *           故"写链接"= 在 relations 里插一条端点含 node 的边，前提是 entity_nodes
 *           里已有该领域条目。本文件按"若不存在则建立"的方式落地：
 *             entity_nodes(node_type='domain_entity', domain='animal', label='Fish')
 *             relations(sense 鱼 --domain_link--> node Fish)
 *     领域 'animal'（动物学库）由 schema §14 domains 建好。
 *     注意：domain_link 的闭包策略是 batch（非 incremental），因此不会派生闭包行 ——
 *           这是刻意的：领域链接是稀疏跨库边，物化闭包得不偿失。
 * ========================================================================= */
INSERT INTO entity_nodes(node_id, node_type, domain, label, entity_class, attrs_json)
SELECT 3001,'domain_entity','animal','Fish','Animal',
       '{"taxon":"鱼类","library":"动物学库","linked_word":"鱼"}'
 WHERE NOT EXISTS (SELECT 1 FROM entity_nodes
                    WHERE node_type='domain_entity' AND domain='animal' AND label='Fish');

INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
SELECT 'sense', e.sid, 'node', n.node_id, 'domain_link', 5, 0.9
  FROM ext_sid e
  JOIN entity_nodes n
    ON n.node_type='domain_entity' AND n.domain='animal' AND n.label='Fish'
 WHERE e.word = '鱼' AND e.pos = 'n'
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 任务二：抽象概念与信息人造物分类骨架
 * ######################################################################### */

/* =========================================================================
 * 2.1 词层 —— 35 个骨架词条（ID 2101-2135）
 *     骨架形状（缩进即 is_a 方向，下位 → 上位）：
 *
 *     抽象概念(2101)
 *     ├── 方法(2102) → 算法(2103) / 策略(2104) / 模式(2105)
 *     ├── 属性(2106) → 物理属性(2107) / 抽象属性(2108)
 *     ├── 关系(2109) → 因果(2110) / 部分整体(2111)
 *     └── 规则(2112) → 语法规则(2113) / 推理规则(2114)
 *
 *     人造物(2115)
 *     ├── 物理人造物(2116) → 工具(2117) / 设备(2118) / 建筑(2119)
 *     └── 信息人造物(2120)
 *         ├── 数据对象(2121) → 数据库(2122) / 表(2123) / 视图(2124) / 索引(2125) / 查询(2126)
 *         ├── 信息载体(2127) → 文档(2128) / 书(★复用 sense 21) / 网页(2129) / 文件(2130)
 *         └── 软件产物(2131) → 程序(2132) / 库(2133) / API(2134) / 模型(2135)
 *
 *     「书」不重复建：schema §14 已有 word 20 / sense 21，本文件只补一条 is_a 边。
 * ========================================================================= */
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 (2101,'抽象概念',  NULL,'合成词',0,'通用'),   -- 骨架顶层 A：不占空间、不能直接感知的思维对象
 (2102,'方法',      NULL,'单纯词',0,'通用'),   -- 为达到目的而采取的途径与步骤
 (2103,'算法',      NULL,'合成词',0,'通用'),   -- 解决某类问题的有限步骤序列
 (2104,'策略',      NULL,'合成词',0,'通用'),   -- 为实现目标而制定的行动方针
 (2105,'模式',      NULL,'合成词',0,'通用'),   -- 从反复出现的事物中抽象出的稳定结构
 (2106,'属性',      NULL,'合成词',0,'通用'),   -- 事物固有的性质或特征
 (2107,'物理属性',  NULL,'合成词',0,'通用'),   -- 可由物理量度量的性质
 (2108,'抽象属性',  NULL,'合成词',0,'通用'),   -- 不能由物理量度量、只能由思维把握的性质
 (2109,'关系',      NULL,'合成词',0,'通用'),   -- 事物之间相互联系的状态
 (2110,'因果',      NULL,'合成词',0,'通用'),   -- 原因与结果之间的联系
 (2111,'部分整体',  NULL,'复合词',0,'通用'),   -- 部分与整体之间的联系
 (2112,'规则',      NULL,'合成词',0,'通用'),   -- 规定事物如何运作或行动的准则
 (2113,'语法规则',  NULL,'合成词',0,'通用'),   -- 组词成句所要遵循的规则
 (2114,'推理规则',  NULL,'合成词',0,'通用'),   -- 由已知命题推出新命题所遵循的规则
 (2115,'人造物',    NULL,'合成词',0,'通用'),   -- 骨架顶层 B：由人创造而非自然生成的事物
 (2116,'物理人造物',NULL,'合成词',0,'通用'),   -- 具有实体形态的人造物
 (2117,'工具',      NULL,'合成词',0,'通用'),   -- 用于加工、操作或作业的器具
 (2118,'设备',      NULL,'合成词',0,'通用'),   -- 为特定用途配置的成套器械
 (2119,'建筑',      NULL,'合成词',0,'通用'),   -- 供人居住或使用的构筑物
 (2120,'信息人造物',NULL,'合成词',0,'通用'),   -- 以信息内容为人所使用的人造物
 (2121,'数据对象',  NULL,'合成词',0,'通用'),   -- 信息系统中被操作的抽象数据单位
 (2122,'数据库',    NULL,'合成词',0,'通用'),   -- 按一定结构组织存储的数据集合
 (2123,'表',        NULL,'单纯词',0,'通用'),   -- 数据库中由行和列构成的二维数据集合
 (2124,'视图',      NULL,'合成词',0,'通用'),   -- 数据库中由查询定义、不实际存储数据的虚表
 (2125,'索引',      NULL,'合成词',0,'通用'),   -- 为加快检索而建立的数据结构
 (2126,'查询',      NULL,'合成词',0,'通用'),   -- 向数据库提出的取数请求
 (2127,'信息载体',  NULL,'合成词',0,'通用'),   -- 承载信息的媒介物
 (2128,'文档',      NULL,'合成词',0,'通用'),   -- 以文字图表记录信息的文件
 (2129,'网页',      NULL,'合成词',0,'通用'),   -- 通过浏览器访问的超文本文档
 (2130,'文件',      NULL,'合成词',0,'通用'),   -- 存储介质上按名称组织的记录集合
 (2131,'软件产物',  NULL,'合成词',0,'通用'),   -- 由编程活动产生的人造物
 (2132,'程序',      NULL,'合成词',0,'通用'),   -- 为完成特定任务而编写的指令序列
 (2133,'库',        NULL,'单纯词',0,'通用'),   -- 可供多个程序复用的代码集合
 (2134,'API',       NULL,'缩略语',0,'通用'),   -- 程序之间约定的调用接口
 (2135,'模型',      NULL,'合成词',0,'通用')    -- 对现实对象或过程的简化表示
ON CONFLICT DO NOTHING;

/* 2.1b 量词「份」—— 为「文档 / 文件」提供专用量词（一份文档）
 *      量词在 schema 中是**义项的特化**（classifiers 指向 sense），故先建词与义项。 */
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 (2201,'份',NULL,'单纯词',0,'通用')
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.2 义项层 —— 与 2.1 1:1（sense_id = word_id）
 *     符号语义原则：每条释义都由**已有中文词**组合而成，不用向量、不用外文。
 *     definition_seg 是空格分词的释义，供 FTS5 unicode61 通道检索 1-2 字词。
 *     literalness 标注：表 / 视图 / 库 / 模式 / 模型 是"日常义 → 术语义"的引申。
 *     API 无汉字读音，pinyin 留 NULL，只填去调/声母检索键（去调与声母同为 'api'）。
 * ========================================================================= */
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary) VALUES
 (2101,2101,'n',1,'不占空间、不能直接感知的思维对象',
   '不 占 空间 不 能 直接 感知 的 思维 对象',
   'chou1xiang4gai4nian4','chouxianggainian','cxgn',1400,'本义',NULL,1),
 (2102,2102,'n',1,'为达到某种目的而采取的途径和步骤',
   '为 达到 某种 目的 而 采取 的 途径 和 步骤',
   'fang1fa3','fangfa','ff',2600,'本义',NULL,1),
 (2103,2103,'n',1,'解决某一类问题的有限步骤序列',
   '解决 某一类 问题 的 有限 步骤 序列',
   'suan4fa3','suanfa','sf',2200,'本义',NULL,1),
 (2104,2104,'n',1,'为实现目标而制定的行动方针和方案',
   '为 实现 目标 而 制定 的 行动 方针 和 方案',
   'ce4lve4','celve','cl',1600,'本义',NULL,1),
 (2105,2105,'n',1,'可以从反复出现的事物中抽象出的稳定结构',
   '可以 从 反复 出现 的 事物 中 抽象 出 的 稳定 结构',
   'mo2shi4','moshi','ms',2000,'引申义',NULL,1),
 (2106,2106,'n',1,'事物本身所固有的性质或特征',
   '事物 本身 所 固有 的 性质 或 特征',
   'shu3xing4','shuxing','sx',2400,'本义',NULL,1),
 (2107,2107,'n',1,'可以用物理量度量的性质',
   '可以 用 物理量 度量 的 性质',
   'wu4li3shu3xing4','wulishuxing','wlsx',900,'本义',NULL,1),
 (2108,2108,'n',1,'不能用物理量度量、只能由思维把握的性质',
   '不 能 用 物理量 度量 只能 由 思维 把握 的 性质',
   'chou1xiang4shu3xing4','chouxiangshuxing','cxssx',700,'本义',NULL,1),
 (2109,2109,'n',1,'事物之间相互联系的状态',
   '事物 之间 相互 联系 的 状态',
   'guan1xi4','guanxi','gx',3200,'本义',NULL,1),
 (2110,2110,'n',1,'原因和结果之间的联系',
   '原因 和 结果 之间 的 联系',
   'yin1guo3','yinguo','yg',1500,'本义',NULL,1),
 (2111,2111,'n',1,'部分与它所构成的整体之间的联系',
   '部分 与 它 所 构成 的 整体 之间 的 联系',
   'bu4fen5zheng3ti3','bufenzhengti','bfzt',1100,'本义',NULL,1),
 (2112,2112,'n',1,'规定事物该如何运作或行动的准则',
   '规定 事物 该 如何 运作 或 行动 的 准则',
   'gui1ze2','guize','gz',1900,'本义',NULL,1),
 (2113,2113,'n',1,'语言中组词成句所要遵循的规则',
   '语言 中 组词 成句 所 要 遵循 的 规则',
   'yu3fa3gui1ze2','yufaguize','yfgz',1300,'本义',NULL,1),
 (2114,2114,'n',1,'由已知命题推出新命题所要遵循的规则',
   '由 已知 命题 推出 新 命题 所 要 遵循 的 规则',
   'tui1li3gui1ze2','tuiliguize','tlgz',1200,'本义',NULL,1),
 (2115,2115,'n',1,'由人创造而非自然生成的事物',
   '由 人 创造 而 非 自然 生成 的 事物',
   'ren2zao4wu4','renzaowu','rzw',1000,'本义',NULL,1),
 (2116,2116,'n',1,'具有实体形态的人造物',
   '具有 实体 形态 的 人造物',
   'wu4li3ren2zao4wu4','wulirenzaowu','wlrzw',600,'本义',NULL,1),
 (2117,2117,'n',1,'用于加工、操作或作业的器具',
   '用 于 加工 操作 或 作业 的 器具',
   'gong1ju4','gongju','gj',2800,'本义',NULL,1),
 (2118,2118,'n',1,'为特定用途配置的成套器械',
   '为 特定 用途 配置 的 成套 器械',
   'she4bei4','shebei','sb',2100,'本义',NULL,1),
 (2119,2119,'n',1,'供人居住或使用的构筑物',
   '供 人 居住 或 使用 的 构筑物',
   'jian4zhu4','jianzhu','jz',2300,'本义',NULL,1),
 (2120,2120,'n',1,'以信息内容而非实体形态为人所使用的人造物',
   '以 信息 内容 而 非 实体 形态 为 人 所 使用 的 人造物',
   'xin4xi1ren2zao4wu4','xinxirenzaowu','xxrzw',800,'本义',NULL,1),
 (2121,2121,'n',1,'信息系统中被操作的抽象数据单位',
   '信息 系统 中 被 操作 的 抽象 数据 单位',
   'shu4ju4dui4xiang4','shujuduixiang','sjdx',1200,'本义',NULL,1),
 (2122,2122,'n',1,'按一定结构组织存储的数据集合',
   '按 一定 结构 组织 存储 的 数据 集合',
   'shu4ju4ku4','shujuku','sjk',2500,'本义',NULL,1),
 (2123,2123,'n',1,'数据库中由行和列构成的二维数据集合',
   '数据库 中 由 行 和 列 构成 的 二维 数据 集合',
   'biao3','biao','b',2600,'引申义',NULL,1),
 (2124,2124,'n',1,'数据库中由查询定义、不实际存储数据的虚表',
   '数据库 中 由 查询 定义 不 实际 存储 数据 的 虚表',
   'shi4tu2','shitu','st',1800,'引申义',NULL,1),
 (2125,2125,'n',1,'数据库中为加快检索而建立的数据结构',
   '数据库 中 为 加快 检索 而 建立 的 数据 结构',
   'suo3yin3','suoyin','sy',1900,'本义',NULL,1),
 (2126,2126,'n',1,'向数据库提出的取数请求',
   '向 数据库 提出 的 取数 请求',
   'cha2xun2','chaxun','cx',1700,'本义',NULL,1),
 (2127,2127,'n',1,'承载信息的媒介物',
   '承载 信息 的 媒介物',
   'xin4xi1zai4ti3','xinxizaiti','xxzt',1100,'本义',NULL,1),
 (2128,2128,'n',1,'以文字图表记录信息的文件',
   '以 文字 图表 记录 信息 的 文件',
   'wen2dang4','wendang','wd',2400,'本义',NULL,1),
 (2129,2129,'n',1,'通过浏览器访问的超文本文档',
   '通过 浏览器 访问 的 超文本 文档',
   'wang3ye4','wangye','wy',2000,'本义',NULL,1),
 (2130,2130,'n',1,'在存储介质上按名称组织的记录集合',
   '在 存储 介质 上 按 名称 组织 的 记录 集合',
   'wen2jian4','wenjian','wj',2900,'本义',NULL,1),
 (2131,2131,'n',1,'由编程活动产生、以代码或模型形式存在的人造物',
   '由 编程 活动 产生 以 代码 或 模型 形式 存在 的 人造物',
   'ruan3jian4chan3wu4','ruanjianchanwu','rjcw',900,'本义',NULL,1),
 (2132,2132,'n',1,'为完成特定任务而编写的指令序列',
   '为 完成 特定 任务 而 编写 的 指令 序列',
   'cheng2xu4','chengxu','cx',2700,'本义',NULL,1),
 (2133,2133,'n',1,'可供多个程序复用的代码集合',
   '可 供 多个 程序 复用 的 代码 集合',
   'ku4','ku','k',1600,'引申义',NULL,1),
 (2134,2134,'n',1,'程序之间约定的调用接口',
   '程序 之间 约定 的 调用 接口',
   NULL,'api','api',1500,'本义',NULL,1),
 (2135,2135,'n',1,'对现实对象或过程的简化表示',
   '对 现实 对象 或 过程 的 简化 表示',
   'mo2xing2','moxing','mx',2600,'引申义',NULL,1),
 (2201,2201,'q',1,'用于计量成组或成件的事物',
   '用 于 计量 成组 或 成件 的 事物',
   'fen4','fen','f',2600,NULL,NULL,1)
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.3 义项 ↔ 语义类别
 *     映射策略（类别层取**已有**骨架，不新建类别）：
 *       抽象分支 14 个 → 「抽象」
 *       人造物分支 18 个 → 「人工物」
 *       物理人造物的三个具体类 → 「器物」（工具 / 设备 / 建筑）
 *     注：义项层 is_a（本文件 2.5）与类别层 semantic_categories 是两套正交体系；
 *         类别层只用于量词默认值与约束取值校验，不参与 is_a 遍历。
 * ========================================================================= */
WITH e(word, cat, is_primary) AS (VALUES
    -- 抽象分支
    ('抽象概念','抽象',1), ('方法','抽象',1), ('算法','抽象',1), ('策略','抽象',1),
    ('模式','抽象',1),     ('属性','抽象',1), ('物理属性','抽象',1), ('抽象属性','抽象',1),
    ('关系','抽象',1),     ('因果','抽象',1), ('部分整体','抽象',1), ('规则','抽象',1),
    ('语法规则','抽象',1), ('推理规则','抽象',1),
    -- 人造物分支
    ('人造物','人工物',1),     ('物理人造物','人工物',1), ('信息人造物','人工物',1),
    ('数据对象','人工物',1),   ('数据库','人工物',1),     ('表','人工物',1),
    ('视图','人工物',1),       ('索引','人工物',1),       ('查询','人工物',1),
    ('信息载体','人工物',1),   ('文档','人工物',1),       ('网页','人工物',1),
    ('文件','人工物',1),       ('软件产物','人工物',1),   ('程序','人工物',1),
    ('库','人工物',1),         ('API','人工物',1),        ('模型','人工物',1),
    -- 器物（物理人造物的具体类）
    ('工具','器物',1), ('设备','器物',1), ('建筑','器物',1)
)
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT s.sid, e.cat, e.is_primary
  FROM e JOIN ext_sid s ON s.word = e.word AND s.pos = 'n'
 WHERE 1
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.4 义素分析（sense_features）
 *     +countable：骨架 35 个节点全部可个体计数（一个算法 / 一份文档 / 一张表）
 *     +concrete ：形态可感的载体（物理人造物一族 + 文档/书/网页/文件）
 *     注：书(sense 21) 的这两个义素已由 schema §14 写入，此处 DO NOTHING。
 * ========================================================================= */
WITH skel(word) AS (VALUES
    ('抽象概念'),('方法'),('算法'),('策略'),('模式'),
    ('属性'),('物理属性'),('抽象属性'),
    ('关系'),('因果'),('部分整体'),
    ('规则'),('语法规则'),('推理规则'),
    ('人造物'),('物理人造物'),('工具'),('设备'),('建筑'),
    ('信息人造物'),('数据对象'),('数据库'),('表'),('视图'),('索引'),('查询'),
    ('信息载体'),('文档'),('网页'),('文件'),
    ('软件产物'),('程序'),('库'),('API'),('模型')
)
INSERT INTO sense_features(sense_id, feature, value)
SELECT s.sid, '+countable', '1'
  FROM skel JOIN ext_sid s ON s.word = skel.word AND s.pos = 'n'
 WHERE 1
ON CONFLICT DO NOTHING;

WITH concrete(word) AS (VALUES
    ('物理人造物'),('工具'),('设备'),('建筑'),
    ('文档'),('书'),('网页'),('文件')
)
INSERT INTO sense_features(sense_id, feature, value)
SELECT s.sid, '+concrete', '1'
  FROM concrete JOIN ext_sid s ON s.word = concrete.word AND s.pos = 'n'
 WHERE 1
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.5 骨架 is_a 关系链（34 条边）
 *     写法与 seed §9.1 完全一致：CTE(VALUES) 把「词形对」映射为 sense_id，
 *     本文件因此不硬编码任何 sense_id（除自建节点外）。
 *     插边后由 tr_closure_edge_ai 自动派生多级闭包：
 *       视图 → 数据对象(1) → 信息人造物(2) → 人造物(3)
 *       文档 → 信息载体(1) → 信息人造物(2) → 人造物(3)
 *       程序 → 软件产物(1) → 信息人造物(2) → 人造物(3)
 *     ⚠ 边序无关紧要：触发器同时向"更具体"与"更一般"两侧递归已有闭包，
 *       故任意插入顺序都能得到与全量重建一致的结果（已由 validate_seed §4 对拍验证）。
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    -- 抽象概念分支
    ('方法',    '抽象概念'),
    ('属性',    '抽象概念'),
    ('关系',    '抽象概念'),
    ('规则',    '抽象概念'),
    ('算法',    '方法'),
    ('策略',    '方法'),
    ('模式',    '方法'),
    ('物理属性','属性'),
    ('抽象属性','属性'),
    ('因果',    '关系'),
    ('部分整体','关系'),
    ('语法规则','规则'),
    ('推理规则','规则'),
    -- 人造物分支
    ('物理人造物','人造物'),
    ('信息人造物','人造物'),
    ('工具',      '物理人造物'),
    ('设备',      '物理人造物'),
    ('建筑',      '物理人造物'),
    ('数据对象',  '信息人造物'),
    ('信息载体',  '信息人造物'),
    ('软件产物',  '信息人造物'),
    ('数据库',    '数据对象'),
    ('表',        '数据对象'),
    ('视图',      '数据对象'),
    ('索引',      '数据对象'),
    ('查询',      '数据对象'),
    ('文档',      '信息载体'),
    ('书',        '信息载体'),     -- ★ 复用 schema §14 已有的 sense 21
    ('网页',      '信息载体'),
    ('文件',      '信息载体'),
    ('程序',      '软件产物'),
    ('库',        '软件产物'),
    ('API',       '软件产物'),
    ('模型',      '软件产物')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN ext_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN ext_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.6 量词搭配
 *     视图 / 程序 → 个（通用个体量词，schema 已存在 sense 1023）
 *     文档 / 文件 → 份（本文件新建 sense 2201，名量词·部分）
 *     三层量词体系里的"显式搭配"层；未显式搭配的名词回退到
 *     classifier_class_defaults 的类别默认值。
 * ========================================================================= */
INSERT INTO classifiers(classifier_sense_id, kind, subtype, is_borrowed, has_reduplication) VALUES
 (2201,'名量词','部分',0,0)     -- 份：一份文档 / 一份文件（不可重叠）
ON CONFLICT(classifier_sense_id) DO NOTHING;

WITH e(noun, clf, pref, is_def) AS (VALUES
    ('视图','个',0.90,1),
    ('文档','份',0.95,1),
    ('程序','个',0.90,1),
    ('文件','份',0.90,1)
)
INSERT INTO noun_classifier_pairs(noun_sense_id, classifier_sense_id, preference, is_default, source_id)
SELECT n.sid, cl.classifier_sense_id, e.pref, e.is_def, 5
  FROM e
  JOIN ext_sid   n  ON n.word = e.noun AND n.pos = 'n'
  JOIN ext_sid   cs ON cs.word = e.clf AND cs.pos = 'q'
  JOIN classifiers cl ON cl.classifier_sense_id = cs.sid
 WHERE n.sid <> cs.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.7 论元选择限制（relation_constraints）—— 算法 / 文档 / 程序
 *     语义读法：把这三个节点当作**论元**出现在命题里时的类型约束，
 *     与 seed §10 对动词的做法同构（键值化、可索引、可被规则引擎消费）。
 *       theme      —— 该节点自身充当"被说明的对象"时的类型
 *       instrument —— 实现/操作该节点所凭借的东西
 *       patient    —— 该节点被动作产出的类型（文档是被写出来的）
 *     kind='category' 的取值必须落在 semantic_categories 内：抽象 / 人工物 / 器物。
 * ========================================================================= */
WITH e(node, slot, kind, val) AS (VALUES
    ('算法','theme',     'category','抽象'),
    ('算法','instrument','category','器物'),
    ('文档','theme',     'category','人工物'),
    ('文档','patient',   'category','人工物'),
    ('程序','theme',     'category','人工物'),
    ('程序','instrument','category','器物')
)
INSERT INTO relation_constraints(sense_id, slot, kind, value)
SELECT s.sid, e.slot, e.kind, e.val
  FROM e JOIN ext_sid s ON s.word = e.node AND s.pos = 'n'
 WHERE 1
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3. 附：自检与常用查询（注释形式，供应用层 / validate_seed.py 调用）
 * ========================================================================= */

-- 3.1 【多继承自检】鱼的上位应有 4 个，且两条路径彼此独立（无祖先关系）
-- SELECT a.ancestor, w.word, a.depth
--   FROM closure a JOIN senses s ON s.sense_id = a.descendant
--   JOIN words w ON w.word_id = s.word_id
--  WHERE a.relation='is_a' AND w.word='鱼'
--  ORDER BY a.depth, w.word;
-- 期望：动物(1) 食物(1) 生物(2) 营养物(2)

-- 3.2 【路径独立性】应返回 0 行 —— 「动物」与「食物」之间不存在 is_a 关系
-- SELECT 1 FROM closure
--  WHERE relation='is_a'
--    AND ((ancestor=(SELECT sense_id FROM senses WHERE word_id=(SELECT word_id FROM words WHERE word='动物'))
--          AND descendant=(SELECT sense_id FROM senses WHERE word_id=(SELECT word_id FROM words WHERE word='食物')))
--      OR (ancestor=(SELECT sense_id FROM senses WHERE word_id=(SELECT word_id FROM words WHERE word='食物'))
--          AND descendant=(SELECT sense_id FROM senses WHERE word_id=(SELECT word_id FROM words WHERE word='动物'))));

-- 3.3 【属性继承链】鱼 应通过 is_a 上位继承到「可食用 / 有营养」（inference_rules #2）
-- SELECT p.word AS inherited_property
--   FROM closure a
--   JOIN relations r ON r.src_id = a.ancestor AND r.relation='has_property'
--   JOIN senses ps ON ps.sense_id = r.dst_id
--   JOIN words  p  ON p.word_id  = ps.word_id
--  WHERE a.relation='is_a'
--    AND a.descendant = (SELECT s.sense_id FROM senses s JOIN words w ON w.word_id=s.word_id
--                         WHERE w.word='鱼' AND s.pos='n');
-- 期望：可食用 / 有营养（各由 食物 depth=1 一跳继承）

-- 3.4 【骨架自检】抽象概念 / 人造物 的全部下位（含自身）
-- SELECT w.word, c.depth FROM closure c
--   JOIN senses s ON s.sense_id = c.descendant
--   JOIN words  w ON w.word_id = s.word_id
--  WHERE c.relation='is_a'
--    AND c.ancestor = (SELECT s.sense_id FROM senses s JOIN words w2 ON w2.word_id=s.word_id
--                       WHERE w2.word='人造物' AND s.pos='n')
--  ORDER BY c.depth, w.word;

-- 3.5 【闭包一致性巡检】以下每句正常应返回 0 行
-- (a) 新增 is_a 边的 depth=1 闭包行缺失
-- SELECT r.relation_id FROM relations r
--  WHERE r.relation='is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
--    AND NOT EXISTS (SELECT 1 FROM closure c
--                     WHERE c.relation='is_a' AND c.ancestor=r.dst_id AND c.descendant=r.src_id);
-- (b) domain_link 端点悬空
-- SELECT * FROM relations r WHERE r.relation='domain_link'
--    AND (NOT EXISTS (SELECT 1 FROM senses WHERE sense_id=r.src_id)
--      OR NOT EXISTS (SELECT 1 FROM entity_nodes WHERE node_id=r.dst_id));

COMMIT;

/*****************************************************************************
 * End of seed_general_v3_ext.sql
 *****************************************************************************/
