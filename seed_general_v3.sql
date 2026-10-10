/*****************************************************************************
 * Logos 通用领域种子数据  seed_general_v3.sql   (v0.1 demo)
 * 中文原生符号语义数据库 —— 最小可用数据集
 *
 * 目标：跑通「猫吃鱼」及其扩展句式（把字句 / 被字句 / 话题句 / 存现句）
 *
 * ---------------------------------------------------------------------------
 * 【前置依赖】必须先执行 schema_v3.sql，再执行本文件。
 *   理由：本文件是**叠加式增量种子**，不重建任何表。schema_v3.sql 的 §14 已写入
 *   26 个词 / 27 个义项（猫=word2/sense2、鱼=word26/sense27、动物=word11/sense12、
 *   把=word4/sense4、只=word3/sense3、看=word17/sense18、他=word21/sense22、
 *   桌子=word9/sense10…）。这些行**不重复插入**，一律用自然键子查询引用，
 *   因此本文件对 schema 内部 ID 的取值不敏感。
 *
 * 【幂等性】全部 INSERT 均带 ON CONFLICT DO NOTHING 或 UPSERT，
 *   可重复执行任意次，结果一致（不产生重复行、不触发 FTS 重复写入）。
 *
 * 【ID 分配策略】新实体统一使用 1000 段（words 1001-1027 / senses 1001-1027），
 *   与 schema_v3.sql 的 1-27 段完全隔离，杜绝主键冲撞。
 *   对已有实体一律按 (word, pos) 自然键查找，见临时视图 seed_sid。
 *
 * ---------------------------------------------------------------------------
 * 【写入规模】触及 15 张表，合计 +195 行（下表为 validate_seed.py 实测值）
 *   1  semantic_categories       +1   食物（本体骨架补一个节点）
 *   2  words                    +27   人/我/你/吃/喝/抓/跑/是/有/大/小/好/坏/
 *                                     被/的/了/着/过/在/从/向/条/个/水/食物/生物/游泳
 *   3  senses                   +27   与上表 1:1（每词单义项，v0.1 不做多义）
 *   4  sense_features           +22   义素分析（生命度/具体性/可数性/自主性/极性）
 *   5  sense_categories         +20   义项 ↔ 语义类别（含为既有词条补类别）
 *   6  function_words            +8   被/的/了/着/过/在/从/向（把 已存在，UPSERT 补全）
 *   7  function_word_roles       +5   虚词 → 语义角色（被→施事、在→处所、从→起点/时点、向→方向）
 *   8  classifiers               +2   条 / 个（只、把 已存在）
 *   9  noun_classifier_pairs     +4   鱼→条、人→个、我→个、你→个（猫→只、老虎→只 已存在）
 *  10  classifier_class_defaults +1   人 → 个（动物 → 只 已存在）
 *  11  relations                +23   5×is_a + 8×can_do + 6×agent + 2×patient + 2×opposite
 *  12  relation_constraints     +17   各动词的 agent/patient/theme/instrument 选择限制
 *  13  slot_constraints         +17   5 个目标句式的槽位约束
 *  14  closure                  +16   5 行 is_a depth=1 + 11 行由触发器派生
 *                                     （is_a depth=2 共 3 行 + can_do depth=1 共 8 行）
 *  15  utterances                +5   demo 语料（5 个扩展句子）
 *   注：第 14 行的 11 行由 tr_closure_edge_ai 在 §9 插边时**自动派生**，非手工写入。
 *       本文件另有一条显式 depth=1 兜底语句（§12），在默认配置下命中冲突、
 *       为 DO NOTHING 空操作；它的价值在于 mode='off' 或触发器被禁用的流水线里兜底。
 *
 * 【依赖顺序】（严格按此顺序，不可调换）
 *   semantic_categories ──┐
 *                        ├─→ sense_categories
 *   words → senses ──────┤
 *                        ├─→ sense_features
 *                        ├─→ function_words → function_word_roles
 *                        ├─→ classifiers → noun_classifier_pairs
 *                        │               → classifier_class_defaults（另需 semantic_categories）
 *                        ├─→ relations（需 senses + relation_types + closure_policy）
 *                        │     └─→ 触发 tr_closure_edge_ai（需 closure_policy 已有 is_a 行）
 *                        │     └─→ relation_constraints
 *                        ├─→ closure（depth=1，须在 relations 之后）
 *                        ├─→ slot_constraints（需 pattern_slots，schema 已建）
 *                        └─→ utterances（需 sentence_patterns / sentence_moods）
 *
 * 【一处刻意的设计取舍 —— 请留意】
 *   需求原文写「吃 has_property 需要_施事(猫/人) + 需要_受事(食物)」。
 *   v3 schema 为"论元选择限制"提供了**专用机制**，比 has_property 更精确且可推理：
 *     · relations(吃, agent, 动物/人) / relations(吃, patient, 食物)
 *       —— relation_types 注释："动词→典型施事类别" / "动词→典型受事类别"
 *     · relation_constraints(吃, 'agent', 'category', '动物') …
 *       —— 驱动 inference_rules #3「施事合法性」/ #4「受事合法性」
 *   因此本文件**不用 has_property 承载论元需求**（那会把"需要施事"实体化成伪名词），
 *   改用上面两条正交路径表达。语义等价，且能被索引与规则引擎消费。
 *
 * 【连接级 PRAGMA】必须在每个连接上执行，不随 schema 持久化。
 *   注意：PRAGMA foreign_keys 在**显式事务内是 no-op**，故写在 BEGIN 之前。
 * =========================================================================*/

PRAGMA foreign_keys     = ON;   -- SQLite 默认 OFF；不开则所有 FK 形同虚设
PRAGMA recursive_triggers = ON; -- 闭包触发器依赖触发器级联语义

BEGIN;

/* =========================================================================
 * 0. 自然键查找视图（仅本会话有效，不落盘）
 *    用途：对 schema_v3.sql 已存在的词条按 (word, pos) 取 sense_id，
 *          使本文件不硬编码任何既有 ID。
 *    注意：(word_id, pos, sense_no) 才唯一；(word, pos) 在本 demo 内已足够唯一，
 *          若将来出现同词性多义项，须改用 sense_no 精确定位。
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS seed_sid AS
    SELECT w.word AS word, s.pos AS pos, s.sense_no AS sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;

/* =========================================================================
 * 1. 本体骨架：补一个「食物」类别
 *    用途：作为 relation_constraints(kind='category', value='食物') 的取值依据，
 *          并让「鱼 is_a 食物」这件事在类别层与关系层各有一处落点。
 *    注意：category 是本体骨架（供字/词/量词默认值共用），
 *          与"义项层的 is_a 关系"是两套正交体系，不是重复存储 —— schema §1 注释明示。
 * ========================================================================= */
INSERT INTO semantic_categories(cat, parent_cat, description) VALUES
 ('食物','实体','可供人或动物食用的东西')          -- 用于「需_受事=食物」的选择限制
ON CONFLICT(cat) DO NOTHING;

/* =========================================================================
 * 2. 词层 —— 新增 27 个词条（ID 1001-1027）
 *    用途：补齐 demo 所需的实词与虚词。已存在的猫/鱼/动物/把/只/看/他 不重复插入。
 *    约定：word_type 取 pos_tags 无关的构词类型；register 统一 '通用'；
 *          is_bound_root=0（本批均为能独立成词的词，无"澡/桌"这类黏着词根）。
 * ========================================================================= */
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 (1001,'人',  NULL,'单纯词',0,'通用'),   -- 名词：施事典型类别
 (1002,'我',  NULL,'单纯词',0,'通用'),   -- 第一人称代词
 (1003,'你',  NULL,'单纯词',0,'通用'),   -- 第二人称代词
 (1004,'吃',  NULL,'单纯词',0,'通用'),   -- 核心及物动词，主谓宾 demo 的 V
 (1005,'喝',  NULL,'单纯词',0,'通用'),   -- 及物动词，受事限液体
 (1006,'抓',  NULL,'单纯词',0,'通用'),   -- 及物动词，猫 can_do 抓
 (1007,'跑',  NULL,'单纯词',0,'通用'),   -- 不及物动词，主谓句 demo
 (1008,'是',  NULL,'单纯词',0,'通用'),   -- 判断动词，主系表 demo
 (1009,'有',  NULL,'单纯词',0,'通用'),   -- 存现动词，存现句 demo（subcategory='存现'）
 (1010,'大',  NULL,'单纯词',0,'通用'),   -- 形容词，opposite 小
 (1011,'小',  NULL,'单纯词',0,'通用'),   -- 形容词，opposite 大
 (1012,'好',  NULL,'单纯词',0,'通用'),   -- 形容词，opposite 坏
 (1013,'坏',  NULL,'单纯词',0,'通用'),   -- 形容词，opposite 好
 (1014,'被',  NULL,'单纯词',0,'通用'),   -- 介词/被动 —— 被字句标记词
 (1015,'的',  NULL,'单纯词',0,'通用'),   -- 结构助词，定语标记
 (1016,'了',  NULL,'单纯词',0,'通用'),   -- 动态助词，完成体
 (1017,'着',  NULL,'单纯词',0,'通用'),   -- 动态助词，持续体（存现句用）
 (1018,'过',  NULL,'单纯词',0,'通用'),   -- 动态助词，经历体
 (1019,'在',  NULL,'单纯词',0,'通用'),   -- 介词，引出处所
 (1020,'从',  NULL,'单纯词',0,'通用'),   -- 介词，引出起点
 (1021,'向',  NULL,'单纯词',0,'通用'),   -- 介词，引出方向/对象
 (1022,'条',  NULL,'单纯词',0,'通用'),   -- 名量词：鱼 → 条
 (1023,'个',  NULL,'单纯词',0,'通用'),   -- 通用个体量词：人 → 个
 (1024,'水',  NULL,'单纯词',0,'通用'),   -- 名词：喝的受事
 (1025,'食物',NULL,'合成词',0,'通用'),   -- 名词：吃的受事，鱼 is_a 食物
 (1026,'生物',NULL,'合成词',0,'通用'),   -- 名词：动物 is_a 生物（层级链顶端）
 (1027,'游泳',NULL,'合成词',0,'通用')    -- 动词：鱼 can_do 游泳
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3. 义项层 —— 与上表 1:1，sense_id = 1000 + 序
 *    用途：符号语义的核心载体 —— 每个义项用**中文词语**定义，不用向量。
 *    要点：
 *      · definition_seg 是 jieba 式空格分词，供 FTS5 unicode61 通道检索 1-2 字词
 *        （trigram 通道要求查询 ≥3 字符，短词搜不到，故必须双通道）
 *      · pinyin 用数字标调；轻声记 0（的 de0 / 了 le0 / 着 zhe0）
 *      · subcategory 只在 schema 注释列出的动词次类里取值：判断/存现
 * ========================================================================= */
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary) VALUES
 (1001,1001,'n',1,'会制造和使用工具、能进行社会活动的高级哺乳动物',
   '会 制造 和 使用 工具 能 进行 社会 活动 的 高级 哺乳动物',
   'ren2','ren','r',4800,'本义',NULL,1),
 (1002,1002,'r',1,'第一人称代词，指说话人自己',
   '第一 人称 代词 指 说话人 自己',
   'wo3','wo','w',6000,'本义',NULL,1),
 (1003,1003,'r',1,'第二人称代词，指听话人',
   '第二 人称 代词 指 听话人',
   'ni3','ni','n',5800,'本义',NULL,1),
 (1004,1004,'v',1,'把食物放进嘴里咀嚼后咽下',
   '把 食物 放进 嘴里 咀嚼 后 咽下',
   'chi1','chi','ch',4200,'本义',NULL,1),
 (1005,1005,'v',1,'把液体或流食咽下去',
   '把 液体 或 流食 咽下去',
   'he1','he','h',3100,'本义',NULL,1),
 (1006,1006,'v',1,'手指或爪子用力握住、捉住',
   '手指 或 爪子 用力 握住 捉住',
   'zhua1','zhua','zh',2200,'本义',NULL,1),
 (1007,1007,'v',1,'双脚或四肢快速交替移动前进',
   '双脚 或 四肢 快速 交替 移动 前进',
   'pao3','pao','p',2600,'本义',NULL,1),
 (1008,1008,'v',1,'联系两个成分表示判断或等同',
   '联系 两个 成分 表示 判断 或 等同',
   'shi4','shi','sh',9000,NULL,'判断',1),
 (1009,1009,'v',1,'表示领有或存在',
   '表示 领有 或 存在',
   'you3','you','y',8500,NULL,'存现',1),
 (1010,1010,'a',1,'在体积、面积、数量、年龄等方面超过一般或超过比较对象',
   '在 体积 面积 数量 年龄 等 方面 超过 一般 或 超过 比较 对象',
   'da4','da','d',5000,'本义',NULL,1),
 (1011,1011,'a',1,'在体积、面积、数量、年龄等方面不及一般或不及比较对象',
   '在 体积 面积 数量 年龄 等 方面 不及 一般 或 不及 比较 对象',
   'xiao3','xiao','x',4800,'本义',NULL,1),
 (1012,1012,'a',1,'优点多，令人满意',
   '优点 多 令人 满意',
   'hao3','hao','h',5200,'本义',NULL,1),
 (1013,1013,'a',1,'缺点多，使人不满意；品质恶劣',
   '缺点 多 使 人 不满意 品质 恶劣',
   'huai4','huai','h',2000,'本义',NULL,1),
 (1014,1014,'p',1,'介词，引出施事，表示被动，构成被字句',
   '介词 引出 施事 表示 被动 构成 被字句',
   'bei4','bei','b',3300,NULL,NULL,1),
 (1015,1015,'u',1,'结构助词，用在定语和中心语之间',
   '结构 助词 用在 定语 和 中心语 之间',
   'de0','de','d',12000,NULL,NULL,1),
 (1016,1016,'u',1,'动态助词，表示动作完成或状态实现',
   '动态 助词 表示 动作 完成 或 状态 实现',
   'le0','le','l',11000,NULL,NULL,1),
 (1017,1017,'u',1,'动态助词，表示动作正在进行或状态持续',
   '动态 助词 表示 动作 正在 进行 或 状态 持续',
   'zhe0','zhe','zh',4200,NULL,NULL,1),
 (1018,1018,'u',1,'动态助词，表示曾经有过某种经历',
   '动态 助词 表示 曾经 有 过 某种 经历',
   'guo4','guo','g',3800,NULL,NULL,1),
 (1019,1019,'p',1,'介词，引出处所或时间',
   '介词 引出 处所 或 时间',
   'zai4','zai','z',7000,NULL,NULL,1),
 (1020,1020,'p',1,'介词，引出起点或来源',
   '介词 引出 起点 或 来源',
   'cong2','cong','c',3600,NULL,NULL,1),
 (1021,1021,'p',1,'介词，引出方向或动作的对象',
   '介词 引出 方向 或 动作 的 对象',
   'xiang4','xiang','x',3000,NULL,NULL,1),
 (1022,1022,'q',1,'用于计量细长的东西或分项的事物',
   '用 于 计量 细长 的 东西 或 分项 的 事物',
   'tiao2','tiao','t',2600,NULL,NULL,1),
 (1023,1023,'q',1,'最常用的通用个体量词',
   '最 常用 的 通用 个体 量词',
   'ge4','ge','g',9500,NULL,NULL,1),
 (1024,1024,'n',1,'无色无味的透明液体，是生物生存必需的物质',
   '无色 无味 的 透明 液体 是 生物 生存 必需 的 物质',
   'shui3','shui','sh',4500,'本义',NULL,1),
 (1025,1025,'n',1,'可以充饥、供人或动物食用的东西',
   '可以 充饥 供 人 或 动物 食用 的 东西',
   'shi2wu4','shiwu','shw',3400,'本义',NULL,1),
 (1026,1026,'n',1,'自然界中具有生命现象、能进行新陈代谢的物体',
   '自然界 中 具有 生命 现象 能 进行 新陈代谢 的 物体',
   'sheng1wu4','shengwu','shw',2800,'本义',NULL,1),
 (1027,1027,'v',1,'在水里游动前进',
   '在 水里 游动 前进',
   'you2yong3','youyong','yy',1900,'本义',NULL,1)
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 4. 义素分析（sense_features）
 *    用途：支撑选择限制的**细粒度**判断（比类别层级更精确）。
 *          例：喝 的受事要求 +concrete，抓 的受事要求 +concrete。
 *    取值：boolean 特征用 '1'/'0'；polarity 是 enum，取 positive/negative。
 * ========================================================================= */
INSERT INTO sense_features(sense_id, feature, value) VALUES
 (1001,'+animate','1'), (1001,'+human','1'),        -- 人
 (1002,'+animate','1'), (1002,'+human','1'),        -- 我
 (1003,'+animate','1'), (1003,'+human','1'),        -- 你
 (1004,'+volitional','1'),                          -- 吃：自主动作
 (1005,'+volitional','1'),                          -- 喝
 (1006,'+volitional','1'),                          -- 抓
 (1007,'+volitional','1'),                          -- 跑
 (1012,'polarity','positive'),                      -- 好：正向
 (1013,'polarity','negative'),                      -- 坏：负向
 (1024,'+concrete','1'),                            -- 水
 (1025,'+concrete','1'), (1025,'+countable','1'),   -- 食物
 (1026,'+animate','1'), (1026,'+concrete','1'),     -- 生物
 (1027,'+volitional','1')                           -- 游泳
ON CONFLICT DO NOTHING;

-- 已存在义项补义素（走自然键，不硬编码 ID）：他 / 鱼 / 猫
INSERT INTO sense_features(sense_id, feature, value)
SELECT sid, '+animate', '1' FROM seed_sid WHERE word='他' AND pos='r'
ON CONFLICT DO NOTHING;
INSERT INTO sense_features(sense_id, feature, value)
SELECT sid, '+human', '1' FROM seed_sid WHERE word='他' AND pos='r'
ON CONFLICT DO NOTHING;
INSERT INTO sense_features(sense_id, feature, value)
SELECT sid, '+animate', '1' FROM seed_sid WHERE word='鱼' AND pos='n'
ON CONFLICT DO NOTHING;
INSERT INTO sense_features(sense_id, feature, value)
SELECT sid, '+countable', '1' FROM seed_sid WHERE word='鱼' AND pos='n'
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 5. 义项 ↔ 语义类别（sense_categories）
 *    用途：① 量词类别默认值的查找依据（classifier_class_defaults.cat）
 *          ② 推理规则 #3/#4「施事/受事合法性」校验 relation_constraints.category
 *    注意：一个义项可有多个类别，用 is_primary 标主类别。
 *          鱼 = 动物 是主类别；鱼 = 食物 是次类别（"部分鱼可食用"）。
 * ========================================================================= */
INSERT INTO sense_categories(sense_id, cat, is_primary) VALUES
 (1001,'人',1),      -- 人
 (1002,'人',1),      -- 我
 (1003,'人',1),      -- 你
 (1004,'事件',1),    -- 吃
 (1005,'事件',1),    -- 喝
 (1006,'事件',1),    -- 抓
 (1007,'事件',1),    -- 跑
 (1008,'抽象',1),    -- 是（判断关系，非动作事件）
 (1009,'事件',1),    -- 有（存在/领有）
 (1010,'性状',1),    -- 大
 (1011,'性状',1),    -- 小
 (1012,'性状',1),    -- 好
 (1013,'性状',1),    -- 坏
 (1024,'无机物',1),  -- 水
 (1025,'食物',1),    -- 食物
 (1026,'生物',1),    -- 生物
 (1027,'事件',1)     -- 游泳
ON CONFLICT DO NOTHING;

-- 已存在义项补类别（自然键引用）
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT sid, '人', 1 FROM seed_sid WHERE word='他' AND pos='r'          -- 他是人称代词
ON CONFLICT DO NOTHING;
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT sid, '食物', 0 FROM seed_sid WHERE word='鱼' AND pos='n'        -- 鱼：次类别=食物（可食用）
ON CONFLICT DO NOTHING;
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT sid, '人工物', 1 FROM seed_sid WHERE word='桌子' AND pos='n'    -- 存现句 demo 的处所承载物
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 6. 虚词独立建模（function_words）
 *    用途：虚词不是"没有意义的词"，而是**语法关系的显式载体**。
 *          schema 让它们直接声明引出哪个语义角色（见 §7），
 *          从而把「把/被/在/从/向」从散落的搭配规则升级为可推理的图节点。
 *    字段语义：
 *      category             介词/连词/结构助词/动态助词/… （CHECK 白名单内）
 *      subcategory          介词: 处置/被动/处所/时间/工具/关涉/比较/排除；助词: 体标记等
 *      grammatical_function 该虚词做什么（人读）
 *      attaches_to          附着位置（机读的粗粒度定位）
 *      selects_verb_subcat  体标记专用：约束可附着的谓词次类
 * ========================================================================= */

-- 6.1 把（schema 已存在，此处用 UPSERT 确保"条目完整"，幂等）
INSERT INTO function_words(sense_id, category, subcategory, grammatical_function,
                           attaches_to, selects_verb_subcat, is_sentence_final)
SELECT sid, '介词','处置','引出被处置的受事，与后续动词短语共同构成处置式（把字句）',
       '介宾短语整体作状语，位于动词短语之前',NULL,0
  FROM seed_sid WHERE word='把' AND pos='p'
ON CONFLICT(sense_id) DO UPDATE SET
  category=excluded.category, subcategory=excluded.subcategory,
  grammatical_function=excluded.grammatical_function,
  attaches_to=excluded.attaches_to,
  selects_verb_subcat=excluded.selects_verb_subcat,
  is_sentence_final=excluded.is_sentence_final;

-- 6.2 其余 8 个虚词（新增）
INSERT INTO function_words(sense_id, category, subcategory, grammatical_function,
                           attaches_to, selects_verb_subcat, is_sentence_final) VALUES
 (1014,'介词','被动','引出施事，构成被字句；施事可由语境省略','介宾短语整体作状语，位于动词短语之前',NULL,0),
 (1015,'结构助词','定语标记','连接定语与中心语，构成"X 的 Y"','后置，附着于定语之后',NULL,0),
 (1016,'动态助词','完成体','表示动作完成或状态实现（"了1"，后接宾语）','后置，附着于动词之后',NULL,0),
 (1017,'动态助词','持续体','表示动作进行或状态持续（存现句"V着"）','后置，附着于动词之后',NULL,0),
 (1018,'动态助词','经历体','表示曾经有过某种经历','后置，附着于动词之后',NULL,0),
 (1019,'介词','处所','引出处所，构成"在+处所+V"','介宾短语整体作状语，位于动词短语之前',NULL,0),
 (1020,'介词','起点','引出位移或时间的起点（"从…到…"）','介宾短语整体作状语，位于动词短语之前',NULL,0),
 (1021,'介词','方向','引出动作的方向或所对的对象','介宾短语整体作状语，位于动词短语之前',NULL,0)
ON CONFLICT(sense_id) DO UPDATE SET
  category=excluded.category, subcategory=excluded.subcategory,
  grammatical_function=excluded.grammatical_function,
  attaches_to=excluded.attaches_to,
  selects_verb_subcat=excluded.selects_verb_subcat,
  is_sentence_final=excluded.is_sentence_final;

/* =========================================================================
 * 7. 虚词 → 语义角色映射（function_word_roles）
 *    用途：这是"用数据库存语法"的关键一步 —— 解析器遇到「把」，不必查硬编码规则，
 *          直接查本表即知"它引出的成分应标为受事，位置在前"。
 *    对照表：把→受事/前置   被→施事/前置   在→处所/前置   从→起点/前置   向→方向/前置
 * ========================================================================= */
INSERT INTO function_word_roles(sense_id, role, position)
SELECT sid, '受事', '前置' FROM seed_sid WHERE word='把' AND pos='p'   -- 把字句
ON CONFLICT DO NOTHING;

INSERT INTO function_word_roles(sense_id, role, position) VALUES
 (1014,'施事','前置'),   -- 被：引出施事
 (1019,'处所','前置'),   -- 在：引出处所
 (1020,'起点','前置'),   -- 从：引出起点
 (1020,'时点','前置'),   -- 从：也可引出时间起点（"从明天起"）
 (1021,'方向','前置')    -- 向：引出方向
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 8. 量词层
 *    用途：量词搭配是中文特有的强制语法范畴（"一只猫"不能说"一个猫"）。
 *    三层结构：义项特化 → 名词-量词显式搭配 → 类别默认量词（可扩展性关键）
 * ========================================================================= */

-- 8.1 量词义项特化（只 sense3、把 sense5 已存在，此处只加 条/个）
INSERT INTO classifiers(classifier_sense_id, kind, subtype, is_borrowed, has_reduplication) VALUES
 (1022,'名量词','个体',0,0),   -- 条：细长物/分项（一条鱼、一条新闻）
 (1023,'名量词','个体',0,1)    -- 个：通用个体量词（一个人），可重叠"个个"
ON CONFLICT(classifier_sense_id) DO NOTHING;

-- 8.2 名词-量词显式搭配（preference 0-1；is_default 标记该名词的首选量词）
INSERT INTO noun_classifier_pairs(noun_sense_id, classifier_sense_id, preference, is_default, source_id)
SELECT sid, 1022, 0.95, 1, 5 FROM seed_sid WHERE word='鱼' AND pos='n'   -- 鱼→条
ON CONFLICT DO NOTHING;
INSERT INTO noun_classifier_pairs(noun_sense_id, classifier_sense_id, preference, is_default, source_id)
SELECT sid, 1023, 0.80, 1, 5 FROM seed_sid WHERE word='人' AND pos='n'   -- 人→个
ON CONFLICT DO NOTHING;
INSERT INTO noun_classifier_pairs(noun_sense_id, classifier_sense_id, preference, is_default, source_id)
SELECT sid, 1023, 0.90, 1, 5 FROM seed_sid WHERE word='我' AND pos='r'   -- 我→个
ON CONFLICT DO NOTHING;
INSERT INTO noun_classifier_pairs(noun_sense_id, classifier_sense_id, preference, is_default, source_id)
SELECT sid, 1023, 0.90, 1, 5 FROM seed_sid WHERE word='你' AND pos='r'   -- 你→个
ON CONFLICT DO NOTHING;

-- 8.3 类别默认量词（避免为每个名词逐一建行 —— 加新领域库时的可扩展性保障）
INSERT INTO classifier_class_defaults(cat, classifier_sense_id, preference) VALUES
 ('人',1023,0.80)    -- 人 → 个（动物→只 已由 schema 种入）
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 9. 关系层（relations）—— 用关系数据库替代向量
 *    统一关系表：只存"正向"边，逆向由视图 v_relations_all 在线推导（无冗余）。
 *    端点用 (kind,id) 异构寻址；本节全部为 sense↔sense。
 *    写法说明：用 CTE(VALUES) 把「词形对」映射为 sense_id，
 *      于是本文件完全不依赖任何具体 ID；WHERE 既做过滤也消除
 *      `INSERT ... SELECT ... ON CONFLICT` 的解析歧义。
 * ========================================================================= */

-- 9.1 is_a 层级（下位 → 上位）
/*    猫→动物→生物 构成 2 层链，是本 demo 验证传递闭包的主用例。
 *    → 触发 tr_closure_edge_ai 自动产出 depth=1 与 depth=2 的闭包行。
 *    → 方向约定（务必牢记）：边是 下位→上位；closure 行是 (ancestor=上位, descendant=下位)。 */
WITH e(child, parent) AS (VALUES
    ('猫','动物'),      -- 猫 is_a 动物
    ('鱼','动物'),      -- 鱼 is_a 动物
    ('人','动物'),      -- 人 is_a 动物（生物学分类；汉语口语习惯另论）
    ('动物','生物'),    -- 动物 is_a 生物
    ('鱼','食物')       -- 鱼 is_a 食物（部分鱼可食用 —— 多上位，体现 DAG 而非树）
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN seed_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN seed_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

-- 9.2 can_do 能力/典型行为（主体 → 动词义项）
/*    用途：支撑"猫会做什么"的常识问答与施事合法性校验。
 *    can_do 在 schema 中 is_transitive=1，会走闭包（策略 incremental）。 */
WITH e(subj, verb) AS (VALUES
    ('猫','抓'),        -- 猫 can_do 抓
    ('猫','吃'),        -- 猫 can_do 吃
    ('猫','跑'),        -- 猫 can_do 跑
    ('鱼','游泳'),      -- 鱼 can_do 游泳
    ('鱼','吃'),        -- 鱼 can_do 吃
    ('人','吃'),        -- 人 can_do 吃
    ('人','喝'),        -- 人 can_do 喝
    ('人','跑')         -- 人 can_do 跑
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', s.sid, 'sense', v.sid, 'can_do', 5
  FROM e
  JOIN seed_sid s ON s.word = e.subj AND s.pos = 'n'
  JOIN seed_sid v ON v.word = e.verb AND v.pos = 'v'
 WHERE s.sid <> v.sid
ON CONFLICT DO NOTHING;

-- 9.3 agent / patient 论元关系（动词 → 典型施事/受事**类别**）
/*    用途：这是需求中「吃 需要_施事 / 需要_受事」在关系层的落点，
 *          与 §10 的 relation_constraints 构成同一事实的两条正交路径：
 *            关系层 —— 供图遍历（"谁能吃"）
 *            约束层 —— 供解析期合法性校验（inference_rules #3/#4）
 *    参考 relation_types：agent='动词→典型施事类别'，patient='动词→典型受事类别' */
WITH e(verb, rel, obj) AS (VALUES
    ('吃','agent',  '动物'),      -- 吃的施事可以是动物
    ('吃','agent',  '人'),        -- 吃的施事可以是人
    ('吃','patient','食物'),      -- 吃的受事是食物
    ('喝','agent',  '人'),        -- 喝的施事
    ('喝','patient','水'),        -- 喝的受事（液体）
    ('抓','agent',  '动物'),      -- 抓的施事可以是动物
    ('抓','agent',  '人'),        -- 抓的施事可以是人
    ('看','agent',  '人')         -- 看的施事
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', v.sid, 'sense', o.sid, e.rel, 5
  FROM e
  JOIN seed_sid v ON v.word = e.verb AND v.pos = 'v'
  JOIN seed_sid o ON o.word = e.obj  AND o.pos = 'n'
 WHERE v.sid <> o.sid
ON CONFLICT DO NOTHING;

-- 9.4 opposite 反义（对称关系：只存一条，v_relations_all 自动双向展开）
/*    WHERE a.sid < b.sid 保证对称关系不重复存储（无冗余、无更新异常）。 */
WITH e(a, b) AS (VALUES
    ('大','小'),        -- antonym_gradable（相对反义）
    ('好','坏')         -- antonym_gradable
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', A.sid, 'sense', B.sid, 'opposite', 5
  FROM e
  JOIN seed_sid A ON A.word = e.a AND A.pos = 'a'
  JOIN seed_sid B ON B.word = e.b AND B.pos = 'a'
 WHERE A.sid < B.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 10. 论元选择限制（relation_constraints）
 *     用途：中文解析器判断「猫吃鱼」合法、而「鱼吃猫」可疑的依据。
 *           键值化存储（slot/kind/value）→ 可索引、可被规则引擎消费，
 *           优于 v2.0 把一串限制埋进逗号分隔字符串。
 *     slot 白名单：agent/patient/dative/instrument/location/time/result/target/theme
 *     kind 白名单：category/feature/sense/pos/classifier
 * ========================================================================= */
WITH e(verb, slot, kind, val) AS (VALUES
    ('吃','agent','category','动物'),      -- 吃的施事类别
    ('吃','agent','category','人'),        -- 吃的施事类别
    ('吃','patient','category','食物'),    -- 吃的受事类别（← 需求"需要受事")
    ('吃','instrument','category','器物'), -- 吃的工具（用筷子吃）—— instrument 槽示例
    ('喝','agent','category','人'),
    ('喝','agent','category','动物'),
    ('喝','patient','category','无机物'),  -- 喝的受事是液体/水
    ('抓','agent','category','动物'),
    ('抓','agent','category','人'),
    ('抓','patient','feature','+concrete'),-- 抓的受事须具体可感
    ('跑','agent','category','动物'),      -- 不及物：无 patient 槽
    ('跑','agent','category','人'),
    ('看','agent','category','人'),
    ('看','patient','feature','+concrete'),
    ('游泳','agent','category','动物'),
    ('游泳','agent','category','人'),
    ('是','theme','category','实体'),      -- 系事（判断句被说明的对象）
    ('有','agent','category','人'),        -- 领有者
    ('有','theme','feature','+concrete')   -- 存在物
)
INSERT INTO relation_constraints(sense_id, slot, kind, value)
SELECT v.sid, e.slot, e.kind, e.val
  FROM e
  JOIN seed_sid v ON v.word = e.verb AND v.pos = 'v'
 WHERE 1
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 11. 句式槽位约束（slot_constraints）—— demo 覆盖的 5 个句式
 *     句式：主谓宾(1) / 主谓谓语句·话题句(5) / 把字句(7) / 被字句(8) / 存现句(11)
 *     用途：槽位只声明"这里有个 S/V/O"是不够的 —— 还需声明**能填什么**。
 *           约束让解析器可以拒绝不合法的填充（如把字句的 P 必须有定）。
 *     前提：这 5 个句式的 pattern_slots 已由 schema_v3.sql §14 建好
 *           （32 个句式 80 个槽位，模板占位符与槽位一一对应，无"死表"）。
 *     kind 白名单：pos/role/feature/definiteness/animacy/countability/
 *                  classifier/aspect/subcategory/lexeme/marker
 * ========================================================================= */
INSERT INTO slot_constraints(pattern_id, slot_name, kind, value) VALUES
 -- ① 主谓宾：S 有生、V 是动词、O 是受事
 (1,'S','animacy','animate'),
 (1,'V','pos','v'),
 (1,'O','role','受事'),
 -- ② 主谓谓语句 / 话题句：话题与主语均为名词性，谓语是动词性
 (5,'Topic','pos','n'),          -- "鱼，猫吃了" 的 鱼
 (5,'S','pos','n'),              -- "鱼，猫吃了" 的 猫
 (5,'V','pos','v'),              -- 猫吃了"
 -- ③ 把字句：A 施事、P 受事且有定（"把书"≠"把一本书"）、V 动词、C 效果补语
 (7,'A','role','施事'),
 (7,'P','role','受事'),
 (7,'V','pos','v'),
 (7,'C','role','效果'),
 -- ④ 被字句：P 受事且有定、A 施事（有生）、V 动词、C 效果补语
 (8,'P','definiteness','definite'),
 (8,'P','role','受事'),
 (8,'V','pos','v'),
 (8,'C','role','效果'),
 -- ⑤ 存现句：L 处所、V 动词（次类=存现，schema 已建）、NP 为结果/存在物
 (11,'L','role','处所'),
 (11,'V','pos','v'),
 (11,'NP','role','结果')
ON CONFLICT DO NOTHING;
-- 已由 schema 建好、此处不重复：(7,'P','definiteness','definite') / (8,'A','feature','+animate')
--                              / (11,'L','pos','nl') / (11,'V','subcategory','存现')

/* =========================================================================
 * 12. 闭包表（closure）—— 只补 depth=1 的直接边
 *     关键：depth>1 的传递闭包**绝不手工插入**，由 tr_closure_edge_ai 触发器派生。
 *
 *     为什么这里还要显式写一次 depth=1？
 *       schema 中 closure_policy(is_a) 默认为 'incremental'，§9.1 插入 is_a 边时
 *       触发器已自动写出 depth=1 行。本语句因此通常是 **DO NOTHING 空操作**。
 *       但它是必要的"兜底"：若运维把 mode 切成了 'off'（或在触发器被禁用的导入
 *       流水线里运行本文件），显式补一次能保证 demo 的 depth=1 闭包必定存在。
 *       用 INSERT...SELECT 从 relations 反推 + 回填 first_edge（见证边），
 *       保证闭包行与关系边**永不脱钩**，且天然幂等。
 * ========================================================================= */
INSERT INTO closure(relation, ancestor, descendant, depth, first_edge)
SELECT r.relation, r.dst_id, r.src_id, 1, r.relation_id
  FROM relations r
 WHERE r.relation = 'is_a'
   AND r.src_kind = 'sense' AND r.dst_kind = 'sense'
ON CONFLICT(relation, ancestor, descendant) DO NOTHING;

/* =========================================================================
 * 13. demo 语料（utterances）—— 「猫吃鱼」及其 4 个扩展句式
 *     用途：为 v0.1 demo 提供可解析的输入，并标注其目标句式。
 *           doc_id 用 demo_cat_fish，与 schema 内置的 demo 文档隔离。
 * ========================================================================= */
INSERT INTO utterances(utterance_id, doc_id, seq, text, speaker, mood, pattern_id) VALUES
 (1001,'demo_cat_fish',1,'猫吃鱼。',        'A','陈述', 1),   -- 主谓宾（基础）
 (1002,'demo_cat_fish',2,'猫把鱼吃了。',    'A','陈述', 7),   -- 把字句（处置）
 (1003,'demo_cat_fish',3,'鱼被猫吃了。',    'A','陈述', 8),   -- 被字句（被动）
 (1004,'demo_cat_fish',4,'鱼，猫吃了。',    'A','陈述', 5),   -- 话题句（话题-述题）
 (1005,'demo_cat_fish',5,'桌上有一条鱼。',  'A','陈述',11)    -- 存现句（存在）
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 14. 附：种子自检与常用查询（注释形式，供应用层/validate_seed.py 调用）
 * ========================================================================= */

-- 14.1 【闭包方向自检】应返回 0 行。
--      同时验证"方向不变量 I3"与"传递闭包已生成"：
--      depth=1 行必须满足 closure.ancestor = relations.dst_id，
--      depth=2 行必须链出 生物→猫 这样的两级结果。
-- SELECT c.ancestor, c.descendant, c.depth
--   FROM closure c
--  WHERE c.relation='is_a' AND c.depth=1
--    AND NOT EXISTS (SELECT 1 FROM relations r
--                     WHERE r.relation='is_a' AND r.src_id=c.descendant
--                       AND r.dst_id=c.ancestor);

-- 14.2 【取某义项的全部上位】（猫 → 动物 → 生物）
-- SELECT a.ancestor, wa.word AS ancestor_word, a.depth
--   FROM closure a
--   JOIN words wa ON wa.word_id = (SELECT word_id FROM senses WHERE sense_id = a.ancestor)
--  WHERE a.relation='is_a'
--    AND a.descendant = (SELECT sid FROM seed_sid WHERE word='猫' AND pos='n')
--  ORDER BY a.depth;

-- 14.3 【取某类别下的全部下位】（生物 → 动物/猫/鱼/人）
-- SELECT c.descendant, wd.word AS descendant_word, c.depth
--   FROM closure c
--   JOIN words wd ON wd.word_id = (SELECT word_id FROM senses WHERE sense_id = c.descendant)
--  WHERE c.relation='is_a'
--    AND c.ancestor = (SELECT sid FROM seed_sid WHERE word='生物' AND pos='n')
--  ORDER BY c.depth, descendant_word;

-- 14.4 【属性/能力继承】closure 取上位 → 再取上位的能力（推理规则 #2 的同构查询）
-- SELECT DISTINCT dv.word AS inherited_verb
--   FROM closure a
--   JOIN relations r ON r.src_id = a.ancestor AND r.relation='can_do'
--   JOIN words dv ON dv.word_id = (SELECT word_id FROM senses WHERE sense_id = r.dst_id)
--  WHERE a.relation='is_a' AND a.descendant = (SELECT sid FROM seed_sid WHERE word='猫' AND pos='n');

-- 14.5 【量词推荐】先查显式搭配，再回退类别默认（完整版见 schema §15.7）
-- SELECT cw.word AS classifier, p.preference, 'explicit' AS src
--   FROM noun_classifier_pairs p
--   JOIN senses cs ON cs.sense_id = p.classifier_sense_id
--   JOIN words cw ON cw.word_id = cs.word_id
--  WHERE p.noun_sense_id = (SELECT sid FROM seed_sid WHERE word='鱼' AND pos='n');

-- 14.6 【虚词 → 语义角色】解析「把/被」时的规则来源
-- SELECT fw.category, fw.subcategory, fwr.role, fwr.position, fw.attaches_to
--   FROM function_words fw
--   JOIN function_word_roles fwr ON fwr.sense_id = fw.sense_id
--   JOIN senses s ON s.sense_id = fw.sense_id
--   JOIN words  w ON w.word_id = s.word_id
--  WHERE w.word IN ('把','被','在','从','向');

-- 14.7 【动词论元约束】判断「猫吃鱼」合法性的依据
-- SELECT rc.slot, rc.kind, rc.value
--   FROM relation_constraints rc
--  WHERE rc.sense_id = (SELECT sid FROM seed_sid WHERE word='吃' AND pos='v')
--  ORDER BY rc.slot, rc.kind, rc.value;

-- 14.8 【句式槽位与约束】demo 5 个句式的完整槽位视图
-- SELECT sp.pattern_id, sp.name, ps.position, ps.slot_name, ps.is_required,
--        ps.syntactic_function, ps.default_role,
--        (SELECT group_concat(sc.kind || '=' || sc.value, '; ')
--           FROM slot_constraints sc
--          WHERE sc.pattern_id = ps.pattern_id AND sc.slot_name = ps.slot_name) AS constraints
--   FROM sentence_patterns sp
--   JOIN pattern_slots ps ON ps.pattern_id = sp.pattern_id
--  WHERE sp.pattern_id IN (1,5,7,8,11)
--  ORDER BY sp.pattern_id, ps.position;

-- 14.9 【一致性巡检】以下每句正常应返回 0 行
-- (a) 悬空的闭包见证边（first_edge 指向已删除的关系）
-- SELECT * FROM closure c WHERE c.first_edge IS NOT NULL
--   AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.relation_id = c.first_edge);
-- (b) 闭包与在线可推导结果不一致（is_a 抽样）
-- SELECT c.ancestor, c.descendant FROM closure c
--  WHERE c.relation='is_a' AND c.depth=1
--    AND NOT EXISTS (SELECT 1 FROM relations r
--                     WHERE r.src_id=c.descendant AND r.relation='is_a' AND r.dst_id=c.ancestor);
-- (c) 名词-量词搭配指向不存在的量词义项
-- SELECT * FROM noun_classifier_pairs p
--  WHERE NOT EXISTS (SELECT 1 FROM classifiers c WHERE c.classifier_sense_id = p.classifier_sense_id);
-- (d) 有外键悬空的义项（FK 关闭时写入的脏数据）
-- SELECT * FROM senses s WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.word_id = s.word_id);

COMMIT;

/*****************************************************************************
 * End of seed_general_v3.sql
 *****************************************************************************/
