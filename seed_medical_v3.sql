/*****************************************************************************
 * Logos 医学库种子数据   seed_medical_v3.sql   (v0.1 demo)
 * 中文原生符号语义数据库 —— 医学通用骨架 + 跨库领域链接
 *
 * 目标：让「医学」在**统一义项空间**里长成一棵可被 is_a / has_part 遍历、
 *       可被闭包物化的骨架树，并通过 cross_links 挂回通用库概念。
 *
 *   医学概念
 *   ├── 解剖结构 → 神经系统 → 脑 ─┬─(has_part)→ 大脑 ──(has_part)→ 大脑皮层
 *   │                              └─(has_part)→ 小脑
 *   │                          └ 脊髓
 *   ├── 生理过程 → 神经传导 / 信号处理 / 记忆
 *   ├── 病理状态 → 疾病 → 脑部疾病 → 癫痫 / 脑卒中 / 神经退行性疾病 → 阿尔茨海默病
 *   │            └ 症状 → 抽搐 / 记忆障碍
 *   ├── 干预手段 → 药物治疗 / 神经调控 → 脑深部电刺激
 *   └── 医学检测 → 脑电图
 *
 * ---------------------------------------------------------------------------
 * 【前置依赖】必须按序执行：
 *     schema_v3.sql → seed_general_v3.sql → seed_general_v3_ext.sql → 本文件
 *   理由：
 *     · 「抽象概念」由 seed_general_v3_ext.sql §2.1 建立（医学概念 is_a 抽象概念）；
 *     · 「营养物」由 seed_general_v3_ext.sql §1.2 建立（营养素 is_a 营养物）；
 *     · 「鱼 / 吃」由 schema §14 与 seed_general_v3.sql 建立（跨库链接端点）。
 *   ★ 依赖缺失时本文件**硬失败**（见 §0），不静默降级 —— 理由见 §0 注释。
 *
 * 【幂等性】全部 INSERT 均带 ON CONFLICT DO NOTHING 或 WHERE NOT EXISTS，
 *   可重复执行任意次，结果一致（不重复写 FTS 与 changelog）。
 *
 * 【ID 分配策略】本文件独占 4000-4999 段，与已用段位完全隔离：
 *     schema 1-27 ／ seed 1000-1027 ／ ext 2000-2999 ／ 本文件 4000-4999 ／ 测试夹具 7000+
 *     4001-4030  医学义项 30 个（word_id = sense_id，v0.1 每词单义项）
 *     4101       「信号」——通用库兜底（见 §1.2 说明，实测通用库缺此词）
 *     3101-3103  医学库领域实体节点（cross_links 端点；ext 已占 3001）
 *
 * ---------------------------------------------------------------------------
 * 【写入规模】（下表为实测增量：schema → seed → ext → 本文件，由 validate_medical.py 复核）
 *   relation_types        +11   4 对互逆（has_symptom/symptom_of、treated_by/treats、
 *                               responsible_for/depends_on、indicates/indicated_by）
 *                               + 3 个领域链接类型（nutritional_relevance /
 *                               functional_analogy / domain_specialization）
 *   words                 +31   医学 30 + 信号 1（通用库兜底）
 *   senses                +31   与上表 1:1
 *   sense_categories      +31   抽象 4 / 实体 8 / 事件 9 / 性状 10
 *   relations             +40   28 is_a（骨架树）+ 3 has_part + 6 医学关系
 *                               + 3 跨库领域边（sense → node）
 *   entity_nodes           +3   医学库 营养素 / 药物治疗 / 脑电图信号
 *   closure               +89   全部**自动派生**：is_a 与 has_part 均为 incremental
 *                               策略，由 tr_closure_edge_ai 写出，
 *                               本文件不含任何 closure 的 INSERT（实测 0 条）
 *   changelog             +40   由 tr_relations_audit_ai 记录，同样自动产生
 *   合计：ext 后 words/senses = 92/93 → 123/124；relations = 61 → 101；
 *         closure = 95 → 184；entity_nodes = 1 → 4；relation_types = 32 → 43
 *
 * 【依赖顺序】（严格按此顺序，不可调换）
 *   relation_types ──→ relations（FK: relation 必须已登记）
 *   words → senses ─┬─→ sense_categories
 *                   ├─→ relations ─┬─→ 触发 tr_closure_edge_ai（is_a / has_part）
 *                   │              └─→ entity_nodes → relations(node 端点)
 *                   └─→ （本文件不写 relation_constraints / 量词 / 义素，见 §4 说明）
 *
 * 【五处刻意的设计取舍 —— 请留意】
 *   (1) 【关系类型 vs 复用 domain_link】需求给了三个具体链接类型
 *       （nutritional_relevance / functional_analogy / domain_specialization）。
 *       本文件把它们登记为**独立的 relation_types**（relation_class='领域'），
 *       而不是塞进 domain_link 的某个字段 —— schema 的 relations 表没有子类型列，
 *       若要保留类型语义只能新建类型。三者一律 is_transitive=0，原因见 (2)。
 *   (2) 【为什么新类型都不设 is_transitive=1】闭包物化触发器的条件是
 *       `src_kind='sense' AND dst_kind='sense' AND is_transitive=1 AND policy='incremental'`。
 *       跨库边一端是 node，永远不满足前两个条件；而本项目记录过一个真实坑：
 *       **is_transitive=1 却缺 closure_policy 行 → 子查询返回 NULL → 静默不物化**。
 *       故本文件对"不构成可传递层级"的关系一律显式声明 is_transitive=0，
 *       从源头消除这个坑，而不是补一堆用不上的策略行。
 *   (3) 【has_symptom 与 indicates 并存不是冗余】二者立场不同：
 *       has_symptom（疾病→症状）是**定义性**关联，"AD 有此症状"；
 *       indicates（症状→疾病）是**诊断性**指向，"此症状提示该病"。
 *       方向和语义角色都不同，故各自保留（且互为不同关系类型，不是逆关系）。
 *   (4) 【大脑皮层是为演示 has_part 传递闭包而增设的节点】需求原文的解剖层只到
 *       「脑 → 大脑/小脑」一层，那样 has_part 只会产出 depth=1 闭包。
 *       补「大脑 has_part 大脑皮层」后，`脑 has_part 大脑皮层` 会由触发器派生出
 *       **depth=2** 的行 —— 这是全库唯一一条**非 is_a** 的 depth>1 物化演示。
 *       解剖学上 part_of 对连续结构传递（大脑皮层 part_of 大脑 part_of 脑 ⇒ 成立），
 *       故该行正确。★ 此节点非需求原文列出，如需最小化可整块删除（§2.2 第 3 行）。
 *   (5) 【义素 / 量词 / 论元约束本文件不写】三者都不是本次需求项，且写入会改变
 *       解析器选择限制的判定面（relation_constraints 直接参与「受事合法性」规则）。
 *       留待需要时单独补，见 §4 的扩展建议。
 *
 * 【连接级 PRAGMA】必须在每个连接上执行，不随 schema 持久化。
 *   注意：PRAGMA foreign_keys 在**显式事务内是 no-op**，故写在 BEGIN 之前。
 * =========================================================================*/

PRAGMA foreign_keys     = ON;   -- SQLite 默认 OFF；不开则所有 FK 形同虚设
PRAGMA recursive_triggers = ON; -- 闭包触发器依赖触发器级联语义

BEGIN;

/* =========================================================================
 * 0. 前置依赖硬检查（不满足即中止，不回退为"能跑但断链"）
 * -------------------------------------------------------------------------
 * 为什么不用 `WHERE NOT EXISTS` 兜底（ext 用的是兜底）：
 *   ext 兜底的是「食物 / 生物」这类**词条本身**，缺了就补一条，语义完整。
 *   本文件依赖的是「抽象概念」——医学概念的唯一上位。若它缺失而静默跳过，
 *   整棵医学树会**悬空成孤岛**（医学概念无上位、闭包链断在根上），
 *   而且 SQL 本身不会报错。这类"看起来成功、语义却断裂"的结果最难排查，
 *   故这里选择**显式失败**。
 *
 * 实现手法：SQLite 的 RAISE() 只能出现在触发器体内，脚本里无法直接断言。
 *   改为建一张临时表 + **具名 CHECK 约束**，失败时异常信息会带上**约束名**
 *   （实测 SQLite 3.53.1：报了约束名而非表达式，故把提示写进约束名本身）：
 *     IntegrityError: CHECK constraint failed: _prereq_missing_...
 *   临时表 + IF NOT EXISTS + DELETE 保证本段自身可重复执行。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS _prereq_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _prereq_missing_run_seed_general_v3_and_ext_first CHECK (ok = 1)
);
DELETE FROM _prereq_check;
INSERT INTO _prereq_check(ok)
SELECT CASE WHEN
        EXISTS (SELECT 1 FROM words WHERE word = '抽象概念')   -- ext §2.1
    AND EXISTS (SELECT 1 FROM words WHERE word = '营养物')     -- ext §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '鱼')         -- schema §14
    AND EXISTS (SELECT 1 FROM words WHERE word = '吃')         -- seed_general_v3.sql
     THEN 1 ELSE 0 END;
DROP TABLE _prereq_check;

/* =========================================================================
 * 0.1 自然键查找视图（仅本会话有效，不落盘）
 *     与 seed_general_v3.sql 的 seed_sid、ext 的 ext_sid 同构，**另起名字**：
 *     三者可能被同一连接先后加载，重名会让后者的 CREATE 失效
 *     （IF NOT EXISTS 命中旧定义）。
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS med_sid AS
    SELECT w.word AS word, s.pos AS pos, s.sense_no AS sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;

/* ###########################################################################
 * 1. 词层与义项层
 * ######################################################################### */

/* =========================================================================
 * 1.1 关系类型登记（11 个）
 *     必须排在 relations 之前 —— relations.relation 有 FK 指向 relation_types。
 *     逆关系成对声明后，v_relations_all 视图会**在线推导**反向边，
 *     因此不必（也不应）冗余存储反向行（schema §11 的设计意图）。
 * ========================================================================= */
INSERT INTO relation_types
 (relation, relation_class, direction, inverse_of, is_transitive, is_symmetric,
  is_reflexive, is_functional, max_depth, description, color_code) VALUES
 -- ── 医学：症状与诊疗（4 对互逆）─────────────────────────────────────────
 ('has_symptom',   '属性','directed','symptom_of',   0,0,0,0, 4,'疾病→症状（定义性关联）','#DB2777'),
 ('symptom_of',    '属性','directed','has_symptom',  0,0,0,1, 4,'症状→所属疾病','#F472B6'),
 ('treated_by',    '事件','directed','treats',       0,0,0,0, 4,'疾病→干预手段','#0891B2'),
 ('treats',        '事件','directed','treated_by',   0,0,0,1, 4,'干预手段→适用疾病','#22D3EE'),
 ('responsible_for','属性','directed','depends_on',  0,0,0,0, 4,'结构→其所负责的功能','#7C3AED'),
 ('depends_on',    '属性','directed','responsible_for',0,0,0,1,4,'功能→所依赖的结构','#A78BFA'),
 ('indicates',     '事件','directed','indicated_by', 0,0,0,0, 4,'症状/检查→提示的疾病（诊断指向）','#EA580C'),
 ('indicated_by',  '事件','directed','indicates',    0,0,0,1, 4,'疾病→提示它的症状/检查','#FB923C'),
 -- ── 跨库领域链接（3 个，无逆关系）────────────────────────────────────────
 ('nutritional_relevance','领域','directed',NULL,0,0,0,0,4,'通用概念→营养学相关医学概念','#65A30D'),
 ('functional_analogy',   '领域','directed',NULL,0,0,0,0,4,'通用概念与医学概念的功能类比','#CA8A04'),
 ('domain_specialization','领域','directed',NULL,0,0,0,0,4,'通用概念在医学领域的特化形态','#0D9488')
ON CONFLICT(relation) DO NOTHING;

/* =========================================================================
 * 1.2 词层（31 个：医学 30 + 通用库兜底 1）
 * -------------------------------------------------------------------------
 * ★ 「信号」（4101）为什么由**医学库文件**来建：
 *   需求要求「通用库"信号" → 医学库"脑电图信号"」这条跨库链接。
 *   实测：seed_general_v3.sql 与 seed_general_v3_ext.sql **都没有「信号」**
 *   （grep 全库无此词）。若直接连边，端点不存在 —— 触发器 tr_relations_validate_ai
 *   会直接 RAISE 中止整个加载。
 *   两个选项：(a) 静默跳过该链接；(b) 按"若不存在则建立"补一个核心库词条。
 *   选 (b)：需求的语义是"两个概念之间有关系"，跳过会让需求 3 之一消失而无任何提示。
 *   故此处用 WHERE NOT EXISTS 兜底补建，并**在此显式标注它是通用库词条**，
 *   以便将来通用库自己扩充「信号」时能发现重复并合并（UNIQUE(word) 会直接拦下）。
 * ========================================================================= */
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 -- 顶层
 (4001,'医学概念',NULL,'合成词',0,'通用'),
 -- 解剖结构
 (4002,'解剖结构',NULL,'合成词',0,'通用'),
 (4003,'神经系统',NULL,'合成词',0,'通用'),
 (4004,'脑',NULL,'单纯词',0,'通用'),
 (4005,'脊髓',NULL,'合成词',0,'通用'),
 (4006,'大脑',NULL,'合成词',0,'通用'),
 (4007,'小脑',NULL,'合成词',0,'通用'),
 (4008,'大脑皮层',NULL,'合成词',0,'通用'),   -- ★ 见文件头取舍 (4)
 -- 生理过程
 (4009,'生理过程',NULL,'合成词',0,'通用'),
 (4010,'神经传导',NULL,'合成词',0,'通用'),
 (4011,'信号处理',NULL,'合成词',0,'通用'),
 (4012,'记忆',NULL,'单纯词',0,'通用'),
 -- 病理状态
 (4013,'病理状态',NULL,'合成词',0,'通用'),
 (4014,'疾病',NULL,'单纯词',0,'通用'),
 (4015,'脑部疾病',NULL,'合成词',0,'通用'),
 (4016,'癫痫',NULL,'单纯词',0,'通用'),
 (4017,'脑卒中',NULL,'合成词',0,'通用'),
 (4018,'神经退行性疾病',NULL,'合成词',0,'通用'),
 (4019,'阿尔茨海默病',NULL,'音译词',0,'通用'),
 (4020,'症状',NULL,'单纯词',0,'通用'),
 (4021,'抽搐',NULL,'合成词',0,'通用'),
 (4022,'记忆障碍',NULL,'合成词',0,'通用'),
 -- 干预与检测
 (4023,'干预手段',NULL,'合成词',0,'通用'),
 (4024,'药物治疗',NULL,'合成词',0,'通用'),
 (4025,'神经调控',NULL,'合成词',0,'通用'),
 (4026,'脑深部电刺激',NULL,'合成词',0,'通用'),
 (4027,'医学检测',NULL,'合成词',0,'通用'),
 (4028,'脑电图',NULL,'合成词',0,'通用'),
 -- 营养与信号
 (4029,'营养素',NULL,'合成词',0,'通用'),
 (4030,'脑电图信号',NULL,'合成词',0,'通用')
ON CONFLICT(word_id) DO NOTHING;

-- 通用库兜底：信号（见上）
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register)
SELECT 4101,'信号',NULL,'合成词',0,'通用'
 WHERE NOT EXISTS (SELECT 1 FROM words WHERE word = '信号');

/* =========================================================================
 * 1.3 义项层 —— 与 1.2 一一对应（sense_id = word_id）
 *     释义一律用中文词语组合（符号语义：用词定义词，不引入外部向量）。
 * ========================================================================= */
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary) VALUES
 (4001,4001,'n',1,'研究人体结构、功能与疾病的一类知识的总称',
  '研究 人体 结构 功能 与 疾病 的 一类 知识 的 总称',
  'yi1 xue2 gai4 nian4','yixuegainian','yxgn',2600,'本义',NULL,1),
 (4002,4002,'n',1,'生物体内部由若干组织器官构成的形态结构',
  '生物体 内部 由 若干 组织 器官 构成 的 形态 结构',
  'jie3 pou1 jie2 gou4','jiepoujiegou','jpjg',1400,'本义',NULL,1),
 (4003,4003,'n',1,'由神经细胞构成、传导信号并调控全身活动的系统',
  '由 神经 细胞 构成 传导 信号 并 调控 全身 活动 的 系统',
  'shen2 jing1 xi4 tong3','shenjingxitong','sjxt',1900,'本义',NULL,1),
 (4004,4004,'n',1,'位于颅腔内、主管感觉运动与思维的中枢器官',
  '位于 颅腔 内 主管 感觉 运动 与 思维 的 中枢 器官',
  'nao3','nao','n',2900,'本义',NULL,1),
 (4005,4005,'n',1,'位于脊柱内、连接脑与周围神经的神经组织',
  '位于 脊柱 内 连接 脑 与 周围 神经 的 神经 组织',
  'ji3 sui3','jisui','js',1300,'本义',NULL,1),
 (4006,4006,'n',1,'脑的前部，负责思维、记忆与随意运动的高级中枢',
  '脑 的 前部 负责 思维 记忆 与 随意 运动 的 高级 中枢',
  'da4 nao3','danao','dn',2400,'本义',NULL,1),
 (4007,4007,'n',1,'位于脑的后下方，主管平衡与动作协调的部分',
  '位于 脑 的 后 下方 主管 平衡 与 动作 协调 的 部分',
  'xiao3 nao3','xiaonao','xn',1500,'本义',NULL,1),
 (4008,4008,'n',1,'大脑表面的灰质层，是高级神经活动的主要部位',
  '大脑 表面 的 灰质 层 是 高级 神经 活动 的 主要 部位',
  'da4 nao3 pi2 ceng2','danaopiceng','dnpc',900,'本义',NULL,1),
 (4009,4009,'n',1,'生物体内有规律进行的机能活动过程',
  '生物体 内 有 规律 进行 的 机能 活动 过程',
  'sheng1 li3 guo4 cheng2','shengliguocheng','slgc',1700,'本义',NULL,1),
 (4010,4010,'n',1,'神经纤维上电信号由一处传向另一处的过程',
  '神经 纤维 上 电 信号 由 一处 传向 另一处 的 过程',
  'shen2 jing1 chuan2 dao3','shenjingchuandao','sjcd',800,'本义',NULL,1),
 (4011,4011,'n',1,'对信号进行接收、转换与传递的处理过程',
  '对 信号 进行 接收 转换 与 传递 的 处理 过程',
  'xin4 hao4 chu3 li3','xinhaochuli','xhcl',1100,'本义',NULL,1),
 (4012,4012,'n',1,'对经历过的事物能够再现和再认的高级神经活动',
  '对 经历 过 的 事物 能够 再现 和 再认 的 高级 神经 活动',
  'ji4 yi4','jiyi','jy',2700,'本义',NULL,1),
 (4013,4013,'n',1,'机体在致病因素作用下偏离正常机能的状态',
  '机体 在 致病 因素 作用 下 偏离 正常 机能 的 状态',
  'bing4 li3 zhuang4 tai4','binglizhuangtai','blzt',1600,'本义',NULL,1),
 (4014,4014,'n',1,'机体在一定病因作用下发生的异常生命活动过程',
  '机体 在 一定 病因 作用 下 发生 的 异常 生命 活动 过程',
  'ji2 bing4','jibing','jb',5200,'本义',NULL,1),
 (4015,4015,'n',1,'发生于脑及其血管、神经的疾病的总称',
  '发生 于 脑 及其 血管 神经 的 疾病 的 总称',
  'nao3 bu4 ji2 bing4','naobujibing','nbjb',1200,'本义',NULL,1),
 (4016,4016,'n',1,'脑神经元异常同步放电引起反复发作的脑部疾病',
  '脑 神经元 异常 同步 放电 引起 反复 发作 的 脑部 疾病',
  'dian1 xian2','dianxian','dx',1000,'本义',NULL,1),
 (4017,4017,'n',1,'脑部血管突然破裂或阻塞导致脑组织损伤的疾病',
  '脑部 血管 突然 破裂 或 阻塞 导致 脑 组织 损伤 的 疾病',
  'nao3 cu4 zhong4','naocuzhong','ncz',950,'本义',NULL,1),
 (4018,4018,'n',1,'以神经元进行性退化变性为特征的一类脑部疾病',
  '以 神经元 进行性 退化 变性 为 特征 的 一类 脑部 疾病',
  'shen2 jing1 tui4 xing2 xing4 ji2 bing4','shenjingtuixingxingjibing','sjtxxjb',700,'本义',NULL,1),
 (4019,4019,'n',1,'以记忆减退和认知功能全面衰退为主要表现的神经退行性疾病',
  '以 记忆 减退 和 认知 功能 全面 衰退 为 主要 表现 的 神经 退行性 疾病',
  'a1 er3 ci2 hai3 mo4 bing4','aercihaimobing','aechmb',880,'本义',NULL,1),
 (4020,4020,'n',1,'疾病过程中病人主观感到的异常感觉或客观检查到的异常表现',
  '疾病 过程 中 病人 主观 感到 的 异常 感觉 或 客观 检查 到 的 异常 表现',
  'zheng4 zhuang4','zhengzhuang','zz',2100,'本义',NULL,1),
 (4021,4021,'n',1,'肌肉不自主地快速收缩的现象',
  '肌肉 不 自主 地 快速 收缩 的 现象',
  'chou1 chu4','chouchu','cc',850,'本义',NULL,1),
 (4022,4022,'n',1,'记忆的形成、保持或提取发生障碍的现象',
  '记忆 的 形成 保持 或 提取 发生 障碍 的 现象',
  'ji4 yi4 zhang4 ai4','jiyizhangai','jyza',760,'本义',NULL,1),
 (4023,4023,'n',1,'为改善病情而采取的措施与方法的统称',
  '为 改善 病情 而 采取 的 措施 与 方法 的 统称',
  'gan1 yu4 shou3 duan4','ganyushouduan','gysd',1050,'本义',NULL,1),
 (4024,4024,'n',1,'用药物改善或控制病情的干预手段',
  '用 药物 改善 或 控制 病情 的 干预 手段',
  'yao4 wu4 zhi4 liao2','yaowuzhiliao','ywzl',2300,'本义',NULL,1),
 (4025,4025,'n',1,'用物理刺激调节神经活动以改善症状的干预手段',
  '用 物理 刺激 调节 神经 活动 以 改善 症状 的 干预 手段',
  'shen2 jing1 tiao2 kong4','shenjingtiaokong','sjtk',820,'本义',NULL,1),
 (4026,4026,'n',1,'将电极植入脑深部施加电刺激的神经调控手段',
  '将 电极 植入 脑 深部 施加 电 刺激 的 神经 调控 手段',
  'nao3 shen1 bu4 dian4 ci4 ji1','naoshenbudianciji','nsbdcj',600,'本义',NULL,1),
 (4027,4027,'n',1,'用于查明疾病与机体状况的检查方法的总称',
  '用于 查明 疾病 与 机体 状况 的 检查 方法 的 总称',
  'yi1 xue2 jian3 ce4','yixuejiance','yxjc',1300,'本义',NULL,1),
 (4028,4028,'n',1,'在头皮记录脑电活动信号的医学检测方法',
  '在 头皮 记录 脑 电 活动 信号 的 医学 检测 方法',
  'nao3 dian4 tu2','naodiantu','ndt',1150,'本义',NULL,1),
 (4029,4029,'n',1,'食物中可供机体利用以维持生命活动的成分',
  '食物 中 可 供 机体 利用 以 维持 生命 活动 的 成分',
  'ying2 yang3 su4','yingyangsu','yys',1800,'本义',NULL,1),
 (4030,4030,'n',1,'脑电活动经检测记录下来的信号',
  '脑 电 活动 经 检测 记录 下来 的 信号',
  'nao3 dian4 tu2 xin4 hao4','naodiantuxinhao','ndtxh',640,'本义',NULL,1)
ON CONFLICT(sense_id) DO NOTHING;

-- 通用库兜底：信号（与 1.2 的 4101 一一对应）
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary)
SELECT 4101,4101,'n',1,'用来传递信息或表示某种意义的载体',
       '用来 传递 信息 或 表示 某种 意义 的 载体',
       'xin4 hao4','xinhao','xh',3100,'本义',NULL,1
 WHERE NOT EXISTS (SELECT 1 FROM senses s JOIN words w ON w.word_id = s.word_id
                    WHERE w.word = '信号' AND s.pos = 'n');

/* =========================================================================
 * 1.4 义项 ↔ 语义类别
 *     映射原则：本文件**不新增** semantic_categories 行，只把医学节点挂到
 *     已有的粗类上（与 ext 的取舍一致：类别层与义项层是两套正交体系，
 *     骨架要能被 is_a 遍历就必须落在义项层；类别层只做粗粒度归档）。
 *       抽象 —— 概念/手段/信号        实体 —— 解剖结构与物质
 *       事件 —— 过程/动作/检测        性状 —— 状态与症状
 * ========================================================================= */
WITH e(word, cat) AS (VALUES
    -- 抽象
    ('医学概念','抽象'),('干预手段','抽象'),('信号','抽象'),('脑电图信号','抽象'),
    -- 实体
    ('解剖结构','实体'),('神经系统','实体'),('脑','实体'),('脊髓','实体'),
    ('大脑','实体'),('小脑','实体'),('大脑皮层','实体'),('营养素','实体'),
    -- 事件
    ('生理过程','事件'),('神经传导','事件'),('信号处理','事件'),('记忆','事件'),
    ('医学检测','事件'),('脑电图','事件'),('药物治疗','事件'),
    ('神经调控','事件'),('脑深部电刺激','事件'),
    -- 性状
    ('病理状态','性状'),('疾病','性状'),('脑部疾病','性状'),('癫痫','性状'),
    ('脑卒中','性状'),('神经退行性疾病','性状'),('阿尔茨海默病','性状'),
    ('症状','性状'),('抽搐','性状'),('记忆障碍','性状')
)
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT s.sid, e.cat, 1
  FROM e JOIN med_sid s ON s.word = e.word AND s.pos = 'n'
 WHERE 1
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 2. 关系层
 * ######################################################################### */

/* =========================================================================
 * 2.1 骨架 is_a 关系链（28 条边）
 *     写法与 ext §2.5 一致：CTE(VALUES) 把「词形对」映射为 sense_id，
 *     本文件**不硬编码任何既有 sense_id**（除自建节点外），因此对既有 ID
 *     取值不敏感；父节点不存在时内连接自动跳过该行（不报错、不产生悬空边）。
 *
 *     插边后由 tr_closure_edge_ai 自动派生多级闭包（实测深度）：
 *       癫痫 → 疾病(1) / 脑部疾病(1) / 病理状态(2) / 医学概念(3) / 抽象概念(4)
 *       阿尔茨海默病 → 神经退行性疾病(1) → 脑部疾病(2) → 疾病(3)
 *                      → 病理状态(4) → 医学概念(5) → 抽象概念(6)
 *       ★ 癫痫 → 疾病 同时存在**直接边(1)**与**经脑部疾病(2)**两条路径，
 *         UPSERT 取 MIN(depth) → 闭包只留 depth=1，这是多继承的预期表现。
 *     ⚠ 边序无关紧要：触发器同时向"更具体"与"更一般"两侧递归已有闭包，
 *       故任意插入顺序都能得到与全量重建一致的结果。
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    -- 顶层：医学概念挂到 ext 建的「抽象概念」下
    ('医学概念',  '抽象概念'),
    -- 解剖结构分支
    ('解剖结构',  '医学概念'),
    ('神经系统',  '解剖结构'),
    ('脑',        '神经系统'),
    ('脊髓',      '神经系统'),
    -- 生理过程分支
    ('生理过程',  '医学概念'),
    ('神经传导',  '生理过程'),
    ('信号处理',  '生理过程'),
    ('记忆',      '生理过程'),          -- 记忆建模为高级神经活动（生理过程子类）
    -- 病理状态分支
    ('病理状态',  '医学概念'),
    ('疾病',      '病理状态'),
    ('脑部疾病',  '疾病'),
    ('癫痫',      '疾病'),              -- ★ 多继承 A：癫痫 is_a 疾病
    ('癫痫',      '脑部疾病'),          -- ★ 多继承 B：癫痫 is_a 脑部疾病
    ('脑卒中',    '脑部疾病'),
    ('神经退行性疾病','脑部疾病'),
    ('阿尔茨海默病','神经退行性疾病'),
    ('症状',      '病理状态'),
    ('抽搐',      '症状'),
    ('记忆障碍',  '症状'),
    -- 干预手段分支
    ('干预手段',  '医学概念'),
    ('药物治疗',  '干预手段'),
    ('神经调控',  '干预手段'),
    ('脑深部电刺激','神经调控'),
    -- 医学检测分支
    ('医学检测',  '医学概念'),
    ('脑电图',    '医学检测'),
    -- 与通用库概念接续（跨库层级：义项层是统一空间，此处是普通 is_a）
    ('营养素',    '营养物'),            -- 营养物 由 ext §1.2 建立
    ('脑电图信号','信号')               -- 信号 由本文件 §1.2 兜底建立
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN med_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN med_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.2 has_part 关系链（3 条边）—— 整体 → 部分
 *     has_part 的闭包策略是 incremental / max_depth=8（schema §14 已登记），
 *     故这三条边会派生出**跨层**的部分关系。
 *
 *     ⚠⚠ 方向提示（容易读反，务必先看）：
 *       closure 的行方向**恒为「边的反向」**（不变量 I3：ancestor = 边的 dst）。
 *       has_part 的边是「整体 →(has_part) 部分」，dst 是**部分**，
 *       所以闭包行的 ancestor 落在**部分**一侧、descendant 落在**整体**一侧：
 *         (ancestor=大脑,     descendant=脑,   depth=1)   -- 读作：大脑 是 脑 的部分
 *         (ancestor=小脑,     descendant=脑,   depth=1)
 *         (ancestor=大脑皮层, descendant=大脑, depth=1)
 *         (ancestor=大脑皮层, descendant=脑,   depth=2)  ← 全库唯一非 is_a 的 depth>1 行
 *       即：查「整体 X 由哪些部分构成」→ WHERE descendant = X  （返回 ancestor）
 *           查「部分 Y 属于哪些整体」  → WHERE ancestor   = Y  （返回 descendant）
 *       —— 所以 closure.py 的 get_ancestors(脑, 'has_part') 实际返回的是
 *          「脑的组成部分」。列名 ancestor/descendant 是 is_a 视角的命名，
 *          对 has_part 而言语义正好互换。若改用逆关系 part_of（部分→整体）存储，
 *          ancestor 就恰好是「整体」，与 is_a 直觉一致；
 *          本文件按需求原文用 has_part，故在此显式标注，避免下游读反。
 * ========================================================================= */
WITH e(whole, part) AS (VALUES
    ('脑',   '大脑'),
    ('脑',   '小脑'),
    ('大脑', '大脑皮层')     -- ★ 非需求原文节点；删除本行则 has_part 只剩 depth=1
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', w.sid, 'sense', p.sid, 'has_part', 5
  FROM e
  JOIN med_sid w ON w.word = e.whole AND w.pos = 'n'
  JOIN med_sid p ON p.word = e.part  AND p.pos = 'n'
 WHERE w.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.3 医学关系（6 条边）
 *     一律 is_transitive=0（见文件头取舍 (2)），因此**不产生闭包行** ——
 *     这类关系靠 is_a 继承在规则引擎里传播，而不是靠自身传递：
 *       「阿尔茨海默病 has_symptom 记忆障碍」经 is_a 上位即可被
 *       「脑部疾病 / 疾病」继承（推理规则 #2 的属性继承路径）。
 * ========================================================================= */
WITH e(src, rel, dst) AS (VALUES
    -- 疾病 → 症状（定义性关联）
    ('癫痫',        'has_symptom',   '抽搐'),
    ('阿尔茨海默病','has_symptom',   '记忆障碍'),
    -- 疾病 → 干预手段
    ('癫痫',        'treated_by',    '神经调控'),
    ('癫痫',        'treated_by',    '药物治疗'),
    -- 结构 → 功能
    ('大脑',        'responsible_for','记忆'),
    -- 症状 → 疾病（诊断指向；与 has_symptom 方向相反、语义立场不同）
    ('记忆障碍',    'indicates',     '阿尔茨海默病')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
SELECT 'sense', s.sid, 'sense', d.sid, e.rel, 5, 0.9
  FROM e
  JOIN med_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN med_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.4 医学库领域实体节点（entity_nodes）
 *     为什么跨库链接必须有这一步：
 *       schema 的 cross_links 是**视图**：
 *         SELECT ... FROM relations WHERE src_kind='node' OR dst_kind='node'
 *       即"跨库链接"= relations 里一条**端点带 node** 的边。
 *       而 node 端点必须先在 entity_nodes 注册（tr_relations_validate_ai 会校验），
 *       故领域库要以"库"的身份被链接，就得登记节点。
 *     domain='medical' 由 schema §14 domains 建好（无需新建）。
 *     节点 ID 用 3101+：ext 已占 3001（动物学库 Fish）。
 * ========================================================================= */
INSERT INTO entity_nodes(node_id, node_type, domain, label, entity_class, attrs_json) VALUES
 (3101,'domain_entity','medical','营养素','Nutrient',
  '{"library":"医学库","linked_word":"营养素","facet":"nutrition"}'),
 (3102,'domain_entity','medical','药物治疗','Therapy',
  '{"library":"医学库","linked_word":"药物治疗","facet":"intervention"}'),
 (3103,'domain_entity','medical','脑电图信号','Biosignal',
  '{"library":"医学库","linked_word":"脑电图信号","facet":"signal"}')
ON CONFLICT(node_id) DO NOTHING;

/* =========================================================================
 * 2.5 cross_links —— 通用库 ↔ 医学库（3 条领域边）
 *     source 端是**通用库义项**（sense），target 端是**医学库领域节点**（node），
 *     因此每条都能被 cross_links 视图读出。
 *     三条边的类型各自表达不同关系（不是同一个 domain_link 的三份拷贝）：
 *       鱼   --nutritional_relevance--> 营养素      营养学相关性
 *       吃   --functional_analogy-----> 药物治疗    功能类比（"摄入" vs "给药"）
 *       信号 --domain_specialization--> 脑电图信号  领域特化（一般信号 → 医学形态）
 *     注 1：**源词不一定是名词**。「吃」在通用库是动词（pos='v'），
 *           故 CTE 显式带上 src_pos —— 若照抄 ext §1.9 只按 pos='n' 连接，
 *           这条边会被内连接**静默丢掉**（不报错、只是少一条），
 *           实测踩到过：cross_links 只出 2 行而非 3 行。
 *     注 2：面向 node 的边不参与闭包物化（触发器要求 sense↔sense），
 *         这三个类型也已显式设为 is_transitive=0，行为一致、无隐式默认。
 * ========================================================================= */
WITH e(src_word, src_pos, node_id, rel) AS (VALUES
    ('鱼',   'n', 3101,'nutritional_relevance'),
    ('吃',   'v', 3102,'functional_analogy'),
    ('信号', 'n', 3103,'domain_specialization')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
SELECT 'sense', s.sid, 'node', e.node_id, e.rel, 5, 0.8
  FROM e
  JOIN med_sid s ON s.word = e.src_word AND s.pos = e.src_pos
  JOIN entity_nodes n ON n.node_id = e.node_id
 WHERE 1
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3. 附：自检与常用查询（注释形式，供应用层 / validate_medical.py 调用）
 * ========================================================================= */

-- 3.1 【骨架自检】医学概念的全部下位（含多级），按深度排序
-- SELECT w.word, c.depth FROM closure c
--   JOIN senses s ON s.sense_id = c.descendant
--   JOIN words  w ON w.word_id = s.word_id
--  WHERE c.relation='is_a'
--    AND c.ancestor = (SELECT s.sense_id FROM senses s JOIN words w2 ON w2.word_id=s.word_id
--                       WHERE w2.word='医学概念' AND s.pos='n')
--  ORDER BY c.depth, w.word;
-- 期望：30 个节点（不含医学概念自身）

-- 3.2 【多继承自检】癫痫 的 is_a 出边应恰为 2 条（疾病 / 脑部疾病）
-- SELECT w.word FROM relations r JOIN senses s ON s.sense_id=r.dst_id
--   JOIN words w ON w.word_id=s.word_id
--  WHERE r.relation='is_a' AND r.src_kind='sense'
--    AND r.src_id=(SELECT s.sense_id FROM senses s JOIN words w2 ON w2.word_id=s.word_id
--                   WHERE w2.word='癫痫' AND s.pos='n');
-- 期望：疾病 / 脑部疾病

-- 3.3 【闭包自动派生自检】阿尔茨海默病 的上位链应达 6 跳（含抽象概念）
-- SELECT w.word, c.depth FROM closure c
--   JOIN senses s ON s.sense_id = c.ancestor JOIN words w ON w.word_id = s.word_id
--  WHERE c.relation='is_a'
--    AND c.descendant=(SELECT s.sense_id FROM senses s JOIN words w2 ON w2.word_id=s.word_id
--                       WHERE w2.word='阿尔茨海默病' AND s.pos='n')
--  ORDER BY c.depth;
-- 期望：神经退行性疾病(1) 脑部疾病(2) 疾病(3) 病理状态(4) 医学概念(5) 抽象概念(6)

-- 3.4 【非 is_a 传递自检】has_part 的全部闭包行（含全库唯一一条 depth=2）
-- SELECT wp.word AS 部分, wh.word AS 整体, c.depth FROM closure c
--   JOIN senses sp ON sp.sense_id=c.ancestor   JOIN words wp ON wp.word_id=sp.word_id
--   JOIN senses sh ON sh.sense_id=c.descendant JOIN words wh ON wh.word_id=sh.word_id
--  WHERE c.relation='has_part' ORDER BY c.depth, 部分;
-- 期望：大脑|脑(1) 小脑|脑(1) 大脑皮层|大脑(1) 大脑皮层|脑(2)
-- ⚠ 注意：ancestor 一侧是「部分」、descendant 一侧是「整体」，见 §2.2 方向提示

-- 3.5 【跨库链接自检】
-- SELECT r.relation, w.word AS core_word, n.domain, n.label
--   FROM cross_links r
--   JOIN senses s ON s.sense_id = r.source_id AND r.source_kind='sense'
--   JOIN words  w ON w.word_id  = s.word_id
--   JOIN entity_nodes n ON n.node_id = r.target_id AND r.target_kind='node'
--  WHERE n.domain='medical';
-- 期望 3 行：nutritional_relevance/functional_analogy/domain_specialization

-- 3.6 【闭包一致性巡检】以下每句正常应返回 0 行
-- (a) 新增 is_a / has_part 边的 depth=1 闭包行缺失
-- SELECT r.relation_id, r.relation FROM relations r
--  WHERE r.relation IN ('is_a','has_part') AND r.src_kind='sense' AND r.dst_kind='sense'
--    AND NOT EXISTS (SELECT 1 FROM closure c WHERE c.relation=r.relation
--                     AND c.ancestor=r.dst_id AND c.descendant=r.src_id);
-- (b) 本文件新增的 7 个非传递关系不应有任何闭包行
-- SELECT DISTINCT relation FROM closure
--  WHERE relation IN ('has_symptom','treated_by','responsible_for','indicates',
--                     'nutritional_relevance','functional_analogy','domain_specialization');
-- (c) 医学库义项悬空的类别挂载
-- SELECT sc.sense_id, sc.cat FROM sense_categories sc
--  WHERE sc.sense_id BETWEEN 4001 AND 4030
--    AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat);

COMMIT;

/*****************************************************************************
 * End of seed_medical_v3.sql
 *****************************************************************************/
