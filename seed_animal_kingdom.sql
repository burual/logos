/*****************************************************************************
 * Logos 动物领域种子数据  seed_animal_kingdom.sql
 * 中文原生符号语义数据库 —— 动物分类骨架 + 关系多样的多跳闭包
 *
 * 目标
 * ----
 * 1. 在通用库「动物 / 生物」之上补一层可用的动物分类骨架（深度 >= 3）。
 * 2. 不只训练 / 评估 is_a：同时提供 has_property、typically_has、can_do、
 *    part_of、opposite_of 的真实同关系链，让闭包不再只有 is_a。
 * 3. 所有新增行都可重复执行；不修改任何已有种子文件。
 *
 * ---------------------------------------------------------------------------
 * 【schema 事实：senses 没有 domain 列】
 *   schema_v3.sql 的 senses 表没有 domain 字段。本文件用
 *     · sense_categories.cat = '动物'  表达「动物领域义项」
 *     · 属性 / 部位 / 动作节点分别挂 '性状' / '实体' / '事件'
 *   entity_nodes.domain='animal' 已由 schema 建好，但本文件不新增领域实体节点，
 *   避免改变 cross_links 视图的既有语义。
 *
 * 【为什么需要同关系链，而不是沿 is_a 继承属性】
 *   闭包触发器 tr_closure_edge_ai 只沿**同一 relation** 递归。它不会把
 *   「动物 is_a 生物」和「生物 has_property 有生命」拼成「动物 has_property
 *   有生命」。因此本文件刻意建：
 *     · 动物 / 类 -> 属性组 -> 原子属性       （has_property 深度 2）
 *     · 动物 / 类 -> 身体部位 -> 具体部件     （typically_has 深度 2）
 *     · 动物 / 类 -> 行为类别 -> 具体动作     （can_do 深度 2）
 *   这些链都是**真实的同关系路径**，闭包由触发器派生，没有任何手工 closure
 *   INSERT，因此 I1–I6 与全量重建差分仍然成立。
 *
 * 【typically_has 的取舍】
 *   医学影像种子把 typically_has 登记为 is_transitive=0（6 条边、零闭包）。
 *   本文件为了得到「动物 -> 部位 -> 具体部件」的 20+ 条深度 >=2 闭包，将其
 *   提升为 is_transitive=1 + closure_policy(incremental, max_depth=12)。
 *   这会让本库中已有的 6 条影像 typically_has 边也产生 depth=1 闭包；
 *   在只有医学种子的独立校验库里（validate_medical.py 显式加载文件）不受影响。
 *   若不希望改变该类型的全局语义，应改用 has_part，或放弃 typically_has 的
 *   多跳目标 —— 二者不可兼得。
 *
 * 【opposite_of 的取舍】
 *   「对立」不是传递关系：A 对立 B、B 对立 C 不能推出 A 对立 C。因此本文件
 *   只登记**直接属性轴对立对**，不构造 A->B->C 的对立链。opposite_of 登记为
 *   directed + is_transitive=1 只是为了与项目「可物化关系显式声明」的口径一致；
 *   当前没有同关系链，闭包只有 depth=1。已有的 opposite 仍是词汇反义关系，
 *   保持 symmetric / 不物化，二者不混用。
 *
 * 【加载顺序】
 *   logos/db/connection.py 按文件名字典序自动发现 seed_*.sql，本文件会排在
 *   seed_general_v3.sql **之前**。因此本文件：
 *     · 需要「生物」时先就地兜底（ID 1026，字段与 seed_general_v3.sql 一致）；
 *     · 对「猫 / 鱼 / 动物 / 生物」全部用 (word, pos) 自然键引用；
 *     · 所有 INSERT 均 ON CONFLICT DO NOTHING，后续 seed_general_v3.sql
 *       再插入同名行时不会冲突。
 *   在 apply_seeds.py 中本文件被显式排在 seed_general_v3.sql 之前，与字典序一致。
 *
 * 【ID 分配】words / senses 独占 5001–5080 段，共 80 个新义项。
 *   relation_types 新增 opposite_of；typically_has / typical_of 若缺失则补登记
 *   （全量库中它们已由影像种子登记，不新增关系类型）。
 *
 * 【关系类型标志（实测）】
 *   is_a=1, has_property=1, typically_has=1, can_do=1, part_of=1, opposite_of=1。
 *   has_property / can_do 在 schema 中原本就是 1；若压成 0，就无法得到本文件
 *   要求的深度≥2 闭包。typically_has 为满足 20+ 深度≥2 目标，从影像种子的 0
 *   提升为 1；opposite_of 新登记为 1（但只登记直接对立对，不构造传递链）。
 *
 * 【写入规模（全量重建实测；文件末尾有硬闸门）】
 *   words +80, senses +80, sense_categories +134（80 主类别 + 54 动物域标记）
 *   relations +316：is_a 28 / has_property 72 / typically_has 116 /
 *                    can_do 73 / part_of 20 / opposite_of 7
 *   closure +719：is_a 105 / has_property 147 / typically_has 180 /
 *                 can_do 251 / part_of 29 / opposite_of 7
 *   全部由触发器 / 队列重建派生，无手工 closure INSERT
 *
 * 【幂等性】全部 INSERT 带 ON CONFLICT DO NOTHING；临时表每次重建；
 *   typically_has 的队列重建可重复执行。
 *****************************************************************************/

PRAGMA foreign_keys = ON;
PRAGMA recursive_triggers = ON;

BEGIN;

/* =========================================================================
 * 0. 自然键视图 + 「生物」兜底
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS ak_sid AS
SELECT w.word, s.pos, s.sense_id AS sid
  FROM words w
  JOIN senses s ON s.word_id = w.word_id;

/* 生物在 seed_general_v3.sql 中才建立；本文件可能先加载，先兜底。
 * 字段与 seed_general_v3.sql 的 1026 行保持一致，后续插入会 DO NOTHING。 */
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register)
SELECT 1026, '生物', NULL, '合成词', 0, '通用'
 WHERE NOT EXISTS (SELECT 1 FROM words WHERE word = '生物')
ON CONFLICT DO NOTHING;

INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, frequency,
                   literalness, is_primary)
SELECT 1026, w.word_id, 'n', 1,
       '自然界中具有生命现象、能进行新陈代谢的物体',
       0, '本义', 1
  FROM words w
 WHERE w.word = '生物'
   AND NOT EXISTS (
       SELECT 1 FROM senses s
        WHERE s.word_id = w.word_id AND s.pos = 'n'
          AND s.definition = '自然界中具有生命现象、能进行新陈代谢的物体')
ON CONFLICT DO NOTHING;

INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT s.sense_id, '生物', 1
  FROM senses s JOIN words w ON w.word_id = s.word_id
 WHERE w.word = '生物' AND s.pos = 'n'
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 1. 关系类型与闭包策略
 *    1.1 typically_has / typical_of：若影像种子尚未加载则先登记；
 *        已存在时用 UPDATE 把 typically_has 提升为可传递。
 *    1.2 opposite_of：新增「属性轴直接对立」类型。
 *    1.3 确保 has_property / can_do / part_of / is_a 的策略行存在。
 * ========================================================================= */

INSERT INTO relation_types
 (relation, relation_class, direction, inverse_of, is_transitive, is_symmetric,
  is_reflexive, is_functional, max_depth, description, color_code) VALUES
 ('typically_has','属性','directed',NULL,1,0,0,0,12,
  '类别→其典型身体部位/特征（允许部位层级闭包）','#B91C1C')
ON CONFLICT(relation) DO NOTHING;

INSERT INTO relation_types
 (relation, relation_class, direction, inverse_of, is_transitive, is_symmetric,
  is_reflexive, is_functional, max_depth, description, color_code) VALUES
 ('typical_of','属性','directed','typically_has',0,0,0,1,12,
  '典型身体部位/特征→典型所属类别','#FCA5A5')
ON CONFLICT(relation) DO NOTHING;

UPDATE relation_types
   SET inverse_of = 'typical_of'
 WHERE relation = 'typically_has' AND inverse_of IS NULL;

UPDATE relation_types
   SET is_transitive = 1, max_depth = 12
 WHERE relation = 'typically_has';

INSERT INTO closure_policy(relation, mode, max_depth)
VALUES ('typically_has','incremental',12)
ON CONFLICT(relation) DO UPDATE
   SET mode = 'incremental', max_depth = 12,
       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now');

INSERT INTO relation_types
 (relation, relation_class, direction, inverse_of, is_transitive, is_symmetric,
  is_reflexive, is_functional, max_depth, description, color_code) VALUES
 ('opposite_of','词汇语义','directed',NULL,1,0,0,0,12,
  '属性轴上的直接对立对（不构造传递链）','#DC2626')
ON CONFLICT(relation) DO NOTHING;

INSERT INTO closure_policy(relation, mode, max_depth)
VALUES ('opposite_of','incremental',12)
ON CONFLICT(relation) DO NOTHING;

INSERT INTO closure_policy(relation, mode, max_depth) VALUES
 ('is_a','incremental',12),
 ('has_property','incremental',12),
 ('can_do','incremental',12),
 ('part_of','incremental',12),
 ('has_part','incremental',12)
ON CONFLICT(relation) DO NOTHING;

/* =========================================================================
 * 2. 词层 / 义项层（5001–5080，共 80 个）
 *    cat: 动物=动物分类与物种；性状=属性；实体=部位/器官；事件=行为
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS ak_seed(
    word_id    INTEGER PRIMARY KEY,
    word       TEXT NOT NULL,
    pos        TEXT NOT NULL,
    word_type  TEXT NOT NULL,
    cat        TEXT NOT NULL,
    definition TEXT NOT NULL
);
DELETE FROM ak_seed;

INSERT INTO ak_seed(word_id, word, pos, word_type, cat, definition) VALUES
-- 2.1 分类骨架与动物（26）
(5001,'猫科','n','合成词','动物','猫属及近缘野生猫类组成的哺乳动物科'),
(5002,'犬科','n','合成词','动物','犬、狼、狐等组成的哺乳动物科'),
(5003,'哺乳动物','n','合成词','动物','用乳汁哺育幼崽、体表多有毛的脊椎动物'),
(5004,'脊椎动物','n','合成词','动物','具有脊柱和脊椎骨的动物'),
(5005,'鸟类','n','合成词','动物','体表有羽毛、前肢演化为翅膀的脊椎动物'),
(5006,'雀形目','n','合成词','动物','鸣禽为主的鸟类目'),
(5007,'硬骨鱼','n','合成词','动物','骨骼多为硬骨、用鳃呼吸的鱼类'),
(5008,'鲤科','n','合成词','动物','鲤、鲫等淡水鱼组成的科'),
(5009,'爬行纲','n','合成词','动物','体表有鳞、卵生、冷血的脊椎动物纲'),
(5010,'两栖纲','n','合成词','动物','幼体水生、成体可上陆的脊椎动物纲'),
(5011,'昆虫纲','n','合成词','动物','身体分头胸腹、有三对足的节肢动物纲'),
(5012,'节肢动物','n','合成词','动物','具有外骨骼和分节附肢的无脊椎动物'),
(5013,'无脊椎动物','n','合成词','动物','没有脊柱的动物'),
(5014,'甲壳纲','n','合成词','动物','具坚硬甲壳、多生活在水中的节肢动物纲'),
(5015,'环节动物','n','合成词','动物','身体由许多环节组成的无脊椎动物'),
(5016,'软体动物','n','合成词','动物','身体柔软、多有贝壳的无脊椎动物'),
(5017,'头足纲','n','合成词','动物','具触手、运动迅速的海生软体动物纲'),
(5018,'狗','n','单纯词','动物','犬科家养动物，嗅觉灵敏，常被驯养'),
(5019,'鸟','n','单纯词','动物','有羽毛、翅膀和喙的卵生动物'),
(5020,'蛇','n','单纯词','动物','身体细长、无四肢的爬行动物'),
(5021,'青蛙','n','合成词','动物','幼体为蝌蚪、成体善跳跃的两栖动物'),
(5022,'昆虫','n','合成词','动物','身体分节、有三对足的小型节肢动物'),
(5023,'蝴蝶','n','合成词','动物','翅膀色彩鲜艳、昼间活动的昆虫'),
(5024,'虾','n','单纯词','动物','生活在水中的甲壳类动物'),
(5025,'蚯蚓','n','合成词','动物','生活在土壤中、身体分节的环节动物'),
(5026,'章鱼','n','合成词','动物','有八条触手、能喷墨的海生头足动物'),
-- 2.2 原子属性（13）
(5027,'有毛','a','合成词','性状','体表覆盖毛发'),
(5028,'温血','a','合成词','性状','体温相对恒定'),
(5029,'胎生','a','合成词','性状','幼体在母体内发育后产出'),
(5030,'有羽毛','a','合成词','性状','体表覆盖羽毛'),
(5031,'卵生','a','合成词','性状','以卵的形式繁殖'),
(5032,'有鳞','a','合成词','性状','体表覆盖鳞片'),
(5033,'冷血','a','合成词','性状','体温随环境变化'),
(5034,'无四肢','a','合成词','性状','没有四肢'),
(5035,'有脊柱','a','合成词','性状','具有脊柱'),
(5036,'无脊柱','a','合成词','性状','没有脊柱'),
(5037,'水生','a','合成词','性状','主要生活在水中'),
(5038,'陆生','a','合成词','性状','主要生活在陆地上'),
(5039,'有外骨骼','a','合成词','性状','体表具有坚硬的外骨骼'),
-- 2.3 属性组（6）—— has_property 的中间层
(5040,'哺乳动物属性组','n','合成词','性状','哺乳动物共有的属性集合'),
(5041,'鸟类属性组','n','合成词','性状','鸟类共有的属性集合'),
(5042,'鱼类属性组','n','合成词','性状','鱼类共有的属性集合'),
(5043,'爬行类属性组','n','合成词','性状','爬行动物共有的属性集合'),
(5044,'两栖类属性组','n','合成词','性状','两栖动物共有的属性集合'),
(5045,'无脊椎动物属性组','n','合成词','性状','无脊椎动物共有的属性集合'),
-- 2.4 身体部位层（4 个区域 + 12 个部件）
(5046,'头部','n','合成词','实体','动物身体前端、承载感觉器官的部位'),
(5047,'躯干','n','合成词','实体','动物身体的主体部分'),
(5048,'前肢','n','合成词','实体','动物身体前部的成对附肢'),
(5049,'尾部','n','合成词','实体','动物身体后端的部位'),
(5050,'爪子','n','合成词','实体','动物趾端坚硬的角质结构'),
(5051,'胡须','n','合成词','实体','动物口鼻附近的感觉毛'),
(5052,'翅膀','n','合成词','实体','鸟类用于飞行的前肢'),
(5053,'喙','n','单纯词','实体','鸟类取食的角质嘴'),
(5054,'羽毛','n','合成词','实体','覆盖鸟类体表的角质结构'),
(5055,'鳍','n','单纯词','实体','鱼类用于游动和保持平衡的器官'),
(5056,'鳃','n','单纯词','实体','水生动物用于呼吸的器官'),
(5057,'鳞片','n','合成词','实体','覆盖鱼或爬行动物体表的薄片'),
(5058,'触手','n','合成词','实体','头足类等动物用于捕食和感知的柔软附肢'),
(5059,'外壳','n','合成词','实体','甲壳类等动物体表的坚硬覆盖物'),
(5060,'尾巴','n','合成词','实体','动物身体后端的延伸部分'),
(5061,'鳍条','n','合成词','实体','支撑鱼鳍的细长结构'),
-- 2.5 行为类别与具体动作（4 + 10）
(5062,'捕猎','v','合成词','事件','动物寻找、追击并捕获猎物的行为'),
(5063,'运动','v','合成词','事件','动物改变位置或姿态的行为'),
(5064,'繁殖','v','合成词','事件','动物产生后代的行为'),
(5065,'防御','v','合成词','事件','动物保护自己、躲避敌害的行为'),
(5066,'扑击','v','合成词','事件','突然向前扑向目标'),
(5067,'追踪','v','合成词','事件','沿着目标留下的线索追赶'),
(5068,'跳跃','v','合成词','事件','用腿蹬地使身体腾空'),
(5069,'攀爬','v','合成词','事件','抓住支撑物向上移动'),
(5070,'奔跑','v','合成词','事件','快速移动'),
(5071,'飞行','v','合成词','事件','在空中运动'),
(5072,'游动','v','合成词','事件','在水中前进'),
(5073,'筑巢','v','合成词','事件','建造供栖息或繁殖的巢'),
(5074,'产卵','v','合成词','事件','产出卵'),
(5075,'伪装','v','合成词','事件','改变外观或保持静止以隐蔽自己'),
-- 2.6 器官与器官系统（4）—— part_of 的中间层
(5076,'心脏','n','合成词','实体','推动血液循环的器官'),
(5077,'循环系统','n','合成词','实体','输送血液和物质的器官系统'),
(5078,'肺','n','单纯词','实体','进行气体交换的呼吸器官'),
(5079,'呼吸系统','n','合成词','实体','完成气体交换的器官系统'),
-- 2.7 与「有外骨骼」配对的属性（1）—— opposite_of 的直接对立轴
(5080,'有内骨骼','a','合成词','性状','体内具有由骨骼构成的支持结构（与体外外骨骼相对）');

INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register)
SELECT word_id, word, NULL, word_type, 0, '通用'
  FROM ak_seed
 WHERE 1
ON CONFLICT DO NOTHING;

INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, tone_code, example,
                   frequency, literalness, derived_from, subcategory, is_primary)
SELECT word_id, word_id, pos, 1, definition, NULL,
       NULL, NULL, NULL, NULL, NULL,
       0, '本义', NULL, NULL, 1
  FROM ak_seed
 WHERE 1
ON CONFLICT DO NOTHING;

INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT word_id, cat, 1
  FROM ak_seed
 WHERE 1
ON CONFLICT DO NOTHING;

/* 领域标记：senses 表没有 domain 列，因此给全部 80 个新义项再加一个
 * '动物' 类别（非主类别），查询「动物域全部新增义项」时用
 *   sense_categories.cat='动物' AND sense_id BETWEEN 5001 AND 5080
 * 即可，不必依赖 primary 类别。 */
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT word_id, '动物', 0
  FROM ak_seed
 WHERE cat <> '动物'
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3. is_a：动物分类骨架
 *    边方向：下位 -> 上位（dst = 上位）
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    -- 猫 / 狗 -> 哺乳动物 -> 脊椎动物 -> 动物 -> 生物
    ('猫','猫科'),
    ('猫科','哺乳动物'),
    ('哺乳动物','脊椎动物'),
    ('脊椎动物','动物'),
    ('动物','生物'),
    ('狗','犬科'),
    ('犬科','哺乳动物'),
    -- 鸟
    ('鸟','雀形目'),
    ('雀形目','鸟类'),
    ('鸟类','脊椎动物'),
    -- 鱼
    ('鱼','鲤科'),
    ('鲤科','硬骨鱼'),
    ('硬骨鱼','脊椎动物'),
    -- 爬行 / 两栖
    ('蛇','爬行纲'),
    ('爬行纲','脊椎动物'),
    ('青蛙','两栖纲'),
    ('两栖纲','脊椎动物'),
    -- 昆虫 / 节肢 / 无脊椎
    ('蝴蝶','昆虫'),
    ('昆虫','昆虫纲'),
    ('昆虫纲','节肢动物'),
    ('节肢动物','无脊椎动物'),
    ('无脊椎动物','动物'),
    -- 甲壳 / 环节 / 软体 / 头足
    ('虾','甲壳纲'),
    ('甲壳纲','节肢动物'),
    ('蚯蚓','环节动物'),
    ('环节动物','无脊椎动物'),
    ('章鱼','头足纲'),
    ('头足纲','软体动物'),
    ('软体动物','无脊椎动物'),
    -- 兜底：既有库中的学科链接
    ('鱼','动物')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN ak_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN ak_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 4. has_property：动物/类 -> 属性组 -> 原子属性
 *    4.1 属性组 -> 原子属性（深度 1）
 *    4.2 动物/类 -> 属性组（深度 1；组合出深度 2）
 *    4.3 关键物种直接断言（深度 1，满足字面示例）
 * ========================================================================= */
WITH e(src, dst) AS (VALUES
    ('哺乳动物属性组','有毛'),
    ('哺乳动物属性组','温血'),
    ('哺乳动物属性组','胎生'),
    ('哺乳动物属性组','有脊柱'),
    ('鸟类属性组','有羽毛'),
    ('鸟类属性组','温血'),
    ('鸟类属性组','卵生'),
    ('鸟类属性组','有脊柱'),
    ('鱼类属性组','有鳞'),
    ('鱼类属性组','冷血'),
    ('鱼类属性组','卵生'),
    ('鱼类属性组','有脊柱'),
    ('爬行类属性组','有鳞'),
    ('爬行类属性组','冷血'),
    ('爬行类属性组','卵生'),
    ('爬行类属性组','有脊柱'),
    ('两栖类属性组','冷血'),
    ('两栖类属性组','卵生'),
    ('两栖类属性组','有脊柱'),
    ('无脊椎动物属性组','无脊柱'),
    ('无脊椎动物属性组','有外骨骼'),
    ('无脊椎动物属性组','卵生')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'has_property', 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'a'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

WITH e(src, dst) AS (VALUES
    ('猫','哺乳动物属性组'),
    ('猫科','哺乳动物属性组'),
    ('哺乳动物','哺乳动物属性组'),
    ('狗','哺乳动物属性组'),
    ('犬科','哺乳动物属性组'),
    ('鸟','鸟类属性组'),
    ('雀形目','鸟类属性组'),
    ('鸟类','鸟类属性组'),
    ('鱼','鱼类属性组'),
    ('鲤科','鱼类属性组'),
    ('硬骨鱼','鱼类属性组'),
    ('蛇','爬行类属性组'),
    ('爬行纲','爬行类属性组'),
    ('青蛙','两栖类属性组'),
    ('两栖纲','两栖类属性组'),
    ('昆虫','无脊椎动物属性组'),
    ('蝴蝶','无脊椎动物属性组'),
    ('昆虫纲','无脊椎动物属性组'),
    ('虾','无脊椎动物属性组'),
    ('甲壳纲','无脊椎动物属性组'),
    ('蚯蚓','无脊椎动物属性组'),
    ('环节动物','无脊椎动物属性组'),
    ('章鱼','无脊椎动物属性组'),
    ('头足纲','无脊椎动物属性组'),
    ('软体动物','无脊椎动物属性组'),
    ('节肢动物','无脊椎动物属性组'),
    ('无脊椎动物','无脊椎动物属性组')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'has_property', 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

WITH e(src, dst) AS (VALUES
    ('猫','有毛'), ('猫','温血'), ('猫','胎生'),
    ('狗','有毛'), ('狗','温血'),
    ('鸟','有羽毛'), ('鸟','温血'), ('鸟','卵生'),
    ('鱼','有鳞'), ('鱼','冷血'), ('鱼','卵生'),
    ('蛇','无四肢'), ('蛇','冷血'),
    ('青蛙','冷血'), ('青蛙','卵生'),
    ('昆虫','无脊柱'), ('昆虫','有外骨骼'),
    ('章鱼','无脊柱'), ('章鱼','水生'),
    ('虾','水生'), ('虾','有外骨骼'),
    ('蚯蚓','无脊柱'), ('蚯蚓','陆生')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'has_property', 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'a'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 5. typically_has：动物/类 -> 身体区域 / 类群特征组 -> 具体部件
 *    5.1 部位 -> 子部位（只保留普遍成立的路径）
 *    5.2 动物/类 -> 身体区域（深度 1）
 *    5.3 类群特征组 -> 典型部位/特征（深度 1；组合出深度 2）
 *    5.4 动物/类 -> 类群特征组（深度 1；组合出深度 2）
 *    5.5 关键物种直接断言（深度 1，满足字面示例）
 *
 * 设计修正：不使用「动物 -> 通用躯干 -> 羽毛」这类链。通用身体区域
 * 不能安全地作为部位层级的中间节点（否则猫会经「躯干 -> 羽毛」得到
 * 羽毛）。深度 2 主要走「猫 -> 哺乳动物属性组 -> 胡须」这种按类群
 * 限定的特征组；区域链只保留「前肢 -> 爪子 / 尾部 -> 尾巴 /
 * 鳍 -> 鳍条」等确实普遍成立的路径。
 * ========================================================================= */
WITH e(src, dst) AS (VALUES
    ('前肢','爪子'),
    ('尾部','尾巴'),
    ('鳍','鳍条')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'typically_has', 0.9, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

WITH e(src, dst) AS (VALUES
    ('猫','头部'), ('猫','躯干'), ('猫','前肢'), ('猫','尾部'),
    ('狗','头部'), ('狗','躯干'), ('狗','前肢'), ('狗','尾部'),
    ('鸟','头部'), ('鸟','躯干'),
    ('鱼','头部'), ('鱼','躯干'), ('鱼','尾部'),
    ('蛇','头部'), ('蛇','躯干'), ('蛇','尾部'),
    ('青蛙','头部'), ('青蛙','躯干'),
    ('昆虫','头部'), ('昆虫','躯干'),
    ('蝴蝶','头部'), ('蝴蝶','躯干'),
    ('虾','头部'), ('虾','躯干'),
    ('蚯蚓','头部'), ('蚯蚓','躯干'),
    ('章鱼','头部'), ('章鱼','躯干'),
    ('猫科','头部'), ('猫科','躯干'), ('猫科','前肢'), ('猫科','尾部'),
    ('犬科','头部'), ('犬科','躯干'), ('犬科','前肢'), ('犬科','尾部'),
    ('哺乳动物','头部'), ('哺乳动物','躯干'), ('哺乳动物','前肢'), ('哺乳动物','尾部'),
    ('鸟类','头部'), ('鸟类','躯干'),
    ('雀形目','头部'), ('雀形目','躯干'),
    ('鱼类','头部'), ('鱼类','躯干'), ('鱼类','尾部'),
    ('硬骨鱼','头部'), ('硬骨鱼','躯干'), ('硬骨鱼','尾部'),
    ('鲤科','头部'), ('鲤科','躯干'), ('鲤科','尾部'),
    ('爬行纲','头部'), ('爬行纲','躯干'), ('爬行纲','尾部'),
    ('两栖纲','头部'), ('两栖纲','躯干'),
    ('昆虫纲','头部'), ('昆虫纲','躯干'),
    ('节肢动物','头部'), ('节肢动物','躯干'),
    ('无脊椎动物','头部'), ('无脊椎动物','躯干'),
    ('甲壳纲','头部'), ('甲壳纲','躯干'),
    ('环节动物','头部'), ('环节动物','躯干'),
    ('软体动物','头部'), ('软体动物','躯干'),
    ('头足纲','头部'), ('头足纲','躯干')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'typically_has', 0.8, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* 类群特征组 -> 典型部位（只列对该类群普遍成立的部位） */
WITH e(src, dst) AS (VALUES
    ('哺乳动物属性组','爪子'),
    ('哺乳动物属性组','胡须'),
    ('哺乳动物属性组','尾巴'),
    ('鸟类属性组','翅膀'),
    ('鸟类属性组','喙'),
    ('鸟类属性组','羽毛'),
    ('鱼类属性组','鳍'),
    ('鱼类属性组','鳃'),
    ('鱼类属性组','鳞片'),
    ('爬行类属性组','鳞片'),
    ('爬行类属性组','尾巴')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'typically_has', 0.8, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* 类群特征组 -> 典型特征（形容词也允许作为 typically_has 的目标；
 * 这两条对所有两栖类 / 无脊椎动物都成立，避免把「触手」传播给昆虫）。 */
WITH e(src, dst) AS (VALUES
    ('两栖类属性组','有脊柱'),
    ('无脊椎动物属性组','无脊柱')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'typically_has', 0.8, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'a'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* 动物/类 -> 类群特征组 */
WITH e(src, dst) AS (VALUES
    ('猫','哺乳动物属性组'),
    ('猫科','哺乳动物属性组'),
    ('哺乳动物','哺乳动物属性组'),
    ('狗','哺乳动物属性组'),
    ('犬科','哺乳动物属性组'),
    ('鸟','鸟类属性组'),
    ('雀形目','鸟类属性组'),
    ('鸟类','鸟类属性组'),
    ('鱼','鱼类属性组'),
    ('鲤科','鱼类属性组'),
    ('硬骨鱼','鱼类属性组'),
    ('蛇','爬行类属性组'),
    ('爬行纲','爬行类属性组'),
    ('青蛙','两栖类属性组'),
    ('两栖纲','两栖类属性组'),
    ('昆虫','无脊椎动物属性组'),
    ('蝴蝶','无脊椎动物属性组'),
    ('昆虫纲','无脊椎动物属性组'),
    ('虾','无脊椎动物属性组'),
    ('甲壳纲','无脊椎动物属性组'),
    ('蚯蚓','无脊椎动物属性组'),
    ('环节动物','无脊椎动物属性组'),
    ('章鱼','无脊椎动物属性组'),
    ('头足纲','无脊椎动物属性组'),
    ('软体动物','无脊椎动物属性组'),
    ('节肢动物','无脊椎动物属性组'),
    ('无脊椎动物','无脊椎动物属性组')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'typically_has', 0.8, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* 关键物种直接断言：只保留无法由「物种 -> 类群特征组 / 区域」
 * 派生出的独有部位，避免直接边把同一条记录的深度 2 路径压成深度 1。 */
WITH e(src, dst) AS (VALUES
    ('蝴蝶','翅膀'), ('昆虫','外壳'), ('虾','外壳'), ('章鱼','触手')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'typically_has', 0.9, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 6. can_do：动物/类 -> 行为类别 -> 具体动作
 *    6.1 行为类别 -> 具体动作（深度 1）
 *    6.2 动物/类 -> 行为类别（深度 1；组合出深度 2）
 *    6.3 关键物种直接断言（深度 1，满足字面示例）
 * ========================================================================= */
WITH e(src, dst) AS (VALUES
    ('捕猎','扑击'), ('捕猎','追踪'),
    ('运动','跳跃'), ('运动','攀爬'), ('运动','奔跑'),
    ('运动','飞行'), ('运动','游动'),
    ('繁殖','筑巢'), ('繁殖','产卵'),
    ('防御','伪装')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'can_do', 0.9, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'v'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'v'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

WITH e(src, dst) AS (VALUES
    ('猫','捕猎'), ('猫','运动'), ('猫','防御'),
    ('狗','捕猎'), ('狗','运动'),
    ('鸟','运动'), ('鸟','繁殖'),
    ('鱼','运动'), ('鱼','防御'),
    ('蛇','捕猎'), ('蛇','运动'),
    ('青蛙','捕猎'), ('青蛙','运动'),
    ('昆虫','运动'), ('昆虫','繁殖'), ('昆虫','防御'),
    ('蝴蝶','运动'), ('蝴蝶','繁殖'),
    ('虾','运动'), ('虾','防御'),
    ('蚯蚓','运动'), ('蚯蚓','防御'),
    ('章鱼','运动'), ('章鱼','防御'),
    ('猫科','捕猎'), ('猫科','运动'), ('猫科','防御'),
    ('犬科','捕猎'), ('犬科','运动'),
    ('哺乳动物','捕猎'), ('哺乳动物','运动'), ('哺乳动物','防御'),
    ('鸟类','运动'), ('鸟类','繁殖'),
    ('雀形目','运动'), ('雀形目','繁殖'),
    ('鱼类','运动'), ('鱼类','防御'),
    ('硬骨鱼','运动'), ('硬骨鱼','防御'),
    ('鲤科','运动'), ('鲤科','防御'),
    ('爬行纲','捕猎'), ('爬行纲','运动'),
    ('两栖纲','捕猎'), ('两栖纲','运动'),
    ('昆虫纲','运动'), ('昆虫纲','繁殖'), ('昆虫纲','防御'),
    ('节肢动物','运动'), ('节肢动物','防御'),
    ('无脊椎动物','运动'), ('无脊椎动物','防御'),
    ('甲壳纲','运动'), ('甲壳纲','防御'),
    ('环节动物','运动'), ('环节动物','防御'),
    ('软体动物','运动'), ('软体动物','防御'),
    ('头足纲','运动'), ('头足纲','防御')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'can_do', 0.9, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'v'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

WITH e(src, dst) AS (VALUES
    ('猫','跳跃'),
    ('鸟','飞行'), ('鸟','筑巢'),
    ('鱼','游动')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'can_do', 0.9, 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'v'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 7. part_of：部分 -> 整体
 *    7.1 器官 -> 器官系统 -> 动物/脊椎动物
 *    7.2 身体部件 -> 区域 -> 动物/类群
 * ========================================================================= */
WITH e(part, whole) AS (VALUES
    ('心脏','循环系统'),
    ('循环系统','猫'),
    ('循环系统','脊椎动物'),
    ('肺','呼吸系统'),
    ('呼吸系统','猫'),
    ('呼吸系统','脊椎动物'),
    ('爪子','前肢'),
    ('前肢','猫'),
    ('胡须','头部'),
    ('头部','猫'),
    ('尾巴','尾部'),
    ('尾部','猫'),
    ('羽毛','翅膀'),
    ('翅膀','鸟类'),
    ('鳍条','鳍'),
    ('鳍','鱼'),
    ('鳃','鱼'),
    ('鳞片','鱼'),
    ('触手','章鱼'),
    ('外壳','虾')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', p.sid, 'sense', w.sid, 'part_of', 5
  FROM e
  JOIN ak_sid p ON p.word = e.part  AND p.pos = 'n'
  JOIN ak_sid w ON w.word = e.whole AND w.pos = 'n'
 WHERE p.sid <> w.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 8. opposite_of：属性轴上的直接对立对
 *    只登记直接对，不建立 A->B->C 链（见文件头取舍）。
 * ========================================================================= */
WITH e(src, dst) AS (VALUES
    ('温血','冷血'),
    ('胎生','卵生'),
    ('有毛','有鳞'),
    ('有羽毛','有鳞'),
    ('水生','陆生'),
    ('有脊柱','无脊柱'),
    ('有外骨骼','有内骨骼')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'opposite_of', 5
  FROM e
  JOIN ak_sid s ON s.word = e.src AND s.pos = 'a'
  JOIN ak_sid d ON d.word = e.dst AND d.pos = 'a'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 9. 闭包收尾
 *    - typically_has 在旧库中可能已有 6 条边（类型刚从 0 提升为 1），
 *      这些旧边不会触发 AFTER INSERT，故显式排队重建一次。
 *    - opposite_of / has_property / can_do / part_of 的边缘插入已触发增量；
 *      再排队一次可保证 --keep 与全量重建结果一致（幂等）。
 * ========================================================================= */
INSERT OR REPLACE INTO closure_rebuild_queue(relation) VALUES
 ('typically_has'),
 ('opposite_of'),
 ('has_property'),
 ('can_do'),
 ('part_of');

/* =========================================================================
 * 10. 硬闸门：任何一项不满足就让建库显式失败
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS ak_gate(ok INTEGER NOT NULL CHECK (ok = 1));
DELETE FROM ak_gate;

INSERT INTO ak_gate(ok)
SELECT CASE WHEN
       (SELECT COUNT(*) FROM senses WHERE sense_id BETWEEN 5001 AND 5080) = 80
   AND (SELECT COUNT(*) FROM relations
         WHERE relation = 'has_property'
           AND src_id BETWEEN 5001 AND 5080
           AND dst_id BETWEEN 5001 AND 5080) >= 40
   AND (SELECT COUNT(*) FROM relations
         WHERE relation = 'typically_has'
           AND src_id BETWEEN 5001 AND 5080
           AND dst_id BETWEEN 5001 AND 5080) >= 50
   AND (SELECT COUNT(*) FROM relations
         WHERE relation = 'can_do'
           AND src_id BETWEEN 5001 AND 5080
           AND dst_id BETWEEN 5001 AND 5080) >= 40
   AND (SELECT COUNT(*) FROM relations
         WHERE relation = 'part_of'
           AND src_id BETWEEN 5001 AND 5080
           AND dst_id BETWEEN 5001 AND 5080) >= 10
   AND (SELECT COUNT(*) FROM closure
         WHERE relation = 'has_property' AND depth >= 2) >= 20
   AND (SELECT COUNT(*) FROM closure
         WHERE relation = 'typically_has' AND depth >= 2) >= 20
   AND (SELECT COUNT(*) FROM closure
         WHERE relation = 'can_do' AND depth >= 2) >= 20
   AND (SELECT COUNT(*) FROM closure
         WHERE relation = 'part_of' AND depth >= 2) >= 3
   THEN 1 ELSE 0 END;

DROP TABLE ak_gate;
DROP VIEW ak_sid;
DROP TABLE ak_seed;

COMMIT;
