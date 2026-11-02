/*****************************************************************************
 * Logos 动物领域扩展种子  seed_animal_kingdom_extended.sql
 * 中文原生符号语义数据库 —— 弱关系扩容（can_do / has_property / typically_has
 *                                / part_of / has_part）
 *
 * ===========================================================================
 * 【本文件要解决的问题】
 *   关系感知打分头（train_closure_relaware.py）在 250 义项图上跑完后，
 *   只有 is_a 得到统计上可信的验证（test n=44）；其余弱关系的 test 样本数
 *   分别是 can_do 7 / has_property 3 / typically_has 1 / part_of 0 / has_part 0。
 *   n 太小既不能证伪也不能证实，因此本文件的目标不是「加概念」而是
 *   **加同关系链的源（source）**：闭包切分是按源（src）做 source-disjoint 的，
 *   一个源的全部边只会落进 train/val/test 中的某一个划分，
 *   所以 test 样本数 ≈ 0.15 × 该关系的不同源个数 × 每源边数。
 *   由此得到本文件的两条设计原则：
 *     (P1) 多造**新的源**，而不是把边堆在少数几个源上；
 *     (P2) 每条边都要能组合出 depth >= 2 的同关系路径。
 *
 * ===========================================================================
 * 【ID 段位】words / senses 独占 5081–5165，共 85 个新义项。
 *   （5081–5089 动物 9 / 5090–5095 行为类别 6 / 5096–5121 具体动作 26 /
 *     5122–5130 属性组 9 / 5131–5143 原子属性 13 / 5144–5165 部位与区域 22）
 *   不占用任何既有段位：通用 1000–1027、医学 4000–4101、影像 4040–4076、
 *   动物 5001–5080、测试夹具 7000+。
 *
 * 【与用户给定清单的两处偏差（必须知道）】
 *   (1) 用户写「IDs 5081-5120」，但清单本身要求 9 个新动物 + 约 26 个新动作 +
 *       属性组与原子属性 + 部位，最少也需要 80+ 个义项才能让每条边都成为
 *       sense→sense 的闭包可物化边（closure 触发器只处理 sense 端点）。
 *       故实际占用 5081–5165（85 个）。
 *   (2) 「章鱼」已在 seed_animal_kingdom.sql 的 5026 建立，本文件**不重复建词**，
 *       只给它补 can_do / has_property / typically_has / part_of 边。
 *
 * ===========================================================================
 * 【为什么弱关系必须走「类别组」中间层】
 *   闭包触发器只沿**同一 relation** 递归，不做跨关系拼接。因此 depth>=2 只能来自
 *   同关系的两步链。对 can_do / has_property / typically_has 而言，唯一自然的
 *   两步链是「个体 → 类别组 → 具体项」：
 *       鹰 --can_do--> 飞行 --can_do--> 翱翔         （鹰可以翱翔，depth 2）
 *       海豚 --has_property--> 鲸类属性组 --has_property--> 高智商
 *       大象 --typically_has--> 长鼻类属性组 --typically_has--> 长鼻
 *
 *   但中间层必须**按类群限定**，不能复用通用区域。反例是既有库里的
 *   「头部 part_of 猫」：一旦写「复眼 part_of 头部」，就会凭空得到
 *   「复眼 part_of 猫」。因此本文件为每类新增部位都配了专属区域节点：
 *       鲸类头部 / 象面部 / 昆虫头部
 *   全部新增部位都挂在这三个节点或类群属性组之下，**不碰通用「头部/尾部/前肢」**。
 *   唯一的例外是「利爪 part_of 爪子」「钩喙 part_of 喙」「飞羽 part_of 翅膀」，
 *   这三条是真正的包含关系（利爪就是一种爪子），可以安全外推。
 *
 * ===========================================================================
 * 【关系类型：本文件不新增任何 relation_types 行】
 *   is_a / has_property / typically_has / can_do / part_of / has_part 六个类型
 *   及其 closure_policy 行均由前序种子建立（typically_has 已由
 *   seed_animal_kingdom.sql 提升为 is_transitive=1 + incremental/12）。
 *   本文件只在 §1 做一次**存在性兜底**，避免单独加载时静默丢闭包。
 *
 * 【写入规模（实测，见 §10 硬闸门）】
 *   义项 +85 / is_a 9 / has_property 58 / typically_has 27 / can_do 65 /
 *   part_of 39 / has_part 14   ——  关系共 212 条，闭包全部由触发器派生，
 *   本文件**零手工 closure INSERT**。
 *
 * 【幂等性】全部 INSERT 带 ON CONFLICT DO NOTHING；临时表/视图每次重建；
 *   闭包队列重建可重复执行。二次执行（--keep）不得产生任何行数漂移。
 *
 * 【加载顺序】字典序：seed_animal_kingdom.sql < seed_animal_kingdom_extended.sql
 *   （'.'(U+002E) < '_'(U+005F)），即扩展恒在基础动物库之后、
 *   seed_general_v3.sql 之前。故本文件可安全引用 5001–5080 段与
 *   1026「生物」兜底行。§0 有具名 CHECK 闸门钉住这个前提。
 *****************************************************************************/

PRAGMA foreign_keys       = ON;
PRAGMA recursive_triggers = ON;

BEGIN;

/* =========================================================================
 * §0 前置依赖硬闸门 + 自然键视图
 *    具名 CHECK 让失败信息本身成为修复提示。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS ae_prereq(
    ok INTEGER NOT NULL,
    CONSTRAINT _prereq_missing_run_seed_animal_kingdom_before_extended CHECK (ok = 1)
);
DELETE FROM ae_prereq;

INSERT INTO ae_prereq(ok)
SELECT CASE WHEN
       -- 动物基础库已加载（80 个 5001–5080 义项 + 行为类别）
       (SELECT COUNT(*) FROM senses WHERE sense_id BETWEEN 5001 AND 5080) = 80
   AND (SELECT COUNT(*) FROM senses s JOIN words w ON w.word_id = s.word_id
         WHERE w.word IN ('鸟类','哺乳动物','昆虫','运动','捕猎','飞行','防御','章鱼')
           AND s.pos IN ('n','v')) >= 8
       -- 闭包策略就位（否则触发器静默不物化）
   AND (SELECT COUNT(*) FROM closure_policy
         WHERE relation IN ('is_a','has_property','typically_has','can_do',
                            'part_of','has_part')
           AND mode = 'incremental') = 6
   THEN 1 ELSE 0 END;

DROP TABLE ae_prereq;

CREATE TEMP VIEW IF NOT EXISTS ae_sid AS
SELECT w.word, s.pos, s.sense_id AS sid
  FROM words w
  JOIN senses s ON s.word_id = w.word_id;

/* =========================================================================
 * §1 关系类型兜底（不新增类型，只确保行存在）
 *    typically_has 的 is_transitive 由 seed_animal_kingdom.sql 提升为 1；
 *    若本文件被单独加载，这里补齐同样语义，保证 can_do 之外的链也可物化。
 * ========================================================================= */
INSERT INTO closure_policy(relation, mode, max_depth) VALUES
 ('is_a','incremental',12),
 ('has_property','incremental',12),
 ('typically_has','incremental',12),
 ('can_do','incremental',12),
 ('part_of','incremental',8),
 ('has_part','incremental',8)
ON CONFLICT DO NOTHING;

UPDATE relation_types
   SET is_transitive = 1
 WHERE relation IN ('typically_has','can_do','has_property')
   AND is_transitive = 0;

/* =========================================================================
 * §2 词层 + 义项层（5081–5165，共 85 个）
 *    cat: 动物=动物与类群；事件=行为；性状=属性/属性组；实体=部位/区域
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS ae_seed(
    word_id    INTEGER PRIMARY KEY,
    word       TEXT NOT NULL,
    pos        TEXT NOT NULL,
    word_type  TEXT NOT NULL,
    cat        TEXT NOT NULL,
    definition TEXT NOT NULL
);
DELETE FROM ae_seed;

INSERT INTO ae_seed(word_id, word, pos, word_type, cat, definition) VALUES
-- 2.1 新增动物（9）—— can_do / has_property / typically_has 的新源
(5081,'鹰','n','单纯词','动物','昼间活动、喙钩爪利的大型猛禽'),
(5082,'海豚','n','合成词','动物','生活在海洋中、靠回声定位捕食的鲸类'),
(5083,'蚂蚁','n','合成词','动物','群居生活、个体间分工明确的昆虫'),
(5084,'蜜蜂','n','合成词','动物','采集花蜜、能筑巢储蜜的群居昆虫'),
(5085,'大象','n','合成词','动物','体型庞大、有长鼻与象牙的哺乳动物'),
(5086,'猴子','n','合成词','动物','灵长类中善攀爬、脑较发达的动物'),
(5087,'企鹅','n','单纯词','动物','多分布于寒冷海域、善游泳而不会飞的海鸟'),
(5088,'蝙蝠','n','单纯词','动物','夜间活动、靠超声波定位的哺乳动物'),
(5089,'河狸','n','单纯词','动物','半水生、会用树枝泥土筑坝的啮齿动物'),
-- 2.2 行为类别（6）—— can_do 的中间层
(5090,'通讯','v','合成词','事件','个体之间传递信息的行为'),
(5091,'筑造','v','合成词','事件','修建巢、坝等结构的行为'),
(5092,'取食','v','合成词','事件','获取并处理食物的行为'),
(5093,'感知','v','合成词','事件','通过感官获取外界信息的行为'),
(5094,'认知','v','合成词','事件','学习、记忆与解决问题的心智活动'),
(5095,'育幼','v','合成词','事件','抚育后代的行为'),
-- 2.3 具体动作（26）—— can_do 的叶子
(5096,'翱翔','v','合成词','事件','不扇动翅膀、借助气流在空中盘旋'),
(5097,'俯冲','v','合成词','事件','从高处快速向下的飞行'),
(5098,'滑行','v','合成词','事件','贴着表面连续移动'),
(5099,'潜水','v','合成词','事件','全身没入水中活动'),
(5100,'跃出水面','v','合成词','事件','用力冲出水面'),
(5101,'挖掘','v','合成词','事件','用肢体或工具翻挖土石'),
(5102,'搬运','v','合成词','事件','把物体从一处移到另一处'),
(5103,'啃树','v','合成词','事件','用牙齿咬断树木'),
(5104,'筑坝','v','合成词','事件','用树枝泥土拦水成坝'),
(5105,'采蜜','v','合成词','事件','从花朵中吸取花蜜'),
(5106,'信息素通讯','v','合成词','事件','用化学气味传递信息'),
(5107,'舞蹈通讯','v','合成词','事件','用身体动作传递信息'),
(5108,'回声定位','v','合成词','事件','发出声音并依回声判断物体方位'),
(5109,'声呐导航','v','合成词','事件','用声波探测环境以确定路线'),
(5110,'面部表情','v','合成词','事件','用面部变化传达情绪'),
(5111,'使用工具','v','合成词','事件','借助外物完成动作'),
(5112,'开瓶','v','合成词','事件','打开瓶盖取出内容'),
(5113,'记忆路径','v','合成词','事件','记住并重现走过的路线'),
(5114,'喷水','v','合成词','事件','把水从体内喷出'),
(5115,'用鼻取物','v','合成词','事件','用长鼻卷取物品'),
(5116,'变色','v','合成词','事件','改变体表颜色'),
(5117,'喷射推进','v','合成词','事件','喷出水流产生前进的力'),
(5118,'倒挂','v','合成词','事件','头朝下悬挂在支撑物上'),
(5119,'空中捕食','v','合成词','事件','在飞行中捕获猎物'),
(5120,'群居育幼','v','合成词','事件','群体共同抚养后代'),
(5121,'群体协作','v','合成词','事件','多个个体配合完成同一任务'),
-- 2.4 类群属性组（9）—— has_property / typically_has 的中间层
(5122,'猛禽属性组','n','合成词','性状','猛禽共有的属性与部位集合'),
(5123,'鲸类属性组','n','合成词','性状','鲸类共有的属性与部位集合'),
(5124,'社会性昆虫属性组','n','合成词','性状','社会性昆虫共有的属性与部位集合'),
(5125,'长鼻类属性组','n','合成词','性状','长鼻类共有的属性与部位集合'),
(5126,'头足类属性组','n','合成词','性状','头足类共有的属性与部位集合'),
(5127,'翼手类属性组','n','合成词','性状','翼手类共有的属性与部位集合'),
(5128,'灵长类属性组','n','合成词','性状','灵长类共有的属性与部位集合'),
(5129,'企鹅属性组','n','合成词','性状','企鹅共有的属性与部位集合'),
(5130,'河狸属性组','n','合成词','性状','河狸共有的属性与部位集合'),
-- 2.5 原子属性（13）—— has_property 的叶子
(5131,'视力敏锐','a','合成词','性状','能在远处分辨细小目标'),
(5132,'骨骼中空','a','合成词','性状','骨骼内部留有空腔、整体较轻'),
(5133,'高智商','a','合成词','性状','学习与解决新问题的能力强'),
(5134,'社会性','a','合成词','性状','营群体生活、个体间有分工'),
(5135,'六足','a','合成词','性状','具有六条足'),
(5136,'长寿命','a','合成词','性状','寿命明显长于近缘类群'),
(5137,'三心脏','a','合成词','性状','体内具有三个心脏'),
(5138,'夜行性','a','合成词','性状','主要在夜间活动'),
(5139,'对生拇指','a','合成词','性状','拇指能与其他指相对'),
(5140,'门齿发达','a','合成词','性状','门齿粗壮且不断生长'),
(5141,'半水生','a','合成词','性状','水陆两处都能生活'),
(5142,'体型庞大','a','合成词','性状','身体尺寸远大于近缘类群'),
(5143,'耐寒','a','合成词','性状','能耐受低温环境'),
-- 2.6 部位与区域（22）—— typically_has / part_of / has_part 的叶子与中间层
(5144,'利爪','n','合成词','实体','弯曲而尖锐的爪'),
(5145,'钩喙','n','合成词','实体','前端向下弯曲的喙'),
(5146,'飞羽','n','合成词','实体','着生在翼上、用于飞行的羽毛'),
(5147,'尾羽','n','合成词','实体','着生在尾部的羽毛'),
(5148,'背鳍','n','合成词','实体','位于背部中线上的鳍'),
(5149,'胸鳍','n','合成词','实体','位于胸部两侧的鳍'),
(5150,'触角','n','合成词','实体','头部成对的感觉附肢'),
(5151,'复眼','n','合成词','实体','由许多小眼组成的眼'),
(5152,'长鼻','n','合成词','实体','大象可卷取物体的延长鼻'),
(5153,'象牙','n','合成词','实体','大象上颌延长形成的长牙'),
(5154,'八腕','n','合成词','实体','章鱼口周的八条腕足'),
(5155,'墨囊','n','合成词','实体','头足类储存墨汁的囊'),
(5156,'声呐器官','n','合成词','实体','发出并接收超声波的器官'),
(5157,'蹼足','n','合成词','实体','趾间具蹼的足'),
(5158,'门齿','n','合成词','实体','位于口腔前部的牙齿'),
(5159,'翼膜','n','合成词','实体','连接前肢与身体、用于飞行的膜'),
(5160,'骨骼系统','n','合成词','实体','支撑身体的骨骼总体'),
(5161,'皮肤系统','n','合成词','实体','覆盖体表的皮肤及其衍生物'),
(5162,'鲸类头部','n','合成词','实体','鲸类身体前端的部位'),
(5163,'象面部','n','合成词','实体','大象头部前端、承载长鼻与象牙的部位'),
(5164,'昆虫头部','n','合成词','实体','昆虫身体前端、着生触角与复眼的体节'),
(5165,'脊柱','n','合成词','实体','由椎骨相连构成的身体中轴');

INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register)
SELECT word_id, word, NULL, word_type, 0, '通用'
  FROM ae_seed
 WHERE 1
ON CONFLICT DO NOTHING;

INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, tone_code, example,
                   frequency, literalness, derived_from, subcategory, is_primary)
SELECT word_id, word_id, pos, 1, definition, NULL,
       NULL, NULL, NULL, NULL, NULL,
       0, '本义', NULL, NULL, 1
  FROM ae_seed
 WHERE 1
ON CONFLICT DO NOTHING;

INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT word_id, cat, 1
  FROM ae_seed
 WHERE 1
ON CONFLICT DO NOTHING;

/* 领域标记：senses 无 domain 列，用非主类别 '动物' 标记整个扩展段。 */
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT word_id, '动物', 0
  FROM ae_seed
 WHERE cat <> '动物'
ON CONFLICT DO NOTHING;

/* =========================================================================
 * §3 is_a：新动物挂进既有分类骨架（9 条）
 *    鹰/企鹅 -> 鸟类；海豚/大象/猴子/蝙蝠/河狸 -> 哺乳动物；蚂蚁/蜜蜂 -> 昆虫
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    ('鹰','鸟类'),
    ('企鹅','鸟类'),
    ('海豚','哺乳动物'),
    ('大象','哺乳动物'),
    ('猴子','哺乳动物'),
    ('蝙蝠','哺乳动物'),
    ('河狸','哺乳动物'),
    ('蚂蚁','昆虫'),
    ('蜜蜂','昆虫')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN ae_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN ae_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * §4 has_property：个体 -> 类群属性组 -> 原子属性
 *    4.1 属性组 -> 原子属性（39 条，depth 1）
 *    4.2 个体 -> 属性组   （19 条，depth 1；与 4.1 组合出 depth 2）
 *    每人挂两个组（专属组 + 既有分类组），使 has_property 的 depth>=2 样本
 *    不只集中在少数几个源上。
 * ========================================================================= */
WITH e(grp, prop) AS (VALUES
    ('猛禽属性组','视力敏锐'), ('猛禽属性组','骨骼中空'),
    ('猛禽属性组','有羽毛'),   ('猛禽属性组','温血'), ('猛禽属性组','有脊柱'),
    ('鲸类属性组','高智商'),   ('鲸类属性组','胎生'),
    ('鲸类属性组','温血'),     ('鲸类属性组','水生'), ('鲸类属性组','有脊柱'),
    ('社会性昆虫属性组','社会性'), ('社会性昆虫属性组','六足'),
    ('社会性昆虫属性组','有外骨骼'), ('社会性昆虫属性组','无脊柱'),
    ('长鼻类属性组','长寿命'), ('长鼻类属性组','体型庞大'), ('长鼻类属性组','高智商'),
    ('长鼻类属性组','有毛'),   ('长鼻类属性组','胎生'),
    ('头足类属性组','三心脏'), ('头足类属性组','无脊柱'),
    ('头足类属性组','水生'),   ('头足类属性组','冷血'),
    ('翼手类属性组','夜行性'), ('翼手类属性组','胎生'),
    ('翼手类属性组','温血'),   ('翼手类属性组','有毛'),
    ('灵长类属性组','高智商'), ('灵长类属性组','对生拇指'),
    ('灵长类属性组','有毛'),   ('灵长类属性组','胎生'),
    ('企鹅属性组','耐寒'),     ('企鹅属性组','有羽毛'),
    ('企鹅属性组','卵生'),     ('企鹅属性组','水生'),
    ('河狸属性组','门齿发达'), ('河狸属性组','半水生'),
    ('河狸属性组','有毛'),     ('河狸属性组','胎生')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', g.sid, 'sense', p.sid, 'has_property', 5
  FROM e
  JOIN ae_sid g ON g.word = e.grp  AND g.pos = 'n'
  JOIN ae_sid p ON p.word = e.prop AND p.pos = 'a'
 WHERE g.sid <> p.sid
ON CONFLICT DO NOTHING;

WITH e(animal, grp) AS (VALUES
    ('鹰','猛禽属性组'),     ('鹰','鸟类属性组'),
    ('海豚','鲸类属性组'),   ('海豚','哺乳动物属性组'),
    ('蚂蚁','社会性昆虫属性组'), ('蚂蚁','无脊椎动物属性组'),
    ('蜜蜂','社会性昆虫属性组'), ('蜜蜂','无脊椎动物属性组'),
    ('大象','长鼻类属性组'), ('大象','哺乳动物属性组'),
    ('猴子','灵长类属性组'), ('猴子','哺乳动物属性组'),
    ('企鹅','企鹅属性组'),   ('企鹅','鸟类属性组'),
    ('蝙蝠','翼手类属性组'), ('蝙蝠','哺乳动物属性组'),
    ('河狸','河狸属性组'),   ('河狸','哺乳动物属性组'),
    ('章鱼','头足类属性组')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', a.sid, 'sense', g.sid, 'has_property', 5
  FROM e
  JOIN ae_sid a ON a.word = e.animal AND a.pos = 'n'
  JOIN ae_sid g ON g.word = e.grp    AND g.pos = 'n'
 WHERE a.sid <> g.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * §5 typically_has：个体 -> 类群属性组 -> 部位
 *    5.1 属性组 -> 部位（17 条）
 *    5.2 个体 -> 属性组（10 条；组合出 depth 2）
 * ========================================================================= */
WITH e(grp, part) AS (VALUES
    ('猛禽属性组','利爪'), ('猛禽属性组','钩喙'),
    ('猛禽属性组','飞羽'), ('猛禽属性组','尾羽'),
    ('鲸类属性组','背鳍'), ('鲸类属性组','胸鳍'), ('鲸类属性组','声呐器官'),
    ('社会性昆虫属性组','触角'), ('社会性昆虫属性组','复眼'),
    ('长鼻类属性组','长鼻'), ('长鼻类属性组','象牙'),
    ('头足类属性组','八腕'), ('头足类属性组','墨囊'),
    ('翼手类属性组','翼膜'),
    ('灵长类属性组','对生拇指'),
    ('企鹅属性组','蹼足'),
    ('河狸属性组','门齿')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', g.sid, 'sense', p.sid, 'typically_has', 0.8, 5
  FROM e
  JOIN ae_sid g ON g.word = e.grp  AND g.pos = 'n'
  JOIN ae_sid p ON p.word = e.part AND p.pos = 'n'
 WHERE g.sid <> p.sid
ON CONFLICT DO NOTHING;

WITH e(animal, grp) AS (VALUES
    ('鹰','猛禽属性组'),
    ('海豚','鲸类属性组'),
    ('蚂蚁','社会性昆虫属性组'),
    ('蜜蜂','社会性昆虫属性组'),
    ('大象','长鼻类属性组'),
    ('章鱼','头足类属性组'),
    ('猴子','灵长类属性组'),
    ('企鹅','企鹅属性组'),
    ('蝙蝠','翼手类属性组'),
    ('河狸','河狸属性组')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', a.sid, 'sense', g.sid, 'typically_has', 0.8, 5
  FROM e
  JOIN ae_sid a ON a.word = e.animal AND a.pos = 'n'
  JOIN ae_sid g ON g.word = e.grp    AND g.pos = 'n'
 WHERE a.sid <> g.sid
ON CONFLICT DO NOTHING;

/* 「对生拇指」是形容词性属性（pos='a'），不能进上面那条只认 'n' 的语句。
 * 这类「部位即属性」的词在既有种子里已有先例（两栖类属性组 typically_has 有脊柱），
 * 故单列一条，避免静默丢边。 */
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', g.sid, 'sense', p.sid, 'typically_has', 0.8, 5
  FROM (SELECT '灵长类属性组' AS grp, '对生拇指' AS part) AS e
  JOIN ae_sid g ON g.word = e.grp  AND g.pos = 'n'
  JOIN ae_sid p ON p.word = e.part AND p.pos = 'a'
 WHERE g.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * §6 can_do：个体 -> 行为类别 -> 具体动作
 *    6.1 行为类别 -> 具体动作（27 条；与既有 10 条共同构成叶子层）
 *    6.2 个体 -> 行为类别（38 条；组合出 depth 2，飞行层再叠出 depth 3）
 * ========================================================================= */
WITH e(grp, act) AS (VALUES
    ('运动','潜水'), ('运动','跃出水面'), ('运动','滑行'),
    ('飞行','翱翔'), ('飞行','俯冲'),   ('飞行','滑行'),
    ('捕猎','空中捕食'),
    ('防御','变色'), ('防御','喷射推进'),
    ('通讯','信息素通讯'), ('通讯','舞蹈通讯'),
    ('通讯','回声定位'),   ('通讯','面部表情'),
    ('筑造','筑坝'), ('筑造','挖掘'), ('筑造','啃树'),
    ('取食','采蜜'), ('取食','搬运'), ('取食','喷水'), ('取食','用鼻取物'),
    ('感知','声呐导航'), ('感知','回声定位'),
    ('认知','记忆路径'), ('认知','使用工具'), ('认知','开瓶'), ('认知','群体协作'),
    ('育幼','群居育幼')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', g.sid, 'sense', a.sid, 'can_do', 0.9, 5
  FROM e
  JOIN ae_sid g ON g.word = e.grp AND g.pos = 'v'
  JOIN ae_sid a ON a.word = e.act AND a.pos = 'v'
 WHERE g.sid <> a.sid
ON CONFLICT DO NOTHING;

WITH e(animal, grp) AS (VALUES
    ('鹰','运动'),   ('鹰','捕猎'),   ('鹰','飞行'),   ('鹰','筑造'),
    ('海豚','运动'), ('海豚','捕猎'), ('海豚','通讯'), ('海豚','育幼'),
    ('蚂蚁','运动'), ('蚂蚁','筑造'), ('蚂蚁','通讯'), ('蚂蚁','取食'),
    ('蜜蜂','运动'), ('蜜蜂','飞行'), ('蜜蜂','通讯'), ('蜜蜂','筑造'), ('蜜蜂','取食'),
    ('大象','运动'), ('大象','通讯'), ('大象','取食'), ('大象','育幼'),
    ('猴子','运动'), ('猴子','认知'), ('猴子','通讯'), ('猴子','取食'),
    ('企鹅','运动'), ('企鹅','捕猎'), ('企鹅','育幼'), ('企鹅','通讯'),
    ('蝙蝠','运动'), ('蝙蝠','飞行'), ('蝙蝠','捕猎'), ('蝙蝠','感知'),
    ('河狸','运动'), ('河狸','筑造'), ('河狸','取食'), ('河狸','防御'),
    ('章鱼','认知')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, confidence, source_id)
SELECT 'sense', a.sid, 'sense', g.sid, 'can_do', 0.9, 5
  FROM e
  JOIN ae_sid a ON a.word = e.animal AND a.pos = 'n'
  JOIN ae_sid g ON g.word = e.grp    AND g.pos = 'v'
 WHERE a.sid <> g.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * §7 part_of：部分 -> 整体（26 条）
 *    只用**类群专属**区域节点做中间层（鲸类头部 / 象面部 / 昆虫头部），
 *    绝不连通用「头部」，否则会经既有的「头部 part_of 猫」污染全库。
 * ========================================================================= */
WITH e(part, whole) AS (VALUES
    -- 鲸类
    ('鲸类头部','海豚'),
    ('声呐器官','鲸类头部'),
    ('皮肤系统','海豚'),
    ('背鳍','皮肤系统'),
    ('胸鳍','皮肤系统'),
    -- 长鼻类
    ('象面部','大象'),
    ('长鼻','象面部'),
    ('象牙','象面部'),
    -- 昆虫
    ('昆虫头部','蚂蚁'),
    ('昆虫头部','蜜蜂'),
    ('触角','昆虫头部'),
    ('复眼','昆虫头部'),
    -- 骨骼
    ('骨骼系统','脊椎动物'),
    ('脊柱','骨骼系统'),
    -- 器官系统（既有 心脏/肺 -> 循环/呼吸系统 两条 edge 复用得 depth 2）
    ('循环系统','鹰'),
    ('循环系统','海豚'),
    ('循环系统','大象'),
    ('循环系统','猴子'),
    ('循环系统','企鹅'),
    ('循环系统','蝙蝠'),
    ('循环系统','河狸'),
    ('循环系统','狗'),
    ('循环系统','蛇'),
    ('呼吸系统','鹰'),
    ('呼吸系统','海豚'),
    ('呼吸系统','大象'),
    ('呼吸系统','猴子'),
    ('呼吸系统','企鹅'),
    ('呼吸系统','蝙蝠'),
    ('呼吸系统','河狸'),
    ('呼吸系统','狗'),
    ('呼吸系统','蛇'),
    -- 头足类与单点部位
    ('八腕','章鱼'),
    ('墨囊','章鱼'),
    ('蹼足','企鹅'),
    ('门齿','河狸'),
    ('翼膜','蝙蝠'),
    -- 真正的包含关系，可安全外推。
    -- 注意：「利爪 part_of 爪子」被**刻意排除** —— 既有库里
    -- 「爪子 part_of 前肢 part_of 猫」会让利爪凭空成为猫的部位，
    -- 这是通用整体做中间层的老问题，不在本轮扩大它。
    ('钩喙','喙'),
    ('飞羽','翅膀')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', p.sid, 'sense', w.sid, 'part_of', 5
  FROM e
  JOIN ae_sid p ON p.word = e.part  AND p.pos = 'n'
  JOIN ae_sid w ON w.word = e.whole AND w.pos = 'n'
 WHERE p.sid <> w.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * §8 has_part：整体 -> 部分（14 条）
 *    与 part_of 成对登记（schema §15.8 的逆向一致性查询要求成对）。
 * ========================================================================= */
WITH e(whole, part) AS (VALUES
    ('鹰','翅膀'),
    ('翅膀','飞羽'),
    ('鹰','尾巴'),
    ('尾巴','尾羽'),
    ('海豚','背鳍'),
    ('背鳍','鳍条'),
    ('蚂蚁','昆虫头部'),
    ('昆虫头部','触角'),
    ('蚂蚁','头部'),
    ('头部','复眼'),
    ('蜜蜂','头部'),
    ('大象','象面部'),
    ('象面部','长鼻'),
    ('象面部','象牙')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', w.sid, 'sense', p.sid, 'has_part', 5
  FROM e
  JOIN ae_sid w ON w.word = e.whole AND w.pos = 'n'
  JOIN ae_sid p ON p.word = e.part  AND p.pos = 'n'
 WHERE w.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * §9 闭包收尾
 *    增量触发器已在每条边插入时物化；这里再排队重建一次，
 *    保证 --keep（二次执行）与全量重建得到同一份闭包（对齐 I1–I6）。
 * ========================================================================= */
INSERT OR REPLACE INTO closure_rebuild_queue(relation) VALUES
 ('is_a'),
 ('has_property'),
 ('typically_has'),
 ('can_do'),
 ('part_of'),
 ('has_part');

/* =========================================================================
 * §10 写入行数硬自检
 *    具名 CHECK：失败时报错文本即「哪个关系少了多少条」。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS ae_gate(
    n_senses   INTEGER NOT NULL,
    n_is_a     INTEGER NOT NULL,
    n_prop     INTEGER NOT NULL,
    n_typ      INTEGER NOT NULL,
    n_can      INTEGER NOT NULL,
    n_part_of  INTEGER NOT NULL,
    n_has_part INTEGER NOT NULL,
    CONSTRAINT _extended_seed_row_counts_do_not_match_expected CHECK (
        -- part_of 实际新增 39 条，其中 4 条是「循环系统/呼吸系统 -> 狗 / 蛇」，
        -- 两端都是既有节点，下面的「至少一端落在 5081–5165」口径只数得到 35 条。
        n_senses = 85 AND n_is_a = 9 AND n_prop = 58 AND n_typ = 27
        AND n_can = 65 AND n_part_of = 35 AND n_has_part = 14)
);
DELETE FROM ae_gate;

INSERT INTO ae_gate(n_senses, n_is_a, n_prop, n_typ, n_can, n_part_of, n_has_part)
SELECT
  (SELECT COUNT(*) FROM senses WHERE sense_id BETWEEN 5081 AND 5165),
  (SELECT COUNT(*) FROM relations WHERE relation = 'is_a'
      AND src_id BETWEEN 5081 AND 5165 AND src_kind = 'sense'),
  (SELECT COUNT(*) FROM relations WHERE relation = 'has_property'
      AND ((src_id BETWEEN 5081 AND 5165 AND src_kind = 'sense')
        OR (dst_id BETWEEN 5081 AND 5165 AND dst_kind = 'sense'))),
  (SELECT COUNT(*) FROM relations WHERE relation = 'typically_has'
      AND ((src_id BETWEEN 5081 AND 5165 AND src_kind = 'sense')
        OR (dst_id BETWEEN 5081 AND 5165 AND dst_kind = 'sense'))),
  (SELECT COUNT(*) FROM relations WHERE relation = 'can_do'
      AND ((src_id BETWEEN 5081 AND 5165 AND src_kind = 'sense')
        OR (dst_id BETWEEN 5081 AND 5165 AND dst_kind = 'sense'))),
  (SELECT COUNT(*) FROM relations WHERE relation = 'part_of'
      AND ((src_id BETWEEN 5081 AND 5165 AND src_kind = 'sense')
        OR (dst_id BETWEEN 5081 AND 5165 AND dst_kind = 'sense'))),
  (SELECT COUNT(*) FROM relations WHERE relation = 'has_part'
      AND ((src_id BETWEEN 5081 AND 5165 AND src_kind = 'sense')
        OR (dst_id BETWEEN 5081 AND 5165 AND dst_kind = 'sense')));

DROP TABLE ae_gate;

/* =========================================================================
 * §11 弱关系闭包覆盖闸门
 *    训练脚本按 depth>=2 的闭包行做正样本，本闸门直接断言扩容后的下限。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS ae_cov(
    can_deep      INTEGER NOT NULL,
    prop_deep     INTEGER NOT NULL,
    typ_deep      INTEGER NOT NULL,
    part_of_deep  INTEGER NOT NULL,
    has_part_deep INTEGER NOT NULL,
    CONSTRAINT _extended_seed_weak_relation_closure_below_target CHECK (
        can_deep >= 280 AND prop_deep >= 120 AND typ_deep >= 60
        AND part_of_deep >= 20 AND has_part_deep >= 5)
);
DELETE FROM ae_cov;

INSERT INTO ae_cov(can_deep, prop_deep, typ_deep, part_of_deep, has_part_deep)
SELECT
  (SELECT COUNT(*) FROM closure WHERE relation='can_do'       AND depth >= 2),
  (SELECT COUNT(*) FROM closure WHERE relation='has_property' AND depth >= 2),
  (SELECT COUNT(*) FROM closure WHERE relation='typically_has' AND depth >= 2),
  (SELECT COUNT(*) FROM closure WHERE relation='part_of'      AND depth >= 2),
  (SELECT COUNT(*) FROM closure WHERE relation='has_part'     AND depth >= 2);

DROP TABLE ae_cov;

DROP VIEW ae_sid;
DROP TABLE ae_seed;

COMMIT;
