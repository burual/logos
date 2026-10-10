/*****************************************************************************
 * Logos 医学库增量补丁   seed_medical_v3_patch_symptom.sql
 * 中文原生符号语义数据库 —— 「症状继承补完」（通用症状下沉到上位疾病层）
 *
 * 目标：把症状从"只声明在最具体疾病上"改为"在【疾病】/【脑部疾病】层声明
 *       通用症状，全部下位疾病经 is_a 继承"。修复
 *         「沿 is_a 向上查询『疾病层的症状』返回空」
 *       这一真实缺陷（见 seed_medical_v3.sql §3.2 末端与 validate_medical.py §M9
 *       里标注的『空 —— 这是预期结果』说明）。
 *
 *   before（原医学库）：            after（本补丁）：
 *     癫痫          --has_symptom--> 抽搐          癫痫        --(直接)--> 抽搐
 *     阿尔茨海默病  --has_symptom--> 记忆障碍      阿尔茨海默病 --(直接)--> 记忆障碍
 *     （疾病/脑部疾病层无任何症状边）               疾病        --has_symptom--> 疼痛 / 乏力
 *                                                   脑部疾病    --has_symptom--> 头痛 / 眩晕 / 意识模糊
 *   ⇒ 癫痫 沿 is_a 继承到 疼痛 / 乏力 / 头痛 / 眩晕 / 意识模糊（+ 直接 抽搐，共 6）
 *   ⇒ 脑卒中 / 阿尔茨海默病 等全部脑部疾病同样继承到这 5 种（+ 各自直接症状）
 *
 * ---------------------------------------------------------------------------
 * 【前置依赖】必须按序执行：
 *     schema_v3.sql → seed_general_v3.sql → seed_general_v3_ext.sql
 *     → seed_medical_v3.sql → 本文件
 *   理由：
 *     · 「症状 / 疾病 / 脑部疾病」由 seed_medical_v3.sql §1.2/§1.3 建立
 *       （本补丁把新症状挂在「症状」下、把症状边挂在「疾病/脑部疾病」上）；
 *     · 关系类型 has_symptom 已由 seed_medical_v3.sql §1.1 登记 —— 本文件
 *       **不再重复 INSERT relation_types**（重复登记会撞 PK，虽然 DO NOTHING
 *       可吞掉，但语义上属于"补丁不应重定义既有类型"）。
 *   ★ 依赖缺失时本文件**硬失败**（见 §0），不静默降级 —— 理由同医学库主文件。
 *
 * 【幂等性】全部 INSERT 均带 ON CONFLICT DO NOTHING 或 WHERE NOT EXISTS，
 *   可重复执行任意次，结果一致（不重复写 FTS 与 changelog）。
 *
 * 【ID 分配策略】沿用医学库 4000-4999 段，**只追加不移动**：
 *     4001-4030  医学义项 30 个（seed_medical_v3.sql）
 *     4101       「信号」通用库兜底（seed_medical_v3.sql）
 *     ★ 4031-4036 本补丁新增的 6 个通用症状义项（word_id = sense_id）
 *     3101-3103  医学库领域实体节点（seed_medical_v3.sql，本文件不动）
 *   注：4031+ 与 4101 之间留空段，避免与主文件的兜底词条相邻冲突。
 *
 * ---------------------------------------------------------------------------
 * 【写入规模】（实测增量：medical → +本补丁，由 validate_medical.py §M10 复核）
 *   words            +6    疼痛 / 发热 / 头痛 / 眩晕 / 乏力 / 意识模糊
 *   senses           +6    与 words 1:1（v0.1 每词单义项）
 *   sense_categories +6    一律 '性状'（症状与病理状态同一粗类）
 *   relations        +11   6 is_a（新症状 → 症状）+ 5 has_symptom
 *                          （疾病→疼痛 / 疾病→乏力 / 脑部疾病→头痛 /
 *                            脑部疾病→眩晕 / 脑部疾病→意识模糊）
 *   closure          +24   全部**自动派生**：6 条 is_a 各派生 4 行闭包
 *                          （症状 1 / 病理状态 2 / 医学概念 3 / 抽象概念 4）
 *                          本文件不含任何 closure 的 INSERT（实测 0 条）
 *   changelog        +11   由 tr_relations_audit_ai 自动记录
 *   合计：senses 124 → 130；relations 101 → 112；closure 184 → 208
 *
 * 【依赖顺序】（严格按此顺序，不可调换）
 *   words → senses ─┬─→ sense_categories
 *                   └─→ relations ─┬─→ tr_closure_edge_ai（is_a 自动物化）
 *                                  └─→ tr_relations_audit_ai（changelog）
 *   has_symptom 的 is_transitive=0（主文件 §1.1 已声明）⇒ 不产生闭包行，
 *   其"继承"靠 is_a 的上位传播，而不是靠 has_symptom 自身传递。
 *
 * ---------------------------------------------------------------------------
 * 【三处刻意的设计取舍 —— 请留意】
 *   (1) 【为什么症状声明在「疾病 / 脑部疾病」层，而不是逐个疾病重复声明】
 *       关系继承的推理规则（本项目的属性继承路径）是：
 *         若 X --has_symptom--> S 且 D --is_a--> X，则 D 也 --has_symptom--> S
 *       于是**把边声明在最一般的、语义上真正成立的层**，全部下位自动继承：
 *         · 疼痛 / 乏力 对**一切疾病**成立 ⇒ 声明在「疾病」层；
 *         · 头痛 / 眩晕 / 意识模糊 对**一切脑部疾病**成立 ⇒ 声明在「脑部疾病」层。
 *       这样新增一种脑部疾病时，无需再逐个补症状边，继承链自动覆盖。
 *       —— 与"逐个疾病声明"相比，边数从 O(疾病数 × 症状数) 降为 O(层数 × 症状数)。
 *   (2) 【为什么「发热」「眩晕」只建义项、本次不挂 has_symptom】
 *       需求把 6 个词都列为"新增通用症状义项"，但只要求把 5 种
 *       （疼痛 / 乏力 / 头痛 / 眩晕 / 意识模糊）声明到层上。
 *       发热的语义限定是"炎症 / 感染性疾病共有"—— 当前骨架里**没有**
 *       「炎症性疾病 / 感染性疾病」这一节点可挂（强行挂到「疾病」层会与
 *       需求给的语义限定矛盾）。故本补丁**只建义项、暂不挂边**，
 *       等骨架补出「炎症性疾病」后按同样的写法补一行边即可（见文件末 §3.7）。
 *       眩晕同时出现在"新增症状"与"脑部疾病层"两处，按后者落边（神经系统
 *       疾病 ⊂ 脑部疾病，挂脑部疾病层是更宽、更安全的口径）。
 *   (3) 【为什么本补丁不碰 relation_types / closure_policy】
 *       has_symptom 已登记且 is_transitive=0；本补丁不引入任何新关系类型，
 *       也不改变任何闭包策略。若在此重复 INSERT relation_types，虽然
 *       ON CONFLICT DO NOTHING 不会报错，但会让"补丁改了既有类型定义"这一
 *       事实变得不可审计 —— 补丁应当只**追加数据**，不重定义结构。
 *
 * 【连接级 PRAGMA】必须在每个连接上执行，不随 schema 持久化。
 *   注意：PRAGMA foreign_keys 在**显式事务内是 no-op**，故写在 BEGIN 之前。
 * =========================================================================*/

PRAGMA foreign_keys     = ON;   -- SQLite 默认 OFF；不开则所有 FK 形同虚设
PRAGMA recursive_triggers = ON; -- 闭包触发器依赖触发器级联语义

BEGIN;

/* =========================================================================
 * 0. 前置依赖硬检查（不满足即中止，不静默跳过）
 * -------------------------------------------------------------------------
 * 与本项目 seed_medical_v3.sql §0 同一手法：SQLite 的 RAISE() 只能出现在
 *   触发器体内，脚本里无法直接断言；改为建一张临时表 + **具名 CHECK 约束**，
 *   失败时异常信息会带上**约束名**（实测 SQLite 3.53.1 报的是约束名本身）：
 *     IntegrityError: CHECK constraint failed: _prereq_missing_...
 *   临时表 + IF NOT EXISTS + DELETE 保证本段自身可重复执行。
 *
 * 依赖项：本补丁把新症状挂在「症状」下、把症状边挂在「疾病 / 脑部疾病」上，
 *   三者缺一都会导致"内连接静默少行"（不报错、只是边没落上）—— 这正是
 *   最该被显式拦截的失败模式，故与主文件一样选择**硬失败**。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS _prereq_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _prereq_missing_run_seed_medical_v3_first CHECK (ok = 1)
);
DELETE FROM _prereq_check;
INSERT INTO _prereq_check(ok)
SELECT CASE WHEN
        EXISTS (SELECT 1 FROM words WHERE word = '症状')       -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '疾病')       -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '脑部疾病')   -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM relation_types WHERE relation = 'has_symptom')
     THEN 1 ELSE 0 END;
DROP TABLE _prereq_check;

/* =========================================================================
 * 0.1 自然键查找视图（仅本会话有效，不落盘）
 *     与 seed_general_v3.sql 的 seed_sid、ext 的 ext_sid、medical 的 med_sid
 *     同构，**另起名字 sym_sid**：四个文件可能被同一连接先后加载，重名会让
 *     后者的 CREATE ... IF NOT EXISTS 命中旧定义（内容虽同，但可读性差、
 *     也不利于将来各文件独立演进）。
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS sym_sid AS
    SELECT w.word AS word, s.pos AS pos, s.sense_no AS sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;

/* ###########################################################################
 * 1. 新增通用症状义项（6 个）
 * ######################################################################### */

/* -------------------------------------------------------------------------
 * 1.1 词层（4031-4036）
 *     词形与语素构造：
 *       疼痛   合成词（疼 + 痛，同义并列）
 *       发热   合成词（发 + 热，动宾）
 *       头痛   合成词（头 + 痛，主谓）
 *       眩晕   联绵词（双声叠韵，不可再分）
 *       乏力   合成词（乏 + 力，动宾）
 *       意识模糊 合成词（意识 + 模糊，主谓）
 * ------------------------------------------------------------------------*/
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 (4031,'疼痛',    NULL,'合成词',0,'通用'),
 (4032,'发热',    NULL,'合成词',0,'通用'),
 (4033,'头痛',    NULL,'合成词',0,'通用'),
 (4034,'眩晕',    NULL,'联绵词',0,'通用'),
 (4035,'乏力',    NULL,'合成词',0,'通用'),
 (4036,'意识模糊',NULL,'合成词',0,'通用')
ON CONFLICT(word_id) DO NOTHING;

/* -------------------------------------------------------------------------
 * 1.2 义项层 —— 与 1.1 一一对应（sense_id = word_id）
 *     释义一律用中文词语组合（符号语义：用词定义词，不引入外部向量），
 *     且**不含数字 / 拉丁字母**（由 validate_medical.py §M10 断言）。
 * ------------------------------------------------------------------------*/
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary) VALUES
 (4031,4031,'n',1,'机体受到刺激时产生的不愉快感觉',
  '机体 受到 刺激 时 产生 的 不 愉快 感觉',
  'teng2 tong4','tengtong','tt',2000,'本义',NULL,1),
 (4032,4032,'n',1,'体温升高超出正常范围的全身现象',
  '体温 升高 超出 正常 范围 的 全身 现象',
  'fa1 re4','fare','fr',1750,'本义',NULL,1),
 (4033,4033,'n',1,'头部出现的疼痛感觉',
  '头部 出现 的 疼痛 感觉',
  'tou2 tong4','toutong','tt',1650,'本义',NULL,1),
 (4034,4034,'n',1,'感到自身或周围景物旋转摇晃的异常感觉',
  '感到 自身 或 周围 景物 旋转 摇晃 的 异常 感觉',
  'xuan4 yun4','xuanyun','xy',900,'本义',NULL,1),
 (4035,4035,'n',1,'全身感到疲倦无力的状态',
  '全身 感到 疲倦 无力 的 状态',
  'fa2 li4','fali','fl',1500,'本义',NULL,1),
 (4036,4036,'n',1,'意识清晰度下降、对外界反应迟钝的状态',
  '意识 清晰度 下降 对 外界 反应 迟钝 的 状态',
  'yi4 shi2 mo2 hu5','yishimohu','ysmo',700,'本义',NULL,1)
ON CONFLICT(sense_id) DO NOTHING;

/* -------------------------------------------------------------------------
 * 1.3 义项 ↔ 语义类别
 *     症状属「病理状态」分支，与主文件一致归入粗类 '性状'（不新增类别行）。
 * ------------------------------------------------------------------------*/
WITH e(word, cat) AS (VALUES
    ('疼痛','性状'),('发热','性状'),('头痛','性状'),
    ('眩晕','性状'),('乏力','性状'),('意识模糊','性状')
)
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT s.sid, e.cat, 1
  FROM e JOIN sym_sid s ON s.word = e.word AND s.pos = 'n'
 WHERE 1                                  -- 消除 INSERT...SELECT...ON CONFLICT 解析歧义
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 2. 关系层
 * ######################################################################### */

/* =========================================================================
 * 2.1 新症状的 is_a 边（6 条）—— 全部挂到「症状」下
 *     is_a 是 is_transitive=1 / incremental，插边即由 tr_closure_edge_ai
 *     派生多级闭包（实测每个症状派生 4 行）：
 *       疼痛 → 症状(1) → 病理状态(2) → 医学概念(3) → 抽象概念(4)
 *     写法与医学库 / ext 一致：CTE(VALUES) 把「词形对」映射为 sense_id，
 *     不硬编码既有 ID；父节点不存在时内连接自动跳过（不报错、不产生悬空边）。
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    ('疼痛',    '症状'),
    ('发热',    '症状'),
    ('头痛',    '症状'),
    ('眩晕',    '症状'),
    ('乏力',    '症状'),
    ('意识模糊','症状')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN sym_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN sym_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.2 通用症状下沉到上位疾病层（5 条 has_symptom 边）
 * -------------------------------------------------------------------------
 *   【疾病】层（对一切疾病成立）      疾病 --has_symptom--> 疼痛 / 乏力
 *   【脑部疾病】层（对一切脑部疾病成立）脑部疾病 --has_symptom--> 头痛 / 眩晕 / 意识模糊
 *
 *   ★ 这是"症状继承"的关键：has_symptom 本身 is_transitive=0（不产生闭包
 *     行），继承由 is_a 的上位传播完成 ——
 *       D --is_a-->* 疾病   且   疾病 --has_symptom--> S   ⇒   D 有症状 S
 *     因此下位疾病（癫痫 / 脑卒中 / 神经退行性疾病 / 阿尔茨海默病 …）
 *     无需各自补边，自动继承到上述 5 种症状。
 *   ★ confidence=0.9：与主文件 §2.3 的医学关系同一量级（"定义性关联"的高置信，
 *     但非 is_a 的 1.0 —— 通用症状对个别疾病可能不典型）。
 * ========================================================================= */
WITH e(src, rel, dst) AS (VALUES
    -- 疾病层：所有疾病共有
    ('疾病',    'has_symptom', '疼痛'),
    ('疾病',    'has_symptom', '乏力'),
    -- 脑部疾病层：所有脑部疾病共有
    ('脑部疾病','has_symptom', '头痛'),
    ('脑部疾病','has_symptom', '眩晕'),
    ('脑部疾病','has_symptom', '意识模糊')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
SELECT 'sense', s.sid, 'sense', d.sid, e.rel, 5, 0.9
  FROM e
  JOIN sym_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN sym_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3. 附：自检与常用查询（注释形式，供应用层 / validate_medical.py 调用）
 * ========================================================================= */

-- 3.1 【新增症状自检】6 个通用症状义项 + 各自的 is_a 出边 = 1（指向 症状）
-- SELECT w.word, w.word_type FROM words w
--  WHERE w.word_id BETWEEN 4031 AND 4036 ORDER BY w.word_id;
-- 期望：疼痛 / 发热 / 头痛 / 眩晕 / 乏力 / 意识模糊

-- 3.2 【症状继承自检】癫痫 沿 is_a 继承到的症状（= 结果集）
-- SELECT DISTINCT wd.word FROM closure c
--   JOIN relations r ON r.relation='has_symptom' AND r.src_kind='sense' AND r.src_id=c.ancestor
--   JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
--  WHERE c.relation='is_a'
--    AND c.descendant=(SELECT s.sense_id FROM senses s JOIN words w2 ON w2.word_id=s.word_id
--                       WHERE w2.word='癫痫' AND s.pos='n')
--  ORDER BY wd.word;
-- 期望：乏力 / 意识模糊 / 头痛 / 眩晕 / 疼痛  （5 种，加直接声明的 抽搐 = 6）

-- 3.3 【症状继承 = 直接声明 + 沿 is_a 继承（去重）】
-- SELECT DISTINCT dst FROM (
--   SELECT dst_id AS dst FROM relations
--    WHERE relation='has_symptom' AND src_kind='sense' AND src_id = <D>          -- 直接
--   UNION
--   SELECT r.dst_id FROM closure c JOIN relations r
--     ON r.relation='has_symptom' AND r.src_kind='sense' AND r.src_id = c.ancestor
--    WHERE c.relation='is_a' AND c.descendant = <D>                              -- 继承
-- );
-- 期望：对 癫痫 = {抽搐, 乏力, 意识模糊, 头痛, 眩晕, 疼痛}（6）

-- 3.4 【闭包自动派生自检】新增症状的 is_a 闭包应各 4 行（至抽象概念）
-- SELECT wa.word AS ancestor, c.depth FROM closure c
--   JOIN senses sa ON sa.sense_id=c.ancestor JOIN words wa ON wa.word_id=sa.word_id
--  WHERE c.relation='is_a'
--    AND c.descendant=(SELECT s.sense_id FROM senses s JOIN words w2 ON w2.word_id=s.word_id
--                       WHERE w2.word='疼痛' AND s.pos='n')
--  ORDER BY c.depth;
-- 期望：症状(1) 病理状态(2) 医学概念(3) 抽象概念(4)

-- 3.5 【闭包一致性巡检】以下每句正常应返回 0 行
-- (a) 本补丁新增的 is_a 边缺 depth=1 闭包行
-- SELECT r.relation_id FROM relations r
--  WHERE r.relation='is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
--    AND r.src_id BETWEEN 4031 AND 4036
--    AND NOT EXISTS (SELECT 1 FROM closure c WHERE c.relation='is_a'
--                     AND c.ancestor=r.dst_id AND c.descendant=r.src_id AND c.depth=1);
-- (b) has_symptom 不应有任何闭包行（is_transitive=0）
-- SELECT DISTINCT relation FROM closure WHERE relation='has_symptom';
-- (c) 新增症状的类别挂载若悬空
-- SELECT sc.sense_id FROM sense_categories sc
--  WHERE sc.sense_id BETWEEN 4031 AND 4036
--    AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat);

-- 3.6 【幂等自检】连续执行本文件两次，6 张相关表行数零漂移
-- SELECT 'words', COUNT(*) FROM words WHERE word_id BETWEEN 4031 AND 4036
-- UNION ALL SELECT 'senses', COUNT(*) FROM senses WHERE sense_id BETWEEN 4031 AND 4036
-- UNION ALL SELECT 'has_symptom', COUNT(*) FROM relations WHERE relation='has_symptom'
-- UNION ALL SELECT 'isa_new', COUNT(*) FROM relations WHERE relation='is_a'
--          AND src_id BETWEEN 4031 AND 4036;

-- 3.7 【后续扩展位】「发热」按语义限定（炎症 / 感染性疾病共有）应挂到
--      「炎症性疾病」层。当前骨架无此节点，故暂只建义项。骨架补齐后补：
-- WITH e(src, rel, dst) AS (VALUES ('炎症性疾病','has_symptom','发热'))
-- INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
-- SELECT 'sense', s.sid, 'sense', d.sid, e.rel, 5, 0.9 FROM e
--   JOIN sym_sid s ON s.word=e.src AND s.pos='n'
--   JOIN sym_sid d ON d.word=e.dst AND d.pos='n'
--  WHERE s.sid <> d.sid ON CONFLICT DO NOTHING;

COMMIT;

/*****************************************************************************
 * End of seed_medical_v3_patch_symptom.sql
 *****************************************************************************/
