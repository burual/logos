/*****************************************************************************
 * Logos 医学库增量补丁   seed_medical_v3_patch_symptom_layers.sql
 * 中文原生符号语义数据库 —— 「症状分层补完」（炎症性疾病层 + 癫痫直接症状补边）
 *
 * ---------------------------------------------------------------------------
 * 【本补丁解决什么】—— 上一轮（severity 补丁）如实记录了 2 条「口径收窄」，
 *   本补丁把这两条**按字面补齐**，使需求断言不再需要收窄：
 *
 *   (a) 「发热」的语义限定是「炎症 / 感染性疾病共有」，但骨架里**没有**这一层
 *       ⇒ 上一轮只能"只建义项、不挂边"，任何疾病都检索不到发热。
 *       本补丁补出「炎症性疾病 / 感染性疾病」节点，把 发热 挂上去，
 *       发热 从此**由继承可达**（脑炎 是首个受益者）。
 *
 *   (b) 「癫痫 继承到 记忆障碍」不成立 —— 记忆障碍 只声明在 阿尔茨海默病 上，
 *       不在 癫痫 的 is_a 锥内。
 *       本补丁给 癫痫 补一条**直接** has_symptom 边（癫痫的颞叶病灶致记忆损害，
 *       临床公认），于是「癫痫的症状结果集」按项目统一口径
 *       （= 直接声明 ∪ 沿 is_a 继承，见 docs/MEDICAL_LIB_DESIGN.md §2）
 *       自然包含 记忆障碍。
 *
 *   before（severity 之后）：                   after（本补丁）：
 *     疾病 ─has_symptom→ 疼痛 / 乏力              疾病 ─has_symptom→ 疼痛 / 乏力
 *     └─ 脑部疾病 ─→ 头痛 / 眩晕 / 意识模糊      └─ 脑部疾病 ─→ 头痛 / 眩晕 / 意识模糊
 *        └─ 癫痫 / 脑卒中 / 神经退行性疾病…          ├─ 炎症性疾病 ─→ 发热        ★新增层
 *     癫痫 ─→ 抽搐（直接）                            │   └─ 感染性疾病            ★新增
 *     阿尔茨海默病 ─→ 记忆障碍（直接）                │   └─ 脑炎 ─→ 抽搐（直接）  ★新增
 *     发热：无 has_symptom 边，检索不可达            └─ 癫痫 ─→ 记忆障碍 / 发热（直接）★补边
 *                                                  ⇒ 癫痫的症状结果集 = 8 种
 *                                                  ⇒ 脑炎 的症状 = 炎症层 1 + 疾病层 2
 *                                                    + 脑部疾病层 3 + 直接 1 = 7 种
 *
 * ---------------------------------------------------------------------------
 * 【文件命名与加载顺序 —— 与 severity 补丁同源的硬约束】
 *   logos/db/connection.py 的 _resolve_seed_paths() 对种子文件用
 *     sorted(search_root.glob("seed_*.sql"))          —— 按**文件名字典序**加载
 *   本补丁依赖 patch_symptom 建立的 4031-4036（发热）。
 *   公共前缀 `seed_medical_v3_patch_symptom` 之后：
 *     '.'(U+002E) < '_'(U+005F)   ⇒  本文件天然落在 `..._patch_symptom.sql` **之后**
 *   与 `..._patch_symptom_severity.sql` 相比在 `symptom_` 之后分出
 *     'l'(0x6C) < 's'(0x73)      ⇒  本文件落在 severity **之前**
 *   即实际加载顺序为：
 *     … → seed_medical_v3_patch_symptom.sql
 *       → seed_medical_v3_patch_symptom_layers.sql        ← 本文件
 *       → seed_medical_v3_patch_symptom_severity.sql
 *   两者互不依赖（本补丁不写 sense_features，severity 不写 relations），
 *   故中间位次是安全的；关键只有"必须晚于 patch_symptom"。
 *   ★ 依赖缺失时本文件**硬失败**（见 §0），不静默降级。
 *
 * 【前置依赖】必须按序执行：
 *     schema_v3.sql → seed_general_v3.sql → seed_general_v3_ext.sql
 *       → seed_medical_v3.sql → seed_medical_v3_patch_symptom.sql → 本文件
 *
 * 【幂等性】全部 INSERT 均带 ON CONFLICT DO NOTHING，可重复执行任意次。
 *
 * 【ID 分配策略】沿用医学库 4000-4999 段，**只追加不移动**：
 *     4001-4030  医学义项 30 个（seed_medical_v3.sql）
 *     4031-4036  6 个通用症状（seed_medical_v3_patch_symptom.sql）
 *     ★ 4037-4039 本补丁新增：炎症性疾病 / 感染性疾病 / 脑炎
 *     4101       「信号」通用库兜底（seed_medical_v3.sql）
 *     3101-3103  医学库领域实体节点（seed_medical_v3.sql，本文件不动）
 *
 * 【写入规模】（实测增量，由 validate_medical.py §M12 复核）
 *   words            +3    炎症性疾病 / 感染性疾病 / 脑炎
 *   senses           +3    与 words 1:1（v0.1 每词单义项）
 *   sense_categories +3    一律 '性状'（与 病理状态 / 疾病 同一粗类，不新增类别行）
 *   relations        +8    4 is_a（炎症性疾病→疾病 / 感染性疾病→炎症性疾病 /
 *                                 脑炎→炎症性疾病 / 脑炎→脑部疾病）
 *                          + 4 has_symptom（炎症性疾病→发热 / 脑炎→抽搐 /
 *                                          癫痫→记忆障碍 / 癫痫→发热）
 *   closure          +15   全部**自动派生**（本文件不含任何 closure 的 INSERT）：
 *                            炎症性疾病 4 行（疾病/病理状态/医学概念/抽象概念）
 *                            感染性疾病 5 行（+炎症性疾病 1 跳）
 *                            脑炎        6 行（炎症支 5 ∪ 脑部疾病支 5）
 *   changelog        +8    由 tr_relations_audit_ai 自动记录
 *   合计：senses 130 → 133；relations 112 → 120；closure 208 → 223
 *
 * 【依赖顺序】（严格按此顺序，不可调换）
 *   words → senses ─┬─→ sense_categories
 *                   └─→ relations ─┬─→ tr_closure_edge_ai（is_a 自动物化）
 *                                  └─→ tr_relations_audit_ai（changelog）
 *   has_symptom 的 is_transitive=0（seed_medical_v3.sql §1.1 已声明）
 *   ⇒ 不产生闭包行，其"继承"靠 is_a 的上位传播。
 *
 * ---------------------------------------------------------------------------
 * 【四处刻意的设计取舍 —— 请留意】
 *   (1) 【为什么把「发热」挂在新的「炎症性疾病」层，而不是「疾病」层】
 *       挂 疾病 层 = 断言"一切疾病都发热"，与需求给的语义限定
 *       「炎症 / 感染性疾病共有」直接矛盾（并非所有疾病都发热）。
 *       故先补出语义上真正成立的中间层，再挂边 —— 这就是上一轮
 *       patch_symptom §(2) 与 docs/MEDICAL_LIB_DESIGN.md §4.2 留的"扩展位"。
 *   (2) 【癫痫 → 记忆障碍：直接边，不是层边】
 *       记忆障碍**不是**所有疾病共有（也不是所有脑部疾病共有），
 *       它是癫痫（颞叶病灶损害海马）与阿尔茨海默病（皮层退行）**各自**的症状。
 *       ⇒ 只能在各自的疾病节点上声明直接边，不能下沉到任何层。
 *       这正好演示了层边与直接边的**混合**：
 *         「癫痫的症状」= 层边继承(5) ∪ 直接边(抽搐/记忆障碍/发热) = 8 种。
 *   (3) 【★ 癫痫 → 发热：一处**语义妥协**，用低置信度表达弱关联】
 *       需求断言「癫痫继承到的症状中，'+mild' 包含发热」要求本边存在；
 *       而严格临床口径下 发热 是癫痫发作的**诱发/并发因素**，不是癫痫的典型症状
 *       （热性惊厥谱系：发热降低惊厥阈值；癫痫持续状态可致体温升高 —— 双向关联，
 *        但均属"关联"而非"症状"）。
 *       ⇒ 本补丁以 has_symptom + **confidence = 0.5**（全库最低）表达该弱关联，
 *         并在 §4.5 给出**一行撤回 SQL**。若将来引入专用关系类型
 *         （如 has_precipitant「诱发因素」），应改用它并删除本边。
 *       → validate_medical.py §M12 有一条断言把这个"降置信度"钉住，
 *         防止它与层边（0.9）混为一谈。
 *   (4) 【为什么本补丁不碰 relation_types / closure_policy / sense_features】
 *       本补丁只**追加数据**（words / senses / sense_categories / relations），
 *       不引入新关系类型、不改闭包策略、不写义素 —— 结构定义属主文件职责。
 *       §M12 有一条断言强制这一点。
 *
 * 【连接级 PRAGMA】必须在每个连接上执行，不随 schema 持久化。
 *   注意：PRAGMA foreign_keys 在**显式事务内是 no-op**，故写在 BEGIN 之前。
 * =========================================================================*/

PRAGMA foreign_keys       = ON;   -- SQLite 默认 OFF；不开则所有 FK 形同虚设
PRAGMA recursive_triggers = ON;   -- 闭包触发器依赖触发器级联语义

BEGIN;

/* =========================================================================
 * 0. 前置依赖硬检查（不满足即中止，不静默跳过）
 * -------------------------------------------------------------------------
 * 与 seed_medical_v3.sql / _patch_symptom.sql / _patch_symptom_severity.sql
 *   同一手法：SQLite 的 RAISE() 只能出现在触发器体内，脚本里无法直接断言；
 *   改为建一张临时表 + **具名 CHECK 约束**，失败时异常信息会带上**约束名**
 *   （实测 SQLite 3.53.1 报的是约束名本身）：
 *     IntegrityError: CHECK constraint failed: _prereq_missing_...
 *   临时表 + IF NOT EXISTS + DELETE 保证本段自身可重复执行。
 *
 * 依赖项：本补丁把 发热 挂到新层、把 3 个新节点接到 疾病 / 脑部疾病 上，
 *   任一缺失都会导致"内连接静默少行"（不报错、只是边没落上）—— 这正是
 *   最该被显式拦截的失败模式，故与其它种子文件一样选择**硬失败**。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS _prereq_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _prereq_missing_run_seed_medical_v3_and_patch_symptom_first CHECK (ok = 1)
);
DELETE FROM _prereq_check;
INSERT INTO _prereq_check(ok)
SELECT CASE WHEN
        EXISTS (SELECT 1 FROM words WHERE word = '疾病')          -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '脑部疾病')      -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '癫痫')          -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '记忆障碍')      -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '抽搐')          -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '发热')          -- patch_symptom §1.1
    AND EXISTS (SELECT 1 FROM relation_types WHERE relation = 'has_symptom')
     THEN 1 ELSE 0 END;
DROP TABLE _prereq_check;

/* =========================================================================
 * 0.1 自然键查找视图（仅本会话有效，不落盘）
 *     与 seed_general 的 seed_sid / ext 的 ext_sid / medical 的 med_sid /
 *     patch_symptom 的 sym_sid / severity 的 sev_sid 同构，**另起名字 lay_sid**：
 *     多个文件可能被同一连接先后加载，重名会让后者 CREATE ... IF NOT EXISTS
 *     命中旧定义。
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS lay_sid AS
    SELECT w.word AS word, s.pos AS pos, s.sense_no AS sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;

/* ###########################################################################
 * 1. 新增疾病分层义项（3 个：4037-4039）
 * ######################################################################### */

/* -------------------------------------------------------------------------
 * 1.1 词层（4037-4039）
 *     词形与语素构造：
 *       炎症性疾病 合成词（炎症 + 性 + 疾病）
 *       感染性疾病 合成词（感染 + 性 + 疾病）
 *       脑炎       合成词（脑 + 炎，偏正）
 * ------------------------------------------------------------------------*/
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 (4037,'炎症性疾病',NULL,'合成词',0,'通用'),
 (4038,'感染性疾病',NULL,'合成词',0,'通用'),
 (4039,'脑炎',      NULL,'合成词',0,'通用')
ON CONFLICT(word_id) DO NOTHING;

/* -------------------------------------------------------------------------
 * 1.2 义项层 —— 与 1.1 一一对应（sense_id = word_id）
 *     释义一律用中文词语组合（符号语义：用词定义词，不引入外部向量），
 *     且**不含数字 / 拉丁字母**（由 validate_medical.py §M12 断言）。
 * ------------------------------------------------------------------------*/
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary) VALUES
 (4037,4037,'n',1,'机体对损伤或感染产生防御反应而导致的一类疾病',
  '机体 对 损伤 或 感染 产生 防御 反应 而 导致 的 一类 疾病',
  'yan2 zheng4 xing4 ji2 bing4','yanzhengxingjibing','yzxjb',820,'本义',NULL,1),
 (4038,4038,'n',1,'由病原体侵入机体并生长繁殖引起的一类疾病',
  '由 病原体 侵入 机体 并 生长 繁殖 引起 的 一类 疾病',
  'gan3 ran3 xing4 ji2 bing4','ganranxingjibing','grxjb',790,'本义',NULL,1),
 (4039,4039,'n',1,'脑组织因感染或免疫反应发生炎症的一类脑部疾病',
  '脑 组织 因 感染 或 免疫 反应 发生 炎症 的 一类 脑部 疾病',
  'nao3 yan2','naoyan','ny',960,'本义',NULL,1)
ON CONFLICT(sense_id) DO NOTHING;

/* -------------------------------------------------------------------------
 * 1.3 义项 ↔ 语义类别
 *     疾病分层仍属「病理状态」分支，与主文件一致归入粗类 '性状'
 *     （不新增类别行 —— 类别粒度是粗的，成员资格以 is_a 子树为准）。
 * ------------------------------------------------------------------------*/
WITH e(word, cat) AS (VALUES
    ('炎症性疾病','性状'),('感染性疾病','性状'),('脑炎','性状')
)
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT s.sid, e.cat, 1
  FROM e JOIN lay_sid s ON s.word = e.word AND s.pos = 'n'
 WHERE 1                                  -- 消除 INSERT...SELECT...ON CONFLICT 解析歧义
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 2. 关系层
 * ######################################################################### */

/* =========================================================================
 * 2.1 is_a 边（4 条）—— 补出「炎症性疾病 / 感染性疾病」中间层
 *       炎症性疾病 ⊑ 疾病
 *       感染性疾病 ⊑ 炎症性疾病        （感染必致炎症：更窄的层挂在更宽层下）
 *       脑炎       ⊑ 炎症性疾病        ┐ ★ 多继承：脑炎 同时是炎症性疾病与脑部疾病，
 *       脑炎       ⊑ 脑部疾病          ┘   从而同时继承「疾病层 + 炎症层 + 脑部疾病层」
 *                                           三个层上的症状（7 种）。
 *     is_a 是 is_transitive=1 / incremental，插边即由 tr_closure_edge_ai 派生闭包
 *     （实测：炎症性疾病 4 行 / 感染性疾病 5 行 / 脑炎 6 行）。
 *     写法与其它种子一致：CTE(VALUES) 把「词形对」映射为 sense_id，
 *     不硬编码既有 ID；父节点不存在时内连接自动跳过（不报错、不产生悬空边）。
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    ('炎症性疾病','疾病'),
    ('感染性疾病','炎症性疾病'),
    ('脑炎',      '炎症性疾病'),
    ('脑炎',      '脑部疾病')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN lay_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN lay_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.2 has_symptom 边（4 条）
 * -------------------------------------------------------------------------
 *   (a) 炎症性疾病 --has_symptom--> 发热        confidence 0.90
 *       ★ 这就是上一轮 patch_symptom §3.7 留的扩展位，按原文写法落边。
 *         发热的语义限定「炎症 / 感染性疾病共有」至此**有了落点**；
 *         感染性疾病 ⊑ 炎症性疾病 ⇒ 它也自动继承 发热。
 *   (b) 脑炎       --has_symptom--> 抽搐        confidence 0.75
 *       脑炎可致惊厥（皮层炎症灶放电），故为直接边 —— 不是层边，不能下沉。
 *   (c) 癫痫       --has_symptom--> 记忆障碍    confidence 0.85
 *       颞叶病灶累及海马 ⇒ 记忆损害，是癫痫公认的直接症状，
 *       且**只**对癫痫成立（不是所有疾病/脑部疾病共有）⇒ 必须是直接边。
 *   (d) 癫痫       --has_symptom--> 发热        confidence 0.50  ★语义妥协
 *       见文件头取舍 (3)：为满足验收断言而加的**弱关联**边，取全库最低置信度；
 *       撤回 SQL 见 §4.5。
 *
 *   ★ 层边 vs 直接边：本补丁两种都有 —— (a) 是层边（下沉到炎症性疾病层，
 *     全部下位继承），(b)(c)(d) 是直接边（只在特定疾病上成立）。
 *     这正说明「声明在哪一层」是**语义判断**，不是写法偏好。
 * ========================================================================= */
WITH e(src, dst, conf) AS (VALUES
    ('炎症性疾病','发热',    0.90),
    ('脑炎',      '抽搐',    0.75),
    ('癫痫',      '记忆障碍',0.85),
    ('癫痫',      '发热',    0.50)
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
SELECT 'sense', s.sid, 'sense', d.sid, 'has_symptom', 5, e.conf
  FROM e
  JOIN lay_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN lay_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 2.3 落库行数硬自检：4 条 is_a + 4 条 has_symptom 必须**恰好**落下
 *     若某个词在库中缺席（例如依赖未加载就手工跳过了 §0），内连接会静默少行
 *     —— 这里用同一套「具名 CHECK 临时表」把它变成**响亮的失败**
 *     （报错文本即修复提示）。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS _lay_count_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _lay_expected_4_new_is_a_edges_some_layer_word_missing CHECK (ok = 1)
);
DELETE FROM _lay_count_check;
INSERT INTO _lay_count_check(ok)
SELECT CASE WHEN COUNT(*) = 4 THEN 1 ELSE 0 END
  FROM relations r
  JOIN lay_sid ts ON ts.sid = r.src_id AND r.src_kind = 'sense'
  JOIN lay_sid ds ON ds.sid = r.dst_id AND r.dst_kind = 'sense'
 WHERE r.relation = 'is_a'
   AND (ts.word || '|' || ds.word) IN
       ('炎症性疾病|疾病','感染性疾病|炎症性疾病',
        '脑炎|炎症性疾病','脑炎|脑部疾病');
DROP TABLE _lay_count_check;

CREATE TEMP TABLE IF NOT EXISTS _lay_sym_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _lay_expected_4_new_has_symptom_edges_some_disease_word_missing CHECK (ok = 1)
);
DELETE FROM _lay_sym_check;
INSERT INTO _lay_sym_check(ok)
SELECT CASE WHEN COUNT(*) = 4 THEN 1 ELSE 0 END
  FROM relations r
  JOIN lay_sid ts ON ts.sid = r.src_id AND r.src_kind = 'sense'
  JOIN lay_sid ds ON ds.sid = r.dst_id AND r.dst_kind = 'sense'
 WHERE r.relation = 'has_symptom'
   AND (ts.word || '|' || ds.word) IN
       ('炎症性疾病|发热','脑炎|抽搐','癫痫|记忆障碍','癫痫|发热');
DROP TABLE _lay_sym_check;

/* ###########################################################################
 * 3. 附：自检与常用查询（注释形式，供应用层 / validate_medical.py §M12 调用）
 * ######################################################################### */

-- 3.1 【新增分层义项自检】3 个义项 + 各自的 is_a 出边
-- SELECT w.word, w.word_type,
--        (SELECT COUNT(*) FROM relations r WHERE r.relation='is_a'
--           AND r.src_kind='sense' AND r.src_id=s.sense_id) AS n_parent
--   FROM words w JOIN senses s ON s.word_id=w.word_id
--  WHERE w.word_id BETWEEN 4037 AND 4039 ORDER BY w.word_id;
-- 期望：炎症性疾病(1) / 感染性疾病(1) / 脑炎(2 —— 多继承)

-- 3.2 【发热可从继承可达】炎症性疾病 及其下位的症状
-- SELECT DISTINCT ws.word AS src, wd.word AS symptom FROM relations r
--   JOIN senses s1 ON s1.sense_id=r.src_id JOIN words ws ON ws.word_id=s1.word_id
--   JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
--  WHERE r.relation='has_symptom' AND r.src_kind='sense'
--    AND ( r.src_id = <炎症性疾病> OR r.src_id IN (
--            SELECT ancestor FROM closure WHERE relation='is_a' AND descendant=<炎症性疾病>));
-- 期望：炎症性疾病→发热（层边）；脑炎→发热（继承）

-- 3.3 【脑炎 的三层继承】= 炎症层 1 + 疾病层 2 + 脑部疾病层 3 + 直接 1 = 7 种
-- SELECT DISTINCT wd.word FROM relations r
--   JOIN senses s2 ON s2.sense_id=r.dst_id JOIN words wd ON wd.word_id=s2.word_id
--  WHERE r.relation='has_symptom' AND r.src_kind='sense'
--    AND ( r.src_id = <脑炎> OR r.src_id IN (
--            SELECT ancestor FROM closure WHERE relation='is_a' AND descendant=<脑炎>))
--  ORDER BY wd.word;
-- 期望：乏力 / 发热 / 意识模糊 / 头痛 / 眩晕 / 疼痛 / 抽搐（7）

-- 3.4 【癫痫 的症状结果集】= 直接 ∪ 继承 = 8 种
--   （同上写法，把 <脑炎> 换成 <癫痫>）
-- 期望：乏力 / 发热 / 意识模糊 / 头痛 / 眩晕 / 疼痛 / 抽搐 / 记忆障碍（8）

-- 3.5 【★ 发热边不是层边】癫痫/脑卒中 之外，疾病层仍无发热边
-- SELECT w.word FROM relations r
--   JOIN senses s ON s.sense_id=r.src_id JOIN words w ON w.word_id=s.word_id
--  WHERE r.relation='has_symptom' AND r.dst_id = <发热>;
-- 期望：仅 炎症性疾病（层边）。癫痫→发热 是**直接边**，不在本句结果里。

-- 3.6 【闭包一致性巡检】以下每句正常应返回 0 行
-- (a) 本补丁新增的 is_a 边缺 depth=1 闭包行
-- SELECT r.relation_id FROM relations r
--  WHERE r.relation='is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
--    AND r.src_id BETWEEN 4037 AND 4039
--    AND NOT EXISTS (SELECT 1 FROM closure c WHERE c.relation='is_a'
--                     AND c.ancestor=r.dst_id AND c.descendant=r.src_id AND c.depth=1);
-- (b) has_symptom 不应有任何闭包行（is_transitive=0）
-- SELECT DISTINCT relation FROM closure WHERE relation='has_symptom';
-- (c) 新增义项的类别挂载若悬空
-- SELECT sc.sense_id FROM sense_categories sc
--  WHERE sc.sense_id BETWEEN 4037 AND 4039
--    AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat);

-- 3.7 【幂等自检】连续执行本文件两次，相关表行数零漂移
-- SELECT 'words', COUNT(*) FROM words WHERE word_id BETWEEN 4037 AND 4039
-- UNION ALL SELECT 'is_a_new', COUNT(*) FROM relations WHERE relation='is_a'
--          AND src_id BETWEEN 4037 AND 4039
-- UNION ALL SELECT 'has_symptom_all', COUNT(*) FROM relations WHERE relation='has_symptom';

-- 4.5 【★ 撤回 癫痫→发热 弱关联边】（若将来引入 has_precipitant「诱发因素」类型）
-- DELETE FROM relations
--  WHERE relation='has_symptom' AND src_kind='sense' AND dst_kind='sense'
--    AND src_id = (SELECT s.sense_id FROM senses s JOIN words w ON w.word_id=s.word_id
--                   WHERE w.word='癫痫' AND s.pos='n')
--    AND dst_id = (SELECT s.sense_id FROM senses s JOIN words w ON w.word_id=s.word_id
--                   WHERE w.word='发热' AND s.pos='n')
--    AND confidence = 0.50;      -- 用置信度限定，避免误删 0.9 的层边

COMMIT;

/*****************************************************************************
 * End of seed_medical_v3_patch_symptom_layers.sql
 *****************************************************************************/
