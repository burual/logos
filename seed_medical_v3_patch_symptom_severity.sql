/*****************************************************************************
 * Logos 医学库增量补丁   seed_medical_v3_patch_symptom_severity.sql
 * 中文原生符号语义数据库 —— 「症状严重度义素标注」（用 sense_features 实现）
 *
 * ---------------------------------------------------------------------------
 * 【文件命名说明 —— 为什么叫 ..._patch_symptom_severity.sql 而不是 ..._patch_severity.sql】
 *   本补丁依赖 seed_medical_v3_patch_symptom.sql 建立的 4031-4036 症状义项。
 *   而 logos/db/connection.py 的 _resolve_seed_paths() 对 seed_*.sql 是
 *     sorted(search_root.glob("seed_*.sql"))
 *   —— 按**文件名字典序**加载。若取名 seed_medical_v3_patch_severity.sql，
 *   则 'severity' 的 'e' 与 'symptom' 的 'y' 在第二个字符处分出大小：
 *     'e'(U+0065) < 'y'(U+0079)  ⇒  severity **排在 symptom 之前**
 *   于是所有走 init_database 自动发现的管线（tests/test_e2e_cat_eats_fish.py、
 *   validate_syntax_parser.py …）都会在症状义项**尚不存在**时执行本文件：
 *   内连接不报错、只是少插 8 行 —— 这正是本项目记录过的「静默丢数据」坑
 *   （见 .workbuddy/memory/MEMORY.md §2）。
 *   故本文件取名 ..._patch_symptom_severity.sql：公共前缀 ..._patch_symptom 之后
 *     '.'(U+002E) < '_'(U+005F)  ⇒  本文件**天然落在 symptom 之后**
 *   （实测证据见 docs/MEDICAL_LIB_DESIGN.md §10「加载顺序实证」）。
 *
 * ---------------------------------------------------------------------------
 * 目标：**不改 schema、不改已有数据，纯追加** —— 用 sense_features 义素系统给
 *       8 个症状义项标注严重度，使「重度 / 中度 / 轻度」在查询时 JOIN 即可得到。
 *
 *   before（patch_symptom 之后）：           after（本补丁）：
 *     症状义项 8 个，无任何严重度信息          疼痛/头痛/抽搐/意识模糊  ← +severe
 *     查询只能答"有哪些症状"                  眩晕/乏力/记忆障碍      ← +moderate
 *                                             发热                    ← +mild
 *     ⇒ 查询可答"癫痫的重度症状有哪些"（closure 继承 + 义素过滤）
 *
 * ---------------------------------------------------------------------------
 * 【为什么用 sense_features 而**不用** relations.confidence】
 *   relations.confidence 的语义是「这条**边**有多可信」（当前 has_symptom 统一 0.9）。
 *   而严重度是**症状义项自身**的属性：疼痛无论出现在哪条疾病边上都是重度。
 *   两者正交，混用会造成两个具体问题：
 *     (a) 同一症状在 N 条疾病边上被重复标注 N 次，改一次要改 N 处；
 *     (b) 无法区分「该症状本身很重」与「该症状对某疾病不典型」——
 *         前者是义素，后者才是边可信度。
 *   故严重度落到义素层（sense_features），与边可信度**语义分离**。
 *
 * 【前置依赖】必须按序执行：
 *     schema_v3.sql → seed_general_v3.sql → seed_general_v3_ext.sql
 *       → seed_medical_v3.sql → seed_medical_v3_patch_symptom.sql → 本文件
 *   理由：8 个症状义项中 6 个（4031-4036）由 patch_symptom 建立，
 *         2 个（抽搐 / 记忆障碍）由 seed_medical_v3.sql 建立。
 *   ★ 依赖缺失时本文件**硬失败**（见 §0），不静默降级。
 *
 * 【幂等性】全部 INSERT 均带 ON CONFLICT DO NOTHING，可重复执行任意次。
 *
 * 【ID / 维度分配】本文件**不新增任何 word / sense**，不占用 4000-4999 段位。
 *   新增的只是"标注行"：
 *     sources            +1   source_id = 6（标注来源登记）
 *     semantic_features  +3   +severe / +moderate / +mild（义素定义）
 *     sense_features     +8   每个症状恰 1 行
 *
 * 【写入规模】（实测，由 validate_medical.py §M11 复核）
 *   semantic_features  7 → 10
 *   sense_features    78 → 86
 *   words / senses / relations / closure / changelog / sense_categories  **零增量**
 *   （本文件不含任何 relations / closure / words / senses 的 INSERT）
 *
 * 【依赖顺序】（严格按此顺序）
 *   sources ─────────────────────────────┐
 *   semantic_features ────────────────────┼─→ sense_features
 *   senses(words) ────────────────────────┘   （FK: sense_id→senses, feature→semantic_features）
 *
 * ---------------------------------------------------------------------------
 * 【三处刻意的设计取舍 —— 请留意】
 *   (1) 【feature_group 复用「极性」，不新增「严重度」分组】
 *       schema_v3.sql 里 semantic_features.feature_group 是 **CHECK 白名单**：
 *         ('生命度','具体性','可数性','自主性','有界性','极性')
 *       严重度（+severe/+moderate/+mild）严格说是一个独立的"程度"维度，但
 *       新增分组必须 ALTER TABLE 改 CHECK —— 与"不改 schema"约束冲突。
 *       故复用含义最接近的**「带 +/- 标记、可枚举」**分组「极性」
 *       （该组已是本库承载 polarity:positive/negative 这类标记值的组）。
 *       ⇒ 已知妥协：待 schema 允许扩大白名单时，应迁移到专用分组「程度」。
 *         validate_medical.py §M11 有一条 [ok] 明示此处。
 *   (2) 【严重度用"三个 boolean 义素"，而不是"一个 enum 义素"】
 *       需求明确要求 feature 取 '+severe' / '+moderate' / '+mild' 且 value=1，
 *       与 schema §14 的 boolean 义素写法一致（boolean 特征用 '1'/'0'）。
 *       代价：DB 层无法阻止一个症状同时标两种严重度（PK 是 sense_id+feature）。
 *       由 §M11「每个症状恰有 1 个严重度义素」断言在构建期兜住。
 *   (3) 【sense_features **没有** source_id 列，source_id=6 无法落库】
 *       需求要求 source_id=6，但该表列仅为 (sense_id, feature, value)
 *       （PK = sense_id + feature，见 schema_v3.sql 第 239-244 行）。
 *       "不改 schema"下不能加列，故：
 *         · 在 sources 表登记 source_id = 6，使该标识**有效、可审计**；
 *         · 把来源写进 semantic_features.description 与注释。
 *       §M11 有一条 [ok] 明示该字段不可落库（而非假装写进去了）。
 *
 * 【连接级 PRAGMA】必须在每个连接上执行，不随 schema 持久化。
 *   注意：PRAGMA foreign_keys 在**显式事务内是 no-op**，故写在 BEGIN 之前。
 * =========================================================================*/

PRAGMA foreign_keys       = ON;   -- 不开启则 sense_features 的两个 FK 形同虚设
PRAGMA recursive_triggers = ON;   -- 与其它种子文件保持一致（本文件不插关系边）

BEGIN;

/* =========================================================================
 * 0. 前置依赖硬检查（不满足即中止，不静默跳过）
 * -------------------------------------------------------------------------
 * 与本项目 seed_medical_v3.sql / _patch_symptom.sql §0 同一手法：SQLite 的
 *   RAISE() 只能出现在触发器体内，脚本里无法直接断言；改为建一张临时表 +
 *   **具名 CHECK 约束**，失败时异常信息会带上**约束名**（实测 SQLite 3.53.1）：
 *     IntegrityError: CHECK constraint failed: _prereq_missing_...
 *   临时表 + IF NOT EXISTS + DELETE 保证本段自身可重复执行。
 *
 * 依赖项：8 个症状义项缺任何一个，§3 的内连接都会**静默少行**（不报错、
 *   只是那 1 行没落上）—— 这正是最该被显式拦截的失败模式，故选择硬失败。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS _prereq_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _prereq_missing_run_seed_medical_v3_and_patch_symptom_first CHECK (ok = 1)
);
DELETE FROM _prereq_check;
INSERT INTO _prereq_check(ok)
SELECT CASE WHEN
        EXISTS (SELECT 1 FROM words WHERE word = '症状')
    AND EXISTS (SELECT 1 FROM words WHERE word = '疼痛')        -- patch_symptom §1.1
    AND EXISTS (SELECT 1 FROM words WHERE word = '发热')
    AND EXISTS (SELECT 1 FROM words WHERE word = '头痛')
    AND EXISTS (SELECT 1 FROM words WHERE word = '眩晕')
    AND EXISTS (SELECT 1 FROM words WHERE word = '乏力')
    AND EXISTS (SELECT 1 FROM words WHERE word = '意识模糊')
    AND EXISTS (SELECT 1 FROM words WHERE word = '抽搐')        -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '记忆障碍')
     THEN 1 ELSE 0 END;
DROP TABLE _prereq_check;

/* =========================================================================
 * 0.1 自然键查找视图（仅本会话有效，不落盘）
 *     与 seed_general 的 seed_sid / ext 的 ext_sid / medical 的 med_sid /
 *     patch_symptom 的 sym_sid 同构，**另起名字 sev_sid**：多个文件可能被同一
 *     连接先后加载，重名会让后者 CREATE ... IF NOT EXISTS 命中旧定义。
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS sev_sid AS
    SELECT w.word AS word, s.pos AS pos, s.sense_no AS sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;

/* ###########################################################################
 * 1. 标注来源登记（sources，source_id = 6）
 *    —— sense_features 无 source_id 列（见文件头取舍 (3)），此处登记使
 *       source_id=6 成为一个**有效且可审计**的标识。
 * ######################################################################### */
INSERT INTO sources(source_id, name, kind, license, note) VALUES
 (6,'医学库症状严重度义素标注','人工标注',NULL,
  'source_id=6：seed_medical_v3_patch_symptom_severity.sql 的严重度义素标注来源')
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 2. 严重度义素定义（semantic_features，+3）
 *    三个 boolean 义素，同一维度（严重度）的互斥取值由应用层 / §M11 保证。
 * ######################################################################### */
INSERT INTO semantic_features(feature, feature_group, value_type, description) VALUES
 ('+severe',  '极性','boolean','症状严重度：重度（剧烈、危及生命或显著失能）（来源 source_id=6）'),
 ('+moderate','极性','boolean','症状严重度：中度（明显不适，通常不危及生命）（来源 source_id=6）'),
 ('+mild',    '极性','boolean','症状严重度：轻度（轻微不适，多可自行缓解）（来源 source_id=6）')
ON CONFLICT(feature) DO NOTHING;

/* ###########################################################################
 * 3. 症状严重度标注（sense_features，+8）
 * -------------------------------------------------------------------------
 *   分级依据（需求给定的语义限定）：
 *     +severe   疼痛(4031) / 头痛(4033) / 抽搐(4021)
 *               —— 另补 意识模糊(4036)：需求的分级清单只列了 7 个症状，
 *                  漏了意识模糊；按其语义限定（"重症 / 脑部疾病共有"，
 *                  见 _patch_symptom.sql §2.2 与 docs/MEDICAL_LIB_DESIGN.md §4.1）
 *                  归入重度。合计 8 个症状各 1 行。
 *     +moderate 眩晕(4034) / 乏力(4035) / 记忆障碍(4022)
 *     +mild     发热(4032)
 *
 *   ★ 写法：CTE(VALUES) 把「词形」映射为 sense_id，**不硬编码任何既有 ID**
 *     （4031-4036 虽已知，仍走自然键，保持补丁对既有 ID 取值不敏感）。
 *     `WHERE 1` 用于消除 INSERT...SELECT...ON CONFLICT 的解析歧义
 *     （本项目记录过的坑，见 MEMORY.md §3 铁律 2）。
 * ######################################################################### */
WITH e(word, feature) AS (VALUES
    ('疼痛',    '+severe'),
    ('头痛',    '+severe'),
    ('抽搐',    '+severe'),
    ('意识模糊','+severe'),
    ('眩晕',    '+moderate'),
    ('乏力',    '+moderate'),
    ('记忆障碍','+moderate'),
    ('发热',    '+mild')
)
INSERT INTO sense_features(sense_id, feature, value)
SELECT s.sid, e.feature, '1'
  FROM e
  JOIN sev_sid s ON s.word = e.word AND s.pos = 'n'
 WHERE 1
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3.1 落库行数硬自检：8 个症状必须恰好落下 8 行严重度义素
 *     若某个症状词在库中缺席（例如依赖未加载就手工跳过了 §0），内连接会
 *     静默少行 —— 这里用同一套「具名 CHECK 临时表」把它变成**响亮的失败**。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS _sev_count_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _sev_expected_8_severity_rows_some_symptom_word_missing CHECK (ok = 1)
);
DELETE FROM _sev_count_check;
INSERT INTO _sev_count_check(ok)
SELECT CASE WHEN COUNT(*) = 8 THEN 1 ELSE 0 END
  FROM sense_features sf
 WHERE sf.feature IN ('+severe', '+moderate', '+mild')
   AND sf.sense_id IN (SELECT sid FROM sev_sid WHERE pos = 'n'
                        AND word IN ('疼痛','发热','头痛','眩晕','乏力','意识模糊',
                                     '抽搐','记忆障碍'));
DROP TABLE _sev_count_check;

/* ###########################################################################
 * 4. 附：自检与常用查询（注释形式，供应用层 / validate_medical.py §M11 调用）
 * ######################################################################### */

-- 4.1 【严重度清单】症状 → 严重度
-- SELECT w.word, sf.feature FROM sense_features sf
--   JOIN senses s ON s.sense_id = sf.sense_id JOIN words w ON w.word_id = s.word_id
--  WHERE sf.feature IN ('+severe','+moderate','+mild')
--  ORDER BY sf.feature, w.word;
-- 期望：+mild 发热 / +moderate 乏力,眩晕,记忆障碍 / +severe 抽搐,头痛,意识模糊,疼痛

-- 4.2 【需求要求的查询形态】closure JOIN relations('is_a')
--     JOIN relations('has_symptom') LEFT JOIN sense_features
--     —— 取疾病 D 的「症状 + 严重度」，LEFT JOIN 保证未标严重度的症状也不丢行
-- WITH d AS (SELECT sid FROM sev_sid WHERE word = ? AND pos = 'n')   -- 例：癫痫
-- SELECT DISTINCT wd.word, sf.feature AS severity
--   FROM closure cl                                        -- ① D 的全部上位
--   JOIN relations r ON r.relation='has_symptom' AND r.src_kind='sense'
--                  AND r.src_id = cl.ancestor              -- ② 上位层声明的症状
--   JOIN senses s2 ON s2.sense_id = r.dst_id
--   JOIN words  wd ON wd.word_id  = s2.word_id
--   LEFT JOIN sense_features sf ON sf.sense_id = r.dst_id  -- ③ 症状自身的严重度义素
--                             AND sf.feature IN ('+severe','+moderate','+mild')
--  WHERE cl.relation='is_a' AND cl.descendant = (SELECT sid FROM d)
--  ORDER BY sf.feature, wd.word;
-- 期望（癫痫）：+moderate 乏力/眩晕；+severe 抽搐/头痛/意识模糊/疼痛
--   （发热 不在结果集 —— 它没有 has_symptom 边，见 _patch_symptom.sql 取舍 (2)）

-- 4.3 【按严重度筛选】癫痫 的重度症状
--   ... 同 4.2，再加 WHERE sf.feature = '+severe'
-- 期望：抽搐 / 头痛 / 意识模糊 / 疼痛

-- 4.4 【全库不变量】严重度义素必须只落在「症状」子树上 —— 正常应返回 0 行
-- SELECT sf.sense_id, w.word FROM sense_features sf
--   JOIN senses s ON s.sense_id = sf.sense_id JOIN words w ON w.word_id = s.word_id
--  WHERE sf.feature IN ('+severe','+moderate','+mild')
--    AND sf.sense_id NOT IN (
--        SELECT ? UNION                                        -- 症状自身
--        SELECT descendant FROM closure
--         WHERE relation='is_a' AND ancestor = ?)              -- 症状的全部下位
--   ;

-- 4.5 【幂等自检】连续执行本文件两次，两张相关表行数零漂移
-- SELECT 'semantic_features', COUNT(*) FROM semantic_features
-- UNION ALL SELECT 'sense_features', COUNT(*) FROM sense_features
-- UNION ALL SELECT 'sources', COUNT(*) FROM sources;

-- 4.6 【后续扩展位】严重度的"程度"专用分组
--     schema 的 feature_group 白名单扩大为含 '严重度' 后，改迁：
-- UPDATE semantic_features SET feature_group = '严重度'
--  WHERE feature IN ('+severe','+moderate','+mild');

COMMIT;

/*****************************************************************************
 * End of seed_medical_v3_patch_symptom_severity.sql
 *****************************************************************************/
