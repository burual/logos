/*****************************************************************************
 * Logos 医学库种子（影像子树）   seed_medical_v3_imaging.sql
 * 中文原生符号语义数据库 —— 「影像特征」概念体系（数值 → 符号映射的目标端）
 *
 * ---------------------------------------------------------------------------
 * 【本文件在影像管线中的位置】—— 它是 Layer 2 的**输出端**，Layer 3 的**输入端**
 *
 *   Layer 1  图像：CT/MRI → CNN(YOLO/UNet) → 结节检测 + 原始数值特征
 *            输出 bbox + {直径, 圆形度, 密度HU, 边缘梯度, …}        ← 数值，不入库
 *   Layer 2  映射：轻量分类器(SVM/随机森林/小MLP) 把数值向量判为符号
 *            {形状:+不规则, 大小:+大型, 边缘:+毛刺, 密度:+高密度}   ← 本文件是它的**目标端**
 *   Layer 3  推理：Logos 符号引擎
 *            ├── 义素写入 sense_features            ← 本文件 §5 建好义素与标注
 *            ├── is_a 闭包推理分类                   ← 本文件 §3.1 建好层级
 *            └── 输出诊断倾向 + 可解释推理链          ← 本文件 §7 给出查询模板
 *
 *   ⇒ 本文件**不含任何数值**。数值阈值只出现在下方注释的「映射契约表」里，
 *     它们是 Layer 2 的参数，不是符号库的内容（符号库只认符号）。
 *
 * ---------------------------------------------------------------------------
 * 【本文件建什么】
 *   影像特征 ⊑ 医学概念
 *   ├── 形态学特征 ─┬─ 形状特征 ─ 圆形 / 椭圆形 / 不规则形 / 分叶状
 *   │               └─ 边缘特征 ─ 光滑 / 毛刺状 / 模糊 / 针刺状
 *   ├── 大小特征 ─ 微小 / 小型 / 中等 / 大型
 *   ├── 密度特征 ─ 低密度 / 等密度 / 高密度 / 钙化
 *   ├── 强化特征 ─ 均匀强化 / 不均匀强化 / 环形强化 / 无强化
 *   └── 位置特征 ─ 肺上叶 / 肺下叶 / 肺门 / 外周 / 中央
 *   病理结构 ⊑ 医学概念 ─ 结节 ─┬─ 恶性结节 ─ typically_has → 不规则形 / 毛刺状 / 高密度
 *                               └─ 良性结节 ─ typically_has → 圆形 / 光滑 / 低密度
 *   结节 has_property 形态学特征 / 密度特征          ← 「结节在这些维度上可被刻画」
 *   跨库：通用库「信号」--domain_specialization--> 医学库节点「强化特征」
 *
 * ---------------------------------------------------------------------------
 * 【前置依赖】必须按序执行：
 *     schema_v3.sql → seed_general_v3.sql → seed_general_v3_ext.sql
 *       → seed_medical_v3.sql → 本文件
 *   理由：「医学概念」（4001）由 seed_medical_v3.sql §1.2 建立（影像特征 ⊑ 医学概念）；
 *         「信号」（4101）也由 seed_medical_v3.sql §1.2 兜底建立（跨库边源端）；
 *         has_property / domain_specialization 两个关系类型由 **schema §14** 登记
 *         （本文件**不重复登记**，只登记新类型 typically_has / typical_of）。
 *   ★ 依赖缺失时本文件**硬失败**（见 §0），不静默降级。
 *
 * 【文件命名与加载顺序 —— 已实证安全，但仍在此显式说明】
 *   logos/db/connection.py 的 _resolve_seed_paths() 用
 *     sorted(search_root.glob("seed_*.sql"))      —— 按**文件名字典序**加载
 *   「seed_medical_v3_imaging.sql」排在：
 *     · seed_medical_v3.sql **之后**（同前缀后 '.' U+002E < '_' U+005F）
 *     · seed_medical_v3_patch_symptom*.sql **之前**（'i' 0x69 < 'p' 0x70）
 *   本文件不依赖任何症状补丁（症状与影像特征无交集），故这个位次正确且安全。
 *   §M13 有一条断言把该位次钉住，防止后手改名破坏顺序。
 *
 * 【幂等性】全部 INSERT 均带 ON CONFLICT DO NOTHING，可重复执行任意次。
 *
 * 【ID 分配策略】沿用医学库 4000-4999 段，**只追加不移动**：
 *     4001-4030  医学义项 30 个（seed_medical_v3.sql）
 *     4031-4036  6 个通用症状（seed_medical_v3_patch_symptom.sql）
 *     4037-4039  疾病分层 3 个（seed_medical_v3_patch_symptom_layers.sql）
 *     ★ 4040-4076 本文件新增 37 个影像义项（word_id = sense_id）
 *     4101       「信号」通用库兜底（seed_medical_v3.sql）
 *     3101-3103  医学库领域实体节点（seed_medical_v3.sql）
 *     ★ 3104      本文件新增领域实体节点「强化特征」
 *
 * ---------------------------------------------------------------------------
 * 【写入规模】（实测增量，由 validate_medical.py §M13 复核）
 *   words            +37   4040-4076（见 §1.1 清单）
 *   senses           +37   与 words 1:1（v0.1 每词单义项）
 *   sense_categories +37   抽象 8 / 性状 20 / 实体 4 / 处所 5（不新增类别行）
 *   relation_types    +2   typically_has / typical_of（一对互逆）
 *   relations        +46   37 is_a + 2 has_property + 6 typically_has
 *                           + 1 跨库 domain_specialization
 *   entity_nodes      +1   强化特征（3104）
 *   closure          +148  分两笔，全部**自动派生**（本文件零 closure INSERT）：
 *                            is_a        +146  ★见下方「为什么是 146 而不是 37」
 *                            has_property +2  ★has_property 是 is_transitive=1
 *   changelog        +46   由 tr_relations_audit_ai 自动记录
 *   semantic_features +12  +圆形/+椭圆形/+不规则/+分叶/+光滑/+毛刺/+模糊/+针刺
 *                          /+微小/+小型/+中等/+大型
 *   sense_features    +12  每个形状/边缘/大小特征义项恰 1 行
 *   sources           +1   source_id = 7（影像特征义素标注来源登记）
 *   合计：senses 133 → 170；relations 120 → 166；closure 223 → 371；
 *         entity_nodes 4 → 5；relation_types 43 → 45
 *
 *   ★ 为什么 is_a 增量是 146 而不是 37：
 *     is_a 的 closure_policy 是 incremental / max_depth=12，插一条边即由
 *     tr_closure_edge_ai 派生**该节点到「抽象概念」的整条上位链**。
 *     逐层累加（每个节点的派生行数 = 它到「抽象概念」的跳数）：
 *       影像特征(2) + 形态学特征(3)
 *         + 形状特征(4) + 4 个形状值 ×5 = 24
 *         + 边缘特征(4) + 4 个边缘值 ×5 = 24
 *         + 大小特征(3) + 4 个大小值 ×4 = 19
 *         + 密度特征(3) + 4 个密度值 ×4 = 19
 *         + 强化特征(3) + 4 个强化值 ×4 = 19
 *         + 位置特征(3) + 5 个位置值 ×4 = 23
 *       = 133
 *       病理结构(2) + 结节(3) + 恶性结节(4) + 良性结节(4) = 13
 *       合计 146
 *
 * 【依赖顺序】（严格按此顺序，不可调换）
 *   relation_types ──→ relations（FK: relation 必须已登记）
 *   words → senses ─┬─→ sense_categories
 *                   ├─→ relations ─→ tr_closure_edge_ai（is_a / has_property 物化）
 *                   ├─→ semantic_features → sense_features
 *                   └─→ entity_nodes → relations(node 端点，跨库边)
 *
 * ---------------------------------------------------------------------------
 * 【六处刻意的设计取舍 —— 请留意】
 *
 *  (1) 【★「小」「大」撞词，改用「小型」「大型」】
 *      需求原文的大小分级是：微小(<5mm) / 小(5-10mm) / 中等(10-30mm) / 大(>30mm)。
 *      实测：**「小」（word_id 1011, pos='a'）与「大」（word_id 1010, pos='a'）
 *      已存在于通用库种子**（seed_general_v3.sql 的 1000-1027 段，是普通形容词）。
 *      而 schema 的 words.word 是 **全局 UNIQUE COLLATE BINARY**（不带 pos），
 *      故在「不改 schema、不修改已有数据」的约束下**无法**再建「小」「大」两个词条。
 *
 *      两条备选路，本文件选了 (b)：
 *        (a) 复用既有 sense 1010/1011，加 is_a 边让 大 ⊑ 大小特征。
 *            ✗ 语义方向是**错的**——那条边断言的是「通用形容词『大』是影像大小特征的
 *              下位」，而不是「影像域把通用词『大』特化为尺寸分级」；
 *            ✗ 且会把两个 id<4000 的通用词拖进「抽象概念」子树，
 *              使 apply_seeds.py 的「分类骨架节点数 = 36（ext 时代）」断言失效
 *              （那两个词当前无任何关系，是孤立形容词节点）。
 *        (b) 另立**不撞词**的词形：「小型」「大型」（本文件采用）。
 *            「中等」本身不撞词，保留原词；只有 小/大 两处替换。
 *      ⇒ 实际词形：微小 / 小型 / 中等 / 大型。
 *        附：若坚持要「小」「大」字面，只能改 schema 或复用语义词 ——
 *            SQL 模板见 §6.5，但那两条路都有上述代价，需先决策。
 *
 *  (2) 【为什么「病理结构」挂 医学概念 而不是 病理状态】
 *      需求只说「需新建『病理结构』节点」，未指定上位。
 *      · 挂「病理状态」⇒ 结节 变成一种"状态"，语义不对（结节是**结构**不是状态）；
 *      · 挂「医学概念」⇒ 病理结构 成为与 解剖结构 / 生理过程 / 病理状态 并列的
 *        第六条顶层分支，语义上最诚实的读法（"病变造成的异常结构"）。
 *      本文件取后者。若贵方希望它归入 病理状态 分支，改 §3.1 那一行的 parent 即可
 *      （闭包增量随之 +4：病理结构 3 而非 2，结节/恶性/良性 各 +1）。
 *
 *  (3) 【has_property 会派生闭包 —— 与医学库其它关系不同】
 *      schema §14 把 has_property 登记为 **is_transitive=1 + closure_policy
 *      (incremental, max_depth=12)**，因此「结节 has_property 形态学特征」这 2 条边
 *      会各自派生 1 行闭包（ancestor=形态学特征, descendant=结节, depth=1）。
 *      这与医学库主文件的 6 条医学关系（一律 is_transitive=0、零闭包）**行为不同**，
 *      不要按"非 is_a 就不出闭包"想当然 —— §M13 断言把实测行数钉住。
 *      「结节 在形态学/密度维度上可被刻画」是最贴合的读法；维度**取值**由
 *      typically_has 给出（见 §3.3），不靠 has_property 表达。
 *
 *  (4) 【新类型 typically_has 一律 is_transitive=0】
 *      沿用本项目既有取舍：不构成可传递层级的关系显式声明 is_transitive=0，
 *      从源头消除「is_transitive=1 却缺 closure_policy 行 → 静默不物化」的坑。
 *      逆向侧 typical_of 按既有约定（symptom_of / treats / depends_on 同款）
 *      标 is_functional=1，表示"一个特征典型归属于唯一类别"——该假设在本数据集内成立
 *      （6 条边恰好两两分组），全域未必成立，故在此写明。
 *
 *  (5) 【义素命名：形状/边缘/大小各 4 个，密度/强化/位置暂不标】
 *      需求只要求"为形状/边缘/大小特征打义素"。本文件严格按此落 12 个义素，
 *      密度/强化/位置的取值**只建义项、不标义素**（§5 留下扩展位，两条 SQL 即可补齐）。
 *      义素名与词形**不同名**（词 不规则形 → 义素 +不规则；词 毛刺状 → 义素 +毛刺），
 *      这是刻意的：义素是 Layer 2 分类器的**输出标签**，应当是紧凑的标记名，
 *      而不是把词形抄一遍。词形↔义素对照见 §5.2 表。
 *
 *  (6) 【feature_group 复用「极性」】
 *      semantic_features.feature_group 是 CHECK 白名单
 *      ('生命度','具体性','可数性','自主性','有界性','极性')，**没有「影像」/「形状」分组**，
 *      新增分组须 ALTER TABLE —— 与"不改 schema"冲突。
 *      故与严重度义素（+severe/+moderate/+mild）同款处理：复用承载 "+/- 标记值" 的「极性」。
 *      迁移 SQL 见 §6.6，已知妥协如实登记。
 *
 *  (7) 【不写 relation_constraints】
 *      架构全景里 Layer 3 的「relation_constraints 校验一致性」不由本文件承担：
 *      该表直接参与解析器的「受事合法性」判定面（见 seed_medical_v3.sql 取舍 (5)），
 *      为影像特征添加槽位约束属独立议题，本次不动。
 *
 * 【连接级 PRAGMA】必须在每个连接上执行，不随 schema 持久化。
 *   注意：PRAGMA foreign_keys 在**显式事务内是 no-op**，故写在 BEGIN 之前。
 * =========================================================================*/

PRAGMA foreign_keys       = ON;   -- SQLite 默认 OFF；不开则 sense_features 的两个 FK 形同虚设
PRAGMA recursive_triggers = ON;   -- 闭包触发器依赖触发器级联语义

BEGIN;

/* =========================================================================
 * 0. 前置依赖硬检查（不满足即中止，不静默跳过）
 * -------------------------------------------------------------------------
 * 与 seed_medical_v3.sql / 三份症状补丁 §0 同一手法：SQLite 的 RAISE() 只能
 *   出现在触发器体内，脚本里无法直接断言；改为建一张临时表 + **具名 CHECK 约束**，
 *   失败时异常信息会带上**约束名**（实测 SQLite 3.53.1 报的是约束名本身）：
 *     IntegrityError: CHECK constraint failed: _prereq_missing_...
 *   临时表 + IF NOT EXISTS + DELETE 保证本段自身可重复执行。
 *
 * 依赖项：本文件把 37 个节点接到「医学概念」/「病理结构」上、把跨库边接到「信号」上，
 *   任一缺失都会导致"内连接静默少行"（不报错、只是边没落上）—— 这正是最该被
 *   显式拦截的失败模式，故与其它种子文件一样选择**硬失败**。
 * ========================================================================= */
CREATE TEMP TABLE IF NOT EXISTS _prereq_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _prereq_missing_run_seed_medical_v3_first_for_imaging CHECK (ok = 1)
);
DELETE FROM _prereq_check;
INSERT INTO _prereq_check(ok)
SELECT CASE WHEN
        EXISTS (SELECT 1 FROM words WHERE word = '医学概念')          -- seed_medical_v3 §1.2
    AND EXISTS (SELECT 1 FROM words WHERE word = '信号')              -- seed_medical_v3 §1.2 兜底
    AND EXISTS (SELECT 1 FROM relation_types WHERE relation = 'has_property')
    AND EXISTS (SELECT 1 FROM relation_types WHERE relation = 'domain_specialization')
    AND EXISTS (SELECT 1 FROM semantic_categories WHERE cat = '抽象')
    AND EXISTS (SELECT 1 FROM semantic_categories WHERE cat = '性状')
    AND EXISTS (SELECT 1 FROM semantic_categories WHERE cat = '实体')
    AND EXISTS (SELECT 1 FROM semantic_categories WHERE cat = '处所')
     THEN 1 ELSE 0 END;
DROP TABLE _prereq_check;

/* =========================================================================
 * 0.1 自然键查找视图（仅本会话有效，不落盘）
 *     与 seed_general 的 seed_sid / ext 的 ext_sid / medical 的 med_sid /
 *     patch_symptom 的 sym_sid / layers 的 lay_sid / severity 的 sev_sid 同构，
 *     本文件**另起名字 img_sid**：多个文件可能被同一连接先后加载，
 *     重名会让后者 CREATE ... IF NOT EXISTS 命中旧定义。
 * ========================================================================= */
CREATE TEMP VIEW IF NOT EXISTS img_sid AS
    SELECT w.word AS word, s.pos AS pos, s.sense_no AS sense_no, s.sense_id AS sid
      FROM senses s JOIN words w ON w.word_id = s.word_id;

/* ###########################################################################
 * 1. 影像特征义项（37 个：4040-4076）
 * ######################################################################### */

/* -------------------------------------------------------------------------
 * 1.1 词层（4040-4076）
 *     ★ 见文件头取舍 (1)：大小分级的「小」「大」撞通用库既有词（1010/1011），
 *       此处改用「小型」「大型」；「中等」不撞词，保留。
 * ------------------------------------------------------------------------*/
INSERT INTO words(word_id, word, separator_pos, word_type, is_bound_root, register) VALUES
 -- ── 顶层与维度（8 个）──────────────────────────────────────────────────
 (4040,'影像特征',   NULL,'合成词',0,'通用'),
 (4041,'形态学特征', NULL,'合成词',0,'通用'),
 (4042,'形状特征',   NULL,'合成词',0,'通用'),
 (4047,'边缘特征',   NULL,'合成词',0,'通用'),
 (4052,'大小特征',   NULL,'合成词',0,'通用'),
 (4057,'密度特征',   NULL,'合成词',0,'通用'),
 (4062,'强化特征',   NULL,'合成词',0,'通用'),
 (4067,'位置特征',   NULL,'合成词',0,'通用'),
 -- ── 形状取值（4 个）────────────────────────────────────────────────────
 (4043,'圆形',       NULL,'合成词',0,'通用'),
 (4044,'椭圆形',     NULL,'合成词',0,'通用'),
 (4045,'不规则形',   NULL,'合成词',0,'通用'),
 (4046,'分叶状',     NULL,'合成词',0,'通用'),
 -- ── 边缘取值（4 个）────────────────────────────────────────────────────
 (4048,'光滑',       NULL,'合成词',0,'通用'),
 (4049,'毛刺状',     NULL,'合成词',0,'通用'),
 (4050,'模糊',       NULL,'合成词',0,'通用'),
 (4051,'针刺状',     NULL,'合成词',0,'通用'),
 -- ── 大小取值（4 个）★ 小/大 撞词，见取舍 (1) ────────────────────────────
 (4053,'微小',       NULL,'合成词',0,'通用'),   -- 需求口径：直径不足五毫米
 (4054,'小型',       NULL,'合成词',0,'通用'),   -- 需求口径「小」，因撞词改名
 (4055,'中等',       NULL,'合成词',0,'通用'),   -- 需求口径：直径居中
 (4056,'大型',       NULL,'合成词',0,'通用'),   -- 需求口径「大」，因撞词改名
 -- ── 密度取值（4 个）────────────────────────────────────────────────────
 (4058,'低密度',     NULL,'合成词',0,'通用'),
 (4059,'等密度',     NULL,'合成词',0,'通用'),
 (4060,'高密度',     NULL,'合成词',0,'通用'),
 (4061,'钙化',       NULL,'合成词',0,'通用'),
 -- ── 强化取值（4 个）────────────────────────────────────────────────────
 (4063,'均匀强化',   NULL,'合成词',0,'通用'),
 (4064,'不均匀强化', NULL,'合成词',0,'通用'),
 (4065,'环形强化',   NULL,'合成词',0,'通用'),
 (4066,'无强化',     NULL,'合成词',0,'通用'),
 -- ── 位置取值（5 个）────────────────────────────────────────────────────
 (4068,'肺上叶',     NULL,'合成词',0,'通用'),
 (4069,'肺下叶',     NULL,'合成词',0,'通用'),
 (4070,'肺门',       NULL,'合成词',0,'通用'),
 (4071,'外周',       NULL,'合成词',0,'通用'),
 (4072,'中央',       NULL,'合成词',0,'通用'),
 -- ── 病理结构分支（4 个）★ 见取舍 (2) ───────────────────────────────────
 (4073,'病理结构',   NULL,'合成词',0,'通用'),
 (4074,'结节',       NULL,'单纯词',0,'通用'),
 (4075,'恶性结节',   NULL,'合成词',0,'通用'),
 (4076,'良性结节',   NULL,'合成词',0,'通用')
ON CONFLICT(word_id) DO NOTHING;

/* -------------------------------------------------------------------------
 * 1.2 义项层 —— 与 1.1 一一对应（sense_id = word_id）
 *     释义一律用中文词语组合（符号语义：用词定义词，不引入外部向量），
 *     且**不含数字 / 拉丁字母**（尺寸带 mm 的口径写在 §1.1 注释与下方
 *     「映射契约表」里，不进释义 —— 数值属 Layer 2 参数，符号库只认符号）。
 *     由 validate_medical.py §M13 断言。
 * ------------------------------------------------------------------------*/
INSERT INTO senses(sense_id, word_id, pos, sense_no, definition, definition_seg,
                   pinyin, pinyin_plain, pinyin_initial, frequency,
                   literalness, subcategory, is_primary) VALUES
 (4040,4040,'n',1,'医学影像上可观察到的、用于描述病灶的属性',
  '医学 影像 上 可 观察 到 的 用于 描述 病灶 的 属性',
  'ying3 xiang4 te4 zheng1','yingxiangtezheng','yxtz',900,'本义',NULL,1),
 (4041,4041,'n',1,'描述病灶形状与边缘的影像属性',
  '描述 病灶 形状 与 边缘 的 影像 属性',
  'xing2 tai4 xue2 te4 zheng1','xingtaixuetezheng','xtxtz',620,'本义',NULL,1),
 (4042,4042,'n',1,'描述病灶整体轮廓形态的影像属性',
  '描述 病灶 整体 轮廓 形态 的 影像 属性',
  'xing2 zhuang4 te4 zheng1','xingzhuangtezheng','xztz',700,'本义',NULL,1),
 (4043,4043,'n',1,'轮廓接近圆形的形状',
  '轮廓 接近 圆形 的 形状',
  'yuan2 xing2','yuanxing','yx',1500,'本义',NULL,1),
 (4044,4044,'n',1,'轮廓接近椭圆形的形状',
  '轮廓 接近 椭圆形 的 形状',
  'tuo3 yuan2 xing2','tuoyuanxing','tyx',700,'本义',NULL,1),
 (4045,4045,'n',1,'轮廓不规则、难以归入规则几何形的形状',
  '轮廓 不 规则 难 以 归入 规则 几何形 的 形状',
  'bu4 gui1 ze2 xing2','buguizexing','bgzx',500,'本义',NULL,1),
 (4046,4046,'n',1,'轮廓表面呈多个弧形分叶的形状',
  '轮廓 表面 呈 多个 弧形 分叶 的 形状',
  'fen1 ye4 zhuang4','fenyezhuang','fyz',420,'本义',NULL,1),
 (4047,4047,'n',1,'描述病灶与周围组织交界处表现的影像属性',
  '描述 病灶 与 周围 组织 交界处 表现 的 影像 属性',
  'bian1 yuan2 te4 zheng1','bianyuantezheng','bytz',640,'本义',NULL,1),
 (4048,4048,'n',1,'边缘平整锐利、无突起与毛刺的表现',
  '边缘 平整 锐利 无 突起 与 毛刺 的 表现',
  'guang1 hua2','guanghua','gh',1600,'本义',NULL,1),
 (4049,4049,'n',1,'边缘向周围伸出细短放射状突起的表现',
  '边缘 向 周围 伸出 细 短 放射状 突起 的 表现',
  'mao2 ci4 zhuang4','maocizhuang','mcz',380,'本义',NULL,1),
 (4050,4050,'n',1,'边缘与周围组织分界不清的表现',
  '边缘 与 周围 组织 分界 不 清 的 表现',
  'mo2 hu5','mohu','mh',1900,'本义',NULL,1),
 (4051,4051,'n',1,'边缘向周围伸出细长针样突起的表现',
  '边缘 向 周围 伸出 细长 针样 突起 的 表现',
  'zhen1 ci4 zhuang4','zhencizhuang','zcz',330,'本义',NULL,1),
 (4052,4052,'n',1,'描述病灶直径范围的影像属性',
  '描述 病灶 直径 范围 的 影像 属性',
  'da4 xiao3 te4 zheng1','daxiaotezheng','dxtz',660,'本义',NULL,1),
 (4053,4053,'n',1,'直径很小的病灶表现',
  '直径 很 小 的 病灶 表现',
  'wei1 xiao3','weixiao','wx',1200,'本义',NULL,1),
 (4054,4054,'n',1,'直径较小的病灶表现',
  '直径 较 小 的 病灶 表现',
  'xiao3 xing2','xiaoxing','xx',1400,'本义',NULL,1),
 (4055,4055,'n',1,'直径居中的病灶表现',
  '直径 居中 的 病灶 表现',
  'zhong1 deng3','zhongdeng','zd',1800,'本义',NULL,1),
 (4056,4056,'n',1,'直径较大的病灶表现',
  '直径 较 大 的 病灶 表现',
  'da4 xing2','daxing','dx',1300,'本义',NULL,1),
 (4057,4057,'n',1,'描述病灶对射线吸收程度高低的影像属性',
  '描述 病灶 对 射线 吸收 程度 高低 的 影像 属性',
  'mi4 du4 te4 zheng1','midutezheng','mdtz',800,'本义',NULL,1),
 (4058,4058,'n',1,'对射线吸收低于周围组织的表现',
  '对 射线 吸收 低于 周围 组织 的 表现',
  'di1 mi4 du4','dimidu','dmd',900,'本义',NULL,1),
 (4059,4059,'n',1,'对射线吸收与周围组织相当的表现',
  '对 射线 吸收 与 周围 组织 相当 的 表现',
  'deng3 mi4 du4','dengmidu','dmd',600,'本义',NULL,1),
 (4060,4060,'n',1,'对射线吸收高于周围组织的表现',
  '对 射线 吸收 高于 周围 组织 的 表现',
  'gao1 mi4 du4','gaomidu','gmd',950,'本义',NULL,1),
 (4061,4061,'n',1,'病灶内出现高密度钙质沉积的表现',
  '病灶 内 出现 高密度 钙质 沉积 的 表现',
  'gai4 hua4','gaihua','gh',700,'本义',NULL,1),
 (4062,4062,'n',1,'描述造影剂进入病灶后信号变化方式的影像属性',
  '描述 造影剂 进入 病灶 后 信号 变化 方式 的 影像 属性',
  'qiang2 hua4 te4 zheng1','qianghuatezheng','qhtz',720,'本义',NULL,1),
 (4063,4063,'n',1,'病灶整体强化程度一致的表现',
  '病灶 整体 强化 程度 一致 的 表现',
  'jun1 yun2 qiang2 hua4','junyunqianghua','jyqh',500,'本义',NULL,1),
 (4064,4064,'n',1,'病灶内部强化程度不一致的表现',
  '病灶 内部 强化 程度 不 一致 的 表现',
  'bu4 jun1 yun2 qiang2 hua4','bujunyunqianghua','bjyqh',480,'本义',NULL,1),
 (4065,4065,'n',1,'病灶周边呈环状强化而中心强化较弱的表现',
  '病灶 周边 呈 环状 强化 而 中心 强化 较 弱 的 表现',
  'huan2 xing2 qiang2 hua4','huanxingqianghua','hxqh',450,'本义',NULL,1),
 (4066,4066,'n',1,'病灶在造影后信号无明显变化的表现',
  '病灶 在 造影 后 信号 无 明显 变化 的 表现',
  'wu2 qiang2 hua4','wuqianghua','wqh',430,'本义',NULL,1),
 (4067,4067,'n',1,'描述病灶所在解剖部位的影像属性',
  '描述 病灶 所在 解剖 部位 的 影像 属性',
  'wei4 zhi4 te4 zheng1','weizhitezheng','wztz',700,'本义',NULL,1),
 (4068,4068,'n',1,'位于肺的上部肺叶的位置',
  '位于 肺 的 上部 肺叶 的 位置',
  'fei4 shang4 ye4','feishangye','fsy',800,'本义',NULL,1),
 (4069,4069,'n',1,'位于肺的下部肺叶的位置',
  '位于 肺 的 下部 肺叶 的 位置',
  'fei4 xia4 ye4','feixiaye','fxy',780,'本义',NULL,1),
 (4070,4070,'n',1,'位于肺门结构附近的位置',
  '位于 肺门 结构 附近 的 位置',
  'fei4 men2','feimen','fm',750,'本义',NULL,1),
 (4071,4071,'n',1,'位于肺野外带、靠近胸膜的位置',
  '位于 肺野 外带 靠近 胸膜 的 位置',
  'wai4 zhou1','waizhou','wz',1100,'本义',NULL,1),
 (4072,4072,'n',1,'位于肺野中心、靠近纵隔的位置',
  '位于 肺野 中心 靠近 纵隔 的 位置',
  'zhong1 yang1','zhongyang','zy',1700,'本义',NULL,1),
 (4073,4073,'n',1,'机体组织因病变而形成的异常结构',
  '机体 组织 因 病变 而 形成 的 异常 结构',
  'bing4 li3 jie2 gou4','binglijiegou','bljg',900,'本义',NULL,1),
 (4074,4074,'n',1,'影像上直径较小、边界相对清楚的类圆形病灶',
  '影像 上 直径 较 小 边界 相对 清楚 的 类圆形 病灶',
  'jie2 jie2','jiejie','jj',1600,'本义',NULL,1),
 (4075,4075,'n',1,'具有恶性倾向、需要进一步处置的结节',
  '具有 恶性 倾向 需要 进一步 处置 的 结节',
  'e4 xing4 jie2 jie2','exingjiejie','exjj',1000,'本义',NULL,1),
 (4076,4076,'n',1,'不具有恶性倾向、以随访观察为主的结节',
  '不 具有 恶性 倾向 以 随访 观察 为主 的 结节',
  'liang2 xing4 jie2 jie2','liangxingjiejie','lxjj',980,'本义',NULL,1)
ON CONFLICT(sense_id) DO NOTHING;

/* -------------------------------------------------------------------------
 * 1.3 义项 ↔ 语义类别
 *     本文件**不新增** semantic_categories 行，只挂到既有粗类：
 *       抽象 —— 8 个「…特征」维度（属性名，尚未落到具体取值）
 *       性状 —— 20 个取值（形状 4 + 边缘 4 + 大小 4 + 密度 4 + 强化 4）
 *       处所 —— 5 个位置取值
 *       实体 —— 病理结构 / 结节 / 恶性结节 / 良性结节
 * ------------------------------------------------------------------------*/
WITH e(word, cat) AS (VALUES
    -- 维度（抽象）
    ('影像特征','抽象'),('形态学特征','抽象'),('形状特征','抽象'),('边缘特征','抽象'),
    ('大小特征','抽象'),('密度特征','抽象'),('强化特征','抽象'),('位置特征','抽象'),
    -- 取值（性状）
    ('圆形','性状'),('椭圆形','性状'),('不规则形','性状'),('分叶状','性状'),
    ('光滑','性状'),('毛刺状','性状'),('模糊','性状'),('针刺状','性状'),
    ('微小','性状'),('小型','性状'),('中等','性状'),('大型','性状'),
    ('低密度','性状'),('等密度','性状'),('高密度','性状'),('钙化','性状'),
    ('均匀强化','性状'),('不均匀强化','性状'),('环形强化','性状'),('无强化','性状'),
    -- 位置取值（处所）
    ('肺上叶','处所'),('肺下叶','处所'),('肺门','处所'),('外周','处所'),('中央','处所'),
    -- 病灶结构（实体）
    ('病理结构','实体'),('结节','实体'),('恶性结节','实体'),('良性结节','实体')
)
INSERT INTO sense_categories(sense_id, cat, is_primary)
SELECT s.sid, e.cat, 1
  FROM e JOIN img_sid s ON s.word = e.word AND s.pos = 'n'
 WHERE 1                                  -- 消除 INSERT...SELECT...ON CONFLICT 解析歧义
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 2. 关系类型（本文件只登记 1 对新的互逆类型）
 * ######################################################################### */

/* =========================================================================
 * 2.1 typically_has / typical_of —— 「典型属性」，Layer 3 的诊断倾向来源
 *     · typically_has：结节类别 → 典型影像特征（一对多：一个类别有多条典型特征）
 *     · typical_of  ：影像特征 → 典型所属结节类别（逆向，由 v_relations_all 在线推导，
 *                     本文件不必冗余存反向行 —— 与 has_symptom/symptom_of 同款）
 *     · 两者 is_transitive=0（见取舍 (4)），故**不需要** closure_policy 行。
 *     · relation_class='属性'：与 has_property / has_symptom 同类。
 *     · 「typically（典型）」与 has_property（具有）的分工：
 *         has_property 断言"必然/定义性"（结节必然可在这些维度上被描述）；
 *         typically_has 断言"统计上典型"（恶性结节**多**呈毛刺状，非必然）。
 *       这条区分正是诊断概率推理的落脚点，不要合并成一个类型。
 * ========================================================================= */
INSERT INTO relation_types
 (relation, relation_class, direction, inverse_of, is_transitive, is_symmetric,
  is_reflexive, is_functional, max_depth, description, color_code) VALUES
 ('typically_has','属性','directed','typical_of', 0,0,0,0, 4,
  '类别→其典型影像特征（统计倾向，非必然）','#B91C1C'),
 ('typical_of',   '属性','directed','typically_has',0,0,0,1, 4,
  '影像特征→典型所属类别','#FCA5A5')
ON CONFLICT(relation) DO NOTHING;

/* ###########################################################################
 * 3. 关系层
 * ######################################################################### */

/* =========================================================================
 * 3.1 is_a 骨架（37 条边：每条新节点各 1 条上位边）
 *     写法与其它种子一致：CTE(VALUES) 把「词形对」映射为 sense_id，
 *     不硬编码既有 ID；父节点不存在时内连接自动跳过（不报错、不产生悬空边）。
 *     插边后由 tr_closure_edge_ai 自动派生整条上位链（实测合计 146 行，见文件头）。
 *     顺序无关紧要：触发器同时向"更具体"与"更一般"两侧递归已有闭包。
 * ========================================================================= */
WITH e(child, parent) AS (VALUES
    -- 顶层
    ('影像特征',   '医学概念'),
    -- 形态学分支
    ('形态学特征', '影像特征'),
    ('形状特征',   '形态学特征'),
    ('圆形',       '形状特征'),
    ('椭圆形',     '形状特征'),
    ('不规则形',   '形状特征'),
    ('分叶状',     '形状特征'),
    ('边缘特征',   '形态学特征'),
    ('光滑',       '边缘特征'),
    ('毛刺状',     '边缘特征'),
    ('模糊',       '边缘特征'),
    ('针刺状',     '边缘特征'),
    -- 大小分支
    ('大小特征',   '影像特征'),
    ('微小',       '大小特征'),
    ('小型',       '大小特征'),
    ('中等',       '大小特征'),
    ('大型',       '大小特征'),
    -- 密度分支
    ('密度特征',   '影像特征'),
    ('低密度',     '密度特征'),
    ('等密度',     '密度特征'),
    ('高密度',     '密度特征'),
    ('钙化',       '密度特征'),
    -- 强化分支
    ('强化特征',   '影像特征'),
    ('均匀强化',   '强化特征'),
    ('不均匀强化', '强化特征'),
    ('环形强化',   '强化特征'),
    ('无强化',     '强化特征'),
    -- 位置分支
    ('位置特征',   '影像特征'),
    ('肺上叶',     '位置特征'),
    ('肺下叶',     '位置特征'),
    ('肺门',       '位置特征'),
    ('外周',       '位置特征'),
    ('中央',       '位置特征'),
    -- 病理结构分支（★ 见取舍 (2)：挂 医学概念，不挂 病理状态）
    ('病理结构',   '医学概念'),
    ('结节',       '病理结构'),
    ('恶性结节',   '结节'),
    ('良性结节',   '结节')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', c.sid, 'sense', p.sid, 'is_a', 5
  FROM e
  JOIN img_sid c ON c.word = e.child  AND c.pos = 'n'
  JOIN img_sid p ON p.word = e.parent AND p.pos = 'n'
 WHERE c.sid <> p.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3.2 has_property 边（2 条）—— 「结节在这些维度上可被刻画」
 *     ⚠ schema §14 登记 has_property 为 is_transitive=1 + closure_policy
 *       (incremental, max_depth=12) ⇒ 这 2 条边**会派生闭包**（各 1 行，共 2 行）。
 *       这与医学库主文件的 6 条医学关系（is_transitive=0、零闭包）**行为不同**，
 *       见文件头取舍 (3)。断言见 §M13。
 *     「维度取值」不由本关系表达（取值是具体的 圆形/毛刺状/… ，
 *      由 typically_has 给出），故 has_property 的 dst 是**维度**节点而非取值节点。
 * ========================================================================= */
WITH e(src, dst) AS (VALUES
    ('结节','形态学特征'),
    ('结节','密度特征')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id)
SELECT 'sense', s.sid, 'sense', d.sid, 'has_property', 5
  FROM e
  JOIN img_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN img_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* =========================================================================
 * 3.3 typically_has 边（6 条）—— 正/负两类结节的典型影像特征
 *       恶性结节 typically_has 不规则形 / 毛刺状 / 高密度
 *       良性结节 typically_has 圆形 / 光滑 / 低密度
 *     confidence=0.8：表达"统计上典型但不必然"。低于 has_property 的语义强度
 *       （后者是定义性断言，无 confidence 可言 —— 注意 §3.2 未写 confidence 列，
 *        取 schema 默认值），也低于症状层边的 0.9。
 *     这 6 条边是 Layer 3「诊断倾向」的直接依据，查询模板见 §7.2 / §7.3。
 * ========================================================================= */
WITH e(src, dst, conf) AS (VALUES
    ('恶性结节','不规则形', 0.8),
    ('恶性结节','毛刺状',   0.8),
    ('恶性结节','高密度',   0.8),
    ('良性结节','圆形',     0.8),
    ('良性结节','光滑',     0.8),
    ('良性结节','低密度',   0.8)
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
SELECT 'sense', s.sid, 'sense', d.sid, 'typically_has', 5, e.conf
  FROM e
  JOIN img_sid s ON s.word = e.src AND s.pos = 'n'
  JOIN img_sid d ON d.word = e.dst AND d.pos = 'n'
 WHERE s.sid <> d.sid
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 4. 跨库链接（通用库「信号」--domain_specialization--> 医学库「强化特征」）
 * ######################################################################### */

/* -------------------------------------------------------------------------
 * 4.1 领域实体节点（entity_nodes，+1）
 *     为什么跨库链接必须先登记节点：schema 的 cross_links 是**视图**
 *       SELECT … FROM relations WHERE src_kind='node' OR dst_kind='node'
 *     即"跨库链接"= relations 里一条**端点带 node** 的边；而 node 端点必须先在
 *     entity_nodes 注册（tr_relations_validate_ai 会校验），故领域库要以"库"的
 *     身份被链接，就得登记节点。
 *     node_id 用 3104：3101-3103 由 seed_medical_v3.sql 占用，ext 占 3001。
 * ------------------------------------------------------------------------*/
INSERT INTO entity_nodes(node_id, node_type, domain, label, entity_class, attrs_json) VALUES
 (3104,'domain_entity','medical','强化特征','ImagingFeature',
  '{"library":"医学库","linked_word":"强化特征","facet":"imaging","subtree":"影像特征"}')
ON CONFLICT(node_id) DO NOTHING;

/* -------------------------------------------------------------------------
 * 4.2 跨库边（1 条）
 *       信号 --domain_specialization--> 强化特征
 *     读法：「一般意义上的信号」在医学影像领域**特化**为「造影剂引起的强化信号」。
 *     复用 seed_medical_v3.sql §1.1 已登记的 domain_specialization 类型
 *     （is_transitive=0、无逆关系），**不新登记类型**。
 *     注：「信号」这个 sense 还挂着另一条同类边 → 脑电图信号（3103）。
 *         同一个源挂两条 domain_specialization 到不同 node 是允许的
 *         （UNIQUE 键含 dst_kind/dst_id），语义上是"同一通用概念的多个领域特化形态"。
 *     注 2：面向 node 的边不参与闭包物化（触发器要求 sense↔sense），
 *           且该类型 is_transitive=0，行为一致、无隐式默认。
 * ------------------------------------------------------------------------*/
WITH e(src_word, src_pos, node_id, rel) AS (VALUES
    ('信号','n',3104,'domain_specialization')
)
INSERT INTO relations(src_kind, src_id, dst_kind, dst_id, relation, source_id, confidence)
SELECT 'sense', s.sid, 'node', e.node_id, e.rel, 5, 0.75
  FROM e
  JOIN img_sid s ON s.word = e.src_word AND s.pos = e.src_pos
  JOIN entity_nodes n ON n.node_id = e.node_id
 WHERE 1
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 5. 义素标注（Layer 2 分类器输出标签 ↔ Layer 3 符号）
 * ######################################################################### */

/* -------------------------------------------------------------------------
 * 5.1 标注来源登记（sources，+1）
 *     sense_features 表**没有** source_id 列（列仅 sense_id/feature/value），
 *     故此处登记使 source_id=7 成为一个有效且可审计的标识（同 severity 补丁手法）。
 * ------------------------------------------------------------------------*/
INSERT INTO sources(source_id, name, kind, license, note) VALUES
 (7,'医学库影像特征义素标注','人工标注',NULL,
  'source_id=7：seed_medical_v3_imaging.sql 的形状/边缘/大小义素标注来源')
ON CONFLICT DO NOTHING;

/* -------------------------------------------------------------------------
 * 5.2 影像义素定义（semantic_features，+12）
 *     词形 ↔ 义素对照（★ 见取舍 (5)：义素是紧凑标记名，不照抄词形）：
 *       形状  圆形→+圆形    椭圆形→+椭圆形  不规则形→+不规则  分叶状→+分叶
 *       边缘  光滑→+光滑    毛刺状→+毛刺    模糊→+模糊        针刺状→+针刺
 *       大小  微小→+微小    小型→+小型      中等→+中等        大型→+大型
 *     feature_group 复用「极性」—— 见取舍 (6)，迁移 SQL 见 §6.6。
 * ------------------------------------------------------------------------*/
INSERT INTO semantic_features(feature, feature_group, value_type, description) VALUES
 ('+圆形',  '极性','boolean','影像形状取值：轮廓接近圆形（来源 source_id=7）'),
 ('+椭圆形','极性','boolean','影像形状取值：轮廓接近椭圆形（来源 source_id=7）'),
 ('+不规则','极性','boolean','影像形状取值：轮廓不规则、难以归入规则几何形（来源 source_id=7）'),
 ('+分叶',  '极性','boolean','影像形状取值：轮廓呈多个弧形分叶（来源 source_id=7）'),
 ('+光滑',  '极性','boolean','影像边缘取值：边缘平整锐利、无突起（来源 source_id=7）'),
 ('+毛刺',  '极性','boolean','影像边缘取值：边缘伸出细短放射状突起（来源 source_id=7）'),
 ('+模糊',  '极性','boolean','影像边缘取值：边缘与周围组织分界不清（来源 source_id=7）'),
 ('+针刺',  '极性','boolean','影像边缘取值：边缘伸出细长针样突起（来源 source_id=7）'),
 ('+微小',  '极性','boolean','影像大小取值：直径很小（来源 source_id=7）'),
 ('+小型',  '极性','boolean','影像大小取值：直径较小（来源 source_id=7）'),
 ('+中等',  '极性','boolean','影像大小取值：直径居中（来源 source_id=7）'),
 ('+大型',  '极性','boolean','影像大小取值：直径较大（来源 source_id=7）')
ON CONFLICT(feature) DO NOTHING;

/* -------------------------------------------------------------------------
 * 5.3 影像义素标注（sense_features，+12）—— 每个形状/边缘/大小取值恰 1 行
 *     value='1'：boolean 义素口径，与 schema §14 及严重度义素一致。
 *     写法：CTE(VALUES) 把「词形 → 义素名」映射为 sense_id，不硬编码既有 ID。
 *     WHERE 1 消除 INSERT...SELECT...ON CONFLICT 的解析歧义（本项目铁律 2）。
 * ------------------------------------------------------------------------*/
WITH e(word, feature) AS (VALUES
    -- 形状
    ('圆形',    '+圆形'),
    ('椭圆形',  '+椭圆形'),
    ('不规则形','+不规则'),
    ('分叶状',  '+分叶'),
    -- 边缘
    ('光滑',    '+光滑'),
    ('毛刺状',  '+毛刺'),
    ('模糊',    '+模糊'),
    ('针刺状',  '+针刺'),
    -- 大小
    ('微小',    '+微小'),
    ('小型',    '+小型'),
    ('中等',    '+中等'),
    ('大型',    '+大型')
)
INSERT INTO sense_features(sense_id, feature, value)
SELECT s.sid, e.feature, '1'
  FROM e
  JOIN img_sid s ON s.word = e.word AND s.pos = 'n'
 WHERE 1
ON CONFLICT DO NOTHING;

/* ###########################################################################
 * 6. 落库行数硬自检（缺词/依存缺失 → 响亮失败，而非静默少行）
 * ######################################################################### */

/* -------------------------------------------------------------------------
 * 6.1 37 个义项 + 37 条 is_a 边
 * ------------------------------------------------------------------------*/
CREATE TEMP TABLE IF NOT EXISTS _img_word_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _img_expected_37_new_imaging_words_4040_4076_missing CHECK (ok = 1)
);
DELETE FROM _img_word_check;
INSERT INTO _img_word_check(ok)
SELECT CASE WHEN COUNT(*) = 37 THEN 1 ELSE 0 END FROM words WHERE word_id BETWEEN 4040 AND 4076;
DROP TABLE _img_word_check;

CREATE TEMP TABLE IF NOT EXISTS _img_isa_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _img_expected_37_new_is_a_edges_some_word_missing CHECK (ok = 1)
);
DELETE FROM _img_isa_check;
INSERT INTO _img_isa_check(ok)
SELECT CASE WHEN COUNT(*) = 37 THEN 1 ELSE 0 END
  FROM relations r
  JOIN img_sid t ON t.sid = r.src_id AND r.src_kind = 'sense'
 WHERE r.relation = 'is_a' AND t.sid BETWEEN 4040 AND 4076;
DROP TABLE _img_isa_check;

/* -------------------------------------------------------------------------
 * 6.2 2 条 has_property + 6 条 typically_has
 * ------------------------------------------------------------------------*/
CREATE TEMP TABLE IF NOT EXISTS _img_prop_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _img_expected_2_has_property_edges_on_结节 CHECK (ok = 1)
);
DELETE FROM _img_prop_check;
INSERT INTO _img_prop_check(ok)
SELECT CASE WHEN COUNT(*) = 2 THEN 1 ELSE 0 END
  FROM relations r
  JOIN img_sid s ON s.sid = r.src_id AND r.src_kind = 'sense'
 WHERE r.relation = 'has_property' AND s.word = '结节';
DROP TABLE _img_prop_check;

CREATE TEMP TABLE IF NOT EXISTS _img_typ_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _img_expected_6_typically_has_edges_some_nodule_class_missing CHECK (ok = 1)
);
DELETE FROM _img_typ_check;
INSERT INTO _img_typ_check(ok)
SELECT CASE WHEN COUNT(*) = 6 THEN 1 ELSE 0 END
  FROM relations r
  JOIN img_sid t ON t.sid = r.src_id AND r.src_kind = 'sense'
  JOIN img_sid d ON d.sid = r.dst_id AND r.dst_kind = 'sense'
 WHERE r.relation = 'typically_has'
   AND (t.word || '|' || d.word) IN
       ('恶性结节|不规则形','恶性结节|毛刺状','恶性结节|高密度',
        '良性结节|圆形','良性结节|光滑','良性结节|低密度');
DROP TABLE _img_typ_check;

/* -------------------------------------------------------------------------
 * 6.3 跨库边 + 12 个义素标注
 * ------------------------------------------------------------------------*/
CREATE TEMP TABLE IF NOT EXISTS _img_cross_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _img_expected_1_cross_link_信号_to_强化特征_node_3104 CHECK (ok = 1)
);
DELETE FROM _img_cross_check;
INSERT INTO _img_cross_check(ok)
SELECT CASE WHEN COUNT(*) = 1 THEN 1 ELSE 0 END
  FROM relations r
  JOIN img_sid s ON s.sid = r.src_id AND r.src_kind = 'sense'
 WHERE r.relation = 'domain_specialization' AND r.dst_kind = 'node' AND r.dst_id = 3104
   AND s.word = '信号';
DROP TABLE _img_cross_check;

CREATE TEMP TABLE IF NOT EXISTS _img_feat_check (
    ok INTEGER NOT NULL,
    CONSTRAINT _img_expected_12_imaging_sense_features_4040_4076 CHECK (ok = 1)
);
DELETE FROM _img_feat_check;
INSERT INTO _img_feat_check(ok)
SELECT CASE WHEN COUNT(*) = 12 THEN 1 ELSE 0 END
  FROM sense_features sf
 WHERE sf.feature IN ('+圆形','+椭圆形','+不规则','+分叶','+光滑','+毛刺','+模糊','+针刺',
                      '+微小','+小型','+中等','+大型')
   AND sf.sense_id BETWEEN 4040 AND 4076;
DROP TABLE _img_feat_check;

/* ###########################################################################
 * 7. 附：数值→符号映射契约 / 自检 / 常用查询（注释形式）
 * ######################################################################### */

/* =========================================================================
 * 7.1 【★ 数值 → 符号映射契约表】—— Layer 2 分类器的目标标签（本表是文档，不是数据）
 *     数值阈值属**算法参数**，不进符号库（符号库只认符号）；此处登记以便两端对齐。
 *
 *   维度        数值特征（Layer 1 输出）              符号标签（本文件义项）
 *   ─────────  ────────────────────────────────────  ──────────────────────────
 *   形状        圆形度 = 4πA/P²                       圆形 / 椭圆形 / 不规则形 / 分叶状
 *               > 0.90           → 圆形
 *               0.75 ~ 0.90      → 椭圆形
 *               < 0.75           → 不规则形
 *               轮廓分叶计数 ≥ 2 → 分叶状（优先于上三者）
 *   边缘        边缘梯度方差 / 放射状突起长度          光滑 / 模糊 / 毛刺状 / 针刺状
 *               梯度方差低       → 光滑
 *               梯度方差中       → 模糊
 *               短密放射突起     → 毛刺状
 *               细长放射突起     → 针刺状
 *   大小        最长径 D（毫米）                       微小 / 小型 / 中等 / 大型
 *               D < 5            → 微小
 *               5 ≤ D < 10       → 小型
 *               10 ≤ D < 30      → 中等
 *               D ≥ 30           → 大型
 *   密度        与周围组织的 HU 差                     低密度 / 等密度 / 高密度 / 钙化
 *   强化        造影前后信号变化（均匀性 / 形态）      均匀强化 / 不均匀强化 / 环形强化 / 无强化
 *   位置        病灶质心落点                           肺上叶 / 肺下叶 / 肺门 / 外周 / 中央
 *
 *   注：本文件只含**形状/边缘/大小**的义素标注（§5.3，需求的 3 个维度）；
 *       密度/强化/位置 3 个维度的取值已建义项但**未标义素**，补齐 SQL 见 §7.6。
 * ========================================================================= */

/* =========================================================================
 * 7.2 【正向查询】某结节类别的典型影像特征（Layer 3 诊断倾向）
 *     = typically_has 直接边 ∪ 沿 is_a 继承（与症状继承同一套口径：
 *       has_property 类关系自身 is_transitive 不足以传播，
 *       「子类继承上位类的属性」由 is_a 闭包完成 —— inference_rules #2）
 * -------------------------------------------------------------------------
 * WITH nodule AS (SELECT sid FROM img_sid WHERE word = ? AND pos = 'n')   -- 例：恶性结节
 * SELECT DISTINCT wn.word AS src, wf.word AS feature, r.relation
 *   FROM relations r
 *   JOIN senses s1 ON s1.sense_id = r.src_id JOIN words wn ON wn.word_id = s1.word_id
 *   JOIN senses s2 ON s2.sense_id = r.dst_id JOIN words wf ON wf.word_id = s2.word_id
 *  WHERE r.relation = 'typically_has' AND r.src_kind = 'sense'
 *    AND ( r.src_id = (SELECT sid FROM nodule)
 *          OR r.src_id IN (SELECT ancestor FROM closure
 *                           WHERE relation = 'is_a' AND descendant = (SELECT sid FROM nodule)) )
 *  ORDER BY wn.word, wf.word;
 * 期望（恶性结节）：不规则形 / 毛刺状 / 高密度（3 条，均为直接边）
 * ========================================================================= */

/* =========================================================================
 * 7.3 【反向查询】给定符号特征向量 → 检索最可能的结节类别（诊断意图）
 *     「形态学:不规则 + 边缘:毛刺 + 密度:高密度」→ 哪个类别 typically_has 全中？
 * -------------------------------------------------------------------------
 * WITH f(word) AS (VALUES ('不规则形'),('毛刺状'),('高密度'))   -- Layer 2 的输出
 * SELECT wn.word AS nodule_class, COUNT(*) AS hit
 *   FROM relations r
 *   JOIN senses s1 ON s1.sense_id = r.src_id JOIN words wn ON wn.word_id = s1.word_id
 *   JOIN senses s2 ON s2.sense_id = r.dst_id JOIN words wf ON wf.word_id = s2.word_id
 *  WHERE r.relation = 'typically_has' AND r.src_kind = 'sense'
 *    AND wf.word IN (SELECT word FROM f)
 *  GROUP BY wn.word ORDER BY hit DESC;
 * 期望：恶性结节 3（全中）—— 用它去驱动「诊断倾向 + 推理链输出」
 * ========================================================================= */

/* =========================================================================
 * 7.4 【义素检索】Layer 2 写义素 → Layer 3 按义素筛选（与症状严重度同款 SQL 形态）
 * -------------------------------------------------------------------------
 * SELECT w.word, sf.feature FROM sense_features sf
 *   JOIN senses s ON s.sense_id = sf.sense_id JOIN words w ON w.word_id = s.word_id
 *  WHERE sf.feature IN ('+圆形','+椭圆形','+不规则','+分叶',
 *                       '+光滑','+毛刺','+模糊','+针刺',
 *                       '+微小','+小型','+中等','+大型')
 *  ORDER BY sf.feature, w.word;
 * 期望 12 行（每个取值恰 1 行）
 * ========================================================================= */

/* =========================================================================
 * 7.5 【闭包一致性巡检】以下每句正常应返回 0 行
 * (a) 本文件新增的 is_a 边缺 depth=1 闭包行
 * SELECT r.relation_id FROM relations r
 *  WHERE r.relation='is_a' AND r.src_kind='sense' AND r.dst_kind='sense'
 *    AND r.src_id BETWEEN 4040 AND 4076
 *    AND NOT EXISTS (SELECT 1 FROM closure c WHERE c.relation='is_a'
 *                     AND c.ancestor=r.dst_id AND c.descendant=r.src_id AND c.depth=1);
 * (b) typically_has 不应有任何闭包行（is_transitive=0）
 * SELECT DISTINCT relation FROM closure WHERE relation='typically_has';
 * (c) 新增义项的类别挂载若悬空
 * SELECT sc.sense_id FROM sense_categories sc
 *  WHERE sc.sense_id BETWEEN 4040 AND 4076
 *    AND NOT EXISTS (SELECT 1 FROM semantic_categories c WHERE c.cat = sc.cat);
 * (d) typically_has 的 6 条边若未落全（缺词导致的静默少行）
 * SELECT COUNT(*) FROM relations WHERE relation='typically_has';   -- 期望 6
 * ========================================================================= */

/* =========================================================================
 * 7.6 【扩展位】密度 / 强化 / 位置的义素标注（需求本次只要 3 个维度）
 *     补齐写法与 §5.2/§5.3 完全同构，两步即可：
 *   -- (1) 定义义素（feature_group 同样复用「极性」）
 *   INSERT INTO semantic_features(feature, feature_group, value_type, description) VALUES
 *    ('+低密度','极性','boolean','影像密度取值：低于周围组织（来源 source_id=7）'),
 *    ('+等密度','极性','boolean','影像密度取值：与周围组织相当（来源 source_id=7）'),
 *    ('+高密度','极性','boolean','影像密度取值：高于周围组织（来源 source_id=7）'),
 *    ('+钙化',  '极性','boolean','影像密度取值：高密度钙质沉积（来源 source_id=7）'),
 *    ('+均匀强化','极性','boolean','影像强化取值：整体强度一致（来源 source_id=7）'),
 *    ('+不均匀强化','极性','boolean','影像强化取值：内部强度不一致（来源 source_id=7）'),
 *    ('+环形强化','极性','boolean','影像强化取值：周边环状强化（来源 source_id=7）'),
 *    ('+无强化','极性','boolean','影像强化取值：造影后无明显变化（来源 source_id=7）')
 *   ON CONFLICT(feature) DO NOTHING;
 *   -- (2) 标注（WITH e(word,feature) AS (VALUES ('低密度','+低密度'), …)
 *   --      → INSERT INTO sense_features …，同 §5.3）
 *     注：位置取值（肺上叶/肺下叶/肺门/外周/中央）本质是**处所**而非程度标记，
 *         是否要走义素体系、还是留在 is_a 层（现状）供结构化检索，需先决策。
 * ========================================================================= */

/* =========================================================================
 * 7.7 【幂等自检】连续执行本文件两次，相关表行数零漂移
 * SELECT 'words',   COUNT(*) FROM words   WHERE word_id BETWEEN 4040 AND 4076
 * UNION ALL SELECT 'senses',  COUNT(*) FROM senses  WHERE sense_id BETWEEN 4040 AND 4076
 * UNION ALL SELECT 'isa',     COUNT(*) FROM relations WHERE relation='is_a' AND src_id BETWEEN 4040 AND 4076
 * UNION ALL SELECT 'has_property', COUNT(*) FROM relations WHERE relation='has_property' AND src_id IN (SELECT sid FROM img_sid WHERE word='结节')
 * UNION ALL SELECT 'typically_has',COUNT(*) FROM relations WHERE relation='typically_has'
 * UNION ALL SELECT 'sense_features',COUNT(*) FROM sense_features WHERE feature LIKE '+%'
 *                             AND sense_id BETWEEN 4040 AND 4076;
 * ========================================================================= */

/* =========================================================================
 * 7.8 【撤回 / 变更备选】（若需求方改主意，逐条一行即可）
 * -- (a) 若坚持用「小」「大」字面（须先解决 words.word 全局 UNIQUE 冲突）：
 * --     把 4054 / 4056 的词形改成 '小' / '大' 需要先删除既有 1011 / 1010，
 * --     或给 words 加 pos 维度（ALTER TABLE）—— 两条路都超出"不改 schema、
 * --     不改已有数据"的约束，故本文件未采用。
 * -- (b) 若要把「病理结构」改挂到「病理状态」下：
 * --     DELETE FROM relations WHERE relation='is_a' AND src_kind='sense'
 * --       AND src_id=(SELECT sense_id FROM senses WHERE word_id=4073)
 * --       AND dst_id=(SELECT sense_id FROM senses WHERE word_id=4001);
 * --     然后按 §3.1 把 parent 改为 '病理状态' 重跑（闭包由触发器自动补齐/失效）。
 * -- (c) 若要把 typically_has 的 is_functional 改为 0（不假设"特征只典型属于一类"）：
 * --     UPDATE relation_types SET is_functional=0 WHERE relation='typical_of';
 * ========================================================================= */

COMMIT;

/*****************************************************************************
 * End of seed_medical_v3_imaging.sql
 *****************************************************************************/
