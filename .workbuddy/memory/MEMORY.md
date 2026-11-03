# Logos 长期约定

> 只放「下次开工必须知道」的硬约定；细节在 `docs/`，进展在 `YYYY-MM-DD.md`。

## 1 ID 段（新增数据前先看）

1–27 schema｜1000–1027 general｜2001–2005, 2101–2135, 2201 general_ext（营养 / 骨架 35 /「份」）
3001 动物学 · 3101–3103 医学 · 3104 影像「强化特征」= `entity_nodes`
4001–4030, 4101 医学骨架（4101 兜底「信号」）｜4031–4036 症状｜4037–4039 分层｜4040–4076 影像子树 37
5001–5080 动物 80｜5081–5165 动物扩展 85（关系 +212）｜7000+ 测试夹具（老鼠/石头）
新种子自选空段 + 登记本表 / SQL 头 / 夹具注释。`…_patch_symptom_severity.sql` 不占 ID（纯义素）。

## 1b 加载顺序（十份，靠字典序）

schema → animal_kingdom → animal_kingdom_extended → general → general_ext → medical → medical_imaging → patch_symptom → patch_symptom_layers → patch_symptom_severity

- `.`(2E) < `_`(5F)、`seed_an` < `seed_g` < `seed_m` ⇒ 天然正确；动物库**先于**通用库，就地兜底建 `1026 生物`
- ⚠ `…_patch_severity.sql` 的 `'e'` < `'y'` ⇒ 会排到 `…_patch_symptom.sql` **之前**、静默丢 8 行。起名后实跑 `sorted(glob("seed_*.sql"))`
- `apply_seeds.py` 的 `SQL_FILES` / `EXPECT` 是显式清单，加种子必须同步
- 基线：senses 335 / relations 694 / closure 1794 / words 334 / entity_nodes 5 / relation_types 46 / semantic_features 22 / sense_features 98 / sources 7 / sense_categories 443；自检 39 项（d1 664 / d2 1130）

## 1c closure 方向（I3）

`ancestor` 恒等于**边的 dst**。`is_a` 符合直觉；**`has_part` 相反**（部分在 ancestor 侧）⇒ 查整体 X 的部分用 `WHERE descendant=X`。

## 2 自动发现种子（最高频坑）

`SEED_GLOB="seed_*.sql"` 按文件名排序加载 ⇒ 新种子不用注册就进 e2e / 解析器测试，**会改下游 ID 空间**（只有 7000 段安全）。给补丁加前置依赖硬检查，顺序反了要**响亮报错**。

## 3 种子 SQL 三铁律

① 全部 INSERT 幂等 ② `INSERT…SELECT…ON CONFLICT` **必须带 WHERE**（哪怕 `WHERE 1`）③ `PRAGMA foreign_keys` 在 `BEGIN` 之前，按 `(word,pos)` 自然键子查询。

**两个可复用手法**：① 前置依赖硬检查 = TEMP TABLE + 具名 CHECK（SQLite 3.53 只报约束名）② 自然键 CTE 映射词形→sense_id **必须带 pos**，否则跨词性源词静默丢边。

## 4 校验计数

schema 102 / seed 77 / connection 159 / closure 87 / syntax_parser 214 / medical 230（独立）/ e2e 64+4XFAIL / apply_seeds 39。编号不擅自增删，已知偏差用 XFAIL。

## 5 语义库口径

- 症状继承：`has_symptom` 恒 `is_transitive=0`，继承走 `is_a` 祖先锥（`src_kind='sense'` 别漏）；`癫痫` = 层边 5 + 直接边 3 = 8；`癫痫→发热` 是 0.5 弱关联
- 严重度在 `sense_features`（`+severe/+moderate/+mild`），与 `confidence` 正交；妥协：无「严重度」feature_group、无 `source_id` 列
- 影像：`has_property` 是 `is_transitive=1`；「小 / 大」撞词 ⇒ 用「小型 / 大型」；闸门 `_prereq_missing_run_seed_medical_v3_first_for_imaging`

## 6 训练（`logos_transformer/`）

- `train_closure_relaware.py` 四开关；不带开关与旧版**逐位一致**。`W_shared` 从未收到梯度（每个活跃关系都有 `W_r`）
- 梯度诊断不能取关系词表前 N 个（0 正样本 ⇒ `.grad=None` ⇒ 静默消失），按 `relation_counts` 降序
- `checkpoints_relaware/` 已被覆盖成 1 个 can_do 头；46 头在 `_ext/`
- `train_calibrated.py --cross-pos-ratio 0` 逐项复现基线（含写入 154）；**温度是单调变换**，救不了 OOD

## 6b 写回守卫链

目标词性（默认开）→ 每组 top-3 → 三个 opt-in 符号守卫。写入量 477 / 154 / **118**。

- 源侧词性守卫**结构上不可能触发**（策略 A/B 只对已带该关系的源生成候选），需新增「策略 C」才有意义
- **`MIN(depth)` 不能当特异性**：真边只通过 11/152；正确度量是祖先集大小（152/152）
- is_a 守卫须两条同时成立：`spec[src]>spec[tgt]` ∧ `ancestors(tgt)⊆ancestors(src)`，各拦一类错
- 方向读 `relation_types.inverse_of`，别从 closure 统计「哪对关系双向出现」
- 原子目标守卫 ≥95% 才生效（`part_of` 93.2%、`is_a` 0% 不生效）
- 拦不住：`猫 is_a 人`、`蛇 温血`、`皮肤系统 part_of 猴子`（粒度非方向）
- `fix_pos_annotations.sql` 不叫 `seed_*`；`senses` 有 FTS 触发器，数改动行别用 `total_changes`

## 6c 预训练特征（`train_pretrained_init.py`）

向量 → `node_embedding.weight` 初值。`--init random` 逐位复现基线（0.3609 / 0.6938 / 118 行）。

- **基线正确/错误分离度只有 0.3814**（错条 0.8864 > 对条 0.8641）；mini 冻结 **0.7664**、charngram 0.4151 ⇒ 收益来自语义，非「不共享随机向量」
- 写回错误率仅 **88.1%→81.1%**（对条留 77%、错条留 55%，底数悬殊救不回）
- `猫 is_a 人` 不降反升 0.9915→0.9999；`鱼 is_a 无脊椎动物` 0.8327→0.4849 ⇒ 嵌入修「属性差异」不修「层级互斥」
- 微调步数越多越糟：冻结 0.3687 / 微调 3 步 0.3556 / 10 步 0.3370
- 初值须重标到 `0.1*sqrt(hidden)`（不标则 dense 差 0.027，测的是尺度）
- 坑：`--freeze-gat-epochs` 顺带冻 `node_embedding`（它在 `model.gat` 里）；`charngram` 必须强改 `--oov-mode`，否则全零且**不报错**；dense 要折叠成 table 才能写回
- 指标 `ablation_report.py` / `gold_probe.py`；金标 `residual_gold.json`（13 对 / 96 错 / 9 争议）

## 7 未决

`max_depth` 死列；`mode='off'` 不清旧闭包；`_CLOSED_CLASS_FALLBACK` 90 词桥；e2e K1–K5；影像留白（§13.8）；**假阳靠校准 / 符号守卫 / 预训练特征都解决不了**（`GUARDS_EVALUATION.md` §6、`PRETRAINED_EMBEDDINGS.md` §13）——缺的是「该不该存在」的损失项与互斥公理；`can_do` 应拆两个方向；`load_relation_target_pos` 从 closure 统计会多放行（`typically_has` 的 a 目标 2.0%→8.0%）。

> 改数据 / 代码时同步改 `docs/` 里的实测数字。
