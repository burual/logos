# schema_v3.sql 建库前检查报告

- 目标运行时：**SQLite 3.53.1**（本机 `sqlite3.sqlite_version`）
- 被检文件：`schema_v3.sql`（1653 行，92330 字节）
- 检查方式：静态解析（保留行号）+ 干净 in-memory DB 逐句执行 + 边界语法实测

## 结论

> **✅ 可以建库** —— FAIL 0 项 / WARNING 2 项 / PASS 12 项

建库实测：150 条语句全部执行成功，耗时 **0.008s**，产出 表 62 / 视图 5 / 触发器 10 / 索引 34。

## 一、逐项结论

| 级别 | 编号 | 检查项 | 结论 | 行号 |
|---|---|---|---|---|
| ⚠️ WARN | P1 | 存在以 relations.src_id 打头的索引（闭包重建触发器按 src_id 关联） | 无。当前唯一能服务该关联的是 UNIQUE(src_kind,src_id,...) 的 skip-scan；关系量过万时可考虑补 `CREATE INDEX idx_relations_src_id ON relations(src_id) WHERE src_kind='sense';` | — |
| ⚠️ WARN | T4 | SQLite 不在 CREATE TRIGGER 阶段校验触发器体内的表名/列名 | 实测：`CREATE TRIGGER ... BEGIN INSERT INTO 不存在的表 ... END` 创建成功，错误直到真正触发 DML 才暴露。因此触发器体内的拼写/表名错误**无法在建库期发现**，必须靠功能回归覆盖（本项目由 validate_schema.py 的 3.6/3.7/3.9 与 R7/R8 覆盖）。 | — |
| ℹ️ INFO | D4 | 延迟解析依赖中的前向引用（SQLite 允许，仅记录） | function_word_roles(第272行) → semantic_roles(第420行); relations(第386行) → contexts(第630行) | — |
| ✅ PASS | S1 | 语句切分完整（无未闭合语句） | 共切出 150 条可执行语句；未闭合 0 条 | — |
| ✅ PASS | D1 | 创建期依赖（INDEX 的表 / TRIGGER 的 ON 表）均存在 | 无缺失 | — |
| ✅ PASS | D2 | 创建期依赖无前向引用（被依赖对象在前文已建立） | 无前向引用 | — |
| ✅ PASS | D3 | 延迟解析依赖（FK 目标 / VIEW 引用表）在文件内均有定义 | 无缺失 | — |
| ✅ PASS | S4 | 全部 64 条外键的被引用列均为父表 PK / UNIQUE | 全部合法 | — |
| ✅ PASS | T1 | 触发器体内无 `INSERT ... WITH RECURSIVE` / 顶层 `WITH ... INSERT` 非法写法 | 全部合规 | — |
| ✅ PASS | T2 | UPSERT 的 DO UPDATE SET 中 first_edge 写在 depth 之前 | 合规 | — |
| ✅ PASS | T3 | 触发器体引用的表名均在 schema 中存在（已剔 CTE 名/别名/关键字） | 无未知引用 | — |
| ✅ PASS | X1 | 逐句执行全部成功（foreign_keys=ON + recursive_triggers=ON） | 150 条语句，耗时 0.008s；失败 0 条 | — |
| ✅ PASS | X2 | PRAGMA foreign_keys 在当前连接上实际生效 | 返回 1 | — |
| ✅ PASS | X3 | foreign_key_check 无违规 | 0 行 | — |
| ✅ PASS | X4 | integrity_check = ok | ok | — |

## 二、FAIL 明细

无。所有阻断项为 0。

## 三、WARNING（不致命，但有隐患）

- ⚠️ **P1 存在以 relations.src_id 打头的索引（闭包重建触发器按 src_id 关联）** — 无。当前唯一能服务该关联的是 UNIQUE(src_kind,src_id,...) 的 skip-scan；关系量过万时可考虑补 `CREATE INDEX idx_relations_src_id ON relations(src_id) WHERE src_kind='sense';`
- ⚠️ **T4 SQLite 不在 CREATE TRIGGER 阶段校验触发器体内的表名/列名** — 实测：`CREATE TRIGGER ... BEGIN INSERT INTO 不存在的表 ... END` 创建成功，错误直到真正触发 DML 才暴露。因此触发器体内的拼写/表名错误**无法在建库期发现**，必须靠功能回归覆盖（本项目由 validate_schema.py 的 3.6/3.7/3.9 与 R7/R8 覆盖）。

- ⚠️ 通用提醒：`PRAGMA foreign_keys` 是**连接级**开关且默认为 OFF。schema 第 38 行虽写了 `PRAGMA foreign_keys = ON`，但它只对**执行该脚本的那个连接**生效；应用每次 `connect()` 后必须自己再设一次，否则全部 FK 静默失效。
- ⚠️ 通用提醒：`PRAGMA recursive_triggers` 默认 OFF。闭包维护依赖「删边 → 写队列 → 重建」的触发器级联，未开启则级联不触发。
- ⚠️ 通用提醒：`PRAGMA foreign_keys` 在**事务内是 no-op**。若应用把建库脚本包在显式 `BEGIN` 里，第 38 行会被静默忽略。

## 四、CREATE 执行顺序依赖表

| # | 行号 | 对象 | 类型 | 依赖 | 依赖建立时机 | 是否前向引用 |
|---|---|---|---|---|---|---|
| 1 | 45 | `schema_meta` | TABLE STRICT | — | — | — |
| 2 | 51 | `sources` | TABLE STRICT | — | — | — |
| 3 | 64 | `domains` | TABLE STRICT | — | — | — |
| 4 | 73 | `semantic_categories` | TABLE STRICT | `semantic_categories` | 解析期(软) | ✅ 否 |
| 5 | 80 | `semantic_features` | TABLE STRICT | — | — | — |
| 6 | 88 | `pos_tags` | TABLE STRICT | — | — | — |
| 7 | 96 | `syntactic_functions` | TABLE STRICT | — | — | — |
| 8 | 103 | `sentence_moods` | TABLE STRICT | — | — | — |
| 9 | 110 | `complement_types` | TABLE STRICT | — | — | — |
| 10 | 122 | `radical_semantics` | TABLE STRICT | — | — | — |
| 11 | 128 | `characters` | TABLE STRICT | `radical_semantics` | 解析期(软) | ✅ 否 |
|  |  | `characters` |  | `semantic_categories` | 解析期(软) | ✅ 否 |
|  |  | `characters` |  | `characters` | 解析期(软) | ✅ 否 |
| 12 | 142 | `character_readings` | TABLE STRICT WITHOUT ROWID | `characters` | 解析期(软) | ✅ 否 |
| 13 | 154 | `phonological_rules` | TABLE STRICT | — | — | — |
| 14 | 171 | `morphemes` | TABLE STRICT | — | — | — |
| 15 | 187 | `words` | TABLE STRICT | `words` | 解析期(软) | ✅ 否 |
| 16 | 206 | `word_morphemes` | TABLE STRICT WITHOUT ROWID | `words` | 解析期(软) | ✅ 否 |
|  |  | `word_morphemes` |  | `morphemes` | 解析期(软) | ✅ 否 |
| 17 | 214 | `senses` | TABLE STRICT | `words` | 解析期(软) | ✅ 否 |
|  |  | `senses` |  | `pos_tags` | 解析期(软) | ✅ 否 |
|  |  | `senses` |  | `senses` | 解析期(软) | ✅ 否 |
| 18 | 239 | `sense_features` | TABLE STRICT WITHOUT ROWID | `senses` | 解析期(软) | ✅ 否 |
|  |  | `sense_features` |  | `semantic_features` | 解析期(软) | ✅ 否 |
| 19 | 247 | `sense_categories` | TABLE STRICT WITHOUT ROWID | `senses` | 解析期(软) | ✅ 否 |
|  |  | `sense_categories` |  | `semantic_categories` | 解析期(软) | ✅ 否 |
| 20 | 256 | `function_words` | TABLE STRICT | `senses` | 解析期(软) | ✅ 否 |
| 21 | 272 | `function_word_roles` | TABLE STRICT WITHOUT ROWID | `function_words` | 解析期(软) | ✅ 否 |
|  |  | `function_word_roles` |  | `semantic_roles` | 解析期(软) | ⚠️ 是 |
| 22 | 282 | `classifiers` | TABLE STRICT | `senses` | 解析期(软) | ✅ 否 |
|  |  | `classifiers` |  | `senses` | 解析期(软) | ✅ 否 |
| 23 | 293 | `noun_classifier_pairs` | TABLE STRICT WITHOUT ROWID | `senses` | 解析期(软) | ✅ 否 |
|  |  | `noun_classifier_pairs` |  | `classifiers` | 解析期(软) | ✅ 否 |
|  |  | `noun_classifier_pairs` |  | `sources` | 解析期(软) | ✅ 否 |
| 24 | 303 | `classifier_class_defaults` | TABLE STRICT WITHOUT ROWID | `semantic_categories` | 解析期(软) | ✅ 否 |
|  |  | `classifier_class_defaults` |  | `classifiers` | 解析期(软) | ✅ 否 |
| 25 | 312 | `reduplication_patterns` | TABLE STRICT | `pos_tags` | 解析期(软) | ✅ 否 |
| 26 | 321 | `word_reduplications` | TABLE STRICT WITHOUT ROWID | `words` | 解析期(软) | ✅ 否 |
|  |  | `word_reduplications` |  | `reduplication_patterns` | 解析期(软) | ✅ 否 |
|  |  | `word_reduplications` |  | `words` | 解析期(软) | ✅ 否 |
| 27 | 331 | `separable_patterns` | TABLE STRICT | `words` | 解析期(软) | ✅ 否 |
| 28 | 343 | `bound_objects` | TABLE STRICT WITHOUT ROWID | `words` | 解析期(软) | ✅ 否 |
|  |  | `bound_objects` |  | `words` | 解析期(软) | ✅ 否 |
| 29 | 356 | `entity_nodes` | TABLE STRICT | `domains` | 解析期(软) | ✅ 否 |
| 30 | 367 | `relation_types` | TABLE STRICT | `relation_types` | 解析期(软) | ✅ 否 |
| 31 | 386 | `relations` | TABLE STRICT | `relation_types` | 解析期(软) | ✅ 否 |
|  |  | `relations` |  | `sources` | 解析期(软) | ✅ 否 |
|  |  | `relations` |  | `contexts` | 解析期(软) | ⚠️ 是 |
| 32 | 404 | `relation_constraints` | TABLE STRICT WITHOUT ROWID | `senses` | 解析期(软) | ✅ 否 |
| 33 | 420 | `semantic_roles` | TABLE STRICT | `semantic_roles` | 解析期(软) | ✅ 否 |
| 34 | 429 | `sentence_patterns` | TABLE STRICT | `sentence_moods` | 解析期(软) | ✅ 否 |
| 35 | 445 | `pattern_slots` | TABLE STRICT WITHOUT ROWID | `sentence_patterns` | 解析期(软) | ✅ 否 |
|  |  | `pattern_slots` |  | `syntactic_functions` | 解析期(软) | ✅ 否 |
|  |  | `pattern_slots` |  | `semantic_roles` | 解析期(软) | ✅ 否 |
| 36 | 457 | `slot_constraints` | TABLE STRICT WITHOUT ROWID | `pattern_slots` | 解析期(软) | ✅ 否 |
| 37 | 471 | `complements` | TABLE STRICT | `complement_types` | 解析期(软) | ✅ 否 |
|  |  | `complements` |  | `senses` | 解析期(软) | ✅ 否 |
|  |  | `complements` |  | `senses` | 解析期(软) | ✅ 否 |
| 38 | 488 | `closure_policy` | TABLE STRICT | `relation_types` | 解析期(软) | ✅ 否 |
| 39 | 516 | `closure` | TABLE STRICT WITHOUT ROWID | `relation_types` | 解析期(软) | ✅ 否 |
| 40 | 527 | `closure_rebuild_queue` | TABLE STRICT | `relation_types` | 解析期(软) | ✅ 否 |
| 41 | 532 | `inference_rules` | TABLE STRICT | — | — | — |
| 42 | 548 | `rule_premises` | TABLE STRICT WITHOUT ROWID | `inference_rules` | 解析期(软) | ✅ 否 |
| 43 | 559 | `rule_conclusions` | TABLE STRICT WITHOUT ROWID | `inference_rules` | 解析期(软) | ✅ 否 |
| 44 | 571 | `ellipsis_rules` | TABLE STRICT | `semantic_roles` | 解析期(软) | ✅ 否 |
| 45 | 590 | `utterances` | TABLE STRICT | `sentence_moods` | 解析期(软) | ✅ 否 |
|  |  | `utterances` |  | `sentence_patterns` | 解析期(软) | ✅ 否 |
| 46 | 602 | `topic_chains` | TABLE STRICT | — | — | — |
| 47 | 612 | `anaphora` | TABLE STRICT | `utterances` | 解析期(软) | ✅ 否 |
|  |  | `anaphora` |  | `semantic_roles` | 解析期(软) | ✅ 否 |
|  |  | `anaphora` |  | `utterances` | 解析期(软) | ✅ 否 |
|  |  | `anaphora` |  | `topic_chains` | 解析期(软) | ✅ 否 |
| 48 | 630 | `contexts` | TABLE STRICT | `domains` | 解析期(软) | ✅ 否 |
| 49 | 637 | `sense_contexts` | TABLE STRICT WITHOUT ROWID | `senses` | 解析期(软) | ✅ 否 |
|  |  | `sense_contexts` |  | `contexts` | 解析期(软) | ✅ 否 |
| 50 | 645 | `import_batches` | TABLE STRICT | `sources` | 解析期(软) | ✅ 否 |
| 51 | 656 | `attributes` | TABLE STRICT | `sources` | 解析期(软) | ✅ 否 |
| 52 | 666 | `changelog` | TABLE STRICT | `import_batches` | 解析期(软) | ✅ 否 |
| 53 | 701 | `tr_senses_fts_ai` | TRIGGER | `senses` | 创建期(硬) | ✅ 否 |
| 54 | 708 | `tr_senses_fts_ad` | TRIGGER | `senses` | 创建期(硬) | ✅ 否 |
| 55 | 715 | `tr_senses_fts_au` | TRIGGER | `senses` | 创建期(硬) | ✅ 否 |
| 56 | 732 | `v_relations_all` | VIEW | `relations` | 解析期(软) | ✅ 否 |
|  |  | `v_relations_all` |  | `relations` | 解析期(软) | ✅ 否 |
|  |  | `v_relations_all` |  | `relation_types` | 解析期(软) | ✅ 否 |
|  |  | `v_relations_all` |  | `relations` | 解析期(软) | ✅ 否 |
|  |  | `v_relations_all` |  | `relation_types` | 解析期(软) | ✅ 否 |
| 57 | 745 | `v_senses_full` | VIEW | `senses` | 解析期(软) | ✅ 否 |
|  |  | `v_senses_full` |  | `words` | 解析期(软) | ✅ 否 |
|  |  | `v_senses_full` |  | `function_words` | 解析期(软) | ✅ 否 |
|  |  | `v_senses_full` |  | `classifiers` | 解析期(软) | ✅ 否 |
| 58 | 758 | `v_closure_2` | VIEW | `relations` | 解析期(软) | ✅ 否 |
|  |  | `v_closure_2` |  | `relations` | 解析期(软) | ✅ 否 |
| 59 | 767 | `entries` | VIEW | `words` | 解析期(软) | ✅ 否 |
| 60 | 771 | `cross_links` | VIEW | `relations` | 解析期(软) | ✅ 否 |
| 61 | 796 | `idx_words_separable` | INDEX | `words` | 创建期(硬) | ✅ 否 |
| 62 | 797 | `idx_words_type` | INDEX | `words` | 创建期(硬) | ✅ 否 |
| 63 | 798 | `idx_senses_pos` | INDEX | `senses` | 创建期(硬) | ✅ 否 |
| 64 | 799 | `idx_senses_subcat` | INDEX | `senses` | 创建期(硬) | ✅ 否 |
| 65 | 800 | `idx_senses_derived` | INDEX | `senses` | 创建期(硬) | ✅ 否 |
| 66 | 801 | `idx_senses_plain` | INDEX | `senses` | 创建期(硬) | ✅ 否 |
| 67 | 802 | `idx_senses_initial` | INDEX | `senses` | 创建期(硬) | ✅ 否 |
| 68 | 803 | `idx_senses_freq` | INDEX | `senses` | 创建期(硬) | ✅ 否 |
| 69 | 804 | `idx_function_words_cat` | INDEX | `function_words` | 创建期(硬) | ✅ 否 |
| 70 | 805 | `idx_fw_roles_role` | INDEX | `function_word_roles` | 创建期(硬) | ✅ 否 |
| 71 | 806 | `idx_classifiers_kind` | INDEX | `classifiers` | 创建期(硬) | ✅ 否 |
| 72 | 807 | `idx_nc_pairs_classifier` | INDEX | `noun_classifier_pairs` | 创建期(硬) | ✅ 否 |
| 73 | 808 | `idx_char_readings_char` | INDEX | `character_readings` | 创建期(硬) | ✅ 否 |
| 74 | 811 | `idx_relations_dst` | INDEX | `relations` | 创建期(硬) | ✅ 否 |
| 75 | 812 | `idx_relations_relation` | INDEX | `relations` | 创建期(硬) | ✅ 否 |
| 76 | 813 | `idx_relations_ctx` | INDEX | `relations` | 创建期(硬) | ✅ 否 |
| 77 | 814 | `idx_relations_source` | INDEX | `relations` | 创建期(硬) | ✅ 否 |
| 78 | 815 | `idx_rel_constraints` | INDEX | `relation_constraints` | 创建期(硬) | ✅ 否 |
| 79 | 816 | `idx_entity_nodes_lookup` | INDEX | `entity_nodes` | 创建期(硬) | ✅ 否 |
| 80 | 818 | `idx_closure_relation_desc` | INDEX | `closure` | 创建期(硬) | ✅ 否 |
| 81 | 819 | `idx_closure_first_edge` | INDEX | `closure` | 创建期(硬) | ✅ 否 |
| 82 | 820 | `idx_closure_depth1` | INDEX | `closure` | 创建期(硬) | ✅ 否 |
| 83 | 822 | `idx_slot_constraints` | INDEX | `slot_constraints` | 创建期(硬) | ✅ 否 |
| 84 | 823 | `idx_complements_head` | INDEX | `complements` | 创建期(硬) | ✅ 否 |
| 85 | 825 | `idx_ellipsis_ctx` | INDEX | `ellipsis_rules` | 创建期(硬) | ✅ 否 |
| 86 | 826 | `idx_rule_premises` | INDEX | `rule_premises` | 创建期(硬) | ✅ 否 |
| 87 | 827 | `idx_rule_conclusions` | INDEX | `rule_conclusions` | 创建期(硬) | ✅ 否 |
| 88 | 829 | `idx_anaphora_utt` | INDEX | `anaphora` | 创建期(硬) | ✅ 否 |
| 89 | 830 | `idx_anaphora_chain` | INDEX | `anaphora` | 创建期(硬) | ✅ 否 |
| 90 | 831 | `idx_topic_chains_doc` | INDEX | `topic_chains` | 创建期(硬) | ✅ 否 |
| 91 | 832 | `idx_sense_contexts_ctx` | INDEX | `sense_contexts` | 创建期(硬) | ✅ 否 |
| 92 | 833 | `idx_changelog_record` | INDEX | `changelog` | 创建期(硬) | ✅ 否 |
| 93 | 834 | `idx_changelog_time` | INDEX | `changelog` | 创建期(硬) | ✅ 否 |
| 94 | 835 | `idx_source_id_lookup` | INDEX | `sources` | 创建期(硬) | ✅ 否 |
| 95 | 843 | `tr_relations_validate_ai` | TRIGGER | `relations` | 创建期(硬) | ✅ 否 |
| 96 | 856 | `tr_relations_freeze_kind_bu` | TRIGGER | `relations` | 创建期(硬) | ✅ 否 |
| 97 | 881 | `tr_closure_edge_ai` | TRIGGER | `relations` | 创建期(硬) | ✅ 否 |
| 98 | 927 | `tr_closure_edge_ad` | TRIGGER | `relations` | 创建期(硬) | ✅ 否 |
| 99 | 942 | `tr_closure_rebuild_ai` | TRIGGER | `closure_rebuild_queue` | 创建期(硬) | ✅ 否 |
| 100 | 971 | `tr_relations_audit_ai` | TRIGGER | `relations` | 创建期(硬) | ✅ 否 |
| 101 | 977 | `tr_relations_audit_ad` | TRIGGER | `relations` | 创建期(硬) | ✅ 否 |

> **依赖语义说明**：
> - **创建期（硬）**：`CREATE INDEX ... ON 表`、`CREATE TRIGGER ... ON 表`、`CREATE VIEW` 中的表名在**建对象时**就会被解析，顺序错了直接报错。
> - **解析期（软）**：外键的 `REFERENCES 目标` 和视图体内的表名是**惰性**解析的，SQLite 允许先建引用方；但一旦真正做 DML / 查询，目标不存在就会报错。本项目全部目标都在文件内定义，故无风险。

## 五、语法特性清单（逐项确认 3.53.1 兼容）

| 特性 | 数量/用法 | 3.53.1 支持 | 最低版本 |
|---|---|---|---|
| STRICT 表 | 52 张 | ✅ | 3.37.0 |
| WITHOUT ROWID | 16 张 | ✅ | 3.8.2 |
| 生成列 VIRTUAL | 2 处 | ✅ | 3.31.0 |
| FTS5 虚拟表 | 2 张 | ✅ | 3.9.0 |
| FTS5 trigram 分词器 | 1 处（`tokenize='trigram'`） | ✅ | 3.34.0 |
| 部分索引(WHERE) | 13 条 | ✅ | 3.8.0 |
| 递归 CTE | 闭包触发器 2 处 | ✅ | 3.8.3 |
| UPSERT(ON CONFLICT DO UPDATE) | 闭包触发器 1 处 | ✅ | 3.24.0 |
| JSON 函数 | json_object, json_valid | ✅ | 3.38.0 / 3.9.0 |
| JSONB | **未使用**（仅注释里提到） | ✅ | 3.45.0 |
| `CREATE VIEW` / `CREATE TRIGGER` | 5 / 10 | ✅ | — |

`PRAGMA` 语句：1 条，均为连接级设置，不随文件持久化。

## 六、边界语法实测（跑一遍才知道的）

| 语法 | 结果 | 备注 |
|---|---|---|
| 触发器体内 UPSERT (INSERT..ON CONFLICT DO UPDATE) | ✅ 支持 | 创建并执行均通过 |
| 触发器体内引用 `excluded` 伪表 | ✅ 支持 | 创建并执行均通过 |
| 触发器体内 `WITH ... INSERT`（预期不支持） | ❌ 不支持 | OperationalError: near "INSERT": syntax error |
| 触发器体内 `INSERT ... SELECT ... FROM (WITH RECURSIVE ...)` | ✅ 支持 | 创建并执行均通过 |
| CREATE TRIGGER 时是否校验触发器体内引用的表 | ⚠️ 不校验（错误延迟到 DML 期） | 带错误表名的触发器创建成功，未报错 |

STRICT 类型强制实测：INT 合法: 接受；TEXT 非法（应拒）: 拒绝(IntegrityError)

## 七、检查方法（可复现）

```bash
python precheck_schema.py     # 产出本报告
python validate_schema.py    # 47 项功能回归
python audit_indexes.py      # 冗余索引探测
```
