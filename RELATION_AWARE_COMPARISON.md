# Relation-Aware Closure Head vs Hard-Negative Baseline

**Date**: 2026-10-22  
**Task**: predict `closure` edges at `depth >= 2` from 1-hop messages only.  
**New script**: `logos_transformer/train_closure_relaware.py`  
**Checkpoints**: `logos_transformer/checkpoints_relaware/`  
**Primary report**: `logos_transformer/checkpoints_relaware/training_report.txt`

This experiment replaces the hard-negative model's single bilinear head

```text
score(s, t) = h_s^T W h_t
```

with a per-relation head

```text
score(s, t, r) = h_s^T W_r h_t
```

and evaluates it on the **same source-disjoint split, the same test pairs and
the same hard pools** as `train_closure_hardneg.py`.

The short version: the head is implemented and works mechanically, but this
dataset has almost no relation diversity in the closure labels. **240 of 241
depth>=2 positives are `is_a`; `has_part` has 1; `has_property` has 0.** The test
set is 28/28 `is_a`. Therefore the run cannot show whether relation-specific
scoring helps, and the observed delta is mostly an initialization / GAT-freeze
trajectory effect, not evidence for relation conditioning.

> **Update (2026-10-26):** the animal-kingdom expansion changed this picture.
> The experiment was re-run on 250 senses / 1090 closure rows / 46 relation
> types. Five heads now receive dedicated training, and the test split covers
> four relations. See **§1b** for the new per-relation results and an explicit
> old-vs-new comparison. The original single-relation analysis below is kept
> unchanged as the historical pre-expansion baseline.

---

## 1. Requested comparison

| Metric | Random Neg (`train_closure.py`) | Source-Disjoint (`train_closure_disjoint.py`) | Hard Neg (`train_closure_hardneg.py`) | Relation-Aware (this run) |
|---|---:|---:|---:|---:|
| Test MRR | 0.7656 | 0.5690 | 0.6404 | **0.6889** |
| Hard-neg MRR | — | 0.7411 | 0.9714 | **0.9688** |
| Margin (hard pool) | — | +0.0407 | +0.2292 | **+0.2106** |
| Per-relation MRR | — | — | — | **`is_a` 0.6889 (n=28)** |

The Random-Neg column used a different, pair-random split and is not directly
comparable. The other three columns use the same 15 test sources and 28 test
positives.

Extended metrics:

| Metric | Random Neg | Source-Disjoint | Hard Neg | Relation-Aware |
|---|---:|---:|---:|---:|
| Test Hits@1 | 0.6389 | 0.3571 | 0.4643 | **0.5000** |
| Test Hits@3 | 0.9167 | 0.6429 | 0.7143 | **0.8571** |
| Test Hits@10 | 0.9722 | 1.0000 | 0.9643 | 0.9643 |
| Test P@10 | — | 0.1867 | 0.1800 | 0.1800 |
| Test AUC (run's neg. mix) | 0.9969 | 1.0000 | 0.9940 | **0.9983** |
| Test hard-pool random MRR | — | 0.1951 | 0.1951 | 0.1951 |
| Hard-pool K (mean) | — | 7.5 | 7.5 | 7.5 |
| Full-catalogue margin | — | +0.0431 | +0.1619 | **+0.2032** |

Notes:

- `hard-neg MRR` ranks each positive against **all** eligible hard negatives
  (mean K=7.5), not against the 3 sampled training negatives.
- `Margin (hard pool)` = hard-pool MRR − hard-pool target-prior MRR.
- `Full-catalogue margin` = full-catalogue MRR − full-catalogue target-prior
  MRR. The relation-aware prior is computed per query relation
  (`S_r.mean(dim=0)`); on the all-`is_a` test set this is still one prior.
- AUC is computed on the sampled hard/medium/easy mix; it is not comparable to
  the random-negative AUC in the original pair-random report.

**Point-estimate verdict:** relation-aware improves full-catalogue MRR
(+0.0486) and Hits@3 (+0.1428), but it does **not** preserve the hard-neg
threshold (0.9688 < 0.97) and it **lowers** the hard-pool margin by 0.0186. It
raises the full-catalogue margin by 0.0413.

---

## 1b. Re-run on expanded dataset (250 senses)

This re-run uses the expanded animal-kingdom graph. The script needed only
report-text changes: the relation vocabulary grew from 45 to 46 types, and the
old hardcoded "only `is_a`" caveats are now derived from the data. No modeling
code, split logic, sampler, or hyperparameter changed.

### 1b.1 Dataset and run

| Item | Value |
|---|---|
| Senses / direct edges / message edges | 250 / 477 / 1204 |
| Closure rows | 1090 (depth=1: 452; depth>=2: **638**) |
| Relation vocabulary | **46** types |
| Split | source-disjoint, seed 42; train 95 / val 20 / test 20 sources |
| Positives | train 457 / val 126 / test 55 |
| Test relations | `is_a` 44, `can_do` 7, `has_property` 3, `typically_has` 1 |
| Epochs | 66 (early stop); best val loss **0.2981** @ epoch 46 |
| Best val hard-pool margin | **+0.3360** @ epoch 46 |
| Relation-head params | 48,174 (46 `W_r` + `W_shared` + gates); model 258,542 |

Effective heads after training:

| Relation | Train positives | Effective head |
|---|---:|---|
| `is_a` | 228 | dedicated `W_r` |
| `can_do` | 124 | dedicated `W_r` |
| `has_property` | 55 | dedicated `W_r` |
| `typically_has` | 44 | dedicated `W_r` |
| `part_of` | 5 | dedicated `W_r` |
| `has_part` | 1 | gated `W_r` + `W_shared` |
| other 40 types (incl. `opposite_of`) | 0 | `W_shared` |

`opposite_of` has 7 closure rows but **0 at depth>=2**, so it is not a closure
prediction target and is correctly absent from the per-relation test table.

### 1b.2 Per-relation test MRR (new vs old)

Old = pre-expansion run (170 senses, 45 relation types, 28/28 `is_a` test).
New = this run. "—" means the relation had no depth>=2 test positive and is
skipped, not scored as zero.

| Relation | Old n | Old full MRR | New n | New full MRR | New hard MRR | New hard margin | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `is_a` | 28 | **0.6889** | 44 | **0.5634** | 0.9470 | +0.3713 | strong, still the best-sampled head |
| `has_property` | 0 | — | 3 | **0.6111** | 1.0000 | +0.3889 | non-random, but only 3 test positives |
| `typically_has` | — | — | 1 | **1.0000** | 1.0000 | +0.7500 | train/val evidence only (test n=1) |
| `can_do` | 0 | — | 7 | **0.3563** | 0.8571 | −0.1429 | hard-pool signal, no full-catalogue margin on test |
| `part_of` | — | — | 0 | — | — | — | 5 train positives, no test coverage |
| `has_part` | 0 | — | 0 | — | — | — | 1 gated positive |
| `opposite_of` | — | — | 0 | — | — | — | 0 depth>=2 rows |

Train/val per-relation full MRR confirms that the heads are real rather than
test-set noise:

| Relation | Train n / MRR | Val n / MRR | Test n / MRR |
|---|---|---|---|
| `is_a` | 228 / 0.5831 | 45 / 0.4970 | 44 / 0.5634 |
| `has_property` | 55 / 0.5163 | 17 / 0.5502 | 3 / 0.6111 |
| `typically_has` | 44 / 0.6386 | 13 / 0.6930 | 1 / 1.0000 |
| `can_do` | 124 / 0.3797 | 47 / 0.3734 | 7 / 0.3563 |
| `part_of` | 5 / 0.2452 | 4 / 0.1521 | — |

`can_do` is the interesting caveat: it is well trained (124 train positives,
hard-pool MRR 1.000 on train and val), but on the 7 test positives its full MRR
equals the target prior exactly (0.3563), and the hard-pool prior is 1.000, so
the test set shows **no independent `can_do` signal**. With n=7 this is not
evidence against the head, but it is not evidence for it either.

### 1b.3 Overall metrics (old vs new)

The test set changed materially (28 -> 55 positives, 1 -> 4 relations, 170 ->
250 candidate nodes, hard-pool K 7.5 -> 11.0), so this table is **descriptive,
not a paired comparison**.

| Metric | Old (pre-expansion) | New (expanded) | Direction |
|---|---:|---:|---|
| Test positives / sources | 28 / 15 | 55 / 20 | harder evaluation |
| Test full-catalogue MRR | 0.6889 | **0.5476** | lower, not comparable |
| Test Hits@1 | 0.5000 | 0.3455 | lower |
| Test Hits@3 | 0.8571 | 0.7273 | lower |
| Test Hits@10 | 0.9643 | 0.9455 | slightly lower |
| Test hard-neg MRR | 0.9688 | **0.9394** | lower, K=11.0 vs 7.5 |
| Target-prior MRR (hard pool) | 0.7581 | 0.6257 | prior is weaker here |
| **Hard-pool margin** | **+0.2106** | **+0.3137** | **increased** |
| Target-prior MRR (full catalogue) | 0.4857 | 0.1946 | prior is much weaker here |
| **Full-catalogue margin** | **+0.2032** | **+0.3530** | **increased** |
| `猫 -> 生物` full-catalogue rank | #12 | **#2** | improved |

The pooled MRR drop is explained by composition, not by a broken head: the new
test set includes low-non-`is_a` relations (`can_do` 0.3563, `has_property`
0.6111, `typically_has` 1.0) and 250 instead of 170 candidates. The
prior-relative signal, which is the metric that isolates the learned scorer
from target frequency, improved on both protocols.

### 1b.4 Expected-outcome check

| Expected outcome | Actual | Verdict |
|---|---|---|
| `has_property` head non-random (>0.3) | full MRR **0.6111** (n=3); hard MRR 1.000, margin +0.3889 | **Pass (small n)** |
| `can_do` head strong (>0.5) | hard MRR **0.8571**, but full MRR 0.3563, hard margin **−0.1429** | **Mixed / fail on full MRR** |
| `typically_has` head >0.3 | train 0.6386 / val 0.6930 / test 1.0000 (n=1) | **Pass on train/val, weak test** |
| `is_a` remains strong | full 0.5634, hard 0.9470, margin +0.3713 (prior 0.5757) | **Pass** |
| Overall test MRR improves from 0.6889 | **0.5476** on a harder, differently composed test set | **Not observed; not comparable** |
| Model-vs-prior margin increases | hard margin +0.2106 -> **+0.3137**; full margin +0.2032 -> **+0.3530** | **Pass** |

### 1b.5 Which `W_r` heads are now meaningfully trained

Head accounting, old vs new:

| | Old (pre-expansion) | New (expanded) |
|---|---:|---:|
| Relation types | 45 | **46** |
| Dedicated `W_r` heads | 1 (`is_a`) | **5** |
| Gated `W_r` + `W_shared` | 1 (`has_part`) | 1 (`has_part`) |
| `W_shared`-only heads | **43** | **40** |
| Relations evaluated on test | 1 (`is_a`) | **4** |

- **`is_a`** — yes. 228 train / 44 test positives; best-sampled and strongest
  head. Hard-pool MRR 0.9470 and margin +0.3713.
- **`has_property`** — yes, with a caveat. 55 train / 75 depth>=2 rows; full
  test MRR 0.6111 and hard margin +0.3889, but only 3 test positives.
- **`typically_has`** — yes on train (44) and val (13); the single test positive
  is not enough to claim generalization.
- **`can_do`** — trained on 124 positives with perfect train/val hard pools, but
  the 7 test positives show no margin over the prior. Needs more test coverage.
- **`part_of`** — only 5 train positives and no test row; weak but non-zero.
- **`has_part`** — 1 positive, gated blend, not meaningfully trained.
- **40 other relations** — no depth>=2 positive; `W_shared` fallback.

So the relation-aware mechanism is no longer a single-head experiment: 5 heads
receive dedicated training and 4 relation types are evaluated on test. The two
best-sampled heads (`is_a`, `has_property`) show clear prior-relative gains.
`can_do` is the open case.

### 1b.6 Baseline limitation

The pre-expansion hard-negative checkpoint is structurally incompatible with
the 250-node graph (`gat.node_embedding` is 170 x 32 vs 250 x 32), so the
script's automatic paired comparison is reported as `n/a`. The old numbers in
this section come from the previous `training_report.txt` and are reproduced
here for orientation; they are **not** a paired baseline. Re-training
`train_closure_hardneg.py` on the expanded graph is the correct next step for a
strict head ablation.

---

## 1c. Weak-relation expansion + training dynamics (335 senses, 1794 closure)

§1b 的结论是「弱关系的 test n 太小，既不能证实也不能证伪」。本轮做两件事：
**(A)** 用 `seed_animal_kingdom_extended.sql` 把弱关系的 depth>=2 闭包补到统计可用；
**(B)** 给 `train_closure_relaware.py` 加 `--relation-subset` / `--grad-mask` /
`--early-stop rel_mrr` / `--grad-report-every`，使单关系可以隔离评估。
§1b.6 遗留的基线缺失也在本轮补上：`train_closure_hardneg.py` 已在扩展图上重训。

### 1c.1 Data expansion (Part A)

新种子文件 `seed_animal_kingdom_extended.sql`，ID 段 **5081–5165**（85 义项），
全部为动物领域，纯追加、零改既有行。设计要点是**造新源而不是堆边**：闭包切分
按 source 做 disjoint，一个源的全部边只落进 train/val/test 之一，所以
`test n ≈ 0.15 × 该关系的不同源个数`。

| relation | 扩容前 depth>=2 | 扩容后 depth>=2 | 目标 | 达标 |
|---|---:|---:|---:|:--:|
| `can_do` | 178 | **529** | ≥50 | ✅ |
| `has_property` | 75 | **129** | ≥50 | ✅ |
| `typically_has` | 58 | **77** | ≥40 | ✅ |
| `part_of` | 9 | **38** | ≥20 | ✅ |
| `has_part` | 1 | **9** | ≥5 | ✅ |

图谱规模（`apply_seeds.py` 实测，39 项自检全绿）：

| 项 | 250 义项（§1b） | 335 义项（本节） |
|---|---:|---:|
| senses / words | 250 / — | **335 / 334** |
| relations（义项级直边） | 477 | **694** |
| closure 行 | 1090（d1 452 / d2 638） | **1794（d1 664 / d2 1130）** |
| 消息边 | 1204 | **1713** |
| relation types | 46 | 46 |

test 逐关系正样本数（source-disjoint, seed 42）从
`can_do 7 / has_property 3 / typically_has 1 / part_of 0 / has_part 0`
变为 `can_do 59 / has_property 16 / typically_has 13 / part_of 2 / has_part 1`。
**`part_of` 仍未达 ≥5 的目标**（38 条 depth>=2 分散在 16 个源上，test 只分到
2 个源），这是切分口径的硬约束，不是数据量不足；再扩也只能靠增加 part_of 的
**独立源**，见 §1c.5。

### 1c.2 Paired baseline (重新训练，不再是 n/a)

`train_closure_hardneg.py --epochs 200` 在扩展图上重训（24 epoch 早停）：
test hard MRR **0.6883**、prior 0.5977、margin **+0.0906**（n=133, K=16.6）。
全部五次运行共用同一 test 划分与同一硬负样本池，因此本节所有对比都是**配对**的。

### 1c.3 Per-relation test metrics after expansion

| relation | test n | full MRR | hard MRR | hard margin | 与 §1b 相比 |
|---|---:|---:|---:|---:|---|
| `is_a` | 50 | **0.5843** | 0.8825 | +0.3630 | n 44→50，margin +0.3713→+0.3630（持平） |
| `has_property` | 16 | **0.3218** | 0.4396 | +0.1979 | n 3→**16**，margin +0.3889→+0.1979（n 变大后回落，真实值） |
| `typically_has` | 13 | **0.2433** | 0.3285 | +0.0017 | n 1→**13**，此前的 1.0000 是小样本幻觉 |
| `can_do` | 59 | **0.2089** | 0.6797 | **+0.0003** | margin −0.1429→**+0.0003**，**翻正** |
| `part_of` | 2 | **0.5167** | 0.6667 | +0.1167 | 首次有 test 覆盖 |
| `has_part` | 1 | 0.0037 | 0.1429 | +0.0000 | 仍只有 1 个 test 正样本 |

整体（test，n=141）：

| 指标 | hardneg 基线 | relaware |
|---|---:|---:|
| full MRR | 0.2793 | **0.3609** |
| Hits@1 / @3 / @10 | 0.1418 / 0.3121 / 0.5887 | **0.2057 / 0.4184 / 0.7021** |
| hard MRR | 0.6883 | **0.6938** |
| hard margin | +0.0906 | **+0.1623** |
| `猫 -> 生物` full rank | — | **#1**（准则要求 ≤ #3 ✅） |

### 1c.4 Isolated `can_do` run (`--relation-subset`)

隔离后只剩 1 个关系头、529 个正样本（train 359 / val 93 / test 77），
关系头参数从 48,174 降到 2,049（1 个 `W_r` + `W_shared` + 门控），
模型总计 215,137。

| 运行 | epochs | test full MRR | test hard MRR | prior | hard margin |
|---|---:|---:|---:|---:|---:|
| §1b 联合训练（250 义项，n=7） | 66 | 0.3563 | 0.8571 | — | −0.1429 |
| C1 联合训练（335 义项，n=59） | 33 | 0.2089 | 0.6797 | 0.6794 | +0.0003 |
| **C2a 隔离 `can_do`（n=65）** | 44 | **0.2445** | **0.8923** | 0.7363 | **+0.1560** |
| C2b 隔离 + `--grad-mask` + `--early-stop rel_mrr` | 41 | 0.2446 | **0.9000** | 0.7359 | **+0.1641** |

隔离带来的收益是压倒性的：同一个关系，hard margin 从 **+0.0003 → +0.1560**，
hard MRR 从 0.6797 → 0.8923。这说明 §1b 里 `can_do` 的低分主要来自**与其他关系
共享编码器/优化目标时的相互干扰**，而不是 `can_do` 本身不可学。

**两个必须说明的口径问题：**

1. `can_do` 的 **full-catalogue** MRR 只有 0.2445，**没有达到「> 0.4」的成功准则**。
   该指标在 335 个候选节点上排序，其 prior（源无关频次基线）本身就是 0.2316，
   而 §1b 的 0.3563 是在 250 个候选、n=7 上测的，两者不可比。**达到 > 0.4 的是
   hard-pool MRR（0.8923 / 0.9000）**。
2. 隔离跑的 test 硬负池里，hardneg 基线（共享头）拿到 **1.0000** 的 hard MRR ——
   即该池对频次先验已经饱和（prior 也是 1.0000），没有给任何模型留出空间。
   因此这里可读的信号是 **prior-relative margin（+0.1560）与 full MRR（0.2445 vs
   0.2418）**，不是绝对 hard MRR 的差值（−0.1077）。

### 1c.5 Training dynamics (Part B)

| 开关 | 作用 | 默认 |
|---|---|---|
| `--relation-subset REL...` | 正样本、切分、负样本池、`W_r` 头、全部指标收窄到子集；未知关系名直接报错并列出可用关系 | 关闭（全关系） |
| `--grad-mask` | `loss.backward()` 之后、`optimizer.step()` 之前，清零「本批没有正样本」的关系头梯度；`W_shared` 仅在整批无共享路径关系时清零 | 关闭 |
| `--early-stop {val_loss,rel_mrr}` + `--primary-relation` | 早停依据二选一：整体 val loss（旧行为）或主关系 val 硬池 MRR；后者单独保存 `closure_gat_relaware_best_<rel>.pt` 并优先用它做最终模型 | `val_loss` |
| `--grad-report-every N` | 每 N 个 epoch 打印各关系头与共享头的平均 `|grad|`，据此判断共享头是否压过专属头 | 10 |

**梯度掩码的实际作用（实测，不是推测）：**

- **单关系子集下它是严格的 no-op**：`--relation-subset can_do --grad-mask`
  与不带 `--grad-mask` 的两次运行逐位一致（epochs 44、best val loss 0.2015 @24、
  test hard MRR 0.8923、margin +0.1560 完全相同），诊断行显示
  `head-grads zeroed=0`。原因很直接：子集内每个批次的负样本与正样本同关系，
  批次里总存在该类正样本。
- **联合训练下它改变结果**：D1（全量 + `--grad-mask`）vs C1（全量）test hard MRR
  0.7061 vs 0.6938，margin **+0.1911 vs +0.1623**，其中收益集中在弱关系：
  `has_property` hard margin +0.1979 → **+0.3492**、`typically_has`
  +0.0017 → **+0.1730**；代价是 `part_of`（n=2）+0.1167 → +0.0278、
  `typically_has` full MRR 0.2433 → 0.1595。样本只有 1–2 个的关系上，
  单次运行的涨落不应被读成结论。

**梯度幅度诊断（C1 全量运行，`--grad-report-every 10`）：**

| head | epoch 10（GAT 冻结中） | epoch 20（解冻后） | epoch 30 |
|---|---:|---:|---:|
| `W_r[is_a]` | 9.69e-05 | **1.42e-03** | 1.19e-03 |
| `W_r[can_do]` | 5.98e-05 | 1.40e-03 | **1.52e-03** |
| `W_r[has_property]` | 1.60e-05 | 7.47e-04 | 1.00e-03 |
| `W_r[typically_has]` | 1.27e-05 | 7.89e-04 | 9.37e-04 |
| `W_r[part_of]` | 2.03e-05 | 2.56e-04 | 2.31e-04 |
| `W_r[has_part]` | 4.41e-06 | 3.71e-04 | 4.05e-04 |
| `W_shared` | **0（无梯度）** | **0（无梯度）** | **0（无梯度）** |

两个与原始假设相悖的发现：

1. **`W_shared` 在整个训练中从未收到梯度。** 6 个有正样本的关系全部持有专属
   `W_r`，没有任何批次走共享路径。所以「40 个未训练关系通过共享头注入噪声梯度、
   拖累 `can_do`」这一假说**在本图上不成立** —— 噪声不来自 `W_shared`，
   而来自**共享 GAT 编码器 + 联合优化**，这与 §1c.4 隔离后大涨的结果一致。
2. **弱关系的梯度并没有被 `is_a` 淹没。** 解冻后量级同为 1e-3，到 epoch 30
   `can_do` 的梯度已经超过 `is_a`。梯度的相对大小不是瓶颈，**样本量与负样本池
   构造才是**。

### 1c.6 Success-criteria check

| 准则 | 要求 | 实测 | 判定 |
|---|---|---|:--:|
| `can_do` test MRR | > 0.4（原 0.3563） | hard-pool **0.8923**（隔离）/ 0.6797（联合）；full **0.2445** | ⚠️ 取决于口径，见 §1c.4 |
| `can_do` hard-pool margin | > 0（原 −0.1429） | **+0.0003**（联合）/ **+0.1560**（隔离） | ✅ |
| `has_property` test n | ≥ 10（原 3） | **16** | ✅ |
| `part_of` test n | ≥ 5（原 0） | **2** | ❌ 见 §1c.1 |
| `猫 -> 生物` rank | ≤ #3 | **#1** | ✅ |

### 1c.7 Reproduction

```bash
# 1) 重建数据库（39 项自检；--keep 幂等）
python apply_seeds.py logos.db
python apply_seeds.py logos.db --keep

# 2) 重新训练配对基线（扩展图，24 epoch 早停）
python logos_transformer/train_closure_hardneg.py --epochs 200 \
  --output-dir ./logos_transformer/checkpoints_hardneg_ext

# 3) C1 全量（33 epoch 早停）
python logos_transformer/train_closure_relaware.py --epochs 200 \
  --output-dir ./logos_transformer/checkpoints_relaware_ext \
  --compare-checkpoint ./logos_transformer/checkpoints_hardneg_ext/closure_gat_hardneg_best.pt

# 4) D1 全量 + 梯度掩码
python logos_transformer/train_closure_relaware.py --grad-mask --epochs 200 \
  --output-dir ./logos_transformer/checkpoints_relaware_ext_mask \
  --compare-checkpoint ./logos_transformer/checkpoints_hardneg_ext/closure_gat_hardneg_best.pt

# 5) C2a 隔离 can_do / C2b 隔离 + 掩码 + 按关系早停
python logos_transformer/train_closure_relaware.py --relation-subset can_do --epochs 200 \
  --output-dir ./logos_transformer/checkpoints_relaware_cando
python logos_transformer/train_closure_relaware.py --relation-subset can_do \
  --grad-mask --early-stop rel_mrr --epochs 200 \
  --output-dir ./logos_transformer/checkpoints_relaware_cando_mask
```

`checkpoints*/` 已加入 `.gitignore`：报告由脚本重新生成，不入库。

---

## 2. Setup

### 2.1 Split and data

Identical to the hard-negative run (seed 42):

| Split | Sources | Positives | Depth distribution |
|---|---:|---:|---|
| train | 71 | 180 | `{2: 73, 3: 57, 4: 38, 5: 11, 6: 1}` |
| val | 15 | 33 | `{2: 15, 3: 11, 4: 7}` |
| test | 15 | 28 | `{2: 15, 3: 8, 4: 3, 5: 2}` |

Closure relation distribution at `depth >= 2`:

| Relation | Depth>=2 positives | Split |
|---|---:|---|
| `is_a` | 240 | train/val/test |
| `has_part` | 1 | train only (`脑 -> 大脑皮层`) |
| `has_property` | **0** | — |

The graph vocabulary has **45 relation types**, so the implementation creates
all 45 `W_r` matrices, but only `is_a` receives enough positives to be a
dedicated head.

### 2.2 Relation-aware head

Implementation in `RelationAwareClosurePredictor`:

| Component | Size / rule |
|---|---|
| `W_r` | 45 × 32×32 = **46,080** params |
| `W_shared` | 32×32 = **1,024** params |
| gate `g_r` | 45 scalars = **45** params |
| total relation-head params | **47,149** |
| total model params | **250,605** (GAT 203,456 + heads 47,149) |

Effective matrix per relation:

```text
count >= 5        -> W_r
0 < count < 5     -> sigmoid(g_r) * W_r + (1 - sigmoid(g_r)) * W_shared
count == 0        -> W_shared
```

Actual configuration for this run:

| Relation | Train positives | Effective head |
|---|---:|---|
| `is_a` | 179 | dedicated `W_r` |
| `has_part` | 1 | gated `W_r` + `W_shared` |
| other 43 relations | 0 | `W_shared` |

The counts are stored as buffers (`relation_count`, `relation_seen`), so a
checkpoint retains the same dedicated/gated/unseen decision.

### 2.3 Training

Same protocol as `train_closure_hardneg.py`:

- Adam, lr 1e-3, `ReduceLROnPlateau(factor=0.5, patience=10)`;
- BCEWithLogitsLoss, `pos_weight = 1080 / 180 = 6.00`;
- batch 64, max 200 epochs, early-stopping patience 20, seed 42;
- 6 negatives per positive: 3 hard / 2 medium / 1 easy;
- checkpoint selection by **best val loss**;
- a second best-val-margin checkpoint is saved as a diagnostic only.

New in this run:

- GAT frozen for the first **10** epochs (`--freeze-gat-epochs 10`), then
  unfrozen at epoch 11. During the freeze only the relation heads train, and GAT
  dropout is disabled so the fixed embeddings are deterministic.
- Per-relation full-catalogue MRR / Hits.
- Per-relation hard-pool MRR / margin.
- Layer-1 attention grouped by relation, top-3 edges per relation.

### 2.4 Training run

| Item | Value |
|---|---|
| Epochs run | 50 (early stop) |
| Best val loss | **0.1086** at epoch 30 (primary checkpoint) |
| Best val hard-pool margin | **+0.1862** at epoch 20 (diagnostic checkpoint) |
| GAT freeze | epochs 1–10, unfrozen epoch 11 |
| Train loss / acc at end | ~0.028 / 0.997 |

The val margin is positive from epoch 1 (+0.0145) and stays positive
throughout; it ends at +0.1327. This is a smoother optimization than the
hard-negative baseline, which became positive around epoch 4–5.

---

## 3. Full-catalogue results

### 3.1 Overall

| Split | Sources | Pos | Neg | AUC | MRR | H@1 | H@3 | H@10 | P@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 71 | 180 | 1080 | 0.9999 | 0.6258 | 0.3944 | 0.8056 | 0.9944 | 0.2486 |
| val | 15 | 33 | 198 | 0.9982 | 0.5972 | 0.3636 | 0.7273 | 0.9394 | 0.2067 |
| test | 15 | 28 | 168 | 0.9983 | **0.6889** | **0.5000** | **0.8571** | 0.9643 | 0.1800 |

Target-prior MRR: train 0.4839 / val 0.4836 / test **0.4857**.  
Full-catalogue margin: train +0.1419 / val +0.1136 / test **+0.2032**.

The relation-aware model has a higher full-catalogue MRR than the hard-negative
baseline (0.6889 vs 0.6404), but the paired source-cluster bootstrap is not
decisive: Δ = +0.0484, 95% CI [−0.0002, +0.1408]. The full-catalogue margin
gain is also not decisive: Δ = +0.0412, 95% CI [−0.0039, +0.1319].

### 3.2 Per-relation breakdown

Because the test set is all `is_a`, the per-relation test row is identical to
the pooled row by construction.

| Split | Relation | n | MRR | H@1 | H@10 | Prior | Margin |
|---|---|---:|---:|---:|---:|---:|---:|
| train | `has_part` | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| train | `is_a` | 179 | 0.6237 | 0.3911 | 0.9944 | 0.4810 | +0.1427 |
| val | `is_a` | 33 | 0.5972 | 0.3636 | 0.9394 | 0.4836 | +0.1136 |
| test | `is_a` | 28 | **0.6889** | **0.5000** | 0.9643 | 0.4857 | **+0.2032** |
| test | `has_part` | 0 | — | — | — | — | — |
| any | `has_property` | 0 | — | — | — | — | — |

`has_part` is train-only and has one positive. Its full-catalogue rank is #1,
but that is a single memorized example with no validation/test counterpart. Its
hard-pool metric is skipped because there is no other `has_part` source to form
a hard pool.

### 3.3 Full-catalogue metrics by depth (test)

| Depth | n | MRR | H@1 | H@10 |
|---:|---:|---:|---:|---:|
| 2 | 15 | 0.6082 | 0.4000 | 0.9333 |
| 3 | 8 | 0.7708 | 0.6250 | 1.0000 |
| 4 | 3 | 0.6667 | 0.3333 | 1.0000 |
| 5 | 2 | 1.0000 | 1.0000 | 1.0000 |

Depth 4/5 have n=3/2 and should not be interpreted. The depth-2 and depth-3
numbers are the informative ones.

---

## 4. Hard-negative protocol

### 4.1 Overall

| Split | Positives | hardMRR | Prior | Margin | K mean | n | skipped | Random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 180 | 0.9836 | 0.7496 | +0.2340 | 6.4 | 158 | 22 | 0.2183 |
| val | 33 | 0.9531 | 0.8024 | +0.1507 | 6.7 | 32 | 1 | 0.2091 |
| test | 28 | **0.9688** | 0.7581 | **+0.2106** | 7.5 | 28 | 0 | 0.1951 |

The test hardMRR is 0.9688, narrowly below the requested 0.97 threshold. The
hard-negative baseline was 0.9714. The paired difference is −0.0028 with 95% CI
[−0.0098, +0.0000], i.e. a small but consistently non-positive change, not a
clear win.

The hard-pool margin is 0.2106 vs 0.2292 for hard negatives. The paired
difference is **−0.0191, 95% CI [−0.0400, −0.0041]**, so the regression is
statistically supported on this test set.

### 4.2 Nontrivial subsets

K=1 queries are prior-trivial (model and prior both rank the only candidate
first), so they inflate the headline. Separating them:

| Model | Subset | n | hardMRR | Prior | Margin | K |
|---|---|---:|---:|---:|---:|---:|
| Relation-aware | all | 28 | 0.9688 | 0.7581 | +0.2106 | 7.5 |
| Relation-aware | K>=2 | 23 | 0.9620 | 0.7055 | +0.2564 | 9.0 |
| Relation-aware | depth=2 | 15 | 0.9417 | 0.6551 | +0.2865 | 11.8 |
| Hard Neg | all | 28 | 0.9714 | 0.7422 | +0.2292 | 7.5 |
| Hard Neg | K>=2 | 23 | 0.9652 | 0.6862 | +0.2791 | 9.0 |
| Hard Neg | depth=2 | 15 | 0.9467 | 0.6299 | +0.3168 | 11.8 |

The relation-aware model is slightly worse on every nontrivial subset. The
margin is still positive, but it is smaller than the hard-negative baseline.

### 4.3 Hard-pool metrics by depth (test)

| Depth | n | hardMRR | Prior | Margin | K mean | Random |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 15 | 0.9417 | 0.6551 | +0.2865 | 11.8 | 0.0787 |
| 3 | 8 | 1.0000 | 0.8000 | +0.2000 | 3.6 | 0.2229 |
| 4 | 3 | 1.0000 | 1.0000 | +0.0000 | 1.0 | 0.5000 |
| 5 | 2 | 1.0000 | 1.0000 | +0.0000 | 1.0 | 0.5000 |

The only hard-pool loss is at depth 2, where K=11.8. Depth 3 is perfect on the
hard pool. Depth 4/5 are K=1 and cannot distinguish models.

---

## 5. Paired source-cluster bootstrap

Unit: **test source** (15 sources, 28 positives). 10,000 resamples with
replacement; paired because both models are evaluated on the same resampled
sources.

| Subset | Metric | Δ (relation-aware − hard neg) | 95% CI | Significant? |
|---|---|---|---:|---|
| all | hardMRR | −0.0028 | [−0.0098, +0.0000] | no (marginal negative) |
| all | hard-pool margin | **−0.0191** | **[−0.0400, −0.0041]** | **yes, negative** |
| all | full-catalogue MRR | +0.0484 | [−0.0002, +0.1408] | no (marginal positive) |
| all | full-catalogue margin | +0.0412 | [−0.0039, +0.1319] | no |
| K>=2 | hardMRR | −0.0034 | [−0.0113, +0.0000] | no |
| K>=2 | hard-pool margin | −0.0226 | [−0.0441, −0.0051] | yes, negative |
| K>=2 | full-catalogue MRR | +0.0579 | [−0.0002, +0.1592] | no |
| K>=2 | full-catalogue margin | +0.0492 | [−0.0043, +0.1487] | no |
| depth=2 | hardMRR | −0.0050 | [−0.0150, +0.0000] | no |
| depth=2 | hard-pool margin | −0.0302 | [−0.0578, −0.0081] | yes, negative |
| depth=2 | full-catalogue MRR | +0.0238 | [−0.0004, +0.0660] | no |
| depth=2 | full-catalogue margin | +0.0142 | [−0.0085, +0.0527] | no |

Point estimates: hardMRR 0.9688 vs 0.9714; hard margin 0.2106 vs 0.2292; full
MRR 0.6889 vs 0.6404; full margin 0.2032 vs 0.1619.

Interpretation: the full-catalogue gains are positive but not statistically
decisive with 15 test sources. The hard-pool margin regression **is** supported.
There is no evidence that relation-aware scoring improves source-conditioned
reasoning on this dataset.

---

## 6. `猫 -> 生物` diagnostic

`猫` is a validation source (sense 2); its only closure target is
`生物` (`is_a`, depth 2).

| Model | Hard-pool rank | Prior rank | Full-catalogue rank |
|---|---:|---:|---:|
| Source-disjoint | #3 | #11 | **#3** |
| Hard Neg | #4 | #12 | **#13** |
| Relation-Aware | **#6** | #13 | **#12** |

The relation-aware head moves the full-catalogue rank from #13 to #12, but it
does **not** climb back toward #3. The hard-pool rank worsens from #4 to #6.
Its top-5 full-catalogue targets are `影像特征`, `形态学特征`, `洗澡`, `可食用`,
`大脑皮层`; `生物` is #12. This is the same failure mode as hard negatives:
the hard pool contains cross-branch hubs, so unrelated high-scoring hubs such as
`影像特征` are never penalized.

---

## 7. Expected-outcome check

| Expected outcome | Actual | Verdict |
|---|---|---|
| Per-relation MRR higher for frequent `is_a` / `has_property` | Test is 28/28 `is_a`, so per-relation = pooled = 0.6889. `has_property` has 0 depth>=2 positives. | **Untestable / not supported** |
| `猫 -> 生物` climbs toward #3 | Full rank **#12** (hard neg #13; disjoint #3) | **No** |
| Hard-neg MRR stays >= 0.97 | **0.9688** | **Narrow fail** |
| Model-vs-prior margin increases further | Hard-pool margin **−0.0186**; full-catalogue margin **+0.0413** (not significant) | **Mixed / no** |

The requested architecture is implemented, but the expected empirical outcome
does not hold. This is a data-diversity result, not a hidden implementation
success.

---

## 8. Why this run cannot test `W_r`

The design intent was: `is_a` should learn an ancestor projection, while
`has_property` / `has_part` learn different projections. The labels do not
support that comparison:

| Relation | Depth>=2 positives | Train / val / test |
|---|---:|---|
| `is_a` | 240 | 179 / 33 / 28 |
| `has_part` | 1 | 1 / 0 / 0 |
| `has_property` | **0** | 0 / 0 / 0 |
| other 42 types | 0 | 0 / 0 / 0 |

Consequences:

- 43 of 45 `W_r` heads never receive a closure positive.
- `has_part` has a single training example and no validation/test example.
- The test set exercises exactly one relation, so the "per-relation" test table
  is mathematically identical to the pooled table.
- For `is_a`, the relation-aware head is functionally a single matrix; the
  relation conditioning cannot change the ranking function on test.
- The observed delta vs hard negatives therefore comes from the **different
  initialization, the 10-epoch GAT freeze, and the resulting optimization
  trajectory**, not from switching between relation-specific matrices.

The data bottleneck is the same one identified in
`HARD_NEGATIVE_COMPARISON.md`: the closure targets are shared hubs
(`is_a` depth 2 has 102 rows but only 14 distinct targets), and the hard pools
contain mostly cross-branch hubs. Adding relation heads does not create
same-branch negatives.

---

## 9. Verdict

**Implemented:** yes. `train_closure_relaware.py` trains 45 `W_r` matrices with a
shared/gated fallback, freezes the GAT for 10 epochs, evaluates per-relation
metrics, compares against the hard-negative checkpoint on identical pairs and
pools, and writes per-relation attention diagnostics.

**Empirically supported:** no. On this dataset:

- Hard-neg MRR **0.9688**, narrowly below the 0.97 threshold and slightly below
  the 0.9714 baseline.
- Hard-pool margin **+0.2106**, below the baseline +0.2292; the paired CI
  excludes zero on the negative side.
- Full-catalogue MRR **+0.0486** and margin **+0.0412** vs baseline; both paired
  CIs include zero.
- `猫 -> 生物` full rank **#12**, not #3.
- No test relation diversity, so the relation-aware mechanism itself is
  untested.

**Supported claim:** the per-relation head is a clean, cheap extension that is
ready for a graph with more relation types. At the current graph size it does
not improve hard-negative ranking and slightly hurts the hard-pool margin. The
binding constraint is **closure-label relation diversity and same-branch
negative diversity**, not the scoring head.

---

## 10. Artifacts and reproduction

```bash
# Smoke (5 epochs)
python logos_transformer/train_closure_relaware.py --epochs 5

# Full run (deterministic, seed 42)
python logos_transformer/train_closure_relaware.py

# Disable the GAT freeze
python logos_transformer/train_closure_relaware.py --freeze-gat-epochs 0
```

Outputs:

```text
logos_transformer/checkpoints_relaware/
  closure_gat_relaware_best.pt          # primary, best val loss
  closure_gat_relaware_best_margin.pt   # diagnostic, best val margin
  training_report.txt                   # full tables + attention + caveats
```

The comparison checkpoint is loaded from

```text
logos_transformer/checkpoints_hardneg/closure_gat_hardneg_best.pt
```

and evaluated on the identical test pairs and hard pools. If it is missing or
incompatible, the run still completes and the comparison rows are reported as
`n/a`.

---

## 11. Caveats

- **Single-relation test set.** All 28 test positives are `is_a`. Per-relation
  MRR equals pooled MRR by construction; this is not evidence of relation-aware
  behavior.
- **`has_part` is one train example.** It is not evaluated and its gated blend is
  a single-example fit.
- **`has_property` has zero depth>=2 positives.** The expected
  `is_a` / `has_property` comparison cannot be run.
- **Test size.** 15 sources / 28 positives produce wide bootstrap CIs; a
  +0.048 full-MRR delta is not decisive.
- **Transductive split.** Source-disjoint prevents source overlap, but test
  nodes still appear in the message graph and their embeddings are updated.
- **Freeze is a confound.** The 10-epoch freeze changes the optimization path.
  This run does not isolate "relation-aware head" from "freeze schedule". A
  proper ablation would run `--freeze-gat-epochs 0` and compare.
- **AUC.** The reported AUC is on the sampled hard/medium/easy mix, not the
  random-negative AUC used in the earlier pair-random report.
- **Hard pools.** `is_a` has only 14 / 6 / 3 distinct targets at depth 2 / 3 /
  4+; the signal remains coarse branch membership.
