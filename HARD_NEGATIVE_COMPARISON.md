# Hard-Negative Training vs Random / Source-Disjoint — Closure Prediction

**Date**: 2026-10-22  
**Task**: predict `closure` edges at `depth >= 2` from 1-hop messages only.  
**Model**: 2-layer relation-aware GAT, 4 heads, hidden 32, bilinear head
`h_s^T W h_t` (204,480 params).  
**New script**: `logos_transformer/train_closure_hardneg.py`  
**Checkpoints**: `logos_transformer/checkpoints_hardneg/`

The source-disjoint experiment (§`SOURCE_DISJOINT_COMPARISON.md`) showed that
random closure-free negatives are too easy: a source-independent **target
prior** already reproduced ~92% of the filtered MRR. This experiment keeps the
same split, model and hyperparameters, but replaces most random negatives with
**hard negatives**: closure targets of *other sources* with the same relation
and the same depth band, excluding anything reachable from the query source.

The question is whether this forces the model to learn **source-conditioned**
reasoning beyond "which node looks like an ancestor".

---

## 1. Requested comparison

| Metric | Random Neg (`train_closure.py`) | Source-Disjoint (`train_closure_disjoint.py`) | Hard Neg (this run) |
|---|---|---|---|
| Test MRR | 0.7656 | 0.5690 | **0.6404** |
| **Hard-neg MRR** | — | **0.7411** | **0.9714** |
| **Model-vs-prior margin** | — | **+0.0407** (hard pool) / +0.0431 (full) | **+0.2292** (hard pool) / +0.1619 (full) |

Extended metrics:

| Metric | Random Neg | Source-Disjoint | Hard Neg |
|---|---|---|---|
| Test Hits@1 | 0.6389 | 0.3571 | **0.4643** |
| Test Hits@3 | 0.9167 | 0.6429 | **0.7143** |
| Test Hits@10 | 0.9722 | 1.0000 | **0.9643** |
| Test P@10 | — | 0.1867 | 0.1800 |
| Test AUC (original neg. mix) | 0.9969 | 1.0000 | **0.9940** (hard-mix) |
| Test AUC on shared hard-mix | — | 0.9264 | 0.9940 |
| Test hard-pool random MRR | — | 0.1951 | 0.1951 |
| Hard-pool K (mean) | — | 7.5 | 7.5 |
| Hard-pool queries / skipped | — | 28 / 0 | 28 / 0 |

Notes:

- The Random-Neg column used a **different split** (random pairs), so its MRR
  is not directly comparable to the source-disjoint columns. Its hard-neg
  metrics are not reported: the random-split model has seen many of the
  source-disjoint test sources during training, which would inflate them.
- The Source-Disjoint hard-neg metrics were computed **post-hoc on the exact
  same 28 test positives and hard pools** as the hard-neg model. Its
  full-catalogue MRR reproduces the earlier 0.5690, confirming the split and
  checkpoint.
- AUC is saturated under random negatives (0.9969 / 1.0000) and is not
  informative; on the shared hard-mix pairs, the disjoint model is 0.9264 and
  the hard-neg model 0.9940. The MRR/margin numbers are the primary evidence.
- `hard-neg MRR` ranks each positive against **all** eligible hard negatives
  for that positive (K mean 7.5, median 10, range 1–13), not against the 3
  sampled training negatives. The random baseline is the per-query
  `1/(K+1)` mean, 0.1951.
- `Model-vs-prior margin` = hard-pool MRR − hard-pool target-prior MRR. The
  target prior ranks targets by the model's mean score over all sources
  (`S.mean(0)`), exactly as in the source-disjoint analysis. The full-catalogue
  margin uses the same idea over the 170-node filtered catalogue.

**Point-estimate verdict on the requested thresholds:**

| Criterion | Value | Result |
|---|---|---|
| Hard-neg MRR > 0.3 | **0.9714** | **PASS** |
| Hard-pool margin > 0.1 | **+0.2292** | **PASS** |
| Full-catalogue margin > 0.1 | **+0.1619** | **PASS** |

The paired bootstrap (§5.5) confirms that the *improvement over the
source-disjoint model* is significant on both margins. The absolute hard-pool
margin is noisier: its 95% CI is `[+0.0814, +0.4285]`, so “> 0.1” is a
point-estimate pass, not a 95%-robust pass. See §6 for the precise claim.

---

## 2. Setup

### 2.1 Split (identical to the source-disjoint baseline)

- Unit: **source sense**. Every source appears in exactly one of train / val /
  test; closure positives follow their source, and negatives inherit their
  positive's source.
- 101 unique sources → **71 train / 15 val / 15 test** (seed 42).
- Positives: **180 / 33 / 28** edges.
- Depth distribution (train): `{2: 73, 3: 57, 4: 38, 5: 11, 6: 1}`.
- Test: `{2: 15, 3: 8, 4: 3, 5: 2}`.
- Test sources are the same as in the source-disjoint run; no test source
  appears as a training source.

### 2.2 Hard / medium / easy pools

For each positive `(source_s, target_t, relation R, depth d)`:

- **Hard** candidates: targets `t'` of **other sources** with the same `R` and
  the same depth band (`2`, `3`, `4+`), excluding:
  - `t' = t`, `t' = source_s`;
  - anything direct/closure-related to `source_s` in either direction
    (`facts.blocked_pairs`), i.e. anything reachable from `source_s`.
- **Medium** candidates: same relation, any depth, same exclusions, minus the
  hard pool (the tiers are disjoint).
- **Easy** candidates: uniformly random nodes, excluding the positive target,
  the source, direct/closure neighbours, and already-chosen hard/medium nodes.

Depth bands `2 / 3 / 4+` are used because exact depth gives 31/241 empty hard
pools (depth 6 has a single positive); banding reduces this to 23/241 while
preserving the structural meaning.

Default mix: **6 negatives per positive = 3 hard / 2 medium / 1 easy**
(50% / 30% / 20%). If a pool is too small, the sampler falls back
**hard → medium → easy** and prints an aggregate warning. The fallback is not
silent: the report records shortfalls per split.

### 2.3 Training

Same as `train_closure_disjoint.py`:

- Adam, lr 1e-3, `ReduceLROnPlateau(factor=0.5, patience=10)`;
- BCEWithLogitsLoss, `pos_weight = train_neg / train_pos = 6.00`;
- batch size 64, max 200 epochs, early-stopping patience 20;
- checkpoint selection by **best val loss** (primary), matching the earlier
  runs;
- a second **best val margin** checkpoint is saved purely as a diagnostic
  upper bound and is not the primary result.

After every epoch the script ranks each **validation** positive against all of
its eligible hard negatives and prints:

```
model MRR, target-prior MRR, margin = model MRR - prior MRR
```

Validation positives are used because they are source-disjoint and therefore
honest; the same diagnostic on train would be memorisation.

---

## 3. Pool feasibility (the main structural constraint)

The graph's closure targets are **shared hubs**, so hard pools are intrinsically
small:

| Relation | Band | Rows | Distinct targets | Distinct sources |
|---|---:|---:|---:|---:|
| `is_a` | 2 | 102 | **14** | 101 |
| `is_a` | 3 | 76 | **6** | 76 |
| `is_a` | 4+ | 62 | **3** | 48 |
| `has_part` | 2 | 1 | **1** | 1 |

Per-positive hard-pool size:

| Split | Positives | min | median | mean | max | zero |
|---|---:|---:|---:|---:|---:|---:|
| train | 180 | 0 | 4.0 | 5.6 | 13 | 22 |
| val | 33 | 0 | 5.0 | 6.5 | 13 | 1 |
| test | 28 | 1 | 10.0 | 7.5 | 13 | 0 |

Sampled negatives and fallback:

| Split | Requested hard | Used hard | Used medium | Used easy | Short hard | Empty hard | Short medium | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 540 | 405 | 346 | 329 | 62/180 | 22/180 | 73/180 | 0 |
| val | 99 | 84 | 51 | 63 | 7/33 | 1/33 | 15/33 | 0 |
| test | 84 | 74 | 36 | 58 | 5/28 | 0/28 | 15/28 | 0 |

Why the pools are small:

- Most other-source targets at the same relation are **also ancestors of the
  query source** (e.g. almost every animal source reaches `生物`). Those targets
  are blocked, not sampled.
- What remains is mostly **cross-branch** hubs: for an animal source the hard
  pool contains `人造物`, `医学概念`, `病理状态`, `信息人造物`, etc., but not
  another animal ancestor.
- `has_part` has one positive and one other source, so it always falls back to
  medium/easy; it cannot be hard-negative trained.

This is a real property of the current graph, not a sampling bug. It shapes the
interpretation in §7.

---

## 4. Training run

- Epochs run: **35** (early stop; patience 20).
- Best val loss: **0.2102** at epoch **15**.
- Val hard-pool margin at the best-loss epoch: **+0.1417**.
- Best val hard-pool margin: **+0.1557** at epoch **16**.
- Train loss/accuracy at the end: ~0.0175 / 0.9968; val loss ~0.2571,
  val accuracy ~0.9437. The model overfits the training pairs quickly, but the
  val margin stays positive and stable after ~epoch 10.

Selected validation margin curve (`model / prior -> margin`):

| Epoch | Model hard MRR | Prior hard MRR | Margin | Val loss |
|---:|---:|---:|---:|---:|
| 1 | 0.4688 | 0.5722 | −0.1033 | 1.1878 |
| 5 | 0.8326 | 0.7943 | +0.0383 | 0.6492 |
| 8 | 0.8932 | 0.7872 | +0.1061 | 0.4719 |
| 10 | 0.8993 | 0.7880 | +0.1113 | 0.3752 |
| 14 | 0.9281 | 0.7943 | +0.1339 | 0.2451 |
| **15** | **0.9359** | **0.7943** | **+0.1417** | **0.2102** |
| **16** | **0.9437** | **0.7880** | **+0.1557** | 0.2211 |
| 20 | 0.9102 | 0.7943 | +0.1159 | 0.2475 |
| 35 | 0.9177 | 0.7943 | +0.1234 | 0.2571 |

The margin becomes positive around epoch 4–5 and never returns to zero. This is
the first direct evidence in this project that the model can use the source to
break ties among ancestor-like targets.

---

## 5. Results

### 5.1 Nontrivial subsets

K=1 (all depth-4/5 test queries) is prior-trivial: with one candidate, model
and prior both rank the target first, contributing MRR 1.0 and margin 0.0.
Reporting them separately avoids inflating the headline.

| Model | Subset | n | Hard MRR | Prior | Margin | K | Full MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| Hard Neg | all | 28 | 0.9714 | 0.7422 | **+0.2292** | 7.5 | 0.6404 |
| Hard Neg | K≥2 | 23 | 0.9652 | 0.6862 | **+0.2791** | 9.0 | 0.6057 |
| Hard Neg | depth=2 | 15 | 0.9467 | 0.6299 | **+0.3168** | 11.8 | 0.5842 |
| Source-disjoint | all | 28 | 0.7411 | 0.7004 | +0.0407 | 7.5 | 0.5690 |
| Source-disjoint | K≥2 | 23 | 0.6848 | 0.6352 | +0.0496 | 9.0 | 0.5188 |
| Source-disjoint | depth=2 | 15 | 0.6722 | 0.6140 | +0.0582 | 11.8 | 0.5022 |

The margin is **higher** on the nontrivial subsets, so the headline is not an
artefact of K=1 queries. The meaningful improvement is at depth 2 (K≈12) and
depth 3 (K≈4); depth 4/5 cannot distinguish the models.

### 5.2 Hard-pool metrics by depth (test)

| Depth | n | Disjoint hard MRR | Disjoint prior | Disjoint margin | HardNeg hard MRR | HardNeg prior | HardNeg margin | K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 15 | 0.6722 | 0.6140 | +0.0582 | **0.9467** | 0.6299 | **+0.3168** | 11.8 |
| 3 | 8 | 0.7083 | 0.6750 | +0.0333 | **1.0000** | 0.7917 | **+0.2083** | 3.6 |
| 4 | 3 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0 |
| 5 | 2 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0 |

The hard-pool gain is concentrated at depths 2 and 3. Depth 4/5 have K=1 and no
discrimination power.

### 5.3 Full-catalogue metrics by depth (corrected)

**Important correction.** The earlier `SOURCE_DISJOINT_COMPARISON.md` per-depth
numbers were affected by an alignment bug in
`train_closure.py::filtered_metrics`: it computes the overall MRR correctly, but
its `by_depth` view zips a **set-ordered** rank list with `test_positives`.
When one source has several closure targets, ranks are assigned to the wrong
depth. The overall MRR/Hits/AUC are unaffected; the per-depth breakdown was
wrong.

The new script computes per-depth ranks **per positive** (`full_catalogue_by_depth`)
with the same mask. Corrected test numbers:

| Depth | n | Disjoint MRR | Disjoint H@1 | Disjoint H@10 | HardNeg MRR | HardNeg H@1 | HardNeg H@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 15 | 0.5022 | 0.2667 | 1.0000 | **0.5842** | **0.4000** | 0.9333 |
| 3 | 8 | 0.5500 | 0.3750 | 1.0000 | **0.6458** | **0.5000** | 1.0000 |
| 4 | 3 | 0.6667 | 0.3333 | 1.0000 | 0.6667 | 0.3333 | 1.0000 |
| 5 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Corrected interpretation:

- There is **no monotonic depth decay**. The old numbers (d2 0.7667, d3 0.4000,
  d4 0.3000, d5 0.1667) were misaligned.
- Depth 3 does have top-1 hits (3/8 for the disjoint model, 4/8 for the
  hard-neg model), so the earlier “depth≥3 has no top-1” claim is false.
- Depth 4/5 have n=3/2 and should not be interpreted.
- The hard-neg model improves depth 2 and depth 3, ties depth 4/5. The overall
  MRR gain is therefore broad across the informative depths, not just one
  lucky query.

The correction does **not** change the prior conclusion: the disjoint model's
overall test MRR 0.5690 is still ~92% explained by the target prior 0.5259.

### 5.4 Paired cluster bootstrap

Test has 15 sources / 28 positives; the appropriate unit is the **source**.
The bootstrap resamples the 15 test sources with replacement (10,000 draws) and
recomputes the metric, preserving within-source correlation.

Absolute estimates:

| Metric | Model | Mean | 95% CI |
|---|---|---:|---|
| Hard MRR | Hard Neg | 0.9700 | [0.8957, 1.0000] |
| Hard-pool margin | Hard Neg | 0.2322 | [0.0814, 0.4285] |
| Hard MRR | Source-disjoint | 0.7408 | [0.5962, 0.8676] |
| Hard-pool margin | Source-disjoint | 0.0424 | [0.0094, 0.0914] |
| Full MRR | Hard Neg | 0.6469 | [0.5263, 0.7828] |
| Full MRR | Source-disjoint | 0.5733 | [0.4577, 0.7058] |

Paired difference (Hard Neg − Source-disjoint, same resampled sources):

| Comparison | Metric | Δ | 95% CI | Significant? |
|---|---|---:|---|---|
| Hard Neg − Disjoint | Hard MRR | **+0.2294** | **[+0.1068, +0.3694]** | **yes** |
| Hard Neg − Disjoint | Hard-pool margin | **+0.1904** | **[+0.0599, +0.3582]** | **yes** |
| Hard Neg − Disjoint | Full-catalogue MRR | +0.0737 | [−0.0025, +0.1824] | no (marginal) |
| Hard Neg − Disjoint | Full-catalogue margin | **+0.1211** | **[+0.0330, +0.2444]** | **yes** |
| Hard Neg − Disjoint | K≥2 hard margin | **+0.2285** | **[+0.0714, +0.4095]** | **yes** |
| Hard Neg − Disjoint | depth-2 hard margin | **+0.2585** | **[+0.0963, +0.4392]** | **yes** |

Interpretation:

- The hard-pool improvement is large and robust. The full-catalogue **margin**
  improvement is also robust: the hard-neg model relies less on the global
  target prior.
- The absolute full-catalogue **MRR** gain is positive (+0.074) but the CI
  includes zero; with 15 test sources it is not statistically decisive.
- The absolute hard-pool margin CI lower bound is 0.081, slightly below the
  0.1 threshold. The threshold passes on the point estimate, but a 95%-robust
  claim would need more test sources.
- The hard-neg model improves over its own prior on **12/28** test positives on
  the full catalogue and loses on **0**; the disjoint model wins on 5, ties 21,
  loses on 2. This is the cleanest summary of the source-conditioning gain.

### 5.5 Per-query diagnostics

| Model | Wins vs prior | Ties | Losses | Mean gain on wins |
|---|---:|---:|---:|---:|
| Hard Neg (hard pool) | 9 | 19 | 0 | +0.7131 |
| Source-disjoint (hard pool) | 5 | 23 | 0 | +0.2280 |
| Hard Neg (full catalogue) | 12 | 16 | 0 | +0.3778 |
| Source-disjoint (full catalogue) | 5 | 21 | 2 | +0.2486 |

The hard-neg model fixes more queries without introducing new losses. The
hard-pool gains are concentrated in branch-specific hubs:

- `工具 → 人造物`: prior #14 → model #1
- `设备 → 人造物`: prior #14 → model #1
- `数据库 / 库 / 文件 / 程序 / API → 信息人造物` (depth 2): prior #9 → model #1
- `… → 人造物` (depth 3): prior #6 → model #1
- `人 → 生物`: prior #12 → model #5 (improved but still not top-1)
- `猫 → 生物`: prior #12 → model #4 (improved but not top-1)

The failures are equally informative.

---

## 6. Qualitative counterexamples

### 6.1 `猫 → 生物` (val, depth 2)

| Model | Hard rank | Prior rank | Full-catalogue rank |
|---|---:|---:|---:|
| Source-disjoint | #3 | #11 | **#3** |
| Hard Neg | #4 | #12 | **#13** |

Hard-Neg top-5 for 猫: `作业`, `影像特征`, `大脑皮层`, `形态学特征`, `营养物`;
`生物` is #13.

The hard-neg model still improves over its own prior (1/4 vs 1/12), but it is
**worse than the disjoint model** for this query. The hard pool only contains
cross-branch hubs; once the model is optimized to separate branches, unrelated
high-scoring hubs (`作业`, `影像特征`, ...) are not penalized because they are
not in the pool. This is visible in the full-catalogue ranking.

### 6.2 `人 → 生物` (test, depth 2)

| Model | Hard rank | Prior rank | Full-catalogue rank |
|---|---:|---:|---:|
| Source-disjoint | #4 | #11 | **#4** |
| Hard Neg | #5 | #12 | **#22** |

This is the only test positive with hard rank > 1. It drives the test Hits@10
drop from 1.0000 (disjoint) to 0.9643 (hard neg). The aggregate gain is positive
because the other 27 queries improve, but this is a genuine regression, not
noise to be hidden.

### 6.3 Attention

Mean normalised attention entropy: **0.9997 (init) → 0.9954 (trained)**.
For 猫, the strongest layer-1 attention shifts to `吃`, `抓`, `跑`
(`can_do`), the same qualitative pattern as the earlier runs. Hard-negative
training changes the score surface more than the attention pattern.

---

## 7. Why the margin is real but coarse

The hard pools are **cross-branch** by construction:

- Same-branch ancestors of the query source are, by definition, reachable from
  it and are therefore blocked.
- What remains is other branches' hubs. For an animal source the hard pool
  contains `人造物`, `医学概念`, `病理状态`, `信息人造物`, `解剖结构`, etc.,
  but not another animal ancestor.
- The model therefore learns a **branch classifier**: animal sources → `生物`,
  tool sources → `人造物`, data sources → `信息人造物`, medical sources →
  `医学概念` / `解剖结构` / `病理状态`. This is genuine source-conditioned
  signal, but it is coarse.

The target prior is strong because these hubs are globally common. The hard-neg
model **lowers** the full-catalogue prior MRR from 0.5259 to 0.4784 while
**raising** the per-source MRR from 0.5690 to 0.6404, so the margin improves
from +0.0431 to +0.1619. That is exactly the desired direction: less reliance
on "which target is popular", more reliance on "which target belongs to this
source".

What the current experiment does **not** show:

- Fine-grained same-branch ancestor ranking. The hard pool has almost no
  same-branch negatives, so the model is never forced to choose between, say,
  `猫科` and `犬科`.
- Uniform full-catalogue improvement. `猫` and `人 → 生物` regress, and the
  absolute full-catalogue MRR gain is not significant.
- Depth decay. The corrected per-depth numbers show improvements at depth 2
  and 3, with tiny samples at depth 4/5.

---

## 8. Verdict

**The requested thresholds pass on the point estimates:**

- Hard-neg MRR **0.9714 > 0.3**.
- Hard-pool margin **+0.2292 > 0.1**.
- Full-catalogue margin **+0.1619 > 0.1**.

**The improvement over the source-disjoint baseline is statistically supported**
for hard MRR (+0.229), hard-pool margin (+0.190), and full-catalogue margin
(+0.121); all paired bootstrap CIs exclude zero.

**The stronger claim “hard-negative training uniformly improves multi-hop
ranking” is not supported.**

- Absolute full-catalogue MRR +0.074, CI [−0.003, +0.182], not significant.
- Hits@10 drops from 1.0000 to 0.9643 because of `人 → 生物` (full rank #22).
- `猫 → 生物` regresses from full rank #3 to #13.
- The absolute hard-pool margin CI [0.081, 0.429] includes values below 0.1.
- The learned signal is coarse branch membership, not fine-grained ancestor
  ranking.

**Supported claim:** at this graph size, hard-negative training significantly
increases source-conditioned signal and reduces reliance on a source-independent
target prior. It does so mainly by learning cross-branch membership; it does
not yet solve fine-grained multi-hop ranking.

This is a **partial success**, not the “architecture genuinely cannot do
source-conditioned reasoning” branch (which would have been margin < 0.05), and
not a complete solution.

---

## 9. Recommended next steps

1. **Expand the animal library / target diversity.** The binding constraint is
   the 14 / 6 / 3 distinct targets at depth 2 / 3 / 4+. More distinct
   same-branch ancestors (e.g. `猫科`, `犬科`, `哺乳动物`) would create
   same-branch hard negatives and force finer discrimination.
2. **Add a branch-disjoint or target-disjoint split.** The current protocol is
   source-disjoint but still transductive: test-source embeddings are updated
   as targets/neighbours. A target-disjoint split would test whether the
   branch signal generalizes to unseen hubs.
3. **Try a relation-aware head `W_r`.** It is cheap, but note that 240/241
   closure rows are `is_a`; data diversity is the bigger lever. A pairwise
   margin/ListNet loss with in-batch hard negatives may be more useful than
   `W_r` alone.
4. **Add full-catalogue hard negatives.** The current hard pool excludes
   ancestor-like confounders such as `作业` and `影像特征` because they are not
   same-relation/depth targets of other sources. Sampling from the model's
   top-K full-catalogue mistakes would directly penalize the `猫` failure mode.
5. **Increase the test set.** 15 sources / 28 positives give wide CIs. Even
   with the same model, more sources would resolve whether the absolute margin
   is reliably above 0.1.
6. **Calibrate / temperature-scale the score surface.** The hard-neg model
   assigns high scores to unrelated hubs for 猫. A calibration term or a
   small amount of easy-negative replay may reduce these spurious highs without
   losing the hard-pool gain.

---

## 10. Reproduction

```bash
# Smoke (5 epochs)
python logos_transformer/train_closure_hardneg.py --epochs 5

# Full run (deterministic, seed 42)
python logos_transformer/train_closure_hardneg.py
```

Outputs:

```
logos_transformer/checkpoints_hardneg/
  closure_gat_hardneg_best.pt          # primary, best val loss
  closure_gat_hardneg_best_margin.pt   # diagnostic, best val margin
  training_report.txt                  # all tables + caveats
```

The source-disjoint baseline is loaded automatically from

```
logos_transformer/checkpoints_disjoint/closure_gat_disjoint_best.pt
```

and evaluated on the identical test pairs and hard pools. If the checkpoint is
missing or incompatible, the run still completes and the comparison columns are
reported as `n/a`.

---

## Appendix: correction to the source-disjoint per-depth numbers

The earlier source-disjoint report listed:

| Depth | Old MRR | Old H@1 | Correct MRR | Correct H@1 | Correct H@10 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.7667 | 0.6667 | **0.5022** | **0.2667** | 1.0000 |
| 3 | 0.4000 | 0.0000 | **0.5500** | **0.3750** | 1.0000 |
| 4 | 0.3000 | 0.0000 | **0.6667** | **0.3333** | 1.0000 |
| 5 | 0.1667 | 0.0000 | **1.0000** | **1.0000** | 1.0000 |

Cause: `filtered_metrics` computes `ranks` by iterating
`keep_by_query` and then `for dst in keep` (a set), but computes `by_depth` by
`zip(test_positives, ranks)`. When a source has multiple positives, the two
orders differ, so ranks are assigned to the wrong depth. The overall MRR is a
mean over the same multiset of ranks and is unaffected.

Consequences:

- No monotonic depth decay.
- Depth 3 does have top-1 hits (3/8 disjoint, 4/8 hard neg).
- Depth 4/5 have n=3/2 and are not interpretable.
- The “depth≥3 has no top-1” sentence in the earlier conclusion should be
  retracted. The target-prior conclusion (0.5259 / 0.5690 ≈ 92%) is unchanged.
