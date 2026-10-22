# Source-Disjoint vs Random Split — Closure Prediction

**Date**: 2026-10-22
**Task**: predict `closure` edges at `depth >= 2` from 1-hop messages only.
**Model**: 2-layer relation-aware GAT, 4 heads, hidden 32, bilinear head `h_s^T W h_t`
(204,480 params). Same optimizer/loss/hyperparameters in both runs (seed 42).

The only change is the split unit: pairs (previous run) vs **sources** (this run).
Every source sense appears in exactly one of train / val / test; each closure edge
follows its source, and negatives inherit their positive's source, so no negative
pair can cross splits.

## Split and dataset

| | Random split (`train_closure.py`) | Source-disjoint (`train_closure_disjoint.py`) |
|---|---|---|
| Split unit | pair | source sense |
| Unique sources | may repeat across splits | **101** → 71 / 15 / 15 |
| Positives (train/val/test) | 169 / 36 / 36 | 180 / 33 / 28 |
| Negatives (train/val/test) | 498 / 107 / 107 | 530 / 97 / 83 |
| Test samples | 143 | 111 |
| Early stopping | val pairs | val sources (15) |

The two test sets are different, so small deltas are sampling noise; treat the
table as indicative, not as a controlled paired comparison.

## Requested comparison

| Metric | Random split (previous) | Source-disjoint (this run) | Δ |
|---|---|---|---|
| Test ROC-AUC | 0.9969 | **1.0000** | +0.0031 |
| Test MRR | 0.7656 | **0.5690** | **−0.1966** |
| Hits@1 | 0.6389 | **0.3571** | **−0.2818** |
| Hits@10 | 0.9722 | **1.0000** | +0.0278 |
| Hits@3 | 0.9167 | 0.6429 | −0.2738 |

`Hits@10` and AUC are preserved (AUC saturates), while top-1 precision drops by
about 28 points. The drop is **not** a collapse: every test positive is still
retrieved within the top 10 of the filtered 170-node catalogue.

## Train vs test (sanity check)

| Split | Sources | Pos | Neg | AUC | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|---|---|---|
| train | 71 | 180 | 530 | 1.0000 | 0.5555 | 0.3222 | 0.7111 | 1.0000 |
| val | 15 | 33 | 97 | 1.0000 | 0.5389 | 0.3030 | 0.6667 | 1.0000 |
| **test** | **15** | **28** | **83** | **1.0000** | **0.5690** | **0.3571** | **0.6429** | **1.0000** |

Train-source and test-source MRR are essentially identical (0.556 vs 0.569).
That is the intended check: **no source-specific memorisation**, so the previous
0.766 MRR is consistent with inflation from seeing the same source in train and
test.

**Why isn't train MRR ~1.0?** Because the two objectives differ. Training uses 3
sampled negatives per positive and reaches BCE ≈ 0.001 / AUC 1.000 on that
distribution. MRR ranks against all 170 nodes, including ancestor-like
confounders that were never sampled as negatives (e.g. `洗澡` for a disease
source). A manual check on a train source with 5 positives puts 4 of them in the
top 4 and the fifth at rank 6; the remaining error comes from such unsampled
candidates, not from failed fitting. The sanity check passes on the training
distribution (BCE/AUC), while full-catalogue MRR is the harder, shared metric.

## Why AUC = 1.0 is not the headline

1. **Easy negatives.** Positives are `is_a` ancestors; negatives are uniformly
   sampled closure-free nodes. A global "this node looks like an ancestor" score
   separates them almost perfectly. Against *all* non-related nodes the hard AUC
   (post-hoc, saved checkpoint) is 0.9944, still high but no longer pinned to 1.0.
2. **Most of the ranking is a target prior, not source conditioning.** Ranking
   every target by its mean score over all sources (a source-independent
   baseline) already yields:

   | | Train MRR | Val MRR | Test MRR |
   |---|---|---|---|
   | Per-source model | 0.5555 | 0.5389 | 0.5690 |
   | Target-prior baseline | 0.5312 | 0.5260 | **0.5259** |

   The source-independent prior accounts for ~92% of the test MRR. The learnable
   bilinear head is therefore mostly learning **which nodes are plausible
   ancestor-like targets**, not a source-specific relation operator.
3. **Depth decay** (test sources) confirms the shallow bias:

   | depth | n | MRR | H@1 | H@10 |
   |---|---|---|---|---|
   | 2 | 15 | 0.7667 | 0.6667 | 1.0000 |
   | 3 | 8 | 0.4000 | 0.0000 | 1.0000 |
   | 4 | 3 | 0.3000 | 0.0000 | 1.0000 |
   | 5 | 2 | 0.1667 | 0.0000 | 1.0000 |

   2-hop closure is learned well; depth ≥ 3 is essentially "retrieved somewhere
   in the top 10" without top-1 precision. A 2-layer GAT only sees 2 hops, so
   this is expected, but it bounds the claim.

## Verdict

- **By the AUC rule (> 0.85): passes** — but AUC is saturated and is not the
  discriminating metric. Do not use it alone to claim relational reasoning.
- **By the "drop < 0.10 = good generalisation" rule: does not pass** — MRR
  drops 0.197 and Hits@1 drops 0.282. Hits@10 is unchanged.
- **What is supported**: the model generalises to unseen *query sources*
  (train ≈ test), and it retrieves the correct closure target within the top 10
  for every test positive. Source overlap is the most plausible explanation for
  the previous MRR being higher, as suspected.
- **What is not supported**: that the gain comes from source-conditioned
  relational structure. Roughly 92% of the test MRR is reproduced by a global
  target prior, and depth ≥ 3 has no top-1 hits. The honest claim today is
  "learns a shallow ancestor prior plus a small source-specific signal", not
  "learns multi-hop relational reasoning".

This is a useful negative result: before the source-disjoint run, the 0.766 MRR
looked like relational reasoning; after it, the leakage-free number is 0.569 and
mostly explainable by a prior.

## Recommended next experiment

The current negatives are too easy. The decisive test is **hard negatives**:

- sample negatives from *other sources'* closure targets, matched on relation and
  depth (e.g. for 猫's `is_a` depth-2 target, use another animal's depth-2
  `is_a` target);
- re-run both splits; the gap between model and target-prior baseline then
  measures genuine source conditioning;
- report bootstrap confidence intervals over the 15 test sources (the current
  test set is small).

Other follow-ups: relation-conditioned heads (`W_r`), source-disjoint +
target-disjoint splits, an inductive variant that masks test-source embeddings,
and expanding the animal library to increase the number of sources.

## Reproduction

```bash
# previous (random pair split)
python logos_transformer/train_closure.py

# this run (source-disjoint)
python logos_transformer/train_closure_disjoint.py

# smoke test
python logos_transformer/train_closure_disjoint.py --epochs 5
```

Artifacts:

- `logos_transformer/checkpoints/closure_gat_best.pt` + `training_report.txt` (random)
- `logos_transformer/checkpoints_disjoint/closure_gat_disjoint_best.pt` + `training_report.txt` (source-disjoint)
