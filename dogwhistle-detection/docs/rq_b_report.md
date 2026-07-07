# RQ-B Multiclass Ingroup Classifier — Full Report

> **Audience.** Teammates and outside readers who haven't seen this code before. Read § A first if you're new; skip to § 3 if you only want the numbers.
> **Status.** All four runs complete (jobs `485354`, `485359`, `485367`, finished 2026-04-29 ≈13:30 CEST).
> **Last updated.** 2026-04-29.

---

## TL;DR

We trained a RoBERTa-base text classifier (125 M parameters) to predict, given a Reddit comment containing a dog-whistle phrase, **which ingroup the dog-whistle targets** (one of 17 categories: `racist`, `antisemitic`, `transphobic`, `Islamophobic`, …). On the grouped split (no dog-whistle root appears in both train and test) we get **macro-F1 = 0.353 ± 0.007**. On a leaky alternative split the same model gets **0.996**. The 64-percentage-point gap is the main thing we want to report: it suggests that most of the apparent "performance" on this dataset comes from the model memorising the term→ingroup mapping rather than learning to recognise dog-whistle ingroups in novel comments.

A 0.353 macro-F1 over 17 classes is well above chance (~0.06) but mediocre in absolute terms. The most useful diagnostic is that **the largest test class (`anti-liberal`, 720/2095 rows) gets F1 = 0.000**: every `anti-liberal` term in the test set is a phrase the model has never seen during training, and the surrounding text doesn't carry enough "anti-liberal-ness" signal for the model to fall back on. The model is weak because the task is hard, and the grouped split makes that weakness visible rather than hiding it. § 6 below walks through this.

---

## A · For someone new to the project

### A.1 What is RQ-B?

The Bocconi 597 NLP project on dog-whistle detection has three research questions:

- **RQ-A.** Given a Reddit comment that contains a dog-whistle phrase, is the phrase being used in its **coded** sense or its **literal** sense? (Binary classification.)
- **RQ-B.** Given a Reddit comment that uses a phrase in its coded sense, **which ingroup does the dog-whistle target?** (17-way classification — this report.)
- **RQ-C.** Same input as RQ-B, but produce a *structured* output: the dog-whistle's root form, the ingroup, the dictionary definition of the coded meaning, and a free-text explanation. (Generation.)

RQ-B is the pure-classifier version of "which group is this phrase against?". It does not generate explanations; it picks one of 17 labels.

### A.2 What is `silent_signals`?

`silent_signals` is a corpus released alongside *"Silent Signals, Loud Impact: LLMs for Word-Sense Disambiguation of Coded Dog Whistles"* (Kruk et al. 2024, ACL). The dataset is built from two sources:

- **Allen AI Glossary of Dog-Whistles** — a curated dictionary of ≈300 dog-whistle phrases (called *roots*), each annotated with surface forms, an ingroup, and a coded-meaning definition. This glossary is **deterministic**: every phrase maps to exactly one ingroup. There is no `coincidence → {antisemitic, white supremacist}` ambiguity in the data — it's listed under one and only one ingroup.
- **Reddit + congressional speech corpus** — actual occurrences of the glossary phrases in real text, with human annotation of whether each occurrence uses the phrase in its coded or literal sense.

For RQ-B we use only the **coded** occurrences (the positives). Each row in our data has:

| column | what it is | example |
|---|---|---|
| `content` | the full text of the Reddit comment | `"Probably just a coincidence I'm sure.."` |
| `dog_whistle` | the surface form actually present in this comment | `"coincidence"` |
| `dog_whistle_root` | the canonical root the surface form maps to | `"coincidence"` |
| `ingroup` | the gold answer — which group this dog-whistle targets | `"antisemitic"` |
| `definition` | dictionary-style definition of the coded meaning | `"Something is very odd or coincidental"` |

The model gets `(content, dog_whistle)` and has to predict `ingroup`.

### A.3 Why is this hard?

Three reasons:

1. **Seventeen classes.** Random guessing gets you 1/17 ≈ 0.06 macro-F1. Even a trivial "always pick the most-frequent class" classifier scores ~0.04 macro-F1 (because it gets one class right and 16 classes wrong). So we're fighting against a high ceiling of difficulty even before considering the linguistic challenges.
2. **Test ingroups have totally different priors than train.** In our test set, `anti-liberal` is 34% of the rows. In our training set, `anti-liberal` is only 5.8%. The model has to predict a class it rarely saw, on examples it has *never* seen.
3. **The Reddit text itself is noisy.** Comments are short, fragmentary, and filled with sarcasm, in-jokes, and contextual references. Without seeing the dog-whistle phrase explicitly listed in a glossary, even a human can struggle to assign an ingroup label.

---

## B · Methodology, step by step

### B.1 The data

`data/final/rq_b_multiclass/` holds three Parquet files (`train.parquet`, `val.parquet`, `test.parquet`) with row counts:

| split | rows | distinct ingroups present | notes |
|---|---|---|---|
| train | 9 064 | 17 | `anti-liberal` only 525 rows (5.8 %) |
| val | 1 764 | 13 | includes 10 `misogynistic` rows that are absent from train (dropped before training; see § B.4) |
| test | 2 095 | 7 | `anti-liberal` is **34 %** (720 rows) — major prior shift vs train |

Each row is one Reddit comment containing one dog-whistle phrase used in its coded sense.

### B.2 Why we split by root (and what "grouped" means)

This is the key methodological choice and worth understanding in detail.

**Naive approach (which we did NOT use).** Take all 12,923 rows, shuffle, put 70% in train, 15% in val, 15% in test. Train the model. Test it. Report the number.

**Problem with the naive approach.** The same dog-whistle phrase appears in many rows. For instance, the phrase `"absent fathers"` shows up in dozens of Reddit comments — they're different sentences, but they all contain the same phrase. Under a random split, some of those comments end up in train, some in test. The model trains on "this phrase = racist" examples and is then tested on... a different sentence containing the same phrase. The model just has to remember which phrase mapped to which ingroup. This isn't classification of dog-whistles — it's a glossary lookup.

**What we did instead — group splits by root.** Every dog-whistle phrase belongs to a *root* (e.g. `"absent fathers"`, `"absentee fathers"`, `"absent father"` all map to root `"absent fathers"`). We assigned each *root* to exactly one of {train, val, test}. So if the root `"absent fathers"` is in train, then *every* sentence using *any* surface form of that root is in train. The test set is built from completely different roots that the model has never been told the ingroup of.

**Concretely:**
- Train: 219 distinct roots
- Val: 39 distinct roots
- Test: 40 distinct roots
- **Overlap: 0**

This is what people mean by "grouped split" or "root-stratified split". It tests whether the model has learned to *generalise* the dog-whistle phenomenon — to recognise an ingroup signal even on phrases it has never seen labelled before.

The split assignment lives in `data/manifests/split_manifest.json` (`root_to_split` is the dictionary mapping every root to a split). It was constructed once, with seed 42, and frozen.

### B.3 The alt-split (Run C only) — and why we built it

`data/final/rq_b_multiclass_random/` is a *different* split of the same data, built as a sensitivity check. We pooled the grouped-split train + val + test (12,923 rows) and re-split them stratified by ingroup (not by root) using `sklearn.model_selection.StratifiedShuffleSplit` with random state 42, 70/15/15.

This deliberately allows root-leakage: the same dog-whistle phrase can appear in both train and test. By design, training on this split lets the model see most test phrases during training. It is the *bad* split methodologically — the one that overstates performance.

**We trained on this split on purpose** to measure the cost of root-grouping. The contrast between Run A (grouped) and Run C (alt) isolates: how much of the model's "performance" comes from learning the dog-whistle phenomenon vs. just memorising the corpus's deterministic glossary?

The build script is `scripts/hpc/build_alt_split_rq_b.py`.

### B.4 The label space

We have 17 ingroups in the train set. We use those 17 as the model's output classes. The val set contains 10 rows tagged with an 18th ingroup (`misogynistic`) that doesn't appear in train. We drop those 10 val rows before training, because there is no way to teach the model a class it has zero training examples of, and counting them in macro-F1 was a methodological bug in earlier scaffolding. The drops are recorded in `label_map.json` → `drop_report` per run.

The test set's 7 ingroups are all in train, so the test set is unaffected.

### B.5 Input format (two arms)

Each `(content, dog_whistle)` pair gets serialised into a single text input the model reads. Two arms tested:

- **`term` arm (the headline run):**
  ```
  Text: <content>
  Matched term: <dog_whistle>
  Predict the ingroup category.
  ```
- **`text_only` arm (input ablation):**
  ```
  Text: <content>
  Predict the ingroup category.
  ```

The `text_only` arm asks: how much of the ingroup signal is recoverable from surrounding text alone, when the model isn't told which phrase is the dog-whistle? Compared to the `term` arm, it isolates the contribution of the matched phrase itself.

### B.6 The model — RoBERTa-base, in plain words

**RoBERTa** (Liu et al. 2019) is a transformer-based language model — same family as BERT and GPT, fundamentally a stack of self-attention layers that read text and produce dense vector representations. **RoBERTa-base** has 125 million parameters, 12 transformer layers, and a 50,000-token vocabulary. It was originally pre-trained by Facebook AI on 160 GB of English text.

**Fine-tuning** means we take that pre-trained model and continue training it on our task — predicting an ingroup label given a comment + dog-whistle. We add one extra layer at the top of the model: a "classification head" that takes the model's representation of the input text and produces 17 numbers, one per ingroup. We pass these through softmax to get probabilities, predict the highest, and compare to the gold label. We measure how wrong we were (the loss), and we update the model's parameters slightly to be less wrong next time. Repeat 9,064 times per epoch, for several epochs.

**Why RoBERTa-base?** It's a standard fine-tuning baseline for a task at this scale — small enough to fit on a single A100 MIG slice in mixed precision, big enough to capture the linguistic patterns Reddit uses. We originally tried `microsoft/deberta-v3-large` (a more capable model) but ran into numerical instability during training (notes in `docs/sessions/2026-04-29_rqa_nan_postmortem.md`) and switched.

### B.7 Training procedure

Every model variant in this report was trained with the same hyperparameters (configured in `scripts/configs/multiclass.yaml`):

| Hyperparameter | Value | What it does in plain language |
|---|---|---|
| `model_name` | `FacebookAI/roberta-base` | The pre-trained transformer we start from |
| `max_length` | 512 | We cut comments off at 512 tokens (roughly 350 words). Longer comments get truncated. |
| `epochs` | 5 (max) | One epoch = one pass through the training set. Capped at 5 but allowed to stop earlier (see `early_stopping_patience` below). |
| `batch_size` | 32 | We process 32 rows at a time before doing a parameter update |
| `lr` | 2e-5 | The learning rate — how big a parameter update is. 2e-5 is the canonical value for RoBERTa-base fine-tuning |
| `weight_decay` | 0.01 | A regularisation term that gently nudges parameters toward zero — prevents overfitting |
| `warmup_ratio` | 0.1 | The first 10% of training uses a linearly-ramping learning rate (smaller at the start, full rate by 10%) |
| `max_grad_norm` | 1.0 | Cap on how big a single update can be — protects against rare exploding-gradient cases |
| `early_stopping_patience` | 2 | If validation macro-F1 doesn't improve for 2 consecutive epochs, stop training |
| `bf16` | true | Mixed-precision training — uses 16-bit floats where safe, 32-bit where needed. Faster and half the memory of full FP32 |
| `gradient_checkpointing` | false | Off because we have enough memory; would trade compute for memory |
| `report_to` | none | We don't report to Weights & Biases or Tensorboard; everything lands as JSON |

**Loss function.** Cross-entropy over the 17-way softmax. For Run D (`weighted-CE`), we additionally weight each example's loss by `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` — meaning rare classes get up-weighted. (Run D didn't help; details in § 4.3.)

**Evaluation cadence.** Once per epoch, we run the model on the validation set, compute val macro-F1, and save the model if val macro-F1 improved. After 5 epochs (or earlier if early-stopping fired), we load the best checkpoint and evaluate on the held-out test set.

**Hardware.** A100 80 GB GPU, sliced as a MIG instance (one student gets a fraction of the card). Each run took 2–8 minutes wall-clock. Total compute across all 4 runs (Runs A through D, including 3 seeds for A): 12 minutes 44 seconds.

### B.8 Metrics — what macro-F1 actually means

For each of the 17 ingroups, we count:
- **TP** (true positive) — the model predicted this ingroup and gold was this ingroup
- **FP** (false positive) — the model predicted this ingroup and gold was *not* this ingroup
- **FN** (false negative) — gold was this ingroup but the model predicted otherwise

For each class:
- **Precision** = TP / (TP + FP) — when the model said "racist", how often was it right?
- **Recall** = TP / (TP + FN) — of all the actually-racist rows, how many did the model catch?
- **F1** = 2 · (Precision · Recall) / (Precision + Recall) — the harmonic mean of precision and recall, on a 0-to-1 scale

**Macro-F1** = unweighted average of per-class F1, over the classes actually present in `y_true ∪ y_pred`. Crucially, we do *not* pad to a hypothetical 17-class label space — that was a methodological bug we corrected. Padding artificially inflates the denominator with classes that have zero support in test, dragging the macro down.

**Weighted-F1** = average of per-class F1 weighted by each class's support (number of test rows). It's dominated by the largest classes.

**Macro-F1 is our primary metric** because it gives equal weight to common and rare ingroups. A model that perfectly handles `racist` (the most-common class) and totally fails everywhere else will score high on weighted-F1 but get penalised on macro-F1. The grouped split's prior shift on `anti-liberal` makes macro and weighted disagree informatively (see § 4.2).

#### B.8.1 Reproducibility note on the y_true∪y_pred macro-F1

The "no-padding" macro-F1 has a subtle fragility: it averages over the **set of classes the model actually emits at least one prediction on**, plus the classes present in test. A small training-time perturbation (different MIG slice timing, library patch, eval-order shuffle) can flip one rare-class softmax above/below the argmax threshold for a single test example, **adding that class to the divisor of the macro average without changing the per-class F1 of any other class**. We saw this concretely during the May 4 reproducibility re-run (`docs/reproduction_report.md` § 4.1): seed 7 of Run A reproduced with all per-class F1 values within ±0.02 of the original, but its macro-F1 dropped 6.4 pp (0.347 → 0.283) purely because the reproduced model emitted predictions on two extra zero-support classes (`anti-Asian`, `anti-LGBTQ`) that the original model never emitted, expanding the macro divisor from 9 → 11. Aggregated over the 3-seed sweep, this propagated to a 2.3 pp drift on the headline mean (0.353 → 0.329) — measurement-shift, not learning-curve drift. **The result is reproducible at the per-class level; the headline number is reproducible to about ±3 pp at the seed level due to the metric's set-of-emitted-classes dependence.** A fully-padded macro-F1 (averaging over all 17 ingroups regardless of emission) would be bit-identical between runs at the same seed but would re-introduce the original "0-support drag" we corrected for. We accept the trade-off and flag the fragility here.

---

## 3 · Headline numbers

| Run | split | input | seeds | macro-F1 | weighted-F1 | accuracy |
|---|---|---|---|---|---|---|
| **A** | **grouped (root-stratified)** | term | 3 | **0.353 ± 0.007** | 0.420 ± 0.012 | 0.478 ± 0.009 |
| B | grouped | text-only | 1 | 0.314 | 0.447 | 0.493 |
| **C** | **alt (random, ingroup-stratified)** | term | 1 | **0.996** | 0.999 | 0.999 |
| D | grouped | term + weighted-CE | 1 | 0.335 | 0.360 | 0.450 |

**The headline for the writeup is Run A: macro-F1 = 0.353 ± 0.007.** Run C's 0.996 is a sensitivity check, not a headline result — see § 5 for why.

### 3.1 Per-seed breakdown (Run A)

| seed | macro-F1 | weighted-F1 | acc | best epoch |
|---|---|---|---|---|
| 42 | 0.3491 | 0.4177 | 0.4716 | 1 (stopped epoch 3) |
| 123 | 0.3626 | 0.4353 | 0.4912 | 5 (budget-exhausted, still improving) |
| 7 | 0.3468 | 0.4060 | 0.4711 | 2 (stopped epoch 4) |
| **mean ± std** | **0.353 ± 0.007** | **0.420 ± 0.012** | **0.478 ± 0.009** | — |

**Caveat.** Seed 123 was still improving when we hit the 5-epoch budget cap; given the small std across seeds (0.007), the headline is *if anything* a slight underestimate. Decision: accept current numbers, document the early-stopping pattern.

### 3.2 Run C per-seed (alt-split)

Single seed (42): macro-F1 = 0.9964, weighted-F1 = 0.9985, accuracy = 0.9985.
Per-class F1 = 1.0000 on 12 of 18 classes, ≥ 0.9697 on every other class.

---

## 4 · Per-class breakdown — grouped split (Run A, mean over 3 seeds)

| ingroup | test support | F1 (mean ± std) | comment |
|---|---|---|---|
| anti-liberal | 720 | **0.000 ± 0.000** | **Total failure.** All 720 test rows use terms the model has never seen in training (terms ⊂ held-out roots). The model has no lookup to fall back on. |
| racist | 422 | 0.882 ± 0.013 | Strong head-class learning. |
| antisemitic | 298 | 0.623 ± 0.005 | Stable across seeds. |
| white supremacist | 273 | 0.250 ± 0.029 | Substantially worse than `racist`/`antisemitic` despite similar support — overlapping roots and discourse style. |
| transphobic | 256 | 0.689 ± 0.186 | High variance — one seed achieves 0.86, another 0.42. |
| Islamophobic | 117 | 0.648 ± 0.105 | |
| conservative | 9 | 0.083 ± 0.020 | Trivially small support; not an interpretable result. |

**The `anti-liberal` F1 = 0 result is the single most important diagnostic for RQ-B.** It accounts for ~34% of the test set, so a model that scores 0 on it is conceding a third of its evaluation. The grouped-split structure is the directly observable cause: 720 of 720 `anti-liberal` test rows use a `dog_whistle` term that does not appear *anywhere* in the train set (terms are nested inside held-out roots). The model has no opportunity to learn an `anti-liberal` association for any of those terms; it can only fall back on contextual features, which evidently capture ~no `anti-liberal`-specific signal. Combined with strong head-class scores (e.g. `racist` at 0.88, where the model has seen related-but-distinct racist roots in training), this confirms the model's only reliable ingroup signal *is* the term itself.

`text_only` (Run B) recovers some `anti-liberal` (F1 = 0.16, macro avg = 0.31) — interpretable as: when the model can't anchor on the matched term, it has to use surrounding context, which contains *some* signal even for unseen roots. This trade-off — text-only loses macro-F1 by trading it for a non-zero floor on the dominant test class — is the most defensible reading of the A vs B comparison.

For the alt-split (Run C), per-class F1 is essentially 1.0 across all 18 classes, including very small classes (`misogynistic` n = 2 → F1 = 1.0). This is what "near-trivial when roots leak" looks like.

---

## 5 · Interpretation — why the gap is the result

### 5.1 The grouped vs alt-split gap

Run A: **0.353** macro-F1. Run C: **0.996**. Same model architecture, same hyperparameters, same input format, same dataset rows. The only difference is whether each `dog_whistle_root` is held out from train.

The 64-pp collapse has a precise structural explanation. By construction of the Allen AI glossary, **every dog-whistle term in this dataset maps to exactly one ingroup**:

| dataset property | value |
|---|---|
| Distinct `dog_whistle` terms in train (Run A) | 676 |
| Of those, terms with >1 ingroup label | **0** (every term is unambiguous) |

So `ingroup` is a deterministic function of `dog_whistle` in the training data. The model's task in Run C is therefore "if I have seen this term in training, predict the ingroup I learned for it; otherwise guess from context." The two splits make that "if I have seen this term" prerequisite either trivially true or systematically false:

| split | test rows with a `(term, ingroup)` combo seen in train | test rows whose `dog_whistle` term appears in train at all |
|---|---|---|
| **Grouped (Run A)** | **0 / 2 095** (0 %) | **0 / 2 095** (0 %) — by design, every test root is held out |
| **Alt (Run C)** | 1 934 / 1 947 (99 %) | 1 935 / 1 947 (99 %) |

In Run C, 99 % of test terms appear in train (median 21 train rows per test `(term, ingroup)` combo, up to 315 rows for the most common). The alt-split task collapses to "read the lookup table off the train data." 99.6 % macro-F1 is not the model doing semantic ingroup inference — it's the model reading back the glossary it was trained on.

In Run A, **every** test row's dog-whistle term is novel from the model's perspective. There is no lookup to fall back on. This is the directly observable reason for the F1 = 0 on `anti-liberal`: 720 of 720 `anti-liberal` test rows use roots — and therefore terms — that the model has never seen during training. The model has never been told that any of those terms are `anti-liberal`, and there is no general "anti-liberal-ness" feature it could have learned because all 525 `anti-liberal` *training* rows use phrases that share a root structure but no surface similarity to the test phrases.

### 5.2 What this means for the writeup

1. **The result is the gap, not the alt-split number.** The 0.996 number is uninteresting on its own — the dataset's structural determinism guarantees a high score whenever roots are allowed to leak. We report it only to show the upper bound, and to make explicit how much of the apparent difficulty comes from the partition rather than the prediction.
2. **For deployment on *novel* dog whistles**, the grouped-split number (0.353) is the only one that matters. If new dog-whistle phrases emerge after training — the realistic case — the model has no lookup and falls back on contextual features that yield ~0.35 macro-F1. That is far above chance (1/17 ≈ 0.06) but far below the in-distribution lookup performance.
3. **The Allen AI glossary's term → ingroup determinism is itself the right thing to highlight in the data section.** Any RQ-B-style multiclass setup on `silent_signals` will have this property, and any future evaluation that reports >0.9 macro-F1 without holding out roots is measuring the glossary, not the model.

### 5.3 Cross-fold content duplication in the alt-split (disclosure)

The alt-split is built by pooling the grouped train+val+test rows and re-splitting on rows. The pool itself contains a small number of near-duplicate `content` strings (9 cross-fold pairs after canonicalising whitespace and case): 2 train↔test, 6 train↔val, 1 val↔test, on a total of 12,923 rows (≈0.07 %). This is a property of the underlying corpus (the same Reddit comment surfaced multiple times in `silent_signals`), not of the splitter. It is far too small to explain the 99.6 % macro-F1 — the term lookup explanation above does. Disclosed for completeness; not a methodological issue.

### 5.4 Text+term vs text-only

Surface intuition was "matched term should help". Empirically: term improves macro-F1 (+3.9 pp) but loses on weighted-F1 (−2.7 pp) and accuracy (−1.5 pp) on the prior-shifted test set. Reason: with the matched term as input, the model leans *more* on the (term, ingroup) lookup — boosting tail classes whose terms it has seen, at the cost of the dominant `anti-liberal` class (whose roots are novel). Text-only forces the model to use context for everyone, which yields worse tail performance but slightly better generalisation on the dominant unseen-root class.

Macro-F1 stays the primary metric, so the headline is "term beats text-only on the right metric, but the gap is small and the trade-off is interpretable."

### 5.5 Weighted-CE doesn't help

Run D underperforms Run A on every metric. The diagnostic explanation is that `sklearn`-balanced weights compute `w_c = N / (K · count_c)` with no flooring; tail classes with 1 or 18 train rows get weights of 533× and 30× respectively, which is more noise than signal. We could revisit this with a clipped or effective-number weighting scheme, but the cost is a more complicated story for marginal expected gain — the *correct* lesson here is that the imbalance is a reporting problem, not a loss-function problem.

---

## 6 · "Is this a problem with how we trained the model and split the data?"

Direct answer: **no, this is the methodology working correctly. But the result is a hard, honest fact about the difficulty of the underlying task — not a model that's broken in some fixable way.**

Let's separate three things:

### 6.1 Is the split methodology a bug?

**No.** Grouping by root is the correct way to evaluate generalisation to novel dog-whistles. The alternative (random or content-level split) lets the same dog-whistle phrase appear in both train and test, which means the model can succeed by memorising the corpus's deterministic glossary rather than by learning anything generalisable. We ran *both* splits intentionally to make this visible — Run A is the methodologically correct setting, Run C is the leaky control.

### 6.2 Is the model architecture a bug?

**No.** RoBERTa-base is a standard, well-understood backbone for sentence classification at this scale. The loss function (cross-entropy), the optimiser (AdamW with warmup), the hyperparameters (canonical for RoBERTa fine-tuning) — all of these are textbook. We didn't make a weird choice that broke something. The `text_only` ablation (Run B) further shows the model is doing *something* sensible — it just has a hard upper bound on what's recoverable from context alone.

### 6.3 So what's the limitation?

**The task is intrinsically hard, and the data structure makes it harder.**

- **Intrinsic hardness.** Predicting the targeted ingroup of a dog-whistle from a Reddit comment is hard for humans too. Many comments are short, ironic, or rely on unstated context. Without recognising the specific phrase, the contextual signal is often weak.
- **Data structure.** The Allen AI glossary is one-to-one (term → ingroup). When you hold out roots, you remove the model's only reliable signal. What's left is contextual generalisation — and 35% macro-F1 (vs ~6% chance) suggests contextual generalisation is real but limited.
- **Anti-liberal specifically.** The `anti-liberal` train rows use phrases like *"woke"*, *"social justice warrior"*, *"snowflake"*. The `anti-liberal` test rows use entirely different phrases — held-out roots like *"defund the police"*, *"basement dweller"*, etc. (See per-root breakdown analysis in the deferred list § 8.) The model is being asked to detect anti-liberalism from sentences that don't share *any* anchor terms with what it learned anti-liberalism looks like. F1 = 0 is the honest result; reporting anything else would be wrong.

### 6.4 What WOULD be a methodology bug?

Things we would have done wrong but didn't:

- **Reporting only Run C (0.996) as the headline.** That would mislead readers into thinking the model generalises. We don't.
- **Using a random split as the primary evaluation.** Same problem. We use grouped-by-root.
- **Counting `misogynistic` (10 val rows, 0 train rows) in macro-F1 calculation.** We dropped them; we'd otherwise be averaging in a class with no support, dragging the metric down for irrelevant reasons.
- **Padding macro-F1 to 17 classes when the test set only has 7.** We compute over the labels actually present in `y_true ∪ y_pred`, which is the sklearn default and the right call.

### 6.5 What WOULD be a meaningful improvement?

A sharper marker would push back on a few choices that aren't *bugs* but might be tweaked:

- **More training epochs.** Seed 123 was still improving when training stopped at 5 epochs. Re-running at `epochs=10` is a natural follow-up. We left it at 5 because std = 0.007 across 3 seeds suggests the macro-F1 is unlikely to move much; this is a possible follow-up rather than something we'd call wrong.
- **Different label space.** If we collapsed the 17 ingroups into a smaller hierarchy (e.g. {anti-minority, anti-political, anti-gender, ...}), the task would get easier, and `anti-liberal` would no longer be a unique class. We chose 17 because that's the corpus's annotation scheme; collapsing the labels feels like cooking the books.
- **Larger model.** We started with `microsoft/deberta-v3-large` (435 M params, more capable) but hit numerical instability and switched to RoBERTa-base. Could fix the numerics and try again — outside scope for this iteration.
- **Bring back `dog_whistle_root` as input feature.** Currently we feed `dog_whistle` (the surface form) into the input. We could also feed the *root*. But for held-out test roots, this is exactly the field that's novel — so it would only help if we had some kind of clustering / nearest-neighbour fallback over roots, which is a more complicated system to build and defend.

### 6.6 Calibration — the head is overconfident, fixed at inference

A separate finding surfaced from a demo failure case ("Saw a 109 reference on Twitter today, sad stuff" — RQB predicted `white supremacist` with confidence 1.00; the gold ingroup is `antisemitic`). Pulled the per-row probabilities for the headline variant (`format_term · seed_123`) on the held-out test set and checked confidence-vs-accuracy:

| confidence threshold | n predictions | accuracy at this threshold |
|---|---|---|
| > 0.50 | 1985 / 2095 | 0.51 |
| > 0.75 | 1741 | 0.58 |
| > 0.90 | 1570 | 0.63 |
| > 0.95 | 1405 | 0.69 |
| > 0.99 | 1171 | 0.78 |

Median predicted confidence on the test set is **0.996**; 56 % of predictions sit above 0.99 — but the model is right ~78 % of the time at that confidence level. The head is severely overconfident. The `109` demo failure is not a labelling bug (the dataset–glossary alignment is 0/12 923 mismatches, and `109`'s 8 corpus rows are all correctly tagged `antisemitic`); it is the modal failure mode of an overconfident multiclass softmax meeting an out-of-training-vocabulary phrase.

**Post-hoc fix.** A single temperature `T` was fitted on val by minimising NLL (`scipy.optimize.minimize_scalar`, bounds `[0.05, 50]`), giving **T\* = 2.26** with NLL dropping from 2.33 → 1.53. Argmax is preserved for any `T > 0`, so the fitted-model accuracy is unchanged; only the reported confidences move. After scaling, median confidence drops from 0.997 → 0.750, and zero predictions land above 0.99. The temperature is recorded in `data/manifests/rqb_temperature.json` and `model_inventory.json`, and applied at inference in the demo's `app/pipeline.py`. Same check on RQA showed it is well-calibrated already (median conf 0.75, accuracy at conf > 0.95 is 0.89), so RQA is left at `T = 1`.

This calibration result is independent of the term-determinism finding. It says only that the trained head's softmax does not produce trustable confidences out of the box; it does not change the underlying RQ-B numbers.

### 6.7 The bottom line

We have a well-trained, well-evaluated, honestly-low-scoring classifier whose limitations come from the difficulty of the task and the structure of the data, not from anything fixable in our pipeline. The Run C contrast (0.996) makes this visible: when the model is allowed to memorise, it does — perfectly. When forced to generalise, it cannot. **That's the finding.**

This is closer to a *negative result* than a triumph. Negative results are publishable (and educational) when the negative is the result of asking the right question of the data, which is what we did.

---

## 7 · Files

```
results/multiclass/
├── format_term/                           # Run A (3 seeds)
│   ├── aggregated_results.json
│   └── seed_{42,123,7}/
│       ├── label_map.json                 # 17-class id2label + drop_report
│       ├── test_results.json              # macro-F1, weighted-F1, per-class
│       ├── best_model/                    # final checkpoint (loaded for inference)
│       └── predictions/test_preds.parquet # per-row predictions for error analysis
├── format_text_only/                      # Run B (single seed)
├── format_term_altsplit/                  # Run C (single seed)
└── format_term_weighted/                  # Run D (single seed)
```

Source files (training and split construction):

- `scripts/hpc/train_multiclass_ingroup.py` — main training script
- `scripts/configs/multiclass.yaml` — hyperparameter config
- `scripts/hpc/submit_multiclass.sh`, `submit_multiclass_alt.sh`, `submit_multiclass_weighted.sh` — SLURM submitters
- `scripts/hpc/build_alt_split_rq_b.py` — alt-split construction
- `data/manifests/split_manifest.json` — root-to-split assignment

Session notes: `docs/sessions/2026-04-29_rqb_kickoff.md`.

---

## 8 · Outstanding error analyses

Per-row predictions are saved at `results/multiclass/format_*/seed_*/predictions/test_preds.parquet` (with columns `dog_whistle`, `dog_whistle_root`, `ingroup` (true), `pred_label`, `pred_prob`, `correct`). The artefacts are in place; what's missing is the analysis code:

1. **Confusion matrices, per arm, per seed.** Row-normalised on test. Especially: where do `anti-liberal` examples actually go in Run A — spread evenly, or concentrated on one or two wrong classes?
2. **Per-root accuracy** (grouped split). Some roots will be effectively zero — list them, group by ingroup. Most direct read on what's "easy" vs "impossible".
3. **High-confidence wrong predictions.** Top-N by `pred_prob` among `correct == 0`. Surfaces where the model is *confident* in the wrong class.
4. **Term-level overlap with RQ-A.** Compare misclassified rows in RQ-A's coded-vs-literal sense to misclassified rows in RQ-B's ingroup task — does the model fail on the same roots in both?
5. **Did `text_only` actually help on `anti-liberal`?** Cross-tabulate term vs text_only predictions on the same test rows; agreement / disagreement matrices.
6. **Variance investigation on `transphobic`** (Run A std = 0.186 across 3 seeds — much larger than every other class's std). Likely a small handful of high-leverage roots; identify them.
7. **Macro-F1 vs train-frequency scatter.** Plot per-class F1 against log train count; quantify the relationship.

---

## 9 · Glossary

For readers new to the project / NLP fine-tuning vocabulary.

- **Backbone / pre-trained model.** A large model (here RoBERTa-base) that's been trained on lots of generic text before our task. We start from its weights rather than from scratch.
- **bf16 (bfloat16).** A 16-bit floating-point format that gives faster training than fp32 with less memory, while keeping enough precision for stable optimisation on modern GPUs (A100 supports it natively).
- **Cross-entropy loss.** The standard loss function for classification — measures how far the model's predicted probability distribution is from the gold one-hot label. Lower is better.
- **Dog-whistle.** A coded phrase that signals an in-group meaning to those who recognise it, while looking innocuous to outsiders. Example: `"absent fathers"` literally describes parents who aren't present, but is used as a coded racist phrase about Black communities.
- **Early stopping.** Training trick: if the validation metric doesn't improve for `patience` epochs in a row, stop and use the best-so-far checkpoint. Prevents overfitting and saves compute.
- **Fine-tuning.** Continuing to train a pre-trained model on a new task / dataset, updating its weights for the new objective.
- **Grouped split.** A train/val/test partition where rows sharing a key attribute (here, `dog_whistle_root`) are kept together. Tests generalisation to *unseen* groups.
- **Ingroup.** In sociology, the group that uses the dog-whistle to signal allegiance. Here it's the **target** of the dog-whistle ("racist" = dog-whistles aimed at racial out-groups; "antisemitic" = at Jewish people; etc.).
- **Macro-F1.** Average of per-class F1 scores, equally weighting every class. Penalises models that ignore minority classes.
- **One-hot label.** A vector with all zeros except one 1, indicating the gold class.
- **Root.** The canonical dictionary form of a dog-whistle phrase. Multiple surface forms (e.g. "absent fathers", "absentee fathers") share one root ("absent fathers").
- **RoBERTa.** A transformer language model from Liu et al. 2019 (Facebook AI), descendant of BERT. Pre-trained on 160 GB of English text. Comes in `base` (125 M params) and `large` (355 M) sizes; we use base.
- **Seed.** Random number that initialises everything random in training (weights, dropout, data shuffle order). Different seeds → slightly different models. Reporting mean ± std across seeds quantifies how much that randomness matters.
- **Softmax.** A function that takes 17 raw scores from the model and converts them into a probability distribution over the 17 classes (positive numbers summing to 1).
- **Tokenisation.** The process of splitting text into pieces the model sees as discrete inputs. RoBERTa uses a 50,000-token vocabulary built from byte-pair encoding.
- **Weighted-F1.** Per-class F1, weighted by class support. Dominated by majority classes — useful as a sanity check but not our primary metric here.
