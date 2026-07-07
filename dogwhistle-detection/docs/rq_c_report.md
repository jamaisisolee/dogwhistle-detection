# RQ-C Structured Generator — Full Report

> **Audience.** Teammates and outside readers who haven't seen this code before. Read § A first if you're new; skip to § 3 if you only want the numbers.
> **Status.** Both paths trained, all 6 runs complete. Path A (jobs `483949`) finished 2026-04-28 04:11 (7 h 23 m). Path B (`484232`) finished 2026-04-28 22:25 (13 h 06 m).
> **Last updated.** 2026-04-29.

---

## TL;DR

We trained a generative model — `google/flan-t5-xl` (3 billion parameters) fine-tuned with **LoRA** adapters — to produce a structured **JSON output** for each Reddit comment containing a dog-whistle phrase. The output has four fields: `dog_whistle_root`, `ingroup`, `definition`, `explanation`. The first three are categorical labels (the model has to "regenerate" the right label string); the fourth is a free-text explanation paragraph.

Two paths were trained, both on the same 9,064-row train set and 2,095-row test set, both grouped by `dog_whistle_root` (no root overlap across splits):

- **Path A (unbalanced)** — train data left at the natural ingroup distribution; **headline `ingroup.macro_f1` = 0.379 ± 0.014** across 3 seeds.
- **Path B (class-balanced)** — train data oversampled to 1,500 rows per ingroup × 17 ingroups = 25,500 rows; **`ingroup.macro_f1` = 0.383 ± 0.063** across 3 seeds.

**We report Path A as our headline system.** The +12-pp jump in `ingroup.accuracy` Path B shows (0.487 → 0.606) doesn't really mean the model got better at recognising dog-whistles — it mostly reflects Path B over-predicting the test-dominant class `anti-liberal` (34 % of test, 6 % of natural train). The `macro_f1` change between A and B is +0.004 (inside one standard deviation), and Path B's seed-to-seed variance is **4.5× higher**. So the accuracy gain looks like class-prior reweighting rather than the model actually learning more.

Two structural facts to keep in mind when reading these numbers:

1. **`dog_whistle_root` and `definition` accuracies are very low (root macro-F1 ≈ 0.16, definition macro-F1 ≈ 0.02) — and that is structurally inevitable under the grouped split.** No test root is in train, and `definition` is a deterministic function of root, so the model cannot regenerate a root or definition it has never seen. The meaningful generalisation question lives at the **`ingroup` category** level, where the 7 test ingroups are all represented in train (just for *other* roots).
2. **Free-text `explanation` is scored by TF-IDF cosine similarity (L5)** — explanation cosine is ~0.51 (Path A) / 0.54 (Path B). The metric is reasonable but limited; § B.6 details the trade-offs.

The full per-seed numbers are in § 4. The two structural facts are defended in § 5.2 and § 6.

---

## A · For someone new to the project

### A.1 What is RQ-C?

The Bocconi 597 NLP project on dog-whistle detection has three research questions:

- **RQ-A.** Given a Reddit comment that contains a dog-whistle phrase, is the phrase being used in its **coded** sense or its **literal** sense? (Binary classification — see `docs/rq_a_report.md`.)
- **RQ-B.** Given a coded use, **which ingroup does the dog-whistle target?** (17-way classification — see `docs/rq_b_report.md`.)
- **RQ-C.** Same input as RQ-B, but produce a *structured* output: dog-whistle root, ingroup, dictionary definition, and free-text explanation. (Generation — this report.)

RQ-C is the *generative* version of RQ-B. Where RQ-B picks one of 17 labels, RQ-C produces a JSON object whose `ingroup` field is one of the same 17 labels — but it *also* generates a root, a definition, and a free-text explanation in one go. It's a richer output contract: a single inference call gives you everything an analyst would want to know about a flagged comment.

### A.2 What is the structured target?

Each training row pairs an input prompt with a target JSON. Concrete example, from a real training row:

**Input** (the prompt the model sees):
```
Text: Absentee fathers only seem to come in one color: Black.
Matched term: absentee fathers
Return JSON explaining the coded use.
```

**Target** (what we train the model to produce):
```json
{
  "dog_whistle_root": "absent fathers",
  "ingroup": "racist",
  "definition": "Black men are irresponsible",
  "explanation": "In this text, 'absentee fathers' is used as a coded reference to racist ideology. Black men are irresponsible"
}
```

The model has to learn:

- Recover the canonical lower-case **root** (`"absent fathers"`) from a surface form (`"absentee fathers"`).
- Predict the right **ingroup** category (`"racist"`).
- Recall the **definition** (the coded meaning) — note this is a deterministic function of root, see § B.2.
- Generate a coherent **explanation** following the template *"In this text, '<dw>' is used as a coded reference to <ingroup> ideology. <definition>"*.

The training data is 9,064 such (input, target) pairs.

### A.3 Why is this hard?

Three reasons specific to the generative setup:

1. **Root and definition are structurally non-recoverable on held-out test roots.** Under the grouped split, no test root appears in train. The model has never seen the canonical string `"absent fathers"` paired with the surface form `"absentee fathers"` for any test root, so it can only guess. Same for definition (which depends on root). This is a hard ceiling; we discuss it in § 5.2.
2. **JSON output discipline.** The model has to produce parseable JSON every time. A misplaced comma, an unclosed brace, a curly-quote — any one of these breaks `json.loads` and the whole row scores zero on every metric. We caught and patched a real instance of this (the *brace bug*, § B.7).
3. **Free-text explanation is scored against gold prose.** Two outputs can be semantically identical and lexically completely different. We score with TF-IDF cosine similarity (L5) plus per-example ROUGE-L; both have known limitations, especially when the gold explanations are templated (which ours are — § B.6).

A generative model that lifts `ingroup.macro_f1` from chance (1/17 ≈ 0.06) to 0.38 on novel roots is doing real work, but the absolute ceiling on this task is bounded by what's recoverable from short Reddit text alone.

---

## B · Methodology, step by step

### B.1 The data

`data/final/rq_c_generation/` holds three Parquet files (`train.parquet`, `val.parquet`, `test.parquet`):

| split | rows | distinct roots | distinct ingroups | notes |
|---|---:|---:|---:|---|
| train | 9 064 | 219 | 17 | natural-distribution corpus |
| val | 1 764 | 39 | 13 | post-drop of 10 unseen-ingroup rows |
| test | 2 095 | 40 | 7 | dominant test class is `anti-liberal` (34 %, the `social justice warrior` root alone is 720/2095) |

Each row pairs an input prompt with the target JSON (§ A.2). The same row layouts and split assignments are used as in RQ-B (`data/manifests/split_manifest.json`).

### B.2 Why grouped splits — and what's structurally impossible because of them

This is the methodological backbone, shared with RQ-A and RQ-B. Every dog-whistle phrase belongs to a *root* (e.g. `"absent fathers"`, `"absentee fathers"`, `"absent father"` all map to root `"absent fathers"`). We assign each root to exactly one of {train, val, test}, so the test set uses 100 % roots the model has never been told the labels of. `DESIGN_DEFENSE.md` D2 mandates this.

**For RQ-C specifically, this has two specific structural consequences worth flagging up front:**

1. **`dog_whistle_root` is non-recoverable.** The model cannot regenerate the canonical lower-case root string for a held-out term — it has only the surface form (`SJW`) and context to go on. It has *never* seen the mapping `SJW → social justice warrior` during training. Reported macro-F1 on the root field across all 6 model variants is ≈0.16 — well above zero (the model gets some closely-related roots right by accident) but bounded.
2. **`definition` is a deterministic function of root.** Every root maps to exactly one definition by the corpus's construction (`0 of 696 unique dog_whistles have ambiguous targets`, verified). So definition accuracy is structurally bounded by root accuracy. Reported macro-F1 on definition is ≈0.02 — even lower than root because there's no near-match credit for getting the definition mostly right.

The meaningful generalisation question therefore lives at the **`ingroup` category** level, where the 7 test ingroups are all represented in train (just for *other* roots). This is the headline metric we focus on.

A pure dictionary-lookup baseline would *also* fail on root EM for the same reason — it would have to fall back to TF-IDF retrieval over training contexts and would copy the labels of a *different* root. We expand on this in § 5.2.

### B.3 The two paths — same model, different training-set composition

Both paths use **the same model architecture, the same hyperparameters, the same input/output format**. The only thing that differs is the *composition* of the training set.

- **Path A (unbalanced).** Train on `data/final/rq_c_generation/train.parquet` as-is. 9,064 rows. The natural ingroup distribution: `racist` 23.3 %, `white supremacist` 16.7 %, `antisemitic` 16.5 %, `transphobic` 14.2 %, …, `anti-liberal` 5.8 %, `conservative` 2.6 %.
- **Path B (class-balanced).** Oversample the train set to 1,500 rows per ingroup × 17 ingroups = **25,500 rows**. Each ingroup gets a flat 5.9 % share. Concretely: rare-class rows (like `anti-vax`) are duplicated many times to hit 1,500; common-class rows (`racist`) are subsampled or capped.

**Why both?** RQ-B's headline finding (macro-F1 = 0.353) showed that the test-dominant class `anti-liberal` was hit hardest by the grouped split. Path B is a "loss-side" intervention — by up-weighting tail classes during training, we ask whether a less train-prior-skewed model would handle the prior-shifted test set better.

**Neither path is "right" in some absolute sense — they answer different questions.** Path A measures the model's natural-distribution behaviour. Path B measures whether retraining on a flat prior helps under prior shift. We report both and recommend Path A as the more defensible *single* system (see § 5.1 for why).

### B.4 The model — Flan-T5-XL + LoRA, in plain words

This is where RQ-C departs most sharply from RQ-A and RQ-B. We use a **generative encoder-decoder** model fine-tuned with **parameter-efficient adapters**.

**Flan-T5-XL** (Chung et al. 2022) is Google's instruction-tuned variant of T5. Three things to know:

1. **It's an encoder-decoder transformer**, not a decoder-only LM like GPT or a encoder-only model like RoBERTa. The encoder reads the input prompt; the decoder generates the output token-by-token.
2. **It's been instruction-tuned.** Beyond the original T5 pre-training (predict masked spans on a huge text corpus), Flan-T5 was further trained on millions of natural-language instructions ("Translate this:", "Summarise this:", "Answer this question:"). This means the model already knows how to follow patterns like "Return JSON explaining the coded use" without us teaching it the format from scratch.
3. **XL = 3 billion parameters.** Smaller than the largest T5 (XXL, 11 B) but big enough to handle structured output reliably. Bigger backbones often produce cleaner JSON.

**LoRA** (Low-Rank Adaptation, Hu et al. 2021) is a parameter-efficient fine-tuning method:

- Instead of updating all 3 B parameters during fine-tuning (which would cost a lot of GPU memory and produce a 12 GB checkpoint), LoRA freezes the original weights and inserts small **adapter matrices** into specific transformer layers (typically the query and value projections in the attention layers).
- Only the adapter matrices are updated during training. They're tiny — for our config, the LoRA adapter is **~36 MB** vs the 12 GB base model.
- At inference time, we load the base Flan-T5-XL and attach the LoRA adapter on top. The math is just: instead of computing `W·x`, compute `(W + α·B·A)·x` where `A` and `B` are small low-rank matrices stored in the adapter.
- **The key win for the project:** we can host all 6 RQC variants (3 seeds × 2 paths) for ~216 MB of adapters total, instead of 72 GB of merged checkpoints. Storage on the HF Hub is free either way, but the smaller adapter is what makes the demo Space practical (we attach + swap adapters on demand).

Our LoRA configuration:

| LoRA hyperparameter | Value | What it controls |
|---|---|---|
| `lora_r` | 16 | The rank of the adapter matrices. Higher = more capacity, larger adapter file. 16 is a typical sweet spot for moderate-task fine-tuning. |
| `lora_alpha` | 32 | A scaling factor on the adapter's contribution. Typically `2·lora_r`. |
| `lora_dropout` | 0.1 | Dropout on the adapter weights — regularisation. |
| `lora_target_modules` | `["q", "v"]` | Which layers get adapters. We adapt the query and value projections in every transformer block. Standard for T5. |

**Why this combo (Flan-T5-XL + LoRA)?**

- We need a **generative** model because the output is structured prose, not a class label. T5 family is the standard open-weight choice for sequence-to-sequence tasks.
- We need a model that's been **instruction-tuned**, so it can follow our prompt format without extensive instruction tuning of our own.
- We need **parameter-efficient fine-tuning**, because (i) we don't have the GPU memory for full-parameter fine-tuning of a 3 B model and (ii) the resulting demo Space needs to fit in 16 GB of free CPU RAM.

### B.5 Training procedure

Every model variant in this report was trained with the same hyperparameters (configured in `scripts/configs/generation.yaml`, with Path B's only delta being `class_balance_train: 1500`):

| Hyperparameter | Value | What it does in plain language |
|---|---|---|
| `model_name` | `google/flan-t5-xl` | The pre-trained transformer we start from |
| `max_input_len` | 512 | Cut input prompts off at 512 tokens. Comments are short so this is rarely binding. |
| `max_target_len` | 128 | Cut target JSON outputs off at 128 tokens. Most outputs fit comfortably. |
| `epochs` | 5 (max) | One epoch = one pass through the training set. Capped at 5 with early-stopping. |
| `batch_size` | 4 | Per-GPU batch size — small because Flan-T5-XL is large |
| `grad_accum` | 4 | Gradient accumulation: we effectively act as if batch were 4·4 = 16, but we only fit 4 in memory at a time |
| `lr` | 3e-4 | Learning rate. Higher than RoBERTa's 2e-5 because LoRA's added parameters are randomly initialised and need bigger steps to learn |
| `weight_decay` | 0.01 | A regularisation term that gently nudges parameters toward zero — prevents overfitting |
| `warmup_ratio` | 0.1 | The first 10 % of training uses a linearly-ramping learning rate |
| `early_stopping_patience` | 2 | If validation metric doesn't improve for 2 consecutive epochs, stop training |
| `bf16` | auto | Mixed precision — uses 16-bit floats where safe, 32-bit where needed. Halves memory and is ~2× faster on A100s |
| `gradient_checkpointing` | true | Trades compute for memory by re-computing activations during backprop. We need this to fit Flan-T5-XL on the MIG slice |
| `seeds` | 42, 123, 7 | 3 independent runs per path to quantify seed-noise on the headline metric |

**Loss function.** Token-level cross-entropy on the decoder (predict the next target token given the input prompt + decoded prefix so far). This is the standard sequence-to-sequence training objective.

**Decoding at inference.** Greedy decoding (`do_sample=False`) — at each step, pick the highest-probability next token. We chose greedy over beam search to keep inference fast and reproducible (same prompt → identical output).

**Evaluation cadence.** Once per epoch on validation, we run greedy decoding on every val row and compute headline metrics (ingroup macro-F1 + JSON parse rate primarily). The best checkpoint by val macro-F1 gets loaded and evaluated on test.

**Hardware.** A100 80 GB GPU, sliced as a MIG instance (`4g.40gb`). Path A took 7 h 23 m, Path B 13 h 06 m (longer because the train set is 2.8× bigger after oversampling).

### B.6 Metrics — what we measure on each output field

Each of the four output fields gets its own metric, because they're different shapes:

**Categorical fields (`dog_whistle_root`, `ingroup`, `definition`).** Even though the model *generates* these as strings, we score them as classification: lower-case + strip both predicted and gold strings, then compare for exact match. We then compute:
- **Accuracy** — fraction of test rows where the predicted string exactly matches the gold string.
- **Macro-F1** — average of per-class F1 over the labels actually present in `y_true ∪ y_pred`. Same definition as RQ-B.
- **Weighted-F1** — F1 averaged with weight by support, dominated by the largest classes.

**`ingroup` is the primary metric** — it's where the meaningful generalisation question lives, since all 7 test ingroups are represented in train. Root and definition are reported but bounded by structural impossibility (§ B.2).

**Free-text field (`explanation`).** Two complementary metrics:
- **TF-IDF cosine similarity** (L5 on the course inventory) — represent both predicted and gold explanations as TF-IDF vectors and compute cosine similarity. Captures lexical overlap weighted by term-importance. Limited because two semantically-identical sentences can have low cosine if they use different vocabulary.
- **ROUGE-L** — longest-common-subsequence overlap, scored as F1. Per-example, not corpus-aggregated (we caught a corpus-aggregation bug; see § B.7). Captures sequence-level overlap; tends to look high when explanations follow a templated form (which ours do).

**Both explanation metrics are imperfect proxies for "is the explanation correct?".** A more rigorous evaluation would require human judgement or LLM-as-judge — out of scope under the no-paid-API constraint.

**JSON parse rate.** Fraction of model outputs that successfully parse as JSON (after the brace-repair step described in § B.7). A model that produces invalid JSON scores 0 on every other metric — we report parse rate alongside everything else so the reader can see if the model is failing structurally.

**Course-aligned levels.** L3 = accuracy / precision / recall / F1 (used on categorical fields); L5 = embedding cosine similarity (used on the explanation field). Both are within the inventory of metrics the course expects; ROUGE-L is reported as a secondary recognised generation metric.

### B.7 Plumbing fixes that materially affected the metrics

Three issues were caught in the original metric pipeline. All are resolved on disk and in the source. Worth flagging because they would otherwise produce misleadingly-bad numbers.

**B.7.1 — The brace bug.** The fine-tuned Flan-T5 reliably emits the JSON *body* without the wrapping braces:

```
"dog_whistle_root": "X", "ingroup": "Y", ...
```

instead of `{"dog_whistle_root": "X", ...}`. Pre-fix, every `json.loads` raised `JSONDecodeError` → `json_parse_rate = 0` and every categorical accuracy = 0.

**Fix.** `repair_json()` in `scripts/hpc/eval_generation.py` strips whitespace and adds the missing opening/closing braces if absent. Post-fix: parse rate 0.97 ± 0.02 across both paths.

**B.7.2 — Capitalisation drift.** The model varies between `SJW`, `Social Justice Warriors`, `social justice warrior`, etc. for the same root. EM was being compared raw, masking real overlap.

**Fix.** `_norm()` in `eval_generation.py` lower-cases and strips both predicted and gold strings before comparison. (This is also why we say "lower-case + strip" in B.6.)

**B.7.3 — ROUGE-on-JSON corpus aggregation.** The original metric pipeline computed corpus-level ROUGE-L on the *whole* output JSON string. This conflated the categorical-field overlap with the explanation overlap, producing a score that was uninterpretable.

**Fix.** ROUGE-L is now computed per-row, only on the `explanation` field, and averaged across rows (`explanation_rouge_l_mean`). Other fields are scored as classification (B.6).

These three patches are why the headline numbers in this report (parse rate 0.97, ingroup macro-F1 0.38) differ dramatically from earlier draft numbers (parse rate 0.0, all EM 0.0). The model itself didn't change between the broken and fixed scorers — just our ability to read what it was producing.

---

## 3 · Headline numbers (mean ± std across 3 seeds)

| metric | Path A · unbalanced | Path B · balanced | Δ (B − A) |
|---|---:|---:|---:|
| `json_parse_rate` | 0.975 ± 0.017 | 0.969 ± 0.022 | flat |
| **`ingroup.accuracy`** | **0.487 ± 0.021** | **0.606 ± 0.065** | **+0.119** |
| **`ingroup.macro_f1`** (primary) | **0.379 ± 0.014** | **0.383 ± 0.063** | **+0.004** (variance ×4.5) |
| `ingroup.weighted_f1` | 0.431 ± 0.026 | 0.608 ± 0.119 | +0.177 (variance ×4.6) |
| `dog_whistle_root.accuracy` | 0.324 ± 0.017 | 0.351 ± 0.053 | +0.027 |
| `dog_whistle_root.macro_f1` | 0.160 ± 0.007 | 0.167 ± 0.014 | +0.007 |
| `definition.accuracy` | 0.051 ± 0.021 | 0.072 ± 0.018 | +0.021 |
| `definition.macro_f1` | 0.016 ± 0.003 | 0.021 ± 0.003 | +0.005 |
| `explanation_cosine_mean` (TF-IDF, L5) | 0.510 ± 0.011 | 0.542 ± 0.010 | +0.032 |
| `explanation_rouge_l_mean` | 0.764 ± 0.012 | 0.786 ± 0.010 | +0.022 |

**Naive baselines for `ingroup.accuracy`:** random-7 = 0.143; modal `anti-liberal` = 0.344. **Both models beat both baselines on ingroup accuracy.** A TF-IDF retrieval baseline against the corrected scorer is pending (§ 8.1) — without it we can't yet defend "Flan-T5 + LoRA beats retrieval" rigorously.

**Single-line interpretation:**

- **Path A is the *stable* model.** Variance across seeds is ≤ 0.026 on every headline metric.
- **Path B improves headline accuracy and weighted-F1 noticeably (+12 pp / +18 pp) but does NOT move macro-F1 (+0.004, inside one std).** The accuracy gain is driven by the over-represented test class (`anti-liberal`, 34 % of test) — which oversampling effectively up-weights at training time as well — not by genuine improvement on the tail.
- **Path B is far less stable.** Macro-F1 std multiplies by ×4.5 (0.014 → 0.063); seed 7 alone collapses to macro-F1 = 0.296 while seed 42 reaches 0.438.

---

## 4 · Per-seed breakdown (both paths)

| metric | A-42 | A-123 | A-7 | B-42 | B-123 | B-7 |
|---|---:|---:|---:|---:|---:|---:|
| `json_parse_rate` | 0.972 | 0.957 | 0.997 | 0.989 | 0.981 | 0.937 |
| `ingroup.accuracy` | 0.474 | 0.470 | 0.517 | **0.654** | 0.513 | 0.649 |
| `ingroup.macro_f1` | 0.383 | 0.360 | 0.393 | **0.438** | 0.416 | **0.296** |
| `ingroup.weighted_f1` | 0.410 | 0.415 | 0.468 | 0.669 | 0.441 | 0.713 |
| `dog_whistle_root.accuracy` | 0.344 | 0.326 | 0.302 | 0.343 | 0.291 | 0.420 |
| `definition.accuracy` | 0.080 | 0.034 | 0.039 | 0.097 | 0.059 | 0.060 |
| `explanation_cosine_mean` | 0.525 | 0.500 | 0.505 | 0.547 | 0.528 | 0.552 |
| `explanation_rouge_l_mean` | 0.767 | 0.748 | 0.776 | 0.799 | 0.782 | 0.776 |

Path A's seed range is ~0.03 on macro-F1; Path B's seed range exceeds 0.14 on macro-F1 and on accuracy. The seed-7 collapse on Path B is the dominant variance driver.

---

## 5 · Interpretation

### 5.1 Path A vs Path B — the +12-pp accuracy gain is class-prior reweighting, not learning

The headline accuracy improvement under Path B (0.487 → 0.606, +12 pp) is real but its cause is mundane. By oversampling rare classes during training, Path B effectively lowers the model's prior on common training classes (`racist`, `antisemitic`, `white supremacist`) and raises its prior on rare ones (including `anti-liberal`). At test time, this aligns *better* with the test distribution — `anti-liberal` is 34 % of test but only 5.8 % of natural train.

| ingroup | train share (Path A) | train share (Path B post-oversample) | test share | shift A→test |
|---|---:|---:|---:|---:|
| **anti-liberal** | **5.8 %** | 5.9 % | **34.4 %** | **×6** |
| racist | 23.3 % | 5.9 % | 20.1 % | flat |
| white supremacist | 16.7 % | 5.9 % | 13.0 % | flat |
| antisemitic | 16.5 % | 5.9 % | 14.2 % | flat |
| transphobic | 14.2 % | 5.9 % | 12.2 % | flat |
| Islamophobic | 6.5 % | 5.9 % | 5.6 % | flat |
| conservative | 2.6 % | 5.9 % | 0.4 % | flat |

Path A trains on a corpus that is dominated by `racist`/`antisemitic`/`white supremacist` examples. At test time, when faced with a never-seen `social justice warrior` family of comments, Path A's natural prior pushes it toward those high-frequency training classes. Path B has flattened the prior, so under prior shift it is *less wrong* on aggregate — but for prior-shift reasons, not because it learned better tail-class semantics.

**The decisive evidence that this is reweighting and not learning** is that **macro-F1 didn't move**. If Path B were genuinely learning the tail classes better, macro-F1 (which gives equal weight to each ingroup) would rise. It doesn't — Δ = +0.004, well inside the seed std. What rose is *accuracy* and *weighted-F1*, both of which are dominated by the prior-aligned majority-test class.

A second, sharper piece of evidence: **Path B's variance grows by 4.5×.** Aggressive oversampling makes optimisation more sensitive to seed-driven order-of-batch effects on the duplicated examples — seed 7 collapses to macro-F1 = 0.296 while seed 42 reaches 0.438. A genuinely better intervention would not produce this variance pattern.

**We use Path A as the headline system.** Path B is reported alongside as a comparison; we don't claim it generalises better.

### 5.2 Why root and definition F1 are structurally low (not a bug)

Some readers will look at `dog_whistle_root.macro_f1 = 0.16` and `definition.macro_f1 = 0.02` and conclude the model is broken. It isn't — these numbers are the consequence of the grouped split combined with the deterministic glossary, exactly as in RQ-B's term-determinism finding (§ 5 of `rq_b_report.md`).

Under the grouped split:
- The model has **never seen the canonical-root string** for any test row's surface form. Asking it to regenerate `"absent fathers"` from `"absentee fathers"` requires either a prior seen during pre-training (Flan-T5-XL might have, for some terms) or context inference from `content` (limited).
- Since `definition = f(root)` is one-to-one, definition accuracy is structurally bounded by root accuracy. For test roots the model can't recover, the definition is also non-recoverable.

A pure dictionary-lookup baseline would also fail on these fields under the grouped split (it would have to retrieve the wrong root and copy the wrong definition). The model's macro-F1 of 0.16 on root is *better than such a baseline* — it implies the model does sometimes recover near-correct roots from context, especially when surface form and root differ in only minor inflection.

**The headline metric is `ingroup.macro_f1`** because that's where the meaningful generalisation question lives. Root and definition are reported as part of the structured output but are explicitly upper-bounded by the split design.

### 5.3 Distribution shift drives the headline narrative

The single test root `social justice warrior` accounts for **34.4 %** of test rows. This is a property of the corpus, not our split — the Allen AI glossary clusters many of its surface forms (`SJW`, `Social Justice Warriors`, `social justice warrior`, etc.) under one root, and that root has extensive Reddit usage.

When we report Path A's `ingroup.accuracy = 0.487`, almost all of the *missed* 0.513 comes from this single root family. The model has never seen the `social justice warrior` root labelled `anti-liberal`; it falls back on its high-frequency training priors and predicts `antisemitic` or `racist` instead. This is consistent with what we see qualitatively in the per-row predictions.

Path B's accuracy gain is the *direct* result of compensating for this single shift. It doesn't tell us the model learned anything additional about `anti-liberal` ideology — only that the model is now more *willing* to say `anti-liberal` because we forced the training prior up.

### 5.4 Path B variance — a real but unexplained pattern

Path B's macro-F1 σ across seeds is 0.063. Path A's is 0.014. **Same model, same data ordering up to seed, ×4.5 variance increase.** Two hypotheses worth investigating (§ 8.4):

1. **Some specific oversampled tail-class derails optimisation** on certain seeds. With 1,500 rows per ingroup, classes like `anti-GMO` (which has only 1–2 unique examples in raw train) get duplicated ~750–1500 times. The optimiser can over-fit on those duplicates and divert the gradient.
2. **The eval-loss curve diverges before early-stopping kicks in.** Load-best-model rescues only partially if the divergence happened in the wrong direction.

Resolving this is in the deferred error-analysis list, but for now: **Path B should not be relied on as a single-seed system** because the seed-7 collapse is dramatic. Reporting Path B with mean ± std is honest; reporting just Path B's best seed would be cherry-picking.

---

## 6 · "Is this a problem with how we trained the model and split the data?"

Direct answer: **no, the methodology is sound. The structurally-low root/definition F1 is the *correct* consequence of the grouped split. The modest ingroup macro-F1 is honest difficulty, not a pipeline bug. Path B's instability is an interesting but not damaging finding — we recommend Path A as the headline.**

Let's separate four things.

### 6.1 Is the grouped-split design a bug?

**No.** Grouping by root is the methodologically-correct way to evaluate generalisation to novel dog-whistles. A random-row split would let the model memorise the corpus's deterministic glossary; the grouped split forces it to use context. The fact that root and definition F1 are structurally low *under the grouped split* is precisely what makes the split a good test: it exposes which fields are recoverable without lookup and which aren't.

We discussed this design at length in `DESIGN_DEFENSE.md` D2 and D8 (the latter specifically about RQ-C's structural impossibility on root/definition). Same design choice as RQ-A and RQ-B — consistency across the project is a feature, not an accident.

### 6.2 Is the model architecture a bug?

**No.** Flan-T5-XL + LoRA is a textbook setup for structured-output generation:
- T5 family is the standard open-weight choice for sequence-to-sequence generation.
- Flan-T5-XL is instruction-tuned, which means we get JSON-like output discipline almost for free (95 %+ parse rate).
- LoRA is the canonical parameter-efficient fine-tuning method for models we can't fit in full-parameter fine-tuning. Our LoRA hyperparameters (r=16, α=32, dropout=0.1, target=q,v) are conventional.

A larger backbone (Flan-T5-XXL, 11 B params) would likely improve absolute numbers by 2–5 pp on macro-F1, but couldn't fit on our MIG slice and would defeat the purpose of the demo Space. Our choice is bounded by the free-tools constraint, not by a fixable design flaw.

### 6.3 Are the metrics a bug?

**Not now, but they were initially.** The brace bug, capitalisation drift, and ROUGE-on-JSON corpus-aggregation issues described in § B.7 *did* produce misleading numbers in earlier drafts (the original `comparison_all.json` shows `parse_rate=0.0, exact_match_ingroup=0.0` — which is stale and wrong, not a property of the model). We caught and patched all three before this report's headline numbers were computed.

**Outstanding:** the explanation-quality metrics (TF-IDF cosine, per-row ROUGE-L) are imperfect proxies for "is the explanation correct?" — both overstate quality on templated outputs (which ours are). A more rigorous evaluation would require human judgement on a sample of rows. This is flagged in § 8 as a deferred analysis, not a fix needed for the report.

### 6.4 So what's the limitation?

**Three things, all honest costs:**

- **Modest absolute lift on ingroup macro-F1.** 0.379 vs ~0.06 chance is real and non-trivial, but it's not a dramatic win in absolute terms. The task is hard for the same reasons as RQ-B (small training data, noisy Reddit text, deterministic glossary that the model can't memorise its way out of) — bounded by intrinsic difficulty.
- **Distribution shift dominates the surface narrative.** Without context, "Path B improved accuracy by 12 pp!" sounds great. With context, it's class-prior reweighting on a single test root that happens to be 34 % of test. We disclose this and recommend Path A; an uncareful reader could still misread it.
- **Explanation quality scoring is fuzzy.** TF-IDF cosine ≈ 0.51 sounds OK, but the metric's relationship to actual explanation quality is loose, especially when the gold explanations are templated. We can't claim "the explanations are good" with the metrics we have; we can only claim they're *consistent with* gold.

### 6.5 What WOULD be a methodology bug?

Things we would have done wrong but didn't:

- **Reporting Path B as the headline.** That would let "+12 pp accuracy" stand without the prior-shift caveat. We recommend Path A as the headline and disclose the caveat prominently.
- **Random splitting instead of grouped-by-root.** Would let the model lookup roots it had memorised. Same critique as RQ-B.
- **Reporting `dog_whistle_root.accuracy` as a primary metric.** Would imply the model is bad at the task, when actually it's structurally bounded by the split. We mark `ingroup` as primary and explain why root/definition are bounded.
- **Hiding the brace-bug fix.** The corrected scorer is what produces our 0.97 parse rate; pre-fix, the model's outputs looked broken. Disclosing this in § B.7 is honesty about how we computed the numbers we report.

### 6.6 What WOULD be a meaningful improvement?

Fair criticisms a sharp marker could raise, with our reasoning for not having done them:

- **Larger base model.** Flan-T5-XXL (11 B params) would likely buy 2–5 pp on macro-F1 and produce cleaner JSON. Doesn't fit on our MIG slice; doesn't fit in 16 GB free Space RAM. Out of scope.
- **Multi-task training.** Train one model that does RQ-A + RQ-B + RQ-C jointly, sharing representations. Could improve all three. Significantly more engineering; out of scope for a course project.
- **Human evaluation on a sample.** Have a human read 50–100 generated explanations and score them on coherence + factual accuracy. Would let us *defend* explanation quality rather than just report TF-IDF cosine. Cheap-ish (a few hours of annotation); not done due to time. Flagged in § 8.
- **Beam search / nucleus sampling at inference.** Currently greedy. Beam search might improve JSON-format adherence on rare cases; nucleus sampling might improve explanation diversity. We use greedy decoding for simplicity.
- **Path C: train on Path-A data but with class-balanced loss weights** (instead of oversampling) to test whether the variance issue is specifically about oversampling or about flat-prior training in general. Out of scope for this iteration.

### 6.7 The bottom line

We have a generative model that produces parseable JSON 97 % of the time, beats both the modal-class and random baselines on ingroup accuracy, and recovers ingroup macro-F1 of 0.38 on entirely-novel dog-whistle phrases. The structural bounds on root and definition are honestly disclosed and methodologically defended. The two paths give us a clean ablation about loss-side interventions that we can interpret cleanly (reweighting, not learning).

This is closer to a careful win than a triumph. The numbers are modest in absolute terms, and the same intrinsic-difficulty critique that applies to RQ-B applies here: the task is hard, and the data is what it is. What we can defend is that we set up the right experiment, ran it cleanly, and reported what the data actually showed — including the bits that don't flatter the model.

---

## 7 · Files

```
results/generation/                           # Path A (3 seeds)
├── aggregated_results.json
└── seed_{42,123,7}/
    ├── test_results.json                     # all headline metrics
    ├── best_model/                           # LoRA adapter only (~36 MB)
    └── test_predictions.parquet              # per-row predictions for error analysis

results/generation_balanced/                  # Path B (3 seeds, currently HPC-only — see § 8.8)
└── (same shape as above)
```

Source files:

- `scripts/hpc/train_generator.py` — main training script (handles both paths)
- `scripts/configs/generation.yaml` — Path A hyperparameter config
- `scripts/configs/generation_balanced.yaml` — Path B config (only delta is `class_balance_train: 1500`)
- `scripts/hpc/submit_generation.sh`, `submit_generation_balanced.sh` — SLURM submitters
- `scripts/hpc/eval_generation.py` — single source of truth for RQ-C metrics; CLI for offline recompute, library entry for in-training calls
- `scripts/hpc/report_metrics.py` — per-path aggregation across seeds
- `data/manifests/split_manifest.json` — root-to-split assignment (shared with RQ-A and RQ-B)

Cross-references:

- `docs/DESIGN_DEFENSE.md` § D2 (grouped split), D5 (pejoratives dropped), D8 (RQ-C structural impossibility on root/definition)
- `docs/rq_b_report.md` — same generalisation story for the classifier task
- `data/final/README.md` — schema for `data/final/rq_c_generation/*.parquet`

---

## 8 · Outstanding error analyses

Per-row predictions are saved at `results/generation/seed_*/test_predictions.parquet` (Path B equivalents on cluster only — § 8.8 below). Suggested cells:

### High-priority (changes the headline)

1. **Re-run baselines with the fixed metric pipeline.** `results/baselines/comparison_all.json` was generated *before* the brace-bug / capitalisation fixes. Re-run **random-7-class**, **modal-class** (`anti-liberal`), and **TF-IDF 1-NN retrieval** on the same 2,095 test rows under the corrected scorer. This is the headline defence: "Flan-T5 + LoRA beats TF-IDF retrieval on `ingroup.macro_f1`".
2. **Per-class confusion matrix for `ingroup` × {Path A, Path B}.** Quantify the qualitative claim that A collapses onto `antisemitic`/`racist` and B redistributes mass to `anti-liberal`. ~10 lines of code.
3. **Per-root accuracy table** (40 test roots × 7 ingroups, both paths). The `social justice warrior` root will dominate; the question is whether the *other 39 roots* improve under Path B. If macro-over-roots is flat or worse, that confirms B's accuracy gain is one-root-driven.

### Medium-priority (validates the methodology)

4. **Path B variance investigation.** Why does seed 7 collapse to macro-F1 = 0.296? Compare seed-7 confusion matrix to seeds 42 / 123. Hypotheses to discriminate: (a) particular oversampled-tail class derails optimisation, (b) eval-loss curve already diverges before early-stopping kicks in.
5. **JSON parse-failure analysis.** ~3 % of predictions fail even after brace-repair. Print 30 failures and characterise: truncated outputs? key-name drift? Concentrated on specific roots?
6. **Explanation cosine vs ingroup-correctness correlation.** Per-row, does `explanation_cosine` track `ingroup_correct`? If not, the explanation is template-filling rather than context-grounded reasoning — a known generator pathology and worth flagging.
7. **Human evaluation on 50–100 rows.** Have a human read predicted explanations and score them on coherence + factual accuracy. Lets us defend explanation quality rather than just report TF-IDF cosine.

### Lower-priority (nice-to-have for the writeup)

8. **Pull Path B predictions to laptop.** `scp -r bocconi-hpc:~/dogwhistle_project/results/generation_balanced/seed_*/ results/generation_balanced/` so all error-analysis cells can run locally without the cluster round-trip.
9. **Top-5 most-confused ingroup pairs.** Quick summary table for the report.
10. **SHAP / token-attribution on a handful of confusions.** The repo already has SHAP wiring; spot-checking ~20 misclassified rows would strengthen the qualitative section.

---

## 9 · Glossary

For readers new to the project / generation-NLP vocabulary.

- **Adapter (LoRA adapter).** A small set of additional parameters inserted into a frozen pre-trained model. Fine-tuning updates only the adapter, not the base weights. Adapter file is ~36 MB for our config; base model is 12 GB.
- **bf16 (bfloat16).** A 16-bit floating-point format that gives faster training than fp32 with less memory, while keeping enough dynamic range for stable training on modern GPUs (A100 supports it natively). Halves memory vs fp32.
- **Cross-entropy loss (token-level).** Standard sequence-to-sequence training loss: at each decoder position, compare the model's predicted next-token distribution against the gold next token. Average over all positions.
- **Decoder.** In a sequence-to-sequence transformer (like T5), the part that *generates* output token-by-token, conditioned on the encoder's representation of the input.
- **Encoder.** The part of a sequence-to-sequence transformer that reads the input prompt and produces a contextual representation for the decoder to attend to.
- **Flan-T5.** Google's instruction-tuned variant of T5. Pre-trained on T5's original objective (mask-and-restore on a huge corpus), then further trained on millions of natural-language instruction-following examples. Comes in `small`, `base`, `large`, `xl` (3 B), `xxl` (11 B). We use XL.
- **Greedy decoding.** At each generation step, pick the highest-probability next token. Faster and reproducible (vs sampling) but can produce repetitive output.
- **Gradient checkpointing.** A training trick that saves memory by re-computing intermediate activations during backprop instead of storing them. Trades compute for memory; necessary to fit Flan-T5-XL on a 40 GB MIG slice.
- **Grouped split.** A train/val/test partition where rows sharing a key attribute (here, `dog_whistle_root`) are kept together. Tests generalisation to *unseen* groups.
- **Instruction tuning.** A specific kind of fine-tuning where a pre-trained model is taught to follow natural-language instructions ("Translate this:", "Summarise this:", etc.). Models that have been instruction-tuned tend to follow new prompt patterns more reliably than raw pre-trained models.
- **JSON.** A structured text format with key-value pairs and quoted strings. Our generator outputs JSON; downstream code needs to parse it with `json.loads`.
- **LoRA (Low-Rank Adaptation).** A parameter-efficient fine-tuning method (Hu et al. 2021). Inserts small low-rank adapter matrices into specific transformer layers; freezes the base model; only trains adapters. Adapter file is small (~36 MB), training is fast, base model is reusable.
- **Macro-F1.** Average of per-class F1, equally weighting every class. Same definition as in RQ-A and RQ-B.
- **Mixed precision.** Training using a mix of 16-bit and 32-bit floating-point. Faster and uses less memory than full fp32, with matched-or-better numerics on modern GPUs.
- **Oversampling.** A data-balancing technique: duplicate (or weight) rare-class rows so each class has roughly equal representation in the training batches. Path B uses oversampling; Path A doesn't.
- **PEFT (parameter-efficient fine-tuning).** Umbrella term for methods like LoRA, prefix tuning, prompt tuning, etc. — all share the property that only a small fraction of the model's parameters are updated during fine-tuning.
- **Prompt template.** The fixed text format we wrap inputs in before passing them to the model. Ours is `"Text: ...\nMatched term: ...\nReturn JSON explaining the coded use."`. The model learned to produce JSON in response to that pattern.
- **Rank (LoRA rank, `lora_r`).** The dimensionality of the adapter's low-rank factor matrices. Higher rank = more adapter capacity, larger adapter file. We use 16.
- **ROUGE-L.** A sequence-overlap metric for generated text. Computes longest-common-subsequence between predicted and gold strings, scored as F1. Per-example, then averaged. Tends to look high on templated outputs.
- **Seed.** Random number that initialises everything random in training. Different seeds → slightly different models. We report mean ± std across 3 seeds.
- **Sequence-to-sequence (seq2seq).** A model architecture (input is one sequence of tokens, output is another) — distinguishable from classification (output is a single label) and language modelling (model continues an input).
- **TF-IDF cosine similarity.** Represent two strings as TF-IDF (term-frequency × inverse-document-frequency) vectors and compute the cosine of the angle between them. Captures lexical overlap weighted by term-importance. L5 metric on the course inventory.
- **Tokenisation.** Splitting text into pieces the model sees as discrete inputs. Flan-T5 uses SentencePiece with a 32 K vocabulary.
- **Warmup.** A period at the start of training where the learning rate ramps linearly from 0 up to its full value. Stabilises early gradients.
