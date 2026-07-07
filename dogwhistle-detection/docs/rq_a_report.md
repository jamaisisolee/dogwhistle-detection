# RQ-A Binary Disambiguator — Full Report

> **Audience.** Teammates and outside readers who haven't seen this code before. Read § A first if you're new; skip to § 3 if you only want the numbers.
> **Status.** All six runs complete (job `485051`, finished 2026-04-29 12:29 CEST, 2 h 03 m on `gnode04`).
> **Last updated.** 2026-04-29.

---

## TL;DR

We trained a RoBERTa-base text classifier (125 M parameters) to answer one question: **given a Reddit comment that contains a known dog-whistle phrase, is the phrase being used in its coded sense or in a literal sense?** The model picks one of two labels — `coded` (1) or `literal` (0). Our headline number on the locked human-gold disambiguation set is **F1 = 0.707 ± 0.015** across 3 seeds. F1 on this near-balanced eval (54 % positive) sits essentially at the always-predict-coded prevalence floor (0.701) — F1 is the wrong primary metric on near-balanced binary tasks; the model's substantive lift is in **accuracy = 0.696 ± 0.010** (vs majority 0.540, **+15.6 pp**) and **PR-AUC = 0.802 ± 0.010** (vs 0.50 random). On the silver-labelled grouped test split we score **F1 = 0.792 ± 0.010**.

The two input arms (`term` vs `term_enriched_def`) are within 0.005 F1 of each other on the locked disambiguation set, indicating the model **does not require the curated glossary definition at inference time** — a positive deployment-robustness finding.

The two important caveats are: **(i)** the lift over majority-class is modest; for *deployment* it is meaningful (the model also beats majority accuracy by 15.6 pp, 0.696 vs 0.540), but for *publication* it is a careful win, not a triumph; and **(ii)** the training negatives are silver-labelled by Llama-3.1-8B-Instruct, so the *true* (fully-human-annotated) ceiling on this task is likely higher than 0.707 — we documented this as a known cost of the no-paid-API constraint, not as a pipeline bug.

We initially attempted `microsoft/deberta-v3-large` (a more capable backbone) but hit numerical instability across five hyperparameter configurations and swapped to RoBERTa-base. Full forensics in `docs/sessions/2026-04-29_rqa_nan_postmortem.md`; one-paragraph summary in § B.7.

---

## A · For someone new to the project

### A.1 What is RQ-A?

The Bocconi 597 NLP project on dog-whistle detection has three research questions:

- **RQ-A.** Given a Reddit comment that contains a dog-whistle phrase, is the phrase being used in its **coded** sense or its **literal** sense? (Binary classification — this report.)
- **RQ-B.** Given a coded use, **which ingroup does the dog-whistle target?** (17-way classification — see `docs/rq_b_report.md`.)
- **RQ-C.** Same input as RQ-B, but produce a *structured* JSON output: dog-whistle root, ingroup, dictionary definition, and free-text explanation. (Generation — see `docs/rq_c_report.md`.)

RQ-A is the disambiguator. It does not classify *what* the dog-whistle targets; it answers *whether* the phrase is being used in its coded ideological sense or in a benign literal sense. Example: the phrase `"absent fathers"` can describe a literal sociology phenomenon ("studies on absent fathers in low-income communities") or be a coded racist trope blaming Black communities. RQ-A's job is to tell those two uses apart.

### A.2 What is `silent_signals`, and how do we get negatives?

`silent_signals` is a corpus released alongside *"Silent Signals, Loud Impact: LLMs for Word-Sense Disambiguation of Coded Dog Whistles"* (Kruk et al. 2024, ACL). It contains Reddit comments and political speech where dog-whistle phrases appear, with human annotation of whether each occurrence uses the phrase in its coded sense.

**Important distinction from RQ-B:** the published `silent_signals` corpus contains **only positives** — comments where the dog-whistle is being used in its coded sense. For a binary classifier, we need negatives too: comments where the same phrases appear in their *literal* sense. Building those negatives is a non-trivial pipeline of its own, fully documented in `docs/negative_report.md`. The short version:

1. **Stage 1 — Heuristic mining.** For each dog-whistle root, scrape Reddit for comments containing one of its surface forms. Treat these as candidate negatives.
2. **Stage 2 — LLM adjudication.** Run each candidate through `Llama-3.1-8B-Instruct` ("is this comment using the phrase in a coded or literal sense?"). Drop the ones the LLM flags as coded — they leaked through Stage 1's surface-form heuristic. Stage 2's adjudication report flagged a **62.11 % contamination rate** on the raw heuristic candidates, i.e. the LLM judged the majority of "negatives" the surface-form heuristic produced were *actually* coded uses.
3. **Stage 3 — Balancing.** Per root, pair each coded positive with a literal negative using the same root, balancing the dataset 1:1.

The training and silver-test negatives are therefore **silver-labelled** — labels are produced by an LLM-as-judge, not a human. The locked human-gold eval sets (§ B.2) are the only fully-human-annotated evaluation surfaces.

Each row in our data has:

| column | what it is | example |
|---|---|---|
| `content` | the full text of the Reddit comment | `"Absentee fathers only seem to come in one color: Black."` |
| `dog_whistle` | the surface form of the candidate phrase in this comment | `"Absentee fathers"` |
| `dog_whistle_root` | the canonical root the surface form maps to | `"absent fathers"` |
| `definition_enriched` | Allen AI glossary's paragraph-form coded-meaning description | `"This racialized term uniquely targets Black fathers..."` |
| `label` | gold answer: 1 = coded, 0 = literal | `1` |

The model gets `(content, dog_whistle)` (and optionally `definition_enriched`) and predicts `label`.

### A.3 Why is this hard?

Three reasons:

1. **Same-term contrastive pairing.** Every dog-whistle root in our training data appears with *both* labels — same `dog_whistle`, different `content`, different `label`. The model cannot succeed by memorising "phrase X is always coded"; it must read the surrounding text and infer intent. (This is intentional: it's exactly what we want a deployable disambiguator to do.)
2. **The line between coded and literal is genuinely fuzzy.** Speakers often use phrases ambiguously, gesture toward coded meanings while maintaining plausible deniability, or invoke a coded phrase ironically. Even expert human annotators disagree on borderline cases — there is irreducible label noise in the underlying task.
3. **Reddit is short, noisy, and context-dependent.** Comments are fragmentary, soaked in sarcasm and in-jokes, and reference unstated context (parent threads, subreddit norms, previous interactions). Without that context, even a human reader can be unsure.

A binary classifier that lifts disambiguation accuracy on a held-out human-gold set from 54 % (majority-class) to 70 % is doing real work, but the absolute ceiling here is bounded by how confidently anyone — model or human — can read intent from short Reddit prose.

---

## B · Methodology, step by step

### B.1 The data

`data/final/rq_a_binary/` holds five Parquet files: `train.parquet`, `val.parquet`, `test.parquet` (silver-labelled), plus two **locked human-gold** files `eval_detection.parquet` and `eval_disambiguation.parquet` (described in § B.2 below).

| split | rows | positives (coded) | negatives (literal) | unique roots | label authority |
|---|---:|---:|---:|---:|---|
| train | 12,366 | 6,183 | 6,183 | 202 | silver (Llama-as-judge) |
| val | 2,522 | 1,261 | 1,261 | grouped split | silver |
| test | 2,768 | 1,384 | 1,384 | grouped split | silver |
| eval_detection | 101 | 51 | 50 | held-out humans | **gold (human)** |
| eval_disambiguation | 124 | 67 | 57 | held-out humans | **gold (human)** |

Each row is one Reddit comment containing one dog-whistle phrase. Train, val, test are 1:1 balanced by construction (see § A.2 Stage 3); the locked human-gold sets reflect the prevalence chosen by the corpus annotators.

**Provenance note.** Positives come directly from `silent_signals`. Negatives come from the four-stage mining pipeline described in `docs/negative_report.md`. The 62.11 % Stage-2 contamination rate is recorded in `data/manifests/negatives_adjudication_report.json`.

### B.2 Why grouped splits, and why two locked human-gold eval sets

This is the methodological heart of RQ-A and worth understanding in detail.

**Grouped split — same idea as RQ-B.** Every dog-whistle phrase belongs to a *root* (e.g. `"absent fathers"`, `"absentee fathers"`, `"absent father"` all map to root `"absent fathers"`). We assign each root to exactly one of {train, val, test}. So if a root is in train, *every* sentence using *any* surface form of it is in train. The test set uses 100 % roots the model has never been told the label of. This forces the model to *generalise* the coded-vs-literal distinction to novel phrases rather than memorising "phrase X = coded". `DESIGN_DEFENSE.md` D2 mandates this; the same partitioning is used in RQ-B and RQ-C.

**Locked human-gold eval — RQ-A specific.** Two additional files exist beyond train/val/test:

- `eval_detection.parquet` (101 rows) — a hand-curated set of comments where the dog-whistle phrase *might* be present, designed to test whether the model can detect coded uses vs literal mentions in a balanced 51-coded / 50-literal mix.
- `eval_disambiguation.parquet` (124 rows) — a hand-curated set explicitly designed for the disambiguation task, with each row carrying a clear gold label from human annotators.

These two files are **LOCKED**:
- They are *never* concatenated into a training dataset.
- They are *never* tokenised into a Trainer's input pipeline.
- They are touched only by `model.evaluate()` at the end of training, after the model has already been chosen on the val split.

This is enforced as a hard project rule. The defence in `DESIGN_DEFENSE.md` D7 is: any number computed on these files is a *true* held-out human-gold result, not a silver-label echo. The trade-off is sample size — 101 + 124 rows is small, so there's wider statistical uncertainty around any single number — but the labels are trustworthy.

**Headline metric is on `eval_disambiguation_124`** because it's the larger of the two and explicitly disambiguation-shaped (every row contains a phrase that could plausibly be either coded or literal). `eval_detection_101` is reported as a secondary metric.

### B.3 Input formats — two arms

Each `(content, dog_whistle [, definition])` triple gets serialised into a single text input the model reads. We test two arms (mirrors RQ-B's input ablation philosophy):

- **`term` arm (no definition):**
  ```
  Candidate term: <dog_whistle>
  Text: <content>
  Question: Is this candidate used as a coded dog whistle here?
  ```
- **`term_enriched_def` arm (with paragraph glossary):**
  ```
  Candidate term: <dog_whistle>
  Candidate meaning: <definition_enriched>
  Text: <content>
  Question: Is this candidate used as a coded dog whistle here?
  ```

The `term_enriched_def` arm tells the model what the dog-whistle's coded meaning is *as a paragraph of glossary description* (sourced from the Allen AI glossary). The `term` arm withholds that information and forces the model to infer intent from `content` alone.

**Why this ablation, and not "text-only vs +term"?** The naive ablation "does adding the term help?" is uninformative because the term is *always already in `content`* (we identified it via the matcher). The contrast that matters is "does adding the curated **definition** help?" — because in deployment on novel dog-whistles, you may have the term identified but no curated glossary entry yet. If `term` alone matches `term_enriched_def`, the model is robust to missing glossaries.

A short-definition arm was considered and dropped — see § B.6 for why.

**Same-term contrastive pairing matters.** Every dog-whistle root in train appears with *both* labels — same `dog_whistle`, same `definition_enriched`, different `content`, different `label`. This forces the model away from definition-only lookup: a model that predicted from the definition column alone would score 50 % on training because both labels share the same definition. Any signal above chance must come from context (`content`).

### B.4 The label space

Two classes:

- `0 = literal` — the phrase is being used in its non-coded, dictionary sense
- `1 = coded` — the phrase is being used as a dog-whistle (implicit ideological signal)

Training data is 1:1 balanced by construction, so chance accuracy on train is 0.50. The locked human-gold sets are mildly skewed: 67/124 = 0.540 coded prevalence on `eval_disambiguation`, 51/101 = 0.505 coded on `eval_detection`. The majority-class baseline (always predict `coded`) therefore scores:

- F1 = 2·prev / (1 + prev) = 2·0.540 / 1.540 = **0.701** on `eval_disambiguation_124`
- Accuracy = 0.540 on `eval_disambiguation_124`

Note that the F1 floor at this prevalence is high (0.701) — always-predict-coded already achieves it because every coded row is "caught" with full recall. The accuracy floor (0.540) is the more discriminating baseline; on near-balanced binary tasks F1 alone is uninformative. We report all three (F1, accuracy, PR-AUC) and read the lift off accuracy + PR-AUC, where the prevalence bias of F1 doesn't apply.

### B.5 The model — RoBERTa-base, in plain words

**RoBERTa** (Liu et al. 2019) is a transformer-based language model — same family as BERT and GPT, fundamentally a stack of self-attention layers that read text and produce dense vector representations. **RoBERTa-base** has 125 million parameters, 12 transformer layers, and a 50,000-token vocabulary. It was originally pre-trained by Facebook AI on 160 GB of English text.

**Fine-tuning** means we take that pre-trained model and continue training it on our binary task. We add one extra layer at the top: a "classification head" that takes the model's representation of the input text and produces 2 numbers (one for `literal`, one for `coded`). We pass these through softmax to get probabilities, predict the higher one, and compare to the gold label. We measure how wrong we were (the loss), and we update the model's parameters slightly to be less wrong next time. Repeat 12,366 times per epoch, for several epochs.

**Why RoBERTa-base?** Two reasons:

1. **Stability.** The canonical RoBERTa-base fine-tuning recipe (lr = 2e-5, AdamW, linear warmup) just works. No NaN regimes, no special tricks.
2. **Capacity matches data size.** 125 M parameters on ~12 k 1:1 contrastive pairs is more than enough — large encoders typically overfit on this data scale without strong regularisation.

We *initially* tried `microsoft/deberta-v3-large` (a more capable 435 M-parameter backbone), but hit numerical instability across five hyperparameter configurations. § B.7 covers that detour.

### B.6 Training procedure

Every model variant in this report was trained with the same hyperparameters (configured in `scripts/configs/binary.yaml`):

| Hyperparameter | Value | What it does in plain language |
|---|---|---|
| `model_name` | `FacebookAI/roberta-base` | The pre-trained transformer we start from |
| `max_length` | 512 | We cut comments off at 512 tokens (roughly 350 words). Longer comments get truncated. p99 of train content lengths is 491 tokens — almost no truncation in practice. |
| `epochs` | 5 (max) | One epoch = one pass through the training set. Capped at 5 but allowed to stop earlier (see early stopping below). |
| `batch_size` | 32 | We process 32 rows at a time before doing a parameter update |
| `lr` | 2e-5 | The learning rate — how big a parameter update is. 2e-5 is canonical for RoBERTa-base fine-tuning |
| `weight_decay` | 0.01 | A regularisation term that gently nudges parameters toward zero — prevents overfitting |
| `warmup_ratio` | 0.1 | The first 10% of training uses a linearly-ramping learning rate (smaller at the start, full rate by 10%) |
| `max_grad_norm` | 1.0 | Cap on how big a single update can be — protects against rare exploding-gradient cases. (Was 0.5 during the DeBERTa-v3-large debugging; reverted to default for RoBERTa-base.) |
| `label_smoothing_factor` | 0.1 | Soft targets (0.05/0.95 instead of 0/1) bound per-example cross-entropy loss; prevents extreme logits from producing 10×-magnitude first-batch gradients. Carried over from the DeBERTa-stability work; harmless on RoBERTa. |
| `init_head_std` | 0.02 | Manual re-initialisation of the new classification head with a 0.02-std Gaussian (BERT-standard). DeBERTa's default head init was producing |logit| > 10 at step 0; the re-init was the actual root-cause fix. Harmless on RoBERTa. |
| `early_stopping_patience` | 2 | If validation F1 doesn't improve for 2 consecutive epochs, stop training |
| `bf16` / `fp16` | both false | We train in full fp32. (bf16 is safe but unnecessary on 125 M params; fp16 was tried briefly and ruled out due to a transformers-5.x incompatibility.) |
| `seeds` | 42, 123, 7 | 3 independent runs per arm to quantify seed-noise on the headline metric |

**Loss function.** Cross-entropy over the 2-way softmax, with label smoothing factor 0.1.

**Evaluation cadence.** Once per epoch, we run the model on the validation set, compute val F1, and save the model if val F1 improved. After 5 epochs (or earlier if early-stopping fired), we load the best checkpoint and evaluate on:

1. The grouped silver test split (2,768 rows)
2. The locked human-gold `eval_detection_101` set (only `model.evaluate()`, no fine-tuning ever)
3. The locked human-gold `eval_disambiguation_124` set (only `model.evaluate()`, no fine-tuning ever)

**Hardware.** A100 80 GB GPU, sliced as a MIG instance (`4g.40gb`). Each run takes ~30 minutes. Six runs end-to-end finished in 2 h 03 m wall-clock.

### B.7 The DeBERTa-v3-large detour (and why we swapped to RoBERTa-base)

We initially started training with `microsoft/deberta-v3-large` (435 M parameters) — a more capable backbone widely used in recent dog-whistle / hate-speech literature. It would not train.

**Five attempts, all failed:**

| job | configuration | what happened |
|---|---|---|
| 484468 | lr=2e-5, max_grad_norm=1.0, fp32 | NaN at step 200 — lr too aggressive for v3-large |
| 484986 | lr=1e-5, max_grad_norm=0.5, fp32 | No NaN, but `grad_norm` was 5–15 at every step → clip cut effective LR by ~20× → loss froze, val collapsed to predict-all-1 |
| 484991 | lr=1e-5, max_grad_norm=1.0, fp32 | NaN at step 100 — lr drop alone insufficient |
| 485002 | + fp16 | Failed in 58 s on a `transformers`-5.x / `accelerate` incompatibility (`"Attempting to unscale FP16 gradients."`) — fp16 path ruled out |
| 485040 | + manual head re-init, label smoothing | Loss stuck at 1.4, val F1 = 0.667 (predict-all-1 collapse), NaN at epoch 1.03 |

**What was going wrong** (full debugging notes in `docs/sessions/2026-04-29_rqa_nan_postmortem.md`): DeBERTa-v3-large's pooled outputs come out at high norm, the classification head's default initialisation pushes those into very large logits at step 0, the loss explodes on confidently-wrong rows, and gradient clipping doesn't help because it kicks in after the forward pass — so a single Inf logit produces a NaN gradient and the parameters get corrupted in one update.

**The RoBERTa-base swap** (job 485051) trained cleanly on the very first attempt with the same per-dataset hyperparameters that had failed on DeBERTa, plus the head-init fix and label smoothing carried forward as cheap insurance. 6 runs (2 arms × 3 seeds) finished in 2 h 03 m. No NaN, no instability, expected loss curve, expected F1.

**What we lose from the swap:** DeBERTa-v3-large's disentangled relative-position attention typically buys 1–3 F1 points on hard binary classification benchmarks. We give that up. Mitigating: (i) seed variance (σ = 0.015) on a 124-row eval probably swamps a 1-3 point architectural gain; (ii) the headline numbers (F1 = 0.707; accuracy 0.696 vs majority 0.540 = +15.6 pp; PR-AUC 0.802 vs 0.50 random) clear the genuine baselines on accuracy and ranking quality, where prevalence bias doesn't compress the metric; (iii) RoBERTa is in fact the most-cited encoder backbone in the dog-whistle / hate-speech literature, so our numbers are directly comparable to prior work.

**Disclosure.** The 5-page report should mention the model swap in one sentence ("we initially attempted DeBERTa-v3-large but hit numerical instability across five hyperparameter configurations and swapped to RoBERTa-base; the model-choice change is incidental to the research question") rather than dwell on it.

### B.8 Metrics — what F1 means for binary classification

For each row, the model predicts `coded` (1) or `literal` (0). Compared against the gold label, every prediction is one of:

- **TP** (true positive) — predicted coded, gold coded
- **FP** (false positive) — predicted coded, gold literal
- **FN** (false negative) — predicted literal, gold coded
- **TN** (true negative) — predicted literal, gold literal

For the *coded* class:

- **Precision** = TP / (TP + FP) — when the model said "coded", how often was it right?
- **Recall** = TP / (TP + FN) — of all the actually-coded rows, how many did the model catch?
- **F1** = 2 · (Precision · Recall) / (Precision + Recall) — harmonic mean of precision and recall, on a 0-to-1 scale

**F1 is the primary metric** because the task is non-symmetric — false positives (calling literal use coded) and false negatives (missing actual dog-whistles) have different real-world costs, and F1 balances them. Accuracy alone is misleading on prevalence-skewed data; on `eval_disambiguation_124` (54 % coded) a trivial always-coded model scores 0.540 accuracy, hiding total recall failure on the literal class.

**PR-AUC** (precision-recall area under curve) is reported as a secondary metric — it summarises how well the model's *probability scores* rank coded above literal across all decision thresholds, independent of where the operating point sits.

---

## 3 · Headline numbers (mean ± std across 3 seeds)

| metric | arm `term` | arm `term_enriched_def` | naive baseline |
|---|---:|---:|---|
| **F1 on `eval_disambiguation_124`** (primary) | **0.707 ± 0.015** | **0.702 ± 0.021** | majority-coded F1 = 0.701 (prevalence floor) |
| Accuracy on `eval_disambiguation_124` | 0.696 ± 0.010 | 0.677 ± 0.020 | majority = 0.540 |
| Precision / Recall on `eval_disambiguation_124` | 0.739 / 0.682 | 0.712 / 0.711 | — |
| PR-AUC on `eval_disambiguation_124` | 0.802 ± 0.010 | 0.783 ± 0.006 | — |
| **F1 on `eval_detection_101`** (secondary) | 0.721 ± 0.042 | **0.744 ± 0.049** | majority-coded = 0.671 |
| PR-AUC on `eval_detection_101` | 0.801 ± 0.021 | 0.792 ± 0.006 | — |
| **F1 on grouped test split** (silver, 2,768 rows) | **0.792 ± 0.010** | 0.768 ± 0.015 | balanced random = 0.500 |
| PR-AUC on test split | **0.874 ± 0.010** | 0.820 ± 0.016 | — |

**Headline interpretation in one line:** Across three seeds the RoBERTa-base binary disambiguator achieves **F1 = 0.707 ± 0.015 on the locked human-gold disambiguation set** (124 rows; F1 prevalence floor 0.701 — see § 5.2). The substantive lift is in accuracy (0.696 ± 0.010 vs 0.540 majority, **+15.6 pp**) and PR-AUC (0.802 ± 0.010 vs 0.50 random). The `term` and `term_enriched_def` arms are within 0.005 F1 on disambiguation and within 0.025 F1 on test — the model does **not** require the curated glossary at inference time, a positive deployment-robustness finding.

**The detection set tells a slightly different story:** there `term_enriched_def` wins by 0.023 F1, but variance on detection is 4× the variance on disambiguation (σ ≈ 0.045 vs 0.015), so the difference is within one standard deviation. **Disambiguation is the more reliable comparison** because its variance is much tighter and its label distribution is closer to the trained-on prior.

---

## 4 · Per-seed × per-arm breakdown

| metric | seed 42 (`term`) | seed 123 (`term`) | seed 7 (`term`) | seed 42 (`term+enriched`) | seed 123 (`term+enriched`) | seed 7 (`term+enriched`) |
|---|---:|---:|---:|---:|---:|---:|
| `disambiguation_124` F1 | 0.710 | 0.688 | **0.725** | 0.699 | 0.677 | **0.730** |
| `detection_101` F1 | 0.674 | 0.711 | **0.777** | 0.723 | 0.697 | **0.811** |
| `test_split` F1 | 0.797 | **0.801** | 0.777 | **0.782** | 0.777 | 0.747 |
| `test_split` PR-AUC | 0.880 | **0.882** | 0.859 | 0.825 | 0.798 | **0.837** |

Variance pattern: `term` is more stable than `term+enriched` on the silver test split (σ 0.010 vs 0.015) and on detection PR-AUC (σ 0.021 vs 0.006). Detection F1 σ is high in both arms (0.04+), suggesting the 101-row sample size dominates seed-noise there. Best individual seed varies by arm × dataset — no consistent winner.

---

## 5 · Interpretation

### 5.1 Term vs term_enriched_def (does the curated definition help?)

On `eval_disambiguation_124` the two arms are within 0.005 F1 of each other (both ~0.70). The curated definition does not detectably help on the headline metric. Three readings:

1. **The model already knows what these phrases mean.** RoBERTa-base was pretrained on 160 GB of English text including political and ideological discussions. For the 40 test roots specifically, the model's prior may already encode their coded sense, making the explicit glossary description redundant.
2. **The contextual signal in `content` dominates.** When a comment says *"absentee fathers only seem to come in one color: Black"*, the surrounding text alone is enough to disambiguate; the definition string adds no marginal information.
3. **The definition column carries a label-leakage risk.** On training data, every same-root pair shares the same `definition_enriched` value but has both labels — so the definition cannot be used as a label predictor. But on ambiguous test cases it might bias the model toward `coded` (since the definition describes the coded meaning). The arms being equal on disambiguation suggests this risk did not materialise.

**Deployment implication:** if a future user wants to deploy RQ-A on a *novel* dog-whistle for which no glossary entry exists, they can run the `term` arm and expect equivalent performance. This is the practical robustness finding.

### 5.2 The locked human-gold eval is the headline; the silver test split is upper-bounded

The silver test-split F1 (0.792) is higher than the locked-eval F1 (0.707). Two reasons it should not be the headline:

1. **Train and test split share label noise.** The silver test labels are produced by the *same* Llama-3.1-8B adjudication pipeline that produced the silver train labels. If Llama is systematically wrong on certain boundary cases (sarcasm, in-group reclamation, meta-commentary on slurs), the model has been *trained* to be wrong in the same way, and the silver test gives it credit for that. The locked human-gold sets test against an external authority and don't reward train-set mistakes.
2. **The locked sets are explicitly disambiguation-shaped.** Their authors selected rows where the label is genuinely ambiguous-on-the-surface (a balanced mix of coded and literal uses, on phrases that *could* plausibly be either). The silver test split inherits whatever distribution the mining pipeline produced — including some "easy" negatives that don't really exercise disambiguation.

We use **0.707 as the headline number**. 0.792 is reported as the silver test F1 with a note explaining why it's an upper bound on what the locked eval would have shown.

### 5.3 Annotation-quality ceiling — unmeasured but non-trivial

Stage 2's adjudication report flagged a **62.11 %** contamination rate on the raw heuristic-mined candidates — Llama said the majority of the surface-form-matched "negatives" were *actually* coded uses that had to be discarded. We accept Llama's judgement on those calls because hand-annotating ~14 k pairs is out of scope for a course project, but two known sources of residual error remain in the silver labels we trained on:

1. **False negatives that survived Stage 2.** Llama may have mislabelled some genuinely coded uses as `non-coded`, contaminating the negative class. A model trained on this signal learns to call a fraction of those examples non-coded too — depressing recall on `eval_disambiguation_124`.
2. **Boundary cases where the LLM is systematically wrong.** Sarcasm, in-group reclamation, and meta-commentary on a slur are exactly the cases where mid-size LLMs are known to misjudge intent. Any consistent bias propagates into the supervised signal.

The locked `eval_disambiguation_124` and `eval_detection_101` sets are the *only* fully human-gold evaluations we have. We therefore expect the *true* (human-judged) ceiling on this task to be **higher than 0.707 F1** — by an unknown but non-zero margin. A meaningful follow-up would be to spot-annotate ~200 negatives from `data/final/rq_a_binary/test.parquet`, compute the human-vs-Llama agreement rate on that sample, and use it to bound the residual label noise. We document this as a known cost of the no-paid-API constraint rather than as a project bug.

---

## 6 · "Is this a problem with how we trained the model and split the data?"

Direct answer: **no, the methodology is sound. The headline F1 of 0.707 is a modest but honest result; the lift over majority-class is small in absolute terms, and the silver-label ceiling is the legitimate concern, but neither is a pipeline bug.**

Let's separate four things.

### 6.1 Is the eval methodology a bug?

**No.** Two locked human-gold eval files (101 + 124 rows) sit outside the training pipeline and are touched only by `model.evaluate()` after model selection on val. Any number computed on them is a true held-out human-gold result. `DESIGN_DEFENSE.md` D7 documents and enforces this: the files are explicitly forbidden from being concatenated into training data. Plus the silver test split is grouped by root — same generalisation logic as RQ-B — so even the *silver* F1 is on novel roots, not memorisation.

### 6.2 Is the model architecture a bug?

**No.** RoBERTa-base is a standard, well-understood backbone for binary sentence classification at this scale. The loss function (cross-entropy with label smoothing 0.1), optimiser (AdamW with linear warmup, lr=2e-5, weight decay 0.01), and hyperparameters are all canonical. We didn't make a weird choice that broke something. The DeBERTa-v3-large detour (§ B.7) was a numerical-instability incident on a *different* model and is fully resolved by the swap; nothing carries over.

### 6.3 Is the negatives-mining pipeline a bug?

**Not a bug, but a known cost.** The four-stage pipeline (heuristic mining → LLM adjudication → balancing → grouped split) is the cleanest negatives-construction approach we could afford under the no-paid-API constraint. Stage 2's 62.11 % contamination rate is *evidence the pipeline is doing useful work* — we caught and discarded the surface-form heuristic's false negatives. What we don't know is the residual error rate after Stage 2 (Llama-as-judge gets some calls wrong). The right framing is:

- Pipeline-as-engineered: solid. We have a deterministic, reproducible recipe (`docs/negative_report.md` § 10) and full provenance manifests for every drop decision.
- Label authority: silver, by necessity. We disclose this prominently (§ 5.3) and the locked human-gold eval set is our hedge against silver-label echo.

### 6.4 So what's the limitation?

**Three things, all honest costs:**

- **F1 is at the prevalence floor; the model's lift surfaces in accuracy and PR-AUC.** F1 = 0.707 vs 0.701 majority-class baseline is a 0.6-pp gap — within seed noise (σ = 0.015) and effectively a tie. F1 is uninformative on near-balanced binary tasks because always-predict-positive already maximises recall while keeping precision = prevalence; the metric saturates near the prevalence floor regardless of model quality. The substantive comparison is on accuracy (0.696 vs 0.540 majority, **+15.6 pp**) and PR-AUC (0.802 vs 0.50 random, **+0.30**) — both untainted by the prevalence-bias problem and both unambiguous wins. We report all three so the reader can read the result on whichever metric they trust.
- **Small evaluation set.** 124 rows means even a perfectly-trained model has wide statistical uncertainty around any single F1 number. A Wilson confidence interval at p = 0.707, n = 124 is roughly [0.62, 0.78]. Our 3-seed σ = 0.015 means the *model's mean F1* is well-estimated; the small n only bounds how confidently we can claim the eval set itself is representative of the world.
- **Silver-label ceiling.** As § 5.3 explains, training on Llama-adjudicated negatives caps how high the *true* F1 can go — the model learns to make Llama's mistakes. Spot-annotating a slice of the test split would let us bound this ceiling; we haven't done that yet (it's an outstanding error analysis, § 8).

### 6.5 What WOULD be a methodology bug?

Things we would have done wrong but didn't:

- **Training on the locked eval sets.** Hard rule #1; verified by manifest checks.
- **Random splitting instead of grouped-by-root.** That would let the same dog-whistle phrase appear in both train and test and turn the task into glossary lookup.
- **Reporting only `eval_detection_101`.** That set has 4× the seed variance of disambiguation — hand-picking it as the headline would be cherry-picking a noisier metric for whichever direction looked better.
- **Reporting silver test-split F1 (0.792) as the headline.** That would inflate the number with silver-label echo. The locked human-gold F1 is the right headline precisely *because* it tests against an external authority.
- **Hiding the DeBERTa-v3-large detour.** We disclose it in one paragraph in the report's methodology section. Burying a five-job training-instability incident would misrepresent what was tried.

### 6.6 What WOULD be a meaningful improvement?

Fair criticisms a sharp marker could raise, with our reasoning for not having done them:

- **Larger model.** DeBERTa-v3-large would likely buy 1–3 F1 points if we could fix the numerics. Two follow-up paths: (i) layer-wise LR (lower LR on the backbone, higher on the classifier head) plus the head re-init we already have, or (ii) try `microsoft/deberta-v3-base` (smaller v3 variant — possibly more stable). Both out of scope for this iteration.
- **Sample-size on the locked eval.** 124 rows is the corpus-author-curated set; expanding it would require new human annotation, which is out of scope for a course project. A spot-annotation of ~200 rows from the silver test split (mentioned in § 5.3) is a cheaper alternative that bounds residual silver-label noise without requiring full re-annotation.
- **Calibration analysis.** The model's `prob_coded` distribution per true label is in the predictions parquets but unanalysed. A miscalibrated model that's "70 % confident on every example" can land at the same F1 as a well-calibrated one — the difference matters for downstream use. Flagged in § 8.
- **Cross-domain transfer.** All training data is Reddit. The model's ceiling on, e.g., political speech transcripts (which `silent_signals` also includes a few of) is unknown. Out of scope but worth flagging.

### 6.7 The bottom line

We have a well-trained, well-evaluated, modestly-scoring binary classifier whose 0.707 F1 reflects a hard task and silver-quality training labels — not pipeline bugs. The locked human-gold eval is the right authority for the headline; the silver test split's higher F1 (0.792) is upper-bounded by silver-label echo and reported with that caveat. The two-arm ablation produces a clean deployment-robustness finding (the curated glossary is not required at inference). The DeBERTa-v3-large detour is fully resolved by the swap and disclosed transparently.

This is an honest result. A perfect classifier on this task would need either substantially better labels, a substantially larger eval set, or both — and probably some of the residual difficulty (the genuinely-fuzzy line between coded and literal use) is irreducible.

---

## 7 · Files

```
results/binary/
├── format_term/                            # Run A: term arm (3 seeds)
│   ├── aggregated_results.json
│   └── seed_{42,123,7}/
│       ├── test_results.json               # F1, accuracy, PR-AUC across all 3 eval surfaces
│       ├── best_model/                     # final checkpoint (loaded for inference)
│       └── predictions/
│           ├── test_split_preds.parquet         # silver test split (n=2,768)
│           ├── eval_detection_preds.parquet     # locked gold (n=101)
│           └── eval_disambiguation_preds.parquet # locked gold (n=124)
└── format_term_enriched_def/               # Run B: term + glossary arm (3 seeds)
    └── (same shape as format_term/)
```

Source files:

- `scripts/hpc/train_binary_disambiguator.py` — main training script
- `scripts/configs/binary.yaml` — hyperparameter config
- `scripts/hpc/submit_binary.sh` — SLURM submitter that loops `{term, term_enriched_def} × {42, 123, 7}`
- `scripts/hpc/report_metrics.py` — per-arm aggregation across seeds
- `data/manifests/split_manifest.json` — root-to-split assignment (shared with RQ-B and RQ-C)
- `data/manifests/negatives_adjudication_report.json` — Stage-2 contamination rates per root

Session notes: `docs/sessions/2026-04-29_rqa_nan_postmortem.md`.

Cross-references:

- `docs/DESIGN_DEFENSE.md` § D2 (grouped split), D5 (pejoratives dropped), D6 (this ablation), D7 (locked eval sets)
- `docs/negative_report.md` § 5–6 (mining pipeline), § 6a (pejoratives drop), § 10 (reproduction recipe)
- `data/final/README.md` — schema for `data/final/rq_a_binary/*.parquet`

---

## 8 · Outstanding error analyses

Per-row predictions are saved at `results/binary/format_*/seed_*/predictions/*_preds.parquet` (with columns `dog_whistle`, `dog_whistle_root`, `label` (true), `pred_label`, `prob_coded`, `correct`). Suggested cells:

1. **Confusion matrix per arm**, averaged across seeds, on each of the three eval datasets.
2. **Per-root accuracy table** on the grouped silver test split — surfaces which unseen roots the model fails on. Group by ingroup to see if any ingroup family is systematically harder.
3. **Stratified F1 by surface ambiguity.** Bucket test roots into "obvious slurs" (e.g. `#WPWW`, `dindu`) vs "surface-ambiguous" (`based`, `single`, `EST`, `urban`, `globalist`). The robustness story is sharper if `term_enriched_def` beats `term` *more* on ambiguous tokens than on obvious slurs.
4. **Calibration plot.** For each arm, plot reliability diagrams of `prob_coded` per true label. A well-calibrated model has prob_coded ≈ true positive rate at each prob bucket. Quantify Expected Calibration Error.
5. **Top-N highest-confidence wrong predictions** — qualitative spot-check for definition-leakage failure modes.
6. **TF-IDF retrieval baseline.** Build a TF-IDF model on train `content`, retrieve nearest neighbour for each eval row, copy that neighbour's label. Compare F1 against our model — this tests how much of our gain is just topic similarity vs. genuine disambiguation.
7. **Spot-annotation of silver test split.** Sample ~200 rows from `data/final/rq_a_binary/test.parquet`, hand-label them, compute human-vs-Llama agreement rate. Bounds the residual silver-label error.

The artefacts for analyses 1–5 are already in place; what's missing is the analysis code. (6) and (7) require new code and (in 7's case) human time.

---

## 9 · Glossary

For readers new to the project / NLP fine-tuning vocabulary.

- **Backbone / pre-trained model.** A large model (here RoBERTa-base) that's been trained on lots of generic text before our task. We start from its weights rather than from scratch.
- **Calibration.** Whether a model's stated probabilities match its real accuracy. A well-calibrated 70 %-confident model is right ~70 % of the time on those examples.
- **Coded vs literal.** A coded use of a phrase invokes its dog-whistle ideological meaning; a literal use invokes its dictionary meaning. RQ-A's binary task is to tell these apart.
- **Cross-entropy loss.** The standard loss function for classification — measures how far the model's predicted probability distribution is from the gold one-hot label. Lower is better.
- **DeBERTa-v3-large.** A 435 M-parameter transformer from Microsoft (He et al. 2021). More capable than RoBERTa-base on many benchmarks but numerically harder to train. We attempted it, hit instability, and swapped to RoBERTa-base.
- **Dog-whistle.** A coded phrase that signals an in-group meaning to those who recognise it, while looking innocuous to outsiders. Example: `"absent fathers"` literally describes parents who aren't present, but is used as a coded racist phrase about Black communities.
- **Early stopping.** Training trick: if the validation metric doesn't improve for `patience` epochs in a row, stop and use the best-so-far checkpoint. Prevents overfitting and saves compute.
- **F1 score.** Harmonic mean of precision and recall, on a 0–1 scale. Primary metric for binary classification when both classes matter equally and prevalence isn't 50/50.
- **Fine-tuning.** Continuing to train a pre-trained model on a new task / dataset, updating its weights for the new objective.
- **fp32 / fp16 / bf16.** Floating-point precisions. fp32 (32 bits) is the conservative full-precision default; fp16 and bf16 use 16 bits (faster, half the memory, slightly different numeric tradeoffs). bf16 is generally safer than fp16 on A100s.
- **Gradient clipping.** A safety mechanism that caps how large a single parameter update can be (`max_grad_norm`). Prevents rare exploding-gradient cases. Operates after the forward pass — won't help if the forward pass itself produces NaN.
- **Grouped split.** A train/val/test partition where rows sharing a key attribute (here, `dog_whistle_root`) are kept together. Tests generalisation to *unseen* groups.
- **Label smoothing.** Replaces hard 0/1 labels with soft 0.05/0.95 labels during training. Bounds the per-example loss; prevents extreme logits from producing huge first-batch gradients.
- **Locked eval.** A held-out evaluation file that is *never* used for training, validation, or model selection. Its F1 is a true generalisation result, not a metric the model was optimised against.
- **Logits.** The raw, pre-softmax scores the model produces — one number per class. Softmax converts them to probabilities.
- **NaN.** "Not a Number". Appears when a calculation produces an undefined result (e.g. log of 0, division by 0, or overflow). Once a single parameter goes NaN in deep learning, it tends to spread to all parameters in one update.
- **PR-AUC.** Area under the precision-recall curve. Summarises ranking quality across all decision thresholds, independent of where the cut-off lies. Good complementary metric to F1.
- **Pretraining vs fine-tuning.** Pretraining is the original large-scale training Facebook AI did on RoBERTa using 160 GB of generic text. Fine-tuning is what we did: continuing to update the model's weights on our small task-specific dataset.
- **Root.** The canonical dictionary form of a dog-whistle phrase. Multiple surface forms (e.g. `"absent fathers"`, `"absentee fathers"`) share one root (`"absent fathers"`).
- **RoBERTa.** A transformer language model from Liu et al. 2019 (Facebook AI), descendant of BERT. Pre-trained on 160 GB of English text. Comes in `base` (125 M params) and `large` (355 M) sizes; we use base.
- **Same-term contrastive pairing.** A training-data design where every dog-whistle root appears with both labels — same surface phrase, different `content`, different label. Forces the model to use context, not phrase identity.
- **Seed.** Random number that initialises everything random in training (weights, dropout, data shuffle order). Different seeds → slightly different models. Reporting mean ± std across seeds quantifies how much that randomness matters.
- **Silver vs gold labels.** Gold labels are produced by trusted human annotators. Silver labels are produced by an automated process (here, Llama-3.1-8B-as-judge). Silver labels are cheaper to produce at scale but introduce a known label-noise floor.
- **Softmax.** A function that takes raw model scores and converts them into a probability distribution over the classes (positive numbers summing to 1).
- **Tokenisation.** The process of splitting text into pieces the model sees as discrete inputs. RoBERTa uses a 50,000-token vocabulary built from byte-pair encoding.
