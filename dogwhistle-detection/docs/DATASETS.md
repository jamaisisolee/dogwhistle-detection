# Datasets — Silent Signals Project

A walkthrough of every dataset surfaced in [notebooks/eda.ipynb](../notebooks/eda.ipynb), what each one is, what we use it for, and the two outstanding data problems we had to solve before training:

1. **`silent_signals` is positives-only** — every row is a confirmed coded use. We need a defensible procedure for sourcing negatives.
2. **Definitions live in a separate artefact** — the enriched Allen AI glossary in [resources/](../resources/). We need a concrete plan for concatenating them into the model input.

Both are addressed in [§6](#6--positives-only-problem--negatives-mining-plan) and [§7](#7--definition-concatenation-plan).

---

## 1 · The five SALT-NLP datasets at a glance

| Dataset | Rows | Source | Role in our project |
|---|---|---|---|
| [`SALT-NLP/silent_signals`](https://huggingface.co/datasets/SALT-NLP/silent_signals) | 16,258 | Reddit + US Congressional Record | **Core supervised data.** All rows are confirmed coded uses (positives only). We use the **Informal subset (12,923 rows)** — see [CONCERNS.md §Scoping Decision](CONCERNS.md). |
| [`SALT-NLP/informal_potential_dogwhistles`](https://huggingface.co/datasets/SALT-NLP/informal_potential_dogwhistles) | ~6.03M | Reddit | **Candidate pool for silver negatives.** Keyword-matched comments containing a dog-whistle surface form, mostly literal/in-language uses. |
| [`SALT-NLP/formal_potential_dogwhistles`](https://huggingface.co/datasets/SALT-NLP/formal_potential_dogwhistles) | ~1.10M | US Congress | Candidate pool for the formal side. **Excluded** from training under the Reddit-only scoping decision; kept as a comparator in [notebooks/eda.ipynb](../notebooks/eda.ipynb). |
| [`SALT-NLP/silent_signals_detection`](https://huggingface.co/datasets/SALT-NLP/silent_signals_detection) | 101 | Mixed (formal + informal) | **Locked human-annotated evaluation set.** Has explicit `coded` / `non-coded` labels. Never used in training. |
| [`SALT-NLP/silent_signals_disambiguation`](https://huggingface.co/datasets/SALT-NLP/silent_signals_disambiguation) | 124 | Mixed | **Locked human-annotated evaluation set** for word-sense disambiguation. Never used in training. |

A sixth artefact we attach to these — the **Allen AI dog whistle glossary**, parsed offline from [resources/Glossary_of_Dogwhistles.md](../resources/Glossary_of_Dogwhistles.md) into [resources/dog_whistle_roots_enriched.csv](../resources/dog_whistle_roots_enriched.csv) — provides the definitions/descriptions we concatenate into model inputs ([§7](#7--definition-concatenation-plan)).

---

## 2 · `silent_signals` (the supervised core)

**What it is.** 16,258 sentences (or short passages) where a known dog-whistle surface form was used as a coded reference, validated by the [Kruk et al. (2024)](https://aclanthology.org/2024.acl-long.675) GPT-4 disambiguation pipeline. Two registers are stitched together: ~80% Reddit comments (Informal), ~20% Congressional excerpts (Formal). Our project uses **Informal only**.

**Schema** ([data/manifests/dataset_profiles.json](../data/manifests/dataset_profiles.json)):

| Column | Populated for | Notes |
|---|---|---|
| `content` | both | the text |
| `dog_whistle` | both | matched surface form (706 unique) |
| `dog_whistle_root` | both | canonical root (298 unique) — **group key for splits** |
| `ingroup` | both | one of 18 labels (e.g. `racist`, `transphobic`) — stratification target |
| `definition` | both | one-line covert-meaning gloss (244 unique) |
| `type` | both | `Formal` / `Informal` |
| `source` | both | provenance string |
| `date` | both | sometimes ISO, sometimes loose |
| `speaker`, `chamber`, `party` | **Formal only** | Congressional metadata |
| `subreddit` | **Informal only** | source subreddit |

The "missingness" in the EDA heatmap (cell 9 of the notebook) is **structural** — Reddit rows have no `party`, Congress rows have no `subreddit`. It is not data quality.

**The critical property — positives only.** Every row in `silent_signals` is a positive example by construction: the SALT-NLP authors fed candidate keyword matches through GPT-4 with the dog-whistle disambiguation prompt and **kept only the high-confidence "coded" cases**. There is no `non-coded` label anywhere in this dataset. A binary classifier trained on these rows alone would learn `P(coded | text containing surface form)` — but only ever sees `coded = 1`, so it has no signal for the negative class. This is the central challenge for our Task A (binary disambiguator) and the reason [§6](#6--positives-only-problem--negatives-mining-plan) exists.

**Already on disk.** [data/splits/{train,val,test}.parquet](../data/splits/) contain a grouped-stratified split (groups = `dog_whistle_root`, stratify = `ingroup`) of the **Informal** subset only — 9,064 / 1,764 / 2,095 rows (219 / 39 / 40 roots). The same split underlies the canonical `data/final/rq_b_multiclass/` and `data/final/rq_c_generation/` bundles. See [notebooks/pipeline.ipynb](../notebooks/pipeline.ipynb) and [scripts/hpc/build_grouped_splits.py](../scripts/hpc/build_grouped_splits.py) for the splitting code; see [data/manifests/split_manifest.json](../data/manifests/split_manifest.json) for the root→split mapping.

---

## 3 · `informal_potential_dogwhistles` (Reddit candidate pool)

**What it is.** ~6.03M Reddit comments matched on a dog-whistle surface form **before** any disambiguation. Only ~100k of the 7M-total candidate set were ever fed through GPT-4 by Kruk et al.; the remainder were **never evaluated**. Most matches are presumed literal in-language uses (the canonical example: "based" in `r/cooking` is not a manosphere signal), but a non-trivial unknown share are real coded uses that simply never got adjudicated.

**Schema:** `content`, `subreddit`, `dog_whistle`, `date`, `ingroup` (inherited from the keyword→ingroup mapping, **not** human-verified), `source`.

**Role in our pipeline.** Source pool for silver negatives ([§6](#6--positives-only-problem--negatives-mining-plan)). The post-Stage-3 balanced output lives in [data/processed/negatives_balanced_{train,val,test}.parquet](../data/processed/) and the matched positives in [data/processed/positives_balanced_{train,val,test}.parquet](../data/processed/) — 6,183 / 1,261 / 1,384 rows per side after per-term 1:1 matching (8,828 pairs total). Stage 1 raw was 37,497 candidates; Stage 2 contamination 62.11%; provenance in [data/manifests/negatives_adjudication_report.json](../data/manifests/negatives_adjudication_report.json).

---

## 4 · `formal_potential_dogwhistles` (Congressional candidate pool)

**What it is.** ~1.10M Congressional Record excerpts matched on a dog-whistle surface form, same construction logic as the Reddit pool but on a much smaller, much more formal corpus. Schema includes `speaker`, `chamber`, `party`, `reference`.

**Role in our pipeline.** **Not used for training.** Excluded under the Reddit-only scoping decision in [CONCERNS.md §Scoping Decision: Reddit-only data](CONCERNS.md). Kept in the EDA notebook as a comparator (e.g. the party-majority-coloured time series in [notebooks/eda.ipynb](../notebooks/eda.ipynb)) and as the source of any reportable *out-of-domain* signal in the formal subset of the locked eval sets.

---

## 5 · The two locked human-annotated eval sets

These are the only honest performance measurements we have. Both contain **explicit coded / non-coded labels** assigned by humans, not by an LLM, which means they double as our source of *gold-standard negatives at evaluation time*.

### `silent_signals_detection` (101 rows)
| Column | Notes |
|---|---|
| `idx`, `dog_whistle`, `dog_whistle_root`, `ingroup`, `definition` | term metadata |
| `example` | the text (note: column is `example`, not `content`) |
| `label` | `coded` / `non-coded` |

Used to evaluate the binary disambiguator at the end of training.

### `silent_signals_disambiguation` (124 rows)
| Column | Notes |
|---|---|
| `type`, `dog_whistle`, `dog_whistle_root`, `ingroup`, `definition` | term metadata + register |
| `content` | the text |
| `label` | three values (mapped to binary in [notebooks/pipeline.ipynb](../notebooks/pipeline.ipynb)) |

Same role: locked eval, never seen in training, content-hash-excluded from negatives.

**Coverage caveat.** Both eval sets are tiny (101 and 124 rows). Their per-ingroup share differs from `silent_signals` by up to ~10pp on some classes (cell 46 of the EDA notebook). Any single-number F1 on them will have wide confidence intervals; we should report per-class breakdowns, not just headline numbers.

---

## 6 · Positives-only problem & negatives mining plan

### The problem, stated bluntly
- `silent_signals` has **zero negatives**. Training a binary classifier on it alone is degenerate.
- The candidate pool has ~6.9M never-adjudicated rows. We don't know which are literal and which are unlabelled coded uses.
- `silent_signals` itself has ~15% label noise (Kruk et al. report 95.7% GPT-4 precision on disambiguation, 85.3% on a 400-row manual audit) — see [CONCERNS.md §Concern 3](CONCERNS.md). Both sides of our binary data are noisy; this plan addresses the negatives side.

### Where the implementation lives
The full pipeline is driven by **[notebooks/negatives.ipynb](../notebooks/negatives.ipynb)** (smoke + orchestration) plus the production HPC scripts [scripts/hpc/mine_negatives_full.py](../scripts/hpc/mine_negatives_full.py), [scripts/hpc/adjudicate_negatives_vllm.py](../scripts/hpc/adjudicate_negatives_vllm.py), and [scripts/hpc/balance_negatives_full.py](../scripts/hpc/balance_negatives_full.py). The shipped output is `data/processed/{positives,negatives}_balanced_{train,val,test}.parquet`, which the binary trainer in [scripts/hpc/train_binary_disambiguator.py](../scripts/hpc/train_binary_disambiguator.py) consumes (or its `data/final/rq_a_binary/` derivative).

### Stage 1 — heuristic silver mining
[scripts/hpc/mine_negatives_full.py](../scripts/hpc/mine_negatives_full.py) does:
1. Build an exclusion set of content hashes for all `silent_signals` rows + both locked eval sets.
2. Stream `informal_potential_dogwhistles`, keep rows whose `dog_whistle` field matches one of the surface forms in our vocab.
3. Cap per-term at `min(max(K_positives × buffer_factor, MIN_FLOOR), pool_count, HARD_CEILING)` with `buffer_factor=3`, `MIN_FLOOR=3`, `HARD_CEILING=2000`.
4. Deduplicate by content hash; treat all survivors as label = 0.
5. Inherit the split of each negative from the `dog_whistle_root` → split mapping (no leakage).

The shipped Stage 1 raw output is [data/processed/negatives_stage1_raw.parquet](../data/processed/) with **37,497 candidates**. This is silver: no evidence each row is actually non-coded, no per-term balance — Stages 2 and 3 fix that.

### The defensible four-stage plan

The plan strengthens the silver-mining step by mirroring the original paper's adjudication methodology, then enforcing balance the original paper did not.

#### Stage 1 — Heuristic silver pool (re-done with adaptive cap)
The shipped per-term cap formula `min(max(K_positives × buffer_factor, MIN_FLOOR), pool_count, HARD_CEILING)` (with `buffer_factor=3`, `MIN_FLOOR=3`, `HARD_CEILING=2000`) replaced the early flat cap of `10` so high-frequency terms like `SJW` (~1k positives) get a proportionally larger candidate pool. Implemented in [scripts/hpc/mine_negatives_full.py](../scripts/hpc/mine_negatives_full.py); orchestration in [notebooks/negatives.ipynb](../notebooks/negatives.ipynb). Output: `data/processed/negatives_stage1_raw.parquet`.

#### Stage 2 — LLM-as-judge adjudication
Implemented in [scripts/hpc/adjudicate_negatives_vllm.py](../scripts/hpc/adjudicate_negatives_vllm.py) (production) with a smoke stub in [notebooks/negatives.ipynb](../notebooks/negatives.ipynb). The judge runs over the silver pool with the **same disambiguation prompt** Kruk et al. used for positives, and **keeps only candidates labelled `non-coded` with high confidence**.

**Judge model chosen:** `meta-llama/Llama-3.1-8B-Instruct` via vLLM (batch size 256, offline mode), running on a 40 GB A100 MIG slice. The free open-weight option won out over GPT-4o-mini on cost/reproducibility grounds; a closed-model cross-check was scoped but not run.

Output: a `judge_label` column appended to each negative row, plus the [data/manifests/negatives_adjudication_report.json](../data/manifests/negatives_adjudication_report.json) manifest reporting:
- **Contamination rate: 62.1%** of the 37,497 Stage 1 candidates were judge-flagged as actually coded and dropped. This is itself a headline finding for the writeup — the candidate pool has far more unlabelled coded uses than the original Kruk et al. adjudication captured.
- **Unparseable rate: 0.34%** of judge outputs.

This stage exists in CONCERNS.md as the "ideal fix" for Concern 1.

#### Stage 3 — Strict per-term 1:1 matching, paired output
Implemented in [scripts/hpc/balance_negatives_full.py](../scripts/hpc/balance_negatives_full.py). For each surface form `X` with `K` positives across the splits, let `n_clean` be the count of post-judge `non-coded` candidates whose surface form is `X`. Set `n_pair = min(K, n_clean)` and:

- pick the top `n_pair` judge-confidence negatives for `X`
- random-sample `n_pair` of `X`'s positives (fixed seed)
- assign both to the split forced by `X`'s root (since each root lives in exactly one split)
- if `n_clean < MIN_NEGATIVES_PER_TERM` (default 3), drop `X` from the binary set entirely — *both* its positives and its (zero or near-zero) negatives. The dropped positives still feed the multiclass + generation tasks, which read the unfiltered `data/splits/{split}.parquet`.

The shipped output is paired: `data/processed/positives_balanced_{split}.parquet` and `data/processed/negatives_balanced_{split}.parquet`. By construction `len(pos) == len(neg)` per split *and per term*. Final counts: **8,828 pairs** total (train 6,183, val 1,261, test 1,384). 100 terms (1,037 positives) were dropped at the per-term floor — full list in [data/manifests/negatives_adjudication_report.json](../data/manifests/negatives_adjudication_report.json). The binary trainer reads `data/final/rq_a_binary/{train,val,test}.parquet`, which is the canonical re-bundled form of these.

This addresses both [CONCERNS.md §Concern 2](CONCERNS.md) (term-identity shortcut) and [CONCERNS.md §Concern 4](CONCERNS.md) (pejorative-only terms — the floor catches them automatically even if the upstream `coded_share` flag is mis-tuned). The previous design balanced 1:1 at the *split* level only; a model could still learn "term `tranny` is usually positive" because we kept all 200 of its positives but produced 0–3 negatives. Paired per-term matching forecloses that shortcut.

#### Stage 4 — Gold negatives at evaluation
The locked eval sets supply the only human-verified non-coded examples we have. We never train on them, but they are the negatives that bound our reported performance. Confidence intervals on metrics computed from 101 + 124 rows must be reported alongside point estimates.

### Why this is defensible in the writeup

1. **Methodology symmetry with the source paper.** Kruk et al. used LLM adjudication to *create* `silent_signals` from raw candidates. We use the same procedure to *create* our negatives from the leftover candidates. We are not inventing a new approach — we are completing the one they didn't run on the negative half.

2. **Quantified, not assumed.** Stage 2 produces an explicit contamination estimate. We can write *"the heuristic silver pool was X% contaminated by likely-coded uses, removed before training"* rather than *"we hope the negatives are clean"*.

3. **Term-identity shortcut killed by construction.** Stage 3 makes per-term marginal P(label=1) ≈ 0.5 by design. The model cannot learn term identity as a proxy for label.

4. **Honest evaluation floor.** Stage 4 keeps ground-truth gold negatives entirely outside training. Any reported F1 on the detection set is therefore not a measurement of training-distribution overfitting — it is a real generalisation number, with a tight enough sample size that we can report binomial CIs.

5. **Failure modes are themselves results.** If Stage 2 reports >25% contamination, that *is* the headline finding for the paper's data-quality discussion — the candidate pool has more unlabelled coded uses than the original adjudication captured. If per-term coverage in Stage 3 is sparse, we report which terms are evaluable and which are data-starved.

### Alternatives we considered and rejected (or kept as ablations)

| Option | Verdict |
|---|---|
| Use `silent_signals_disambiguation`'s non-coded rows as training negatives | **No.** Tiny (~30 non-coded rows max), and burning the eval set as training data destroys our only honest evaluation. |
| Hard negatives via embedding similarity to positives | Defensible as a *secondary* training set, not the primary negatives. Risks teaching the model that "context similar to a coded use → not coded", which inverts the signal. Park as an ablation. |
| Time-window / subreddit-matched negatives (sample literal uses from the same subreddit + week as each positive) | Strong as a *control* against register/subreddit confounds. Worth doing as an ablation report (does the model degrade when negatives can't be distinguished by subreddit?). Adds an extra mining axis we don't strictly need for the headline result. |
| Synthetic negatives from an LLM (prompt: "rewrite this coded use as a literal use") | **No.** Distribution shift — we'd be testing whether the model can detect synthetic vs. real text, not whether it can detect coding. |
| Treat unmatched candidate-pool rows as negatives unfiltered | What the old project did. Already known broken — produces the contamination problem we are solving. |

### The pejorative-only edge case

A subset of surface forms (slurs, manosphere coinages, white-nationalist tags) have `coded_share` ≈ 1.0 in the candidate pool — every occurrence in the wild is the coded use, by construction. We cannot mine literal negatives for them no matter how good the judge is. These terms are flagged via [data/manifests/term_coded_share.csv](../data/manifests/term_coded_share.csv) (produced by [notebooks/negatives.ipynb](../notebooks/negatives.ipynb)), dropped from binary training, and reported as a known-restricted scope. They stay in the multiclass and generation tasks where the framing remains well-posed. Full argument in [CONCERNS.md §Concern 4](CONCERNS.md).

### Concrete deliverables produced

- [data/processed/positives_balanced_{train,val,test}.parquet](../data/processed/) — positives subsampled to per-term parity with the matching negatives. Pejorative-only positives whose term was dropped at the floor are *not* present here (they remain in `data/splits/{split}.parquet` for the multiclass + generation tasks).
- [data/processed/negatives_balanced_{train,val,test}.parquet](../data/processed/) — adjudicated, per-term-balanced 1:1 negatives. Same row count as `positives_balanced_*` by construction.
- [data/final/rq_a_binary/{train,val,test}.parquet](../data/final/rq_a_binary/) — the canonical re-bundled form of the above, what the binary trainer actually reads.
- [data/manifests/negatives_adjudication_report.json](../data/manifests/) — judge model, contamination rate (62.1%), unparseable rate (0.34%), per-split counts, dropped-term list with reasons, total positives lost to per-term floor (1,037).
- [data/manifests/term_coded_share.csv](../data/manifests/) — `coded_share` per surface form + `pejorative_flag`. Load-bearing for both binary scope and the writeup.

---

## 7 · Definition concatenation plan

### Why definitions matter
Task A is framed as **NLI-style disambiguation**: given a candidate term, its meaning, and a passage, decide whether the term is being used in its coded sense in this passage. The "meaning" half of that input is the definition. A model with no definition has to learn from term identity alone — which is exactly the shortcut Stage 3 of the negatives plan exists to prevent. Concatenating a definition gives the model a non-shortcut signal it can actually use.

### Source of definitions
Two columns are available for each `dog_whistle_root`:

| Column | Source | Length | What it gives the model |
|---|---|---|---|
| `definition` | already in `silent_signals` | 1 sentence (e.g. *"Lesbian, gay, and bisexual, but not trans, rights"*) | The covert meaning, terse |
| `glossary_description` | [resources/dog_whistle_roots_enriched.csv](../resources/dog_whistle_roots_enriched.csv) | 1–3 paragraphs | The covert meaning + register, history, in-group context |

The enriched CSV also exposes `glossary_example`, `glossary_type`, `glossary_register`, `glossary_speaker`, `glossary_date`. Coverage is **298/298 roots resolved** (297 exact + 1 alphanum match, see [resources/README.md](../resources/README.md)).

### The plan

#### Step 1 — Attach the enriched columns at data prep time
The enriched glossary columns are left-joined onto the balanced positives/negatives parquets on `dog_whistle_root` and persisted into [data/final/rq_a_binary/{train,val,test}.parquet](../data/final/rq_a_binary/). Splits are byte-stable — joining is on the group key, so the existing root→split mapping is preserved.

#### Step 2 — Two arms for the RQ-A ablation (D6)
Per [DESIGN_DEFENSE.md D6](DESIGN_DEFENSE.md):

| Arm (`input_format`) | `Candidate meaning` source | Hypothesis |
|---|---|---|
| **`term`** | (no definition concatenated) | sufficient for unambiguous terms; the term itself carries the signal |
| **`term_enriched_def`** | `definition_enriched` (long glossary paragraph) | needed for ambiguous terms where the surface form alone is too thin |

This is **not a new research question** — it's a definition-richness ablation slotted into RQ-A. We report Δ-F1 between the two on the locked eval sets.

#### Step 3 — Where the concatenation happens in code
`build_binary_input` lives in [scripts/hpc/train_binary_disambiguator.py](../scripts/hpc/train_binary_disambiguator.py) and switches on the `input_format` config field (`"term"` vs `"term_enriched_def"`). The arm is selectable at run time via the `--input_format` CLI flag and is appended to the output dir as `format_<arm>/seed_<n>/`. See [scripts/configs/binary.yaml](../scripts/configs/binary.yaml) for the default.

#### Step 4 — Token budget
- `definition` is ~10–25 tokens. RoBERTa-base at `max_length=256` would be fine.
- `definition_enriched` is ~80–180 tokens; with a 100-token Reddit comment that pushes us past 256.
- **All shipped runs use `max_length=512`** (the default in [scripts/configs/binary.yaml](../scripts/configs/binary.yaml)) on RoBERTa-base, so truncation is rare.

#### Step 5 — Generation task input
The generation trainer in [scripts/hpc/train_generator.py](../scripts/hpc/train_generator.py) builds its prompt as `"Text: {content}\nMatched term: {dog_whistle}\n..."` and uses the enriched glossary fields (`definition_enriched`) as input context. The target JSON's `definition` field stays at the short form so ROUGE-L on the explanation field remains meaningful.

---

## 8 · Decisions resolved before training

1. **Judge model.** Llama-3.1-8B-Instruct (free, on-cluster) was chosen over GPT-4o-mini on cost/reproducibility grounds. The closed-model cross-check on a stratified subsample was scoped but not run.
2. **Per-term floor.** `MIN_NEGATIVES_PER_TERM = 3`. Any term with fewer than 3 adjudicated `non-coded` candidates is excluded from binary training and flagged in [data/manifests/negatives_adjudication_report.json](../data/manifests/negatives_adjudication_report.json); 100 terms (1,037 positives) were dropped this way.
3. **Eval-set filtering.** [CONCERNS.md §Scoping Decision](CONCERNS.md) flagged Reddit-only training but kept eval mixed. Stayed mixed; formal-row degradation is a reportable finding.
4. **Definition for negatives.** Negatives have a `dog_whistle_root` (mined by surface match), so the enriched-definition join works. The definition for a literal use is conceptually a coded-meaning hypothesis being tested against the literal text — the right framing for NLI-style disambiguation, flagged in the writeup.

---

## Source

- [Kruk et al. (2024), "Silent Signals, Loud Impact: LLMs for Word-Sense Disambiguation of Coded Dog Whistles", ACL 2024.](https://aclanthology.org/2024.acl-long.675)
- HuggingFace dataset cards for the five SALT-NLP datasets above.
- Local artefacts: [notebooks/eda.ipynb](../notebooks/eda.ipynb), [CONCERNS.md](CONCERNS.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [NOTEBOOK_IMPLEMENTATION.md](NOTEBOOK_IMPLEMENTATION.md), [resources/](../resources/).
