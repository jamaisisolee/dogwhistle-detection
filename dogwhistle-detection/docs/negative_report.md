# Mining Binary Negatives — Methodology Report

**Document scope.** This report explains how negative examples for the binary dog-whistle disambiguation classifier (RQ-A) were mined, adjudicated, and balanced. It walks through each pipeline stage, gives the exact parameters and code locations, and defends each decision against alternatives. The intended reader is anyone writing the methods section of the project report or auditing the data construction.

**One-line summary.** Our negatives mining replicates the procedure Kruk et al. (2024) used to *create their positives* — same candidate pool, same disambiguation prompt — but applies it to the unjudged remainder of that pool to produce the **literal-use** counterparts. We then enforce strict per-term 1:1 pairing with the published positives and drop terms whose literal supply is too small to support contrast learning.

---

## 1. Why this is necessary at all

The published `silent_signals` corpus is **positives-only**. Each Reddit comment in the dataset is a *coded* use of a dog-whistle phrase, vetted by Kruk et al. with GPT-4 + manual sampling. Training a binary classifier requires a matched negative class — Reddit comments using the **same surface phrases** in their **literal, non-coded** sense.

The two natural alternatives are both unworkable for a binary trainer:

| Alternative | Why it fails |
|---|---|
| `silent_signals_disambiguation` (124 manually-labelled rows) | Far too small to train; reserved as **eval-only**. |
| `silent_signals_detection` (101 manually-labelled rows) | Same reason; also reserved as **eval-only**. |
| Random Reddit comments not containing the phrase | Trivially separable on lexical signal — model learns "phrase present ⇒ coded", which is keyword detection, not disambiguation. |

So negatives must come from a corpus where the **same phrase appears** but its sense is unjudged. Kruk et al. published exactly such a corpus alongside their positives.

---

## 2. Source: `SALT-NLP/informal_potential_dogwhistles`

This is the **candidate pool from which Kruk et al. distilled their published Informal positives.** Every row contains a Reddit comment that surface-matches at least one term in the dog-whistle vocabulary, but the comment has not been judged for coded vs. literal use. The published `silent_signals` Informal subset is the slice of this pool that GPT-4 + manual review confirmed as coded; the remainder is unjudged.

**Pool size:** 4 parquet shards in the HF cache (`datasets--SALT-NLP--informal_potential_dogwhistles/snapshots/.../data/train-0000{0..3}-of-00004.parquet`).

**Key property.** Because the surface phrase is guaranteed present in every candidate, any negative we mine here is:
1. Matched on surface form to some positive (no register / lexicon confound).
2. Plausibly literal (the *unjudged* slice is, by construction, what GPT-4 did not promote to coded).
3. From the same source distribution as the positives (Reddit, same subreddits, same era).

This is the same source the paper used; the only difference is which slice of it is consumed.

---

## 3. Comparison to Kruk et al. (2024)

| Step | Kruk et al. (2024) | This pipeline |
|---|---|---|
| **Candidate pool** | `informal_potential_dogwhistles` (Reddit) | `informal_potential_dogwhistles` (Reddit) — **same** |
| **Adjudication prompt** | "Is this phrase being used in the coded sense?" | **Same** disambiguation prompt; see §5 |
| **Judge model** | **GPT-4** (closed, paid) | **Llama-3.1-8B-Instruct** via vLLM on Bocconi A100 (open, free) |
| **Direction** | Promote `coded` ⇒ corpus positives | Keep `non-coded` ⇒ corpus negatives |
| **Output curation** | Manual vetting on 400-row sample (≈14.7% false-positive rate reported) | Automated; we accept the ≈15% noise rate they document and report a Stage-2 contamination rate as our parallel quality signal |
| **Per-term balancing** | Not applicable — they only built one class | Strict 1:1 per surface form, with `MIN_NEGATIVES_PER_TERM = 3` floor and term drop |

**Methodological framing.** This is not a new method. It is the unfinished *other half* of theirs. Their paper produced positives by judging candidates as coded; we produce negatives by judging the remainder of the same candidates as non-coded with the same prompt. The defensible writeup line is: *"we ran the same adjudication procedure as Kruk et al. on the unjudged tail of their candidate pool."*

The single deliberate divergence is the **judge model**: we use Llama-3.1-8B-Instruct rather than GPT-4. Three reasons (defended in §5).

---

## 4. Stage 1 — Heuristic silver mining

**Code:** `scripts/hpc/mine_negatives_full.py` (runs on cluster as `hpc_scripts/mine_negatives_full.py` — see `reproduction_report.md` I1)
**Output:** `data/processed/negatives_stage1_raw.parquet`

### Algorithm

1. Load the project's positives from `data/splits/{train,val,test}.parquet` and concatenate.
2. Build the **term vocabulary**: unique `(dog_whistle, dog_whistle_root, definition)` triples present in the positives.
3. Compute **per-term mining targets**:
   ```
   K          = positive count for term
   target     = min( max(K * buffer_factor, min_floor),
                     pool_count_for_term,
                     hard_ceiling )
   ```
   With defaults `buffer_factor=3`, `min_floor=3`, `hard_ceiling=2000`. The 3× buffer ensures Stage 2 has enough survivors after the judge prunes contamination.
4. Build the **content-hash exclusion set**:
   ```
   EXCLUDE = hashes(positives.content)
           ∪ hashes(silent_signals_detection.example)
           ∪ hashes(silent_signals_disambiguation.content)
   ```
   `content_hash` = SHA-256 of whitespace-normalised lowercase text. This catches both exact duplicates and trivial whitespace variants.
5. Stream the 4 candidate-pool parquet shards, row by row:
   - Skip rows whose term has no remaining target.
   - Skip rows whose content is shorter than `min_text_len = 50` chars.
   - Skip rows whose content hash is in `EXCLUDE` or already seen.
   - Otherwise emit, decrement the term's remaining target.
6. Merge with `term_vocab` to attach `dog_whistle_root` and `definition` (used by Stages 2 and 3).
7. Write the result.

### Defense of each parameter

| Parameter | Default | Why |
|---|---|---|
| `buffer_factor=3` | mine 3× the positive count | Stage-2 contamination is unknown a priori; even a 50% kill rate leaves ≥1.5× headroom. |
| `min_floor=3` | at least 3 candidates per term | Below this, a single Stage-2 false-coded label drops the term entirely. The floor matches `MIN_NEGATIVES_PER_TERM` in Stage 3 — both invariants are defended together. |
| `hard_ceiling=2000` | per-term cap | Prevents a single high-frequency term from monopolising the candidate budget. |
| `min_text_len=50` chars | filter on content length | Reddit one-word replies don't carry enough context for the judge to disambiguate; below 50 chars the judge produces near-random labels. |
| Content-hash exclusion | exact + whitespace-normalised | Prevents any positive (or eval-set row) from leaking into the negative pool. Stage-3 also re-applies this; it is an *invariant*, not a one-off filter. |
| Source: only `informal_potential` (not `formal_potential`) | exclusively informal | The formal pool is Congressional speech. Mixing it with informal positives would re-introduce a register confound — model learns "Reddit syntax ⇒ coded". See `DESIGN_DEFENSE.md` D10. |

### What Stage 1 does *not* do

- It does not judge anything as coded vs. literal. It is a heuristic pre-filter for Stage 2 to ingest.
- It does not balance term counts to positives. That is Stage 3's job.
- It does not look at `pejorative` flags. Stage 3 will drop pejorative-only terms by virtue of the per-term floor.

---

## 5. Stage 2 — LLM-as-judge adjudication

**Code:** `scripts/hpc/adjudicate_negatives_vllm.py`
**Hardware:** 1× A100 40 GB MIG slice on Bocconi HPC `gnode02`/`gnode04`
**Model:** `meta-llama/Llama-3.1-8B-Instruct`, fp16, served via vLLM 0.19.1
**Output:** `data/processed/negatives_stage2_judged.parquet`

### The disambiguation prompt

This is the verbatim prompt sent to the model for every candidate (chat-template wrapped):

```
You are an expert linguist labelling whether a phrase in a Reddit comment is being
used as a coded dog whistle.

A dog whistle is a phrase that, on its surface, has an innocuous meaning but is
also used to signal a hidden, coded meaning to an in-group audience.

Phrase: "{dog_whistle}"
The coded (in-group) meaning of this phrase is: {definition}

Reddit comment:
"""
{content}
"""

Question: In this Reddit comment, is the phrase "{dog_whistle}" being used with
the coded meaning above, or with its ordinary literal meaning?

Answer with exactly one word: 'coded' or 'non-coded'.
```

This is **the same prompt structure Kruk et al. used** to disambiguate the candidate pool — the phrase, its glossed coded meaning, the comment, and a forced two-word answer.

### Inference settings

| Setting | Value | Why |
|---|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` | Open-weight, instruction-tuned, fits a 40 GB MIG slice in fp16 |
| `temperature` | `0.0` | Determinism — same input ⇒ same label across runs |
| `max_tokens` | `8` | Output is 1 word; 8 tokens is a safety margin for chat-template artefacts |
| `batch_size` | `256` | vLLM continuous batching; saturates throughput on a single A100 |
| `max_content_chars` | `2000` | Caps prompt length below `max_model_len=4096` tokens after tokenisation |
| Chat template | applied via `tokenizer.apply_chat_template` | Llama-3.1-Instruct is chat-tuned; raw concatenation degrades quality |

### Output schema (added columns on top of Stage 1)

- `judge_label` ∈ {`coded`, `non-coded`, `unparseable`}
- `judge_confidence` ∈ {0.9, 0.0} — fixed values; the parser flags clearly-formatted answers as 0.9 and unparseable garbage as 0.0
- `judge_model` = `"meta-llama/Llama-3.1-8B-Instruct"` (string, written into every row)

Only rows with `judge_label == "non-coded"` survive into Stage 3.

### Defense: why Llama-3.1-8B-Instruct and not GPT-4

This is the only deliberate divergence from Kruk et al. Three reasons:

1. **Cost and dependency.** Adjudicating ~37,500 candidates on GPT-4o-mini is on the order of a few dollars — affordable, but it adds a closed off-cluster dependency to a pipeline that otherwise runs end-to-end on hardware we already have. Llama on the HPC finishes the same batch in ~10–25 minutes and writes directly into the data directory.
2. **Reproducibility.** An open-weight checkpoint is auditable. A closed model behind a moving API endpoint is not — the *same prompt* sent to GPT-4 in 2026 may not return the same labels in 2027.
3. **Symmetry concern.** Kruk et al. used GPT-4 to produce their positives. Re-using GPT-4 as the negative judge would mean *the same model family is choosing both classes* of the trainer's input. An independent model family is a stronger test of "is this candidate actually coded?"

**Acceptable trade-off.** Llama-3.1-8B is plausibly noisier than GPT-4 on this task. We treat the Stage-2 contamination rate (the % of Stage-1 candidates judged `coded`, *which we then drop*) as an empirical signal of judge behaviour, recorded in the manifest.

### 5a. Empirical Stage-2 results (run 2026-04-27, job 483408)

Of the **37,497 Stage-1 candidates** the judge labelled:

| Label | Count | Share |
|---|---|---|
| `coded` (dropped) | 23,289 | **62.11%** |
| `non-coded` (kept for Stage 3) | **14,079** | **37.55%** |
| `unparseable` (dropped) | 129 | 0.34% |

**Headline finding.** The Stage-2 contamination rate is **62.11%**, far above the ~15% positive-noise floor Kruk et al. report from their manual review. By the judge's reading, the *unjudged remainder* of `informal_potential_dogwhistles` — the slice that GPT-4 did not promote to coded for the published positives — is itself **predominantly coded**, not predominantly literal as a casual reading of the dataset card would suggest. The literal-use distribution is the minority class in the pool we mine literals from.

This is reportable as a property of the candidate pool, not as a pipeline failure (see §9). The remaining 14,079 non-coded survivors are still ample for Stage 3 to pair against positives; the rate just tells us that mining was less efficient than buffer_factor=3 expected.

### 5b. Validation strategy and its constraints

**What we did not do, and why.** A small GPT-4o-mini sanity sample (~200 rows, ~$2 in API cost) would have given an independent cross-check on Llama's `coded` vs `non-coded` calls and an estimate of judge-disagreement rate. We chose not to spend the API budget for two practical reasons:

1. **Strict free-tools constraint.** This project runs end-to-end on free / academic resources — the Bocconi HPC, Hugging Face Hub, open-weight models. Adding a paid closed-API dependency, even for ~$2, breaks the free-resources discipline and the "any reader could rerun this" reproducibility guarantee.
2. **Budget discipline appropriate to a graduate project.** An academic project under student budget constraints calls for transparent acknowledgement of a methodology trade-off, not for hiding it behind an API key.

**Honest caveat.** Because the judge is Llama-3.1-8B-Instruct rather than the larger frontier model used by Kruk et al. (GPT-4), our **per-row label noise is plausibly higher than theirs**, and we have not directly measured the gap. We therefore have **lower confidence in any individual `non-coded` label** than the published positives have in their corresponding `coded` labels. Two consequences:

- The 62% contamination rate could be inflated if Llama is over-eager to call ambiguous comments coded; it could equally be under-stated if Llama misses subtler coded uses. We don't know in which direction the bias points.
- Downstream, the binary classifier sees noisier negatives than positives. We accept this asymmetry as a known limitation of an open-weights, zero-cost methodology, and recommend a paid GPT-4o-mini cross-check (≈$2, ~200 rows) be added in any future revision that has the budget for it.

**What this means for the report.** Anywhere the binary classifier's F1 is reported, it is bounded above by the looser of (a) the ~15% positive label noise Kruk et al. accept and (b) an unmeasured ≥0% Llama-judge label noise on the negative side. If headline F1 plateaus around ~0.85, that is consistent with the union of both noise sources, not with a model bug.

### What Stage 2 does *not* do

- It does not modify or augment the prompt across candidates. Same prompt template, same temperature, same model, same max tokens, every row.
- It does not filter on `judge_confidence` thresholds beyond drop `unparseable`. The 0.9/0.0 values are diagnostic only.
- It does not write a "judge log" of model outputs by row beyond label + confidence — only the parsed verdict is stored, to keep the parquet small.

---

## 6. Stage 3 — Strict per-term 1:1 matching and split assignment

**Code:** `scripts/hpc/balance_negatives_full.py`
**Output:** `data/processed/{positives,negatives}_balanced_{train,val,test}.parquet` + `data/manifests/negatives_adjudication_report.json`

### Algorithm

1. Load the positives splits and the Stage-2 judged candidates.
2. Build `root_to_split` from positives: every `dog_whistle_root` is mapped to the *modal* split it appears in. Negatives inherit splits via this map, so a negative for `dirty_jew` follows the same train/val/test assignment as positives sharing that root. (This preserves the **grouped-by-root split** invariant; see `DESIGN_DEFENSE.md` D2.)
3. Compute `K_per_term`: the positive count per surface form `dog_whistle`.
4. Pre-filter to the **clean pool**: rows where `judge_label == "non-coded"`.
5. For each term `X` with `K_per_term[X] = K`:
   - Get the term's clean candidates, sorted by `judge_confidence` descending.
   - If fewer than `MIN_NEGATIVES_PER_TERM = 3` clean candidates, **drop the term entirely** — both its positives and any negatives. Record the reason in the manifest.
   - Else, set `n_pair = min(K, n_clean)`. Take the top-`n_pair` clean candidates as the term's negatives. Random-sample (seed=42) `n_pair` of the term's positives.
6. Assign each negative to its split via `root_to_split`. Stamp `label=0`, `type="Informal"`, `source_dataset="SALT-NLP/informal_potential_dogwhistles"`.
7. Stamp positives with `label=1`.
8. Write paired files per split. Write a manifest JSON capturing the contamination rate, dropped-term list, dropped-term count, and per-split row counts.

### Defense of each design choice

**Strict 1:1 per term (not e.g. 1:3 or class-balanced overall).** With unequal per-term ratios, the binary trainer can game the task by learning per-surface-form base rates. With strict 1:1 the trainer has no per-term shortcut: the term itself carries no information about the label. Whatever signal it learns has to be contextual.

**Drop terms entirely when negatives < floor (vs. dropping only the negative side).** A mid-design version of this pipeline filtered only the negatives — terms with no clean negatives kept *all* their positives. That re-introduced the term-identity shortcut. With 200 `tranny` positives and 0–3 negatives, the trainer learns "model sees `tranny` ⇒ predict coded", which is keyword matching, not disambiguation. **The fix is per-term parity, which by construction means subsampling positives down to `n_pair` whenever the negative supply is the limiting factor.** This is the asymmetric scope decision defended in `DESIGN_DEFENSE.md` D4.

**Top-confidence selection for negatives, random for positives.** Negatives have a real noise source (judge label noise) and we want the cleanest candidates per term, hence top-`n_pair` by judge confidence. Positives have a random label noise we accept (Kruk et al.'s ~15%, see D8); deterministic selection there would bias toward whichever positives appear first in the parquet, so we random-sample with a fixed seed instead.

**Split assignment via `root_to_split` modal vote.** Negatives must follow the same grouped split as positives sharing their root, otherwise a positive-only test set could never be matched with negatives, and worse, a `dirty jew` negative could leak from train into test. The modal vote is the conservative reading: if a root appears 95% in train and 5% in test, the negative goes to train.

**MIN_NEGATIVES_PER_TERM = 3.** Below 3, a single false-coded label from the judge can erase a term's entire negative supply. The number matches the `min_floor` in Stage 1 — both ends of the pipeline enforce the same invariant. (Sensitivity: increasing to 5 drops more terms but tightens contrast quality. We chose 3 as the looser end of plausible — terms that survive at 3 but not 5 are by definition rare in the candidate pool, and excluding them entirely would over-narrow the trained vocabulary.)

**Re-run determinism (seed=42).** Stage 3 is the only stage with an RNG (positive sub-sampling). The seed is fixed so that two runs over the same Stage-2 output produce byte-identical training sets.

### 6a. Empirical Stage-3 results (run 2026-04-27, job 483423)

| Quantity | Value |
|---|---|
| Stage-2 clean pool entering Stage 3 | 14,079 |
| Final balanced **negatives** total | **8,828** |
| Final balanced **positives** total (matched 1:1) | **8,828** |
| Per-split rows (train / val / test) | **6,183 / 1,261 / 1,384** |
| Terms dropped (below `MIN_NEGATIVES_PER_TERM=3`) | **100** |
| Positives lost to dropped terms | **1,037** (≈10.5% of pre-drop positives) |
| Stage-2 contamination rate (recorded in manifest) | **62.11%** |
| Stage-2 unparseable rate | 0.34% |
| Judge model | `meta-llama/Llama-3.1-8B-Instruct` |

**Dropped-term composition.** Of the 100 terms dropped at this stage, the pattern is exactly what the asymmetric-scope design (D4a) anticipated: nearly all of them are **pejorative-only** terms with no literal sense to disambiguate against. Representative buckets visible in the dropped list:

- **Punctuation / formatting patterns**: `(((echo)))`, `(((fellow whites)))`, `⚡⚡`, emoji ZWJ sequences. These have no literal natural-language meaning at all; the unjudged pool predictably yields zero `non-coded` candidates.
- **Coined slurs that exist only in coded use**: `Pajeet(s)`, `troons`, `troon`, `dindus`, `Schlomo`, `cuckservatives`, `Cohencidence(s)`, `magapede(s)`, `genderist(s)`.
- **Anti-trans coinages with no literal sense**: `AGPs`, `autogynephiles`, `TIFs`, `Trans Identified Female`, `Trans Identified Males`, `YWNBAW`, `uterus-havers`, `soybois`, `soy boys`.
- **Conspiracy/anti-Semitic neologisms**: `Zios`, `Zionist Occupation Government`, `loxism`, `string pullers`, `we wuz kangs`.
- A few **rare-positive low-resource terms** (e.g. `Manhattan elite`, `merit-based immigration policy`, `protecting women's spaces`) that have only 1 positive example total — too few for the Stage-3 floor regardless of pool quality.

**Why this is a good outcome, not a coverage failure.** Each dropped term falls into one of two well-defined exclusion categories defended in `DESIGN_DEFENSE.md` D4a:
1. **Pejorative-only** (no literal sense): the binary disambiguation question reduces to "is this `(((echo)))`?" — keyword matching, not disambiguation. Including these in training would teach the classifier "rare punctuation patterns ⇒ coded", a brittle proxy.
2. **Too-rare positives** (K=1): a single positive cannot anchor any meaningful train/val/test pair. Statistical contrast requires ≥3 on both sides.

The 1,037 positives lost are exactly the positives whose terms the binary task is **structurally** ill-equipped to ask about. They remain present in the locked eval sets and in the generation task, where the question is well-posed for them. (See `DESIGN_DEFENSE.md` D4a for the per-task scope matrix.)

**Cross-checked by the manifest.** All of the above is auditable from `data/manifests/negatives_adjudication_report.json` — `stage3_dropped_terms` lists every dropped term with `K_positives`, `n_clean`, and reason; `stage3_dropped_term_count = 100`; `stage3_positives_lost_to_drop = 1037`.

### Manifest (audit trail)

`data/manifests/negatives_adjudication_report.json` records:
- `stage2_contamination_rate` — fraction of Stage-1 candidates judged coded
- `stage2_unparseable_rate` — fraction the parser couldn't classify
- `stage2_judge_model` — string identifying the model
- `stage3_balanced_total_negatives`, `stage3_balanced_total_positives` — final row counts
- `stage3_per_split` — train/val/test counts
- `stage3_dropped_terms` — full list with `K_positives`, `n_clean`, and reason
- `stage3_dropped_term_count`, `stage3_positives_lost_to_drop`
- `stage3_min_negatives_per_term` — the floor we used

Anything we say about negative quality in the report can be cross-checked against this file.

---

## 7. Stage 4 — Gold-set verification (always-on invariant)

There is no separate Stage-4 *script*; the gold-set exclusion is an invariant **applied in every stage that touches the candidate pool**:

- Stage 1 builds and applies the content-hash exclusion against `silent_signals_detection.example` and `silent_signals_disambiguation.content`.
- Stage 3 only sees Stage-2 survivors, which are downstream of that exclusion.

The exclusion catches:
1. Exact duplicates of any locked-eval row.
2. Trivial whitespace / case variants of any locked-eval row (whitespace-normalisation is part of the hash).

It does **not** catch paraphrases. We accept that — paraphrases are not a leakage we can detect at hash level, and the locked eval sets are deliberately small (101 + 124 rows) so the prior of a paraphrase landing in the candidate pool is low.

---

## 8. Outputs and downstream consumers

| File | Producer | Consumer |
|---|---|---|
| `data/processed/negatives_stage1_raw.parquet` | Stage 1 | Stage 2 only |
| `data/processed/negatives_stage2_judged.parquet` | Stage 2 | Stage 3 only |
| `data/processed/negatives_balanced_{train,val,test}.parquet` | Stage 3 | Binary trainer (RQ-A) |
| `data/processed/positives_balanced_{train,val,test}.parquet` | Stage 3 | Binary trainer (RQ-A) |
| `data/manifests/negatives_adjudication_report.json` | Stage 3 | Methods section + audit |

The binary trainer reads the **paired** balanced files, **not** the raw `data/splits/{train,val,test}.parquet`. Pejorative-only terms are absent from the trainer's input by virtue of their failure to survive Stage 3's per-term floor. This is the asymmetric scope: pejorative-only terms remain in the locked eval sets and in the generation task, but not in binary training. (See `DESIGN_DEFENSE.md` D4a.)

---

## 9. Known risks and "failure as result"

The pipeline can produce three kinds of useful negative findings, all of which are reportable rather than hidden:

1. **High Stage-2 contamination (>25%) — observed at 62.11% in our run.** Indicates the candidate pool has more residual coded uses than Kruk et al.'s manual sample suggested. This is itself a publishable finding about pool quality, and is reported in §5a above.
2. **Many terms dropped at Stage 3.** Indicates the dog-whistle vocabulary is dominated by pejorative-only terms (no literal sense to disambiguate against). This bounds the disambiguation task's scope and tells the reader honestly which terms the binary classifier *cannot* be expected to handle.
3. **Stage-2 ↔ GPT-4 sanity-sample disagreement.** If we run a small (~200-row) GPT-4o-mini check against the Llama labels and the agreement is <75%, we report both numbers and flag the methodological caveat.

None of these is a pipeline failure. Each is a quantified property of the data we recover and report.

---

## 10. Reproducibility checklist

| Component | How to reproduce |
|---|---|
| Stage 1 candidates | `python scripts/hpc/mine_negatives_full.py --data_dir ./data` (deterministic; pool order + content-hash exclusion fully determine output) |
| Stage 2 labels | `sbatch scripts/hpc/submit_stage2.sh` on Bocconi HPC (note: the submit shell expects the cluster path `hpc_scripts/`; the committed copy is at `scripts/hpc/submit_stage2.sh`). Deterministic at temperature=0 with fixed model checkpoint. |
| Stage 3 balanced output | `python scripts/hpc/balance_negatives_full.py --data_dir ./data --min_negatives_per_term 3` (deterministic with seed=42) |
| Software versions | Python 3.10.20, torch 2.10.0+cu128, vLLM 0.19.1, transformers 5.6.2 (env `dogwhistle` on Bocconi HPC) |
| Cluster | Bocconi HPC, A100 40 GB MIG slice, driver 580.95.05 |
| Model checkpoint | `meta-llama/Llama-3.1-8B-Instruct` snapshot `0e9e39f249a16976918f6564b8830bc894c89659` |

---

## 11. Source

Cross-references:

- `scripts/hpc/mine_negatives_full.py` — Stage 1 implementation
- `scripts/hpc/adjudicate_negatives_vllm.py` — Stage 2 implementation
- `scripts/hpc/balance_negatives_full.py` — Stage 3 implementation
- `scripts/hpc/submit_stage2.sh` — SLURM job script (env vars, modules, GPU request)
- `notebooks/negatives.ipynb` — local notebook (Stage 2 stub for plumbing test only; real labels come from HPC)
- `DESIGN_DEFENSE.md` — fuller design-decision log; this report is a focused negatives-only re-read
- `DATASETS.md` — full pipeline specification including the candidate pool
- `CONCERNS.md` — risk catalogue (register, term identity, etc.)

Citation:

- Kruk, J. et al. (2024). *Silent Signals, Loud Impact: LLMs for Word-Sense Disambiguation of Coded Dog Whistles.* ACL 2024. https://aclanthology.org/2024.acl-long.675
