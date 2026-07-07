# Design Defense

Each major design choice in this project, and the argument for it. This document is the place to point at when a reviewer asks "why did you do X?". It pairs with [CONCERNS.md](CONCERNS.md) (which catalogues *risks*) — this file catalogues *decisions*.

Sections are independent; read in any order.

---

## D1. Reddit-only scope

**Decision.** All training and mining are restricted to `type == "Informal"` rows (Reddit). The formal subset of `silent_signals` (~3.3k Congressional speech rows) and the `formal_potential_dogwhistles` candidate pool are excluded from training. Locked eval sets are kept *unfiltered* — formal examples appear in the eval but never in training.

**Why.**
1. **Register confound.** Congressional speech and Reddit comments differ on lexicon, syntax, and topic distribution. Training on a mixed corpus risks the model learning `register == Formal ⇒ literal use` rather than learning disambiguation. This was raised early in the project, see [CONCERNS.md §"Scoping Decision: Reddit-only data"](CONCERNS.md).
2. **Volume sufficiency.** The informal subset is ~12.9k positives over 298 roots — already enough for grouped 70/15/15 splits with multiple positives per root in each split. Adding ~3.3k formal rows would double the distribution-shift risk without changing the order of magnitude of available data.
3. **Candidate-pool capacity.** The informal candidate pool is ~6M rows; even after content-hash exclusion, exclusion-of-positives, and per-term targeting we have plenty of slack. Adding the formal pool gives no marginal benefit.

**What we give up.** The model is not evaluated as a generic dog-whistle detector across registers; it is evaluated as a *Reddit* dog-whistle detector that we *also test* on Congressional speech without training on it. If formal-eval performance is poor, this is a reportable cross-register generalisation gap, not a failure of the method.

**Falsifier.** If, on the locked detection set, formal-row F1 is within 5 points of informal-row F1, this decision was conservative; we'd note that mixed-register training was unnecessary. If the gap is large, it confirms that mixing registers without a register feature would have been a worse choice.

---

## D2. Grouped splits by `dog_whistle_root`

**Decision.** Train/val/test splits are constructed with `StratifiedGroupKFold(group=dog_whistle_root)`. No surface form (and therefore no root) appears in more than one split. Stratification is on `ingroup`.

**Why.** With a per-row split, the same root (e.g. `the_dirty_jew`) would appear in train and test with different surface forms (`dirty jew`, `Dirty Jew!`). The model would memorise the root's lexical signal in training and "generalise" it in test by recognising the same root. That is term-identity inflation, not disambiguation. Grouping by root forces test-time generalisation to *unseen* lexical material, which is what RQ-A actually asks.

**Cost.** Some roots have only 1–2 positives and end up entirely in one split, which means certain ingroups are slightly under-represented in val/test. We accept this — better that than an artificially inflated headline number.

**Cross-check.** [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb) §2 asserts zero root overlap and prints per-split ingroup distributions. The split manifest (`data/manifests/splits.json`) records the exact assignment so the splits are reproducible.

---

## D3. Four-stage negatives pipeline (over a one-stage cap)

**Decision.** Negatives are produced by [notebooks/negatives.ipynb](notebooks/negatives.ipynb) as a four-stage pipeline:
1. **Stage 1 — heuristic silver mining** with adaptive per-term cap `min(max(K × buffer, MIN_FLOOR), pool_avail, HARD_CEILING)`.
2. **Stage 2 — LLM-as-judge adjudication** via vLLM + Llama-3.1-8B-Instruct, same disambiguation prompt as Kruk et al. (2024). Only `non-coded` survivors are kept.
3. **Stage 3 — strict per-term 1:1 matching, paired output**. For each surface form `X`: keep `n_pair = min(K, n_clean_post_judge)` of *both* positives and negatives; drop terms entirely when `n_clean < MIN_NEGATIVES_PER_TERM`.
4. **Stage 4 — gold-negative verification** against locked eval sets (content-hash exclusion).

**Why each stage is necessary:**

| Stage | What it prevents |
|---|---|
| 1 | Under-mining rare terms (would give zero negatives for tail). The flat-cap-of-10 used by the legacy notebook starves common terms relative to their positive counts. |
| 2 | Training the model on negatives that are themselves coded uses (Kruk et al.'s own manual vetting found 14.7% positive label noise; the candidate pool, being unjudged, is at least as contaminated). |
| 3 | Term-identity shortcut. Without per-term parity, a model can score well by memorising "term X usually labelled positive". |
| 4 | Eval-set leakage masquerading as generalisation. |

**Symmetric to Kruk et al.'s methodology.** Their original paper used GPT-4 to *create* the positives by adjudicating 100k candidates. We use the same prompt, against the same candidates, to *create* the negatives. This is not a new method — it's the unfinished other half of theirs. The defensive line in the writeup is "we ran the same adjudication procedure on the unjudged tail".

**Failure-as-result.** If Stage 2 reports >25% contamination, that *is* a publishable finding about the candidate pool. If certain terms have zero clean negatives, that itself is the data-availability story for the pejorative-only edge case.

---

## D4. Paired strict per-term 1:1 (positives subsampled, terms dropped)

**Decision.** Stage 3 of the negatives pipeline writes *paired* `positives_balanced_{split}.parquet` and `negatives_balanced_{split}.parquet`. The binary trainer reads these instead of the unfiltered `data/splits/{split}.parquet`. For each term `X`:

- If post-judge clean negatives `n_clean < MIN_NEGATIVES_PER_TERM` (default 3), drop the term entirely from binary training (positives **and** negatives).
- Otherwise, keep `n_pair = min(K, n_clean)` of each side: top-confidence negatives, random-sampled positives (fixed seed).

**Why this is the right call (not a hack).** A mid-design version of this pipeline filtered only the negatives — terms with no clean negatives kept all their positives in `data/splits/`. That re-introduces the term-identity shortcut Stage 3 exists to prevent: with all 200 `tranny` positives and 0–3 negatives, the trainer learns "model sees `tranny` ⇒ predict coded", which is keyword matching, not disambiguation. The fix is per-term parity, which by construction means subsampling positives down to `n_pair` whenever the negative supply is the limiting factor.

**What about the dropped positives?** They are *not* discarded from the project — they remain in `data/splits/{split}.parquet`, which is what the multiclass and generation tasks read. The argument is asymmetric:
- **Binary disambiguation** is only well-posed for terms that have a literal sense to disambiguate against. For pejorative-only terms it collapses to "is this `tranny`?" — keyword matching.
- **Multiclass ingroup** ("which group is being signalled?") and **explanation generation** ("explain the coded use") are well-posed even when the term is always coded. Pejorative-only positives are valid training data for those tasks.

So the asymmetric scope is the *correct* scope. Reporting binary F1 over a vocabulary of pejorative-only terms would conflate disambiguation with keyword detection and inflate the headline number.

**Reporting contract.** We report two binary F1 numbers:
1. **Full-vocab** — over whatever evaluable subset the locked eval sets contain.
2. **Ambiguity-restricted** — over terms with `coded_share < 0.5`.

The second is the methodologically honest disambiguation number; the first is a generous reading. Both are reported; neither is hidden.

**Manifest evidence.** `data/manifests/negatives_adjudication_report.json` records `stage3_dropped_terms` (the term list with reasons), `stage3_dropped_term_count`, and `stage3_positives_lost_to_drop` so the choice is auditable.

---

## D4a. What the asymmetric scope means task-by-task

D4 establishes that pejorative-only terms are dropped from the *binary* training set and survive in the multiclass and generation training sets. This section makes that concrete for each research question, because the asymmetry is the most important methodological claim in the project and the easiest one to misread. Closely paired with [CONCERNS.md §Concern 4](CONCERNS.md), which catalogues the *risk* that motivates this scoping.

### Where each task sees pejorative-only terms

| Task | RQ | Training data source | Sees pejorative-only terms in training? | Sees them at eval? |
|---|---|---|---|---|
| Binary disambiguator | RQ-A | `data/processed/positives_balanced_{split}.parquet` + `data/processed/negatives_balanced_{split}.parquet` (Stage 3 paired output, pejoratives dropped) | **No** | **Yes** — locked eval kept unfiltered |
| Multiclass ingroup | RQ-B | `data/splits/{split}.parquet` (unfiltered) | **Yes** | **Yes** |
| Explanation generation | RQ-C | `data/splits/{split}.parquet` (unfiltered) | **Yes** | **Yes** |

The asymmetry is intentional: the eval sets are the same human-labelled rows for all three tasks, but each task draws its training data from the source most appropriate for the question it answers.

### What the binary model is asked to do at eval anyway

Per [CONCERNS.md §Concern 4](CONCERNS.md), we keep the locked eval sets *unfiltered* — they include some pejorative-only terms with human labels. So:

- The binary model is tested on pejorative-only terms it has never seen in training.
- We expect it to underperform on those terms. That underperformance is real and reportable, not a flaw to hide.
- This is exactly why we report **two** F1 numbers (see also [§D9 below](#d9-dual-binary-f1-reporting-full-vocab--ambiguity-restricted)):
  - **Full-vocab F1** — over the full eval set, including the unseen pejoratives. The model will likely fail on those rows.
  - **Restricted-vocab F1** — over only `coded_share < 0.5` terms. This is the methodologically honest measure of whether the disambiguation framing works.

**The defence.** Binary disambiguation is only well-posed for terms with both senses. Asking the model *"is this the coded use of `(((echo)))`?"* is asking the wrong question — `(((echo)))` has only the coded use, so the answer reduces to *"is this `(((echo)))`?"*, which is keyword matching, not disambiguation. Including those terms in *training* would teach the model that "rare punctuation patterns ⇒ coded", a brittle proxy that doesn't transfer to ambiguous terms. Including them in *eval* keeps the report honest about coverage — we are not claiming the model handles them; we are showing what the model does on them, and explaining why.

### So is the project still expected to "classify and define" these terms?

Yes — but split by which task does the work:

- **"What ingroup is `(((echo)))` signalling?"** — RQ-B answers this. The multiclass model *has seen* the term in training with its `ingroup` label (anti-Semitic), so it can map term → group. This is a well-posed question for pejorative-only terms.
- **"Explain the coded use of `(((echo)))` in this comment"** — RQ-C answers this. The generation model has seen the term + its definition in training. Again well-posed: explaining a coded use does not require a contrasting literal use to exist.
- **"Is this comment using `(((echo)))` as a dog whistle vs. literally?"** — RQ-A *cannot* honestly answer this for pejoratives, because there is no literal use to disambiguate against. Reporting the binary model's accuracy on this question for `(((echo)))` as if it were a disambiguation result would be misleading; reporting that the question is ill-posed is correct.

### Why this is the right scoping (not a gap)

Each task's training data matches the question that task is supposed to answer:

- The binary task answers *"is this the coded sense of an ambiguous term?"* and is trained on terms where the question has a meaningful answer.
- The multiclass and generation tasks answer *"which group is being signalled?"* and *"explain the coded use"*, which remain meaningful even when only the coded sense exists.

Reporting binary F1 on a vocabulary that includes pejorative-only terms would conflate keyword detection (easy) with disambiguation (the actual research question), inflating the headline number. Reporting it on the disambiguation-relevant subset, alongside the full-vocab number for transparency, is the correct contract.

---

## D5. `coded_share` threshold for pejorative flagging

**Decision.** Surface forms with `coded_share = positives_in_silent_signals / candidate_pool_occurrences ≥ 0.5` are flagged as pejorative-only. Threshold is a knob, currently set at 0.5.

**Why a threshold and not a hand-picked list.** A threshold is reproducible, falsifiable, and dataset-derived. A hand-picked slur list is none of those things. It also requires sociolinguistic judgement we are not the right people to make.

**Why 0.5.** If more than half of the candidate-pool occurrences of `X` are confirmed coded (the `silent_signals` adjudication captured them), then by construction the *unjudged remainder* is at least as coded — there is essentially no literal-use distribution to mine. The 0.5 line is the point at which "mostly used as coded" stops being a contamination concern and starts being the modal use of the word.

**Backstop.** Even if the 0.5 threshold is mistuned, Stage 3's `MIN_NEGATIVES_PER_TERM` floor catches the same terms by a different route: they fail to produce enough clean negatives post-judge, so they're dropped regardless. The threshold is the *upstream* signal; the floor is the *downstream* invariant.

**Sensitivity sweep.** Before committing the binary headline number we will sweep the threshold on a small validation slice {0.3, 0.5, 0.7} and report the per-term F1 distribution under each. If the number is unstable across the sweep, we report the range, not a single point.

---

## D6. Two-arm robustness ablation (term vs term + enriched definition)

**Hypothesis under RQ-A.** "Can a fine-tuned LM identify when a phrase is being used as a dog whistle vs. literally?" — a capability question, not a comparative claim about definitions.

**Decision.** Two input formats are run at the same three seeds (`{42, 123, 7}`), giving 6 trained models total:

| Arm | Input format | Tests |
|---|---|---|
| `term` | `Candidate term: {dw}\nText: {content}\nQuestion: …` | the model relies on term identity + content alone |
| `term_enriched_def` | `Candidate term: {dw}\nCandidate meaning: {enriched_def}\nText: {content}\nQuestion: …` | adding curated background context (Allen AI `glossary_description`) changes performance |

**Why both, given the hypothesis is now non-comparative.** The two arms are a *robustness signal*, not a hypothesis test:
- If `term` alone matches `term_enriched_def`, the model is using the term and content; the result is more deployable (works on novel terms not yet glossed).
- If only `term_enriched_def` reaches the headline F1, the model depends on external curated knowledge — a deployment caveat to flag.
- The same-term contrastive pairing of positives and negatives (every dog-whistle root appears with both a coded use and a literal use sharing the same term and definition) defuses the obvious leakage path: the model cannot trivially predict from the definition alone, because the definition is held constant within each contrastive pair.

**Why enriched, not short.** The short `definition` column in `silent_signals` is often label-revealing on its face (e.g., `#BlueLivesMatter` → `"Black lives don't matter"`). The enriched `glossary_description` is a paragraph about register, history, and in-group context — richer context, less direct label leakage. The enriched variant also has full coverage on the locked eval sets (101/101 detection, 124/124 disambiguation), whereas the short variant is missing for 25/50 detection literals — a per-class artifact that would inflate the short-arm number. Running only the enriched arm cleanly avoids both confounds.

**Coverage check.** [resources/dog_whistle_roots_enriched.csv](../resources/dog_whistle_roots_enriched.csv) resolves 298/298 roots (297 exact, 1 alphanumeric match), so the enriched arm has zero coverage holes — see [resources/README.md](../resources/README.md). The column is pre-joined into every parquet in `data/final/rq_a_binary/` as `definition_enriched`.

**What we don't do.** We don't synthesise definitions from scratch with an LLM. The Allen AI glossary is already curated by humans for a published dataset, and synthetic definitions would introduce a definition-drift confound on top of the disambiguation question. We also no longer run the short-vs-enriched ablation that an earlier draft of D6 proposed — the short variant's coverage gap on `eval_detection` made it an unreliable arm and the new framing doesn't ask the question it was designed to answer.

---

## D7. Locked eval sets are never trained on

**Decision.** `silent_signals_detection` (101 rows) and `silent_signals_disambiguation` (124 rows) are loaded only to (a) build content-hash exclusions during negatives mining, and (b) compute test metrics. They are never tokenised into a training dataset.

**Why.** These are the only human-annotated examples we have. They are the only honest measure of model performance; everything else (the silver positives, the silver negatives) has known noise. If we contaminate them with training data we lose our only ground-truth instrument.

**How it's enforced.**
1. The Stage 1 streaming code in [notebooks/negatives.ipynb](notebooks/negatives.ipynb) §4 builds `EXCLUDE = hashes(positives) | hashes(detection.example) | hashes(disambiguation.content)` and drops candidates whose hash is in `EXCLUDE`.
2. Stage 3 asserts `bal_hashes & EXCLUDE == ∅` before writing the balanced parquet — a fatal-on-overlap check.
3. The HPC `mine_negatives_full.py` and `balance_negatives_full.py` carry the same exclusion logic.

**What this gives us.** When we report F1 on the detection set, the only inflation that can come from is silver-set noise (which will *deflate* our number if anything, by training the model on noisy boundaries). It cannot come from leakage. That makes the eval F1 a lower bound under the silver-data assumption, which is the right direction for a defensible number.

---

## D8. Acceptance of ~15% positive label noise

**Decision.** We accept the ~15% false-positive rate Kruk et al. report in their manual vetting of 400 silver-positive rows. We do not re-vet, do not relabel, do not filter. Our positives are exactly the published `silent_signals` Informal subset.

**Why.**
1. **Out of scope to refix the source dataset.** Re-running adjudication on 12.9k rows is a project-sized effort by itself.
2. **Symmetric noise on the eval side.** The locked detection / disambiguation sets were *manually labelled* (see HuggingFace dataset cards), so they are the gold against which the noisy silver model is measured. Eval honesty does not depend on training-set cleanliness.
3. **Noise floor as a reportable headline.** The model trained on this data cannot exceed roughly `1 - 0.15 = 0.85` precision on positives by any honest argument — that ceiling *is* a finding about the upstream data, not a failure of the method.

**Pairing with negatives noise.** Stage 2 of the negatives pipeline produces a quantified contamination rate (Section D3 above). Both halves of the training data therefore have a *reported* noise estimate. Hiding either would be the methodological error; reporting both lets readers calibrate.

---

## D9. Dual binary F1 reporting (full-vocab + ambiguity-restricted)

**Decision.** Every binary number in the writeup carries two F1 values:
- **Full-vocab F1** — over the locked eval set as supplied, including pejorative-only terms.
- **Restricted-vocab F1** — over the subset of eval rows whose `dog_whistle_root` has `coded_share < PEJORATIVE_THRESHOLD`.

**Why.** Pejorative-only terms are well-defined under "is this `X`?" but ill-defined under "is this *the coded use* of `X`?". The full-vocab number measures keyword + disambiguation jointly; the restricted number measures disambiguation alone.

**How readers should interpret the gap.** A wide gap (full-vocab >> restricted) means most of the headline performance comes from terms that are easy because they have no literal use — the model is getting credit for keyword matching. A narrow gap means the disambiguation framing is doing real work across the vocabulary. Either reading is informative; neither is hidden.

---

## D10. Exclusion of `formal_potential_dogwhistles` candidate pool

**Decision.** The 30M-row formal candidate pool is not used in mining. Negatives come exclusively from `informal_potential_dogwhistles`.

**Why.** Direct corollary of D1. Mixing formal-pool negatives with informal-pool positives would re-introduce the register confound: the model could distinguish "Reddit comment" from "Congressional speech" perfectly without learning anything about coding.

**Cost.** The formal pool contains literal uses of phrases we'd otherwise have very thin coverage for. We accept narrower term coverage in exchange for a cleaner register distribution.

---

## D11. LLM judge: open-weight, not GPT-4

**Decision.** Stage 2 uses `meta-llama/Llama-3.1-8B-Instruct` via vLLM on the HPC A100, not GPT-4 or GPT-4o-mini.

**Why.**
1. **Cost.** ~7k judgments at ~600 input + 8 output tokens is on the order of $1–3 in GPT-4o-mini, which is fine — but it adds an off-cluster dependency to a pipeline that otherwise runs end-to-end on the HPC we already have. Llama-3.1-8B at vLLM throughput finishes the same batch in well under an hour and writes directly into the data dir.
2. **Reproducibility.** An open-weight checkpoint is reproducible by anyone reading the paper. A closed model behind a moving API endpoint is not.
3. **Symmetry concern.** Kruk et al. used GPT-4. Using GPT-4 again on the candidates GPT-4 already saw could leak training-set artefacts. Using a different family of model is a stronger test of "is this candidate actually coded?".

**Falsifier.** If the contamination rate Llama reports is wildly different from a small GPT-4o-mini sanity sample (say, on 200 rows for a few dollars), we report both and treat the gap as a known judge-disagreement effect.

---

## D12. Stub judge in the local notebook

**Decision.** The local copy of the negatives notebook uses a deterministic stub judge — pejorative-flagged terms ⇒ `coded`, everything else ⇒ `non-coded` — when `FULL=False`. This is *not* a real adjudication and is documented in-cell as such. The real labels come from the HPC vLLM run.

**Why.** A real GPU-only model in the local notebook would gate the whole pipeline behind hardware most local environments don't have. The stub lets the local run validate the *plumbing* (does the prompt format right, does the output get parsed, do the manifests get written) without needing a GPU. The HPC run is what produces the real labels for the headline numbers.

**Failure mode it does not have.** The stub never silently looks like a real model; it stamps `judge_model = "stub"` into every row, and the manifest records `stage2_judge_model: "stub-deterministic"` for any local run. There is no path by which a stub-labelled run ends up in a results table.

---

## Summary table — design choices and their reportability

| Decision | Section | What we lose | What we gain |
|---|---|---|---|
| Reddit-only training | D1 | ~3.3k formal positives | Single-register training; no register confound |
| Group-by-root splits | D2 | Some ingroup coverage in val/test | Honest generalisation to unseen lexical material |
| Four-stage negatives | D3 | 1–2 hours HPC runtime per refresh | Quantified contamination rate; defensible negatives |
| Paired per-term 1:1 + drop | D4 | Pejorative-only positives leave binary set | No term-identity shortcut; honest disambiguation framing |
| Asymmetric task scope | D4a | Binary won't recognise pejoratives at eval | Each task trained on the data its research question actually asks for |
| `coded_share ≥ 0.5` flag | D5 | Threshold sensitivity | Reproducible, dataset-derived pejorative identification |
| Definition ablation | D6 | One extra training run per model | Tests whether enrichment is worth its tokens |
| Locked eval sets | D7 | Cannot use eval rows for training | Only honest measurement instrument we have |
| ~15% positive noise accepted | D8 | F1 ceiling around 0.85 | Avoids project-scope label refixing |
| Dual F1 reporting | D9 | One extra column in every results table | Reader can decompose keyword vs disambiguation effect |
| Formal pool excluded | D10 | Lower per-term negative supply for some terms | No register-shortcut training signal |
| Open-weight Llama judge | D11 | Slightly noisier than GPT-4 | Reproducible, on-cluster, family-independent of source |
| Stub judge locally | D12 | Local run is plumbing-only | Notebook works without a GPU |

---

## Source

These defenses correspond to design decisions made between project start (2026-04-01) and the negatives-pipeline overhaul (2026-04-25), with the paired-output fix added on 2026-04-26 after the per-term imbalance question was raised. Cross-references:

- [CONCERNS.md](CONCERNS.md) — risk catalogue
- [DATASETS.md](DATASETS.md) — full pipeline specification
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — section-by-section plan
- [NOTEBOOK_IMPLEMENTATION.md](NOTEBOOK_IMPLEMENTATION.md) — notebook structure
- [notebooks/negatives.ipynb](notebooks/negatives.ipynb) — the negatives pipeline implementation
- [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb) — the training pipeline that consumes its outputs
- Kruk et al. (2024), "Silent Signals, Loud Impact: LLMs for Word-Sense Disambiguation of Coded Dog Whistles", ACL 2024 — https://aclanthology.org/2024.acl-long.675
