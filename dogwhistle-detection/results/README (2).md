# `results/`: trained-model outputs (metrics + per-row predictions)

Everything in this folder is **post-training output**: per-seed metric JSONs and per-row prediction parquets. The trained model weights themselves are not shipped here — they live on the Hugging Face Hub under `calerio/silent-signals-{rqa,rqb,rqc}` (one branch per variant, indexed by `model_inventory.json` → `hf_branch`).

```
results/
├── README.md                             ← you are here
│
├── binary/                               RQ-A: RoBERTa-base binary disambiguator (coded vs literal)
│   ├── format_term/                      ← term arm: input = matched term + content
│   │   └── seed_{42, 123, 7}/
│   │       ├── test_results.json         metrics on all 3 eval sets + confusion matrices
│   │       └── predictions/
│   │           ├── test_split_preds.parquet           (2,768 rows · internal silver test)
│   │           ├── detection_101_preds.parquet        (   101 rows · LOCKED gold eval)
│   │           └── disambiguation_124_preds.parquet   (   124 rows · LOCKED gold eval · HEADLINE)
│   └── format_term_enriched_def/         ← enriched-def ablation arm (3 seeds, same files)
│
├── multiclass/                           RQ-B: RoBERTa-base 17-way ingroup classifier
│   ├── format_term/                      ← Run A: term + grouped split (3 seeds · HEADLINE)
│   │   └── seed_{42, 123, 7}/
│   │       ├── label_map.json            label2id / id2label (17 ingroups) + drop_report
│   │       ├── test_results.json         macro/weighted F1, accuracy, per-class, confusion matrix
│   │       └── predictions/test_preds.parquet         (2,095 rows on grouped test)
│   ├── format_text_only/seed_42/         ← Run B: text-only ablation (1 seed)
│   ├── format_term_altsplit/seed_42/     ← Run C: alt-split sensitivity check (1 seed; 1,947 rows)
│   └── format_term_weighted/seed_42/     ← Run D: weighted-CE ablation (1 seed)
│
├── generation/                           RQ-C Path A: Flan-T5-XL + LoRA, unbalanced (HEADLINE)
│   └── seed_{42, 123, 7}/
│       ├── test_results.json             json_parse_rate + per-field {accuracy, macro_f1, per-class}
│       └── test_predictions.parquet      (2,095 rows: input/target/pred + parsed fields + scores)
│
└── generation_balanced/                  RQ-C Path B: oversampled-balanced
    └── seed_{42, 123, 7}/                same file schema as `generation/`, plus optional `sample_outputs`
```

The per-RQ subfolders are organised `<arm>/seed_<seed>/`. Aggregated 3-seed means (the actual headline numbers in the report) are derived by `scripts/build_model_inventory.py`, which writes them to `data/manifests/model_inventory.json` — go there for headline metrics, not to this folder.

---

## Per-RQ details

### `binary/` — RQ-A

**Two arms** (`format_term`, `format_term_enriched_def`) × **three seeds** (42, 123, 7) = 6 trained variants. Each evaluated on three datasets:
1. **`test_split`** (2,768 rows): the internal split's silver test set. Useful for sanity-checking, not the headline.
2. **`detection_101`** (101 rows): the locked HF `silent_signals_detection` set. Gold-labelled.
3. **`disambiguation_124`** (124 rows): the locked HF `silent_signals_disambiguation` set. Gold-labelled. **This is the headline metric** (`disambiguation_124_f1`).

The two arms differ only in input format — `term` uses the dataset's short `definition`; `term_enriched_def` uses the Allen AI glossary's longer `definition_enriched` (cleanly available since enriched coverage is 298/298 roots). The headline RQ-A finding is that the per-arm mean F1 is essentially identical (term = 0.7074, enriched = 0.7021), so the curated glossary isn't required at inference.

### `multiclass/` — RQ-B

**Four runs**, varying one knob each relative to Run A:

| Folder | Run | Differs from A in | Seeds | n_test |
|---|---|---|---|---|
| `format_term/` | **A** (headline) | — | 42, 123, 7 | 2,095 (grouped) |
| `format_text_only/` | **B** (input ablation) | input arm = text only | 42 | 2,095 |
| `format_term_altsplit/` | **C** (sensitivity check) | split = ingroup-stratified random | 42 | 1,947 (alt-split) |
| `format_term_weighted/` | **D** (loss ablation) | loss = sklearn-balanced weighted CE | 42 | 2,095 |

Run A vs Run C is the central RQ-B contrast (macro-F1 0.353 vs 0.996; ~64 pp gap = result of glossary determinism, see `docs/rq_b_report.md`).

### `generation/` and `generation_balanced/` — RQ-C

**Two paths** × **three seeds** = 6 LoRA adapters fine-tuned on Flan-T5-XL. Both produce structured JSON `{dog_whistle_root, ingroup, definition, explanation}` for each test row.

- **Path A** (`generation/`): trained on the unbalanced positives (matches RQ-B's class distribution; defensible system, the headline).
- **Path B** (`generation_balanced/`): trained on an oversample-balanced version. Lifts headline `ingroup.accuracy` by +12 pp under prior shift but moves macro-F1 by ~+0.004 — class-prior reweighting, not learned generalisation. Some Path B seeds also include a `sample_outputs` field with 5 example predictions for inspection.

---

## File schemas

### `test_results.json` — RQ-A (`binary/`)

Flat metric keys per eval set, followed by a `_full` block with confusion matrices:

```json
{
  "test_split_n": 2768.0, "test_split_accuracy": 0.79, "test_split_f1": 0.80,
  "test_split_precision": 0.77, "test_split_recall": 0.82, "test_split_pr_auc": 0.88,
  "detection_101_*": ..., "disambiguation_124_*": ...,
  "_full": {
    "<dataset>": { "n": ..., "accuracy": ..., "f1": ..., "confusion_matrix": [[TN, FP], [FN, TP]] }
  },
  "_input_format": "term"
}
```

Eval-set order (rows × cols of confusion matrix): `[[label=0 actual], [label=1 actual]]`, where `1 = coded`, `0 = literal/non-coded`.

### `test_results.json` — RQ-B (`multiclass/`)

```json
{
  "input_format": "term", "seed": 42, "n_test": 2095,
  "accuracy": ..., "f1_macro": ..., "f1_weighted": ...,
  "classification_report": { "<class>": {precision, recall, f1-score, support}, ... },
  "confusion_matrix": [[...]],
  "confusion_matrix_label_order": ["Islamophobic", ...],
  "drop_report": { "val": {before, after, dropped, unseen_label_counts}, "test": {...} }
}
```

Note: `f1_macro` is the "no-padding" variant — averages over the union of classes that the model emits at least one prediction on plus classes present in test, which excludes some 0-support classes from the divisor. See `docs/rq_b_report.md` § B.8.1 for the reproducibility caveat. `drop_report` records val rows dropped because their label wasn't in the train-only label space (e.g. 10 `misogynistic` rows in val).

### `test_results.json` — RQ-C (`generation/`, `generation_balanced/`)

```json
{
  "n_samples": 2095, "json_parse_rate": 0.97,
  "dog_whistle_root": { "accuracy": ..., "macro_f1": ..., "weighted_f1": ..., "per_class": {...} },
  "ingroup":          { same shape },
  "definition":       { same shape },
  "explanation_cosine_mean": ..., "explanation_cosine_median": ...,
  "explanation_rouge_l_mean": ...,
  "sample_outputs": [ {input, predicted, target}, ... ]   // present in some Path B runs
}
```

`json_parse_rate` is post-`repair_json()`. `ingroup.macro_f1` is the **headline metric** for RQ-C.

### `label_map.json` — RQ-B only

Pinned label space (17 classes) for the trained variant: `label2id`, `id2label`, plus the same `drop_report` as in `test_results.json`. The model's logit columns map to ids 0…16 in this order; use this to decode `pred_label` from `test_preds.parquet` if you're loading the raw model.

### Predictions parquets

| File | Rows | Key columns beyond input metadata |
|---|---|---|
| `binary/.../predictions/test_split_preds.parquet` | 2,768 | `label`, `pred`, `prob_coded`, `correct` |
| `binary/.../predictions/detection_101_preds.parquet` | 101 | same |
| `binary/.../predictions/disambiguation_124_preds.parquet` | 124 | same |
| `multiclass/.../predictions/test_preds.parquet` | 2,095 (1,947 for altsplit) | `label` (int), `pred` (int), `pred_label` (str), `correct`, `pred_prob` |
| `generation/seed_*/test_predictions.parquet` | 2,095 | `input_text`, `target_json`, `pred_text`, `pred_text_repaired`, `parsed_ok`, `pred_{root,ingroup,definition,explanation}`, `explanation_cosine`, `explanation_rouge_l`, `rouge_l` |
| `generation_balanced/seed_*/test_predictions.parquet` | 2,095 | same as Path A minus the standalone `rouge_l` column |

Every predictions parquet preserves the original input metadata (`dog_whistle`, `dog_whistle_root`, `ingroup`, `content`, `subreddit`/`type`, `definition` where present) so you can do error analysis without re-joining against `data/final/`.

---

## Reproducibility pointers

- Headline aggregate metrics are computed by `scripts/build_model_inventory.py`, which reads `test_results.json` files here and emits `data/manifests/model_inventory.json`. Re-run after any new training run.
- Training scripts that produced these outputs live in `scripts/hpc/` (e.g. `train_binary.py`, `train_multiclass_ingroup.py`, `train_generation.py`) with YAML configs under `scripts/configs/`.
- The 3-seed sweeps used seeds **42, 123, 7**; single-seed runs (RQ-B Runs B/C/D) used seed **42**.
- Trained model weights are on the HF Hub at `calerio/silent-signals-{rqa,rqb,rqc}`, branched by variant (`hf_branch` in `model_inventory.json`).
