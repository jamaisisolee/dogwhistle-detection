# Notebook Implementation Guide

## What was built

Three Jupyter notebooks:

| Notebook | Purpose |
|---|---|
| [notebooks/eda.ipynb](notebooks/eda.ipynb) | Exploratory data analysis across all five SALT-NLP datasets |
| [notebooks/negatives.ipynb](notebooks/negatives.ipynb) | Negatives mining pipeline (stages 1–4 from [DATASETS.md §6](DATASETS.md)). Produces paired `data/processed/positives_balanced_{split}.parquet` + `data/processed/negatives_balanced_{split}.parquet`. |
| [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb) | End-to-end training pipeline (RQ-A binary disambiguator, RQ-B multiclass ingroup, RQ-C generation). Consumes the balanced negatives from the previous notebook. |

The pipeline notebook runs locally on a ~1000-row sample as a smoke test, then exports standalone HPC scripts via `%%writefile` cells at the end. Each notebook does the same: smoke-runnable locally, HPC-runnable via written-out scripts.

---

## Notebook structure (64 cells)

### Section 0: Setup & Utilities (cells 0–3)

**Cell 0 (markdown)**: Title and overview.

**Cell 1 (markdown)**: Section header.

**Cell 2 (code)**: All imports and constants.
- Sets `SAMPLE_SIZE = 1000` (change to 0 for full dataset)
- Sets `SEED = 42`
- Creates directory structure: `data/{manifests,processed,splits}`, `results/local`, `hpc_scripts/configs`
- Imports: torch, transformers, datasets, sklearn, peft, evaluate, pandas, numpy

**Cell 3 (code)**: All utility functions used throughout the notebook.
- `set_seed(seed)` — seeds random, numpy, torch, torch.cuda
- `print_diagnostics()` — prints Python/PyTorch/CUDA/GPU info
- `sanitize_text_column(df, col)` — drops nulls, coerces to str, strips whitespace, removes empties
- `content_hash(text)` — SHA-256 of normalized (lowered, stripped, whitespace-collapsed) text
- `deduplicate_df(df, text_col)` — removes duplicate rows by content hash
- `get_content_hashes(df, text_col)` — returns set of hashes for exclusion
- `resolve_bf16()` — returns True only if CUDA available AND bf16 supported
- `get_device_str()` — returns "cuda", "mps", or "cpu"

---

### Section 1: Inspect Datasets (cells 4–5)

**Cell 5 (code)**: Loads and profiles all 5 HuggingFace datasets.
- `SALT-NLP/silent_signals` (16,258 rows) — loaded fully into `df_ss`
- `SALT-NLP/silent_signals_detection` (101 rows) — loaded into `df_det`
- `SALT-NLP/silent_signals_disambiguation` (124 rows) — loaded into `df_dis`
- `SALT-NLP/informal_potential_dogwhistles` (6.03M rows) — streamed, first 1000 rows profiled
- `SALT-NLP/formal_potential_dogwhistles` (1.1M rows) — streamed, first 1000 rows profiled

Prints schema, row count, null count, unique value count per column. Prints type distribution and ingroup distribution for silent_signals. Saves `data/manifests/dataset_profiles.json`.

Variables persisted for later cells: `df_ss`, `df_det`, `df_dis`.

---

### Section 2: Build Grouped Splits (cells 6–8)

**Cell 7 (code)**: The core splitting logic.
1. Copies `df_ss` into `df_full`, sanitizes the `content` column
2. Uses `StratifiedGroupKFold(n_splits=7)` to split (train+val) vs test (~14.3% test)
3. Uses `StratifiedGroupKFold(n_splits=6)` on the remainder to split train vs val (~14.3% val)
4. Groups = `dog_whistle_root` (298 unique), stratify on `ingroup` (18 classes)
5. Result: ~71% train, ~14.3% val, ~14.3% test

**Cell 8 (code)**: Assertions and manifest saving.
- Asserts zero root overlap between all three split pairs
- Checks all 18 ingroups are present in train
- Prints per-split statistics (rows, roots, type distribution, top ingroups)
- Builds `root_to_split` mapping dict (used later for negative mining)
- Saves `data/splits/{train,val,test}.parquet`
- Saves `data/manifests/split_manifest.json` with root-to-split mapping, counts, distributions, timestamp

Variables persisted: `df_train`, `df_val`, `df_test`, `df_full`, `root_to_split`.

---

### Section 3: Mine Binary Negatives (cells 9–12)

**Cell 10 (code)**: Builds the exclusion set and term vocabulary.
- Computes content hashes of all silent_signals rows (positives must never be reused as negatives)
- Also hashes locked eval sets (detection `example` column, disambiguation `content` column)
- Builds `term_vocab`: maps each of the 706 surface forms to its root and definition
- Builds `valid_terms` set

**Cell 11 (code)**: Streams the two candidate pools and collects negatives.
- Defines `collect_from_stream()` which iterates through a streaming HF dataset
- For each row: checks if `dog_whistle` field matches a term in `valid_terms`, skips if already at cap, skips if content hash is in exclusion set
- `MAX_PER_TERM = 10` for smoke test (100 for HPC)
- Assigns `type` label ("Informal" or "Formal") based on source dataset
- Streams `informal_potential_dogwhistles` first, then `formal_potential_dogwhistles`

**Cell 12 (code)**: Deduplicates, balances, assigns splits, and saves.
- Deduplicates by content hash
- Assigns each negative to the split of its `dog_whistle_root` (via `root_to_split`)
- Balances: caps negatives per split at the number of positives in that split (1:1 ratio)
- Saves `data/processed/negatives_{train,val,test}.parquet`
- Saves `data/manifests/negatives_manifest.json` with provenance, counts, terms not found

---

### Section 4: Build Generation Targets (cells 13–14)

**Cell 14 (code)**: Builds structured JSON targets for the generation task.
- For each row in each split parquet, constructs:
  - `input_text`: `"Text: {content}\nMatched term: {dog_whistle}\nReturn JSON explaining the coded use."`
  - `target_json`: JSON with fields `dog_whistle_root`, `ingroup`, `definition`, `explanation`
  - The `explanation` field is a deterministic template: `"In this text, '{dog_whistle}' is used as a coded reference to {ingroup} ideology. {definition}"`
- Missing definitions get fallback text: `"No standard definition available"`
- Saves `data/processed/generation_{train,val,test}.parquet`
- Prints a sample input/target pair

---

### Section 5: Train Binary Disambiguator (cells 15–18)

**Cell 16 (code)**: Prepares binary classification data.
- `build_binary_input(row, input_format)` formats the input. Two arms (DESIGN_DEFENSE.md D6):
  - `input_format="term"`:
    ```
    Candidate term: {dog_whistle}
    Text: {content}
    Question: Is this candidate used as a coded dog whistle here?
    ```
  - `input_format="term_enriched_def"` (default):
    ```
    Candidate term: {dog_whistle}
    Candidate meaning: {definition_enriched}
    Text: {content}
    Question: Is this candidate used as a coded dog whistle here?
    ```
- The HPC pipeline runs both arms × 3 seeds = 6 trained models per RQ-A.
- Loads positives (label=1) from split parquets + negatives (label=0) from negatives parquets
- If `SAMPLE_SIZE > 0`: stratified sample by label (equal pos/neg)

**Cell 17 (code)**: Tokenizes and trains.
- Model: `microsoft/deberta-v3-base` (local smoke test)
- Max length: 256, batch size: 8, 1 epoch, lr: 2e-5
- `bf16=False`, `gradient_checkpointing=False` (DeBERTa-v3 incompatible)
- `use_mps_device=True` if on Apple Silicon
- Metrics: accuracy, precision, recall, F1, PR-AUC
- Best model selection: highest F1
- EarlyStoppingCallback with patience=2

**Cell 18 (code)**: Evaluates on three datasets.
1. **Internal test split**: positives + negatives from the held-out test
2. **Locked detection set (101 rows)**: loads `SALT-NLP/silent_signals_detection`, maps `"coded"`→1 / `"non-coded"`→0, uses `example` column as text
3. **Locked disambiguation set (124 rows)**: loads `SALT-NLP/silent_signals_disambiguation`, maps labels to binary (anything with "coded" that isn't "non-coded" → 1), uses `content` column as text

For each dataset: computes F1, precision, recall, PR-AUC, confusion matrix. For the test split: also computes per-type (Formal/Informal) breakdown.

Saves `results/local/binary/eval_binary_results.json` and `results/local/binary/best_model/`.

Cleans up model from memory with `del` + `gc.collect()` + `torch.cuda.empty_cache()`.

---

### Section 6: Train Multiclass Ingroup Classifier (cells 19–22)

**Cell 20 (code)**: Prepares multiclass data.
- Input format: `"Text: {content}\nMatched term: {dog_whistle}\nPredict the ingroup category."`
- Builds `label2id` map: 18 ingroups sorted alphabetically → integers 0–17
- Saves `label_map.json`
- If `SAMPLE_SIZE > 0`: proportional stratified sample by label

**Cell 21 (code)**: Tokenizes and trains.
- Same model/hyperparameters as binary
- Metrics: accuracy, macro-F1, weighted-F1
- Best model selection: highest macro-F1

**Cell 22 (code)**: Evaluates.
- Computes macro-F1, weighted-F1, accuracy
- Per-class precision/recall/F1 via `classification_report`
- Per-type (Formal/Informal) macro-F1 breakdown
- Confusion matrix (18x18)
- Saves `results/local/multiclass/eval_multiclass_results.json` and `best_model/`

---

### Section 7: Train Explanation Generator (cells 23–26)

**Cell 24 (code)**: Loads generation data, samples if needed.
- Model: `google/flan-t5-base` (local; `flan-t5-xl` on HPC)
- Max input: 256, max target: 128

**Cell 25 (code)**: Tokenizes and trains with LoRA.
- LoRA config: r=8, alpha=16, dropout=0.1, target_modules=["q", "v"]
- `bf16=resolve_bf16()` (True only if CUDA+BF16 supported)
- `gradient_checkpointing=True` (safe for Flan-T5, unlike DeBERTa)
- 2 epochs, batch size 4, lr 3e-4
- Seq2SeqTrainer with `predict_with_generate=True`
- Metrics: ROUGE-L
- Best model selection: highest ROUGE-L
- Prints trainable parameter count (LoRA = ~0.1-0.2% of total)

**Cell 26 (code)**: Evaluates the generator.
- Runs `model.generate()` on test set
- Computes:
  - **JSON parse rate**: % of outputs that are valid JSON
  - **Exact match** on `dog_whistle_root`, `ingroup`, `definition` fields
  - **ROUGE-L** on the `explanation` field
- Prints 3 sample outputs (input, predicted, gold)
- Saves `results/local/generation/eval_generator_results.json` and `best_model/`

---

### Section 8: Unified Report (cells 27–28)

**Cell 28 (code)**: Aggregates all eval results.
- Loads all three eval JSON files
- Prints formatted summary with all metrics
- Saves `results/local/final_report.json` (machine-readable)
- Saves `results/local/final_report.md` (markdown tables for binary, multiclass, generation)

---

### Section 9: Write HPC Files (cells 29–63)

**Cell 30 (code)**: Creates HPC directory structure.

The remaining cells use `%%writefile` to export files into `hpc_scripts/`:

| Cell | File | Type | Description |
|------|------|------|-------------|
| 32 | `requirements.txt` | pip deps | All Python dependencies with version pins |
| 34 | `configs/binary.yaml` | config | FacebookAI/roberta-base, 512 max_len, 5 epochs, bf16=false (originally deberta-v3-large; swapped 2026-04-29 after the NaN saga, see `docs/sessions/2026-04-29_rqa_nan_postmortem.md`) |
| 36 | `configs/multiclass.yaml` | config | FacebookAI/roberta-base, same hyperparams (bf16=true on multiclass) |
| 38 | `configs/generation.yaml` | config | flan-t5-xl, LoRA r=16/alpha=32, bf16=auto |
| 40 | `utils.py` | Python | Shared: `load_config()`, `parse_args()`, `set_seed()`, `print_diagnostics()`, `sanitize_text_column()`, `content_hash()`, `get_content_hashes()` |
| 42 | `train_binary_disambiguator.py` | Python | Standalone binary trainer: loads config via `--config`, accepts `--seed` override, trains with HF Trainer, evaluates on test, saves best model |
| 44 | `train_multiclass_ingroup.py` | Python | Standalone multiclass trainer: 18-class ingroup, saves label_map.json, classification report |
| 46 | `train_generator.py` | Python | Standalone generator trainer: Flan-T5 + LoRA, JSON field-level eval |
| 48 | `report_metrics.py` | Python | Aggregates results across seeds (mean +/- std) |
| 50 | `build_grouped_splits.py` | Python | Standalone split builder for HPC data prep |
| 52 | `mine_binary_negatives.py` | Python | Standalone negative miner: `--skip_llm`, `--max_candidates_per_term` |
| 54 | `build_generation_targets.py` | Python | Standalone generation target builder |
| 56 | `submit_data_prep.sh` | SLURM | CPU-only, 2h, runs split+mine+gen targets |
| 58 | `submit_binary.sh` | SLURM | 1 GPU, 24h, loops seeds [42, 123, 7] |
| 60 | `submit_multiclass.sh` | SLURM | 1 GPU, 24h, loops seeds [42, 123, 7] |
| 62 | `submit_generation.sh` | SLURM | 1 GPU, 24h, loops seeds [42, 123, 7] |

**Cell 63 (markdown)**: SCP and submission instructions.

---

## Key design decisions implemented

| Decision | Implementation |
|----------|---------------|
| **Grouped splits** | `StratifiedGroupKFold` by `dog_whistle_root` — cell 7. Zero-overlap assertion in cell 8. |
| **Silver negatives** | Streamed from candidate pools, filtered by term match, deduplicated, balanced 1:1 — cells 10-12. |
| **Locked eval sets** | Detection (101) and disambiguation (124) loaded only at eval time, with content-hash exclusion check — cell 18. |
| **BF16 gating** | `resolve_bf16()` checks both `is_available()` AND `is_bf16_supported()`. DeBERTa defaults to `bf16=False`. |
| **DeBERTa-v3 safety** | `bf16=False` and `gradient_checkpointing=False` in all DeBERTa training args (disentangled attention incompatibility). |
| **Candidate disambiguator framing** | Binary input includes term + definition + text + question — not a generic free-text detector. |
| **Structured generation** | Target is JSON with 4 fields, not free-form prose. `explanation` field uses deterministic template from gold labels. |
| **No data leakage** | Negatives inherit the split of their `dog_whistle_root`. Eval sets excluded from all training data via content hashing. |

---

## How to run

### Local smoke test
1. Open `notebooks/pipeline.ipynb` in Jupyter/VSCode
2. Run all cells top-to-bottom (Sections 0–8)
3. Takes ~10 min on MacBook (CPU), trains on ~1000 rows per split, 1–2 epochs
4. Results saved to `results/local/`

### Export HPC files
1. Run Section 9 cells (they write files to `hpc_scripts/`)
2. Copy to cluster: `scp -r hpc_scripts/ bocconi-hpc:~/dogwhistle_project/`
3. SSH in and submit:
   ```bash
   cd ~/dogwhistle_project
   mkdir -p logs results data/manifests data/processed data/splits
   sbatch submit_data_prep.sh        # CPU-only, ~2h
   # After data prep completes:
   sbatch submit_binary.sh           # 1 GPU, 24h, 3 seeds
   sbatch submit_multiclass.sh       # 1 GPU, 24h, 3 seeds
   sbatch submit_generation.sh       # 1 GPU, 24h, 3 seeds
   ```

### SLURM details
- Account: <YOUR_MATRICOLA>, partition: stud, QOS: stud
- Email notifications: <YOUR_EMAIL>
- Module: `sw/miniconda3`, conda env: `dogwhistle`
- GPU training scripts loop over seeds [42, 123, 7], each seed gets its own `seed_N/` output directory
- `report_metrics.py` aggregates across seeds with mean +/- std

---

## Files created

```
notebooks/pipeline.ipynb        # training pipeline, 64 cells
notebooks/negatives.ipynb       # negatives mining, 33 cells (stages 1-4)
notebooks/eda.ipynb             # EDA across all 5 SALT-NLP datasets
DATASETS.md                          # per-dataset breakdown + negatives plan
CONCERNS.md                          # known data-quality concerns
IMPLEMENTATION_PLAN.md               # Design plan (pre-existing)
NOTEBOOK_IMPLEMENTATION.md           # This file
data/
  manifests/                         # populated by running notebooks
  processed/                         # populated by running notebooks
  splits/                            # populated by running notebooks
results/
  local/                             # populated by running pipeline notebook
hpc_scripts/                         # populated by running Section 9 / negatives §8
  configs/{binary,multiclass,generation}.yaml
  utils.py
  train_binary_disambiguator.py
  train_multiclass_ingroup.py
  train_generator.py
  report_metrics.py
  build_grouped_splits.py
  mine_binary_negatives.py           # legacy single-stage miner
  build_generation_targets.py
  mine_negatives_full.py             # NEW: stage 1 of negatives notebook
  adjudicate_negatives_vllm.py       # NEW: stage 2, LLM-as-judge via vLLM
  balance_negatives_full.py          # NEW: stage 3, per-term 1:1 matching
  submit_negatives.sh                # NEW: SLURM submitter, chains stages 1-3
  submit_data_prep.sh
  submit_binary.sh
  submit_multiclass.sh
  submit_generation.sh
  requirements.txt
```

## Negatives mining notebook — what it does

`notebooks/negatives.ipynb` (33 cells) is a standalone pipeline that produces the binary task's negatives. It exists separately from the training pipeline so the negatives can be re-run, audited, and the contamination metrics inspected in isolation.

| Section | What it does |
|---|---|
| §0 Setup | Imports, paths, `FULL=False` smoke knob |
| §1 Load existing data | Positives from `data/splits/`, eval sets, term vocab; build content-hash exclusion set |
| §2 `coded_share` per term | Full scan of 6.03M-row Reddit candidate pool; flag pejorative-only terms (`coded_share ≥ 0.5`); save `data/manifests/term_coded_share.csv` |
| §3 Per-term capacity targets | Compute `mine_target = min(max(K × BUFFER_FACTOR, MIN_FLOOR), pool_avail, HARD_CEILING)` for each surface form |
| §4 Stage 1 — silver mining | Stream candidate pool with adaptive per-term cap; save `negatives_stage1_raw.parquet` |
| §5 Stage 2 — LLM judge | Prompt template (Kruk et al. style) + smoke stub; production runs on HPC |
| §6 Stage 3 — paired 1:1 matching | For each term `X`: pick top `n_pair = min(K, n_clean)` judge-confident negatives **and** random-sample `n_pair` of `X`'s positives; drop terms below `MIN_NEGATIVES_PER_TERM` on both sides. Asserts per-term parity, no content-hash leakage. Saves paired `positives_balanced_{split}.parquet` + `negatives_balanced_{split}.parquet` |
| §7 Stage 4 — manifest | Write `negatives_adjudication_report.json` with contamination rate, dropped terms, total positives lost to per-term floor |
| §8 `%%writefile` HPC | Production scripts for stages 1–3 + SLURM submitter (paired output) |
| §9 How to run | Smoke-vs-HPC instructions, knobs to tune |

The notebook integrates with the existing pipeline by writing paired `data/processed/positives_balanced_{train,val,test}.parquet` + `data/processed/negatives_balanced_{train,val,test}.parquet`. Cells 22 (local) and 64 (HPC `train_binary_disambiguator.py` writefile) of `notebooks/pipeline.ipynb` already prefer these balanced parquets, with a graceful fallback to the un-paired layout when the negatives pipeline hasn't been run yet. The multiclass and generation tasks read the unfiltered `data/splits/{split}.parquet` so dropped-term positives still feed those tasks.
