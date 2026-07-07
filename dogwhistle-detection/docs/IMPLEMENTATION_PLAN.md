# Silent Signals Dog Whistle Detection — Implementation Plan

## Context

Build a complete NLP pipeline for dog whistle detection, classification, and explanation using the SALT-NLP/silent_signals dataset (16,258 supervised examples). The old project (Group Project Beta) had critical flaws: trained on the 6M-row candidate pool instead of the supervised dataset, used GoEmotions (unrelated) as negatives, had data leakage via non-grouped splits, and never used the locked evaluation sets. This new project fixes all of these.

**Core principle**: `silent_signals` is the only supervised dataset. The candidate pools (6M informal, 1.1M formal) are for mining silver negatives. The detection (101) and disambiguation (124) sets are locked for evaluation only.

---

## Deliverable Structure

**Three Jupyter notebooks**:
- `notebooks/pipeline.ipynb` — the end-to-end training pipeline (binary + multiclass + generation), runnable as a smoke test locally and exporting HPC scripts via `%%writefile`.
- `notebooks/eda.ipynb` — exploratory data analysis across all five SALT-NLP datasets.
- `notebooks/negatives.ipynb` — standalone negatives mining pipeline (stages 1–4 from [DATASETS.md §6](DATASETS.md)). Produces paired `data/processed/positives_balanced_{split}.parquet` + `data/processed/negatives_balanced_{split}.parquet` and writes its own HPC scripts (`mine_negatives_full.py`, `adjudicate_negatives_vllm.py`, `balance_negatives_full.py`, `submit_negatives.sh`). The training pipeline notebook consumes these in §5.

```
project/
  notebooks/pipeline.ipynb        # training pipeline (RQ-A/B/C)
  notebooks/eda.ipynb             # EDA notebook
  notebooks/negatives.ipynb       # negatives mining pipeline
  data/                                # created by notebook cells
    manifests/
    processed/
    splits/
  results/                             # created by notebook cells
    local/
  hpc_scripts/                         # written by %%writefile cells at the end
    configs/
      binary.yaml
      multiclass.yaml
      generation.yaml
    train_binary_disambiguator.py
    train_multiclass_ingroup.py
    train_generator.py
    eval_binary.py
    eval_multiclass.py
    eval_generator.py
    report_metrics.py
    submit_binary.sh
    submit_multiclass.sh
    submit_generation.sh
    submit_data_prep.sh
    requirements.txt
  IMPLEMENTATION_PLAN.md               # this file
  bocconi-hpc-skill.md                 # HPC reference (already exists)
```

---

## Notebook Outline (cell-by-cell)

### Section 0: Setup & Utilities

**Cell 0.1 — Imports & constants**
```python
import os, json, hashlib, random, gc, time, warnings
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, TrainingArguments, Seq2SeqTrainingArguments,
    Trainer, Seq2SeqTrainer, DataCollatorWithPadding, DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    average_precision_score, classification_report, confusion_matrix,
    f1_score,
)
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, TaskType
import evaluate as hf_evaluate

SAMPLE_SIZE = 1000  # for local smoke test; set 0 for full
SEED = 42
DATA_DIR = "./data"
RESULTS_DIR = "./results/local"
os.makedirs(f"{DATA_DIR}/manifests", exist_ok=True)
os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
os.makedirs(f"{DATA_DIR}/splits", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
```

**Cell 0.2 — Utility functions**
```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_diagnostics():
    """Print Python/torch/CUDA/GPU info."""
    ...

def sanitize_text_column(df, col):
    """Drop nulls, coerce to str, strip, remove empties, log stats."""
    before = len(df)
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  sanitize '{col}': dropped {dropped:,} rows ({dropped/before*100:.1f}%)")
    return df

def content_hash(text):
    """SHA-256 of normalized text."""
    normalized = " ".join(str(text).lower().strip().split())
    return hashlib.sha256(normalized.encode()).hexdigest()

def deduplicate_df(df, text_col="content"):
    df["_hash"] = df[text_col].apply(content_hash)
    before = len(df)
    df = df.drop_duplicates(subset="_hash", keep="first").drop(columns="_hash")
    print(f"  Dedup: {before:,} -> {len(df):,} (removed {before - len(df):,})")
    return df.reset_index(drop=True)

def get_content_hashes(df, text_col="content"):
    return set(df[text_col].apply(content_hash))

def resolve_bf16():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True
    return False

set_seed(SEED)
print_diagnostics()
```

---

### Section 1: Inspect Datasets

**Cell 1.1 — Load & profile all 5 datasets**
- Load `SALT-NLP/silent_signals` fully (16K rows)
- Load `SALT-NLP/silent_signals_detection` (101 rows)
- Load `SALT-NLP/silent_signals_disambiguation` (124 rows)
- Stream `SALT-NLP/informal_potential_dogwhistles` — profile first 1000 rows
- Stream `SALT-NLP/formal_potential_dogwhistles` — profile first 1000 rows
- Print: schemas, row counts, null counts, unique values per column, type distribution
- Save `data/manifests/dataset_profiles.json`

---

### Section 2: Build Grouped Splits

**Cell 2.1 — Grouped stratified splitting**
```python
from sklearn.model_selection import StratifiedGroupKFold

df = load_dataset("SALT-NLP/silent_signals", split="train").to_pandas()
df = sanitize_text_column(df, "content")

X = np.arange(len(df))
y = df["ingroup"].values
groups = df["dog_whistle_root"].values

# Step 1: (train+val) vs test — n_splits=7 gives ~14.3% test
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
train_val_idx, test_idx = next(iter(sgkf.split(X, y, groups)))

# Step 2: train vs val — n_splits=6 gives ~16.7% of remaining
df_tv = df.iloc[train_val_idx]
sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
X_tv = np.arange(len(df_tv))
y_tv = df_tv["ingroup"].values
groups_tv = df_tv["dog_whistle_root"].values
train_idx_local, val_idx_local = next(iter(sgkf2.split(X_tv, y_tv, groups_tv)))

df_train = df_tv.iloc[train_idx_local].reset_index(drop=True)
df_val = df_tv.iloc[val_idx_local].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)
```

**Cell 2.2 — Assertions & manifest**
- Assert zero root overlap between all split pairs
- Assert every major ingroup in train
- Print split stats: rows, roots, type distribution, ingroup distribution per split
- Save `data/splits/{train,val,test}.parquet`
- Save `data/manifests/split_manifest.json` with root→split mapping, counts, timestamp

---

### Section 3: Mine Binary Negatives

**Cell 3.1 — Build exclusion set**
- Content hashes of all silent_signals rows + locked eval rows
- Load term vocabulary: 706 surface forms, their roots and definitions

**Cell 3.2 — Stream candidate pools & collect negatives**
- Stream `informal_potential_dogwhistles` and `formal_potential_dogwhistles`
- Filter: `dog_whistle` field matches a term in silent_signals
- Exclude: content hash already in exclusion set
- Cap: `max_candidates_per_term = 10` (smoke test; 100 for HPC)
- Deduplicate via `deduplicate_df()`

**Cell 3.3 — Balance & assign splits**
- Skip LLM adjudication for notebook (all candidates treated as non-coded)
- Balance 1:1 with positives, proportional by type and root
- Each negative inherits split of its `dog_whistle_root` from split manifest
- Save `data/processed/negatives_{train,val,test}.parquet`
- Save `data/manifests/negatives_manifest.json`

---

### Section 4: Build Generation Targets

**Cell 4.1 — Build structured JSON targets**
- For each split parquet, build:
  - `input_text`: `"Text: {content}\nMatched term: {dog_whistle}\nReturn JSON explaining the coded use."`
  - `target_json`: `{"dog_whistle_root": "...", "ingroup": "...", "definition": "...", "explanation": "In this text, '{dog_whistle}' is used as a coded reference to {ingroup} ideology. {definition}"}`
- Handle missing definitions with fallback
- Save `data/processed/generation_{train,val,test}.parquet`

---

### Section 5: Train Binary Disambiguator (local smoke)

**Cell 5.1 — Prepare binary data**
- Load positives (label=1) from `data/processed/positives_balanced_{split}.parquet` (paired per-term 1:1; produced by [notebooks/negatives.ipynb](notebooks/negatives.ipynb) §6). Falls back to `data/splits/{split}.parquet` only if the balanced parquet is missing.
- Load negatives (label=0) from `data/processed/negatives_balanced_{split}.parquet`. Falls back to `data/processed/negatives_{split}.parquet`.
- The binary trainer therefore sees a per-term-balanced set with pejorative-only terms removed on both sides; the multiclass and generation tasks continue to read the unfiltered `data/splits/{split}.parquet`.
- Build input: `"Candidate term: {dog_whistle}\nCandidate meaning: {definition}\nText: {content}\nQuestion: Is this candidate used as a coded dog whistle here?"`
- If SAMPLE_SIZE > 0: stratified sample

**Cell 5.2 — Tokenize & train**
- Model: `microsoft/deberta-v3-base` (small, local)
- 1 epoch, batch=8, max_length=256
- Metrics: accuracy, precision, recall, F1, PR-AUC
- EarlyStoppingCallback(patience=2)
- `bf16=False` (DeBERTa unstable), `gradient_checkpointing=False`

**Cell 5.3 — Evaluate binary**
- Evaluate on held-out test split
- Evaluate on locked detection (101 rows): map "coded"→1, "non-coded"→0, use `example` as text
- Evaluate on locked disambiguation (124 rows): use `content` as text
- Assert no content hash overlap with training data
- Compute per-set: F1, precision, recall, PR-AUC, confusion matrix, per-type breakdown
- Save `results/local/eval_binary_results.json`

---

### Section 6: Train Multiclass Ingroup Classifier (local smoke)

**Cell 6.1 — Prepare multiclass data**
- Load `data/splits/{split}.parquet` (all positives)
- Build input: `"Text: {content}\nMatched term: {dog_whistle}\nPredict the ingroup category."`
- Label: ingroup → integer via sorted label2id map
- Save `label_map.json`
- If SAMPLE_SIZE > 0: stratified sample

**Cell 6.2 — Tokenize & train**
- Model: `microsoft/deberta-v3-base`
- 1 epoch, batch=8, max_length=256
- Metrics: macro-F1, weighted-F1, accuracy
- Best model criterion: macro-F1

**Cell 6.3 — Evaluate multiclass**
- Per-class precision/recall/F1 for all 18 ingroups
- Per-type (Formal/Informal) breakdown
- Full classification report + confusion matrix
- Save `results/local/eval_multiclass_results.json`

---

### Section 7: Train Explanation Generator (local smoke)

**Cell 7.1 — Prepare generation data**
- Load `data/processed/generation_{split}.parquet`
- If SAMPLE_SIZE > 0: sample

**Cell 7.2 — Tokenize & train with LoRA**
- Model: `google/flan-t5-base` (small, local)
- LoRA: r=8, alpha=16, dropout=0.1, target_modules=["q", "v"]
- 1–2 epochs, batch=4, max_input=256, max_target=128
- `bf16=resolve_bf16()`, `gradient_checkpointing=True`
- Metrics: ROUGE-L

**Cell 7.3 — Evaluate generator**
- Run `model.generate()` on test inputs
- JSON parse rate
- Exact match on `dog_whistle_root`, `ingroup`, `definition`
- ROUGE-L on `explanation`
- Per-type breakdown
- Save sample generations for inspection
- Save `results/local/eval_generator_results.json`

---

### Section 8: Unified Report

**Cell 8.1 — Aggregate & display all metrics**
- Load all eval JSON files
- Print summary tables (markdown)
- Save `results/local/final_report.json` and `results/local/final_report.md`

---

### Section 9: Write HPC Files (%%writefile)

This section generates all files needed for full-scale HPC training. Each cell uses `%%writefile hpc_scripts/...` to write a standalone file.

#### Cell 9.1 — `%%writefile hpc_scripts/requirements.txt`
```
torch>=2.6
transformers>=4.40
datasets>=2.18
evaluate>=0.4
peft>=0.10
accelerate>=0.28
sentencepiece>=0.1.99
scikit-learn>=1.4
pandas>=2.2
pyarrow>=15.0
numpy>=1.26
pyyaml>=6.0
tqdm>=4.66
rouge-score>=0.1.2
```

#### Cell 9.2 — `%%writefile hpc_scripts/configs/binary.yaml`
```yaml
task: binary
model_name: microsoft/deberta-v3-large
max_length: 512
epochs: 5
batch_size: 16
grad_accum: 2
lr: 2e-5
weight_decay: 0.01
warmup_ratio: 0.1
sample_size: 0
seed: 42
seeds: [42, 123, 7]
data_dir: ./data
output_dir: ./results/hpc/binary
split_dir: ./data/splits
bf16: false
gradient_checkpointing: false
early_stopping_patience: 2
logging_steps: 200
save_total_limit: 2
num_workers: 4
report_to: none
```

#### Cell 9.3 — `%%writefile hpc_scripts/configs/multiclass.yaml`
Same structure, `task: multiclass`, same model.

#### Cell 9.4 — `%%writefile hpc_scripts/configs/generation.yaml`
```yaml
task: generation
model_name: google/flan-t5-xl
max_input_len: 512
max_target_len: 128
epochs: 5
batch_size: 4
grad_accum: 4
lr: 3e-4
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: ["q", "v"]
sample_size: 0
seed: 42
seeds: [42, 123, 7]
bf16: auto
gradient_checkpointing: true
early_stopping_patience: 2
logging_steps: 200
save_total_limit: 2
num_workers: 4
report_to: none
```

#### Cell 9.5 — `%%writefile hpc_scripts/train_binary_disambiguator.py`
Standalone script: loads YAML config via `--config`, accepts `--seed` override, loads split parquets + negatives, builds binary inputs, tokenizes, trains with HF Trainer, evaluates on test + locked eval sets, saves results. Self-contained (includes all utility functions inline or via a shared utils section at top).

#### Cell 9.6 — `%%writefile hpc_scripts/train_multiclass_ingroup.py`
Same pattern for 18-class ingroup classifier.

#### Cell 9.7 — `%%writefile hpc_scripts/train_generator.py`
Same pattern for Flan-T5 + LoRA seq2seq.

#### Cell 9.8 — `%%writefile hpc_scripts/eval_binary.py`
Standalone eval: loads trained model, evaluates on test split + locked detection + disambiguation sets.

#### Cell 9.9 — `%%writefile hpc_scripts/eval_multiclass.py`
Standalone eval for multiclass.

#### Cell 9.10 — `%%writefile hpc_scripts/eval_generator.py`
Standalone eval for generator.

#### Cell 9.11 — `%%writefile hpc_scripts/report_metrics.py`
Aggregates all eval results across seeds into final report.

#### Cell 9.12 — `%%writefile hpc_scripts/submit_data_prep.sh`
CPU-only SLURM job: builds splits, mines negatives, builds generation targets.
```bash
#!/bin/bash
#SBATCH --job-name=dw-data-prep
#SBATCH --account=<YOUR_MATRICOLA>
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail
module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate dogwhistle
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python build_grouped_splits.py
python mine_binary_negatives.py --skip_llm --max_candidates_per_term 100
python build_generation_targets.py
```

*Note*: The data prep scripts (split builder, negative miner, gen target builder) will also be written via `%%writefile` cells, as standalone `.py` files that can run on HPC.

#### Cell 9.13 — `%%writefile hpc_scripts/submit_binary.sh`
```bash
#!/bin/bash
#SBATCH --job-name=dw-binary
#SBATCH --account=<YOUR_MATRICOLA>
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail
module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate dogwhistle
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

for SEED in 42 123 7; do
    echo "=== Seed $SEED ==="
    python train_binary_disambiguator.py --config configs/binary.yaml --seed $SEED
    python eval_binary.py --config configs/binary.yaml --model_path results/hpc/binary/seed_${SEED}/best_model --seed $SEED
done
python report_metrics.py --results_dir results/hpc/binary
```

#### Cell 9.14 — `%%writefile hpc_scripts/submit_multiclass.sh`
Same pattern, calling multiclass scripts.

#### Cell 9.15 — `%%writefile hpc_scripts/submit_generation.sh`
Same pattern, calling generation scripts.

---

## Key Design Decisions (vs. Old Project)

| Issue | Old Project | New Project |
|-------|-------------|-------------|
| Training data | 6M candidate pool | 16K supervised silent_signals |
| Negatives | GoEmotions (unrelated) | Silver negatives from candidate pools |
| Splits | Random stratified | Grouped by dog_whistle_root (no leakage) |
| Split persistence | In-memory per run | Parquet + JSON manifests on disk |
| Config | Hardcoded argparse | YAML configs + CLI overrides |
| BF16 | `torch.cuda.is_available()` | `torch.cuda.is_bf16_supported()` |
| Locked eval sets | Never used | Explicit eval on detection (101) + disambiguation (124) |
| Deduplication | None | Content-hash based |
| Binary framing | Generic classifier | Candidate disambiguator with term+definition context |
| Generation target | Free-form prose | Structured JSON with deterministic template |
| Format | 20+ separate .py files | 1 notebook + %%writefile HPC export |

---

## Dataset Schemas (reference)

### silent_signals (16,258 rows) — CORE SUPERVISED
| Column | Unique | Notes |
|--------|--------|-------|
| dog_whistle | 706 | surface form |
| dog_whistle_root | 298 | group key for splits |
| ingroup | 18 | stratification target |
| content | — | text |
| definition | 244 | used in binary input |
| type | 2 | "Formal" / "Informal" (~80/20) |
| speaker, chamber, party | — | formal rows only |
| subreddit | 47 | informal rows only |

### silent_signals_detection (101 rows) — LOCKED EVAL
Columns: idx, dog_whistle, dog_whistle_root, ingroup, definition, **example** (text), **label** ("coded"/"non-coded")

### silent_signals_disambiguation (124 rows) — LOCKED EVAL
Columns: type, dog_whistle (21), dog_whistle_root (17), ingroup (7), **content** (text), **label** (3 values), definition

### informal_potential_dogwhistles (6.03M rows) — CANDIDATE POOL
Columns: dog_whistle, ingroup, content, date, subreddit, source, split

### formal_potential_dogwhistles (1.1M rows) — CANDIDATE POOL
Columns: date, speaker, chamber, reference, source, party, content, dog_whistle, ingroup

---

## Split Strategy

Use `StratifiedGroupKFold` from sklearn:
- Groups = `dog_whistle_root` (298 unique) — no root in multiple splits
- Stratify on = `ingroup` (18 classes)
- Step 1: n_splits=7, take fold 0 as test (~14.3%)
- Step 2: n_splits=6 on remainder, take fold 0 as val (~16.7% of remainder ≈ 14.3% overall)
- Result: ~71% train, ~14.3% val, ~14.3% test
- Assert zero root overlap between all pairs
- Assert all major ingroups present in train

---

## Negative Mining (Binary Task)

The original (legacy) flow is documented below for reference. The defensible flow now lives in [notebooks/negatives.ipynb](notebooks/negatives.ipynb) and is summarised in [DATASETS.md §6](DATASETS.md):

**Defensible flow (current):**
1. **Stage 1** — adaptive per-term cap: `max(K_positives × buffer, MIN_FLOOR)` instead of flat 10.
2. **Stage 2** — LLM-as-judge adjudication via vLLM + Llama-3.1-8B-Instruct on HPC (smoke stub locally).
3. **Stage 3** — *paired* strict per-term 1:1 matching. For each term `X` with `K` positives and `n_clean` post-judge negatives, sample `min(K, n_clean)` of each side. Drop the term entirely (both positives and negatives) when `n_clean < MIN_NEGATIVES_PER_TERM` (default 3). Output is paired `positives_balanced_{split}.parquet` + `negatives_balanced_{split}.parquet`.
4. **Stage 4** — verify zero overlap with locked eval sets (gold negatives stay locked).
5. Pejorative-only terms (slurs / manosphere coinages with `coded_share ≥ 0.5`) are flagged via [data/manifests/term_coded_share.csv](data/manifests/term_coded_share.csv), dropped from binary training on both sides, kept for multiclass + generation. See [CONCERNS.md §Concern 4](CONCERNS.md).

Output consumed by §5 of the training pipeline: `data/processed/positives_balanced_{train,val,test}.parquet` + `data/processed/negatives_balanced_{train,val,test}.parquet`. The multiclass and generation tasks continue to read the unfiltered `data/splits/{train,val,test}.parquet`, so dropped-term positives are not wasted.

**Legacy flow (deprecated, kept for reference):**
1. Get 706 surface forms + definitions from silent_signals
2. Build exclusion set: content hashes of silent_signals + locked eval sets
3. Stream candidate pools, filter by `dog_whistle` field matching vocabulary
4. Cap per term (10 for notebook, 100 for HPC)
5. Deduplicate by content hash
6. Skip LLM adjudication in notebook (treat all as non-coded); optional for HPC
7. Balance 1:1 with positives by type and root
8. Assign splits based on root→split mapping from manifest

---

## HPC Details

- Account: <YOUR_MATRICOLA>, Partition: stud, QOS: stud
- Email: <YOUR_EMAIL>
- Module: `sw/miniconda3`, Conda env: `dogwhistle`
- GPU: A100 80GB, CUDA 12.1
- Home quota: 50GB
- DeBERTa-v3: bf16=false, gradient_checkpointing=false (unstable)
- Flan-T5: bf16=auto (resolves via `is_bf16_supported()`), gradient_checkpointing=true

