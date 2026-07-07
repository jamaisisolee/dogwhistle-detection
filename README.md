# Hidden in Plain Sight: Dog Whistle Detection, Classification, and Explanation

> ### Project Summary

> Developed an end-to-end NLP pipeline for detecting coded dog-whistle language on Reddit, classifying ideological ingroups, and generating structured explanations. Fine-tuned RoBERTa for binary and multiclass classification and Flan-T5-XL + LoRA for structured explanation generation. The project's central finding demonstrates that apparent near-perfect performance on ingroup classification is largely explained by evaluation leakage, with macro-F1 dropping from 0.996 to 0.353 when dog-whistle roots are properly held out during testing.

This project was developed for the Natural Language Processing course at Universita Bocconi. It studies how transformer-based NLP systems handle politically and ideologically coded language: phrases that appear innocuous in surface form but communicate a hidden meaning to an intended audience.

The project addresses three linked tasks:

1. **Binary disambiguation** - determine whether a phrase is being used literally or as a coded dog whistle.
2. **Ingroup classification** - classify the ideological ingroup associated with a coded expression.
3. **Structured explanation generation** - generate a JSON-style explanation containing the dog-whistle root, ingroup, definition, and rationale.

The initial report and project documentation created by the team are available in the original repository here [`calerio/dog-whistle-detection`](https://github.com/calerio/dog-whistle-detection/tree/main/docs) documentation.

The full submitted report is included in [`docs/submission/final_report.pdf`](docs/submission/final_report.pdf).

---

## Project Overview

Dog whistles are difficult for standard hate-speech or toxicity systems because the harmful meaning is not always visible from the words alone. The same expression can be literal in one context and coded in another. This project treats dog-whistle detection as a context-sensitive NLP problem rather than a simple keyword-matching task.

The work uses the `silent_signals` Reddit corpus, derived from the Allen AI Glossary of Dog Whistles. The dataset contains coded examples across hundreds of canonical dog-whistle roots and multiple ingroup labels. The main experimental design choice is **root-stratified evaluation**: each dog-whistle root is assigned to only one of train, validation, or test, so models must generalize to unseen expressions instead of memorizing a glossary.

---

## Main Research Finding

The central methodological finding is that evaluation leakage can make ingroup classification appear almost solved.

When dog-whistle roots are allowed to appear in both training and testing:

- **Macro-F1 = 0.996**

When dog-whistle roots are fully held out during testing:

- **Macro-F1 = 0.353**

This gap shows that random splits can measure glossary memorization rather than contextual understanding. Root-stratified splits provide a more honest evaluation of whether a model can generalize to new coded expressions.

---

## Pipeline

```text
Reddit comments
      |
      v
Binary dog-whistle disambiguation
RoBERTa-base
      |
      v
Ingroup classification
RoBERTa-base
      |
      v
Structured explanation generation
Flan-T5-XL + LoRA
      |
      v
JSON output: root, ingroup, definition, explanation
```

---

## Dataset

The project uses the `silent_signals` dataset from Kruk et al. (2024), restricted to the informal Reddit split.

Dataset characteristics:

- Approximately 13,000 coded Reddit examples after filtering and deduplication
- 298 canonical dog-whistle roots
- 17 ideological ingroup categories
- English-language Reddit data
- Root-level train/validation/test split to prevent term leakage

For the binary disambiguation task, literal-sense negatives were mined from a larger Reddit candidate pool and filtered using Llama-3.1-8B-Instruct as an adjudicator. Locked human-gold evaluation files were kept outside the training pipeline and used only after model selection.

---

## Methodology

### Task 1 - Binary Disambiguation

**Objective:** classify whether a known phrase is being used literally or as a coded dog whistle.

**Model:**

- `FacebookAI/roberta-base`
- Binary classification head

**Input framing:**

- Dog-whistle term
- Reddit comment context
- Binary classification prompt
- Optional glossary definition ablation

**Evaluation:**

- F1 score for the coded class
- Accuracy
- PR-AUC
- Locked human-gold disambiguation and detection sets

---

### Task 2 - Ingroup Classification

**Objective:** predict the ideological ingroup associated with a coded expression.

**Model:**

- `FacebookAI/roberta-base`
- 17-class classification head

**Evaluation protocol:**

- Root-stratified split for the main result
- Alternative random split as a leakage control
- Macro-F1 as the primary metric
- Confusion matrix and per-class error analysis

The grouped-vs-random comparison is the key analysis point: when test terms appear in training, the classifier can effectively learn a glossary lookup table. When roots are held out, the model must rely on context and semantic transfer.

---

### Task 3 - Structured Explanation Generation

**Objective:** generate machine-readable explanations for coded language.

**Model:**

- `google/flan-t5-xl`
- LoRA adapters
- Frozen base model with trainable parameter-efficient adapters

**Output format:**

```json
{
  "dog_whistle_root": "...",
  "ingroup": "...",
  "definition": "...",
  "explanation": "..."
}
```

The generation pipeline includes JSON repair to recover malformed outputs and score only parseable structured predictions. Ingroup prediction is treated as the main generalization metric because held-out roots make exact root and definition generation structurally difficult.

---

## Key Results

| Task | Model | Result |
|---|---|---:|
| Binary disambiguation | RoBERTa-base | F1 = 0.707 +/- 0.015 |
| Binary disambiguation | RoBERTa-base | PR-AUC = 0.802 +/- 0.010 |
| Ingroup classification, root-stratified | RoBERTa-base | Macro-F1 = 0.353 +/- 0.007 |
| Ingroup classification, random split control | RoBERTa-base | Macro-F1 = 0.996 |
| Structured generation | Flan-T5-XL + LoRA | Ingroup macro-F1 = 0.379 +/- 0.014 |
| Structured generation | Flan-T5-XL + LoRA | JSON parse rate = 97.5% +/- 1.7% |

---

## What This Project Demonstrates

- Fine-tuning transformer models for context-sensitive NLP classification
- Building binary and multiclass text classifiers with RoBERTa
- Parameter-efficient generation using Flan-T5-XL + LoRA
- Root-stratified evaluation to prevent data leakage
- LLM-assisted negative mining and data curation
- Structured JSON generation and output repair
- Multi-seed evaluation and error analysis
- Ethical handling of harmful-language datasets

---

## Technical Stack

### Machine Learning

- PyTorch
- Hugging Face Transformers
- RoBERTa-base
- Flan-T5-XL
- LoRA / PEFT

### Data Processing

- Pandas
- NumPy
- Parquet files
- JSON outputs

### Experimentation

- Multi-seed training
- Root-stratified splitting
- Class-balancing experiments
- Confusion matrix analysis
- Human-gold and silver-label evaluation surfaces

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── dogwhistle-detection
      ├── requirements.txt
      ├── .gitignore
      ├── model_weights.txt
      ├── data/                 # Dataset placeholders; raw Reddit data is not committed
      ├── docs/
      │   ├── repository_notes.md
      │   └── submission/
      │       └── final_report.pdf
      ├── notebooks/            # EDA and experiment notebooks
      ├── resources/            # Supporting project resources
      ├── results/              # Headline metrics, confusion matrices, and result summaries
      ├── scripts/              # Training, evaluation, and data-preparation scripts
      └── src/                  # Reusable project modules
```

---

## Related Project Links

- Team repository: [https://github.com/ceciliaalocicero/dogwhistle-detection](https://github.com/calerio/dog-whistle-detection)
- Dataset paper: Kruk et al. (2024), *Silent Signals, Loud Impact: LLMs for Word-Sense Disambiguation of Coded Dog Whistles*

---

## Authors

Facundo Lucero, Alissa Sharuda, Jan Szkulepa, Valerio Costa, Cecilia Lo Cicero  
Universita Bocconi - Course 20597, Natural Language Processing

---

## Ethics Note

This project studies harmful and ideologically coded language for research purposes. A deployed detector could create both false positives, by over-flagging legitimate uses of ambiguous terms, and false negatives, by missing emerging coded language outside the studied vocabulary. Results should therefore be interpreted as research findings on a fixed dataset and time window rather than as a moderation-ready system.

---

## License

This repository uses the MIT License for shareable code. Dataset files, model weights, and the final report may be governed by separate licenses, terms, or attribution requirements from their original sources.
