# Headline Results

| Task | Model | Metric | Result |
|---|---|---|---:|
| Binary disambiguation | RoBERTa-base | F1 | 0.707 +/- 0.015 |
| Binary disambiguation | RoBERTa-base | PR-AUC | 0.802 +/- 0.010 |
| Ingroup classification | RoBERTa-base | Macro-F1, root-stratified | 0.353 +/- 0.007 |
| Ingroup classification | RoBERTa-base | Macro-F1, random split control | 0.996 |
| Structured generation | Flan-T5-XL + LoRA | Ingroup macro-F1 | 0.379 +/- 0.014 |
| Structured generation | Flan-T5-XL + LoRA | JSON parse rate | 97.5% +/- 1.7% |

The main result is the grouped-vs-random split contrast: when dog-whistle roots leak between train and test, the task becomes close to glossary lookup; when roots are held out, the model must generalize from context.
