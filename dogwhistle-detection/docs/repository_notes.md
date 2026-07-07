# Repository Notes

This repository is organized as a GitHub-facing version of the final NLP project, with the project explanation, submitted report, and space for clean code/results.

## Main README direction

The README should present the project as an end-to-end NLP pipeline first. It should lead with the motivation, three research tasks, pipeline, methodology, and results. Notes about omitted raw data or checkpoints should appear later as artifact-management guidance, not as the opening framing.

## Strong sections to keep

- Project title and summary
- Three-task project overview
- Main research finding about root leakage
- Pipeline diagram
- Dataset summary
- Methodology by task
- Key results table
- Technical stack
- Repository structure
- Data and model artifact notes
- Team/project attribution
- Ethics note

## Files worth adding next

```text
notebooks/
  01_data_overview.ipynb
  02_binary_disambiguation.ipynb
  03_ingroup_classification.ipynb
  04_structured_generation.ipynb

scripts/
  prepare_splits.py
  mine_negatives.py
  train_rqa_roberta.py
  train_rqb_roberta.py
  train_rqc_lora.py
  evaluate_predictions.py

results/
  headline_metrics.json
  rqb_confusion_matrix.csv
  rqc_confusion_matrix.csv
  per_class_f1.csv
```

## Files to keep out of GitHub

- Raw Reddit comments
- Full dataset dumps, unless redistribution is explicitly allowed
- Model checkpoints and large LoRA adapters
- Cluster logs containing local paths
- Credentials, API keys, or environment-specific configuration
- Per-row predictions if they expose raw user text

## Suggested GitHub description

> End-to-end NLP pipeline for dog-whistle detection, ingroup classification, and explainable generation using RoBERTa and Flan-T5-XL.

## Suggested topics

`nlp`, `machine-learning`, `text-classification`, `transformers`, `pytorch`, `roberta`, `flan-t5`, `lora`, `explainable-ai`, `computational-social-science`, `hate-speech-detection`
