# Phase 4 — Transliteration-Aware Retrieval (Shishya Run)

This directory contains the complete codebase, configurations, metrics, logs, and report for **Phase 4: Transliteration-Aware Retrieval** of the Amazon ML Challenge 2026 Business Entity Resolution project.

## Directory Structure

```text
phase_4_Shishya/
├── config/
│   └── phase4_config.json                 # Machine-readable experiment parameters
├── src/
│   └── phase4_transliteration_retrieval.py # Full Phase 4 experimental pipeline
├── metrics/
│   ├── phase4_transliteration_metrics.csv # Overall & per-K recall/precision metrics
│   ├── phase4_candidate_statistics.csv    # Candidate count distributions & quantiles
│   └── phase4_script_country_metrics.csv  # Script classification & breakdown by country
├── report/
│   └── phase4_transliteration_retrieval.md# Final Phase 4 comprehensive report
├── logs/
│   └── phase4_run.log                     # Live execution logs
└── README.md                              # Module README
```

## Objective

Determine whether adding a transliterated representation (e.g. Devanagari ↔ Latin) recovers ground-truth entity pairs that standard character n-gram retrieval misses, without creating an unmanageable candidate volume explosion.

## Transliteration Library

- **Library**: `indic-transliteration` (sanscript module)
- **Version**: `2.3.82`
- **License**: MIT License
- **Schemes**: Devanagari to Latin (ITRANS / ISO)
- **Determinism**: Fully deterministic, local, offline execution (no external APIs).

## Hardware & AWS Guidance

- **Execution Environment**: AWS EC2 instance recommended (`c6i.4xlarge`: 16 vCPU, 32 GB RAM, 100 GB EBS).
- **GPU Usage**: Not required. Workload is sparse TF-IDF matrix multiplication (`sparse_dot_topn`).

## Usage

1. Preprocess raw TSV data to Parquet (if not already completed):
   ```bash
   python src/prepare_data.py
   ```

2. Run Phase 4 experiment:
   ```bash
   python phase_4_Shishya/src/phase4_transliteration_retrieval.py --config phase_4_Shishya/config/phase4_config.json
   ```
