# Amazon ML Entity Resolution

Analysis and baseline experiments for the Amazon ML 2026 Business Entity Resolution challenge.
The project compares noisy business records from three independent sources and identifies
Source 2 and Source 3 records that refer to each Source 1 business.

## Repository layout

- `src/prepare_data.py` - normalizes the TSV data and creates Parquet files for analysis.
- `src/baseline_experiment.py` - evaluates deterministic exact-match signals with DuckDB.
- `src/ngram_retrieval_experiment.py` - evaluates TF-IDF n-gram candidate retrieval.
- `src/data_audit.py` and `src/dataset_recon.py` - inspect data quality and dataset structure.
- `src/check_*.py` and `src/find_hard_negatives.py` - targeted diagnostics.
- `reports/` - experiment metrics, audit results, and failure analysis.
- `Dataset/student_resource/` - challenge documentation and the submission validator.

## Data

The raw challenge data is intentionally excluded from this repository because it is several
gigabytes in size and is not suitable for a normal GitHub repository. Obtain the challenge
data separately and place it under:

```text
Dataset/student_resource/dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Generated Parquet files belong in `data/`. The `data/`, `duckdb_temp/`, `output/`, and
submission-data directories are ignored by Git.

See [Dataset/student_resource/README.md](Dataset/student_resource/README.md) for the full
challenge specification and output format.

## Setup

Python 3.10 or newer is recommended. Create an environment and install the analysis
 dependencies:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install duckdb numpy pandas pyarrow scikit-learn scipy
```

## Run the analysis

Run commands from the repository root:

```bash
python src/prepare_data.py
python src/baseline_experiment.py
python src/ngram_retrieval_experiment.py
```

Preprocessing reads the training TSV files from `Dataset/student_resource/dataset/train/`
and writes normalized Parquet files to `data/`. The experiments read those Parquet files
and write metrics and diagnostics to `reports/`.

The diagnostic scripts are intended to run after preprocessing and the relevant experiment:

```bash
python src/data_audit.py
python src/dataset_recon.py
python src/check_gt_counts.py
python src/check_union_recall.py
```

Some scripts process millions of rows and require substantial disk space and memory. Review
the configuration constants near the top of each script before running on a smaller machine.

## Validate a submission

After creating `output/matching_results.tsv` and `output/candidate_pairs.tsv`, run the
standard-library validator from the challenge directory:

```bash
python Dataset/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir Dataset/student_resource/dataset/test
```

A successful validation exits with status 0 and prints `PASS`.

## Notes

This repository contains research and baseline analysis code rather than a packaged
production inference service. The reports document the current experiments and their
limitations.
