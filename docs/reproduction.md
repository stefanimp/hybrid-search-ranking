# Reproduction

## Installation

Python 3.12 and 3.13 are supported. `uv.lock` records the dependency resolution.
Use `uv sync --python 3.12 --locked --extra dev` for the locked development environment.
Alternatively, create a virtual environment with Python 3.12/3.13 and install with pip:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
hybrid-search demo
```

On Windows, use `py -3.12 -m venv .venv` if the default Python is another version.
The pip route uses pinned direct dependencies but does not enforce the uv transitive lock.

## Inputs

```text
assets/private/
  data/
    corpus_100k.csv        # document_id,title,text
    train_queries.csv     # query_id,query
    test_queries.csv      # query_id,query
    train_qrels.csv       # Id,Expected
  cache/
    basic_precision_candidates.joblib
    cross_encoder_dense_100k_top50_scores.joblib
    cross_encoder_dense_top100_scores.joblib
    dense_candidates.joblib
    tfidf_word_titlechar_100k_candidates.joblib
```

The original artifacts are available in the author's local copy. They are not a public release:
the repository alone supports the synthetic demo and tests, not full tournament replay.
No scripts for regenerating the five neural retrieval caches were present in the original delivery.
Use only trusted joblib files: loading them can execute Python code.

`hybrid-search check` validates CSV columns, IDs, query splits and relevance lists, and checks
that cache files exist. It does not deserialize or certify their contents.

## Full run

PowerShell:

```powershell
$env:PYTHONHASHSEED="0"
uv run hybrid-search run --data-dir assets/private/data --cache-dir assets/private/cache --output-dir artifacts/run-001
```

macOS/Linux:

```bash
PYTHONHASHSEED=0 uv run hybrid-search run --data-dir assets/private/data --cache-dir assets/private/cache --output-dir artifacts/run-001
```

The run performs five internal folds, fits the final ranker, saves
`models/modelo_final.joblib` and exports `submision_final.csv`. Historical Spanish artifact
names are retained to make the output easy to compare with the archived delivery.
Use `--include-variants` to export `submision_pruebas_1.csv` through `submision_pruebas_3.csv`.

## Verification

The original script and maintained pipeline produced identical CSV bytes on Windows with
Python 3.12.3, NumPy 2.3.4, pandas 3.0.2, scikit-learn 1.8.0, LightGBM 4.6.0 and joblib 1.5.3.
The maintenance cleanup was checked again against the same reference. All four CSVs also
coincide with one another on these inputs; the optional variants did not change predictions.
The baseline output digest and current checks are in `verification.json`.

[GitHub Actions](https://github.com/stefanimp/hybrid-search-ranking/actions/runs/39994015289) passed the core tests, lint and formatting checks
on Windows/Linux with Python 3.12/3.13. The verification record identifies the tested commit.
No hidden Kaggle labels were reevaluated.
