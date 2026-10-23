# Development

Use Python 3.12 or 3.13. Install the locked development environment:

```bash
uv sync --python 3.12 --locked --extra dev
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
```

Keep changes focused. Explain the behavior being changed and add a regression test when a
failure would affect predictions, input validation or output files. Synthetic fixtures belong
in `tests/fixtures/`; tournament text and model weights do not.

Before changing model settings, preprocessing, feature order, ranking ties or class order,
run a separate experiment. Record its input hashes and compare its output with the archived
reference. A higher local score alone does not establish an improvement in generalization.

Use a new output path for each run. Never overwrite reference artifacts. Keep experiment
results under `artifacts/`, which is ignored by Git. Do not commit datasets, credentials,
downloaded models or generated training checkpoints.

The technical report is generated from `docs/report.md`:

```bash
uv sync --locked --extra dev --extra report
uv run python scripts/build_report.py
```

Review the rendered PDF after editing it. The report and README must distinguish measured
software checks, historical competition scores and results confirmed by the author.
