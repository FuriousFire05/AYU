# AYU: Reliable Remaining Useful Life Prediction

AYU is a BITS F464 machine-learning project using NASA C-MAPSS. The current
milestone implements reproducible classical baselines for FD001 through FD004.
It intentionally uses the exact, uncapped target
`RUL = final engine cycle - current cycle`.

## Reproduce the four-subset milestone

Place the official C-MAPSS text files under `data/raw/CMaps/`, then install the
locked environment and run development experiments:

```powershell
uv sync
uv run python scripts/run_baselines.py
```

Development runs use an engine-disjoint seeded 80/20 split and do not open the
official `RUL_FD00x.txt` labels. To perform the explicit, locked endpoint
evaluation after model selection:

```powershell
uv run python scripts/run_baselines.py --evaluate-test
```

An existing official endpoint metrics file prevents accidental repeat testing.
Only an explicitly authorized rerun should use `--force-endpoint`. A subset can
be run alone with `--subsets FD002`.

Each `outputs/baselines/fd00x/` directory records the configuration and software
versions, seed and engine split, training-only sensor selection, exact feature
list/count, validation metrics and cycle-level predictions, model-selection
decision, plots, and (when unlocked) official endpoint metrics and predictions.
Cross-subset summaries distinguish validation cycles from one endpoint per test
engine.

## Methodological boundaries

- Complete engines—not rows—are split between development training and validation.
- Sensor variance filtering is learned only from development-training engines.
- Rolling mean and standard-deviation features use trailing windows 5, 10, and 20,
  reset for each engine, with no future observations.
- `unit_id` is never a predictor. The controlled XGBoost comparison includes and
  excludes observable `cycle` while keeping all other features and parameters fixed.
- Official test labels are never used for feature selection, early stopping, model
  choice, or hyperparameter tuning. Only the validation-selected XGBoost variant
  is refitted on all training engines and evaluated at official test endpoints.
- These are uncalibrated classical point-prediction baselines. Condition-aware
  normalization, uncertainty calibration, deep learning, APIs, and dashboards are
  outside this milestone.

## Verification

```powershell
uv run pytest -q
uv run ruff check src tests scripts
uv run pyright
```

The protected CIA1 checkpoint is tagged `cia1-fd001-baseline`. Its FD001
reference is validation RMSE 32.0213 / R² 0.7621 for XGBoost with cycle, and
official endpoint MAE 19.0819, RMSE 25.4518, R² 0.6249, NASA score 12111.8971.
