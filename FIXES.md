# What was fixed

## Version 2.1 (one-command Docker, new dashboard)

- **"API unreachable" / `{"detail":"Not Found"}` in the dashboard.** The Vite dev proxy
  targeted `localhost:8000`. On Windows, Node resolves `localhost` to IPv6 `::1`
  first, where an old container from the original project was listening, so the
  dashboard talked to the wrong API. The proxy now targets `127.0.0.1`. A new
  `/api/info` endpoint lets the dashboard confirm it reached *this* API; if not, it
  says so and explains how to free the port.
- **One command.** `docker-compose.yml` now lives in the project root:
  `docker compose up --build` starts MLflow, the API, the dashboard and Airflow.
  The API trains the models automatically on first start; services wait for each
  other with health checks; ports are set in `.env`.
- **Airflow in the default stack**, with no login, DAGs unpaused, and no ML libraries
  in its image (the DAGs call the API over HTTP).
- **Windows line endings can't break containers.** The container entrypoint is a
  Python script, and `.gitattributes` enforces LF.
- **Data lives in Docker volumes** instead of a Windows bind mount (no permission or
  speed problems); `docker compose down -v` resets everything.
- **Smaller images:** `xgboost-cpu` (same `import xgboost`) instead of `xgboost`,
  which pulled ~300 MB of NVIDIA GPU libraries. macOS uses `xgboost` automatically.
- **New dashboard:** light layout with KPI cards, one card per disease with an AUC
  chart (baseline and drift markers), a drift bar chart, a version chart with
  rollback, a risk gauge for predictions and a batch simulator; responsive layout.

## Version 2.3 (CSV upload)

- **Upload batch files** from the dashboard next to the existing simulator, with a
  column template and sample files per disease. Files are validated first (missing
  columns, text in number columns, out-of-range values, bad Outcome values, too few
  rows) with messages naming the columns and rows to fix.
- **Batches without outcomes** get a drift-only check and are never used for
  retraining. Drift in such a batch raises an alert but does not start a retrain.
- Fixed a crash when monitoring a batch without an Outcome column.
- nginx accepts uploads up to 10 MB (the default 1 MB limit rejected larger CSVs).

## Version 2.2 (reliable builds on slow connections)

- **`pip install` failing inside Docker** (hash mismatch / timeouts on numpy and other
  large packages). The download was cut off on a slow connection. Now:
  `scripts/download_wheels.ps1` downloads the Linux packages once on Windows into
  `wheels/`, the image installs from there first, and a BuildKit pip cache keeps
  finished downloads between build attempts, so a retry continues instead of
  starting over.
- **MLflow and the API share one image** (the API image already contains mlflow),
  so the Python packages are downloaded and installed once instead of twice.

## Version 2.0

Each item below is a bug from the review of the original project, followed by what changed.

## Blockers: the platform could not run

1. **Empty `src/monitoring/` package imported everywhere.** Real modules now live
   there: `drift_detector`, `performance_monitor`, `alerting` and `monitor`.
   `monitoring_dag` no longer imports a `monitoring.monitor` that didn't exist.
2. **Batch folders didn't match the code** (`data/batches/batch_1` vs `data/batches/<disease>/batch_1`).
   There is now one layout, `data/batches/<disease>/batch_NNN/`, used by the
   generator, loader, monitor and API. Batches sort numerically, so `batch_10`
   comes after `batch_2`.
3. **MLflow DB held Windows paths to another folder.** The old `mlflow.db` and
   `mlruns/` are removed and regenerated on setup. Locally, artifacts are pinned
   inside `mlflow/artifacts`. In Docker, a tracking server proxies artifacts
   (`mlflow-artifacts:/...`), so no machine-specific paths are stored.
4. **Model artifacts never reached the containers.** The API talks to the MLflow
   server, which owns a named volume containing both the database and the artifacts.
5. **The Airflow volume hid `/opt/airflow/src`.** Airflow now uses the stock image
   and calls the API over HTTP. Nothing is baked into the image, so nothing can be hidden.
6. **MLflow UI SQLite URI was relative** (3 slashes). This was replaced by the
   tracking server (`sqlite:////mlflow/mlflow.db`).
7. **Monitoring always triggered retraining.** `check_<disease>` is now a
   `ShortCircuitOperator`: the trigger runs only when retraining is needed.

## Wrong results

8. **Batches overlapped train/test rows.** Batches are drawn only from a 10%
   *future pool* that is never used for training, validation or testing. Each
   batch carries `source_id`, so its evaluation rows are excluded from retraining.
9. **Model selection used the test set.** Selection now uses the validation set;
   test is reported only.
10. **Comparison ignored new data, and the 0.9994 + 0.01 rule was impossible.**
    Champion and challenger are compared on recent held-out batch data plus the
    test set, using configurable rules (see README).
11. **Train/serve skew (`Insulin=0`).** One fitted `DiseasePreprocessor` does the
    zero-as-missing handling, imputation and scaling. It is logged with each model
    version and used for serving.
12. **Endless retrain loop.** The baseline is read from the champion's registry
    tags. The drift reference moves with each promotion. Each batch is checked
    once (`state.json`), and a cooldown stops repeated retraining.
13. **Rollback ping-pong.** A promotion history is stored on the registered model,
    and rollback walks back through it.
14. **Multiple "Production" versions.** Deprecated stages are replaced by a single
    `champion` alias.
15. **Fragile new-version lookup.** The code now uses `model_info.registered_model_version`.
16. **Imputation leaked across splits.** The preprocessor is fitted on train only.
17. **`create_batches.py` copied `val.csv` and had the wrong docstring path.** It is
    now a real simulator with `--drift`, `--concept-shift` and `--label-noise`.

## API

18. **Model and scaler reloaded on every request.** There is now an in-memory
    `ModelStore` that refreshes when the champion alias changes.
19. **`/health` wasn't registered, used the wrong model name, and `.seconds` wrapped.**
    It is registered, reports per disease, and uses `total_seconds()`.
20. **Prediction middleware wasn't wired, was diabetes-only, and read missing fields.**
    `prediction_logger.py` logs every disease to its own CSV, and the response
    now includes `model_version` and `model_algorithm`.
21. **Broad XGBoost fallback hid errors.** All models are logged with the sklearn
    flavour, so there is one loader and no fallback.
22. **Every error was a 500, and there was no input validation.** Errors are now
    503 (no model), 404 (unknown disease), 409 (busy) and 422 (invalid input).
    Fields have ranges and extra fields are rejected.

## Infrastructure

23. **Unpinned UTF-16 `requirements.txt`.** Requirements are pinned and UTF-8, with
    `requirements-dev.txt` for tests. Airflow no longer installs ML libraries at all.
24. **Paths depended on the working directory.** `common/config.py` resolves
    everything from the project root (`MLOPS_PROJECT_ROOT`).
25. **Parallel writes to one SQLite DB.** API jobs run one at a time; Docker uses
    the MLflow server; `retraining_dag` has `max_active_runs=1`.
26. **No `.dockerignore`, and a ~80,000-file `venv` in the project.** Both are fixed.
27. **`validation.py` imported `ingestion_v2` and was never called.** It is fixed
    and runs before preprocessing.
28. **`retraining_dag` silently defaulted to diabetes.** It now fails with a clear
    message when `disease` is missing or unknown.
29. **Deprecated APIs.** Aliases replace stages, `name=` replaces `artifact_path=`,
    and `model_dump()` replaces `.dict()`.
30. **Empty or stale files.** The four empty config YAMLs became one
    `config/settings.yaml`. The empty tests became 26 real tests. Empty scripts,
    `f.txt`, `model_registry.py`, `mlflow_config.py` and stale CSVs are removed or
    replaced. The data duplicated at the top level of `data/processed/` and `data/reference/` is gone.

## Found while fixing

31. **Heart-disease labels were inverted.** `Outcome=1` meant *no* disease, so the
    API called healthy patients sick. Fixed with `invert_outcome: true`.
32. **`ca=4` and `thal=0` are "unknown" codes**, not values. They are now imputed.
33. **Small batches caused false alarms.** A performance drop must be statistically
    significant (bootstrap CI of the test-vs-batch AUC difference).
34. **Recent-data comparison used only ~10 patients for heart disease.** It now uses
    the held-out parts of the newest 3 batches.
35. **numpy/NaN values could break alert logging.** All logs go through `to_jsonable`.

## Added

- A React dashboard (`dashboard/`) with a per-disease monitor strip, activity
  timeline, model versions with rollback, a prediction form, a new-data
  simulator and initial setup.
- Operations endpoints under `/api/...`, a background job queue, and an
  optional alert webhook.
- `scripts/bootstrap.py`, `scripts/simulate_drift.py`, and a single docker-compose
  file that runs MLflow, the API, the dashboard and optionally Airflow.
