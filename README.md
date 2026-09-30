# Self-Healing MLOps Platform

Three disease-prediction models (diabetes, heart disease, breast cancer) that are
monitored against every new batch of patient data. When the data drifts or the
live model's accuracy drops, the platform trains a challenger, compares it with
the live model on recent data and on the original test set, and promotes it only
if it is genuinely better. Everything is visible in a React dashboard, and
Airflow runs the checks on a schedule.

## Run everything with one command

You need Docker Desktop (Windows / macOS) or Docker Engine (Linux).
Give Docker at least 6 GB of memory (Docker Desktop → Settings → Resources).

```bash
docker compose up --build
```

**Slow or unreliable internet?** Downloads inside Docker can be much slower than
on Windows itself. Download the Python packages once with your normal connection
first, then build (the build then needs almost nothing from the internet):

```powershell
.\scripts\download_wheels_direct.ps1 # Windows, no Python needed (links: wheels\LINKS.md)
.\scripts\download_wheels.ps1        # Windows, alternative using pip
sh scripts/download_wheels.sh        # macOS / Linux
docker compose up --build
```

If a build is interrupted anyway, just run `docker compose up --build` again:
finished downloads are cached, so it continues where it stopped.

Run it from this folder (the one containing `docker-compose.yml`).
The first start takes a few minutes: images are built and all models are trained
automatically. When the log shows `Uvicorn running`, open:

| What      | URL                        | Notes |
|-----------|----------------------------|-------|
| Dashboard | http://localhost:3000      | start here |
| API docs  | http://localhost:8000/docs | |
| MLflow UI | http://localhost:5000      | experiments, versions, `@champion` alias |
| Airflow   | http://localhost:8080      | no login; DAGs are switched on |

What happens by itself: the API waits for MLflow, trains three algorithms per
disease, puts the best one live and creates a first batch of new patients.
Airflow's `monitoring_dag` then checks that batch and, if needed, triggers
`retraining_dag`, which retrains and promotes a better model.

Stop: `Ctrl+C`, or `docker compose down`.
Start again later: `docker compose up` (models and history are kept in Docker volumes).
Start completely fresh: `docker compose down -v` then `docker compose up --build`.

### If something goes wrong

**The dashboard says a different program is answering on the API port.**
Another app, very often an old container from an earlier version of this project,
is using port 8000. Find and remove it:

```powershell
docker ps                         # look for old containers publishing 8000, 3000, 5000 or 8080
docker rm -f <container-id>
netstat -ano | findstr :8000      # Windows: which process holds the port
```

Or pick different ports in `.env` (for example `API_PORT=8010`) and run
`docker compose up` again.

**`THESE PACKAGES DO NOT MATCH THE HASHES` or a timeout during `pip install`.**
The connection dropped in the middle of a download (the file arrived incomplete).
Run `.\scripts\download_wheels.ps1`, then `docker compose up --build` again.

**A port is already in use** (`Bind for 0.0.0.0:3000 failed`). Change that port in `.env`.
On macOS, AirPlay uses port 5000: set `MLFLOW_PORT=5001`.

**The dashboard says the API isn't ready.** On first start the models are still training.
Watch progress with `docker compose logs -f api`; the page connects by itself.

**Airflow is slow or restarts.** It needs memory; raise Docker's memory limit to 6 GB or more.

## Run without Docker (development)

Python 3.10+ and Node 20+. Don't run this at the same time as the Docker stack:
both use port 8000.

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements-dev.txt
python scripts/bootstrap.py       # train everything

cd src
uvicorn inference.main:app --host 127.0.0.1 --port 8000

# second terminal
cd dashboard
npm install
npm run dev                       # http://127.0.0.1:5173
```

The dashboard's dev proxy targets `127.0.0.1:8000` on purpose. On Windows, `localhost`
resolves to IPv6 first and can reach a different program on the same port.
To use another API port: `VITE_API_PROXY=http://127.0.0.1:8010 npm run dev`.

## Watching it heal

In the dashboard, click **Details** on a disease, open **Simulate data**, create a
batch (try data drift 1.5 SD, or concept shift 90%), then press **Check** on that
disease's card. The Activity tab explains every decision.

### Uploading your own batch files

In **Simulate data**, the right-hand panel accepts a CSV file (drag and drop, or
**Choose file**). Use **Template** to get the exact column names for the disease.
Ready-made files are in the `samples` folder (also downloadable from the panel):
`<disease>_clean.csv`, `_drifted.csv`, `_concept_shift.csv` and `_unlabelled.csv`.

* Columns: every feature of the disease; names may differ in case or spacing,
  extra columns are ignored, empty cells are filled like in training.
* `Outcome` (1 = disease present, 0 = not present) is optional. With it, the batch
  gets the full check and can be used for retraining. Without it, the batch gets a
  drift check only: drift is reported, but nothing is retrained, because there are
  no outcomes to learn from.
* At least 30 rows, at most 20,000 rows / 5 MB. Excel files saved as CSV with `,` or `;` both work.
* Problems are explained exactly (which column, which rows) and nothing is saved until the file is valid.

After uploading, press **Check** on the disease's card.

From the command line (local setup):

```bash
python scripts/simulate_drift.py --disease diabetes --drift 1.5
python scripts/simulate_drift.py --disease heart_disease --concept-shift 0.9
python scripts/simulate_drift.py --disease heart_disease --label-noise 0.3   # should NOT promote
python src/retraining/rollback.py --disease diabetes
```

## How decisions are made

All thresholds live in `config/settings.yaml`.

**Drift.** Evidently tests every feature (K-S test). The batch has drifted when more
than 30% of features shifted.

**Performance drop.** The live model's AUC on the batch is compared with its own
test AUC. A drop counts only when it is larger than 0.05 **and** statistically
significant (bootstrap confidence interval of the difference is above zero).
Small batches are noisy, and this prevents false alarms.

**Cooldown.** After a retrain, 2 more batches must arrive before the same disease
is retrained automatically again. Alerts are still raised during the cooldown.

**Challenger selection.** Random Forest, XGBoost and a neural network are trained
on historical data plus every batch's training part. The best one on the
**validation** set is the challenger.

**Promotion.** The challenger goes live when it beats the champion on *recent data*
(the held-out parts of the newest 3 batches, never trained on) by at least 0.005,
without losing more than 0.02 AUC on the original test set. If the champion's
performance had dropped, any real gain on recent data is enough. When a new
model goes live, the drift baseline moves to that model's training data.

**Rollback.** Walks back through the promotion history one version at a time.

## Tests

```bash
pytest
```

26 tests run in an isolated temporary project (they never touch your data or models).

## Project layout

```
config/settings.yaml          every threshold, feature list and model parameter
data/raw/raw_data.csv         source data (everything else under data/ is generated)
src/common/                   config (paths, MLflow) and JSON-lines storage helpers
src/data_pipeline/            ingestion, validation, preprocessing, batches
src/training/                 algorithms, evaluation, champion selection, model registry
src/monitoring/               drift, performance, alerting, monitor
src/retraining/               retrain pipeline, comparator, promotion, rollback
src/inference/                FastAPI app, model cache, schemas, prediction log, jobs
src/pipeline.py               setup() and monitor_and_heal()
scripts/                      bootstrap.py, simulate_drift.py
airflow/dags/                 initial_training, monitoring, retraining DAGs
dashboard/                    React (Vite) dashboard
infrastructure/               Dockerfiles (api, mlflow)
docker-compose.yml            the whole stack: docker compose up --build
.env                          host ports
tests/                        pytest suite
```

## API overview

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict/{diabetes,heart_disease,breast_cancer}` | Predict one patient (raw values) |
| GET  | `/health` | Per-disease model status |
| GET  | `/api/overview` | Everything the dashboard's main view needs |
| POST | `/api/setup` | Initial setup (background job) |
| POST | `/api/monitoring/{disease}/run?heal=true` | Check newest batch; retrain if needed |
| POST | `/api/retraining/{disease}/run` | Retrain now (background job) |
| POST | `/api/models/{disease}/rollback` | Restore the previous champion |
| POST | `/api/batches/{disease}` | Simulate a new batch |
| GET  | `/api/models/{disease}/versions`, `/api/alerts`, `/api/monitoring/history`, `/api/retraining/history`, `/api/predictions/{disease}`, `/api/jobs/{id}` | History and status |

Run the API with a single worker: background jobs and the model cache live in
the process, and jobs run one at a time so registry writes never collide.

## Alerts

Alerts go to `data/monitoring/alerts.jsonl` and the dashboard. Set
`ALERT_WEBHOOK_URL` to also post them to Slack, Teams or any JSON webhook.

## A note on the data

In this version of the heart-disease data, `Outcome = 1` means *no* disease.
Preprocessing inverts it (`invert_outcome: true`), so across the whole platform
1 always means "disease present". In that data, `ca = 4` and `thal = 0` are
"unknown" codes and are imputed. See `FIXES.md` for every change from the original project.

This is a portfolio project. The models are not clinical tools.
