# Retraining & Monitoring — Group Task 3

Automated MLOps pipeline that retrains and monitors a 1D-CNN regression model
whenever new data is pushed.

## Triggers
- **Push** to `data/**`, `train/**`, `src/**`, or `dvc.yaml`
- **Schedule** weekly (Monday 02:00 UTC)
- **Manual** via `workflow_dispatch` (Actions → *Run workflow*)

## Pipeline (DVC)
```
preprocess → train → evaluate → monitor
```
Run locally:
```bash
pip install -r requirements.txt
dvc repro
```

## Triggering a retrain
```bash
cp /path/to/new_data.csv data/
git add data/new_data.csv
git commit -m "data: add new batch for retraining"
git push
```

## Layout
```
.github/workflows/retrain-on-push.yml   # CI/CD
src/                                    # pipeline scripts
data/        train/        test/        # input data
artifacts/   models, data, preprocessing, metrics, metadata
monitoring/  reports, logs, alerts
reports/     plots + HTML dashboard
dvc.yaml                                # pipeline definition
```

See [REPORT.md](REPORT.md) for the Group Task 3 write-up.
