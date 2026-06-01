# CL09_T05 — COS40007 AI Engineering

Group project repository. Each task lives in its own folder with its own
GitHub Actions workflow.

## task2/ — GitHub Actions CI/CD (Mercedes-Benz dataset)
1D CNN regression model. Workflow `.github/workflows/train.yml` trains the
model on every push to `task2/`.

## task3/ — Scaling & Monitoring (Obesity dataset)
Random Forest classifier with a DVC pipeline (preprocess → train → evaluate →
monitor) and drift monitoring. Workflow `.github/workflows/retrain-on-push.yml`
retrains on push, on a weekly schedule, and manually.
