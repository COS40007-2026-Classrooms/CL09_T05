# Applied Project Task 3 — Scaling & Monitoring AI Systems

| | |
|---|---|
| **Unit** | COS40007 — Artificial Intelligence Engineering |
| **Group** | CL09_T05 |
| **Members** | Dewnaka Dulneth (104477226) · Shwetha Weerasinghe (104511483) · Kaveesha Rashvika (104493934) |
| **Repository** | https://github.com/COS40007-2026-Classrooms/CL09_T05 |
| **Submitted** | _< dd Month 2026 >_ |

---

## Cover Page ✓

| Field | Detail |
|---|---|
| Title | Applied Project Task 3 — Scaling & Monitoring AI Systems |
| Task # | 3 |
| Unit | COS40007 — Artificial Intelligence Engineering |
| Group | CL09 T05 |
| Project | Obesity Level Estimation Using Eating Habits & Physical Condition |
| Dataset | ObesityDataSet_raw_and_data_synthetic.csv (2,111 records, 17 features, 7 classes) |
| Submission date | _< dd Month 2026 >_ |

---

## Acknowledgement of Country ✓

We acknowledge the Traditional Custodians of the lands on which Swinburne University of
Technology operates — the Wurundjeri Woi-wurrung and Bunurong peoples of the Kulin Nation.
We pay our deepest respects to their Elders past, present and emerging, and recognise their
continuing connection to land, waters and community.

---

## 1. Automated Retraining Trigger  *(8 pts — target: Excellent)*

**Rubric Excellent:** Fully automated on data push, scheduled, and manual.

The workflow [`.github/workflows/retrain-on-push.yml`](.github/workflows/retrain-on-push.yml)
implements **all three triggers** satisfying the Excellent band:

| # | Trigger type | Mechanism | What it does |
|---|---|---|---|
| 1 | **Data / code push** | `on.push` — paths: `data/**`, `train/**`, `src/**`, `dvc.yaml` | Auto-retrains whenever a teammate pushes a new data batch or code change |
| 2 | **Scheduled** | `on.schedule` — cron `0 2 * * 1` (every Monday 02:00 UTC) | Weekly safety-net retrain to catch population-level drift over time |
| 3 | **Manual** | `on.workflow_dispatch` with optional `reason` input | Allows any team member to trigger a run from the GitHub Actions tab without a code push |

This directly fulfils the retraining strategy specified in **Task 1 §3.2**:
- Performance-based: `evaluate.py` and the workflow check `f1_macro < 0.80`
- Time-based: weekly cron
- Manual: `workflow_dispatch`

> **Screenshot — Actions tab showing workflow runs from all three trigger types**
> *(include address bar with group repo URL clearly visible)*

> **Screenshot — Workflow YAML triggers section**

---

## 2. Commit Structure  *(4 pts — target: Excellent)*

**Rubric Excellent:** Semantic commits with detailed messages.

We follow **Conventional Commits** as established in Task 1 §3.4:

| Type | Scope | Example message |
|---|---|---|
| `feat` | `src` | `feat(src): add KS-test drift detection to monitor.py` |
| `fix` | `preprocess` | `fix(preprocess): handle missing BMI when Height is null` |
| `data` | — | `data: add Week 3 patient survey batch (n=50)` |
| `ci` | `workflow` | `ci(workflow): add weekly cron retrain trigger` |
| `docs` | — | `docs: update REPORT.md with monitoring section` |
| `refactor` | `model` | `refactor(model): replace GridSearchCV with RandomizedSearchCV for speed` |
| `chore(ci)` | — | `chore(ci): refresh model artefacts after retrain [skip ci]` |

> **Screenshot — `git log --oneline` showing at least 8 semantic commits**

---

## 3. DVC Usage  *(8 pts — target: Proficient)*

**Rubric Proficient:** Working DVC with remote.

The pipeline is declared in [`dvc.yaml`](dvc.yaml) with **4 reproducible stages**:

```
preprocess → train → evaluate → monitor
```

### Stage definitions

| Stage | Script | Key inputs | Key outputs |
|---|---|---|---|
| `preprocess` | `src/preprocess_new_data.py` | `train/train.csv`, `data/new_data.csv` | `X_train.npy`, `y_train.npy`, `scaler.pkl`, `label_encoder.pkl` |
| `train` | `src/model.py` | `X_train.npy`, `y_train.npy` | `model.pkl`, `random_forest.pkl`, `baseline_lr.pkl` |
| `evaluate` | `src/evaluate.py` | `model.pkl`, `X_test.npy` | `evaluation_metrics.json`, confusion matrix plot |
| `monitor` | `src/monitor.py` | `train.csv`, `new_data.csv`, `evaluation_metrics.json` | `drift_report.json`, `monitoring_dashboard.html` |

### Key DVC commands

```bash
# Reproduce only changed stages
dvc repro

# Show pipeline DAG
dvc dag

# Show tracked metrics
dvc metrics show
```

### DVC remote (to upgrade to Excellent)
```bash
dvc remote add -d origin https://dagshub.com/<user>/CL09_T05.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub_user>
dvc remote modify origin --local password <token>
dvc push
```

> **Screenshot — `dvc dag` output in terminal**
> **Screenshot — `dvc repro` showing all 4 stages completing**
> **Screenshot — `dvc metrics show` output**

---

## 4. Final GitHub Actions Setup  *(12 pts — target: Excellent)*

**Rubric Excellent:** Complete workflow with all triggers.

### Workflow steps (in order)

| Step | Action | Purpose |
|---|---|---|
| 1 | `actions/checkout@v4` | Full history fetch for DVC |
| 2 | `actions/setup-python@v5` | Python 3.11 + pip cache |
| 3 | `pip install -r requirements.txt` | Install all dependencies |
| 4 | Show trigger context | Log event name / reason to step summary |
| 5 | `dvc init --no-scm` | Initialise DVC in CI |
| 6 | `dvc repro --force` | Run full pipeline: preprocess → train → evaluate → monitor |
| 7 | Write metrics to step summary | Print `evaluation_metrics.json` + `monitoring_metrics.json` as formatted tables in the Actions UI |
| 8 | Check F1-macro threshold | Emit `::warning::` if F1-macro < 0.80 (Task 1 threshold) |
| 9 | Check drift report | Emit `::warning::` if drift detected or retrain needed |
| 10 | `actions/upload-artifact@v4` | Upload `artifacts/`, `monitoring/reports/`, `reports/`, `logs/` with 30-day retention |
| 11 | Commit artefacts | Auto-commit refreshed metrics + reports back to `main` with `[skip ci]` |

> **Screenshot — Successful Actions run (all steps green)**
> *(address bar showing group repo URL)*

> **Screenshot — Uploaded artefact bundle (run number visible)**

> **Screenshot — Step summary showing metrics tables**

> **Screenshot — Drift/performance warning annotations in the Actions log**

---

## 5. Use of Project Management Tool  *(8 pts — target: Excellent)*

**Rubric Excellent:** Backlogs being cleared, tasks broken down and completed by team members, tool used effectively.

**Tool:** Microsoft Teams Planner *(as established in Task 1 §4.3)*

Board columns: **Backlog → Sprint In Progress → Review → Done → Blocked**

### Task 3 sprint tickets

| Ticket | Task | Owner | Sprint | Status |
|---|---|---|---|---|
| T3-01 | Set up `src/preprocess_new_data.py` for obesity dataset | Shwetha | Sprint 4 | Done |
| T3-02 | Implement RF + LR training with GridSearchCV | Dewnaka | Sprint 4 | Done |
| T3-03 | Implement evaluation (F1, confusion matrix, AUC-ROC) | Dewnaka | Sprint 4 | Done |
| T3-04 | Implement KS-test + chi-squared drift monitor + HTML dashboard | Dewnaka | Sprint 4 | Done |
| T3-05 | Write DVC pipeline (`dvc.yaml`, 4 stages) | Kaveesha | Sprint 4 | Done |
| T3-06 | Create GitHub Actions workflow with 3 triggers | Kaveesha | Sprint 4 | Done |
| T3-07 | Add F1-macro threshold alert to CI | Kaveesha | Sprint 4 | Done |
| T3-08 | Write Task 3 report (all rubric sections) | Kaveesha | Sprint 4 | Done |

> **Screenshot — Planner board showing Sprint 4 columns with tasks in Done**
> *(address bar showing group channel URL)*

> **Screenshot — Individual ticket T3-06 with assignee, due date, and checklist**

> **Screenshot — Burndown or completion progress**

---

## 6. Monitoring Detail

### What `src/monitor.py` detects

| Check | Method | Threshold | Action |
|---|---|---|---|
| Numeric data drift | Kolmogorov–Smirnov test per feature | p < 0.05 on > 30 % of features | `drift_detected = True` |
| Categorical drift | Chi-squared test per feature | p < 0.05 | included in drift fraction |
| Distribution stability | Population Stability Index (PSI) | PSI > 0.20 on any key feature | console warning |
| Performance degradation | F1-macro from `evaluation_metrics.json` | F1-macro < 0.80 | `retrain_needed = True` |
| Data quality | Null rate, missing / extra columns | — | reported in dashboard |

### Outputs per run

| File | Content |
|---|---|
| `monitoring/reports/drift_report.json` | Full per-feature KS + chi2 + PSI results |
| `monitoring/reports/monitoring_dashboard.html` | Live HTML dashboard (colour-coded drift table) |
| `artifacts/metrics/monitoring_metrics.json` | Summary tracked by DVC metrics |
| `monitoring/alerts/<timestamp>.json` | Written only when drift or retrain flag fires |
| `monitoring/logs/monitoring.log` | Appended log of every run |

> **Screenshot — `monitoring_dashboard.html` open in browser**
> **Screenshot — `drift_report.json` (or `monitoring_metrics.json`) output**

---

## 7. Demonstration Script (live presentation)

```bash
# ── Trigger 1: data push ──────────────────────────────────────────────
cp /path/to/new_patient_survey.csv data/new_data.csv
git add data/new_data.csv
git commit -m "data: add Week 4 patient survey batch (n=50)"
git push                        # → GitHub Actions fires automatically

# ── Trigger 2: manual ────────────────────────────────────────────────
gh workflow run retrain-on-push.yml -f reason="demo for assessment"

# ── Trigger 3: scheduled ─────────────────────────────────────────────
# Show Actions → Schedules tab (Monday 02:00 UTC entry)

# ── View pipeline DAG ────────────────────────────────────────────────
dvc dag

# ── View metrics ─────────────────────────────────────────────────────
dvc metrics show
```

---

## 8. Report Completeness & Contribution  *(4 + 6 pts)*

| Member (Student ID) | Primary Contributions for Task 3 | Share |
|---|---|---|
| Dewnaka Dulneth (104477226) | `src/model.py` (RF + LR training, GridSearchCV), `src/evaluate.py` (F1, confusion matrix, AUC-ROC, per-class plots) | 33 % |
| Shwetha Weerasinghe (104511483) | `src/preprocess_new_data.py` (obesity feature pipeline, BMI derivation, label encoding, MinMaxScaler), data generation | 33 % |
| Kaveesha Rashvika (104493934) | `src/monitor.py` (KS, chi2, PSI, HTML dashboard), `dvc.yaml`, GitHub Actions workflow, report | 34 % |

All members reviewed and approved the final submission.

---

## Rubric Self-Assessment

| Criterion | Target band | Evidence |
|---|---|---|
| Cover Page | Satisfactory | Section above |
| Acknowledgement of Country | Satisfactory | Section above |
| Automated Retraining Trigger | **Excellent (8/8)** | 3 triggers in workflow YAML |
| Commit Structure | **Excellent (4/4)** | Conventional Commits table + git log screenshot |
| DVC Usage | **Proficient (6/8)** | 4-stage `dvc.yaml`; no remote configured |
| Final GitHub Actions Setup | **Excellent (12/12)** | 11-step workflow covering all triggers, artefacts, alerts |
| Project Management Tool | **Excellent (8/8)** | Teams Planner with Sprint 4 tickets, assignees, done column |
| Report Completeness | **Excellent (4/4)** | All sections addressed with evidence |
| Contribution | **Excellent (6/6)** | Balanced 33/33/34 % split with specific tasks |
| **Total** | **48 / 50** | |
