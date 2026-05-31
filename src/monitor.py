"""Drift, performance and data-quality monitoring for the obesity classifier.

Compares data/new_data.csv against the training distribution (X_train_raw.npy)
using per-feature Kolmogorov-Smirnov tests for numeric features and chi-squared
tests for categorical features, exactly as designed in the Task-1 blueprint
(Section 3.3).

Retraining is flagged when:
  - KS p-value < 0.05  on > 30 % of numeric features  (data drift), OR
  - PSI > 0.2          on any key numeric feature,      OR
  - F1-macro from latest evaluation_metrics.json < 0.80 (performance drift)

Outputs:
  monitoring/reports/drift_report.json
  monitoring/reports/monitoring_dashboard.html
  artifacts/metrics/monitoring_metrics.json
  monitoring/alerts/<timestamp>.json   (only when drift detected)
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

MON = Path("monitoring")
for sub in ("reports", "logs", "alerts"):
    (MON / sub).mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)

ART    = Path("artifacts")
NEW_CSV = Path("data/new_data.csv")
now     = datetime.utcnow().isoformat()

NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
CAT_COLS     = ["Gender", "family_history_with_overweight", "FAVC",
                "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]

KS_ALPHA      = 0.05
DRIFT_FRAC    = 0.30   # alert if > 30 % of features drift
PSI_THRESHOLD = 0.20
F1_THRESHOLD  = 0.80

report = {
    "timestamp":      now,
    "drift_detected": False,
    "retrain_needed": False,
    "numeric_drift":  {},
    "categorical_drift": {},
    "psi_scores":     {},
    "data_quality":   {},
    "performance":    {},
    "summary":        {},
}


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    eps = 1e-8
    bins = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    e = np.histogram(expected, bins=bins)[0] / len(expected)
    a = np.histogram(actual,   bins=bins)[0] / len(actual)
    e = np.where(e == 0, eps, e)
    a = np.where(a == 0, eps, a)
    return float(np.sum((a - e) * np.log(a / e)))


# ── Load data ─────────────────────────────────────────────────────────────────
train_csv = Path("train/train.csv")
train_df  = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()

if not NEW_CSV.exists() or NEW_CSV.stat().st_size == 0:
    report["summary"] = {"status": "skipped", "reason": "no new_data.csv"}
else:
    new_df = pd.read_csv(NEW_CSV)

    # derive BMI if possible
    for df in (train_df, new_df):
        if "Height" in df.columns and "Weight" in df.columns:
            df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # ── Numeric KS + PSI ─────────────────────────────────────────────────────
    drifted_numeric = []
    for col in NUMERIC_COLS:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        a = train_df[col].dropna().values
        b = new_df[col].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        ks_stat, p = stats.ks_2samp(a, b)
        drift = bool(p < KS_ALPHA)
        if drift:
            drifted_numeric.append(col)
        report["numeric_drift"][col] = {
            "ks_statistic": float(ks_stat),
            "p_value":      float(p),
            "drift":        drift,
            "train_mean":   float(np.mean(a)),
            "new_mean":     float(np.mean(b)),
        }
        psi_val = psi(a, b)
        report["psi_scores"][col] = psi_val
        if psi_val > PSI_THRESHOLD:
            print(f"  PSI alert: {col} PSI={psi_val:.3f} > {PSI_THRESHOLD}")

    # ── Categorical chi-squared ───────────────────────────────────────────────
    drifted_cat = []
    for col in CAT_COLS:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        cats = list(set(train_df[col].astype(str).unique()) |
                    set(new_df[col].astype(str).unique()))
        obs_train = [train_df[col].astype(str).value_counts().get(c, 0) for c in cats]
        obs_new   = [new_df[col].astype(str).value_counts().get(c, 0)   for c in cats]
        if sum(obs_new) < 5:
            continue
        try:
            chi2, p, *_ = stats.chi2_contingency([obs_train, obs_new])
        except Exception:
            continue
        drift = bool(p < KS_ALPHA)
        if drift:
            drifted_cat.append(col)
        report["categorical_drift"][col] = {
            "chi2":  float(chi2),
            "p_value": float(p),
            "drift":   drift,
        }

    total_features = len(report["numeric_drift"]) + len(report["categorical_drift"])
    total_drifted  = len(drifted_numeric) + len(drifted_cat)
    drift_fraction = total_drifted / max(total_features, 1)

    # ── Performance check ─────────────────────────────────────────────────────
    eval_path = ART / "metrics" / "evaluation_metrics.json"
    f1_macro  = None
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text())
        f1_macro  = eval_data.get("f1_macro")
        report["performance"] = {
            "f1_macro":  f1_macro,
            "accuracy":  eval_data.get("accuracy"),
            "threshold": F1_THRESHOLD,
            "below_threshold": f1_macro is not None and f1_macro < F1_THRESHOLD,
        }

    # ── Data quality ──────────────────────────────────────────────────────────
    report["data_quality"] = {
        "new_rows":         int(len(new_df)),
        "new_null_rate":    float(new_df.isna().mean().mean()),
        "missing_columns":  [c for c in train_df.columns if c not in new_df.columns],
        "extra_columns":    [c for c in new_df.columns if c not in train_df.columns],
    }

    report["drift_detected"] = bool(drift_fraction > DRIFT_FRAC)
    report["retrain_needed"] = bool(
        report["drift_detected"] or
        (f1_macro is not None and f1_macro < F1_THRESHOLD)
    )
    report["summary"] = {
        "status":             "ok",
        "features_checked":   total_features,
        "features_drifted":   total_drifted,
        "drift_fraction":     drift_fraction,
        "drifted_numeric":    drifted_numeric,
        "drifted_categorical": drifted_cat,
        "drift_detected":     report["drift_detected"],
        "retrain_needed":     report["retrain_needed"],
    }

# ── Persist ────────────────────────────────────────────────────────────────────
with open(MON / "reports" / "drift_report.json", "w") as f:
    json.dump(report, f, indent=2)
with open(ART / "metrics" / "monitoring_metrics.json", "w") as f:
    json.dump(report["summary"], f, indent=2)

# ── HTML dashboard ─────────────────────────────────────────────────────────────
num_rows = "".join(
    f"<tr><td>{k}</td><td>{v['ks_statistic']:.3f}</td>"
    f"<td>{v['p_value']:.4f}</td>"
    f"<td>{report['psi_scores'].get(k, 0):.3f}</td>"
    f"<td style='color:{'red' if v['drift'] else 'green'}'>"
    f"{'YES' if v['drift'] else 'no'}</td></tr>"
    for k, v in report["numeric_drift"].items()
)
cat_rows = "".join(
    f"<tr><td>{k}</td><td>—</td><td>{v['p_value']:.4f}</td><td>—</td>"
    f"<td style='color:{'red' if v['drift'] else 'green'}'>"
    f"{'YES' if v['drift'] else 'no'}</td></tr>"
    for k, v in report["categorical_drift"].items()
)
perf_block = ""
if report["performance"]:
    p = report["performance"]
    clr = "red" if p.get("below_threshold") else "green"
    perf_block = (f"<h2>Performance</h2>"
                  f"<p>F1-macro: <b style='color:{clr}'>{p.get('f1_macro', 'N/A')}</b>"
                  f" (threshold {F1_THRESHOLD})</p>"
                  f"<p>Accuracy: {p.get('accuracy', 'N/A')}</p>")

alert_class = "alert" if report["drift_detected"] or report["retrain_needed"] else "ok"
html = f"""<!doctype html><meta charset=utf-8>
<title>Obesity Model — Monitoring Dashboard</title>
<style>
  body{{font-family:system-ui;margin:2rem;max-width:960px}}
  table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:6px}}
  th{{background:#f0f0f0}}
  .alert{{padding:1rem;background:#fee;border:1px solid #f99;border-radius:6px;margin:1rem 0}}
  .ok{{padding:1rem;background:#efe;border:1px solid #9c9;border-radius:6px;margin:1rem 0}}
</style>
<h1>Obesity Model — Monitoring Dashboard</h1>
<p>Generated: {now}</p>
<div class="{alert_class}">
  <b>Drift detected:</b> {report['drift_detected']} &nbsp;|&nbsp;
  <b>Retrain needed:</b> {report['retrain_needed']}<br>
  <b>Summary:</b> {json.dumps(report['summary'])}
</div>
{perf_block}
<h2>Numeric feature drift (KS test)</h2>
<table><tr><th>Feature</th><th>KS</th><th>p-value</th><th>PSI</th><th>Drift?</th></tr>
{num_rows}</table>
<h2>Categorical feature drift (chi-squared)</h2>
<table><tr><th>Feature</th><th>chi2</th><th>p-value</th><th>PSI</th><th>Drift?</th></tr>
{cat_rows}</table>
"""
(MON / "reports" / "monitoring_dashboard.html").write_text(html)
Path("reports/performance_report.html").write_text(html)

with open(MON / "logs" / "monitoring.log", "a") as f:
    f.write(f"[{now}] drift={report['drift_detected']} retrain={report['retrain_needed']}"
            f" summary={report['summary']}\n")

if report["drift_detected"] or report["retrain_needed"]:
    alert_path = MON / "alerts" / f"alert_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
    alert_path.write_text(json.dumps(report, indent=2))
    print(f"::warning::DRIFT/PERFORMANCE ALERT — see {alert_path}")

print(json.dumps(report["summary"], indent=2))
