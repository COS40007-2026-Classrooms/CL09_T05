"""Evaluate the trained obesity classifier on the held-out test split.

Outputs:
  artifacts/metrics/evaluation_metrics.json  — accuracy, F1-macro, per-class F1
  reports/confusion_matrix.png
  reports/per_class_f1.png
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)

ART = Path("artifacts")
(ART / "metrics").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)

X_test = np.load(ART / "data" / "X_test.npy")
y_test = np.load(ART / "data" / "y_test.npy")
model  = joblib.load(ART / "models" / "model.pkl")
le     = joblib.load(ART / "preprocessing" / "label_encoder.pkl")

preds      = model.predict(X_test)
proba      = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
f1_macro   = float(f1_score(y_test, preds, average="macro"))
f1_weighted= float(f1_score(y_test, preds, average="weighted"))
accuracy   = float(accuracy_score(y_test, preds))
clf_report = classification_report(y_test, preds,
                                   target_names=le.classes_, output_dict=True)

# AUC-ROC (one-vs-rest)
auc_ovr = None
if proba is not None:
    try:
        auc_ovr = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    except Exception:
        pass

metrics = {
    "timestamp":    datetime.utcnow().isoformat(),
    "accuracy":     accuracy,
    "f1_macro":     f1_macro,
    "f1_weighted":  f1_weighted,
    "auc_roc_ovr":  auc_ovr,
    "n_test":       int(len(y_test)),
    "per_class_f1": {cls: clf_report[cls]["f1-score"]
                     for cls in le.classes_ if cls in clf_report},
}
with open(ART / "metrics" / "evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── Confusion matrix plot ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix  (Accuracy={accuracy:.3f})")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Per-class F1 bar chart ────────────────────────────────────────────────────
f1_per_class = [clf_report.get(c, {}).get("f1-score", 0) for c in le.classes_]
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(le.classes_, f1_per_class, color="steelblue", edgecolor="white")
ax.axhline(0.80, color="red", linestyle="--", label="threshold 0.80")
ax.set_ylim(0, 1.05)
ax.set_ylabel("F1-score")
ax.set_title(f"Per-class F1  (macro={f1_macro:.3f})")
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("reports/per_class_f1.png", dpi=150, bbox_inches="tight")
plt.close()

print(json.dumps(metrics, indent=2))

# Alert if below Task-1 threshold
if f1_macro < 0.80:
    print(f"::warning::F1-macro {f1_macro:.4f} is BELOW 0.80 threshold — retraining recommended")
