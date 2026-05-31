"""Train obesity level classifier.

Primary model : Random Forest (GridSearchCV-tuned)
Baseline model: Logistic Regression
Both are evaluated and the better F1-macro is saved as the production model.
All runs are logged to artifacts/metrics/training_history.json.
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")

ART = Path("artifacts")
for sub in ("models", "metrics", "metadata"):
    (ART / sub).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

X_train = np.load(ART / "data" / "X_train.npy")
y_train = np.load(ART / "data" / "y_train.npy")
X_test  = np.load(ART / "data" / "X_test.npy")
y_test  = np.load(ART / "data" / "y_test.npy")
le      = joblib.load(ART / "preprocessing" / "label_encoder.pkl")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Baseline: Logistic Regression ────────────────────────────────────────────
print("Training baseline: Logistic Regression ...")
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_f1    = f1_score(y_test, lr_preds, average="macro")
lr_acc   = accuracy_score(y_test, lr_preds)
print(f"  Baseline  F1-macro={lr_f1:.4f}  acc={lr_acc:.4f}")

# ── Primary: Random Forest ────────────────────────────────────────────────────
print("Training primary: Random Forest (GridSearchCV) ...")
param_grid = {
    "n_estimators":     [100, 200],
    "max_depth":        [None, 20],
    "min_samples_leaf": [1, 2],
}
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
gs = GridSearchCV(rf_base, param_grid, cv=cv,
                  scoring="f1_macro", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
rf = gs.best_estimator_
rf_preds = rf.predict(X_test)
rf_f1    = f1_score(y_test, rf_preds, average="macro")
rf_acc   = accuracy_score(y_test, rf_preds)
print(f"  RF best   F1-macro={rf_f1:.4f}  acc={rf_acc:.4f}  params={gs.best_params_}")

# ── Pick winner ───────────────────────────────────────────────────────────────
if rf_f1 >= lr_f1:
    best_model, best_name, best_preds = rf,  "RandomForest",        rf_preds
    best_f1,    best_acc             = rf_f1, rf_acc
else:
    best_model, best_name, best_preds = lr,  "LogisticRegression",  lr_preds
    best_f1,    best_acc             = lr_f1, lr_acc

print(f"\nSelected model: {best_name}  F1-macro={best_f1:.4f}")

joblib.dump(best_model, ART / "models" / "model.pkl")
joblib.dump(lr,         ART / "models" / "baseline_lr.pkl")
joblib.dump(rf,         ART / "models" / "random_forest.pkl")

# ── Persist metrics ───────────────────────────────────────────────────────────
clf_report = classification_report(
    y_test, best_preds,
    target_names=le.classes_, output_dict=True
)

history = {
    "timestamp": datetime.utcnow().isoformat(),
    "selected_model": best_name,
    "best_params": gs.best_params_,
    "baseline_logistic_regression": {"f1_macro": lr_f1, "accuracy": lr_acc},
    "random_forest": {"f1_macro": rf_f1, "accuracy": rf_acc},
    "selected": {"f1_macro": best_f1, "accuracy": best_acc},
    "classification_report": clf_report,
}
with open(ART / "metrics" / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
(ART / "metadata" / "model_version.txt").write_text(f"version={version}\nmodel={best_name}\n")
(ART / "metadata" / "last_retrain.txt").write_text(datetime.utcnow().isoformat() + "\n")

with open("logs/training.log", "a") as f:
    f.write(f"[{datetime.utcnow().isoformat()}] {best_name} "
            f"f1_macro={best_f1:.4f} acc={best_acc:.4f} version={version}\n")

print(f"\nSaved: artifacts/models/model.pkl  (version={version})")
print(f"F1-macro={best_f1:.4f}  Accuracy={best_acc:.4f}")
