"""Preprocess the obesity dataset (+ optional new_data.csv) into numpy artefacts.

Reads train/train.csv and test/test.csv.
If data/new_data.csv exists and contains the target column it is appended to
the training set, so a `git push data/new_data.csv` triggers retraining on the
combined corpus.

Outputs into artifacts/:
  data/X_train.npy, y_train.npy, X_test.npy, y_test.npy, X_new.npy
  preprocessing/feature_columns.json, scaler.pkl, label_encoder.pkl
  metadata/data_version.txt
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ART = Path("artifacts")
for sub in ("data", "preprocessing", "metadata"):
    (ART / sub).mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("train/train.csv")
TEST_CSV  = Path("test/test.csv")
NEW_CSV   = Path("data/new_data.csv")
TARGET    = "NObeyesdad"

CATEGORICAL = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]
NUMERIC = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

if not TRAIN_CSV.exists():
    sys.exit(f"ERROR: {TRAIN_CSV} not found")

print(f"Loading {TRAIN_CSV} ...")
train_df = pd.read_csv(TRAIN_CSV)

# Merge new data if available and labelled
if NEW_CSV.exists() and NEW_CSV.stat().st_size > 0:
    try:
        new_df = pd.read_csv(NEW_CSV)
        if TARGET in new_df.columns:
            print(f"  Merging {len(new_df)} labelled rows from {NEW_CSV}")
            train_df = pd.concat([train_df, new_df], ignore_index=True, sort=False)
        else:
            print(f"  WARN: {NEW_CSV} has no '{TARGET}' column — skipping merge")
    except Exception as e:
        print(f"  WARN: could not read {NEW_CSV}: {e}")

print(f"  Training rows after merge: {len(train_df)}")

if TARGET not in train_df.columns:
    sys.exit(f"ERROR: '{TARGET}' column missing from training data")

# ── Label-encode the target ──────────────────────────────────────────────────
le = LabelEncoder()
y_train_raw = le.fit_transform(train_df[TARGET])
joblib.dump(le, ART / "preprocessing" / "label_encoder.pkl")
print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")

# ── Feature engineering ──────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, cat_encoders: dict | None = None) -> tuple:
    """Return (X_array, feature_cols, updated_cat_encoders)."""
    df = df.copy()
    encoders = cat_encoders or {}

    # derive BMI
    if "Height" in df.columns and "Weight" in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # ordinal / frequency encode categoricals
    for col in CATEGORICAL:
        if col not in df.columns:
            continue
        if cat_encoders is None:
            enc = LabelEncoder().fit(df[col].astype(str))
            encoders[col] = enc
        else:
            enc = encoders.get(col)
            if enc is None:
                enc = LabelEncoder().fit(df[col].astype(str))
                encoders[col] = enc
        df[col] = df[col].astype(str).map(
            lambda v, e=enc: e.transform([v])[0] if v in e.classes_ else -1
        )

    feature_cols = NUMERIC + ["BMI"] + CATEGORICAL
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean()).fillna(0)
    return X.values.astype("float32"), feature_cols, encoders

X_train_raw, feature_cols, cat_encoders = build_features(train_df)

# ── Scale ────────────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw).astype("float32")
joblib.dump(scaler, ART / "preprocessing" / "scaler.pkl")
joblib.dump(cat_encoders, ART / "preprocessing" / "cat_encoders.pkl")

with open(ART / "preprocessing" / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

# ── Test set ─────────────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_CSV) if TEST_CSV.exists() else pd.DataFrame()
if not test_df.empty:
    y_test_raw = le.transform(test_df[TARGET]) if TARGET in test_df.columns else None
    X_test_raw, _, _ = build_features(test_df, cat_encoders)
    X_test = scaler.transform(X_test_raw).astype("float32")
else:
    X_test = np.zeros((0, X_train.shape[1]), dtype="float32")
    y_test_raw = np.zeros(0, dtype="int64")

# ── Save artefacts ───────────────────────────────────────────────────────────
np.save(ART / "data" / "X_train.npy", X_train)
np.save(ART / "data" / "y_train.npy", y_train_raw.astype("int64"))
np.save(ART / "data" / "X_test.npy",  X_test)
np.save(ART / "data" / "y_test.npy",  np.array(y_test_raw, dtype="int64"))

# save raw (unscaled) training features for drift reference
np.save(ART / "data" / "X_train_raw.npy", X_train_raw)

# X_new: unlabelled new data for inference
if NEW_CSV.exists() and NEW_CSV.stat().st_size > 0:
    new_df2 = pd.read_csv(NEW_CSV)
    X_new_raw, _, _ = build_features(new_df2, cat_encoders)
    X_new = scaler.transform(X_new_raw).astype("float32")
    np.save(ART / "data" / "X_new.npy", X_new)
    np.save(ART / "data" / "X_new_raw.npy", X_new_raw)
    print(f"  Saved X_new: {X_new.shape}")

version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
(ART / "metadata" / "data_version.txt").write_text(
    f"version={version}\n"
    f"train_rows={len(train_df)}\n"
    f"test_rows={len(test_df)}\n"
    f"features={len(feature_cols)}\n"
    f"classes={len(le.classes_)}\n"
    f"new_data_merged={NEW_CSV.exists()}\n"
)

print(f"Done. X_train={X_train.shape}  X_test={X_test.shape}  version={version}")
