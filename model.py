import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt
from datetime import datetime

artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

print("=" * 50)
print("STARTING CNN REGRESSION MODEL")
print("=" * 50)

if not os.path.exists("train/train.csv"):
    print("ERROR: train/train.csv not found!")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    sys.exit(1)

print("\nLoading data...")
data = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")
print(f"Train data shape: {data.shape}")
print(f"Test data shape: {dtest.shape}")

# Drop constant columns
suspiciousData = [col for col in data.columns if len(data[col].unique()) == 1]
if suspiciousData:
    print(f"Dropping {len(suspiciousData)} constant columns")
    for dataset in [data, dtest]:
        dataset.drop(suspiciousData, axis=1, inplace=True)
else:
    print("No constant columns found")

# Encode categorical variables
cat_vars = [var for var in data.columns if data[var].dtypes == 'O' and var not in ['ID', 'y']]
print(f"Number of categorical variables: {len(cat_vars)}")
if cat_vars:
    for var in cat_vars:
        freq_encoding = data[var].value_counts().to_dict()
        data[f"{var}_freq"] = data[var].map(freq_encoding)
        dtest[f"{var}_freq"] = dtest[var].map(freq_encoding).fillna(0)
    data = data.drop(cat_vars, axis=1)
    dtest = dtest.drop(cat_vars, axis=1)

# Prepare features and target
if 'ID' in data.columns:
    data = data.drop("ID", axis=1)
if 'y' not in data.columns:
    print("ERROR: 'y' column not found!")
    sys.exit(1)

X = data.drop("y", axis=1).apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean()).fillna(0)
y = data["y"].values
X = X.values

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Build 1D CNN model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

batch_size = 16
epochs = 10

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1), padding='same'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='linear'),
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', r2_metric])
model.summary()

with open('model_summary.txt', 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    model.summary()
    sys.stdout = original_stdout
print("Saved: model_summary.txt")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

print("\nTraining model...")
history = model.fit(
    X_train_cnn, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test_cnn, y_test),
    callbacks=callbacks,
)
print("Training completed!")

score = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"Test Loss (MSE): {score[0]:.4f}, MAE: {score[1]:.4f}, R2: {score[2]:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True)
if 'r2_metric' in history.history:
    axes[1].plot(history.history['r2_metric'], label='Train R2')
    axes[1].plot(history.history['val_r2_metric'], label='Validation R2')
    axes[1].set_title('Model R2 Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R2 Score')
    axes[1].legend()
    axes[1].grid(True)
plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: model_results.png")

# Predictions
preds = model.predict(X_test_cnn).flatten()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True, alpha=0.3)
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: predictions_vs_actual.png")

residuals = y_test - preds
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Residuals Distribution')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0].grid(True, alpha=0.3)
axes[1].scatter(preds, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_title('Residuals vs Predictions')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: residuals_analysis.png")

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mse)

with open('metrics.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Mean Squared Error (MSE):        {mse:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE):  {rmse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE):        {mae:.4f}\n")
    f.write(f"R2 Score:                         {r2:.4f}\n")
    f.write("=" * 50 + "\n")
print("Saved: metrics.txt")

metrics_dict = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "1D CNN",
    "metrics": {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2_score": float(r2)},
    "training_history": {
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
    },
}
with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)
print("Saved: metrics.json")

data_info = {
    "features_count": X.shape[1],
    "target_mean": float(y.mean()),
    "target_std": float(y.std()),
    "target_min": float(y.min()),
    "target_max": float(y.max()),
}
with open('data_info.json', 'w') as f:
    json.dump(data_info, f, indent=4)
print("Saved: data_info.json")

model.save('cnn_regression_model.h5')
print("Saved: cnn_regression_model.h5")

# Submission
test_data = dtest.drop("ID", axis=1).copy().apply(pd.to_numeric, errors='coerce')
test_data = test_data.fillna(test_data.mean()).fillna(0).values
X_test_final = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
predictions = model.predict(X_test_final).flatten()
submission = pd.DataFrame({"ID": dtest["ID"], "y": predictions})
submission.to_csv('submission_5.csv', index=False)
print("Saved: submission_5.csv")

summary_report = f"""
================================================================================
CNN REGRESSION MODEL - TRAINING SUMMARY
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL PERFORMANCE:
- MSE:      {mse:.4f}
- RMSE:     {rmse:.4f}
- MAE:      {mae:.4f}
- R2 Score: {r2:.4f}

FILES GENERATED:
[+] model_results.png
[+] predictions_vs_actual.png
[+] residuals_analysis.png
[+] metrics.txt
[+] metrics.json
[+] model_summary.txt
[+] submission_5.csv
[+] cnn_regression_model.h5
[+] data_info.json
[+] summary_report.txt
================================================================================
"""

with open('summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)
print("Saved: summary_report.txt")

print("\n" + summary_report)
print("=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)
