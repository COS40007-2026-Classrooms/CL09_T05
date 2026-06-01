# %%
import os
import sys
import json
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Create artifacts directory
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

print("="*50)
print("STARTING CNN REGRESSION MODEL")
print("="*50)

# %%
# Check if data files exist
if not os.path.exists("train/train.csv"):
    print("ERROR: train/train.csv not found!")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    exit(1)

# %%
print("\nLoading data...")
data = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")
print(f"Train data shape: {data.shape}")
print(f"Test data shape: {dtest.shape}")

# %% [markdown]
# ### Missing Values

# %%
vars_with_na = data.columns[data.isnull().any()].tolist()
vars_with_na_test = dtest.columns[dtest.isnull().any()].tolist()
print(f"Missing values in train: {len(vars_with_na)}")
print(f"Missing values in test: {len(vars_with_na_test)}")

# %% [markdown]
# ### Group the Data [train and test]

# %%
train_test_data = [data, dtest]

# %%
# list of numerical variables
for dataset in train_test_data:
    num_vars = [var for var in dataset.columns if dataset[var].dtypes != 'O']
    print('Number of numerical variables: ', len(num_vars))

# %% [markdown]
# ### Suspicious data (constant columns)

# %%
suspiciousData = []
for col in data:
    if len(data[col].unique()) == 1:
        suspiciousData.append(col)
        
if suspiciousData:
    print(f"Dropping {len(suspiciousData)} constant columns: {suspiciousData[:5]}...")
else:
    print("No constant columns found")

# %% [markdown]
# ### Drop suspicious features

# %%
for dataset in train_test_data:
    if suspiciousData:
        dataset.drop(suspiciousData, axis=1, inplace=True)

# %% [markdown]
# ### Identify categorical variables

# %%
# list of categorical variables
cat_vars = [var for var in data.columns if data[var].dtypes == 'O' and var not in ['ID', 'y']]
print(f'Number of categorical variables: {len(cat_vars)}')

# %% [markdown]
# ## Categorical data processing

# %%
if len(cat_vars) > 0:
    print(f"Processing {len(cat_vars)} categorical variables")
    
    # Use frequency encoding
    for var in cat_vars:
        # Create frequency encoding
        freq_encoding = data[var].value_counts().to_dict()
        data[f"{var}_freq"] = data[var].map(freq_encoding)
        dtest[f"{var}_freq"] = dtest[var].map(freq_encoding).fillna(0)
    
    # Drop original categorical columns
    data = data.drop(cat_vars, axis=1)
    dtest = dtest.drop(cat_vars, axis=1)
    
    print("Categorical variables encoded successfully")
else:
    print("No categorical variables to process")

# %% [markdown]
# ## Prepare data for modeling

# %%
# Drop ID column
if 'ID' in data.columns:
    data = data.drop("ID", axis=1)

# Define X and y
if 'y' not in data.columns:
    print("ERROR: 'y' column not found in data!")
    print("Columns:", data.columns.tolist())
    exit(1)

X = data.drop("y", axis=1)
y = data["y"]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# %%
# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')
# Fill NaN values with column means
X = X.fillna(X.mean())
# Fill any remaining NaN with 0
X = X.fillna(0)

# %%
# Convert to numpy arrays
X = X.values
y = y.values

print(f"X array shape: {X.shape}")
print(f"y array shape: {y.shape}")

# %% [markdown]
# ## Train test split

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# %% [markdown]
# # Build and train the CNN model

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# %%
# For 1D CNN (better for tabular data)
batch_size = 16
epochs = 10

# Reshape for 1D CNN: (samples, features, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f'X_train shape for CNN: {X_train_cnn.shape}')
print(f'X_test shape for CNN: {X_test_cnn.shape}')

# %%
# Define R2 metric using TensorFlow
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# %%
# Build 1D CNN model
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
    
    Dense(1, activation='linear')
])

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae', r2_metric]
)

model.summary()

# Save model summary
with open('model_summary.txt', 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    model.summary()
    sys.stdout = original_stdout
print("Model summary saved to model_summary.txt")

# %%
# Callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# %%
# Train the model
print("\nTraining model...")
history = model.fit(
    X_train_cnn, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test_cnn, y_test),
    callbacks=callbacks
)

print("Training completed successfully!")

# %%
# Evaluate the model
score = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f'Test Loss (MSE): {score[0]:.4f}')
print(f'Test MAE: {score[1]:.4f}')
print(f'Test R2: {score[2]:.4f}')

# %% [markdown]
# # Visualize training history

# %%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True)

# Plot R2
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
plt.savefig(f'{artifacts_dir}/model_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: model_results.png")

# %% [markdown]
# # Make predictions on test set

# %%
# Make predictions
preds = model.predict(X_test_cnn)
preds = preds.flatten()

# %%
# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True, alpha=0.3)
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: predictions_vs_actual.png")

# %%
# Plot residuals
residuals = y_test - preds

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of residuals
axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Residuals Distribution')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0].grid(True, alpha=0.3)

# Residuals vs predictions
axes[1].scatter(preds, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predictions')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Predictions')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: residuals_analysis.png")

# %% [markdown]
# # Calculate and save metrics

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mse)

# Save metrics as text file
with open('metrics.txt', 'w') as f:
    f.write("="*50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("="*50 + "\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write("="*50 + "\n\n")
    f.write("TRAINING HISTORY\n")
    f.write("="*50 + "\n")
    f.write(f"Final Train Loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
    if 'r2_metric' in history.history:
        f.write(f"Final Train R2: {history.history['r2_metric'][-1]:.4f}\n")
        f.write(f"Final Validation R2: {history.history['val_r2_metric'][-1]:.4f}\n")
    f.write("="*50 + "\n")

print("Saved: metrics.txt")

# %%
# Save metrics as JSON
metrics_dict = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "1D CNN",
    "test_size": 0.2,
    "batch_size": batch_size,
    "epochs": epochs,
    "final_epoch": len(history.history['loss']),
    "metrics": {
        "mean_squared_error": float(mse),
        "root_mean_squared_error": float(rmse),
        "mean_absolute_error": float(mae),
        "r2_score": float(r2)
    },
    "training_history": {
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1])
    }
}

if 'r2_metric' in history.history:
    metrics_dict["training_history"]["final_train_r2"] = float(history.history['r2_metric'][-1])
    metrics_dict["training_history"]["final_val_r2"] = float(history.history['val_r2_metric'][-1])

with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("Saved: metrics.json")

# %% [markdown]
# # Save data information

# %%
data_info = {
    "train_samples": len(data),
    "test_samples": len(dtest),
    "features_count": X.shape[1],
    "categorical_variables_original": len(cat_vars) if 'cat_vars' in locals() else 0,
    "suspicious_features_dropped": len(suspiciousData) if 'suspiciousData' in locals() else 0,
    "target_mean": float(y.mean()),
    "target_std": float(y.std()),
    "target_min": float(y.min()),
    "target_max": float(y.max())
}

with open('data_info.json', 'w') as f:
    json.dump(data_info, f, indent=4)

print("Saved: data_info.json")

# %% [markdown]
# # Save model

# %%
# Save the model
model.save('cnn_regression_model.h5')
model.save(f'{artifacts_dir}/cnn_regression_model.h5')
print("Saved: cnn_regression_model.h5")

# %% [markdown]
# # Make predictions on test set and create submission

# %%
# Prepare test data
test_data = dtest.drop("ID", axis=1).copy()

# Convert to numeric
test_data = test_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.fillna(test_data.mean())
test_data = test_data.fillna(0)

# Reshape for CNN
X_test_final = test_data.values
X_test_final = X_test_final.reshape(X_test_final.shape[0], X_test_final.shape[1], 1)

print(f"Test data shape: {X_test_final.shape}")

# Make predictions
predictions = model.predict(X_test_final)
predictions = predictions.flatten()

# Create submission file
submission = pd.DataFrame({
    "ID": dtest["ID"],
    "y": predictions
})

submission.to_csv('submission_5.csv', index=False)
submission.to_csv(f'{artifacts_dir}/submission_5.csv', index=False)
print("Saved: submission_5.csv")

# ============================================================================
# Create summary report (ASCII only - no Unicode)
# ============================================================================

summary_report = f"""
================================================================================
CNN REGRESSION MODEL - TRAINING SUMMARY
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA INFORMATION:
- Training samples: {data_info['train_samples']}
- Test samples: {data_info['test_samples']}
- Features used: {data_info['features_count']}
- Target range: [{data_info['target_min']:.4f}, {data_info['target_max']:.4f}]

MODEL PERFORMANCE:
- MSE: {mse:.4f}
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
- R2 Score: {r2:.4f}

MODEL ARCHITECTURE:
- Type: 1D Convolutional Neural Network
- Conv1D layers: 3
- Dense layers: 3
- Total parameters: {model.count_params():,}

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

# Save summary report
try:
    with open('summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print("Saved: summary_report.txt")
except UnicodeEncodeError:
    # Fallback to ASCII-only
    with open('summary_report.txt', 'w') as f:
        f.write(summary_report)
    print("Saved: summary_report.txt (ASCII)")

print("\n" + summary_report)

# ============================================================================
# List all generated files
# ============================================================================

print("\n" + "="*50)
print("ALL GENERATED FILES:")
print("="*50)
files = ['model_results.png', 'predictions_vs_actual.png', 'residuals_analysis.png', 
         'metrics.txt', 'metrics.json', 'model_summary.txt', 'submission_5.csv', 
         'cnn_regression_model.h5', 'data_info.json', 'summary_report.txt']

for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024
        print(f"[+] {file} ({size:.2f} KB)")
    else:
        print(f"[-] {file} (NOT FOUND)")

print("="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
