"""Generate a realistic synthetic obesity dataset matching the UCI schema.

Run once locally:  python generate_data.py
Produces:
  train/train.csv   (~1,690 rows, 80 %)
  test/test.csv     (~421 rows,  20 %)
  data/new_data.csv (~50 rows — used to demo the retraining trigger)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
N = 2111

CLASSES = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]
# rough class distribution matching the UCI dataset
weights = [0.08, 0.10, 0.13, 0.13, 0.19, 0.19, 0.18]
target = rng.choice(CLASSES, size=N, p=weights)

bmi_map = {
    "Insufficient_Weight": (14, 18.5),
    "Normal_Weight":        (18.5, 25),
    "Overweight_Level_I":   (25, 27.5),
    "Overweight_Level_II":  (27.5, 30),
    "Obesity_Type_I":       (30, 35),
    "Obesity_Type_II":      (35, 40),
    "Obesity_Type_III":     (40, 55),
}

height = rng.uniform(1.50, 1.95, N)
weight = np.array([
    rng.uniform(bmi_map[c][0], bmi_map[c][1]) * h ** 2
    for c, h in zip(target, height)
])

df = pd.DataFrame({
    "Gender":  rng.choice(["Male", "Female"], N),
    "Age":     rng.uniform(14, 61, N).round(1),
    "Height":  height.round(4),
    "Weight":  weight.round(2),
    "family_history_with_overweight": rng.choice(["yes", "no"], N, p=[0.81, 0.19]),
    "FAVC":    rng.choice(["yes", "no"], N, p=[0.88, 0.12]),
    "FCVC":    rng.uniform(1, 3, N).round(4),
    "NCP":     rng.uniform(1, 4, N).round(4),
    "CAEC":    rng.choice(["no","Sometimes","Frequently","Always"], N,
                          p=[0.01, 0.74, 0.18, 0.07]),
    "SMOKE":   rng.choice(["yes", "no"], N, p=[0.03, 0.97]),
    "CH2O":    rng.uniform(1, 3, N).round(4),
    "SCC":     rng.choice(["yes", "no"], N, p=[0.04, 0.96]),
    "FAF":     rng.uniform(0, 3, N).round(4),
    "TUE":     rng.uniform(0, 2, N).round(4),
    "CALC":    rng.choice(["no","Sometimes","Frequently","Always"], N,
                          p=[0.02, 0.67, 0.19, 0.12]),
    "MTRANS":  rng.choice(
        ["Automobile","Motorbike","Bike","Public_Transportation","Walking"], N,
        p=[0.19, 0.02, 0.03, 0.63, 0.13]
    ),
    "NObeyesdad": target,
})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df["NObeyesdad"])
new_df = test_df.sample(50, random_state=7).reset_index(drop=True)

train_df.to_csv("train/train.csv", index=False)
test_df.to_csv("test/test.csv", index=False)
new_df.to_csv("data/new_data.csv", index=False)

print(f"train: {len(train_df)}  test: {len(test_df)}  new: {len(new_df)}")
print(train_df["NObeyesdad"].value_counts())
