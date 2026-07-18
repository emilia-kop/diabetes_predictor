"""
find_probability_fast.py
Same idea as find_probability.py, but builds the whole grid as one big
array and calls predict_proba ONCE on all rows together, instead of looping
row by row. Finishes in a few seconds instead of minutes.

Run this in the same folder as diabetes_model.json and scaler.pkl:
    python find_probability_fast.py
"""

import pickle
import numpy as np
from xgboost import XGBClassifier

TARGET_PCT = 34.5

model = XGBClassifier()
model.load_model('diabetes_model.json')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

gender = 0
hypertension = 0
heart_disease = 0
smoking_never = 1
smoking_current = smoking_ever = smoking_former = smoking_notcurrent = 0

ages = np.arange(20, 75, 5)
bmis = np.arange(20, 40, 2.0)
hba1cs = np.arange(4.5, 9.0, 0.1)
glucoses = np.arange(80, 260, 10)

# Build every combination as one big grid
grid = np.array(np.meshgrid(ages, bmis, hba1cs, glucoses)).T.reshape(-1, 4)
print(f"Testing {len(grid)} combinations in one batch...")

# Scale all the numeric columns at once
scaled_numeric = scaler.transform(grid)  # columns: age, bmi, hba1c, glucose

n = len(grid)
features = np.column_stack([
    np.full(n, gender),
    scaled_numeric[:, 0],           # age
    np.full(n, hypertension),
    np.full(n, heart_disease),
    scaled_numeric[:, 1],           # bmi
    scaled_numeric[:, 2],           # hba1c
    scaled_numeric[:, 3],           # glucose
    np.full(n, smoking_current),
    np.full(n, smoking_ever),
    np.full(n, smoking_former),
    np.full(n, smoking_never),
    np.full(n, smoking_notcurrent),
])

probs = model.predict_proba(features)[:, 1] * 100

diffs = np.abs(probs - TARGET_PCT)
top10 = np.argsort(diffs)[:10]

print(f"\nClosest matches to {TARGET_PCT}% probability:\n")
for i in top10:
    age, bmi, hba1c, glucose = grid[i]
    print(f"Age={age:.0f}, BMI={bmi:.1f}, HbA1c={hba1c:.1f}, Glucose={glucose:.0f} -> {probs[i]:.1f}%")