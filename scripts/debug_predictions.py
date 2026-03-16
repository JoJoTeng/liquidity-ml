"""Quick diagnostic: train XGBoost on one window (matching real pipeline)."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.config import load_config
from src.data.loader import load_panel, get_feature_names, normalize_features
from src.models import create_model

config = load_config()
panel = load_panel()
features = get_feature_names(panel)
target_col = config["data"]["target_col"]

print(f"Panel shape: {panel.shape}")
print(f"Features: {len(features)}")
print(f"Target: {target_col}")
print(f"Date range: {panel['yyyymm'].min()} - {panel['yyyymm'].max()}")

train_months = config["training"]["train_window"]
val_months = config["training"]["validation_window"]

dates = sorted(panel["yyyymm"].unique())
print(f"Total unique months: {len(dates)}")

t = train_months + val_months
train_end_idx = train_months - 1
val_end_idx = train_months + val_months - 1

train_dates = dates[:train_months]
val_dates = dates[train_months:train_months + val_months]
test_date = dates[t]

print(f"\nTrain: {train_dates[0]} - {train_dates[-1]} ({len(train_dates)} months)")
print(f"Val:   {val_dates[0]} - {val_dates[-1]} ({len(val_dates)} months)")
print(f"Test:  {test_date}")

train_mask = panel["yyyymm"].isin(train_dates)
val_mask = panel["yyyymm"].isin(val_dates)
test_mask = panel["yyyymm"] == test_date

train_df = panel[train_mask].copy()
val_df = panel[val_mask].copy()
test_df = panel[test_mask].copy()

# Normalize (matching real pipeline: train+val together, test independently)
train_val = pd.concat([train_df, val_df])
train_val_norm = normalize_features(train_val, features)
test_norm = normalize_features(test_df, features)

train_norm = train_val_norm.loc[train_df.index]
val_norm = train_val_norm.loc[val_df.index]

# Fill NaN features with 0.0
for df in [train_norm, val_norm, test_norm]:
    df[features] = df[features].fillna(0.0)

# Drop NaN targets
train_norm = train_norm.dropna(subset=[target_col])
val_norm = val_norm.dropna(subset=[target_col])
test_norm = test_norm.dropna(subset=[target_col])

X_train = train_norm[features].values
y_train = train_norm[target_col].values
X_val = val_norm[features].values
y_val = val_norm[target_col].values
X_test = test_norm[features].values

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"X_test shape:  {X_test.shape}")

print(f"\ny_train: mean={np.mean(y_train):.6f}, std={np.std(y_train):.6f}, range=[{np.min(y_train):.4f}, {np.max(y_train):.4f}]")
print(f"X_train NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")

# Train model (no tuning, use defaults)
model = create_model("xgboost", config=config)
model.fit(X_train, y_train, X_val, y_val)

# Predict
preds = model.predict(X_test)
print(f"\nPredictions: mean={np.mean(preds):.8f}, std={np.std(preds):.8f}")
print(f"Range: [{np.min(preds):.8f}, {np.max(preds):.8f}]")
print(f"Unique values: {len(np.unique(preds))}")
print(f"Sample (first 20): {preds[:20]}")

# Test qcut behavior
pred_series = pd.Series(preds)
print(f"\n--- qcut diagnosis ---")
for n_q in [10, 5, 3]:
    try:
        q = pd.qcut(pred_series, q=n_q, labels=False, duplicates="drop")
        print(f"qcut(n={n_q}): {q.nunique()} bins, dist: {q.value_counts().sort_index().to_dict()}")
    except Exception as e:
        print(f"qcut(n={n_q}): FAILED - {e}")

# Rank-based approach
rank_q = pd.qcut(pred_series.rank(method="first"), q=10, labels=False) + 1
print(f"\nRank-based qcut(10): {rank_q.nunique()} bins")
print(f"Distribution: {rank_q.value_counts().sort_index().to_dict()}")
