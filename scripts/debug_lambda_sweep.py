"""Quick sweep: test different lambda values for softmax_rank weighting."""
import sys, copy
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.config import load_config
from src.data.loader import load_panel, get_feature_names, normalize_features
from src.models import create_model
from src.weighting import compute_weights
from src.portfolio.construction import build_portfolio_timeseries

config = load_config()
panel = load_panel()
features = get_feature_names(panel)
target_col = config["data"]["target_col"]

dates = sorted(panel["yyyymm"].unique())
train_window = config["training"]["train_window"]
val_window = config["training"]["validation_window"]

# Use a small OOS window: last 24 months
oos_months = dates[-24:]
print(f"OOS months: {oos_months[0]} - {oos_months[-1]} ({len(oos_months)} months)")

# Train one standard and one weighted model on a fixed window
train_end_idx = len(dates) - 24 - val_window
train_dates = dates[train_end_idx - train_window:train_end_idx]
val_dates = dates[train_end_idx:train_end_idx + val_window]

print(f"Train: {train_dates[0]} - {train_dates[-1]}")
print(f"Val:   {val_dates[0]} - {val_dates[-1]}")

train_df = panel[panel["yyyymm"].isin(train_dates)]
val_df = panel[panel["yyyymm"].isin(val_dates)]
oos_df = panel[panel["yyyymm"].isin(oos_months)]

# Normalize
train_val = pd.concat([train_df, val_df])
train_val_norm = normalize_features(train_val, features)
oos_norm = normalize_features(oos_df, features)

train_norm = train_val_norm.loc[train_df.index].dropna(subset=features + [target_col])
val_norm = train_val_norm.loc[val_df.index].dropna(subset=features + [target_col])
oos_norm = oos_norm.dropna(subset=features + [target_col])

X_train = train_norm[features].values
y_train = train_norm[target_col].values
X_val = val_norm[features].values
y_val = val_norm[target_col].values
X_oos = oos_norm[features].values

print(f"\nTrain: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, OOS: {X_oos.shape[0]:,}")

# Train standard model
print("\nTraining standard model...")
model_std = create_model("xgboost", config=config)
model_std.fit(X_train, y_train, X_val, y_val)
pred_std = model_std.predict(X_oos)

# Test different lambdas for weighted training
lambdas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

results = []
for lam in lambdas:
    print(f"\nLambda={lam}: ", end="")

    # Compute weights with this lambda
    if lam == 0.0:
        w_train = np.ones(len(train_norm))
        w_val = np.ones(len(val_norm))
    else:
        w_train = compute_weights(train_norm, scheme="softmax_rank", lam=lam).values
        w_val = compute_weights(val_norm, scheme="softmax_rank", lam=lam).values

    # Train weighted model
    model_wt = create_model("xgboost", config=config)
    model_wt.fit(X_train, y_train, X_val, y_val,
                 sample_weight=w_train, sample_weight_val=w_val)
    pred_wt = model_wt.predict(X_oos)

    # Check prediction spread
    pred_std_val = np.std(pred_wt)
    pred_spread = np.percentile(pred_wt, 90) - np.percentile(pred_wt, 10)

    # Rank correlation with actual returns
    oos_with_preds = oos_norm.copy()
    oos_with_preds["_pred"] = pred_wt
    corrs = oos_with_preds.groupby("yyyymm").apply(
        lambda g: g["_pred"].corr(g[target_col], method="spearman"),
        include_groups=False
    )
    mean_corr = corrs.mean()
    pct_pos = (corrs > 0).mean() * 100

    # Build portfolio
    pred_series = pd.Series(pred_wt, index=oos_norm.index)
    ts_df, _ = build_portfolio_timeseries(oos_norm, pred_series, weighted=False, config=config)
    sr = ts_df["ret_long_short"].mean() / ts_df["ret_long_short"].std() * np.sqrt(12)

    print(f"spread={pred_spread:.6f}, rank_corr={mean_corr:.4f} ({pct_pos:.0f}% pos), SR={sr:.3f}")
    results.append({"lambda": lam, "spread": pred_spread, "rank_corr": mean_corr,
                     "pct_positive": pct_pos, "sharpe": sr})

# Also show standard model performance
oos_with_std = oos_norm.copy()
oos_with_std["_pred"] = pred_std
corrs_std = oos_with_std.groupby("yyyymm").apply(
    lambda g: g["_pred"].corr(g[target_col], method="spearman"),
    include_groups=False
)
pred_series_std = pd.Series(pred_std, index=oos_norm.index)
ts_std, _ = build_portfolio_timeseries(oos_norm, pred_series_std, weighted=False, config=config)
sr_std = ts_std["ret_long_short"].mean() / ts_std["ret_long_short"].std() * np.sqrt(12)

print(f"\n--- Standard model (no weighting): rank_corr={corrs_std.mean():.4f}, SR={sr_std:.3f} ---")
print("\n--- Summary ---")
df = pd.DataFrame(results)
print(df.to_string(index=False))
