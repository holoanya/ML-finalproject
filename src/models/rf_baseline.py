"""Random Forest baseline on tabular + spatial features.

Pipeline: prepare.load_split (frozen, host-grouped) -> FeatureBuilder (fit on
train only) -> RandomForestRegressor on log1p(price) -> held-out metrics in EUR,
compared against two trivial baselines. Saves metrics + feature importances.

Run from the project root:  python src/models/rf_baseline.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
for sub in ("data", "features", "eval"):
    sys.path.insert(0, str(ROOT / "src" / sub))

from prepare import load_split          # noqa: E402
from features import build_features     # noqa: E402
from metrics import regression_report, print_report  # noqa: E402

RESULTS = ROOT / "results"


def trivial_baselines(train_df, test_df):
    """(a) global-mean predictor, (b) neighbourhood x room_type median."""
    y_tr, y_te = train_df["y_log"], test_df["y_log"]

    # (a) constant = train mean
    rep_mean = regression_report(y_te.to_numpy(), np.full(len(y_te), y_tr.mean()))

    # (b) group median by (neighbourhood, room_type), fallback to global median
    key = ["neighbourhood_cleansed", "room_type"]
    grp = train_df.groupby(key)["y_log"].median()
    glob = y_tr.median()
    pred_b = test_df.set_index(key).index.map(grp).to_numpy(dtype="float")
    pred_b = np.where(np.isnan(pred_b), glob, pred_b)
    rep_grp = regression_report(y_te.to_numpy(), pred_b)
    return rep_mean, rep_grp


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    train_df, test_df = load_split()
    y_train, y_test = train_df["y_log"], test_df["y_log"]

    X_train, X_test = build_features(train_df, test_df, y_train)
    print(f"[features] {X_train.shape[1]} features | train {X_train.shape[0]} | test {X_test.shape[0]}")

    print("\n=== held-out test metrics (EUR scale) ===")
    rep_mean, rep_grp = trivial_baselines(train_df, test_df)
    print_report("trivial: global mean", rep_mean)
    print_report("trivial: neigh x room", rep_grp)

    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rep_rf = regression_report(y_test.to_numpy(), pred)
    print_report("RandomForest", rep_rf)

    # feature importances
    imp = (
        pd.Series(rf.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    print("\n--- top 15 feature importances ---")
    print(imp.head(15).round(4).to_string())

    # save
    (RESULTS / "rf_baseline_metrics.json").write_text(json.dumps({
        "trivial_global_mean": rep_mean,
        "trivial_neigh_room": rep_grp,
        "random_forest": rep_rf,
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }, indent=2))
    imp.to_csv(RESULTS / "rf_feature_importances.csv", header=["importance"])
    print(f"\nsaved -> {RESULTS / 'rf_baseline_metrics.json'}")


if __name__ == "__main__":
    main()
