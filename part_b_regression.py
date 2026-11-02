import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES

def run_part_b():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # 70% / 15% / 15% Split using fixed seeds
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale exclusively via training rows to safely isolate against pipeline leakages
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Integrity check logs
    checks = {
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_scaled_max_abs_mean": float(np.max(np.abs(np.mean(X_train_scaled, axis=0)))),
        "test_scaled_max_abs_mean": float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))
    }
    pd.DataFrame(list(checks.items()), columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)

    # Task 3: Evaluate Base Baseline
    base_pred = np.full(shape=y_val.shape, fill_value=y_train.mean())
    metrics_list = [[
        "mean_baseline", "val",
        round(mean_absolute_error(y_val, base_pred), 4),
        round(mean_squared_error(y_val, base_pred), 4),
        round(np.sqrt(mean_squared_error(y_val, base_pred)), 4),
        round(r2_score(y_val, base_pred), 4)
    ]]

    # Task 4: Structural Tree depth hyperparameter scans
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_tree_depth = None
    best_tree_rmse = float('inf')

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        t_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
        v_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
        d_label = "unlimited" if d is None else d
        tree_records.append([d_label, round(t_rmse, 4), round(v_rmse, 4)])

        if v_rmse < best_tree_rmse:
            best_tree_rmse = v_rmse
            best_tree_depth = d

    pd.DataFrame(tree_records, columns=["max_depth", "train_rmse", "val_rmse"]).to_csv("outputs/part_b_tree_depth.csv", index=False)

    # Candidate architectures
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_tree_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    best_candidate_name = None
    best_candidate_rmse = float('inf')

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        metrics_list.append([name, "val", round(mae,4), round(mse,4), round(rmse,4), round(r2,4)])

        if rmse < best_candidate_rmse:
            best_candidate_rmse = rmse
            best_candidate_name = name

    # Task 6: Learning rates check
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gbr.fit(X_train, y_train)
        preds = gbr.predict(X_val)
        lr_records.append([lr, round(np.sqrt(mean_squared_error(y_val, preds)), 4), round(r2_score(y_val, preds), 4)])
    pd.DataFrame(lr_records, columns=["learning_rate", "val_rmse", "val_r2"]).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # Task 7: Cross validation extraction
    # Sort top two validation models
    val_ranks = sorted([(m, metrics_list[i][4]) for i, m in enumerate(["linear_regression", "decision_tree", "random_forest", "gradient_boosting"], start=1)], key=lambda x: x[1])
    top_two = [val_ranks[0][0], val_ranks[1][0]]

    X_cv = pd.concat([X_train, X_val])
    y_cv = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_records = []
    for name in top_two:
        scores = cross_val_score(candidates[name], X_cv, y_cv, cv=kf)
        cv_records.append([name, round(scores.mean(), 4), round(scores.std(), 4)])
    pd.DataFrame(cv_records, columns=["model", "mean_r2", "std_r2"]).to_csv("outputs/part_b_cv.csv", index=False)

    # Final evaluation mapping against selected test rows
    final_model = candidates[best_candidate_name]
    test_preds = final_model.predict(X_test)
    metrics_list.append([
        best_candidate_name, "test",
        round(mean_absolute_error(y_test, test_preds), 4),
        round(mean_squared_error(y_test, test_preds), 4),
        round(np.sqrt(mean_squared_error(y_test, test_preds)), 4),
        round(r2_score(y_test, test_preds), 4)
    ])

    pd.DataFrame(metrics_list, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]).to_csv("outputs/part_b_metrics.csv", index=False)

if __name__ == "__main__":
    run_part_b()
