import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


def evaluate_regressor(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return round(mae, 4), round(mse, 4), round(rmse, 4), round(r2, 4)


def run_part_b():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # 1. Train / Validation / Test split (70 / 15 / 15 -> 840 / 180 / 180)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 2. Scaling without leakage check
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    test_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))

    b_checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(train_scaled_max_abs_mean, 8)),
        ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4)),
    ]
    pd.DataFrame(b_checks, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)
    print("outputs/part_b_checks.csv written.")

    # 3. Baseline model: predict mean of y_train
    baseline_val_pred = np.full_like(y_val, y_train.mean(), dtype=float)
    base_mae, base_mse, base_rmse, base_r2 = evaluate_regressor(y_val, baseline_val_pred)

    # 4. Tree-depth diagnostic
    tree_depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_val_rmse = float("inf")
    best_depth = None

    for d in tree_depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_pred = dt.predict(X_train)
        va_pred = dt.predict(X_val)

        tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
        va_rmse = np.sqrt(mean_squared_error(y_val, va_pred))

        depth_label = "unlimited" if d is None else d
        tree_records.append({
            "max_depth": depth_label,
            "train_rmse": round(float(tr_rmse), 4),
            "val_rmse": round(float(va_rmse), 4),
        })

        if va_rmse < best_val_rmse:
            best_val_rmse = va_rmse
            best_depth = d

    pd.DataFrame(tree_records).to_csv("outputs/part_b_tree_depth.csv", index=False)
    print(f"outputs/part_b_tree_depth.csv written. Best depth: {best_depth}")

    # 5. Four candidate models
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {"model": "mean_baseline", "split": "val", "MAE": base_mae, "MSE": base_mse, "RMSE": base_rmse, "R2": base_r2}
    ]

    fitted_models = {}
    val_rmses = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        mae, mse, rmse, r2 = evaluate_regressor(y_val, val_pred)
        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })
        fitted_models[name] = model
        val_rmses[name] = rmse

    # 6. Learning-rate experiment
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        va_pred = gb.predict(X_val)
        va_mse = mean_squared_error(y_val, va_pred)
        va_rmse = float(np.sqrt(va_mse))
        va_r2 = float(r2_score(y_val, va_pred))
        lr_records.append({
            "learning_rate": lr,
            "val_rmse": round(va_rmse, 4),
            "val_r2": round(va_r2, 4),
        })
    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    print("outputs/part_b_gb_learning_rate.csv written.")

    # 7. Cross-validated comparison and final model
    # Rank candidates by validation RMSE (lower is better)
    ranked_candidates = sorted(val_rmses.items(), key=lambda x: x[1])
    top_two = [item[0] for item in ranked_candidates[:2]]

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []
    for model_name in top_two:
        # Re-instantiate a clean instance with same hyperparameters
        if model_name == "linear_regression":
            m_cv = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        elif model_name == "decision_tree":
            m_cv = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
        elif model_name == "random_forest":
            m_cv = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "gradient_boosting":
            m_cv = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

        scores = cross_val_score(m_cv, X_train_val, y_train_val, cv=cv, scoring="r2")
        cv_records.append({
            "model": model_name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })
    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)
    print("outputs/part_b_cv.csv written.")

    # Final model is candidate with lowest validation RMSE
    final_model_name = ranked_candidates[0][0]
    final_model = fitted_models[final_model_name]
    test_pred = final_model.predict(X_test)
    test_mae, test_mse, test_rmse, test_r2 = evaluate_regressor(y_test, test_pred)

    metrics_rows.append({
        "model": final_model_name,
        "split": "test",
        "MAE": test_mae,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "R2": test_r2,
    })

    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
    print("outputs/part_b_metrics.csv written.")


if __name__ == "__main__":
    run_part_b()