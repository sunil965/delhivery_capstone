"""Part C — Delay Classifier and Alert Threshold.

Builds classification models to predict shipment delays, evaluates feature importance,
selects top k features, performs hyperparameter tuning, picks an operational
probability threshold to meet the 80% recall target, and evaluates on the test set.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.shipment_prep import FEATURES, NOISE_COLUMNS


def get_clf_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    return tn, fp, fn, tp, acc, prec, rec, f1


def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # 1. Stratified split: 70 / 15 / 15 -> 840 / 180 / 180
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Split sizes: train={len(X_train)} (delayed={y_train.sum()}), "
          f"val={len(X_val)} (delayed={y_val.sum()}), test={len(X_test)} (delayed={y_test.sum()})")

    # 2. Blind baseline and three classifiers
    # (a) blind_all_on_time
    blind_pred = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, blind_pred)
    models_rows = [{
        "model": "blind_all_on_time",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": 0.0, "recall": rec, "f1": f1
    }]

    # (b) logistic_regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, lr_pred)
    models_rows.append({
        "model": "logistic_regression",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    # (c) decision_tree_depth_<d> with d in [2, 3, 4, 5, 6, 8, 10]
    dt_candidates = [2, 3, 4, 5, 6, 8, 10]
    best_dt_f1 = -1.0
    best_dt_d = None
    best_dt_pred = None

    for d in dt_candidates:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        pred = dt.predict(X_val)
        val_f1 = f1_score(y_val, pred, zero_division=0)
        if val_f1 > best_dt_f1:
            best_dt_f1 = val_f1
            best_dt_d = d
            best_dt_pred = pred

    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, best_dt_pred)
    models_rows.append({
        "model": f"decision_tree_depth_{best_dt_d}",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    # (d) random_forest_default
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    rf_pred = rf_def.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, rf_pred)
    models_rows.append({
        "model": "random_forest_default",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    pd.DataFrame(models_rows).to_csv("outputs/part_c_models.csv", index=False)
    print("outputs/part_c_models.csv written.")

    # 3. Feature signals and SelectKBest
    rf_feat = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_feat.fit(X_train, y_train)
    rf_importances = rf_feat.feature_importances_

    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    feat_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "rf_importance": rf_importances,
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    feat_df["f_rank"] = feat_df["f_score"].rank(ascending=False, method="min").astype(int)
    # Arrange columns: feature, f_score, f_rank, rf_importance, is_noise
    feat_df = feat_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feat_df["f_score"] = feat_df["f_score"].round(4)
    feat_df["rf_importance"] = feat_df["rf_importance"].round(4)
    feat_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    print("outputs/part_c_feature_scores.csv written.")

    # k experiment for k in [5, 7, 10, 12, 17]
    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        val_f1 = round(float(f1_score(y_val, pred, zero_division=0)), 4)
        selectk_rows.append({"k": k, "val_f1": val_f1})
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)
    print("outputs/part_c_selectk.csv written.")

    # 4. Tuning with k=7
    selector_k7 = SelectKBest(f_classif, k=7)
    X_train_k7 = selector_k7.fit_transform(X_train, y_train)
    X_val_k7 = selector_k7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    # Grid search (candidates=27, cv=5 -> 135 fits)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
    )
    grid.fit(X_train_k7, y_train)
    grid_val_pred = grid.predict(X_val_k7)
    grid_val_f1 = round(float(f1_score(y_val, grid_val_pred, zero_division=0)), 4)
    grid_best_acc = round(float(grid.best_score_), 4)

    # Randomized search (n_iter=9, cv=5 -> 45 fits)
    rand = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
    )
    rand.fit(X_train_k7, y_train)
    rand_val_pred = rand.predict(X_val_k7)
    rand_val_f1 = round(float(f1_score(y_val, rand_val_pred, zero_division=0)), 4)
    rand_best_acc = round(float(rand.best_score_), 4)

    # Default forest on same 7 features
    def_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    def_rf.fit(X_train_k7, y_train)
    def_val_pred = def_rf.predict(X_val_k7)
    def_val_f1 = round(float(f1_score(y_val, def_val_pred, zero_division=0)), 4)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid.best_params_),
            "best_cv_accuracy": grid_best_acc,
            "val_f1": grid_val_f1,
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rand.best_params_),
            "best_cv_accuracy": rand_best_acc,
            "val_f1": rand_val_f1,
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": def_val_f1,
        },
    ]
    pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)
    print("outputs/part_c_tuning.csv written.")

    # 5. Choosing the final configuration
    # Ties: default_forest_same_7_features, then grid_search, then randomized_search
    priority = {"default_forest_same_7_features": 3, "grid_search": 2, "randomized_search": 1}
    candidates_eval = [
        ("grid_search", grid_val_f1, grid.best_params_),
        ("randomized_search", rand_val_f1, rand.best_params_),
        ("default_forest_same_7_features", def_val_f1, {}),
    ]
    # Sort descending by val_f1, then priority
    candidates_eval.sort(key=lambda x: (x[1], priority[x[0]]), reverse=True)
    winner_name, winner_f1, winner_params = candidates_eval[0]
    print(f"Final model chosen: {winner_name} with val_f1={winner_f1} and params={winner_params}")

    # Build final Pipeline fitted on training rows
    if winner_name == "default_forest_same_7_features":
        final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        n_est = 100
        max_d = "unlimited"
        min_leaf = 1
    else:
        final_clf = RandomForestClassifier(random_state=42, **winner_params)
        n_est = winner_params.get("n_estimators", 100)
        max_d = winner_params.get("max_depth", "unlimited")
        min_leaf = winner_params.get("min_samples_leaf", 1)

    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", final_clf),
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Alert threshold
    # Predict positive class probabilities on validation rows
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []
    chosen_threshold = 0.1

    for t in thresholds:
        pred_t = (val_probs >= t).astype(int)
        tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_val, pred_t)
        thresh_records.append({
            "threshold": t,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })
        # Operations rule: highest t whose validation recall >= 0.80
        if rec >= 0.80:
            if t > chosen_threshold:
                chosen_threshold = t

    pd.DataFrame(thresh_records).to_csv("outputs/part_c_thresholds.csv", index=False)
    print(f"outputs/part_c_thresholds.csv written. Chosen threshold: {chosen_threshold}")

    # 7. One test evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_pred_thresh = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_clf_metrics(y_test, test_pred_thresh)
    final_test_row = [{
        "threshold": chosen_threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }]
    pd.DataFrame(final_test_row).to_csv("outputs/part_c_final_test.csv", index=False)
    print("outputs/part_c_final_test.csv written.")

    # outputs/part_c_final_config.csv
    final_config_row = [{
        "selector_k": 7,
        "n_estimators": n_est,
        "max_depth": max_d,
        "min_samples_leaf": min_leaf,
        "threshold": chosen_threshold,
        "chosen_from": winner_name,
    }]
    pd.DataFrame(final_config_row).to_csv("outputs/part_c_final_config.csv", index=False)
    print("outputs/part_c_final_config.csv written.")


if __name__ == "__main__":
    run_part_c()
