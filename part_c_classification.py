import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, NOISE_COLUMNS

def run_part_c():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # Task 1: Stratified partitions keeping class representation balanced
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Task 2: Standard baseline setups
    model_rows = []
    # Blind Baseline
    blind_preds = np.zeros(shape=y_val.shape)
    tn, fp, fn, tp = confusion_matrix(y_val, blind_preds).ravel()
    model_rows.append(["blind_all_on_time", tn, fp, fn, tp, round(accuracy_score(y_val, blind_preds), 4), 0.0, 0.0, 0.0])

    # Logistic Pipeline
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, lr_preds).ravel()
    model_rows.append(["logistic_regression", tn, fp, fn, tp, round(accuracy_score(y_val, lr_preds), 4), round(precision_score(y_val, lr_preds), 4), round(recall_score(y_val, lr_preds), 4), round(f1_score(y_val, lr_preds), 4)])

    # Tune Tree structural validation limits
    best_dt_d, best_dt_f1 = None, -1
    for d in:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        score = f1_score(y_val, dt.predict(X_val))
        if score > best_dt_f1:
            best_dt_f1, best_dt_d = score, d

    dt_best = DecisionTreeClassifier(criterion="gini", max_depth=best_dt_d, random_state=42)
    dt_best.fit(X_train, y_train)
    dt_preds = dt_best.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, dt_preds).ravel()
    model_rows.append([f"decision_tree_depth_{best_dt_d}", tn, fp, fn, tp, round(accuracy_score(y_val, dt_preds), 4), round(precision_score(y_val, dt_preds), 4), round(recall_score(y_val, dt_preds), 4), round(f1_score(y_val, dt_preds), 4)])

    # Untuned Default Forest Baseline
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    rf_preds = rf_def.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, rf_preds).ravel()
    model_rows.append(["random_forest_default", tn, fp, fn, tp, round(accuracy_score(y_val, rf_preds), 4), round(precision_score(y_val, rf_preds), 4), round(recall_score(y_val, rf_preds), 4), round(f1_score(y_val, rf_preds), 4)])
    pd.DataFrame(model_rows, columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).to_csv("outputs/part_c_models.csv", index=False)

    # Task 3: Signal Ranking & SelectKBest Filtering extraction
    selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    f_scores = selector_all.scores_
    f_ranks = pd.Series(f_scores).rank(ascending=False, method='min').astype(int).values
    rf_imp = rf_def.feature_importances_

    feat_scores = pd.DataFrame({
        "feature": FEATURES,
        "f_score": np.round(f_scores, 4),
        "f_rank": f_ranks,
        "rf_importance": np.round(rf_imp, 4),
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
    })
    feat_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

    # Evaluate Validation K-signal curves
    k_curves = []
    for k_val in:
        k_pipe = Pipeline([("selector", SelectKBest(f_classif, k=k_val)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
        k_pipe.fit(X_train, y_train)
        k_curves.append([k_val, round(f1_score(y_val, k_pipe.predict(X_val)), 4)])
    pd.DataFrame(k_curves, columns=["k", "val_f1"]).to_csv("outputs/part_c_selectk.csv", index=False)

    # Task 4: Hyperparameter grid space exploration (Targeting K=7 features)
    X_tr_7 = SelectKBest(f_classif, k=7).fit_transform(X_train, y_train)
    X_va_7 = SelectKBest(f_classif, k=7).fit(X_train, y_train).transform(X_val)

    param_grid = {
        "n_estimators":,
        "max_depth":,
        "min_samples_leaf": [1, 5, 10]
    }

    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(X_tr_7, y_train)
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42).fit(X_tr_7, y_train)

    # Default untuned control benchmark on top 7 features
    rf_7_def = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_7, y_train)

    tuning_rows = [
        ["grid_search", 27, 5, 135, str(gs.best_params_).replace("'", '"'), round(gs.best_score_, 4), round(f1_score(y_val, gs.predict(X_va_7)), 4)],
        ["randomized_search", 9, 5, 45, str(rs.best_params_).replace("'", '"'), round(rs.best_score_, 4), round(f1_score(y_val, rs.predict(X_va_7)), 4)],
        ["default_forest_same_7_features", 1, 0, 1, "defaults", "", round(f1_score(y_val, rf_7_def.predict(X_va_7)), 4)]
    ]
    pd.DataFrame(tuning_rows, columns=["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"]).to_csv("outputs/part_c_tuning.csv", index=False)

    # Task 5: Select final optimal parameters mapping tied entries safely
    f1_scores = [tuning_rows[2][6], tuning_rows[0][6], tuning_rows[1][6]]
    winning_idx = int(np.argmax(f1_scores))
    winners = ["default_forest_same_7_features", "grid_search", "randomized_search"]
    final_source = winners[winning_idx]

    if final_source == "grid_search":
        params = gs.best_params_
    elif final_source == "randomized_search":
        params = rs.best_params_
    else:
        params = {}

    final_pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **params))
    ]).fit(X_train, y_train)

    # Task 6: Risk Threshold calibrations
    val_probs = final_pipe.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []
    chosen_t = 0.1

    for t in thresholds:
        t_preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, t_preds).ravel()
        rec = recall_score(y_val, t_preds)
        thresh_records.append([t, tn, fp, fn, tp, round(accuracy_score(y_val, t_preds), 4), round(precision_score(y_val, t_preds, zero_division=0), 4), round(rec, 4), round(f1_score(y_val, t_preds, zero_division=0), 4)])

        if rec >= 0.80:
            chosen_t = max(chosen_t, t)

    pd.DataFrame(thresh_records, columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).to_csv("outputs/part_c_thresholds.csv", index=False)

    # Task 7: Evaluate general performance limits against test partitions
    t_preds_test = (final_pipe.predict_proba(X_test)[:, 1] >= chosen_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, t_preds_test).ravel()
    pd.DataFrame([[chosen_t, tn, fp, fn, tp, round(accuracy_score(y_test, t_preds_test), 4), round(precision_score(y_test, t_preds_test), 4), round(recall_score(y_test, t_preds_test), 4), round(f1_score(y_test, t_preds_test), 4)]],
                 columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).to_csv("outputs/part_c_final_test.csv", index=False)

    pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": params.get("n_estimators", 100),
        "max_depth": params.get("max_depth", "unlimited"),
        "min_samples_leaf": params.get("min_samples_leaf", 1),
        "threshold": chosen_t,
        "chosen_from": final_source
    }]).to_csv("outputs/part_c_final_config.csv", index=False)

if __name__ == "__main__":
    run_part_c()
