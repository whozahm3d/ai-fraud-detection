# Used as the primary gradient boosting model due to its superior performance on structured/tabular datasets.
# It handles class imbalance well, captures complex feature interactions, and provides high predictive accuracy for fraud detection.

# scale_pos_weight REMOVED — SMOTE inside CV pipeline already handles imbalance.
# Keeping both would double-compensate → model over-predicts fraud → low precision + low AUPRC.

# Used to model deeper non-linear representations of transaction patterns.
# It learns hierarchical feature interactions automatically and serves as the most
# expressive model in the pipeline for fraud classification.


def run_xgboost(X_train, X_test, y_train, y_test, X_cv, y_cv):
    """
    XGBoost Classifier — main fraud detection model.
    X_cv, y_cv : pre-SMOTE pre-scaling data passed to run_kfold pipeline
    X_train    : post-SMOTE post-scaling data used for final model.fit()

    scale_pos_weight intentionally removed:
    - SMOTE in CV pipeline brings fraud ratio to ~23% per fold
    - SMOTE on X_train already balanced the final training set
    - Adding scale_pos_weight on top makes model overly aggressive → test AUPRC collapses
    """
    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        # scale_pos_weight REMOVED — SMOTE handles imbalance, double compensation hurts performance
        eval_metric      = 'logloss',
        random_state     = 42,
        n_jobs           = -1,
    )

    # CV — uses X_cv (pre-SMOTE), pipeline handles SMOTE + scaling per fold
    cv_metrics = run_kfold(model, X_cv, y_cv, "xgboost")

    # Final training — uses X_train (post-SMOTE, post-scaling)
    model.fit(X_train, y_train)

    test_metrics = evaluate_model(model, X_test, y_test, "xgboost")
    all_metrics  = {**cv_metrics, **test_metrics}

    weights_path = os.path.join(MODEL_SAVE_PATH, "xgboost.pkl")
    joblib.dump(model, weights_path)
    print(f"  Weights saved  → {weights_path}")

    save_metrics(all_metrics, "xgboost")
    save_experiment_results(all_metrics, "XGBoost")

    return model, all_metrics

xgb_model, xgb_metrics = run_xgboost(X_tr, X_ts, y_tr, y_ts, X_cv, y_cv)
print("\nXGBoost Model trained and saved.")

# MEMORY ADDITION #21
gc.collect()
