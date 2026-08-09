# Used to capture non-linear relationships and feature interactions in transaction data.
# It improves robustness over linear models by reducing variance through ensemble bagging of decision trees.

# scripts/random_forest.py
# Captures non-linear relationships and feature interactions.
# Ensemble bagging reduces variance over a single decision tree.

def run_random_forest(X_train, X_test, y_train, y_test, X_cv, y_cv):
    """
    Random Forest.
    X_cv, y_cv : pre-SMOTE pre-scaling data passed to run_kfold pipeline
    X_train    : post-SMOTE post-scaling data used for final model.fit()
    """
    model = RandomForestClassifier(
        n_estimators      = 30,
        max_depth         = 12,
        min_samples_split = 20,
        min_samples_leaf  = 5,
        max_features      = 'sqrt',
        max_samples       = 0.5,
        # class_weight    REMOVED — SMOTE handles imbalance, double compensation hurts performance
        random_state      = 42,
        n_jobs            = -1,
    )

    # CV — uses X_cv (pre-SMOTE), pipeline handles SMOTE + scaling per fold
    cv_metrics = run_kfold(model, X_cv, y_cv, "random_forest")

    # Final training — uses X_train (post-SMOTE, post-scaling)
    model.fit(X_train, y_train)

    test_metrics = evaluate_model(model, X_test, y_test, "random_forest")
    all_metrics  = {**cv_metrics, **test_metrics}

    weights_path = os.path.join(MODEL_SAVE_PATH, "random_forest.pkl")
    joblib.dump(model, weights_path)
    print(f"  Weights saved  → {weights_path}")

    save_metrics(all_metrics, "random_forest")
    save_experiment_results(all_metrics, "Random Forest")

    return model, all_metrics


rf_model, rf_metrics = run_random_forest(X_tr, X_ts, y_tr, y_ts, X_cv, y_cv)
print("\nRandom Forest Model trained and saved.")

# MEMORY ADDITION #20
gc.collect()
