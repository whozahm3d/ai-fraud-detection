# Used as a baseline linear classifier to establish a reference performance benchmark. 
# It helps measure how well linearly separable the fraud patterns are and provides interpretability for feature impact

print_memory("[READY FOR/BEFORE TRAINING OF ALL MODELS]")

# scripts/logistic_regression.py
# Used as a baseline linear classifier to establish a reference performance benchmark.
# Measures how linearly separable fraud patterns are.

def run_logistic(X_train, X_test, y_train, y_test, X_cv, y_cv):
    """
    Logistic Regression — baseline model.
    X_cv, y_cv : pre-SMOTE pre-scaling data passed to run_kfold pipeline
    X_train    : post-SMOTE post-scaling data used for final model.fit()
    """
    model = LogisticRegression(
        max_iter     = 1000,
        C            = 1.0,
        solver = 'saga',   # swap from 'lbfgs' — faster on large datasets
        n_jobs = -1,       # saga supports parallelism, lbfgs does not
        random_state = 42,
        # class_weight REMOVED — ImbPipeline SMOTE handles imbalance
        # keeping balanced = double compensation → precision collapses to 0.006
    )

    # CV — uses X_cv (pre-SMOTE), pipeline handles SMOTE + scaling per fold
    cv_metrics = run_kfold(model, X_cv, y_cv, "logistic_regression")

    # Final training — uses X_train (post-SMOTE, post-scaling)
    model.fit(X_train, y_train)

    test_metrics = evaluate_model(model, X_test, y_test, "logistic_regression")
    all_metrics  = {**cv_metrics, **test_metrics}

    weights_path = os.path.join(MODEL_SAVE_PATH, "logistic_regression.pkl")
    joblib.dump(model, weights_path)
    print(f"  Weights saved  → {weights_path}")

    save_metrics(all_metrics, "logistic_regression")
    save_experiment_results(all_metrics, "Logistic Regression")

    return model, all_metrics

lr_model, lr_metrics = run_logistic(X_tr, X_ts, y_tr, y_ts, X_cv, y_cv)
print("\nLogistic Regression Model trained and saved.")

# FIX: Keep X_train_final, X_test_final, y_train_smote alive for later model cells
# Removed: del X_train_final, X_test_final, y_train_smote
gc.collect()
