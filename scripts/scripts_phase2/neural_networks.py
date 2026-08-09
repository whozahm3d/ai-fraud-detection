def run_neural_network(X_train, X_test, y_train, y_test, X_cv, y_cv):
    """
    Neural Network — plain sklearn MLPClassifier.
    Wrapper class removed — was causing NaN in K-Fold due to cloning issues inside ImbPipeline.
    Direct sklearn MLP works cleanly with cross_validate + ImbPipeline.
    """
    y_train = np.array(y_train, dtype=int)
    y_test  = np.array(y_test,  dtype=int)
    y_cv    = np.array(y_cv,    dtype=int)

    model = SklearnMLP(
        hidden_layer_sizes = (32, 16),
        activation         = 'relu',
        solver             = 'adam',
        alpha              = 0.03,
        batch_size         = 256,
        learning_rate      = 'adaptive',
        learning_rate_init = 0.001,
        max_iter           = 50,
        early_stopping     = False,
        random_state       = 42,
        tol                = 1e-4,
        n_iter_no_change   = 10,
    )

    cv_metrics = run_kfold(model, X_cv, y_cv, "neural_network")
    print("Cross-validation complete")

    model.fit(X_train, y_train)

    test_metrics = evaluate_model(model, X_test, y_test, "neural_network")
    all_metrics  = {**cv_metrics, **test_metrics}

    weights_path = os.path.join(MODEL_SAVE_PATH, "neural_network.pkl")
    joblib.dump(model, weights_path)
    print(f"  Weights saved  → {weights_path}")

    save_metrics(all_metrics, "neural_network")
    save_experiment_results(all_metrics, "Neural Network")
    print("Metrics saved")

    return model, all_metrics


# Note: early_stopping=False is intentional.
# With early_stopping=True inside ImbPipeline, the internal 10% val split
# can produce zero-fraud folds → NaN loss → training crash.
# Convergence is handled instead by tol + n_iter_no_change.
# max_iter=50 is a compute-constrained choice (200 is ideal but takes 2-4 hrs on CPU).
# 50 iterations still outperforms the original 20 and converges sufficiently for comparison but takes alot of time.

nn_model, nn_metrics = run_neural_network(X_tr, X_ts, y_tr, y_ts, X_cv, y_cv)
print("\nNeural Network Model trained and saved.")

# MEMORY ADDITION #22
gc.collect()
print_memory("[AFTER ALL MODELS TRAINED]")


def clean_experiment_log():
    """
    Keep only the most recent run per model.
    Earlier broken runs (zero metrics, NaN CV) are removed.
    """
    log_path = os.path.join(EXPERIMENTS_SAVE_PATH, "experiment_results.csv")
    if not os.path.exists(log_path):
        print("No experiment log found.")
        return

    exp = pd.read_csv(log_path)

    # Keep last occurrence per model_name — most recent run
    exp_clean = exp.groupby('model_name').last().reset_index()
    exp_clean.to_csv("outputs/experiments/cleaned_experiment_results.csv", index=False)

    print("Experiment log cleaned — kept latest run per model:")
    print(exp_clean[['model_name','test_recall','test_f1',
                      'test_avg_prec']].to_string(index=False))

clean_experiment_log()
