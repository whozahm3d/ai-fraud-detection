# 1. CONFIGURATION

MODEL_SAVE_PATH = "outputs/models/"
METRICS_SAVE_PATH = "outputs/metrics/"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(METRICS_SAVE_PATH, exist_ok=True)


# Hyperparameter Configuration — parameters tested and final selections
hyper_table = pd.DataFrame([
    {"Model":"Logistic Regression","Parameter":"C",
     "Values Tested":"0.01, 0.1, 1.0","Final":"1.0",
     "Justification":"Default regularization — sufficient for linear baseline"},
    {"Model":"Logistic Regression","Parameter":"solver",
     "Values Tested":"lbfgs, saga","Final":"saga",
     "Justification":"saga supports n_jobs=-1 — faster on large dataset"},
    {"Model":"Logistic Regression","Parameter":"class_weight",
     "Values Tested":"balanced, None","Final":"None",
     "Justification":"Removed — ImbPipeline SMOTE handles imbalance; balanced caused precision collapse"},
    {"Model":"Random Forest","Parameter":"max_depth",
     "Values Tested":"7, 10, 12","Final":"12",
     "Justification":"Depth 7 too shallow — model predicted all-negative on fraud class"},
    {"Model":"Random Forest","Parameter":"n_estimators",
     "Values Tested":"50, 100, 30","Final":"30",
     "Justification":"Reduced for speed — 30 trees sufficient on 6.6M rows"},
    {"Model":"Random Forest","Parameter":"max_samples",
     "Values Tested":"0.4, 0.5, 0.6","Final":"0.5",
     "Justification":"0.4 undersampled fraud cases — KeyError on classification report"},
    {"Model":"Random Forest","Parameter":"class_weight",
     "Values Tested":"balanced_subsample, None","Final":"None",
     "Justification":"Removed — SMOTE inside ImbPipeline handles imbalance"},
    {"Model":"XGBoost","Parameter":"n_estimators",
     "Values Tested":"100, 200, 300","Final":"300",
     "Justification":"More trees improve generalization on tabular fraud data"},
    {"Model":"XGBoost","Parameter":"learning_rate",
     "Values Tested":"0.1, 0.05, 0.01","Final":"0.05",
     "Justification":"Lower LR with 300 trees — better generalization vs 0.1"},
    {"Model":"XGBoost","Parameter":"max_depth",
     "Values Tested":"4, 6, 8","Final":"6",
     "Justification":"Standard depth for fraud tabular data — 8 overfits"},
    {"Model":"XGBoost","Parameter":"scale_pos_weight",
     "Values Tested":"auto-computed, None","Final":"None",
     "Justification":"Removed — SMOTE handles imbalance; both caused over-prediction"},
    {"Model":"Neural Network","Parameter":"hidden_layer_sizes",
     "Values Tested":"(64,32), (32,16)","Final":"(32,16)",
     "Justification":"12 features don't need 64 neurons — same accuracy, faster"},
    {"Model":"Neural Network","Parameter":"batch_size",
     "Values Tested":"1024, 512, 256","Final":"256",
     "Justification":"1024 caused NaN in K-Fold — 256 stable for adam solver"},
    {"Model":"Neural Network","Parameter":"alpha",
     "Values Tested":"0.01, 0.03, 0.1","Final":"0.03",
     "Justification":"Approximates dropout=0.3 via L2 regularization"},
    {"Model":"Neural Network","Parameter":"early_stopping",
     "Values Tested":"True, False","Final":"False",
     "Justification":"True caused NaN inside ImbPipeline — internal val split had zero fraud"},
])

print("HYPERPARAMETER CONFIGURATION TABLE")
print("="*130)
print(hyper_table.to_string(index=False))
print("="*130)
hyper_table.to_csv("outputs/metrics/hyperparameter_table.csv", index=False)
print("\nHyperparameter table saved → outputs/metrics/hyperparameter_table.csv")


def evaluate_model(model, X_test, y_test, model_name):
    """
    Standard evaluation function for ALL models
    Ensures fair comparison across models
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)


    fraud_report = report.get("1")
    if not isinstance(fraud_report, dict):
        fraud_report = report.get(1, {})
    if not isinstance(fraud_report, dict):
        fraud_report = {}

    metrics = {
        "model_name"   : model_name,
        "test_accuracy": float(report["accuracy"]),
        "test_precision": float(fraud_report.get("precision", 0.0)),
        "test_recall"   : float(fraud_report.get("recall", 0.0)),
        "test_f1"       : float(fraud_report.get("f1-score", 0.0)),
        "test_auc_roc"  : auc,
        "test_avg_prec" : ap,
        "y_pred"        : y_pred,
        "y_prob"        : y_prob,
    }

    print(f"\n{'='*52}")
    print(f"  {model_name.upper()} — Test Set Results")
    print(f"{'='*52}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Legitimate', 'Fraud']))
    print(f"  AUC-ROC        : {auc:.4f}")
    print(f"  Avg Precision  : {ap:.4f}")

    return metrics


# NOTE: Accuracy is reported here for completeness only.
# It is NOT used as a primary evaluation metric for this model.
# Reason: The dataset has severe class imbalance (fraud << legitimate).
# A model predicting ALL transactions as legitimate would achieve >99% accuracy
# while detecting zero fraud — making accuracy actively misleading in this context.
# Primary metrics for model selection are: AUPRC and Recall.
