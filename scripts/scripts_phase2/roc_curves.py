ALL_MODELS  = {
    "Logistic Regression": lr_model,
    "Random Forest"      : rf_model,
    "XGBoost"            : xgb_model,
    "Neural Network"     : nn_model,
}
ALL_METRICS = {
    "Logistic Regression": lr_metrics,
    "Random Forest"      : rf_metrics,
    "XGBoost"            : xgb_metrics,
    "Neural Network"     : nn_metrics,
}

# scripts/roc_curves.py
# Plot ROC and Precision-Recall curves for all models on a single canvas.

def plot_roc_pr_curves(all_models, X_test, y_test):
    """
    Side-by-side ROC curve and Precision-Recall curve for all models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for (name, model), color in zip(all_models.items(), MODEL_COLORS.values()):
        y_prob = model.predict_proba(X_test)[:, 1]

        # ── ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc         = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=color, linewidth=2.5,
                     label=f"{name}  (AUC = {auc:.3f})")

        # ── Precision-Recall
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap           = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, color=color, linewidth=2.5,
                     label=f"{name}  (AP = {ap:.3f})")

    # ROC reference line
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5,
                 label='Random Classifier')
    axes[0].fill_between([0, 1], [0, 1], alpha=0.04, color='gray')
    axes[0].set_title("ROC Curves", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate",  fontsize=12)
    axes[0].legend(fontsize=10, loc='lower right')

    # PR baseline — fraud rate in test set
    baseline = y_test.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=1.2,
                    alpha=0.5, label=f'No-Skill Baseline ({baseline:.4f})')
    axes[1].set_title("Precision-Recall Curves", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Recall",    fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].legend(fontsize=10, loc='upper right')

    fig.suptitle("Model Discrimination Curves", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/plots/roc_pr_curves.png", bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("ROC and Precision-Recall curves saved.")

plot_roc_pr_curves(ALL_MODELS, X_ts, y_ts)
