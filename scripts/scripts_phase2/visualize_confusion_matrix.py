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

# scripts/visualize_confusion_matrix.py
# Plot annotated confusion matrices for all four models.

def plot_confusion_matrices_from_models(all_models, X_test, y_test):
    """
    Plot confusion matrices for all models in a 2x2 grid.
    Each cell shows count and row-normalised percentage.
    """
    cmaps = ['Greens', 'Blues', 'Purples', 'Oranges']
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, ((name, model), cmap) in enumerate(zip(all_models.items(), cmaps)):
        y_pred         = model.predict(X_test)
        cm             = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cm_norm        = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, ax=axes[i], cmap=cmap,
                    annot=False, fmt='.2f',
                    linewidths=2, linecolor='white',
                    cbar_kws={'shrink': 0.75, 'label': 'Row-normalised rate'},
                    vmin=0, vmax=1)

        cell_data = [
            [f"TN\n{tn:,}\n{cm_norm[0,0]:.1%}", f"FP\n{fp:,}\n{cm_norm[0,1]:.1%}"],
            [f"FN\n{fn:,}\n{cm_norm[1,0]:.1%}", f"TP\n{tp:,}\n{cm_norm[1,1]:.1%}"],
        ]
        for r in range(2):
            for c in range(2):
                txt_color = 'white' if cm_norm[r, c] > 0.55 else '#2c3e50'
                axes[i].text(c + 0.5, r + 0.5, cell_data[r][c],
                             ha='center', va='center',
                             fontsize=11, fontweight='bold', color=txt_color)

        axes[i].set_title(name, fontsize=14, fontweight='bold', pad=12)
        axes[i].set_xlabel("Predicted Label", fontsize=11)
        axes[i].set_ylabel("True Label",      fontsize=11)
        axes[i].set_xticklabels(['Legitimate', 'Fraud'], fontsize=10)
        axes[i].set_yticklabels(['Legitimate', 'Fraud'], fontsize=10, rotation=0)

    fig.suptitle("Confusion Matrices — All Models", fontsize=17,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrices.png", bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("Confusion matrices saved.")


plot_confusion_matrices_from_models(ALL_MODELS, X_ts, y_ts)
