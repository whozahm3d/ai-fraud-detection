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

# Analyse which features contribute most to fraud detection in tree-based models.

print("Feature names:")
print(FEATURE_NAMES)
print("\nColumns with 'type':")
print([c for c in FEATURE_NAMES if 'type' in c.lower()])

# scripts/feature_importance.py
# Tree-based models only — LR and NN do not have feature_importances_

def plot_feature_importance(rf_model, xgb_model, feature_names, top_n=12):
    """
    Horizontal bar chart of feature importances for RF and XGBoost.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, model, name, color in zip(
        axes,
        [rf_model, xgb_model],
        ["Random Forest", "XGBoost"],
        [C['blue'], C['orange']],
    ):
        importances  = model.feature_importances_
        top_idx      = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in top_idx]
        top_vals     = importances[top_idx]

        bars = ax.barh(top_features, top_vals, color=color,
                       edgecolor='white', linewidth=0.8, height=0.65)
        for bar, val in zip(bars, top_vals):
            ax.text(bar.get_width() + 0.0005,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)

        ax.set_title(f"{name}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_xlim(0, top_vals.max() * 1.2)

    fig.suptitle(f"Top-{top_n} Feature Importances", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance.png", bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("Feature importance charts saved.")

plot_feature_importance(rf_model, xgb_model, FEATURE_NAMES)
