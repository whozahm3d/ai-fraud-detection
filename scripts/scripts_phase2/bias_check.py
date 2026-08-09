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

# Run this to diagnose
import pandas as pd
df_debug = pd.DataFrame(X_ts, columns=FEATURE_NAMES)
type_cols = [c for c in FEATURE_NAMES if c.startswith('type_')]
print("Type columns found:", type_cols)
print("\nUnique values in each type column:")
for col in type_cols:
    print(f"  {col}: {df_debug[col].unique()[:5]}")
print("\nFraud count in y_ts:", np.sum(y_ts == 1))


# scripts/bias_check.py
# FIX — was called with X_test, y_test (undefined). Now uses X_ts, y_ts.

def bias_check(model, model_name, X_test, y_test, feature_names):
    """
    Evaluate fraud detection performance broken down by transaction type.
    Identifies whether the model is systematically biased toward or against
    specific transaction categories (TRANSFER, CASH_OUT, PAYMENT, etc.).
    """
    y_pred  = model.predict(X_test)           # predictions use full float array — correct
    df_bias = pd.DataFrame(X_test, columns=feature_names)
    df_bias['y_true'] = np.array(y_test)
    df_bias['y_pred'] = y_pred

    type_cols = [c for c in feature_names if c.startswith('type_')]

    if not type_cols:
        print(f"{model_name}: no transaction-type columns found.")
        return None

    print(f"\n{'='*68}")
    print(f"  BIAS CHECK: {model_name.upper()}")
    print(f"{'='*68}")
    print(f"  {'Type':<18} {'Total':>7} {'Fraud':>7} {'Recall':>8} {'Precision':>11} {'F1':>7}")
    print("  " + "─" * 62)

    rows = []
    for col in type_cols:
        subset = df_bias[df_bias[col] == 1]
        if len(subset) == 0 or subset['y_true'].sum() == 0:
            continue
        rpt = classification_report(
            subset['y_true'], subset['y_pred'],
            output_dict=True, zero_division=0
        )
        fm    = rpt.get('1', {'precision': 0, 'recall': 0, 'f1-score': 0})
        label = col.replace('type_', '')
        print(f"  {label:<18} {len(subset):>7,} {int(subset['y_true'].sum()):>7,} "
              f"{fm['recall']:>8.3f} {fm['precision']:>11.3f} {fm['f1-score']:>7.3f}")
        rows.append({
            'type'       : label,
            'total'      : len(subset),
            'fraud_count': int(subset['y_true'].sum()),
            'recall'     : fm['recall'],
            'precision'  : fm['precision'],
            'f1'         : fm['f1-score'],
        })

    return pd.DataFrame(rows)


def plot_bias_check(bias_dfs, model_names):
    """
    Grouped bar chart of Recall and F1-Score by transaction type per model.
    """
    type_color = {
        'CASH_IN' : C['green'],  'CASH_OUT': C['red'],
        'DEBIT'   : C['blue'],   'PAYMENT' : C['teal'],
        'TRANSFER': C['orange'], 'DEFAULT' : C['gray'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax, (name, bdf) in zip(axes, zip(model_names, bias_dfs)):
        if bdf is None or len(bdf) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=13)
            ax.set_title(name, fontsize=13, fontweight='bold')
            continue

        x      = np.arange(len(bdf))
        width  = 0.35
        colors = [type_color.get(t, C['gray']) for t in bdf['type']]

        bars_r = ax.bar(x - width/2, bdf['recall'],    width, color=colors,
                        alpha=0.85, edgecolor='white', linewidth=1, label='Recall')
        bars_f = ax.bar(x + width/2, bdf['f1'],        width, color=colors,
                        alpha=0.50, edgecolor='white', linewidth=1,
                        hatch='///', label='F1-Score')

        for bars in [bars_r, bars_f]:
            for bar in bars:
                if bar.get_height() > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.01,
                            f'{bar.get_height():.2f}',
                            ha='center', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(bdf['type'], fontsize=10)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    fig.suptitle("Bias Check: Recall & F1-Score by Transaction Type",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/plots/bias_check.png", bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("Bias check chart saved.")


# X_test_df was created in Cell 57 — pre-scaling, post-encoding, has named columns
# It has binary 0/1 for type_ columns — correct for bias check
# Use this instead of X_ts (which is scaled)

# Bias check — correct data per model type
# Tree models (RF, XGB): scale-invariant → X_test_df.values (unscaled) is fine
# Linear/NN models (LR, NN): scale-sensitive → must use X_ts (scaled)
#
# The bias_check function filters by type_ columns (binary).
# In X_ts (scaled), these columns are near 0 and near 1 due to StandardScaler.
# We round them back to integer so the == 1 filter works correctly.

def prepare_for_bias(X_arr, feature_names):
    """
    Round type_ columns back to 0/1 integers so bias_check filtering works
    correctly on scaled data. Does not affect model prediction (predict() uses
    the raw array, not this rounded version).
    This is only used for the subset-filtering logic inside bias_check.
    """
    df_b = pd.DataFrame(X_arr, columns=feature_names)
    type_cols = [c for c in feature_names if c.startswith('type_')]
    df_b[type_cols] = df_b[type_cols].round().astype(int)
    return df_b.values

# Scale-invariant models — unscaled input is acceptable
bias_rf  = bias_check(rf_model,  "Random Forest", X_test_df.values,              y_ts, FEATURE_NAMES)
bias_xgb = bias_check(xgb_model, "XGBoost",       X_test_df.values,              y_ts, FEATURE_NAMES)

# Scale-sensitive models — must use scaled input; round type_ cols for filter
bias_lr  = bias_check(lr_model,  "Logistic Regression", prepare_for_bias(X_ts, FEATURE_NAMES), y_ts, FEATURE_NAMES)
bias_nn  = bias_check(nn_model,  "Neural Network",      prepare_for_bias(X_ts, FEATURE_NAMES), y_ts, FEATURE_NAMES)


plot_bias_check(
    [bias_lr, bias_rf, bias_xgb, bias_nn],
    ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network'],
)

print_memory("[PIPELINE COMPLETE]")
print("\nAll done! Models, metrics, and plots saved in outputs/")
