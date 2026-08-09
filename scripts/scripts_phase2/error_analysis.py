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

# scripts/error_analysis.py
# FIX — was called with X_test, y_test (undefined). Now uses X_ts, y_ts.

def error_analysis(model, model_name, X_test, y_test, feature_names):
    """
    Analyse False Positives (legitimate flagged as fraud)
    and False Negatives (fraud missed by the model).
    """
    y_pred = model.predict(X_test)
    df_err = pd.DataFrame(X_test, columns=feature_names)
    df_err['y_true'] = np.array(y_test)
    df_err['y_pred'] = y_pred

    fp    = df_err[(df_err['y_true'] == 0) & (df_err['y_pred'] == 1)]
    fn    = df_err[(df_err['y_true'] == 1) & (df_err['y_pred'] == 0)]
    total = len(df_err)

    print(f"\n{'='*58}")
    print(f"  ERROR ANALYSIS: {model_name.upper()}")
    print(f"{'='*58}")
    print(f"  Total test samples : {total:,}")
    print(f"  False Positives    : {len(fp):,} "
          f"({len(fp)/total*100:.2f}%)  — legit flagged as fraud")
    print(f"  False Negatives    : {len(fn):,} "
          f"({len(fn)/total*100:.2f}%)  — fraud missed by model")

    key_cols = ['amount', 'balanceDiff', 'amount_ratio']
    existing = [c for c in key_cols if c in feature_names]
    if existing:
        print("\n  ── False Positive Key Features ──")
        for col in existing:
            print(f"    {col:<18}: mean={fp[col].mean():.4f}  std={fp[col].std():.4f}")
        print("\n  ── False Negative Key Features ──")
        for col in existing:
            print(f"    {col:<18}: mean={fn[col].mean():.4f}  std={fn[col].std():.4f}")

    return fp, fn



# Run error analysis for all four models
lr_fp,  lr_fn  = error_analysis(lr_model,  "Logistic Regression", X_ts, y_ts, FEATURE_NAMES)
rf_fp,  rf_fn  = error_analysis(rf_model,  "Random Forest",       X_ts, y_ts, FEATURE_NAMES)
xgb_fp, xgb_fn = error_analysis(xgb_model, "XGBoost",             X_ts, y_ts, FEATURE_NAMES)
nn_fp,  nn_fn  = error_analysis(nn_model,  "Neural Network",      X_ts, y_ts, FEATURE_NAMES)


def plot_error_analysis(all_fps, all_fns, model_names, feature_names):
    """
    Bar chart of FP and FN counts per model.
    """
    fp_counts = [len(fp) for fp in all_fps]
    fn_counts = [len(fn) for fn in all_fns]

    x     = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width/2, fp_counts, width, color=C['orange'],
                   edgecolor='white', linewidth=1.2, label='False Positives (FP)')
    bars2 = ax.bar(x + width/2, fn_counts, width, color=C['red'],
                   edgecolor='white', linewidth=1.2, label='False Negatives (FN)')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Error Count", fontsize=12)
    ax.set_title("Error Analysis: False Positives vs False Negatives per Model",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/plots/error_analysis.png", bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("Error analysis chart saved.")

plot_error_analysis(
    [lr_fp,  rf_fp,  nn_fp,  xgb_fp],
    [lr_fn,  rf_fn,  nn_fn,  xgb_fn],
    list(ALL_MODELS.keys()),
    FEATURE_NAMES,
)
