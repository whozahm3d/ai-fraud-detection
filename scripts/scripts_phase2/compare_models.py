# scripts/compare_model.py
# FIX — sorted by Test Avg Prec (AUPRC) not Test F1
# AUPRC is primary metric per TA instruction for imbalanced fraud detection

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


def compare_models(all_metrics):
    """
    Aggregate CV and test metrics for all models into a comparison table.
    Sorted by Test Avg Prec (AUPRC) descending — primary metric for fraud detection.
    Saves comparison as both CSV and JSON.
    """
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model"          : name,
            "CV Precision"   : f"{m['cv_precision_mean']:.3f} ± {m['cv_precision_std']:.3f}",
            "CV Recall"      : f"{m['cv_recall_mean']:.3f} ± {m['cv_recall_std']:.3f}",
            "CV F1"          : f"{m['cv_f1_mean']:.3f} ± {m['cv_f1_std']:.3f}",
            "CV AUC-ROC"     : f"{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}",
            "CV Avg Prec"    : f"{m['cv_average_precision_mean']:.3f} ± {m['cv_average_precision_std']:.3f}",
            "Test Precision" : round(m['test_precision'], 4),
            "Test Recall"    : round(m['test_recall'],    4),
            "Test F1"        : round(m['test_f1'],        4),
            "Test AUC-ROC"   : round(m['test_auc_roc'],   4),
            "Test Avg Prec"  : round(m['test_avg_prec'],  4),
        })

    # FIX — sort by Test Avg Prec (AUPRC), not Test F1
    # AUPRC is the primary metric for imbalanced fraud detection
    # Test F1 is misleading when precision and recall are both low
    df_cmp = (pd.DataFrame(rows)
                .sort_values("Test Avg Prec", ascending=False)
                .reset_index(drop=True))

    print("\n" + "=" * 100)
    print("  MODEL COMPARISON TABLE  (sorted by Test AUPRC ↓ — primary metric)")
    print("=" * 100)
    print(df_cmp.to_string(index=False))
    print("=" * 100)

    best = df_cmp.iloc[0]
    print(f"\n  🏆 BEST MODEL  : {best['Model']}")
    print(f"     Test AUPRC  : {best['Test Avg Prec']}")
    print(f"     Test Recall : {best['Test Recall']}")
    print(f"     Test AUC-ROC: {best['Test AUC-ROC']}")

    # NOTE: Accuracy is excluded from this table intentionally.
    # A model predicting all transactions as legitimate achieves >99% accuracy
    # while detecting zero fraud. AUPRC and Recall are the honest metrics here.

    # ── CSV
    csv_path = os.path.join(METRICS_SAVE_PATH, "model_comparison.csv")
    df_cmp.to_csv(csv_path, index=False)
    print(f"\n  Comparison table saved → {csv_path}")

    # ── JSON
    json_path = os.path.join(METRICS_SAVE_PATH, "model_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(df_cmp.to_dict(orient='records'), f, indent=4)
    print(f"  Comparison table saved → {json_path}")

    return df_cmp

df_compare = compare_models(ALL_METRICS)

# NOTE: Models are compared and ranked by AUPRC and Recall — not accuracy.
# Accuracy is included in the table for reference only.
# In imbalanced fraud detection, a high-accuracy model that misses fraud cases
# is worse than a lower-accuracy model that catches them.

def plot_model_comparison(df_compare):
    """
    4-panel bar chart — one panel per key test metric.
    Sorted by AUPRC to match compare_models ranking.
    """
    metrics = ['Test Avg Prec', 'Test Recall', 'Test F1', 'Test AUC-ROC']
    labels  = ['Avg Precision (AUPRC)', 'Recall', 'F1-Score', 'AUC-ROC']
    colors  = [C['blue'], C['red'], C['green'], C['orange']]
    models  = df_compare['Model'].tolist()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = df_compare[metric].tolist()
        bars = ax.bar(models, vals, color=color, edgecolor='white',
                      linewidth=1.2, width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)

    fig.suptitle("Model Comparison — Test Set Metrics (sorted by AUPRC)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/plots/model_comparison_bars.png", bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("Model comparison bar chart saved.")

plot_model_comparison(df_compare)

# Plot 3 : Per-metric bar chart — F1, Precision, Recall, AUC, AUPRC
# Adapted to fraud detection pipeline — uses y_prob from ALL_METRICS

def plot_metrics_bar_comparison(ALL_METRICS):
    """
    5-panel bar chart — one panel per key metric.
    Sorted by AUPRC (primary metric) to match compare_models ranking.
    Directly comparable to the 4-panel plot_model_comparison above —
    this version adds AUPRC as a dedicated panel and annotates the best bar.
    """
    metric_keys = [
        ('test_avg_prec',  'AUPRC (Avg Precision)',  C['blue']),
        ('test_recall',    'Recall',                  C['red']),
        ('test_f1',        'F1-Score',                C['green']),
        ('test_auc_roc',   'AUC-ROC',                 C['orange']),
        ('test_precision', 'Precision',               C['purple']),
    ]

    # Sort models by AUPRC descending — consistent with compare_models
    sorted_models = sorted(
        ALL_METRICS.items(),
        key=lambda x: x[1].get('test_avg_prec', 0),
        reverse=True
    )
    model_names = [m[0] for m in sorted_models]
    colors_list = [MODEL_COLORS.get(m, C['gray']) for m in model_names]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    for ax, (metric_key, metric_label, bar_color) in zip(axes, metric_keys):
        vals = [m[1].get(metric_key, 0.0) for m in sorted_models]

        bars = ax.bar(model_names, vals,
                      color=colors_list, edgecolor='white',
                      linewidth=1.2, width=0.5)

        # Annotate value on top of each bar
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        # Highlight best bar with black border
        best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2.0)

        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        "Model Comparison — All Metrics (sorted by AUPRC ↓)\n"
        "Black border = best model per metric",
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig("outputs/plots/model_comparison_all_metrics.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Extended metrics bar chart saved.")
    gc.collect()

plot_metrics_bar_comparison(ALL_METRICS)


def plot_fraud_prob_distribution(ALL_METRICS, y_test):
    """
    Equivalent of Actual vs Predicted for classification.
    Shows predicted fraud probability distribution split by actual class
    for each model — tells you how well each model separates fraud from legit.
    A good model shows two clearly separated peaks (legit near 0, fraud near 1).
    """
    n_models = len(ALL_METRICS)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=False)

    if n_models == 1:
        axes = [axes]

    y_test_arr = np.array(y_test)

    for ax, (model_name, metrics) in zip(axes, ALL_METRICS.items()):
        y_prob = metrics.get('y_prob')
        if y_prob is None:
            ax.set_visible(False)
            continue

        y_prob = np.array(y_prob)
        color  = MODEL_COLORS.get(model_name, C['gray'])

        # Legitimate transactions — predicted probs
        prob_legit = y_prob[y_test_arr == 0]
        # Fraud transactions — predicted probs
        prob_fraud = y_prob[y_test_arr == 1]

        ax.hist(prob_legit, bins=50, color=C['green'], alpha=0.6,
                label='Legitimate', density=True, edgecolor='white')
        ax.hist(prob_fraud, bins=50, color=C['red'], alpha=0.7,
                label='Fraud', density=True, edgecolor='white')

        # Decision threshold line
        ax.axvline(x=0.5, color='black', lw=1.5, ls='--', label='Threshold = 0.5')

        auprc = metrics.get('test_avg_prec', 0.0)
        recall = metrics.get('test_recall', 0.0)

        ax.set_title(f"{model_name}\nAUPRC={auprc:.3f}  Recall={recall:.3f}",
                     fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel("Predicted Fraud Probability", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Predicted Fraud Probability Distribution — All Models\n"
        "Good model: Legitimate peaks near 0, Fraud peaks near 1 (clear separation)",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig("outputs/plots/fraud_prob_distribution_all_models.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Fraud probability distribution plot saved.")
    gc.collect()

plot_fraud_prob_distribution(ALL_METRICS, y_ts)


# FIX — runs on all 4 models, not just NN
# Best threshold per model saved for deployment reference

def threshold_optimization(model, X_test, y_test, model_name="Model"):
    """
    Find optimal decision threshold that maximises F1-Score.
    Plots Precision/Recall/F1 vs threshold and the PR curve.
    """
    print(f"\n{'='*70}")
    print(f"  THRESHOLD OPTIMIZATION: {model_name.upper()}")
    print(f"{'='*70}")

    y_proba     = model.predict_proba(X_test)[:, 1]
    thresholds  = np.linspace(0, 1, 200)
    precisions, recalls, f1_scores = [], [], []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if len(np.unique(y_pred)) == 1:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            continue
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    optimal_idx       = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1        = f1_scores[optimal_idx]
    optimal_precision = precisions[optimal_idx]
    optimal_recall    = recalls[optimal_idx]

    print(f"\n  Optimal Threshold : {optimal_threshold:.3f}")
    print(f"  Precision         : {optimal_precision:.4f}")
    print(f"  Recall            : {optimal_recall:.4f}")
    print(f"  F1-Score          : {optimal_f1:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(thresholds, precisions, label='Precision', linewidth=2, color=C['blue'])
    axes[0].plot(thresholds, recalls,    label='Recall',    linewidth=2, color=C['red'])
    axes[0].plot(thresholds, f1_scores,  label='F1-Score',  linewidth=2,
                 color=C['green'], linestyle='--')
    axes[0].axvline(optimal_threshold, color='black', linestyle=':',
                    linewidth=2, label=f'Optimal ({optimal_threshold:.3f})')
    axes[0].set_xlabel('Decision Threshold', fontsize=11)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title(f'{model_name} — Metrics vs Threshold', fontsize=12, fontweight='bold')
    axes[0].legend()

    axes[1].plot(recalls, precisions, linewidth=2, color=C['purple'])
    axes[1].scatter([optimal_recall], [optimal_precision],
                    color=C['red'], s=120, zorder=5,
                    label=f'Optimal point')
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title(f'{model_name} — Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(f'outputs/plots/threshold_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

    return optimal_threshold, optimal_f1

# Run for all 4 models
thresholds_summary = {}
for name, model in ALL_MODELS.items():
    t, f1 = threshold_optimization(model, X_ts, y_ts, name)
    thresholds_summary[name] = {'optimal_threshold': round(t, 3), 'optimal_f1': round(f1, 4)}

print("\n── Optimal Thresholds Summary ──")
for name, vals in thresholds_summary.items():
    print(f"  {name:<22}: threshold={vals['optimal_threshold']}  F1={vals['optimal_f1']}")

# CONFUSION MATRIX GRID — all models side by side
# Shows raw TP, FP, TN, FN counts per model
# Reveals which model catches fraud vs which over-flags legit

def plot_confusion_matrices(ALL_METRICS, y_test):
    """
    1-row grid of confusion matrices — one per model.
    Normalized by true class (rows) so imbalance doesn't hide recall.
    Both raw counts and normalized rates shown.
    """
    n_models = len(ALL_METRICS)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    y_test_arr = np.array(y_test)

    for ax, (model_name, metrics) in zip(axes, ALL_METRICS.items()):
        y_pred = metrics.get('y_pred')
        if y_pred is None:
            ax.set_visible(False)
            continue

        y_pred = np.array(y_pred)
        color  = MODEL_COLORS.get(model_name, C['blue'])

        cm = confusion_matrix(y_test_arr, y_pred)
        cm_norm = confusion_matrix(y_test_arr, y_pred, normalize='true')

        # Plot normalized matrix for color scale
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm,
            display_labels=['Legitimate', 'Fraud']
        )
        disp.plot(ax=ax, colorbar=True, cmap='rocket')

        # Overwrite cell text with both raw count and percentage
        tn, fp, fn, tp = cm.ravel()
        tn_r, fp_r, fn_r, tp_r = cm_norm.ravel()

        cell_texts = [
            (0, 0, tn,  tn_r,  'TN'),
            (0, 1, fp,  fp_r,  'FP'),
            (1, 0, fn,  fn_r,  'FN'),
            (1, 1, tp,  tp_r,  'TP'),
        ]

        for row, col, raw, rate, label in cell_texts:
            ax.texts[row * 2 + col].set_text(
                f"{label}\n{raw:,}\n({rate:.1%})"
            )
            ax.texts[row * 2 + col].set_fontsize(10)
            ax.texts[row * 2 + col].set_fontweight('bold')

        auprc  = metrics.get('test_avg_prec', 0.0)
        recall = metrics.get('test_recall',   0.0)

        ax.set_title(
            f"{model_name}\nAUPRC={auprc:.3f}  Recall={recall:.3f}",
            fontsize=11, fontweight='bold', color=color
        )
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label",      fontsize=10)

    fig.suptitle(
        "Confusion Matrices — All Models (Normalized by True Class)\n"
        "FN = missed fraud (critical)  |  FP = false alarms",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrices_all_models.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Confusion matrix grid saved.")
    gc.collect()

plot_confusion_matrices(ALL_METRICS, y_test)


# For XGBoost, extract from training data patterns instead
# XGBoost INTERPRETABILITY ANALYSIS
# XGBoost has feature_importances_ (gain-based importance from tree splits).
# Below: show top features ranked by importance, plus mean feature values
# in fraud-predicted vs legit-predicted subsets for additional context.

print(f"\n{'='*70}")
print(f"  INTERPRETABILITY ANALYSIS: XGBOOST")
print(f"{'='*70}")

# Feature importances — directly from the model
importances = xgb_model.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features by XGBoost Importance (gain):")
print(feat_importance_df.head(10).to_string(index=False))

# Mean feature values split by model prediction — supplementary context
print("\nFeature Mean Values: Fraud-Predicted vs Legit-Predicted")
print(f"{'Feature':<20} {'Fraud Mean':>12} {'Legit Mean':>12} {'Difference':>12}")
print("-" * 58)

fraud_mask = (xgb_model.predict(X_ts) == 1)
legit_mask = ~fraud_mask

for i, feat_name in enumerate(FEATURE_NAMES):
    fraud_mean = X_ts[fraud_mask, i].mean()
    legit_mean = X_ts[legit_mask, i].mean()
    diff = abs(fraud_mean - legit_mean)
    print(f"{feat_name:<20} {fraud_mean:>12.4f} {legit_mean:>12.4f} {diff:>12.4f}")


def explain_fraud_cases(model, X_test, y_test, feature_names, model_name="Model", n_examples=5):
    """Show why model flagged specific transactions as fraud."""
    print(f"\n{'='*70}")
    print(f"  EXAMPLE FRAUD CASES: {model_name.upper()}")
    print(f"{'='*70}")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fraud_mask = (y_pred == 1) & (y_test == 1)
    fraud_indices = np.where(fraud_mask)[0]
    
    if len(fraud_indices) == 0:
        print("No fraud cases detected in test set")
        return
    
    fraud_indices = fraud_indices[np.argsort(y_proba[fraud_indices])[::-1]][:n_examples]
    
    print(f"\nShowing {min(n_examples, len(fraud_indices))} high-confidence fraud cases:")
    for i, idx in enumerate(fraud_indices):
        print(f"\n  Case {i+1}:")
        print(f"    Fraud Probability: {y_proba[idx]:.4f}")
        
        if hasattr(model, 'feature_importances_'):
            top_features = np.argsort(model.feature_importances_)[::-1][:3]
            print(f"    Top contributing features:")
            for feat_idx in top_features:
                value = X_test.iloc[idx, feat_idx] if hasattr(X_test, 'iloc') else X_test[idx, feat_idx]
                print(f"      - {feature_names[feat_idx]}: {value:.4f}")

print("Analysis functions loaded!")

explain_fraud_cases(xgb_model, X_ts, y_ts, FEATURE_NAMES, "XGBoost", n_examples=10)

print("\n" + "="*70)
print("DETAILED FRAUD CASE ANALYSIS")
print("="*70)

# Get fraud cases
y_proba = xgb_model.predict_proba(X_ts)[:, 1]
y_pred  = xgb_model.predict(X_ts)
fraud_mask = (y_pred == 1) & (y_ts == 1)
fraud_indices = np.where(fraud_mask)[0]

# Show top 5 fraud cases
for i, idx in enumerate(fraud_indices[:5]):
    print(f"\n Case {i+1} (Fraud Probability: {y_proba[idx]:.4f}):")
    print("  Features:")
    
    for j, feat_name in enumerate(FEATURE_NAMES):
        value = X_ts[idx, j]
        print(f"    {feat_name:<20}: {value:>12.4f}")
print_memory("[PIPELINE COMPLETE]")
print("\nAll done! Models, metrics, and plots saved in outputs/")



print(len(FEATURE_NAMES), xgb_model.feature_importances_.shape[0])

# FEATURE IMPORTANCE — All models except NN Model
# Shows which features drive fraud detection decisions
# Directly validates feature engineering (balanceDiff, amount_ratio)
# Ties to ablation study — if engineered features rank high,
# ablation C conclusion is visually confirmed here

def plot_feature_importance_comparison(models_dict, feature_names):
    """
    Side-by-side horizontal bar chart of feature importances across all models.
    Models included:
      - XGBoost   : feature_importances_ (gain-based internally)
      - Random Forest : feature_importances_ (mean decrease impurity)
      - Logistic Regression : abs(coef_) — magnitude = importance
      - Neural Network (SklearnMLP) : excluded — MLP has no feature importance
    Engineered features (balanceDiff, amount_ratio) highlighted in red.
    """
    from matplotlib.patches import Patch

    ENGINEERED = {'balanceDiff', 'amount_ratio'}

    # ── Collect importances per model
    importance_map = {}

    # XGBoost
    if 'XGBoost' in models_dict:
        imp = models_dict['XGBoost'].feature_importances_
        importance_map['XGBoost'] = imp / imp.sum() if imp.sum() > 0 else imp

    # Random Forest
    if 'Random Forest' in models_dict:
        imp = models_dict['Random Forest'].feature_importances_
        importance_map['Random Forest'] = imp / imp.sum() if imp.sum() > 0 else imp

    # Logistic Regression — abs coefficients as proxy for importance
    if 'Logistic Regression' in models_dict:
        coef = np.abs(models_dict['Logistic Regression'].coef_[0])
        importance_map['Logistic Regression'] = coef / coef.sum() if coef.sum() > 0 else coef

    # SklearnMLP has no feature_importances_ — excluded by design
    # Neural Network skipped — weights are distributed, not attributable per feature

    n_models = len(importance_map)
    if n_models == 0:
        print("No models with feature importance available.")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, max(6, len(feature_names) * 0.55)),
                             sharey=True)

    if n_models == 1:
        axes = [axes]

    model_colors_map = {
        'XGBoost'             : C['orange'],
        'Random Forest'       : C['blue'],
        'Logistic Regression' : C['red'],
    }

    for ax, (model_name, importances) in zip(axes, importance_map.items()):

        # Sort each model independently — different models rank features differently
        sorted_idx  = np.argsort(importances)
        sorted_feat = [feature_names[i] for i in sorted_idx]
        sorted_imp  = importances[sorted_idx]

        bar_colors = [
            C['red'] if f in ENGINEERED else model_colors_map.get(model_name, C['gray'])
            for f in sorted_feat
        ]

        bars = ax.barh(
            sorted_feat, sorted_imp,
            color=bar_colors, edgecolor='white', linewidth=1.0
        )

        # Annotate values
        for bar, val in zip(bars, sorted_imp):
            if val > 0.001:
                ax.text(
                    bar.get_width() + sorted_imp.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}',
                    va='center', ha='left',
                    fontsize=8.5, fontweight='bold'
                )

        ax.set_title(model_name, fontsize=13, fontweight='bold',
                     color=model_colors_map.get(model_name, C['gray']))
        ax.set_xlabel("Normalized Importance", fontsize=10)
        ax.set_xlim(0, sorted_imp.max() * 1.25)
        ax.grid(True, alpha=0.3, axis='x')

        if ax == axes[0]:
            ax.set_ylabel("Feature", fontsize=10)

    # Shared legend
    legend_elements = [
        Patch(facecolor=C['red'],  label='Engineered Feature (balanceDiff, amount_ratio)'),
        Patch(facecolor=C['gray'], label='Raw Feature (color = model)'),
    ]
    fig.legend(handles=legend_elements, fontsize=10,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Feature Importance Comparison — XGBoost vs Random Forest vs Logistic Regression\n"
        "Red = engineered features | Neural Network excluded (MLP has no feature importance)",
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance_comparison.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Feature importance comparison chart saved.")
    gc.collect()


# ── Call it
plot_feature_importance_comparison(
    models_dict = {
        'Logistic Regression' : lr_model,
        'Random Forest'       : rf_model,
        'XGBoost'             : xgb_model,
    },
    feature_names = FEATURE_NAMES,
)


def plot_metrics_lineplot(ALL_METRICS):
    """
    Line chart — each line is a metric, each point is a model.
    Same style as ResearchGate academic comparison charts.
    Shows performance trends across models per metric simultaneously.
    """
    metrics_to_show = [
        ('test_precision', 'Precision',  C['blue']),
        ('test_recall',    'Recall',     C['green']),
        ('test_f1',        'F1-Score',   C['orange']),
        ('test_auc_roc',   'AUC-ROC',    C['purple']),
        ('test_avg_prec',  'AUPRC',      C['red']),
        ('test_accuracy',  'Accuracy*',  C['gray']),
    ]

    model_names = list(ALL_METRICS.keys())
    x           = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(13, 6))

    for metric_key, metric_label, color in metrics_to_show:
        vals = [float(ALL_METRICS[m].get(metric_key, 0.0)) for m in model_names]

        # Primary metrics (AUPRC, Recall) get thicker line + larger markers
        is_primary = metric_label in ('AUPRC', 'Recall')
        lw         = 2.5 if is_primary else 1.5
        ms         = 9   if is_primary else 6
        ls         = '-' if is_primary else '--'

        ax.plot(x, vals,
                color=color, lw=lw, ls=ls,
                marker='o', markersize=ms,
                label=metric_label, zorder=3 if is_primary else 2)

        # Annotate each point with its value
        for xi, val in zip(x, vals):
            ax.text(xi, val + 0.008, f'{val:.3f}',
                    ha='center', fontsize=7.5,
                    color=color, fontweight='bold' if is_primary else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Performance Comparison of ML Models — All Metrics\n"
        "Solid lines = primary metrics (AUPRC, Recall) | "
        "Dashed = secondary | *Accuracy misleading on imbalanced data",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/model_metrics_lineplot.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Metrics line plot saved.")
    gc.collect()


plot_metrics_lineplot(ALL_METRICS)


def plot_model_heatmap(ALL_METRICS):
    """
    Heatmap of all key metrics across all models.
    Rows = models, Columns = metrics.
    Color intensity = performance (darker = better).
    Instantly shows which model dominates which metric.
    """
    metrics_to_show = [
        ('test_avg_prec',  'AUPRC'),
        ('test_recall',    'Recall'),
        ('test_f1',        'F1-Score'),
        ('test_auc_roc',   'AUC-ROC'),
        ('test_precision', 'Precision'),
        ('test_accuracy',  'Accuracy*'),
    ]

    model_names  = list(ALL_METRICS.keys())
    metric_labels = [m[1] for m in metrics_to_show]

    # Build matrix — rows=models, cols=metrics
    matrix = []
    for model_name in model_names:
        row = []
        for metric_key, _ in metrics_to_show:
            val = ALL_METRICS[model_name].get(metric_key, 0.0)
            row.append(float(val))
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Annotate every cell with its value
    for i in range(len(model_names)):
        for j in range(len(metric_labels)):
            val  = matrix[i, j]
            # Dark text on light cells, white text on dark cells
            text_color = 'black' if val < 0.6 else 'white'
            ax.text(j, i, f'{val:.4f}',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color)

    # Axes
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax.set_yticklabels(model_names,   fontsize=11, fontweight='bold')

    # Highlight primary metrics (AUPRC, Recall) with box
    for j, (_, label) in enumerate(metrics_to_show):
        if label in ('AUPRC', 'Recall'):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, -0.5), 1, len(model_names),
                fill=False, edgecolor='black', linewidth=2.5
            ))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Score', fontsize=10)

    ax.set_title(
        "Model Performance Heatmap — All Metrics\n"
        "Black border = primary metrics (AUPRC, Recall) per TA instruction | "
        "*Accuracy is misleading on imbalanced data",
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/models_comparison_heatmap.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Model heatmap saved.")
    gc.collect()


plot_model_heatmap(ALL_METRICS)


def plot_fp_fn_tradeoff(ALL_METRICS, y_test):
    """
    False Positive vs False Negative bar chart — all models side by side.
    FN = missed fraud (critical — real financial loss)
    FP = false alarm (customer friction — less critical)
    Directly answers: which model is safer for deployment?
    Lower FN = better for fraud detection at cost of higher FP.
    """
    y_test_arr = np.array(y_test, dtype=int)
    model_names, fp_counts, fn_counts = [], [], []

    for model_name, metrics in ALL_METRICS.items():
        y_pred = metrics.get('y_pred')
        if y_pred is None:
            continue
        y_pred = np.array(y_pred, dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred).ravel()
        model_names.append(model_name)
        fp_counts.append(int(fp))
        fn_counts.append(int(fn))

    x     = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── LEFT: Raw counts
    bars_fp = axes[0].bar(x - width/2, fp_counts, width,
                          color=C['orange'], edgecolor='white',
                          linewidth=1.2, label='False Positives (legit flagged as fraud)')
    bars_fn = axes[0].bar(x + width/2, fn_counts, width,
                          color=C['red'], edgecolor='white',
                          linewidth=1.2, label='False Negatives (fraud missed)')

    for bar, val in zip(bars_fp, fp_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(fp_counts) * 0.01,
                     f'{val:,}', ha='center', fontsize=9, fontweight='bold',
                     color=C['orange'])
    for bar, val in zip(bars_fn, fn_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(fp_counts) * 0.01,
                     f'{val:,}', ha='center', fontsize=9, fontweight='bold',
                     color=C['red'])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    axes[0].set_title("FP vs FN — Raw Counts\n(Lower FN = fewer missed frauds)",
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{int(val):,}')
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ── RIGHT: FN as % of total actual fraud — more meaningful
    total_fraud = int(y_test_arr.sum())
    fn_pct      = [fn / total_fraud * 100 for fn in fn_counts]
    fp_total    = int((y_test_arr == 0).sum())
    fp_pct      = [fp / fp_total * 100 for fp in fp_counts]

    bars_fp2 = axes[1].bar(x - width/2, fp_pct, width,
                           color=C['orange'], edgecolor='white',
                           linewidth=1.2, label='FP Rate (% of legit flagged)')
    bars_fn2 = axes[1].bar(x + width/2, fn_pct, width,
                           color=C['red'], edgecolor='white',
                           linewidth=1.2, label='FN Rate (% of fraud missed)')

    for bar, val in zip(bars_fp2, fp_pct):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{val:.2f}%', ha='center', fontsize=9,
                     fontweight='bold', color=C['orange'])
    for bar, val in zip(bars_fn2, fn_pct):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{val:.2f}%', ha='center', fontsize=9,
                     fontweight='bold', color=C['red'])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    axes[1].set_title(
        "FP vs FN — As % of Actual Class\n"
        "FN% = fraud missed rate (critical) | FP% = false alarm rate",
        fontsize=12, fontweight='bold'
    )
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"False Positive vs False Negative Tradeoff — All Models\n"
        f"Total fraud in test set: {total_fraud:,} | "
        f"Total legit: {fp_total:,} | FN is more costly than FP in fraud detection",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("outputs/plots/fp_fn_tradeoff.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("FP vs FN tradeoff chart saved.")
    gc.collect()

plot_fp_fn_tradeoff(ALL_METRICS, y_ts)



def plot_cost_benefit_analysis(ALL_METRICS, y_test, avg_fraud_amount=500):
    """
    Cost-Benefit Analysis — assigns financial cost to FP and FN.
    FN cost = avg fraud transaction amount (full loss — fraud goes undetected)
    FP cost = fixed investigation cost (customer friction, manual review)
    Shows which model minimizes total financial loss.
    avg_fraud_amount: average fraud transaction in your dataset (adjust if known)
    """
    y_test_arr = np.array(y_test, dtype=int)

    # Cost assumptions — adjust based on domain knowledge
    FN_COST = avg_fraud_amount  # cost of missing one fraud = full transaction lost
    FP_COST = 10                # cost of false alarm = manual review / customer friction

    model_names  = []
    total_costs  = []
    fn_costs     = []
    fp_costs     = []

    for model_name, metrics in ALL_METRICS.items():
        y_pred = metrics.get('y_pred')
        if y_pred is None:
            continue
        y_pred = np.array(y_pred, dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred).ravel()

        fn_cost    = fn * FN_COST
        fp_cost    = fp * FP_COST
        total_cost = fn_cost + fp_cost

        model_names.append(model_name)
        fn_costs.append(fn_cost)
        fp_costs.append(fp_cost)
        total_costs.append(total_cost)

    x     = np.arange(len(model_names))
    width = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── LEFT: Stacked bar — FN cost + FP cost = total cost per model
    bars_fn = axes[0].bar(x, fn_costs, width,
                          color=C['red'], edgecolor='white',
                          linewidth=1.2, label=f'FN Cost (${FN_COST}/missed fraud)')
    bars_fp = axes[0].bar(x, fp_costs, width,
                          color=C['orange'], edgecolor='white',
                          linewidth=1.2, label=f'FP Cost (${FP_COST}/false alarm)',
                          bottom=fn_costs)

    # Annotate total cost on top
    for i, (fn_c, fp_c, total) in enumerate(zip(fn_costs, fp_costs, total_costs)):
        axes[0].text(i, total + max(total_costs) * 0.01,
                     f'${total:,.0f}',
                     ha='center', fontsize=9, fontweight='bold')

    # Highlight best (lowest cost) model
    best_idx = int(np.argmin(total_costs))
    axes[0].patches[best_idx].set_edgecolor('black')
    axes[0].patches[best_idx].set_linewidth(2.5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    axes[0].set_title(
        f"Total Financial Cost per Model\n"
        f"FN=${FN_COST}/fraud missed | FP=${FP_COST}/false alarm | "
        f"Black border = lowest cost",
        fontsize=11, fontweight='bold'
    )
    axes[0].set_ylabel("Estimated Total Cost ($)", fontsize=11)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'${int(val):,}')
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ── RIGHT: Cost breakdown % — what fraction is FN vs FP per model
    fn_pct = [fn / tot * 100 if tot > 0 else 0 for fn, tot in zip(fn_costs, total_costs)]
    fp_pct = [fp / tot * 100 if tot > 0 else 0 for fp, tot in zip(fp_costs, total_costs)]

    axes[1].bar(x, fn_pct, width, color=C['red'],    edgecolor='white',
                linewidth=1.2, label='FN Cost %')
    axes[1].bar(x, fp_pct, width, color=C['orange'], edgecolor='white',
                linewidth=1.2, label='FP Cost %', bottom=fn_pct)

    for i, (fn_p, fp_p) in enumerate(zip(fn_pct, fp_pct)):
        axes[1].text(i, fn_p / 2,        f'{fn_p:.1f}%',
                     ha='center', fontsize=9, fontweight='bold', color='white')
        axes[1].text(i, fn_p + fp_p / 2, f'{fp_p:.1f}%',
                     ha='center', fontsize=9, fontweight='bold', color='black')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    axes[1].set_title(
        "Cost Breakdown — FN vs FP Contribution (%)\n"
        "FN dominates = model missing too much fraud",
        fontsize=11, fontweight='bold'
    )
    axes[1].set_ylabel("Cost Contribution (%)", fontsize=11)
    axes[1].set_ylim(0, 115)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        "Cost-Benefit Analysis — Financial Impact of Model Errors\n"
        "Assumptions: avg fraud = $500 | false alarm = $10 | "
        "adjust avg_fraud_amount to match your dataset",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("outputs/plots/cost_benefit_analysis.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Cost-benefit analysis saved.")
    gc.collect()

plot_cost_benefit_analysis(ALL_METRICS, y_ts, avg_fraud_amount=500)


def plot_classification_report_heatmap(ALL_METRICS, y_test):
    """
    Heatmap of full classification report per model.
    Rows = models, Columns = class-level metrics.
    Breaks down performance by class 0 (legit) and class 1 (fraud) separately.
    Critical for imbalanced data — aggregate metrics hide per-class failures.
    """
    y_test_arr = np.array(y_test, dtype=int)

    # Metrics to extract per class
    metric_cols = [
        ('0', 'precision', 'Legit\nPrecision'),
        ('0', 'recall',    'Legit\nRecall'),
        ('0', 'f1-score',  'Legit\nF1'),
        ('1', 'precision', 'Fraud\nPrecision'),
        ('1', 'recall',    'Fraud\nRecall'),
        ('1', 'f1-score',  'Fraud\nF1'),
        ('macro avg', 'precision', 'Macro\nPrecision'),
        ('macro avg', 'recall',    'Macro\nRecall'),
        ('macro avg', 'f1-score',  'Macro\nF1'),
    ]

    model_names  = []
    matrix_rows  = []

    for model_name, metrics in ALL_METRICS.items():
        y_pred = metrics.get('y_pred')
        if y_pred is None:
            continue

        y_pred = np.array(y_pred, dtype=int)
        report = classification_report(
            y_test_arr, y_pred,
            output_dict=True, zero_division=0
        )

        row = []
        for cls, metric_key, _ in metric_cols:
            cls_report = report.get(cls, report.get(int(cls) if cls.isdigit() else cls, {}))
            val = float(cls_report.get(metric_key, 0.0)) if isinstance(cls_report, dict) else 0.0
            row.append(val)

        model_names.append(model_name)
        matrix_rows.append(row)

    matrix      = np.array(matrix_rows)
    col_labels  = [c[2] for c in metric_cols]

    fig, ax = plt.subplots(figsize=(16, max(4, len(model_names) * 1.2)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(col_labels)):
            val        = matrix[i, j]
            text_color = 'black' if 0.3 < val < 0.8 else 'white'
            ax.text(j, i, f'{val:.4f}',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color=text_color)

    # Separate fraud vs legit columns with vertical lines
    ax.axvline(x=2.5,  color='black', lw=2.0)  # legit | fraud
    ax.axvline(x=5.5,  color='black', lw=2.0)  # fraud | macro

    # Section labels above columns
    ax.text(1.0,  -0.7, 'Legitimate (Class 0)', ha='center', fontsize=11,
            fontweight='bold', color=C['green'],
            transform=ax.get_xaxis_transform())
    ax.text(4.0,  -0.7, 'Fraud (Class 1)',       ha='center', fontsize=11,
            fontweight='bold', color=C['red'],
            transform=ax.get_xaxis_transform())
    ax.text(7.0,  -0.7, 'Macro Average',          ha='center', fontsize=11,
            fontweight='bold', color=C['blue'],
            transform=ax.get_xaxis_transform())

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(col_labels,   fontsize=10, fontweight='bold')
    ax.set_yticklabels(model_names,  fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Score', fontsize=10)

    ax.set_title(
        "Classification Report Heatmap — Per-Class Breakdown\n"
        "Green = high performance | Red = low | "
        "Fraud column is primary — legitimate column should remain high",
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/classification_report_heatmap.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Classification report heatmap saved.")
    gc.collect()


plot_classification_report_heatmap(ALL_METRICS, y_ts)


# MODEL COMPARISON SUMMARY
# ── Four models evaluated: Logistic Regression, Random Forest,
#    XGBoost, Neural Network (MLP).
# ── Primary metrics: AUPRC and Recall — accuracy suppressed due to
#    severe class imbalance (0.13% fraud in test set).
# ── XGBoost expected to lead on AUPRC and F1 — strongest model for
#    tabular fraud detection with non-linear feature interactions.
# ── Logistic Regression serves as linear baseline — performance gap
#    vs tree models quantifies the value of non-linearity.
# ── Random Forest included for ensemble bagging comparison vs
#    XGBoost boosting — different variance reduction strategies.
# ── Neural Network (MLP) included to test deep representation
#    learning on tabular data — typically underperforms trees here.
# ── Bias check confirms whether models generalize across transaction
#    types or overfit to dominant categories (CASH_OUT, TRANSFER).
