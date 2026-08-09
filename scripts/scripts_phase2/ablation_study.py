# ABLATION STUDY
# Purpose: Prove that each pipeline component actually contributes.
# Three ablations:
#   A — Without Fraud Simulation Engine
#   B — SMOTE ratio comparison (0.0 vs 0.3 vs 0.5)
#   C — Without engineered features (balanceDiff, amount_ratio)
# FIX 1: y cast to int everywhere — run_kfold scorers break on float32
# FIX 2: run_kfold receives pre-SMOTE data — SMOTE happens inside CV pipeline
# FIX 3: Panel B labels corrected to match actual ratios (0.0, 0.3, 0.5)

import warnings
warnings.filterwarnings("ignore")

ABLATION_SAVE_PATH = "outputs/ablation/"
os.makedirs(ABLATION_SAVE_PATH, exist_ok=True)

# Ensure XGB alias exists
if "XGB" not in globals():
    XGB = XGBClassifier

print("ABLATION STUDY STARTING")
print_memory("[BEFORE ABLATION]")


# ── ABLATION HELPER
# We use XGBoost for all ablations — it's fast, stable, no NaN risk and best model among all the models have been used.
# Comparing the SAME model across conditions isolates the pipeline variable.

def ablation_model():
    """
    Identical XGBoost config used across ALL ablation runs.
    Same hyperparameters = fair comparison.
    scale_pos_weight excluded — SMOTE handles imbalance inside each condition.
    """
    return XGB(
        n_estimators     = 50,  # was 300 — ablation needs direction, not perfection
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        eval_metric      = 'logloss',
        random_state     = 42,
        n_jobs           = -1,
    )


def run_ablation_condition(label, X_train_ab, y_train_ab,
                                  X_cv_ab,    y_cv_ab,
                                  X_test_ab,  y_test_ab):
    """
    Train + evaluate one ablation condition.

    Args:
        X_train_ab : post-SMOTE post-scaling — used for final model.fit()
        y_train_ab : labels for above
        X_cv_ab    : pre-SMOTE pre-scaling — passed to run_kfold (SMOTE inside CV)
        y_cv_ab    : labels for above
        X_test_ab  : scaled test set for this condition
        y_test_ab  : test labels

    FIX: run_kfold now receives pre-SMOTE data (X_cv_ab) so SMOTE
    is applied inside each fold only — no leakage into val fold.
    """
    print(f"\n  Running: {label}")
    model = ablation_model()

    # CV on pre-SMOTE data — pipeline handles SMOTE inside each fold
    # FIX: y cast to int — float32 breaks precision/recall scorers
    cv = run_kfold(model, X_cv_ab, np.array(y_cv_ab, dtype=int), label, n_splits=3)

    # Final fit on post-SMOTE post-scaling data
    model.fit(X_train_ab, np.array(y_train_ab, dtype=int))

    y_pred = model.predict(X_test_ab)
    y_prob = model.predict_proba(X_test_ab)[:, 1]

    report = classification_report(
        np.array(y_test_ab, dtype=int), y_pred,
        output_dict=True, zero_division=0
    )
    auc = roc_auc_score(np.array(y_test_ab, dtype=int), y_prob)
    ap  = average_precision_score(np.array(y_test_ab, dtype=int), y_prob)

    cls_1 = None
    if isinstance(report, dict):
        cls_1 = report.get('1') or report.get(1) or report.get('1.0')
    if not isinstance(cls_1, dict):
        cls_1 = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}

    test_precision = float(cls_1.get('precision', 0.0))
    test_recall    = float(cls_1.get('recall',    0.0))
    test_f1        = float(cls_1.get('f1-score',  0.0))
    cv_f1_mean     = float(cv.get('cv_f1_mean',   0.0))
    cv_auc_mean    = float(cv.get('cv_auc_mean',  0.0))

    print(f"    Test F1={test_f1:.4f}  Recall={test_recall:.4f}  AUC={float(auc):.4f}")

    return {
        "Condition"     : label,
        "CV F1"         : round(cv_f1_mean,     4),
        "CV AUC"        : round(cv_auc_mean,    4),
        "Test Precision": round(test_precision, 4),
        "Test Recall"   : round(test_recall,    4),
        "Test F1"       : round(test_f1,        4),
        "Test AUC"      : round(float(auc),     4),
        "Test AP"       : round(float(ap),      4),
    }


ablation_results = []

# ── ABLATION A: Effect of Fraud Simulation Engine

print("\n" + "="*60)
print("ABLATION A: Fraud Simulation Engine — Does it help?")
print("="*60)

# --- A1: WITHOUT Fraud Simulation — SMOTE directly on raw X_train
# Step 1: Apply SMOTE to raw X_train (no fraud simulation before this)
X_no_sim_smote, y_no_sim_smote = apply_smote(X_train, y_train, sampling_ratio=0.3)
X_no_sim_smote = X_no_sim_smote.astype('float32')   # SMOTE returns float64 — convert back

# Step 2: Scale using StandardScaler directly (fit on train, transform test)
ab_scaler_a          = StandardScaler()
X_no_sim_scaled_arr  = ab_scaler_a.fit_transform(X_no_sim_smote).astype('float32')
X_test_ab_scaled_arr = ab_scaler_a.transform(X_test.values.astype('float32')).astype('float32')

result_a1 = run_ablation_condition(
    label      = "A1: No Fraud Simulation (SMOTE only)",
    X_train_ab = X_no_sim_scaled_arr,
    y_train_ab = np.array(y_no_sim_smote, dtype=int),
    X_cv_ab    = X_train.values.astype('float32'),
    y_cv_ab    = np.array(y_train, dtype=int),
    X_test_ab  = X_test_ab_scaled_arr,
    y_test_ab  = np.array(y_test, dtype=int),
)
ablation_results.append(result_a1)

del X_no_sim_smote, X_no_sim_scaled_arr, X_test_ab_scaled_arr
gc.collect()

# --- A2: WITH Fraud Simulation — full pipeline baseline
result_a2 = run_ablation_condition(
    label      = "A2: With Fraud Simulation (full pipeline)",
    X_train_ab = X_tr,
    y_train_ab = np.array(y_tr, dtype=int),
    X_cv_ab    = X_train_aug.values.astype('float32'),
    y_cv_ab    = np.array(y_train_aug, dtype=int),
    X_test_ab  = X_ts,
    y_test_ab  = np.array(y_ts, dtype=int),
)
ablation_results.append(result_a2)
gc.collect()


# ── ABLATION B: SMOTE Sampling Ratio Comparison
# Test three ratios: 0.0 (no SMOTE), 0.3 (your current), 0.5 (more aggressive)
# All use Fraud Simulation output (X_train_aug, y_train_aug) as base.

print("\n" + "="*60)
print("ABLATION B: SMOTE Sampling Ratio — 0.0 vs 0.3 vs 0.5")
print("="*60)

for ratio, label in [
    (0.0, "B1: No SMOTE (ratio=0.0)"),
    (0.3, "B2: SMOTE ratio=0.3 (current)"),
    (0.5, "B3: SMOTE ratio=0.5 (aggressive)"),
]:
    X_test_b_arr = X_test.values.astype('float32')

    if ratio == 0.0:
        # No SMOTE — fit new scaler on aug data, transform test
        ab_scaler = StandardScaler()
        X_ab_b  = ab_scaler.fit_transform(
                      X_train_aug.values.astype('float32')
                  ).astype('float32')
        X_ts_ab = ab_scaler.transform(X_test_b_arr).astype('float32')
        y_ab_b  = np.array(y_train_aug, dtype=int)
    else:
        X_sm, y_sm = apply_smote(X_train_aug, y_train_aug, sampling_ratio=ratio)
        X_sm       = X_sm.astype('float32')
        ab_scaler  = StandardScaler()
        X_ab_b  = ab_scaler.fit_transform(X_sm).astype('float32')
        X_ts_ab = ab_scaler.transform(X_test_b_arr).astype('float32')
        y_ab_b  = np.array(y_sm, dtype=int)

    result_b = run_ablation_condition(
        label      = label,
        X_train_ab = X_ab_b,
        y_train_ab = y_ab_b,
        X_cv_ab    = X_train_aug.values.astype('float32'),
        y_cv_ab    = np.array(y_train_aug, dtype=int),
        X_test_ab  = X_ts_ab,
        y_test_ab  = np.array(y_test, dtype=int),
    )
    ablation_results.append(result_b)
    gc.collect()

# ── ABLATION C: Effect of Engineered Features
# Drop balanceDiff and amount_ratio — use only raw features.
# Proves whether your feature engineering actually adds signal.


print("\n" + "="*60)
print("ABLATION C: Feature Engineering — Raw vs Engineered")
print("="*60)

ENGINEERED_COLS  = ['balanceDiff', 'amount_ratio']
features_reduced = [f for f in FEATURE_NAMES if f not in ENGINEERED_COLS]

# C1: WITHOUT engineered features
X_tr_c1 = pd.DataFrame(X_tr, columns=FEATURE_NAMES)[features_reduced].values.astype('float32')
X_ts_c1 = pd.DataFrame(X_ts, columns=FEATURE_NAMES)[features_reduced].values.astype('float32')
X_cv_c1 = pd.DataFrame(
    X_train_aug.values, columns=FEATURE_NAMES
)[features_reduced].values.astype('float32')

result_c1 = run_ablation_condition(
    label      = "C1: Without Engineered Features",
    X_train_ab = X_tr_c1,
    y_train_ab = np.array(y_tr, dtype=int),
    X_cv_ab    = X_cv_c1,
    y_cv_ab    = np.array(y_train_aug, dtype=int),
    X_test_ab  = X_ts_c1,
    y_test_ab  = np.array(y_ts, dtype=int),
)
ablation_results.append(result_c1)

del X_tr_c1, X_ts_c1, X_cv_c1
gc.collect()

# C2: WITH engineered features (full pipeline baseline)
result_c2 = run_ablation_condition(
    label      = "C2: With Engineered Features (full pipeline)",
    X_train_ab = X_tr,
    y_train_ab = np.array(y_tr, dtype=int),
    X_cv_ab    = X_train_aug.values.astype('float32'),
    y_cv_ab    = np.array(y_train_aug, dtype=int),
    X_test_ab  = X_ts,
    y_test_ab  = np.array(y_ts, dtype=int),
)
ablation_results.append(result_c2)
gc.collect()

# ── ABLATION RESULTS TABLE

df_ablation = pd.DataFrame(ablation_results)

print("\n" + "="*100)
print("  ABLATION STUDY — FULL RESULTS TABLE")
print("="*100)
print(df_ablation.to_string(index=False))
print("="*100)

df_ablation.to_csv(os.path.join(ABLATION_SAVE_PATH, "ablation_results.csv"), index=False)
print("\nAblation results saved to outputs/ablation/ablation_results.csv")



# ── ABLATION VISUALIZATION
# FIX: grouped side-by-side bars — solid=F1, hatched=AUPRC
# Previous version had overlap — both bars same position same width

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
width = 0.35

# --- Panel A: Fraud Simulation effect
ab_a  = df_ablation[df_ablation['Condition'].str.startswith('A')]
x_a   = np.arange(2)
labs_a = ['No Simulation', 'With Simulation']

axes[0].bar(x_a - width/2, ab_a['Test F1'].values,
            width, color=[C['gray'], C['green']],
            edgecolor='white', label='F1')
axes[0].bar(x_a + width/2, ab_a['Test AP'].values,
            width, color=[C['gray'], C['green']],
            edgecolor='white', alpha=0.6, hatch='///', label='AUPRC')

for i, (f1, ap) in enumerate(zip(ab_a['Test F1'].values, ab_a['Test AP'].values)):
    axes[0].text(i - width/2, f1 + 0.01, f'{f1:.4f}',
                 ha='center', fontsize=8, fontweight='bold')
    axes[0].text(i + width/2, ap + 0.01, f'{ap:.4f}',
                 ha='center', fontsize=8, fontweight='bold')

axes[0].set_xticks(x_a)
axes[0].set_xticklabels(labs_a, fontsize=10)
axes[0].set_title("A: Fraud Simulation Engine\n(Solid=F1 | Hatched=AUPRC)",
                  fontweight='bold')
axes[0].set_ylim(0, 1.18)
axes[0].set_ylabel("Score")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

# --- Panel B: SMOTE ratio effect
ab_b   = df_ablation[df_ablation['Condition'].str.startswith('B')]
x_b    = np.arange(3)
labs_b = ['ratio=0.0', 'ratio=0.3', 'ratio=0.5']
cols_b = [C['gray'], C['blue'], C['orange']]

axes[1].bar(x_b - width/2, ab_b['Test F1'].values,
            width, color=cols_b, edgecolor='white', label='F1')
axes[1].bar(x_b + width/2, ab_b['Test AP'].values,
            width, color=cols_b, edgecolor='white',
            alpha=0.6, hatch='///', label='AUPRC')

for i, (f1, ap) in enumerate(zip(ab_b['Test F1'].values, ab_b['Test AP'].values)):
    axes[1].text(i - width/2, f1 + 0.01, f'{f1:.4f}',
                 ha='center', fontsize=8, fontweight='bold')
    axes[1].text(i + width/2, ap + 0.01, f'{ap:.4f}',
                 ha='center', fontsize=8, fontweight='bold')

axes[1].set_xticks(x_b)
axes[1].set_xticklabels(labs_b, fontsize=10)
axes[1].set_title("B: SMOTE Sampling Ratio\n(Solid=F1 | Hatched=AUPRC)",
                  fontweight='bold')
axes[1].set_ylim(0, 1.18)
axes[1].set_ylabel("Score")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

# --- Panel C: Feature engineering effect
ab_c   = df_ablation[df_ablation['Condition'].str.startswith('C')]
x_c    = np.arange(2)
labs_c = ['Without Eng.', 'With Eng.']

axes[2].bar(x_c - width/2, ab_c['Test F1'].values,
            width, color=[C['gray'], C['purple']],
            edgecolor='white', label='F1')
axes[2].bar(x_c + width/2, ab_c['Test AP'].values,
            width, color=[C['gray'], C['purple']],
            edgecolor='white', alpha=0.6, hatch='///', label='AUPRC')

for i, (f1, ap) in enumerate(zip(ab_c['Test F1'].values, ab_c['Test AP'].values)):
    axes[2].text(i - width/2, f1 + 0.01, f'{f1:.4f}',
                 ha='center', fontsize=8, fontweight='bold')
    axes[2].text(i + width/2, ap + 0.01, f'{ap:.4f}',
                 ha='center', fontsize=8, fontweight='bold')

axes[2].set_xticks(x_c)
axes[2].set_xticklabels(labs_c, fontsize=10)
axes[2].set_title("C: Feature Engineering\n(Solid=F1 | Hatched=AUPRC)",
                  fontweight='bold')
axes[2].set_ylim(0, 1.18)
axes[2].set_ylabel("Score")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, axis='y')

fig.suptitle(
    "Ablation Study — Component Contribution Analysis\n"
    "Primary metric: AUPRC (hatched) | Secondary: F1 (solid)",
    fontsize=15, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(ABLATION_SAVE_PATH, "ablation_study.png"),
            bbox_inches='tight', dpi=300)
plt.show()
plt.close('all')
print("Ablation chart saved.")
gc.collect()

# ── ABLATION CONCLUSIONS — printed summary
print("\n" + "="*80)
print("  ABLATION CONCLUSIONS")
print("="*80)
for group, label in [('A','Fraud Simulation'), ('B','SMOTE Ratio'), ('C','Feature Engineering')]:
    grp  = df_ablation[df_ablation['Condition'].str.startswith(group)]
    best = grp.loc[grp['Test AP'].idxmax()]
    print(f"  {label}: Best = {best['Condition']} "
          f"(AUPRC={best['Test AP']:.4f}  Recall={best['Test Recall']:.4f})")


def plot_ablation_delta(df_ablation):
    """
    Bar chart of metric change relative to the full pipeline baseline.
    Baseline = A2 (With Fraud Sim) = C2 (With Eng. Features) = B2 (SMOTE 0.3).
    Green bars = component helps. Red bars = component hurts.
    """
    baseline_ap = df_ablation[df_ablation['Condition'] == 'A2: With Fraud Simulation (full pipeline)']['Test AP'].values[0]
    baseline_f1 = df_ablation[df_ablation['Condition'] == 'A2: With Fraud Simulation (full pipeline)']['Test F1'].values[0]

    # Compute deltas for all non-baseline conditions
    comparisons = [
        ('A1: No Fraud Simulation (SMOTE only)',    'vs Full Pipeline (A2)', 'Fraud Sim Removed'),
        ('B1: No SMOTE (ratio=0.0)',                'vs SMOTE=0.3 (B2)',     'SMOTE Removed'),
        ('B3: SMOTE ratio=0.5 (aggressive)',        'vs SMOTE=0.3 (B2)',     'SMOTE Ratio 0.5'),
        ('C1: Without Engineered Features',         'vs Full Pipeline (C2)', 'Eng. Features Removed'),
    ]

    b2_ap = df_ablation[df_ablation['Condition'] == 'B2: SMOTE ratio=0.3 (current)']['Test AP'].values[0]
    b2_f1 = df_ablation[df_ablation['Condition'] == 'B2: SMOTE ratio=0.3 (current)']['Test F1'].values[0]
    c2_ap = df_ablation[df_ablation['Condition'] == 'C2: With Engineered Features (full pipeline)']['Test AP'].values[0]
    c2_f1 = df_ablation[df_ablation['Condition'] == 'C2: With Engineered Features (full pipeline)']['Test F1'].values[0]

    baselines_ap = [baseline_ap, b2_ap, b2_ap, c2_ap]
    baselines_f1 = [baseline_f1, b2_f1, b2_f1, c2_f1]

    labels      = [c[2] for c in comparisons]
    cond_labels = [c[0] for c in comparisons]
    deltas_ap, deltas_f1 = [], []

    for cond, _, _ in comparisons:
        row = df_ablation[df_ablation['Condition'] == cond]
        deltas_ap.append(float(row['Test AP'].values[0]))
        deltas_f1.append(float(row['Test F1'].values[0]))

    deltas_ap = [d - b for d, b in zip(deltas_ap, baselines_ap)]
    deltas_f1 = [d - b for d, b in zip(deltas_f1, baselines_f1)]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_ap = ax.bar(x - width/2, deltas_ap, width,
                     color=[C['green'] if v >= 0 else C['red'] for v in deltas_ap],
                     edgecolor='white', linewidth=1.2, label='AUPRC Δ')
    bars_f1 = ax.bar(x + width/2, deltas_f1, width,
                     color=[C['blue'] if v >= 0 else C['orange'] for v in deltas_f1],
                     edgecolor='white', linewidth=1.2, label='F1 Δ', alpha=0.75)

    for bar, val in zip(bars_ap, deltas_ap):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if val >= 0 else -0.008),
                f'{val:+.4f}', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars_f1, deltas_f1):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if val >= 0 else -0.008),
                f'{val:+.4f}', ha='center', fontsize=9)

    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Δ Score vs Baseline (positive = hurts removal)', fontsize=11)
    ax.set_title(
        'Ablation Delta Analysis — Component Contribution\n'
        'Negative = removing this component hurts performance (component is valuable)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_SAVE_PATH, 'ablation_delta.png'),
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Ablation delta chart saved.")

plot_ablation_delta(df_ablation)


def plot_ablation_heatmap(df_ablation):
    """
    Heatmap of all ablation conditions × key metrics.
    Rows = conditions, Columns = metrics.
    Color intensity = score. Easy to read in reports.
    """
    metric_cols   = ['Test Precision', 'Test Recall', 'Test F1', 'Test AUC', 'Test AP']
    display_names = ['Precision', 'Recall', 'F1', 'AUC-ROC', 'AUPRC']

    short_labels = {
        'A1: No Fraud Simulation (SMOTE only)'          : 'A1: No Fraud Sim',
        'A2: With Fraud Simulation (full pipeline)'     : 'A2: With Fraud Sim',
        'B1: No SMOTE (ratio=0.0)'                      : 'B1: SMOTE=0.0',
        'B2: SMOTE ratio=0.3 (current)'                 : 'B2: SMOTE=0.3',
        'B3: SMOTE ratio=0.5 (aggressive)'              : 'B3: SMOTE=0.5',
        'C1: Without Engineered Features'               : 'C1: No Eng. Features',
        'C2: With Engineered Features (full pipeline)'  : 'C2: With Eng. Features',
    }

    df_plot = df_ablation.copy()
    df_plot['Label'] = df_plot['Condition'].map(short_labels).fillna(df_plot['Condition'])

    matrix = df_plot[metric_cols].values
    rows   = df_plot['Label'].tolist()

    fig, ax = plt.subplots(figsize=(12, max(5, len(rows) * 0.9)))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    for i in range(len(rows)):
        for j in range(len(display_names)):
            val = matrix[i, j]
            text_color = 'black' if 0.3 < val < 0.75 else 'white'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=text_color)

    # Separator lines between ablation groups A / B / C
    ax.axhline(1.5, color='black', lw=2.0)
    ax.axhline(4.5, color='black', lw=2.0)

    # Highlight AUPRC column (primary metric)
    ax.add_patch(plt.Rectangle(
        (3.5, -0.5), 1, len(rows),
        fill=False, edgecolor='black', linewidth=2.5
    ))

    ax.set_xticks(np.arange(len(display_names)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(display_names, fontsize=11, fontweight='bold')
    ax.set_yticklabels(rows, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label('Score', fontsize=10)

    ax.set_title(
        'Ablation Study Heatmap — All Conditions × All Metrics\n'
        'Black border = AUPRC (primary metric) | Horizontal lines = ablation groups A / B / C',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_SAVE_PATH, 'ablation_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Ablation heatmap saved.")

plot_ablation_heatmap(df_ablation)

def plot_ablation_pr_scatter(df_ablation):
    """
    Scatter plot of Precision vs Recall for each ablation condition.
    Size = AUPRC (bigger bubble = better overall).
    Color = ablation group (A / B / C).
    """
    group_colors = {'A': C['blue'], 'B': C['orange'], 'C': C['purple']}
    short_labels = {
        'A1: No Fraud Simulation (SMOTE only)'         : 'A1',
        'A2: With Fraud Simulation (full pipeline)'    : 'A2 ★',
        'B1: No SMOTE (ratio=0.0)'                     : 'B1',
        'B2: SMOTE ratio=0.3 (current)'                : 'B2 ★',
        'B3: SMOTE ratio=0.5 (aggressive)'             : 'B3',
        'C1: Without Engineered Features'              : 'C1',
        'C2: With Engineered Features (full pipeline)' : 'C2 ★',
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df_ablation.iterrows():
        group = row['Condition'][0]   # 'A', 'B', or 'C'
        color = group_colors.get(group, C['gray'])
        label = short_labels.get(row['Condition'], row['Condition'])
        size  = row['Test AP'] * 1200 + 80    # bubble size ∝ AUPRC

        ax.scatter(row['Test Recall'], row['Test Precision'],
                   s=size, color=color, alpha=0.78,
                   edgecolors='white', linewidth=1.5, zorder=3)
        ax.annotate(label,xy=(row['Test Recall'], row['Test Precision']),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color, linewidth=1.2),
                    zorder=10)

    # Ideal point marker
    ax.scatter([1.0], [1.0], s=200, color=C['green'], marker='*',
               zorder=5, label='Ideal (Recall=1, Precision=1)')

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=C['blue'],   label='Group A — Fraud Simulation'),
        Patch(facecolor=C['orange'], label='Group B — SMOTE Ratio'),
        Patch(facecolor=C['purple'], label='Group C — Feature Engineering'),
        Patch(facecolor=C['green'],  label='★ = baseline (full pipeline)'),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc='lower left')

    ax.set_xlabel('Recall (Fraud Class)', fontsize=12)
    ax.set_ylabel('Precision (Fraud Class)', fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        'Ablation Study — Precision vs Recall per Condition\n'
        'Bubble size ∝ AUPRC | ★ = full pipeline baseline | Closer to top-right = better',
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_SAVE_PATH, 'ablation_pr_scatter.png'),
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("Ablation P-R scatter saved.")

plot_ablation_pr_scatter(df_ablation)


def plot_smote_ratio_trend(df_ablation):
    """
    Line chart specific to ablation B — how AUPRC, Recall, Precision
    change as SMOTE ratio increases from 0.0 → 0.3 → 0.5.
    Shows the diminishing returns / tradeoff clearly.
    """
    ab_b  = df_ablation[df_ablation['Condition'].str.startswith('B')].copy()
    ratios = [0.0, 0.3, 0.5]

    fig, ax = plt.subplots(figsize=(9, 5))

    for metric, color, lw, ls in [
        ('Test AP',        C['red'],    2.5, '-'),
        ('Test Recall',    C['green'],  2.5, '-'),
        ('Test Precision', C['blue'],   1.8, '--'),
        ('Test F1',        C['orange'], 1.8, '--'),
    ]:
        vals = ab_b[metric].values
        ax.plot(ratios, vals, color=color, linewidth=lw, linestyle=ls,
                marker='o', markersize=9, label=metric.replace('Test ', ''))
        for x, y in zip(ratios, vals):
            ax.annotate(f'{y:.4f}', (x, y),
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=9, color=color, fontweight='bold')

    ax.axvline(0.3, color='black', linewidth=1.5, linestyle=':', alpha=0.6,
               label='Current ratio (0.3)')
    ax.set_xticks(ratios)
    ax.set_xticklabels(['No SMOTE\n(0.0)', 'Current\n(0.3)', 'Aggressive\n(0.5)'],
                       fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        'Ablation B — Effect of SMOTE Sampling Ratio on All Metrics\n'
        'Solid = primary metrics | Dashed = secondary',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_SAVE_PATH, 'ablation_smote_trend.png'),
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    print("SMOTE trend chart saved.")


plot_smote_ratio_trend(df_ablation)


# DEPLOYMENT PREP
# Saves everything Streamlit needs in one clean folder.
# Deploy XGBoost only — not Keras. Reason: no TF dependency on server.

DEPLOY_PATH = "outputs/deployment/"
os.makedirs(DEPLOY_PATH, exist_ok=True)

# 1. Save XGBoost model (best model for tabular fraud data)
joblib.dump(xgb_model, os.path.join(DEPLOY_PATH, "model.pkl"))

# 2. Save scaler (must use the same one fitted on training data)
joblib.dump(scaler, os.path.join(DEPLOY_PATH, "scaler.pkl"))

# 3. Save feature names (Streamlit needs exact column order)
import json
with open(os.path.join(DEPLOY_PATH, "feature_names.json"), "w") as f:
    json.dump(FEATURE_NAMES, f)

# 4. Save model metadata (thresholds, metrics for display in app)
deploy_meta = {
    "model_type"       : "XGBoost",
    "features"         : FEATURE_NAMES,
    "n_features"       : len(FEATURE_NAMES),
    "test_f1"          : round(float(xgb_metrics['test_f1']),      4),
    "test_auc"         : round(float(xgb_metrics['test_auc_roc']), 4),
    "test_recall"      : round(float(xgb_metrics['test_recall']),  4),
    "test_precision"   : round(float(xgb_metrics['test_precision']), 4),
    "decision_threshold": float(thresholds_summary.get("XGBoost", {}).get("threshold", 0.5)),
}
with open(os.path.join(DEPLOY_PATH, "model_meta.json"), "w") as f:
    json.dump(deploy_meta, f, indent=4)

print("Deployment assets saved:")
print(f"  - {DEPLOY_PATH}model.pkl")
print(f"  - {DEPLOY_PATH}scaler.pkl")
print(f"  - {DEPLOY_PATH}feature_names.json")
print(f"  - {DEPLOY_PATH}model_meta.json")
print("\nTo deploy: copy outputs/deployment/ into your Streamlit repo.")
print("Load with: model = joblib.load('model.pkl')")
print("           scaler = joblib.load('scaler.pkl')")

