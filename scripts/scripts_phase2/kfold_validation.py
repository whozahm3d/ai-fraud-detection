# Implement 3-Fold Stratified Cross-Validation for robust performance estimation.
# FIX — Three changes from previous version:
#   1. scoring changed from list → dict using make_scorer
#      Reason: string scorer 'precision' uses zero_division='warn' → NaN when a fold
#              predicts all-negative. make_scorer lets us set zero_division=0 explicitly.
#   2. for-loop over scoring iterates dict keys — behaviour identical to before
#   3. All results[] key references unchanged — dict keys match old string names exactly

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# SMOTE leakage fix — pipeline applies SMOTE + scaling inside each fold
# Val fold always sees real data only, never synthetic samples

def run_kfold(model, X_cv, y_cv, model_name, n_splits=3):
    """
    Run Stratified K-Fold Cross-Validation (3-Fold).
    SMOTE + StandardScaler applied INSIDE each fold via ImbPipeline.
    Val fold always sees real data only — zero synthetic samples.
    """
    y_cv = np.array(y_cv, dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        'precision'         : make_scorer(precision_score, zero_division=0),
        'recall'            : make_scorer(recall_score,    zero_division=0),
        'f1'                : make_scorer(f1_score,        zero_division=0),
        'roc_auc'           : 'roc_auc',
        'average_precision' : 'average_precision',
    }

    # FIX: ImbPipeline actually applies SMOTE + scaling inside each fold
    # Without this, X_cv is 0.13% fraud — model predicts all-negative — NaN scores
    pipeline = ImbPipeline([
        ('smote',  SMOTE(sampling_strategy=0.3, random_state=42)),
        ('scaler', StandardScaler()),
        ('model',  model),
    ])

    results = cross_validate(
        pipeline, X_cv, y_cv,
        cv                 = skf,
        scoring            = scoring,
        n_jobs             = 1,
        return_train_score = True,
    )

    print(f"\n{'─'*60}")
    print(f"  K-Fold CV ({n_splits}-Fold): {model_name.upper()}")
    print(f"{'─'*60}")
    print(f"  {'Metric':<20} {'Train':>10} {'Val':>10} {'Gap':>10}")
    print(f"  {'─'*50}")
    for metric in scoring:
        tr   = results[f'train_{metric}']
        vl   = results[f'test_{metric}']
        gap  = tr.mean() - vl.mean()
        flag = "  ⚠ overfit" if gap > 0.10 else ""
        print(f"  {metric:<20}: {tr.mean():.4f}±{tr.std():.4f}  "
              f"{vl.mean():.4f}±{vl.std():.4f}  {gap:+.4f}{flag}")

    return {
        "model"                     : model_name,
        "cv_precision_mean"         : results['test_precision'].mean(),
        "cv_precision_std"          : results['test_precision'].std(),
        "cv_recall_mean"            : results['test_recall'].mean(),
        "cv_recall_std"             : results['test_recall'].std(),
        "cv_f1_mean"                : results['test_f1'].mean(),
        "cv_f1_std"                 : results['test_f1'].std(),
        "cv_auc_mean"               : results['test_roc_auc'].mean(),
        "cv_auc_std"                : results['test_roc_auc'].std(),
        "cv_average_precision_mean" : results['test_average_precision'].mean(),
        "cv_average_precision_std"  : results['test_average_precision'].std(),
        "cv_train_f1_mean"          : results['train_f1'].mean(),
        "cv_train_auc_mean"         : results['train_roc_auc'].mean(),
    }

import json

# Separate folder for experiment CSVs
EXPERIMENTS_SAVE_PATH = "outputs/experiments/"
os.makedirs(EXPERIMENTS_SAVE_PATH, exist_ok=True)

def _make_serialisable(metrics: dict) -> dict:
    """
    Strip non-serialisable objects (numpy arrays, numpy scalars)
    from a metrics dict before JSON / CSV save.
    y_pred and y_prob are numpy arrays — they must be removed here;
    they are only needed in-memory for plots.
    """
    clean = {}
    skip  = {'y_pred', 'y_prob'}          # arrays — never serialise into metrics files
    for k, v in metrics.items():
        if k in skip:
            continue
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    return clean

def save_metrics(metrics: dict, model_name: str):
    """
    Save model metrics in BOTH JSON and CSV format.

    JSON  → outputs/metrics/<model_name>_metrics.json
    CSV   → outputs/metrics/<model_name>_metrics.csv

    Arrays (y_pred, y_prob) are intentionally excluded — they belong
    in memory only, not in metrics files.
    """
    clean = _make_serialisable(metrics)
    safe_name = model_name.lower().replace(' ', '_')

    # ── JSON
    json_path = os.path.join(METRICS_SAVE_PATH, f"{safe_name}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(clean, f, indent=4)

    # ── CSV  (single-row)
    csv_path = os.path.join(METRICS_SAVE_PATH, f"{safe_name}_metrics.csv")
    pd.DataFrame([clean]).to_csv(csv_path, index=False)

    print(f"  Metrics saved → {json_path}")
    print(f"  Metrics saved → {csv_path}")

def save_experiment_results(metrics: dict, model_name: str):
    """
    Append this run's experiment results to a master CSV log.
    Creates the file if it doesn't exist; appends if it does.
    Saves model weights path as a reference column.

    CSV → outputs/experiments/experiment_results.csv
    """
    clean = _make_serialisable(metrics)
    safe_name = model_name.lower().replace(' ', '_')

    clean['model_name']    = model_name
    clean['weights_path']  = os.path.join(MODEL_SAVE_PATH, f"{safe_name}.pkl")
    clean['run_timestamp'] = pd.Timestamp.now().isoformat()

    exp_path = os.path.join(EXPERIMENTS_SAVE_PATH, "experiment_results.csv")
    row_df   = pd.DataFrame([clean])

    if os.path.exists(exp_path):
        existing = pd.read_csv(exp_path)
        pd.concat([existing, row_df], ignore_index=True).to_csv(exp_path, index=False)
    else:
        row_df.to_csv(exp_path, index=False)

    print(f"  Experiment results appended → {exp_path}")


print("Save utilities loaded.")

# ── Arrays for final model training (post-SMOTE, post-scaling)
X_tr = X_train_final.values.astype('float32')
X_ts = X_test_final.values.astype('float32')
y_tr = np.array(y_train_smote, dtype=int)
y_ts = np.array(y_test,        dtype=int)

# ── Arrays for CV (post-fraud-simulation, pre-SMOTE, pre-scaling)
# Passing X_train_aug ensures val fold has zero synthetic samples
# ── Prepare arrays for CV (pre-SMOTE, pre-scaling — fraud simulation output only)
# SMOTE and scaling happen inside run_kfold pipeline per fold
X_cv = X_train_aug.values.astype('float32')
y_cv = np.array(y_train_aug, dtype=int)

print(f"Final train : {X_tr.shape}  fraud ratio: {y_tr.mean():.4f}")
print(f"Test        : {X_ts.shape}  fraud ratio: {y_ts.mean():.4f}")
print(f"CV input    : {X_cv.shape}  fraud ratio: {y_cv.mean():.4f}")

