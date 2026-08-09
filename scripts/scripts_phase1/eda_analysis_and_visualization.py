# Perform visual analysis to understand patterns in transactions and identify potential fraud indicators.

# 1. Transaction Volume Over Time
# Concept: Detect seasonal or temporal trends in transaction activity. Sudden spikes may indicate fraud bursts.
# Technical Insight: Use time-series aggregation of transaction counts by day/week/month.

def eda_validation(y_train, y_test):
    """
    Validate class distribution in train vs test
    """

    print("\nTRAIN SET DISTRIBUTION")
    print(pd.Series(y_train).value_counts())

    print("\nFraud % (Train):")
    print(pd.Series(y_train).value_counts(normalize=True) * 100)

    print("\nTEST SET DISTRIBUTION")
    print(pd.Series(y_test).value_counts())

    print("\nFraud % (Test):")
    print(pd.Series(y_test).value_counts(normalize=True) * 100)

eda_validation(y_train, y_test)

def dataset_integrity_check(X_train, X_test):
    """
    Ensure no data leakage between train and test
    """

    print("\nDATA LEAKAGE CHECK")

    overlap = len(set(X_train.index).intersection(set(X_test.index)))

    print("Overlapping rows:", overlap)

    if overlap == 0:
        print("No Data Leakage Detected")
    else:
        print("Warning: Data Leakage Exists")

dataset_integrity_check(X_train, X_test)


# ── EDA: Class distribution at each pipeline stage

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

stages = [
    ("Original Train Split",        y_train),
    ("After Fraud Simulation",      y_train_aug),
    ("After SMOTE (Final Train)",   pd.Series(y_train_smote)),
]

for ax, (title, y_data) in zip(axes, stages):
    counts  = pd.Series(y_data).value_counts().sort_index()
    labels  = ['Legitimate', 'Fraud']
    colors  = [C['green'], C['red']]
    bars    = ax.bar(labels, counts.values, color=colors, width=0.45,
                     edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{val:,}', ha='center', fontsize=11, fontweight='bold')
    pct = counts[1] / counts.sum() * 100
    ax.set_title(f"{title}\nFraud: {pct:.2f}%", fontsize=12, fontweight='bold')
    ax.set_ylabel("Transaction Count", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.18)

fig.suptitle("Class Distribution Across Pipeline Stages", fontsize=15,
             fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig("outputs/plots/eda_class_distribution.png", bbox_inches='tight')
plt.show()
# MEMORY ADDITION #8 — close figure from memory
plt.close('all')
print("Class distribution chart saved.")
# MEMORY ADDITION #9
gc.collect()



# ── EDA: Feature distributions — fraud vs legitimate

X_eda  = X_train.copy()
X_eda['isFraud'] = y_train.values

plot_cols = ['amount', 'balanceDiff', 'amount_ratio', 'oldbalanceOrg']
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, col in zip(axes, plot_cols):
    if col in X_eda.columns:
        legit = X_eda[X_eda['isFraud'] == 0][col].clip(
            upper=X_eda[col].quantile(0.98))
        fraud = X_eda[X_eda['isFraud'] == 1][col].clip(
            upper=X_eda[col].quantile(0.98))
        ax.hist(legit, bins=60, color=C['green'], alpha=0.65, label='Legitimate', density=True)
        ax.hist(fraud, bins=60, color=C['red'],   alpha=0.65, label='Fraud',      density=True)
        ax.set_title(col, fontsize=13, fontweight='bold')
        ax.set_xlabel("Value (clipped at 98th pct)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=10)

fig.suptitle("Feature Distributions: Fraud vs Legitimate", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/eda_feature_distributions.png", bbox_inches='tight')
plt.show()
# MEMORY ADDITION #10 — close figure from memory
plt.close('all')
gc.collect()
print("Feature distribution chart saved.")


# 1. Transaction Volume Over Time
# Concept: Detect seasonal or temporal trends in transaction activity. Sudden spikes may indicate fraud bursts.
# Technical Insight: Use time-series aggregation of transaction counts by day/week/month.

# The 'transaction_date' column does not exist. Using 'step' as a proxy for time.
time_series = df.groupby('step').size()

plt.figure(figsize=(12,5))
plt.plot(time_series, marker='o')
plt.title('Transaction Volume Over Time (by Step)')
plt.xlabel('Step')
plt.ylabel('Number of Transactions')

plt.grid(True)
plt.savefig("outputs/plots/transaction_volume_over_time.png", dpi=300, bbox_inches="tight")
plt.show()
# MEMORY ADDITION #11 — close figure from memory
plt.close('all')
del time_series
gc.collect()

# 2. Fraudulent vs Legitimate Transaction Distribution
# Concept: Compare class imbalance visually. Essential for imbalanced classification problems.
# Technical Insight: Bar charts or pie charts for categorical understanding.

def fraud_distribution(df_smote):

    plt.figure(figsize=(6,4))
    sns.countplot(
        x="isFraud",
        hue="isFraud",
        data=df_smote,
        order=[0, 1],
        hue_order=[0, 1],
        palette=[C['green'], C['red']],
        legend=False
    )
    plt.xlabel("Transaction Type", fontsize=10)
    plt.ylabel("Number of Transactions", fontsize=10)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xticks([0, 1], ['Legitimate', 'Fraud'])
    plt.tight_layout()
    plt.grid(True)

    plt.savefig("outputs/plots/fraud_distribution.png")
    plt.show()
    plt.close('all')

fraud_distribution(df_smote)

# 3. Transaction types Distribution
def transaction_types(df_original, n_rows=30000):
    # Use only first n_rows to speed up plotting and reduce memory usage
    df_plot = df_original.head(n_rows)

    # medium-dark, readable colors (no pink)
    type_palette = ['#2E86C1', '#16A085', '#27AE60', '#E67E22', '#7F8C8D']

    plt.figure(figsize=(8,5))
    sns.countplot(
        x="type",
        hue="type",          # avoids seaborn palette deprecation warning
        data=df_plot,
        palette=type_palette,
        legend=False
    )

    plt.title(f"Transaction Types Distribution (first {len(df_plot):,} rows)")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("outputs/plots/transaction_types_distribution.png")
    plt.show()
    plt.close('all')


# Reconstruct human-readable 'type' column from one-hot encoded columns
# This avoids reloading the full CSV just for one plot
df_for_type_plot = df.copy()
df_for_type_plot['type'] = np.select(
    [
        df_for_type_plot.get('type_CASH_OUT', pd.Series(False, index=df.index)).astype(bool),
        df_for_type_plot.get('type_DEBIT',    pd.Series(False, index=df.index)).astype(bool),
        df_for_type_plot.get('type_PAYMENT',  pd.Series(False, index=df.index)).astype(bool),
        df_for_type_plot.get('type_TRANSFER', pd.Series(False, index=df.index)).astype(bool),
    ],
    ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
    default='CASH_IN'
)

transaction_types(df_for_type_plot, 30000)
del df_for_type_plot
gc.collect()

# 4. Fraud Transaction by Type

# Identify the one-hot encoded 'type' columns
type_columns = [col for col in df_smote.columns if col.startswith('type_')]

# Melt the DataFrame to bring the one-hot encoded columns into a single 'variable' column
# and their values into a 'value' column.
# Filter only for rows where 'value' is True (i.e., the transaction is of that type)
df_melted_types = df_smote.melt(id_vars=['isFraud'], value_vars=type_columns, var_name='transaction_type_encoded', value_name='is_this_type')
df_melted_types = df_melted_types[df_melted_types['is_this_type'] == True]

# Rename the transaction_type_encoded to a cleaner 'type' for plotting
df_melted_types['type'] = df_melted_types['transaction_type_encoded'].str.replace('type_', '')

plt.figure(figsize=(10, 6))
sns.countplot(x="type", hue="isFraud", data=df_melted_types)

plt.title("Fraud Transactions by Type (from final processed data)")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/plots/fraud_by_transaction_type.png", dpi=300)
plt.show()

# MEMORY ADDITION #12 — close figure from memory and free melted DataFrame
plt.close('all')
del df_melted_types
gc.collect()


# 5. Transaction Amount Distribution
# Visualizing transaction amount distribution helps understand
# skewness and the need for feature scaling.

plt.figure(figsize=(8,5))

sns.histplot(df["amount"].iloc[:2_000_000], bins=50)

plt.title("Transaction Amount Distribution")

plt.savefig("outputs/plots/transaction_amount_distribution.png", dpi=300, bbox_inches="tight")

plt.show()
# MEMORY ADDITION #13 — close figure from memory
plt.close('all')
gc.collect()


# 6. Transaction Amount Distribution by Fraud Status
# Concept: Fraud often shows unusual transaction amounts.
# Technical Insight: Boxplots or violin plots highlight outliers and distribution skew.

sns.boxplot(x='isFraud', y='amount', data=df_smote)
plt.title('Transaction Amount Distribution by Fraud Status')
plt.savefig("outputs/plots/fraud_vs_normal_transaction_amounts.png", dpi=300, bbox_inches="tight")
plt.show()
# MEMORY ADDITION #14 — close figure from memory
plt.close('all')
gc.collect()

# 8. Correlation Heatmap
# Concept: Identify relationships between numeric features; helps detect potential predictive features.
# Technical Insight: Strong correlations can guide feature engineering or indicate redundancy.
# This heatmao shows the numeric features distribution before scalarization, fraud simulation and SMOTE
def correlation_heatmap(df):

    plt.figure(figsize=(10,8))

    # Select only numerical columns for correlation calculation
    numerical_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numerical_df.corr(), cmap="rocket", annot=True, fmt=".2f")

    plt.title("Feature Correlation Heatmap")

    plt.savefig("outputs/plots/correlation_heatmap.png")
    plt.show()
    plt.close('all')


correlation_heatmap(df)


# 10. Top Fraud Patterns (Mean comparison)
def plot_fraud_patterns(df_smote):
    mean_df = df_smote.groupby('isFraud')[['amount', 'balanceDiff']].mean()

    ax = mean_df.plot(
        kind='bar',
        figsize=(8, 5),
        color=[C['blue'], C['orange']],   # reasonable contrasting colors
        edgecolor='white',
        width=0.75
    )

    ax.set_title("Fraud Pattern Comparison")
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean Value")
    ax.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
    ax.legend(title="Feature")
    plt.tight_layout()

    plt.savefig("outputs/plots/top_fraud_patterns.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close('all')

plot_fraud_patterns(df_smote)

# 11. Class Imbalance Visualization
def plot_class_imbalance(df_smote):
    counts = df_smote['isFraud'].value_counts(normalize=True).sort_index()

    ax = counts.plot(
        kind='pie',
        figsize=(6, 6),
        autopct='%1.1f%%',
        startangle=90,
        colors=[C['green'], C['red']],   # Legitimate, Fraud
        labels=['Legitimate', 'Fraud'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
        textprops={'fontsize': 10}
    )

    ax.set_ylabel('')
    plt.title("Class Imbalance Ratio")
    plt.tight_layout()

    plt.savefig("outputs/plots/class_imbalance.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close('all')

plot_class_imbalance(df_smote)


# 14. Feature Importance using Correlation (No ML model needed)
features = df.drop(columns=['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud'])

importances = features.corrwith(df['isFraud']).abs().sort_values()

# soft-to-strong color ramp (low -> high importance)
bar_colors = sns.color_palette("blend:#d6eaf8,#1f618d", n_colors=len(importances))

ax = importances.plot(
    kind='barh',
    figsize=(10, 7),
    color=bar_colors,
    edgecolor='white',
    linewidth=1.0
)

ax.set_title('Feature Importance for Fraud Detection')
ax.set_xlabel('Absolute Correlation with isFraud')
ax.set_ylabel('Features')
plt.tight_layout()

plt.savefig("outputs/plots/feature_importance_1.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close('all')
del features, importances
gc.collect()


# ── EDA: Fraud Rate and Volume by Transaction Type
# Two panels:
#   Left  — count of fraud vs legitimate per transaction type (raw volume)
#   Right — fraud rate (%) per type (shows which type is MOST DANGEROUS)
# Uses df_original before encoding so type labels are human-readable.
from matplotlib.ticker import FuncFormatter

def plot_fraud_by_transaction_type(df_original, n_rows=None):
    """
    Show fraud volume and fraud rate broken down by transaction type.
    Uses original (pre-encoded) DataFrame so type names are readable.
    
    Args:
        df_original : raw DataFrame with 'type' and 'isFraud' columns
        n_rows      : if set, sample this many rows to speed up (None = use all)
    """
    df_plot = df_original if n_rows is None else df_original.head(n_rows)

    # ── Compute stats per type
    type_stats = (
        df_plot.groupby('type')['isFraud']
        .agg(total='count', fraud_count='sum')
        .reset_index()
    )
    type_stats['legit_count'] = type_stats['total'] - type_stats['fraud_count']
    type_stats['fraud_rate']  = (type_stats['fraud_count'] / type_stats['total']) * 100
    type_stats = type_stats.sort_values('fraud_rate', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── LEFT PANEL: Stacked bar — volume of fraud vs legit per type
    x      = np.arange(len(type_stats))
    width  = 0.5

    bars_l = axes[0].bar(
        x, type_stats['legit_count'],
        width=width, color=C['green'], edgecolor='white',
        linewidth=1.5, label='Legitimate'
    )
    bars_f = axes[0].bar(
        x, type_stats['fraud_count'],
        width=width, color=C['red'], edgecolor='white',
        linewidth=1.5, label='Fraud',
        bottom=type_stats['legit_count']   # stacked on top of legit
    )

    # Annotate fraud count on top of each stack
    for bar_l, bar_f, fraud_val in zip(bars_l, bars_f, type_stats['fraud_count']):
        total_h = bar_l.get_height() + bar_f.get_height()
        axes[0].text(
            bar_l.get_x() + bar_l.get_width() / 2,
            total_h * 1.02,
            f'Fraud: {int(fraud_val):,}',
            ha='center', fontsize=9, fontweight='bold', color=C['red']
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(type_stats['type'], rotation=30, ha='right', fontsize=10)
    axes[0].set_title("Transaction Volume by Type\n(Fraud vs Legitimate)",
                      fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Transaction Count", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda val, _: f'{int(val):,}')
    )

    # ── RIGHT PANEL: Fraud rate % per type — this is the important one
    # Shows TRANSFER and CASH_OUT are dangerous even if low volume
    bar_colors = [C['red'] if r > 5 else C['orange'] if r > 1 else C['gray']
                  for r in type_stats['fraud_rate']]

    bars_r = axes[1].bar(
        x, type_stats['fraud_rate'],
        width=0.5, color=bar_colors, edgecolor='white', linewidth=1.5
    )

    # Annotate exact % on each bar
    for bar, rate in zip(bars_r, type_stats['fraud_rate']):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f'{rate:.2f}%',
            ha='center', fontsize=10, fontweight='bold'
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(type_stats['type'], rotation=30, ha='right', fontsize=10)
    axes[1].set_title("Fraud Rate (%) by Transaction Type\n(Red = >5%, Orange = 1–5%, Gray = <1%)",
                      fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Fraud Rate (%)", fontsize=11)
    axes[1].set_ylim(0, type_stats['fraud_rate'].max() * 1.25)

    fig.suptitle("Fraud Analysis by Transaction Type",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/plots/fraud_by_type_detailed.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    gc.collect()
    print("Fraud by transaction type chart saved.")


# Reconstruct human-readable transaction type from one-hot columns
df_plot = df.copy()

df_plot["type"] = np.select(
    [
        df_plot["type_CASH_OUT"],
        df_plot["type_DEBIT"],
        df_plot["type_PAYMENT"],
        df_plot["type_TRANSFER"],
    ],
    ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
    default="CASH_IN",
)

plot_fraud_by_transaction_type(df_plot)

# ── EDA: Transaction Amount Distribution per Type — Fraud vs Legitimate
# Violin plot — shows full distribution shape, not just mean
# Critical insight: fraud transactions tend to cluster at higher amounts
# in TRANSFER and CASH_OUT specifically

def plot_amount_by_type_and_fraud(df_original, n_rows=200000):
    """
    Violin plot of transaction amount split by type and fraud status.
    Clips at 98th percentile to suppress extreme outliers distorting the plot.
    
    Args:
        df_original : raw DataFrame with 'type', 'amount', 'isFraud' columns
        n_rows      : sample size — violin plots are slow on 6M rows
    """
    df_plot = df_original.sample(n=min(n_rows, len(df_original)), random_state=42)

    # Clip amount at 98th pct to avoid outlier distortion
    cap = df_plot['amount'].quantile(0.98)
    df_plot = df_plot.copy()
    df_plot['amount_clipped'] = df_plot['amount'].clip(upper=cap)
    df_plot['Class'] = df_plot['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.violinplot(
        data    = df_plot,
        x       = 'type',
        y       = 'amount_clipped',
        hue     = 'Class',
        split   = True,           # mirror halves — legit left, fraud right
        inner   = 'quartile',     # show median + IQR lines inside violin
        palette = {'Legitimate': C['green'], 'Fraud': C['red']},
        linewidth = 1.2,
        ax      = ax,
        order   = sorted(df_plot['type'].unique())
    )

    ax.set_title(
        f"Transaction Amount Distribution by Type & Fraud Status\n"
        f"(Clipped at 98th percentile = {cap:,.0f} | Sample = {len(df_plot):,} rows)",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Transaction Type", fontsize=11)
    ax.set_ylabel(f"Transaction Amount (max {cap:,.0f})", fontsize=11)
    ax.legend(title='Class', fontsize=10, title_fontsize=10)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, _: f'{int(val):,}')
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/amount_distribution_by_type_fraud.png",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')
    del df_plot
    gc.collect()
    print("Amount distribution by type chart saved.")


# Ensure DataFrame passed to the plot has a 'type' column (reconstruct from one-hot if needed)
if 'type' in df.columns:
    plot_amount_by_type_and_fraud(df)
elif 'type_CASH_OUT' in df.columns:
    tmp = df.copy()
    tmp['type'] = np.select(
        [
            tmp['type_CASH_OUT'].astype(bool),
            tmp['type_DEBIT'].astype(bool),
            tmp['type_PAYMENT'].astype(bool),
            tmp['type_TRANSFER'].astype(bool),
        ],
        ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        default="CASH_IN",
    )
    plot_amount_by_type_and_fraud(tmp)
    del tmp
    gc.collect()
else:
    raise KeyError("DataFrame must contain 'type' column or one-hot 'type_*' columns.")


print_memory("[AFTER LOAD]")

## Key Insights from EDA

# - The original dataset exhibited severe class imbalance, with fraudulent transactions representing a very small percentage of the total data. This imbalance was identified as a major challenge for model training.
# - Fraudulent activities were primarily concentrated in specific transaction types, particularly TRANSFER and CASH_OUT, indicating that certain transaction categories carry higher fraud risk.
# - Transaction amounts displayed a highly skewed distribution, with a small number of very large transactions. Fraud cases were often associated with higher transaction amounts and unusual balance changes.
# - Feature engineering, such as the creation of balance difference variables, helped capture hidden transactional anomalies and improved the interpretability of fraud patterns.
# - After applying the fraud simulation engine, the diversity and frequency of fraudulent transactions increased, making fraud patterns more visible and learnable.
# - The application of SMOTE with a sampling ratio of 0.1 significantly improved class balance while maintaining a realistic distribution between fraudulent and non-fraudulent transactions.
# - Comparative analysis between training and testing datasets confirmed that the test set remained untouched, ensuring no data leakage and preserving unbiased evaluation.
# - Feature scaling was applied as the final preprocessing step to standardize numerical features, enabling better performance and stability of machine learning models.
# - Post-processing EDA confirmed that the dataset is now more balanced, structured, and suitable for training robust fraud detection models.
