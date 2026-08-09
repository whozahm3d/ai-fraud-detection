def apply_smote(X_train, y_train, sampling_ratio=0.3):
    """
    Apply SMOTE to balance dataset.
    sampling_ratio=0.3 means the minority class will reach
    10% of the size of the majority class (reduced to prevent excessive synthetic data).
    """

    # FIX #2: Reduced sampling_ratio from 0.5 to 0.3 to prevent 40-60% training set bloat
    smote = SMOTE(sampling_strategy=sampling_ratio, random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("\nSMOTE Applied Successfully")
    print("New Fraud Ratio:", round(y_resampled.mean(), 4))

    return X_resampled, y_resampled

print("SMOTE RESAMPLING")

print_memory("[BEFORE SMOTE]")


# Apply SMOTE
X_train_smote, y_train_smote = apply_smote(X_train_aug, y_train_aug)

# MEMORY ADDITION #6 — THE BIGGEST WIN
# SMOTE always returns float64. Our data was float32.
# Converting back to float32 cuts this array's memory in HALF.
X_train_smote = X_train_smote.astype('float32')
gc.collect()

print("Final Train Fraud Ratio:", round(y_train_smote.mean(), 4))

df_smote = pd.concat([
    pd.DataFrame(X_train_smote, columns=FEATURE_NAMES),
    pd.Series(y_train_smote, name='isFraud')
], axis=1)
df_smote.head()

print_memory("[AFTER SMOTE]")

# Fraud vs Non-Fraud distribution
# Fraud detection datasets are typically highly imbalanced,
# meaning fraudulent transactions represent a very small
# percentage of the total transactions.

fraud_counts = df_smote["isFraud"].value_counts()

print(fraud_counts)

fraud_percentage = df_smote["isFraud"].value_counts(normalize=True) * 100

print("\nFraud Percentage:")
print(fraud_percentage)


# Handling Class Imbalance
# ------------------------------------------------------------
# The dataset is highly imbalanced (fraud vs non-fraud cases).
# To address this:
# 1. Fraud simulation is applied to generate synthetic fraud-like patterns.
# 2. SMOTE (Synthetic Minority Oversampling Technique) is used
#    to balance the class distribution by generating synthetic samples.
# This improves model performance and prevents bias toward majority class.
