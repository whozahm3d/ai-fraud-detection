def fraud_simulation_engine(X_train, y_train):
    """
    Inject synthetic fraud based on realistic PaySim fraud patterns.
    
    Real fraud in PaySim occurs only in TRANSFER and CASH_OUT transactions.
    Behavioral pattern: amount ≈ entire origin balance, origin is drained to 0.
    
    Strategy:
    - Sample 5% of TRANSFER or CASH_OUT legitimate transactions from training set
    - Set amount = oldbalanceOrg (full account drain)
    - Set newbalanceOrig = 0 (account drained)
    - Recompute balanceDiff and amount_ratio to reflect drain
    - Label as fraud
    """
    df_temp = X_train.copy()
    df_temp['isFraud'] = y_train.values

    # Only sample from high-risk types — TRANSFER or CASH_OUT (one-hot encoded)
    # type_CASH_OUT and type_TRANSFER are binary columns from get_dummies
    high_risk_mask = (
        ((df_temp.get('type_CASH_OUT', 0) == 1) | (df_temp.get('type_TRANSFER', 0) == 1))
        & (df_temp['isFraud'] == 0)
        & (df_temp['oldbalanceOrg'] > 0)   # only accounts with money to drain
    )
    candidate_pool = df_temp[high_risk_mask]

    if len(candidate_pool) == 0:
        print("Warning: No high-risk candidates found. Falling back to all legitimate.")
        candidate_pool = df_temp[df_temp['isFraud'] == 0]

    synthetic = candidate_pool.sample(frac=0.05, random_state=42).copy()

    # Realistic fraud pattern: drain the full account
    synthetic['amount']         = synthetic['oldbalanceOrg']
    synthetic['newbalanceOrig'] = 0.0
    synthetic['balanceDiff']    = synthetic['oldbalanceOrg']          # all money moved out
    synthetic['amount_ratio']   = 1.0                                  # 100% of balance taken
    synthetic['isFraud']        = 1

    df_aug = pd.concat([df_temp, synthetic], ignore_index=True)

    X_aug = df_aug.drop('isFraud', axis=1)
    y_aug = df_aug['isFraud']

    print("Fraud Simulation Completed")
    print(f"  Sampled from {len(candidate_pool):,} high-risk candidates")
    print(f"  Injected {len(synthetic):,} synthetic fraud records")
    print("  Augmented fraud ratio:", round(y_aug.mean(), 4))

    return X_aug, y_aug


# Apply Fraud Simulation Engine to training data
X_train_aug, y_train_aug = fraud_simulation_engine(X_train, y_train)

print("Final Train Fraud Ratio:", round(y_train_aug.mean(), 4))

df_aug = pd.concat([X_train_aug, y_train_aug], axis=1)
df_aug.head()

print_memory("[AFTER FRAUD SIMULATION]")

# Fraud vs Non-Fraud distribution
# Fraud detection datasets are typically highly imbalanced,
# meaning fraudulent transactions represent a very small
# percentage of the total transactions.

fraud_counts = df_aug["isFraud"].value_counts()

print(fraud_counts)

fraud_percentage = df_aug["isFraud"].value_counts(normalize=True) * 100

print("\nFraud Percentage:")
print(fraud_percentage)

# MEMORY ADDITION #4
gc.collect()
