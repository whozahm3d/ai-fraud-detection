def scale_features(X_train, X_test):
    """
    Apply StandardScaler.
    Fit on training data only — transform both train and test.
    Uses shallow copies to minimize memory allocation.
    """
    scaler       = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns

    # Create shallow copies only (metadata copy, not data copy)
    X_train_scaled = X_train.copy(deep=False)
    X_test_scaled  = X_test.copy(deep=False)

    # Scale numeric columns - convert to float32 to save memory
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols]).astype('float32')
    X_test_scaled[numeric_cols]  = scaler.transform(X_test[numeric_cols]).astype('float32')

    print("Scaling Completed")
    return X_train_scaled, X_test_scaled, scaler

os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/plots",   exist_ok=True)


print(" FEATURE SCALING")
print_memory("[BEFORE SCALING]")


# Wrap SMOTE output into a DataFrame so scale_features works correctly
X_train_smote_df = pd.DataFrame(X_train_smote, columns=FEATURE_NAMES)
X_test_df        = pd.DataFrame(X_test.values,  columns=FEATURE_NAMES)

# Apply Scaling
X_train_final, X_test_final, scaler = scale_features(X_train_smote_df, X_test_df)
joblib.dump(scaler, "outputs/models/scaler.pkl")

print("Train size :", X_train_final.shape)
print("Test  size :", X_test_final.shape)
print("Train fraud ratio:", round(pd.Series(y_train_smote).mean(), 4))
print("Test  fraud ratio:", round(y_test.mean(), 4))

# MEMORY ADDITION #7
gc.collect()

print_memory("[AFTER SCALING & CLEANUP]")

df_train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

df_final = pd.concat([
    X_train_final.reset_index(drop=True),
    pd.Series(y_train_smote, name='isFraud')
], axis=1)

df_scaled = X_train_final
df_scaled.head()

print_memory("[AFTER FINAL DATAFRAMES CREATED]")
