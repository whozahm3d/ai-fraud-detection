# Feature scaling was applied to numerical variables using StandardScaler to normalize the distribution of transaction-related variables. 
# This step ensures that features with larger ranges do not dominate the learning process of machine learning models.

def preprocess_data(df):

    df = remove_duplicates(df)
    df = drop_null(df)
    df = feature_engineering(df)
    df = encode_features(df)

    return df


# Return the processed DataFrame
#return df
print("PREPROCESSING STARTED")
print_memory("[BEFORE PREPROCESS]")

# Use existing preprocess_data function
df = preprocess_data(df)

print("\nPREPROCESING COMPLETED!")
print_memory("[AFTER PREPROCESS]")
print(f"Processed shape: {df.shape}")

df.head()

# MEMORY ADDITION #2 — free preprocessing garbage
gc.collect()


# Fraud vs Non-Fraud distribution
# Fraud detection datasets are typically highly imbalanced,
# meaning fraudulent transactions represent a very small
# percentage of the total transactions.

fraud_counts = df["isFraud"].value_counts()

print(fraud_counts)

fraud_percentage = df["isFraud"].value_counts(normalize=True) * 100

print("\nFraud Percentage:")
print(fraud_percentage)
