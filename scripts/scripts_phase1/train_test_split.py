print("TRAIN/TEST SPLIT")
print_memory("[BEFORE SPLIT]")

# Step Split
# Ensure df is the preprocessed DataFrame before this step.
# Feature Selection
# Drop columns not used for model training:
#   isFraud       — target variable (label)
#   nameOrig      — high-cardinality ID, not generalizable
#   nameDest      — high-cardinality ID, not generalizable
#   isFlaggedFraud — INTENTIONALLY DROPPED to prevent data leakage.
#                    This column is a simulator-generated rule-based flag (transactions > 200k
#                    to empty accounts). It was created by the PaySim simulator using the same
#                    logic that generates isFraud. Keeping it would inflate all model metrics
#                    artificially — models would learn to rely on the flag rather than the
#                    underlying transaction features. Dropped for honest evaluation.

X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y = df['isFraud']


# Save feature names for later (visualizations, error analysis)
FEATURE_NAMES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train size :", X_train.shape)
print("Test  size :", X_test.shape)
print("Train fraud ratio:", round(y_train.mean(), 4))
print("Test  fraud ratio:", round(y_test.mean(), 4))


print_memory("[AFTER TRAIN/TEST SPLIT]")

