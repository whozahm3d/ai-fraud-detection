# Convert categorical variables into numerical form by using LabelEnoder so they can be used by machine learning models.

def feature_engineering(df):
    # Use inplace operations to avoid unnecessary copy
    df['balanceDiff'] = (df['oldbalanceOrg'] - df['newbalanceOrig']).astype('float32')
    df['amount_ratio'] = (df['amount'] / (df['oldbalanceOrg'] + 1)).astype('float32')

    print("Feature Engineering is Done!")
    return df


def encode_features(df):
    """
    Convert categorical type columns into numeric values/ Do one-hot encoding

    """

    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    print("Feature encoding done.")
    return df



