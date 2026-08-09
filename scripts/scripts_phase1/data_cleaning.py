# Check for missing values to ensure data quality before performing preprocessing and analysis.

def remove_duplicates(df):
    """
    Remove duplicate rows
    """

    before = df.shape[0]

    df = df.drop_duplicates()

    after = df.shape[0]

    print("Duplicates removed:", before - after)

    return df

def drop_null(df):
    """
    Remove duplicate rows
    """

    before = df.shape[0]

    df = df.dropna()

    after = df.shape[0]

    print("Drop missing or null values:", before - after)

    return df
    
