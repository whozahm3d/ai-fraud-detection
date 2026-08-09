# Perform initial dataset inspection including shape, column names, data types, and statistical summary to understand the dataset structure.
# MEMORY ADDITION #1 — free loader garbage
gc.collect()

def dataset_shape(df):
    """
    Display number of rows and columns
    """
    print("\nDataset Shape:", df.shape)

def dataset_columns(df):
    """
    Display column names
    """
    print("\nColumns:")
    print(df.columns)


def dataset_datatypes(df):
    """
    Display datatypes of each column
    """
    print("\nData Types:")
    print(df.dtypes)

def dataset_info(df):
    """
    Display overall info of the entire dataset
    """
    print("\nDataset Info:")
    print(df.info())

def dataset_description(df):
    """
    Statistical summary of dataset
    """
    print("\nDataset Statistical Summary:")
    print(df.describe())


def missing_values(df):
    """
    Check missing values in dataset
    """
    print("\nMissing Values:")
    print(df.isnull().sum())

def dataset_fraud(df):
    """
    Display fraud distribution
    """
    print(df['isFraud'].value_counts(normalize=True))


def dataset_head(df):
    """
    Display first rows
    """
    print("\nFirst 5 Rows:")
    print(df.head())



dataset_shape(df)
dataset_columns(df)
dataset_datatypes(df)
dataset_info(df)
dataset_description(df)
missing_values(df)
dataset_fraud(df)
dataset_head(df)


