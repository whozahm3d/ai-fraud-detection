# Load the PaySim dataset containing synthetic financial transaction records used for fraud detection analysis.

def print_memory(label=""):
    """
    Print current memory usage for monitoring.
    Call this after major operations to track memory freed.
    """
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory Usage {label:30s}: {mem_mb:>8,.0f} MB")
    except Exception as e:
        print(f"Memory check failed: {e}")

# Test it works
print_memory("[STARTUP]")

def load_dataset(path):
    """
    Load dataset from CSV file with optimized data types to reduce memory usage
    """

    # Define optimized dtypes to reduce memory footprint
    dtype_dict = {
        'step': 'int32',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'category',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'category',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8',
    }    
    df = pd.read_csv(path, dtype=dtype_dict)

    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# Load dataset using the function
dataset_path = r"E:\Documents\Artifical Intelligence\AI Project\Dataset\PaySim - Synthetic Financial Dataset for Fraud Detection.csv"
print("Loading dataset...")
print_memory("[BEFORE LOAD]")

df = load_dataset(dataset_path)

print_memory("[AFTER LOAD]")

