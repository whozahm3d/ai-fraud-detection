# Save the cleaned and preprocessed DataFrame (post-cleaning, post-encoding, pre-SMOTE, pre-scaling)
# This is the base dataset BEFORE train/test split augmentation.
# Fraud simulation and SMOTE are applied only to X_train — not to df.
# Use df_final (Cell 63) for the fully processed training set.


def save_dataset(df, filename, folder="outputs/processed_data/"):
    """
    Generic save function
    """

    # Ensure the target folder exists
     # Cleaned + Processed Dataset
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)

    print(f"Saved: {path}")
    print(f"Shape: {df.shape}")

save_dataset(df, "processed_paysim.csv")
Saved: outputs/processed_data/processed_paysim.csv
Shape: (6362620, 16)

print(df.head())
# output
   step        amount     nameOrig  oldbalanceOrg  newbalanceOrig  \
0     1   9839.639648  C1231006815       170136.0   160296.359375   
1     1   1864.280029  C1666544295        21249.0    19384.720703   
2     1    181.000000  C1305486145          181.0        0.000000   
3     1    181.000000   C840083671          181.0        0.000000   
4     1  11668.139648  C2048537720        41554.0    29885.859375   

      nameDest  oldbalanceDest  newbalanceDest  isFraud  isFlaggedFraud  \
0  M1979787155             0.0             0.0        0               0   
1  M2044282225             0.0             0.0        0               0   
2   C553264065             0.0             0.0        1               0   
3    C38997010         21182.0             0.0        1               0   
4  M1230701703             0.0             0.0        0               0   

    balanceDiff  amount_ratio  type_CASH_OUT  type_DEBIT  type_PAYMENT  \
0   9839.640625      0.057834          False       False          True   
1   1864.279297      0.087731          False       False          True   
2    181.000000      0.994505          False       False         False   
3    181.000000      0.994505           True       False         False   
4  11668.140625      0.280788          False       False          True   

   type_TRANSFER  
0          False  
1          False  
2           True  
3          False  
4          False  



def save_pipeline_outputs(df_train, df_test, df_aug, df_final):
    """
    Save all pipeline stages
    """

    print("\n===== SAVING ALL DATASETS =====")

    # 1. Original Train/Test
    save_dataset(df_train, "train_original.csv")
    save_dataset(df_test, "test_original.csv")

    # 2. After Fraud Simulation
    save_dataset(df_aug, "train_after_simulation.csv")

    # 3. After SMOTE (Final Training Data)
    save_dataset(df_final, "train_final_smote.csv")

    print("\n All datasets saved successfully!")


save_pipeline_outputs(df_train, df_test, df_aug, df_final)
# output
===== SAVING ALL DATASETS =====
Saved: outputs/processed_data/train_original.csv
Shape: (6107909, 13)
Saved: outputs/processed_data/test_original.csv
Shape: (1272524, 13)
Saved: outputs/processed_data/train_after_simulation.csv
Shape: (5148253, 13)
Saved: outputs/processed_data/train_final_smote.csv
Shape: (6608583, 13)

 All datasets saved successfully!



print(df_final.head())
       step    amount  oldbalanceOrg  newbalanceOrig  oldbalanceDest  \
0 -1.652397 -0.264755      -0.249410       -0.241122       -0.328275   
1 -1.616715 -0.269301      -0.266346       -0.254292       -0.328275   
2 -0.110956  0.257120      -0.266346       -0.254292       -0.181858   
3 -0.075275 -0.268960      -0.266346       -0.254292       -0.328275   
4 -0.738951  0.044399       4.727031        4.833709        0.400641   

   newbalanceDest  balanceDiff  amount_ratio  type_CASH_OUT  type_DEBIT  \
0       -0.353642    -0.051772     -0.122299      -0.947041      -0.071   
1       -0.353642    -0.074055     -0.106909      -0.947041      -0.071   
2       -0.118310    -0.074055      0.688691       1.055920      -0.071   
3       -0.353642    -0.074055     -0.106395      -0.947041      -0.071   
4        0.260328    -0.564035     -0.122299      -0.947041      -0.071   

   type_PAYMENT  type_TRANSFER  isFraud  
0      1.685065      -0.386579        0  
1      1.685065      -0.386579        0  
2     -0.593449      -0.386579        0  
3      1.685065      -0.386579        0  
4     -0.593449      -0.386579        0  
