import numpy as np
import cv2 
import pandas as pd


df = pd.read_csv('ICESING.tsv', delimiter='\t')

# Step 2: Get the list of column names
columns = df.columns.to_list()

# Step 3: Replace 0 with NaN in the numeric columns before starting the loop
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].replace(0, pd.NA)

# Step 4: Loop through columns and create new normalized columns
for colmns in columns:
    # Avoid division by 'n_token' itself and check if the column is numeric
    if colmns != 'n_token' and pd.api.types.is_numeric_dtype(df[colmns]):
        new_col_name = f'n_token_by_{colmns}'  # Name of the new column
        # Perform the division and create the new column
        df[new_col_name] = df['n_token'] / df[colmns]
        # Print for debugging: check the newly created column and original columns
        print(f"Created column: {new_col_name}")
        print(df[[new_col_name]].head())  # Display the first few rows of the new column
    else:
        # If the column is not numeric, you can either skip it or handle it differently
        print(f"Skipping column: {colmns} (not numeric)")

# Step 5: Replace NaN (pd.NA) with 0 after the loop
df[columns] = df[columns].fillna(0)

# Step 6: Save the resulting DataFrame to a new CSV file
df.to_csv('normalized.csv', index=False)