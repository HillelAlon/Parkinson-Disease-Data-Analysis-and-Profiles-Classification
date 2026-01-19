#Imports libraries 
import pandas as pd

def load_and_clean_data(csv_path, index_col_name, cols_to_remove):
    """
    checking type for inputs.
    Loads a CSV, sets an index, removes specific columns, and drops duplicates.
    """
    
    # chacks if path is string and if its csv
    if not isinstance(csv_path, str):
        raise TypeError(f"csv_path must be a string, got {type(csv_path)}")
    
    if not csv_path.lower().endswith('.csv'):
        raise ValueError("The file path must end with '.csv'")
        

    # chacks if index col is string
    if not isinstance(index_col_name, str):
        raise TypeError(f"index_col_name must be a string, got {type(index_col_name)}")
    
      # chacks if index col is exsist in the CSV
    if not isinstance(index_col_name, str):
        raise TypeError(f"index_col_name must be a string, got {type(index_col_name)}")

    # chacks if cols_to_remove is list
    if not isinstance(cols_to_remove, list):
        raise TypeError(f"cols_to_remove must be a list, got {type(cols_to_remove)}")
    
    # chacks if every element in cols_to_remove is string
    if not all(isinstance(col, str) for col in cols_to_remove):
        raise ValueError("All items in 'cols_to_remove' must be strings.")
    
     # chacks if index col is exsist in the CSV
    if not isinstance(index_col_name, str):
        raise TypeError(f"index_col_name must be a string, got {type(index_col_name)}")

    # if all checks worked :)
    print(f"Loading data from: {csv_path}...")
   
   #load and set index columns
    df = pd.read_csv(csv_path, index_col=index_col_name)
    
    #remove irralavant columns
    df_clean = df.drop(columns=cols_to_remove, errors='ignore')
    
    #remove duplucate
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_rows = initial_rows - len(df_clean)
    
    print(f"Successfully loaded. Dropped {dropped_rows} duplicates. Final shape: {df_clean.shape}")
    
    return df_clean
