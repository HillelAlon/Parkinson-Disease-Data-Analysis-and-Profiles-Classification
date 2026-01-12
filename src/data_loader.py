import pandas as pd
import logging

def load_raw_data(file_path):
    """
    Step 1 & 2: Load the raw CSV and set PatientID as the index.
    """
    logging.info(f"Loading raw data from {file_path}")
    # Setting PatientID as index right at the start
    df = pd.read_csv(file_path, index_col='PatientID')
    return df

def clean_data_structure(df):
    """
    Step 3 & 4: Remove irrelevant columns and duplicate rows.
    """
    logging.info("Cleaning data: Dropping 'DoctorInCharge' and removing duplicates.")
    # Drop DoctorInCharge if it exists
    df_clean = df.drop(columns=['DoctorInCharge'], errors='ignore')
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    return df_clean

def prepare_for_analysis(df):
    """
    Step 5: Map Ethnicity values to names for better visualization.
    """
    logging.info("Preparing data for analysis: Mapping ethnicity labels.")
    df_analysis = df.copy()
    ethnicity_map = {0: 'Caucasian', 1: 'African_American', 2: 'Asian', 3: 'Other'} 
    df_analysis['Ethnicity'] = df_analysis['Ethnicity'].map(ethnicity_map)
    return df_analysis

def prepare_for_modeling(df):
    """
    Step 6: One-Hot Encoding for Ethnicity to prepare for Machine Learning.
    """
    logging.info("Preparing data for modeling: Applying One-Hot Encoding to Ethnicity.")
    # Using get_dummies to create binary columns for Ethnicity
    df_model = pd.get_dummies(df, columns=['Ethnicity'], drop_first=False, dtype=int)
    return df_model

def save_data(df, output_path):
    """
    Helper function to save the processed DataFrames.
    """
    df.to_csv(output_path, index=True)
    logging.info(f"Successfully saved file to: {output_path}")