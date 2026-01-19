# install requirements
pip install -r ../requirements.txt

#Imports libraries 
import pandas as pd
from IPython.display import display
import pandas as pd
import os

# --- 1. Load Data ---
# --- 2. PatientID to Index ---
df = pd.read_csv('../data/parkinsons_disease_data.csv', index_col='PatientID')

# --- 3. Remove Irrelevant Columns ---
cols_to_drop = ['DoctorInCharge']
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# --- 4. Remove Duplicates ---
df_clean = df_clean.drop_duplicates()

#--- Outlier Detection (Sanity Check) ---
print("Data Statistics Check:")
display(df_clean[['Age', 'BMI', 'SystolicBP']].describe())

# --- 5. Save Cleaned Data ---
# index=True to save the IDs back to the CSV
output_path = '../data/parkinsons_cleaned.csv'
df_clean.to_csv(output_path, index=True) 

print("Cleaned data saved.")
display(df.head())
