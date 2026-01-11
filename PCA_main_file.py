import pandas as pd

df = pd.read_csv("data/parkinsons_cleaned.csv")
print(df.head())

# Keep only the sick patients in the data frame.
df = df[df['Diagnosis'] != 0]
print(df.head())
print(df.info())

# Keep only age, lifestyle and clinical measurements categories
bool_cols_to_drop = [col for col in df.columns if len(set(df[col].unique())) < 5] #Remove boolean and categorical columns
df = df.drop(columns=bool_cols_to_drop)
df = df.drop(columns=["PatientID","UPDRS", "MoCA", "FunctionalAssessment"]) #Remove categories: ID and assessments

print(df.info())
print(df.head(50))

df.to_csv("data/parkinsons_lifestyle_clinical_for_PCA.csv",index=True)


"""
df1 = df
print(df1.info())
print(df1.head())
for col in df.columns:
    df1[f'{col} Z-Scores'] = (df[col] - df[col].mean()) / df[col].std()
"""

