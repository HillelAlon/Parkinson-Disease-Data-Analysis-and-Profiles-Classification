import pandas as pd

df = pd.read_csv("../data/parkinsons_cleaned.csv")
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

#0. Make a new df without nominal and ordinal columns.
#1. Transformation the df to Z-Scores values.
#2. PCA: from 12D to new combined 3D. (sklearn.decomposition.PCA)
#3. Clustering.
#4. Compare between the clustering (Age, Lifestyle and Clinical Measures) and Assessments.
#4.1 Correlate between clustering and UPDRS (Unified Parkinson's Disease Rating Scale).
#4.2 Correlate between clustering and MoCA (Montreal Cognitive Assessment).
#4.3 Correlate between clustering and FunctionalAssessment.


df1 = df
print(df1.info())
print(df1.head())
print(df1.describe())
for col in df.columns:
    df1[f'{col} Z-Scores'] = (df[col] - df[col].mean()) / df[col].std()
    print(round(df1[f'{col} Z-Scores'].mean(),2))
print(df1.info())
print(df1.head())
print(df1.describe())
print(df.info())
print(df.head())
print(df.describe())
