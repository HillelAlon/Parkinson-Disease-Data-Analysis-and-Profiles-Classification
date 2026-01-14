import pandas as pd
import logging
from clusteringTA.pca_functions import *

# def cleand df to pca
def cleand_df_to_pca(cleand_data_path):
    df = pd.read_csv(cleand_data_path)
    # Keep only the sick patients in the data frame.
    df = df[df['Diagnosis'] != 0]
    # Keep only age, lifestyle and clinical measurements categories
    bool_cols_to_drop = [col for col in df.columns if len(set(df[col].unique())) < 5] #Remove boolean and categorical columns
    df = df.drop(columns=bool_cols_to_drop)
    df = df.drop(columns=["PatientID","UPDRS", "MoCA", "FunctionalAssessment"]) #Remove categories: ID and assessments
    df.to_csv("data/parkinsons_lifestyle_clinical_for_PCA.csv",index=True)

# Make a new df without nominal and ordinal columns.
cleand_data_path = "data/parkinsons_cleaned.csv"
cleand_df_to_pca(cleand_data_path)


#1 Data and loading
path = '../data/parkinsons_lifestyle_clinical_for_PCA.csv'
df_pca = load_dataset(path)

#2. Transformation the df to Z-Scores values.
if df_pca is not None:
    scaled_data = standardize(df_pca)

#3. PCA: from 12D to new combined 3D. (sklearn.decomposition.PCA)
if scaled_data is not None:
    pca,df_pca_output,pca_results = our_pca(scaled_data)
total_variance = explained_variance_analysis(pca)
threshold = 0.7
variance_analysis(scaled_data,threshold)
scree_plot(scaled_data,threshold)
clusters_plot(df_pca_output)

#4. Clustering.
elbow_method(df_pca_output)
k_means_clustering(df_pca_output)
cluster_profiles = cluster_profile(df_pca,df_pca_output)
cluster_heat_map(df_pca,cluster_profiles)

#5. Compare between the clustering (Age, Lifestyle and Clinical Measures) and every assessment (UPDRS,MoCA,FunctionalAssessment).
new_path = "../data/parkinsons_cleaned.csv"
our_p_value = 0.2
clusters_per_assessment(new_path,our_p_value,df_pca)