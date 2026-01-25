"""
=== PARKINSON'S RESEARCH PROJECT ===
Hypothesis: Parkinson's symptoms follow unique, dissociated trajectories per patient.
"""


# Create Logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Import all functions
from data_cleaning_functions import load_and_clean_data
from analysis_functions import *
from clustering_functions import *


"""
[Step 0] Cleaning Raw Data
"""
#Raw cleaning function
file_path = 'data/parkinsons_disease_data.csv'
index_col = 'PatientID'
columns_to_drop = ['DoctorInCharge']
cleand_data_path = "data/parkinsons_cleaned.csv"
# ---Cleaning data function ---
    #GET - csv file, string that represent index column, list of irrelevant Columns.
    #CHECKS - if all the inputs are right type. ELSE - value error
    #RETURN - data frame without the irrelevant Columns, index column set as index column, without duplicates
df_clean = load_and_clean_data(file_path, index_col, columns_to_drop)
df_clean.to_csv(cleand_data_path, index=True)
print(df_clean.head())


"""
[Step 1] Clinical Analysis Pipeline
"""
#1 load dataset
df = load_dataset(cleand_data_path)

#2 plot_global_heatmap
if df is not None:
    plot_global_heatmap(df)

#3 plot_feature_correlation_profile
if 'df' in locals() and df is not None and not df.empty:
    plot_feature_correlation_profile(df, 'Diagnosis')

#4 extract_sick_population
if df is not None:
    sick_df = extract_sick_population(df)

#5 plot_sick_population_heatmap
if 'sick_df' in locals() and not sick_df.empty:
    plot_sick_population_heatmap(sick_df)

#6 plot_severity_distributions
plot_severity_distributions(sick_df)

#7 analyze_metric_dissociation
if 'sick_df' in locals() and not sick_df.empty:
    analyze_metric_dissociation(sick_df)
else:
    logger.error("Analysis failed: 'sick_df' is not defined or empty.")


"""
[Step 2] Running PCA & Unsupervised Clustering...
"""
#1 Make a new df without nominal and ordinal columns
df_pca = cleand_df_to_pca(cleand_data_path)

#2 Transformation the df to Z-Scores values
if df_pca is not None:
    scaled_data = standardize(df_pca)

#3 PCA: from 12D to new combined 3D
if scaled_data is not None:
    pca,df_pca_output,pca_results = our_pca(scaled_data,df_pca)
total_variance = explained_variance_analysis(pca)
threshold = 0.7
variance_analysis(scaled_data,threshold)
scree_plot(scaled_data,threshold, total_variance)
clusters_plot(df_pca_output)

#4 Clustering
elbow_method(df_pca_output,pca_results)
k_means_clustering(df_pca_output,pca_results)
clusters_3d_plot(df_pca_output)
cluster_profiles = cluster_profile(df_pca,df_pca_output)
cluster_heat_map(df_pca,cluster_profiles)


"""
[Step 3] Statistical Cluster Validation (ANOVA Results):
"""
#Compare between the clustering (Age, Lifestyle and Clinical Measures) and every assessment (UPDRS,MoCA,FunctionalAssessment)
clusters_per_assessment(cleand_data_path,df_pca)


"""
[Step 4] Advanced Research Modules (bonus)
"""
#1 poisson_analysis
if 'sick_df' in locals():
    run_poisson_analysis(sick_df)

#2 gatekeeper_analysis
if 'sick_df' in locals():
    run_gatekeeper_analysis(sick_df)