from research_archive.clusteringTA.pca_functions import *

#1 Make a new df without nominal and ordinal columns.
cleand_data_path = "../../data/parkinsons_cleaned.csv"
df_pca = cleand_df_to_pca(cleand_data_path)

#2. Transformation the df to Z-Scores values.
if df_pca is not None:
    scaled_data = standardize(df_pca)

#3. PCA: from 12D to new combined 3D. (sklearn.decomposition.PCA)
if scaled_data is not None:
    pca,df_pca_output,pca_results = our_pca(scaled_data,df_pca)
total_variance = explained_variance_analysis(pca)
threshold = 0.7
variance_analysis(scaled_data,threshold)
scree_plot(scaled_data,threshold, total_variance)
clusters_plot(df_pca_output)

#4. Clustering.
elbow_method(df_pca_output,pca_results)
k_means_clustering(df_pca_output,pca_results)
clusters_3d_plot(df_pca_output)
cluster_profiles = cluster_profile(df_pca,df_pca_output)
cluster_heat_map(df_pca,cluster_profiles)

#5. Compare between the clustering (Age, Lifestyle and Clinical Measures) and every assessment (UPDRS,MoCA,FunctionalAssessment).
clusters_per_assessment(cleand_data_path,df_pca)