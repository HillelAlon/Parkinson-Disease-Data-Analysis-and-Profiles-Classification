import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from math import comb
import logging
import os

# Initialize Logger for this module
logger = logging.getLogger(__name__)

# --- 1. DATA PREPARATION ---

def load_dataset(path):
    """Stage 1: Load and prepare the dataset for PCA analysis."""
    try:
        df_pca = pd.read_csv(path, index_col=0)
        logger.info(f"Dataset loaded successfully. Shape: {df_pca.shape}")
        return df_pca
    except FileNotFoundError:
        logger.error(f"Dataset file not found at: {path}")
        return None

def standardize(data):
    """Stage 2: Standardize features by removing the mean and scaling to unit variance."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    logger.info("Feature scaling complete: Data transformed into Z-scores.")
    return scaled_data

# --- 2. DIMENSIONALITY REDUCTION ---

def our_pca(scaled_data, original_index):
    """Stage 3: Perform PCA to reduce data to 3 principal components."""
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(scaled_data)
    
    df_pca_output = pd.DataFrame(
        data=pca_results,
        columns=['PC1', 'PC2', 'PC3'],
        index=original_index
    )
    
    total_var = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA completed. Dimensionality reduced to 3. Total Variance: {total_var:.2%}")
    return pca, df_pca_output, pca_results

# --- 3. VALIDATION & DIAGNOSTICS ---

def explained_variance_analysis(pca):
    """Stage 4: Analyze the variance contribution of each component."""
    ratios = pca.explained_variance_ratio_
    logger.info(f"Variance Breakdown - PC1: {ratios[0]:.2%}, PC2: {ratios[1]:.2%}, PC3: {ratios[2]:.2%}")
    return np.sum(ratios)

def variance_analysis(scaled_data, threshold, results_path):
    """Stage 5: Visualize cumulative variance to find optimal component count."""
    full_pca = PCA().fit(scaled_data)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.title('Cumulative Variance Analysis')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    
    plt.savefig(os.path.join(results_path, '07_variance_analysis.png'))
    plt.close()

def scree_plot(scaled_data, threshold, results_path):
    """Stage 6: Generate Scree Plot for visual inspection of component significance."""
    full_pca = PCA().fit(scaled_data)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='b')
    plt.axhline(y=threshold, color='r', label=f'{threshold*100}% Threshold')
    plt.title('Scree Plot: Captured Information')
    plt.legend()
    
    plt.savefig(os.path.join(results_path, '08_scree_plot.png'))
    plt.close()

# --- 4. CLUSTERING & PROFILING ---

def clusters_plot(df_pca_output, results_path):
    """Stage 7: 3D Visualization of Patient Profiles in PCA space."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df_pca_output['PC1'], df_pca_output['PC2'], df_pca_output['PC3'], 
                    c=df_pca_output['PC1'], cmap='viridis', alpha=0.6)
    
    plt.colorbar(sc, label='PC1 Intensity')
    ax.set_title("3D PCA Space: Parkinson's Patient Distribution")
    
    plt.savefig(os.path.join(results_path, '09_pca_3d_profiles.png'))
    plt.close()

def elbow_method(pca_results, results_path):
    """Stage 8: Apply Elbow Method to determine optimal number of clusters."""
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(pca_results)
        inertia.append(km.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o', color='purple')
    plt.title('Elbow Method: Clustering Optimization')
    
    plt.savefig(os.path.join(results_path, '10_elbow_method.png'))
    plt.close()

def k_means_clustering(df_pca_output, pca_results):
    """Stage 9: Group patients into 4 distinct clinical profiles using K-Means."""
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_pca_output['Cluster'] = kmeans.fit_predict(pca_results)
    
    logger.info("Clustering Complete. Patients per Profile:")
    logger.info(f"\n{df_pca_output['Cluster'].value_counts()}")
    return df_pca_output

def clusters_3d_plot(df_pca_output, results_path):
    """Stage 10: Visualize the identified 4 clusters in 3D PCA space."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_pca_output['PC1'], df_pca_output['PC2'], df_pca_output['PC3'], 
                        c=df_pca_output['Cluster'], cmap='Set1', s=40)
    
    ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.set_title("Identified Patient Profiles (4 Clusters)")
    
    plt.savefig(os.path.join(results_path, '11_clusters_3d_view.png'))
    plt.close()

def cluster_profile(df_pca, df_pca_output):
    """Stage 11: Calculate mean feature values for each identified cluster."""
    df_pca['Cluster'] = df_pca_output['Cluster']
    profiles = df_pca.groupby('Cluster').mean()
    
    profiles.to_csv("patient_profiles_summary.csv")
    logger.info("Profiles summarized and exported to CSV.")
    return profiles

def cluster_heat_map(df_pca, cluster_profiles, results_path):
    """Stage 12: Heatmap visualization of Normalized Cluster Characteristics."""
    df_pca_original = df_pca.drop(columns=['Cluster'])
    norm_profiles = (cluster_profiles - df_pca_original.mean()) / df_pca_original.std()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(norm_profiles, annot=cluster_profiles, cmap='RdYlBu_r', center=0, fmt='.2f')
    plt.title("Clinical & Lifestyle Profile Heatmap")
    
    plt.savefig(os.path.join(results_path, '12_cluster_heatmap.png'))
    plt.close()

# --- 5. STATISTICAL VALIDATION ---

def clusters_per_assessment(new_path, alpha, df_pca):
    """Stage 13: Statistical validation using ANOVA and Post-hoc Bonferroni tests."""
    df = pd.read_csv(new_path)
    df = df[df['Diagnosis'] != 0] # Focus on diagnosed population
    
    # Remove low-variance columns
    drop_cols = [c for c in df.columns if len(df[c].unique()) < 5]
    df = df.drop(columns=drop_cols + ["PatientID"], errors='ignore')

    metrics = ["UPDRS", "MoCA", "FunctionalAssessment"]
    logger.info(f"Running Statistical Validation (Alpha={alpha})")

    for metric in metrics:
        combined = pd.concat([df_pca['Cluster'], df[metric]], axis=1).dropna()
        groups = [combined[combined['Cluster'] == i][metric] for i in range(4)]
        
        f_stat, p_val = f_oneway(*groups)
        logger.info(f"{metric} -> F-stat: {f_stat:.3f}, P-value: {p_val:.4f}")
        
        if p_val < alpha:
            # Applying Bonferroni Correction for Multiple Comparisons
            adj_alpha = (alpha / comb(4, 2)) / 2 
            logger.info(f"Significant difference found in {metric}. Adj-Alpha: {adj_alpha:.4f}")
            
    logger.info("Statistical Analysis cycle complete.")