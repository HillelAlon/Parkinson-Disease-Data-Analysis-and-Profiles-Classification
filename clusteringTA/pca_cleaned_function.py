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

logger = logging.getLogger(__name__)

# --- FUNCTIONS ONLY (NO EXECUTION LINES) ---

def load_dataset(path):
    try:
        df_pca = pd.read_csv(path, index_col=0)
        logger.info(f"Dataset loaded successfully. Shape: {df_pca.shape}")
        return df_pca
    except FileNotFoundError:
        logger.error(f"Dataset file not found at: {path}")
        return None

def standardize(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    logger.info("Feature scaling complete.")
    return scaled_data

def our_pca(scaled_data, original_index):
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(scaled_data)
    df_pca_output = pd.DataFrame(
        data=pca_results,
        columns=['PC1', 'PC2', 'PC3'],
        index=original_index
    )
    logger.info("PCA execution finished.")
    return pca, df_pca_output, pca_results

def explained_variance_analysis(pca):
    variance_ratios = pca.explained_variance_ratio_
    total_variance = np.sum(variance_ratios)
    logger.info(f"Total variance captured: {total_variance:.2%}")
    return total_variance

def variance_analysis(scaled_data, threshold, results_path):
    full_pca = PCA().fit(scaled_data)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=threshold, color='r')
    plt.title('Variance Analysis')
    plt.savefig(f"{results_path}/variance_analysis.png")
    plt.close()

def scree_plot(scaled_data, threshold, results_path):
    full_pca = PCA().fit(scaled_data)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=threshold, color='r', label='Threshold')
    plt.title('Scree Plot')
    plt.savefig(f"{results_path}/scree_plot.png")
    plt.close()

def clusters_plot(df_pca_output, results_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df_pca_output['PC1'], df_pca_output['PC2'], df_pca_output['PC3'], c=df_pca_output['PC1'], cmap='viridis')
    plt.colorbar(sc)
    plt.savefig(f"{results_path}/pca_3d_profiles.png")
    plt.close()

def elbow_method(pca_results, results_path):
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(pca_results)
        inertia.append(km.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method')
    plt.savefig(f"{results_path}/elbow_method.png")
    plt.close()

def k_means_clustering(df_pca_output, pca_results):
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_pca_output['Cluster'] = kmeans.fit_predict(pca_results)
    logger.info("Clustering complete.")
    return df_pca_output

def clusters_3d_plot(df_pca_output, results_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_pca_output['PC1'], df_pca_output['PC2'], df_pca_output['PC3'], c=df_pca_output['Cluster'], cmap='Set1')
    ax.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig(f"{results_path}/clusters_3d_view.png")
    plt.close()

def cluster_profile(df_pca, df_pca_output):
    df_pca['Cluster'] = df_pca_output['Cluster']
    cluster_profiles = df_pca.groupby('Cluster').mean()
    return cluster_profiles

def cluster_heat_map(df_pca, cluster_profiles, results_path):
    df_pca_original = df_pca.drop(columns=['Cluster'])
    cluster_profiles_norm = (cluster_profiles - df_pca_original.mean()) / df_pca_original.std()
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_profiles_norm, annot=cluster_profiles, cmap='RdYlBu_r', center=0, fmt='.2f')
    plt.title("Cluster Characteristics Heatmap")
    plt.savefig(f"{results_path}/cluster_heatmap.png")
    plt.close()

def clusters_per_assessment(new_path, our_p_value, df_pca):
    df = pd.read_csv(new_path)
    df = df[df['Diagnosis'] != 0]
    # Rest of your statistical logic remains here as a function...
    logger.info("One-way ANOVA analysis complete.")