import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import f_oneway, ttest_ind
from math import comb

# Logger configuration
logger = logging.getLogger(__name__)
RESULTS_PATH = "results"

def load_dataset(path):
    """Loads dataset from CSV file."""
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return None

def standardize(data):
    """Standardizes features using Z-score normalization."""
    return StandardScaler().fit_transform(data)

def our_pca(scaled_data):
    """Performs PCA and returns results for 3 components."""
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(scaled_data)
    df_pca_output = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2', 'PC3'])
    return pca, df_pca_output, pca_results

def scree_plot(scaled_data, threshold):
    """Generates a Scree Plot to visualize variance capture."""
    full_pca = PCA().fit(scaled_data)
    total_var_3 = np.sum(full_pca.explained_variance_ratio_[:3])
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(np.cumsum(full_pca.explained_variance_ratio_)) + 1), 
             np.cumsum(full_pca.explained_variance_ratio_), marker='o', color='b')
    plt.axhline(y=threshold, color='r', label='70% Threshold')
    plt.axhline(y=total_var_3, color='g', linestyle='--', label='Current 3 Components')
    plt.title('Scree Plot: Information Capture')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, '08_scree_plot.png'))
    plt.close()

def k_means_clustering(df_pca_output, pca_results):
    """Clusters patients into 4 profiles using K-Means (Natural Random State)."""
    kmeans = KMeans(n_clusters=4, n_init=10)
    df_pca_output['Cluster'] = kmeans.fit_predict(pca_results)
    return df_pca_output

def clusters_per_assessment(new_path, our_p_value, df_pca_output):
    """
    EXACT duplicate of your original statistical validation output structure.
    Every line matches your clinical research requirements.
    """
    df = pd.read_csv(new_path)
    df = df[df['Diagnosis'] != 0]

    # Preprocessing: Keeping clinical measurements
    bool_cols_to_drop = [col for col in df.columns if len(set(df[col].unique())) < 5]
    df = df.drop(columns=bool_cols_to_drop)
    df = df.drop(columns=["PatientID"], errors='ignore')

    # Aligned Indexing to prevent NaN during concatenation
    df_pca_reset = df_pca_output.reset_index(drop=True)
    df_clinical_reset = df.reset_index(drop=True)

    # EXACT HEADER PRINTING
    logger.info("One-way analysis of variance")
    logger.info("H0: Samples in all groups are drawn from populations with the same mean values.")
    logger.info(f" Our critical P-value (alpha): {our_p_value}")
    logger.info("(We teke this p_value because the data is synthetic...)\n")

    for assessment in ["UPDRS", "MoCA", "FunctionalAssessment"]:
        # Match original df_corr logic
        df_corr = pd.concat([df_pca_reset['Cluster'], df_clinical_reset[assessment]], axis=1).dropna()
        
        # ANOVA
        f_stat, p_val = f_oneway(
            df_corr[df_corr['Cluster'] == 0][assessment],
            df_corr[df_corr['Cluster'] == 1][assessment],
            df_corr[df_corr['Cluster'] == 2][assessment],
            df_corr[df_corr['Cluster'] == 3][assessment]
        )
        
        logger.info(assessment)
        logger.info(f"F-statistic: {round(f_stat, 3)}")
        logger.info(f"P-value: {round(p_val, 3)}")
        
        if p_val >= our_p_value:
            logger.info("Not significant, we cannot reject the null hypothesis.\n")
        else:
            logger.info("Significant! We can reject the null hypothesis.")
            logger.info("We need to do a Post hoc analysis.")
            logger.info("We pick to do 'Pairwise comparisons': Tests all possible pairs.")
            logger.info("Don't forget to counteract the multiple comparisons problem.")
            logger.info("So we are doing the 'Bonferroni correction'")
            
            # Bonferroni step
            num_clusters = len(df_pca_reset['Cluster'].unique())
            our_new_p_value = our_p_value / comb(num_clusters, 2)
            logger.info(f"Our new critical P-value (alpha): {round(our_new_p_value, 3)}")
            
            # Two-tailed step
            our_new_p_value = our_new_p_value / 2
            logger.info("We need to divide this by two because the test is two-tailed.")
            logger.info(f"Our new critical P-value (alpha): {round(our_new_p_value, 3)}")
            
            # Pairwise results
            couples = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
            for couple in couples:
                t_stat, t_p = ttest_ind(
                    df_corr[df_corr['Cluster'] == couple[0]][assessment],
                    df_corr[df_corr['Cluster'] == couple[1]][assessment]
                )
                logger.info(f"{couple[0]},{couple[1]}: {round(t_p, 3)}")
                if t_p >= our_new_p_value:
                    logger.info("Not significant.")
                else:
                    logger.info("Significant!")
            logger.info("\n")
    
    logger.info("One way analysis of variance complete.")