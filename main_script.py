import os
import sys
import logging
import pandas as pd
import numpy as np

# --- 1. PATH MANAGEMENT SYSTEM ---
# Ensures all project subdirectories are accessible for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'analysis'))
sys.path.append(os.path.join(BASE_DIR, 'clusteringTA'))
sys.path.append(os.path.join(BASE_DIR, 'cleaning_data'))

# --- 2. MODULE IMPORTS ---
try:
    from cleaning_data.data_cleaning import load_and_clean_data
    import functions_analysis as clinical_utils
    import bonus_analysis as bonus_utils
    # Importing from your specialized cleaned file
    from clusteringTA.pca_cleaned_function import *
    MODULES_READY = True
except ImportError as e:
    print(f"IMPORT ERROR: Check __init__.py files in subfolders. Details: {e}")
    MODULES_READY = False

# --- 3. LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 4. GLOBAL PATH CONFIGURATIONS ---
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'parkinsons_disease_data.csv')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'parkinsons_cleaned.csv')
PCA_DATA_PATH = os.path.join(BASE_DIR, 'data', 'parkinsons_lifestyle_clinical_for_PCA.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Global Constants
DIAGNOSIS_COL = 'Diagnosis'
SICK_CODE = 1
CLINICAL_METRICS = ['UPDRS', 'MoCA', 'FunctionalAssessment']

def main():
    if not MODULES_READY:
        return
    
    logger.info("=== MASTER RESEARCH PIPELINE: FULL SYSTEM CHECK & EXECUTION ===")

    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # --- PHASE A: DATA PREPROCESSING ---
    logger.info("--- Phase A: Data Cleaning and Standardization ---")
    df_clean = load_and_clean_data(RAW_DATA_PATH, 'PatientID', ['DoctorInCharge'])
    if df_clean is not None:
        df_clean.to_csv(CLEANED_DATA_PATH, index=True)
        logger.info(f"Cleaned dataset ready at: {CLEANED_DATA_PATH}")
    else:
        logger.error("Data preprocessing failed. Terminating.")
        return

    # --- PHASE B: CLINICAL ANALYSIS (7 STAGES) ---
    logger.info("--- Phase B: Executing 7 Stages of Clinical Analysis ---")
    
    # Stage 2 & 3: Global Heatmap and Z-Score Profiles
    clinical_utils.plot_global_heatmap(df_clean, RESULTS_DIR)
    clinical_utils.plot_feature_profile(df_clean, DIAGNOSIS_COL, RESULTS_DIR)
    
    # Stage 4: Population Filtering
    sick_df = clinical_utils.extract_sick_population(df_clean, DIAGNOSIS_COL, SICK_CODE)
    
    # Stage 5, 6 & 7: Masked Heatmaps, Distributions, and Dissociation
    if not sick_df.empty:
        clinical_utils.plot_sick_population_heatmap(sick_df, DIAGNOSIS_COL, RESULTS_DIR)
        clinical_utils.plot_severity_distributions(sick_df, CLINICAL_METRICS, RESULTS_DIR)
        clinical_utils.analyze_metric_dissociation(sick_df, CLINICAL_METRICS, RESULTS_DIR)
    
    logger.info("Phase B complete: All clinical visualization stages executed.")

    # --- PHASE C: PCA & UNSUPERVISED PROFILING ---
    logger.info("--- Phase C: Algorithmic PCA & Clustering Execution ---")
    
    # Prep data for PCA (Filtering Numerical Features)
    df_pca_prep = df_clean[df_clean[DIAGNOSIS_COL] != 0].copy()
    bool_cols = [c for c in df_pca_prep.columns if len(df_pca_prep[c].unique()) < 5]
    df_pca_prep = df_pca_prep.drop(columns=bool_cols + ["PatientID"] + CLINICAL_METRICS, errors='ignore')
    df_pca_prep.to_csv(PCA_DATA_PATH, index=True)

    df_pca_loaded = load_dataset(PCA_DATA_PATH)
    if df_pca_loaded is not None:
        scaled_data = standardize(df_pca_loaded)
        # Passing original index to the PCA function
        pca_model, df_pca_output, pca_results = our_pca(scaled_data, df_pca_loaded.index)
        
        # Diagnostics, Scree Plots and 3D PCA
        explained_variance_analysis(pca_model)
        variance_analysis(scaled_data, 0.7, RESULTS_DIR)
        scree_plot(scaled_data, 0.7, RESULTS_DIR)
        clusters_plot(df_pca_output, RESULTS_DIR)
        
        # K-Means Clustering and Profile Identification
        elbow_method(pca_results, RESULTS_DIR)
        df_clustered = k_means_clustering(df_pca_output, pca_results)
        clusters_3d_plot(df_clustered, RESULTS_DIR)
        
        # Final Profiling Heatmaps
        profiles = cluster_profile(df_pca_loaded, df_clustered)
        cluster_heat_map(df_pca_loaded, profiles, RESULTS_DIR)

    # --- PHASE D: STATISTICAL VALIDATION ---
    logger.info("--- Phase D: Cluster-Assessment ANOVA Validation ---")
    clusters_per_assessment(CLEANED_DATA_PATH, 0.2, df_pca_loaded)

    # --- PHASE E: BONUS RESEARCH MODELS (NOW COMPLETE) ---
    logger.info("--- Phase E: Advanced Bonus Experimental Modules ---")
    if not sick_df.empty:
        # 1. Symptom Aggregation (Poisson)
        bonus_utils.run_poisson_analysis(sick_df, RESULTS_DIR)
        
        # 2. Key Symptom Identification (Gatekeeper)
        bonus_utils.run_gatekeeper_analysis(sick_df, RESULTS_DIR)
        
        logger.info("Phase E complete: All bonus research modules executed.")

    logger.info(f"=== MASTER PIPELINE COMPLETE: All outputs stored in {RESULTS_DIR} ===")

if __name__ == "__main__":
    main()