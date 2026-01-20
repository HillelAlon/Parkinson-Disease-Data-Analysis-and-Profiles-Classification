import os
import sys
import logging
import pandas as pd

# --- 1. Path Management ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'analysis'))
sys.path.append(os.path.join(BASE_DIR, 'clusteringTA'))

# --- 2. Module Imports ---
from cleaning_data.data_cleaning import load_and_clean_data
import functions_analysis as clinical_utils
import bonus_analysis as bonus_utils
from clusteringTA.pca_functions import * # --- 3. Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
RAW_DATA = os.path.join(BASE_DIR, 'data', 'parkinsons_disease_data.csv')
CLEANED_DATA = os.path.join(BASE_DIR, 'data', 'parkinsons_cleaned.csv')
PCA_DATA = os.path.join(BASE_DIR, 'data', 'parkinsons_lifestyle_clinical_for_PCA.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def main():
    logger.info("=== STARTING MASTER PIPELINE ===")
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    # PHASE A: CLEANING
    logger.info("--- Phase A: Cleaning ---")
    df_clean = load_and_clean_data(RAW_DATA, 'PatientID', ['DoctorInCharge'])
    if df_clean is None: return
    df_clean.to_csv(CLEANED_DATA, index=True)

    # PHASE B: CLINICAL ANALYSIS
    logger.info("--- Phase B: Clinical Analysis ---")
    clinical_utils.run_full_analysis_pipeline(df_clean, 'Diagnosis', 1, ['UPDRS', 'MoCA'], RESULTS_DIR)

    # PHASE C: ALGORITHM (PCA & CLUSTERING)
    logger.info("--- Phase C: Algorithm ---")
    
    # Pre-processing for PCA
    df_pca_prep = df_clean[df_clean['Diagnosis'] != 0].copy()
    # Drop categorical columns
    cols_to_drop = [c for c in df_pca_prep.columns if len(df_pca_prep[c].unique()) < 5]
    df_pca_prep = df_pca_prep.drop(columns=cols_to_drop + ["UPDRS", "MoCA", "FunctionalAssessment"], errors='ignore')
    df_pca_prep.to_csv(PCA_DATA, index=True)

    # Now we call your functions in order!
    df_pca_loaded = load_dataset(PCA_DATA) # This comes from pca_functions.py
    if df_pca_loaded is not None:
        # Step 1: Scale
        scaled_data = standardize(df_pca_loaded)
        # Step 2: PCA
        pca_model, df_pca_output, pca_results = our_pca(scaled_data)
        # Step 3: Variance & Plots
        explained_variance_analysis(pca_model)
        variance_analysis(scaled_data, 0.7)
        scree_plot(scaled_data, 0.7)
        clusters_plot(df_pca_output)
        
        # Step 4: Clustering
        elbow_method(df_pca_output)
        df_clustered = k_means_clustering(df_pca_output)
        clusters_3d_plot(df_clustered)
        
        # Step 5: Profiles
        profiles = cluster_profile(df_pca_loaded, df_clustered)
        cluster_heat_map(df_pca_loaded, profiles)

    # PHASE D: STATISTICAL TEST
    logger.info("--- Phase D: Statistics ---")
    clusters_per_assessment(CLEANED_DATA, 0.2, df_pca_loaded)

    logger.info(f"=== ALL DONE! Results in: {RESULTS_DIR} ===")

if __name__ == "__main__":
    main()