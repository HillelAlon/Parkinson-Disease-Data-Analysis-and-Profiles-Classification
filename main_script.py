import os
import sys
import logging
import platform
import subprocess
import pandas as pd
import numpy as np
import matplotlib

# Set matplotlib to non-interactive mode
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 1. LOGGING CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

log_path = os.path.join(RESULTS_DIR, '00_research_log_and_conclusions.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler(log_path, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# --- 2. MODULAR IMPORTS ---
sys.path.extend([BASE_DIR, os.path.join(BASE_DIR, 'analysis'), os.path.join(BASE_DIR, 'clusteringTA')])
from cleaning_data.data_cleaning import load_and_clean_data
import functions_analysis as clinical_utils
import bonus_analysis as bonus_utils
import clusteringTA.pca_cleaned_function as pca_utils

# File Paths
RAW_DATA = os.path.join(BASE_DIR, 'data', 'parkinsons_disease_data.csv')
CLEAN_DATA = os.path.join(BASE_DIR, 'data', 'parkinsons_cleaned.csv')
PCA_DATA = os.path.join(BASE_DIR, 'data', 'parkinsons_lifestyle_clinical_for_PCA.csv')

def open_folder(path):
    """Universal folder opener for Windows, Mac, and Linux."""
    try:
        if platform.system() == "Windows": os.startfile(path)
        elif platform.system() == "Darwin": subprocess.Popen(["open", path])
        else: subprocess.Popen(["xdg-open", path])
    except: pass

def main():
    # HEADER SUMMARY - EXACT MATCH
    logger.info("=== PARKINSON'S RESEARCH PROJECT: AUTOMATED SUMMARY ===")
    logger.info("Hypothesis: Parkinson's symptoms follow unique, dissociated trajectories per patient.\n")

    # [Step 1]
    logger.info("[Step 1] Cleaning Raw Data...")
    df_clean = load_and_clean_data(RAW_DATA, 'PatientID', ['DoctorInCharge'])
    df_clean.to_csv(CLEAN_DATA, index=True)

    # [Step 2]
    logger.info("\n[Step 2] Executing Clinical Analysis Pipeline...")
    clinical_utils.plot_global_heatmap(df_clean, RESULTS_DIR)
    # Removing the redundant logger.info here as the function already reports it
    sick_df = clinical_utils.extract_sick_population(df_clean, 'Diagnosis', 1)
    clinical_utils.analyze_metric_dissociation(sick_df, ['UPDRS', 'MoCA', 'FunctionalAssessment'], RESULTS_DIR)

    # [Step 3]
    logger.info("\n[Step 3] Running PCA & Unsupervised Clustering...")
    df_pca_loaded = pca_utils.load_dataset(PCA_DATA)
    if df_pca_loaded is not None:
        logger.info(f"Dataset loaded successfully. Shape: {df_pca_loaded.shape}")
        scaled = pca_utils.standardize(df_pca_loaded)
        logger.info("Feature scaling complete: Data transformed into Z-scores.")
        
        pca_model, df_pca_out, pca_res = pca_utils.our_pca(scaled)
        var = pca_model.explained_variance_ratio_
        logger.info(f"PCA completed. Dimensionality reduced to 3. Total Variance: {np.sum(var):.2%}")
        logger.info(f"Variance Breakdown - PC1: {var[0]:.2%}, PC2: {var[1]:.2%}, PC3: {var[2]:.2%}")
        
        df_pca_out = pca_utils.k_means_clustering(df_pca_out, pca_res)
        logger.info("Clustering Complete. Patients per Profile:")
        logger.info(f"\n{df_pca_out['Cluster'].value_counts()}")
        logger.info("\nProfiles summarized and exported to CSV.")

    # [Step 4]
    logger.info("\n[Step 4] Statistical Cluster Validation (ANOVA Results):")
    pca_utils.clusters_per_assessment(CLEAN_DATA, 0.2, df_pca_out)

    # [Step 5]
    logger.info("\n[Step 5] Advanced Research Modules...")
    bonus_utils.run_poisson_analysis(sick_df, RESULTS_DIR)
    bonus_utils.run_gatekeeper_analysis(sick_df, RESULTS_DIR)

    logger.info("\n=== PIPELINE COMPLETE: ALL FINDINGS SAVED IN RESULTS FOLDER ===")
    open_folder(RESULTS_DIR)

if __name__ == "__main__":
    main()