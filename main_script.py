import os
import sys
import logging
import subprocess
import platform

# --- SYSTEM & LOGGING SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

# Configure logging to show in console AND save to a numbered text file 
log_file = os.path.join(RESULTS_DIR, '00_research_log_and_conclusions.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# System paths for modular imports 
sys.path.extend([BASE_DIR, os.path.join(BASE_DIR, 'analysis'), os.path.join(BASE_DIR, 'clusteringTA')])
from cleaning_data.data_cleaning import load_and_clean_data
import functions_analysis as clinical_utils
import bonus_analysis as bonus_utils
from clusteringTA.pca_cleaned_function import *

# Constants [cite: 53]
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'parkinsons_disease_data.csv')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'parkinsons_cleaned.csv')
PCA_DATA_PATH = os.path.join(BASE_DIR, 'data', 'parkinsons_lifestyle_clinical_for_PCA.csv')

def open_results_folder(path):
    """Automatically opens the results folder based on OS[cite: 35]."""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin": # macOS
        subprocess.Popen(["open", path])
    else: # Linux
        subprocess.Popen(["xdg-open", path])

def main():
    logger.info("=== PARKINSON'S RESEARCH PROJECT: AUTOMATED SUMMARY ===")
    logger.info("Hypothesis: Parkinson's symptoms follow unique, dissociated trajectories per patient.\n")

    # PHASE A: PREPROCESSING
    logger.info("[Step 1] Cleaning Raw Data...")
    df_clean = load_and_clean_data(RAW_DATA_PATH, 'PatientID', ['DoctorInCharge'])
    df_clean.to_csv(CLEANED_DATA_PATH, index=True)

    # PHASE B: CLINICAL ANALYSIS (Numbered 01-06)
    logger.info("\n[Step 2] Executing Clinical Analysis Pipeline...")
    clinical_utils.plot_global_heatmap(df_clean, RESULTS_DIR) # Saves as 01_
    clinical_utils.plot_feature_profile(df_clean, 'Diagnosis', RESULTS_DIR) # Saves as 02_
    
    sick_df = clinical_utils.extract_sick_population(df_clean, 'Diagnosis', 1)
    clinical_utils.plot_sick_population_heatmap(sick_df, 'Diagnosis', RESULTS_DIR) # 03_
    clinical_utils.plot_severity_distributions(sick_df, ['UPDRS', 'MoCA', 'FunctionalAssessment'], RESULTS_DIR) # 04_
    clinical_utils.analyze_metric_dissociation(sick_df, ['UPDRS', 'MoCA', 'FunctionalAssessment'], RESULTS_DIR) # 05_ & 06_

    # PHASE C: ALGORITHMIC PCA & CLUSTERING (Numbered 07-12)
    logger.info("\n[Step 3] Running PCA & Unsupervised Clustering...")
    df_pca_loaded = load_dataset(PCA_DATA_PATH)
    if df_pca_loaded is not None:
        scaled_data = standardize(df_pca_loaded)
        pca_model, df_pca_output, pca_results = our_pca(scaled_data, df_pca_loaded.index)
        
        explained_variance_analysis(pca_model) # Output will be saved in the text log
        variance_analysis(scaled_data, 0.7, RESULTS_DIR) # 07_
        scree_plot(scaled_data, 0.7, RESULTS_DIR) # 08_
        clusters_plot(df_pca_output, RESULTS_DIR) # 09_
        
        elbow_method(pca_results, RESULTS_DIR) # 10_
        df_clustered = k_means_clustering(df_pca_output, pca_results)
        clusters_3d_plot(df_clustered, RESULTS_DIR) # 11_
        
        profiles = cluster_profile(df_pca_loaded, df_clustered)
        cluster_heat_map(df_pca_loaded, profiles, RESULTS_DIR) # 12_

    # PHASE D: STATISTICAL VALIDATION (ANOVA)
    logger.info("\n[Step 4] Statistical Cluster Validation (ANOVA Results):")
    clusters_per_assessment(CLEANED_DATA_PATH, 0.2, df_pca_loaded) # Findings logged to text file

    # PHASE E: BONUS MODULES (Numbered 13-14)
    logger.info("\n[Step 5] Advanced Research Modules...")
    bonus_utils.run_poisson_analysis(sick_df, RESULTS_DIR) # 13_
    bonus_utils.run_gatekeeper_analysis(sick_df, RESULTS_DIR) # 14_

    logger.info("\n=== PIPELINE COMPLETE: ALL FINDINGS SAVED IN RESULTS FOLDER ===")
    open_results_folder(RESULTS_DIR)

if __name__ == "__main__":
    main()