import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Professional logging setup
logger = logging.getLogger(__name__)

def load_dataset(path):
    """Stage 1: Data Import."""
    try:
        data = pd.read_csv(path, index_col='PatientID')
        logger.info(f"Dataset loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def plot_global_heatmap(data, save_path):
    """
    Stage 2: Global Correlation Heatmap.
    Uses a mask to remove the redundant upper triangle for better readability.
    """
    logger.info("Generating masked global correlation heatmap.")
    corr_matrix = data.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(16, 10))
    # Apply the mask here
    sns.heatmap(corr_matrix, mask=mask, cmap='RdYlBu_r', center=0, 
                linewidths=0.5, annot=False)
    
    plt.title('Global Feature Correlation Landscape (Masked)', fontsize=16, fontweight='bold')
    
    file_path = os.path.join(save_path, '01_global_heatmap.png')
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Global heatmap saved to: {file_path}")

def plot_feature_profile(data, diagnosis_col, save_path):
    """Stage 3: Correlation Profile for Diagnosis."""
    logger.info(f"Generating correlation profile for: {diagnosis_col}")
    target_corrs = data.corr()[diagnosis_col].drop(diagnosis_col).sort_values()
    plt.figure(figsize=(16, 8))
    target_corrs.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Correlation Profile: {diagnosis_col}', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_path, '02_feature_profile.png'))
    plt.close()

def extract_sick_population(data, diagnosis_col, sick_value):
    """Stage 4: Population Filtering."""
    sick_subset = data[data[diagnosis_col] == sick_value].copy()
    logger.info(f"Filtered {len(sick_subset)} diagnosed patients.")
    return sick_subset


def plot_sick_population_heatmap(sick_data, diagnosis_col, save_path):
    """
    Stage 5: Internal Dynamics Heatmap.
    Visualizes correlations within the sick population, removing redundant tiles.
    """
    logger.info("Generating masked internal heatmap for sick population.")
    # Drop the diagnosis column as it's constant (all are sick)
    plot_df = sick_data.drop(columns=[diagnosis_col], errors='ignore')
    corr_matrix = plot_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(18, 12))
    # Apply the mask here
    sns.heatmap(corr_matrix, mask=mask, cmap='RdYlBu_r', center=0, 
                linewidths=0.5, annot=False)
    
    plt.title('Internal Disease Dynamics Map (Masked)', fontsize=18, fontweight='bold')
    
    file_path = os.path.join(save_path, '03_sick_population_heatmap.png')
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Sick population heatmap saved to: {file_path}")



def plot_severity_distributions(sick_data, metrics, save_path):
    """Stage 6: Clinical Metric Distributions."""
    logger.info(f"Plotting distributions for: {metrics}")
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.histplot(sick_data[metric], kde=True, color='purple')
        plt.title(f'{metric} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '04_severity_distributions.png'))
    plt.close()

def analyze_metric_dissociation(sick_data, metrics, save_path):
    """Stage 7: Dissociation & Driver Analysis (Saves 2 Plots)."""
    logger.info("Executing clinical metric dissociation analysis.")
    
    # Plot A: Inter-metric correlation
    plt.figure(figsize=(8, 4))
    sns.heatmap(sick_data[metrics].corr(), annot=True, cmap='Blues', center=0)
    plt.title('Inter-Metric Correlation (Checking Dissociation)')
    plt.savefig(os.path.join(save_path, '05_metric_correlation.png'))
    plt.close()

    # Plot B: Absolute Impact Comparison
    features = [c for c in sick_data.columns if c not in metrics and c != 'Diagnosis']
    comparison = pd.DataFrame({m: sick_data[features + [m]].corr()[m].drop(m).abs() for m in metrics})
    plt.figure(figsize=(12, 8))
    sns.heatmap(comparison.sort_values(by=metrics[0], ascending=False).head(15), annot=True, cmap='YlGnBu')
    plt.title('Absolute Feature Impact Comparison')
    plt.savefig(os.path.join(save_path, '06_feature_impact_comparison.png'))
    plt.close()

def run_full_analysis_pipeline(data, diagnosis_col, sick_value, metrics, save_path):
    """Main Orchestrator: Calls ALL 7 research stages."""
    logger.info(">>> Starting Full Clinical Analysis Pipeline <<<")
    
    plot_global_heatmap(data, save_path)
    plot_feature_profile(data, diagnosis_col, save_path)
    
    sick_df = extract_sick_population(data, diagnosis_col, sick_value)
    
    if not sick_df.empty:
        plot_sick_population_heatmap(sick_df, diagnosis_col, save_path)
        plot_severity_distributions(sick_df, metrics, save_path)
        analyze_metric_dissociation(sick_df, metrics, save_path)
        
    logger.info(">>> Full Clinical Analysis Pipeline Finished <<<")