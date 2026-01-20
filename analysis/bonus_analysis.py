import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from scipy.stats import poisson

# Professional logging setup matching the main module
logger = logging.getLogger(__name__)

def run_poisson_analysis(data, save_path):
    """
    Compares actual symptom clustering against a random Poisson model.
    Saves the resulting plot to the results directory.
    """
    logger.info("Running Poisson Distribution analysis.")
    symptom_list = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 
                    'SpeechProblems', 'SleepDisorders', 'Constipation']
    
    # Calculation logic
    data['SymptomCount'] = data[symptom_list].sum(axis=1)
    mu = data['SymptomCount'].mean()
    
    actual = data['SymptomCount'].value_counts(normalize=True).sort_index()
    theoretical = [poisson.pmf(k, mu) for k in range(len(actual))]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(actual.index, actual.values, alpha=0.5, label='Observed Data', color='grey')
    plt.plot(actual.index, theoretical, 'ro-', linewidth=2, label='Poisson (Random Theory)')
    plt.title(f'Symptom Aggregation vs Random Chance (Mean: {mu:.2f})')
    plt.legend()
    
    # Save Logic
    file_path = os.path.join(save_path, 'bonus_01_poisson_distribution.png')
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Poisson analysis plot saved to: {file_path}")

def run_gatekeeper_analysis(data, save_path):
    """
    Identifies which symptoms correlate with higher overall symptom burden.
    Fixes Seaborn palette warning and saves output as PNG.
    """
    logger.info("Executing Gatekeeper Analysis (Tremor Paradox).")
    symptoms = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 
                'SpeechProblems', 'SleepDisorders', 'Constipation']
    results = []

    for s in symptoms:
        others = [sym for sym in symptoms if sym != s]
        impact = data[data[s] == 1][others].sum(axis=1).mean() - data[data[s] == 0][others].sum(axis=1).mean()
        results.append({'Symptom': s, 'Impact': impact})

    impact_df = pd.DataFrame(results).sort_values(by='Impact')
    
    # Plotting with palette fix
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in impact_df['Impact']]
    sns.barplot(data=impact_df, x='Impact', y='Symptom', hue='Symptom', palette=colors, legend=False)
    plt.title('Gatekeeper Effect: Impact of Specific Symptoms on Overall Burden')
    
    # Save Logic
    file_path = os.path.join(save_path, 'bonus_02_gatekeeper_analysis.png')
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Gatekeeper analysis plot saved to: {file_path}")

