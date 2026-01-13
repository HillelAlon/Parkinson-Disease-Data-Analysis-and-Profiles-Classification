import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy.stats import poisson

# --- LOGGING SETUP ---
# Standard logging configuration to track progress and debug errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. DATA LOADING FUNCTION ---

def load_dataset(path):
    """
    Loads the cleaned dataset from a CSV file.
    Returns an empty DataFrame if the file is missing to prevent crashes.
    """
    try:
        data = pd.read_csv(path, index_col='PatientID')
        logger.info(f"Successfully loaded data. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame() # Return empty table instead of None

# --- 2. GLOBAL VISUALIZATION FUNCTIONS (General Population) ---

def plot_global_heatmap(data):
    """
    Generates a correlation heatmap for all features in the dataset.
    Helps identify broad relationships between lifestyle and clinical data.
    """
    if data.empty: 
        logger.warning("Empty data provided to plot_global_heatmap.")
        return
        
    logger.info("Generating global correlation heatmap.")
    corr_matrix = data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdYlBu_r', center=0, linewidths=0.5)
    plt.title('Global Feature Correlation Landscape', fontsize=16, fontweight='bold')
    plt.show()

def plot_diagnosis_correlations(data):
    """
    Visualizes correlations between features and the Diagnosis variable.
    Splits features into Lifestyle and Clinical categories for better clarity.
    """
    if data.empty: return
    logger.info("Visualizing risk factors vs. clinical symptoms relative to Diagnosis.")
    corr = data.corr()['Diagnosis']
    
    lifestyle = ['Age', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    clinical = ['UPDRS', 'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Lifestyle factors bar plot
    ls_data = corr[lifestyle].sort_values()
    ls_data.plot(kind='barh', color=['#2ecc71' if x < 0 else '#e74c3c' for x in ls_data], ax=ax1)
    ax1.set_title('Lifestyle: Risk (+) vs Protective (-)')
    
    # Clinical symptoms bar plot
    corr[clinical].sort_values().plot(kind='barh', color='skyblue', ax=ax2)
    ax2.set_title('Clinical Indicators (Symptom Presence)')
    
    plt.tight_layout()
    plt.show()

# --- 3. POPULATION FILTERING ---

def extract_sick_population(data):
    """
    Filters the dataset to include only patients diagnosed with Parkinson's (Diagnosis == 1).
    """
    if data.empty: return pd.DataFrame()
    sick_subset = data[data['Diagnosis'] == 1].copy()
    logger.info(f"Population filtered. Analyzing {len(sick_subset)} diagnosed patients.")
    return sick_subset

# --- 4. CLINICAL ANALYSIS FUNCTIONS (Sick Population Only) ---

def plot_severity_distributions(data):
    """
    Compares the value distribution of core severity metrics: UPDRS, MoCA, and Functional Assessment.
    """
    if data.empty: return
    logger.info("Comparing distributions of primary clinical metrics.")
    metrics = ['UPDRS', 'MoCA', 'FunctionalAssessment']
    
    plt.figure(figsize=(15, 5))
    for i, m in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.histplot(data[m], kde=True, color='purple')
        plt.title(f'{m} Distribution')
    plt.tight_layout()
    plt.show()

def analyze_metric_dissociation(data):
    """
    Analyzes absolute correlations to prove that physical, cognitive, and functional declines are dissociated.
    """
    if data.empty: return
    logger.info("Executing dissociation analysis with absolute correlations.")
    metrics = ['UPDRS', 'MoCA', 'FunctionalAssessment']
    
    # Inter-metric internal correlation
    inter_corr = data[metrics].corr()
    plt.figure(figsize=(8, 4))
    sns.heatmap(inter_corr, annot=True, cmap='Blues')
    plt.title('Inter-Metric Correlation (Proving Dissociation)')
    plt.show()

    # Comparative feature impact (Absolute correlations)
    numeric_data = data.select_dtypes(include=[np.number])
    features = [c for c in numeric_data.columns if c not in metrics and c != 'Diagnosis']
    
    comparison = pd.DataFrame({
        m: numeric_data[features + [m]].corr()[m].drop(m).abs() 
        for m in metrics
    })
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(comparison.sort_values(by='UPDRS', ascending=False).head(15), 
                annot=True, cmap='YlGnBu')
    plt.title('Absolute Impact Comparison (Physical vs Cognitive vs Functional)')
    plt.show()

def run_poisson_analysis(data):
    """
    Compares actual symptom count distribution against a random Poisson model.
    Proves that symptoms aggregate biologically rather than randomly.
    """
    if data.empty: return
    logger.info("Running Poisson Distribution analysis.")
    symptom_list = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 
                    'SpeechProblems', 'SleepDisorders', 'Constipation']
    
    data['SymptomCount'] = data[symptom_list].sum(axis=1)
    mu = data['SymptomCount'].mean()
    
    actual = data['SymptomCount'].value_counts(normalize=True).sort_index()
    theoretical = [poisson.pmf(k, mu) for k in range(len(actual))]
    
    plt.figure(figsize=(10, 6))
    plt.bar(actual.index, actual.values, alpha=0.5, label='Observed Data', color='grey')
    plt.plot(actual.index, theoretical, 'ro-', linewidth=2, label='Poisson (Random Theory)')
    plt.title(f'Symptom Aggregation vs Random Chance (Mean: {mu:.2f})')
    plt.legend()
    plt.show()

def run_gatekeeper_analysis(data):
    """
    Identifies 'Gatekeeper' symptoms that pull other symptoms into the clinical profile.
    Updated to resolve Seaborn palette warnings.
    """
    if data.empty: return
    logger.info("Executing Gatekeeper Analysis (Tremor Paradox).")
    symptoms = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 
                'SpeechProblems', 'SleepDisorders', 'Constipation']
    results = []

    for s in symptoms:
        others = [sym for sym in symptoms if sym != s]
        impact = data[data[s] == 1][others].sum(axis=1).mean() - data[data[s] == 0][others].sum(axis=1).mean()
        results.append({'Symptom': s, 'Impact': impact})

    impact_df = pd.DataFrame(results).sort_values(by='Impact')
    
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in impact_df['Impact']]
    # Updated: Assigning 'y' to 'hue' and setting legend=False to comply with new Seaborn standards
    sns.barplot(data=impact_df, x='Impact', y='Symptom', hue='Symptom', palette=colors, legend=False)
    plt.title('Gatekeeper Effect: Impact of specific symptoms on overall burden')
    plt.show()

def plot_tremor_comparison(data):
    """
    Directly compares non-tremor symptom prevalence between Tremor vs. No-Tremor groups.
    """
    if data.empty: return
    logger.info("Comparing symptom prevalence: Tremor vs. No-Tremor groups.")
    other_symptoms = ['Rigidity', 'Bradykinesia', 'PosturalInstability', 
                      'SpeechProblems', 'SleepDisorders', 'Constipation']
    
    comparison_df = data.groupby('Tremor')[other_symptoms].mean().T
    comparison_df.columns = ['Group: No Tremor', 'Group: With Tremor']
    
    comparison_df.plot(kind='bar', color=['#e74c3c', '#2ecc71'], figsize=(12, 6))
    plt.title('Clinical Dissociation: Tremor vs. No-Tremor Patients')
    plt.ylabel('Prevalence (Percentage)')
    plt.show()

def plot_disease_roadmap(data):
    """
    Uses rolling averages (window=100) to reveal physical and cognitive trajectories across age.
    """
    if data.empty: return
    logger.info("Generating the final disease progression roadmap.")
    roadmap_df = data.sort_values('Age')
    
    roadmap_df['Physical_Trend'] = roadmap_df['UPDRS'].rolling(window=100, center=True).mean()
    roadmap_df['Cognitive_Trend'] = roadmap_df['MoCA'].rolling(window=100, center=True).mean()
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Physical Axis (Red)
    ax1.plot(roadmap_df['Age'], roadmap_df['Physical_Trend'], color='#e74c3c', linewidth=3, label='Physical')
    ax1.set_xlabel('Patient Age', fontweight='bold')
    ax1.set_ylabel('Physical Severity (UPDRS)', color='#e74c3c', fontweight='bold')
    
    # Cognitive Axis (Blue)
    ax2 = ax1.twinx()
    ax2.plot(roadmap_df['Age'], roadmap_df['Cognitive_Trend'], color='#3498db', linewidth=3, label='Cognitive')
    ax2.set_ylabel('Cognitive Score (MoCA)', color='#3498db', fontweight='bold')
    
    plt.title('The Parkinson\'s Roadmap: Physical vs. Cognitive Trajectories', fontsize=16, fontweight='bold')
    plt.show()

# --- 5. MAIN EXECUTION BLOCK ---
# This block ensures that the analysis runs only when executed directly.
if __name__ == "__main__":
    data_path = 'data/parkinsons_cleaned.csv'
    df = load_dataset(data_path)
    
    if not df.empty:
        # Step 1: Global population-wide visualizations
        plot_global_heatmap(df)
        plot_diagnosis_correlations(df)
        
        # Step 2: Focus on sick population for in-depth clinical analysis
        sick_df = extract_sick_population(df)
        
        if not sick_df.empty:
            plot_severity_distributions(sick_df)
            analyze_metric_dissociation(sick_df)
            run_poisson_analysis(sick_df)
            run_gatekeeper_analysis(sick_df)
            plot_tremor_comparison(sick_df)
            plot_disease_roadmap(sick_df)
            
        logger.info("Complete analysis sequence finalized.")


