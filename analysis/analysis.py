import pandas as pd
import logging

def calculate_correlations(df, target_col='Diagnosis'):
    """
    Part 1 & 2: Calculates the correlation matrix and focuses on the target column.
    """
    logging.info(f"Calculating correlations for target column: {target_col}")
    corr_matrix = df.corr()
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    # Identify top 10 positive and top 10 negative correlations
    top_positive = target_corr.head(11).drop(target_col, errors='ignore') # Exclude itself
    top_negative = target_corr.tail(10)
    
    return corr_matrix, top_positive, top_negative

def analyze_sick_severity(df):
    """
    Part 4: Filters sick patients and calculates correlations with severity (UPDRS).
    """
    logging.info("Filtering diagnosed patients for severity analysis (UPDRS).")
    # Filter ONLY on diagnosed patients
    sick_df = df[df['Diagnosis'] == 1].copy()
    
    # Calculate correlation specifically with UPDRS (Severity)
    severity_corr = sick_df.corr()['UPDRS'].drop(['UPDRS', 'Diagnosis'], errors='ignore')
    
    # Take the top 10 influential factors
    top_drivers = severity_corr.reindex(severity_corr.abs().sort_values(ascending=False).index).head(10)
    
    return top_drivers

def get_top_impact_factors(df, metric_name):
    """
    Identifies top 2 aggravating and top 2 alleviating factors for a specific metric.
    Adjusts logic based on whether higher metric values are 'good' or 'bad'.
    """
    # 1. Calculate correlation for the specific metric
    # Drop IDs and other targets to stay clean
    exclude = ['PatientID', 'Diagnosis', 'DoctorInCharge', 'UPDRS', 'MoCA', 'FunctionalAssessment']
    features = [col for col in df.columns if col not in exclude]
    
    corrs = df[features + [metric_name]].corr()[metric_name].drop(metric_name)
    
    # 2. Logic Adjustment
    # UPDRS: High is BAD. MoCA/Functional: High is GOOD.
    if metric_name == 'UPDRS':
        aggravating = corrs.sort_values(ascending=False).head(2) # Top positive
        alleviating = corrs.sort_values(ascending=True).head(2)  # Top negative
    else:
        # For MoCA and Functional Assessment
        aggravating = corrs.sort_values(ascending=True).head(2)  # Top negative (makes score lower/worse)
        alleviating = corrs.sort_values(ascending=False).head(2) # Top positive (makes score higher/better)
        
    return aggravating, alleviating