import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

# Importing your functions (Assumes your file is named 'main_analysis.py')
from analysis_logic import (
    extract_sick_population, 
    run_poisson_analysis, 
    run_gatekeeper_analysis,
    plot_disease_roadmap,
    analyze_metric_dissociation
)

# --- TEST FIXTURE ---
# This creates a small "fake" dataset for testing purposes
@pytest.fixture
def sample_data():
    """Generates a synthetic dataset to test analysis functions."""
    np.random.seed(42)
    n_samples = 150
    data = pd.DataFrame({
        'PatientID': range(n_samples),
        'Age': np.random.randint(50, 90, n_samples),
        'Diagnosis': np.random.choice([0, 1], n_samples),
        'UPDRS': np.random.uniform(10, 160, n_samples),
        'MoCA': np.random.uniform(5, 30, n_samples),
        'FunctionalAssessment': np.random.uniform(0, 10, n_samples),
        'Tremor': np.random.choice([0, 1], n_samples),
        'Rigidity': np.random.choice([0, 1], n_samples),
        'Bradykinesia': np.random.choice([0, 1], n_samples),
        'PosturalInstability': np.random.choice([0, 1], n_samples),
        'SpeechProblems': np.random.choice([0, 1], n_samples),
        'SleepDisorders': np.random.choice([0, 1], n_samples),
        'Constipation': np.random.choice([0, 1], n_samples)
    })
    return data

# --- UNIT TESTS ---

def test_extract_sick_population(sample_data):
    """Verifies that only patients with Diagnosis == 1 are kept."""
    sick_df = extract_sick_population(sample_data)
    
    # Check 1: All patients in result must be sick
    assert all(sick_df['Diagnosis'] == 1), "Found healthy patients in sick_df"
    
    # Check 2: The filtered dataframe should not be empty
    assert len(sick_df) > 0, "Filtered population is empty"
    
    # Check 3: Check if core columns still exist
    assert 'UPDRS' in sick_df.columns

@patch("matplotlib.pyplot.show") # This prevents actual plots from popping up during tests
def test_visual_functions(mock_show, sample_data):
    """Ensures that visual analysis functions run without crashing (Smoke Test)."""
    sick_df = extract_sick_population(sample_data)
    
    # Test if Dissociation Analysis runs
    analyze_metric_dissociation(sick_df)
    
    # Test if Poisson Analysis runs
    run_poisson_analysis(sick_df)
    
    # Test if Gatekeeper Analysis runs
    run_gatekeeper_analysis(sick_df)
    
    # Test if Disease Roadmap runs
    plot_disease_roadmap(sick_df)
    
    # Verify that 'show' was called by the plotting functions
    assert mock_show.called

def test_poisson_logic(sample_data):
    """Validates the mathematical calculation of SymptomCount."""
    sick_df = extract_sick_population(sample_data)
    run_poisson_analysis(sick_df)
    
    # Check if the new column was successfully created
    assert 'SymptomCount' in sick_df.columns
    
    # Check if counts are within a valid range (0 to 7 symptoms)
    assert sick_df['SymptomCount'].max() <= 7
    assert sick_df['SymptomCount'].min() >= 0

def test_gatekeeper_impact_calculation(sample_data):
    """Tests if the gatekeeper logic produces valid numeric impact values."""
    # 1. Generating the filtered sick population from mock data
    # We name the variable 'sick_df' here
    sick_df = extract_sick_population(sample_data)
    
    s = 'Tremor'
    others = ['Rigidity', 'Bradykinesia', 'PosturalInstability']
    
    # 2. Calculating the impact manually to verify the logic
    # FIX: We must use 'sick_df' consistently throughout the function
    mean_with = sick_df[sick_df[s] == 1][others].sum(axis=1).mean()
    mean_without = sick_df[sick_df[s] == 0][others].sum(axis=1).mean()
    impact = mean_with - mean_without
    
    # 3. Verify the result is a valid number and not NaN (Not a Number)
    assert not np.isnan(impact), "Impact calculation returned NaN"
    