import pytest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from unittest.mock import patch

# Importing the 7 analysis functions 
from functions_analysis import (
    load_dataset,
    plot_global_heatmap,
    plot_feature_profile, 
    extract_sick_population,
    plot_sick_population_heatmap,
    plot_severity_distributions,
    analyze_metric_dissociation
)

# --- TEST FIXTURE ---
@pytest.fixture
def sample_data():
    """Generates a synthetic dataset for clinical analysis testing."""
    np.random.seed(42)
    n_samples = 50
    data = pd.DataFrame({
        'PatientID': range(n_samples),
        'Age': np.random.randint(50, 90, n_samples),
        'Diagnosis': np.random.choice([0, 1], n_samples),
        'UPDRS': np.random.uniform(10, 160, n_samples),
        'MoCA': np.random.uniform(5, 30, n_samples),
        'FunctionalAssessment': np.random.uniform(0, 10, n_samples),
        'Tremor': np.random.choice([0, 1], n_samples),
        'Rigidity': np.random.choice([0, 1], n_samples)
    })
    data.set_index('PatientID', inplace=True)
    return data

@pytest.fixture
def temp_results_dir(tmp_path):
    """Temporary directory for test plot exports."""
    d = tmp_path / "results"
    d.mkdir()
    return str(d)

# --- UNIT TESTS ---

def test_extract_sick_population(sample_data):
    """Verifies population filtering logic."""
    sick_df = extract_sick_population(sample_data, 'Diagnosis', 1)
    assert all(sick_df['Diagnosis'] == 1)
    assert len(sick_df) == (sample_data['Diagnosis'] == 1).sum()

@patch("matplotlib.pyplot.savefig")
def test_visual_functions_smoke(mock_save, sample_data, temp_results_dir):
    """Ensures all visual functions run with the corrected names."""
    diagnosis_col = 'Diagnosis'
    metrics = ['UPDRS', 'MoCA']
    sick_df = extract_sick_population(sample_data, diagnosis_col, 1)

    # Calling the corrected function name
    plot_global_heatmap(sample_data, temp_results_dir)
    plot_feature_profile(sample_data, diagnosis_col, temp_results_dir)
    plot_sick_population_heatmap(sick_df, diagnosis_col, temp_results_dir)
    plot_severity_distributions(sick_df, metrics, temp_results_dir)
    analyze_metric_dissociation(sick_df, metrics, temp_results_dir)

    assert mock_save.called
    # We expect at least 6 saves (1 global, 1 profile, 1 sick heatmap, 1 dist, 2 dissociation)
    assert mock_save.call_count >= 6
    