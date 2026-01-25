import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import patch

# --- DYNAMIC PATH CONFIGURATION ---
# Adds the project root to sys.path to ensure 'analysis_functions' can be imported
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent # Adjust based on your folder structure
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the functions to be tested
try:
    from analysis_functions import extract_sick_population, run_poisson_analysis
except ImportError:
    # Fallback if the folder structure is different
    from analysis_functions import extract_sick_population, run_poisson_analysis

class TestParkinsonAnalysis(unittest.TestCase):

    def setUp(self):
        """
        Setup: Generates the synthetic dataset for clinical analysis testing.
        This follows the Modularity principle by centralizing data creation.
        """
        np.random.seed(42)
        n_samples = 50
        
        # Creating the fake data structure 
        self.sample_data = pd.DataFrame({
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
        self.sample_data.set_index('PatientID', inplace=True)

    # --- TEST 1: extract_sick_population (Logical/Positive Test Case) ---
    
    def test_extract_sick_population_positive(self):
        """
        Positive Test Case: Verifies that the system works as expected with valid input data.
        Checks if only patients with Diagnosis=1 are returned.
        """
        result = extract_sick_population(self.sample_data)
        
        # Verify the result is not empty (assuming seed 42 produces sick patients)
        self.assertFalse(result.empty, "The sick population subset should not be empty.")
        # Logical check: all diagnosis values must be 1
        self.assertTrue(all(result['Diagnosis'] == 1), "Extracted population must only contain diagnosed patients.")

    def test_extract_sick_population_null(self):
        """
        Null Test Case: Tests the system with empty values to confirm it handles them without crashing.
        """
        empty_df = pd.DataFrame(columns=['Diagnosis'])
        result = extract_sick_population(empty_df)
        self.assertTrue(result.empty, "Result should be empty when input is empty.")

    # --- TEST 2: run_poisson_analysis (Logical/Error Test Case) ---

    def test_poisson_logic_and_side_effects(self):
        """
        Logical Error Check: Verifies that the code produces the correct 'SymptomCount' column.
        Handles the visual output using a mock to avoid runtime interruptions.
        """
        data_copy = self.sample_data.copy()
        
        # Using patch to prevent plt.show() from opening windows during test execution
        with patch("matplotlib.pyplot.show"):
            run_poisson_analysis(data_copy)
            
        # Verify that the new column was added to the dataframe
        self.assertIn('SymptomCount', data_copy.columns, "SymptomCount column was not added.")
        
        # Boundary Test: Verify the count is within the possible range (0 to 7 symptoms)
        self.assertTrue(data_copy['SymptomCount'].max() <= 7, "Symptom count exceeds the number of defined symptoms.")

if __name__ == '__main__':
    # Execute the tests with verbose output
    unittest.main()