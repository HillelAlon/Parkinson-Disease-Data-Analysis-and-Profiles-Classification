import unittest
import pandas as pd
import numpy as np

# Importing functions
from clustering_functions import (standardize, our_pca)

class TestPCAPipeline(unittest.TestCase):

    def setUp(self):
        """
        Set up synthetic data for testing.
        This runs before every test method.
        """
        np.random.seed(42)
        n_samples = 20
        # Creating a sample dataframe with 5 features
        self.raw_data = pd.DataFrame(
            np.random.rand(n_samples, 5),
            columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
            index=[f"Patient_{i}" for i in range(n_samples)]
        )

    def test_standardize_logic(self):
        """
        Validates that standardization results in:
        1. Mean approximately 0.
        2. Standard deviation of 1.
        """
        scaled = standardize(self.raw_data)

        # Test mean (should be ~0)
        mean_after_scaling = np.mean(scaled, axis=0)
        for val in mean_after_scaling:
            self.assertAlmostEqual(val, 0, places=7, msg="Mean after standardization is not 0")

        # Test standard deviation (should be 1)
        std_after_scaling = np.std(scaled, axis=0)
        for val in std_after_scaling:
            self.assertAlmostEqual(val, 1, places=7, msg="Standard deviation after standardization is not 1")

    def test_our_pca_functionality(self):
        """
        Validates the PCA function:
        1. Dimensionality reduction to 3 components.
        2. Preservation of original index (Patient IDs).
        3. Correct column naming.
        """
        # First, we need standardized data for PCA
        scaled = standardize(self.raw_data)

        # Run your PCA function
        pca_model, df_output, results = our_pca(scaled, self.raw_data)

        # Check number of columns (Should be 3)
        self.assertEqual(df_output.shape[1], 3, "Output should have exactly 3 columns (PC1, PC2, PC3)")

        # Check column names
        self.assertListEqual(list(df_output.columns), ['PC1', 'PC2', 'PC3'], "Column names are incorrect")

        # Check if index is preserved
        self.assertTrue(all(df_output.index == self.raw_data.index), "The original Patient IDs index was lost")

        # Check if the output is a DataFrame
        self.assertIsInstance(df_output, pd.DataFrame, "The output must be a pandas DataFrame")


if __name__ == '__main__':
    unittest.main()