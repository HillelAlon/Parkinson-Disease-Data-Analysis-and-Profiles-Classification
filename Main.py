from src.data_loader import load_raw_data, clean_data_structure, prepare_for_modeling, save_data

def main():
    # Pipeline
    df = load_raw_data("data/parkinsons_disease_data.csv")
    df_clean = clean_data_structure(df)
    
    # Save the ready-to-use data
    df_ready = prepare_for_modeling(df_clean)
    save_data(df_ready, "data/parkinsons_for_model.csv")
    
    print("Data Pipeline Completed!")

if __name__ == "__main__":
    main()


import logging
import os
# Importing our modular functions
from src.data_loader import load_raw_data, clean_data_structure
from src.analysis import calculate_correlations, analyze_sick_severity
from src.plotting import plot_correlation_heatmap, plot_correlation_bar, plot_severity_drivers

def main():
    # 1. Setup Logging (Professional alternative to print)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 2. Data Loading & Cleaning Phase
    # Path is relative to the root directory where main.py sits
    data_path = os.path.join('data', 'parkinsons_disease_data.csv')
    df_raw = load_raw_data(data_path)
    df_clean = clean_data_structure(df_raw)
    
    logging.info("--- Starting Analysis Phase ---")

    # 3. General Correlation Analysis (Risk vs Protective Factors)
    # This calls the modular function that returns 3 items
    matrix, risk_factors, protective_factors = calculate_correlations(df_clean)
    
    # Visualizing general correlations
    plot_correlation_heatmap(matrix, title="General Clinical Correlations")
    plot_correlation_bar(risk_factors, title="Top 10 Risk Factors", color_palette='Reds_r')
    plot_correlation_bar(protective_factors, title="Top 5 Protective Factors", color_palette='Blues_r')

    # 4. Disease Severity Analysis (Sick Patients Sub-group)
    # Analyzing only patients with Diagnosis == 1
    severity_drivers = analyze_sick_severity(df_clean)
    
    # Visualizing what drives UPDRS scores higher or lower
    plot_severity_drivers(severity_drivers, title="Key Drivers of Parkinson's Severity (UPDRS)")

    logging.info("--- Analysis Completed Successfully ---")

if __name__ == "__main__":
    main()