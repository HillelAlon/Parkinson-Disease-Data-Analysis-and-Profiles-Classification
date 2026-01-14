from analysis.main_analysis import run_analysis_pipeline

def main():
    """
    Main entry point for the Parkinson's Research Project.
    """
    print("--- PARKINSONS RESEARCH PROJECT START ---")
    
    # Define data path
    raw_data_path = 'data/parkinsons_cleaned.csv'
    
    # Execute Analysis Module
    # This single call triggers all 10 functions inside the analysis package
    sick_df = run_analysis_pipeline(raw_data_path)
    
    if sick_df is not None:
        print("Analysis Module completed successfully.")
        # Future: clustering_results = run_clustering_pipeline(sick_df)
    else:
        print("Project stopped due to an error in the Analysis phase.")

if __name__ == "__main__":
    main()