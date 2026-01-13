from .analysis_logic import *

def run_analysis_pipeline(data_path):
    """
    Executes the full 10-step analysis pipeline.
    Returns the sick_df for potential use in other modules.
    """
    logger.info("Starting the Analysis Pipeline orchestration.")
    
    # 1. Load data
    df = load_dataset(data_path)
    if df is None:
        logger.error("Analysis aborted: Data could not be loaded.")
        return None

    # 2-3. General Population Analysis
    plot_global_heatmap(df)
    plot_diagnosis_correlations(df)

    # 4. Filter Sick Population
    sick_df = extract_sick_population(df)
    
    # 5-7. Severity and Dissociation Analysis
    plot_severity_distributions(sick_df)
    analyze_metric_dissociation(sick_df)
    
    severity_metrics = ['UPDRS', 'MoCA', 'FunctionalAssessment']
    impact_df = get_severity_impact(sick_df, severity_metrics)
    plot_impact_profiles(impact_df)

    # 8-10. Biological Patterns and Evolution
    run_poisson_analysis(sick_df)
    run_gatekeeper_analysis(sick_df)
    plot_tremor_comparison(sick_df)
    plot_disease_roadmap(sick_df)

    logger.info("Analysis Pipeline orchestration completed successfully.")
    return sick_df