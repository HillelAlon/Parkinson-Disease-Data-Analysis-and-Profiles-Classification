## Parkinson Disease Data Analysis and Profiles Classification

This project investigates comprehensive health information for 2,105 patients to uncover patterns in Parkinson's Disease progression. By integrating statistical analysis with clustering, we aim to move toward personalized treatment strategies through advanced patient profiling.

## Project Overview
This research explores the relationships between lifestyle choices, clinical characteristics, and disease severity. 
**Hypothesis:** Patients can be categorized into distinct phenotypes based on clinical and lifestyle data, and these groups show different trajectories of physical and cognitive decline.
**link for gitHub:** https://github.com/HillelAlon/Parkinson-Disease-Data-Analysis-and-Profiles-Classification
**Link for slides:** [Project Presentation](https://docs.google.com/presentation/d/1nOysAlVYcGIlKAsOH9k4TYhBiuq9o8w6vciK3Ff_9Ow/edit?slide=id.g3b572de3954_2_103#slide=id.g3b572de3954_2_103).


 # Research Goal
To use machine learning (K-Means) and dimensionality reduction (PCA) to group patients into 4 distinct clusters, enabling a more personalized approach to wellness and treatment strategies.
The overarching goal of this study was to identify key factors influencing patient health and to categorize patients into meaningful subgroups for personalized care.

The project consists of several core analyses:
- Correlation & Trend Analysis: Investigating how specific lifestyle choices (diet, exercise) correlate with clinical symptoms.
- Predictive Modeling/Statistical Testing
- Advanced Patient Profiling: Using Machine Learning to identify distinct patient phenotypes.

 # Methodology
1. **Data cleaning** - convert PatientID to Index, Remove Irrelevant Columns, Remove Duplicates, Save Cleaned Data
2.  **Analysis:** Global Screening: Establishing a baseline across all 2,105 subjects to prevent statistical bias.

    Diagnosis "Zoom-In": Identifying hidden lifestyle markers that emerge only when isolating the diagnosis variable.

    Intra-Cohort Analysis: Uncovering high-resolution correlations specific to the pathological state of the diseased population.

    Metric Dissociation: Testing the three severity scales to prove motor, cognitive, and functional decline are independent pathways.

    Domain-Specific Influencers: Analyzing factors that selectively worsen or improve one severity metric without affecting others.

    **Advanced Bonus Modules:**

    Symptom Aggregation (Poisson): Proving that symptoms cluster biologically rather than appearing as independent random noise.

    Heterogeneity Validation: Confirming the absence of a "leader" symptom, reinforcing that disease progression is unique to each patient.

3.  **Clustering**
    - **Normalization:** Scaling data using Z-Scores.
    - **PCA:** Reducing 12 variables into 3 Principal Components (capturing >70% variance).
    - **K-Means:** Grouping patients into 4 profiles using the Elbow Method.
4.  **Visualization:** 3D plotting of clusters, standardized heatmaps, and disease progression roadmaps (Physical vs. Cognitive trajectories).

# Key Findings 
- A synthetic data is not always realistic :(
- Metric Dissociation: Clinical evidence suggests that motor (UPDRS) and cognitive (MoCA) declines progress as independent pathways, highlighting the need for multi-domain treatment.

- Symptom Logic: Poisson analysis confirmed that symptoms are biologically clustered rather than appearing randomly.

## Data Description
The dataset contains 2,105 patient records with features including:
- **Lifestyle:** Diet Quality, Physical Activity, Sleep Quality, BMI.
- **Clinical:** UPDRS (Physical severity), MoCA (Cognitive score), and specific symptoms like Tremor and Rigidity.

## Folder Structure
* `Main.py`: The central entry point for running the cleaning and analysis pipeline.
* `cleaning_data/`: 
    * `data_cleaning.py`: Functions for data loading, validation, and preprocessing.
* `analysis/`:
    * `functios_analysis.py`: 7-stage clinical research logic and visualizations.
    * `bonus_analysis.py`: Poisson distribution and Gatekeeper analysis.
    * `test_analysis.py`: Unit tests using `pytest` for validating analysis functions.
* `clusteringTA/`:
    * `PCA_main_file.py`: Implementation of Z-score normalization, PCA, and Clustering.
* `data/`: Contains raw and processed CSV files.

- # Tech Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Pytest.

## How to Run
1. **Install requirements:** pip install -r requirements.txt
2.  **Run main:** python Main.py
    - note - for MAC OS: python3 Main.py
3.  **Run unit tests:** pytest analysis/test_analysis.py

**Created by: Tamar Kan, Alon Hillel, Or Galifat and Roni Itay | Neuroscience, Bar-Ilan University**