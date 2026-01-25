## Parkinson Disease Data Analysis and Profiles Classification

This project investigates comprehensive health information for 2,105 patients to uncover patterns in Parkinson's Disease progression. By integrating statistical analysis with clustering, we aim to move toward personalized treatment strategies through advanced patient profiling.

## Project Overview
This research explores the relationships between lifestyle choices, clinical characteristics, and disease severity. 
**Hypothesis:** Patients can be categorized into distinct phenotypes based on clinical and lifestyle data, and these groups show different trajectories of physical and cognitive decline.
**link for gitHub:** https://github.com/HillelAlon/Parkinson-Disease-Data-Analysis-and-Profiles-Classification
**Link for slides:** [Project Presentation](https://docs.google.com/presentation/d/1nOysAlVYcGIlKAsOH9k4TYhBiuq9o8w6vciK3Ff_9Ow/edit?slide=id.g3b572de3954_2_103#slide=id.g3b572de3954_2_103).
**link for Dataset:**: https://www.kaggle.com/datasets/rabieelkharoua/parkinsons-disease-dataset-analysis (Note - also in folder data > parkinson_disease_data.csv)

### Metric Dissociation Logic
This study specifically tests the **Metric Dissociation Hypothesis**. By analyzing the lack of linear correlation between UPDRS (motor severity), MoCA (cognitive score), and Functional Assessment scores, we aim to provide data-driven evidence for the heterogeneous nature of the disease, supporting the transition toward personalized medicine in PD care.
**link for gitHub:** https://github.com/HillelAlon/Parkinson-Disease-Data-Analysis-and-Profiles-Classification
**Link for slides:** [Project Presentation](https://docs.google.com/presentation/d/  
**link for supporting academic paper** Title: New Clinical Subtypes of Parkinson Disease and Their Longitudinal Progression: A Prospective Cohort Comparison With Other Phenotypes Link: https://pubmed.ncbi.nlm.nih.gov/26076039/

# Research Goal
To use machine learning (K-Means) and dimensionality reduction (PCA) to group patients into 4 distinct clusters, enabling a more personalized approach to wellness and treatment strategies.
The overarching goal of this study was to identify key factors influencing patient health and to categorize patients into meaningful subgroups for personalized care.

The project consists of several core analyses:
- Correlation & Trend Analysis: Investigating how specific lifestyle choices (diet, exercise) correlate with clinical symptoms.
- Predictive Modeling/Statistical Testing
- Advanced Patient Profiling: Using Machine Learning to identify distinct patient phenotypes.

 # Methodology
Methodology & Research Flow
0. Data Preparation (data_cleaning.py)

    Validation & Cleaning: We implemented a rigorous cleaning pipeline to remove non-informative features like DoctorInCharge and eliminate duplicate records.

    Indexing: PatientID was converted to the primary index to ensure consistent tracking across all analysis modules.

    Rationale: Establishing a clean "Ground Truth" is essential to prevent clustering algorithms from focusing on noisy or redundant data.

1. Multi-Stage Clinical Analysis (functions_analysis.py)

    Global Screening: We established a baseline correlation landscape across all 2,105 subjects.

        Rationale: This prevents Confounding Bias by ensuring that observed patterns are not driven by external factors like Age or BMI that affect the general population.

    Intra-Cohort Zoom-In: We isolated the diagnosed population (n=1,304) to uncover high-resolution dynamics specific to the pathological state.

    Metric Dissociation Testing: We analyzed the Physical (UPDRS), Cognitive (MoCA), and Functional assessment scales.

        Rationale: Finding wide distributions and low correlations between these pillars supports our hypothesis that Parkinson's decline follows independent, dissociated pathways.

2. Advanced Modeling & Dimensionality Reduction (pca_cleaned_function.py)

    Normalization: Continuous variables were standardized using Z-score scaling (Z=σx−μ​) to ensure all 35 features contribute equally to the model.

    PCA Navigation: We implemented Principal Component Analysis to reduce data complexity into a visualizable 3D space.

    K-Means Clustering: We attempted to categorize patients into 4 clinical profiles using the Elbow Method.

        Finding: The resulting overlapping clusters confirm that Parkinson’s exists on a clinical spectrum rather than in isolated silos.

3.  Statistical Validation: A One-Way ANOVA was performed to verify if the identified profiles represent statistically distinct groups despite their clinical overlap.

4. Stochastic Analysis & Complexity Modules (bonus_analysis.py)

    Poisson Distribution: We compared symptom aggregation against a random model to prove that symptom clustering is a structured biological process rather than random noise.

    Gatekeeper Analysis: We tested for "leader" symptoms that might drive overall burden.

        Rationale: Confirming the absence of a single "Gatekeeper" reinforces the theory that disease progression is systemic and unique to each individual.
# Key Findings 
- A synthetic data is not always realistic :(
- Metric Dissociation: Clinical evidence suggests that motor (UPDRS) and cognitive (MoCA) declines progress as independent pathways, highlighting the need for multi-domain treatment.

- Symptom Logic: Poisson analysis confirmed that symptoms are biologically clustered rather than appearing randomly.

## Data Description
The dataset contains 2,105 patient records with features including:
- Patient ID
- **Demographic Details:** Age, Gender, Ethnicity, EducationLevel
- **Lifestyle Factors:** BMI, Smoking, Alcohol Consumption, Physical Activity, Diet Quality,Sleep Quality
- **Medical History** (presence of): Traumatic Brain Injury, Hypertension, Diabetes, Depression, Stroke. 
- **Clinical Measurements:** Systolic BP, DiastolicBP, Cholesterol Total, Cholesterol LDL, Cholesterol HDL, Cholesterol Triglycerides.
- **Cognitive and Functional Assessments:** Unified Parkinson's Disease Rating Scale, Montreal Cognitive Assessment, Functional assessment score
- **Symptoms:** Presence of tremor, Presence of muscle rigidity, Slowness of movement, Stability/balance issues, Presence of speech problems, Presence of sleep disorders, Presence of constipation.
- **Diagnosis Information:** Parkinson's Disease diagnosis status

**Folder Structure**
Each module folder contains the source code (.py) for automated execution and a Jupyter notebook (.ipynb) used for exploratory analysis and presentation.

    main_script.py: The central entry point that orchestrates the entire research pipeline from data cleaning to statistical validation.

    cleaning_data/:

        data_cleaning.py: Functions for data loading, integrity validation, and preprocessing.

    analysis/:

        functions_analysis.py: Implementation of the 7-stage clinical research logic and visualizations.

        bonus_analysis.py: Advanced statistical modules (Poisson and Gatekeeper analysis).

    clusteringTA/:

        pca_cleaned_function.py: Professional implementation of PCA dimensionality reduction and K-Means clustering.

    testsing/:

        test_analysis.py: Unit tests using the pytest framework to ensure the reliability of research functions.

    data/: Directory containing raw and processed CSV datasets.

    results/: Automatically generated directory where all 15+ visualizations and the research log (00_research_log_and_conclusions.txt) are stored.

    requirements.txt: List of all necessary Python dependencies and their versions.

    README.md: This documentation file, providing a project overview, hypothesis, and execution instructions.

- # Tech Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Pytest.

## How to Run
1. **Install requirements:** pip install -r requirements.txt
2.  **Run main:** python Main.py
    - note - for MAC OS: python3 Main.py
3.  **Run unit tests:** pytest analysis/test_analysis.py

**Created by: Tamar Kan, Alon Hillel, Or Galifat and Roni Itay | Neuroscience, Bar-Ilan University**

## References
Fereshtehnejad, S.-M., Romenets, S. R., Anang, J. B. M., Latreille, V., Gagnon, J.-F., & Postuma, R. B. (2015). New Clinical Subtypes of Parkinson Disease and Their Longitudinal Progression. JAMA Neurology, 72(8), 863. https://doi.org/10.1001/jamaneurol.2015.0703
