## Parkinson Disease Data Analysis and Profiles Classification
This dataset comprises comprehensive health information for 2,105 patients, some diagnosed with Parkinson's Disease. 
Patient Profiling in Parkinson's Disease using PCA & Clustering.

 # Project Overview
This collaborative research project investigates the complex relationships between clinical characteristics, lifestyle factors, and disease progression in patients with Parkinson's Disease. Using a dataset of over 2,100 patients, our team conducted several analyses to uncover hidden patterns and correlations that contribute to a deeper understanding of the disease's heterogeneity.
Link for slides: https://docs.google.com/presentation/d/1nOysAlVYcGIlKAsOH9k4TYhBiuq9o8w6vciK3Ff_9Ow/edit?slide=id.g3b572de3954_2_103#slide=id.g3b572de3954_2_103

 # Research Goal
To use machine learning (K-Means) and dimensionality reduction (PCA) to group patients into 4 distinct clusters, enabling a more personalized approach to wellness and treatment strategies.
The overarching goal of this study was to identify key factors influencing patient health and to categorize patients into meaningful subgroups for personalized care.

The project consists of several core analyses:

Correlation & Trend Analysis: Investigating how specific lifestyle choices (diet, exercise) correlate with clinical symptoms.

Predictive Modeling/Statistical Testing

Advanced Patient Profiling: Using Machine Learning to identify distinct patient phenotypes.

 # Methodology
- Data cleaning - convert PatientID to Index, Remove Irrelevant Columns, Remove Duplicates, Save Cleaned Data
- Dimensionality Reduction (PCA): Synthesized 12 complex lifestyle and clinical variables into 3 Principal Components, capturing over 70% of the data's variance.

- Optimal Clustering: Applied the Elbow Method to determine the most stable number of patient subgroups.

- Phenotype Identification: Utilized K-Means Clustering to divide the population into 4 distinct profiles, visualized through 3D plotting and standardized Heatmaps.


# Key Findings 
- A synthetic data is not always realistic :(

- # Tech Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

🚀 How to Run
Install requirements: pip install -r requirements.txt

Run the main script or Jupyter Notebook.

**Created by: Tamar Kan, Alon Hillel, Or Galifat and Roni Itay | Neuroscience, Bar-Ilan University**
