import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(corr_matrix, title='Full Correlation Heatmap'):
    """
    Generates a heatmap to visualize relationships between all clinical features.
    """
    plt.figure(figsize=(15, 10))
    # Using 'coolwarm' to show positive (red) and negative (blue) correlations
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_correlation_bar(data, title, color_palette='Reds_r'):
    """
    Standard bar plot for displaying top correlated factors (Risk or Protective).
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=data.values, y=data.index, palette=color_palette)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()

def plot_severity_drivers(top_drivers, title='Key Drivers of Disease Severity (UPDRS)'):
    """
    Specialized bar plot for UPDRS drivers with color-coding:
    Red for factors that increase severity, Blue for those that decrease it.
    """
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Logic: Red for positive correlation (worsening), Blue for negative (improvement)
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_drivers]
    
    sns.barplot(x=top_drivers.values, y=top_drivers.index, palette=colors)
    
    # Add a vertical line at zero for visual clarity
    plt.axvline(0, color='black', linewidth=1.5)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Correlation Coefficient (r)', fontsize=12)
    plt.tight_layout()
    plt.show()