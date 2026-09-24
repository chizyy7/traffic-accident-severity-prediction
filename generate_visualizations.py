"""
Script to generate visualization assets for the AI Traffic Accident Severity Prediction System.
This script creates the required plots and saves them to the visualizations directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

def create_sample_data():
    """Create sample data for generating visualizations."""
    np.random.seed(42)
    n_samples = 10000

    # Create timestamps
    start_dates = pd.date_range('2016-01-01', '2023-12-31', freq='H')
    start_times = np.random.choice(start_dates, n_samples)

    # Create dataframe with features needed for visualizations
    df = pd.DataFrame({
        'ID': [f'A{i:06d}' for i in range(n_samples)],
        'Start_Time': start_times,
        'End_Time': start_times + pd.to_timedelta(np.random.randint(5, 300, n_samples), unit='m'),
        'Severity': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'Distance(mi)': np.random.exponential(0.5, n_samples),
        'Street': np.random.choice(['Main St', 'Oak Ave', 'Pine St', 'Elm St', 'Washington Blvd', 'Unknown'], n_samples),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia'], n_samples),
        'County': np.random.choice(['Los Angeles County', 'Cook County', 'Harris County', 'Maricopa County', 'Unknown'], n_samples),
        'State': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC'], n_samples),
        'Zipcode': np.random.choice(['10001', '90210', '60601', '75201', 'Unknown'], n_samples),
        'Country': 'US',
        'Timezone': np.random.choice(['America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles'], n_samples),
        'Airport_Code': np.random.choice(['JFK', 'LAX', 'ORD', 'DFW', 'UNKNOWN'], n_samples),
        'Weather_Timestamp': start_times,
        'Temperature(F)': np.random.normal(60, 20, n_samples).clip(-20, 120),
        'Wind_Chill(F)': np.random.normal(58, 20, n_samples).clip(-30, 100),
        'Humidity(%)': np.random.uniform(10, 100, n_samples),
        'Pressure(in)': np.random.normal(29.92, 0.5, n_samples).clip(28, 32),
        'Visibility(mi)': np.random.exponential(5, n_samples).clip(0, 50),
        'Wind_Direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Variable'], n_samples),
        'Wind_Speed(mph)': np.random.exponential(5, n_samples).clip(0, 50),
        'Precipitation(in)': np.random.exponential(0.01, n_samples).clip(0, 2),
        'Weather_Condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog', 'Storm', 'Haze'], n_samples, p=[0.4, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05]),
        'Amenity': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'Bump': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        'Crossing': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
        'Give_Way': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'Junction': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'No_Exit': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'Railway': np.random.choice([True, False], n_samples, p=[0.02, 0.98]),
        'Roundabout': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'Station': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'Stop': np.random.choice([True, False], n_samples, p=[0.25, 0.75]),
        'Traffic_Calming': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        'Traffic_Signal': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'Turning_Loop': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'Sunrise_Sunset': np.random.choice(['Day', 'Night'], n_samples, p=[0.7, 0.3]),
        'Civil_Twilight': np.random.choice(['Day', 'Night'], n_samples, p=[0.7, 0.3]),
        'Nautical_Twilight': np.random.choice(['Day', 'Night'], n_samples, p=[0.6, 0.4]),
        'Astronomical_Twilight': np.random.choice(['Day', 'Night'], n_samples, p=[0.5, 0.5])
    })

    return df

def generate_severity_distribution(df):
    """Generate severity distribution plot."""
    severity_counts = df['Severity'].value_counts().sort_index()
    severity_labels = {1: 'Minor', 2: 'Moderate', 3: 'Serious', 4: 'Severe'}
    severity_counts_named = severity_counts.rename(severity_labels)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(severity_counts_named.index, severity_counts_named.values,
                   color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
    plt.title('Distribution of Accident Severity', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Severity Level', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(severity_counts_named.values),
                 f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/severity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: severity_distribution.png")

def generate_accidents_by_year(df):
    """Generate accidents by year plot."""
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Year'] = df['Start_Time'].dt.year

    yearly_counts = df.groupby('Year').size()

    plt.figure(figsize=(12, 6))
    plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6, color='#1f77b4')
    plt.title('Accidents by Year', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Add value labels on points
    for x, y in zip(yearly_counts.index, yearly_counts.values):
        plt.text(x, y + 0.01*max(yearly_counts.values), f'{int(y):,}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/accidents_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: accidents_by_year.png")

def generate_severity_by_weather(df):
    """Generate severity by weather condition plot."""
    # Create a cross-tabulation of weather condition and severity
    weather_severity = pd.crosstab(df['Weather_Condition'], df['Severity'], normalize='index') * 100

    # Select top weather conditions for clarity
    top_weather = df['Weather_Condition'].value_counts().head(8).index
    weather_severity_top = weather_severity.loc[top_weather]

    # Rename columns for clarity
    weather_severity_top.columns = ['Minor', 'Moderate', 'Serious', 'Severe']

    plt.figure(figsize=(12, 8))
    weather_severity_top.plot(kind='bar', stacked=True, color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
    plt.title('Accident Severity Distribution by Weather Condition', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Weather Condition', fontsize=12)
    plt.ylabel('Percentage of Accidents', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Severity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/severity_by_weather.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: severity_by_weather.png")

def generate_severity_by_road_condition(df):
    """Generate severity by road condition plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Severity by visibility condition
    if 'Visibility(mi)' in df.columns:
        df['Visibility_Category'] = pd.cut(df['Visibility(mi)'],
                                           bins=[0, 1, 3, 5, 10, 50],
                                           labels=['Very Poor', 'Poor', 'Moderate', 'Good', 'Excellent'],
                                           include_lowest=True)

        visibility_severity = pd.crosstab(df['Visibility_Category'], df['Severity'], normalize='index') * 100
        visibility_severity.columns = ['Minor', 'Moderate', 'Serious', 'Severe']

        visibility_severity.plot(kind='bar', stacked=True, ax=axes[0,0],
                                color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
        axes[0,0].set_title('Accident Severity by Visibility Condition', fontsize=14, fontweight='bold', pad=15)
        axes[0,0].set_xlabel('Visibility Condition', fontsize=12)
        axes[0,0].set_ylabel('Percentage of Accidents', fontsize=12)
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(title='Severity Level')
    else:
        axes[0,0].text(0.5, 0.5, 'Visibility data not available', ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Accident Severity by Visibility Condition', fontsize=14, fontweight='bold', pad=15)

    # Plot 2: Severity by lighting condition
    if 'Sunrise_Sunset' in df.columns:
        lighting_severity = pd.crosstab(df['Sunrise_Sunset'], df['Severity'], normalize='index') * 100
        lighting_severity.columns = ['Minor', 'Moderate', 'Serious', 'Severe']

        lighting_severity.plot(kind='bar', stacked=True, ax=axes[0,1],
                              color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
        axes[0,1].set_title('Accident Severity by Lighting Condition', fontsize=14, fontweight='bold', pad=15)
        axes[0,1].set_xlabel('Lighting Condition', fontsize=12)
        axes[0,1].set_ylabel('Percentage of Accidents', fontsize=12)
        axes[0,1].tick_params(axis='x', rotation=0)
        axes[0,1].legend(title='Severity Level')
    else:
        axes[0,1].text(0.5, 0.5, 'Lighting data not available', ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Accident Severity by Lighting Condition', fontsize=14, fontweight='bold', pad=15)

    # Plot 3: Severity by road type (junction features)
    junction_cols = ['Junction', 'Crossing', 'Traffic_Signal', 'Stop', 'Roundabout']
    available_junction_cols = [col for col in junction_cols if col in df.columns]
    if available_junction_cols:
        df['Junction_Count'] = df[available_junction_cols].sum(axis=1)
        df['Road_Type'] = pd.cut(df['Junction_Count'],
                                 bins=[-1, 0, 1, 2, 5],
                                 labels=['No Junction', 'Single Junction', 'Multiple Junctions', 'Complex Intersection'],
                                 include_lowest=True)

        road_type_severity = pd.crosstab(df['Road_Type'], df['Severity'], normalize='index') * 100
        road_type_severity.columns = ['Minor', 'Moderate', 'Serious', 'Severe']

        road_type_severity.plot(kind='bar', stacked=True, ax=axes[1,0],
                               color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
        axes[1,0].set_title('Accident Severity by Road Junction Type', fontsize=14, fontweight='bold', pad=15)
        axes[1,0].set_xlabel('Road Junction Type', fontsize=12)
        axes[1,0].set_ylabel('Percentage of Accidents', fontsize=12)
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Severity Level')
    else:
        axes[1,0].text(0.5, 0.5, 'Junction data not available', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Accident Severity by Road Junction Type', fontsize=14, fontweight='bold', pad=15)

    # Plot 4: Severity by distance category
    if 'Distance(mi)' in df.columns:
        df['Distance_Category'] = pd.cut(df['Distance(mi)'],
                                         bins=[0, 0.1, 0.5, 1, 2, 5, 50],
                                         labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Extreme'],
                                         include_lowest=True)

        distance_severity = pd.crosstab(df['Distance_Category'], df['Severity'], normalize='index') * 100
        distance_severity.columns = ['Minor', 'Moderate', 'Serious', 'Severe']

        distance_severity.plot(kind='bar', stacked=True, ax=axes[1,1],
                              color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
        axes[1,1].set_title('Accident Severity by Distance Category', fontsize=14, fontweight='bold', pad=15)
        axes[1,1].set_xlabel('Distance Category', fontsize=12)
        axes[1,1].set_ylabel('Percentage of Accidents', fontsize=12)
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(title='Severity Level')
    else:
        axes[1,1].text(0.5, 0.5, 'Distance data not available', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Accident Severity by Distance Category', fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('visualizations/severity_by_road_condition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: severity_by_road_condition.png")

def generate_model_comparison():
    """Generate model comparison plot."""
    # Create sample model performance data
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM']
    accuracy = [0.672, 0.685, 0.712, 0.734, 0.738]
    precision = [0.658, 0.662, 0.698, 0.721, 0.725]
    recall = [0.672, 0.685, 0.712, 0.734, 0.738]
    f1 = [0.665, 0.673, 0.705, 0.727, 0.731]
    roc_auc = [0.721, 0.708, 0.745, 0.768, 0.772]

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - 2*width, accuracy, width, label='Accuracy', color='#1f77b4')
    bars2 = ax.bar(x - width, precision, width, label='Precision', color='#ff7f0e')
    bars3 = ax.bar(x, recall, width, label='Recall', color='#2ca02c')
    bars4 = ax.bar(x + width, f1, width, label='F1-Score', color='#d62728')
    bars5 = ax.bar(x + 2*width, roc_auc, width, label='ROC-AUC', color='#9467bd')

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    add_value_labels(bars5)

    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: model_comparison.png")

def generate_confusion_matrix():
    """Generate confusion matrix plot."""
    # Create sample confusion matrix
    # Assume 4 classes: Minor, Moderate, Serious, Severe
    cm = np.array([
        [1200, 300, 100, 50],   # Minor: True Negatives, False Positives, etc.
        [250, 800, 150, 100],   # Moderate
        [80, 120, 600, 100],    # Serious
        [30, 50, 80, 400]       # Severe
    ])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Minor', 'Moderate', 'Serious', 'Severe'],
                yticklabels=['Minor', 'Moderate', 'Serious', 'Severe'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: confusion_matrix.png")

def generate_feature_importance():
    """Generate feature importance plot."""
    # Create sample feature importance data
    features = [
        'Visibility(mi)', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)',
        'Wind_Speed(mph)', 'Pressure(in)', 'Precipitation(in)', 'Junction',
        'Crossing', 'Traffic_Signal', 'Stop', 'Weather_Condition_Encoded',
        'Time_Hour', 'Is_Weekend', 'Is_Rush_Hour', 'Is_Night'
    ]
    importance = [0.18, 0.15, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01]
    # Normalize to sum to 1
    importance = np.array(importance) / np.sum(importance)

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = importance[sorted_idx]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(features_sorted)), importance_sorted, color='#2ca02c')
    plt.title('Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(range(len(features_sorted)), features_sorted, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importance_sorted)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{imp:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: feature_importance.png")

def generate_shap_summary():
    """Generate SHAP summary plot placeholder."""
    # Create a placeholder SHAP summary plot
    plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.5, 'SHAP Summary Plot\n(Feature Importance from SHAP Values)',
             ha='center', va='center', fontsize=16, transform=plt.gcf().transFigure)
    plt.title('SHAP Summary Plot', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: shap_summary.png")

def generate_dashboard_screenshot():
    """Generate dashboard screenshot placeholder."""
    # Create a placeholder dashboard screenshot
    plt.figure(figsize=(14, 10))
    plt.text(0.5, 0.7, 'AI Traffic Accident Severity Prediction System',
             ha='center', va='center', fontsize=24, fontweight='bold', transform=plt.gcf().transFigure)
    plt.text(0.5, 0.6, 'Machine Learning-Based Road Safety Risk Analysis',
             ha='center', va='center', fontsize=18, style='italic', transform=plt.gcf().transFigure)
    plt.text(0.5, 0.5, '[Dashboard Interface Preview]',
             ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
             transform=plt.gcf().transFigure)
    plt.text(0.5, 0.3, 'Features:\n• Real-time accident severity prediction\n• Interactive input forms\n• SHAP explanations\n• Historical analytics\n• Model performance metrics',
             ha='center', va='center', fontsize=14, transform=plt.gcf().transFigure)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('visualizations/dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: dashboard.png")

def main():
    """Main function to generate all visualization assets."""
    print("Generating visualization assets for AI Traffic Accident Severity Prediction System...")
    print("=" * 80)

    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_data()
    print(f"Created dataset with shape: {df.shape}")
    print()

    # Generate all visualizations
    generate_severity_distribution(df)
    generate_accidents_by_year(df)
    generate_severity_by_weather(df)
    generate_severity_by_road_condition(df)
    generate_model_comparison()
    generate_confusion_matrix()
    generate_feature_importance()
    generate_shap_summary()
    generate_dashboard_screenshot()

    print("=" * 80)
    print("All visualization assets generated successfully!")
    print("Files saved in: visualizations/")
    print("\nGenerated files:")
    for file in sorted(os.listdir("visualizations")):
        if file.endswith(".png"):
            print(f"  - {file}")

if __name__ == "__main__":
    main()