# AI Traffic Accident Severity Prediction System

## Overview
A machine learning system to predict the severity of traffic accidents using historical accident data from the US Accidents dataset (2016-2023). The system analyzes environmental, temporal, road, vehicle, and traffic-related features to accurately classify accident severity into four levels: Minor, Moderate, Serious, and Severe.

![Dashboard](visualizations/dashboard.png)

## Project Motivation
Traffic accidents remain a leading cause of injury and death worldwide. According to the World Health Organization, approximately 1.35 million people die each year as a result of road traffic crashes. Predicting accident severity can help:
- Emergency services prioritize responses
- Transportation agencies implement preventive measures
- Drivers make informed decisions about route safety
- Urban planners design safer road infrastructure

## Objectives
1. Develop an end-to-end ML pipeline for accident severity prediction
2. Implement proper temporal validation to prevent data leakage
3. Create interpretable models using SHAP values for explainability
4. Build an interactive Streamlit dashboard for real-world usage
5. Generate comprehensive documentation and visualizations for portfolio presentation

## Dataset
- **Name**: US Accidents Dataset
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/
- **Time Period**: February 2016 - March 2023
- **Number of Records**: 7.7M+ accident records (sample used for development)
- **Features**: 40+ including temporal, location, weather, road conditions, and point-of-interest features
- **Target Variable**: Severity (1-4 scale)
  - 1: Minor (least severe)
  - 2: Moderate
  - 3: Serious
  - 4: Severe (most severe)
- **License**: Publicly available for research and educational purposes

## Methodology
The project follows a structured ML pipeline:

![Model Comparison](visualizations/model_comparison.png)

1. **Data Collection**: Acquisition of US Accidents dataset from Kaggle
2. **Data Cleaning**: Handling missing values, duplicate records, and inconsistent data
3. **Feature Engineering**: Creation of temporal, weather, road, and traffic-related features
4. **Exploratory Data Analysis**: Understanding patterns and relationships in the data
5. **Data Leakage Prevention**: Removal of post-accident features that would not be available at prediction time
6. **Model Training**: Training multiple ML algorithms with consistent preprocessing
7. **Hyperparameter Tuning**: Optimization of top-performing models
8. **Model Evaluation**: Comprehensive assessment using multiple metrics
9. **Model Interpretation**: SHAP analysis for explainability
10. **Deployment**: Interactive Streamlit dashboard for real-time predictions

### Temporal Validation Strategy
To prevent temporal leakage and better simulate real-world prediction scenarios, the data was split temporally:
- **Training**: 2016-2020 (older historical data)
- **Validation**: 2021 (recent historical data for tuning)
- **Testing**: 2022-2023 (most recent data for final evaluation)

This approach ensures the model learns from past accidents to predict future ones, eliminating look-ahead bias.

### Data Leakage Prevention
Features that could reveal accident severity after the accident occurred were identified and removed:

**Removed Features:**
- `End_Time`: Only available after accident concludes
- `Distance(mi)`: While partially available, exact measurement requires post-accident assessment
- `Description`: Text field containing post-accident details
- `Number of Casualties`: Directly related to severity (would create circular reasoning)
- `Vehicle Count`: Often determined after accident investigation

**Retained Features (Available at Prediction Time):**
- Temporal: Time of day, day of week, month, year, season
- Environmental: Temperature, precipitation, visibility, wind conditions, weather type
- Road: Junction presence, traffic signals, road curvature indicators
- Location: City, state, zipcode (general area, not exact coordinates)
- Point-of-interest: Amenities, crossings, railway proximity (known Infrastructure)

![Severity Distribution](visualizations/severity_distribution.png)
*Figure: Distribution of accident severity levels in the dataset*

## Machine Learning Models
The following models were trained and compared:

1. **Logistic Regression**: Interpretable baseline model
2. **Decision Tree**: Non-parametric model for capturing non-linear relationships
3. **Random Forest**: Ensemble method reducing overfitting
4. **XGBoost**: Gradient boosting with regularization
5. **LightGBM**: Efficient gradient boosting implementation

All models used identical preprocessing pipelines to ensure fair comparison.

## Results

### Model Performance Comparison
The table below shows the performance of each model on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------:|----------:|-------:|---------:|---------:|
| Logistic Regression | 0.672 | 0.658 | 0.672 | 0.665 | 0.721 |
| Decision Tree | 0.685 | 0.662 | 0.685 | 0.673 | 0.708 |
| Random Forest | 0.712 | 0.698 | 0.712 | 0.705 | 0.745 |
| XGBoost | 0.734 | 0.721 | 0.734 | 0.727 | 0.768 |
| LightGBM | 0.738 | 0.725 | 0.738 | 0.731 | 0.772 |

![Confusion Matrix](visualizations/confusion_matrix.png)
*Figure: Confusion matrix for the best performing model (LightGBM)*

### Feature Importance
The most influential features for predicting accident severity:

![Feature Importance](visualizations/feature_importance.png)
*Figure: Relative importance of features in the LightGBM model*

### SHAP Analysis
SHAP (SHapley Additive exPlanations) values provide interpretable explanations for individual predictions:

![SHAP Summary](visualizations/shap_summary.png)
*Figure: SHAP summary plot showing feature impact on model predictions*

### Key Insights from EDA
Exploratory data analysis revealed important patterns:

**Temporal Patterns:**
- Accidents peak during rush hours (7-9 AM, 4-6 PM)
- Higher accident rates on weekends
- Seasonal variations with increased incidents in adverse weather months

**Environmental Factors:**
- Severe accidents more likely in poor visibility conditions
- Nighttime accidents show higher severity rates
- Rain and snow conditions correlate with increased accident severity

**Road Characteristics:**
- Junctions and intersections increase accident risk
- Poor road visibility significantly impacts severity
- Urban areas with complex intersections show higher severe accident rates

![Severity by Weather](visualizations/severity_by_weather.png)
*Figure: Accident severity distribution by weather condition*

![Severity by Road Condition](visualizations/severity_by_road_condition.png)
*Figure: Accident severity by visibility, lighting, junction type, and distance categories*

## Model Comparison Visualization
![Model Comparison](visualizations/model_comparison.png)
*Figure: Performance comparison across all evaluated models*

## Confusion Matrix
![Confusion Matrix](visualizations/confusion_matrix.png)
*Figure: Detailed confusion matrix showing classification performance per severity class*

## Feature Importance
![Feature Importance](visualizations/feature_importance.png)
*Figure: Relative importance of top 16 features in the final model*

## SHAP Analysis
![SHAP Summary](visualizations/shap_summary.png)
*Figure: SHAP summary plot illustrating feature contributions to predictions*

## Dashboard Screenshots
![Dashboard](visualizations/dashboard.png)
*Figure: Main interface of the Streamlit dashboard showing prediction and analytics sections*

## Demo
To launch the interactive dashboard:
```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
1. **Prediction Interface**: Input accident conditions to get severity predictions with confidence scores
2. **Model Insights**: View feature importance, SHAP explanations, and model performance metrics
3. **Data Analytics**: Explore historical patterns and trends in accident data
4. **About Section**: Learn about the project, dataset, and limitations

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-accident-severity-prediction.git
cd traffic-accident-severity-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the data:
   - Download the US Accidents dataset from Kaggle
   - Place `US_Accidents.csv` in the `data/raw/` directory
   - (For development, the system includes sample data generation)

4. Train the model (optional - pre-trained model included):
```bash
python src/train.py
```

5. Launch the dashboard:
```bash
streamlit run app/streamlit_app.py
```

## Project Structure
```
traffic-accident-severity-prediction/
├── data/
│   ├── raw/              # Raw dataset (US_Accidents.csv)
│   └── processed/        # Cleaned and processed data
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and visualizations
│   ├── 02_data_cleaning.ipynb         # Data cleaning procedures
│   ├── 03_feature_engineering.ipynb   # Feature engineering steps
│   ├── 04_model_training.ipynb        # Model training and evaluation
│   └── 05_model_explainability.ipynb  # SHAP analysis and interpretation
├── src/
│   ├── data_processing.py     # Data loading and cleaning
│   ├── feature_engineering.py # Feature creation and encoding
│   ├── train.py               # Model training and evaluation
│   ├── predict.py             # Prediction interface
│   └── explain.py             # SHAP and feature importance explanations
├── models/
│   ├── best_model.joblib      # Trained LightGBM model
│   ├── feature_encoders.joblib # Encoders and scalers
│   └── training_results.joblib # Model performance metrics
├── app/
│   └── streamlit_app.py       # Streamlit dashboard interface
├── visualizations/            # Generated plots for README and documentation
│   ├── severity_distribution.png
│   ├── accidents_by_year.png
│   ├── severity_by_weather.png
│   ├── severity_by_road_condition.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── dashboard.png
├── tests/
│   ├── test_data_processing.py
│   └── test_feature_engineering.py
├── requirements.txt
├── README.md
├── MODEL_CARD.md
└── LICENSE
```

## Technologies Used
- **Python 3.9+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and utilities
- **XGBoost**: Extreme gradient boosting
- **LightGBM**: Light gradient boosting machine
- **SHAP**: Model interpretability and explanation
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization
- **Streamlit**: Interactive web application framework
- **Joblib**: Model persistence

## Limitations
1. **Dataset Limitations**: 
   - Sample data used for development (full dataset recommended for production)
   - Potential reporting biases in accident data
   - Limited geographical coordinates in public dataset

2. **Temporal Limitations**:
   - Model trained on 2016-2023 data may not capture recent changes in road safety
   - Seasonal patterns may vary by region

3. **Class Imbalance**: 
   - Severe accidents are rarer (approx. 5% of dataset)
   - Model may be conservative in predicting severe cases

4. **Geographical Limitations**:
   - Model may not generalize equally across all regions
   - Local road conditions and regulations not fully captured

5. **Potential Bias**:
   - Historical reporting biases may affect predictions
   - Socioeconomic factors not explicitly modeled

6. **Generalization Limitations**:
   - Model optimized for US accident patterns
   - May require retraining for other countries

## Future Improvements
1. **Real-time Data Integration**:
   - Live weather API integration
   - Real-time traffic flow data
   - Dynamic road condition updates

2. **Enhanced Modeling**:
   - Deep learning approaches (LSTM for temporal patterns)
   - Geographic modeling with spatial algorithms
   - Ensemble methods with uncertainty quantification

3. **Dashboard Enhancements**:
   - Real-time prediction updates
   - Geographic heatmaps
   - Custom scenario building
   - Export functionality for reports

4. **Model Improvements**:
   - Federated learning for multi-institution collaboration
   - Continuous learning from new accident data
   - Uncertainty estimation for prediction confidence

5. **Expansion**:
   - International datasets for global applicability
   - Integration with navigation systems
   - API for third-party applications

## Author
**chizy** - Machine Learning Engineer and Data Scientist
- Portfolio project demonstrating end-to-end ML system development
- Expertise in machine learning, data science, and software engineering
- Focus on interpretable, ethical, and socially responsible AI applications

## Acknowledgements
- US Accidents Dataset contributors and maintainers
- Open-source ML library developers (scikit-learn, XGBoost, LightGBM, SHAP)
- Streamlit community for the excellent framework
- Kaggle for providing access to the dataset

## References
1. US Accidents Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/
2. World Health Organization. (2023). Road traffic injuries. 
3. Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games.
4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.