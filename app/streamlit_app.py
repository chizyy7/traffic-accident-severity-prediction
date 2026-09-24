"""
Streamlit Dashboard for AI Traffic Accident Severity Prediction System.
Provides an interactive interface for predicting accident severity and exploring data insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import AccidentPredictor
from explain import AccidentExplainer
from evaluate import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="AI Traffic Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .severity-minor { border-left-color: #2ca02c; }
    .severity-moderate { border-left-color: #ff7f0e; }
    .severity-serious { border-left-color: #d62728; }
    .severity-severe { border-left-color: #9467bd; }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_model_and_explainer():
    """Load the trained model and explainer."""
    try:
        predictor = AccidentPredictor()
        explainer = AccidentExplainer()
        return predictor, explainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def load_visualizations():
    """Load pre-generated visualizations."""
    viz_path = "visualizations/"
    viz_files = {}

    if os.path.exists(viz_path):
        for file in os.listdir(viz_path):
            if file.endswith('.png'):
                viz_files[file.replace('.png', '')] = os.path.join(viz_path, file)

    return viz_files


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<h1 class="main-header">AI Traffic Accident Severity Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning-Based Road Safety Risk Analysis</p>', unsafe_allow_html=True)

    # Load model and explainer
    predictor, explainer = load_model_and_explainer()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔮 Make Prediction", "📊 Model Insights", "📈 Data Analytics", "ℹ️ About"]
    )

    # Route to selected page
    if page == "🏠 Home":
        show_home_page(predictor, explainer)
    elif page == "🔮 Make Prediction":
        show_prediction_page(predictor, explainer)
    elif page == "📊 Model Insights":
        show_insights_page()
    elif page == "📈 Data Analytics":
        show_analytics_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page(predictor, explainer):
    """Display the home page."""

    st.header("Welcome to the Accident Severity Prediction System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This system uses machine learning to predict the severity of traffic accidents
        based on historical data and environmental conditions.

        ### Features:
        - **Accurate Predictions**: Trained on 7.7M+ real accident records from 2016-2023
        - **Interpretability**: SHAP values explain why each prediction was made
        - **User-Friendly**: Simple interface for domain experts and the public
        - **Comprehensive Analytics**: Explore patterns in accident data

        ### How it works:
        1. Enter accident conditions (weather, time, location, etc.)
        2. The model processes the features using the same pipeline as training
        3. Get a severity prediction with confidence scores
        4. View explanations for why the model made that prediction
        """)

    with col2:
        st.markdown("### System Status")
        if predictor is not None and predictor.model is not None:
            st.success("✅ Model Loaded Successfully")
            st.info(f"Model Type: {type(predictor.model).__name__}")
        else:
            st.warning("⚠️ No Model Loaded")
            st.info("Please train a model first using the training script")

    # Display key metrics if available
    st.header("System Overview")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Dataset Size",
            value="7.7M+",
            help="Number of accident records in the training dataset"
        )

    with metric_col2:
        st.metric(
            label="Years Covered",
            value="2016-2023",
            help="Temporal range of the training data"
        )

    with metric_col3:
        st.metric(
            label="Severity Classes",
            value="4",
            help="Minor, Moderate, Serious, Severe"
        )

    with metric_col4:
        if predictor is not None and predictor.model is not None:
            st.metric(
                label="Model Status",
                value="Ready",
                help="Model is loaded and ready for predictions"
            )
        else:
            st.metric(
                label="Model Status",
                value="Not Loaded",
                help="Please train and load a model"
            )

    # Sample visualization
    viz_files = load_visualizations()
    if 'model_comparison' in viz_files:
        st.header("Model Performance Overview")
        st.image(viz_files['model_comparison'], caption="Model Comparison Chart")
    elif 'dashboard' in viz_files:
        st.header("System Preview")
        st.image(viz_files['dashboard'], caption="Dashboard Preview")


def show_prediction_page(predictor, explainer):
    """Display the prediction page."""

    st.header("Make Accident Severity Prediction")

    if predictor is None or predictor.model is None:
        st.error("⚠️ No model available. Please train a model first.")
        st.info("Run: python src/train.py to train a model")
        return

    # Input form
    st.subheader("Enter Accident Conditions")

    # Create input form with organized sections
    with st.form("prediction_form"):
        # Temporal Information
        st.markdown("### 🕐 Time Information")
        temp_col1, temp_col2, temp_col3 = st.columns(3)

        with temp_col1:
            accident_date = st.date_input(
                "Date",
                value=datetime.now().date(),
                min_value=datetime(2016, 1, 1).date(),
                max_value=datetime(2023, 12, 31).date()
            )

        with temp_col2:
            accident_time = st.time_input(
                "Time",
                value=datetime.now().time().replace(second=0, microsecond=0)
            )

        with temp_col3:
            # Create combined datetime
            accident_datetime = datetime.combine(accident_date, accident_time)
            st.text_input("Combined DateTime", value=accident_datetime.strftime("%Y-%m-%d %H:%M:%S"), disabled=True)

        # Environmental Conditions
        st.markdown("### 🌤️ Environmental Conditions")
        env_col1, env_col2, env_col3, env_col4 = st.columns(4)

        with env_col1:
            temperature = st.number_input(
                "Temperature (°F)",
                min_value=-20.0,
                max_value=130.0,
                value=70.0,
                step=0.5
            )

        with env_col2:
            visibility = st.number_input(
                "Visibility (miles)",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1
            )

        with env_col3:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=1
            )

        with env_col4:
            pressure = st.number_input(
                "Pressure (in)",
                min_value=25.0,
                max_value=32.0,
                value=30.0,
                step=0.01
            )

        # Weather Conditions
        st.markdown("### 🌧️ Weather Details")
        weather_col1, weather_col2 = st.columns(2)

        with weather_col1:
            weather_condition = st.selectbox(
                "Weather Condition",
                ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Storm", "Windy", "Haze", "Sleet", "Hail"]
            )

        with weather_col2:
            wind_speed = st.number_input(
                "Wind Speed (mph)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1
            )

        # Location Information
        st.markdown("### 📍 Location Information")
        loc_col1, loc_col2, loc_col3 = st.columns(3)

        with loc_col1:
            street = st.text_input("Street", value="Main Street")
            city = st.text_input("City", value="Unknown")

        with loc_col2:
            county = st.text_input("County", value="Unknown")
            state = st.selectbox(
                "State",
                ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                 "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                 "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                 "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                 "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"],
                index=32  # Default to TX
            )

        with loc_col3:
            zipcode = st.text_input("ZIP Code", value="00000")
            country = st.selectbox("Country", ["US"], index=0)

        # Road Characteristics
        st.markdown("### 🛣️ Road Characteristics")
        road_col1, road_col2, road_col3, road_col4 = st.columns(4)

        with road_col1:
            distance = st.number_input(
                "Distance (miles)",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1
            )

        with road_col2:
            junction = st.checkbox("Junction")

        with road_col3:
            traffic_signal = st.checkbox("Traffic Signal")

        with road_col4:
            stop = st.checkbox("Stop Sign")

        # Additional road features
        road_col5, road_col6, road_col7, road_col8 = st.columns(4)

        with road_col5:
            bump = st.checkbox("Speed Bump")

        with road_col6:
            crossing = st.checkbox("Pedestrian Crossing")

        with road_col7:
            railway = st.checkbox("Railway Crossing")

        with road_col8:
            roundabout = st.checkbox("Roundabout")

        # Time-related features (derived)
        st.markdown("### ⏰ Derived Features (Auto-calculated)")
        derived_col1, derived_col2, derived_col3 = st.columns(3)

        with derived_col1:
            # Calculate day of week
            day_of_week = accident_datetime.weekday()
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            st.text_input("Day of Week", value=day_names[day_of_week], disabled=True)

        with derived_col2:
            # Calculate hour
            hour = accident_datetime.hour
            st.text_input("Hour of Day", value=str(hour), disabled=True)

            # Rush hour indicator
            is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 18)
            st.text_input("Rush Hour", value="Yes" if is_rush_hour else "No", disabled=True)

        with derived_col3:
            # Night indicator
            is_night = hour >= 20 or hour <= 5
            st.text_input("Night Time", value="Yes" if is_night else "No", disabled=True)

            # Weekend indicator
            is_weekend = day_of_week >= 5  # Saturday or Sunday
            st.text_input("Weekend", value="Yes" if is_weekend else "No", disabled=True)

        # Submit button
        submitted = st.form_submit_button("🔍 Predict Accident Severity")

        if submitted:
            # Prepare input data
            input_data = {
                'ID': 'PRED001',
                'Start_Time': accident_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                'End_Time': (accident_datetime.replace(minute=accident_datetime.minute+30)).strftime("%Y-%m-%d %H:%M:%S"),  # 30 minutes later
                'Temperature(F)': temperature,
                'Visibility(mi)': visibility,
                'Wind_Speed(mph)': wind_speed,
                'Humidity(%)': humidity,
                'Pressure(in)': pressure,
                'Weather_Condition': weather_condition,
                'Distance(mi)': distance,
                'Street': street,
                'City': city,
                'County': county,
                'State': state,
                'Zipcode': zipcode,
                'Country': country,
                'Timezone': 'America/New_York',  # Simplified
                'Airport_Code': 'UNKNOWN',
                'Weather_Timestamp': accident_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                'Amenity': False,
                'Bump': bump,
                'Crossing': crossing,
                'Give_Way': False,
                'Junction': junction,
                'No_Exit': False,
                'Railway': railway,
                'Roundabout': roundabout,
                'Station': False,
                'Stop': stop,
                'Traffic_Calming': False,
                'Traffic_Signal': traffic_signal,
                'Turning_Loop': False,
                'Sunrise_Sunset': 'Day' if 6 <= hour <= 18 else 'Night',
                'Civil_Twilight': 'Day' if 6 <= hour <= 18 else 'Night',
                'Nautical_Twilight': 'Night',
                'Astronomical_Twilight': 'Night'
            }

            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    # Get prediction
                    prediction_result = predictor.predict(input_data)

                    # Get explanation
                    explanation_result = explainer.explain_prediction(input_data)

                    # Display results
                    st.header("Prediction Results")

                    # Main prediction box
                    severity = prediction_result['prediction']
                    confidence = prediction_result['confidence']

                    # Determine CSS class based on severity
                    severity_class = f"severity-{severity.lower()}"

                    st.markdown(f"""
                    <div class="prediction-box {severity_class}">
                        <h2>Predicted Severity: {severity}</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability visualization
                    st.subheader("Severity Probabilities")
                    probs = prediction_result['class_probabilities']

                    if probs:
                        # Create probability bars
                        prob_df = pd.DataFrame(list(probs.items()), columns=['Severity', 'Probability'])
                        prob_df = prob_df.sort_values('Probability', ascending=False)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(prob_df['Severity'], prob_df['Probability'],
                                     color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
                        ax.set_ylabel('Probability')
                        ax.set_title('Accident Severity Prediction Probabilities')
                        ax.set_ylim(0, 1)

                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.2f}', ha='center', va='bottom')

                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()

                    # Explanation section
                    st.subheader("Explanation")

                    if explanation_result.get('explanation_method') == 'shap':
                        st.write("**Why did the model make this prediction?**")

                        # Top positive features (increase severity)
                        pos_features = explanation_result.get('top_positive_features', [])
                        if pos_features:
                            st.write("**Factors increasing severity:**")
                            for feat in pos_features[:3]:
                                st.write(f"• {feat['feature']} (SHAP value: {feat['shap_value']:.3f})")

                        # Top negative features (decrease severity)
                        neg_features = explanation_result.get('top_negative_features', [])
                        if neg_features:
                            st.write("**Factors decreasing severity:**")
                            for feat in neg_features[:3]:
                                st.write(f"• {feat['feature']} (SHAP value: {feat['shap_value']:.3f})")

                    elif explanation_result.get('explanation_method') == 'feature_importance':
                        st.write("**Feature Importance Explanation:**")
                        top_features = explanation_result.get('top_features', [])
                        if top_features:
                            for feat in top_features[:5]:
                                st.write(f"• {feat['feature']} (Importance: {feat['importance']:.3f})")

                    else:
                        st.info("Explanation not available for this model type.")

                    # Show input summary
                    with st.expander("View Input Summary"):
                        st.json(input_data)

                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.exception(e)


def show_insights_page():
    """Display the model insights page."""

    st.header("Model Insights & Performance")

    # Load visualizations
    viz_files = load_visualizations()

    if not viz_files:
        st.warning("No visualization files found. Please run the training and evaluation scripts first.")
        st.info("Expected location: visualizations/ directory")
        return

    # Model performance comparison
    if 'model_comparison' in viz_files:
        st.subheader("Model Performance Comparison")
        st.image(viz_files['model_comparison'], use_column_width=True)

    # Individual model metrics
    metric_cols = st.columns(2)

    with metric_cols[0]:
        if 'confusion_matrix' in viz_files:
            st.subheader("Confusion Matrix")
            st.image(viz_files['confusion_matrix'], use_column_width=True)

    with metric_cols[1]:
        if 'feature_importance' in viz_files:
            st.subheader("Feature Importance")
            st.image(viz_files['feature_importance'], use_column_width=True)

    # SHAP explanation
    if 'shap_summary' in viz_files:
        st.subheader("SHAP Summary Plot")
        st.image(viz_files['shap_summary'], use_column_width=True)

    # ROC Curve
    if 'roc_curve' in viz_files:
        st.subheader("ROC Curve")
        st.image(viz_files['roc_curve'], use_column_width=True)

    # Model details
    st.subheader("Model Details")

    # Try to load training results
    try:
        training_results = joblib.load("models/training_results.joblib")
        if training_results:
            st.write("**Best Model:**", training_results.get('best_model_name', 'Unknown'))

            if 'model_scores' in training_results:
                st.write("**Model Performance:**")
                scores_df = pd.DataFrame(training_results['model_scores']).T
                # Select relevant metrics for display
                display_cols = [col for col in scores_df.columns if any(metric in col for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])]
                if display_cols:
                    st.dataframe(scores_df[display_cols].round(4))
    except Exception as e:
        st.info("Training results not available. Please run the training script first.")


def show_analytics_page():
    """Display the data analytics page."""

    st.header("Data Analytics & Historical Trends")

    # Load visualizations
    viz_files = load_visualizations()

    if not viz_files:
        st.warning("No visualization files found. Please run the data processing and analysis scripts first.")
        return

    # Accidents over time
    if 'accidents_by_year' in viz_files:
        st.subheader("Accidents by Year")
        st.image(viz_files['accidents_by_year'], use_column_width=True)

    # Accidents by severity
    if 'severity_distribution' in viz_files:
        st.subheader("Severity Distribution")
        st.image(viz_files['severity_distribution'], use_column_width=True)

    # Weather impact
    weather_cols = st.columns(2)

    with weather_cols[0]:
        if 'severity_by_weather' in viz_files:
            st.subheader("Severity by Weather Condition")
            st.image(viz_files['severity_by_weather'], use_column_width=True)

    with weather_cols[1]:
        if 'severity_by_road_condition' in viz_files:
            st.subheader("Severity by Road Condition")
            st.image(viz_files['severity_by_road_condition'], use_column_width=True)

    # Lighting and time impact
    lighting_cols = st.columns(2)

    with lighting_cols[0]:
        if 'severity_by_lighting' in viz_files:
            st.subheader("Severity by Lighting Condition")
            st.image(viz_files['severity_by_lighting'], use_column_width=True)

    with lighting_cols[1]:
        if 'severity_by_hour' in viz_files:
            st.subheader("Accidents by Hour of Day")
            st.image(viz_files['severity_by_hour'], use_column_width=True)

    # Road type impact
    if 'severity_by_road_type' in viz_files:
        st.subheader("Severity by Road Type")
        st.image(viz_files['severity_by_road_type'], use_column_width=True)

    # Speed limit impact
    if 'severity_by_speed_limit' in viz_files:
        st.subheader("Severity by Speed Limit")
        st.image(viz_files['severity_by_speed_limit'], use_column_width=True)

    # Geographic visualization (if available)
    if 'geographic_distribution' in viz_files:
        st.subheader("Geographic Distribution of Accidents")
        st.image(viz_files['geographic_distribution'], use_column_width=True)

    # Correlation matrix
    if 'correlation_matrix' in viz_files:
        st.subheader("Feature Correlation Matrix")
        st.image(viz_files['correlation_matrix'], use_column_width=True)


def show_about_page():
    """Display the about page."""

    st.header("About This System")

    st.markdown("""
    ### AI Traffic Accident Severity Prediction System

    This is a machine learning system designed to predict the severity of traffic accidents
    based on historical accident data from the United States.

    #### 🎯 Purpose
    To provide accurate, interpretable predictions of accident severity to help with:
    - Road safety planning and prevention
    - Emergency response preparation
    - Traffic management decisions
    - Public awareness of risk factors

    #### 📊 Dataset
    - **Source**: US Accidents Dataset (Kaggle)
    - **Period**: February 2016 - March 2023
    - **Records**: 7.7M+ accident records
    - **Features**: 46+ including temporal, spatial, weather, and road conditions
    - **Target**: Severity level (1-4 scale)

    #### 🤖 Models Evaluated
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost
    - LightGBM

    #### 🔍 Explainability
    - SHAP (SHapley Additive exPlanations) values for individual predictions
    - Feature importance analysis
    - Global and local explanations

    #### 💻 Technology Stack
    - **Language**: Python 3.9+
    - **ML Libraries**: scikit-learn, XGBoost, LightGBM
    - **Explainability**: SHAP
    - **Web Interface**: Streamlit
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Data Processing**: Pandas, NumPy

    #### ⚠️ Important Limitations
    - This is a research/educational system
    - Predictions are based on historical patterns and may not account for real-time changes
    - Should not be used as the sole basis for critical safety or legal decisions
    - Model performance varies by region and accident type
    - Data may have reporting biases and inconsistencies

    #### 📝 Ethical Considerations
    - Geographic bias: Model may perform differently in underrepresented areas
    - Historical bias: Learns from past reporting patterns which may not reflect current realities
    - Class imbalance: Severe accidents are rarer and may be predicted less accurately
    - Privacy: No personally identifiable information is used or stored

    #### 👨‍💻 Developer
    Created by chizy as a portfolio project demonstrating end-to-end ML system development.

    #### 📚 References
    - US Accidents Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/
    - SHAP Library: https://shap.readthedocs.io/
    - Streamlit: https://streamlit.io/
    """)

    # Model card section
    st.subheader("Model Card")

    try:
        # Try to load or display model card information
        model_info = {
            "Model Purpose": "Predict traffic accident severity (1-4 scale)",
            "Training Data": "US Accidents 2016-2023 (7.7M+ records)",
            "Evaluation": "Temporal split: Train on 2016-2020, Validate on 2021, Test on 2022",
            "Key Features": "Temporal, weather, road, location characteristics",
            "Performance": "Varies by model - see Model Insights page",
            "Limitations": "Temporal limitations, geographic bias, class imbalance",
            "Ethical Use": "Educational/research purposes only",
            "Inappropriate Use": "Emergency response, legal decisions, insurance underwriting without validation"
        }

        for key, value in model_info.items():
            st.text(f"{key}: {value}")

    except Exception:
        st.info("Model card information not available.")


if __name__ == "__main__":
    main()