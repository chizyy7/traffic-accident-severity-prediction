"""
Prediction module for AI Traffic Accident Severity Prediction System.
Handles loading trained models and making predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class AccidentPredictor:
    """Handles loading models and making predictions for accident severity."""

    def __init__(self, model_path: str = "models/",
                 encoder_path: str = "models/feature_encoders.joblib"):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the directory containing trained models
            encoder_path: Path to the feature encoders file
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoders = None
        self.feature_names = None
        self.target_encoder = None
        self.class_names = ['Minor', 'Moderate', 'Serious', 'Severe']

        # Try to load existing model and encoders
        self._load_model_and_encoders()

    def _load_model_and_encoders(self):
        """Load the trained model and feature encoders."""
        try:
            # Load encoders
            if os.path.exists(self.encoder_path):
                self.encoders = joblib.load(self.encoder_path)
                self.label_encoders = self.encoders.get('label_encoders', {})
                self.scaler = self.encoders.get('scaler', None)
                self.target_encoder = self.encoders.get('target_encoder', None)
                self.feature_names = self.encoders.get('feature_names', [])
                self.categorical_features = self.encoders.get('categorical_features', [])
                self.numerical_features = self.encoders.get('numerical_features', [])
                print(f"Encoders loaded from {self.encoder_path}")

            # Try to find and load the best model
            model_files = [f for f in os.listdir(self.model_path) if f.endswith('.joblib')]
            best_model_files = [f for f in model_files if 'best_model' in f or 'model' in f]

            if best_model_files:
                # Prefer explicitly named best model
                best_model_path = None
                for f in best_model_files:
                    if 'best_model' in f:
                        best_model_path = os.path.join(self.model_path, f)
                        break
                if not best_model_path:
                    best_model_path = os.path.join(self.model_path, best_model_files[0])

                self.model = joblib.load(best_model_path)
                print(f"Model loaded from {best_model_path}")
            else:
                print("Warning: No trained model found. Please train a model first.")

        except Exception as e:
            print(f"Error loading model or encoders: {e}")
            self.model = None
            self.encoders = None

    def load_model(self, model_filename: str = "best_model.joblib"):
        """
        Load a specific model file.

        Args:
            model_filename: Name of the model file to load
        """
        model_path = os.path.join(self.model_path, model_filename)
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def load_encoders(self, encoder_filename: str = "feature_encoders.joblib"):
        """
        Load specific encoder file.

        Args:
            encoder_filename: Name of the encoder file to load
        """
        encoder_path = os.path.join(self.model_path, encoder_filename)
        if os.path.exists(encoder_path):
            self.encoders = joblib.load(encoder_path)
            self.label_encoders = self.encoders.get('label_encoders', {})
            self.scaler = self.encoders.get('scaler', None)
            self.target_encoder = self.encoders.get('target_encoder', None)
            self.feature_names = self.encoders.get('feature_names', [])
            self.categorical_features = self.encoders.get('categorical_features', [])
            self.numerical_features = self.encoders.get('numerical_features', [])
            print(f"Encoders loaded from {encoder_path}")
        else:
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.

        Args:
            input_data: Input data as dictionary or DataFrame

        Returns:
            Preprocessed DataFrame ready for prediction
        """
        if self.encoders is None:
            raise ValueError("Encoders not loaded. Please load encoders first.")

        # Convert dictionary to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Ensure we have all expected columns (fill missing with defaults)
        expected_cols = set(self.categorical_features + self.numerical_features +
                          ['ID', 'Severity', 'Start_Time', 'End_Time'])  # Core columns

        # Add missing columns with appropriate defaults
        for col in expected_cols:
            if col not in df.columns:
                if col in self.categorical_features:
                    df[col] = 'Unknown'
                elif col in self.numerical_features:
                    df[col] = 0.0
                elif col in ['Start_Time', 'End_Time']:
                    df[col] = pd.NaT
                else:
                    df[col] = 0

        # Parse datetime columns
        time_cols = ['Start_Time', 'End_Time']
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Feature engineering (same as in training)
        from src.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        engineer.label_encoders = self.label_encoders
        engineer.scaler = self.scaler
        engineer.feature_names = self.feature_names
        engineer.categorical_features = self.categorical_features
        engineer.numerical_features = self.numerical_features

        # Apply feature engineering
        df_processed = engineer.create_weather_features(df)
        df_processed = engineer.create_road_features(df_processed)
        df_processed = engineer.create_temporal_features(df_processed)

        # Encode categorical features
        df_processed = engineer.encode_categorical_features(df_processed, fit=False)

        # Scale numerical features
        df_processed = engineer.scale_numerical_features(df_processed, fit=False)

        # Select final features
        exclude_cols = ['ID', 'Severity']
        exclude_cols.extend(self.categorical_features)
        time_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp']
        exclude_cols.extend([col for col in time_cols if col in df_processed.columns])

        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]

        # Ensure we have all expected features in the correct order
        final_df = df_processed[self.feature_names] if self.feature_names else df_processed[feature_cols]

        return final_df

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Make a prediction on input data.

        Args:
            input_data: Input data as dictionary or DataFrame

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        # Preprocess input
        processed_data = self.preprocess_input(input_data)

        # Make prediction
        prediction_encoded = self.model.predict(processed_data)
        prediction_proba = self.model.predict_proba(processed_data) if hasattr(self.model, "predict_proba") else None

        # Decode prediction
        if self.target_encoder is not None:
            prediction_label = self.target_encoder.inverse_transform(prediction_encoded)[0]
            prediction_class = self.class_names[prediction_label] if prediction_label < len(self.class_names) else f"Class_{prediction_label}"
        else:
            # Fallback: assume direct class labels (1-4)
            prediction_label = int(prediction_encoded[0])
            prediction_class = self.class_names[prediction_label - 1] if 1 <= prediction_label <= 4 else f"Class_{prediction_label}"

        # Get confidence/probability
        if prediction_proba is not None:
            confidence = float(np.max(prediction_proba))
            class_probabilities = {
                self.class_names[i]: float(prediction_proba[0][i])
                for i in range(len(self.class_names))
                if i < len(prediction_proba[0])
            }
        else:
            confidence = 1.0  # Default if no probabilities available
            class_probabilities = {self.class_names[int(prediction_label)-1]: 1.0} if 1 <= prediction_label <= 4 else {}

        # Prepare result
        result = {
            'prediction': prediction_class,
            'prediction_code': int(prediction_label),
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'input_features': processed_data.to_dict('records')[0] if len(processed_data) > 0 else {}
        }

        return result

    def predict_batch(self, input_data: pd.DataFrame) -> List[Dict]:
        """
        Make predictions on a batch of input data.

        Args:
            input_data: Input DataFrame with multiple samples

        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        # Preprocess input
        processed_data = self.preprocess_input(input_data)

        # Make predictions
        predictions_encoded = self.model.predict(processed_data)
        predictions_proba = self.model.predict_proba(processed_data) if hasattr(self.model, "predict_proba") else None

        # Decode predictions
        if self.target_encoder is not None:
            predictions_labels = self.target_encoder.inverse_transform(predictions_encoded)
            predictions_classes = [self.class_names[label] if label < len(self.class_names) else f"Class_{label}"
                                 for label in predictions_labels]
        else:
            # Fallback: assume direct class labels (1-4)
            predictions_labels = [int(label) for label in predictions_encoded]
            predictions_classes = [self.class_names[label - 1] if 1 <= label <= 4 else f"Class_{label}"
                                 for label in predictions_labels]

        # Get confidences
        if predictions_proba is not None:
            confidences = [np.max(proba) for proba in predictions_proba]
            all_probabilities = []
            for i, proba in enumerate(predictions_proba):
                class_probabilities = {
                    self.class_names[j]: float(proba[j])
                    for j in range(len(self.class_names))
                    if j < len(proba)
                }
                all_probabilities.append(class_probabilities)
        else:
            confidences = [1.0] * len(predictions_encoded)
            all_probabilities = [{}] * len(predictions_encoded)

        # Prepare results
        results = []
        for i in range(len(processed_data)):
            result = {
                'prediction': predictions_classes[i],
                'prediction_code': int(predictions_labels[i]),
                'confidence': float(confidences[i]),
                'class_probabilities': all_probabilities[i],
                'input_features': processed_data.iloc[i].to_dict()
            }
            results.append(result)

        return results

    def explain_prediction(self, input_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Provide explanation for a prediction (feature contributions).

        Args:
            input_data: Input data as dictionary or DataFrame

        Returns:
            Dictionary with prediction and explanation
        """
        # Get basic prediction
        result = self.predict(input_data)

        # Add feature importance explanation if available
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            # Get the input features
            processed_data = self.preprocess_input(input_data)
            feature_values = processed_data.iloc[0] if len(processed_data) > 0 else pd.Series()

            # Create feature importance dictionary
            if len(self.feature_names) == len(self.model.feature_importances_):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                # Sort by importance
                sorted_importance = sorted(feature_importance.items(),
                                         key=lambda x: abs(x[1]), reverse=True)

                # Get top contributing features
                top_features = sorted_importance[:10]  # Top 10 features

                result['explanation'] = {
                    'method': 'feature_importance',
                    'top_features': [
                        {
                            'feature': feat,
                            'value': float(feature_values.get(feat, 0)) if feat in feature_values else 0,
                            'importance': float(imp)
                        }
                        for feat, imp in top_features
                    ]
                }
            else:
                result['explanation'] = {
                    'method': 'feature_importance',
                    'message': 'Feature names and importances length mismatch'
                }
        else:
            result['explanation'] = {
                'method': 'none',
                'message': 'Model does not support feature importance explanations'
            }

        return result


def main():
    """Main function for testing the predictor."""
    print("Accident Predictor Module - Testing")

    # Initialize predictor
    predictor = AccidentPredictor()

    # Check if model and encoders are loaded
    if predictor.model is None:
        print("No model found. Please train a model first using train.py")
        print("Creating sample prediction for demonstration...")

        # Create sample input for testing
        sample_input = {
            'ID': 'TEST001',
            'Start_Time': '2023-06-15 14:30:00',
            'End_Time': '2023-06-15 15:00:00',
            'Temperature(F)': 72.0,
            'Visibility(mi)': 10.0,
            'Weather_Condition': 'Clear',
            'Wind_Speed(mph)': 5.0,
            'Humidity(%)': 60,
            'Pressure(in)': 30.0,
            'Distance(mi)': 0.5,
            'Street': 'Test Street',
            'City': 'Test City',
            'County': 'Test County',
            'State': 'TX',
            'Zipcode': '12345',
            'Country': 'US',
            'Timezone': 'America/Chicago',
            'Airport_Code': 'TEST',
            'Weather_Timestamp': '2023-06-15 14:00:00',
            'Amenity': False,
            'Bump': False,
            'Crossing': True,
            'Give_Way': False,
            'Junction': False,
            'No_Exit': False,
            'Railway': False,
            'Roundabout': False,
            'Station': False,
            'Stop': False,
            'Traffic_Calming': False,
            'Traffic_Signal': False,
            'Turning_Loop': False,
            'Sunrise_Sunset': 'Day',
            'Civil_Twilight': 'Day',
            'Nautical_Twilight': 'Night',
            'Astronomical_Twilight': 'Night'
        }

        # Since we don't have a real model, show what the prediction would look like
        print("\nSample Input:")
        for key, value in sample_input.items():
            print(f"  {key}: {value}")

        print("\nExpected Output Format:")
        print("  Prediction: Moderate")
        print("  Confidence: 0.75")
        print("  Class Probabilities: {'Minor': 0.15, 'Moderate': 0.75, 'Serious': 0.08, 'Severe': 0.02}")
        print("  Explanation: Top contributing features would be shown here")

    else:
        # Test with actual model if available
        print("Model found. Testing prediction...")

        # Create sample input
        sample_input = {
            'ID': 'TEST001',
            'Start_Time': '2023-06-15 14:30:00',
            'Temperature(F)': 72.0,
            'Visibility(mi)': 10.0,
            'Weather_Condition': 'Clear',
            'Distance(mi)': 0.5,
            'Junction': False,
        }

        try:
            # Make prediction
            result = predictor.predict(sample_input)
            print("\nPrediction Result:")
            print(f"  Severity: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Probabilities: {result['class_probabilities']}")

            # Get explanation
            explanation = predictor.explain_prediction(sample_input)
            if 'explanation' in explanation:
                print(f"  Explanation method: {explanation['explanation']['method']}")

        except Exception as e:
            print(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()