"""
Model explainability module for AI Traffic Accident Severity Prediction System.
Handles SHAP values and feature interpretation for model predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class AccidentExplainer:
    """Handles model explainability using SHAP and feature importance."""

    def __init__(self, model_path: str = "models/",
                 encoder_path: str = "models/feature_encoders.joblib"):
        """
        Initialize the explainer.

        Args:
            model_path: Path to the directory containing trained models
            encoder_path: Path to the feature encoders file
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoders = None
        self.feature_names = None
        self.class_names = ['Minor', 'Moderate', 'Serious', 'Severe']
        self.explainer = None
        self.shap_values = None

        # Try to load existing model and encoders
        self._load_model_and_encoders()

        # Initialize SHAP explainer if model is loaded and SHAP is available
        if self.model is not None and SHAP_AVAILABLE:
            self._initialize_shap_explainer()

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

    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Cannot initialize explainer.")
            return

        try:
            # Determine appropriate explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models, use TreeExplainer
                if any(tree_type in str(type(self.model)).lower() for tree_type in ['tree', 'forest', 'boost', 'gbm']):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # For other models, use KernelExplainer (slower but works with any model)
                    # We need background data for KernelExplainer - this is a limitation
                    print("Warning: Using KernelExplainer requires background data. Consider using TreeExplainer for tree-based models.")
                    self.explainer = None
            else:
                print("Warning: Model does not have predict_proba method. SHAP explanation may be limited.")

            if self.explainer is not None:
                print("SHAP explainer initialized successfully.")

        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            self.explainer = None

    def explain_prediction(self, input_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Explain a single prediction using SHAP values.

        Args:
            input_data: Input data as dictionary or DataFrame

        Returns:
            Dictionary with prediction and SHAP explanation
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        # Import predictor to use its preprocessing
        from src.predict import AccidentPredictor
        predictor = AccidentPredictor(self.model_path, self.encoder_path)
        predictor.model = self.model
        predictor.encoders = self.encoders
        predictor.feature_names = self.feature_names
        predictor.label_encoders = self.label_encoders
        predictor.scaler = self.scaler
        predictor.target_encoder = self.target_encoder

        # Get basic prediction
        prediction_result = predictor.predict(input_data)

        # Preprocess input for SHAP
        processed_data = predictor.preprocess_input(input_data)

        explanation = {
            'prediction': prediction_result['prediction'],
            'prediction_code': prediction_result['prediction_code'],
            'confidence': prediction_result['confidence'],
            'class_probabilities': prediction_result['class_probabilities'],
            'explanation_method': 'none'
        }

        # Generate SHAP explanation if available
        if self.explainer is not None and SHAP_AVAILABLE:
            try:
                # Calculate SHAP values
                if hasattr(self.explainer, 'shap_values'):
                    # TreeExplainer
                    shap_vals = self.explainer.shap_values(processed_data)
                else:
                    # Generic explainer
                    shap_vals = self.explainer(processed_data)

                # Handle multiclass case
                if isinstance(shap_vals, list):
                    # List of arrays, one per class
                    pred_class = prediction_result['prediction_code']
                    if isinstance(pred_class, (np.integer, int)) and pred_class < len(shap_vals):
                        shap_values = shap_vals[pred_class]
                    else:
                        # Use first class or average
                        shap_values = np.mean(np.abs(shap_vals), axis=0)
                else:
                    # Single array
                    shap_values = shap_vals

                # For single prediction, take first row
                if len(shap_values.shape) > 1:
                    shap_values_single = shap_vals[0]
                else:
                    shap_values_single = shap_vals

                # Create feature importance dictionary
                if len(self.feature_names) == len(shap_values_single):
                    feature_importance = dict(zip(self.feature_names, shap_values_single))
                    # Sort by absolute importance
                    sorted_importance = sorted(feature_importance.items(),
                                             key=lambda x: abs(x[1]), reverse=True)

                    # Get top positive and negative contributors
                    top_positive = [(k, v) for k, v in sorted_importance if v > 0][:5]
                    top_negative = [(k, v) for k, v in sorted_importance if v < 0][:5]

                    explanation.update({
                        'explanation_method': 'shap',
                        'shap_values': dict(zip(self.feature_names, shap_values_single.tolist())),
                        'top_positive_features': [{'feature': k, 'shap_value': v} for k, v in top_positive],
                        'top_negative_features': [{'feature': k, 'shap_value': v} for k, v in top_negative],
                        'expected_value': getattr(self.explainer, 'expected_value', None)
                    })
                else:
                    explanation['explanation_method'] = 'shap_error'
                    explanation['explanation_message'] = 'Feature names and SHAP values length mismatch'

            except Exception as e:
                explanation['explanation_method'] = 'shap_error'
                explanation['explanation_message'] = f'Error computing SHAP values: {str(e)}'

        else:
            # Fallback to feature importance if SHAP not available
            if self.model is not None and hasattr(self.model, 'feature_importances_'):
                try:
                    if len(self.feature_names) == len(self.model.feature_importances_):
                        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                        sorted_importance = sorted(feature_importance.items(),
                                                 key=lambda x: abs(x[1]), reverse=True)

                        explanation.update({
                            'explanation_method': 'feature_importance',
                            'top_features': [{'feature': k, 'importance': v} for k, v in sorted_importance[:10]]
                        })
                    else:
                        explanation['explanation_method'] = 'feature_importance_error'
                        explanation['explanation_message'] = 'Feature names and importances length mismatch'
                except Exception as e:
                    explanation['explanation_method'] = 'feature_importance_error'
                    explanation['explanation_message'] = f'Error computing feature importance: {str(e)}'
            else:
                explanation['explanation_method'] = 'none'
                explanation['explanation_message'] = 'Model does not support explanations'

        return explanation

    def explain_batch(self, input_data: pd.DataFrame,
                     sample_size: int = 100) -> Dict:
        """
        Explain predictions for a batch of input data.

        Args:
            input_data: Input DataFrame with multiple samples
            sample_size: Number of samples to explain (for computational efficiency)

        Returns:
            Dictionary with batch explanation
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        # Sample data if too large
        if len(input_data) > sample_size:
            # Stratified sampling if we have target, otherwise random
            explained_data = input_data.sample(n=sample_size, random_state=42)
        else:
            explained_data = input_data

        # Preprocess data
        from src.predict import AccidentPredictor
        predictor = AccidentPredictor(self.model_path, self.encoder_path)
        predictor.model = self.model
        predictor.encoders = self.encoders
        predictor.feature_names = self.feature_names
        predictor.label_encoders = self.label_encoders
        predictor.scaler = self.scaler
        predictor.target_encoder = self.target_encoder

        processed_data = predictor.preprocess_input(explained_data)

        explanation = {
            'batch_size': len(processed_data),
            'explanation_method': 'none'
        }

        # Generate SHAP explanation if available
        if self.explainer is not None and SHAP_AVAILABLE:
            try:
                # Calculate SHAP values
                if hasattr(self.explainer, 'shap_values'):
                    shap_vals = self.explainer.shap_values(processed_data)
                else:
                    shap_vals = self.explainer(processed_data)

                # Handle multiclass case
                if isinstance(shap_vals, list):
                    # Average absolute SHAP values across classes
                    mean_abs_shap = np.mean([np.abs(vals) for vals in shap_vals], axis=0)
                else:
                    mean_abs_shap = np.abs(shap_vals)

                # Create feature importance dictionary
                if len(self.feature_names) == len(mean_ashap):
                    feature_importance = dict(zip(self.feature_names, mean_abs_shap))
                    sorted_importance = sorted(feature_importance.items(),
                                             key=lambda x: abs(x[1]), reverse=True)

                    explanation.update({
                        'explanation_method': 'shap',
                        'global_feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
                        'top_features': [{'feature': k, 'importance': v} for k, v in sorted_importance[:15]]
                    })
                else:
                    explanation['explanation_method'] = 'shap_error'
                    explanation['explanation_message'] = 'Feature names and SHAP values length mismatch'

            except Exception as e:
                explanation['explanation_method'] = 'shap_error'
                explanation['explanation_message'] = f'Error computing SHAP values: {str(e)}'

        else:
            # Fallback to feature importance
            if self.model is not None and hasattr(self.model, 'feature_importances_'):
                try:
                    if len(self.feature_names) == len(self.model.feature_importances_):
                        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                        sorted_importance = sorted(feature_importance.items(),
                                                 key=lambda x: abs(x[1]), reverse=True)

                        explanation.update({
                            'explanation_method': 'feature_importance',
                            'global_feature_importance': dict(zip(self.feature_names, self.model.feature_importances_.tolist())),
                            'top_features': [{'feature': k, 'importance': v} for k, v in sorted_importance[:15]]
                        })
                    else:
                        explanation['explanation_method'] = 'feature_importance_error'
                        explanation['explanation_message'] = 'Feature names and importances length mismatch'
                except Exception as e:
                    explanation['explanation_method'] = 'feature_importance_error'
                    explanation['explanation_message'] = f'Error computing feature importance: {str(e)}'
            else:
                explanation['explanation_method'] = 'none'
                explanation['explanation_message'] = 'Model does not support explanations'

        return explanation

    def plot_shap_summary(self, input_data: pd.DataFrame,
                         max_display: int = 20,
                         save_path: str = None) -> plt.Figure:
        """
        Create SHAP summary plot.

        Args:
            input_data: Input DataFrame for explanation
            max_display: Maximum number of features to display
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")

        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Check if model is tree-based.")

        # Preprocess data
        from src.predict import AccidentPredictor
        predictor = AccidentPredictor(self.model_path, self.encoder_path)
        predictor.model = self.model
        predictor.encoders = self.encoders
        predictor.feature_names = self.feature_names
        predictor.label_encoders = self.label_encoders
        predictor.scaler = self.scaler
        predictor.target_encoder = self.target_encoder

        processed_data = predictor.preprocess_input(input_data)

        # Calculate SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_vals = self.explainer.shap_values(processed_data)
        else:
            shap_vals = self.explainer(processed_data)

        # Handle multiclass case for summary plot
        if isinstance(shap_vals, list):
            # For multiclass, we'll show the first class or combine
            # Using absolute mean SHAP values for simplicity
            mean_abs_shap = np.mean([np.abs(vals) for vals in shap_vals], axis=0)
            shap_vals_for_plot = mean_abs_shap
        else:
            shap_vals_for_plot = shap_vals

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals_for_plot, processed_data,
                         feature_names=self.feature_names,
                         max_display=max_display, show=False)
        fig = plt.gcf()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")

        return fig

    def plot_shap_dependence(self, input_data: pd.DataFrame,
                           feature_name: str,
                           interaction_index: str = None,
                           save_path: str = None) -> plt.Figure:
        """
        Create SHAP dependence plot for a specific feature.

        Args:
            input_data: Input DataFrame
            feature_name: Name of the feature to plot
            interaction_index: Name of feature for interaction (optional)
            save_path: Path to save the figure (optional)

        Returns:
            Matplotlib figure object
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")

        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized.")

        # Preprocess data
        from src.predict import AccidentPredictor
        predictor = AccidentPredictor(self.model_path, self.encoder_path)
        predictor.model = self.model
        predictor.encoders = self.encoders
        predictor.feature_names = self.feature_names
        predictor.label_encoders = self.label_encoders
        predictor.scaler = self.scaler
        predictor.target_encoder = self.target_encoder

        processed_data = predictor.preprocess_input(input_data)

        # Calculate SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_vals = self.explainer.shap_values(processed_data)
        else:
            shap_vals = self.explainer(processed_data)

        # Handle multiclass case - use first class for simplicity
        if isinstance(shap_vals, list):
            shap_vals_to_use = shap_vals[0]  # First class
        else:
            shap_vals_to_use = shap_vals

        # Create dependence plot
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(feature_name, shap_vals_to_use, processed_data,
                           interaction_index=interaction_index, show=False)
        fig = plt.gcf()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP dependence plot saved to {save_path}")

        return fig

    def save_explanations(self, explanations: Dict,
                         filename: str = "explanations.joblib"):
        """
        Save explanations to file.

        Args:
            explanations: Dictionary of explanations
            filename: Output filename
        """
        os.makedirs("models/", exist_ok=True)
        filepath = os.path.join("models/", filename)
        joblib.dump(explanations, filepath)
        print(f"Explanations saved to {filepath}")


def main():
    """Main function for testing the explainer."""
    print("Accident Explainer Module - Testing")

    # Initialize explainer
    explainer = AccidentExplainer()

    # Check if model and SHAP are available
    if explainer.model is None:
        print("No model found. Please train a model first using train.py")
        print("SHAP availability:", SHAP_AVAILABLE)
        return

    if not SHAP_AVAILABLE:
        print("SHAP not installed. Install with: pip install shap")
        return

    # Create sample input for testing
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
        # Explain prediction
        explanation = explainer.explain_prediction(sample_input)
        print("\nExplanation Result:")
        print(f"  Prediction: {explanation['prediction']}")
        print(f"  Confidence: {explanation['confidence']:.3f}")
        print(f"  Explanation method: {explanation.get('explanation_method', 'none')}")

        if explanation.get('explanation_method') == 'shap':
            print(f"  Top positive features: {explanation.get('top_positive_features', [])[:3]}")
            print(f"  Top negative features: {explanation.get('top_negative_features', [])[:3]}")
        elif explanation.get('explanation_method') == 'feature_importance':
            print(f"  Top features: {explanation.get('top_features', [])[:3]}")

    except Exception as e:
        print(f"Error generating explanation: {e}")


if __name__ == "__main__":
    main()