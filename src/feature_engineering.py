"""
Feature engineering module for US Accidents dataset.
Handles encoding, scaling, and creation of meaningful features for accident severity prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FeatureEngineer:
    """Handles feature engineering for accident severity prediction."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = None
        self.column_transformer = None
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.target_encoder = LabelEncoder()

    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify categorical and numerical features.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        # Exclude target and ID columns
        exclude_cols = ['ID', 'Severity']  # Target and identifier

        # Get columns that exist in dataframe
        available_cols = [col for col in df.columns if col not in exclude_cols]

        # Identify feature types
        categorical_features = []
        numerical_features = []

        for col in available_cols:
            if df[col].dtype == 'object' or df[col].nunique() < 50:
                # Treat as categorical if object type or low cardinality
                categorical_features.append(col)
            else:
                numerical_features.append(col)

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        return categorical_features, numerical_features

    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features from raw weather data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional weather features
        """
        df_processed = df.copy()

        # Weather condition grouping if Weather_Condition exists
        if 'Weather_Condition' in df_processed.columns:
            # Group similar weather conditions
            weather_groups = {
                'Clear': ['Clear', 'Fair'],
                'Cloudy': ['Cloudy', 'Overcast'],
                'Rain': ['Rain', 'Light Rain', 'Heavy Rain', 'Drizzle'],
                'Snow': ['Snow', 'Light Snow', 'Heavy Snow', 'Sleet', 'Hail'],
                'Fog': ['Fog', 'Smoke', 'Haze'],
                'Storm': ['Thunderstorm', 'Storm', 'Lightning'],
                'Windy': ['Windy'],
                'Other': []  # Catch-all
            }

            def map_weather(condition):
                if pd.isna(condition):
                    return 'Unknown'
                condition_str = str(condition).strip()
                for group, conditions in weather_groups.items():
                    if condition in conditions or any(c in condition_str for c in conditions):
                        return group
                # Check exact match
                for group, conditions in weather_groups.items():
                    if condition_str in conditions:
                        return group
                return 'Other'

            df_processed['weather_category'] = df_processed['Weather_Condition'].apply(map_weather)

        # Visibility categories
        if 'Visibility(mi)' in df_processed.columns:
            df_processed['visibility_category'] = pd.cut(
                df_processed['Visibility(mi)'],
                bins=[-np.inf, 1, 3, 5, 10, np.inf],
                labels=['Very Poor', 'Poor', 'Moderate', 'Good', 'Excellent'],
                include_lowest=True
            )

        # Temperature categories
        if 'Temperature(F)' in df_processed.columns:
            df_processed['temp_category'] = pd.cut(
                df_processed['Temperature(F)'],
                bins=[-np.inf, 32, 50, 70, 85, np.inf],
                labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot'],
                include_lowest=True
            )

        return df_processed

    def create_road_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create road-related features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional road features
        """
        df_processed = df.copy()

        # Junction features
        junction_cols = ['Junction', 'Crossing', 'Stop', 'Traffic_Signal', 'Roundabout']
        available_junction_cols = [col for col in junction_cols if col in df_processed.columns]

        if available_junction_cols:
            # Count number of junction features present
            df_processed['junction_count'] = df_processed[available_junction_cols].sum(axis=1)
            df_processed['is_intersection'] = (df_processed['junction_count'] > 0).astype(int)

        # Road surface indicators (from binary features)
        road_features = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                        'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                        'Turning_Loop']
        available_road_features = [col for col in road_features if col in df_processed.columns]

        if available_road_features:
            df_processed['total_road_features'] = df_processed[available_road_features].sum(axis=1)

        # Distance categories
        if 'Distance(mi)' in df_processed.columns:
            df_processed['distance_category'] = pd.cut(
                df_processed['Distance(mi)'],
                bins=[-np.inf, 0.1, 0.5, 1, 2, 5, np.inf],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Extreme'],
                include_lowest=True
            )

        return df_processed

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional temporal features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional temporal features
        """
        df_processed = df.copy()

        # Already created in data_processing.py, but ensuring consistency
        if 'Start_Time' in df_processed.columns:
            df_processed['Start_Time'] = pd.to_datetime(df_processed['Start_Time'], errors='coerce')

        if 'Start_Time' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['Start_Time']):
            # Hour of day bins
            df_processed['time_of_day'] = pd.cut(
                df_processed['Start_Time'].dt.hour,
                bins=[-1, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )

            # Day type
            df_processed['day_type'] = df_processed['Start_Time'].dt.dayofweek.apply(
                lambda x: 'Weekend' if x >= 5 else 'Weekday'
            )

        return df_processed

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders (True for training, False for inference)

        Returns:
            DataFrame with encoded categorical features
        """
        df_processed = df.copy()

        # Identify categorical features if not already done
        if not self.categorical_features:
            self.categorical_features, _ = self.identify_feature_types(df_processed)

        # Encode each categorical feature
        for col in self.categorical_features:
            if col in df_processed.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        # Handle NaN values
                        col_series = df_processed[col].astype(str)
                        col_series = col_series.replace('nan', 'Unknown')
                        self.label_encoders[col].fit(col_series)
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(
                        df_processed[col].astype(str).replace('nan', 'Unknown')
                    )
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels by mapping to a default value
                        col_series = df_processed[col].astype(str)
                        col_series = col_series.replace('nan', 'Unknown')
                        # Transform known labels, map unknown to most frequent or -1
                        try:
                            encoded_vals = self.label_encoders[col].transform(col_series)
                        except ValueError:
                            # Handle unseen labels
                            known_labels = set(self.label_encoders[col].classes_)
                            col_series = col_series.apply(
                                lambda x: x if x in known_labels else 'Unknown'
                            )
                            # If 'Unknown' not in classes, add it or use first class
                            if 'Unknown' not in known_labels:
                                # Use the most frequent class as fallback
                                encoded_vals = np.full(len(col_series),
                                                     self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])[0])
                            else:
                                encoded_vals = self.label_encoders[col].transform(col_series)
                        df_processed[f'{col}_encoded'] = encoded_vals
                    else:
                        # If encoder doesn't exist, create a default encoding
                        df_processed[f'{col}_encoded'] = 0

        return df_processed

    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            DataFrame with scaled numerical features
        """
        df_processed = df.copy()

        # Identify numerical features if not already done
        if not self.numerical_features:
            _, self.numerical_features = self.identify_feature_types(df_processed)

        # Select only numerical columns that exist in dataframe
        numerical_cols = [col for col in self.numerical_features if col in df_processed.columns]

        if numerical_cols:
            if fit:
                self.scaler = StandardScaler()
                df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
            else:
                if self.scaler is not None:
                    df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])

        return df_processed

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for modeling.

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training)

        Returns:
            Tuple of (processed DataFrame, feature column names)
        """
        df_processed = df.copy()

        # Create engineered features
        df_processed = self.create_weather_features(df_processed)
        df_processed = self.create_road_features(df_processed)
        df_processed = self.create_temporal_features(df_processed)

        # Identify feature types
        self.categorical_features, self.numerical_features = self.identify_feature_types(df_processed)

        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=fit)

        # Scale numerical features
        df_processed = self.scale_numerical_features(df_processed, fit=fit)

        # Identify final feature columns (excluding target and ID)
        exclude_cols = ['ID', 'Severity']
        # Also exclude original categorical columns that we've encoded
        exclude_cols.extend(self.categorical_features)
        # Exclude timestamp columns if they exist
        time_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp']
        exclude_cols.extend([col for col in time_cols if col in df_processed.columns])

        self.feature_names = [col for col in df_processed.columns if col not in exclude_cols]

        return df_processed, self.feature_names

    def encode_target(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode target variable.

        Args:
            y: Target series
            fit: Whether to fit the encoder

        Returns:
            Encoded target array
        """
        if fit:
            return self.target_encoder.fit_transform(y)
        else:
            return self.target_encoder.transform(y)

    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Decode target variable back to original labels.

        Args:
            y_encoded: Encoded target array

        Returns:
            Decoded target array
        """
        return self.target_encoder.inverse_transform(y_encoded)

    def save_encoders(self, filepath: str = "models/feature_encoders.joblib"):
        """
        Save fitted encoders and scaler.

        Args:
            filepath: Path to save the encoders
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        encoders_dict = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_names': self.feature_names
        }
        joblib.dump(encoders_dict, filepath)
        print(f"Encoders saved to {filepath}")

    def load_encoders(self, filepath: str = "models/feature_encoders.joblib"):
        """
        Load fitted encoders and scaler.

        Args:
            filepath: Path to load the encoders from
        """
        if os.path.exists(filepath):
            encoders_dict = joblib.load(filepath)
            self.label_encoders = encoders_dict['label_encoders']
            self.scaler = encoders_dict['scaler']
            self.target_encoder = encoders_dict['target_encoder']
            self.categorical_features = encoders_dict['categorical_features']
            self.numerical_features = encoders_dict['numerical_features']
            self.feature_names = encoders_dict['feature_names']
            print(f"Encoders loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Encoders file not found: {filepath}")


def main():
    """Main function for testing the feature engineer."""
    engineer = FeatureEngineer()

    # Create sample data for testing
    sample_data = {
        'ID': [1, 2, 3],
        'Severity': [1, 2, 3],
        'Start_Time': ['2020-01-01 08:00:00', '2020-01-01 14:00:00', '2020-01-01 20:00:00'],
        'Temperature(F)': [32.0, 75.0, 60.0],
        'Visibility(mi)': [10.0, 5.0, 2.0],
        'Weather_Condition': ['Clear', 'Rain', 'Fog'],
        'City': ['New York', 'Los Angeles', 'Chicago'],
        'State': ['NY', 'CA', 'IL'],
        'Distance(mi': [0.5, 2.0, 0.1],
        'Junction': [True, False, True],
        'Crossing': [False, True, False]
    }

    df = pd.DataFrame(sample_data)
    print("Sample data:")
    print(df.head())

    # Test feature engineering
    try:
        processed_df, feature_names = engineer.prepare_features(df, fit=True)
        print("\nProcessed features:")
        print(processed_df[feature_names].head())
        print(f"\nFeature names: {feature_names}")

        # Test target encoding
        y_encoded = engineer.encode_target(df['Severity'], fit=True)
        print(f"\nEncoded target: {y_encoded}")
        print(f"Decoded target: {engineer.decode_target(y_encoded)}")

        # Save encoders
        engineer.save_encoders()

    except Exception as e:
        print(f"Error in feature engineering: {e}")


if __name__ == "__main__":
    main()