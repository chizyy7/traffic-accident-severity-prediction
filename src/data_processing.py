"""
Data processing module for US Accidents dataset.
Handles loading, cleaning, and preprocessing of traffic accident data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import joblib
import os
from datetime import datetime


class AccidentDataProcessor:
    """Handles loading and preprocessing of US Accidents dataset."""

    def __init__(self, data_path: str = "data/raw/"):
        """
        Initialize the data processor.

        Args:
            data_path: Path to the raw data directory
        """
        self.data_path = data_path
        self.processed_path = "data/processed/"
        self.feature_config = None

        # Expected columns based on dataset research
        self.expected_columns = [
            'ID', 'Start_Time', 'End_Time', 'Severity', 'Distance(mi)',
            'Street', 'City', 'County', 'State', 'Zipcode', 'Country',
            'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Temperature(F)',
            'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
            'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)',
            'Weather_Condition', 'Amenity', 'Bump', 'Crossing',
            'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout',
            'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
            'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight',
            'Nautical_Twilight', 'Astronomical_Twilight'
        ]

        # Target variable
        self.target_column = 'Severity'

        # Severity mapping (1: least severe, 4: most severe)
        self.severity_mapping = {
            1: 'Minor',
            2: 'Moderate',
            3: 'Serious',
            4: 'Severe'
        }

    def load_data(self, filename: str = "US_Accidents.csv") -> pd.DataFrame:
        """
        Load the US Accidents dataset.

        Args:
            filename: Name of the CSV file to load

        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        print(f"Loading data from {filepath}...")
        # For large datasets, we might want to load in chunks or sample initially
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with shape: {df.shape}")

        return df

    def inspect_data(self, df: pd.DataFrame) -> dict:
        """
        Perform initial data inspection.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with inspection results
        """
        inspection_results = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'target_distribution': df[self.target_column].value_counts().sort_index().to_dict() if self.target_column in df.columns else None,
            'date_range': None
        }

        # Check temporal columns if they exist
        time_cols = ['Start_Time', 'End_Time']
        for col in time_cols:
            if col in df.columns:
                try:
                    df[f'{col}_parsed'] = pd.to_datetime(df[col])
                    inspection_results['date_range'] = {
                        'start': df[f'{col}_parsed'].min(),
                        'end': df[f'{col}_parsed'].max()
                    }
                    # Clean up temporary column
                    df = df.drop(columns=[f'{col}_parsed'])
                except Exception as e:
                    print(f"Could not parse {col} as datetime: {e}")

        return inspection_results

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        df_processed = df.copy()

        # For numerical columns, fill with median (robust to outliers)
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)

        # For categorical columns, fill with mode or 'Unknown'
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()
                if not mode_val.empty:
                    df_processed[col].fillna(mode_val.iloc[0], inplace=True)
                else:
                    df_processed[col].fillna('Unknown', inplace=True)

        return df_processed

    def parse_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse datetime columns and extract temporal features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional temporal features
        """
        df_processed = df.copy()

        # Parse Start_Time and End_Time
        for col in ['Start_Time', 'End_Time']:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')

        # Extract temporal features from Start_Time (accident start time)
        if 'Start_Time' in df_processed.columns:
            df_processed['accident_year'] = df_processed['Start_Time'].dt.year
            df_processed['accident_month'] = df_processed['Start_Time'].dt.month
            df_processed['accident_day'] = df_processed['Start_Time'].dt.day
            df_processed['accident_day_of_week'] = df_processed['Start_Time'].dt.dayofweek  # Monday=0
            df_processed['accident_hour'] = df_processed['Start_Time'].dt.hour

            # Derived features
            df_processed['is_weekend'] = df_processed['accident_day_of_week'].isin([5, 6]).astype(int)  # Sat, Sun
            df_processed['is_rush_hour'] = ((df_processed['accident_hour'].between(7, 9)) |
                                          (df_processed['accident_hour'].between(16, 18))).astype(int)
            df_processed['is_night'] = ((df_processed['accident_hour'] >= 20) |
                                      (df_processed['accident_hour'] <= 5)).astype(int)  # 8PM-5AM

            # Season
            df_processed['season'] = df_processed['accident_month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

        # Calculate accident duration if both timestamps exist
        if 'Start_Time' in df_processed.columns and 'End_Time' in df_processed.columns:
            valid_mask = df_processed['Start_Time'].notna() & df_processed['End_Time'].notna()
            df_processed.loc[valid_mask, 'accident_duration_minutes'] = (
                (df_processed.loc[valid_mask, 'End_Time'] -
                 df_processed.loc[valid_mask, 'Start_Time']).dt.total_seconds() / 60
            )
            # Cap unreasonable durations (e.g., > 24 hours)
            df_processed.loc[df_processed['accident_duration_minutes'] > 1440, 'accident_duration_minutes'] = np.nan

        return df_processed

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_accidents.csv"):
        """
        Save processed data to CSV.

        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        os.makedirs(self.processed_path, exist_ok=True)
        filepath = os.path.join(self.processed_path, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")

    def save_feature_config(self, config: dict, filename: str = "feature_config.joblib"):
        """
        Save feature configuration for later use.

        Args:
            config: Configuration dictionary
            filename: Output filename
        """
        os.makedirs("models/", exist_ok=True)
        filepath = os.path.join("models/", filename)
        joblib.dump(config, filepath)
        print(f"Feature config saved to {filepath}")

    def load_feature_config(self, filename: str = "feature_config.joblib") -> dict:
        """
        Load feature configuration.

        Args:
            filename: Configuration filename

        Returns:
            Loaded configuration dictionary
        """
        filepath = os.path.join("models/", filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            raise FileNotFoundError(f"Feature config not found: {filepath}")


def main():
    """Main function for testing the data processor."""
    processor = AccidentDataProcessor()

    try:
        # Try to load data (will fail if file doesn't exist, which is expected initially)
        df = processor.load_data()

        # Inspect data
        inspection_results = processor.inspect_data(df)
        print("Data inspection completed.")
        print(f"Dataset shape: {inspection_results['shape']}")

        # Process data
        df_cleaned = processor.handle_missing_values(df)
        df_processed = processor.parse_datetime_features(df_cleaned)

        # Save processed data
        processor.save_processed_data(df_processed)

        # Save feature config
        feature_config = {
            'target_column': processor.target_column,
            'severity_mapping': processor.severity_mapping,
            'temporal_features': ['accident_year', 'accident_month', 'accident_day',
                                'accident_day_of_week', 'accident_hour', 'is_weekend',
                                'is_rush_hour', 'is_night', 'season']
        }
        processor.save_feature_config(feature_config)

    except FileNotFoundError:
        print("Data file not found. Please download the US Accidents dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/")
        print("Place the CSV file in the data/raw/ directory.")

        # Create placeholder files for testing structure
        os.makedirs("data/raw/", exist_ok=True)
        os.makedirs("data/processed/", exist_ok=True)
        os.makedirs("models/", exist_ok=True)
        print("Directory structure created.")


if __name__ == "__main__":
    main()