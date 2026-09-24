"""
Unit tests for feature engineering module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from feature_engineering import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer."""

    def setUp(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()

        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'ID': ['A001', 'A002', 'A003', 'A004'],
            'Severity': [1, 2, 3, 4],
            'Start_Time': ['2020-01-01 08:00:00', '2020-01-01 14:00:00', '2020-01-01 20:00:00', '2020-01-02 02:00:00'],
            'Temperature(F)': [32.0, 75.0, 60.0, 90.0],
            'Visibility(mi)': [10.0, 5.0, 2.0, 15.0],
            'Weather_Condition': ['Clear', 'Rain', 'Fog', 'Clear'],
            'Distance(mi)': [0.5, 2.0, 0.1, 5.0],
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
            'State': ['NY', 'CA', 'IL', 'TX'],
            'Junction': [True, False, True, False],
            'Crossing': [False, True, False, True]
        })

    def test_initialization(self):
        """Test engineer initialization."""
        self.assertIsInstance(self.engineer, FeatureEngineer)
        self.assertEqual(len(self.engineer.categorical_features), 0)
        self.assertEqual(len(self.engineer.numerical_features), 0)

    def test_identify_feature_types(self):
        """Test feature type identification."""
        categorical, numerical = self.engineer.identify_feature_types(self.sample_data)

        # Check that we identified some features
        self.assertGreater(len(categorical), 0)
        self.assertGreater(len(numerical), 0)

        # Check specific classifications
        # These should be categorical (string or low cardinality)
        expected_categorical = ['Weather_Condition', 'City', 'State']
        for col in expected_categorical:
            if col in self.sample_data.columns:
                self.assertIn(col, categorical)

        # These should be numerical
        expected_numerical = ['Temperature(F)', 'Visibility(mi)', 'Distance(mi)']
        for col in expected_numerical:
            if col in self.sample_data.columns:
                self.assertIn(col, numerical)

    def test_create_weather_features(self):
        """Test weather feature creation."""
        df_processed = self.engineer.create_weather_features(self.sample_data)

        # Check that new columns were created
        self.assertIn('weather_category', df_processed.columns)
        self.assertIn('visibility_category', df_processed.columns)
        self.assertIn('temp_category', df_processed.columns)

        # Check specific mappings
        # Clear should map to Clear category
        clear_rows = df_processed[df_processed['Weather_Condition'] == 'Clear']
        self.assertTrue(all(clear_rows['weather_category'] == 'Clear'))

        # Rain should map to Rain category
        rain_rows = df_processed[df_processed['Weather_Condition'] == 'Rain']
        self.assertTrue(all(rain_rows['weather_category'] == 'Rain'))

        # Fog should map to Fog category
        fog_rows = df_processed[df_processed['Weather_Condition'] == 'Fog']
        self.assertTrue(all(fog_rows['weather_category'] == 'Fog'))

    def test_create_road_features(self):
        """Test road feature creation."""
        df_processed = self.engineer.create_road_features(self.sample_data)

        # Check that new columns were created
        self.assertIn('junction_count', df_processed.columns)
        self.assertIn('is_intersection', df_processed.columns)
        self.assertIn('total_road_features', df_processed.columns)

        # Check junction count calculation
        # Row 0: Junction=True, Crossing=False -> count=1
        self.assertEqual(df_processed.loc[0, 'junction_count'], 1)
        # Row 1: Junction=False, Crossing=True -> count=1
        self.assertEqual(df_processed.loc[1, 'junction_count'], 1)
        # Row 2: Junction=True, Crossing=False -> count=1
        self.assertEqual(df_processed.loc[2, 'junction_count'], 1)
        # Row 3: Junction=False, Crossing=True -> count=1
        self.assertEqual(df_processed.loc[3, 'junction_count'], 1)

        # Check is_intersection
        self.assertTrue(all(df_processed['is_intersection'] == 1))  # All have at least one junction feature

    def test_create_temporal_features(self):
        """Test temporal feature creation."""
        df_processed = self.engineer.create_temporal_features(self.sample_data)

        # Check that new columns were created
        self.assertIn('time_of_day', df_processed.columns)
        self.assertIn('day_type', df_processed.columns)

        # Check specific values
        # 8 AM should be Morning
        self.assertEqual(df_processed.loc[0, 'time_of_day'], 'Morning')
        # 2 PM should be Afternoon
        self.assertEqual(df_processed.loc[1, 'time_of_day'], 'Afternoon')
        # 8 PM should be Evening
        self.assertEqual(df_processed.loc[2, 'time_of_day'], 'Evening')
        # 2 AM should be Night
        self.assertEqual(df_processed.loc[3, 'time_of_day'], 'Night')

        # Check day_type
        # Jan 1, 2020 was Wednesday (weekday)
        self.assertEqual(df_processed.loc[0, 'day_type'], 'Weekday')
        # Jan 2, 2020 was Thursday (weekday)
        self.assertEqual(df_processed.loc[3, 'day_type'], 'Weekday')

    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        # First identify features
        self.engineer.identify_feature_types(self.sample_data)

        # Test fitting
        df_encoded = self.engineer.encode_categorical_features(self.sample_data, fit=True)

        # Check that encoded columns were created
        for col in self.engineer.categorical_features:
            if col in self.sample_data.columns:
                encoded_col = f'{col}_encoded'
                self.assertIn(encoded_col, df_encoded.columns)
                # Check that values are integers (encoded)
                self.assertTrue(pd.api.types.is_integer_dtype(df_encoded[encoded_col]))

        # Test transforming (should produce same results when fit=False)
        df_encoded2 = self.engineer.encode_categorical_features(self.sample_data, fit=False)
        # For columns that were fitted, results should be identical
        for col in self.engineer.categorical_features:
            if col in self.sample_data.columns:
                encoded_col = f'{col}_encoded'
                if encoded_col in df_encoded.columns and encoded_col in df_encoded2.columns:
                    pd.testing.assert_series_equal(df_encoded[encoded_col], df_encoded2[encoded_col])

    def test_scale_numerical_features(self):
        """Test numerical feature scaling."""
        # First identify features
        self.engineer.identify_feature_types(self.sample_data)

        # Test fitting
        df_scaled = self.engineer.scale_numerical_features(self.sample_data, fit=True)

        # Check that numerical columns were scaled (mean approx 0, std approx 1)
        for col in self.engineer.numerical_features:
            if col in self.sample_data.columns:
                # Check that scaling was applied
                self.assertAlmostEqual(df_scaled[col].mean(), 0.0, places=10)
                self.assertAlmostEqual(df_scaled[col].std(), 1.0, places=10)

    def test_prepare_features(self):
        """Test full feature preparation pipeline."""
        # Test with fitting
        df_processed, feature_names = self.engineer.prepare_features(self.sample_data, fit=True)

        # Check that we got feature names
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)

        # Check that processed data has the expected columns
        self.assertEqual(set(df_processed.columns), set(feature_names + ['ID', 'Severity'] +
                                                    self.engineer.categorical_features +
                                                    ['Start_Time', 'End_Time']))

        # Check that we can transform new data
        df_new = self.sample_data.copy()
        # Modify one value to test
        df_new.loc[0, 'Temperature(F)'] = 50.0

        df_processed_new, _ = self.engineer.prepare_features(df_new, fit=False)
        self.assertEqual(set(df_processed_new.columns), set(df_processed.columns))


if __name__ == '__main__':
    unittest.main()