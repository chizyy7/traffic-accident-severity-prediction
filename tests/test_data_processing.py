"""
Unit tests for data processing module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_processing import AccidentDataProcessor


class TestAccidentDataProcessor(unittest.TestCase):
    """Test cases for AccidentDataProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = AccidentDataProcessor()

        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'ID': ['A001', 'A002', 'A003'],
            'Start_Time': ['2020-01-01 08:00:00', '2020-01-01 14:00:00', '2020-01-01 20:00:00'],
            'End_Time': ['2020-01-01 08:30:00', '2020-01-01 14:45:00', '2020-01-01 20:30:00'],
            'Severity': [1, 2, 3],
            'Temperature(F)': [32.0, 75.0, 60.0],
            'Visibility(mi)': [10.0, 5.0, 2.0],
            'Weather_Condition': ['Clear', 'Rain', 'Fog'],
            'Distance(mi)': [0.5, 2.0, 0.1],
            'City': ['New York', 'Los Angeles', 'Chicago'],
            'State': ['NY', 'CA', 'IL'],
            'Junction': [True, False, True]
        })

    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsInstance(self.processor, AccidentDataProcessor)
        self.assertEqual(self.processor.target_column, 'Severity')
        self.assertIn('Minor', self.processor.severity_mapping.values())

    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create data with missing values
        df_with_nan = self.sample_data.copy()
        df_with_nan.loc[0, 'Temperature(F)'] = np.nan
        df_with_nan.loc[1, 'Weather_Condition'] = np.nan

        # Process
        df_processed = self.processor.handle_missing_values(df_with_nan)

        # Check that no NaN values remain
        self.assertFalse(df_processed.isnull().any().any())

        # Check that numerical missing values were filled with median
        expected_temp_median = df_with_nan['Temperature(F)'].median()
        self.assertEqual(df_processed.loc[0, 'Temperature(F)'], expected_temp_median)

    def test_parse_datetime_features(self):
        """Test datetime feature extraction."""
        # Process the sample data
        df_processed = self.processor.parse_datetime_features(self.sample_data)

        # Check that new columns were created
        expected_columns = ['accident_year', 'accident_month', 'accident_day',
                          'accident_day_of_week', 'accident_hour', 'is_weekend',
                          'is_rush_hour', 'is_night', 'season']
        for col in expected_columns:
            self.assertIn(col, df_processed.columns)

        # Check specific values
        # First row: 2020-01-01 08:00:00 should be Wednesday (day_of_week=2), hour=8
        self.assertEqual(df_processed.loc[0, 'accident_year'], 2020)
        self.assertEqual(df_processed.loc[0, 'accident_month'], 1)
        self.assertEqual(df_processed.loc[0, 'accident_day'], 1)
        self.assertEqual(df_processed.loc[0, 'accident_day_of_week'], 2)  # Wednesday
        self.assertEqual(df_processed.loc[0, 'accident_hour'], 8)

        # Check weekend calculation (Jan 1, 2020 was Wednesday)
        self.assertEqual(df_processed.loc[0, 'is_weekend'], 0)

        # Check rush hour (8 AM is rush hour)
        self.assertEqual(df_processed.loc[0, 'is_rush_hour'], 1)

        # Check night (8 AM is not night)
        self.assertEqual(df_processed.loc[0, 'is_night'], 0)

        # Check season (January is Winter)
        self.assertEqual(df_processed.loc[0, 'season'], 'Winter')

    def test_accident_duration_calculation(self):
        """Test accident duration calculation."""
        df_processed = self.processor.parse_datetime_features(self.sample_data)

        # Check that accident_duration_minutes column exists
        self.assertIn('accident_duration_minutes', df_processed.columns)

        # Check specific durations
        # First row: 08:00 to 08:30 = 30 minutes
        self.assertAlmostEqual(df_processed.loc[0, 'accident_duration_minutes'], 30.0, places=1)

        # Second row: 14:00 to 14:45 = 45 minutes
        self.assertAlmostEqual(df_processed.loc[1, 'accident_duration_minutes'], 45.0, places=1)

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file raises appropriate error."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_data("nonexistent_file.csv")


if __name__ == '__main__':
    unittest.main()