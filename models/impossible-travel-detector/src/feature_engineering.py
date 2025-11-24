"""
Feature Engineering Module for Impossible Travel Detection

This module handles creation of derived features to improve model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class FeatureEngineer:
    """Creates derived features for impossible travel detection."""
    
    def __init__(self, config):
        """
        Initialize the feature engineer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def create_log_features(self, df):
        """
        Create logarithmic transformations of skewed features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with log features
        """
        df = df.copy()
        
        # Add small constant to avoid log(0)
        epsilon = 1e-6
        
        if 'travel_speed_kmh' in df.columns:
            # Ensure numeric type and handle missing values
            df['travel_speed_kmh'] = pd.to_numeric(df['travel_speed_kmh'], errors='coerce').fillna(0)
            df['speed_log'] = np.log1p(df['travel_speed_kmh'] + epsilon)
            
        if 'distance_km' in df.columns:
            df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce').fillna(0)
            df['distance_log'] = np.log1p(df['distance_km'] + epsilon)
            
        if 'time_diff_hours' in df.columns:
            df['time_diff_hours'] = pd.to_numeric(df['time_diff_hours'], errors='coerce').fillna(0)
            df['time_diff_log'] = np.log1p(df['time_diff_hours'] + epsilon)
            
        return df
    
    def create_boolean_features(self, df):
        """
        Create boolean indicator features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with boolean features
        """
        df = df.copy()
        
        # Check if same country
        if 'country' in df.columns and 'prev_country' in df.columns:
            df['is_same_country'] = (df['country'] == df['prev_country']).astype(int)
        
        # Check if first login
        if 'prev_country' in df.columns:
            df['is_first_login'] = (df['prev_country'] == 'FIRST_LOGIN').astype(int)
            
        return df
    
    def create_temporal_features(self, df):
        """
        Create time-based features from timestamp.
        
        Args:
            df: Input DataFrame with timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        if 'timestamp' in df.columns:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract hour and day of week
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Cyclical encoding for hour (0-23)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            
            # Cyclical encoding for day of week (0-6)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
        return df
    
    def create_geographical_features(self, df):
        """
        Create geographical features from coordinates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with geographical features
        """
        df = df.copy()
        
        # Calculate latitude and longitude differences
        if all(col in df.columns for col in ['latitude', 'longitude', 'prev_latitude', 'prev_longitude']):
            df['lat_diff'] = abs(df['latitude'] - df['prev_latitude'])
            df['lon_diff'] = abs(df['longitude'] - df['prev_longitude'])
            
            # Calculate manhattan distance (in degrees)
            df['manhattan_distance_deg'] = df['lat_diff'] + df['lon_diff']
            
        return df
    
    def create_velocity_features(self, df):
        """
        Create velocity-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with velocity features
        """
        df = df.copy()
        
        if all(col in df.columns for col in ['distance_km', 'time_diff_hours']):
            # Ensure numeric types
            df['travel_speed_kmh'] = pd.to_numeric(df['travel_speed_kmh'], errors='coerce').fillna(0)
            df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce').fillna(0)
            df['time_diff_hours'] = pd.to_numeric(df['time_diff_hours'], errors='coerce').fillna(0)
            
            # Avoid division by zero
            epsilon = 1e-6
            
            # Speed categories
            df['speed_category'] = pd.cut(
                df['travel_speed_kmh'],
                bins=[-np.inf, 100, 500, 900, np.inf],
                labels=[0, 1, 2, 3]  # 0=normal, 1=fast, 2=very fast, 3=impossible
            ).astype(int)
            
            # Distance categories
            df['distance_category'] = pd.cut(
                df['distance_km'],
                bins=[-np.inf, 1000, 5000, 10000, np.inf],
                labels=[0, 1, 2, 3]  # 0=local, 1=regional, 2=continental, 3=intercontinental
            ).astype(int)
            
            # Speed anomaly score (how many standard deviations from mean)
            if df['travel_speed_kmh'].std() > 0:
                df['speed_zscore'] = (
                    df['travel_speed_kmh'] - df['travel_speed_kmh'].mean()
                ) / df['travel_speed_kmh'].std()
            else:
                df['speed_zscore'] = 0
                
        return df
    
    def create_risk_features(self, df):
        """
        Create risk-related features for impossible travel detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with risk features
        """
        df = df.copy()
        
        # Speed threshold indicators (common thresholds for impossible travel)
        if 'travel_speed_kmh' in df.columns:
            df['travel_speed_kmh'] = pd.to_numeric(df['travel_speed_kmh'], errors='coerce').fillna(0)
            df['exceeds_900kmh'] = (df['travel_speed_kmh'] > 900).astype(int)  # Faster than commercial flight
            df['exceeds_500kmh'] = (df['travel_speed_kmh'] > 500).astype(int)  # Faster than typical driving
            df['exceeds_300kmh'] = (df['travel_speed_kmh'] > 300).astype(int)  # Faster than high-speed train
            
        # Short time interval indicators
        if 'time_diff_hours' in df.columns:
            df['time_diff_hours'] = pd.to_numeric(df['time_diff_hours'], errors='coerce').fillna(0)
            df['very_short_interval'] = (df['time_diff_hours'] < 1).astype(int)
            df['short_interval'] = (df['time_diff_hours'] < 3).astype(int)
            
        # Long distance indicators
        if 'distance_km' in df.columns:
            df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce').fillna(0)
            df['very_long_distance'] = (df['distance_km'] > 10000).astype(int)
            df['long_distance'] = (df['distance_km'] > 5000).astype(int)
            
        # Combined risk: long distance in short time
        if 'distance_km' in df.columns and 'time_diff_hours' in df.columns:
            df['high_risk_combo'] = (
                (df['distance_km'] > 5000) & (df['time_diff_hours'] < 6)
            ).astype(int)
            
        return df
    
    def engineer_features(self, df):
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        
        # Create temporal features first (before dropping timestamp)
        df = self.create_temporal_features(df)
        
        # Create log features
        df = self.create_log_features(df)
        
        # Create boolean features
        df = self.create_boolean_features(df)
        
        # Create geographical features
        df = self.create_geographical_features(df)
        
        # Create velocity features
        df = self.create_velocity_features(df)
        
        # Create risk features
        df = self.create_risk_features(df)
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_importance_names(self, df):
        """
        Get list of all feature names after engineering.
        
        Args:
            df: DataFrame after feature engineering
            
        Returns:
            List of feature names
        """
        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'prev_timestamp', 'username', 'ip_address',
                       'device_id', 'session_id', 'city', 'prev_city', 
                       'is_impossible_travel']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols


if __name__ == "__main__":
    # Example usage
    import yaml
    from data_preprocessing import DataPreprocessor
    
    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load raw data
    df = pd.read_csv(config['data']['raw_data_path'])
    
    # Parse datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['prev_timestamp'] = pd.to_datetime(df['prev_timestamp'])
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Engineer features
    df_engineered = feature_engineer.engineer_features(df)
    
    print("\nEngineered features:")
    print(df_engineered.columns.tolist())
    print(f"\nDataFrame shape: {df_engineered.shape}")
