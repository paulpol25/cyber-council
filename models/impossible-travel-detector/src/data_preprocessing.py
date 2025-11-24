"""
Data Preprocessing Module for Impossible Travel Detection

This module handles loading, cleaning, and preprocessing of the impossible travel dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from datetime import datetime


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self, config):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, file_path):
        """
        Load the dataset from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pandas DataFrame
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        print("Handling missing values...")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            print("No missing values found")
            
        return df
    
    def parse_datetime(self, df):
        """
        Parse datetime columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with parsed datetime columns
        """
        print("Parsing datetime columns...")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        if 'prev_timestamp' in df.columns:
            df['prev_timestamp'] = pd.to_datetime(df['prev_timestamp'])
            
        return df
    
    def encode_categorical_features(self, df, categorical_features, fit=True):
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df: Input DataFrame
            categorical_features: List of categorical feature names
            fit: Whether to fit the encoder (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        print("Encoding categorical features...")
        df = df.copy()
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    # Handle unseen categories
                    df[feature] = df[feature].astype(str)
                    seen_labels = set(self.label_encoders[feature].classes_)
                    df[feature] = df[feature].apply(
                        lambda x: x if x in seen_labels else 'unknown'
                    )
                    
                    # Add unknown to encoder if needed
                    if 'unknown' not in self.label_encoders[feature].classes_:
                        self.label_encoders[feature].classes_ = np.append(
                            self.label_encoders[feature].classes_, 'unknown'
                        )
                    
                    df[feature] = self.label_encoders[feature].transform(df[feature])
                    
        return df
    
    def normalize_features(self, df, numerical_features, fit=True):
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            numerical_features: List of numerical feature names
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with normalized numerical features
        """
        print("Normalizing numerical features...")
        df = df.copy()
        
        # Filter features that exist in the dataframe
        available_features = [f for f in numerical_features if f in df.columns]
        
        if fit:
            df[available_features] = self.scaler.fit_transform(df[available_features])
        else:
            df[available_features] = self.scaler.transform(df[available_features])
            
        return df
    
    def split_data(self, df, target_column='is_impossible_travel'):
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Splitting data...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Get split ratios
        train_ratio = self.config['split']['train_ratio']
        val_ratio = self.config['split']['val_ratio']
        test_ratio = self.config['split']['test_ratio']
        random_seed = self.config['split']['random_seed']
        
        # First split: separate out test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Print class distribution
        print(f"\nClass distribution in training set:")
        print(y_train.value_counts(normalize=True))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess(self, file_path, save_splits=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to the raw data CSV
            save_splits: Whether to save train/val/test splits
            
        Returns:
            Dictionary containing preprocessed data splits
        """
        # Load data
        df = self.load_data(file_path)
        
        # Parse datetime
        df = self.parse_datetime(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Drop columns that won't be used as features
        columns_to_drop = ['timestamp', 'prev_timestamp', 'username', 'ip_address', 
                          'device_id', 'session_id', 'city', 'prev_city']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Encode categorical features
        categorical_features = self.config['features']['categorical_features']
        df = self.encode_categorical_features(df, categorical_features, fit=True)
        
        # Split data before normalization to prevent data leakage
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Normalize numerical features (fit on training data only)
        numerical_features = self.config['features']['numerical_features']
        X_train = self.normalize_features(X_train, numerical_features, fit=True)
        X_val = self.normalize_features(X_val, numerical_features, fit=False)
        X_test = self.normalize_features(X_test, numerical_features, fit=False)
        
        # Save splits if requested
        if save_splits:
            self.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Save preprocessor state
        self.save_preprocessor()
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def save_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save data splits to CSV files."""
        print("Saving data splits...")
        
        # Create directory if it doesn't exist
        splits_dir = os.path.dirname(self.config['data']['train_split_path'])
        os.makedirs(splits_dir, exist_ok=True)
        
        # Combine features and target
        train_df = X_train.copy()
        train_df['is_impossible_travel'] = y_train.values
        
        val_df = X_val.copy()
        val_df['is_impossible_travel'] = y_val.values
        
        test_df = X_test.copy()
        test_df['is_impossible_travel'] = y_test.values
        
        # Save to CSV
        train_df.to_csv(self.config['data']['train_split_path'], index=False)
        val_df.to_csv(self.config['data']['val_split_path'], index=False)
        test_df.to_csv(self.config['data']['test_split_path'], index=False)
        
        print(f"Saved splits to {splits_dir}")
    
    def save_preprocessor(self):
        """Save the preprocessor state (scalers, encoders, feature columns)."""
        print("Saving preprocessor state...")
        
        # Create directory if it doesn't exist
        processed_dir = os.path.dirname(self.config['data']['processed_data_path'])
        os.makedirs(processed_dir, exist_ok=True)
        
        preprocessor_state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        with open(self.config['data']['processed_data_path'], 'wb') as f:
            pickle.dump(preprocessor_state, f)
            
        print(f"Saved preprocessor to {self.config['data']['processed_data_path']}")
    
    def load_preprocessor(self):
        """Load the preprocessor state."""
        print("Loading preprocessor state...")
        
        with open(self.config['data']['processed_data_path'], 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.scaler = preprocessor_state['scaler']
        self.label_encoders = preprocessor_state['label_encoders']
        self.feature_columns = preprocessor_state['feature_columns']
        
        print("Preprocessor loaded successfully")


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Preprocess data
    data = preprocessor.preprocess(config['data']['raw_data_path'])
    
    print("\nPreprocessing complete!")
    print(f"Training set shape: {data['X_train'].shape}")
    print(f"Validation set shape: {data['X_val'].shape}")
    print(f"Test set shape: {data['X_test'].shape}")
