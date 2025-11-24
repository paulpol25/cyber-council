"""
Inference Script for Impossible Travel Detector

This script provides a simple interface to check new login events for impossible travel patterns
using a trained model. It handles all the necessary preprocessing and feature engineering.
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import FeatureEngineer
from model import ImpossibleTravelDetector


# Country capital coordinates (for quick lookups without API calls)
COUNTRY_COORDS = {
    'USA': (38.9072, -77.0369),  # Washington DC
    'UK': (51.5074, -0.1278),  # London
    'France': (48.8566, 2.3522),  # Paris
    'Germany': (52.5200, 13.4050),  # Berlin
    'China': (39.9042, 116.4074),  # Beijing
    'Japan': (35.6762, 139.6503),  # Tokyo
    'Australia': (-33.8688, 151.2093),  # Sydney
    'Brazil': (-15.8267, -47.9218),  # Brasília
    'India': (28.6139, 77.2090),  # New Delhi
    'Russia': (55.7558, 37.6173),  # Moscow
    'Canada': (45.4215, -75.6972),  # Ottawa
    'Mexico': (19.4326, -99.1332),  # Mexico City
    'Spain': (40.4168, -3.7038),  # Madrid
    'Italy': (41.9028, 12.4964),  # Rome
    'Netherlands': (52.3676, 4.9041),  # Amsterdam
    'Sweden': (59.3293, 18.0686),  # Stockholm
    'Norway': (59.9139, 10.7522),  # Oslo
    'Poland': (52.2297, 21.0122),  # Warsaw
    'Romania': (44.4268, 26.1025),  # Bucharest
    'Turkey': (39.9334, 32.8597),  # Ankara
    'South Korea': (37.5665, 126.9780),  # Seoul
    'Singapore': (1.3521, 103.8198),  # Singapore
    'UAE': (24.4539, 54.3773),  # Abu Dhabi
    'Saudi Arabia': (24.7136, 46.6753),  # Riyadh
    'South Africa': (-25.7479, 28.2293),  # Pretoria
    'Egypt': (30.0444, 31.2357),  # Cairo
    'Argentina': (-34.6037, -58.3816),  # Buenos Aires
    'Chile': (-33.4489, -70.6693),  # Santiago
    'Colombia': (4.7110, -74.0721),  # Bogotá
    'Thailand': (13.7563, 100.5018),  # Bangkok
    'Vietnam': (21.0285, 105.8542),  # Hanoi
    'Indonesia': (-6.2088, 106.8456),  # Jakarta
    'Philippines': (14.5995, 120.9842),  # Manila
    'Malaysia': (3.1390, 101.6869),  # Kuala Lumpur
    'New Zealand': (-41.2865, 174.7762),  # Wellington
    'Greece': (37.9838, 23.7275),  # Athens
    'Portugal': (38.7223, -9.1393),  # Lisbon
    'Austria': (48.2082, 16.3738),  # Vienna
    'Switzerland': (46.9480, 7.4474),  # Bern
    'Belgium': (50.8503, 4.3517),  # Brussels
    'Denmark': (55.6761, 12.5683),  # Copenhagen
    'Finland': (60.1699, 24.9384),  # Helsinki
    'Ireland': (53.3498, -6.2603),  # Dublin
    'Israel': (31.7683, 35.2137),  # Jerusalem
    'Pakistan': (33.6844, 73.0479),  # Islamabad
    'Bangladesh': (23.8103, 90.4125),  # Dhaka
    'Nigeria': (9.0765, 7.3986),  # Abuja
    'Kenya': (-1.2921, 36.8219),  # Nairobi
    'Morocco': (34.0209, -6.8416),  # Rabat
}


class GeoCalculator:
    """Calculate geographic distances and travel speeds."""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="impossible_travel_detector")
    
    def get_country_coordinates(self, country):
        """
        Get coordinates for a country.
        
        Args:
            country: Country name
            
        Returns:
            Tuple of (latitude, longitude)
        """
        # Check hardcoded list first
        if country in COUNTRY_COORDS:
            return COUNTRY_COORDS[country]
        
        # Try to geocode
        try:
            location = self.geolocator.geocode(country, timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"Warning: Could not geocode {country}: {e}")
        
        # Default to 0, 0 if not found
        return (0.0, 0.0)
    
    def calculate_distance(self, coord1, coord2):
        """
        Calculate distance between two coordinates in kilometers.
        
        Args:
            coord1: Tuple of (lat, lon)
            coord2: Tuple of (lat, lon)
            
        Returns:
            Distance in kilometers
        """
        return geodesic(coord1, coord2).kilometers
    
    def calculate_speed(self, distance_km, time_diff_hours):
        """
        Calculate travel speed.
        
        Args:
            distance_km: Distance in kilometers
            time_diff_hours: Time difference in hours
            
        Returns:
            Speed in km/h
        """
        if time_diff_hours == 0:
            return 0.0
        return distance_km / time_diff_hours


class ImpossibleTravelChecker:
    """Simple interface for checking impossible travel in new login events."""
    
    def __init__(self, config_path, model_path=None):
        """
        Initialize the checker with a trained model.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained model (optional, uses latest if not provided)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load preprocessor state (scalers, encoders, feature columns)
        preprocessor_path = self.config['data']['processed_data_path']
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                "Please run training first to generate the preprocessor."
            )
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.scaler = preprocessor_state['scaler']
        self.label_encoders = preprocessor_state['label_encoders']
        self.feature_columns = preprocessor_state['feature_columns']
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Initialize geo calculator
        self.geo_calculator = GeoCalculator()
        
        # Load model
        if model_path is None:
            # Find latest model
            save_dir = self.config['model_save']['save_dir']
            models = [f for f in os.listdir(save_dir) if f.endswith('.keras')]
            if not models:
                raise ValueError(f"No trained models found in {save_dir}")
            model_path = os.path.join(save_dir, sorted(models)[-1])
        
        input_dim = len(self.feature_columns)
        self.detector = ImpossibleTravelDetector(self.config, input_dim)
        self.detector.load_model(model_path)
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Ready to check for impossible travel")
    
    def enrich_login_data(self, login_data):
        """
        Enrich login data with calculated geographic values.
        
        Args:
            login_data: Dictionary or DataFrame with login data
            
        Returns:
            Enriched login data with coordinates, distance, and speed
        """
        if isinstance(login_data, dict):
            df = pd.DataFrame([login_data])
            was_dict = True
        else:
            df = login_data.copy()
            was_dict = False
        
        # Get coordinates if not provided
        for idx, row in df.iterrows():
            # Current location
            if pd.isna(row.get('latitude', None)) or row.get('latitude') == 0.0:
                country = row.get('country', 'USA')
                lat, lon = self.geo_calculator.get_country_coordinates(country)
                df.at[idx, 'latitude'] = lat
                df.at[idx, 'longitude'] = lon
            
            # Previous location
            if pd.isna(row.get('prev_latitude', None)) or row.get('prev_latitude') == 0.0:
                prev_country = row.get('prev_country', 'USA')
                if prev_country != 'FIRST_LOGIN':
                    prev_lat, prev_lon = self.geo_calculator.get_country_coordinates(prev_country)
                    df.at[idx, 'prev_latitude'] = prev_lat
                    df.at[idx, 'prev_longitude'] = prev_lon
                else:
                    df.at[idx, 'prev_latitude'] = df.at[idx, 'latitude']
                    df.at[idx, 'prev_longitude'] = df.at[idx, 'longitude']
            
            # Calculate distance if not provided
            if pd.isna(row.get('distance_km', None)) or row.get('distance_km') == 0.0:
                coord1 = (df.at[idx, 'latitude'], df.at[idx, 'longitude'])
                coord2 = (df.at[idx, 'prev_latitude'], df.at[idx, 'prev_longitude'])
                distance = self.geo_calculator.calculate_distance(coord1, coord2)
                df.at[idx, 'distance_km'] = distance
            
            # Calculate time difference if timestamps provided
            if 'timestamp' in df.columns and 'prev_timestamp' in df.columns:
                if pd.notna(row.get('timestamp')) and pd.notna(row.get('prev_timestamp')):
                    ts = pd.to_datetime(row['timestamp'])
                    prev_ts = pd.to_datetime(row['prev_timestamp'])
                    time_diff = (ts - prev_ts).total_seconds() / 3600  # hours
                    df.at[idx, 'time_diff_hours'] = time_diff
            
            # If time_diff_hours not set, use default
            if pd.isna(row.get('time_diff_hours', None)) or row.get('time_diff_hours') == 0.0:
                df.at[idx, 'time_diff_hours'] = 1.0  # Default to 1 hour
            
            # Calculate speed
            distance = df.at[idx, 'distance_km']
            time_diff = df.at[idx, 'time_diff_hours']
            speed = self.geo_calculator.calculate_speed(distance, time_diff)
            df.at[idx, 'travel_speed_kmh'] = speed
        
        return df.to_dict('records')[0] if was_dict else df
    
    def preprocess_input(self, login_data):
        """
        Preprocess a login event for prediction.
        
        Args:
            login_data: Dictionary or DataFrame with login event data
            
        Returns:
            Preprocessed features ready for model input
        """
        # Convert to DataFrame if dictionary
        if isinstance(login_data, dict):
            df = pd.DataFrame([login_data])
        else:
            df = login_data.copy()
        
        # Parse datetime columns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'prev_timestamp' in df.columns:
            df['prev_timestamp'] = pd.to_datetime(df['prev_timestamp'])
        
        # Apply feature engineering
        df = self.feature_engineer.engineer_features(df)
        
        # Drop columns not used as features
        columns_to_drop = ['timestamp', 'prev_timestamp', 'username', 'ip_address', 
                          'device_id', 'session_id', 'city', 'prev_city', 'is_impossible_travel']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Encode categorical features
        for feature in self.config['features']['categorical_features']:
            if feature in df.columns:
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
        
        # Normalize numerical features
        numerical_features = self.config['features']['numerical_features']
        available_features = [f for f in numerical_features if f in df.columns]
        if available_features:
            df[available_features] = self.scaler.transform(df[available_features])
        
        # Ensure all expected features are present and in correct order
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing features with default value
        
        # Select and reorder features to match training
        df = df[self.feature_columns]
        
        return df
    
    def check_login(self, login_data, threshold=None, enrich=True):
        """
        Check a single login event for impossible travel.
        
        Args:
            login_data: Dictionary with login event data
            threshold: Classification threshold (optional, uses config default)
            enrich: Whether to enrich data with calculated values (default: True)
            
        Returns:
            Dictionary with prediction results
        """
        if threshold is None:
            threshold = self.config['evaluation']['threshold']
        
        # Enrich login data with calculated geographic values
        if enrich:
            login_data = self.enrich_login_data(login_data)
        
        # Preprocess the input
        X = self.preprocess_input(login_data)
        
        # Make prediction
        probability = self.detector.predict_proba(X.values)[0][0]
        is_impossible = probability > threshold
        
        # Create result
        result = {
            'is_impossible_travel': bool(is_impossible),
            'probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 2),  # 0-1 scale
            'risk_level': self._get_risk_level(probability),
            'threshold': threshold
        }
        
        return result
    
    def check_multiple_logins(self, login_data_list, threshold=None):
        """
        Check multiple login events for impossible travel.
        
        Args:
            login_data_list: List of dictionaries or DataFrame with login events
            threshold: Classification threshold (optional)
            
        Returns:
            List of prediction results
        """
        if isinstance(login_data_list, pd.DataFrame):
            df = login_data_list
        else:
            df = pd.DataFrame(login_data_list)
        
        # Preprocess all inputs
        X = self.preprocess_input(df)
        
        # Make predictions
        probabilities = self.detector.predict_proba(X.values).flatten()
        
        if threshold is None:
            threshold = self.config['evaluation']['threshold']
        
        # Create results
        results = []
        for i, prob in enumerate(probabilities):
            is_impossible = prob > threshold
            result = {
                'index': i,
                'is_impossible_travel': bool(is_impossible),
                'probability': float(prob),
                'confidence': float(abs(prob - 0.5) * 2),
                'risk_level': self._get_risk_level(prob),
                'threshold': threshold
            }
            results.append(result)
        
        return results
    
    def _get_risk_level(self, probability):
        """
        Determine risk level based on probability.
        
        Args:
            probability: Prediction probability
            
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return "LOW"
        elif probability < 0.5:
            return "MODERATE"
        elif probability < 0.7:
            return "HIGH"
        elif probability < 0.9:
            return "VERY HIGH"
        else:
            return "CRITICAL"
    
    def format_result(self, result, login_data=None):
        """
        Format prediction result for display.
        
        Args:
            result: Prediction result dictionary
            login_data: Optional original login data for context
            
        Returns:
            Formatted string
        """
        output = []
        output.append("=" * 60)
        output.append("IMPOSSIBLE TRAVEL CHECK RESULT")
        output.append("=" * 60)
        
        if login_data:
            output.append("\nLogin Event:")
            output.append(f"  User: {login_data.get('username', 'N/A')}")
            output.append(f"  Location: {login_data.get('city', 'N/A')}, {login_data.get('country', 'N/A')}")
            output.append(f"  Timestamp: {login_data.get('timestamp', 'N/A')}")
            output.append(f"  Previous: {login_data.get('prev_city', 'N/A')}, {login_data.get('prev_country', 'N/A')}")
            output.append(f"  Distance: {login_data.get('distance_km', 'N/A')} km")
            output.append(f"  Time Diff: {login_data.get('time_diff_hours', 'N/A')} hours")
            output.append(f"  Speed: {login_data.get('travel_speed_kmh', 'N/A')} km/h")
        
        output.append("\nPrediction:")
        output.append(f"  Result: {'🚨 IMPOSSIBLE TRAVEL DETECTED' if result['is_impossible_travel'] else '✓ Normal Travel'}")
        output.append(f"  Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
        output.append(f"  Confidence: {result['confidence']:.4f}")
        output.append(f"  Risk Level: {result['risk_level']}")
        output.append(f"  Threshold: {result['threshold']}")
        
        output.append("=" * 60)
        
        return "\n".join(output)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Check login events for impossible travel patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple check - just provide username and countries (distance/speed auto-calculated)
  python predict.py --username user001 --country Australia --prev_country USA
  
  # With time difference specified
  python predict.py --username user001 --country France --prev_country China --time_diff_hours 2
  
  # With exact timestamps
  python predict.py --username user001 --country Japan --prev_country USA \\
                    --timestamp "2024-11-23T14:00:00" --prev_timestamp "2024-11-23T08:00:00"

  # Check login events from a CSV file
  python predict.py --input new_logins.csv --output predictions.csv
        """
    )
    
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (optional, uses latest)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Classification threshold (optional, uses config default)')
    
    # Input methods
    parser.add_argument('--input', type=str, default=None,
                       help='Path to CSV file with login events to check')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save prediction results (CSV)')
    
    # Single event parameters (simplified - most can be auto-calculated)
    parser.add_argument('--username', type=str, help='Username (required)')
    parser.add_argument('--country', type=str, help='Current country (required)')
    parser.add_argument('--prev_country', type=str, help='Previous country (required)')
    parser.add_argument('--event_type', type=str, default='LOGIN_SUCCESS',
                       help='Event type (LOGIN_SUCCESS or LOGIN_FAILED)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Login timestamp (ISO format, optional)')
    parser.add_argument('--prev_timestamp', type=str, default=None,
                       help='Previous login timestamp (ISO format, optional)')
    
    # Optional parameters (will be auto-calculated if not provided)
    parser.add_argument('--time_diff_hours', type=float, default=None,
                       help='Time difference in hours (optional, will be calculated)')
    parser.add_argument('--distance_km', type=float, default=None,
                       help='Distance in km (optional, will be calculated)')
    parser.add_argument('--travel_speed_kmh', type=float, default=None,
                       help='Travel speed in km/h (optional, will be calculated)')
    parser.add_argument('--latitude', type=float, default=None,
                       help='Current latitude (optional, will be geocoded)')
    parser.add_argument('--longitude', type=float, default=None,
                       help='Current longitude (optional, will be geocoded)')
    parser.add_argument('--prev_latitude', type=float, default=None,
                       help='Previous latitude (optional, will be geocoded)')
    parser.add_argument('--prev_longitude', type=float, default=None,
                       help='Previous longitude (optional, will be geocoded)')
    
    args = parser.parse_args()
    
    # Initialize checker
    print("\nInitializing Impossible Travel Checker...")
    checker = ImpossibleTravelChecker(args.config, args.model)
    
    # Process input
    if args.input:
        # Load from CSV file
        print(f"\nLoading login events from: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} login events")
        
        # Check all events
        print("\nChecking for impossible travel patterns...")
        results = checker.check_multiple_logins(df, args.threshold)
        
        # Add results to dataframe
        df['is_impossible_travel'] = [r['is_impossible_travel'] for r in results]
        df['probability'] = [r['probability'] for r in results]
        df['risk_level'] = [r['risk_level'] for r in results]
        
        # Print summary
        impossible_count = sum(r['is_impossible_travel'] for r in results)
        print(f"\nResults:")
        print(f"  Total events: {len(results)}")
        print(f"  Impossible travel detected: {impossible_count}")
        print(f"  Normal travel: {len(results) - impossible_count}")
        
        # Save results
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")
        else:
            print("\nDetailed results:")
            for i, result in enumerate(results):
                if result['is_impossible_travel']:
                    print(f"  Event {i}: 🚨 IMPOSSIBLE TRAVEL (prob={result['probability']:.4f})")
    
    elif args.username and args.country:
        # Single event from command line
        print("\nPreparing login event...")
        
        # Set timestamps
        current_time = datetime.now()
        timestamp = args.timestamp if args.timestamp else current_time.isoformat()
        
        # Calculate previous timestamp if time_diff_hours is provided
        if args.prev_timestamp:
            prev_timestamp = args.prev_timestamp
        elif args.time_diff_hours:
            prev_timestamp = (current_time - timedelta(hours=args.time_diff_hours)).isoformat()
        else:
            prev_timestamp = (current_time - timedelta(hours=1)).isoformat()  # Default 1 hour ago
        
        # Build login data with minimal required fields
        login_data = {
            'username': args.username,
            'event_type': args.event_type,
            'country': args.country,
            'prev_country': args.prev_country if args.prev_country else 'FIRST_LOGIN',
            'timestamp': timestamp,
            'prev_timestamp': prev_timestamp,
            # Optional fields - will be calculated if not provided
            'distance_km': args.distance_km if args.distance_km is not None else 0.0,
            'time_diff_hours': args.time_diff_hours if args.time_diff_hours is not None else 0.0,
            'travel_speed_kmh': args.travel_speed_kmh if args.travel_speed_kmh is not None else 0.0,
            'latitude': args.latitude if args.latitude is not None else 0.0,
            'longitude': args.longitude if args.longitude is not None else 0.0,
            'prev_latitude': args.prev_latitude if args.prev_latitude is not None else 0.0,
            'prev_longitude': args.prev_longitude if args.prev_longitude is not None else 0.0,
        }
        
        print(f"User: {login_data['username']}")
        print(f"Travel: {login_data['prev_country']} → {login_data['country']}")
        print("Calculating distance and speed...")
        
        # Check the login (with enrichment enabled by default)
        result = checker.check_login(login_data, args.threshold, enrich=True)
        
        # Get enriched data for display
        enriched_data = checker.enrich_login_data(login_data)
        
        # Print formatted result
        print("\n" + checker.format_result(result, enriched_data))
    
    else:
        print("\nError: Please provide either:")
        print("  1. --input <csv_file> for batch processing")
        print("  2. Individual event parameters (--username, --country, etc.)")
        print("\nUse --help for more information")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
