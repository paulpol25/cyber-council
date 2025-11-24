"""
Impossible Travel Detection Function

This module provides impossible travel detection capabilities for the MCP server.
Uses the trained ML model and preprocessing pipeline from predict.py.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yaml

# Add models directory to path
models_path = Path(__file__).parent.parent.parent / "models" / "impossible-travel-detector"
sys.path.insert(0, str(models_path))

from src.feature_engineering import FeatureEngineer
from src.model import ImpossibleTravelDetector

# Import for geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("Warning: geopy not available, geocoding will use fallback coordinates")


# Country capital coordinates (for quick lookups without API calls)
COUNTRY_COORDS = {
    'USA': (38.9072, -77.0369), 'United States': (38.9072, -77.0369),
    'UK': (51.5074, -0.1278), 'United Kingdom': (51.5074, -0.1278),
    'France': (48.8566, 2.3522),
    'Germany': (52.5200, 13.4050),
    'China': (39.9042, 116.4074),
    'Japan': (35.6762, 139.6503),
    'Australia': (-33.8688, 151.2093),
    'Brazil': (-15.8267, -47.9218),
    'India': (28.6139, 77.2090),
    'Russia': (55.7558, 37.6173),
    'Canada': (45.4215, -75.6972),
    'Mexico': (19.4326, -99.1332),
    'Spain': (40.4168, -3.7038),
    'Italy': (41.9028, 12.4964),
    'Netherlands': (52.3676, 4.9041),
    'Sweden': (59.3293, 18.0686),
    'Norway': (59.9139, 10.7522),
    'Poland': (52.2297, 21.0122),
    'Romania': (44.4268, 26.1025),
    'Turkey': (39.9334, 32.8597),
    'South Korea': (37.5665, 126.9780),
    'Singapore': (1.3521, 103.8198),
    'UAE': (24.4539, 54.3773),
    'Saudi Arabia': (24.7136, 46.6753),
    'South Africa': (-25.7479, 28.2293),
    'Egypt': (30.0444, 31.2357),
    'Argentina': (-34.6037, -58.3816),
    'Chile': (-33.4489, -70.6693),
    'Colombia': (4.7110, -74.0721),
    'Thailand': (13.7563, 100.5018),
    'Vietnam': (21.0285, 105.8542),
    'Indonesia': (-6.2088, 106.8456),
    'Philippines': (14.5995, 120.9842),
    'Malaysia': (3.1390, 101.6869),
    'New Zealand': (-41.2865, 174.7762),
    'Greece': (37.9838, 23.7275),
    'Portugal': (38.7223, -9.1393),
    'Austria': (48.2082, 16.3738),
    'Switzerland': (46.9480, 7.4474),
    'Belgium': (50.8503, 4.3517),
    'Denmark': (55.6761, 12.5683),
    'Finland': (60.1699, 24.9384),
    'Ireland': (53.3498, -6.2603),
    'Israel': (31.7683, 35.2137),
    'Pakistan': (33.6844, 73.0479),
    'Bangladesh': (23.8103, 90.4125),
    'Nigeria': (9.0765, 7.3986),
    'Kenya': (-1.2921, 36.8219),
    'Morocco': (34.0209, -6.8416),
}


class GeoCalculator:
    """Calculate geographic distances and travel speeds."""
    
    def __init__(self):
        if GEOPY_AVAILABLE:
            self.geolocator = Nominatim(user_agent="impossible_travel_detector")
        else:
            self.geolocator = None
    
    def get_coordinates(self, country, city=None):
        """
        Get coordinates for a country/city.
        
        Args:
            country: Country name
            city: City name (optional)
            
        Returns:
            Tuple of (latitude, longitude)
        """
        # Check hardcoded list first
        if country in COUNTRY_COORDS:
            return COUNTRY_COORDS[country]
        
        # Try to geocode if geopy is available
        if self.geolocator:
            try:
                location_str = f"{city}, {country}" if city else country
                location = self.geolocator.geocode(location_str, timeout=10)
                if location:
                    return (location.latitude, location.longitude)
            except Exception as e:
                print(f"Warning: Could not geocode {country}: {e}")
        
        # Default to 0, 0 if not found
        return (0.0, 0.0)
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two coordinates in kilometers.
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
            
        Returns:
            Distance in kilometers
        """
        if GEOPY_AVAILABLE:
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        else:
            # Haversine formula fallback
            from math import radians, cos, sin, asin, sqrt
            
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            km = 6371 * c
            return km


class ImpossibleTravelChecker:
    """Checks login events for impossible travel patterns using trained ML model."""
    
    def __init__(self, config_path=None, model_path=None):
        """
        Initialize the checker with a trained model.
        
        Args:
            config_path: Path to configuration file (optional)
            model_path: Path to trained model (optional, uses latest if not provided)
        """
        # Set default config path
        if config_path is None:
            config_path = models_path / "configs" / "config.yaml"
        
        # Load configuration
        if not os.path.exists(config_path):
            # Fallback to heuristic-based detection if config not found
            print(f"Warning: Config not found at {config_path}, using heuristic-based detection")
            self.use_ml = False
            self.geo_calculator = GeoCalculator()
            self.HIGH_RISK_SPEED = 800
            self.MEDIUM_RISK_SPEED = 600
            self.LOW_RISK_SPEED = 400
            return
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Try to load preprocessor and model
        try:
            preprocessor_path = self.config['data']['processed_data_path']
            
            if not os.path.exists(preprocessor_path):
                print(f"Warning: Preprocessor not found at {preprocessor_path}, using heuristic-based detection")
                self.use_ml = False
                self.geo_calculator = GeoCalculator()
                self.HIGH_RISK_SPEED = 800
                self.MEDIUM_RISK_SPEED = 600
                self.LOW_RISK_SPEED = 400
                return
            
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
                save_dir = self.config['model_save']['save_dir']
                if not os.path.exists(save_dir):
                    raise FileNotFoundError(f"Model directory not found: {save_dir}")
                
                models = [f for f in os.listdir(save_dir) if f.endswith('.keras')]
                if not models:
                    raise ValueError(f"No trained models found in {save_dir}")
                model_path = os.path.join(save_dir, sorted(models)[-1])
            
            input_dim = len(self.feature_columns)
            self.detector = ImpossibleTravelDetector(self.config, input_dim)
            self.detector.load_model(model_path)
            
            self.use_ml = True
            print(f"✓ ML model loaded from: {model_path}")
            
        except Exception as e:
            print(f"Warning: Could not load ML model ({e}), falling back to heuristic-based detection")
            self.use_ml = False
            self.geo_calculator = GeoCalculator()
            self.HIGH_RISK_SPEED = 800
            self.MEDIUM_RISK_SPEED = 600
            self.LOW_RISK_SPEED = 400
    
    def enrich_login_data(self, login_data):
        """
        Enrich login data with calculated geographic values.
        
        Args:
            login_data: Dictionary with login data
            
        Returns:
            Enriched login data with coordinates, distance, and speed
        """
        df = pd.DataFrame([login_data])
        
        # Get coordinates if not provided
        for idx, row in df.iterrows():
            # Current location
            if pd.isna(row.get('latitude', None)) or row.get('latitude') == 0.0:
                country = row.get('country', 'USA')
                coords = self.geo_calculator.get_coordinates(country, row.get('city'))
                df.at[idx, 'latitude'] = coords[0]
                df.at[idx, 'longitude'] = coords[1]
            
            # Previous location
            if pd.isna(row.get('prev_latitude', None)) or row.get('prev_latitude') == 0.0:
                prev_country = row.get('prev_country', 'USA')
                if prev_country != 'FIRST_LOGIN':
                    prev_coords = self.geo_calculator.get_coordinates(prev_country, row.get('prev_city'))
                    df.at[idx, 'prev_latitude'] = prev_coords[0]
                    df.at[idx, 'prev_longitude'] = prev_coords[1]
                else:
                    df.at[idx, 'prev_latitude'] = df.at[idx, 'latitude']
                    df.at[idx, 'prev_longitude'] = df.at[idx, 'longitude']
            
            # Calculate distance if not provided
            if pd.isna(row.get('distance_km', None)) or row.get('distance_km') == 0.0:
                coord1 = (df.at[idx, 'latitude'], df.at[idx, 'longitude'])
                coord2 = (df.at[idx, 'prev_latitude'], df.at[idx, 'prev_longitude'])
                distance = self.geo_calculator.calculate_distance(coord1[0], coord1[1], coord2[0], coord2[1])
                df.at[idx, 'distance_km'] = distance
            
            # Calculate time difference if timestamps provided
            if 'timestamp' in df.columns and 'prev_timestamp' in df.columns:
                if pd.notna(row.get('timestamp')) and pd.notna(row.get('prev_timestamp')):
                    ts = pd.to_datetime(row['timestamp'])
                    prev_ts = pd.to_datetime(row['prev_timestamp'])
                    time_diff = (ts - prev_ts).total_seconds() / 3600  # hours
                    df.at[idx, 'time_diff_hours'] = time_diff
            
            # If time_diff_hours not set, use provided value
            if pd.isna(row.get('time_diff_hours', None)) or row.get('time_diff_hours') == 0.0:
                df.at[idx, 'time_diff_hours'] = 1.0  # Default to 1 hour
            
            # Calculate speed
            distance = df.at[idx, 'distance_km']
            time_diff = df.at[idx, 'time_diff_hours']
            if time_diff > 0:
                speed = distance / time_diff
            else:
                speed = 0.0
            df.at[idx, 'travel_speed_kmh'] = speed
        
        return df.to_dict('records')[0]
    
    def preprocess_input(self, login_data):
        """
        Preprocess a login event for ML prediction.
        
        Args:
            login_data: Dictionary with login event data
            
        Returns:
            Preprocessed features ready for model input
        """
        df = pd.DataFrame([login_data])
        
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
    
    def check(
        self,
        username: str,
        current_country: str,
        previous_country: str,
        time_diff_hours: float,
        current_city: Optional[str] = None,
        previous_city: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if a login represents impossible travel.
        
        Args:
            username: User identifier
            current_country: Current login country
            previous_country: Previous login country
            time_diff_hours: Hours between logins
            current_city: Current login city (optional)
            previous_city: Previous login city (optional)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Prepare login data
            current_time = datetime.now()
            timestamp = current_time.isoformat()
            prev_timestamp = (current_time - timedelta(hours=time_diff_hours)).isoformat()
            
            login_data = {
                'username': username,
                'event_type': 'LOGIN_SUCCESS',
                'country': current_country,
                'prev_country': previous_country if previous_country else 'FIRST_LOGIN',
                'city': current_city,
                'prev_city': previous_city,
                'timestamp': timestamp,
                'prev_timestamp': prev_timestamp,
                'distance_km': 0.0,
                'time_diff_hours': time_diff_hours,
                'travel_speed_kmh': 0.0,
                'latitude': 0.0,
                'longitude': 0.0,
                'prev_latitude': 0.0,
                'prev_longitude': 0.0,
            }
            
            # Enrich with calculated values
            enriched_data = self.enrich_login_data(login_data)
            
            distance_km = enriched_data['distance_km']
            required_speed = enriched_data['travel_speed_kmh']
            
            # Use ML model if available
            if self.use_ml:
                # Preprocess for ML
                X = self.preprocess_input(enriched_data)
                
                # Make prediction
                threshold = self.config.get('evaluation', {}).get('threshold', 0.5)
                probability = self.detector.predict_proba(X.values)[0][0]
                ml_prediction = probability > threshold
                
                # Combine ML prediction with heuristic
                heuristic_classification, heuristic_confidence, reasoning = self._classify_risk_heuristic(
                    distance_km, required_speed, time_diff_hours, current_country, previous_country
                )
                
                # Use ML probability but enhance reasoning with heuristics
                confidence = float(abs(probability - 0.5) * 2)
                
                if ml_prediction:
                    classification = "IMPOSSIBLE_TRAVEL" if probability > 0.7 else "SUSPICIOUS"
                else:
                    classification = "LEGITIMATE"
                
                # Enhance reasoning with ML confidence
                reasoning = f"ML Model Confidence: {probability:.2%}. {reasoning}"
                
            else:
                # Fall back to heuristic-based detection
                classification, confidence, reasoning = self._classify_risk_heuristic(
                    distance_km, required_speed, time_diff_hours, current_country, previous_country
                )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                classification,
                required_speed,
                username
            )
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(required_speed, time_diff_hours)
            
            return {
                "username": username,
                "classification": classification,
                "risk_score": risk_score,
                "confidence": confidence,
                "distance_km": round(distance_km, 2),
                "time_diff_hours": time_diff_hours,
                "required_speed_kmh": round(required_speed, 2),
                "locations": {
                    "previous": {
                        "country": previous_country,
                        "city": previous_city,
                        "coordinates": [enriched_data['prev_latitude'], enriched_data['prev_longitude']]
                    },
                    "current": {
                        "country": current_country,
                        "city": current_city,
                        "coordinates": [enriched_data['latitude'], enriched_data['longitude']]
                    }
                },
                "reasoning": reasoning,
                "recommendation": recommendation
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "classification": "ERROR",
                "risk_score": 0,
                "confidence": 0.0
            }
    
    def _classify_risk_heuristic(
        self,
        distance_km: float,
        required_speed: float,
        time_diff_hours: float,
        current_country: str,
        previous_country: str
    ) -> tuple[str, float, str]:
        """
        Classify the risk level based on travel parameters using heuristics.
        
        Returns:
            Tuple of (classification, confidence, reasoning)
        """
        # Same country logins
        if current_country == previous_country:
            if distance_km < 100:
                return (
                    "LEGITIMATE",
                    0.95,
                    "Same country with minimal distance - likely normal user movement"
                )
            elif required_speed < self.LOW_RISK_SPEED:
                return (
                    "LEGITIMATE",
                    0.85,
                    f"Same country - travel at {required_speed:.0f} km/h is feasible by car/train"
                )
        
        # Very short time difference
        if time_diff_hours < 0.5:  # Less than 30 minutes
            if distance_km > 50:
                return (
                    "SUSPICIOUS",
                    0.90,
                    f"Only {time_diff_hours:.1f} hours to travel {distance_km:.0f} km - highly unlikely"
                )
        
        # High speed required
        if required_speed > self.HIGH_RISK_SPEED:
            return (
                "IMPOSSIBLE_TRAVEL",
                0.95,
                f"Required speed of {required_speed:.0f} km/h exceeds typical airplane speeds - physically impossible"
            )
        elif required_speed > self.MEDIUM_RISK_SPEED:
            return (
                "SUSPICIOUS",
                0.85,
                f"Required speed of {required_speed:.0f} km/h is very high - would require direct flight with no delays"
            )
        elif required_speed > self.LOW_RISK_SPEED:
            return (
                "SUSPICIOUS",
                0.70,
                f"Required speed of {required_speed:.0f} km/h is high but possible with air travel"
            )
        
        # Long time between logins
        if time_diff_hours > 24:
            return (
                "LEGITIMATE",
                0.90,
                f"Sufficient time ({time_diff_hours:.1f} hours) to travel {distance_km:.0f} km - normal travel"
            )
        
        # Moderate risk
        return (
            "LEGITIMATE",
            0.75,
            f"Travel at {required_speed:.0f} km/h over {time_diff_hours:.1f} hours is feasible"
        )
    
    def _calculate_risk_score(self, required_speed: float, time_diff_hours: float) -> int:
        """
        Calculate a risk score from 0-100.
        
        Args:
            required_speed: Required speed in km/h
            time_diff_hours: Time difference in hours
            
        Returns:
            Risk score (0-100)
        """
        # Base score on speed
        if required_speed > self.HIGH_RISK_SPEED:
            base_score = 95
        elif required_speed > self.MEDIUM_RISK_SPEED:
            base_score = 75
        elif required_speed > self.LOW_RISK_SPEED:
            base_score = 55
        else:
            base_score = 20
        
        # Adjust for time factor
        if time_diff_hours < 1:
            base_score = min(100, base_score + 15)
        elif time_diff_hours < 3:
            base_score = min(100, base_score + 5)
        elif time_diff_hours > 24:
            base_score = max(0, base_score - 20)
        
        return int(base_score)
    
    def _generate_recommendation(
        self,
        classification: str,
        required_speed: float,
        username: str
    ) -> str:
        """
        Generate a recommendation based on the classification.
        
        Args:
            classification: Risk classification
            required_speed: Required speed in km/h
            username: Username
            
        Returns:
            Recommendation string
        """
        if classification == "IMPOSSIBLE_TRAVEL":
            return (
                f"🚨 IMMEDIATE ACTION REQUIRED: Account '{username}' shows impossible travel pattern. "
                "Recommend: 1) Force password reset, 2) Terminate all active sessions, "
                "3) Enable MFA if not present, 4) Contact user to verify legitimate access, "
                "5) Review account activity for past 7 days"
            )
        elif classification == "SUSPICIOUS":
            return (
                f"⚠️ INVESTIGATE: Account '{username}' shows suspicious travel pattern. "
                "Recommend: 1) Verify with user through secondary channel, "
                "2) Review recent account activity, 3) Check for other anomalies, "
                "4) Consider temporary account restrictions"
            )
        else:
            return (
                f"✓ MONITOR: Account '{username}' travel pattern appears legitimate. "
                "Continue normal monitoring."
            )
