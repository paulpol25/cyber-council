"""
Quick test for the updated impossible_travel.py with ML integration
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.functions.impossible_travel import ImpossibleTravelChecker


def test_impossible_travel_ml():
    """Test impossible travel with ML model."""
    print("=" * 60)
    print("Testing Impossible Travel with ML Model")
    print("=" * 60)
    
    try:
        # Initialize checker (will try to load ML model)
        print("\nInitializing checker...")
        checker = ImpossibleTravelChecker()
        
        if checker.use_ml:
            print("✓ Using ML model for detection")
        else:
            print("⚠️  Using heuristic-based detection (ML model not available)")
        
        # Test case: Tokyo to New York in 2 hours (impossible)
        print("\n" + "-" * 60)
        print("Test Case: Tokyo → New York in 2 hours")
        print("-" * 60)
        
        result = checker.check(
            username="test_user",
            current_country="United States",
            previous_country="Japan",
            time_diff_hours=2,
            current_city="New York",
            previous_city="Tokyo"
        )
        
        print(f"\nResults:")
        print(f"  Classification: {result['classification']}")
        print(f"  Risk Score: {result['risk_score']}/100")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Distance: {result['distance_km']} km")
        print(f"  Required Speed: {result['required_speed_kmh']} km/h")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"\n  Recommendation:")
        print(f"  {result['recommendation']}")
        
        # Test case: London to Paris in 3 hours (legitimate)
        print("\n" + "-" * 60)
        print("Test Case: London → Paris in 3 hours")
        print("-" * 60)
        
        result2 = checker.check(
            username="test_user2",
            current_country="France",
            previous_country="United Kingdom",
            time_diff_hours=3,
            current_city="Paris",
            previous_city="London"
        )
        
        print(f"\nResults:")
        print(f"  Classification: {result2['classification']}")
        print(f"  Risk Score: {result2['risk_score']}/100")
        print(f"  Confidence: {result2['confidence']:.2f}")
        print(f"  Distance: {result2['distance_km']} km")
        print(f"  Required Speed: {result2['required_speed_kmh']} km/h")
        print(f"  Reasoning: {result2['reasoning']}")
        
        # Test case: Romania to Japan in 3 hours (like the command line test)
        print("\n" + "-" * 60)
        print("Test Case: Japan → Romania in 3 hours")
        print("-" * 60)
        
        result3 = checker.check(
            username="user001",
            current_country="Romania",
            previous_country="Japan",
            time_diff_hours=3
        )
        
        print(f"\nResults:")
        print(f"  Classification: {result3['classification']}")
        print(f"  Risk Score: {result3['risk_score']}/100")
        print(f"  Confidence: {result3['confidence']:.2f}")
        print(f"  Distance: {result3['distance_km']} km")
        print(f"  Required Speed: {result3['required_speed_kmh']} km/h")
        print(f"  Reasoning: {result3['reasoning']}")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_impossible_travel_ml()
    sys.exit(0 if success else 1)
