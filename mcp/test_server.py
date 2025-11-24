"""
Test script to verify MCP server functionality
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.functions.impossible_travel import ImpossibleTravelChecker
from mcp.functions.alert_analyzer import AlertAnalyzer


def test_impossible_travel():
    """Test impossible travel detection."""
    print("\n" + "=" * 60)
    print("TEST 1: Impossible Travel Detection")
    print("=" * 60)
    
    try:
        checker = ImpossibleTravelChecker()
        
        # Test case: Tokyo to New York in 2 hours (should be impossible)
        result = checker.check(
            username="test_user",
            current_country="United States",
            previous_country="Japan",
            time_diff_hours=2,
            current_city="New York",
            previous_city="Tokyo"
        )
        
        print(f"Classification: {result['classification']}")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Distance: {result['distance_km']} km")
        print(f"Required Speed: {result['required_speed_kmh']} km/h")
        print(f"Reasoning: {result['reasoning']}")
        
        if result['classification'] in ['IMPOSSIBLE_TRAVEL', 'SUSPICIOUS']:
            print("✅ PASS - Correctly identified as suspicious/impossible")
            return True
        else:
            print("❌ FAIL - Should have been flagged as suspicious")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alert_analyzer():
    """Test alert analysis."""
    print("\n" + "=" * 60)
    print("TEST 2: Alert Analysis")
    print("=" * 60)
    
    try:
        analyzer = AlertAnalyzer()
        
        # Test case: Malware detection
        result = analyzer.analyze(
            alert_type="malware_detection",
            description="Ransomware detected on workstation",
            source_ip="192.168.1.100",
            username="test_user"
        )
        
        print(f"Threat Level: {result['threat_level']}")
        print(f"Threat Score: {result['threat_score']}/100")
        print(f"Confidence: {result['confidence']}")
        print(f"Recommended Actions: {len(result['recommended_actions'])} actions")
        
        if result['threat_level'] in ['CRITICAL', 'HIGH']:
            print("✅ PASS - Correctly identified as high/critical threat")
            return True
        else:
            print("❌ FAIL - Malware should be high threat")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threat_assessment():
    """Test threat level assessment."""
    print("\n" + "=" * 60)
    print("TEST 3: Threat Level Assessment")
    print("=" * 60)
    
    try:
        analyzer = AlertAnalyzer()
        
        # Test case: Multiple critical indicators
        result = analyzer.assess_threat(
            indicators=[
                {"type": "ip", "value": "192.0.2.1", "severity": "critical"},
                {"type": "domain", "value": "evil.com", "severity": "critical"},
                {"type": "file_hash", "value": "abc123", "severity": "high"}
            ],
            context="Multiple IOCs from same attack campaign"
        )
        
        print(f"Threat Level: {result['threat_level']}")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Response Level: {result['response_level']}")
        print(f"Total Indicators: {result['indicator_count']}")
        
        if result['threat_level'] in ['CRITICAL', 'HIGH']:
            print("✅ PASS - Correctly assessed as high threat")
            return True
        else:
            print("❌ FAIL - Multiple critical indicators should be high threat")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MCP SERVER FUNCTIONALITY TESTS")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Impossible Travel Detection", test_impossible_travel()))
    results.append(("Alert Analysis", test_alert_analyzer()))
    results.append(("Threat Assessment", test_threat_assessment()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! The MCP server is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
