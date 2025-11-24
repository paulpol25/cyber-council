"""
MCP Tool Adapter for AG2

Adapts MCP server tools to AG2 function calling format.
"""

import asyncio
from typing import Dict, Any, Callable, Optional
import json
import sys
import os

# Add parent directory to path to import from mcp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.functions.impossible_travel import ImpossibleTravelChecker
from mcp.functions.alert_analyzer import AlertAnalyzer
from mcp.functions.virustotal_check import VirusTotalChecker
from mcp.functions.abuseipdb_check import AbuseIPDBChecker
from mcp.functions.ioc_extractor import extract_iocs as _extract_iocs


class MCPToolAdapter:
    """Adapts MCP tools for use with AG2 agents."""
    
    def __init__(self):
        """Initialize MCP tool checkers."""
        self.travel_checker = ImpossibleTravelChecker()
        self.alert_analyzer = AlertAnalyzer()
        self.vt_checker = VirusTotalChecker()
        self.abuse_checker = AbuseIPDBChecker()
    
    def check_impossible_travel(
        self,
        username: str,
        source_country: str,
        target_country: str,
        time_diff_hours: float,
        source_city: str = None,
        target_city: str = None
    ) -> str:
        """
        Check for impossible travel patterns.
        
        Args:
            username: User account
            source_country: Country of first login
            target_country: Country of second login
            time_diff_hours: Hours between logins
            source_city: City of first login (optional)
            target_city: City of second login (optional)
            
        Returns:
            JSON string with detection results
        """
        print(f"🔍 [DEBUG] check_impossible_travel called: user={username}, {source_country}->{target_country}")
        result = self.travel_checker.check(
            username=username,
            previous_country=source_country,
            current_country=target_country,
            time_diff_hours=time_diff_hours,
            previous_city=source_city,
            current_city=target_city
        )
        print(f"✅ [DEBUG] check_impossible_travel result: {result.get('is_impossible_travel', 'unknown')}")
        return json.dumps(result, indent=2)
    
    def analyze_security_alert(
        self,
        alert_type: str = None,
        alert_data: Dict[str, Any] = None
    ) -> str:
        """
        Analyze a security alert. Can auto-detect alert type from data.
        
        Args:
            alert_type: Type of security alert (optional - will auto-detect)
            alert_data: Alert details (dict with username, description, etc.)
            
        Returns:
            JSON string with analysis results
        """
        print(f"🔍 [DEBUG] analyze_security_alert called: type={alert_type}, data_keys={list(alert_data.keys()) if alert_data else 'None'}")
        result = self.alert_analyzer.analyze(alert_type=alert_type, alert_data=alert_data)
        print(f"✅ [DEBUG] analyze_security_alert result: threat_level={result.get('threat_level')}")
        return json.dumps(result, indent=2)
    
    def assess_threat_level(
        self,
        indicators: Dict[str, Any]
    ) -> str:
        """
        Assess overall threat level from multiple indicators.
        
        Args:
            indicators: Dictionary of security indicators
            
        Returns:
            JSON string with threat assessment
        """
        print(f"🔍 [DEBUG] assess_threat_level called with indicators: {list(indicators.keys())[:5]}")
        
        # Convert the indicators dict to a list with a single item
        # The underlying assess_threat expects List[Dict]
        indicators_list = [indicators] if isinstance(indicators, dict) else indicators
        
        result = self.alert_analyzer.assess_threat(indicators_list)
        print(f"✅ [DEBUG] assess_threat_level result: threat_level={result.get('threat_level')}, risk_score={result.get('risk_score')}")
        return json.dumps(result, indent=2)
    
    def check_virustotal(
        self,
        indicator: str,
        indicator_type: str
    ) -> str:
        """
        Check indicator against VirusTotal.
        
        Args:
            indicator: IP, domain, hash, or URL to check
            indicator_type: Type (ip, domain, hash, url)
            
        Returns:
            JSON string with VirusTotal results
        """
        print(f"🔍 [DEBUG] check_virustotal called: indicator={indicator}, type={indicator_type}")
        result = self.vt_checker.check(indicator, indicator_type)
        print(f"✅ [DEBUG] check_virustotal result: {json.dumps(result, indent=2)[:200]}...")
        return json.dumps(result, indent=2)
    
    def check_abuseipdb(
        self,
        ip_address: str,
        max_age_days: int = 90,
        verbose: bool = True
    ) -> str:
        """
        Check IP reputation on AbuseIPDB.
        
        Args:
            ip_address: IP to check
            max_age_days: Look back period
            verbose: Include detailed report info
            
        Returns:
            JSON string with AbuseIPDB results
        """
        print(f"🔍 [DEBUG] check_abuseipdb called: ip={ip_address}")
        result = self.abuse_checker.check(ip_address, max_age_days, verbose)
        print(f"✅ [DEBUG] check_abuseipdb result: {json.dumps(result, indent=2)[:200]}...")
        return json.dumps(result, indent=2)
    
    def extract_iocs(self, text: str) -> str:
        """
        Extract indicators of compromise from log text or alerts.
        
        Args:
            text: Log text, alert description, or incident details to analyze
            
        Returns:
            JSON string with extracted IoCs categorized by type
        """
        print(f"🔍 [DEBUG] extract_iocs called with text length: {len(text)} chars")
        result = _extract_iocs(text)
        print(f"✅ [DEBUG] extract_iocs result: {result[:200]}...")
        return result
    
    def get_function_map(self) -> Dict[str, Callable]:
        """
        Get mapping of function names to callables for AG2.
        
        Returns:
            Dictionary of function names to methods
        """
        return {
            "extract_iocs": self.extract_iocs,
            "check_impossible_travel": self.check_impossible_travel,
            "analyze_security_alert": self.analyze_security_alert,
            "assess_threat_level": self.assess_threat_level,
            "check_virustotal": self.check_virustotal,
            "check_abuseipdb": self.check_abuseipdb,
        }
