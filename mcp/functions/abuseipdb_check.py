"""
AbuseIPDB Integration for MCP Server

Checks IP addresses against AbuseIPDB's threat intelligence database for abuse reports.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
import requests


class AbuseIPDBChecker:
    """Check IP addresses for abuse reports using AbuseIPDB API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AbuseIPDB checker.
        
        Args:
            api_key: AbuseIPDB API key (if not provided, reads from env)
        """
        self.api_key = api_key or os.getenv('ABUSEIPDB_API_KEY')
        self.base_url = "https://api.abuseipdb.com/api/v2"
        
        if not self.api_key:
            print("Warning: AbuseIPDB API key not found. Set ABUSEIPDB_API_KEY environment variable.")
        
        self.headers = {
            "Key": self.api_key,
            "Accept": "application/json"
        } if self.api_key else {}
    
    def check(
        self,
        ip_address: str,
        max_age_days: int = 90,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Check an IP address against AbuseIPDB.
        
        Args:
            ip_address: The IP address to check
            max_age_days: Maximum age of reports to consider (default: 90 days)
            verbose: Whether to include detailed report info
            
        Returns:
            Dictionary with abuse report results
        """
        if not self.api_key:
            return {
                "error": "AbuseIPDB API key not configured",
                "ip_address": ip_address,
                "status": "error"
            }
        
        try:
            url = f"{self.base_url}/check"
            params = {
                "ipAddress": ip_address,
                "maxAgeInDays": max_age_days,
                "verbose": verbose
            }
            
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                return {
                    "error": "Unexpected API response format",
                    "ip_address": ip_address,
                    "status": "error"
                }
            
            ip_data = data['data']
            
            # Parse the response
            abuse_confidence_score = ip_data.get('abuseConfidenceScore', 0)
            total_reports = ip_data.get('totalReports', 0)
            num_distinct_users = ip_data.get('numDistinctUsers', 0)
            is_whitelisted = ip_data.get('isWhitelisted', False)
            country_code = ip_data.get('countryCode', 'Unknown')
            usage_type = ip_data.get('usageType', 'Unknown')
            isp = ip_data.get('isp', 'Unknown')
            domain = ip_data.get('domain', 'Unknown')
            
            # Determine threat level based on abuse confidence score
            if is_whitelisted:
                threat_level = "WHITELISTED"
                severity = "info"
            elif abuse_confidence_score >= 75:
                threat_level = "HIGH_RISK"
                severity = "critical"
            elif abuse_confidence_score >= 50:
                threat_level = "MODERATE_RISK"
                severity = "high"
            elif abuse_confidence_score >= 25:
                threat_level = "LOW_RISK"
                severity = "medium"
            elif total_reports > 0:
                threat_level = "REPORTED"
                severity = "low"
            else:
                threat_level = "CLEAN"
                severity = "info"
            
            # Get report categories
            reports = ip_data.get('reports', [])
            categories = []
            if reports:
                category_counts = {}
                for report in reports[:10]:  # Last 10 reports
                    for cat in report.get('categories', []):
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                
                # Map category IDs to names
                category_names = self._get_category_names(list(category_counts.keys()))
                categories = [
                    {"id": cat_id, "name": category_names.get(cat_id, f"Category {cat_id}"), "count": count}
                    for cat_id, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                ]
            
            result = {
                "ip_address": ip_address,
                "status": "success",
                "is_whitelisted": is_whitelisted,
                "threat_level": threat_level,
                "severity": severity,
                "abuse_confidence_score": abuse_confidence_score,
                "total_reports": total_reports,
                "num_distinct_reporters": num_distinct_users,
                "last_reported": ip_data.get('lastReportedAt', 'Never'),
                "context": {
                    "country": country_code,
                    "usage_type": usage_type,
                    "isp": isp,
                    "domain": domain
                },
                "report_categories": categories[:5],  # Top 5 categories
                "recommendation": self._get_recommendation(
                    threat_level,
                    abuse_confidence_score,
                    total_reports
                ),
                "abuseipdb_url": f"https://www.abuseipdb.com/check/{ip_address}"
            }
            
            # Add recent reports if verbose
            if verbose and reports:
                result["recent_reports"] = [
                    {
                        "reported_at": report.get('reportedAt'),
                        "comment": report.get('comment', 'No comment')[:200],  # Truncate
                        "categories": self._get_category_names(report.get('categories', []))
                    }
                    for report in reports[:3]  # Last 3 reports
                ]
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "ip_address": ip_address,
                "status": "error"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "ip_address": ip_address,
                "status": "error"
            }
    
    def _get_category_names(self, category_ids: list) -> dict:
        """Map category IDs to human-readable names."""
        # AbuseIPDB category mappings
        categories = {
            1: "DNS Compromise",
            2: "DNS Poisoning",
            3: "Fraud Orders",
            4: "DDoS Attack",
            5: "FTP Brute-Force",
            6: "Ping of Death",
            7: "Phishing",
            8: "Fraud VoIP",
            9: "Open Proxy",
            10: "Web Spam",
            11: "Email Spam",
            12: "Blog Spam",
            13: "VPN IP",
            14: "Port Scan",
            15: "Hacking",
            16: "SQL Injection",
            17: "Spoofing",
            18: "Brute-Force",
            19: "Bad Web Bot",
            20: "Exploited Host",
            21: "Web App Attack",
            22: "SSH",
            23: "IoT Targeted"
        }
        
        return {cat_id: categories.get(cat_id, f"Category {cat_id}") 
                for cat_id in category_ids}
    
    def _get_recommendation(
        self,
        threat_level: str,
        abuse_score: int,
        total_reports: int
    ) -> str:
        """Generate recommendation based on threat assessment."""
        if threat_level == "WHITELISTED":
            return (
                "✓ WHITELISTED: This IP is on AbuseIPDB's whitelist. "
                "Likely a legitimate service or CDN."
            )
        elif threat_level == "HIGH_RISK":
            return (
                f"🚨 HIGH RISK: Abuse confidence score {abuse_score}% with {total_reports} reports. "
                "BLOCK IMMEDIATELY. This IP has a strong history of malicious activity. "
                "Add to firewall blocklist and review any connections from this IP."
            )
        elif threat_level == "MODERATE_RISK":
            return (
                f"⚠️ MODERATE RISK: Abuse confidence score {abuse_score}% with {total_reports} reports. "
                "Block or heavily restrict. Monitor any existing connections. "
                "Consider rate limiting if blocking is not feasible."
            )
        elif threat_level == "LOW_RISK":
            return (
                f"⚠️ LOW RISK: Abuse confidence score {abuse_score}% with {total_reports} reports. "
                "Monitor closely. Some abuse reports exist but confidence is low. "
                "Investigate context before blocking."
            )
        elif threat_level == "REPORTED":
            return (
                f"ℹ️ REPORTED: {total_reports} abuse report(s) exist but confidence score is low. "
                "Review reports to understand context. May be false positives."
            )
        else:
            return (
                "✓ CLEAN: No abuse reports found in AbuseIPDB. "
                "However, absence of reports doesn't guarantee safety. "
                "Continue monitoring."
            )
    
    def report_ip(
        self,
        ip_address: str,
        categories: list,
        comment: str
    ) -> Dict[str, Any]:
        """
        Report an IP address for abuse (requires API key with reporting permissions).
        
        Args:
            ip_address: The IP to report
            categories: List of category IDs (see _get_category_names)
            comment: Description of the abuse
            
        Returns:
            Result of the report submission
        """
        if not self.api_key:
            return {
                "error": "AbuseIPDB API key not configured",
                "status": "error"
            }
        
        try:
            url = f"{self.base_url}/report"
            data = {
                "ip": ip_address,
                "categories": ",".join(map(str, categories)),
                "comment": comment
            }
            
            response = requests.post(
                url,
                headers=self.headers,
                data=data,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "status": "success",
                "ip_address": ip_address,
                "message": "IP reported successfully",
                "abuse_confidence_score": result.get('data', {}).get('abuseConfidenceScore', 0)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Failed to report IP: {str(e)}",
                "ip_address": ip_address,
                "status": "error"
            }
