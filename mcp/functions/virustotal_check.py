"""
VirusTotal Integration for MCP Server

Checks IPs, domains, file hashes, and URLs against VirusTotal's threat intelligence database.
"""

import os
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import requests


class VirusTotalChecker:
    """Check threat intelligence using VirusTotal API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize VirusTotal checker.
        
        Args:
            api_key: VirusTotal API key (if not provided, reads from env)
        """
        self.api_key = api_key or os.getenv('VIRUSTOTAL_API_KEY')
        self.base_url = "https://www.virustotal.com/api/v3"
        
        if not self.api_key:
            print("Warning: VirusTotal API key not found. Set VIRUSTOTAL_API_KEY environment variable.")
        
        self.headers = {
            "x-apikey": self.api_key
        } if self.api_key else {}
    
    def check(
        self,
        indicator: str,
        indicator_type: Literal["ip", "domain", "hash", "url"]
    ) -> Dict[str, Any]:
        """
        Check an indicator against VirusTotal.
        
        Args:
            indicator: The indicator to check (IP, domain, hash, or URL)
            indicator_type: Type of indicator
            
        Returns:
            Dictionary with threat intelligence results
        """
        if not self.api_key:
            return {
                "error": "VirusTotal API key not configured",
                "indicator": indicator,
                "indicator_type": indicator_type,
                "status": "error"
            }
        
        try:
            # Route to appropriate check method
            if indicator_type == "ip":
                return self._check_ip(indicator)
            elif indicator_type == "domain":
                return self._check_domain(indicator)
            elif indicator_type == "hash":
                return self._check_hash(indicator)
            elif indicator_type == "url":
                return self._check_url(indicator)
            else:
                return {
                    "error": f"Unknown indicator type: {indicator_type}",
                    "indicator": indicator,
                    "status": "error"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "indicator": indicator,
                "indicator_type": indicator_type,
                "status": "error"
            }
    
    def _check_ip(self, ip_address: str) -> Dict[str, Any]:
        """Check IP address against VirusTotal."""
        try:
            url = f"{self.base_url}/ip_addresses/{ip_address}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 404:
                return {
                    "indicator": ip_address,
                    "indicator_type": "ip",
                    "status": "not_found",
                    "malicious": False,
                    "message": "IP not found in VirusTotal database"
                }
            
            response.raise_for_status()
            data = response.json()
            
            # Parse the response
            attributes = data.get('data', {}).get('attributes', {})
            stats = attributes.get('last_analysis_stats', {})
            
            malicious_count = stats.get('malicious', 0)
            suspicious_count = stats.get('suspicious', 0)
            harmless_count = stats.get('harmless', 0)
            total_scans = sum(stats.values())
            
            # Determine threat level
            if malicious_count > 0:
                threat_level = "MALICIOUS"
                severity = "critical" if malicious_count > 3 else "high"
            elif suspicious_count > 0:
                threat_level = "SUSPICIOUS"
                severity = "medium"
            else:
                threat_level = "CLEAN"
                severity = "low"
            
            # Get additional context
            country = attributes.get('country', 'Unknown')
            as_owner = attributes.get('as_owner', 'Unknown')
            
            return {
                "indicator": ip_address,
                "indicator_type": "ip",
                "status": "success",
                "malicious": malicious_count > 0,
                "threat_level": threat_level,
                "severity": severity,
                "detections": {
                    "malicious": malicious_count,
                    "suspicious": suspicious_count,
                    "harmless": harmless_count,
                    "total_engines": total_scans
                },
                "detection_rate": f"{malicious_count}/{total_scans}",
                "context": {
                    "country": country,
                    "as_owner": as_owner
                },
                "recommendation": self._get_recommendation(threat_level, malicious_count),
                "virustotal_url": f"https://www.virustotal.com/gui/ip-address/{ip_address}"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "indicator": ip_address,
                "indicator_type": "ip",
                "status": "error"
            }
    
    def _check_domain(self, domain: str) -> Dict[str, Any]:
        """Check domain against VirusTotal."""
        try:
            url = f"{self.base_url}/domains/{domain}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 404:
                return {
                    "indicator": domain,
                    "indicator_type": "domain",
                    "status": "not_found",
                    "malicious": False,
                    "message": "Domain not found in VirusTotal database"
                }
            
            response.raise_for_status()
            data = response.json()
            
            attributes = data.get('data', {}).get('attributes', {})
            stats = attributes.get('last_analysis_stats', {})
            
            malicious_count = stats.get('malicious', 0)
            suspicious_count = stats.get('suspicious', 0)
            harmless_count = stats.get('harmless', 0)
            total_scans = sum(stats.values())
            
            if malicious_count > 0:
                threat_level = "MALICIOUS"
                severity = "critical" if malicious_count > 5 else "high"
            elif suspicious_count > 0:
                threat_level = "SUSPICIOUS"
                severity = "medium"
            else:
                threat_level = "CLEAN"
                severity = "low"
            
            categories = attributes.get('categories', {})
            
            return {
                "indicator": domain,
                "indicator_type": "domain",
                "status": "success",
                "malicious": malicious_count > 0,
                "threat_level": threat_level,
                "severity": severity,
                "detections": {
                    "malicious": malicious_count,
                    "suspicious": suspicious_count,
                    "harmless": harmless_count,
                    "total_engines": total_scans
                },
                "detection_rate": f"{malicious_count}/{total_scans}",
                "categories": list(categories.values())[:5] if categories else [],
                "recommendation": self._get_recommendation(threat_level, malicious_count),
                "virustotal_url": f"https://www.virustotal.com/gui/domain/{domain}"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "indicator": domain,
                "indicator_type": "domain",
                "status": "error"
            }
    
    def _check_hash(self, file_hash: str) -> Dict[str, Any]:
        """Check file hash against VirusTotal."""
        try:
            url = f"{self.base_url}/files/{file_hash}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 404:
                return {
                    "indicator": file_hash,
                    "indicator_type": "hash",
                    "status": "not_found",
                    "malicious": False,
                    "message": "File hash not found in VirusTotal database"
                }
            
            response.raise_for_status()
            data = response.json()
            
            attributes = data.get('data', {}).get('attributes', {})
            stats = attributes.get('last_analysis_stats', {})
            
            malicious_count = stats.get('malicious', 0)
            suspicious_count = stats.get('suspicious', 0)
            undetected_count = stats.get('undetected', 0)
            total_scans = sum(stats.values())
            
            if malicious_count > 0:
                threat_level = "MALICIOUS"
                severity = "critical"
            elif suspicious_count > 0:
                threat_level = "SUSPICIOUS"
                severity = "high"
            else:
                threat_level = "CLEAN"
                severity = "low"
            
            # Get file details
            file_type = attributes.get('type_description', 'Unknown')
            file_name = attributes.get('meaningful_name', 'Unknown')
            file_size = attributes.get('size', 0)
            
            return {
                "indicator": file_hash,
                "indicator_type": "hash",
                "status": "success",
                "malicious": malicious_count > 0,
                "threat_level": threat_level,
                "severity": severity,
                "detections": {
                    "malicious": malicious_count,
                    "suspicious": suspicious_count,
                    "undetected": undetected_count,
                    "total_engines": total_scans
                },
                "detection_rate": f"{malicious_count}/{total_scans}",
                "file_info": {
                    "type": file_type,
                    "name": file_name,
                    "size": file_size
                },
                "recommendation": self._get_recommendation(threat_level, malicious_count),
                "virustotal_url": f"https://www.virustotal.com/gui/file/{file_hash}"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "indicator": file_hash,
                "indicator_type": "hash",
                "status": "error"
            }
    
    def _check_url(self, url_to_check: str) -> Dict[str, Any]:
        """Check URL against VirusTotal."""
        try:
            import base64
            # VirusTotal requires URL to be base64 encoded without padding
            url_id = base64.urlsafe_b64encode(url_to_check.encode()).decode().strip("=")
            
            url = f"{self.base_url}/urls/{url_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 404:
                # URL not scanned yet, submit it
                return {
                    "indicator": url_to_check,
                    "indicator_type": "url",
                    "status": "not_found",
                    "malicious": False,
                    "message": "URL not yet scanned in VirusTotal. Submit it for scanning."
                }
            
            response.raise_for_status()
            data = response.json()
            
            attributes = data.get('data', {}).get('attributes', {})
            stats = attributes.get('last_analysis_stats', {})
            
            malicious_count = stats.get('malicious', 0)
            suspicious_count = stats.get('suspicious', 0)
            harmless_count = stats.get('harmless', 0)
            total_scans = sum(stats.values())
            
            if malicious_count > 0:
                threat_level = "MALICIOUS"
                severity = "critical"
            elif suspicious_count > 0:
                threat_level = "SUSPICIOUS"
                severity = "high"
            else:
                threat_level = "CLEAN"
                severity = "low"
            
            return {
                "indicator": url_to_check,
                "indicator_type": "url",
                "status": "success",
                "malicious": malicious_count > 0,
                "threat_level": threat_level,
                "severity": severity,
                "detections": {
                    "malicious": malicious_count,
                    "suspicious": suspicious_count,
                    "harmless": harmless_count,
                    "total_engines": total_scans
                },
                "detection_rate": f"{malicious_count}/{total_scans}",
                "recommendation": self._get_recommendation(threat_level, malicious_count),
                "virustotal_url": f"https://www.virustotal.com/gui/url/{url_id}"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "indicator": url_to_check,
                "indicator_type": "url",
                "status": "error"
            }
    
    def _get_recommendation(self, threat_level: str, detection_count: int) -> str:
        """Generate recommendation based on threat level."""
        if threat_level == "MALICIOUS":
            if detection_count > 5:
                return (
                    "🚨 CRITICAL THREAT: Block immediately. "
                    f"{detection_count} security vendors flagged this as malicious. "
                    "Isolate affected systems, review logs, and initiate incident response."
                )
            else:
                return (
                    "⚠️ HIGH THREAT: Block and investigate. "
                    f"{detection_count} vendors detected malicious activity. "
                    "Review context and consider blocking at perimeter."
                )
        elif threat_level == "SUSPICIOUS":
            return (
                "⚠️ SUSPICIOUS: Monitor and investigate. "
                "Some vendors flagged this as suspicious. "
                "Gather more context before taking action."
            )
        else:
            return (
                "✓ CLEAN: No malicious activity detected. "
                "However, absence of detection doesn't guarantee safety. "
                "Continue monitoring."
            )
