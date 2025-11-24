"""
Alert Analyzer Function

This module provides security alert analysis capabilities for the MCP server.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re


class AlertAnalyzer:
    """Analyzes security alerts and provides threat assessments."""
    
    # Known malicious patterns and indicators
    MALICIOUS_PATTERNS = {
        "sql_injection": [
            r"(\bunion\b.*\bselect\b)",
            r"(\bor\b\s+1\s*=\s*1)",
            r"(\bdrop\b.*\btable\b)",
            r"(--|\#|\/\*.*\*\/)",
        ],
        "xss": [
            r"(<script[^>]*>.*<\/script>)",
            r"(javascript:)",
            r"(onerror\s*=)",
            r"(onload\s*=)",
        ],
        "command_injection": [
            r"(;\s*(ls|cat|wget|curl)\b)",
            r"(\|.*\b(bash|sh|cmd)\b)",
            r"(`.*`)",
        ],
    }
    
    # Known malicious IPs (example - in production, use threat intelligence feeds)
    KNOWN_MALICIOUS_IPS = {
        "192.0.2.1",  # Example malicious IP
        "198.51.100.1",  # Example malicious IP
    }
    
    # Severity weights for threat level calculation
    SEVERITY_WEIGHTS = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1
    }
    
    def __init__(self):
        """Initialize the alert analyzer."""
        self.alert_types = {
            "authentication_anomaly": self._analyze_auth_anomaly,
            "malware_detection": self._analyze_malware,
            "network_intrusion": self._analyze_network_intrusion,
            "data_exfiltration": self._analyze_data_exfiltration,
            "privilege_escalation": self._analyze_privilege_escalation,
            "suspicious_process": self._analyze_suspicious_process,
        }
    
    def analyze(
        self,
        alert_type: Optional[str] = None,
        description: Optional[str] = None,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None,
        username: Optional[str] = None,
        timestamp: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        alert_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a security alert.
        
        Args:
            alert_type: Type of security alert (optional - will auto-detect if not provided)
            description: Detailed description of the alert
            source_ip: Source IP address
            destination_ip: Destination IP address
            username: Associated username
            timestamp: When the event occurred
            additional_context: Additional metadata
            alert_data: Alternative to individual params - dict with all alert info
            
        Returns:
            Analysis results with threat assessment
        """
        # If alert_data dict is provided, extract parameters from it
        if alert_data:
            username = alert_data.get('username', username)
            source_ip = alert_data.get('source_ip', source_ip)
            destination_ip = alert_data.get('destination_ip', destination_ip)
            timestamp = alert_data.get('timestamp', timestamp)
            description = alert_data.get('description', alert_data.get('reasoning', str(alert_data)))
            additional_context = alert_data if not additional_context else {**alert_data, **additional_context}
        
        # Auto-detect alert type if not provided
        if not alert_type and additional_context:
            alert_type = self._detect_alert_type(additional_context, description or "")
        
        # Basic validation
        if not description:
            description = str(additional_context) if additional_context else "Unknown alert"
        
        # Initialize result
        result = {
            "alert_type": alert_type,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "username": username,
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "description": description,
        }
        
        # Check for known malicious IPs
        ip_threat_score = 0
        if source_ip and source_ip in self.KNOWN_MALICIOUS_IPS:
            ip_threat_score = 50
            result["ip_reputation"] = "MALICIOUS"
        elif source_ip:
            result["ip_reputation"] = "UNKNOWN"
        
        # Analyze based on alert type
        if alert_type in self.alert_types:
            analysis = self.alert_types[alert_type](
                description, source_ip, destination_ip, username, additional_context or {}
            )
        else:
            analysis = self._analyze_generic(description, additional_context or {})
        
        # Combine results
        result.update(analysis)
        
        # Add IP threat score to overall score
        if "threat_score" in result:
            result["threat_score"] = min(100, result["threat_score"] + ip_threat_score)
        
        return result
    
    def assess_threat(
        self,
        indicators: List[Dict[str, Any]],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Assess overall threat level from multiple indicators.
        
        Args:
            indicators: List of security indicators
            context: Additional context
            
        Returns:
            Comprehensive threat assessment
        """
        if not indicators:
            return {
                "threat_level": "UNKNOWN",
                "risk_score": 0,
                "message": "No indicators provided"
            }
        
        # Calculate weighted threat score
        total_weight = 0
        for indicator in indicators:
            severity = indicator.get("severity", "low")
            total_weight += self.SEVERITY_WEIGHTS.get(severity, 1)
        
        # Normalize to 0-100 scale
        max_possible = len(indicators) * 4  # All critical
        risk_score = int((total_weight / max_possible) * 100)
        
        # Determine threat level
        if risk_score >= 75:
            threat_level = "CRITICAL"
            response = "IMMEDIATE"
        elif risk_score >= 50:
            threat_level = "HIGH"
            response = "URGENT"
        elif risk_score >= 25:
            threat_level = "MEDIUM"
            response = "INVESTIGATE"
        else:
            threat_level = "LOW"
            response = "MONITOR"
        
        # Generate recommendations
        recommendations = self._generate_threat_recommendations(
            threat_level, indicators
        )
        
        return {
            "threat_level": threat_level,
            "risk_score": risk_score,
            "response_level": response,
            "indicator_count": len(indicators),
            "indicators_by_severity": self._count_by_severity(indicators),
            "recommendations": recommendations,
            "context": context
        }
    
    def _analyze_auth_anomaly(
        self, description: str, source_ip: Optional[str],
        destination_ip: Optional[str], username: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze authentication anomaly."""
        threat_score = 40  # Base score
        
        # Check for multiple failed attempts
        if any(word in description.lower() for word in ["failed", "multiple", "repeated"]):
            threat_score += 20
        
        # Check for brute force indicators
        if any(word in description.lower() for word in ["brute force", "password spray"]):
            threat_score += 30
        
        threat_level = "CRITICAL" if threat_score >= 70 else "HIGH" if threat_score >= 50 else "MEDIUM"
        
        return {
            "threat_level": threat_level,
            "threat_score": threat_score,
            "iocs": {
                "source_ip": source_ip,
                "username": username
            },
            "recommended_actions": [
                "Review authentication logs for this user",
                "Check if account is compromised",
                "Consider implementing MFA",
                "Temporarily lock account if high confidence"
            ],
            "confidence": 0.75
        }
    
    def _analyze_malware(
        self, description: str, source_ip: Optional[str],
        destination_ip: Optional[str], username: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze malware detection."""
        threat_score = 70  # Malware is always high severity
        
        # Check for specific malware types
        malware_types = []
        if "ransomware" in description.lower():
            threat_score = 95
            malware_types.append("ransomware")
        if "trojan" in description.lower():
            threat_score = max(threat_score, 80)
            malware_types.append("trojan")
        if "worm" in description.lower():
            threat_score = max(threat_score, 75)
            malware_types.append("worm")
        
        return {
            "threat_level": "CRITICAL",
            "threat_score": threat_score,
            "malware_types": malware_types or ["unknown"],
            "iocs": {
                "affected_host": source_ip or "unknown",
                "user": username
            },
            "recommended_actions": [
                "🚨 ISOLATE affected system immediately",
                "Run full system scan with updated definitions",
                "Preserve forensic evidence",
                "Check for lateral movement",
                "Review recent file modifications",
                "Notify security team and management"
            ],
            "confidence": 0.90
        }
    
    def _analyze_network_intrusion(
        self, description: str, source_ip: Optional[str],
        destination_ip: Optional[str], username: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze network intrusion."""
        threat_score = 60
        
        # Check for scanning activity
        if any(word in description.lower() for word in ["scan", "probe", "reconnaissance"]):
            threat_score += 15
        
        # Check for exploitation attempts
        if any(word in description.lower() for word in ["exploit", "vulnerability", "cve"]):
            threat_score += 25
        
        threat_level = "CRITICAL" if threat_score >= 80 else "HIGH" if threat_score >= 60 else "MEDIUM"
        
        return {
            "threat_level": threat_level,
            "threat_score": threat_score,
            "iocs": {
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "attack_vector": self._identify_attack_vector(description)
            },
            "recommended_actions": [
                "Block source IP at firewall",
                "Review IDS/IPS signatures",
                "Check for successful exploitation",
                "Patch vulnerable systems",
                "Monitor for similar activity"
            ],
            "confidence": 0.80
        }
    
    def _analyze_data_exfiltration(
        self, description: str, source_ip: Optional[str],
        destination_ip: Optional[str], username: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze data exfiltration."""
        return {
            "threat_level": "CRITICAL",
            "threat_score": 90,
            "iocs": {
                "source": source_ip,
                "destination": destination_ip,
                "user": username
            },
            "recommended_actions": [
                "🚨 IMMEDIATE: Block destination IP/domain",
                "Identify what data was accessed",
                "Preserve network logs and packet captures",
                "Investigate user account for compromise",
                "Review DLP policies",
                "Notify legal and compliance teams"
            ],
            "confidence": 0.85
        }
    
    def _analyze_privilege_escalation(
        self, description: str, source_ip: Optional[str],
        destination_ip: Optional[str], username: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze privilege escalation attempt."""
        return {
            "threat_level": "HIGH",
            "threat_score": 75,
            "iocs": {
                "user": username,
                "source_ip": source_ip
            },
            "recommended_actions": [
                "Review user's current permissions",
                "Check for unauthorized access to sensitive resources",
                "Audit recent privilege changes",
                "Investigate user account activity",
                "Review sudo/admin logs"
            ],
            "confidence": 0.80
        }
    
    def _analyze_suspicious_process(
        self, description: str, source_ip: Optional[str],
        destination_ip: Optional[str], username: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze suspicious process."""
        threat_score = 50
        
        if any(word in description.lower() for word in ["powershell", "cmd.exe", "bash"]):
            threat_score += 20
        
        return {
            "threat_level": "MEDIUM" if threat_score < 70 else "HIGH",
            "threat_score": threat_score,
            "iocs": {
                "host": source_ip,
                "user": username
            },
            "recommended_actions": [
                "Analyze process behavior and network connections",
                "Check parent process and command line",
                "Review file system changes",
                "Collect process memory dump if suspicious",
                "Correlate with other endpoint events"
            ],
            "confidence": 0.70
        }
    
    def _analyze_generic(
        self, description: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze unknown alert type."""
        return {
            "threat_level": "MEDIUM",
            "threat_score": 50,
            "iocs": {},
            "recommended_actions": [
                "Review alert details manually",
                "Correlate with other security events",
                "Check SIEM for similar patterns"
            ],
            "confidence": 0.50
        }
    
    def _identify_attack_vector(self, description: str) -> str:
        """Identify the attack vector from description."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["sql", "injection"]):
            return "SQL Injection"
        elif any(word in description_lower for word in ["xss", "script"]):
            return "Cross-Site Scripting"
        elif any(word in description_lower for word in ["buffer", "overflow"]):
            return "Buffer Overflow"
        elif any(word in description_lower for word in ["dos", "ddos"]):
            return "Denial of Service"
        else:
            return "Unknown"
    
    def _generate_threat_recommendations(
        self, threat_level: str, indicators: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on threat level."""
        recommendations = []
        
        if threat_level == "CRITICAL":
            recommendations.extend([
                "🚨 Activate incident response team immediately",
                "Isolate affected systems from network",
                "Preserve all forensic evidence",
                "Notify senior management and stakeholders"
            ])
        elif threat_level == "HIGH":
            recommendations.extend([
                "⚠️ Escalate to security operations center",
                "Begin detailed investigation",
                "Review and update security controls",
                "Prepare incident report"
            ])
        elif threat_level == "MEDIUM":
            recommendations.extend([
                "📋 Document findings thoroughly",
                "Monitor affected systems closely",
                "Review security policies",
                "Consider additional monitoring"
            ])
        else:
            recommendations.extend([
                "✓ Continue routine monitoring",
                "Log event for future correlation",
                "No immediate action required"
            ])
        
        return recommendations
    
    def _count_by_severity(self, indicators: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count indicators by severity level."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for indicator in indicators:
            severity = indicator.get("severity", "low")
            if severity in counts:
                counts[severity] += 1
        return counts
    
    def _detect_alert_type(self, context: Dict[str, Any], description: str) -> str:
        """Auto-detect alert type from context and description."""
        context_str = str(context).lower()
        desc_lower = description.lower() if description else ""
        
        # Check for impossible travel
        if any(key in context for key in ['classification', 'locations', 'distance_km']):
            if context.get('classification') == 'IMPOSSIBLE_TRAVEL':
                return 'authentication_anomaly'
        
        # Check for malware indicators
        if any(word in context_str or word in desc_lower for word in 
               ['malware', 'virus', 'trojan', 'ransomware', 'malicious_process']):
            return 'malware_detection'
        
        # Check for data exfiltration
        if any(word in context_str or word in desc_lower for word in 
               ['exfiltration', 'data_leak', 'download', 'upload', 'transfer']):
            if any(word in context_str or word in desc_lower for word in ['bulk', 'massive', 'unusual']):
                return 'data_exfiltration'
        
        # Check for privilege escalation
        if any(word in context_str or word in desc_lower for word in 
               ['privilege', 'escalation', 'sudo', 'admin', 'root']):
            return 'privilege_escalation'
        
        # Check for network intrusion
        if any(word in context_str or word in desc_lower for word in 
               ['intrusion', 'scan', 'probe', 'exploit', 'vulnerability']):
            return 'network_intrusion'
        
        # Check for suspicious process
        if any(word in context_str or word in desc_lower for word in 
               ['process', 'executable', 'command', 'powershell', 'cmd']):
            return 'suspicious_process'
        
        # Check for authentication issues
        if any(word in context_str or word in desc_lower for word in 
               ['authentication', 'login', 'failed', 'brute', 'password']):
            return 'authentication_anomaly'
        
        # Default to generic analysis
        return 'unknown'
