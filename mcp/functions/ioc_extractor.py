"""
IoC (Indicator of Compromise) Extractor

Extracts IPs, domains, file hashes, URLs, and other indicators from logs and alert text.
"""

import re
from typing import Dict, List, Any


class IoCExtractor:
    """Extract indicators of compromise from security logs and alerts."""
    
    def __init__(self):
        """Initialize IoC extraction patterns."""
        # Regex patterns for different IoC types
        self.patterns = {
            'ipv4': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            'ipv6': r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Extract emails first
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'domain': r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
            'md5': r'\b[a-fA-F0-9]{32}\b',
            'sha1': r'\b[a-fA-F0-9]{40}\b',
            'sha256': r'\b[a-fA-F0-9]{64}\b',
            'cve': r'CVE-\d{4}-\d{4,7}',
        }
    
    def extract_iocs(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all IoCs from text.
        
        Args:
            text: Log text or alert description to analyze
            
        Returns:
            Dictionary mapping IoC types to lists of found indicators
            
        Example:
            >>> extractor = IoCExtractor()
            >>> text = "Login from 93.115.92.100 to malicious.com"
            >>> result = extractor.extract_iocs(text)
            >>> print(result)
            {
                "ipv4": ["93.115.92.100"],
                "domain": ["malicious.com"],
                "ipv6": [],
                "url": [],
                "email": [],
                "md5": [],
                "sha1": [],
                "sha256": [],
                "cve": []
            }
        """
        results = {}
        extracted_emails = []
        
        for ioc_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            # Remove duplicates while preserving order
            unique_matches = list(dict.fromkeys(matches))
            
            # Store emails to filter them from domains later
            if ioc_type == 'email':
                extracted_emails = unique_matches
            
            # Filter out common false positives for domains
            if ioc_type == 'domain':
                unique_matches = self._filter_false_positive_domains(unique_matches, extracted_emails)
            
            results[ioc_type] = unique_matches
        
        return results
    
    def _filter_false_positive_domains(self, domains: List[str], emails: List[str]) -> List[str]:
        """Filter out common false positive domains and email usernames."""
        # Common internal/safe domains to exclude
        false_positives = {
            'company.com',
            'localhost.localdomain',
            'internal.local',
            'test.local',
            'example.com',
            'example.org',
            'example.net',
            'microsoft.com',
        }
        
        # Extract email usernames and domain parts to filter out
        email_parts_to_filter = set()
        for email in emails:
            if '@' in email:
                username, email_domain = email.split('@', 1)
                email_parts_to_filter.add(username)  # e.g., "hr.manager"
                # Also add any dotted parts of username (e.g., "hr" from "hr.manager")
                if '.' in username:
                    for part in username.split('.'):
                        if len(part) > 2:  # Only add substantial parts
                            email_parts_to_filter.add(part)
        
        # File extensions and executables to filter
        file_extensions = {'.exe', '.dll', '.sys', '.dat', '.log', '.txt', '.json', '.xml', 
                          '.html', '.pdf', '.doc', '.docx', '.ps1', '.bat', '.cmd', '.msi'}
        
        filtered = []
        for domain in domains:
            domain_lower = domain.lower()
            
            # Skip if ends with file extension
            if any(domain_lower.endswith(ext) for ext in file_extensions):
                continue
            
            # Skip if it's a Windows executable or class name
            if any(word in domain_lower for word in ['winword', 'explorer', 'powershell', 
                                                       'svchost', 'webclient', 'rundll']):
                continue
            
            # Skip if it's just a filename without proper TLD
            if '.' in domain:
                parts = domain.split('.')
                tld = parts[-1].lower()
                # Real domains should have valid TLDs (2-6 chars, letters only)
                if not (2 <= len(tld) <= 6 and tld.isalpha()):
                    continue
            
            # Skip if it's a false positive, email part, or too short
            if (domain_lower not in false_positives and 
                domain not in email_parts_to_filter and
                len(domain) > 3):  # Minimum reasonable domain length
                filtered.append(domain)
        
        return filtered
    
    def extract_structured(self, text: str) -> Dict[str, Any]:
        """
        Extract IoCs and return structured data suitable for threat intelligence queries.
        
        Args:
            text: Log text or alert description
            
        Returns:
            Structured dictionary with categorized IoCs and statistics
            
        Example:
            >>> result = extractor.extract_structured(log_text)
            >>> print(result["summary"])
            "Found 2 IPs, 1 domain, 0 hashes"
        """
        iocs = self.extract_iocs(text)
        
        # Count non-empty findings
        counts = {k: len(v) for k, v in iocs.items() if v}
        total_count = sum(counts.values())
        
        # Generate summary
        summary_parts = []
        if counts.get('ipv4', 0) > 0:
            summary_parts.append(f"{counts['ipv4']} IPv4")
        if counts.get('ipv6', 0) > 0:
            summary_parts.append(f"{counts['ipv6']} IPv6")
        if counts.get('domain', 0) > 0:
            summary_parts.append(f"{counts['domain']} domain(s)")
        if counts.get('url', 0) > 0:
            summary_parts.append(f"{counts['url']} URL(s)")
        if counts.get('email', 0) > 0:
            summary_parts.append(f"{counts['email']} email(s)")
        
        hash_count = counts.get('md5', 0) + counts.get('sha1', 0) + counts.get('sha256', 0)
        if hash_count > 0:
            summary_parts.append(f"{hash_count} hash(es)")
        
        if counts.get('cve', 0) > 0:
            summary_parts.append(f"{counts['cve']} CVE(s)")
        
        summary = f"Found {', '.join(summary_parts)}" if summary_parts else "No IoCs found"
        
        return {
            "iocs": iocs,
            "counts": counts,
            "total_count": total_count,
            "summary": summary,
            "threat_intel_ready": {
                "ips_to_check": iocs['ipv4'] + iocs['ipv6'],
                "domains_to_check": iocs['domain'],
                "hashes_to_check": iocs['md5'] + iocs['sha1'] + iocs['sha256']
            }
        }


# Singleton instance
_extractor = IoCExtractor()


def extract_iocs(text: str) -> str:
    """
    Extract indicators of compromise (IoCs) from log text or alerts.
    
    This function scans text for:
    - IP addresses (IPv4 and IPv6)
    - Domain names
    - URLs
    - Email addresses
    - File hashes (MD5, SHA1, SHA256)
    - CVE identifiers
    
    Args:
        text: Log text, alert description, or incident details to analyze
        
    Returns:
        JSON string with extracted IoCs categorized by type
        
    Example:
        >>> result = extract_iocs("Login from 93.115.92.100 accessing evil.com")
        >>> # Returns: {"ipv4": ["93.115.92.100"], "domain": ["evil.com"], ...}
    """
    import json
    result = _extractor.extract_structured(text)
    return json.dumps(result, indent=2)
