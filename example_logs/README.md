# Example Security Logs for LLM Council Analysis

This directory contains realistic security log files that demonstrate various attack scenarios. These logs can be fed directly to the LLM Council for analysis and decision-making.

## Log Files Overview

### 1. `impossible_travel.log`
**Scenario:** User account compromised with impossible travel pattern
- User logs in from Bucharest, Romania at 08:00 UTC
- Same user logs in from Tokyo, Japan at 09:30 UTC (1.5 hours later)
- Distance: 8,847 km requiring 5,898 km/h speed
- Post-compromise activities: email access, bulk downloads, email forwarding rules
- **Council Actions:** TravelDetective checks impossible travel, AlertTriager assesses anomaly, ThreatAnalyst evaluates risk

### 2. `malware_detection.log`
**Scenario:** Ransomware (LockBit 3.0) infection via email attachment
- Phishing email with double extension (.pdf.exe)
- Malware execution with shadow copy deletion
- C2 communication to 45.142.212.100 (Russia)
- Mass file encryption (1,247 files)
- Ransom note creation
- **Council Actions:** ThreatIntelSpecialist checks file hash & C2 IP, AlertTriager analyzes malware behavior, ThreatAnalyst assesses severity

### 3. `brute_force_attack.log`
**Scenario:** Successful credential stuffing attack against domain admin
- 147 failed login attempts from 5 Chinese IPs
- Successful login from 203.0.113.50 (China)
- Post-compromise: created backdoor admin account, modified Group Policy, exported AD database
- Automated attack pattern (Python-requests user agent)
- **Council Actions:** AlertTriager identifies brute force pattern, ThreatIntelSpecialist checks source IPs, ThreatAnalyst evaluates post-compromise impact

### 4. `c2_communication.log`
**Scenario:** Cobalt Strike C2 beaconing detected
- Regular beaconing pattern (60-second intervals) to 185.220.101.50
- IDS signatures: "ET MALWARE Cobalt Strike Beacon"
- Self-signed SSL certificate
- Executable download (50 MB payload)
- **Council Actions:** ThreatIntelSpecialist checks C2 IP reputation, AlertTriager analyzes network anomaly, ThreatAnalyst assesses APT indicators

### 5. `data_exfiltration.log`
**Scenario:** Insider threat - contractor exfiltrates 250GB of sensitive data
- Off-hours activity (02:00 AM)
- Database exports: 1.5M customer records, 250K financial transactions
- Compressed 250GB of data to external cloud storage
- PII exposure: 1.5M records, 45K credit cards, 120K SSNs
- **Council Actions:** AlertTriager flags off-hours anomaly, ThreatIntelSpecialist checks destination domain, ThreatAnalyst evaluates regulatory impact

### 6. `privilege_escalation.log`
**Scenario:** Zero-day exploit (CVE-2024-38063) for privilege escalation
- Standard user opens malicious document
- Obfuscated PowerShell command exploits TCP/IP vulnerability
- Escalation from standard_user to SYSTEM
- Post-exploitation: disabled Windows Defender, credential dumping with Mimikatz, SAM database access
- **Council Actions:** AlertTriager detects exploit indicators, ThreatAnalyst assesses privilege escalation risk, IncidentCommander coordinates response

### 7. `suspicious_process.log`
**Scenario:** Office macro malware with process masquerading
- Malicious Word document with auto-executing macro
- Downloads fake svchost.exe to user temp directory
- Process masquerading (wrong location for svchost.exe)
- Typosquatting domain: update-microsoft-services.com
- Process injection into explorer.exe
- Document exfiltration (50 MB)
- **Council Actions:** AlertTriager identifies suspicious process behavior, ThreatIntelSpecialist checks domain reputation, ThreatAnalyst evaluates overall threat

### 8. `comprehensive_apt_attack.log`
**Scenario:** Full APT kill chain - all attack stages in one incident
- **Stage 1:** Spear phishing email
- **Stage 2:** Credential harvesting (phishing site)
- **Stage 3:** Initial access from Romania
- **Stage 4:** Impossible travel (Romania → Japan in 1.17 hours)
- **Stage 5:** Ransomware deployment (LockBit 3.0)
- **Stage 6:** Privilege escalation (CVE-2024-38063)
- **Stage 7:** C2 communication (Cobalt Strike)
- **Stage 8:** Lateral movement (Pass-the-Hash)
- **Stage 9:** Data staging (250 GB compressed)
- **Stage 10:** Mass encryption (487K files, 5 TB)
- **Stage 11:** Data exfiltration (250 GB to Bulgaria)
- **Stage 12:** Backdoor persistence (3 systems)
- **Attribution:** APT29 (Cozy Bear)
- **Impact:** 15 systems compromised, $15M estimated loss, 1.5M PII records exposed
- **Council Actions:** ALL council members engaged, full multi-agent analysis, CRITICAL incident response

## How to Use These Logs

### Option 1: Direct CLI Input
```bash
cd council
python cli.py

# At the prompt, paste log contents
> Please analyze these security logs:
> [paste log file contents]
```

### Option 2: Feed Entire Log File
```powershell
# PowerShell
Get-Content ..\example_logs\impossible_travel.log | python cli.py analyze

# Or manually copy-paste entire log into CLI
```

### Option 3: Programmatic Analysis
```python
from council.orchestrator import CyberCouncil

council = CyberCouncil()

# Read log file
with open('../example_logs/malware_detection.log', 'r') as f:
    log_data = f.read()

# Analyze with council
result = council.analyze_incident(f"Analyze these security logs:\n\n{log_data}")
print(result)
```

## Expected Council Behavior

When you feed these logs to the LLM Council:

1. **IncidentCommander** reads the logs and identifies key events
2. **TravelDetective** extracts login locations/times and checks for impossible travel
3. **ThreatIntelSpecialist** extracts IPs, domains, and hashes, then queries threat intelligence
4. **AlertTriager** identifies security events and categorizes alerts
5. **ThreatAnalyst** aggregates findings and provides risk assessment
6. **IncidentCommander** synthesizes all analyses into final decision

## Log Format Details

All logs follow standard security logging patterns:
- **Timestamp:** ISO 8601 format (2024-11-23T08:00:00Z)
- **Log Level:** INFO, WARN, HIGH, CRITICAL
- **Source:** [AUTH], [EDR], [NETWORK], [ANOMALY], [INCIDENT], etc.
- **Key-Value Pairs:** Structured data for parsing
- **IOCs:** IP addresses, domains, file hashes embedded in logs

## Threat Intelligence Notes

These logs contain realistic IOCs:
- **IP Ranges:** 
  - 93.115.92.100 (Romania - attacker source)
  - 103.79.141.100 (Japan - attacker source)
  - 45.142.212.100 (Russia - C2 server)
  - 185.220.101.50 (Bulgaria - exfiltration destination)
  - 192.0.2.* (RFC 5737 documentation range)
  - 203.0.113.* (RFC 5737 documentation range)
  
- **Malicious Domains:**
  - company-login-portal-secure.xyz (phishing)
  - update-services-microsoft.com (C2, typosquatting)
  - cloud-backup-services.xyz (exfiltration)
  - suspicious-cloud-storage.xyz (data staging)

- **Malware Hashes:**
  - 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f (LockBit 3.0 ransomware)
  - def789abc456ghi123jkl456mno789pqr012stu345vwx678yza901bcd234 (backdoor)
  - abc123def456789ghi012jkl345mno678pqr901stu234vwx567yza890bcd (info stealer)

## Demonstration Value

These logs are perfect for demonstrating:
- ✅ Multi-agent collaboration (all 5 agents working together)
- ✅ Real-world attack patterns (APT tactics, techniques, procedures)
- ✅ Tool integration (impossible travel ML model, VirusTotal, AbuseIPDB)
- ✅ Complex decision-making (weighing multiple indicators)
- ✅ Incident response recommendations (containment, eradication, recovery)
- ✅ Regulatory impact assessment (GDPR, HIPAA, PCI-DSS)

## Testing Recommendations

**Start with simple cases:**
1. `impossible_travel.log` - Tests travel detection + basic threat assessment
2. `malware_detection.log` - Tests threat intel integration + file hash analysis

**Move to complex cases:**
3. `brute_force_attack.log` - Tests pattern recognition + post-compromise analysis
4. `privilege_escalation.log` - Tests exploit detection + impact assessment

**Finish with comprehensive:**
5. `comprehensive_apt_attack.log` - Tests full council collaboration on complex APT

**Expected Processing Time:**
- Simple logs: 30-60 seconds
- Complex logs: 2-3 minutes
- Comprehensive APT: 3-5 minutes (with max_rounds=25)

## Customization

To create your own log files:
1. Follow the timestamp + log level + source pattern
2. Include structured key-value pairs
3. Embed IOCs (IPs, domains, hashes) naturally in events
4. Maintain chronological order
5. Include anomaly/incident summary entries

Example log entry template:
```
2024-11-23T12:34:56Z [SOURCE] LEVEL key1=value1 key2=value2 description="text"
```
