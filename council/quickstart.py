"""
Quick Start Guide - Cyber Council
"""

from council.orchestrator import CyberCouncil


def example_1_simple_analysis():
    """Simple incident analysis."""
    print("\n" + "="*80)
    print("Example 1: Simple Incident Analysis")
    print("="*80)
    
    council = CyberCouncil()
    
    incident = "User logged in from US at 8am, then from China at 9am"
    
    results = council.analyze_incident(incident, max_rounds=5)
    

def example_2_quick_checks():
    """Demonstrate quick single-agent checks."""
    print("\n" + "="*80)
    print("Example 2: Quick Checks (No Multi-Agent)")
    print("="*80)
    
    council = CyberCouncil()
    
    # Check impossible travel
    print("\n🌍 Checking impossible travel...")
    result = council.quick_check(
        "travel",
        username="john.doe",
        source_ip="93.115.92.100",
        source_country="RO",
        target_ip="103.79.141.100", 
        target_country="JP",
        time_diff_hours=1.5
    )
    print(result)
    
    # Check IP reputation (if API keys configured)
    print("\n🔍 Checking IP reputation...")
    try:
        result = council.quick_check(
            "virustotal",
            indicator="192.0.2.1",
            indicator_type="ip"
        )
        print(result)
    except Exception as e:
        print(f"VirusTotal check requires API key: {e}")


def example_3_complex_incident():
    """Complex multi-faceted incident."""
    print("\n" + "="*80)
    print("Example 3: Complex Incident with Multiple IOCs")
    print("="*80)
    
    council = CyberCouncil()
    
    incident = """
    INCIDENT REPORT - ID: INC-2024-001
    
    Timeline:
    - 08:00 UTC: User alice@company.com login from London, UK (51.5074, -0.1278)
    - 09:30 UTC: Same user login from Tokyo, Japan (35.6762, 139.6503)
    - 09:45 UTC: Large data transfer detected (500 MB) to external IP: 203.0.113.42
    - 10:00 UTC: Malware alert triggered on alice's workstation
    
    Indicators of Compromise (IOCs):
    - Source IP (London): 93.115.92.100
    - Source IP (Tokyo): 103.79.141.100
    - External IP: 203.0.113.42
    - File hash: 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f
    - Suspicious domain: evil-c2.com
    
    Assets accessed:
    - Financial database server
    - Customer PII records
    - Internal source code repository
    
    Please provide comprehensive threat assessment and response recommendations.
    """
    
    results = council.analyze_incident(incident, max_rounds=15)


def example_4_alert_triage():
    """Alert triage scenario."""
    print("\n" + "="*80)
    print("Example 4: Security Alert Triage")
    print("="*80)
    
    council = CyberCouncil()
    
    incident = """
    Multiple security alerts in last hour:
    
    1. Authentication Anomaly - alice@company.com
       - Failed login attempts: 50
       - Source IPs: 10 unique IPs from China
       - Account locked automatically
    
    2. Malware Detection - bob-laptop
       - File: invoice.pdf.exe
       - Detection: Trojan.Generic
       - Action: Quarantined
    
    3. Network Intrusion - web-server-01
       - SQL injection attempt detected
       - Source: 192.0.2.100
       - WAF blocked request
    
    Which alerts should we prioritize for immediate investigation?
    """
    
    results = council.analyze_incident(incident, max_rounds=8)


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              Cyber Council - Quick Start Examples            ║
╚═══════════════════════════════════════════════════════════════╝

Choose an example to run:
1. Simple incident analysis (quick demo)
2. Quick checks without multi-agent (fast)
3. Complex incident with multiple IOCs (comprehensive)
4. Alert triage scenario (prioritization)
""")
    
    choice = input("Enter choice (1-4) or 'all': ").strip()
    
    if choice == "1":
        example_1_simple_analysis()
    elif choice == "2":
        example_2_quick_checks()
    elif choice == "3":
        example_3_complex_incident()
    elif choice == "4":
        example_4_alert_triage()
    elif choice.lower() == "all":
        example_1_simple_analysis()
        example_2_quick_checks()
        example_3_complex_incident()
        example_4_alert_triage()
    else:
        print("Invalid choice")
