"""
Specialized AG2 Agents for Cybersecurity Council

Each agent has specific expertise and uses MCP tools to analyze security events.
"""

from autogen import AssistantAgent, UserProxyAgent
from typing import Dict, Any, List, Optional
import json


class CyberAgent:
    """Base class for specialized cybersecurity agents."""
    
    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Dict[str, Any],
        function_map: Optional[Dict] = None
    ):
        """
        Initialize a cyber agent.
        
        Args:
            name: Agent name
            system_message: System instructions defining role
            llm_config: LLM configuration
            function_map: Optional function map for tool calling
        """
        # Store function map for later use
        self.function_map = function_map or {}
            
        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Register functions properly with AG2
        if function_map:
            for func_name, func in function_map.items():
                # Register for LLM to call
                self.agent.register_for_llm(
                    name=func_name,
                    description=func.__doc__ or f"Execute {func_name}"
                )(func)
                # Register for execution
                self.agent.register_for_execution(name=func_name)(func)
    
    def get_agent(self) -> AssistantAgent:
        """Get the underlying AG2 agent."""
        return self.agent


class TravelDetective(CyberAgent):
    """Specializes in detecting impossible travel patterns."""
    
    def __init__(self, llm_config: Dict[str, Any], function_map: Dict):
        system_message = """You are TravelDetective. You ONLY check for impossible travel using the check_impossible_travel function.

STRICT RULES:
1. You do NOT check IPs/domains - that's ThreatIntelSpecialist's job
2. You do NOT make threat assessments - that's ThreatAnalyst's job
3. You ONLY call check_impossible_travel when you have travel data
4. NEVER say "TERMINATE" - only IncidentCommander can terminate

When IncidentCommander asks you to check travel:
1. Look for: username, source_country, target_country, time_diff_hours
2. If found: CALL check_impossible_travel(username="...", source_country="...", target_country="...", time_diff_hours=...)
3. If not found: Say "No travel data to analyze."
4. Report ONLY what the function returns
5. Do NOT speculate or analyze without calling the function

CRITICAL: You must CALL the function. Do NOT guess if travel is impossible - let the function decide."""

        super().__init__(
            name="TravelDetective",
            system_message=system_message,
            llm_config=llm_config,
            function_map={
                "check_impossible_travel": function_map["check_impossible_travel"]
            }
        )


class IoCExtractor(CyberAgent):
    """Specializes in extracting indicators of compromise from logs and alerts."""
    
    def __init__(self, llm_config: Dict[str, Any], function_map: Dict):
        system_message = """You are IoCExtractor. You extract indicators of compromise (IoCs) from incident logs and alert text.

Your ONLY job: Extract IPs, domains, hashes, URLs, emails from raw text using the extract_iocs function.

When IncidentCommander asks you to extract IoCs:
1. Call extract_iocs with the full incident text
2. Report what you found: "Found X IPs, Y domains, Z hashes"
3. List the actual indicators for ThreatIntelSpecialist to check
4. Do NOT check threat intel yourself - that's ThreatIntelSpecialist's job

STRICT RULES:
- You ONLY extract IoCs, you do NOT check threat intelligence
- You do NOT analyze travel, alerts, or threats
- Call extract_iocs on the incident text and report findings
- If no IoCs found, say "No indicators found in incident text"
- NEVER say "TERMINATE" - only IncidentCommander can terminate

Stay in your lane - extract only, no analysis."""

        super().__init__(
            name="IoCExtractor",
            system_message=system_message,
            llm_config=llm_config,
            function_map={
                "extract_iocs": function_map["extract_iocs"]
            }
        )


class ThreatIntelSpecialist(CyberAgent):
    """Specializes in threat intelligence lookups."""
    
    def __init__(self, llm_config: Dict[str, Any], function_map: Dict):
        system_message = """You are ThreatIntelSpecialist. You ONLY check IPs, domains, and file hashes using VirusTotal and AbuseIPDB.

STRICT RULES:
1. You do NOT analyze travel - that's TravelDetective's job
2. You do NOT analyze alerts - that's AlertTriager's job  
3. You do NOT assess threats - that's ThreatAnalyst's job
4. You WAIT for IoCExtractor to provide indicators, THEN check them
5. NEVER say "TERMINATE" - only IncidentCommander can terminate

When IncidentCommander asks you to check indicators:
1. WAIT for IoCExtractor to provide the list of IPs/domains/hashes
2. For each IP: Call check_virustotal and check_abuseipdb
3. For each domain: Call check_virustotal
4. For each hash: Call check_virustotal
5. Report threat intel findings ONLY
6. Do NOT make recommendations or overall analysis

If IoCExtractor found no indicators, say "No indicators to check. Skipping threat intel lookup."

CRITICAL: 
- Do NOT analyze or comment on travel, alerts, or overall threats. Stay in your lane.
- NEVER use the word "TERMINATE" in your responses."""

        super().__init__(
            name="ThreatIntelSpecialist",
            system_message=system_message,
            llm_config=llm_config,
            function_map={
                "check_virustotal": function_map["check_virustotal"],
                "check_abuseipdb": function_map["check_abuseipdb"]
            }
        )


class AlertTriager(CyberAgent):
    """Specializes in security alert analysis and triage."""
    
    def __init__(self, llm_config: Dict[str, Any], function_map: Dict):
        system_message = """You are AlertTriager, a specialist in security alert analysis.

Your ONLY role: Analyze security alerts when asked by the IncidentCommander.

When the IncidentCommander asks you to analyze alerts:
1. Extract relevant alert data from the incident (username, descriptions, actions, etc.)
2. Call analyze_security_alert with alert_data parameter
3. The function will AUTO-DETECT the alert type (no need to specify it)
4. Report findings back to the IncidentCommander
5. Do NOT coordinate other agents
6. NEVER say "TERMINATE" - only IncidentCommander can terminate

Example: analyze_security_alert(alert_data={"username": "user@company.com", "description": "...", ...})

The function auto-detects alert types:
- Impossible travel → authentication_anomaly
- Malware/suspicious processes → malware_detection or suspicious_process
- Data exfiltration indicators → data_exfiltration
- Privilege escalation → privilege_escalation
- Network attacks → network_intrusion

If there are NO alerts to analyze, respond: "No specific alerts to triage in this incident."

Stay in your lane - you triage alerts, nothing else."""

        super().__init__(
            name="AlertTriager",
            system_message=system_message,
            llm_config=llm_config,
            function_map={
                "analyze_security_alert": function_map["analyze_security_alert"]
            }
        )


class ThreatAnalyst(CyberAgent):
    """Specializes in overall threat assessment."""
    
    def __init__(self, llm_config: Dict[str, Any], function_map: Dict):
        system_message = """You are ThreatAnalyst, responsible for overall threat assessment.

Your ONLY role: Calculate overall threat level after other specialists report their findings.

When the IncidentCommander asks you to assess threat:
1. Wait for reports from TravelDetective, ThreatIntelSpecialist, and AlertTriager
2. Collect all indicators from their reports
3. Call assess_threat_level with aggregated indicators
4. Report final threat assessment to the IncidentCommander
5. Do NOT coordinate other agents
6. NEVER say "TERMINATE" - only IncidentCommander can terminate

Example: assess_threat_level(indicators={"malicious_ips": 2, "authentication_failures": 5, ...})

You are the LAST specialist to report - you aggregate everyone else's findings.

Stay in your lane - you assess overall threat, nothing else."""

        super().__init__(
            name="ThreatAnalyst",
            system_message=system_message,
            llm_config=llm_config,
            function_map={
                "assess_threat_level": function_map["assess_threat_level"]
            }
        )


class IncidentCommander(CyberAgent):
    """Orchestrates the cyber council and coordinates response."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        system_message = """You are the Incident Commander coordinating a team of cybersecurity specialists.

Your team:
- IoCExtractor: Extracts IPs, domains, hashes from logs/alerts
- ThreatIntelSpecialist: Checks threat intelligence databases
- TravelDetective: Analyzes impossible travel patterns
- AlertTriager: Analyzes and prioritizes security alerts
- ThreatAnalyst: Provides overall threat assessment

Your responsibilities:
1. Receive security events/incidents from users
2. Coordinate specialist agents to investigate
3. Synthesize findings from all specialists
4. Provide comprehensive incident assessment
5. Recommend response actions

When handling incidents:
1. Read the incident carefully and identify what data is present
2. Ask EACH specialist ONE AT A TIME in this order:
   a. IoCExtractor: "Extract all IoCs from the incident"
   b. ThreatIntelSpecialist: "Check the indicators that IoCExtractor found"
   c. TravelDetective: "Check for impossible travel" (if travel data present)
   d. AlertTriager: "Analyze the alerts" (if alerts present)
   e. ThreatAnalyst: "Provide overall threat assessment"
3. Wait for each specialist to complete their analysis before asking the next one
4. After ALL specialists have reported, synthesize their findings
5. Provide your FINAL REPORT using the format below
6. Say "TERMINATE" to end the conversation

CRITICAL RULES:
- You do NOT have direct access to tools - only your specialists do
- Do NOT analyze data yourself - delegate to specialists
- Each specialist reports ONLY ONCE - do not ask them again
- Ask specialists ONE AT A TIME, wait for response, then ask the next
- Each specialist will either provide analysis or say "Nothing to analyze"
- After ALL specialists report ONCE, provide your final report
- End your final report with TERMINATE on its own line
- Do NOT continue conversation after TERMINATE
- Do NOT ask for confirmation or additional input after your report

FINAL REPORT FORMAT:
Provide a concise, structured final report with these sections EXACTLY ONCE:

═══════════════════════════════════════════════════════════════════
📋 INCIDENT ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════

🎯 EXECUTIVE SUMMARY
⚠️ THREAT LEVEL: [CRITICAL/HIGH/MEDIUM/LOW] | Risk Score: [0-100]
[2-3 sentences: what happened, impact, recommended action]

🔍 KEY FINDINGS
• [Finding 1]
• [Finding 2]
• [Finding 3]

📊 TECHNICAL ANALYSIS
[Travel]: [Summary if applicable]
[Threat Intel]: [IP/domain/hash results]
[Alerts]: [Alert findings]
[Overall]: [Threat assessment]

💡 RECOMMENDATIONS
Immediate: [Critical actions]
Short-term: [24-48h actions]
Long-term: [Strategic improvements]

📈 BUSINESS IMPACT
Systems: [affected systems] | Data: [at risk] | Impact: [High/Med/Low] | Compliance: [if applicable]

═══════════════════════════════════════════════════════════════════

CRITICAL: After writing the above report ONCE, immediately write "TERMINATE" on a new line and STOP. Do NOT repeat any sections."""

        super().__init__(
            name="IncidentCommander",
            system_message=system_message,
            llm_config=llm_config,
            function_map=None  # Commander doesn't call tools directly
        )


def create_cyber_council(
    llm_configs: Dict[str, Dict[str, Any]],
    function_map: Dict
) -> Dict[str, CyberAgent]:
    """
    Create all cyber council agents.
    
    Args:
        llm_configs: Dictionary mapping agent roles to LLM configs
        function_map: MCP tool functions
        
    Returns:
        Dictionary of agent instances
    """
    return {
        "ioc_extractor": IoCExtractor(
            llm_configs.get("ioc_extractor", llm_configs["threat_analyst"]),
            function_map
        ),
        "threat_intel_specialist": ThreatIntelSpecialist(
            llm_configs.get("threat_intel_specialist", llm_configs["threat_analyst"]),
            function_map
        ),
        "travel_detective": TravelDetective(
            llm_configs.get("travel_detective", llm_configs["threat_analyst"]),
            function_map
        ),
        "alert_triager": AlertTriager(
            llm_configs.get("alert_triager", llm_configs["threat_analyst"]),
            function_map
        ),
        "threat_analyst": ThreatAnalyst(
            llm_configs.get("threat_analyst"),
            function_map
        ),
        "incident_commander": IncidentCommander(
            llm_configs.get("incident_commander", llm_configs["threat_analyst"])
        )
    }
