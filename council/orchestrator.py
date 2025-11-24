"""
Cybersecurity Council Orchestrator

Coordinates multiple AG2 agents to collaboratively analyze security incidents.
"""

from autogen import GroupChat, GroupChatManager
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from council.agents import create_cyber_council
from council.mcp_adapter import MCPToolAdapter
from council.llm_config import get_council_llm_configs, DEFAULT_CONFIG
from council.mcp_toolkit import create_mcp_toolkit


class CyberCouncil:
    """
    Orchestrates a council of specialized cybersecurity AI agents.
    """
    
    def __init__(self, use_all_agents: bool = True):
        """
        Initialize the cyber council.
        
        Args:
            use_all_agents: Whether to use all agents (default: True)
        """
        print("🔧 Initializing Cyber Council...")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize MCP tool adapter
        print("📦 Loading MCP tools...")
        self.mcp_adapter = MCPToolAdapter()
        self.toolkit = create_mcp_toolkit(self.mcp_adapter)
        self.function_map = self.toolkit  # For backward compatibility
        
        # Get LLM configurations
        print("🤖 Configuring LLMs...")
        self.llm_configs = get_council_llm_configs()


        
        # Create agents (they will register their own functions via function_map)
        print("👥 Creating specialist agents...")
        self.agents = create_cyber_council(self.llm_configs, self.toolkit)
        
        # Get list of agents for group chat
        self.agent_list = [agent.get_agent() for agent in self.agents.values()]
        
        print(f"✅ Council initialized with {len(self.agents)} agents + 1 executor")
        print(f"   Agents: {', '.join(self.agents.keys())}")
    
    def analyze_incident(
        self,
        incident_description: str,
        max_rounds: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze a security incident using the council.
        
        Args:
            incident_description: Description of the security event
            max_rounds: Maximum conversation rounds
            
        Returns:
            Analysis results from the council
        """
        print("\n" + "="*80)
        print("🚨 NEW INCIDENT ANALYSIS")
        print("="*80)
        print(f"\nIncident: {incident_description}\n")
        
        # Define workflow order
        workflow_order = [
            "IoCExtractor",
            "ThreatIntelSpecialist", 
            "TravelDetective",
            "AlertTriager",
            "ThreatAnalyst",
            "IncidentCommander"
        ]
        
        def select_next_speaker(last_speaker, groupchat):
            """Enforce strict workflow order."""
            messages = groupchat.messages
            
            # First message - start with Commander to kick things off
            if len(messages) == 0:
                return self.agents["incident_commander"].get_agent()
            
            last_message = messages[-1]
            last_name = last_message.get("name", "")
            last_content = str(last_message.get("content", ""))
            
            # Check for termination (only Commander can terminate)
            if last_name == "IncidentCommander":
                content_stripped = last_content.rstrip()
                if content_stripped.endswith("\nTERMINATE") or content_stripped == "TERMINATE":
                    return None  # End conversation
            
            # Check if we're in the middle of function execution (last 2 messages only)
            recent_messages = messages[-2:] if len(messages) >= 2 else messages
            
            # Only keep agent as speaker if they JUST called a tool or JUST received a response
            # and haven't sent a substantive analysis yet
            for msg in reversed(recent_messages):
                msg_content = str(msg.get("content", ""))
                msg_name = msg.get("name", "")
                
                # If the agent just suggested a tool call, keep them to receive the response
                if "***** Suggested tool call" in msg_content:
                    for agent_key, agent_wrapper in self.agents.items():
                        if agent_wrapper.get_agent().name == msg_name:
                            return agent_wrapper.get_agent()
                
                # If the agent just received a tool response, keep them to analyze it
                # BUT only if they haven't sent their analysis yet (next message after response)
                if "***** Response from calling tool" in msg_content:
                    # Check if there's a substantive message after this response
                    msg_index = messages.index(msg)
                    if msg_index == len(messages) - 1:
                        # Response is the last message, agent needs to process it
                        for agent_key, agent_wrapper in self.agents.items():
                            if agent_wrapper.get_agent().name == msg_name:
                                return agent_wrapper.get_agent()
                    # If there's already a message after the response, agent has processed it
            
            # Track how many times each agent has spoken (excluding tool call/response messages)
            agent_turns = {}
            commander_messages = 0
            
            for msg in messages:
                name = msg.get("name", "")
                content = str(msg.get("content", ""))
                
                if not name:
                    continue
                
                # Skip function call/response system messages
                if ("***** Suggested tool call" in content or 
                    "***** Response from calling tool" in content or
                    "EXECUTING FUNCTION" in content or
                    "EXECUTED FUNCTION" in content):
                    continue
                
                # Count substantial messages only
                if len(content.strip()) > 10:
                    if name == "IncidentCommander":
                        commander_messages += 1
                    else:
                        agent_turns[name] = agent_turns.get(name, 0) + 1
            
            # Strict workflow: each agent speaks once (except Commander who gets intro + final)
            # Only allow agent to continue if they haven't completed their turn yet
            
            ioc_turns = agent_turns.get("IoCExtractor", 0)
            intel_turns = agent_turns.get("ThreatIntelSpecialist", 0)
            travel_turns = agent_turns.get("TravelDetective", 0)
            alert_turns = agent_turns.get("AlertTriager", 0)
            analyst_turns = agent_turns.get("ThreatAnalyst", 0)
            
            # Workflow progression: each agent gets exactly 1 turn
            if ioc_turns == 0:
                return self.agents["ioc_extractor"].get_agent()
            elif intel_turns == 0:
                return self.agents["threat_intel_specialist"].get_agent()
            elif travel_turns == 0:
                return self.agents["travel_detective"].get_agent()
            elif alert_turns == 0:
                return self.agents["alert_triager"].get_agent()
            elif analyst_turns == 0:
                return self.agents["threat_analyst"].get_agent()
            elif commander_messages < 2:
                # All specialists done, Commander gives final report
                return self.agents["incident_commander"].get_agent()
            else:
                # Commander has spoken twice (intro + final report), terminate
                return None
        
        # Create group chat with custom speaker selection
        groupchat = GroupChat(
            agents=self.agent_list,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=select_next_speaker,
            allow_repeat_speaker=True  # Allow agents to call their own functions
        )
        
        # Create manager with incident commander's config
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_configs.get(
                "incident_commander",
                self.llm_configs["threat_analyst"]
            )
        )
        
        # Start the analysis
        commander = self.agents["incident_commander"].get_agent()
        
        try:
            # Termination function - only IncidentCommander can terminate
            def is_termination_msg(msg):
                """Stop when IncidentCommander says TERMINATE on its own line."""
                content = str(msg.get("content", ""))
                speaker = msg.get("name", "")
                
                # Only allow IncidentCommander to terminate
                if speaker != "IncidentCommander":
                    return False
                
                # Check if TERMINATE appears at the end (ignore trailing whitespace)
                stripped = content.rstrip()
                return stripped.endswith("\nTERMINATE") or stripped == "TERMINATE"
            
            commander.initiate_chat(
                manager,
                message=f"""You are the Incident Commander. Your specialist team will now analyze this security incident in order:

INCIDENT:
{incident_description}

Your specialists will report in this order:
1. IoCExtractor will extract all indicators (IPs, domains, hashes, URLs)
2. ThreatIntelSpecialist will check extracted indicators against threat intel databases
3. TravelDetective will check for impossible travel patterns
4. AlertTriager will analyze security alerts
5. ThreatAnalyst will provide overall threat assessment
6. You will provide the FINAL REPORT

Each specialist will report automatically in order. Do NOT ask them - they will speak when ready.

Your job:
- Listen to all specialist reports
- After ALL specialists have reported, synthesize their findings
- Provide comprehensive FINAL REPORT
- End with "TERMINATE" on its own line

CRITICAL: Say nothing until all 5 specialists have reported. Then provide your final report.""",
                clear_history=True,
                is_termination_msg=is_termination_msg
            )
            
            # Extract results
            chat_history = groupchat.messages
            
            result = {
                "incident": incident_description,
                "agents_consulted": [msg.get("name") for msg in chat_history if "name" in msg],
                "conversation": chat_history,
                "summary": chat_history[-1].get("content", "No summary available") if chat_history else "No analysis completed"
            }
            
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETE")
            print("="*80)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            return {
                "incident": incident_description,
                "error": str(e),
                "summary": "Analysis failed due to error"
            }
    
    def quick_check(
        self,
        check_type: str,
        **kwargs
    ) -> str:
        """
        Perform a quick single-agent check.
        
        Args:
            check_type: Type of check (travel, virustotal, abuseipdb, alert, threat)
            **kwargs: Parameters for the check
            
        Returns:
            Check results as JSON string
        """
        function_mapping = {
            "travel": "check_impossible_travel",
            "virustotal": "check_virustotal",
            "abuseipdb": "check_abuseipdb",
            "alert": "analyze_security_alert",
            "threat": "assess_threat_level"
        }
        
        if check_type not in function_mapping:
            return f"Unknown check type: {check_type}"
        
        function_name = function_mapping[check_type]
        function = self.function_map[function_name]
        
        try:
            result = function(**kwargs)
            return result
        except Exception as e:
            return f"Error performing {check_type} check: {e}"


def main():
    """Example usage of the Cyber Council."""
    
    # Initialize council
    council = CyberCouncil()
    
    # Example incident
    incident = """
    We detected suspicious activity from user john.doe:
    
    1. Login from Romania (IP: 93.115.92.100) at 08:00 UTC
    2. Login from Japan (IP: 103.79.141.100) at 09:30 UTC (1.5 hours later)
    3. Alert triggered: "Impossible travel detected"
    4. Both IPs flagged by threat intelligence
    5. User accessed sensitive financial data during Japan session
    
    Please analyze this incident and provide recommendations.
    """
    
    # Analyze
    results = council.analyze_incident(incident)
    
    print("\n" + "="*80)
    print("📊 FINAL REPORT")
    print("="*80)
    print(f"\n{results['summary']}\n")


if __name__ == "__main__":
    main()
