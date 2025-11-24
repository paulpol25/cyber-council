#!/usr/bin/env python3
"""
Cybersecurity Council MCP Server

This MCP server provides cybersecurity analysis tools for LLMs, including:
- Impossible travel detection
- Alert analysis and threat assessment
- Security event correlation
- Threat intelligence lookups
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

# Add parent directory to path to import from models
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Import our security tools (use relative imports since we're in the mcp directory)
from functions.impossible_travel import ImpossibleTravelChecker
from functions.alert_analyzer import AlertAnalyzer
from functions.virustotal_check import VirusTotalChecker
from functions.abuseipdb_check import AbuseIPDBChecker


class CybersecCouncilServer:
    """MCP Server for cybersecurity analysis tools."""

    def __init__(self):
        self.server = Server("cybersec-council")
        self.impossible_travel_checker: Optional[ImpossibleTravelChecker] = None
        self.alert_analyzer: Optional[AlertAnalyzer] = None
        self.virustotal_checker: Optional[VirusTotalChecker] = None
        self.abuseipdb_checker: Optional[AbuseIPDBChecker] = None
        
        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP protocol handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available security analysis tools."""
            return [
                Tool(
                    name="check_impossible_travel",
                    description=(
                        "Analyzes a login event to detect impossible travel patterns. "
                        "Checks if the user could physically travel between two locations "
                        "in the given time frame. Returns risk score, classification, and reasoning."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "Username or user ID of the account"
                            },
                            "current_country": {
                                "type": "string",
                                "description": "Current login country"
                            },
                            "previous_country": {
                                "type": "string",
                                "description": "Previous login country"
                            },
                            "time_diff_hours": {
                                "type": "number",
                                "description": "Time difference between logins in hours"
                            },
                            "current_city": {
                                "type": "string",
                                "description": "Current login city (optional)"
                            },
                            "previous_city": {
                                "type": "string",
                                "description": "Previous login city (optional)"
                            }
                        },
                        "required": ["username", "current_country", "previous_country", "time_diff_hours"]
                    }
                ),
                Tool(
                    name="analyze_security_alert",
                    description=(
                        "Analyzes a security alert or event to assess threat level. "
                        "Provides risk assessment, IOCs (Indicators of Compromise), "
                        "recommended actions, and confidence level. Supports various alert types "
                        "including authentication anomalies, malware detections, network intrusions, etc."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "alert_type": {
                                "type": "string",
                                "description": "Type of alert (e.g., 'authentication_anomaly', 'malware_detection', 'network_intrusion', 'data_exfiltration')"
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the security event or alert"
                            },
                            "source_ip": {
                                "type": "string",
                                "description": "Source IP address (if applicable)"
                            },
                            "destination_ip": {
                                "type": "string",
                                "description": "Destination IP address (if applicable)"
                            },
                            "username": {
                                "type": "string",
                                "description": "Associated username (if applicable)"
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "When the event occurred (ISO 8601 format)"
                            },
                            "additional_context": {
                                "type": "object",
                                "description": "Any additional context or metadata about the alert"
                            }
                        },
                        "required": ["alert_type", "description"]
                    }
                ),
                Tool(
                    name="assess_threat_level",
                    description=(
                        "Provides a comprehensive threat assessment by analyzing multiple signals "
                        "and indicators. Combines various detection methods to give an overall "
                        "risk score and recommended response level."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "indicators": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Indicator type (e.g., 'ip', 'domain', 'file_hash', 'behavior')"
                                        },
                                        "value": {
                                            "type": "string",
                                            "description": "The indicator value"
                                        },
                                        "severity": {
                                            "type": "string",
                                            "enum": ["low", "medium", "high", "critical"],
                                            "description": "Individual indicator severity"
                                        }
                                    },
                                    "required": ["type", "value", "severity"]
                                },
                                "description": "List of security indicators to assess"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context about the situation"
                            }
                        },
                        "required": ["indicators"]
                    }
                ),
                Tool(
                    name="check_virustotal",
                    description=(
                        "Checks IPs, domains, file hashes, or URLs against VirusTotal's threat intelligence database. "
                        "Returns detection rates from multiple security vendors, threat classification, and recommendations."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "indicator": {
                                "type": "string",
                                "description": "The indicator to check (IP address, domain, file hash, or URL)"
                            },
                            "indicator_type": {
                                "type": "string",
                                "enum": ["ip", "domain", "hash", "url"],
                                "description": "Type of indicator being checked"
                            }
                        },
                        "required": ["indicator", "indicator_type"]
                    }
                ),
                Tool(
                    name="check_abuseipdb",
                    description=(
                        "Checks IP addresses against AbuseIPDB for abuse reports and malicious activity. "
                        "Returns abuse confidence score, number of reports, report categories, and risk assessment."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ip_address": {
                                "type": "string",
                                "description": "The IP address to check"
                            },
                            "max_age_days": {
                                "type": "number",
                                "description": "Maximum age of reports to consider in days (default: 90)"
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Include detailed report information (default: true)"
                            }
                        },
                        "required": ["ip_address"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle tool execution requests."""
            
            if name == "check_impossible_travel":
                return await self._check_impossible_travel(arguments)
            elif name == "analyze_security_alert":
                return await self._analyze_security_alert(arguments)
            elif name == "assess_threat_level":
                return await self._assess_threat_level(arguments)
            elif name == "check_virustotal":
                return await self._check_virustotal(arguments)
            elif name == "check_abuseipdb":
                return await self._check_abuseipdb(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _check_impossible_travel(self, args: dict) -> list[TextContent]:
        """Check for impossible travel patterns."""
        try:
            # Initialize checker if needed
            if self.impossible_travel_checker is None:
                self.impossible_travel_checker = ImpossibleTravelChecker()
            
            # Perform the check
            result = self.impossible_travel_checker.check(
                username=args["username"],
                current_country=args["current_country"],
                previous_country=args["previous_country"],
                time_diff_hours=args["time_diff_hours"],
                current_city=args.get("current_city"),
                previous_city=args.get("previous_city")
            )
            
            # Format response
            response = {
                "classification": result["classification"],
                "risk_score": result["risk_score"],
                "confidence": result["confidence"],
                "distance_km": result["distance_km"],
                "required_speed_kmh": result["required_speed_kmh"],
                "reasoning": result["reasoning"],
                "recommendation": result["recommendation"]
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    async def _analyze_security_alert(self, args: dict) -> list[TextContent]:
        """Analyze a security alert."""
        try:
            # Initialize analyzer if needed
            if self.alert_analyzer is None:
                self.alert_analyzer = AlertAnalyzer()
            
            # Perform analysis
            result = self.alert_analyzer.analyze(
                alert_type=args["alert_type"],
                description=args["description"],
                source_ip=args.get("source_ip"),
                destination_ip=args.get("destination_ip"),
                username=args.get("username"),
                timestamp=args.get("timestamp"),
                additional_context=args.get("additional_context", {})
            )
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    async def _assess_threat_level(self, args: dict) -> list[TextContent]:
        """Assess overall threat level from multiple indicators."""
        try:
            # Initialize analyzer if needed
            if self.alert_analyzer is None:
                self.alert_analyzer = AlertAnalyzer()
            
            # Assess threat
            result = self.alert_analyzer.assess_threat(
                indicators=args["indicators"],
                context=args.get("context", "")
            )
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    async def _check_virustotal(self, args: dict) -> list[TextContent]:
        """Check indicator against VirusTotal."""
        try:
            # Initialize checker if needed
            if self.virustotal_checker is None:
                self.virustotal_checker = VirusTotalChecker()
            
            # Perform check
            result = self.virustotal_checker.check(
                indicator=args["indicator"],
                indicator_type=args["indicator_type"]
            )
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    async def _check_abuseipdb(self, args: dict) -> list[TextContent]:
        """Check IP against AbuseIPDB."""
        try:
            # Initialize checker if needed
            if self.abuseipdb_checker is None:
                self.abuseipdb_checker = AbuseIPDBChecker()
            
            # Perform check
            result = self.abuseipdb_checker.check(
                ip_address=args["ip_address"],
                max_age_days=args.get("max_age_days", 90),
                verbose=args.get("verbose", True)
            )
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = CybersecCouncilServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
