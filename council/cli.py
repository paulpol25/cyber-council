"""
Command-Line Interface for Cyber Council

Provides an interactive interface to the cybersecurity council.
"""

import argparse
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from council.orchestrator import CyberCouncil


def print_banner():
    """Print the cyber council banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                   🛡️  CYBER COUNCIL 🛡️                        ║
║                                                               ║
║           Multi-Agent AI Security Analysis System            ║
║                                                               ║
║  Powered by AG2 (AutoGen) + MCP Tools                       ║
╚═══════════════════════════════════════════════════════════════╝
""")


def cmd_analyze(args):
    """Handle analyze command."""
    # Council initialization will print model info
    council = CyberCouncil()
    
    if args.file:
        with open(args.file, 'r') as f:
            incident = f.read()
    else:
        incident = args.incident
    
    results = council.analyze_incident(incident, max_rounds=args.max_rounds)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    else:
        print("\n📊 ANALYSIS SUMMARY")
        print("="*80)
        print(results['summary'])


def cmd_check_travel(args):
    """Handle travel check command."""
    council = CyberCouncil()
    
    result = council.quick_check(
        "travel",
        username=args.username,
        source_ip=args.source_ip,
        source_country=args.source_country,
        target_ip=args.target_ip,
        target_country=args.target_country,
        time_diff_hours=args.time_diff
    )
    
    print("\n🌍 TRAVEL ANALYSIS")
    print("="*80)
    print(result)


def cmd_check_ip(args):
    """Handle IP check command."""
    council = CyberCouncil()
    
    print("\n🔍 Checking VirusTotal...")
    vt_result = council.quick_check("virustotal", indicator=args.ip, indicator_type="ip")
    print(vt_result)
    
    print("\n🔍 Checking AbuseIPDB...")
    abuse_result = council.quick_check("abuseipdb", ip_address=args.ip, max_age_days=args.max_age)
    print(abuse_result)


def cmd_check_hash(args):
    """Handle hash check command."""
    council = CyberCouncil()
    
    result = council.quick_check("virustotal", indicator=args.hash, indicator_type="hash")
    print("\n🔍 HASH ANALYSIS")
    print("="*80)
    print(result)


def cmd_check_domain(args):
    """Handle domain check command."""
    council = CyberCouncil()
    
    result = council.quick_check("virustotal", indicator=args.domain, indicator_type="domain")
    print("\n🔍 DOMAIN ANALYSIS")
    print("="*80)
    print(result)


def cmd_interactive(args):
    """Start interactive mode."""
    print_banner()
    
    # Council initialization will print model info
    council = CyberCouncil()
    
    print("\n📝 Interactive Mode - Enter your security incident description")
    print("   Type 'quit' or 'exit' to leave\n")
    
    while True:
        try:
            print("─" * 80)
            incident = input("\n🚨 Describe the incident (or 'quit' to exit):\n> ")
            
            if incident.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if not incident.strip():
                continue
            
            results = council.analyze_incident(incident)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cyber Council - Multi-Agent AI Security Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  OPENAI_API_KEY      - OpenAI API key (required if using OpenAI)
  OPENAI_MODEL        - OpenAI model to use (default: gpt-4o-mini-2024-07-18)
  GOOGLE_API_KEY      - Google API key (required if using Gemini)
  GEMINI_MODEL        - Gemini model to use (default: gemini-pro)
  OLLAMA_BASE_URL     - Ollama base URL (default: http://localhost:11434)
  OLLAMA_MODEL        - Ollama model to use (default: llama2)

Examples:
  # Interactive mode
  python cli.py interactive
  
  # Analyze an incident
  python cli.py analyze "Suspicious login from multiple countries"
  
  # Check impossible travel
  python cli.py check-travel --username john.doe --source-ip 1.2.3.4 \\
      --source-country US --target-ip 5.6.7.8 --target-country CN --time-diff 2
  
  # Check an IP
  python cli.py check-ip 192.0.2.1
  
  # Check a file hash
  python cli.py check-hash 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f
  
  # Check a domain
  python cli.py check-domain evil.com
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive mode
    parser_interactive = subparsers.add_parser('interactive', help='Start interactive mode')
    parser_interactive.set_defaults(func=cmd_interactive)
    
    # Analyze command
    parser_analyze = subparsers.add_parser('analyze', help='Analyze a security incident')
    parser_analyze.add_argument('incident', nargs='?', help='Incident description')
    parser_analyze.add_argument('--file', '-f', help='Read incident from file')
    parser_analyze.add_argument('--output', '-o', help='Save results to file')
    parser_analyze.add_argument('--max-rounds', type=int, default=10, help='Max conversation rounds')
    parser_analyze.set_defaults(func=cmd_analyze)
    
    # Check travel command
    parser_travel = subparsers.add_parser('check-travel', help='Check impossible travel')
    parser_travel.add_argument('--username', required=True, help='Username')
    parser_travel.add_argument('--source-ip', required=True, help='Source IP')
    parser_travel.add_argument('--source-country', required=True, help='Source country code')
    parser_travel.add_argument('--target-ip', required=True, help='Target IP')
    parser_travel.add_argument('--target-country', required=True, help='Target country code')
    parser_travel.add_argument('--time-diff', type=float, required=True, help='Time difference in hours')
    parser_travel.set_defaults(func=cmd_check_travel)
    
    # Check IP command
    parser_ip = subparsers.add_parser('check-ip', help='Check IP reputation')
    parser_ip.add_argument('ip', help='IP address to check')
    parser_ip.add_argument('--max-age', type=int, default=90, help='Max age for reports (days)')
    parser_ip.set_defaults(func=cmd_check_ip)
    
    # Check hash command
    parser_hash = subparsers.add_parser('check-hash', help='Check file hash')
    parser_hash.add_argument('hash', help='File hash to check')
    parser_hash.set_defaults(func=cmd_check_hash)
    
    # Check domain command
    parser_domain = subparsers.add_parser('check-domain', help='Check domain')
    parser_domain.add_argument('domain', help='Domain to check')
    parser_domain.set_defaults(func=cmd_check_domain)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
