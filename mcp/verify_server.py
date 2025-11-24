"""
Verify MCP Server Configuration

This script checks that the MCP server is properly configured and ready to use.
It does NOT run the server (which requires an MCP client), but verifies all
components are available.
"""

import sys
from pathlib import Path

print("=" * 60)
print("MCP SERVER CONFIGURATION VERIFICATION")
print("=" * 60)

# Check imports
print("\n1. Checking imports...")
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    print("   ✓ MCP SDK imports successful")
except ImportError as e:
    print(f"   ✗ MCP SDK import failed: {e}")
    print("   Run: pip install mcp")
    sys.exit(1)

try:
    from functions.impossible_travel import ImpossibleTravelChecker
    from functions.alert_analyzer import AlertAnalyzer
    print("   ✓ Function modules import successful")
except ImportError as e:
    print(f"   ✗ Function import failed: {e}")
    sys.exit(1)

# Check if ML model is available
print("\n2. Checking ML model availability...")
try:
    checker = ImpossibleTravelChecker()
    if checker.use_ml:
        print("   ✓ ML model loaded successfully")
    else:
        print("   ⚠ ML model not available (using heuristic fallback)")
except Exception as e:
    print(f"   ✗ Error initializing checker: {e}")

# Check tools can be instantiated
print("\n3. Checking tool initialization...")
try:
    analyzer = AlertAnalyzer()
    print("   ✓ Alert analyzer initialized")
except Exception as e:
    print(f"   ✗ Error initializing analyzer: {e}")
    sys.exit(1)

# Check server can be created
print("\n4. Checking server creation...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from server import CybersecCouncilServer
    server = CybersecCouncilServer()
    print("   ✓ MCP server created successfully")
except Exception as e:
    print(f"   ✗ Error creating server: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ SERVER CONFIGURATION VERIFIED!")
print("=" * 60)

print("\nThe MCP server is properly configured.")
print("\nNOTE: The error you saw is NORMAL when running the server")
print("directly from the command line. The server expects to be")
print("run by an MCP client (like Claude Desktop) which will")
print("communicate via the MCP protocol.")

print("\n" + "-" * 60)
print("HOW TO USE THE SERVER:")
print("-" * 60)

print("\n1. Test functions directly:")
print("   python test_impossible_travel.py")
print("   python examples.py")

print("\n2. Configure Claude Desktop:")
print("   Edit: %APPDATA%\\Claude\\claude_desktop_config.json")
print("   Add the configuration from claude_config.example.json")
print("   Restart Claude Desktop")

print("\n3. The server will run automatically when Claude needs it")
print("   and will communicate using the MCP protocol.")

print("\n" + "=" * 60)
