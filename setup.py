"""
Setup and Configuration Script for Cyber Council
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def check_python_version():
    """Check if Python version is 3.10+."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    
    print("✅ Python version OK")
    return True


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")
    
    requirements_files = [
        "mcp/requirements.txt",
        "council/requirements.txt"
    ]
    
    for req_file in requirements_files:
        if not Path(req_file).exists():
            print(f"⚠️  {req_file} not found, skipping")
            continue
        
        print(f"📦 Installing from {req_file}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", req_file
            ])
            print(f"✅ Installed {req_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {req_file}: {e}")
            return False
    
    return True


def setup_environment():
    """Setup .env file."""
    print_header("Setting Up Environment")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠️  .env already exists")
        response = input("Overwrite? (y/N): ").lower()
        if response != 'y':
            print("Keeping existing .env")
            return True
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy example to .env
    with open(env_example, 'r') as src:
        content = src.read()
    
    with open(env_file, 'w') as dst:
        dst.write(content)
    
    print("✅ Created .env from .env.example")
    print("\n📝 Please edit .env and add your API keys:")
    print("   - OPENAI_API_KEY (recommended)")
    print("   - GOOGLE_API_KEY (optional)")
    print("   - VIRUSTOTAL_API_KEY (recommended)")
    print("   - ABUSEIPDB_API_KEY (recommended)")
    
    return True


def check_api_keys():
    """Check if API keys are configured."""
    print_header("Checking API Keys")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Google Gemini": os.getenv("GOOGLE_API_KEY"),
        "VirusTotal": os.getenv("VIRUSTOTAL_API_KEY"),
        "AbuseIPDB": os.getenv("ABUSEIPDB_API_KEY"),
    }
    
    configured = []
    missing = []
    
    for name, value in keys.items():
        if value and not value.startswith("your_") and not value.startswith("sk-your"):
            configured.append(name)
            print(f"✅ {name} configured")
        else:
            missing.append(name)
            print(f"⚠️  {name} not configured")
    
    print(f"\nConfigured: {len(configured)}/4 API keys")
    
    if not configured:
        print("\n⚠️  No API keys configured!")
        print("At least one LLM provider (OpenAI or Gemini) is required")
        print("Or install Ollama for local LLM: https://ollama.ai")
        return False
    
    if "OpenAI" not in configured and "Google Gemini" not in configured:
        print("\n⚠️  No LLM provider configured")
        print("Checking for Ollama...")
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✅ Ollama is running locally")
                return True
        except:
            pass
        
        print("❌ No LLM provider available")
        print("Please configure OpenAI, Gemini, or install Ollama")
        return False
    
    return True


def verify_imports():
    """Verify all required imports."""
    print_header("Verifying Imports")
    
    required_modules = [
        ("autogen", "pyautogen"),
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("tensorflow", "tensorflow"),
        ("sklearn", "scikit-learn"),
    ]
    
    all_ok = True
    
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} not installed")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick test of the system."""
    print_header("Running Quick Test")
    
    try:
        print("Importing council modules...")
        from council.mcp_adapter import MCPToolAdapter
        
        print("Initializing MCP tools...")
        adapter = MCPToolAdapter()
        
        print("Testing travel detection...")
        result = adapter.check_impossible_travel(
            username="test_user",
            source_ip="1.2.3.4",
            source_country="US",
            target_ip="5.6.7.8",
            target_country="CN",
            time_diff_hours=2.0
        )
        
        print("✅ Travel detection test passed")
        
        # Test function map
        function_map = adapter.get_function_map()
        print(f"✅ Loaded {len(function_map)} MCP tools")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete!")
    
    print("""
🎉 Cyber Council is ready to use!

Next Steps:
-----------

1. Configure API Keys (if not done):
   Edit .env and add your API keys

2. Try Interactive Mode:
   python council/cli.py interactive

3. Run Example Scenarios:
   python council/quickstart.py

4. Quick Checks:
   python council/cli.py check-ip 192.0.2.1
   python council/cli.py check-hash <file_hash>

5. Analyze Incidents:
   python council/cli.py analyze "Suspicious login from multiple countries"

Documentation:
--------------
- Council: council/README.md
- MCP Server: mcp/README.md
- ML Model: models/impossible-travel-detector/README.md

For help:
   python council/cli.py --help
""")


def main():
    """Main setup script."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              Cyber Council - Setup Script                    ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Run setup steps
    steps = [
        ("Check Python Version", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Setup Environment", setup_environment),
        ("Check API Keys", check_api_keys),
        ("Verify Imports", verify_imports),
        ("Run Quick Test", run_quick_test),
    ]
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                print(f"\n⚠️  Setup incomplete: {step_name} failed")
                print("Please resolve the issues above and run setup again")
                return False
        except Exception as e:
            print(f"\n❌ Error in {step_name}: {e}")
            return False
    
    print_next_steps()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
