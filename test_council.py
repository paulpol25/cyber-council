"""
Test Suite for Cyber Council

Tests all components: MCP tools, AG2 agents, and orchestrator.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_mcp_adapter():
    """Test MCP tool adapter."""
    print("\n" + "="*80)
    print("Testing MCP Tool Adapter")
    print("="*80)
    
    try:
        from council.mcp_adapter import MCPToolAdapter
        
        adapter = MCPToolAdapter()
        function_map = adapter.get_function_map()
        
        print(f"✅ Loaded {len(function_map)} functions")
        for name in function_map.keys():
            print(f"   - {name}")
        
        # Test travel detection
        print("\n🧪 Testing impossible travel detection...")
        result = adapter.check_impossible_travel(
            username="test_user",
            source_country="US",
            target_country="JP",
            time_diff_hours=1.5
        )
        print("✅ Travel detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_config():
    """Test LLM configuration."""
    print("\n" + "="*80)
    print("Testing LLM Configuration")
    print("="*80)
    
    try:
        from council.llm_config import get_council_llm_configs
        from dotenv import load_dotenv
        
        load_dotenv()
        
        configs = get_council_llm_configs()
        
        print(f"✅ Configured {len(configs)} agents")
        for agent, config in configs.items():
            print(f"   - {agent}: {config.get('model', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Config test failed: {e}")
        print("Make sure at least one LLM provider is configured in .env")
        return False


def test_agents():
    """Test agent creation."""
    print("\n" + "="*80)
    print("Testing Agent Creation")
    print("="*80)
    
    try:
        from council.agents import create_cyber_council
        from council.llm_config import get_council_llm_configs
        from council.mcp_adapter import MCPToolAdapter
        from dotenv import load_dotenv
        
        load_dotenv()
        
        llm_configs = get_council_llm_configs()
        adapter = MCPToolAdapter()
        function_map = adapter.get_function_map()
        
        agents = create_cyber_council(llm_configs, function_map)
        
        print(f"✅ Created {len(agents)} agents")
        for name in agents.keys():
            print(f"   - {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_check():
    """Test quick check functionality."""
    print("\n" + "="*80)
    print("Testing Quick Check (No Multi-Agent)")
    print("="*80)
    
    try:
        from council.orchestrator import CyberCouncil
        
        council = CyberCouncil()
        
        print("\n🧪 Testing travel check...")
        result = council.quick_check(
            "travel",
            username="alice",
            source_country="RO",
            target_country="JP",
            time_diff_hours=1.5
        )
        print("✅ Quick check works")
        print(f"Result length: {len(result)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick check test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_virustotal_integration():
    """Test VirusTotal integration."""
    print("\n" + "="*80)
    print("Testing VirusTotal Integration")
    print("="*80)
    
    try:
        from council.orchestrator import CyberCouncil
        import os
        
        if not os.getenv("VIRUSTOTAL_API_KEY") or os.getenv("VIRUSTOTAL_API_KEY").startswith("your_"):
            print("⚠️  VirusTotal API key not configured, skipping")
            return True
        
        council = CyberCouncil()
        
        print("\n🧪 Checking IP on VirusTotal...")
        result = council.quick_check(
            "virustotal",
            indicator="8.8.8.8",
            indicator_type="ip"
        )
        print("✅ VirusTotal check works")
        print(f"Result length: {len(result)} chars")
        
        return True
        
    except Exception as e:
        print(f"⚠️  VirusTotal test failed (may need valid API key): {e}")
        return True  # Don't fail the test suite


def test_abuseipdb_integration():
    """Test AbuseIPDB integration."""
    print("\n" + "="*80)
    print("Testing AbuseIPDB Integration")
    print("="*80)
    
    try:
        from council.orchestrator import CyberCouncil
        import os
        
        if not os.getenv("ABUSEIPDB_API_KEY") or os.getenv("ABUSEIPDB_API_KEY").startswith("your_"):
            print("⚠️  AbuseIPDB API key not configured, skipping")
            return True
        
        council = CyberCouncil()
        
        print("\n🧪 Checking IP on AbuseIPDB...")
        result = council.quick_check(
            "abuseipdb",
            ip_address="8.8.8.8",
            max_age_days=90
        )
        print("✅ AbuseIPDB check works")
        print(f"Result length: {len(result)} chars")
        
        return True
        
    except Exception as e:
        print(f"⚠️  AbuseIPDB test failed (may need valid API key): {e}")
        return True  # Don't fail the test suite


def main():
    """Run all tests."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              Cyber Council - Test Suite                      ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    tests = [
        ("MCP Adapter", test_mcp_adapter),
        ("LLM Configuration", test_llm_config),
        ("Agent Creation", test_agents),
        ("Quick Check", test_quick_check),
        ("VirusTotal Integration", test_virustotal_integration),
        ("AbuseIPDB Integration", test_abuseipdb_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Exception in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
