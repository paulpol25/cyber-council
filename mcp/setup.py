"""
Setup script for Cybersecurity Council MCP Server
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def verify_model():
    """Check if the impossible travel model exists."""
    print("\n🔍 Checking for trained model...")
    
    model_dir = Path(__file__).parent.parent / "models" / "impossible-travel-detector" / "models" / "saved_models"
    
    if model_dir.exists():
        models = list(model_dir.glob("*.keras"))
        if models:
            print(f"✅ Found trained model: {models[-1].name}")
            return True
        else:
            print("⚠️  No trained model found")
            print("   Run this to train the model:")
            print("   cd ..\\models\\impossible-travel-detector")
            print("   python src\\train.py --config configs\\config.yaml")
            return False
    else:
        print("⚠️  Model directory doesn't exist")
        print("   The impossible travel detector will work with basic heuristics")
        print("   For ML-based detection, train the model first")
        return False


def test_imports():
    """Test that all required modules can be imported."""
    print("\n🧪 Testing imports...")
    
    modules = [
        ("mcp", "MCP SDK"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} - not found")
            all_ok = False
    
    return all_ok


def create_config_example():
    """Create example configuration file."""
    print("\n📝 Configuration example created at:")
    config_path = Path(__file__).parent / "claude_config.example.json"
    print(f"   {config_path}")
    print("\nTo use with Claude Desktop:")
    print("1. Copy the configuration from claude_config.example.json")
    print("2. Edit %APPDATA%\\Claude\\claude_desktop_config.json")
    print("3. Add or merge the configuration")
    print("4. Restart Claude Desktop")


def main():
    """Main setup function."""
    print("=" * 60)
    print("CYBERSECURITY COUNCIL MCP SERVER - SETUP")
    print("=" * 60)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at requirements installation")
        return False
    
    # Step 2: Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed. Please install missing packages.")
        print("   Try: pip install -r requirements.txt")
    
    # Step 3: Check model
    verify_model()
    
    # Step 4: Show config info
    create_config_example()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the functions: python examples.py")
    print("2. Configure Claude Desktop (see QUICKSTART.md)")
    print("3. Start using the security analysis tools!")
    print("\n" + "=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
