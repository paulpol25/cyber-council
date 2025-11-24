"""
AG2 (AutoGen) Configuration for Multi-LLM Support

Configures different LLM providers (OpenAI, Gemini, Ollama) for use with AG2 agents.
"""

import os
from typing import Dict, Any, Optional


def get_openai_config(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get OpenAI configuration for AG2.
    
    Args:
        model: OpenAI model to use (default: from env or gpt-4o-mini-2024-07-18)
        
    Returns:
        Configuration dictionary for AG2
    """
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment variables")
    
    # Get model from env or use provided or default
    if model is None:
        model = os.getenv('OPENAI_MODEL', '')
    
    return {
        "config_list": [{
            "model": model,
            "api_key": api_key,
        }],
    }


def get_gemini_config(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Google Gemini configuration for AG2.
    
    Args:
        model: Gemini model to use (default: from env or gemini-pro)
        
    Returns:
        Configuration dictionary for AG2
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment variables")
    
    # Get model from env or use provided or default
    if model is None:
        model = os.getenv('GEMINI_MODEL', 'gemini-pro')
    
    return {
        "model": model,
        "api_key": api_key,
        "api_type": "google",
        "temperature": 0.7,
        "max_tokens": 2000,
    }


def get_ollama_config(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Ollama (local LLM) configuration for AG2.
    
    Args:
        model: Ollama model to use (default: from env or llama2)
        
    Returns:
        Configuration dictionary for AG2
    """
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    model = model or os.getenv('OLLAMA_MODEL', 'llama2')
    
    return {
        "model": model,
        "base_url": f"{base_url}/v1",
        "api_key": "ollama",  # Ollama doesn't need real API key
        "temperature": 0.7,
        "max_tokens": 2000,
    }


def get_llm_config(
    provider: str = "openai",
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get LLM configuration based on provider.
    
    Args:
        provider: LLM provider (openai, gemini, ollama)
        model: Specific model to use (optional, will use env default if not provided)
        
    Returns:
        Configuration dictionary for AG2
    """
    if provider.lower() == "openai":
        return get_openai_config(model)
    elif provider.lower() == "gemini":
        return get_gemini_config(model)
    elif provider.lower() == "ollama":
        return get_ollama_config(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_council_llm_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get LLM configurations for the cyber council.
    Returns a mix of different providers if available.
    
    Returns:
        Dictionary mapping agent names to LLM configs
    """
    configs = {}
    
    # Try to use different providers for diversity
    providers = []
    
    if os.getenv('OPENAI_API_KEY'):
        # Get OpenAI model from env or use default
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini-2024-07-18')
        providers.append(('openai', openai_model))
    
    if os.getenv('GOOGLE_API_KEY'):
        # Get Gemini model from env or use default
        gemini_model = os.getenv('GEMINI_MODEL', 'gemini-pro')
        providers.append(('gemini', gemini_model))
    
    # Check if Ollama is running
    try:
        import requests
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            providers.append(('ollama', os.getenv('OLLAMA_MODEL', 'llama2')))
    except:
        pass
    
    if not providers:
        raise ValueError(
            "No LLM providers configured. Please set at least one of: "
            "OPENAI_API_KEY, GOOGLE_API_KEY, or run Ollama locally"
        )
    
    # Print configured providers and models
    print("🤖 Configured LLM Providers:")
    for provider, model in providers:
        print(f"  • {provider}: {model}")
    print()
    
    # Assign providers to agents (cycle through available providers)
    agent_roles = [
        "threat_analyst",
        "travel_detective", 
        "alert_triager",
        "threat_intel_specialist",
        "incident_commander"
    ]
    
    for i, role in enumerate(agent_roles):
        provider, model = providers[i % len(providers)]
        configs[role] = get_llm_config(provider, model)
    
    return configs


# Default configurations
DEFAULT_CONFIG = {
    "timeout": 600,  # 10 minutes
    "cache_seed": None,  # Disable caching for more diverse responses
}
