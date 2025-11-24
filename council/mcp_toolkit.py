"""
MCP Toolkit Adapter for AG2

Creates AG2-compatible tools from MCP functions.
"""

from typing import Dict, Any, Callable
import sys
import os
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from council.mcp_adapter import MCPToolAdapter


def create_mcp_toolkit(mcp_adapter: MCPToolAdapter) -> Dict[str, Callable]:
    """
    Create AG2-compatible toolkit from MCP adapter.
    
    Converts bound methods to standalone functions that AG2 can register.
    
    Args:
        mcp_adapter: MCPToolAdapter instance
        
    Returns:
        Dictionary of tool functions ready for AG2
    """
    # Get the raw function map (these are bound methods)
    function_map = mcp_adapter.get_function_map()
    
    # Convert bound methods to standalone functions
    toolkit = {}
    
    for name, method in function_map.items():
        # Get the underlying function and instance
        func = method.__func__
        instance = method.__self__
        
        # Get the original signature and remove 'self' parameter
        sig = inspect.signature(func)
        params = list(sig.parameters.values())[1:]  # Skip 'self'
        new_sig = sig.replace(parameters=params)
        
        # Create standalone function with proper signature
        def make_standalone(f, inst):
            def standalone_func(*args, **kwargs):
                return f(inst, *args, **kwargs)
            return standalone_func
        
        standalone = make_standalone(func, instance)
        standalone.__name__ = func.__name__
        standalone.__doc__ = func.__doc__
        standalone.__signature__ = new_sig
        standalone.__annotations__ = {k: v for k, v in func.__annotations__.items() if k != 'self'}
        
        toolkit[name] = standalone
    
    return toolkit
