"""
API Key Management Utility

This module provides utilities for loading API keys from a JSON file.
The api_keys.json file should be in the project root and is gitignored.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict

from .constants import API_KEYS_FILENAME

def load_api_keys(api_keys_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load API keys from a JSON file.
    
    Args:
        api_keys_file: Path to the API keys JSON file. If None, looks for
                      'api_keys.json' in the project root.
    
    Returns:
        Dictionary mapping API key names to their values.
        Empty dict if file doesn't exist or is invalid.
    """
    if api_keys_file is None:
        # Look for api_keys.json in project root
        project_root = Path(__file__).parent.parent.parent
        api_keys_file = project_root / API_KEYS_FILENAME
    else:
        api_keys_file = Path(api_keys_file)
    
    if not api_keys_file.exists():
        return {}
    
    try:
        with open(api_keys_file, "r", encoding="utf-8") as f:
            keys = json.load(f)
            # Filter out empty values
            return {k: v for k, v in keys.items() if v}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load API keys from {api_keys_file}: {e}")
        return {}


def get_api_key_for_mode(mode: str, api_keys_file: Optional[str] = None) -> Optional[str]:
    """
    Get API key for a specific mode from the API keys file or environment.
    
    Priority:
    1. API keys file: api_keys.json file
    2. Environment variable
    
    Args:
        mode: API mode ("gemini", "deepseek", "cgpt", "together", "siliconflow")
        api_keys_file: Path to the API keys JSON file. If None, looks for
                      'api_keys.json' in the project root.
    
    Returns:
        API key string, or None if not found.
    """
    # Map mode to environment variable name
    env_var_map = {
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "cgpt": "OPENAI_API_KEY",
        "together": "TOGETHER_API_KEY",
        "siliconflow": "SILICONFLOW_API_KEY"
    }
    
    env_var = env_var_map.get(mode.lower())
    if not env_var:
        return None
    
    # Try loading from file first
    keys = load_api_keys(api_keys_file)
    if env_var in keys and keys[env_var]:
        return keys[env_var]
    
    # Fall back to environment variable
    return os.getenv(env_var)


def get_api_key(api_key_name: str, api_keys_file: Optional[str] = None) -> Optional[str]:
    """
    Get a specific API key by name from the API keys file or environment.
    
    Priority:
    1. API keys file: api_keys.json file
    2. Environment variable
    
    Args:
        api_key_name: Name of the API key (e.g., "GEMINI_API_KEY")
        api_keys_file: Path to the API keys JSON file. If None, looks for
                      'api_keys.json' in the project root.
    
    Returns:
        API key string, or None if not found.
    """
    # Try loading from file first
    keys = load_api_keys(api_keys_file)
    if api_key_name in keys and keys[api_key_name]:
        return keys[api_key_name]
    
    # Fall back to environment variable
    return os.getenv(api_key_name)
