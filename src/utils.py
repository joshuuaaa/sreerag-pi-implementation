# src/utils.py
"""
Utility functions
"""

import yaml
import os
from pathlib import Path

def load_config(config_path: str = "configs/pi_production.yaml") -> dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return {}
    except Exception as e:
        print(f"❌ Config load error: {e}")
        return {}

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
