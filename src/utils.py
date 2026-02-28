# src/utils.py
"""
Utility functions
"""

import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger("utils")

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
        logger.info("Config loaded from %s", config_path)
        return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        return {}
    except Exception as e:
        logger.error("Config load error: %s", e)
        return {}

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
