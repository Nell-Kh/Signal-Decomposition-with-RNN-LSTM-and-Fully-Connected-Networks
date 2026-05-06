"""Loads project configuration from config/setup.json."""
import json
from pathlib import Path


def load_config():
    """
    Load settings from config/setup.json.
    Finds the project root by locating the config folder
    relative to this file's actual location.
    """
    # This file is at: src/signal_decomp/shared/config.py
    # Project root is 3 levels up
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "config" / "setup.json"

    with open(config_path) as f:
        return json.load(f)
