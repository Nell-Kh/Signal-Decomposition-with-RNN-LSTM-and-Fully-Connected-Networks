"""Loads project configuration from config/setup.json."""
import json
from pathlib import Path


def load_config():
    """
    Load settings from config/setup.json.
    Uses pathlib to find the project root reliably
    regardless of where the script is called from.
    """
    project_root = Path(__file__).resolve().parents[4]
    config_path = project_root / "config" / "setup.json"

    with open(config_path, "r") as f:
        return json.load(f)