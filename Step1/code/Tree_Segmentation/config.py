# config.py
import yaml
import os

def load_config(config_path=None):
    # Default to the config file that lives next to this module.
    default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    # If no config path is provided, use the default one
    if config_path is None:
        config_path = default_config_path

    # Check existence of config file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load YAML config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


