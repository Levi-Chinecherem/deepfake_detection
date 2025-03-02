# src/test_config.py

import argparse
import yaml
import os

def load_config(config_path):
    """
    Load the YAML configuration file.
    
    Args:
        config_path (str): Path to the config.yaml file.
    
    Returns:
        dict: Configuration settings.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

def main():
    # Parse command-line argument for config file path
    parser = argparse.ArgumentParser(description="Test loading the config file.")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to the config YAML file")
    args = parser.parse_args()

    # Load and print the config
    config = load_config(args.config)
    print("Configuration loaded successfully!")
    print("Settings:")
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()