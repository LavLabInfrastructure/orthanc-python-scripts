"""Defines Config class to read the config file and return the values"""
from typing import Any
import yaml

class Config:
    """Class to read the config file and return the values"""
    def __init__(self, config_file: str = 'config.yaml'):
        with open(config_file, 'r', encoding='utf-8') as file:
            self.config: dict = yaml.safe_load(file)

    def get(self, key: str) -> Any:
        """Return the value for the given key."""
        return self.config.get(key)

    def get_orthanc_config(self) -> dict[str, Any]:
        """Return the Orthanc configuration."""
        return self.config.get('orthanc')

    def get_sheets(self) -> list[dict[str, Any]]:
        """Return the list of sheets."""
        return self.config.get('sheets')

    def get_microsoft_config(self) -> dict[str, Any]:
        """Return the Microsoft configuration."""
        return self.config.get('microsoft')
