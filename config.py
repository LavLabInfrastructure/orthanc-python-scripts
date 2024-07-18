import yaml
from typing import Any

class Config:
    def __init__(self, config_file: str = 'config.yaml'):
        with open(config_file, 'r') as file:
            self.config: dict = yaml.safe_load(file)
    
    def get(self, key: str) -> Any:
        return self.config.get(key)
    
    def get_orthanc_config(self) -> dict[str, Any]:
        return self.config.get('orthanc')
    
    def get_sheets(self) -> list[dict[str, Any]]:
        return self.config.get('sheets')
    
    def get_microsoft_config(self) -> dict[str, Any]:
        return self.config.get('microsoft')