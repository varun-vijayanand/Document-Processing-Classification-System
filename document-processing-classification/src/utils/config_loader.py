import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

class ConfigLoader:
    """Configuration loader for the application."""
    
    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        self.config_path = config_path
        self.env_path = env_path
        self.config = self._load_config()
        self._load_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            "paths": {
                "raw_data": "data/raw",
                "processed_data": "data/processed",
                "ocr_output": "data/ocr",
                "embeddings": "data/embeddings",
                "models": "models"
            },
            "ocr": {
                "language": "eng",
                "config": "--oem 3 --psm 6",
                "timeout": 30
            },
            "text_processing": {
                "min_text_length": 50,
                "remove_stopwords": True,
                "lemmatize": True
            },
            "embeddings": {
                "model_name": "all-MiniLM-L6-v2",
                "batch_size": 32,
                "device": "cpu"
            },
            "models": {
                "test_size": 0.2,
                "random_state": 42,
                "n_folds": 5
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with default config
                default_config.update(loaded_config)
        
        return default_config
    
    def _load_env(self):
        """Load environment variables."""
        if os.path.exists(self.env_path):
            load_dotenv(self.env_path)
        
        # Set Tesseract path from environment
        tesseract_path = os.getenv("TESSERACT_PATH")
        if tesseract_path:
            self.config["ocr"]["tesseract_path"] = tesseract_path
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value