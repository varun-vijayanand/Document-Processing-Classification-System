import os
import json
import yaml
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import magic
from datetime import datetime

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine file type using python-magic."""
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            return file_type
        except Exception as e:
            logger.warning(f"Could not determine file type for {file_path}: {e}")
            return "unknown"
    
    @staticmethod
    def ensure_dir(directory: str) -> None:
        """Create directory if it doesn't exist."""
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_all_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """Get all files in directory with optional extension filter."""
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                if extensions:
                    if any(file.lower().endswith(ext.lower()) for ext in extensions):
                        file_list.append(os.path.join(root, file))
                else:
                    file_list.append(os.path.join(root, file))
        return file_list
    
    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> None:
        """Save data as JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    
    @staticmethod
    def load_json(file_path: str) -> Any:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, file_path: str) -> None:
        """Save data as pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """Load data from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")