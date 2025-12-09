import os
import logging
from typing import Dict, List, Optional, Union, BinaryIO
from pathlib import Path
import PyPDF2
from PIL import Image
import pandas as pd

from src.utils.file_utils import FileUtils

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loader for various document formats."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF',
        '.jpg': 'Image',
        '.jpeg': 'Image',
        '.png': 'Image',
        '.tiff': 'Image',
        '.tif': 'Image',
        '.txt': 'Text',
        '.csv': 'CSV',
        '.json': 'JSON'
    }
    
    def __init__(self):
        self.file_utils = FileUtils()
    
    def load_document(self, file_path: str) -> Dict[str, Union[str, bytes]]:
        """Load a document and return metadata and content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(file_ext, 'Unknown')
        
        metadata = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'file_type': file_type,
            'file_size': os.path.getsize(file_path),
            'extension': file_ext
        }
        
        try:
            if file_type == 'PDF':
                content = self._load_pdf(file_path)
            elif file_type == 'Image':
                content = self._load_image(file_path)
            elif file_type == 'Text':
                content = self._load_text(file_path)
            elif file_type == 'CSV':
                content = self._load_csv(file_path)
            elif file_type == 'JSON':
                content = self._load_json(file_path)
            else:
                content = self._load_binary(file_path)
            
            return {
                'metadata': metadata,
                'content': content,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return {
                'metadata': metadata,
                'content': None,
                'success': False,
                'error': str(e)
            }
    
    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def _load_image(self, file_path: str) -> bytes:
        """Load image as binary data."""
        with open(file_path, 'rb') as file:
            return file.read()
    
    def _load_text(self, file_path: str) -> str:
        """Load text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file as DataFrame."""
        return pd.read_csv(file_path)
    
    def _load_json(self, file_path: str) -> Dict:
        """Load JSON file."""
        return self.file_utils.load_json(file_path)
    
    def _load_binary(self, file_path: str) -> bytes:
        """Load any file as binary."""
        with open(file_path, 'rb') as file:
            return file.read()
    
    def load_directory(self, directory: str) -> List[Dict]:
        """Load all supported documents from a directory."""
        documents = []
        files = self.file_utils.get_all_files(
            directory, 
            extensions=list(self.SUPPORTED_EXTENSIONS.keys())
        )
        
        logger.info(f"Found {len(files)} files in {directory}")
        
        for file_path in files:
            document = self.load_document(file_path)
            documents.append(document)
        
        return documents