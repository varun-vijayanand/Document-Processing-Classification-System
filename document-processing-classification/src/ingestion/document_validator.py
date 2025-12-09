import os
import logging
from typing import Dict, List, Tuple
from PIL import Image
import PyPDF2

logger = logging.getLogger(__name__)

class DocumentValidator:
    """Validator for document files."""
    
    MIN_FILE_SIZE = 100  # bytes
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    SUPPORTED_IMAGE_FORMATS = ['JPEG', 'PNG', 'TIFF', 'BMP', 'GIF']
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_document(self, file_path: str) -> Dict:
        """Validate a single document."""
        results = {
            'file_path': file_path,
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check file exists
        if not os.path.exists(file_path):
            results['is_valid'] = False
            results['issues'].append('File does not exist')
            return results
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size < self.MIN_FILE_SIZE:
            results['issues'].append(f'File too small ({file_size} bytes)')
            results['is_valid'] = False
        
        if file_size > self.MAX_FILE_SIZE:
            results['warnings'].append(f'File very large ({file_size} bytes)')
        
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        if not ext:
            results['warnings'].append('No file extension')
        
        # For images, check if it's readable
        if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']:
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify it's a valid image
                    if img.format not in self.SUPPORTED_IMAGE_FORMATS:
                        results['warnings'].append(f'Uncommon image format: {img.format}')
            except Exception as e:
                results['issues'].append(f'Invalid image: {str(e)}')
                results['is_valid'] = False
        
        # For PDFs, check if it's not corrupted
        elif ext == '.pdf':
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    if len(pdf_reader.pages) == 0:
                        results['warnings'].append('PDF has no pages')
            except Exception as e:
                results['issues'].append(f'Invalid PDF: {str(e)}')
                results['is_valid'] = False
        
        return results
    
    def validate_batch(self, file_paths: List[str]) -> Dict:
        """Validate multiple documents."""
        results = {
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'detailed_results': []
        }
        
        for file_path in file_paths:
            validation = self.validate_document(file_path)
            results['detailed_results'].append(validation)
            
            if validation['is_valid']:
                results['valid_files'] += 1
            else:
                results['invalid_files'] += 1
        
        return results