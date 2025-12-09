import os
import logging
from typing import Optional, List, Dict
import pytesseract
from PIL import Image
import cv2
import numpy as np

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR processor using Tesseract."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigLoader(config_path) if config_path else ConfigLoader()
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Setup Tesseract path from configuration."""
        tesseract_path = self.config.get("ocr.tesseract_path")
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Tesseract path set to: {tesseract_path}")
        else:
            logger.warning("Tesseract path not configured, using default")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        return denoised
    
    def extract_text(self, image_path: str, lang: Optional[str] = None) -> Dict:
        """Extract text from image file."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # OCR parameters
            lang = lang or self.config.get("ocr.language", "eng")
            config = self.config.get("ocr.config", "--oem 3 --psm 6")
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang=lang,
                config=config,
                timeout=self.config.get("ocr.timeout", 30)
            )
            
            # Get additional information
            data = pytesseract.image_to_data(
                processed_image,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            return {
                'success': True,
                'text': text.strip(),
                'confidence': np.mean([float(c) for c in data['conf'] if float(c) > 0]),
                'image_path': image_path,
                'language': lang,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'image_path': image_path,
                'error': str(e)
            }
    
    def extract_text_from_bytes(self, image_bytes: bytes, lang: Optional[str] = None) -> Dict:
        """Extract text from image bytes."""
        try:
            # Convert bytes to image
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image from bytes")
            
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # OCR parameters
            lang = lang or self.config.get("ocr.language", "eng")
            config = self.config.get("ocr.config", "--oem 3 --psm 6")
            
            # Perform OCR
            text = pytesseract.image_to_string(
                pil_image,
                lang=lang,
                config=config,
                timeout=self.config.get("ocr.timeout", 30)
            )
            
            return {
                'success': True,
                'text': text.strip(),
                'confidence': 90.0,  # Approximate confidence
                'language': lang,
                'word_count': len(text.split())
            }
            
        except Exception as e:
            logger.error(f"OCR from bytes failed: {e}")
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def batch_process(self, image_paths: List[str], lang: Optional[str] = None) -> List[Dict]:
        """Process multiple images."""
        results = []
        for image_path in image_paths:
            result = self.extract_text(image_path, lang)
            results.append(result)
        return results