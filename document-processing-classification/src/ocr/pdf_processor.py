import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
from pdf2image import convert_from_path
import pytesseract

from src.ocr.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processor for PDF documents with OCR support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.ocr_processor = OCRProcessor(config_path)
        self.config = self.ocr_processor.config
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = False) -> Dict:
        """Extract text from PDF file."""
        try:
            if use_ocr:
                return self._extract_with_ocr(pdf_path)
            else:
                return self._extract_without_ocr(pdf_path)
        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {e}")
            return {
                'success': False,
                'text': '',
                'pages': 0,
                'method': 'none',
                'error': str(e)
            }
    
    def _extract_without_ocr(self, pdf_path: str) -> Dict:
        """Extract text from searchable PDF."""
        try:
            import PyPDF2
            text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n\n"
            
            return {
                'success': True,
                'text': text.strip(),
                'pages': num_pages,
                'method': 'direct_extraction',
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
        except Exception as e:
            logger.warning(f"Direct extraction failed, falling back to OCR: {e}")
            return self._extract_with_ocr(pdf_path)
    
    def _extract_with_ocr(self, pdf_path: str) -> Dict:
        """Extract text using OCR on PDF pages."""
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=300,
                fmt='jpeg',
                thread_count=2
            )
            
            text = ""
            page_results = []
            
            for i, image in enumerate(images):
                # Save temporary image
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    image.save(tmp.name, 'JPEG')
                    tmp_path = tmp.name
                
                # OCR the image
                result = self.ocr_processor.extract_text(tmp_path)
                
                # Clean up
                os.unlink(tmp_path)
                
                if result['success']:
                    text += result['text'] + "\n\n"
                    page_results.append({
                        'page': i + 1,
                        'success': True,
                        'confidence': result['confidence'],
                        'word_count': result['word_count']
                    })
                else:
                    page_results.append({
                        'page': i + 1,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
            
            # Calculate overall confidence
            successful_pages = [p for p in page_results if p['success']]
            avg_confidence = (
                sum(p['confidence'] for p in successful_pages) / len(successful_pages)
                if successful_pages else 0.0
            )
            
            return {
                'success': True,
                'text': text.strip(),
                'pages': len(images),
                'method': 'ocr',
                'avg_confidence': avg_confidence,
                'page_results': page_results,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return {
                'success': False,
                'text': '',
                'pages': 0,
                'method': 'ocr',
                'error': str(e)
            }
    
    def batch_process_pdfs(self, pdf_paths: List[str], use_ocr: bool = False) -> List[Dict]:
        """Process multiple PDF files."""
        results = []
        for pdf_path in pdf_paths:
            result = self.extract_text_from_pdf(pdf_path, use_ocr)
            result['file_path'] = pdf_path
            results.append(result)
        return results