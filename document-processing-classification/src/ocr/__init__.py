"""
OCR processing module for extracting text from images and scanned documents.
"""

from .ocr_processor import OCRProcessor
from .pdf_processor import PDFProcessor

__all__ = ["OCRProcessor", "PDFProcessor"]