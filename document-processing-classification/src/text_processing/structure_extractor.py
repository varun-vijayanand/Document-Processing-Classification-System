import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import dateutil.parser as dparser
from datetime import datetime

logger = logging.getLogger(__name__)

class StructureExtractor:
    """Extract structured information from text."""
    
    def __init__(self):
        # Common patterns for document structure
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[-\.\s]??)?\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}',
            'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+\s+\d{1,2},?\s+\d{4})\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'amount': r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?\%',
        }
        
        # Common document sections
        self.common_sections = [
            'abstract', 'introduction', 'methodology', 'results',
            'discussion', 'conclusion', 'references', 'appendix',
            'executive summary', 'background', 'analysis', 'recommendations'
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract various entities from text."""
        entities = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract and parse dates from text."""
        dates = []
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',   # MM/DD/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b',   # DD-MM-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    date_str = match.group()
                    parsed_date = dparser.parse(date_str, fuzzy=True)
                    dates.append({
                        'text': date_str,
                        'date': parsed_date.isoformat(),
                        'position': match.start(),
                        'format': pattern
                    })
                except:
                    continue
        
        return dates
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract document sections."""
        sections = {}
        lines = text.split('\n')
        
        current_section = 'header'
        section_content = []
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Check if line is a section header
            is_section_header = False
            for section in self.common_sections:
                if section in line_lower and len(line_lower.split()) < 10:
                    is_section_header = True
                    break
            
            if is_section_header and section_content:
                # Save previous section
                sections[current_section] = '\n'.join(section_content)
                current_section = line_lower
                section_content = []
            else:
                section_content.append(line)
        
        # Add the last section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tabular data from text (simple heuristic)."""
        tables = []
        lines = text.split('\n')
        
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            # Simple heuristic: lines with consistent delimiter patterns
            if re.search(r'(\t|\|){2,}', line) or re.search(r'\s{2,}.+\s{2,}', line):
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            else:
                if in_table and len(current_table) >= 2:  # At least header + one row
                    tables.append({
                        'rows': len(current_table),
                        'content': '\n'.join(current_table),
                        'start_line': i - len(current_table),
                        'end_line': i - 1
                    })
                in_table = False
                current_table = []
        
        return tables
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text."""
        from collections import Counter
        import string
        
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into words and remove stopwords
        words = text.split()
        stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        ])
        
        words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Count frequency
        word_counts = Counter(words)
        
        # Return top N keywords
        return [word for word, _ in word_counts.most_common(top_n)]
    
    def get_document_structure(self, text: str) -> Dict[str, Any]:
        """Get comprehensive document structure analysis."""
        return {
            'entities': self.extract_entities(text),
            'dates': self.extract_dates(text),
            'sections': self.extract_sections(text),
            'tables': self.extract_tables(text),
            'keywords': self.extract_keywords(text),
            'metadata': {
                'line_count': len(text.split('\n')),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
                'estimated_pages': max(1, len(text) // 2500)  # Rough estimate
            }
        }