import re
import logging
from typing import List, Optional, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

logger = logging.getLogger(__name__)

class TextCleaner:
    """Text cleaning and normalization utilities."""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self._download_nltk_resources()
        
        # Initialize NLTK resources
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize spaCy if available
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def clean_text(self, text: str, 
                   remove_stopwords: bool = True,
                   lemmatize: bool = True,
                   remove_numbers: bool = True,
                   remove_punctuation: bool = True,
                   remove_extra_spaces: bool = True,
                   min_word_length: int = 2) -> str:
        """Clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        cleaned = text.lower()
        
        # Remove special characters and numbers
        if remove_numbers:
            cleaned = re.sub(r'\d+', ' ', cleaned)
        
        if remove_punctuation:
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        
        # Remove extra whitespace
        if remove_extra_spaces:
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Tokenize
        tokens = word_tokenize(cleaned)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        if lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Filter by word length
        tokens = [word for word in tokens if len(word) >= min_word_length]
        
        return ' '.join(tokens)
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = re.compile(r'\S+@\S+')
        return email_pattern.sub('', text)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        return sent_tokenize(text)
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text."""
        paragraphs = text.split('\n\n')
        return [para.strip() for para in paragraphs if para.strip()]
    
    def get_word_frequency(self, text: str, top_n: int = 20) -> Dict[str, int]:
        """Get word frequency distribution."""
        cleaned = self.clean_text(text, remove_stopwords=False, lemmatize=False)
        tokens = word_tokenize(cleaned)
        
        freq_dist = {}
        for token in tokens:
            freq_dist[token] = freq_dist.get(token, 0) + 1
        
        # Sort by frequency
        sorted_freq = dict(sorted(freq_dist.items(), key=lambda x: x[1], reverse=True))
        
        # Return top N
        return dict(list(sorted_freq.items())[:top_n])
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """Get text statistics."""
        sentences = self.extract_sentences(text)
        paragraphs = self.extract_paragraphs(text)
        words = word_tokenize(text)
        
        cleaned_text = self.clean_text(text)
        cleaned_words = word_tokenize(cleaned_text)
        
        return {
            'original_length': len(text),
            'cleaned_length': len(cleaned_text),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'word_count': len(words),
            'cleaned_word_count': len(cleaned_words),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1)
        }