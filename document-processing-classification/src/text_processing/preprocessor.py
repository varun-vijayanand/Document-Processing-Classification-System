import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

from .cleaner import TextCleaner

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing for machine learning."""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.vectorizers = {}
        self.models = {}
    
    def prepare_corpus(self, documents: List[str], 
                       clean: bool = True,
                       min_length: int = 50) -> List[str]:
        """Prepare corpus for vectorization."""
        prepared = []
        
        for doc in documents:
            if not doc or not isinstance(doc, str):
                continue
                
            if len(doc) < min_length:
                logger.warning(f"Document too short ({len(doc)} chars), skipping")
                continue
            
            if clean:
                cleaned = self.cleaner.clean_text(doc)
                if cleaned and len(cleaned) >= min_length:
                    prepared.append(cleaned)
            else:
                prepared.append(doc)
        
        return prepared
    
    def create_tfidf_vectors(self, documents: List[str],
                             max_features: int = 5000,
                             ngram_range: tuple = (1, 2),
                             min_df: int = 2,
                             max_df: float = 0.8) -> np.ndarray:
        """Create TF-IDF vectors from documents."""
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        self.vectorizers['tfidf'] = vectorizer
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix.toarray()
    
    def create_count_vectors(self, documents: List[str],
                             max_features: int = 5000,
                             ngram_range: tuple = (1, 1)) -> np.ndarray:
        """Create bag-of-words vectors from documents."""
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        count_matrix = vectorizer.fit_transform(documents)
        self.vectorizers['count'] = vectorizer
        
        logger.info(f"Count matrix shape: {count_matrix.shape}")
        return count_matrix.toarray()
    
    def reduce_dimensionality(self, vectors: np.ndarray,
                              n_components: int = 100,
                              method: str = 'svd') -> np.ndarray:
        """Reduce dimensionality of document vectors."""
        if method == 'svd':
            reducer = TruncatedSVD(
                n_components=min(n_components, vectors.shape[1]),
                random_state=42
            )
        elif method == 'lda':
            reducer = LatentDirichletAllocation(
                n_components=n_components,
                random_state=42,
                learning_method='online'
            )
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        reduced = reducer.fit_transform(vectors)
        self.models[f'reducer_{method}'] = reducer
        
        logger.info(f"Reduced shape: {reduced.shape}")
        return reduced
    
    def create_document_features(self, documents: List[str]) -> Dict[str, Any]:
        """Create multiple feature representations for documents."""
        prepared_docs = self.prepare_corpus(documents)
        
        features = {
            'documents': prepared_docs,
            'tfidf_vectors': self.create_tfidf_vectors(prepared_docs),
            'count_vectors': self.create_count_vectors(prepared_docs),
            'stats': [self.cleaner.get_text_stats(doc) for doc in documents]
        }
        
        # Add reduced dimensions
        features['svd_vectors'] = self.reduce_dimensionality(
            features['tfidf_vectors'],
            n_components=50,
            method='svd'
        )
        
        return features