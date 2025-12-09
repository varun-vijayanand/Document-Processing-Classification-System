import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for documents using various models."""
    
    def __init__(self, model_name: Optional[str] = None, config_path: Optional[str] = None):
        self.config = ConfigLoader(config_path) if config_path else ConfigLoader()
        self.model_name = model_name or self.config.get("embeddings.model_name", "all-MiniLM-L6-v2")
        self.device = self.config.get("embeddings.device", "cpu")
        self.batch_size = self.config.get("embeddings.batch_size", 32)
        
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load embedding models."""
        try:
            # Load Sentence Transformer model
            self.sentence_model = SentenceTransformer(self.model_name)
            self.sentence_model.to(self.device)
            logger.info(f"Loaded Sentence Transformer model: {self.model_name}")
            
            # Also load BERT model for comparison
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
                self.bert_model.to(self.device)
                logger.info("Loaded BERT model for embeddings")
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}")
                self.bert_model = None
            
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise
    
    def generate_sentence_embeddings(self, texts: List[str], 
                                     batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings using Sentence Transformers."""
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        
        # Process in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.sentence_model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            if self.device == "cuda":
                batch_embeddings = batch_embeddings.cpu()
            
            embeddings.append(batch_embeddings.numpy())
        
        return np.vstack(embeddings)
    
    def generate_bert_embeddings(self, texts: List[str], 
                                 pooling: str = 'mean') -> np.ndarray:
        """Generate embeddings using BERT model."""
        if not self.bert_model or not texts:
            return np.array([])
        
        self.bert_model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.bert_model(**inputs)
                
                if pooling == 'mean':
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                elif pooling == 'cls':
                    # CLS token pooling
                    pooled = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")
                
                if self.device == "cuda":
                    pooled = pooled.cpu()
                
                embeddings.append(pooled.numpy())
        
        return np.vstack(embeddings)
    
    def generate_tfidf_weighted_embeddings(self, texts: List[str],
                                           tfidf_vectors: np.ndarray) -> np.ndarray:
        """Generate TF-IDF weighted sentence embeddings."""
        if len(texts) != tfidf_vectors.shape[0]:
            raise ValueError("Number of texts must match number of TF-IDF vectors")
        
        # Get sentence embeddings for each document
        sentence_embeddings = self.generate_sentence_embeddings(texts)
        
        # Normalize TF-IDF vectors
        tfidf_normalized = tfidf_vectors / (np.linalg.norm(tfidf_vectors, axis=1, keepdims=True) + 1e-8)
        
        # Apply TF-IDF weighting (conceptual - in practice, might need different approach)
        weighted_embeddings = sentence_embeddings * tfidf_normalized.mean(axis=1, keepdims=True)
        
        return weighted_embeddings
    
    def generate_document_embeddings(self, documents: List[Dict[str, Any]], 
                                     method: str = 'sentence_transformer') -> Dict[str, Any]:
        """Generate embeddings for documents with metadata."""
        results = {
            'method': method,
            'model_name': self.model_name,
            'embeddings': None,
            'document_ids': [],
            'metadata': []
        }
        
        texts = []
        for doc in documents:
            texts.append(doc.get('text', ''))
            results['document_ids'].append(doc.get('id', len(results['document_ids'])))
            results['metadata'].append(doc.get('metadata', {}))
        
        if method == 'sentence_transformer':
            embeddings = self.generate_sentence_embeddings(texts)
        elif method == 'bert':
            embeddings = self.generate_bert_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        results['embeddings'] = embeddings
        results['embedding_dim'] = embeddings.shape[1]
        results['num_documents'] = len(documents)
        
        logger.info(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        return results