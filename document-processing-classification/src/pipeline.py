import logging
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
import json

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.document_validator import DocumentValidator
from src.ocr.pdf_processor import PDFProcessor
from src.ocr.ocr_processor import OCRProcessor
from src.text_processing.cleaner import TextCleaner
from src.text_processing.structure_extractor import StructureExtractor
from src.text_processing.preprocessor import TextPreprocessor
from src.models.embeddings import EmbeddingGenerator
from src.models.clustering import DocumentClustering
from src.models.classification import DocumentClassifier
from src.models.model_comparison import ModelComparison
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import ResultVisualizer
from src.utils.file_utils import FileUtils
from src.utils.config_loader import ConfigLoader
from src.utils.logging_config import setup_logging
import numpy as np

logger = logging.getLogger(__name__)

class DocumentPipeline:
    """Main pipeline for document processing and classification."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Setup logging
        self.logger = setup_logging()
        
        # Load configuration
        self.config = ConfigLoader(config_path)
        
        # Initialize components
        self.file_utils = FileUtils()
        self.document_loader = DocumentLoader()
        self.document_validator = DocumentValidator()
        self.pdf_processor = PDFProcessor(config_path)
        self.ocr_processor = OCRProcessor(config_path)
        self.text_cleaner = TextCleaner()
        self.structure_extractor = StructureExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator(config_path=config_path)
        self.document_clustering = DocumentClustering(config_path)
        self.document_classifier = DocumentClassifier(config_path)
        self.model_comparison = ModelComparison(config_path)
        self.model_evaluator = ModelEvaluator()
        self.result_visualizer = ResultVisualizer()
        
        # Pipeline state
        self.documents = []
        self.processed_documents = []
        self.embeddings = None
        self.results = {}
        
        # Ensure directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config.get("paths.raw_data"),
            self.config.get("paths.processed_data"),
            self.config.get("paths.ocr_output"),
            self.config.get("paths.embeddings"),
            self.config.get("paths.models"),
            "logs",
            "reports",
            "visualizations"
        ]
        
        for directory in directories:
            self.file_utils.ensure_dir(directory)
    
    def process_folder(self, folder_path: str) -> Dict[str, Any]:
        """Process all documents in a folder."""
        self.logger.info(f"Processing folder: {folder_path}")
        
        # Load documents
        self.documents = self.document_loader.load_directory(folder_path)
        self.logger.info(f"Loaded {len(self.documents)} documents")
        
        # Process each document
        processed_docs = []
        
        for doc in self.documents:
            if not doc['success']:
                self.logger.warning(f"Skipping failed document: {doc['metadata']['file_path']}")
                continue
            
            processed = self._process_document(doc)
            if processed:
                processed_docs.append(processed)
        
        self.processed_documents = processed_docs
        self.logger.info(f"Successfully processed {len(processed_docs)} documents")
        
        # Save processed documents
        self._save_processed_documents()
        
        return {
            'total_documents': len(self.documents),
            'processed_documents': len(processed_docs),
            'failed_documents': len([d for d in self.documents if not d['success']])
        }
    
    def _process_document(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single document."""
        try:
            file_path = document['metadata']['file_path']
            file_type = document['metadata']['file_type']
            content = document['content']
            
            # Extract text based on file type
            if file_type == 'PDF':
                # Try direct extraction first, then OCR if needed
                result = self.pdf_processor.extract_text_from_pdf(file_path, use_ocr=False)
                if not result['success'] or len(result['text']) < 50:
                    result = self.pdf_processor.extract_text_from_pdf(file_path, use_ocr=True)
            elif file_type == 'Image':
                result = self.ocr_processor.extract_text(file_path)
            else:
                # Text or other files
                result = {
                    'success': True,
                    'text': str(content) if not isinstance(content, bytes) else '',
                    'method': 'direct'
                }
            
            if not result['success'] or not result.get('text'):
                self.logger.warning(f"No text extracted from {file_path}")
                return None
            
            # Clean and structure text
            cleaned_text = self.text_cleaner.clean_text(result['text'])
            structure = self.structure_extractor.get_document_structure(result['text'])
            
            # Create processed document
            processed_doc = {
                'id': len(self.processed_documents),
                'file_path': file_path,
                'file_type': file_type,
                'original_text': result['text'][:1000] + '...' if len(result['text']) > 1000 else result['text'],
                'cleaned_text': cleaned_text,
                'structure': structure,
                'metadata': {
                    'extraction_method': result.get('method', 'unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text)
                }
            }
            
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Error processing document {document['metadata']['file_path']}: {e}")
            return None
    
    def _save_processed_documents(self):
        """Save processed documents to disk."""
        output_dir = self.config.get("paths.processed_data")
        timestamp = self.file_utils.get_timestamp()
        
        # Save as JSON
        output_file = Path(output_dir) / f"processed_documents_{timestamp}.json"
        
        # Convert to serializable format
        serializable_docs = []
        for doc in self.processed_documents:
            serializable_doc = {
                'id': doc['id'],
                'file_path': str(doc['file_path']),
                'file_type': doc['file_type'],
                'cleaned_text': doc['cleaned_text'],
                'metadata': doc['metadata']
            }
            serializable_docs.append(serializable_doc)
        
        self.file_utils.save_json(serializable_docs, str(output_file))
        self.logger.info(f"Saved {len(serializable_docs)} processed documents to {output_file}")
    
    def generate_embeddings(self, method: str = 'sentence_transformer') -> np.ndarray:
        """Generate embeddings for processed documents."""
        if not self.processed_documents:
            raise ValueError("No documents to process. Run process_folder() first.")
        
        self.logger.info(f"Generating embeddings using {method}")
        
        # Prepare documents for embedding
        docs_for_embedding = []
        for doc in self.processed_documents:
            docs_for_embedding.append({
                'id': doc['id'],
                'text': doc['cleaned_text'],
                'metadata': doc['metadata']
            })
        
        # Generate embeddings
        embedding_results = self.embedding_generator.generate_document_embeddings(
            docs_for_embedding, method=method
        )
        
        self.embeddings = embedding_results['embeddings']
        
        # Save embeddings
        embeddings_dir = self.config.get("paths.embeddings")
        timestamp = self.file_utils.get_timestamp()
        embedding_file = Path(embeddings_dir) / f"embeddings_{method}_{timestamp}.npy"
        
        np.save(str(embedding_file), self.embeddings)
        self.logger.info(f"Saved embeddings to {embedding_file}")
        
        # Save metadata
        metadata_file = Path(embeddings_dir) / f"embeddings_metadata_{method}_{timestamp}.json"
        metadata = {
            'method': method,
            'model_name': embedding_results['model_name'],
            'embedding_dim': embedding_results['embedding_dim'],
            'num_documents': embedding_results['num_documents'],
            'document_ids': embedding_results['document_ids']
        }
        self.file_utils.save_json(metadata, str(metadata_file))
        
        return self.embeddings
    
    def cluster_documents(self, method: str = 'kmeans', n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster documents based on embeddings."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run generate_embeddings() first.")
        
        self.logger.info(f"Clustering documents using {method}")
        
        clustering_results = self.document_clustering.cluster_documents(
            self.embeddings, method=method, n_clusters=n_clusters
        )
        
        # Add document information to results
        cluster_assignments = []
        for i, label in enumerate(clustering_results['labels']):
            cluster_assignments.append({
                'document_id': i,
                'file_path': self.processed_documents[i]['file_path'],
                'cluster': int(label),
                'is_noise': label == -1
            })
        
        clustering_results['assignments'] = cluster_assignments
        
        # Save results
        results_dir = "reports/clustering"
        self.file_utils.ensure_dir(results_dir)
        timestamp = self.file_utils.get_timestamp()
        
        results_file = Path(results_dir) / f"clustering_{method}_{timestamp}.json"
        self.file_utils.save_json(clustering_results, str(results_file))
        
        # Create visualization
        viz_dir = "visualizations/clustering"
        self.file_utils.ensure_dir(viz_dir)
        
        viz_file = Path(viz_dir) / f"clustering_{method}_{timestamp}.png"
        fig = self.result_visualizer.plot_clustering_results(
            self.embeddings, clustering_results['labels'],
            title=f"Document Clustering ({method})"
        )
        fig.savefig(str(viz_file), dpi=300, bbox_inches='tight')
        
        self.logger.info(f"Clustering completed: {clustering_results['n_clusters']} clusters found")
        self.results['clustering'] = clustering_results
        
        return clustering_results
    
    def train_classifier(self, texts: List[str], labels: List[str],
                         model_type: str = 'random_forest',
                         test_size: float = 0.2) -> Dict[str, Any]:
        """Train a document classifier."""
        self.logger.info(f"Training {model_type} classifier")
        
        if model_type == 'bert':
            classification_results = self.document_classifier.train_bert_model(
                texts, labels
            )
        else:
            classification_results = self.document_classifier.train_traditional_model(
                texts, labels, model_type=model_type
            )
        
        # Save results
        results_dir = "reports/classification"
        self.file_utils.ensure_dir(results_dir)
        timestamp = self.file_utils.get_timestamp()
        
        results_file = Path(results_dir) / f"classification_{model_type}_{timestamp}.json"
        self.file_utils.save_json(classification_results, str(results_file))
        
        # Create visualizations
        viz_dir = "visualizations/classification"
        self.file_utils.ensure_dir(viz_dir)
        
        # Confusion matrix
        cm_file = Path(viz_dir) / f"confusion_matrix_{model_type}_{timestamp}.png"
        fig = self.result_visualizer.plot_confusion_matrix(
            classification_results['y_true'],
            classification_results['y_pred'],
            title=f"Confusion Matrix - {model_type}"
        )
        fig.savefig(str(cm_file), dpi=300, bbox_inches='tight')
        
        self.results['classification'] = classification_results
        self.logger.info(f"Classification completed: Accuracy = {classification_results['accuracy']:.3f}")
        
        return classification_results
    
    def compare_classification_models(self, texts: List[str], labels: List[str],
                                      models: List[str] = None) -> Dict[str, Any]:
        """Compare multiple classification models."""
        self.logger.info("Comparing classification models")
        
        comparison_results = self.model_comparison.compare_models(
            texts, labels, models=models
        )
        
        # Save results
        results_dir = "reports/comparison"
        self.file_utils.ensure_dir(results_dir)
        timestamp = self.file_utils.get_timestamp()
        
        results_file = Path(results_dir) / f"model_comparison_{timestamp}.json"
        self.file_utils.save_json(comparison_results, str(results_file))
        
        # Create visualization
        viz_dir = "visualizations/comparison"
        self.file_utils.ensure_dir(viz_dir)
        
        viz_file = Path(viz_dir) / f"model_comparison_{timestamp}.png"
        fig = self.model_comparison.plot_model_comparison()
        if fig:
            fig.savefig(str(viz_file), dpi=300, bbox_inches='tight')
        
        # Generate report
        report = self.model_comparison.generate_report()
        report_file = Path(results_dir) / f"comparison_report_{timestamp}.json"
        self.file_utils.save_json(report, str(report_file))
        
        self.logger.info("Model comparison completed")
        return comparison_results
    
    def run_full_pipeline(self, data_folder: str, 
                          classification_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the full document processing pipeline."""
        pipeline_results = {}
        
        # Step 1: Process documents
        self.logger.info("Step 1: Processing documents")
        processing_results = self.process_folder(data_folder)
        pipeline_results['processing'] = processing_results
        
        # Step 2: Generate embeddings
        self.logger.info("Step 2: Generating embeddings")
        try:
            embeddings = self.generate_embeddings()
            pipeline_results['embeddings'] = {
                'shape': embeddings.shape,
                'method': 'sentence_transformer'
            }
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            pipeline_results['embeddings'] = {'error': str(e)}
        
        # Step 3: Cluster documents
        self.logger.info("Step 3: Clustering documents")
        try:
            clustering_results = self.cluster_documents(method='hdbscan')
            pipeline_results['clustering'] = clustering_results['evaluation']
        except Exception as e:
            self.logger.error(f"Failed to cluster documents: {e}")
            pipeline_results['clustering'] = {'error': str(e)}
        
        # Step 4: Classification (if data provided)
        if classification_data:
            self.logger.info("Step 4: Training classification models")
            try:
                texts = classification_data.get('texts', [])
                labels = classification_data.get('labels', [])
                
                if texts and labels:
                    comparison_results = self.compare_classification_models(texts, labels)
                    pipeline_results['classification'] = comparison_results
                else:
                    self.logger.warning("No classification data provided")
            except Exception as e:
                self.logger.error(f"Failed to train classifiers: {e}")
                pipeline_results['classification'] = {'error': str(e)}
        
        # Save pipeline results
        results_dir = "reports/pipeline"
        self.file_utils.ensure_dir(results_dir)
        timestamp = self.file_utils.get_timestamp()
        
        results_file = Path(results_dir) / f"pipeline_results_{timestamp}.json"
        self.file_utils.save_json(pipeline_results, str(results_file))
        
        self.logger.info("Pipeline execution completed")
        return pipeline_results