#!/usr/bin/env python3
"""
Main script for Document Processing and Classification System.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import DocumentPipeline
from src.utils.logging_config import setup_logging

def main():
    """Main function."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Document Processing and Classification System"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input folder with documents"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["process", "cluster", "classify", "full"],
        default="full",
        help="Pipeline mode"
    )
    
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="hdbscan",
        choices=["kmeans", "hdbscan", "hierarchical"],
        help="Clustering method"
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters (for kmeans/hierarchical)"
    )
    
    parser.add_argument(
        "--classification-model",
        type=str,
        default="random_forest",
        help="Classification model type"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Document Processing and Classification System")
    
    try:
        # Initialize pipeline
        pipeline = DocumentPipeline(config_path=args.config)
        
        # Run pipeline based on mode
        if args.mode == "process":
            logger.info(f"Processing documents from: {args.input}")
            results = pipeline.process_folder(args.input)
            logger.info(f"Processed {results['processed_documents']} documents")
        
        elif args.mode == "cluster":
            logger.info(f"Clustering documents from: {args.input}")
            
            # First process documents
            pipeline.process_folder(args.input)
            
            # Generate embeddings
            pipeline.generate_embeddings()
            
            # Cluster documents
            clustering_results = pipeline.cluster_documents(
                method=args.clustering_method,
                n_clusters=args.n_clusters
            )
            
            logger.info(f"Found {clustering_results['n_clusters']} clusters")
        
        elif args.mode == "full":
            logger.info(f"Running full pipeline on: {args.input}")
            results = pipeline.run_full_pipeline(args.input)
            logger.info("Full pipeline completed successfully")
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
        
        logger.info("Pipeline execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())