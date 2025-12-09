import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class DocumentClustering:
    """Document clustering using various algorithms."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigLoader(config_path) if config_path else ConfigLoader()
        self.models = {}
        self.results = {}
    
    def reduce_dimensions(self, embeddings: np.ndarray, 
                          n_components: int = 50) -> np.ndarray:
        """Reduce dimensionality for clustering using PCA."""
        reducer = PCA(n_components=min(n_components, embeddings.shape[1]))
        reduced = reducer.fit_transform(embeddings)
        self.models['pca_reducer'] = reducer
        
        logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {reduced.shape[1]} dimensions")
        return reduced
    
    def kmeans_clustering(self, embeddings: np.ndarray,
                          n_clusters: int = 5,
                          random_state: int = 42) -> Dict[str, Any]:
        """Perform K-Means clustering."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init='auto'
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        results = {
            'method': 'kmeans',
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_clusters': n_clusters,
            'model': kmeans
        }
        
        self.models['kmeans'] = kmeans
        return results
    
    def dbscan_clustering(self, embeddings: np.ndarray,
                          eps: float = 0.5,
                          min_samples: int = 5) -> Dict[str, Any]:
        """Perform DBSCAN clustering."""
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        labels = dbscan.fit_predict(embeddings)
        
        results = {
            'method': 'dbscan',
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'outliers': np.sum(labels == -1),
            'model': dbscan
        }
        
        self.models['dbscan'] = dbscan
        return results
    
    def hierarchical_clustering(self, embeddings: np.ndarray,
                                n_clusters: int = 5,
                                linkage: str = 'ward') -> Dict[str, Any]:
        """Perform hierarchical clustering."""
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric='euclidean'
        )
        
        labels = hierarchical.fit_predict(embeddings)
        
        results = {
            'method': 'hierarchical',
            'labels': labels,
            'n_clusters': n_clusters,
            'linkage': linkage,
            'model': hierarchical
        }
        
        self.models['hierarchical'] = hierarchical
        return results
    
    def evaluate_clustering(self, embeddings: np.ndarray, 
                           labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality."""
        # Remove noise points for evaluation
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            return {
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': float('inf'),
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'outlier_ratio': np.mean(labels == -1)
            }
        
        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(set(valid_labels)) < 2:
            return {
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': float('inf'),
                'n_clusters': 1,
                'outlier_ratio': np.mean(labels == -1)
            }
        
        try:
            silhouette = silhouette_score(valid_embeddings, valid_labels)
        except:
            silhouette = -1
        
        try:
            calinski = calinski_harabasz_score(valid_embeddings, valid_labels)
        except:
            calinski = -1
        
        try:
            davies = davies_bouldin_score(valid_embeddings, valid_labels)
        except:
            davies = float('inf')
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'n_clusters': len(set(valid_labels)),
            'outlier_ratio': np.mean(labels == -1),
            'cluster_sizes': [np.sum(valid_labels == label) for label in set(valid_labels)]
        }
    
    def cluster_documents(self, embeddings: np.ndarray,
                          method: str = 'kmeans',
                          n_clusters: Optional[int] = None,
                          reduce_dimensions: bool = True) -> Dict[str, Any]:
        """Cluster documents using specified method."""
        # Reduce dimensions if needed
        if reduce_dimensions and embeddings.shape[1] > 50:
            logger.info(f"Reducing dimensions from {embeddings.shape[1]} to 50")
            embeddings = self.reduce_dimensions(embeddings, n_components=50)
        
        # Perform clustering
        if method == 'kmeans':
            n_clusters = n_clusters or 5
            results = self.kmeans_clustering(embeddings, n_clusters=n_clusters)
        elif method == 'dbscan':
            results = self.dbscan_clustering(embeddings)
        elif method == 'hierarchical':
            n_clusters = n_clusters or 5
            results = self.hierarchical_clustering(embeddings, n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Evaluate clustering
        evaluation = self.evaluate_clustering(embeddings, results['labels'])
        results['evaluation'] = evaluation
        
        logger.info(f"Clustering completed: {results['n_clusters']} clusters found")
        return results
    
    def compare_clustering_methods(self, embeddings: np.ndarray,
                                   methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Compare multiple clustering methods."""
        if methods is None:
            methods = ['kmeans', 'dbscan', 'hierarchical']
        
        comparisons = {}
        
        for method in methods:
            try:
                results = self.cluster_documents(embeddings, method=method)
                comparisons[method] = results
                logger.info(f"Method {method}: {results['evaluation']}")
            except Exception as e:
                logger.error(f"Failed to run {method} clustering: {e}")
                comparisons[method] = {'error': str(e)}
        
        return comparisons