import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
import json

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_classification(self, y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray] = None,
                                labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate classification model performance."""
        
        # Basic metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        }
        
        # ROC AUC if probabilities are available
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    # Multiclass classification
                    metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
                    metrics['roc_auc_ovo'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo'))
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['per_class'] = class_report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional statistics
        metrics['statistics'] = {
            'total_samples': len(y_true),
            'class_distribution': {
                str(cls): int(count) for cls, count in enumerate(np.bincount(y_true))
            },
            'prediction_distribution': {
                str(cls): int(count) for cls, count in enumerate(np.bincount(y_pred))
            }
        }
        
        return metrics
    
    def evaluate_clustering(self, embeddings: np.ndarray,
                           labels: np.ndarray,
                           true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate clustering performance."""
        
        from sklearn.metrics import (
            silhouette_score, calinski_harabasz_score,
            davies_bouldin_score, adjusted_rand_score,
            normalized_mutual_info_score, homogeneity_score,
            completeness_score, v_measure_score
        )
        
        metrics = {}
        
        # Internal metrics (don't require true labels)
        valid_mask = labels != -1  # Exclude noise points
        
        if np.sum(valid_mask) >= 2:
            valid_embeddings = embeddings[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(np.unique(valid_labels)) >= 2:
                try:
                    metrics['silhouette_score'] = float(
                        silhouette_score(valid_embeddings, valid_labels)
                    )
                except:
                    metrics['silhouette_score'] = -1.0
                
                try:
                    metrics['calinski_harabasz_score'] = float(
                        calinski_harabasz_score(valid_embeddings, valid_labels)
                    )
                except:
                    metrics['calinski_harabasz_score'] = -1.0
                
                try:
                    metrics['davies_bouldin_score'] = float(
                        davies_bouldin_score(valid_embeddings, valid_labels)
                    )
                except:
                    metrics['davies_bouldin_score'] = float('inf')
        
        # External metrics (require true labels)
        if true_labels is not None:
            # Remove noise points for external metrics too
            if np.any(labels == -1):
                non_noise_mask = labels != -1
                filtered_labels = labels[non_noise_mask]
                filtered_true = true_labels[non_noise_mask]
            else:
                filtered_labels = labels
                filtered_true = true_labels
            
            if len(filtered_labels) > 0:
                try:
                    metrics['adjusted_rand_score'] = float(
                        adjusted_rand_score(filtered_true, filtered_labels)
                    )
                except:
                    metrics['adjusted_rand_score'] = -1.0
                
                try:
                    metrics['normalized_mutual_info'] = float(
                        normalized_mutual_info_score(filtered_true, filtered_labels)
                    )
                except:
                    metrics['normalized_mutual_info'] = -1.0
                
                try:
                    metrics['homogeneity_score'] = float(
                        homogeneity_score(filtered_true, filtered_labels)
                    )
                    metrics['completeness_score'] = float(
                        completeness_score(filtered_true, filtered_labels)
                    )
                    metrics['v_measure_score'] = float(
                        v_measure_score(filtered_true, filtered_labels)
                    )
                except:
                    metrics['homogeneity_score'] = -1.0
                    metrics['completeness_score'] = -1.0
                    metrics['v_measure_score'] = -1.0
        
        # Clustering statistics
        unique_clusters = np.unique(labels)
        cluster_sizes = []
        
        for cluster in unique_clusters:
            if cluster != -1:  # Exclude noise
                size = np.sum(labels == cluster)
                cluster_sizes.append(size)
        
        metrics['cluster_statistics'] = {
            'num_clusters': len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            'num_noise_points': int(np.sum(labels == -1)),
            'noise_ratio': float(np.mean(labels == -1)),
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            'std_cluster_size': float(np.std(cluster_sizes)) if cluster_sizes else 0.0
        }
        
        return metrics
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models."""
        comparison = {
            'model_comparison': {},
            'best_models': {},
            'summary': {}
        }
        
        best_accuracy = -1
        best_f1 = -1
        best_model_accuracy = None
        best_model_f1 = None
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                comparison['model_comparison'][model_name] = {'error': model_results['error']}
                continue
            
            metrics = model_results.get('metrics', {})
            comparison['model_comparison'][model_name] = metrics
            
            # Track best models
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_weighted', 0)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_accuracy = model_name
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_f1 = model_name
        
        comparison['best_models'] = {
            'best_accuracy': {
                'model': best_model_accuracy,
                'score': best_accuracy
            },
            'best_f1': {
                'model': best_model_f1,
                'score': best_f1
            }
        }
        
        # Summary statistics
        accuracies = []
        f1_scores = []
        
        for model_name, model_info in comparison['model_comparison'].items():
            if 'error' not in model_info:
                accuracies.append(model_info.get('accuracy', 0))
                f1_scores.append(model_info.get('f1_weighted', 0))
        
        if accuracies:
            comparison['summary'] = {
                'avg_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'avg_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'num_models': len(accuracies)
            }
        
        return comparison
    
    def save_evaluation_report(self, results: Dict[str, Any], 
                              filepath: str) -> None:
        """Save evaluation results to JSON file."""
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        logger.info(f"Evaluation report saved to {filepath}")