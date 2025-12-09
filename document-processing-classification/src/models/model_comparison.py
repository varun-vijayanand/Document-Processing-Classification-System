import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from src.models.classification import DocumentClassifier

logger = logging.getLogger(__name__)

class ModelComparison:
    """Compare multiple classification models."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.classifier = DocumentClassifier(config_path)
        self.comparison_results = {}
        self.visualizations = {}
    
    def compare_models(self, texts: List[str], labels: List[Any],
                       models: List[str] = None,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """Compare multiple classification models."""
        if models is None:
            models = [
                'random_forest',
                'svm',
                'logistic_regression',
                'naive_bayes',
                'gradient_boosting',
                'xgboost'
            ]
        
        # Prepare data
        data = self.classifier.prepare_data(
            texts, labels,
            test_size=test_size,
            random_state=random_state
        )
        
        comparison = {
            'data_info': {
                'total_samples': len(texts),
                'train_samples': len(data['X_train']),
                'test_samples': len(data['X_test']),
                'num_classes': len(data['classes']),
                'classes': data['classes'].tolist(),
                'class_distribution': data['class_distribution']
            },
            'models': {}
        }
        
        # Train and evaluate each model
        for model_type in models:
            try:
                logger.info(f"Training {model_type}...")
                
                if model_type == 'bert':
                    results = self.classifier.train_bert_model(
                        data['X_train'], data['y_train'],
                        data['X_test'], data['y_test']
                    )
                else:
                    results = self.classifier.train_traditional_model(
                        data['X_train'], data['y_train'],
                        data['X_test'], data['y_test'],
                        model_type=model_type
                    )
                
                comparison['models'][model_type] = {
                    'metrics': {
                        'accuracy': results['accuracy'],
                        'f1_weighted': results['f1_weighted'],
                        'f1_macro': results['f1_macro'],
                        'precision_weighted': results['precision_weighted'],
                        'recall_weighted': results['recall_weighted']
                    },
                    'training_time': results.get('training_time', None),
                    'model_size': results.get('model_size', None)
                }
                
                logger.info(f"{model_type}: Accuracy = {results['accuracy']:.3f}, F1 = {results['f1_weighted']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                comparison['models'][model_type] = {'error': str(e)}
        
        self.comparison_results = comparison
        return comparison
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with model comparison results."""
        rows = []
        
        for model_name, model_info in self.comparison_results.get('models', {}).items():
            if 'error' in model_info:
                continue
                
            row = {
                'model': model_name,
                'accuracy': model_info['metrics']['accuracy'],
                'f1_weighted': model_info['metrics']['f1_weighted'],
                'f1_macro': model_info['metrics']['f1_macro'],
                'precision_weighted': model_info['metrics']['precision_weighted'],
                'recall_weighted': model_info['metrics']['recall_weighted']
            }
            
            if 'training_time' in model_info and model_info['training_time']:
                row['training_time'] = model_info['training_time']
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_model_comparison(self, metric: str = 'f1_weighted',
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot model comparison results."""
        df = self.create_comparison_dataframe()
        
        if df.empty:
            logger.warning("No model results to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        titles = ['Accuracy', 'F1 Score (Weighted)', 'Precision (Weighted)', 'Recall (Weighted)']
        
        for idx, (metric_name, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if metric_name in df.columns:
                bars = ax.bar(df['model'], df[metric_name])
                ax.set_title(title)
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                ax.set_xticklabels(df['model'], rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.visualizations['model_comparison'] = fig
        return fig
    
    def plot_confusion_matrices(self, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Plot confusion matrices for all models."""
        figures = {}
        
        for model_name, model_info in self.classifier.results.items():
            if 'confusion_matrix' not in model_info:
                continue
            
            cm = np.array(model_info['confusion_matrix'])
            classes = self.classifier.label_encoder.classes_
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes,
                       ax=ax)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - {model_name}')
            
            figures[model_name] = fig
            
            if save_path:
                fig.savefig(f"{save_path}_{model_name}.png", dpi=300, bbox_inches='tight')
        
        self.visualizations['confusion_matrices'] = figures
        return figures
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            'summary': {
                'total_models_tested': len(self.comparison_results.get('models', {})),
                'successful_models': sum(1 for m in self.comparison_results.get('models', {}).values() 
                                        if 'error' not in m),
                'best_model': None,
                'best_accuracy': 0
            },
            'detailed_results': self.comparison_results,
            'recommendations': []
        }
        
        # Find best model
        best_model = None
        best_score = 0
        
        for model_name, model_info in self.comparison_results.get('models', {}).items():
            if 'error' in model_info:
                continue
            
            accuracy = model_info['metrics']['accuracy']
            if accuracy > best_score:
                best_score = accuracy
                best_model = model_name
        
        if best_model:
            report['summary']['best_model'] = best_model
            report['summary']['best_accuracy'] = best_score
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(best_model)
        
        return report
    
    def _generate_recommendations(self, best_model: str) -> List[str]:
        """Generate recommendations based on best model."""
        recommendations = []
        
        if best_model == 'bert':
            recommendations.extend([
                "BERT model achieved highest accuracy but requires significant computational resources",
                "Consider fine-tuning on more domain-specific data",
                "Use GPU acceleration for training and inference"
            ])
        elif best_model in ['xgboost', 'gradient_boosting']:
            recommendations.extend([
                "Gradient boosting models performed well and are relatively fast",
                "Consider hyperparameter tuning for better performance",
                "These models handle imbalanced data well"
            ])
        elif best_model == 'random_forest':
            recommendations.extend([
                "Random Forest provides good interpretability",
                "Consider increasing n_estimators for better performance",
                "Feature importance analysis can provide insights"
            ])
        elif best_model == 'svm':
            recommendations.extend([
                "SVM works well with high-dimensional data",
                "Consider using different kernels for non-linear problems",
                "Memory usage can be high with large datasets"
            ])
        
        recommendations.append(f"Recommend using {best_model} for production deployment")
        
        return recommendations