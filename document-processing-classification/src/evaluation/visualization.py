import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """Visualization utilities for model results."""
    
    def __init__(self):
        self.figures = {}
    
    def plot_confusion_matrix(self, y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              title: str = "Confusion Matrix",
                              normalize: bool = True,
                              figsize: tuple = (10, 8)) -> plt.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(y_true)))]
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        self.figures['confusion_matrix'] = fig
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray,
                        y_prob: np.ndarray,
                        class_names: Optional[List[str]] = None,
                        title: str = "ROC Curves") -> plt.Figure:
        """Plot ROC curves for multiclass classification."""
        n_classes = y_prob.shape[1]
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        self.figures['roc_curves'] = fig
        return fig
    
    def plot_precision_recall_curves(self, y_true: np.ndarray,
                                     y_prob: np.ndarray,
                                     class_names: Optional[List[str]] = None,
                                     title: str = "Precision-Recall Curves") -> plt.Figure:
        """Plot precision-recall curves."""
        n_classes = y_prob.shape[1]
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compute precision-recall curve for each class
        precision = {}
        recall = {}
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true == i, y_prob[:, i]
            )
            ax.plot(recall[i], precision[i], lw=2,
                   label=f'{class_names[i]}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        self.figures['pr_curves'] = fig
        return fig
    
    def plot_clustering_results(self, embeddings: np.ndarray,
                                labels: np.ndarray,
                                title: str = "Document Clustering",
                                reduce_dim: bool = True) -> plt.Figure:
        """Visualize clustering results."""
        if reduce_dim and embeddings.shape[1] > 2:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            reduced_embeddings = reducer.fit_transform(embeddings)
        else:
            reduced_embeddings = embeddings[:, :2]
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                mask = labels == label
                ax.scatter(reduced_embeddings[mask, 0],
                          reduced_embeddings[mask, 1],
                          c='gray', alpha=0.3, s=10,
                          label='Noise')
            else:
                mask = labels == label
                ax.scatter(reduced_embeddings[mask, 0],
                          reduced_embeddings[mask, 1],
                          c=[color], alpha=0.7, s=30,
                          label=f'Cluster {label}')
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        self.figures['clustering'] = fig
        return fig
    
    def plot_model_comparison_bar(self, comparison_results: Dict[str, Dict[str, float]],
                                  metric: str = 'accuracy',
                                  title: str = "Model Comparison") -> plt.Figure:
        """Plot bar chart comparing models."""
        models = list(comparison_results.keys())
        scores = [comparison_results[model].get(metric, 0) for model in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(models)), scores, color='skyblue', alpha=0.8)
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        
        self.figures['model_comparison_bar'] = fig
        return fig
    
    def create_interactive_clustering_plot(self, embeddings: np.ndarray,
                                          labels: np.ndarray,
                                          texts: Optional[List[str]] = None,
                                          title: str = "Interactive Clustering") -> go.Figure:
        """Create interactive clustering plot with Plotly."""
        if embeddings.shape[1] > 3:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3)
            reduced_embeddings = reducer.fit_transform(embeddings)
        else:
            reduced_embeddings = embeddings
        
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Plotly
        
        fig = go.Figure()
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = colors[i % len(colors)]
            
            if label == -1:
                name = 'Noise'
                marker_size = 5
                opacity = 0.3
            else:
                name = f'Cluster {label}'
                marker_size = 8
                opacity = 0.7
            
            # Prepare hover text
            if texts is not None:
                hover_texts = [texts[j][:100] + '...' for j in np.where(mask)[0]]
            else:
                hover_texts = [f'Point {j}' for j in np.where(mask)[0]]
            
            if reduced_embeddings.shape[1] == 3:
                # 3D plot
                fig.add_trace(go.Scatter3d(
                    x=reduced_embeddings[mask, 0],
                    y=reduced_embeddings[mask, 1],
                    z=reduced_embeddings[mask, 2],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=marker_size,
                        color=color,
                        opacity=opacity
                    ),
                    text=hover_texts,
                    hoverinfo='text'
                ))
            else:
                # 2D plot
                fig.add_trace(go.Scatter(
                    x=reduced_embeddings[mask, 0],
                    y=reduced_embeddings[mask, 1],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=marker_size,
                        color=color,
                        opacity=opacity
                    ),
                    text=hover_texts,
                    hoverinfo='text'
                ))
        
        if reduced_embeddings.shape[1] == 3:
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Component 1',
                    yaxis_title='Component 2',
                    zaxis_title='Component 3'
                ),
                height=800
            )
        else:
            fig.update_layout(
                title=title,
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                height=600
            )
        
        self.figures['interactive_clustering'] = fig
        return fig
    
    def save_all_figures(self, directory: str, format: str = 'png', dpi: int = 300) -> None:
        """Save all generated figures."""
        import os
        from pathlib import Path
        
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for name, fig in self.figures.items():
            if isinstance(fig, plt.Figure):
                filepath = os.path.join(directory, f"{name}.{format}")
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
                logger.info(f"Saved figure: {filepath}")
            elif isinstance(fig, go.Figure):
                filepath = os.path.join(directory, f"{name}.html")
                fig.write_html(filepath)
                logger.info(f"Saved interactive plot: {filepath}")