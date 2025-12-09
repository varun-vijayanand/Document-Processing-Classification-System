import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DocumentClassifier:
    """Document classification using various models."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigLoader(config_path) if config_path else ConfigLoader()
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.results = {}
    
    def prepare_data(self, texts: List[str], labels: List[Any],
                     test_size: float = 0.2,
                     random_state: int = 42) -> Dict[str, Any]:
        """Prepare data for classification."""
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=encoded_labels
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'classes': self.label_encoder.classes_,
            'class_distribution': {
                'train': np.bincount(y_train),
                'test': np.bincount(y_test)
            }
        }
    
    def train_traditional_model(self, X_train, y_train, X_test, y_test,
                                model_type: str = 'random_forest',
                                vectorizer: Any = None) -> Dict[str, Any]:
        """Train traditional ML model."""
        # Vectorize text if needed
        if vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
        else:
            X_train_vec = vectorizer.transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            model = SVC(
                kernel='linear',
                probability=True,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'naive_bayes':
            model = MultinomialNB()
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        results['model'] = model
        results['vectorizer'] = vectorizer
        results['model_type'] = model_type
        
        self.models[model_type] = model
        self.results[model_type] = results
        
        return results
    
    def train_bert_model(self, X_train, y_train, X_test, y_test,
                         model_name: str = "bert-base-uncased",
                         num_epochs: int = 3,
                         batch_size: int = 16) -> Dict[str, Any]:
        """Train BERT-based classification model."""
        try:
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.label_encoder.classes_)
            )
            
            # Create datasets
            train_dataset = TextDataset(X_train, y_train, tokenizer)
            test_dataset = TextDataset(X_test, y_test, tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./bert_results',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1"
            )
            
            # Define compute_metrics function
            def compute_metrics(p):
                predictions, labels = p
                predictions = np.argmax(predictions, axis=1)
                
                return {
                    'accuracy': accuracy_score(labels, predictions),
                    'f1': f1_score(labels, predictions, average='weighted'),
                    'precision': precision_score(labels, predictions, average='weighted'),
                    'recall': recall_score(labels, predictions, average='weighted')
                }
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics
            )
            
            # Train model
            trainer.train()
            
            # Evaluate model
            eval_results = trainer.evaluate()
            
            # Make predictions
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
            
            results = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            results['model'] = model
            results['tokenizer'] = tokenizer
            results['model_type'] = 'bert'
            results['training_args'] = training_args
            results['eval_results'] = eval_results
            
            self.models['bert'] = model
            self.results['bert'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to train BERT model: {e}")
            raise
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'y_true': y_true.tolist() if isinstance(y_true, np.ndarray) else y_true,
            'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
            'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
        
        return metrics
    
    def cross_validate(self, texts: List[str], labels: List[Any],
                       model_type: str = 'random_forest',
                       n_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Vectorize texts
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVC(kernel='linear', random_state=42)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X, encoded_labels,
            cv=cv, scoring='f1_weighted',
            n_jobs=-1
        )
        
        return {
            'model_type': model_type,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_min': np.min(cv_scores),
            'cv_max': np.max(cv_scores)
        }