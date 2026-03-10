"""
Machine Learning Models Module for Twitter Sentiment Analysis

This module implements various ML models for sentiment classification:
- Naive Bayes (Multinomial and Gaussian)
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- LSTM Neural Network
- BERT Transformer (optional)

Includes model training, hyperparameter tuning, and prediction capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM models will be disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. BERT models will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentClassifier:
    """
    Comprehensive sentiment classifier with multiple ML models.
    """
    
    def __init__(self, model_dir: str = "../models"):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Models dictionary
        self.models = {}
        self.trained_models = {}
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Model performance tracking
        self.model_performance = {}
        
        logger.info("SentimentClassifier initialized")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            test_size (float): Test set size
            random_state (int): Random state
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features for models that need it
        # Note: We'll scale per model as needed
        
        logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_naive_bayes(self, X_train: np.ndarray, y_train: np.ndarray, 
                         model_type: str = 'multinomial') -> Dict[str, Any]:
        """
        Train Naive Bayes classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            model_type (str): 'multinomial' or 'gaussian'
            
        Returns:
            Dict: Training results
        """
        logger.info(f"Training {model_type} Naive Bayes...")
        
        if model_type == 'multinomial':
            model = MultinomialNB()
            # Ensure non-negative features for MultinomialNB
            X_train_nb = np.abs(X_train)
        else:  # gaussian
            model = GaussianNB()
            X_train_nb = X_train
        
        # Train model
        model.fit(X_train_nb, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_nb, y_train, cv=5, scoring='accuracy')
        
        # Store model
        model_name = f'naive_bayes_{model_type}'
        self.trained_models[model_name] = model
        
        # Save model
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        results = {
            'model': model,
            'model_name': model_name,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"{model_type} Naive Bayes trained - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, 
                  kernel: str = 'rbf', C: float = 1.0) -> Dict[str, Any]:
        """
        Train Support Vector Machine classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            kernel (str): SVM kernel type
            C (float): Regularization parameter
            
        Returns:
            Dict: Training results
        """
        logger.info(f"Training SVM with {kernel} kernel...")
        
        # Scale features for SVM
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Store model and scaler
        model_name = f'svm_{kernel}'
        self.trained_models[model_name] = {'model': model, 'scaler': self.scaler}
        
        # Save model and scaler
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': self.scaler}, f)
        
        results = {
            'model': model,
            'model_name': model_name,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"SVM trained - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 C: float = 1.0, max_iter: int = 1000) -> Dict[str, Any]:
        """
        Train Logistic Regression classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            C (float): Regularization parameter
            max_iter (int): Maximum iterations
            
        Returns:
            Dict: Training results
        """
        logger.info("Training Logistic Regression...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42, multi_class='auto')
        model.fit(X_train_scaled, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Store model and scaler
        model_name = 'logistic_regression'
        self.trained_models[model_name] = {'model': model, 'scaler': self.scaler}
        
        # Save model and scaler
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': self.scaler}, f)
        
        results = {
            'model': model,
            'model_name': model_name,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"Logistic Regression trained - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100, max_depth: int = None) -> Dict[str, Any]:
        """
        Train Random Forest classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
            
        Returns:
            Dict: Training results
        """
        logger.info("Training Random Forest...")
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Store model
        model_name = 'random_forest'
        self.trained_models[model_name] = model
        
        # Save model
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        results = {
            'model': model,
            'model_name': model_name,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': model.feature_importances_
        }
        
        logger.info(f"Random Forest trained - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_lstm(self, X_text: List[str], y_train: np.ndarray, 
                   max_words: int = 5000, max_len: int = 100,
                   embedding_dim: int = 128, lstm_units: int = 64) -> Dict[str, Any]:
        """
        Train LSTM neural network for text classification.
        
        Args:
            X_text (List[str]): Original text data (not features)
            y_train (np.ndarray): Training labels
            max_words (int): Maximum vocabulary size
            max_len (int): Maximum sequence length
            embedding_dim (int): Embedding dimension
            lstm_units (int): LSTM units
            
        Returns:
            Dict: Training results
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot train LSTM.")
            return {}
        
        logger.info("Training LSTM neural network...")
        
        # Tokenize text
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_text)
        
        # Convert text to sequences
        sequences = tokenizer.texts_to_sequences(X_text)
        X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        # Build LSTM model
        model = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(lstm_units)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
        
        # Train model
        history = model.fit(
            X_padded, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Store model and tokenizer
        model_name = 'lstm'
        self.trained_models[model_name] = {'model': model, 'tokenizer': tokenizer}
        
        # Save model and tokenizer
        model.save(os.path.join(self.model_dir, f'{model_name}.h5'))
        with open(os.path.join(self.model_dir, f'{model_name}_tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
        
        results = {
            'model': model,
            'model_name': model_name,
            'tokenizer': tokenizer,
            'history': history,
            'max_len': max_len
        }
        
        logger.info("LSTM trained successfully")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, X_text: List[str], y_train: np.ndarray) -> Dict[str, Dict]:
        """
        Train all available models.
        
        Args:
            X_train (np.ndarray): Training features
            X_text (List[str]): Original text data (for LSTM)
            y_train (np.ndarray): Training labels
            
        Returns:
            Dict: All training results
        """
        logger.info("Training all models...")
        
        all_results = {}
        
        # Train Naive Bayes
        try:
            all_results['naive_bayes_multinomial'] = self.train_naive_bayes(X_train, y_train, 'multinomial')
        except Exception as e:
            logger.error(f"Error training Multinomial NB: {e}")
        
        # Train SVM
        try:
            all_results['svm_rbf'] = self.train_svm(X_train, y_train, 'rbf')
        except Exception as e:
            logger.error(f"Error training SVM: {e}")
        
        # Train Logistic Regression
        try:
            all_results['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
        
        # Train Random Forest
        try:
            all_results['random_forest'] = self.train_random_forest(X_train, y_train)
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
        
        # Train LSTM (if TensorFlow available)
        try:
            if TENSORFLOW_AVAILABLE:
                all_results['lstm'] = self.train_lstm(X_text, y_train)
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
        
        # Store performance summary
        self.model_performance = {
            name: {'cv_mean': result.get('cv_mean', 0), 
                   'cv_std': result.get('cv_std', 0)}
            for name, result in all_results.items() if result
        }
        
        logger.info("All models training completed")
        
        return all_results
    
    def predict(self, model_name: str, X: np.ndarray, X_text: List[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the model to use
            X (np.ndarray): Feature matrix
            X_text (List[str]): Original text (for LSTM)
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model_data = self.trained_models[model_name]
        
        if model_name == 'lstm':
            # Special handling for LSTM
            if X_text is None:
                raise ValueError("X_text required for LSTM predictions")
            
            tokenizer = model_data['tokenizer']
            model = model_data['model']
            max_len = model_data.get('max_len', 100)
            
            sequences = tokenizer.texts_to_sequences(X_text)
            X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            predictions = model.predict(X_padded)
            return np.argmax(predictions, axis=1)
        
        elif 'scaler' in model_data:
            # Models that require scaling
            scaler = model_data['scaler']
            model = model_data['model']
            X_scaled = scaler.transform(X)
            return model.predict(X_scaled)
        
        else:
            # Models that don't require scaling
            model = model_data if isinstance(model_data, (MultinomialNB, GaussianNB, RandomForestClassifier)) else model_data['model']
            
            if isinstance(model, MultinomialNB):
                X = np.abs(X)  # Ensure non-negative features
            
            return model.predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray, X_text: List[str] = None) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name (str): Name of the model
            X (np.ndarray): Feature matrix
            X_text (List[str]): Original text (for LSTM)
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model_data = self.trained_models[model_name]
        
        if model_name == 'lstm':
            # Special handling for LSTM
            if X_text is None:
                raise ValueError("X_text required for LSTM predictions")
            
            tokenizer = model_data['tokenizer']
            model = model_data['model']
            max_len = model_data.get('max_len', 100)
            
            sequences = tokenizer.texts_to_sequences(X_text)
            X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            return model.predict(X_padded)
        
        elif 'scaler' in model_data:
            # Models that require scaling
            scaler = model_data['scaler']
            model = model_data['model']
            X_scaled = scaler.transform(X)
            return model.predict_proba(X_scaled)
        
        else:
            # Models that don't require scaling
            model = model_data if isinstance(model_data, (MultinomialNB, GaussianNB, RandomForestClassifier)) else model_data['model']
            
            if isinstance(model, MultinomialNB):
                X = np.abs(X)  # Ensure non-negative features
            
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                raise ValueError(f"Model {model_name} does not support probability predictions")
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all trained models' performance.
        
        Returns:
            pd.DataFrame: Model performance summary
        """
        if not self.model_performance:
            return pd.DataFrame()
        
        summary_df = pd.DataFrame.from_dict(self.model_performance, orient='index')
        summary_df = summary_df.sort_values('cv_mean', ascending=False)
        summary_df.columns = ['CV Accuracy', 'CV Std']
        
        return summary_df
    
    def load_model(self, model_name: str) -> None:
        """
        Load a saved model.
        
        Args:
            model_name (str): Name of the model to load
        """
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.trained_models[model_name] = pickle.load(f)
            logger.info(f"Model {model_name} loaded successfully")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")


def main():
    """
    Example usage of the SentimentClassifier.
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples)  # 3 sentiment classes
    
    # Sample texts for LSTM
    sample_texts = [
        "love product amazing excellent quality",
        "terrible experience waste money poor service",
        "okay nothing special average quality"
    ] * (n_samples // 3)
    
    # Initialize classifier
    classifier = SentimentClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
    
    # Train all models
    results = classifier.train_all_models(X_train, sample_texts[:len(X_train)], y_train)
    
    # Print model summary
    summary = classifier.get_model_summary()
    print("Model Performance Summary:")
    print(summary)
    
    # Make predictions
    if 'svm_rbf' in classifier.trained_models:
        predictions = classifier.predict('svm_rbf', X_test)
        print(f"\nSample predictions: {predictions[:10]}")
        print(f"Actual labels: {y_test[:10]}")


if __name__ == "__main__":
    main()
