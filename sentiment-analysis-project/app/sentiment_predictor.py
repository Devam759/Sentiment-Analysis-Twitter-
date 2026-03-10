"""
Twitter Sentiment Prediction Demo Application

This module provides a simple interactive interface for sentiment prediction.
Users can input tweets and get real-time sentiment predictions using trained models.

Features:
- Text preprocessing
- Feature extraction
- Model prediction
- Confidence scores
- Batch prediction
- Model comparison
- Simple web interface (optional)
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import TwitterTextPreprocessor
from feature_engineering import FeatureExtractor
from models import SentimentClassifier
from evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Interactive sentiment prediction application.
    """
    
    def __init__(self, model_dir: str = "../models"):
        """
        Initialize the sentiment predictor.
        
        Args:
            model_dir (str): Directory containing trained models
        """
        self.model_dir = model_dir
        self.models_dir = os.path.join(model_dir, "models")
        self.results_dir = os.path.join(model_dir, "results")
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.feature_extractor = None
        self.classifier = None
        self.evaluator = None
        
        # Available models
        self.available_models = {}
        
        # Prediction history
        self.prediction_history = []
        
        logger.info("SentimentPredictor initialized")
    
    def load_models(self) -> bool:
        """
        Load all trained models and components.
        
        Returns:
            bool: True if models loaded successfully
        """
        logger.info("Loading models and components...")
        
        try:
            # Load preprocessor configuration
            self.preprocessor = TwitterTextPreprocessor()
            
            # Load feature extractor
            self.feature_extractor = FeatureExtractor(self.models_dir)
            
            # Try to load TF-IDF vectorizer
            tfidf_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.feature_extractor.tfidf_vectorizer = pickle.load(f)
                logger.info("TF-IDF vectorizer loaded")
            
            # Try to load Word2Vec model
            word2vec_path = os.path.join(self.models_dir, 'word2vec_model.bin')
            if os.path.exists(word2vec_path):
                from gensim.models import Word2Vec
                self.feature_extractor.word2vec_model = Word2Vec.load(word2vec_path)
                logger.info("Word2Vec model loaded")
            
            # Load classifier
            self.classifier = SentimentClassifier(self.models_dir)
            
            # Check for available models
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            self.available_models = {f.replace('.pkl', ''): f for f in model_files}
            
            # Load label encoder if exists
            label_encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.classifier.label_encoder = pickle.load(f)
            
            # Initialize evaluator
            self.evaluator = ModelEvaluator(self.results_dir)
            
            logger.info(f"Loaded {len(self.available_models)} models: {list(self.available_models.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        return self.preprocessor.preprocess_text(text)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from texts.
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: Feature matrix
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not loaded")
        
        # Extract features (only TF-IDF and statistical features for demo)
        features = self.feature_extractor.extract_all_features(
            texts,
            use_tfidf=True,
            use_word2vec=self.feature_extractor.word2vec_model is not None,
            use_statistical=True,
            use_lexicon=True,
            train_word2vec_if_needed=False
        )
        
        return features
    
    def predict_sentiment(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text
            model_name (str): Name of the model to use (None for best available)
            
        Returns:
            Dict: Prediction results
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        # Select model
        if model_name is None:
            if not self.available_models:
                raise ValueError("No models available")
            model_name = list(self.available_models.keys())[0]  # Use first available
        
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract features
        features = self.extract_features([processed_text])
        
        # Make prediction
        try:
            prediction = self.classifier.predict(model_name, features)
            prediction_proba = None
            
            try:
                prediction_proba = self.classifier.predict_proba(model_name, features)
            except:
                logger.warning("Probability predictions not available for this model")
            
            # Convert prediction to sentiment label
            if hasattr(self.classifier.label_encoder, 'inverse_transform'):
                sentiment_label = self.classifier.label_encoder.inverse_transform([prediction])[0]
            else:
                # Default mapping
                sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                sentiment_label = sentiment_map.get(prediction[0], 'unknown')
            
            # Calculate confidence
            confidence = 0.0
            if prediction_proba is not None:
                confidence = np.max(prediction_proba[0])
            
            result = {
                'text': text,
                'processed_text': processed_text,
                'predicted_sentiment': sentiment_label,
                'prediction_id': int(prediction[0]),
                'confidence': confidence,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat(),
                'probabilities': prediction_proba[0].tolist() if prediction_proba is not None else None
            }
            
            # Add to history
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'text': text,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, texts: List[str], model_name: str = None) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            model_name (str): Name of the model to use
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict_sentiment(text, model_name)
            results.append(result)
        
        return results
    
    def compare_models(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare predictions from all available models.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Model comparison results
        """
        comparison_results = {}
        
        for model_name in self.available_models.keys():
            try:
                result = self.predict_sentiment(text, model_name)
                comparison_results[model_name] = result
            except Exception as e:
                comparison_results[model_name] = {'error': str(e)}
        
        return comparison_results
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prediction history.
        
        Returns:
            Dict: Prediction statistics
        """
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.prediction_history)
        
        # Remove error predictions
        valid_predictions = df[df['predicted_sentiment'].notna()]
        
        if valid_predictions.empty:
            return {'message': 'No valid predictions yet'}
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'valid_predictions': len(valid_predictions),
            'error_predictions': len(self.prediction_history) - len(valid_predictions),
            'sentiment_distribution': valid_predictions['predicted_sentiment'].value_counts().to_dict(),
            'average_confidence': valid_predictions['confidence'].mean(),
            'models_used': valid_predictions['model_used'].value_counts().to_dict(),
            'most_common_sentiment': valid_predictions['predicted_sentiment'].mode().iloc[0] if not valid_predictions.empty else None
        }
        
        return stats
    
    def export_predictions(self, filename: str = None) -> str:
        """
        Export prediction history to CSV.
        
        Args:
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.results_dir, filename)
        
        if self.prediction_history:
            df = pd.DataFrame(self.prediction_history)
            df.to_csv(filepath, index=False)
            logger.info(f"Predictions exported to {filepath}")
        else:
            logger.warning("No predictions to export")
        
        return filepath
    
    def interactive_demo(self) -> None:
        """
        Run interactive demo in console.
        """
        print("=" * 60)
        print("Twitter Sentiment Analysis - Interactive Demo")
        print("=" * 60)
        print()
        
        if not self.load_models():
            print("Error: Could not load models. Please train models first.")
            return
        
        print(f"Available models: {list(self.available_models.keys())}")
        print()
        
        while True:
            print("\nOptions:")
            print("1. Predict sentiment for a single tweet")
            print("2. Compare all models")
            print("3. Batch prediction")
            print("4. View prediction statistics")
            print("5. Export predictions")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self._single_prediction_demo()
            elif choice == '2':
                self._model_comparison_demo()
            elif choice == '3':
                self._batch_prediction_demo()
            elif choice == '4':
                self._show_statistics()
            elif choice == '5':
                self._export_predictions_demo()
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _single_prediction_demo(self) -> None:
        """Demo for single prediction."""
        print("\n" + "-" * 40)
        print("Single Tweet Prediction")
        print("-" * 40)
        
        text = input("Enter tweet text (or 'back' to return): ").strip()
        
        if text.lower() == 'back':
            return
        
        if not text:
            print("Please enter some text.")
            return
        
        # Select model
        print(f"\nAvailable models: {list(self.available_models.keys())}")
        model_choice = input("Enter model name (or press Enter for default): ").strip()
        
        model_name = model_choice if model_choice else None
        
        # Make prediction
        print("\nAnalyzing...")
        result = self.predict_sentiment(text, model_name)
        
        # Display result
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPrediction Results:")
            print(f"Text: {result['text']}")
            print(f"Predicted Sentiment: {result['predicted_sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Model Used: {result['model_used']}")
            
            if result['probabilities']:
                print("\nProbabilities:")
                if hasattr(self.classifier.label_encoder, 'classes_'):
                    classes = self.classifier.label_encoder.classes_
                    for i, (cls, prob) in enumerate(zip(classes, result['probabilities'])):
                        print(f"  {cls}: {prob:.3f}")
                else:
                    print(f"  Negative: {result['probabilities'][0]:.3f}")
                    print(f"  Neutral: {result['probabilities'][1]:.3f}")
                    print(f"  Positive: {result['probabilities'][2]:.3f}")
    
    def _model_comparison_demo(self) -> None:
        """Demo for model comparison."""
        print("\n" + "-" * 40)
        print("Model Comparison")
        print("-" * 40)
        
        text = input("Enter tweet text (or 'back' to return): ").strip()
        
        if text.lower() == 'back':
            return
        
        if not text:
            print("Please enter some text.")
            return
        
        print("\nComparing all models...")
        results = self.compare_models(text)
        
        print(f"\nModel Comparison Results:")
        print(f"Text: {text}")
        print("-" * 40)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Sentiment: {result['predicted_sentiment'].upper()}")
                print(f"  Confidence: {result['confidence']:.3f}")
    
    def _batch_prediction_demo(self) -> None:
        """Demo for batch prediction."""
        print("\n" + "-" * 40)
        print("Batch Prediction")
        print("-" * 40)
        
        print("Enter multiple tweets (one per line). Enter 'done' when finished:")
        
        texts = []
        while True:
            text = input(f"Tweet {len(texts) + 1}: ").strip()
            
            if text.lower() == 'done':
                break
            elif text.lower() == 'back':
                return
            elif text:
                texts.append(text)
        
        if not texts:
            print("No tweets entered.")
            return
        
        # Select model
        print(f"\nAvailable models: {list(self.available_models.keys())}")
        model_choice = input("Enter model name (or press Enter for default): ").strip()
        
        model_name = model_choice if model_choice else None
        
        print(f"\nAnalyzing {len(texts)} tweets...")
        results = self.predict_batch(texts, model_name)
        
        # Display results
        print(f"\nBatch Prediction Results:")
        print("-" * 60)
        
        for i, result in enumerate(results):
            print(f"\nTweet {i + 1}: {result['text'][:50]}...")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Sentiment: {result['predicted_sentiment'].upper()}")
                print(f"  Confidence: {result['confidence']:.3f}")
    
    def _show_statistics(self) -> None:
        """Show prediction statistics."""
        print("\n" + "-" * 40)
        print("Prediction Statistics")
        print("-" * 40)
        
        stats = self.get_prediction_statistics()
        
        if 'message' in stats:
            print(stats['message'])
        else:
            print(f"Total Predictions: {stats['total_predictions']}")
            print(f"Valid Predictions: {stats['valid_predictions']}")
            print(f"Error Predictions: {stats['error_predictions']}")
            print(f"Average Confidence: {stats['average_confidence']:.3f}")
            print(f"Most Common Sentiment: {stats['most_common_sentiment']}")
            
            print(f"\nSentiment Distribution:")
            for sentiment, count in stats['sentiment_distribution'].items():
                print(f"  {sentiment}: {count}")
            
            print(f"\nModels Used:")
            for model, count in stats['models_used'].items():
                print(f"  {model}: {count}")
    
    def _export_predictions_demo(self) -> None:
        """Demo for exporting predictions."""
        print("\n" + "-" * 40)
        print("Export Predictions")
        print("-" * 40)
        
        if not self.prediction_history:
            print("No predictions to export.")
            return
        
        filename = input("Enter filename (or press Enter for default): ").strip()
        
        if not filename:
            filename = None
        
        filepath = self.export_predictions(filename)
        print(f"Predictions exported to: {filepath}")


def main():
    """
    Main function to run the sentiment predictor demo.
    """
    predictor = SentimentPredictor()
    predictor.interactive_demo()


if __name__ == "__main__":
    main()
