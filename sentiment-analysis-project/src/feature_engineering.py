"""
Feature Engineering Module for Twitter Sentiment Analysis

This module implements various feature extraction techniques for text data:
- TF-IDF Vectorization
- Word2Vec Embeddings
- N-gram features
- Statistical features
- Sentiment lexicon features

Following research methodology for comprehensive feature extraction.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import os
from typing import Tuple, Dict, List, Optional, Union
import logging
from tqdm import tqdm
from collections import Counter
import re

# Import textblob for additional features
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Some features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Comprehensive feature extractor for Twitter sentiment analysis.
    Implements multiple feature extraction techniques.
    """
    
    def __init__(self, output_dir: str = "../models"):
        """
        Initialize the feature extractor.
        
        Args:
            output_dir (str): Directory to save trained models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature extractors
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.doc2vec_model = None
        
        # Feature names for tracking
        self.feature_names = []
        
        logger.info("FeatureExtractor initialized")
    
    def extract_tfidf_features(self, 
                             texts: List[str], 
                             max_features: int = 5000,
                             ngram_range: Tuple[int, int] = (1, 2),
                             min_df: int = 2,
                             max_df: float = 0.95,
                             fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features from text data.
        
        Args:
            texts (List[str]): List of preprocessed texts
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): Range of n-grams to extract
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            fit (bool): Whether to fit the vectorizer or use existing one
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        logger.info(f"Extracting TF-IDF features with max_features={max_features}")
        
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=False,  # Already preprocessed
                token_pattern=r'\b\w+\b'
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.feature_names.extend([f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()])
            
            # Save the vectorizer
            with open(os.path.join(self.output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix.toarray()
    
    def train_word2vec(self, 
                      texts: List[str],
                      vector_size: int = 100,
                      window: int = 5,
                      min_count: int = 2,
                      epochs: int = 10) -> None:
        """
        Train Word2Vec model on the corpus.
        
        Args:
            texts (List[str]): List of preprocessed texts
            vector_size (int): Dimensionality of word vectors
            window (int): Maximum distance between current and predicted word
            min_count (int): Minimum count of words to consider
            epochs (int): Number of training epochs
        """
        logger.info("Training Word2Vec model...")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs,
            sg=1  # Skip-gram model
        )
        
        # Save the model
        self.word2vec_model.save(os.path.join(self.output_dir, 'word2vec_model.bin'))
        
        logger.info(f"Word2Vec model trained with vocabulary size: {len(self.word2vec_model.wv)}")
    
    def extract_word2vec_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract Word2Vec features by averaging word vectors.
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: Word2Vec feature matrix
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained. Call train_word2vec() first.")
        
        logger.info("Extracting Word2Vec features...")
        
        features = []
        vector_size = self.word2vec_model.vector_size
        
        for text in tqdm(texts, desc="Extracting Word2Vec"):
            words = text.split()
            word_vectors = []
            
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            # Average word vectors (or zero vector if no words found)
            if word_vectors:
                text_vector = np.mean(word_vectors, axis=0)
            else:
                text_vector = np.zeros(vector_size)
            
            features.append(text_vector)
        
        features_array = np.array(features)
        self.feature_names.extend([f"w2v_{i}" for i in range(vector_size)])
        
        logger.info(f"Word2Vec features shape: {features_array.shape}")
        return features_array
    
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract statistical features from text.
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: Statistical feature matrix
        """
        logger.info("Extracting statistical features...")
        
        features = []
        
        for text in texts:
            words = text.split()
            
            # Basic statistical features
            char_count = len(text)
            word_count = len(words)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Punctuation and special character counts
            exclamation_count = text.count('!')
            question_count = text.count('?')
            period_count = text.count('.')
            
            # Capitalization features (less relevant after preprocessing but kept for completeness)
            uppercase_count = sum(1 for c in text if c.isupper())
            uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
            
            # Digit count
            digit_count = sum(c.isdigit() for c in text)
            digit_ratio = digit_count / char_count if char_count > 0 else 0
            
            # Unique words ratio
            unique_words = set(words)
            unique_ratio = len(unique_words) / word_count if word_count > 0 else 0
            
            text_features = [
                char_count, word_count, avg_word_length,
                exclamation_count, question_count, period_count,
                uppercase_ratio, digit_ratio, unique_ratio
            ]
            
            features.append(text_features)
        
        features_array = np.array(features)
        stat_feature_names = [
            'char_count', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'period_count',
            'uppercase_ratio', 'digit_ratio', 'unique_ratio'
        ]
        self.feature_names.extend(stat_feature_names)
        
        logger.info(f"Statistical features shape: {features_array.shape}")
        return features_array
    
    def extract_sentiment_lexicon_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features using sentiment lexicons.
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            np.ndarray: Sentiment lexicon features
        """
        if not TEXTBLOB_AVAILABLE:
            logger.warning("TextBlob not available. Skipping sentiment lexicon features.")
            return np.zeros((len(texts), 0))
        
        logger.info("Extracting sentiment lexicon features...")
        
        features = []
        
        for text in texts:
            try:
                blob = TextBlob(text)
                
                # Polarity and subjectivity scores
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Additional sentiment features
                positive_words = 0
                negative_words = 0
                neutral_words = 0
                
                for word in text.split():
                    word_blob = TextBlob(word)
                    word_polarity = word_blob.sentiment.polarity
                    
                    if word_polarity > 0.1:
                        positive_words += 1
                    elif word_polarity < -0.1:
                        negative_words += 1
                    else:
                        neutral_words += 1
                
                total_words = len(text.split())
                positive_ratio = positive_words / total_words if total_words > 0 else 0
                negative_ratio = negative_words / total_words if total_words > 0 else 0
                neutral_ratio = neutral_words / total_words if total_words > 0 else 0
                
                text_features = [
                    polarity, subjectivity,
                    positive_ratio, negative_ratio, neutral_ratio,
                    positive_words, negative_words, neutral_words
                ]
                
            except Exception as e:
                logger.warning(f"Error processing text with TextBlob: {e}")
                text_features = [0] * 8
            
            features.append(text_features)
        
        features_array = np.array(features)
        lexicon_feature_names = [
            'polarity', 'subjectivity',
            'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'positive_words', 'negative_words', 'neutral_words'
        ]
        self.feature_names.extend(lexicon_feature_names)
        
        logger.info(f"Sentiment lexicon features shape: {features_array.shape}")
        return features_array
    
    def extract_all_features(self, 
                           texts: List[str],
                           use_tfidf: bool = True,
                           use_word2vec: bool = True,
                           use_statistical: bool = True,
                           use_lexicon: bool = True,
                           train_word2vec_if_needed: bool = True) -> np.ndarray:
        """
        Extract all types of features.
        
        Args:
            texts (List[str]): List of preprocessed texts
            use_tfidf (bool): Whether to use TF-IDF features
            use_word2vec (bool): Whether to use Word2Vec features
            use_statistical (bool): Whether to use statistical features
            use_lexicon (bool): Whether to use sentiment lexicon features
            train_word2vec_if_needed (bool): Whether to train Word2Vec if not available
            
        Returns:
            np.ndarray: Combined feature matrix
        """
        logger.info("Extracting all features...")
        
        feature_matrices = []
        
        # TF-IDF features
        if use_tfidf:
            tfidf_features = self.extract_tfidf_features(texts)
            feature_matrices.append(tfidf_features)
        
        # Word2Vec features
        if use_word2vec:
            if self.word2vec_model is None and train_word2vec_if_needed:
                self.train_word2vec(texts)
            
            if self.word2vec_model is not None:
                word2vec_features = self.extract_word2vec_features(texts)
                feature_matrices.append(word2vec_features)
            else:
                logger.warning("Word2Vec model not available. Skipping Word2Vec features.")
        
        # Statistical features
        if use_statistical:
            stat_features = self.extract_statistical_features(texts)
            feature_matrices.append(stat_features)
        
        # Sentiment lexicon features
        if use_lexicon:
            lexicon_features = self.extract_sentiment_lexicon_features(texts)
            if lexicon_features.shape[1] > 0:  # Only add if features were extracted
                feature_matrices.append(lexicon_features)
        
        # Combine all features
        if feature_matrices:
            combined_features = np.hstack(feature_matrices)
        else:
            logger.warning("No features extracted!")
            combined_features = np.zeros((len(texts), 0))
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        logger.info(f"Total feature types: {len(feature_matrices)}")
        
        return combined_features
    
    def get_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray) -> pd.DataFrame:
        """
        Create a DataFrame of feature importance scores.
        
        Args:
            feature_names (List[str]): List of feature names
            importance_scores (np.ndarray): Importance scores
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_features(self, features: np.ndarray, filename: str) -> None:
        """
        Save extracted features to disk.
        
        Args:
            features (np.ndarray): Feature matrix
            filename (str): Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        np.save(filepath, features)
        logger.info(f"Features saved to {filepath}")
    
    def load_features(self, filename: str) -> np.ndarray:
        """
        Load features from disk.
        
        Args:
            filename (str): Input filename
            
        Returns:
            np.ndarray: Loaded feature matrix
        """
        filepath = os.path.join(self.output_dir, filename)
        features = np.load(filepath)
        logger.info(f"Features loaded from {filepath}")
        return features


def main():
    """
    Example usage of the FeatureExtractor.
    """
    # Sample texts
    sample_texts = [
        "love product amazing excellent quality",
        "terrible experience waste money poor service",
        "okay nothing special average quality",
        "great service excellent quality highly recommended",
        "poor quality bad customer service disappointed"
    ]
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract all features
    features = extractor.extract_all_features(
        sample_texts,
        use_tfidf=True,
        use_word2vec=True,
        use_statistical=True,
        use_lexicon=True
    )
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature names count: {len(extractor.feature_names)}")
    
    # Save features
    extractor.save_features(features, 'sample_features.npy')
    
    # Save feature names
    with open(os.path.join(extractor.output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(extractor.feature_names, f)


if __name__ == "__main__":
    main()
