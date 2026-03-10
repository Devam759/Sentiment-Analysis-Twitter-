"""
Data Collection and Loading Module for Twitter Sentiment Analysis

This module handles:
- Loading Twitter sentiment datasets from various sources
- Data validation and basic cleaning
- Dataset preparation for preprocessing pipeline
"""

import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple, Dict, List
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataLoader:
    """
    A comprehensive data loader for Twitter sentiment analysis datasets.
    Supports multiple dataset formats and sources.
    """
    
    def __init__(self, data_dir: str = "../dataset"):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory to store/load datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_sentiment140(self, sample_size: int = None) -> pd.DataFrame:
        """
        Load the Sentiment140 dataset.
        This is a popular benchmark dataset for Twitter sentiment analysis.
        
        Args:
            sample_size (int): Number of samples to load (None for all)
            
        Returns:
            pd.DataFrame: Processed dataset with 'text' and 'sentiment' columns
        """
        logger.info("Loading Sentiment140 dataset...")
        
        # Column names for Sentiment140 dataset
        columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
        
        try:
            # Try to load from local file first
            file_path = os.path.join(self.data_dir, 'sentiment140.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='latin-1', header=None, names=columns)
            else:
                logger.warning("Local Sentiment140 file not found. Creating sample dataset...")
                # Create a sample dataset for demonstration
                df = self._create_sample_dataset()
                
            # Convert polarity to sentiment labels (0=negative, 2=neutral, 4=positive)
            sentiment_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
            df['sentiment'] = df['polarity'].map(sentiment_map)
            
            # Sample if specified
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                
            # Keep only relevant columns
            df = df[['text', 'sentiment']].copy()
            
            logger.info(f"Loaded {len(df)} samples from Sentiment140")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Sentiment140: {e}")
            # Fallback to sample dataset
            return self._create_sample_dataset(sample_size)
    
    def load_custom_dataset(self, file_path: str, text_column: str, 
                          sentiment_column: str) -> pd.DataFrame:
        """
        Load a custom dataset with specified text and sentiment columns.
        
        Args:
            file_path (str): Path to the dataset file
            text_column (str): Name of the text column
            sentiment_column (str): Name of the sentiment column
            
        Returns:
            pd.DataFrame: Processed dataset
        """
        logger.info(f"Loading custom dataset from {file_path}")
        
        try:
            # Detect file format
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV, JSON, or Excel.")
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in dataset")
            if sentiment_column not in df.columns:
                raise ValueError(f"Sentiment column '{sentiment_column}' not found in dataset")
            
            # Select and rename columns
            df = df[[text_column, sentiment_column]].copy()
            df.columns = ['text', 'sentiment']
            
            logger.info(f"Loaded {len(df)} samples from custom dataset")
            return df
            
        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            raise
    
    def _create_sample_dataset(self, size: int = 1000) -> pd.DataFrame:
        """
        Create a sample dataset for demonstration purposes.
        
        Args:
            size (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Sample dataset
        """
        logger.info("Creating sample dataset...")
        
        # Sample tweets with different sentiments
        positive_tweets = [
            "I love this product! It's amazing! 😍",
            "Great service and excellent quality! 👏",
            "Best purchase I've made this year! Highly recommended!",
            "Fantastic experience with customer support!",
            "This is exactly what I was looking for! Perfect!",
            "Outstanding quality and fast delivery! ⭐⭐⭐⭐⭐",
            "Really impressed with the features and performance!",
            "Absolutely love it! Worth every penny!",
            "Exceeded my expectations! Amazing product!",
            "Great value for money! Very satisfied!"
        ]
        
        negative_tweets = [
            "Terrible experience! Waste of money! 😡",
            "Poor quality and bad customer service!",
            "Not worth the price! Very disappointed!",
            "Worst purchase ever! Don't buy this!",
            "Broken after one week! Useless product!",
            "Very poor quality! Not recommended at all!",
            "Complete waste of time and money! 😤",
            "Defective product! Bad customer support!",
            "Very disappointed! Does not work as advertised!",
            "Poor quality control! Avoid this product!"
        ]
        
        neutral_tweets = [
            "The product is okay, nothing special.",
            "Average quality, meets basic expectations.",
            "It works as described, nothing more.",
            "Standard product, nothing extraordinary.",
            "Decent value for the price.",
            "Functionality is basic but sufficient.",
            "Product meets minimum requirements.",
            "Nothing impressive but works fine.",
            "Average experience overall.",
            "Typical product with standard features."
        ]
        
        # Generate balanced dataset
        tweets_per_category = size // 3
        remaining = size - (tweets_per_category * 3)
        
        data = []
        sentiments = []
        
        # Add positive tweets
        for _ in range(tweets_per_category):
            data.append(np.random.choice(positive_tweets))
            sentiments.append('positive')
        
        # Add negative tweets
        for _ in range(tweets_per_category):
            data.append(np.random.choice(negative_tweets))
            sentiments.append('negative')
        
        # Add neutral tweets
        for _ in range(tweets_per_category + remaining):
            data.append(np.random.choice(neutral_tweets))
            sentiments.append('neutral')
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': data,
            'sentiment': sentiments
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created sample dataset with {len(df)} samples")
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the dataset and return statistics.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            Dict: Validation statistics
        """
        logger.info("Validating dataset...")
        
        stats = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_text_length': df['text'].str.len().mean(),
            'max_text_length': df['text'].str.len().max(),
            'min_text_length': df['text'].str.len().min()
        }
        
        # Check for required columns
        required_columns = ['text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty texts
        empty_texts = df['text'].str.strip().eq('').sum()
        if empty_texts > 0:
            logger.warning(f"Found {empty_texts} empty text entries")
        
        logger.info(f"Dataset validation complete: {stats['total_samples']} samples")
        return stats
    
    def save_dataset(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save the processed dataset.
        
        Args:
            df (pd.DataFrame): Dataset to save
            filename (str): Output filename
        """
        file_path = os.path.join(self.data_dir, filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Dataset saved to {file_path}")


def main():
    """
    Example usage of the TwitterDataLoader.
    """
    # Initialize the data loader
    loader = TwitterDataLoader()
    
    # Load dataset (try Sentiment140 first, fallback to sample)
    try:
        df = loader.load_sentiment140(sample_size=5000)
    except Exception as e:
        logger.warning(f"Could not load Sentiment140: {e}")
        df = loader._create_sample_dataset(1000)
    
    # Validate dataset
    stats = loader.validate_dataset(df)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save processed dataset
    loader.save_dataset(df, 'processed_twitter_sentiment.csv')
    
    return df


if __name__ == "__main__":
    main()
