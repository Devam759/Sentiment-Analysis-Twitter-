"""
Text Preprocessing Module for Twitter Sentiment Analysis

This module implements comprehensive text preprocessing steps specifically
designed for Twitter data, following research paper methodology.

Preprocessing steps include:
- Lowercase conversion
- URL removal
- Mention and hashtag removal
- Punctuation removal
- Stopword removal
- Tokenization
- Optional stemming/lemmatization
- Emoji handling
- Contractions expansion
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TwitterTextPreprocessor:
    """
    Comprehensive text preprocessor for Twitter sentiment analysis.
    Implements all standard preprocessing steps for social media text.
    """
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = True,
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = True,
                 use_stemming: bool = False,
                 use_lemmatization: bool = True,
                 remove_emojis: bool = False,
                 expand_contractions_enabled: bool = True,
                 min_word_length: int = 2):
        """
        Initialize the preprocessor with specified options.
        
        Args:
            remove_urls (bool): Whether to remove URLs
            remove_mentions (bool): Whether to remove @mentions
            remove_hashtags (bool): Whether to remove #hashtags
            remove_punctuation (bool): Whether to remove punctuation
            remove_stopwords (bool): Whether to remove stopwords
            use_stemming (bool): Whether to apply stemming
            use_lemmatization (bool): Whether to apply lemmatization
            remove_emojis (bool): Whether to remove emojis
            expand_contractions_enabled (bool): Whether to expand contractions
            min_word_length (int): Minimum word length to keep
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_emojis = remove_emojis
        self.should_expand_contractions = expand_contractions_enabled
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common contractions dictionary
        self.contractions_dict = {
            "aren't": "are not", "can't": "cannot", "could've": "could have",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
            "how's": "how is", "i'd": "I would", "i'd've": "I would have",
            "i'll": "I will", "i'm": "I am", "i've": "I have", "isn't": "is not",
            "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
            "it's": "it is", "let's": "let us", "ma'am": "madam", "might've": "might have",
            "mightn't": "might not", "must've": "must have", "mustn't": "must not",
            "needn't": "need not", "o'clock": "of the clock", "shan't": "shall not",
            "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
            "she's": "she is", "should've": "should have", "shouldn't": "should not",
            "that'd": "that would", "that'd've": "that would have", "that's": "that is",
            "there'd": "there would", "there'd've": "there would have", "there's": "there is",
            "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
            "they're": "they are", "they've": "they have", "wasn't": "was not",
            "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what'll": "what will", "what're": "what are", "what's": "what is",
            "what've": "what have", "when's": "when is", "when've": "when have",
            "where'd": "where did", "where's": "where is", "where've": "where have",
            "who'd": "who would", "who'd've": "who would have", "who'll": "who will",
            "who's": "who is", "who've": "who have", "why'd": "why did",
            "why're": "why are", "why's": "why is", "why've": "why have",
            "won't": "will not", "would've": "would have", "wouldn't": "would not",
            "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
            "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
            "you'd've": "you would have", "you'll": "you will", "you're": "you are",
            "you've": "you have", "u": "you", "r": "are", "ur": "your", "lol": "laugh out loud",
            "lmao": "laugh my ass off", "rofl": "rolling on floor laughing", "btw": "by the way",
            "idk": "I don't know", "imo": "in my opinion", "imho": "in my humble opinion",
            "smh": "shaking my head", "fyi": "for your information", "tbh": "to be honest"
        }
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        self.punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
        
        logger.info("TwitterTextPreprocessor initialized")
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand common English contractions.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with expanded contractions
        """
        if not self.should_expand_contractions:
            return text
            
        # Create pattern for contractions
        contractions_pattern = re.compile('({})'.format('|'.join(self.contractions_dict.keys())), 
                                        flags=re.IGNORECASE|re.DOTALL)
        
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = self.contractions_dict.get(match.lower(), match)
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction
        
        expanded_text = contractions_pattern.sub(expand_match, text)
        return expanded_text
    
    def remove_urls_mentions_hashtags(self, text: str) -> str:
        """
        Remove URLs, mentions, and hashtags from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags (keep the text, remove the # symbol)
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        
        return text.strip()
    
    def handle_emojis(self, text: str) -> str:
        """
        Handle emojis in text - either remove or convert to text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with emojis handled
        """
        if self.remove_emojis:
            return self.emoji_pattern.sub('', text)
        else:
            # Convert emojis to text descriptions (simplified)
            # This is a basic implementation - could be enhanced with emoji library
            return text
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize text and apply cleaning steps.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of cleaned tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove URLs, mentions, hashtags
        text = self.remove_urls_mentions_hashtags(text)
        
        # Handle emojis
        text = self.handle_emojis(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self.punctuation_pattern.sub('', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        if self.remove_stopwords:
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) >= self.min_word_length]
        else:
            tokens = [token for token in tokens if len(token) >= self.min_word_length]
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        tokens = self.tokenize_and_clean(text)
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess text in a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text
        """
        logger.info(f"Preprocessing {len(df)} texts...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Apply preprocessing
        tqdm.pandas(desc="Preprocessing")
        df_processed['processed_text'] = df_processed[text_column].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        df_processed = df_processed[df_processed['processed_text'].str.strip() != '']
        
        logger.info(f"Preprocessing complete. {len(df_processed)} texts remaining.")
        
        return df_processed
    
    def get_preprocessing_stats(self, df_original: pd.DataFrame, 
                              df_processed: pd.DataFrame) -> Dict[str, any]:
        """
        Get statistics about the preprocessing process.
        
        Args:
            df_original (pd.DataFrame): Original DataFrame
            df_processed (pd.DataFrame): Processed DataFrame
            
        Returns:
            Dict: Preprocessing statistics
        """
        stats = {
            'original_count': len(df_original),
            'processed_count': len(df_processed),
            'texts_removed': len(df_original) - len(df_processed),
            'removal_rate': (len(df_original) - len(df_processed)) / len(df_original) * 100,
            'avg_original_length': df_original['text'].str.len().mean(),
            'avg_processed_length': df_processed['processed_text'].str.len().mean(),
            'length_reduction': (df_original['text'].str.len().mean() - 
                              df_processed['processed_text'].str.len().mean()) / 
                              df_original['text'].str.len().mean() * 100
        }
        
        return stats


def main():
    """
    Example usage of the TwitterTextPreprocessor.
    """
    # Sample data
    sample_texts = [
        "I love this product! It's amazing! 😍 Check it out at https://example.com #bestproduct",
        "Terrible experience! Waste of money! 😡 Don't buy this @company",
        "It's okay, nothing special... #average",
        "Great service and excellent quality! 👏 Highly recommended!",
        "Poor quality and bad customer service! Very disappointed!"
    ]
    
    df = pd.DataFrame({
        'text': sample_texts,
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
    })
    
    print("Original texts:")
    for i, text in enumerate(df['text']):
        print(f"{i+1}. {text}")
    
    # Initialize preprocessor
    preprocessor = TwitterTextPreprocessor(
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        remove_punctuation=True,
        remove_stopwords=True,
        use_lemmatization=True,
        remove_emojis=True
    )
    
    # Preprocess
    df_processed = preprocessor.preprocess_dataframe(df)
    
    print("\nProcessed texts:")
    for i, text in enumerate(df_processed['processed_text']):
        print(f"{i+1}. {text}")
    
    # Get statistics
    stats = preprocessor.get_preprocessing_stats(df, df_processed)
    print("\nPreprocessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
