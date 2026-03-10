"""
Advanced Analysis Module for Twitter Sentiment Analysis

This module implements advanced NLP analysis techniques:
- Named Entity Recognition (NER) to identify brands/products
- Topic modeling using LDA
- Semantic similarity analysis
- Sentiment trend analysis
- Aspect-based sentiment analysis
- Emotion detection
- Text summarization

Following research methodology for comprehensive text analysis.
"""

import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# NLP imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. NER and advanced features will be limited.")

try:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some features will be disabled.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Some sentiment features will be disabled.")

try:
    import gensim
    from gensim.models import LdaModel, Word2Vec
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logging.warning("Gensim not available. Advanced topic modeling will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTextAnalyzer:
    """
    Advanced text analysis toolkit for sentiment analysis research.
    """
    
    def __init__(self, output_dir: str = "../results", spacy_model: str = "en_core_web_sm"):
        """
        Initialize the advanced analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
            spacy_model (str): Name of spaCy model to use
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize NLP components
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(f"spaCy model {spacy_model} not found. NER will be disabled.")
                logger.info("To install: python -m spacy download en_core_web_sm")
        
        # Brand/product keywords for targeted analysis
        self.brand_keywords = {
            'apple': ['apple', 'iphone', 'ipad', 'macbook', 'ios', 'app store'],
            'google': ['google', 'android', 'pixel', 'gmail', 'chrome', 'youtube'],
            'microsoft': ['microsoft', 'windows', 'office', 'xbox', 'surface', 'azure'],
            'amazon': ['amazon', 'aws', 'kindle', 'prime', 'alexa', 'echo'],
            'tesla': ['tesla', 'model s', 'model 3', 'model x', 'elon musk', 'electric car'],
            'netflix': ['netflix', 'streaming', 'binge watch', 'original series'],
            'facebook': ['facebook', 'meta', 'instagram', 'whatsapp', 'zuckerberg'],
            'twitter': ['twitter', 'x', 'tweet', 'retweet', 'elon', 'blue check'],
            'samsung': ['samsung', 'galaxy', 'smartphone', 'tv', 'electronics']
        }
        
        # Aspect keywords for aspect-based sentiment analysis
        self.aspect_keywords = {
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money'],
            'quality': ['quality', 'build', 'material', 'durable', 'sturdy', 'well-made'],
            'service': ['service', 'support', 'customer', 'help', 'staff', 'assistance'],
            'features': ['features', 'functionality', 'capabilities', 'options', 'settings'],
            'design': ['design', 'look', 'appearance', 'style', 'aesthetic', 'color'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'responsive', 'lag'],
            'battery': ['battery', 'charge', 'battery life', 'power', 'lasting'],
            'usability': ['easy', 'simple', 'intuitive', 'user-friendly', 'complicated']
        }
        
        logger.info("AdvancedTextAnalyzer initialized")
    
    def extract_named_entities(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract named entities from texts using spaCy.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            pd.DataFrame: Named entities analysis
        """
        if self.nlp is None:
            logger.warning("spaCy not available. Cannot extract named entities.")
            return pd.DataFrame()
        
        logger.info("Extracting named entities...")
        
        entities_data = []
        
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities_data.append({
                    'text_index': i,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'entity': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_)
                })
        
        entities_df = pd.DataFrame(entities_data)
        
        # Analysis summary
        if not entities_df.empty:
            entity_summary = entities_df['label'].value_counts().to_frame('count')
            entity_summary['description'] = entity_summary.index.map(lambda x: spacy.explain(x))
            
            # Save results
            entities_df.to_csv(os.path.join(self.output_dir, 'named_entities.csv'), index=False)
            entity_summary.to_csv(os.path.join(self.output_dir, 'entity_summary.csv'))
            
            logger.info(f"Extracted {len(entities_df)} named entities")
        
        return entities_df
    
    def identify_brands_products(self, texts: List[str]) -> pd.DataFrame:
        """
        Identify brands and products mentioned in texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            pd.DataFrame: Brand/product mentions analysis
        """
        logger.info("Identifying brands and products...")
        
        brand_mentions = []
        
        for i, text in enumerate(tqdm(texts, desc="Analyzing brands")):
            text_lower = text.lower()
            
            for brand, keywords in self.brand_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        brand_mentions.append({
                            'text_index': i,
                            'brand': brand,
                            'keyword_found': keyword,
                            'text': text[:100] + '...' if len(text) > 100 else text
                        })
        
        brand_df = pd.DataFrame(brand_mentions)
        
        if not brand_df.empty:
            # Analysis summary
            brand_summary = brand_df['brand'].value_counts().to_frame('mentions')
            brand_summary['percentage'] = (brand_summary['mentions'] / len(texts)) * 100
            
            # Save results
            brand_df.to_csv(os.path.join(self.output_dir, 'brand_mentions.csv'), index=False)
            brand_summary.to_csv(os.path.join(self.output_dir, 'brand_summary.csv'))
            
            logger.info(f"Found {len(brand_df)} brand/product mentions")
        
        return brand_df
    
    def perform_topic_modeling(self, texts: List[str], n_topics: int = 5,
                             method: str = 'lda', max_features: int = 1000) -> Dict[str, Any]:
        """
        Perform topic modeling using LDA.
        
        Args:
            texts (List[str]): List of texts to analyze
            n_topics (int): Number of topics
            method (str): Topic modeling method ('lda' or 'gensim_lda')
            max_features (int): Maximum features for vectorization
            
        Returns:
            Dict: Topic modeling results
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Cannot perform topic modeling.")
            return {}
        
        logger.info(f"Performing topic modeling with {n_topics} topics...")
        
        results = {}
        
        if method == 'lda':
            # Use scikit-learn LDA
            vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights
                })
            
            results = {
                'method': 'lda',
                'n_topics': n_topics,
                'topics': topics,
                'feature_names': feature_names,
                'model': lda,
                'vectorizer': vectorizer
            }
        
        elif method == 'gensim_lda' and GENSIM_AVAILABLE:
            # Use Gensim LDA
            tokenized_texts = [text.split() for text in texts]
            dictionary = Dictionary(tokenized_texts)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42)
            
            topics = []
            for topic_idx in range(n_topics):
                topic_words = lda_model.show_topic(topic_idx, topn=10)
                topics.append({
                    'topic_id': topic_idx,
                    'words': [word for word, weight in topic_words],
                    'weights': [weight for word, weight in topic_words]
                })
            
            results = {
                'method': 'gensim_lda',
                'n_topics': n_topics,
                'topics': topics,
                'model': lda_model,
                'dictionary': dictionary,
                'corpus': corpus
            }
        
        # Save results
        topics_df = pd.DataFrame([
            {'topic_id': topic['topic_id'], 'words': ', '.join(topic['words'])}
            for topic in results['topics']
        ])
        topics_df.to_csv(os.path.join(self.output_dir, 'topics.csv'), index=False)
        
        logger.info(f"Topic modeling completed: {len(results['topics'])} topics found")
        
        return results
    
    def analyze_semantic_similarity(self, texts: List[str], sample_size: int = 100) -> pd.DataFrame:
        """
        Analyze semantic similarity between texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            sample_size (int): Number of texts to sample for analysis
            
        Returns:
            pd.DataFrame: Semantic similarity analysis
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Cannot analyze semantic similarity.")
            return pd.DataFrame()
        
        logger.info("Analyzing semantic similarity...")
        
        # Sample texts if needed
        if len(texts) > sample_size:
            sample_indices = np.random.choice(len(texts), sample_size, replace=False)
            sample_texts = [texts[i] for i in sample_indices]
        else:
            sample_texts = texts
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find most similar pairs
        similarity_data = []
        n_texts = len(sample_texts)
        
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                similarity_data.append({
                    'text1_index': i,
                    'text2_index': j,
                    'similarity': similarity_matrix[i, j],
                    'text1': sample_texts[i][:100] + '...' if len(sample_texts[i]) > 100 else sample_texts[i],
                    'text2': sample_texts[j][:100] + '...' if len(sample_texts[j]) > 100 else sample_texts[j]
                })
        
        similarity_df = pd.DataFrame(similarity_data)
        
        # Sort by similarity
        similarity_df = similarity_df.sort_values('similarity', ascending=False)
        
        # Save results
        similarity_df.to_csv(os.path.join(self.output_dir, 'semantic_similarity.csv'), index=False)
        
        logger.info(f"Semantic similarity analysis completed for {len(sample_texts)} texts")
        
        return similarity_df
    
    def aspect_based_sentiment_analysis(self, texts: List[str], sentiments: List[str]) -> pd.DataFrame:
        """
        Perform aspect-based sentiment analysis.
        
        Args:
            texts (List[str]): List of texts to analyze
            sentiments (List[str]): Corresponding sentiment labels
            
        Returns:
            pd.DataFrame: Aspect-based sentiment analysis
        """
        logger.info("Performing aspect-based sentiment analysis...")
        
        aspect_data = []
        
        for i, (text, sentiment) in enumerate(tqdm(zip(texts, sentiments), desc="Analyzing aspects")):
            text_lower = text.lower()
            
            for aspect, keywords in self.aspect_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        aspect_data.append({
                            'text_index': i,
                            'aspect': aspect,
                            'keyword_found': keyword,
                            'sentiment': sentiment,
                            'text': text[:100] + '...' if len(text) > 100 else text
                        })
        
        aspect_df = pd.DataFrame(aspect_data)
        
        if not aspect_df.empty:
            # Aspect-sentiment analysis
            aspect_sentiment = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
            
            # Calculate sentiment percentages for each aspect
            aspect_percentages = aspect_sentiment.div(aspect_sentiment.sum(axis=1), axis=0) * 100
            
            # Save results
            aspect_df.to_csv(os.path.join(self.output_dir, 'aspect_mentions.csv'), index=False)
            aspect_sentiment.to_csv(os.path.join(self.output_dir, 'aspect_sentiment_counts.csv'))
            aspect_percentages.to_csv(os.path.join(self.output_dir, 'aspect_sentiment_percentages.csv'))
            
            logger.info(f"Aspect-based analysis completed: {len(aspect_df)} aspect mentions found")
        
        return aspect_df
    
    def detect_emotions(self, texts: List[str]) -> pd.DataFrame:
        """
        Detect emotions in texts using TextBlob.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            pd.DataFrame: Emotion analysis results
        """
        if not TEXTBLOB_AVAILABLE:
            logger.warning("TextBlob not available. Cannot detect emotions.")
            return pd.DataFrame()
        
        logger.info("Detecting emotions...")
        
        emotion_data = []
        
        # Simple emotion keyword mapping
        emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'pleased', 'satisfied', 'glad'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'fearful', 'anxious', 'worried', 'nervous'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'disappointed', 'upset'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened']
        }
        
        for i, text in enumerate(tqdm(texts, desc="Detecting emotions")):
            text_lower = text.lower()
            
            # Get sentiment from TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Detect emotions based on keywords
            detected_emotions = []
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected_emotions.append(emotion)
            
            if detected_emotions:
                for emotion in detected_emotions:
                    emotion_data.append({
                        'text_index': i,
                        'emotion': emotion,
                        'polarity': polarity,
                        'subjectivity': subjectivity,
                        'text': text[:100] + '...' if len(text) > 100 else text
                    })
            else:
                # Assign emotion based on polarity if no keywords found
                if polarity > 0.3:
                    primary_emotion = 'joy'
                elif polarity < -0.3:
                    primary_emotion = 'sadness'
                else:
                    primary_emotion = 'neutral'
                
                emotion_data.append({
                    'text_index': i,
                    'emotion': primary_emotion,
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'text': text[:100] + '...' if len(text) > 100 else text
                })
        
        emotion_df = pd.DataFrame(emotion_data)
        
        if not emotion_df.empty:
            # Emotion analysis summary
            emotion_summary = emotion_df['emotion'].value_counts().to_frame('count')
            emotion_summary['percentage'] = (emotion_summary['count'] / len(texts)) * 100
            
            # Average polarity and subjectivity by emotion
            emotion_stats = emotion_df.groupby('emotion')[['polarity', 'subjectivity']].mean()
            
            # Save results
            emotion_df.to_csv(os.path.join(self.output_dir, 'emotion_analysis.csv'), index=False)
            emotion_summary.to_csv(os.path.join(self.output_dir, 'emotion_summary.csv'))
            emotion_stats.to_csv(os.path.join(self.output_dir, 'emotion_statistics.csv'))
            
            logger.info(f"Emotion detection completed: {len(emotion_df)} emotion instances found")
        
        return emotion_df
    
    def analyze_sentiment_trends(self, df: pd.DataFrame, date_column: str = 'date',
                               sentiment_column: str = 'sentiment',
                               text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment trends over time.
        
        Args:
            df (pd.DataFrame): DataFrame with date, sentiment, and text columns
            date_column (str): Name of the date column
            sentiment_column (str): Name of the sentiment column
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: Sentiment trend analysis
        """
        logger.info("Analyzing sentiment trends...")
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and sentiment
        trend_data = df.groupby([df[date_column].dt.date, sentiment_column]).size().unstack(fill_value=0)
        
        # Calculate daily totals and percentages
        trend_data['total'] = trend_data.sum(axis=1)
        
        for sentiment in trend_data.columns:
            if sentiment != 'total':
                trend_data[f'{sentiment}_percentage'] = (trend_data[sentiment] / trend_data['total']) * 100
        
        # Calculate rolling averages (7-day and 30-day)
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in trend_data.columns:
                trend_data[f'{sentiment}_7day_avg'] = trend_data[sentiment].rolling(window=7, min_periods=1).mean()
                trend_data[f'{sentiment}_30day_avg'] = trend_data[sentiment].rolling(window=30, min_periods=1).mean()
        
        # Trend analysis metrics
        trend_analysis = []
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in trend_data.columns:
                # Calculate trend direction (simple linear trend)
                sentiment_values = trend_data[sentiment].values
                x = np.arange(len(sentiment_values))
                
                # Simple linear regression to detect trend
                if len(x) > 1:
                    slope = np.polyfit(x, sentiment_values, 1)[0]
                    trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                else:
                    trend_direction = 'insufficient_data'
                
                # Calculate volatility
                volatility = sentiment_values.std() / sentiment_values.mean() if sentiment_values.mean() > 0 else 0
                
                trend_analysis.append({
                    'sentiment': sentiment,
                    'trend_direction': trend_direction,
                    'slope': slope if len(x) > 1 else 0,
                    'volatility': volatility,
                    'avg_daily_count': sentiment_values.mean(),
                    'max_daily_count': sentiment_values.max(),
                    'min_daily_count': sentiment_values.min()
                })
        
        trend_analysis_df = pd.DataFrame(trend_analysis)
        
        # Save results
        trend_data.to_csv(os.path.join(self.output_dir, 'sentiment_trends.csv'))
        trend_analysis_df.to_csv(os.path.join(self.output_dir, 'trend_analysis.csv'))
        
        logger.info("Sentiment trend analysis completed")
        
        return trend_analysis_df
    
    def generate_comprehensive_report(self, texts: List[str], sentiments: List[str],
                                    df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive advanced analysis report.
        
        Args:
            texts (List[str]): List of texts to analyze
            sentiments (List[str]): Corresponding sentiment labels
            df (Optional[pd.DataFrame]): DataFrame with additional columns
            
        Returns:
            Dict: Comprehensive analysis results
        """
        logger.info("Generating comprehensive advanced analysis report...")
        
        results = {}
        
        # Named Entity Recognition
        ner_results = self.extract_named_entities(texts)
        if not ner_results.empty:
            results['named_entities'] = ner_results
        
        # Brand/Product Identification
        brand_results = self.identify_brands_products(texts)
        if not brand_results.empty:
            results['brand_mentions'] = brand_results
        
        # Topic Modeling
        topic_results = self.perform_topic_modeling(texts)
        if topic_results:
            results['topic_modeling'] = topic_results
        
        # Semantic Similarity
        similarity_results = self.analyze_semantic_similarity(texts)
        if not similarity_results.empty:
            results['semantic_similarity'] = similarity_results
        
        # Aspect-based Sentiment Analysis
        aspect_results = self.aspect_based_sentiment_analysis(texts, sentiments)
        if not aspect_results.empty:
            results['aspect_sentiment'] = aspect_results
        
        # Emotion Detection
        emotion_results = self.detect_emotions(texts)
        if not emotion_results.empty:
            results['emotion_detection'] = emotion_results
        
        # Sentiment Trends (if date column available)
        if df is not None and 'date' in df.columns:
            trend_results = self.analyze_sentiment_trends(df)
            if not trend_results.empty:
                results['sentiment_trends'] = trend_results
        
        # Save summary report
        summary_text = self._generate_summary_text(results)
        with open(os.path.join(self.output_dir, 'advanced_analysis_summary.txt'), 'w') as f:
            f.write(summary_text)
        
        logger.info("Comprehensive advanced analysis report completed")
        
        return results
    
    def _generate_summary_text(self, results: Dict[str, Any]) -> str:
        """
        Generate a summary text of the analysis results.
        
        Args:
            results (Dict[str, Any]): Analysis results
            
        Returns:
            str: Summary text
        """
        summary = []
        summary.append("Advanced Text Analysis Summary Report")
        summary.append("=" * 50)
        summary.append("")
        
        # Named Entities
        if 'named_entities' in results:
            ner_df = results['named_entities']
            summary.append(f"Named Entity Recognition:")
            summary.append(f"  - Total entities found: {len(ner_df)}")
            summary.append(f"  - Unique entity types: {ner_df['label'].nunique()}")
            summary.append(f"  - Most common entity type: {ner_df['label'].mode().iloc[0]}")
            summary.append("")
        
        # Brand Mentions
        if 'brand_mentions' in results:
            brand_df = results['brand_mentions']
            summary.append(f"Brand/Product Mentions:")
            summary.append(f"  - Total mentions: {len(brand_df)}")
            summary.append(f"  - Unique brands mentioned: {brand_df['brand'].nunique()}")
            if not brand_df.empty:
                top_brand = brand_df['brand'].value_counts().index[0]
                summary.append(f"  - Most mentioned brand: {top_brand}")
            summary.append("")
        
        # Topic Modeling
        if 'topic_modeling' in results:
            topic_data = results['topic_modeling']
            summary.append(f"Topic Modeling:")
            summary.append(f"  - Number of topics: {topic_data['n_topics']}")
            summary.append(f"  - Method used: {topic_data['method']}")
            summary.append("")
        
        # Aspect-based Analysis
        if 'aspect_sentiment' in results:
            aspect_df = results['aspect_sentiment']
            summary.append(f"Aspect-based Sentiment Analysis:")
            summary.append(f"  - Total aspect mentions: {len(aspect_df)}")
            summary.append(f"  - Unique aspects: {aspect_df['aspect'].nunique()}")
            summary.append("")
        
        # Emotion Detection
        if 'emotion_detection' in results:
            emotion_df = results['emotion_detection']
            summary.append(f"Emotion Detection:")
            summary.append(f"  - Total emotion instances: {len(emotion_df)}")
            summary.append(f"  - Unique emotions: {emotion_df['emotion'].nunique()}")
            if not emotion_df.empty:
                top_emotion = emotion_df['emotion'].value_counts().index[0]
                summary.append(f"  - Most common emotion: {top_emotion}")
            summary.append("")
        
        return "\n".join(summary)


def main():
    """
    Example usage of the AdvancedTextAnalyzer.
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    
    sentiments = np.random.choice(['positive', 'negative', 'neutral'], size=n_samples)
    texts = [
        f"I {'love' if s == 'positive' else 'hate' if s == 'negative' else 'feel okay about'} my new {'iPhone' if i % 3 == 0 else 'Samsung' if i % 3 == 1 else 'Google Pixel'}! The {'battery life' if i % 4 == 0 else 'camera quality' if i % 4 == 1 else 'screen' if i % 4 == 2 else 'performance'} is {'amazing' if s == 'positive' else 'terrible' if s == 'negative' else 'decent'}."
        for i, s in enumerate(sentiments)
    ]
    
    # Create DataFrame with dates
    df = pd.DataFrame({
        'text': texts,
        'sentiment': sentiments,
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    })
    
    # Initialize analyzer
    analyzer = AdvancedTextAnalyzer()
    
    # Generate comprehensive report
    results = analyzer.generate_comprehensive_report(texts, sentiments, df)
    
    print("Advanced analysis completed successfully!")
    print(f"Analysis components generated: {list(results.keys())}")


if __name__ == "__main__":
    main()
