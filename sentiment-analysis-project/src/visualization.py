"""
Visualization Module for Twitter Sentiment Analysis

This module provides comprehensive visualization capabilities:
- Sentiment distribution plots
- Word clouds for different sentiments
- Model performance comparison charts
- Topic modeling visualizations
- Time series sentiment trends
- Feature importance plots
- Interactive plots with Plotly

Following research standards for data visualization in NLP projects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from collections import Counter
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Topic modeling imports
try:
    from sklearn.decomposition import LatentDirichletAllocation
    import pyLDAvis
    import pyLDAvis.lda_model
    LDA_AVAILABLE = True
except ImportError:
    LDA_AVAILABLE = False
    logging.warning("pyLDAvis not available. Topic visualization will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentVisualizer:
    """
    Comprehensive visualization toolkit for sentiment analysis.
    """
    
    def __init__(self, output_dir: str = "../results"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color schemes
        self.sentiment_colors = {
            'positive': '#2E8B57',
            'negative': '#DC143C', 
            'neutral': '#708090'
        }
        
        logger.info("SentimentVisualizer initialized")
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, sentiment_column: str = 'sentiment',
                                   figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot sentiment distribution with multiple chart types.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            sentiment_column (str): Name of the sentiment column
            figsize (Tuple[int, int]): Figure size
        """
        logger.info("Plotting sentiment distribution...")
        
        # Count sentiments
        sentiment_counts = df[sentiment_column].value_counts()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Sentiment Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Bar plot
        colors = [self.sentiment_colors.get(sentiment.lower(), '#3498DB') for sentiment in sentiment_counts.index]
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        axes[0, 0].set_title('Sentiment Count Distribution')
        axes[0, 0].set_xlabel('Sentiment')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                      colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Sentiment Percentage Distribution')
        
        # Box plot of text length by sentiment
        if 'text' in df.columns:
            df['text_length'] = df['text'].str.len()
            sns.boxplot(data=df, x=sentiment_column, y='text_length', ax=axes[1, 0])
            axes[1, 0].set_title('Text Length Distribution by Sentiment')
            axes[1, 0].set_ylabel('Text Length')
        
        # Histogram of text lengths
        if 'text' in df.columns:
            for sentiment in sentiment_counts.index:
                subset = df[df[sentiment_column] == sentiment]
                axes[1, 1].hist(subset['text_length'], alpha=0.6, label=sentiment, 
                               bins=20, color=self.sentiment_colors.get(sentiment.lower(), '#3498DB'))
            axes[1, 1].set_title('Text Length Histogram by Sentiment')
            axes[1, 1].set_xlabel('Text Length')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'sentiment_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Sentiment distribution plot saved")
    
    def create_word_cloud(self, texts: List[str], sentiment: str, 
                         max_words: int = 100, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Create word cloud for specific sentiment.
        
        Args:
            texts (List[str]): List of texts
            sentiment (str): Sentiment label
            max_words (int): Maximum number of words
            figsize (Tuple[int, int]): Figure size
        """
        logger.info(f"Creating word cloud for {sentiment} sentiment...")
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            max_words=max_words,
            background_color='white',
            colormap='viridis',
            stopwords=STOPWORDS,
            collocations=False
        ).generate(combined_text)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment.capitalize()} Sentiment', fontsize=16, fontweight='bold')
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'wordcloud_{sentiment.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Word cloud saved for {sentiment}")
    
    def create_sentiment_word_clouds(self, df: pd.DataFrame, text_column: str = 'text',
                                   sentiment_column: str = 'sentiment') -> None:
        """
        Create word clouds for all sentiments.
        
        Args:
            df (pd.DataFrame): DataFrame with text and sentiment
            text_column (str): Name of the text column
            sentiment_column (str): Name of the sentiment column
        """
        logger.info("Creating word clouds for all sentiments...")
        
        sentiments = df[sentiment_column].unique()
        
        for sentiment in sentiments:
            sentiment_texts = df[df[sentiment_column] == sentiment][text_column].tolist()
            if sentiment_texts:
                self.create_word_cloud(sentiment_texts, sentiment)
    
    def plot_model_performance_comparison(self, comparison_df: pd.DataFrame,
                                        figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create comprehensive model performance comparison plots.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison data
            figsize (Tuple[int, int]): Figure size
        """
        logger.info("Creating model performance comparison plots...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        axes = axes.ravel()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        if 'ROC AUC' in comparison_df.columns:
            metrics.append('ROC AUC')
        
        # Bar plots for each metric
        for i, metric in enumerate(metrics[:5]):
            if metric in comparison_df.columns:
                ax = axes[i]
                bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                            color='skyblue', edgecolor='navy', alpha=0.7)
                ax.set_title(f'{metric} by Model')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
        
        # Radar chart for overall comparison
        if len(metrics) >= 4:
            ax = axes[5]
            
            # Prepare data for radar chart
            models = comparison_df['Model'].tolist()
            angles = np.linspace(0, 2 * np.pi, len(metrics[:4]), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax = plt.subplot(2, 3, 6, projection='polar')
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                values = comparison_df[comparison_df['Model'] == model][metrics[:4]].iloc[0].tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics[:4])
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Radar Chart')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'model_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Model performance comparison plots saved")
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot feature importance for models that support it.
        
        Args:
            feature_names (List[str]): List of feature names
            importance_scores (np.ndarray): Importance scores
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size
        """
        logger.info("Plotting feature importance...")
        
        # Create DataFrame and sort
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'],
                       color='lightcoral', edgecolor='darkred')
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Feature importance plot saved")
    
    def plot_sentiment_timeline(self, df: pd.DataFrame, date_column: str = 'date',
                              sentiment_column: str = 'sentiment',
                              figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Plot sentiment trends over time.
        
        Args:
            df (pd.DataFrame): DataFrame with date and sentiment columns
            date_column (str): Name of the date column
            sentiment_column (str): Name of the sentiment column
            figsize (Tuple[int, int]): Figure size
        """
        logger.info("Plotting sentiment timeline...")
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and sentiment
        timeline_data = df.groupby([df[date_column].dt.date, sentiment_column]).size().unstack(fill_value=0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle('Sentiment Trends Over Time', fontsize=16, fontweight='bold')
        
        # Stacked area chart
        ax1 = axes[0]
        timeline_data.plot(kind='area', stacked=True, ax=ax1, 
                          color=[self.sentiment_colors.get(col.lower(), '#3498DB') 
                                for col in timeline_data.columns])
        ax1.set_title('Sentiment Distribution Over Time (Stacked)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count')
        ax1.legend(title='Sentiment')
        ax1.grid(True, alpha=0.3)
        
        # Line chart for each sentiment
        ax2 = axes[1]
        for sentiment in timeline_data.columns:
            ax2.plot(timeline_data.index, timeline_data[sentiment], 
                    marker='o', label=sentiment,
                    color=self.sentiment_colors.get(sentiment.lower(), '#3498DB'),
                    linewidth=2, markersize=4)
        ax2.set_title('Sentiment Count Trends')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Count')
        ax2.legend(title='Sentiment')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'sentiment_timeline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Sentiment timeline plot saved")
    
    def plot_topic_modeling(self, texts: List[str], n_topics: int = 5,
                          max_features: int = 1000, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Perform and visualize topic modeling using LDA.
        
        Args:
            texts (List[str]): List of texts
            n_topics (int): Number of topics
            max_features (int): Maximum features for TF-IDF
            figsize (Tuple[int, int]): Figure size
        """
        if not LDA_AVAILABLE:
            logger.warning("LDA visualization not available")
            return
        
        logger.info("Performing topic modeling...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Perform LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Topic Modeling Results', fontsize=16, fontweight='bold')
        axes = axes.ravel()
        
        # Plot top words for each topic
        top_words = 10
        for topic_idx in range(n_topics):
            if topic_idx >= len(axes):
                break
                
            # Get top words for this topic
            topic_words = lda.components_[topic_idx]
            top_indices = topic_words.argsort()[-top_words:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_scores = topic_words[top_indices]
            
            # Create horizontal bar chart
            ax = axes[topic_idx]
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_scores, color=plt.cm.Set3(topic_idx / n_topics))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Word Weight')
            ax.set_title(f'Topic {topic_idx + 1}')
            ax.invert_yaxis()
        
        # Hide unused subplots
        for i in range(n_topics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'topic_modeling.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create interactive LDA visualization if available
        try:
            vis = pyLDAvis.lda_model.prepare(lda, tfidf_matrix, vectorizer)
            pyLDAvis.save_html(vis, os.path.join(self.output_dir, 'lda_interactive.html'))
            logger.info("Interactive LDA visualization saved")
        except Exception as e:
            logger.warning(f"Could not create interactive LDA visualization: {e}")
        
        logger.info("Topic modeling visualization completed")
    
    def create_interactive_dashboard(self, df: pd.DataFrame, 
                                  text_column: str = 'text',
                                  sentiment_column: str = 'sentiment') -> None:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            text_column (str): Name of the text column
            sentiment_column (str): Name of the sentiment column
        """
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Text Length by Sentiment',
                          'Sentiment Timeline', 'Word Frequency'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Sentiment distribution
        sentiment_counts = df[sentiment_column].value_counts()
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                  name='Count', marker_color=['#2E8B57', '#DC143C', '#708090']),
            row=1, col=1
        )
        
        # Text length by sentiment
        df['text_length'] = df[text_column].str.len()
        for sentiment in df[sentiment_column].unique():
            subset = df[df[sentiment_column] == sentiment]
            fig.add_trace(
                go.Box(y=subset['text_length'], name=sentiment,
                      marker_color=self.sentiment_colors.get(sentiment.lower(), '#3498DB')),
                row=1, col=2
            )
        
        # Word frequency
        all_words = ' '.join(df[text_column]).split()
        word_freq = Counter(all_words).most_common(10)
        words, frequencies = zip(*word_freq)
        fig.add_trace(
            go.Bar(x=list(words), y=list(frequencies), name='Word Frequency',
                  marker_color='lightblue'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Twitter Sentiment Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save interactive plot
        pyo.plot(fig, filename=os.path.join(self.output_dir, 'interactive_dashboard.html'), 
                auto_open=False)
        
        logger.info("Interactive dashboard saved")
    
    def plot_word_frequency_analysis(self, texts: List[str], sentiments: List[str],
                                   top_n: int = 20, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Analyze and plot word frequencies by sentiment.
        
        Args:
            texts (List[str]): List of texts
            sentiments (List[str]): Corresponding sentiment labels
            top_n (int): Number of top words to show
            figsize (Tuple[int, int]): Figure size
        """
        logger.info("Analyzing word frequencies...")
        
        # Create DataFrame
        df = pd.DataFrame({'text': texts, 'sentiment': sentiments})
        
        # Get unique sentiments
        unique_sentiments = df['sentiment'].unique()
        n_sentiments = len(unique_sentiments)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Word Frequency Analysis by Sentiment', fontsize=16, fontweight='bold')
        axes = axes.ravel()
        
        for i, sentiment in enumerate(unique_sentiments[:4]):
            if i >= len(axes):
                break
                
            # Get texts for this sentiment
            sentiment_texts = df[df['sentiment'] == sentiment]['text']
            
            # Count word frequencies
            all_words = ' '.join(sentiment_texts).split()
            word_freq = Counter(all_words).most_common(top_n)
            words, frequencies = zip(*word_freq)
            
            # Plot
            ax = axes[i]
            bars = ax.barh(range(len(words)), frequencies, 
                          color=self.sentiment_colors.get(sentiment.lower(), '#3498DB'))
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel('Frequency')
            ax.set_title(f'Top {top_n} Words - {sentiment.capitalize()}')
            ax.invert_yaxis()
        
        # Hide unused subplots
        for i in range(n_sentiments, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'word_frequency_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Word frequency analysis plot saved")
    
    def create_comprehensive_report(self, df: pd.DataFrame, 
                                  text_column: str = 'text',
                                  sentiment_column: str = 'sentiment') -> None:
        """
        Create a comprehensive visualization report.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            text_column (str): Name of the text column
            sentiment_column (str): Name of the sentiment column
        """
        logger.info("Creating comprehensive visualization report...")
        
        # Create all visualizations
        self.plot_sentiment_distribution(df, sentiment_column)
        self.create_sentiment_word_clouds(df, text_column, sentiment_column)
        self.plot_word_frequency_analysis(df[text_column].tolist(), df[sentiment_column].tolist())
        
        # Topic modeling if enough data
        if len(df) > 100:
            self.plot_topic_modeling(df[text_column].tolist())
        
        # Interactive dashboard
        self.create_interactive_dashboard(df, text_column, sentiment_column)
        
        logger.info("Comprehensive visualization report completed")


def main():
    """
    Example usage of the SentimentVisualizer.
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    sentiments = np.random.choice(['positive', 'negative', 'neutral'], size=n_samples)
    texts = [
        f"Sample text {i} with {'good' if s == 'positive' else 'bad' if s == 'negative' else 'okay'} sentiment"
        for i, s in enumerate(sentiments)
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'sentiment': sentiments,
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    })
    
    # Initialize visualizer
    visualizer = SentimentVisualizer()
    
    # Create visualizations
    visualizer.plot_sentiment_distribution(df)
    visualizer.create_word_cloud(texts[:100], 'positive')
    visualizer.plot_word_frequency_analysis(texts, sentiments)
    
    print("Visualization examples completed successfully")


if __name__ == "__main__":
    main()
