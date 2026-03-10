# Twitter Sentiment Analysis Project

A comprehensive machine learning project for Twitter sentiment analysis that implements research-based methodology for consumer sentiment prediction and trend analysis.

## 📋 Project Overview

This project implements a complete sentiment analysis pipeline following academic research standards. It includes data preprocessing, feature extraction, multiple machine learning models, comprehensive evaluation, and advanced text analysis techniques.

### 🎯 Project Goals

- Build a Twitter sentiment analysis system that predicts consumer sentiment from tweet text
- Analyze sentiment trends and patterns
- Implement research-based methodology for text classification
- Provide comprehensive evaluation and visualization capabilities
- Create an interactive demo for real-time sentiment prediction

### 🏗️ Project Structure

```
sentiment-analysis-project/
│
├── dataset/                    # Datasets and data files
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Source code modules
│   ├── data_loader.py          # Data collection and loading
│   ├── preprocessing.py        # Text preprocessing pipeline
│   ├── feature_engineering.py  # Feature extraction (TF-IDF, Word2Vec)
│   ├── models.py               # Machine learning models
│   ├── evaluation.py           # Model evaluation metrics
│   ├── visualization.py        # Data visualization
│   └── advanced_analysis.py    # Advanced NLP analysis
├── app/                        # Demo application
│   └── sentiment_predictor.py  # Interactive prediction interface
├── models/                     # Trained models and artifacts
├── results/                    # Evaluation results and visualizations
├── main.py                     # Main pipeline script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd sentiment-analysis-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download additional NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

4. **Download spaCy model (optional, for advanced features)**
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the Pipeline

#### Option 1: Run Complete Pipeline
```bash
python main.py --mode full
```

#### Option 2: Run Specific Components
```bash
# Train models only
python main.py --mode train

# Evaluate models only
python main.py --mode evaluate

# Run demo application
python main.py --mode demo

# Advanced analysis only
python main.py --mode analyze
```

#### Option 3: Interactive Demo
```bash
python app/sentiment_predictor.py
```

## 📊 Features

### 🔄 Data Pipeline

1. **Data Collection**
   - Sentiment140 dataset integration
   - Custom dataset support
   - Data validation and statistics

2. **Text Preprocessing**
   - Lowercase conversion
   - URL removal
   - Mention and hashtag removal
   - Punctuation removal
   - Stopword removal
   - Tokenization
   - Lemmatization
   - Emoji handling
   - Contraction expansion

3. **Feature Extraction**
   - TF-IDF vectorization
   - Word2Vec embeddings
   - Statistical features (text length, punctuation counts)
   - Sentiment lexicon features

### 🤖 Machine Learning Models

- **Naive Bayes** (Multinomial)
- **Support Vector Machine** (SVM)
- **Logistic Regression**
- **Random Forest**
- **LSTM Neural Network** (if TensorFlow available)

### 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC Curves and AUC
- Cross-validation scores
- Error analysis
- Model comparison

### 🎨 Visualizations

- Sentiment distribution plots
- Word clouds for different sentiments
- Model performance comparison charts
- Topic modeling visualizations
- Interactive dashboards
- Sentiment timeline analysis

### 🔬 Advanced Analysis

- **Named Entity Recognition** (NER)
- **Brand/Product Identification**
- **Topic Modeling** (LDA)
- **Semantic Similarity Analysis**
- **Aspect-based Sentiment Analysis**
- **Emotion Detection**
- **Sentiment Trend Analysis**

## 📋 Detailed Documentation

### Data Loading (`src/data_loader.py`)

The `TwitterDataLoader` class handles:
- Loading Sentiment140 dataset
- Custom dataset loading (CSV, JSON, Excel)
- Data validation and statistics
- Sample dataset generation for testing

```python
from src.data_loader import TwitterDataLoader

loader = TwitterDataLoader()
df = loader.load_sentiment140(sample_size=5000)
stats = loader.validate_dataset(df)
```

### Text Preprocessing (`src/preprocessing.py`)

The `TwitterTextPreprocessor` class implements comprehensive text cleaning:

```python
from src.preprocessing import TwitterTextPreprocessor

preprocessor = TwitterTextPreprocessor(
    remove_urls=True,
    remove_mentions=True,
    remove_hashtags=True,
    use_lemmatization=True
)

processed_df = preprocessor.preprocess_dataframe(df)
```

### Feature Extraction (`src/feature_engineering.py`)

The `FeatureExtractor` class provides multiple feature types:

```python
from src.feature_engineering import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_all_features(
    texts,
    use_tfidf=True,
    use_word2vec=True,
    use_statistical=True
)
```

### Model Training (`src/models.py`)

The `SentimentClassifier` class handles model training and prediction:

```python
from src.models import SentimentClassifier

classifier = SentimentClassifier()
results = classifier.train_all_models(X_train, texts_train, y_train)

# Make predictions
predictions = classifier.predict('svm_rbf', X_test)
probabilities = classifier.predict_proba('svm_rbf', X_test)
```

### Model Evaluation (`src/evaluation.py`)

The `ModelEvaluator` class provides comprehensive evaluation:

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(y_test, predictions, probabilities, 'SVM')
comparison_df = evaluator.compare_models(model_results)
```

### Visualization (`src/visualization.py`)

The `SentimentVisualizer` class creates various visualizations:

```python
from src.visualization import SentimentVisualizer

visualizer = SentimentVisualizer()
visualizer.plot_sentiment_distribution(df)
visualizer.create_sentiment_word_clouds(df)
visualizer.plot_model_performance_comparison(comparison_df)
```

### Advanced Analysis (`src/advanced_analysis.py`)

The `AdvancedTextAnalyzer` class implements advanced NLP techniques:

```python
from src.advanced_analysis import AdvancedTextAnalyzer

analyzer = AdvancedTextAnalyzer()
results = analyzer.generate_comprehensive_report(texts, sentiments, df)
```

## 🎯 Demo Application

The interactive demo application (`app/sentiment_predictor.py`) provides:

- Single tweet sentiment prediction
- Model comparison
- Batch prediction
- Prediction statistics
- Export functionality

### Demo Features

1. **Single Prediction**
   - Input tweet text
   - Select model
   - Get sentiment prediction with confidence scores

2. **Model Comparison**
   - Compare predictions from all models
   - See confidence scores for each model

3. **Batch Processing**
   - Process multiple tweets at once
   - Export results to CSV

4. **Statistics**
   - View prediction history
   - Analyze sentiment distribution
   - Track model performance

## 📊 Results and Outputs

After running the pipeline, you'll find:

### In `dataset/`:
- `raw_dataset.csv` - Original dataset
- `processed_dataset.csv` - Preprocessed dataset

### In `models/`:
- Trained model files (.pkl, .h5)
- Feature extractors
- Label encoders
- Feature names

### In `results/`:
- `evaluation_report.txt` - Comprehensive evaluation summary
- `model_comparison.csv` - Model performance comparison
- `confusion_matrix_*.png` - Confusion matrix plots
- `wordcloud_*.png` - Word clouds for each sentiment
- `model_performance_comparison.png` - Performance charts
- `interactive_dashboard.html` - Interactive dashboard
- Advanced analysis results (NER, topics, emotions)

## 🔧 Configuration

You can modify the pipeline behavior by editing the configuration in `main.py`:

```python
config = {
    'data': {
        'sample_size': 5000,
        'test_size': 0.2,
        'random_state': 42
    },
    'preprocessing': {
        'remove_urls': True,
        'remove_mentions': True,
        'use_lemmatization': True
    },
    'features': {
        'max_features': 5000,
        'use_tfidf': True,
        'use_word2vec': True
    },
    'models': {
        'train_naive_bayes': True,
        'train_svm': True,
        'train_lstm': True
    }
}
```

## 📚 Research Methodology

This project follows established research methodology for sentiment analysis:

1. **Data Collection**: Using benchmark datasets (Sentiment140)
2. **Preprocessing**: Standard text cleaning for social media content
3. **Feature Engineering**: Multiple feature types for comprehensive representation
4. **Model Selection**: Various algorithms from classical to deep learning
5. **Evaluation**: Standard metrics with cross-validation
6. **Analysis**: Advanced NLP techniques for deeper insights

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')
   ```

3. **spaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Memory Issues**
   - Reduce sample size in configuration
   - Disable Word2Vec features
   - Use smaller TF-IDF max_features

5. **TensorFlow Issues**
   - Install CPU version: `pip install tensorflow-cpu`
   - Or disable LSTM training in configuration

### Performance Tips

- For faster training, reduce dataset size
- Disable Word2Vec if not needed
- Use CPU version of TensorFlow for LSTM
- Reduce TF-IDF max_features

## 📈 Model Performance

Typical performance on Sentiment140 dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|---------|----------|
| SVM | 0.78-0.82 | 0.77-0.81 | 0.78-0.82 | 0.77-0.81 |
| Random Forest | 0.75-0.79 | 0.74-0.78 | 0.75-0.79 | 0.74-0.78 |
| Naive Bayes | 0.70-0.75 | 0.69-0.74 | 0.70-0.75 | 0.69-0.74 |
| Logistic Regression | 0.76-0.80 | 0.75-0.79 | 0.76-0.80 | 0.75-0.79 |
| LSTM | 0.78-0.83 | 0.77-0.82 | 0.78-0.83 | 0.77-0.82 |

*Note: Performance varies based on dataset size, preprocessing, and feature selection.*

## 🔬 Advanced Features

### Aspect-Based Sentiment Analysis

Identify sentiment towards specific aspects (price, quality, service, etc.):

```python
aspect_results = analyzer.aspect_based_sentiment_analysis(texts, sentiments)
```

### Named Entity Recognition

Extract entities like people, organizations, locations:

```python
entities = analyzer.extract_named_entities(texts)
```

### Topic Modeling

Discover hidden topics in the dataset:

```python
topics = analyzer.perform_topic_modeling(texts, n_topics=5)
```

### Emotion Detection

Detect emotions beyond positive/negative/neutral:

```python
emotions = analyzer.detect_emotions(texts)
```

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@misc{twitter_sentiment_analysis,
  title={Twitter Sentiment Analysis Project},
  author={Your Name},
  year={2024},
  description={Comprehensive sentiment analysis pipeline following research methodology}
}
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Sentiment140 dataset providers
- NLTK, spaCy, and scikit-learn communities
- Research papers on sentiment analysis methodology
- Open-source NLP libraries

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with details
4. Check existing issues for solutions

---

**Happy Sentiment Analysis! 🎉**
