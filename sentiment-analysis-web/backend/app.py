"""
Flask Backend for Twitter Sentiment Analysis Web Application
Connects the React frontend with the Python ML models
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import numpy as np
from datetime import datetime
import logging

# Add the sentiment analysis project to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sentiment-analysis-project', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for models and components
preprocessor = None
feature_extractor = None
classifier = None
label_encoder = None

def load_models():
    """Load the trained models and components"""
    global preprocessor, feature_extractor, classifier, label_encoder
    
    try:
        # Import the modules (this will work if the Python project is available)
        from preprocessing import TwitterTextPreprocessor
        from feature_engineering import FeatureExtractor
        from models import SentimentClassifier
        import pickle
        
        logger.info("Loading models...")
        
        # Initialize components
        preprocessor = TwitterTextPreprocessor()
        feature_extractor = FeatureExtractor()
        classifier = SentimentClassifier()
        
        # Try to load trained models (fallback to mock if not available)
        models_path = os.path.join(os.path.dirname(__file__), '..', 'sentiment-analysis-project', 'models')
        
        # Load TF-IDF vectorizer if available
        tfidf_path = os.path.join(models_path, 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                feature_extractor.tfidf_vectorizer = pickle.load(f)
            logger.info("TF-IDF vectorizer loaded")
        
        # Load Word2Vec model if available
        word2vec_path = os.path.join(models_path, 'word2vec_model.bin')
        if os.path.exists(word2vec_path):
            from gensim.models import Word2Vec
            feature_extractor.word2vec_model = Word2Vec.load(word2vec_path)
            logger.info("Word2Vec model loaded")
        
        # Load label encoder if available
        label_encoder_path = os.path.join(models_path, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            classifier.label_encoder = label_encoder
            logger.info("Label encoder loaded")
        
        # Load trained models
        model_files = [f for f in os.listdir(models_path) if f.endswith('.pkl')]
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            try:
                with open(os.path.join(models_path, model_file), 'rb') as f:
                    model_data = pickle.load(f)
                    classifier.trained_models[model_name] = model_data
                logger.info(f"Model {model_name} loaded")
            except Exception as e:
                logger.warning(f"Could not load model {model_name}: {e}")
        
        logger.info("Models loaded successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"Could not import ML modules: {e}")
        logger.info("Running in mock mode")
        return False
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def mock_predict(text):
    """Mock prediction function for development"""
    import random
    
    sentiments = ['positive', 'negative', 'neutral']
    sentiment = random.choice(sentiments)
    confidence = 0.7 + random.random() * 0.3
    
    return {
        'text': text,
        'predicted_sentiment': sentiment,
        'confidence': confidence,
        'model_used': 'mock_model',
        'timestamp': datetime.now().isoformat(),
        'probabilities': {
            'positive': confidence if sentiment == 'positive' else random.random() * 0.3,
            'negative': confidence if sentiment == 'negative' else random.random() * 0.3,
            'neutral': confidence if sentiment == 'neutral' else random.random() * 0.3,
        }
    }

def real_predict(text, model_name='svm_rbf'):
    """Real prediction using loaded models"""
    try:
        # Preprocess text
        processed_text = preprocessor.preprocess_text(text)
        
        # Extract features
        features = feature_extractor.extract_all_features(
            [processed_text],
            use_tfidf=True,
            use_word2vec=feature_extractor.word2vec_model is not None,
            use_statistical=True,
            use_lexicon=True,
            train_word2vec_if_needed=False
        )
        
        # Make prediction
        prediction = classifier.predict(model_name, features)
        prediction_proba = None
        
        try:
            prediction_proba = classifier.predict_proba(model_name, features)
        except:
            pass
        
        # Convert prediction to sentiment label
        if hasattr(classifier.label_encoder, 'inverse_transform'):
            sentiment_label = classifier.label_encoder.inverse_transform([prediction])[0]
        else:
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment_label = sentiment_map.get(prediction[0], 'unknown')
        
        # Calculate confidence
        confidence = 0.0
        if prediction_proba is not None:
            confidence = np.max(prediction_proba[0])
        
        result = {
            'text': text,
            'predicted_sentiment': sentiment_label,
            'confidence': float(confidence),
            'model_used': model_name,
            'timestamp': datetime.now().isoformat(),
            'probabilities': {}
        }
        
        if prediction_proba is not None and hasattr(classifier.label_encoder, 'classes_'):
            classes = classifier.label_encoder.classes_
            for i, cls in enumerate(classes):
                result['probabilities'][cls] = float(prediction_proba[0][i])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in real prediction: {e}")
        return mock_predict(text)

@app.route('/')
def index():
    """Serve the React app (for production)"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    """Predict sentiment for a single text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Use real prediction if models are loaded, otherwise mock
        if classifier and len(classifier.trained_models) > 0:
            result = real_predict(text)
        else:
            result = mock_predict(text)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_sentiment: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = []
        for text in texts:
            if classifier and len(classifier.trained_models) > 0:
                result = real_predict(text)
            else:
                result = mock_predict(text)
            results.append(result)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in batch_predict: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analyze-all', methods=['POST'])
def analyze_with_all_models():
    """Analyze text with all available models"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        results = {}
        
        if classifier and len(classifier.trained_models) > 0:
            # Use real models
            for model_name in classifier.trained_models.keys():
                try:
                    result = real_predict(text, model_name)
                    results[model_name] = result
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {e}")
                    results[model_name] = mock_predict(text)
        else:
            # Mock all models
            models = ['svm_rbf', 'random_forest', 'logistic_regression', 'naive_bayes_multinomial']
            for model in models:
                result = mock_predict(text)
                result['model_used'] = model
                results[model] = result
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in analyze_with_all_models: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        # Mock stats for now
        stats = {
            'total_predictions': 1250,
            'accuracy': 0.94,
            'models_trained': len(classifier.trained_models) if classifier else 4,
            'avg_processing_time': 0.15,
            'sentiment_distribution': {
                'positive': 450,
                'negative': 380,
                'neutral': 420
            }
        }
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/model-comparison')
def get_model_comparison():
    """Get model comparison data"""
    try:
        # Mock model comparison data
        comparison = [
            {'model': 'SVM', 'accuracy': 0.94, 'precision': 0.93, 'recall': 0.95, 'f1_score': 0.94},
            {'model': 'Random Forest', 'accuracy': 0.92, 'precision': 0.91, 'recall': 0.93, 'f1_score': 0.92},
            {'model': 'Logistic Regression', 'accuracy': 0.89, 'precision': 0.88, 'recall': 0.90, 'f1_score': 0.89},
            {'model': 'Naive Bayes', 'accuracy': 0.85, 'precision': 0.84, 'recall': 0.86, 'f1_score': 0.85},
        ]
        return jsonify(comparison)
        
    except Exception as e:
        logger.error(f"Error in get_model_comparison: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': classifier is not None and len(classifier.trained_models) > 0,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Load models on startup
    models_loaded = load_models()
    
    if models_loaded:
        logger.info("Running with real ML models")
    else:
        logger.info("Running in mock mode")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
