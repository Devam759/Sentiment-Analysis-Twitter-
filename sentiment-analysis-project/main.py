"""
Twitter Sentiment Analysis - Main Pipeline Script

This script runs the complete sentiment analysis pipeline from data loading
to model training, evaluation, and visualization.

Pipeline Steps:
1. Data Loading and Validation
2. Text Preprocessing
3. Feature Extraction
4. Model Training
5. Model Evaluation
6. Visualization
7. Advanced Analysis
8. Demo Application

Usage:
    python main.py [--mode MODE] [--config CONFIG]

Modes:
    - full: Run complete pipeline (default)
    - train: Only train models
    - evaluate: Only evaluate models
    - demo: Run demo application
    - analyze: Only advanced analysis
"""

import argparse
import os
import sys
import logging
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import TwitterDataLoader
from preprocessing import TwitterTextPreprocessor
from feature_engineering import FeatureExtractor
from models import SentimentClassifier
from evaluation import ModelEvaluator
from visualization import SentimentVisualizer
from advanced_analysis import AdvancedTextAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SentimentAnalysisPipeline:
    """
    Complete sentiment analysis pipeline orchestrator.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the pipeline.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or self._get_default_config()
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.feature_extractor = None
        self.classifier = None
        self.evaluator = None
        self.visualizer = None
        self.advanced_analyzer = None
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info("Sentiment Analysis Pipeline initialized")
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'data': {
                'sample_size': 5000,
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'remove_urls': True,
                'remove_mentions': True,
                'remove_hashtags': True,
                'remove_punctuation': True,
                'remove_stopwords': True,
                'use_lemmatization': True,
                'remove_emojis': True
            },
            'features': {
                'use_tfidf': True,
                'use_word2vec': True,
                'use_statistical': True,
                'use_lexicon': True
            },
            'models': {
                'train_naive_bayes': True,
                'train_svm': True,
                'train_logistic_regression': True,
                'train_random_forest': True,
                'train_lstm': True
            },
            'evaluation': {
                'cross_validation': True,
                'roc_curves': True,
                'error_analysis': True
            },
            'visualization': {
                'sentiment_distribution': True,
                'word_clouds': True,
                'model_comparison': True,
                'topic_modeling': True
            },
            'advanced_analysis': {
                'ner': True,
                'brand_identification': True,
                'topic_modeling': True,
                'semantic_similarity': True,
                'aspect_sentiment': True,
                'emotion_detection': True
            }
        }
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            'dataset', 'models', 'results', 'notebooks', 'app'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Directories created")
    
    def load_data(self) -> bool:
        """
        Load and validate dataset.
        
        Returns:
            bool: True if data loaded successfully
        """
        logger.info("Loading dataset...")
        
        try:
            self.data_loader = TwitterDataLoader()
            
            # Load dataset (try Sentiment140 first, fallback to sample)
            try:
                self.raw_data = self.data_loader.load_sentiment140(
                    sample_size=self.config['data']['sample_size']
                )
            except Exception as e:
                logger.warning(f"Could not load Sentiment140: {e}")
                self.raw_data = self.data_loader._create_sample_dataset(1000)
            
            # Validate dataset
            stats = self.data_loader.validate_dataset(self.raw_data)
            logger.info(f"Dataset loaded: {stats}")
            
            # Save dataset
            self.data_loader.save_dataset(self.raw_data, 'raw_dataset.csv')
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self) -> bool:
        """
        Preprocess text data.
        
        Returns:
            bool: True if preprocessing successful
        """
        logger.info("Preprocessing data...")
        
        try:
            # Initialize preprocessor
            self.preprocessor = TwitterTextPreprocessor(**self.config['preprocessing'])
            
            # Preprocess data
            self.processed_data = self.preprocessor.preprocess_dataframe(self.raw_data)
            
            # Get preprocessing statistics
            stats = self.preprocessor.get_preprocessing_stats(self.raw_data, self.processed_data)
            logger.info(f"Preprocessing completed: {stats}")
            
            # Save processed data
            self.processed_data.to_csv('dataset/processed_dataset.csv', index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return False
    
    def extract_features(self) -> bool:
        """
        Extract features from preprocessed text.
        
        Returns:
            bool: True if feature extraction successful
        """
        logger.info("Extracting features...")
        
        try:
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor()
            
            # Extract features
            texts = self.processed_data['processed_text'].tolist()
            self.features = self.feature_extractor.extract_all_features(
                texts, **self.config['features']
            )
            
            logger.info(f"Features extracted: {self.features.shape}")
            
            # Save features
            self.feature_extractor.save_features(self.features, 'extracted_features.npy')
            
            # Save feature names
            import pickle
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_extractor.feature_names, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return False
    
    def train_models(self) -> bool:
        """
        Train machine learning models.
        
        Returns:
            bool: True if training successful
        """
        logger.info("Training models...")
        
        try:
            # Initialize classifier
            self.classifier = SentimentClassifier()
            
            # Prepare data
            labels = self.processed_data['sentiment'].values
            self.X_train, self.X_test, self.y_train, self.y_test = self.classifier.prepare_data(
                self.features, labels, 
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_state']
            )
            
            # Train models
            original_texts = self.processed_data['text'].tolist()
            train_texts = [original_texts[i] for i in range(len(self.y_train))]
            
            training_results = self.classifier.train_all_models(
                self.X_train, train_texts, self.y_train
            )
            
            # Save label encoder
            import pickle
            with open('models/label_encoder.pkl', 'wb') as f:
                pickle.dump(self.classifier.label_encoder, f)
            
            logger.info(f"Models trained: {list(training_results.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def evaluate_models(self) -> bool:
        """
        Evaluate trained models.
        
        Returns:
            bool: True if evaluation successful
        """
        logger.info("Evaluating models...")
        
        try:
            # Initialize evaluator
            self.evaluator = ModelEvaluator()
            
            evaluation_results = {}
            
            # Evaluate each model
            for model_name in self.classifier.trained_models.keys():
                try:
                    # Make predictions
                    predictions = self.classifier.predict(model_name, self.X_test)
                    
                    # Get probabilities if available
                    probabilities = None
                    try:
                        probabilities = self.classifier.predict_proba(model_name, self.X_test)
                    except:
                        pass
                    
                    # Evaluate model
                    class_names = self.classifier.label_encoder.classes_.tolist()
                    results = self.evaluator.evaluate_model(
                        self.y_test, predictions, probabilities,
                        model_name, class_names
                    )
                    
                    evaluation_results[model_name] = results
                    
                    logger.info(f"Model {model_name} evaluated - F1: {results['f1_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating model {model_name}: {e}")
            
            # Generate comparison
            if evaluation_results:
                comparison_df = self.evaluator.compare_models(evaluation_results)
                logger.info("Model comparison completed")
                
                # Generate comprehensive report
                report = self.evaluator.generate_evaluation_report(evaluation_results)
                logger.info("Evaluation report generated")
                
                # Save results
                self.evaluator.save_results()
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return False
    
    def create_visualizations(self) -> bool:
        """
        Create visualizations.
        
        Returns:
            bool: True if visualization successful
        """
        logger.info("Creating visualizations...")
        
        try:
            # Initialize visualizer
            self.visualizer = SentimentVisualizer()
            
            # Sentiment distribution
            if self.config['visualization']['sentiment_distribution']:
                self.visualizer.plot_sentiment_distribution(self.processed_data)
            
            # Word clouds
            if self.config['visualization']['word_clouds']:
                self.visualizer.create_sentiment_word_clouds(self.processed_data)
            
            # Model performance comparison
            if self.config['visualization']['model_comparison'] and self.evaluator:
                comparison_df = self.evaluator.compare_models(self.evaluator.evaluation_results)
                self.visualizer.plot_model_performance_comparison(comparison_df)
            
            # Topic modeling
            if self.config['visualization']['topic_modeling']:
                texts = self.processed_data['processed_text'].tolist()
                self.visualizer.plot_topic_modeling(texts)
            
            # Word frequency analysis
            texts = self.processed_data['processed_text'].tolist()
            sentiments = self.processed_data['sentiment'].tolist()
            self.visualizer.plot_word_frequency_analysis(texts, sentiments)
            
            # Interactive dashboard
            self.visualizer.create_interactive_dashboard(self.processed_data)
            
            logger.info("Visualizations created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return False
    
    def perform_advanced_analysis(self) -> bool:
        """
        Perform advanced text analysis.
        
        Returns:
            bool: True if analysis successful
        """
        logger.info("Performing advanced analysis...")
        
        try:
            # Initialize advanced analyzer
            self.advanced_analyzer = AdvancedTextAnalyzer()
            
            texts = self.processed_data['text'].tolist()
            sentiments = self.processed_data['sentiment'].tolist()
            
            # Generate comprehensive report
            results = self.advanced_analyzer.generate_comprehensive_report(
                texts, sentiments, self.processed_data
            )
            
            logger.info(f"Advanced analysis completed: {list(results.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in advanced analysis: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        logger.info("Starting full sentiment analysis pipeline...")
        start_time = time.time()
        
        steps = [
            ("Loading Data", self.load_data),
            ("Preprocessing Data", self.preprocess_data),
            ("Extracting Features", self.extract_features),
            ("Training Models", self.train_models),
            ("Evaluating Models", self.evaluate_models),
            ("Creating Visualizations", self.create_visualizations),
            ("Advanced Analysis", self.perform_advanced_analysis)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            try:
                if not step_func():
                    logger.error(f"Pipeline failed at step: {step_name}")
                    return False
                logger.info(f"Completed: {step_name}")
            except Exception as e:
                logger.error(f"Error in {step_name}: {e}")
                return False
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
        
        # Generate final summary
        self._generate_pipeline_summary(duration)
        
        return True
    
    def _generate_pipeline_summary(self, duration: float) -> None:
        """Generate pipeline execution summary."""
        summary = f"""
Twitter Sentiment Analysis Pipeline - Execution Summary
{'=' * 60}

Execution Time: {duration:.2f} seconds
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Information:
- Raw samples: {len(self.raw_data) if self.raw_data is not None else 0}
- Processed samples: {len(self.processed_data) if self.processed_data is not None else 0}
- Feature dimensions: {self.features.shape if self.features is not None else 'N/A'}

Models Trained: {list(self.classifier.trained_models.keys()) if self.classifier else 'None'}

Generated Outputs:
- Dataset: dataset/
- Models: models/
- Results: results/
- Visualizations: results/
- Advanced Analysis: results/

Next Steps:
1. Review evaluation results in results/evaluation_report.txt
2. Explore visualizations in results/
3. Run demo application: python app/sentiment_predictor.py
4. Check advanced analysis results in results/

Pipeline completed successfully!
"""
        
        with open('results/pipeline_summary.txt', 'w') as f:
            f.write(summary)
        
        logger.info("Pipeline summary generated")
    
    def run_training_only(self) -> bool:
        """Run only training steps."""
        return (self.load_data() and 
                self.preprocess_data() and 
                self.extract_features() and 
                self.train_models())
    
    def run_evaluation_only(self) -> bool:
        """Run only evaluation steps."""
        if self.load_data() and self.preprocess_data() and self.extract_features():
            # Load existing models
            self.classifier = SentimentClassifier()
            return self.evaluate_models()
        return False
    
    def run_demo_only(self) -> None:
        """Run demo application only."""
        from app.sentiment_predictor import SentimentPredictor
        predictor = SentimentPredictor()
        predictor.interactive_demo()
    
    def run_analysis_only(self) -> bool:
        """Run only advanced analysis."""
        return (self.load_data() and 
                self.preprocess_data() and 
                self.perform_advanced_analysis())


def main():
    """
    Main function to run the sentiment analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Twitter Sentiment Analysis Pipeline')
    parser.add_argument('--mode', choices=['full', 'train', 'evaluate', 'demo', 'analyze'],
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SentimentAnalysisPipeline()
    
    # Run based on mode
    if args.mode == 'full':
        success = pipeline.run_full_pipeline()
    elif args.mode == 'train':
        success = pipeline.run_training_only()
    elif args.mode == 'evaluate':
        success = pipeline.run_evaluation_only()
    elif args.mode == 'demo':
        pipeline.run_demo_only()
        success = True
    elif args.mode == 'analyze':
        success = pipeline.run_analysis_only()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated outputs:")
        print("- Dataset: dataset/")
        print("- Models: models/")
        print("- Results: results/")
        print("\nTo run the demo application:")
        print("python app/sentiment_predictor.py")
    else:
        logger.error("Pipeline execution failed!")
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION FAILED!")
        print("=" * 60)
        print("\nCheck the log file: sentiment_analysis.log")


if __name__ == "__main__":
    main()
