import React, { useState, useEffect } from 'react';
import SentimentPredictor from '../ui/SentimentPredictor';
import { sentimentAPI } from '../../api/api';

const Home = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await sentimentAPI.getStats();
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const features = [
    {
      title: 'Advanced NLP',
      description: 'State-of-the-art natural language processing with multiple ML models',
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      ),
    },
    {
      title: 'Real-time Analysis',
      description: 'Get instant sentiment predictions with confidence scores',
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
    },
    {
      title: 'Multiple Models',
      description: 'Compare results from SVM, Random Forest, Logistic Regression, and Naive Bayes',
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      title: 'Detailed Insights',
      description: 'Comprehensive analysis with probability distributions and confidence metrics',
      icon: (
        <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v1a1 1 0 001 1h4a1 1 0 001-1v-1m3-2V8a2 2 0 00-2-2H9a2 2 0 00-2 2v6m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2H9a2 2 0 01-2-2v-6z" />
        </svg>
      ),
    },
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-primary-50 to-white py-20 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center animate-in">
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              Twitter Sentiment Analysis
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Advanced Machine Learning for Consumer Sentiment Prediction
            </p>
            <div className="flex flex-wrap justify-center gap-4 mb-8">
              <div className="badge badge-success">94% Accuracy</div>
              <div className="badge badge-neutral">4 ML Models</div>
              <div className="badge badge-neutral">Real-time Processing</div>
            </div>
          </div>
        </div>
        
        {/* Background decoration */}
        <div className="absolute top-0 right-0 -mt-4 w-64 h-64 bg-primary-100 rounded-full opacity-20 blur-3xl"></div>
        <div className="absolute bottom-0 left-0 -mb-4 w-96 h-96 bg-primary-50 rounded-full opacity-30 blur-3xl"></div>
      </section>

      {/* Stats Section */}
      {stats && (
        <section className="py-16 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="card card-hover text-center p-6">
                <div className="text-3xl font-bold text-primary-600 mb-2">
                  {stats.total_predictions.toLocaleString()}
                </div>
                <div className="text-gray-600">Total Predictions</div>
              </div>
              <div className="card card-hover text-center p-6">
                <div className="text-3xl font-bold text-success mb-2">
                  {(stats.accuracy * 100).toFixed(1)}%
                </div>
                <div className="text-gray-600">Accuracy</div>
              </div>
              <div className="card card-hover text-center p-6">
                <div className="text-3xl font-bold text-primary-600 mb-2">
                  {stats.models_trained}
                </div>
                <div className="text-gray-600">Models Trained</div>
              </div>
              <div className="card card-hover text-center p-6">
                <div className="text-3xl font-bold text-primary-600 mb-2">
                  {stats.avg_processing_time}s
                </div>
                <div className="text-gray-600">Avg Processing Time</div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Key Features
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Comprehensive tools and capabilities for advanced sentiment analysis
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className="card card-hover p-6 text-center group"
              >
                <div className="flex justify-center mb-4 group-hover:scale-110 transition-transform duration-200">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Sentiment Predictor */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Try It Now
            </h2>
            <p className="text-xl text-gray-600">
              Enter text below to analyze its sentiment using our advanced ML models
            </p>
          </div>
          <SentimentPredictor />
        </div>
      </section>

      {/* Sentiment Distribution */}
      {stats && (
        <section className="py-20 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
                Sentiment Distribution
              </h2>
              <p className="text-xl text-gray-600">
                Overview of sentiment analysis results across all predictions
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="card card-hover p-8 text-center">
                <div className="text-4xl font-bold text-success mb-4">
                  {stats.sentiment_distribution.positive.toLocaleString()}
                </div>
                <div className="text-lg font-medium text-gray-900 mb-2">
                  Positive Sentiments
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-success h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(stats.sentiment_distribution.positive / Object.values(stats.sentiment_distribution).reduce((a, b) => a + b, 0)) * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="card card-hover p-8 text-center">
                <div className="text-4xl font-bold text-error mb-4">
                  {stats.sentiment_distribution.negative.toLocaleString()}
                </div>
                <div className="text-lg font-medium text-gray-900 mb-2">
                  Negative Sentiments
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-error h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(stats.sentiment_distribution.negative / Object.values(stats.sentiment_distribution).reduce((a, b) => a + b, 0)) * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="card card-hover p-8 text-center">
                <div className="text-4xl font-bold text-warning mb-4">
                  {stats.sentiment_distribution.neutral.toLocaleString()}
                </div>
                <div className="text-lg font-medium text-gray-900 mb-2">
                  Neutral Sentiments
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-warning h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(stats.sentiment_distribution.neutral / Object.values(stats.sentiment_distribution).reduce((a, b) => a + b, 0)) * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default Home;
