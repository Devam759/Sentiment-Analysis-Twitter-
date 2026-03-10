import React, { useState } from 'react';
import { sentimentAPI } from '../../api/api';

const SentimentPredictor = () => {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const result = await sentimentAPI.predictSentiment(text);
      setPrediction(result);
    } catch (err) {
      setError('Failed to analyze sentiment. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return (
          <svg className="w-6 h-6 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
        );
      case 'negative':
        return (
          <svg className="w-6 h-6 text-error" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
          </svg>
        );
      case 'neutral':
        return (
          <svg className="w-6 h-6 text-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h8m-4-4v8" />
          </svg>
        );
      default:
        return null;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'text-success bg-green-50 border-green-200';
      case 'negative':
        return 'text-error bg-red-50 border-red-200';
      case 'neutral':
        return 'text-warning bg-yellow-50 border-yellow-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="card p-8 max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Sentiment Predictor
        </h2>
        <p className="text-gray-600">
          Enter text below to analyze its sentiment using our advanced ML models
        </p>
      </div>

      <div className="space-y-6">
        {/* Text Input */}
        <div>
          <textarea
            className="input min-h-[120px] resize-none"
            placeholder="Enter text to analyze... (e.g., 'I love this product! It's amazing!')"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={loading}
          />
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        {/* Predict Button */}
        <div className="flex justify-center">
          <button
            onClick={handlePredict}
            disabled={loading || !text.trim()}
            className="btn btn-primary px-8 py-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing...
              </span>
            ) : (
              <span className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
                Analyze Sentiment
              </span>
            )}
          </button>
        </div>

        {/* Results */}
        {prediction && (
          <div className="animate-in space-y-6">
            <div className="border-l-4 border-primary-500 pl-4">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Analysis Results
              </h3>
            </div>

            {/* Main Result */}
            <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  {getSentimentIcon(prediction.predicted_sentiment)}
                  <div>
                    <div className="text-lg font-medium text-gray-900">
                      Predicted Sentiment:
                    </div>
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getSentimentColor(prediction.predicted_sentiment)}`}>
                      {prediction.predicted_sentiment.toUpperCase()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Confidence */}
              <div className="mb-4">
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>Confidence</span>
                  <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="text-sm text-gray-600">
                Model: <span className="font-medium">{prediction.model_used}</span>
              </div>
            </div>

            {/* Probability Distribution */}
            {prediction.probabilities && (
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                <h4 className="text-lg font-medium text-gray-900 mb-4">
                  Probability Distribution
                </h4>
                <div className="space-y-3">
                  {Object.entries(prediction.probabilities).map(([sentiment, prob]) => (
                    <div key={sentiment}>
                      <div className="flex justify-between text-sm text-gray-600 mb-1">
                        <span className="capitalize">{sentiment}</span>
                        <span>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-500 ${
                            sentiment === 'positive' ? 'bg-success' :
                            sentiment === 'negative' ? 'bg-error' : 'bg-warning'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentPredictor;
