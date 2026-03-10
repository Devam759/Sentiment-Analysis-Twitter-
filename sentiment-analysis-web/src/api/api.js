import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Mock API responses for development (since we don't have a real backend yet)
const mockResponses = {
  predict: (text) => {
    const sentiments = ['positive', 'negative', 'neutral'];
    const randomSentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
    const confidence = 0.7 + Math.random() * 0.3;
    
    return {
      text,
      predicted_sentiment: randomSentiment,
      confidence,
      model_used: 'svm_rbf',
      timestamp: new Date().toISOString(),
      probabilities: {
        positive: randomSentiment === 'positive' ? confidence : Math.random() * 0.3,
        negative: randomSentiment === 'negative' ? confidence : Math.random() * 0.3,
        neutral: randomSentiment === 'neutral' ? confidence : Math.random() * 0.3,
      }
    };
  },
  
  batchPredict: (texts) => {
    return texts.map(text => mockResponses.predict(text));
  },
  
  getStats: () => ({
    total_predictions: 1250,
    accuracy: 0.94,
    models_trained: 4,
    avg_processing_time: 0.15,
    sentiment_distribution: {
      positive: 450,
      negative: 380,
      neutral: 420
    }
  }),
  
  getModelComparison: () => [
    { model: 'SVM', accuracy: 0.94, precision: 0.93, recall: 0.95, f1_score: 0.94 },
    { model: 'Random Forest', accuracy: 0.92, precision: 0.91, recall: 0.93, f1_score: 0.92 },
    { model: 'Logistic Regression', accuracy: 0.89, precision: 0.88, recall: 0.90, f1_score: 0.89 },
    { model: 'Naive Bayes', accuracy: 0.85, precision: 0.84, recall: 0.86, f1_score: 0.85 },
  ]
};

// API Service Functions
export const sentimentAPI = {
  // Predict sentiment for a single text
  predictSentiment: async (text) => {
    try {
      // For development, use mock response
      // In production, this would be: const response = await api.post('/predict', { text });
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockResponses.predict(text)), 500);
      });
    } catch (error) {
      console.error('Error predicting sentiment:', error);
      throw error;
    }
  },

  // Predict sentiment for multiple texts
  batchPredict: async (texts) => {
    try {
      // For development, use mock response
      // In production: const response = await api.post('/batch-predict', { texts });
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockResponses.batchPredict(texts)), 1000);
      });
    } catch (error) {
      console.error('Error in batch prediction:', error);
      throw error;
    }
  },

  // Get system statistics
  getStats: async () => {
    try {
      // For development, use mock response
      // In production: const response = await api.get('/stats');
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockResponses.getStats()), 300);
      });
    } catch (error) {
      console.error('Error fetching stats:', error);
      throw error;
    }
  },

  // Get model comparison data
  getModelComparison: async () => {
    try {
      // For development, use mock response
      // In production: const response = await api.get('/model-comparison');
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockResponses.getModelComparison()), 400);
      });
    } catch (error) {
      console.error('Error fetching model comparison:', error);
      throw error;
    }
  },

  // Analyze text with all models
  analyzeWithAllModels: async (text) => {
    try {
      // For development, use mock response
      // In production: const response = await api.post('/analyze-all', { text });
      const models = ['svm_rbf', 'random_forest', 'logistic_regression', 'naive_bayes_multinomial'];
      return new Promise((resolve) => {
        setTimeout(() => {
          const results = {};
          models.forEach(model => {
            results[model] = mockResponses.predict(text);
            results[model].model_used = model;
          });
          resolve(results);
        }, 800);
      });
    } catch (error) {
      console.error('Error analyzing with all models:', error);
      throw error;
    }
  }
};

export default api;
