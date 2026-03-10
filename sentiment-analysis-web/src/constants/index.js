// Application constants

export const API_ENDPOINTS = {
  PREDICT: '/api/predict',
  BATCH_PREDICT: '/api/batch-predict',
  ANALYZE_ALL: '/api/analyze-all',
  STATS: '/api/stats',
  MODEL_COMPARISON: '/api/model-comparison',
  HEALTH: '/api/health',
};

export const SENTIMENT_COLORS = {
  positive: '#4caf50',
  negative: '#f44336',
  neutral: '#ff9800',
};

export const SENTIMENT_ICONS = {
  positive: '📈',
  negative: '📉',
  neutral: '➡️',
};

export const MODEL_COLORS = {
  SVM: '#667eea',
  'Random Forest': '#764ba2',
  'Logistic Regression': '#4caf50',
  'Naive Bayes': '#ff9800',
};

export const CHART_CONFIG = {
  FONT_COLOR: '#ffffff',
  GRID_COLOR: 'rgba(255, 255, 255, 0.1)',
  BACKGROUND_COLOR: 'rgba(255, 255, 255, 0.05)',
};

export const ROUTES = {
  HOME: '/',
  ANALYSIS: '/analysis',
  ABOUT: '/about',
};

export const LOADING_MESSAGES = {
  PREDICTING: 'Analyzing sentiment...',
  LOADING_MODELS: 'Loading ML models...',
  FETCHING_STATS: 'Fetching statistics...',
  ANALYZING_ALL: 'Comparing all models...',
};

export const ERROR_MESSAGES = {
  NO_TEXT: 'Please enter some text to analyze',
  API_ERROR: 'Failed to connect to the server. Please try again.',
  MODEL_ERROR: 'Error loading models. Please refresh the page.',
  NETWORK_ERROR: 'Network error. Please check your connection.',
};

export const SUCCESS_MESSAGES = {
  PREDICTION_COMPLETE: 'Sentiment analysis complete!',
  MODELS_LOADED: 'ML models loaded successfully.',
};

export const VALIDATION = {
  MIN_TEXT_LENGTH: 1,
  MAX_TEXT_LENGTH: 500,
  BATCH_SIZE_LIMIT: 100,
};

export const ANIMATION_DURATION = {
  FAST: 300,
  NORMAL: 500,
  SLOW: 1000,
};

export const BREAKPOINTS = {
  XS: 0,
  SM: 600,
  MD: 960,
  LG: 1280,
  XL: 1920,
};
