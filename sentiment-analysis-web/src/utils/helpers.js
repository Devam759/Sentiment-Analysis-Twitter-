// Utility functions

/**
 * Format confidence score as percentage
 * @param {number} confidence - Confidence value between 0 and 1
 * @returns {string} Formatted percentage string
 */
export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(1)}%`;
};

/**
 * Format timestamp to readable date
 * @param {string} timestamp - ISO timestamp string
 * @returns {string} Formatted date string
 */
export const formatDate = (timestamp) => {
  return new Date(timestamp).toLocaleString();
};

/**
 * Get sentiment color class based on sentiment
 * @param {string} sentiment - Sentiment value
 * @returns {string} CSS class name
 */
export const getSentimentClass = (sentiment) => {
  return `sentiment-${sentiment}`;
};

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength = 100) => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

/**
 * Validate text input
 * @param {string} text - Text to validate
 * @returns {Object} Validation result
 */
export const validateTextInput = (text) => {
  const trimmedText = text.trim();
  
  if (!trimmedText) {
    return { isValid: false, error: 'Please enter some text to analyze' };
  }
  
  if (trimmedText.length < 1) {
    return { isValid: false, error: 'Text is too short' };
  }
  
  if (trimmedText.length > 500) {
    return { isValid: false, error: 'Text is too long (max 500 characters)' };
  }
  
  return { isValid: true, error: null };
};

/**
 * Calculate percentage from value and total
 * @param {number} value - Current value
 * @param {number} total - Total value
 * @returns {number} Percentage (0-100)
 */
export const calculatePercentage = (value, total) => {
  if (total === 0) return 0;
  return Math.round((value / total) * 100);
};

/**
 * Sort model comparison data by accuracy
 * @param {Array} models - Array of model objects
 * @returns {Array} Sorted array
 */
export const sortModelsByAccuracy = (models) => {
  return [...models].sort((a, b) => b.accuracy - a.accuracy);
};

/**
 * Generate random mock data for development
 * @param {string} text - Input text
 * @returns {Object} Mock prediction result
 */
export const generateMockPrediction = (text) => {
  const sentiments = ['positive', 'negative', 'neutral'];
  const randomSentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
  const confidence = 0.7 + Math.random() * 0.3;
  
  return {
    text,
    predicted_sentiment: randomSentiment,
    confidence,
    model_used: 'mock_model',
    timestamp: new Date().toISOString(),
    probabilities: {
      positive: randomSentiment === 'positive' ? confidence : Math.random() * 0.3,
      negative: randomSentiment === 'negative' ? confidence : Math.random() * 0.3,
      neutral: randomSentiment === 'neutral' ? confidence : Math.random() * 0.3,
    }
  };
};

/**
 * Debounce function to limit API calls
 * @param {Function} func - Function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} Debounced function
 */
export const debounce = (func, delay) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(null, args), delay);
  };
};

/**
 * Check if a value is empty or null
 * @param {*} value - Value to check
 * @returns {boolean} Whether value is empty
 */
export const isEmpty = (value) => {
  return value === null || value === undefined || value === '' || 
         (Array.isArray(value) && value.length === 0) ||
         (typeof value === 'object' && Object.keys(value).length === 0);
};

/**
 * Get model display name from technical name
 * @param {string} modelName - Technical model name
 * @returns {string} Display name
 */
export const getModelDisplayName = (modelName) => {
  const nameMap = {
    'svm_rbf': 'SVM',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'naive_bayes_multinomial': 'Naive Bayes',
  };
  
  return nameMap[modelName] || modelName.replace('_', ' ').toUpperCase();
};

/**
 * Format processing time
 * @param {number} seconds - Processing time in seconds
 * @returns {string} Formatted time string
 */
export const formatProcessingTime = (seconds) => {
  if (seconds < 0.001) return '< 1ms';
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(2)}s`;
};

/**
 * Generate chart colors array
 * @param {number} count - Number of colors needed
 * @returns {Array} Array of colors
 */
export const generateChartColors = (count) => {
  const baseColors = [
    '#667eea', '#764ba2', '#4caf50', '#ff9800', '#f44336',
    '#2196f3', '#9c27b0', '#00bcd4', '#8bc34a', '#ffc107'
  ];
  
  const colors = [];
  for (let i = 0; i < count; i++) {
    colors.push(baseColors[i % baseColors.length]);
  }
  
  return colors;
};

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise} Promise that resolves when text is copied
 */
export const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    return true;
  }
};
