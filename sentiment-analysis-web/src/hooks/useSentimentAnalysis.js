import { useState, useCallback, useEffect } from 'react';
import { sentimentAPI } from '../api/api';
import { LOADING_MESSAGES, ERROR_MESSAGES } from '../constants';
import { validateTextInput, generateMockPrediction } from '../utils/helpers';

/**
 * Custom hook for sentiment analysis functionality
 * @returns {Object} Sentiment analysis state and functions
 */
export const useSentimentAnalysis = () => {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);

  // Clear error when text changes
  useEffect(() => {
    if (text) setError('');
  }, [text]);

  // Predict sentiment for single text
  const predictSentiment = useCallback(async (inputText = text) => {
    const validation = validateTextInput(inputText);
    if (!validation.isValid) {
      setError(validation.error);
      return null;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const result = await sentimentAPI.predictSentiment(inputText);
      setPrediction(result);
      
      // Add to history
      setHistory(prev => [result, ...prev.slice(0, 9)]); // Keep last 10
      
      return result;
    } catch (err) {
      console.error('Prediction error:', err);
      setError(ERROR_MESSAGES.API_ERROR);
      
      // Fallback to mock prediction
      const mockResult = generateMockPrediction(inputText);
      setPrediction(mockResult);
      return mockResult;
    } finally {
      setLoading(false);
    }
  }, [text]);

  // Batch predict multiple texts
  const batchPredict = useCallback(async (texts) => {
    if (!texts || texts.length === 0) {
      setError('Please provide texts to analyze');
      return [];
    }

    setLoading(true);
    setError('');

    try {
      const results = await sentimentAPI.batchPredict(texts);
      return results;
    } catch (err) {
      console.error('Batch prediction error:', err);
      setError(ERROR_MESSAGES.API_ERROR);
      
      // Fallback to mock predictions
      return texts.map(text => generateMockPrediction(text));
    } finally {
      setLoading(false);
    }
  }, []);

  // Analyze with all models
  const analyzeWithAllModels = useCallback(async (inputText = text) => {
    const validation = validateTextInput(inputText);
    if (!validation.isValid) {
      setError(validation.error);
      return null;
    }

    setLoading(true);
    setError('');

    try {
      const results = await sentimentAPI.analyzeWithAllModels(inputText);
      return results;
    } catch (err) {
      console.error('Multi-model analysis error:', err);
      setError(ERROR_MESSAGES.API_ERROR);
      return null;
    } finally {
      setLoading(false);
    }
  }, [text]);

  // Clear prediction
  const clearPrediction = useCallback(() => {
    setPrediction(null);
    setError('');
  }, []);

  // Clear history
  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  return {
    // State
    text,
    setText,
    prediction,
    loading,
    error,
    history,
    
    // Actions
    predictSentiment,
    batchPredict,
    analyzeWithAllModels,
    clearPrediction,
    clearHistory,
    
    // Computed
    hasPrediction: prediction !== null,
    hasHistory: history.length > 0,
    isLoading: loading,
  };
};
