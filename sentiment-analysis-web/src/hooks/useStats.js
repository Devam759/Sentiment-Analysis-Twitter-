import { useState, useEffect, useCallback } from 'react';
import { sentimentAPI } from '../api/api';

/**
 * Custom hook for application statistics
 * @returns {Object} Stats state and functions
 */
export const useStats = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState(null);

  // Load statistics
  const loadStats = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      
      const data = await sentimentAPI.getStats();
      
      // Process stats
      const processedStats = {
        ...data,
        totalPredictionsFormatted: data.total_predictions.toLocaleString(),
        accuracyFormatted: (data.accuracy * 100).toFixed(1),
        avgProcessingTimeFormatted: `${data.avg_processing_time}s`,
        sentimentDistribution: {
          ...data.sentiment_distribution,
          total: Object.values(data.sentiment_distribution).reduce((sum, count) => sum + count, 0),
        },
        sentimentPercentages: Object.entries(data.sentiment_distribution).reduce((acc, [sentiment, count]) => {
          const total = Object.values(data.sentiment_distribution).reduce((sum, c) => sum + c, 0);
          acc[sentiment] = total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
          return acc;
        }, {}),
      };
      
      setStats(processedStats);
      setLastUpdated(new Date());
      
    } catch (err) {
      console.error('Error loading stats:', err);
      setError('Failed to load statistics');
      
      // Fallback mock stats
      const mockStats = {
        total_predictions: 1250,
        accuracy: 0.94,
        models_trained: 4,
        avg_processing_time: 0.15,
        sentiment_distribution: {
          positive: 450,
          negative: 380,
          neutral: 420,
        },
        totalPredictionsFormatted: '1,250',
        accuracyFormatted: '94.0',
        avgProcessingTimeFormatted: '0.15s',
        sentimentDistribution: {
          positive: 450,
          negative: 380,
          neutral: 420,
          total: 1250,
        },
        sentimentPercentages: {
          positive: '36.0',
          negative: '30.4',
          neutral: '33.6',
        },
      };
      
      setStats(mockStats);
      setLastUpdated(new Date());
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadStats();
  }, [loadStats]);

  // Refresh stats
  const refreshStats = useCallback(() => {
    return loadStats();
  }, [loadStats]);

  // Get sentiment chart data
  const getSentimentChartData = useCallback(() => {
    if (!stats) return null;
    
    return {
      labels: Object.keys(stats.sentiment_distribution).map(s => 
        s.charAt(0).toUpperCase() + s.slice(1)
      ),
      datasets: [{
        label: 'Sentiment Distribution',
        data: Object.values(stats.sentiment_distribution),
        backgroundColor: ['#4caf50', '#f44336', '#ff9800'],
        borderColor: ['#4caf50', '#f44336', '#ff9800'],
        borderWidth: 1,
      }],
    };
  }, [stats]);

  // Get performance metrics
  const getPerformanceMetrics = useCallback(() => {
    if (!stats) return null;
    
    return {
      accuracy: stats.accuracy,
      accuracyPercent: stats.accuracyFormatted,
      modelsTrained: stats.models_trained,
      avgProcessingTime: stats.avg_processing_time,
      avgProcessingTimeFormatted: stats.avgProcessingTimeFormatted,
    };
  }, [stats]);

  return {
    // State
    stats,
    loading,
    error,
    lastUpdated,
    
    // Actions
    loadStats: refreshStats,
    
    // Computed
    sentimentChartData: getSentimentChartData(),
    performanceMetrics: getPerformanceMetrics(),
    hasStats: stats !== null,
  };
};
