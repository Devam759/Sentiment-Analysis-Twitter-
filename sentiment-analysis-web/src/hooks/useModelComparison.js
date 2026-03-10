import { useState, useEffect, useCallback } from 'react';
import { sentimentAPI } from '../api/api';
import { sortModelsByAccuracy, getModelDisplayName } from '../utils/helpers';

/**
 * Custom hook for model comparison functionality
 * @returns {Object} Model comparison state and functions
 */
export const useModelComparison = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  // Load model comparison data
  useEffect(() => {
    const loadModelComparison = async () => {
      try {
        setLoading(true);
        const data = await sentimentAPI.getModelComparison();
        
        // Process and sort data
        const processedModels = data.map(model => ({
          ...model,
          displayName: getModelDisplayName(model.model),
          accuracyPercent: (model.accuracy * 100).toFixed(1),
          precisionPercent: (model.precision * 100).toFixed(1),
          recallPercent: (model.recall * 100).toFixed(1),
          f1Percent: (model.f1_score * 100).toFixed(1),
        }));
        
        const sortedModels = sortModelsByAccuracy(processedModels);
        setModels(sortedModels);
        
        // Select best model by default
        if (sortedModels.length > 0) {
          setSelectedModel(sortedModels[0].model);
        }
        
      } catch (err) {
        console.error('Error loading model comparison:', err);
        setError('Failed to load model comparison data');
        
        // Fallback mock data
        const mockModels = [
          { 
            model: 'SVM', 
            accuracy: 0.94, 
            precision: 0.93, 
            recall: 0.95, 
            f1_score: 0.94,
            displayName: 'SVM',
            accuracyPercent: '94.0',
            precisionPercent: '93.0',
            recallPercent: '95.0',
            f1Percent: '94.0',
          },
          { 
            model: 'Random Forest', 
            accuracy: 0.92, 
            precision: 0.91, 
            recall: 0.93, 
            f1_score: 0.92,
            displayName: 'Random Forest',
            accuracyPercent: '92.0',
            precisionPercent: '91.0',
            recallPercent: '93.0',
            f1Percent: '92.0',
          },
          { 
            model: 'Logistic Regression', 
            accuracy: 0.89, 
            precision: 0.88, 
            recall: 0.90, 
            f1_score: 0.89,
            displayName: 'Logistic Regression',
            accuracyPercent: '89.0',
            precisionPercent: '88.0',
            recallPercent: '90.0',
            f1Percent: '89.0',
          },
          { 
            model: 'Naive Bayes', 
            accuracy: 0.85, 
            precision: 0.84, 
            recall: 0.86, 
            f1_score: 0.85,
            displayName: 'Naive Bayes',
            accuracyPercent: '85.0',
            precisionPercent: '84.0',
            recallPercent: '86.0',
            f1Percent: '85.0',
          },
        ];
        
        setModels(mockModels);
        setSelectedModel(mockModels[0].model);
      } finally {
        setLoading(false);
      }
    };

    loadModelComparison();
  }, []);

  // Analyze text with all models
  const analyzeWithAllModels = useCallback(async (text) => {
    if (!text || !text.trim()) {
      setError('Please enter text to analyze');
      return;
    }

    setAnalyzing(true);
    setError('');
    setAnalysisResults(null);

    try {
      const results = await sentimentAPI.analyzeWithAllModels(text);
      
      // Process results
      const processedResults = {};
      Object.entries(results).forEach(([modelName, result]) => {
        processedResults[modelName] = {
          ...result,
          displayName: getModelDisplayName(modelName),
          confidencePercent: (result.confidence * 100).toFixed(1),
        };
      });
      
      setAnalysisResults(processedResults);
      return processedResults;
      
    } catch (err) {
      console.error('Error analyzing with all models:', err);
      setError('Failed to analyze text with all models');
      return null;
    } finally {
      setAnalyzing(false);
    }
  }, []);

  // Get best performing model
  const getBestModel = useCallback(() => {
    if (models.length === 0) return null;
    return models[0]; // Already sorted by accuracy
  }, [models]);

  // Get model by name
  const getModelByName = useCallback((modelName) => {
    return models.find(model => model.model === modelName);
  }, [models]);

  // Get model statistics
  const getModelStats = useCallback(() => {
    if (models.length === 0) return null;
    
    const avgAccuracy = models.reduce((sum, model) => sum + model.accuracy, 0) / models.length;
    const avgF1 = models.reduce((sum, model) => sum + model.f1_score, 0) / models.length;
    
    return {
      totalModels: models.length,
      averageAccuracy: (avgAccuracy * 100).toFixed(1),
      averageF1: (avgF1 * 100).toFixed(1),
      bestModel: getBestModel(),
      worstModel: models[models.length - 1],
    };
  }, [models, getBestModel]);

  return {
    // State
    models,
    loading,
    error,
    selectedModel,
    setSelectedModel,
    analysisResults,
    analyzing,
    
    // Actions
    analyzeWithAllModels,
    
    // Computed
    bestModel: getBestModel(),
    modelStats: getModelStats(),
    hasModels: models.length > 0,
    hasResults: analysisResults !== null,
  };
};
