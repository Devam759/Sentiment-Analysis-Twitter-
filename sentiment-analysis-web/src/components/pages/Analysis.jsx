import React, { useState, useEffect } from 'react';
import { sentimentAPI } from '../../api/api';

const Analysis = () => {
  const [modelComparison, setModelComparison] = useState([]);
  const [loading, setLoading] = useState(true);
  const [analysisText, setAnalysisText] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchModelComparison = async () => {
      try {
        setLoading(true);
        const data = await sentimentAPI.getModelComparison();
        
        const processedModels = data.map(model => ({
          ...model,
          displayName: model.model.replace('_', ' ').toUpperCase(),
          accuracyPercent: (model.accuracy * 100).toFixed(1),
          precisionPercent: (model.precision * 100).toFixed(1),
          recallPercent: (model.recall * 100).toFixed(1),
          f1Percent: (model.f1_score * 100).toFixed(1),
        }));
        
        const sortedModels = processedModels.sort((a, b) => b.accuracy - a.accuracy);
        setModelComparison(sortedModels);
        
      } catch (err) {
        console.error('Error loading model comparison:', err);
        setError('Failed to load model comparison data');
        
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
            displayName: 'RANDOM FOREST',
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
            displayName: 'LOGISTIC REGRESSION',
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
            displayName: 'NAIVE BAYES',
            accuracyPercent: '85.0',
            precisionPercent: '84.0',
            recallPercent: '86.0',
            f1Percent: '85.0',
          },
        ];
        
        setModelComparison(mockModels);
      } finally {
        setLoading(false);
      }
    };

    fetchModelComparison();
  }, []);

  const handleAnalyze = async () => {
    if (!analysisText.trim()) {
      setError('Please enter text to analyze');
      return;
    }

    setAnalyzing(true);
    setError('');
    setAnalysisResults(null);

    try {
      const results = await sentimentAPI.analyzeWithAllModels(analysisText);
      
      const processedResults = {};
      Object.entries(results).forEach(([modelName, result]) => {
        processedResults[modelName] = {
          ...result,
          displayName: modelName.replace('_', ' ').toUpperCase(),
          confidencePercent: (result.confidence * 100).toFixed(1),
        };
      });
      
      setAnalysisResults(processedResults);
      
    } catch (err) {
      console.error('Error analyzing with all models:', err);
      setError('Failed to analyze text with all models');
    } finally {
      setAnalyzing(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center pt-16">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Model Analysis & Comparison
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Compare performance metrics across different machine learning models
          </p>
        </div>

        {/* Model Comparison Table */}
        <div className="card p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Model Performance Metrics
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Accuracy
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Precision
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Recall
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    F1 Score
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {modelComparison.map((model, index) => (
                  <tr key={index} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {model.displayName}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <div className="flex items-center justify-center">
                        <span className="text-sm text-gray-900 mr-2">{model.accuracyPercent}%</span>
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${model.accuracy * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <div className="flex items-center justify-center">
                        <span className="text-sm text-gray-900 mr-2">{model.precisionPercent}%</span>
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${model.precision * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <div className="flex items-center justify-center">
                        <span className="text-sm text-gray-900 mr-2">{model.recallPercent}%</span>
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${model.recall * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <div className="flex items-center justify-center">
                        <span className="text-sm text-gray-900 mr-2">{model.f1Percent}%</span>
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${model.f1_score * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Multi-Model Analysis */}
        <div className="card p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Compare All Models
          </h2>
          <p className="text-gray-600 mb-6">
            Enter text to see how all models analyze the sentiment
          </p>
          
          <div className="space-y-6">
            <div>
              <textarea
                className="input min-h-[100px] resize-none"
                placeholder="Enter text to analyze with all models..."
                value={analysisText}
                onChange={(e) => setAnalysisText(e.target.value)}
                disabled={analyzing}
              />
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                {error}
              </div>
            )}

            <div className="flex justify-center">
              <button
                onClick={handleAnalyze}
                disabled={analyzing || !analysisText.trim()}
                className="btn btn-primary px-8 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {analyzing ? (
                  <span className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing with All Models...
                  </span>
                ) : (
                  <span className="flex items-center">
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Analyze with All Models
                  </span>
                )}
              </button>
            </div>
          </div>

          {analysisResults && (
            <div className="mt-8 space-y-4">
              <h3 className="text-xl font-semibold text-gray-900">
                Analysis Results
              </h3>
              {Object.entries(analysisResults).map(([modelName, result]) => (
                <div key={modelName} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-medium text-gray-900">
                      {result.displayName}
                    </h4>
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${
                      result.predicted_sentiment === 'positive' ? 'text-success bg-green-50 border-green-200' :
                      result.predicted_sentiment === 'negative' ? 'text-error bg-red-50 border-red-200' : 
                      'text-warning bg-yellow-50 border-yellow-200'
                    }`}>
                      {result.predicted_sentiment.toUpperCase()}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-600 mb-1">Predicted Sentiment</div>
                      <div className="font-medium">{result.predicted_sentiment}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600 mb-1">Confidence</div>
                      <div className="font-medium">{result.confidencePercent}%</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;
