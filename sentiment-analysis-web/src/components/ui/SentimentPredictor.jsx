import React, { useState } from 'react';
import {
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Box,
  CircularProgress,
  Chip,
  LinearProgress,
  Alert,
  Grid
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import RemoveIcon from '@mui/icons-material/Remove';
import { sentimentAPI } from '../services/api';

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
        return <TrendingUpIcon sx={{ color: '#4caf50' }} />;
      case 'negative':
        return <TrendingDownIcon sx={{ color: '#f44336' }} />;
      case 'neutral':
        return <RemoveIcon sx={{ color: '#ff9800' }} />;
      default:
        return null;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return '#4caf50';
      case 'negative':
        return '#f44336';
      case 'neutral':
        return '#ff9800';
      default:
        return '#666';
    }
  };

  return (
    <Card sx={{ maxWidth: 600, mx: 'auto', mt: 4 }}>
      <CardContent sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', fontWeight: 600 }}>
          Sentiment Predictor
        </Typography>
        
        <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', mb: 3 }}>
          Enter text below to analyze its sentiment using our advanced ML models
        </Typography>

        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth
            multiline
            rows={4}
            variant="outlined"
            placeholder="Enter text to analyze... (e.g., 'I love this product! It's amazing!')"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={loading}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.3)',
                },
                '&:hover fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.5)',
                },
                '&.Mui-focused fieldset': {
                  borderColor: 'primary.main',
                },
              },
            }}
          />
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <Button
            variant="contained"
            size="large"
            onClick={handlePredict}
            disabled={loading || !text.trim()}
            startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
            sx={{
              px: 4,
              py: 1.5,
              fontSize: '1.1rem',
            }}
          >
            {loading ? 'Analyzing...' : 'Analyze Sentiment'}
          </Button>
        </Box>

        {prediction && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Results
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Box sx={{ 
                  p: 2, 
                  bgcolor: 'rgba(255, 255, 255, 0.05)', 
                  borderRadius: 2,
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {getSentimentIcon(prediction.predicted_sentiment)}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      Predicted Sentiment:
                    </Typography>
                    <Chip
                      label={prediction.predicted_sentiment.toUpperCase()}
                      sx={{
                        ml: 2,
                        backgroundColor: getSentimentColor(prediction.predicted_sentiment),
                        color: 'white',
                        fontWeight: 600,
                      }}
                    />
                  </Box>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={prediction.confidence * 100}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                        },
                      }}
                    />
                  </Box>

                  <Typography variant="body2" color="text.secondary">
                    Model: {prediction.model_used}
                  </Typography>
                </Box>
              </Grid>

              {prediction.probabilities && (
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Probability Distribution
                  </Typography>
                  {Object.entries(prediction.probabilities).map(([sentiment, prob]) => (
                    <Box key={sentiment} sx={{ mb: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2">
                          {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                        </Typography>
                        <Typography variant="body2">
                          {(prob * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={prob * 100}
                        sx={{
                          height: 6,
                          borderRadius: 3,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getSentimentColor(sentiment),
                          },
                        }}
                      />
                    </Box>
                  ))}
                </Grid>
              )}
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SentimentPredictor;
