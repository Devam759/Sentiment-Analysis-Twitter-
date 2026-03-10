import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Chip,
  CircularProgress,
} from '@mui/material';
import SentimentPredictor from '../components/SentimentPredictor';
import { sentimentAPI } from '../services/api';
import PsychologyIcon from '@mui/icons-material/Psychology';
import SpeedIcon from '@mui/icons-material/Speed';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import AnalyticsIcon from '@mui/icons-material/Analytics';

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
      icon: <PsychologyIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Advanced NLP',
      description: 'State-of-the-art natural language processing with multiple ML models',
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Real-time Analysis',
      description: 'Get instant sentiment predictions with confidence scores',
    },
    {
      icon: <ModelTrainingIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Multiple Models',
      description: 'Compare results from SVM, Random Forest, Logistic Regression, and Naive Bayes',
    },
    {
      icon: <AnalyticsIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Detailed Insights',
      description: 'Comprehensive analysis with probability distributions and confidence metrics',
    },
  ];

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ display: 'flex', justifyContent: 'center', mt: 8 }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" className="page-container">
      {/* Hero Section */}
      <Box className="hero-section">
        <Typography className="hero-title" variant="h2">
          Twitter Sentiment Analysis
        </Typography>
        <Typography className="hero-subtitle" variant="h5">
          Advanced Machine Learning for Consumer Sentiment Prediction
        </Typography>
        <Box sx={{ mt: 3 }}>
          <Chip
            label="94% Accuracy"
            color="primary"
            sx={{ mr: 1, fontSize: '1rem', py: 0.5 }}
          />
          <Chip
            label="4 ML Models"
            color="secondary"
            sx={{ mr: 1, fontSize: '1rem', py: 0.5 }}
          />
          <Chip
            label="Real-time Processing"
            variant="outlined"
            sx={{ fontSize: '1rem', py: 0.5, borderColor: '#667eea', color: '#667eea' }}
          />
        </Box>
      </Box>

      {/* Stats Section */}
      {stats && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card className="stat-card">
              <CardContent>
                <Typography className="stat-number" variant="h3">
                  {stats.total_predictions.toLocaleString()}
                </Typography>
                <Typography className="stat-label">
                  Total Predictions
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card className="stat-card">
              <CardContent>
                <Typography className="stat-number" variant="h3">
                  {(stats.accuracy * 100).toFixed(1)}%
                </Typography>
                <Typography className="stat-label">
                  Accuracy
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card className="stat-card">
              <CardContent>
                <Typography className="stat-number" variant="h3">
                  {stats.models_trained}
                </Typography>
                <Typography className="stat-label">
                  Models Trained
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card className="stat-card">
              <CardContent>
                <Typography className="stat-number" variant="h3">
                  {stats.avg_processing_time}s
                </Typography>
                <Typography className="stat-label">
                  Avg Processing Time
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Features Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" sx={{ textAlign: 'center', mb: 4, fontWeight: 600 }}>
          Key Features
        </Typography>
        <Grid container spacing={3} className="feature-grid">
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card className="feature-card">
                <CardContent>
                  <Box sx={{ textAlign: 'center', mb: 2 }}>
                    {feature.icon}
                  </Box>
                  <Typography className="feature-title" variant="h6">
                    {feature.title}
                  </Typography>
                  <Typography className="feature-description" variant="body2">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Sentiment Predictor */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" sx={{ textAlign: 'center', mb: 4, fontWeight: 600 }}>
          Try It Now
        </Typography>
        <SentimentPredictor />
      </Box>

      {/* Sentiment Distribution */}
      {stats && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h3" sx={{ textAlign: 'center', mb: 4, fontWeight: 600 }}>
            Sentiment Distribution
          </Typography>
          <Grid container spacing={3}>
            {Object.entries(stats.sentiment_distribution).map(([sentiment, count]) => (
              <Grid item xs={12} sm={4} key={sentiment}>
                <Card className="stat-card">
                  <CardContent>
                    <Typography variant="h4" sx={{ 
                      color: sentiment === 'positive' ? '#4caf50' : 
                             sentiment === 'negative' ? '#f44336' : '#ff9800',
                      fontWeight: 600
                    }}>
                      {count.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                      {sentiment} Sentiments
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Container>
  );
};

export default Home;
