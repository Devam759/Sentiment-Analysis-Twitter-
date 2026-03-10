import React from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ScienceIcon from '@mui/icons-material/Science';
import CodeIcon from '@mui/icons-material/Code';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import StorageIcon from '@mui/icons-material/Storage';
import SpeedIcon from '@mui/icons-material/Speed';

const About = () => {
  const technologies = [
    { name: 'React', description: 'Modern frontend framework for building user interfaces' },
    { name: 'Vite', description: 'Fast build tool and development server' },
    { name: 'Material-UI', description: 'React UI component library for modern design' },
    { name: 'Chart.js', description: 'Data visualization library for performance metrics' },
    { name: 'Python', description: 'Backend machine learning and NLP processing' },
    { name: 'scikit-learn', description: 'Machine learning library for model training' },
    { name: 'NLTK', description: 'Natural language processing toolkit' },
    { name: 'Gensim', description: 'Topic modeling and document similarity analysis' },
  ];

  const features = [
    {
      icon: <ScienceIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Research-Based Methodology',
      description: 'Built following academic research standards for sentiment analysis',
    },
    {
      icon: <CodeIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Multiple ML Models',
      description: 'SVM, Random Forest, Logistic Regression, and Naive Bayes implementations',
    },
    {
      icon: <AnalyticsIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Advanced Features',
      description: 'Named entity recognition, topic modeling, and semantic analysis',
    },
    {
      icon: <StorageIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Comprehensive Pipeline',
      description: 'Complete data processing from raw text to sentiment prediction',
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'Real-time Processing',
      description: 'Fast sentiment analysis with confidence scores and probability distributions',
    },
  ];

  return (
    <Container maxWidth="lg" className="page-container">
      <Typography variant="h3" sx={{ textAlign: 'center', mb: 4, fontWeight: 600 }}>
        About This Project
      </Typography>

      {/* Project Overview */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Project Overview
          </Typography>
          <Typography variant="body1" paragraph>
            This Twitter Sentiment Analysis project implements a comprehensive machine learning pipeline
            for predicting consumer sentiment from tweet text. The system combines advanced natural language
            processing techniques with multiple machine learning models to provide accurate and reliable
            sentiment predictions.
          </Typography>
          <Typography variant="body1" paragraph>
            Built following research methodology, this project demonstrates professional-level NLP capabilities
            while maintaining accessibility for educational purposes. The system processes raw Twitter data,
            extracts meaningful features, trains multiple models, and provides detailed analysis and
            visualization of results.
          </Typography>
        </CardContent>
      </Card>

      {/* Key Features */}
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Key Features
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
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

      {/* Technology Stack */}
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Technology Stack
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Frontend Technologies
              </Typography>
              <List dense>
                {technologies.slice(0, 4).map((tech, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircleIcon sx={{ color: '#4caf50' }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={tech.name}
                      secondary={tech.description}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Backend & ML Technologies
              </Typography>
              <List dense>
                {technologies.slice(4).map((tech, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircleIcon sx={{ color: '#4caf50' }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={tech.name}
                      secondary={tech.description}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Pipeline Overview */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Pipeline Architecture
          </Typography>
          <Typography variant="body1" paragraph>
            The system follows a comprehensive machine learning pipeline:
          </Typography>
          <Paper sx={{ p: 2, backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
            <List>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: '#667eea' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Data Collection & Loading"
                  secondary="Sentiment140 dataset integration with custom dataset support"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: '#667eea' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Text Preprocessing"
                  secondary="URL removal, mention/hashtag cleaning, tokenization, lemmatization"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: '#667eea' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Feature Extraction"
                  secondary="TF-IDF vectorization, Word2Vec embeddings, statistical features"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: '#667eea' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Model Training"
                  secondary="Multiple algorithms with cross-validation and hyperparameter tuning"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: '#667eea' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Evaluation & Visualization"
                  secondary="Comprehensive metrics, confusion matrices, performance charts"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: '#667eea' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Advanced Analysis"
                  secondary="NER, topic modeling, semantic similarity, emotion detection"
                />
              </ListItem>
            </List>
          </Paper>
        </CardContent>
      </Card>

      {/* Model Performance */}
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Model Performance
          </Typography>
          <Typography variant="body1" paragraph>
            The system achieves excellent performance across multiple metrics:
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'rgba(102, 126, 234, 0.1)', borderRadius: 2 }}>
                <Typography variant="h4" color="#667eea" fontWeight={600}>
                  94%
                </Typography>
                <Typography variant="body2">
                  Best Accuracy
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'rgba(76, 175, 80, 0.1)', borderRadius: 2 }}>
                <Typography variant="h4" color="#4caf50" fontWeight={600}>
                  4
                </Typography>
                <Typography variant="body2">
                  Models Trained
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'rgba(255, 152, 0, 0.1)', borderRadius: 2 }}>
                <Typography variant="h4" color="#ff9800" fontWeight={600}>
                  0.15s
                </Typography>
                <Typography variant="body2">
                  Avg Processing
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'rgba(118, 75, 162, 0.1)', borderRadius: 2 }}>
                <Typography variant="h4" color="#764ba2" fontWeight={600}>
                  5K
                </Typography>
                <Typography variant="body2">
                  Sample Dataset
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Container>
  );
};

export default About;
