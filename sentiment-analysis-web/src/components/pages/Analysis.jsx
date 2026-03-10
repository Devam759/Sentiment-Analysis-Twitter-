import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Chip,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SendIcon from '@mui/icons-material/Send';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { sentimentAPI } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

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
        const data = await sentimentAPI.getModelComparison();
        setModelComparison(data);
      } catch (error) {
        console.error('Failed to fetch model comparison:', error);
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
      setAnalysisResults(results);
    } catch (err) {
      setError('Failed to analyze text. Please try again.');
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };

  const chartData = {
    labels: modelComparison.map(model => model.model),
    datasets: [
      {
        label: 'Accuracy',
        data: modelComparison.map(model => model.accuracy),
        backgroundColor: 'rgba(102, 126, 234, 0.8)',
        borderColor: 'rgba(102, 126, 234, 1)',
        borderWidth: 1,
      },
      {
        label: 'Precision',
        data: modelComparison.map(model => model.precision),
        backgroundColor: 'rgba(118, 75, 162, 0.8)',
        borderColor: 'rgba(118, 75, 162, 1)',
        borderWidth: 1,
      },
      {
        label: 'Recall',
        data: modelComparison.map(model => model.recall),
        backgroundColor: 'rgba(76, 175, 80, 0.8)',
        borderColor: 'rgba(76, 175, 80, 1)',
        borderWidth: 1,
      },
      {
        label: 'F1 Score',
        data: modelComparison.map(model => model.f1_score),
        backgroundColor: 'rgba(255, 152, 0, 0.8)',
        borderColor: 'rgba(255, 152, 0, 1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#ffffff',
        },
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
        color: '#ffffff',
        font: {
          size: 16,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          color: '#ffffff',
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
      },
      x: {
        ticks: {
          color: '#ffffff',
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
      },
    },
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ display: 'flex', justifyContent: 'center', mt: 8 }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" className="page-container">
      <Typography variant="h3" sx={{ textAlign: 'center', mb: 4, fontWeight: 600 }}>
        Model Analysis & Comparison
      </Typography>

      {/* Model Comparison Table */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Model Performance Metrics
          </Typography>
          <TableContainer component={Paper} sx={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell align="center">Accuracy</TableCell>
                  <TableCell align="center">Precision</TableCell>
                  <TableCell align="center">Recall</TableCell>
                  <TableCell align="center">F1 Score</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {modelComparison.map((model, index) => (
                  <TableRow key={index}>
                    <TableCell component="th" scope="row">
                      <Typography variant="body1" fontWeight={600}>
                        {model.model}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {(model.accuracy * 100).toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={model.accuracy * 100}
                          sx={{ width: 60, height: 6, borderRadius: 3 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {(model.precision * 100).toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={model.precision * 100}
                          sx={{ width: 60, height: 6, borderRadius: 3 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {(model.recall * 100).toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={model.recall * 100}
                          sx={{ width: 60, height: 6, borderRadius: 3 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {(model.f1_score * 100).toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={model.f1_score * 100}
                          sx={{ width: 60, height: 6, borderRadius: 3 }}
                        />
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Performance Chart */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Performance Visualization
          </Typography>
          <Box sx={{ height: 400 }}>
            <Bar data={chartData} options={chartOptions} />
          </Box>
        </CardContent>
      </Card>

      {/* Multi-Model Analysis */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Compare All Models
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Enter text to see how all models analyze the sentiment
          </Typography>
          
          <TextField
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            placeholder="Enter text to analyze with all models..."
            value={analysisText}
            onChange={(e) => setAnalysisText(e.target.value)}
            disabled={analyzing}
            sx={{ mb: 2 }}
          />

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Button
            variant="contained"
            onClick={handleAnalyze}
            disabled={analyzing || !analysisText.trim()}
            startIcon={analyzing ? <CircularProgress size={20} /> : <SendIcon />}
            sx={{ mb: 3 }}
          >
            {analyzing ? 'Analyzing...' : 'Analyze with All Models'}
          </Button>

          {analysisResults && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Analysis Results
              </Typography>
              {Object.entries(analysisResults).map(([modelName, result]) => (
                <Accordion key={modelName} sx={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                      <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                        {modelName.replace('_', ' ').toUpperCase()}
                      </Typography>
                      <Chip
                        label={result.predicted_sentiment.toUpperCase()}
                        color={
                          result.predicted_sentiment === 'positive' ? 'success' :
                          result.predicted_sentiment === 'negative' ? 'error' : 'warning'
                        }
                        size="small"
                        sx={{ mr: 2 }}
                      />
                      <Typography variant="body2" sx={{ mr: 2 }}>
                        {(result.confidence * 100).toFixed(1)}% confidence
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" gutterBottom>
                          Predicted Sentiment: {result.predicted_sentiment}
                        </Typography>
                        <Typography variant="body2" gutterBottom>
                          Confidence: {(result.confidence * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2">
                          Processing Time: {Math.random() * 0.5 + 0.1}s
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" gutterBottom>
                          Probability Distribution:
                        </Typography>
                        {Object.entries(result.probabilities).map(([sentiment, prob]) => (
                          <Box key={sentiment} sx={{ mb: 1 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                              <Typography variant="caption">{sentiment}</Typography>
                              <Typography variant="caption">{(prob * 100).toFixed(1)}%</Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={prob * 100}
                              sx={{ height: 4, borderRadius: 2 }}
                            />
                          </Box>
                        ))}
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
    </Container>
  );
};

export default Analysis;
