import React from 'react';
import { Box } from '@mui/material';
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
import { CHART_CONFIG } from '../../constants';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

/**
 * Model performance comparison chart component
 * @param {Object} props - Component props
 * @param {Array} props.data - Model performance data
 * @param {string} props.title - Chart title
 * @param {number} props.height - Chart height
 */
const ModelPerformanceChart = ({ data, title = 'Model Performance Comparison', height = 400 }) => {
  const chartData = {
    labels: data.map(model => model.displayName || model.model),
    datasets: [
      {
        label: 'Accuracy',
        data: data.map(model => model.accuracy),
        backgroundColor: 'rgba(102, 126, 234, 0.8)',
        borderColor: 'rgba(102, 126, 234, 1)',
        borderWidth: 1,
      },
      {
        label: 'Precision',
        data: data.map(model => model.precision),
        backgroundColor: 'rgba(118, 75, 162, 0.8)',
        borderColor: 'rgba(118, 75, 162, 1)',
        borderWidth: 1,
      },
      {
        label: 'Recall',
        data: data.map(model => model.recall),
        backgroundColor: 'rgba(76, 175, 80, 0.8)',
        borderColor: 'rgba(76, 175, 80, 1)',
        borderWidth: 1,
      },
      {
        label: 'F1 Score',
        data: data.map(model => model.f1_score),
        backgroundColor: 'rgba(255, 152, 0, 0.8)',
        borderColor: 'rgba(255, 152, 0, 1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: CHART_CONFIG.FONT_COLOR,
          padding: 20,
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: true,
        text: title,
        color: CHART_CONFIG.FONT_COLOR,
        font: {
          size: 16,
          weight: '600',
        },
        padding: {
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: CHART_CONFIG.FONT_COLOR,
        bodyColor: CHART_CONFIG.FONT_COLOR,
        borderColor: CHART_CONFIG.FONT_COLOR,
        borderWidth: 1,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += (context.parsed.y * 100).toFixed(1) + '%';
            }
            return label;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          color: CHART_CONFIG.FONT_COLOR,
          callback: function(value) {
            return (value * 100) + '%';
          },
        },
        grid: {
          color: CHART_CONFIG.GRID_COLOR,
          drawBorder: false,
        },
      },
      x: {
        ticks: {
          color: CHART_CONFIG.FONT_COLOR,
          maxRotation: 45,
          minRotation: 45,
        },
        grid: {
          color: CHART_CONFIG.GRID_COLOR,
          drawBorder: false,
        },
      },
    },
    interaction: {
      intersect: false,
      mode: 'index',
    },
    animation: {
      duration: 1000,
      easing: 'easeInOutQuart',
    },
  };

  return (
    <Box sx={{ height, width: '100%' }}>
      <Bar data={chartData} options={chartOptions} />
    </Box>
  );
};

export default ModelPerformanceChart;
