import React from 'react';
import { Box } from '@mui/material';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  Title,
} from 'chart.js';
import { SENTIMENT_COLORS, CHART_CONFIG } from '../../constants';

ChartJS.register(ArcElement, Tooltip, Legend, Title);

/**
 * Sentiment distribution doughnut chart component
 * @param {Object} props - Component props
 * @param {Object} props.data - Sentiment distribution data
 * @param {string} props.title - Chart title
 * @param {number} props.height - Chart height
 */
const SentimentDistributionChart = ({ data, title = 'Sentiment Distribution', height = 300 }) => {
  const chartData = {
    labels: Object.keys(data).map(sentiment => 
      sentiment.charAt(0).toUpperCase() + sentiment.slice(1)
    ),
    datasets: [
      {
        data: Object.values(data),
        backgroundColor: [
          SENTIMENT_COLORS.positive,
          SENTIMENT_COLORS.negative,
          SENTIMENT_COLORS.neutral,
        ],
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 2,
        hoverOffset: 4,
      },
    ],
  };

  const total = Object.values(data).reduce((sum, count) => sum + count, 0);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: CHART_CONFIG.FONT_COLOR,
          padding: 20,
          font: {
            size: 12,
          },
          generateLabels: function(chart) {
            const data = chart.data;
            if (data.labels.length && data.datasets.length) {
              const dataset = data.datasets[0];
              const total = dataset.data.reduce((a, b) => a + b, 0);
              return data.labels.map((label, i) => {
                const value = dataset.data[i];
                const percentage = ((value / total) * 100).toFixed(1);
                return {
                  text: `${label}: ${percentage}%`,
                  fillStyle: dataset.backgroundColor[i],
                  strokeStyle: dataset.borderColor[i],
                  lineWidth: dataset.borderWidth,
                  hidden: false,
                  index: i,
                };
              });
            }
            return [];
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
            const label = context.label || '';
            const value = context.parsed;
            const percentage = ((value / total) * 100).toFixed(1);
            return `${label}: ${value} (${percentage}%)`;
          },
        },
      },
    },
    cutout: '60%',
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 1000,
      easing: 'easeInOutQuart',
    },
  };

  return (
    <Box sx={{ height, width: '100%', display: 'flex', justifyContent: 'center' }}>
      <Doughnut data={chartData} options={chartOptions} />
    </Box>
  );
};

export default SentimentDistributionChart;
