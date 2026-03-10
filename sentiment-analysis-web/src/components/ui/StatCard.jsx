import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

/**
 * Statistics card component
 * @param {Object} props - Component props
 * @param {string|number} props.value - Main value to display
 * @param {string} props.label - Label for the value
 * @param {string} props.color - Color theme
 * @param {React.ReactNode} props.icon - Icon to display
 * @param {string} props.unit - Unit to display after value
 */
const StatCard = ({ 
  value, 
  label, 
  color = '#667eea', 
  icon, 
  unit = '',
  subtitle 
}) => {
  return (
    <Card 
      className="stat-card"
      sx={{
        background: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(10px)',
        transition: 'transform 0.3s ease, box-shadow 0.3s ease',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: '0 10px 30px rgba(102, 126, 234, 0.3)',
        },
      }}
    >
      <CardContent sx={{ textAlign: 'center', py: 3 }}>
        {icon && (
          <Box sx={{ mb: 2, color }}>
            {icon}
          </Box>
        )}
        
        <Typography 
          className="stat-number"
          variant="h3" 
          sx={{ 
            fontWeight: 700, 
            color,
            mb: 0.5,
            fontSize: { xs: '2rem', md: '2.5rem' }
          }}
        >
          {value}{unit}
        </Typography>
        
        <Typography 
          className="stat-label"
          variant="body2" 
          sx={{ 
            color: '#b8bcc8',
            mb: subtitle ? 1 : 0,
            fontSize: { xs: '0.875rem', md: '1rem' }
          }}
        >
          {label}
        </Typography>
        
        {subtitle && (
          <Typography 
            variant="caption" 
            sx={{ 
              color: '#888',
              fontSize: '0.75rem'
            }}
          >
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard;
