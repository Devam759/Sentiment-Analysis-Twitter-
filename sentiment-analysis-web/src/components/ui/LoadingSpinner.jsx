import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

/**
 * Loading spinner component with optional text
 * @param {Object} props - Component props
 * @param {string} props.message - Loading message
 * @param {number} props.size - Spinner size
 * @param {boolean} props.centered - Whether to center the spinner
 */
const LoadingSpinner = ({ 
  message = 'Loading...', 
  size = 40, 
  centered = true,
  color = 'primary' 
}) => {
  const content = (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      gap: 2 
    }}>
      <CircularProgress 
        size={size} 
        color={color}
        sx={{
          '& .MuiCircularProgress-circle': {
            strokeLinecap: 'round',
          },
        }}
      />
      <Typography 
        variant="body2" 
        color="text.secondary" 
        sx={{ textAlign: 'center' }}
      >
        {message}
      </Typography>
    </Box>
  );

  if (centered) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          minHeight: 200,
          width: '100%'
        }}
      >
        {content}
      </Box>
    );
  }

  return content;
};

export default LoadingSpinner;
