import React from 'react';
import { Alert, AlertTitle, IconButton, Collapse } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

/**
 * Error alert component with dismiss functionality
 * @param {Object} props - Component props
 * @param {string} props.message - Error message
 * @param {string} props.title - Error title (optional)
 * @param {boolean} props.severity - Alert severity
 * @param {boolean} props.dismissible - Whether alert can be dismissed
 */
const ErrorAlert = ({ 
  message, 
  title, 
  severity = 'error',
  dismissible = true,
  onClose 
}) => {
  const [open, setOpen] = React.useState(true);

  const handleClose = () => {
    setOpen(false);
    if (onClose) onClose();
  };

  if (!open) return null;

  return (
    <Alert
      severity={severity}
      action={
        dismissible && (
          <IconButton
            aria-label="close"
            color="inherit"
            size="small"
            onClick={handleClose}
          >
            <CloseIcon fontSize="inherit" />
          </IconButton>
        )
      }
      sx={{ mb: 2 }}
    >
      {title && <AlertTitle>{title}</AlertTitle>}
      {message}
    </Alert>
  );
};

export default ErrorAlert;
