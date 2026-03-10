import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container
} from '@mui/material';
import SentimentSatisfiedAltIcon from '@mui/icons-material/SentimentSatisfiedAlt';

const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { label: 'Home', path: '/' },
    { label: 'Analysis', path: '/analysis' },
    { label: 'About', path: '/about' },
  ];

  return (
    <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Container maxWidth="lg">
        <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <SentimentSatisfiedAltIcon sx={{ mr: 2, fontSize: 32 }} />
            <Typography
              variant="h6"
              noWrap
              component={Link}
              to="/"
              sx={{
                mr: 2,
                fontWeight: 700,
                color: 'inherit',
                textDecoration: 'none',
              }}
            >
              Sentiment Analysis
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            {navItems.map((item) => (
              <Button
                key={item.path}
                component={Link}
                to={item.path}
                sx={{
                  color: location.pathname === item.path ? 'primary.main' : 'inherit',
                  backgroundColor: location.pathname === item.path ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  },
                }}
              >
                {item.label}
              </Button>
            ))}
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Navbar;
