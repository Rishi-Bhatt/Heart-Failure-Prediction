import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CardHeader, 
  Divider, 
  FormControl,
  FormControlLabel,
  FormGroup,
  Grid, 
  InputLabel, 
  MenuItem, 
  Select, 
  Slider, 
  Switch,
  TextField, 
  Typography 
} from '@mui/material';
import { API_BASE_URL } from '../config';

const ModelTrainingConfig = () => {
  const [config, setConfig] = useState({
    epochs: 50,
    use_neural_network: true,
    retraining_threshold: 20,
    drift_detection_threshold: 0.1,
    records_since_last_retraining: 0,
    last_retraining_date: null,
    retraining_count: 0
  });
  
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  
  // Fetch current configuration
  useEffect(() => {
    fetchConfig();
  }, []);
  
  const fetchConfig = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/retraining/config`);
      setConfig(response.data);
      setError('');
    } catch (err) {
      setError('Failed to fetch configuration: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };
  
  const handleChange = (e) => {
    const { name, value, checked } = e.target;
    if (name === 'use_neural_network') {
      setConfig({ ...config, [name]: checked });
    } else {
      setConfig({ ...config, [name]: value });
    }
  };
  
  const handleSliderChange = (name) => (e, value) => {
    setConfig({ ...config, [name]: value });
  };
  
  const handleSubmit = async () => {
    try {
      setLoading(true);
      setMessage('');
      setError('');
      
      // Only send the configurable parameters
      const configToSend = {
        epochs: config.epochs,
        use_neural_network: config.use_neural_network
      };
      
      const response = await axios.post(`${API_BASE_URL}/api/retrain`, configToSend);
      
      if (response.data.success) {
        setMessage('Model retraining configuration updated and retraining started successfully!');
        // Refresh config after a short delay to allow retraining to start
        setTimeout(fetchConfig, 1000);
      } else {
        setError('Retraining failed: ' + response.data.message);
      }
    } catch (err) {
      setError('Error: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Card>
      <CardHeader 
        title="Model Training Configuration" 
        subheader="Configure neural network training parameters"
      />
      <Divider />
      <CardContent>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h6">Neural Network Configuration</Typography>
          </Grid>
          
          <Grid item xs={12}>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.use_neural_network}
                    onChange={handleChange}
                    name="use_neural_network"
                    color="primary"
                  />
                }
                label="Use Neural Network Model"
              />
            </FormGroup>
          </Grid>
          
          <Grid item xs={12}>
            <Typography id="epochs-slider" gutterBottom>
              Training Epochs: {config.epochs}
            </Typography>
            <Slider
              value={config.epochs}
              onChange={handleSliderChange('epochs')}
              aria-labelledby="epochs-slider"
              valueLabelDisplay="auto"
              step={10}
              marks
              min={10}
              max={200}
              disabled={!config.use_neural_network}
            />
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Number of Epochs"
              name="epochs"
              type="number"
              value={config.epochs}
              onChange={handleChange}
              variant="outlined"
              disabled={!config.use_neural_network}
              InputProps={{ inputProps: { min: 1, max: 1000 } }}
            />
          </Grid>
          
          <Grid item xs={12}>
            <Divider />
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="h6">Current Status</Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              Records since last retraining: {config.records_since_last_retraining}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              Retraining threshold: {config.retraining_threshold}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              Last retraining date: {config.last_retraining_date ? new Date(config.last_retraining_date).toLocaleString() : 'Never'}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              Total retraining count: {config.retraining_count}
            </Typography>
          </Grid>
          
          <Grid item xs={12}>
            <Divider />
          </Grid>
          
          {message && (
            <Grid item xs={12}>
              <Typography variant="body1" color="primary">{message}</Typography>
            </Grid>
          )}
          
          {error && (
            <Grid item xs={12}>
              <Typography variant="body1" color="error">{error}</Typography>
            </Grid>
          )}
          
          <Grid item xs={12}>
            <Button
              color="primary"
              variant="contained"
              onClick={handleSubmit}
              disabled={loading}
              fullWidth
            >
              {loading ? 'Updating...' : 'Update Configuration & Retrain Model'}
            </Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ModelTrainingConfig;
