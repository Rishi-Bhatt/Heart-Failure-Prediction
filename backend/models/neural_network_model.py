"""
Neural Network Model for Heart Failure Prediction

This module implements a neural network model for heart failure prediction
with configurable epochs and early stopping.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import io
import base64

class NeuralNetworkModel:
    """
    Neural Network model for heart failure prediction
    """
    
    def __init__(self, input_dim=25, model_path='models/nn_model.h5', scaler_path='models/nn_scaler.pkl'):
        """
        Initialize the neural network model
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        model_path : str
            Path to save/load the model
        scaler_path : str
            Path to save/load the scaler
        """
        self.input_dim = input_dim
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Initialize or load model and scaler
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"Neural network model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading neural network model: {str(e)}")
                self._create_model()
                self.scaler = StandardScaler()
        else:
            self._create_model()
            self.scaler = StandardScaler()
    
    def _create_model(self):
        """
        Create a new neural network model
        """
        model = Sequential([
            Dense(64, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        print("Created new neural network model")
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target vector
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        verbose : int
            Verbosity level
            
        Returns:
        --------
        history : dict
            Training history
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save scaler
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        
        return history.history
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target vector
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        X_scaled = self.scaler.transform(X)
        loss, accuracy, auc, precision, recall = self.model.evaluate(X_scaled, y, verbose=0)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(2 * precision * recall / (precision + recall + 1e-10))
        }
    
    def get_training_plot(self, history):
        """
        Generate training history plot
        
        Parameters:
        -----------
        history : dict
            Training history from fit method
            
        Returns:
        --------
        plot_data : str
            Base64 encoded PNG image of the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot AUC
        plt.subplot(2, 2, 3)
        plt.plot(history['auc'], label='Training AUC')
        plt.plot(history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        # Plot learning rate if available
        if 'lr' in history:
            plt.subplot(2, 2, 4)
            plt.plot(history['lr'], label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
        
        plt.tight_layout()
        
        # Convert plot to base64 encoded PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return plot_data
