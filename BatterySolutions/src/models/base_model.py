from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union
import os
from pathlib import Path
import pickle
import time

from src.config.config import MODEL_DIR


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    Follows Interface Segregation Principle by defining common interfaces.
    """
    
    def __init__(self, name: str = "base_model", model_dir: Path = MODEL_DIR):
        self.name = name
        self.model_dir = model_dir
        self.model = None
        self.is_trained = False
        self.training_time = None
        self.prediction_time = None
        self.model_metrics = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    @abstractmethod
    def build_model(self, **kwargs):
        """Build the machine learning model with specified hyperparameters."""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train the model with training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model using test data and return performance metrics."""
        pass
    
    def measure_training_time(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> float:
        """Measure the time taken to train the model."""
        start_time = time.time()
        self.train(X_train, y_train, **kwargs)
        end_time = time.time()
        self.training_time = end_time - start_time
        return self.training_time
    
    def measure_prediction_time(self, X: np.ndarray) -> float:
        """Measure the time taken to make predictions."""
        start_time = time.time()
        _ = self.predict(X)
        end_time = time.time()
        self.prediction_time = end_time - start_time
        return self.prediction_time
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        if filename is None:
            filename = f"{self.name}_model.pkl"
        
        file_path = os.path.join(self.model_dir, filename)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            print(f"Model saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving model to {file_path}: {e}")
            return ""
    
    def load_model(self, filename: Optional[str] = None) -> bool:
        """Load a trained model from disk."""
        if filename is None:
            filename = f"{self.name}_model.pkl"
        
        file_path = os.path.join(self.model_dir, filename)
        
        try:
            with open(file_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_trained = True
            print(f"Model loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
            return False
    
    def save_metrics(self, metrics: Dict[str, float], filename: Optional[str] = None) -> str:
        """Save model performance metrics to disk."""
        if filename is None:
            filename = f"{self.name}_metrics.pkl"
        
        file_path = os.path.join(self.model_dir, filename)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            print(f"Metrics saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving metrics to {file_path}: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including training time, prediction time, and metrics."""
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'metrics': self.model_metrics
        } 