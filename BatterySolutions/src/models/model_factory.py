from typing import Dict, Any, Union, Optional
from pathlib import Path

from src.models.base_model import BaseModel
# Import models with try-except to handle missing dependencies
try:
    from src.models.lstm_model import LSTMModel, TF_AVAILABLE
except ImportError:
    TF_AVAILABLE = False
    print("LSTM model not available due to missing dependencies")

try:
    from src.models.random_forest_model import RandomForestModel
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False
    print("Random Forest model not available due to missing dependencies")

try:
    from src.models.xgboost_model import XGBoostModel, XGB_AVAILABLE
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost model not available due to missing dependencies")

from src.config.config import MODEL_DIR


class ModelFactory:
    """
    Factory class to create different types of models.
    Implements Factory Method pattern to abstract model creation.
    """

    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create ('lstm', 'random_forest', 'xgboost')
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If an unsupported model type is provided
            ImportError: If the required dependencies for the model are not available
        """
        model_type = model_type.lower()
        
        if model_type == 'lstm':
            if not TF_AVAILABLE:
                raise ImportError("Cannot create LSTM model: TensorFlow is not available")
            return LSTMModel(**kwargs)
        elif model_type in ['random_forest', 'rf']:
            if not RF_AVAILABLE:
                raise ImportError("Cannot create Random Forest model: scikit-learn is not available")
            return RandomForestModel(**kwargs)
        elif model_type in ['xgboost', 'xgb']:
            if not XGB_AVAILABLE:
                raise ImportError("Cannot create XGBoost model: XGBoost is not available")
            return XGBoostModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                            f"Supported types are: 'lstm', 'random_forest', 'xgboost'")
    
    @staticmethod
    def get_available_models() -> Dict[str, bool]:
        """
        Return a dictionary of available models based on installed dependencies.
        
        Returns:
            Dictionary with model names as keys and availability status as values
        """
        return {
            'lstm': TF_AVAILABLE,
            'random_forest': RF_AVAILABLE,
            'xgboost': XGB_AVAILABLE
        }
    
    @staticmethod
    def load_model(model_type: str, filename: Optional[str] = None, **kwargs) -> BaseModel:
        """
        Load a trained model from disk.
        
        Args:
            model_type: Type of model to load ('lstm', 'random_forest', 'xgboost')
            filename: Name of the file to load (without extension)
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            Loaded model instance
        """
        model = ModelFactory.create_model(model_type, **kwargs)
        
        if filename is None:
            filename = f"{model.name}"
        
        # Load the model
        success = model.load_model(filename)
        
        if not success:
            raise ValueError(f"Failed to load model from {filename}")
        
        return model 