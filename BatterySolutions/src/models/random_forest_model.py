import numpy as np
from typing import Dict, Any, Optional, List, Union
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import psutil
import joblib

from src.models.base_model import BaseModel
from src.config.config import RF_PARAMS


class RandomForestModel(BaseModel):
    """
    Random Forest model implementation for battery SoH prediction.
    Follows Single Responsibility Principle by focusing on Random Forest-specific implementation.
    With added optimizations for performance and accuracy.
    """
    
    def __init__(self, name: str = "random_forest_model", **kwargs):
        """Initialize the Random Forest model with default or custom parameters."""
        super().__init__(name=name, **kwargs)
        self.params = RF_PARAMS.copy()
        # Update params with any provided kwargs
        self.params.update(kwargs)
        self.use_ensemble = kwargs.get('use_ensemble', False)
        self.ensemble_type = kwargs.get('ensemble_type', 'voting')
        self.base_models = []
    
    def build_model(self, **kwargs) -> Union[RandomForestRegressor, VotingRegressor, BaggingRegressor]:
        """
        Build the Random Forest model with specified hyperparameters.
        
        Args:
            **kwargs: Parameters to override defaults from RF_PARAMS
            
        Returns:
            Initialized model (RandomForestRegressor, VotingRegressor, or BaggingRegressor)
        """
        # Update parameters if provided
        if kwargs:
            self.params.update(kwargs)
        
        # Get parameters
        n_estimators = self.params.get('n_estimators', 100)
        max_depth = self.params.get('max_depth', None)
        min_samples_split = self.params.get('min_samples_split', 2)
        min_samples_leaf = self.params.get('min_samples_leaf', 1)
        random_state = self.params.get('random_state', 42)
        n_jobs = self.params.get('n_jobs', -1)
        criterion = self.params.get('criterion', 'squared_error')
        max_features = self.params.get('max_features', 'sqrt')
        bootstrap = self.params.get('bootstrap', True)
        oob_score = self.params.get('oob_score', True)
        
        # Create base Random Forest model
        base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            criterion=criterion,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            warm_start=self.params.get('warm_start', False)
        )
        
        # Check if we should use an ensemble approach
        if self.use_ensemble:
            if self.ensemble_type == 'voting':
                # Create diversified Random Forest models
                rf1 = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else 10,
                    min_samples_split=min_samples_split,
                    max_features='sqrt',
                    random_state=random_state
                )
                
                rf2 = RandomForestRegressor(
                    n_estimators=n_estimators + 50,
                    max_depth=max_depth if max_depth else 15, 
                    min_samples_split=min_samples_split + 1,
                    max_features='log2',
                    random_state=random_state + 1
                )
                
                rf3 = RandomForestRegressor(
                    n_estimators=n_estimators + 100,
                    max_depth=max_depth if max_depth else 20,
                    min_samples_split=min_samples_split - 1 if min_samples_split > 2 else 2,
                    max_features=0.8,
                    random_state=random_state + 2
                )
                
                # Save the base models
                self.base_models = [("rf1", rf1), ("rf2", rf2), ("rf3", rf3)]
                
                # Create voting ensemble
                model = VotingRegressor(
                    estimators=self.base_models,
                    weights=self.params.get('ensemble_weights', [1, 1, 1]),
                    n_jobs=n_jobs
                )
            else:  # Bagging
                model = BaggingRegressor(
                    base_estimator=base_rf,
                    n_estimators=self.params.get('bagging_estimators', 10),
                    max_samples=self.params.get('max_samples', 0.8),
                    max_features=self.params.get('bagging_max_features', 0.8),
                    bootstrap=True,
                    bootstrap_features=False,
                    random_state=random_state,
                    n_jobs=n_jobs
                )
        else:
            model = base_rf
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Any:
        """
        Train the Random Forest model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target values
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Trained model
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(**kwargs)
        
        # Check if auto tuning is enabled
        auto_tune = kwargs.get('auto_tune', self.params.get('auto_tune', False))
        if auto_tune:
            self._hyperparameter_tuning(X_train, y_train)
        
        # Use warm-up approach for better initialization if enabled
        warm_up = kwargs.get('warm_up', self.params.get('warm_up', False))
        if warm_up and not self.use_ensemble and hasattr(self.model, 'warm_start'):
            # Warm-up training with incremental trees
            self.model.n_estimators = 10
            self.model.fit(X_train, y_train)
            
            self.model.n_estimators = 50
            self.model.fit(X_train, y_train)
            
            self.model.n_estimators = self.params.get('n_estimators', 100)
            self.model.fit(X_train, y_train)
        else:
            # Train the model
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Print OOB score if available
        if hasattr(self.model, 'oob_score_'):
            print(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        elif hasattr(self.model, 'estimator_') and hasattr(self.model.estimator_, 'oob_score_'):
            print(f"Out-of-bag score: {self.model.estimator_.oob_score_:.4f}")
            
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained Random Forest model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predicted values as numpy array
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 80.0) -> Dict[str, float]:
        """
        Evaluate the model using test data and return performance metrics.
        
        Args:
            X_test: Test features
            y_test: Test target values
            threshold: Threshold for binary classification metrics (e.g., F1 score)
            
        Returns:
            Dictionary of evaluation metrics (MAE, RMSE, R^2, F1, CPU usage)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation.")
        
        # Generate predictions
        y_pred = self.predict(X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate binary classification metrics based on threshold
        # (e.g., if SoH < threshold, it's considered degraded)
        y_test_binary = (y_test < threshold).astype(int)
        y_pred_binary = (y_pred < threshold).astype(int)
        
        try:
            f1 = f1_score(y_test_binary, y_pred_binary)
        except:
            f1 = np.nan  # In case of division by zero or undefined
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent()
        
        # Store metrics
        self.model_metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R^2': r2,
            'F1': f1,
            'CPU_usage': cpu_usage
        }
        
        # Add additional metrics
        self.model_metrics['predicted_mean'] = np.mean(y_pred)
        self.model_metrics['actual_mean'] = np.mean(y_test)
        self.model_metrics['prediction_std'] = np.std(y_pred)
        
        # Add feature importance if available
        try:
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
                self.model_metrics['feature_importance'] = feature_importance.tolist()
            elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'feature_importances_'):
                # For ensemble models
                importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
                self.model_metrics['feature_importance'] = importances.tolist()
        except:
            pass
        
        return self.model_metrics
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Args:
            feature_names: Names of features to match with importance scores
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained to get feature importance.")
        
        # Get importances based on model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_') and isinstance(self.model, VotingRegressor):
            # For voting ensemble, average the feature importances of each base estimator
            all_importances = []
            for name, est in self.model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    all_importances.append(est.feature_importances_)
            
            if all_importances:
                importances = np.mean(all_importances, axis=0)
            else:
                return {'error': 'No feature importances available for this model'}
        elif hasattr(self.model, 'estimators_') and not isinstance(self.model, VotingRegressor):
            # For other ensemble methods
            importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        else:
            return {'error': 'No feature importances available for this model'}
        
        if feature_names is None:
            return {f"feature_{i}": importance for i, importance in enumerate(importances)}
        else:
            if len(feature_names) != len(importances):
                raise ValueError("Length of feature_names must match number of features.")
            return {name: importance for name, importance in zip(feature_names, importances)}
    
    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Perform quick hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target values
        """
        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.7, 0.8]
        }
        
        # Create the base model
        base_model = RandomForestRegressor(
            random_state=self.params.get('random_state', 42),
            n_jobs=self.params.get('n_jobs', -1)
        )
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            verbose=1,
            random_state=self.params.get('random_state', 42),
            n_jobs=self.params.get('n_jobs', -1),
            scoring='neg_mean_squared_error'
        )
        
        # Perform random search
        random_search.fit(X_train, y_train)
        
        # Update parameters with best ones found
        best_params = random_search.best_params_
        print(f"Best parameters found: {best_params}")
        
        # Update model parameters
        self.params.update(best_params)
        
        # Rebuild model with best parameters
        self.model = RandomForestRegressor(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            max_features=best_params.get('max_features', 'sqrt'),
            random_state=self.params.get('random_state', 42),
            n_jobs=self.params.get('n_jobs', -1),
            oob_score=True
        )
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained Random Forest model to disk using joblib.
        
        Args:
            filename: Name of the file to save the model (without extension)
            
        Returns:
            Path where the model was saved
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving.")
        
        if filename is None:
            filename = f"{self.name}"
        
        file_path = f"{self.model_dir}/{filename}.joblib"
        
        try:
            joblib.dump(self.model, file_path)
            print(f"Model saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving model to {file_path}: {e}")
            return ""
    
    def load_model(self, filename: Optional[str] = None) -> bool:
        """
        Load a trained Random Forest model from disk using joblib.
        
        Args:
            filename: Name of the file to load the model from (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        if filename is None:
            filename = f"{self.name}"
        
        file_path = f"{self.model_dir}/{filename}.joblib"
        
        try:
            self.model = joblib.load(file_path)
            self.is_trained = True
            print(f"Model loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
            return False 