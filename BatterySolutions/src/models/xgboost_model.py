import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available. XGBoost model functionality will be limited.")
    XGB_AVAILABLE = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import psutil
import joblib
import time

from src.models.base_model import BaseModel
from src.config.config import XGB_PARAMS


class XGBoostModel(BaseModel):
    """
    XGBoost model implementation for battery SoH prediction.
    Follows Single Responsibility Principle by focusing on XGBoost-specific implementation.
    Optimized version with advanced features for better performance.
    """
    
    def __init__(self, name: str = "xgboost_model", **kwargs):
        """Initialize the XGBoost model with default or custom parameters."""
        super().__init__(name=name, **kwargs)
        self.params = XGB_PARAMS.copy()
        # Update params with any provided kwargs
        self.params.update(kwargs)
        
        # Track optimization metrics
        self.train_time = None
        self.inference_time = None
        self.memory_usage = None
        
        if not XGB_AVAILABLE:
            print("Warning: XGBoost is not available. XGBoost model will not work.")
    
    def build_model(self, **kwargs) -> Any:
        """
        Build the XGBoost model with specified parameters.
        
        Args:
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Initialized XGBoost model
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not available. Cannot build XGBoost model.")
            
        # Update parameters if provided
        if kwargs:
            self.params.update(kwargs)
        
        # Extract optimized parameters
        # Core parameters
        n_estimators = self.params.get('n_estimators', 200)  # Increased from default
        max_depth = self.params.get('max_depth', 6)  # Increased from default
        learning_rate = self.params.get('learning_rate', 0.05)  # Decreased from default for better generalization
        booster = self.params.get('booster', 'gbtree')  # Can be 'gbtree', 'gblinear', or 'dart'
        
        # Learning task parameters
        objective = self.params.get('objective', 'reg:squarederror')
        eval_metric = self.params.get('eval_metric', ['rmse', 'mae'])
        
        # Regularization parameters
        gamma = self.params.get('gamma', 0.1)  # Minimum loss reduction to create a new tree split
        reg_alpha = self.params.get('reg_alpha', 0.1)  # L1 regularization
        reg_lambda = self.params.get('reg_lambda', 1.0)  # L2 regularization
        
        # Tree parameters
        subsample = self.params.get('subsample', 0.8)  # Sampling ratio for training instances
        colsample_bytree = self.params.get('colsample_bytree', 0.8)  # Feature sampling ratio per tree
        colsample_bylevel = self.params.get('colsample_bylevel', 1.0)  # Feature sampling ratio per level
        min_child_weight = self.params.get('min_child_weight', 1)  # Minimum sum of instance weight in a child
        
        # Other parameters
        random_state = self.params.get('random_state', 42)
        n_jobs = self.params.get('n_jobs', -1)
        tree_method = self.params.get('tree_method', 'auto')  # 'auto', 'exact', 'approx', 'hist', 'gpu_hist'
        
        # Create XGBoost model with optimized parameters
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            objective=objective,
            eval_metric=eval_metric,
            booster=booster,
            tree_method=tree_method,
            random_state=random_state,
            n_jobs=n_jobs,
            use_label_encoder=False if hasattr(xgb.XGBRegressor(), 'use_label_encoder') else None
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the XGBoost model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Validation features (optional)
            y_val: Validation target values (optional)
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Dictionary containing training information
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not available. Cannot train XGBoost model.")
        
        # Build model if not already built
        if self.model is None:
            self.build_model(**kwargs)
        
        # Update parameters if provided
        if kwargs:
            self.params.update(kwargs)
        
        # Check if auto-tuning is enabled
        auto_tune = kwargs.get('auto_tune', self.params.get('auto_tune', False))
        if auto_tune:
            self._hyperparameter_tuning(X_train, y_train)
        
        # Start measuring time and memory
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
        
        # Prepare early stopping and validation parameters
        fit_params = {}
        
        # Enable custom callbacks
        callbacks = []
        
        # Add learning rate scheduler if enabled
        use_lr_scheduler = self.params.get('use_lr_scheduler', False)
        if use_lr_scheduler:
            scheduler = kwargs.get('lr_scheduler', lambda epoch: max(0.01, learning_rate * 0.95 ** epoch))
            lr_callback = self._create_learning_rate_callback(scheduler)
            callbacks.append(lr_callback)
        
        # Set up validation and early stopping
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # If we have both train and val, we can monitor both
            if self.params.get('monitor_train', False):
                eval_set.insert(0, (X_train, y_train))
                
            fit_params.update({
                'eval_set': eval_set,
                'early_stopping_rounds': self.params.get('early_stopping_rounds', 20),
                'verbose': self.params.get('verbose', True)
            })
        
        # Add callbacks to fit_params if any
        if callbacks:
            fit_params['callbacks'] = callbacks
        
        # Train the model
        self.model.fit(X_train, y_train, **fit_params)
        self.is_trained = True
        
        # Record training time and memory usage
        self.train_time = time.time() - start_time
        self.memory_usage = (psutil.Process().memory_info().rss / (1024 * 1024)) - start_mem
        
        # Get best iteration if available
        best_iteration = getattr(self.model, 'best_iteration', None)
        best_score = getattr(self.model, 'best_score', None)
        
        return {
            'model': self.model,
            'params': self.params,
            'is_trained': self.is_trained,
            'train_time': self.train_time,
            'memory_usage': self.memory_usage,
            'best_iteration': best_iteration,
            'best_score': best_score
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained XGBoost model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predicted values as numpy array
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        # Time inference
        start_time = time.time()
        
        # Use best iteration if available for prediction
        if hasattr(self.model, 'best_ntree_limit'):
            predictions = self.model.predict(X, ntree_limit=self.model.best_ntree_limit)
        else:
            predictions = self.model.predict(X)
        
        # Record inference time
        self.inference_time = time.time() - start_time
        
        return predictions
    
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
            'CPU_usage': cpu_usage,
            'inference_time': self.inference_time,
            'train_time': self.train_time,
            'memory_usage': self.memory_usage
        }
        
        # Add error analysis
        error_abs = np.abs(y_test - y_pred)
        self.model_metrics.update({
            'error_max': np.max(error_abs),
            'error_min': np.min(error_abs),
            'error_std': np.std(error_abs),
            'error_median': np.median(error_abs),
            'error_mean': np.mean(error_abs)
        })
        
        # Add feature importance
        try:
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
                self.model_metrics['feature_importance'] = feature_importance.tolist()
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
        
        # Use XGBoost's get_score method if possible for more detailed feature importance
        if hasattr(self.model, 'get_booster'):
            try:
                # Try to get feature importance by weight (covers most XGBoost versions)
                score_dict = self.model.get_booster().get_score(importance_type='weight')
                
                # If feature_names provided, map them to the scores
                if feature_names is not None:
                    # Create a mapping from default feature names (f0, f1, etc.) to provided names
                    feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}
                    
                    # Map the scores to the provided feature names
                    result = {feature_map.get(f, f): v for f, v in score_dict.items()}
                    return result
                return score_dict
            except:
                # Fall back to standard feature_importances_ attribute
                pass
        
        # Standard approach using feature_importances_ attribute
        importances = self.model.feature_importances_
        
        if feature_names is None:
            return {f"feature_{i}": importance for i, importance in enumerate(importances)}
        else:
            if len(feature_names) != len(importances):
                raise ValueError("Length of feature_names must match number of features.")
            return {name: importance for name, importance in zip(feature_names, importances)}
    
    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 3) -> Dict:
        """
        Perform hyperparameter tuning using randomized search cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target values
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and results
        """
        print("Starting XGBoost hyperparameter tuning...")
        
        # Define parameter space for advanced tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0.1, 0.5, 1, 5],
        }
        
        # Initialize base model for randomized search
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.params.get('random_state', 42),
            n_jobs=self.params.get('n_jobs', -1),
            tree_method=self.params.get('tree_method', 'auto')
        )
        
        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter settings sampled
            cv=cv,
            verbose=2,
            random_state=self.params.get('random_state', 42),
            n_jobs=self.params.get('n_jobs', -1),
            scoring='neg_mean_squared_error'
        )
        
        # Perform randomized search
        random_search.fit(X_train, y_train)
        
        # Update model parameters with best parameters
        best_params = random_search.best_params_
        print(f"Best parameters found: {best_params}")
        self.params.update(best_params)
        
        # Rebuild and train model with best parameters
        self.build_model()
        
        # Return results
        return {
            'best_params': best_params,
            'best_score': random_search.best_score_,
            'results': random_search.cv_results_
        }
    
    def _create_learning_rate_callback(self, scheduler_func):
        """
        Create a learning rate scheduler callback for XGBoost.
        
        Args:
            scheduler_func: Function that takes epoch number and returns learning rate
            
        Returns:
            Callback object
        """
        # Define a custom callback to adjust learning rate
        class LearningRateScheduler:
            def __init__(self, scheduler_func):
                self.scheduler = scheduler_func
                self.current_epoch = 0
                
            def __call__(self, env):
                # Called after each iteration
                if env.iteration % env.end_iteration == 0 and env.iteration > 0:
                    self.current_epoch += 1
                    new_lr = self.scheduler(self.current_epoch)
                    env.model.set_param('learning_rate', new_lr)
                    print(f"Epoch {self.current_epoch}: learning rate set to {new_lr}")
        
        return LearningRateScheduler(scheduler_func)
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained XGBoost model to disk.
        
        Args:
            filename: Name of the file to save the model (without extension)
            
        Returns:
            Path where the model was saved
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving.")
        
        if filename is None:
            filename = f"{self.name}"
        
        # Save with both XGBoost native format and joblib
        native_file_path = f"{self.model_dir}/{filename}.json"
        joblib_file_path = f"{self.model_dir}/{filename}.joblib"
        
        try:
            # Save model in XGBoost's native format
            if hasattr(self.model, 'get_booster'):
                self.model.get_booster().save_model(native_file_path)
                print(f"Model saved in XGBoost native format to {native_file_path}")
            
            # Save full model with joblib
            joblib.dump(self.model, joblib_file_path)
            print(f"Model saved with joblib to {joblib_file_path}")
            
            return native_file_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return ""
    
    def load_model(self, filename: Optional[str] = None) -> bool:
        """
        Load a trained XGBoost model from disk.
        
        Args:
            filename: Name of the file to load the model from (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        if filename is None:
            filename = f"{self.name}"
        
        # Try to load from joblib first, then fall back to XGBoost native format
        joblib_file_path = f"{self.model_dir}/{filename}.joblib"
        native_file_path = f"{self.model_dir}/{filename}.json"
        
        try:
            # Try loading with joblib first (preserves all attributes)
            if joblib_file_path:
                self.model = joblib.load(joblib_file_path)
                self.is_trained = True
                print(f"Model loaded from {joblib_file_path}")
                return True
        except Exception as e:
            print(f"Error loading model from {joblib_file_path}: {e}")
            
            # Fall back to XGBoost native format
            try:
                if self.model is None:
                    self.build_model()
                self.model.get_booster().load_model(native_file_path)
                self.is_trained = True
                print(f"Model loaded from {native_file_path}")
                return True
            except Exception as e2:
                print(f"Error loading model from {native_file_path}: {e2}")
                
        return False

    def feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None,
                          threshold: float = 0.01) -> Dict:
        """
        Perform feature selection based on feature importance.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Names of features (optional)
            threshold: Importance threshold for feature selection
            
        Returns:
            Dictionary with selected features and their importance
        """
        if self.model is None:
            # Train a simple model for feature selection
            self.build_model(n_estimators=100)
            self.train(X, y)
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create feature map with or without names
        if feature_names is None:
            feature_map = {i: f"feature_{i}" for i in range(len(importances))}
        else:
            if len(feature_names) != len(importances):
                raise ValueError("Length of feature_names must match number of features.")
            feature_map = {i: name for i, name in enumerate(feature_names)}
        
        # Select features above threshold
        selected_indices = np.where(importances > threshold)[0]
        selected_features = {feature_map[i]: importances[i] for i in selected_indices}
        
        # Sort by importance (descending)
        selected_features = dict(sorted(selected_features.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'selected_features': selected_features,
            'feature_indices': selected_indices.tolist(),
            'threshold': threshold,
            'n_selected': len(selected_features)
        } 