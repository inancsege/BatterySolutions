import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Data directories
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "Dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# Model directories
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Reports directory for visualizations
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Model parameters
# LSTM parameters - Optimized
LSTM_PARAMS = {
    # Training parameters
    "epochs": 150,
    "batch_size": 32,
    "validation_split": 0.2,
    "patience": 20,
    "verbose": 1,
    
    # Model architecture
    "model_type": "complex",  # Options: 'lstm', 'bidirectional', 'cnn_lstm', 'gru', 'complex'
    "use_attention": False,
    "learning_rate": 0.001,
    "use_batch_norm": True,
    "dropout_rate": 0.3,
    "recurrent_dropout": 0.2,
    "units": [128, 64],
    "dense_layers": [32],
    "activation": "relu",
    
    # Regularization
    "regularization": "l2",  # Options: None, 'l1', 'l2', 'l1_l2'
    "regularization_factor": 0.001,
    "l1_factor": 0.001,
    "l2_factor": 0.001,
    
    # Learning rate schedule
    "learning_rate_decay": True,
    "decay_steps": 1000,
    "decay_rate": 0.9,
    
    # Computation control
    "use_artificial_delay": True,  # Add artificial computation to prevent instant epochs
    "delay_factor": 1.0,  # Higher value means more computation
    "workers": 0,  # Set to 0 to avoid TensorFlow worker issues
    "use_multiprocessing": False,
    
    # Additional features
    "use_tensorboard": True,
    "log_dir": "./logs",
    "model_dir": "./models",
    
    # Loss and metrics
    "loss": "mean_squared_error",
    "metrics": ["mae"]
}

# Random Forest parameters - Optimized
RF_PARAMS = {
    # Core parameters
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
    "criterion": "squared_error",
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": True,
    
    # Training options
    "warm_start": False,
    "auto_tune": True,  # Perform hyperparameter tuning automatically
    
    # Ensemble options
    "use_ensemble": True,  # Use ensemble methods for better performance
    "ensemble_type": "voting",  # Options: 'voting', 'bagging'
    "ensemble_weights": [1, 1, 1],  # Weights for voting ensemble
    "bagging_estimators": 10,  # Number of estimators for bagging
    "max_samples": 0.8,  # Sampling ratio for bagging
    "bagging_max_features": 0.8,  # Feature sampling ratio for bagging
    
    # Advanced parameters
    "compute_importance": True,  # Calculate feature importance
    "class_weight": "balanced",  # Weight classes inversely proportional to frequency
    "ccp_alpha": 0.0  # Complexity parameter for pruning
}

# XGBoost parameters - Optimized
XGB_PARAMS = {
    # Core parameters
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "gamma": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 1.0,
    "min_child_weight": 1,
    "random_state": 42,
    "n_jobs": -1,
    
    # Boosting parameters
    "booster": "gbtree",  # Options: 'gbtree', 'gblinear', 'dart'
    "tree_method": "auto",  # Options: 'auto', 'exact', 'approx', 'hist', 'gpu_hist'
    "grow_policy": "depthwise",  # Options: 'depthwise', 'lossguide'
    
    # Objective and metrics
    "objective": "reg:squarederror",
    "eval_metric": ["rmse", "mae"],
    "early_stopping_rounds": 20,
    
    # Regularization
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
    
    # Training control
    "verbose": 1,
    "auto_tune": True,  # Perform hyperparameter tuning
    "use_lr_scheduler": True,  # Use learning rate scheduling
    "monitor_train": True,  # Monitor training set performance
    
    # Advanced parameters
    "max_bin": 256,  # Maximum number of discrete bins for continuous features
    "scale_pos_weight": 1.0,  # Control balance of positive and negative weights
    "num_parallel_tree": 1,  # Number of parallel trees constructed
    "importance_type": "gain"  # Feature importance type: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
}

# Data processing parameters
DATA_PROCESSING_PARAMS = {
    # Sequence parameters for LSTM
    "sequence_length": 30,
    "test_size": 0.2,
    
    # Data augmentation
    "enable_augmentation": True,
    "augmentation_factor": 3,
    
    # Feature engineering
    "use_advanced_features": True,
    "lag_features": 5,
    
    # Normalization
    "scaler_type": "standard",  # Options: 'standard', 'minmax', 'robust'
    
    # Stratification
    "stratify_bins": 5
}

# Battery specifications
ORIGINAL_CAPACITY_AH = 140  # Adjust based on battery specifications

# Feature selection
FEATURES = [
    "voltage (V)",
    "current (A)",
    "temperature (°C)",
    "available_capacity (Ah)",
    "available_energy (kw)"
]

# Advanced features derived from base features
DERIVED_FEATURES = [
    "SoH_velocity",
    "SoH_acceleration",
    "capacity_variation",
    "capacity_range",
    "day_of_week",
    "month"
]

TARGET = "SoH_capacity" 