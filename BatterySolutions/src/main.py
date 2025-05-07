#!/usr/bin/env python
"""
Battery State of Health (SoH) Prediction System

This script provides a command-line interface to train and evaluate
different machine learning models for predicting battery SoH.
"""

import argparse
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Optional

# Add the project root to the path to allow absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Fix imports to use the correct paths
from src.data.data_loader import DataLoader
from src.models.model_factory import ModelFactory
from src.visualization.visualization import Visualizer
try:
    from src.utils.energy_monitor import EnergyMonitor
    ENERGY_MONITOR_AVAILABLE = True
except ImportError:
    ENERGY_MONITOR_AVAILABLE = False
    print("Energy monitoring is not available due to missing dependencies")
from src.config.config import DATASET_DIR, ORIGINAL_CAPACITY_AH

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'application.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse() -> argparse.Namespace:
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Battery State of Health (SoH) Prediction System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    parser.add_argument('--data-file', type=str, default=None,
                      help='Specific data file to use (if None, will use the first file found)')
    
    # Model options
    available_models = list(ModelFactory.get_available_models().keys())
    available_models.append('all')
    parser.add_argument('--model-type', type=str, choices=available_models,
                      default='all', help='Type of model to train')
    
    # Training options
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--sequence-length', type=int, default=30,
                      help='Sequence length for LSTM model')
    parser.add_argument('--lag-features', type=int, default=5,
                      help='Number of lag features for Random Forest and XGBoost models')
    
    # Execution options
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--monitor-energy', action='store_true', help='Monitor energy consumption')
    
    # Output options
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--load-model', action='store_true', help='Load a previously trained model')
    
    # Add a list option to show available models
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    return parser.parse_args()


def train_evaluate_model(model_type: str, data_loader: DataLoader, args: argparse.Namespace, 
                        energy_monitor: Optional[EnergyMonitor] = None) -> Dict[str, Any]:
    """Train and evaluate a specific model type."""
    results = {}
    
    # Check if model type is available
    available_models = ModelFactory.get_available_models()
    if model_type not in available_models or not available_models[model_type]:
        logger.error(f"{model_type} model is not available due to missing dependencies")
        return results
    
    # Get data file path
    if args.data_file:
        data_file = os.path.join(DATASET_DIR, args.data_file)
    else:
        # Use the first CSV file in the dataset directory
        csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
        if not csv_files:
            logger.error("No CSV files found in the dataset directory")
            return results
        data_file = os.path.join(DATASET_DIR, csv_files[0])
    
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} does not exist")
        return results
    
    logger.info(f"Using data file: {data_file}")
    
    try:
        # Create model
        logger.info(f"Creating {model_type} model")
        model = ModelFactory.create_model(model_type)
        results['model'] = model
        
        # Load model if requested
        if args.load_model:
            logger.info(f"Loading trained {model_type} model")
            try:
                success = model.load_model()
                if not success:
                    logger.warning("Failed to load model. Will train a new one if training is enabled.")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        # Train model if requested
        if args.train:
            logger.info(f"Training {model_type} model")
            
            # Prepare data with better error handling
            try:
                if model_type == 'lstm':
                    # Prepare data for LSTM
                    data = data_loader.prepare_data_for_lstm(
                        data_file, 
                        sequence_length=args.sequence_length,
                        test_size=args.test_size
                    )
                else:
                    # Prepare data for Random Forest or XGBoost
                    data = data_loader.prepare_data_for_rf_xgb(
                        data_file,
                        lag_features=args.lag_features,
                        test_size=args.test_size
                    )
                
                # Verify data shapes
                if any(key not in data for key in ['X_train', 'X_test', 'y_train', 'y_test']):
                    raise ValueError("Missing required data components")
                
                if len(data['X_train']) == 0 or len(data['y_train']) == 0:
                    raise ValueError("Empty training dataset")
                
                # Store data for later use
                results['data'] = data
                
                # Log data shapes for debugging
                logger.info(f"Data shapes - X_train: {data['X_train'].shape}, "
                           f"y_train: {data['y_train'].shape}, "
                           f"X_test: {data['X_test'].shape}, "
                           f"y_test: {data['y_test'].shape}")
                
                # Train with energy monitoring if requested
                if energy_monitor and args.monitor_energy and ENERGY_MONITOR_AVAILABLE:
                    logger.info("Monitoring energy during training")
                    
                    @energy_monitor.energy_monitor_decorator
                    def train_model():
                        return model.train(data['X_train'], data['y_train'])
                    
                    training_result = train_model()
                    results['training_result'] = training_result
                else:
                    # Train without energy monitoring
                    training_start = time.time()
                    training_result = model.train(data['X_train'], data['y_train'])
                    training_time = time.time() - training_start
                    logger.info(f"Training completed in {training_time:.2f} seconds")
                    results['training_result'] = training_result
                    results['training_time'] = training_time
                
                # Save model if requested
                if args.save_model:
                    logger.info(f"Saving {model_type} model")
                    try:
                        model.save_model()
                    except Exception as e:
                        logger.error(f"Error saving model: {e}")
            
            except ValueError as e:
                logger.error(f"Data preparation error: {e}")
                return results
            except Exception as e:
                logger.error(f"Error during training: {e}")
                return results
        
        # Evaluate model if requested
        if args.evaluate:
            if 'data' not in results:
                if not model.is_trained:
                    logger.warning("No data available for evaluation and model is not trained. Skipping evaluation.")
                    return results
                
                # Try to prepare data for evaluation only
                try:
                    logger.info("Preparing data for evaluation only")
                    if model_type == 'lstm':
                        data = data_loader.prepare_data_for_lstm(
                            data_file, 
                            sequence_length=args.sequence_length,
                            test_size=args.test_size
                        )
                    else:
                        data = data_loader.prepare_data_for_rf_xgb(
                            data_file,
                            lag_features=args.lag_features,
                            test_size=args.test_size
                        )
                    results['data'] = data
                except Exception as e:
                    logger.error(f"Error preparing data for evaluation: {e}")
                    return results
            
            logger.info(f"Evaluating {model_type} model")
            data = results['data']
            
            try:
                # Evaluate with energy monitoring if requested
                if energy_monitor and args.monitor_energy and ENERGY_MONITOR_AVAILABLE:
                    logger.info("Monitoring energy during prediction")
                    
                    @energy_monitor.energy_monitor_decorator
                    def predict():
                        return model.predict(data['X_test'])
                    
                    y_pred = predict()
                else:
                    # Predict without energy monitoring
                    prediction_start = time.time()
                    y_pred = model.predict(data['X_test'])
                    prediction_time = time.time() - prediction_start
                    logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
                    results['prediction_time'] = prediction_time
                
                # Calculate and store evaluation metrics
                metrics = model.evaluate(data['X_test'], data['y_test'])
                logger.info(f"Evaluation metrics: {metrics}")
                results['metrics'] = metrics
                results['y_pred'] = y_pred
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
    
    except ImportError as e:
        logger.error(f"Error creating or using {model_type} model: {e}")
    except Exception as e:
        logger.error(f"Error during {model_type} model training/evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    return results


def run_all_models(data_loader: DataLoader, args: argparse.Namespace, 
                   energy_monitor: Optional[EnergyMonitor] = None) -> Dict[str, Dict[str, Any]]:
    """Run all models and return their results."""
    model_results = {}
    
    # Get available models
    available_models = ModelFactory.get_available_models()
    
    for model_type, is_available in available_models.items():
        if is_available:
            logger.info(f"Running {model_type} model")
            results = train_evaluate_model(model_type, data_loader, args, energy_monitor)
            model_results[model_type] = results
        else:
            logger.warning(f"Skipping {model_type} model as it is not available")
    
    return model_results


def visualize_results(model_results: Dict[str, Dict[str, Any]], visualizer: Visualizer, data_loader: DataLoader):
    """Create visualizations for model results."""
    logger.info("Creating visualizations")
    
    # Combine metrics from all models
    metrics_dict = {}
    for model_type, results in model_results.items():
        if 'metrics' in results:
            metrics_dict[model_type] = results['metrics']
    
    # Plot performance metrics comparison
    if metrics_dict:
        logger.info("Creating model comparison plots")
        visualizer.plot_model_comparison(metrics_dict, metric_name='MAE', 
                                       save_path='model_comparison_mae')
        visualizer.plot_model_comparison(metrics_dict, metric_name='RMSE', 
                                       save_path='model_comparison_rmse')
        visualizer.plot_model_comparison(metrics_dict, metric_name='R^2', 
                                       save_path='model_comparison_r2')
        
        # Create performance metrics table
        visualizer.plot_performance_metrics_table(metrics_dict, 
                                               save_path='model_performance_metrics')
    
    # Plot feature importance for tree-based models
    for model_type in ['random_forest', 'xgboost']:
        if model_type in model_results and 'model' in model_results[model_type]:
            model = model_results[model_type]['model']
            data = model_results[model_type].get('data', {})
            feature_names = data.get('feature_names', None)
            
            if hasattr(model, 'get_feature_importance') and feature_names:
                logger.info(f"Creating feature importance plot for {model_type}")
                importance_dict = model.get_feature_importance(feature_names)
                visualizer.plot_feature_importance(importance_dict, 
                                                title=f"{model_type} Feature Importance",
                                                save_path=f"{model_type}_feature_importance")
    
    # Plot training history for LSTM
    if 'lstm' in model_results and 'training_result' in model_results['lstm']:
        logger.info("Creating LSTM training history plot")
        training_result = model_results['lstm']['training_result']
        visualizer.plot_training_history(training_result, 
                                      title='LSTM Training History',
                                      save_path='lstm_training_history')


def main():
    """Main entry point of the application."""
    try:
        # Parse command-line arguments
        args = setup_argparse()
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
        
        logger.info("Starting Battery SoH Prediction System")
        
        # List available models if requested
        if args.list_models:
            available_models = ModelFactory.get_available_models()
            print("\nAvailable Models:")
            for model_name, is_available in available_models.items():
                status = "Available" if is_available else "Not available (missing dependencies)"
                print(f"  - {model_name}: {status}")
            return
        
        # Initialize components
        data_loader = DataLoader()
        visualizer = Visualizer()
        energy_monitor = EnergyMonitor() if args.monitor_energy and ENERGY_MONITOR_AVAILABLE else None
        
        if args.monitor_energy and not ENERGY_MONITOR_AVAILABLE:
            logger.warning("Energy monitoring was requested but is not available. Continuing without energy monitoring.")
        
        # Run the selected model or all models
        model_results = {}
        try:
            if args.model_type == 'all':
                logger.info("Running all available models")
                model_results = run_all_models(data_loader, args, energy_monitor)
            else:
                logger.info(f"Running {args.model_type} model")
                results = train_evaluate_model(args.model_type, data_loader, args, energy_monitor)
                model_results = {args.model_type: results}
        except Exception as e:
            logger.error(f"Error running models: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Create visualizations if requested
        if args.visualize and model_results:
            try:
                visualize_results(model_results, visualizer, data_loader)
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")
        
        logger.info("Completed Battery SoH Prediction System")

    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main() 