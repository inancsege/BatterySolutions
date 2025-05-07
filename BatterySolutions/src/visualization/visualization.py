import os
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib or Seaborn not available. Visualization functionality will be limited.")
    MATPLOTLIB_AVAILABLE = False

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from src.config.config import REPORTS_DIR


class Visualizer:
    """
    Class responsible for creating and saving visualizations.
    Follows Single Responsibility Principle by focusing only on visualization.
    """
    
    def __init__(self, reports_dir: Path = REPORTS_DIR):
        """Initialize the Visualizer with the reports directory."""
        self.reports_dir = reports_dir
        
        # Create reports directory if it doesn't exist
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
        
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        if not self.matplotlib_available:
            print("Warning: Visualizations will not be available due to missing dependencies.")
    
    def plot_battery_soh_over_time(self, daily_df: pd.DataFrame, date_col: str = 'date', 
                                  soh_col: str = 'SoH_capacity',
                                  save_path: Optional[str] = None) -> None:
        """
        Plot battery SoH degradation over time.
        
        Args:
            daily_df: DataFrame with date and SoH capacity columns
            date_col: Name of the date column
            soh_col: Name of the SoH column
            save_path: Path to save the figure (if None, not saved)
        """
        if not self.matplotlib_available:
            print(f"Cannot create battery SoH plot: matplotlib not available")
            return
            
        # Check if DataFrame has the necessary columns
        if date_col not in daily_df.columns or soh_col not in daily_df.columns:
            print(f"DataFrame does not contain required columns: {date_col} and/or {soh_col}")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Plot SoH over time
        plt.plot(daily_df[date_col], daily_df[soh_col], 'o-', markersize=4, alpha=0.7)
        
        plt.title('Battery State of Health (SoH) Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('SoH (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure if path provided
        if save_path:
            full_path = os.path.join(self.reports_dir, f"{save_path}.png")
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {full_path}")
        
        plt.close()
    
    def plot_model_comparison(self, metrics_dict: Dict[str, Dict[str, float]], 
                            metric_name: str = 'RMSE', save_path: Optional[str] = None) -> None:
        """
        Create a bar plot comparing different models based on a specific metric.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metrics dictionaries as values
            metric_name: The metric to use for comparison ('RMSE', 'MAE', 'R^2', etc.)
            save_path: Optional path to save the figure
        """
        if not self.matplotlib_available:
            print(f"Cannot create model comparison plot: matplotlib not available")
            return
            
        # Extract values for the specified metric
        models = []
        values = []
        
        for model_name, metrics in metrics_dict.items():
            if metric_name in metrics:
                models.append(model_name)
                values.append(metrics[metric_name])
        
        if not models:
            print(f"No models have the metric '{metric_name}'")
            return
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, color=['skyblue', 'lightgreen', 'salmon'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.title(f'Model Comparison - {metric_name}')
        plt.ylabel(metric_name)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        # Save if path is provided
        if save_path:
            full_path = os.path.join(self.reports_dir, f"{save_path}.png")
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {full_path}")
        
        plt.close()
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                               title: str = 'Feature Importance', 
                               save_path: Optional[str] = None) -> None:
        """
        Create a horizontal bar plot showing feature importance.
        
        Args:
            importance_dict: Dictionary with feature names as keys and importance values as values
            title: Plot title
            save_path: Path to save the figure (if None, not saved)
        """
        if not self.matplotlib_available:
            print(f"Cannot create feature importance plot: matplotlib not available")
            return
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_features]
        importances = [x[1] for x in sorted_features]
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, max(6, len(features) * 0.4)))
        plt.barh(features, importances, color='skyblue')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.tight_layout()
        
        # Save the figure if path provided
        if save_path:
            full_path = os.path.join(self.reports_dir, f"{save_path}.png")
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {full_path}")
        
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            title: str = 'Training History', 
                            save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss over epochs.
        
        Args:
            history: Dictionary with lists of training and validation metrics
            title: Plot title
            save_path: Path to save the figure (if None, not saved)
        """
        if not self.matplotlib_available:
            print(f"Cannot create training history plot: matplotlib not available")
            return
            
        # Check if history has the necessary keys
        if 'loss' not in history:
            print("Training history does not contain 'loss' key")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.plot(history['loss'], label='Training Loss', color='blue')
        
        # Plot validation loss if available
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', color='orange')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure if path provided
        if save_path:
            full_path = os.path.join(self.reports_dir, f"{save_path}.png")
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {full_path}")
        
        plt.close()
    
    def plot_performance_metrics_table(self, metrics_dict: Dict[str, Dict[str, float]], 
                                     save_path: Optional[str] = None) -> None:
        """
        Create a table visualization of model performance metrics.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metrics dictionaries as values
            save_path: Path to save the figure (if None, not saved)
        """
        if not self.matplotlib_available:
            print(f"Cannot create performance metrics table: matplotlib not available")
            return
            
        # Extract model names and convert to DataFrame
        models = list(metrics_dict.keys())
        metrics_df = pd.DataFrame(index=models)
        
        # Get all unique metrics
        all_metrics = set()
        for metrics in metrics_dict.values():
            all_metrics.update(metrics.keys())
        
        # Fill DataFrame with metrics
        for metric in all_metrics:
            metrics_df[metric] = [metrics.get(metric, float('nan')) for metrics in metrics_dict.values()]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, len(models) * 0.5 + 2))
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=metrics_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)).values,
            rowLabels=metrics_df.index,
            colLabels=metrics_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Model Performance Metrics', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save the figure if path provided
        if save_path:
            full_path = os.path.join(self.reports_dir, f"{save_path}.png")
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {full_path}")
        
        plt.close()
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 title: str = 'Predictions vs Actual',
                                 save_path: Optional[str] = None) -> None:
        """
        Create a scatter plot of predicted vs actual values.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            title: Plot title
            save_path: Path to save the figure (if None, not saved)
        """
        if not self.matplotlib_available:
            print(f"Cannot create predictions vs actual plot: matplotlib not available")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Plot scatter of actual vs predicted
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Plot perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure if path provided
        if save_path:
            full_path = os.path.join(self.reports_dir, f"{save_path}.png")
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {full_path}")
        
        plt.close()
    
    def close_all_plots(self):
        """Close all open matplotlib plots."""
        if self.matplotlib_available:
            plt.close('all')
        else:
            print("Matplotlib is not available, no plots to close.") 