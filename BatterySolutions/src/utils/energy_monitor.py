import time
import os
import psutil
from typing import Dict, Any, Optional, Callable
import functools
import numpy as np
try:
    import pyRAPL
    PYRAPL_AVAILABLE = True
except ImportError:
    PYRAPL_AVAILABLE = False

try:
    from pyJoules.energy_meter import EnergyContext
    from pyJoules.handler.pandas_handler import PandasHandler
    PYJOULES_AVAILABLE = True
except ImportError:
    PYJOULES_AVAILABLE = False


class EnergyMonitor:
    """
    Class for monitoring energy consumption during model training and inference.
    """
    
    def __init__(self):
        """Initialize energy monitoring capabilities."""
        self.pyrapl_available = PYRAPL_AVAILABLE
        self.pyjoules_available = PYJOULES_AVAILABLE
        
        if self.pyrapl_available:
            pyRAPL.setup()
        
    def measure_energy(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure energy consumption of a function execution.
        Uses either pyRAPL or pyJoules if available, otherwise only time and CPU usage.
        
        Args:
            func: Function to measure
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary with execution metrics including energy consumption if available
        """
        result = None
        metrics = {}
        
        # Measure basic metrics (time and CPU)
        start_time = time.time()
        start_cpu_percent = psutil.cpu_percent()
        
        # Use pyRAPL for energy measurement if available
        if self.pyrapl_available:
            meter = pyRAPL.Measurement('func_energy')
            meter.begin()
            result = func(*args, **kwargs)
            meter.end()
            
            # Extract energy metrics
            metrics['energy_pkg'] = meter.result.pkg[0]  # Package energy (CPU + integrated GPU)
            metrics['energy_dram'] = meter.result.dram[0]  # DRAM energy
            metrics['energy_total'] = metrics['energy_pkg'] + metrics['energy_dram']
            
        # Use pyJoules for energy measurement if available and pyRAPL is not
        elif self.pyjoules_available:
            handler = PandasHandler()
            with EnergyContext(handler=handler) as ctx:
                result = func(*args, **kwargs)
            
            # Extract metrics from the pandas DataFrame
            energy_df = handler.get_dataframe()
            if not energy_df.empty:
                # Get the total energy consumption
                metrics['energy_total'] = energy_df['package_0'].sum()
        
        # If no energy measurement is available, just run the function
        else:
            result = func(*args, **kwargs)
        
        end_time = time.time()
        end_cpu_percent = psutil.cpu_percent()
        
        # Calculate basic metrics
        metrics['execution_time'] = end_time - start_time
        metrics['avg_cpu_percent'] = (start_cpu_percent + end_cpu_percent) / 2
        metrics['memory_usage_mb'] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        return {'result': result, 'metrics': metrics}
    
    def energy_monitor_decorator(self, func: Callable) -> Callable:
        """
        Decorator for measuring energy consumption of a function.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function that measures energy
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result_dict = self.measure_energy(func, *args, **kwargs)
            print(f"Energy metrics for {func.__name__}:")
            for k, v in result_dict['metrics'].items():
                print(f"  {k}: {v}")
            return result_dict['result']
        return wrapper
    
    def compare_model_efficiency(self, models: Dict[str, Callable], X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Compare energy efficiency of multiple models during inference.
        
        Args:
            models: Dictionary mapping model names to prediction functions
            X: Input data for prediction
            
        Returns:
            Dictionary with energy metrics for each model
        """
        results = {}
        
        for model_name, predict_func in models.items():
            print(f"Measuring energy for {model_name}")
            # Run multiple predictions to get a better average
            energy_results = []
            for _ in range(5):  # Run 5 times for better statistics
                result = self.measure_energy(predict_func, X)
                energy_results.append(result['metrics'])
            
            # Calculate average metrics
            avg_metrics = {}
            for metric in energy_results[0].keys():
                avg_metrics[metric] = sum(result[metric] for result in energy_results) / len(energy_results)
            
            results[model_name] = avg_metrics
        
        return results 