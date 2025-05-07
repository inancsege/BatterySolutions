import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import glob
from datetime import datetime
from pathlib import Path
import random
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.config.config import DATASET_DIR, PROCESSED_DATA_DIR, ORIGINAL_CAPACITY_AH


class DataLoader:
    """
    Responsible for loading and preprocessing battery dataset.
    Single Responsibility: Handles data loading and initial preprocessing.
    """
    
    def __init__(self, dataset_dir: Path = DATASET_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        """Initialize the DataLoader with dataset directory paths."""
        self.dataset_dir = dataset_dir
        self.processed_dir = processed_dir
        
        # Create processed directory if it doesn't exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a single CSV file into a pandas DataFrame."""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded file: {file_path}")
            return df
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_all_files(self, pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
        """Load all CSV files in the dataset directory matching the pattern."""
        file_pattern = os.path.join(self.dataset_dir, pattern)
        files = glob.glob(file_pattern)
        
        data_dict = {}
        for file_path in files:
            file_name = os.path.basename(file_path)
            data_dict[file_name] = self.load_single_file(file_path)
        
        return data_dict
    
    def preprocess_datetime(self, df: pd.DataFrame, time_col: str = 'record_time') -> pd.DataFrame:
        """Convert record_time column to datetime format."""
        df_copy = df.copy()
        
        # Check for different datetime formats and handle accordingly
        if time_col in df_copy.columns:
            # Try different datetime formats
            try:
                # YYYYMMDDHHMMSS format
                df_copy[time_col] = pd.to_datetime(df_copy[time_col], format='%Y%m%d%H%M%S')
            except:
                try:
                    # Standard format
                    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
                except Exception as e:
                    print(f"Error converting datetime: {e}")
        
        return df_copy
    
    def calculate_daily_metrics(self, df: pd.DataFrame, capacity_col: str = 'available_capacity (Ah)', 
                              time_col: str = 'record_time') -> pd.DataFrame:
        """Calculate daily average capacity and State of Health (SoH)."""
        # Ensure datetime format
        df = self.preprocess_datetime(df, time_col)
        
        # Add date column
        df['date'] = df[time_col].dt.date
        
        # Group by date and calculate daily averages
        daily_df = df.groupby('date').agg({
            capacity_col: ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten multi-level columns
        daily_df.columns = ['_'.join(col).strip('_') for col in daily_df.columns.values]
        
        # Calculate SoH based on original capacity
        daily_df['SoH_capacity'] = (daily_df[f'{capacity_col}_mean'] / ORIGINAL_CAPACITY_AH) * 100
        
        # Add time-based features
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['day'] = daily_df['date'].dt.day
        daily_df['dayofyear'] = daily_df['date'].dt.dayofyear
        
        # Add smoothed SoH using Savitzky-Golay filter
        daily_df['SoH_capacity_smooth'] = savgol_filter(
            daily_df['SoH_capacity'], 
            window_length=min(5, len(daily_df)), 
            polyorder=min(2, len(daily_df)-1)
        ) if len(daily_df) > 4 else daily_df['SoH_capacity']
        
        # Add first and second derivatives of SoH
        daily_df['SoH_velocity'] = daily_df['SoH_capacity'].diff().fillna(0)
        daily_df['SoH_acceleration'] = daily_df['SoH_velocity'].diff().fillna(0)
        
        # Add capacity variation features
        daily_df['capacity_variation'] = daily_df[f'{capacity_col}_std'] / daily_df[f'{capacity_col}_mean']
        daily_df['capacity_range'] = (daily_df[f'{capacity_col}_max'] - daily_df[f'{capacity_col}_min']) / daily_df[f'{capacity_col}_mean']
        
        return daily_df
    
    def create_timeseries_features(self, df: pd.DataFrame, target_col: str = 'SoH_capacity', 
                                 sequence_length: int = 7, additional_features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series features by creating sequences of data.
        Returns X (sequences) and y (targets) arrays for time series modeling.
        
        Args:
            df: DataFrame with time series data
            target_col: Column name of the target variable
            sequence_length: Length of input sequences
            additional_features: Additional features to include beyond the target
            
        Returns:
            X: numpy array of shape (samples, sequence_length, features)
            y: numpy array of target values
        """
        if additional_features is None:
            additional_features = ['SoH_velocity', 'SoH_acceleration', 'capacity_variation']
        
        # Filter additional features to only include those in the dataframe
        available_additional_features = [f for f in additional_features if f in df.columns]
        
        # Initialize lists for features and target
        X, y = [], []
        
        # Column indices for features
        feature_cols = [target_col] + available_additional_features
        num_features = len(feature_cols)
        
        # Create sequences
        for i in range(len(df) - sequence_length):
            # Get feature values for current sequence
            sequence_data = []
            for feature in feature_cols:
                sequence_data.append(df[feature].values[i:i + sequence_length])
            
            # Reformat to have shape (sequence_length, num_features)
            sequence = np.column_stack(sequence_data)
            X.append(sequence)
            
            # Target is the next value of target column
            y.append(df[target_col].values[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def time_series_augmentation(self, X: np.ndarray, y: np.ndarray, 
                               augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment time series data with synthetic samples.
        
        Args:
            X: Input sequences of shape (samples, sequence_length, features)
            y: Target values
            augmentation_factor: Number of times to augment the data
            
        Returns:
            Augmented X and y arrays
        """
        X_aug, y_aug = X.copy(), y.copy()
        
        for _ in range(augmentation_factor - 1):
            X_new, y_new = [], []
            
            for i in range(len(X)):
                sequence = X[i].copy()
                target = y[i]
                
                # Apply one of several augmentation techniques
                augmentation_technique = random.choice(['jitter', 'scale', 'shift', 'none'])
                
                if augmentation_technique == 'jitter':
                    # Add small noise
                    noise_factor = 0.02
                    noise = np.random.normal(0, noise_factor, sequence.shape)
                    sequence_aug = sequence + noise
                    
                elif augmentation_technique == 'scale':
                    # Scale by a small random factor
                    scale_factor = np.random.uniform(0.97, 1.03, (1, sequence.shape[1]))
                    sequence_aug = sequence * scale_factor
                    
                elif augmentation_technique == 'shift':
                    # Small vertical shift
                    shift_factor = np.random.uniform(-0.03, 0.03, (1, sequence.shape[1]))
                    sequence_aug = sequence + shift_factor
                    
                else:  # 'none'
                    sequence_aug = sequence
                
                X_new.append(sequence_aug)
                y_new.append(target)
            
            X_aug = np.vstack([X_aug, np.array(X_new)])
            y_aug = np.concatenate([y_aug, np.array(y_new)])
        
        return X_aug, y_aug
    
    def split_train_test(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                       random_state: int = 42, stratify_bins: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets with optional stratification for regression.
        
        Args:
            X: Feature matrix
            y: Target values
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            stratify_bins: Number of bins to use for stratification (0 for no stratification)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        # Check if we have enough data to split
        if len(X) < 2:
            raise ValueError(f"Not enough samples to split. Got {len(X)} samples, need at least 2.")
        
        # Adjust test_size if needed to ensure at least one sample in each split
        min_test_samples = 1
        min_train_samples = 1
        max_test_size = (len(X) - min_train_samples) / len(X)
        min_test_size = min_test_samples / len(X)
        
        if test_size > max_test_size:
            test_size = max_test_size
            print(f"Adjusted test_size to {test_size:.2f} to ensure at least one training sample")
        elif test_size < min_test_size:
            test_size = min_test_size
            print(f"Adjusted test_size to {test_size:.2f} to ensure at least one test sample")
        
        if stratify_bins > 0 and len(X) >= stratify_bins:
            # Create bins for stratification
            try:
                y_binned = pd.qcut(y, q=min(stratify_bins, len(X) // 2), labels=False, duplicates='drop')
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y_binned
                )
            except ValueError:
                print("Could not create stratified bins, falling back to random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Verify splits are not empty
        if len(X_train) == 0 or len(X_test) == 0:
            print("Warning: Empty split detected, using manual split")
            # Manual split to ensure non-empty sets
            split_idx = max(1, int((1 - test_size) * len(X)))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                    scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Normalize features using different scaling methods.
        
        Args:
            X_train: Training features
            X_test: Testing features
            scaler_type: Type of scaler to use ('standard', 'minmax', or 'robust')
            
        Returns:
            Scaled training features, scaled testing features, and scaler info
        """
        # Reshape data for normalization if needed
        original_shape = X_train.shape
        if len(original_shape) > 2:
            # For 3D data (samples, sequence_length, features)
            # Reshape to 2D (samples * sequence_length, features)
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        else:
            X_train_reshaped = X_train
            X_test_reshaped = X_test
        
        # Initialize appropriate scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:  # default to minmax
            scaler = MinMaxScaler()
        
        # Fit on training data and transform both training and test data
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        
        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Return scaled data and scaler for inverse transform later
        scaler_info = {'scaler': scaler, 'type': scaler_type}
        return X_train_scaled, X_test_scaled, scaler_info
    
    def prepare_data_for_lstm(self, file_path: Union[str, Path], sequence_length: int = 30, 
                           test_size: float = 0.2, augment: bool = True,
                           augmentation_factor: int = 3) -> Dict:
        """
        Prepare data specifically for LSTM model with enhanced features and augmentation.
        
        Args:
            file_path: Path to the data file
            sequence_length: Length of input sequences
            test_size: Proportion of data to use for testing
            augment: Whether to apply data augmentation
            augmentation_factor: Factor by which to augment data
            
        Returns:
            Dictionary with train/test data and scaler
        """
        # Load and preprocess data with enhanced feature engineering
        df = self.load_single_file(file_path)
        daily_df = self.calculate_daily_metrics(df)
        
        # Check if we have enough data
        if len(daily_df) <= sequence_length:
            # Adjust sequence length if not enough data
            original_sequence_length = sequence_length
            sequence_length = max(1, len(daily_df) // 2)
            print(f"Warning: Not enough data for requested sequence length {original_sequence_length}. "
                  f"Adjusted to {sequence_length}.")
            
            # If still not enough data, generate synthetic data
            if len(daily_df) <= sequence_length + 1:
                print(f"Warning: Not enough real data. Generating synthetic data.")
                # Create synthetic data by adding noise to existing data
                synthetic_size = 100  # Generate enough synthetic points
                base_capacity = daily_df['SoH_capacity'].mean() if not daily_df.empty else 80.0
                
                # Create a declining trend for capacity
                trend = np.linspace(base_capacity, base_capacity * 0.7, synthetic_size)
                
                # Add noise to the trend
                noise = np.random.normal(0, 2, synthetic_size)
                synthetic_capacity = trend + noise
                
                # Generate dates
                start_date = pd.Timestamp.now() - pd.Timedelta(days=synthetic_size)
                dates = [start_date + pd.Timedelta(days=i) for i in range(synthetic_size)]
                
                # Create synthetic DataFrame
                synthetic_df = pd.DataFrame({
                    'date': dates,
                    'SoH_capacity': synthetic_capacity,
                    'SoH_velocity': np.diff(synthetic_capacity, prepend=synthetic_capacity[0]),
                    'SoH_acceleration': np.diff(np.diff(synthetic_capacity, prepend=synthetic_capacity[0]), prepend=0),
                    'capacity_variation': np.random.uniform(0.01, 0.05, synthetic_size),
                    'capacity_range': np.random.uniform(0.05, 0.15, synthetic_size),
                    'day_of_week': [d.dayofweek for d in dates],
                    'month': [d.month for d in dates]
                })
                
                # Use synthetic data instead
                daily_df = synthetic_df
                
                # Set a reasonable sequence length
                sequence_length = min(30, len(daily_df) // 3)
        
        # Create sequences with additional features
        additional_features = [
            'SoH_velocity', 'SoH_acceleration', 
            'capacity_variation', 'capacity_range',
            'day_of_week', 'month'
        ]
        
        # Filter features to only include those available in the DataFrame
        available_features = [f for f in additional_features if f in daily_df.columns]
        
        X, y = self.create_timeseries_features(
            daily_df, 
            sequence_length=sequence_length,
            additional_features=available_features
        )
        
        # Check if we have any sequences
        if len(X) == 0:
            raise ValueError("Could not create any sequences from the data. Please check your data or decrease sequence_length.")
        
        # Split before augmentation to avoid data leakage
        try:
            X_train, X_test, y_train, y_test = self.split_train_test(
                X, y, test_size=test_size, stratify_bins=min(5, len(X) // 2)
            )
        except ValueError as e:
            print(f"Warning: {e}. Creating manual split.")
            # Manual split for extremely small datasets
            split_idx = max(1, int(0.8 * len(X)))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Augment training data if enabled
        if augment and len(X_train) > 0:
            X_train, y_train = self.time_series_augmentation(
                X_train, y_train, augmentation_factor=augmentation_factor
            )
        
        # Normalize data
        X_train_scaled, X_test_scaled, scaler_info = self.normalize_data(
            X_train, X_test, scaler_type='standard'
        )
        
        # Return prepared data
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler_info': scaler_info,
            'sequence_length': sequence_length,
            'num_features': X_train.shape[2] if X_train.size > 0 else 0,
            'feature_names': ['SoH_capacity'] + [f for f in available_features if f in daily_df.columns]
        }
    
    def prepare_data_for_rf_xgb(self, file_path: Union[str, Path], lag_features: int = 5, 
                             test_size: float = 0.2, use_advanced_features: bool = True) -> Dict:
        """
        Prepare data specifically for Random Forest and XGBoost models with advanced features.
        
        Args:
            file_path: Path to the data file
            lag_features: Number of lag features to create
            test_size: Proportion of data to use for testing
            use_advanced_features: Whether to use advanced feature engineering
            
        Returns:
            Dictionary with prepared data
        """
        # Load and preprocess data
        df = self.load_single_file(file_path)
        daily_df = self.calculate_daily_metrics(df)
        
        # Check if we have enough data
        if len(daily_df) <= lag_features:
            # Adjust lag features if not enough data
            original_lag_features = lag_features
            lag_features = max(1, len(daily_df) // 2 - 1)
            print(f"Warning: Not enough data for requested lag_features {original_lag_features}. "
                  f"Adjusted to {lag_features}.")
            
            # If still not enough data, generate synthetic data
            if len(daily_df) <= lag_features + 1:
                print(f"Warning: Not enough real data. Generating synthetic data.")
                # Create synthetic data by adding noise to existing data
                synthetic_size = 100  # Generate enough synthetic points
                base_capacity = daily_df['SoH_capacity'].mean() if not daily_df.empty else 80.0
                
                # Create a declining trend for capacity
                trend = np.linspace(base_capacity, base_capacity * 0.7, synthetic_size)
                
                # Add noise to the trend
                noise = np.random.normal(0, 2, synthetic_size)
                synthetic_capacity = trend + noise
                
                # Generate dates
                start_date = pd.Timestamp.now() - pd.Timedelta(days=synthetic_size)
                dates = [start_date + pd.Timedelta(days=i) for i in range(synthetic_size)]
                
                # Create synthetic DataFrame
                synthetic_df = pd.DataFrame({
                    'date': dates,
                    'SoH_capacity': synthetic_capacity,
                    'SoH_velocity': np.diff(synthetic_capacity, prepend=synthetic_capacity[0]),
                    'SoH_acceleration': np.diff(np.diff(synthetic_capacity, prepend=synthetic_capacity[0]), prepend=0),
                    'capacity_variation': np.random.uniform(0.01, 0.05, synthetic_size),
                    'capacity_range': np.random.uniform(0.05, 0.15, synthetic_size),
                    'day_of_week': [d.dayofweek for d in dates],
                    'month': [d.month for d in dates]
                })
                
                # Use synthetic data instead
                daily_df = synthetic_df
                
                # Set a reasonable lag feature count
                lag_features = min(5, len(daily_df) // 3 - 1)
        
        # Create base feature dataframe
        feature_df = daily_df.copy()
        
        # Add lag features
        for i in range(1, lag_features + 1):
            feature_df[f'SoH_capacity_lag{i}'] = feature_df['SoH_capacity'].shift(i)
            
            # Add velocity and acceleration lags if advanced features are enabled
            if use_advanced_features and 'SoH_velocity' in feature_df.columns:
                feature_df[f'SoH_velocity_lag{i}'] = feature_df['SoH_velocity'].shift(i)
            if use_advanced_features and 'SoH_acceleration' in feature_df.columns:
                feature_df[f'SoH_acceleration_lag{i}'] = feature_df['SoH_acceleration'].shift(i)
        
        # Add moving averages and other statistical features
        if use_advanced_features:
            windows = [min(w, len(feature_df) // 2) for w in [3, 5, 7]]
            for window in windows:
                if len(feature_df) > window:
                    # Add rolling statistics
                    feature_df[f'SoH_capacity_rolling_mean_{window}'] = feature_df['SoH_capacity'].rolling(window=window).mean()
                    feature_df[f'SoH_capacity_rolling_std_{window}'] = feature_df['SoH_capacity'].rolling(window=window).std()
                    
                    # Add rolling min/max range as a degradation signal
                    feature_df[f'SoH_capacity_rolling_range_{window}'] = (
                        feature_df['SoH_capacity'].rolling(window=window).max() - 
                        feature_df['SoH_capacity'].rolling(window=window).min()
                    )
                    
                    # Exponential weighted moving average
                    feature_df[f'SoH_capacity_ewm_{window}'] = feature_df['SoH_capacity'].ewm(span=window).mean()
            
            # Add cyclical encoding of time features
            if 'day_of_week' in feature_df.columns:
                feature_df['day_of_week_sin'] = np.sin(2 * np.pi * feature_df['day_of_week'] / 7)
                feature_df['day_of_week_cos'] = np.cos(2 * np.pi * feature_df['day_of_week'] / 7)
            
            if 'month' in feature_df.columns:
                feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
                feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
            
            # Add polynomial features of SoH
            feature_df['SoH_capacity_squared'] = feature_df['SoH_capacity'] ** 2
            feature_df['SoH_capacity_cubed'] = feature_df['SoH_capacity'] ** 3
        
        # Drop rows with NaN values (from lag feature creation)
        feature_df = feature_df.dropna()
        
        # Check if we have any data after dropping NaNs
        if len(feature_df) == 0:
            raise ValueError("No data available after creating lag features. Please check your data or decrease lag_features.")
        
        # Define features and target
        feature_cols = [col for col in feature_df.columns if col != 'SoH_capacity' and col != 'date']
        X = feature_df[feature_cols]
        y = feature_df['SoH_capacity']
        
        # Split and normalize data
        try:
            X_train, X_test, y_train, y_test = self.split_train_test(
                X.values, y.values, test_size=test_size, stratify_bins=min(5, len(X) // 2)
            )
        except ValueError as e:
            print(f"Warning: {e}. Creating manual split.")
            # Manual split for extremely small datasets
            split_idx = max(1, int(0.8 * len(X)))
            X_train, X_test = X.values[:split_idx], X.values[split_idx:]
            y_train, y_test = y.values[:split_idx], y.values[split_idx:]
        
        X_train_scaled, X_test_scaled, scaler_info = self.normalize_data(
            X_train, X_test, scaler_type='standard'
        )
        
        return {
            'X_train': X_train_scaled, 
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'scaler_info': scaler_info
        }
    
    def save_processed_data(self, data: Dict, filename: str) -> str:
        """Save processed data to processed directory."""
        import pickle
        
        file_path = os.path.join(self.processed_dir, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved processed data to {file_path}")
        return file_path
    
    def load_processed_data(self, filename: str) -> Dict:
        """Load processed data from processed directory."""
        import pickle
        
        file_path = os.path.join(self.processed_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Loaded processed data from {file_path}")
            return data
        except Exception as e:
            print(f"Error loading processed data from {file_path}: {e}")
            return {} 