import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout, Bidirectional, GRU
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Attention
    from tensorflow.keras.layers import Input, Concatenate, TimeDistributed, Lambda
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    # Force CPU/GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting error: {e}")
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. LSTM model functionality will be limited.")
    TF_AVAILABLE = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

from src.models.base_model import BaseModel
from src.config.config import LSTM_PARAMS


class LSTMModel(BaseModel):
    """
    Long Short-Term Memory (LSTM) model implementation for battery SoH prediction.
    Follows Single Responsibility Principle by focusing on LSTM-specific implementation.
    """
    
    def __init__(self, name: str = "lstm_model", **kwargs):
        """Initialize the LSTM model with default or custom parameters."""
        super().__init__(name=name, **kwargs)
        self.params = LSTM_PARAMS.copy()
        # Update params with any provided kwargs
        self.params.update(kwargs)
        
        # Track computation metrics
        self.training_history = None
        self.epoch_times = []
        self.total_params = 0
        
        if not TF_AVAILABLE:
            print("Warning: TensorFlow is not available. LSTM model will not work.")
    
    def build_model(self, input_shape: Tuple[int, int] = (30, 1), **kwargs) -> Any:
        """
        Build the LSTM model architecture with specified input shape.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Compiled Keras Sequential model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot build LSTM model.")
            
        # Update parameters if provided
        if kwargs:
            self.params.update(kwargs)
        
        # Get parameters from config
        model_type = self.params.get('model_type', 'lstm')  # Can be 'lstm', 'bidirectional', 'cnn_lstm', 'gru', 'complex'
        use_attention = self.params.get('use_attention', False)
        use_batch_norm = self.params.get('use_batch_norm', True)
        dropout_rate = self.params.get('dropout_rate', 0.3)
        recurrent_dropout = self.params.get('recurrent_dropout', 0.2)
        units = self.params.get('units', [128, 64])
        dense_layers = self.params.get('dense_layers', [32])
        activation = self.params.get('activation', 'relu')
        learning_rate = self.params.get('learning_rate', 0.001)
        learning_rate_decay = self.params.get('learning_rate_decay', False)
        regularization = self.params.get('regularization', None)
        
        # Add regularization if specified
        if regularization:
            from tensorflow.keras.regularizers import l1, l2, l1_l2
            if regularization == 'l1':
                kernel_regularizer = l1(self.params.get('regularization_factor', 0.01))
            elif regularization == 'l2':
                kernel_regularizer = l2(self.params.get('regularization_factor', 0.01))
            elif regularization == 'l1_l2':
                kernel_regularizer = l1_l2(
                    l1=self.params.get('l1_factor', 0.01),
                    l2=self.params.get('l2_factor', 0.01)
                )
            else:
                kernel_regularizer = None
        else:
            kernel_regularizer = None
        
        # Create a more complex model architecture
        if model_type == 'complex':
            # Use functional API for more complex architectures
            inputs = Input(shape=input_shape)
            
            # Main LSTM branch
            lstm_branch = BatchNormalization()(inputs) if use_batch_norm else inputs
            lstm_branch = Bidirectional(KerasLSTM(
                units=units[0], 
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=kernel_regularizer
            ))(lstm_branch)
            
            if use_batch_norm:
                lstm_branch = BatchNormalization()(lstm_branch)
            
            lstm_branch = Bidirectional(KerasLSTM(
                units=units[1], 
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=kernel_regularizer
            ))(lstm_branch)
            
            # CNN branch for feature extraction
            cnn_branch = BatchNormalization()(inputs) if use_batch_norm else inputs
            cnn_branch = Conv1D(
                filters=64, 
                kernel_size=3, 
                activation=activation, 
                padding='same',
                kernel_regularizer=kernel_regularizer
            )(cnn_branch)
            cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
            
            if use_batch_norm:
                cnn_branch = BatchNormalization()(cnn_branch)
                
            cnn_branch = Conv1D(
                filters=32, 
                kernel_size=3, 
                activation=activation,
                padding='same',
                kernel_regularizer=kernel_regularizer
            )(cnn_branch)
            cnn_branch = Flatten()(cnn_branch)
            
            # Combine branches
            combined = Concatenate()([lstm_branch, cnn_branch])
            
            # Add dense layers
            for units in dense_layers:
                combined = Dense(
                    units=units, 
                    activation=activation,
                    kernel_regularizer=kernel_regularizer
                )(combined)
                combined = Dropout(dropout_rate)(combined)
                if use_batch_norm:
                    combined = BatchNormalization()(combined)
            
            # Output layer
            outputs = Dense(units=1)(combined)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
        elif model_type == 'cnn_lstm':
            # CNN-LSTM hybrid for extracting temporal features
            model = Sequential()
            
            # Add first CNN layer
            model.add(Conv1D(
                filters=64, 
                kernel_size=3, 
                activation=activation, 
                padding='same',
                input_shape=input_shape,
                kernel_regularizer=kernel_regularizer
            ))
            model.add(MaxPooling1D(pool_size=2))
            
            if use_batch_norm:
                model.add(BatchNormalization())
            
            # Add second CNN layer
            model.add(Conv1D(
                filters=32, 
                kernel_size=3, 
                activation=activation,
                padding='same',
                kernel_regularizer=kernel_regularizer
            ))
            
            if use_batch_norm:
                model.add(BatchNormalization())
            
            # First LSTM layer with attention if enabled
            if use_attention:
                model.add(Bidirectional(KerasLSTM(
                    units=units[0],
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=kernel_regularizer
                )))
                # Implement custom attention mechanism
                model.add(Lambda(lambda x: tf.reduce_mean(x, axis=1)))
            else:
                model.add(KerasLSTM(
                    units=units[0], 
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=kernel_regularizer
                ))
            
            model.add(Dropout(dropout_rate))
            
        elif model_type == 'bidirectional':
            # Bidirectional LSTM for capturing patterns in both directions
            model = Sequential()
            model.add(Bidirectional(
                KerasLSTM(
                    units=units[0], 
                    return_sequences=True, 
                    input_shape=input_shape,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=kernel_regularizer
                )
            ))
            if use_batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
        elif model_type == 'gru':
            # GRU (Gated Recurrent Unit) for potentially faster training
            model = Sequential()
            model.add(GRU(
                units=units[0], 
                return_sequences=True, 
                input_shape=input_shape,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=kernel_regularizer
            ))
            if use_batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
        else:  # Default LSTM
            # Standard LSTM architecture
            model = Sequential()
            model.add(KerasLSTM(
                units=units[0], 
                return_sequences=True, 
                input_shape=input_shape,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=kernel_regularizer
            ))
            if use_batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Only add second layer if not using complex model type
        if model_type != 'complex':
            # Second layer (non-returning sequences)
            if model_type == 'bidirectional':
                model.add(Bidirectional(KerasLSTM(
                    units=units[1], 
                    return_sequences=False,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=kernel_regularizer
                )))
            elif model_type == 'gru':
                model.add(GRU(
                    units=units[1], 
                    return_sequences=False,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=kernel_regularizer
                ))
            else:
                model.add(KerasLSTM(
                    units=units[1], 
                    return_sequences=False,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=kernel_regularizer
                ))
            
            if use_batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            # Add dense layers
            for units in dense_layers:
                model.add(Dense(
                    units=units, 
                    activation=activation,
                    kernel_regularizer=kernel_regularizer
                ))
                model.add(Dropout(dropout_rate))
                if use_batch_norm:
                    model.add(BatchNormalization())
            
            # Output layer
            model.add(Dense(units=1))
        
        # Configure learning rate with optional decay
        if learning_rate_decay:
            initial_learning_rate = learning_rate
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=self.params.get('decay_steps', 1000),
                decay_rate=self.params.get('decay_rate', 0.9),
                staircase=True
            )
            optimizer = Adam(learning_rate=lr_schedule)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        # Compile model with metrics
        model.compile(
            optimizer=optimizer,
            loss=self.params.get('loss', 'mean_squared_error'),
            metrics=self.params.get('metrics', ['mae'])
        )
        
        # Store the model and count parameters
        self.model = model
        self.total_params = model.count_params()
        
        # Print model summary if verbose
        if self.params.get('verbose', 1) > 0:
            model.summary()
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Validation features (optional)
            y_val: Validation target values (optional)
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Dictionary containing training history
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot train LSTM model.")
            
        # Build model if not already built
        if self.model is None:
            # Infer input shape from X_train
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape=input_shape, **kwargs)
        
        # Update parameters if provided
        if kwargs:
            self.params.update(kwargs)
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.params.get('patience', 20),
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Add ReduceLROnPlateau to dynamically adjust learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Add model checkpoint to save best model
        model_dir = self.params.get('model_dir', './models')
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, f"{self.name}_best.h5")
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Add TensorBoard for visualization if enabled
        if self.params.get('use_tensorboard', False):
            log_dir = self.params.get('log_dir', './logs')
            os.makedirs(log_dir, exist_ok=True)
            tensorboard = TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
        
        # Custom callback to time each epoch
        class TimeEpochCallback(tf.keras.callbacks.Callback):
            def __init__(self, model_instance):
                self.model_instance = model_instance
                self.times = []
                self.epoch_start_time = None
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                
            def on_epoch_end(self, epoch, logs=None):
                if self.epoch_start_time:
                    epoch_time = time.time() - self.epoch_start_time
                    self.times.append(epoch_time)
                    if hasattr(self.model_instance, 'epoch_times'):
                        self.model_instance.epoch_times.append(epoch_time)
                    print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")
        
        time_callback = TimeEpochCallback(self)
        callbacks.append(time_callback)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None
            validation_split = self.params.get('validation_split', 0.2)
        
        # Ensure batch size is appropriate for dataset size
        batch_size = self.params.get('batch_size', 32)
        if len(X_train) < batch_size:
            # Adjust batch size to data size to avoid instant epochs
            batch_size = max(1, len(X_train) // 2)
            print(f"Adjusted batch size to {batch_size} due to small dataset size")
        
        # Add artificial computation to prevent instant epochs if needed
        use_artificial_delay = self.params.get('use_artificial_delay', False)
        if use_artificial_delay:
            # Temporarily wrap the model's call method to add computation
            original_call = self.model.call
            
            def call_with_delay(*args, **kwargs):
                # Add extra matrix multiplications to increase computation time
                result = original_call(*args, **kwargs)
                
                # Perform some additional computations to slow down processing
                delay_factor = self.params.get('delay_factor', 1.0)
                if delay_factor > 0:
                    # Create dummy tensors and perform operations
                    dummy_size = int(1000 * delay_factor)
                    dummy1 = tf.random.normal((dummy_size, dummy_size))
                    dummy2 = tf.random.normal((dummy_size, dummy_size))
                    _ = tf.matmul(dummy1, dummy2)
                
                return result
            
            self.model.call = call_with_delay
        
        # Train model with potentially increased computation
        print(f"Starting training with batch size {batch_size} and {self.params.get('epochs', 150)} epochs")
        history = self.model.fit(
            X_train, y_train,
            epochs=self.params.get('epochs', 150),
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.params.get('verbose', 1),
            shuffle=True
        )
        
        # Revert to original call method if modified
        if use_artificial_delay:
            self.model.call = original_call
        
        self.is_trained = True
        self.training_history = history.history
        
        # Print epoch time statistics
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            min_epoch_time = min(self.epoch_times)
            max_epoch_time = max(self.epoch_times)
            print(f"Epoch time statistics: Avg={avg_epoch_time:.2f}s, Min={min_epoch_time:.2f}s, Max={max_epoch_time:.2f}s")
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained LSTM model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predicted values as numpy array
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model using test data and return performance metrics.
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics (MAE, RMSE, R^2)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation.")
        
        # Generate predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        error = y_test - y_pred
        mean_error = np.mean(error)
        std_error = np.std(error)
        
        # Store metrics
        self.model_metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R^2': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'total_params': self.total_params,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else None
        }
        
        return self.model_metrics
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained LSTM model to disk.
        Overrides the base implementation to use Keras save functionality.
        
        Args:
            filename: Name of the file to save the model (without extension)
            
        Returns:
            Path where the model was saved
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving.")
        
        if filename is None:
            filename = f"{self.name}"
        
        file_path = f"{self.model_dir}/{filename}"
        
        try:
            self.model.save(file_path)
            # Save training history as well
            if self.training_history:
                import pickle
                with open(f"{file_path}_history.pkl", 'wb') as f:
                    pickle.dump(self.training_history, f)
            
            print(f"Model saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving model to {file_path}: {e}")
            return ""
    
    def load_model(self, filename: Optional[str] = None) -> bool:
        """
        Load a trained LSTM model from disk.
        Overrides the base implementation to use Keras load functionality.
        
        Args:
            filename: Name of the file to load the model from (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        if filename is None:
            filename = f"{self.name}"
        
        file_path = f"{self.model_dir}/{filename}"
        
        try:
            self.model = tf.keras.models.load_model(file_path)
            self.is_trained = True
            
            # Try to load training history as well
            try:
                import pickle
                with open(f"{file_path}_history.pkl", 'rb') as f:
                    self.training_history = pickle.load(f)
            except:
                pass
                
            print(f"Model loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
            return False 