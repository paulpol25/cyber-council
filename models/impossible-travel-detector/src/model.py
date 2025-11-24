"""
Neural Network Model for Impossible Travel Detection

This module defines the neural network architecture for binary classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np


class ImpossibleTravelDetector:
    """Neural network model for detecting impossible travel patterns."""
    
    def __init__(self, config, input_dim):
        """
        Initialize the model with configuration.
        
        Args:
            config: Configuration dictionary
            input_dim: Number of input features
        """
        self.config = config
        self.input_dim = input_dim
        self.model = None
        
    def build_model(self):
        """
        Build the neural network architecture.
        
        Returns:
            Compiled Keras model
        """
        print("Building model architecture...")
        
        # Get model configuration
        hidden_layers = self.config['model']['hidden_layers']
        dropout_rate = self.config['model']['dropout_rate']
        activation = self.config['model']['activation']
        batch_norm = self.config['model']['batch_norm']
        
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        # First hidden layer
        x = layers.Dense(
            hidden_layers[0],
            activation=activation,
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_1'
        )(inputs)
        
        if batch_norm:
            x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        x = layers.Dropout(dropout_rate, name='dropout_1')(x)
        
        # Additional hidden layers
        for i, units in enumerate(hidden_layers[1:], start=2):
            x = layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(0.001),
                name=f'dense_{i}'
            )(x)
            
            if batch_norm:
                x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
            
            x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            name='output'
        )(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='impossible_travel_detector')
        
        # Compile model
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        
        print(f"Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def get_callbacks(self, class_weights=None):
        """
        Create training callbacks.
        
        Args:
            class_weights: Dictionary of class weights for imbalanced data
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor=self.config['checkpoint']['monitor'],
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            mode=self.config['checkpoint']['mode'],
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=self.config['checkpoint']['monitor'],
            factor=self.config['training']['reduce_lr_factor'],
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=self.config['training']['min_lr'],
            mode=self.config['checkpoint']['mode'],
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        import os
        checkpoint_dir = self.config['checkpoint']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.config['checkpoint']['monitor'],
            save_best_only=self.config['checkpoint']['save_best_only'],
            mode=self.config['checkpoint']['mode'],
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # TensorBoard logging (if enabled)
        if self.config['logging'].get('tensorboard', False):
            log_dir = self.config['logging']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
        
        return callbacks
    
    def calculate_class_weights(self, y_train):
        """
        Calculate class weights for imbalanced dataset.
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Calculate class weights
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        class_weights = dict(zip(classes, weights))
        
        print(f"Class weights: {class_weights}")
        
        return class_weights
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history
        """
        print("\nStarting training...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Calculate class weights if enabled
        class_weights = None
        if self.config['training'].get('class_weight') == 'auto':
            class_weights = self.calculate_class_weights(y_train)
        
        # Get callbacks
        callbacks = self.get_callbacks(class_weights)
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=self.config['logging']['verbose']
        )
        
        print("\nTraining complete!")
        
        return history
    
    def save_model(self, filename=None):
        """
        Save the trained model.
        
        Args:
            filename: Optional custom filename
        """
        import os
        from datetime import datetime
        
        # Create save directory
        save_dir = self.config['model_save']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config['model_save']['model_name']
            filename = f"{model_name}_{timestamp}.keras"
        
        # Save path
        save_path = os.path.join(save_dir, filename)
        
        # Save model
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")
        
        return save_path
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        
    def predict(self, X, threshold=0.5):
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        probabilities = self.model.predict(X)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        return self.model.predict(X)
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Example: Create model with 20 input features
    input_dim = 20
    
    # Initialize model
    detector = ImpossibleTravelDetector(config, input_dim)
    
    # Build and display model
    detector.build_model()
    detector.summary()
