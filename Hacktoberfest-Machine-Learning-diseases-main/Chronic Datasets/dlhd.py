"""
Deep Learning Model for Chronic Disease Prediction
Supports multiple disease datasets with neural networks
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class ChronicDiseasePredictor:
    """
    Deep Learning model for chronic disease prediction
    """
    
    def __init__(self, task_type='binary', random_state=42):
        """
        Initialize the predictor
        
        Args:
            task_type: 'binary' or 'multiclass'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def build_model(self, input_dim, num_classes=1):
        """
        Build deep neural network architecture
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (1 for binary, n for multiclass)
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # First hidden layer
            layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Fourth hidden layer
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(
                num_classes if self.task_type == 'multiclass' else 1,
                activation='softmax' if self.task_type == 'multiclass' else 'sigmoid'
            )
        ])
        
        # Compile model
        if self.task_type == 'binary':
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        self.model = model
        return model
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess the data
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            fit: Whether to fit the scaler (True for training data)
        """
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            if y is not None and self.task_type == 'multiclass':
                y = self.label_encoder.fit_transform(y)
        else:
            X_scaled = self.scaler.transform(X)
            if y is not None and self.task_type == 'multiclass':
                y = self.label_encoder.transform(y)
        
        return X_scaled, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the deep learning model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
        """
        # Preprocess data
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit=True)
        
        if X_val is not None:
            X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val, fit=False)
            validation_data = (X_val_scaled, y_val_encoded)
        else:
            validation_data = None
        
        # Build model if not already built
        if self.model is None:
            num_classes = len(np.unique(y_train)) if self.task_type == 'multiclass' else 1
            self.build_model(X_train_scaled.shape[1], num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train_encoded,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        X_test_scaled, y_test_encoded = self.preprocess_data(X_test, y_test, fit=False)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_scaled)
        
        if self.task_type == 'binary':
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            auc = roc_auc_score(y_test_encoded, y_pred_proba)
            print(f"AUC Score: {auc:.4f}")
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        return y_pred, cm
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, classes=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            classes: Class labels
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
        """
        X_scaled, _ = self.preprocess_data(X, fit=False)
        predictions = self.model.predict(X_scaled)
        
        if self.task_type == 'binary':
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def save_model(self, filepath='disease_prediction_model.h5'):
        """
        Save the trained model
        """
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='disease_prediction_model.h5'):
        """
        Load a trained model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# Example usage for different datasets
if __name__ == "__main__":
    
    # Example 1: Cardiovascular Disease (Binary Classification)
    print("=" * 50)
    print("Example: Cardiovascular Disease Prediction")
    print("=" * 50)
    
    # Load your dataset
    df = pd.read_csv('Cardiovascular_Diseases_Risk_Prediction_Dataset.csv')
    X = df.drop('target_column', axis=1)
    y = df['target_column']
    
    # # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # # Initialize and train model
    predictor = ChronicDiseasePredictor(task_type='binary')
    predictor.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # # Evaluate
    y_pred, cm = predictor.evaluate(X_test, y_test)
    
    # # Plot results
    predictor.plot_training_history()
    predictor.plot_confusion_matrix(cm, classes=['No Disease', 'Disease'])
    
    # # Save model
    predictor.save_model('cardiovascular_model.h5')
    
    print("\nReplace the commented code with your actual dataset loading and training.")
    print("\nModel Architecture:")
    print("- Input Layer")
    print("- Dense(256) + BatchNorm + ReLU + Dropout(0.3)")
    print("- Dense(128) + BatchNorm + ReLU + Dropout(0.3)")
    print("- Dense(64) + BatchNorm + ReLU + Dropout(0.2)")
    print("- Dense(32) + BatchNorm + ReLU + Dropout(0.2)")
    print("- Output Layer (Sigmoid for binary, Softmax for multiclass)")