"""
Training Pipeline for CNN-LSTM Drowsiness Detection Model
Handles data preparation, augmentation, training, and evaluation
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import json

from model import build_cnn_lstm_model, create_callbacks


class DrowsinessDataGenerator:
    """
    Custom data generator for drowsiness detection
    Creates sequences of face frames with temporal augmentation
    """
    
    def __init__(self, sequence_length=16, img_size=(48, 48)):
        """
        Initialize data generator
        
        Args:
            sequence_length: Number of frames per sequence
            img_size: Target image size (width, height)
        """
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Image augmentation
        self.augmenter = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def load_video_sequences(self, video_path, label, max_sequences=100):
        """
        Load sequences from a single video file
        
        Args:
            video_path: Path to video file
            label: Label for this video (0=Alert, 1=Warning, 2=Drowsy)
            max_sequences: Maximum number of sequences to extract
        
        Returns:
            tuple: (sequences, labels) as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        sequences = []
        labels = []
        
        frame_buffer = []
        frame_count = 0
        
        print(f"Loading video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.img_size)
            normalized = resized.astype(np.float32) / 255.0
            
            frame_buffer.append(normalized)
            frame_count += 1
            
            # Create sequence when buffer is full
            if len(frame_buffer) == self.sequence_length:
                sequence = np.array(frame_buffer)
                sequence = np.expand_dims(sequence, axis=-1)  # Add channel dimension
                sequences.append(sequence)
                labels.append(label)
                
                # Slide window by half sequence length for overlap
                frame_buffer = frame_buffer[self.sequence_length // 2:]
                
                if len(sequences) >= max_sequences:
                    break
        
        cap.release()
        
        print(f"Extracted {len(sequences)} sequences from {video_path}")
        
        return np.array(sequences), np.array(labels)
    
    def generate_synthetic_data(self, num_samples=1000):
        """
        Generate synthetic drowsiness data for initial training
        Creates realistic face-like patterns with varying EAR/MAR characteristics
        
        Args:
            num_samples: Number of sequences to generate
        
        Returns:
            tuple: (X, y) training data
        """
        print("Generating synthetic training data...")
        
        sequences = []
        labels = []
        
        for i in range(num_samples):
            # Randomly assign state
            state = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])  # More alert samples
            
            sequence = []
            
            for frame_idx in range(self.sequence_length):
                # Create synthetic face-like image
                img = np.random.rand(self.img_size[0], self.img_size[1]) * 0.3
                
                # Add face-like features
                center_x, center_y = self.img_size[0] // 2, self.img_size[1] // 2
                
                # Eyes (darker regions)
                eye_y = center_y - 8
                eye_openness = 1.0
                
                if state == 2:  # Drowsy - eyes gradually close
                    eye_openness = max(0.2, 1.0 - (frame_idx / self.sequence_length) * 0.8)
                elif state == 1:  # Warning - occasional blinks
                    if frame_idx % 5 == 0:
                        eye_openness = 0.3
                
                # Left eye
                cv2.ellipse(img, (center_x - 12, eye_y), (4, int(3 * eye_openness)), 
                           0, 0, 360, 0.1, -1)
                # Right eye
                cv2.ellipse(img, (center_x + 12, eye_y), (4, int(3 * eye_openness)), 
                           0, 0, 360, 0.1, -1)
                
                # Mouth
                mouth_y = center_y + 12
                mouth_openness = 1.0
                
                if state == 2:  # Drowsy - occasional yawn
                    if 8 <= frame_idx <= 12:
                        mouth_openness = 2.5
                
                cv2.ellipse(img, (center_x, mouth_y), (6, int(2 * mouth_openness)), 
                           0, 0, 360, 0.1, -1)
                
                # Add noise
                noise = np.random.rand(self.img_size[0], self.img_size[1]) * 0.1
                img = np.clip(img + noise, 0, 1)
                
                # Apply Gaussian blur for smoothness
                img = cv2.GaussianBlur(img, (5, 5), 0)
                
                sequence.append(img)
            
            sequence = np.array(sequence)
            sequence = np.expand_dims(sequence, axis=-1)
            sequences.append(sequence)
            labels.append(state)
        
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"Generated {len(X)} synthetic sequences")
        print(f"  Alert: {np.sum(y == 0)}")
        print(f"  Warning: {np.sum(y == 1)}")
        print(f"  Drowsy: {np.sum(y == 2)}")
        
        return X, y


def train_model(X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=32, 
                model_save_path='models/drowsiness_detector.h5'):
    """
    Train CNN-LSTM drowsiness detection model
    
    Args:
        X_train: Training sequences (N, 16, 48, 48, 1)
        y_train: Training labels (N,)
        X_val: Validation sequences
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        model_save_path: Path to save trained model
    
    Returns:
        tuple: (model, history)
    """
    print("\n" + "="*70)
    print("Training CNN-LSTM Drowsiness Detection Model")
    print("="*70)
    
    # Convert labels to categorical
    num_classes = 3
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    
    print(f"\nTraining set: {X_train.shape[0]} sequences")
    print(f"Validation set: {X_val.shape[0]} sequences")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Classes: {num_classes} (Alert, Warning, Drowsy)")
    
    # Build model
    print("\nBuilding model...")
    model = build_cnn_lstm_model(input_shape=(16, 48, 48, 1), num_classes=num_classes)
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(model_save_path=model_save_path, patience=15)
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print("="*70 + "\n")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Test labels
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*70)
    print("Evaluating Model Performance")
    print("="*70)
    
    # Convert labels to categorical
    y_test_cat = to_categorical(y_test, 3)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    
    # Per-class accuracy
    class_names = ['Alert', 'Warning', 'Drowsy']
    class_accuracies = {}
    
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")
    print("Per-Class Performance:")
    print("-" * 50)
    
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            class_accuracies[class_name] = class_acc
            print(f"{class_name:10s}: {class_acc * 100:6.2f}% ({np.sum(mask):4d} samples)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print("-" * 50)
    print("         " + "  ".join(f"{name:8s}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:8s} " + "  ".join(f"{val:8d}" for val in row))
    
    print("="*70 + "\n")
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist()
    }


def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot training history curves
    
    Args:
        history: Keras training history object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("Driver Drowsiness Detection - Training Pipeline")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize data generator
    data_gen = DrowsinessDataGenerator(sequence_length=16, img_size=(48, 48))
    
    # Generate synthetic data for demonstration
    # In production, replace this with real video data loading
    print("Step 1: Generating training data...")
    X, y = data_gen.generate_synthetic_data(num_samples=2000)
    
    # Shuffle data
    X, y = shuffle(X, y, random_state=42)
    
    # Split into train/val/test
    print("\nStep 2: Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\nStep 3: Training model...")
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=50,  # Reduced for synthetic data
        batch_size=32,
        model_save_path='models/drowsiness_detector.h5'
    )
    
    # Plot training history
    print("\nStep 4: Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\nStep 5: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics
    metrics_path = 'models/evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    print("\n" + "="*70)
    print("Training pipeline completed successfully!")
    print("="*70)
    print(f"\nTrained model saved to: models/drowsiness_detector.h5")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}%")
    print("\nYou can now run the real-time detection system with:")
    print("  python main.py --model models/drowsiness_detector.h5")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
