"""
CNN-LSTM Model Architecture for Driver Drowsiness Detection
Implements hybrid model with CNN feature extraction and LSTM temporal learning
"""

import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except (ImportError, Exception):
    tf = None
    keras = None


def build_cnn_lstm_model(input_shape=(16, 48, 48, 1), num_classes=3):
    """
    Build CNN-LSTM hybrid model for drowsiness detection
    
    Architecture:
    - TimeDistributed CNN for spatial feature extraction
    - LSTM layers for temporal pattern learning
    - Dense classifier for 3-class output (Alert, Warning, Drowsy)
    
    Args:
        input_shape: Input tensor shape (sequence_length, height, width, channels)
        num_classes: Number of output classes (default 3)
    
    Returns:
        keras.Model: Compiled CNN-LSTM model
    """
    # Input layer: (batch, 16 frames, 48x48, 1 channel)
    input_layer = layers.Input(shape=input_shape)
    
    # --- CNN Feature Extractor (applied to each frame independently) ---
    # Conv Block 1
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    )(input_layer)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)
    
    # Conv Block 2
    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same')
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)
    
    # Conv Block 3
    x = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='relu', padding='same')
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    
    # Flatten CNN output for each timestep
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # Dense layer to reduce dimensionality
    x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    
    # --- LSTM Temporal Learner ---
    # First LSTM layer with return sequences
    x = layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    
    # Second LSTM layer (final temporal features)
    x = layers.LSTM(128, dropout=0.3, recurrent_dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    
    # --- Classifier Head ---
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer: 3 classes (Alert, Warning, Drowsy)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def create_callbacks(model_save_path='models/best_model.h5', patience=10):
    """
    Create training callbacks for model optimization
    
    Args:
        model_save_path: Path to save best model weights
        patience: Patience for early stopping
    
    Returns:
        list: List of Keras callbacks
    """
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


class DrowsinessPredictor:
    """
    Wrapper class for drowsiness prediction with pre/post-processing
    Handles model loading, inference, and result interpretation
    """
    
    def __init__(self, model_path=None, threshold_warning=0.4, threshold_drowsy=0.8):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model file (.h5)
            threshold_warning: Probability threshold for warning state
            threshold_drowsy: Probability threshold for drowsy state
        """
        self.model = None
        self.threshold_warning = threshold_warning
        self.threshold_drowsy = threshold_drowsy
        
        # State smoothing (require N consecutive frames for state change)
        self.state_history = []
        self.state_smoothing_window = 5
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load trained model from file
        
        Args:
            model_path: Path to model file (.h5)
        """
        print(f"[DrowsinessPredictor] Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        print(f"[DrowsinessPredictor] Model loaded successfully")
    
    def reset_state_history(self):
        """Clear the temporal state smoothing buffer"""
        self.state_history = []
        
    def predict(self, sequence):
        """
        Predict drowsiness state from face sequence
        
        Args:
            sequence: Numpy array of shape (16, 48, 48, 1) - 16 grayscale face frames
        
        Returns:
            dict: Prediction results with state, probabilities, and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure correct input shape
        if sequence.shape != (16, 48, 48, 1):
            raise ValueError(f"Expected shape (16, 48, 48, 1), got {sequence.shape}")
        
        # Add batch dimension
        input_batch = np.expand_dims(sequence, axis=0)
        
        # Get prediction
        predictions = self.model.predict(input_batch, verbose=0)[0]
        
        # predictions shape: (3,) -> [P(Alert), P(Warning), P(Drowsy)]
        prob_alert = predictions[0]
        prob_warning = predictions[1]
        prob_drowsy = predictions[2]
        
        # Determine state based on thresholds
        if prob_drowsy > self.threshold_drowsy:
            state = 2  # DROWSY
            state_name = "DROWSY"
            confidence = prob_drowsy
        elif prob_drowsy > self.threshold_warning:
            state = 1  # WARNING
            state_name = "WARNING"
            confidence = prob_drowsy
        else:
            state = 0  # ALERT
            state_name = "ALERT"
            confidence = prob_alert
        
        # Apply state smoothing
        self.state_history.append(state)
        if len(self.state_history) > self.state_smoothing_window:
            self.state_history.pop(0)
        
        # Use majority vote for smoothed state
        if len(self.state_history) == self.state_smoothing_window:
            smoothed_state = max(set(self.state_history), key=self.state_history.count)
        else:
            smoothed_state = state
        
        result = {
            'state': smoothed_state,
            'state_name': ["ALERT", "WARNING", "DROWSY"][smoothed_state],
            'raw_state': state,
            'raw_state_name': state_name,
            'probabilities': {
                'alert': float(prob_alert),
                'warning': float(prob_warning),
                'drowsy': float(prob_drowsy)
            },
            'confidence': float(confidence)
        }
        
        return result
    
    def reset_state(self):
        """Reset state smoothing history"""
        self.state_history = []


class OnnxDrowsinessPredictor:
    """
    High-performance ONNX Runtime predictor for CNN-LSTM drowsiness detection.
    Zero C++ DLL dependencies on Windows, 2x-3x faster CPU execution via SIMD/AVX2.
    """
    def __init__(self, model_path, threshold_warning=0.4, threshold_drowsy=0.8):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX models. Install with: pip install onnxruntime")
            
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.threshold_warning = threshold_warning
        self.threshold_drowsy = threshold_drowsy
        self.state_history = []
        self.state_smoothing_window = 5
        self.model = self.session
        print(f"[OnnxPredictor] Successfully loaded ONNX model: {model_path}")

    def predict(self, sequence):
        # sequence: (16, 48, 48, 1)
        inp = np.expand_dims(sequence, axis=0).astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: inp})[0][0]
        
        prob_alert = float(outputs[0])
        prob_warning = float(outputs[1])
        prob_drowsy = float(outputs[2])
        
        if prob_drowsy >= self.threshold_drowsy:
            raw_state = 2
        elif prob_drowsy >= self.threshold_warning or prob_warning >= 0.5:
            raw_state = 1
        else:
            raw_state = 0
            
        self.state_history.append(raw_state)
        if len(self.state_history) > self.state_smoothing_window:
            self.state_history.pop(0)
            
        counts = np.bincount(self.state_history, minlength=3)
        smoothed_state = int(np.argmax(counts))
        state_names = ["ALERT", "WARNING", "DROWSY"]
        
        return {
            'state': smoothed_state,
            'state_name': state_names[smoothed_state],
            'raw_state': raw_state,
            'raw_state_name': state_names[raw_state],
            'probabilities': {
                'alert': prob_alert,
                'warning': prob_warning,
                'drowsy': prob_drowsy
            },
            'confidence': float(outputs[smoothed_state])
        }

    def reset_state(self):
        self.state_history = []


def export_to_onnx(keras_model_path='models/drowsiness_detector.h5', onnx_output_path='models/drowsiness_detector.onnx'):
    """
    Export Keras .h5 model to ONNX runtime format
    """
    try:
        import tf2onnx
        if keras is None:
            raise ImportError("TensorFlow/Keras is required to load .h5 before exporting to ONNX.")
        model = keras.models.load_model(keras_model_path)
        spec = (tf.TensorSpec((None, 16, 48, 48, 1), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_output_path)
        print(f"[INFO] Exported ONNX model successfully to {onnx_output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to export to ONNX: {e}")
        return False


def load_pretrained_model(model_path='models/drowsiness_detector.h5'):
    """
    Load pretrained drowsiness detection model.
    Automatically detects and prioritizes ONNX Runtime if an ONNX model is provided or present.
    
    Args:
        model_path: Path to model file (.onnx or .h5)
    
    Returns:
        DrowsinessPredictor or OnnxDrowsinessPredictor: Loaded model predictor
    """
    # 1. Check if direct ONNX model specified
    if model_path.endswith('.onnx') and os.path.exists(model_path):
        return OnnxDrowsinessPredictor(model_path)
        
    # 2. Check if sibling .onnx file exists for faster, dependency-free inference
    sibling_onnx = os.path.splitext(model_path)[0] + '.onnx'
    if os.path.exists(sibling_onnx):
        try:
            return OnnxDrowsinessPredictor(sibling_onnx)
        except Exception:
            pass

    # 3. Fallback to Keras / TensorFlow if installed
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file not found at {model_path}")
        if keras is not None:
            model = build_cnn_lstm_model()
            predictor = DrowsinessPredictor()
            predictor.model = model
            return predictor
        else:
            predictor = DrowsinessPredictor()
            predictor.model = None
            return predictor
    
    if keras is None:
        raise ImportError("TensorFlow/Keras is not installed. To run with deep learning, install onnxruntime or tensorflow.")
        
    predictor = DrowsinessPredictor(model_path=model_path)
    return predictor


def get_model_summary():
    """
    Print model architecture summary for debugging
    
    Returns:
        keras.Model: Model instance
    """
    model = build_cnn_lstm_model()
    print("\n" + "="*70)
    print("CNN-LSTM Model Architecture")
    print("="*70)
    model.summary()
    print("="*70)
    print(f"Total Parameters: {model.count_params():,}")
    print("="*70 + "\n")
    return model


if __name__ == "__main__":
    # Test model architecture
    print("Building and testing CNN-LSTM model...")
    model = get_model_summary()
    
    # Test with dummy input
    print("Testing with dummy input...")
    dummy_input = np.random.rand(2, 16, 48, 48, 1).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities (sample 1): {output[0]}")
    print(f"Sum of probabilities: {output[0].sum():.6f}")
    print("\nModel test completed successfully!")
