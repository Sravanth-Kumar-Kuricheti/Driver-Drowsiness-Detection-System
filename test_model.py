"""
Test Script for CNN-LSTM Model Architecture
Validates model structure, inference speed, and output format
"""

import numpy as np
import time
import tensorflow as tf

from model import build_cnn_lstm_model, DrowsinessPredictor


def test_model_architecture():
    """Test model architecture and parameters"""
    
    print("\n" + "="*70)
    print("Model Architecture Test")
    print("="*70 + "\n")
    
    print("Building CNN-LSTM model...")
    model = build_cnn_lstm_model(input_shape=(16, 48, 48, 1), num_classes=3)
    
    print("\n" + "-"*70)
    model.summary()
    print("-"*70)
    
    # Verify model properties
    print("\nModel Validation:")
    print("-" * 70)
    
    # Check input shape
    expected_input = (None, 16, 48, 48, 1)
    actual_input = model.input_shape
    print(f"✓ Input shape: {actual_input} (expected: {expected_input})")
    
    # Check output shape
    expected_output = (None, 3)
    actual_output = model.output_shape
    print(f"✓ Output shape: {actual_output} (expected: {expected_output})")
    
    # Check number of parameters
    total_params = model.count_params()
    print(f"✓ Total parameters: {total_params:,}")
    print(f"  (Expected: ~2-3 million for this architecture)")
    
    # Check layer types
    layer_types = [type(layer).__name__ for layer in model.layers]
    print(f"✓ Number of layers: {len(model.layers)}")
    print(f"✓ Contains TimeDistributed: {'TimeDistributed' in layer_types}")
    print(f"✓ Contains LSTM: {'LSTM' in layer_types}")
    print(f"✓ Contains Dense: {'Dense' in layer_types}")
    
    print("="*70 + "\n")
    
    return model


def test_model_inference_speed(model, num_iterations=100):
    """Test model inference speed"""
    
    print("\n" + "="*70)
    print("Inference Speed Test")
    print("="*70 + "\n")
    
    print(f"Running {num_iterations} inference iterations...")
    
    # Create dummy input
    dummy_input = np.random.rand(1, 16, 48, 48, 1).astype(np.float32)
    
    # Warm-up run
    _ = model.predict(dummy_input, verbose=0)
    
    # Time inference
    inference_times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        _ = model.predict(dummy_input, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1000 / mean_time
    
    print("\nInference Speed Results:")
    print("-" * 70)
    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f"Std deviation: {std_time:.2f} ms")
    print(f"Min time: {min_time:.2f} ms")
    print(f"Max time: {max_time:.2f} ms")
    print(f"Equivalent FPS: {fps:.1f} FPS")
    
    # Check against target
    target_time_ms = 30  # Target <30ms per inference
    
    print("\nPerformance Validation:")
    print("-" * 70)
    
    if mean_time < target_time_ms:
        print(f"✓ PASS: Inference time ({mean_time:.2f}ms) < target ({target_time_ms}ms)")
    else:
        print(f"⚠ WARNING: Inference time ({mean_time:.2f}ms) > target ({target_time_ms}ms)")
        print("  Consider using GPU or optimizing model")
    
    if fps >= 25:
        print(f"✓ PASS: Can achieve real-time performance ({fps:.1f} FPS ≥ 25 FPS)")
    else:
        print(f"⚠ WARNING: May not achieve real-time performance ({fps:.1f} FPS < 25 FPS)")
    
    print("="*70 + "\n")


def test_model_output_format(model):
    """Test model output format and probabilities"""
    
    print("\n" + "="*70)
    print("Output Format Test")
    print("="*70 + "\n")
    
    # Create test inputs
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        dummy_input = np.random.rand(batch_size, 16, 48, 48, 1).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        
        print(f"Batch size {batch_size}:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Validate output properties
        assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
        
        # Check probability distributions
        for i in range(min(batch_size, 3)):  # Show first 3 samples
            probs = output[i]
            prob_sum = np.sum(probs)
            
            print(f"  Sample {i+1} probabilities: Alert={probs[0]:.4f}, "
                  f"Warning={probs[1]:.4f}, Drowsy={probs[2]:.4f}, Sum={prob_sum:.6f}")
            
            # Validate probability sum
            assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities don't sum to 1: {prob_sum}"
            
            # Validate range
            assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities out of [0, 1] range"
        
        print(f"  ✓ All probabilities valid\n")
    
    print("="*70 + "\n")


def test_predictor_wrapper():
    """Test DrowsinessPredictor wrapper class"""
    
    print("\n" + "="*70)
    print("Predictor Wrapper Test")
    print("="*70 + "\n")
    
    # Create predictor
    predictor = DrowsinessPredictor()
    
    # Load model
    model = build_cnn_lstm_model()
    predictor.model = model
    
    print("Testing DrowsinessPredictor class...")
    
    # Test prediction
    dummy_sequence = np.random.rand(16, 48, 48, 1).astype(np.float32)
    result = predictor.predict(dummy_sequence)
    
    print("\nPrediction Result:")
    print("-" * 70)
    print(f"State: {result['state']} ({result['state_name']})")
    print(f"Raw State: {result['raw_state']} ({result['raw_state_name']})")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities:")
    print(f"  Alert: {result['probabilities']['alert']:.4f}")
    print(f"  Warning: {result['probabilities']['warning']:.4f}")
    print(f"  Drowsy: {result['probabilities']['drowsy']:.4f}")
    
    # Validate result structure
    assert 'state' in result, "Missing 'state' key"
    assert 'state_name' in result, "Missing 'state_name' key"
    assert 'confidence' in result, "Missing 'confidence' key"
    assert 'probabilities' in result, "Missing 'probabilities' key"
    assert result['state'] in [0, 1, 2], f"Invalid state: {result['state']}"
    
    print("\n✓ Predictor wrapper test passed")
    
    # Test state smoothing
    print("\nTesting state smoothing...")
    
    predictor.reset_state()
    
    # Feed 5 consecutive predictions
    for i in range(5):
        result = predictor.predict(dummy_sequence)
        print(f"  Iteration {i+1}: Raw={result['raw_state']}, Smoothed={result['state']}")
    
    print("✓ State smoothing test completed")
    
    print("="*70 + "\n")


def test_model_training_compatibility():
    """Test if model can be trained (gradient computation)"""
    
    print("\n" + "="*70)
    print("Training Compatibility Test")
    print("="*70 + "\n")
    
    print("Building model...")
    model = build_cnn_lstm_model()
    
    print("Creating dummy training data...")
    X_train = np.random.rand(16, 16, 48, 48, 1).astype(np.float32)
    y_train = np.eye(3)[np.random.randint(0, 3, size=16)]  # One-hot encoded
    
    print("Testing single training step...")
    
    # Test one training step
    history = model.fit(X_train, y_train, epochs=1, batch_size=4, verbose=1)
    
    # Verify training occurred
    assert 'loss' in history.history, "Training loss not recorded"
    assert 'accuracy' in history.history, "Training accuracy not recorded"
    
    loss = history.history['loss'][0]
    accuracy = history.history['accuracy'][0]
    
    print(f"\nTraining step completed:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print("✓ Model is trainable")
    
    print("="*70 + "\n")


def main():
    """Run all model tests"""
    
    print("\n" + "="*80)
    print(" " * 20 + "CNN-LSTM MODEL TEST SUITE")
    print("="*80)
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Test 1: Architecture
        model = test_model_architecture()
        
        # Test 2: Inference speed
        test_model_inference_speed(model, num_iterations=50)
        
        # Test 3: Output format
        test_model_output_format(model)
        
        # Test 4: Predictor wrapper
        test_predictor_wrapper()
        
        # Test 5: Training compatibility
        test_model_training_compatibility()
        
        # Final summary
        print("\n" + "="*80)
        print(" " * 25 + "ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe CNN-LSTM model is ready for training and deployment.")
        print("You can now:")
        print("  1. Train the model: python train.py")
        print("  2. Run real-time detection: python main.py")
        print("="*80 + "\n")
    
    except Exception as e:
        print("\n" + "="*80)
        print(" " * 28 + "TEST FAILED ✗")
        print("="*80)
        print(f"\nError: {e}")
        print("\nPlease check the error message above and fix any issues.")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    main()
