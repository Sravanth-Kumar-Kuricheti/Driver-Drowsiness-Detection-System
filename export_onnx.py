"""
Export Keras .h5 Drowsiness Model to ONNX format
Enables 2x-3x faster CPU execution and eliminates TensorFlow / C++ DLL dependencies.
"""

import os
import sys

def convert_h5_to_onnx(h5_path="models/drowsiness_detector.h5", onnx_path="models/drowsiness_detector.onnx"):
    if not os.path.exists(h5_path):
        print(f"[Error] Source model file not found: {h5_path}")
        return False

    print(f"[1/3] Loading Keras model from {h5_path}...")
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("[Error] TensorFlow is required for the one-time export from .h5 to .onnx.")
        print("Install with: pip install tensorflow tf2onnx onnxruntime")
        return False

    try:
        import tf2onnx
    except ImportError:
        print("[Error] tf2onnx is required. Install with: pip install tf2onnx")
        return False

    try:
        model = keras.models.load_model(h5_path)
        print("      ✓ Keras model loaded successfully")
        
        print(f"[2/3] Converting to ONNX format...")
        input_spec = (tf.TensorSpec((None, 16, 48, 48, 1), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_spec,
            opset=13,
            output_path=onnx_path
        )
        print(f"      ✓ Model converted and saved to {onnx_path}")
        
        print("[3/3] Validating ONNX model with onnxruntime...")
        try:
            import onnxruntime as ort
            import numpy as np
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            dummy = np.random.rand(1, 16, 48, 48, 1).astype(np.float32)
            out = session.run(None, {session.get_inputs()[0].name: dummy})
            print(f"      ✓ Validation passed! Output shape: {out[0].shape}, probabilities: {out[0]}")
        except ImportError:
            print("      (onnxruntime not installed yet, skipping inference check)")
            
        print("\n" + "="*60)
        print("SUCCESS! You can now run the system using ONNX:")
        print(f"  python main.py --model {onnx_path}")
        print("="*60 + "\n")
        return True
    except Exception as e:
        print(f"[Error] Failed to convert model: {e}")
        return False

if __name__ == "__main__":
    h5_file = sys.argv[1] if len(sys.argv) > 1 else "models/drowsiness_detector.h5"
    onnx_file = sys.argv[2] if len(sys.argv) > 2 else "models/drowsiness_detector.onnx"
    convert_h5_to_onnx(h5_file, onnx_file)
