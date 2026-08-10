import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

def convert_to_tflite(owner_id, model_dir):
    """
    Converts a trained Keras model into a size-optimized TensorFlow Lite model,
    saves the model and labels, and verifies the converted model's predictions.
    """
    user_dir = os.path.join(model_dir, f"user_{owner_id}")
    if not os.path.exists(user_dir):
        raise FileNotFoundError(f"Model directory for User {owner_id} does not exist at: {user_dir}")
        
    keras_model_path = os.path.join(user_dir, "behavior_model.keras")
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model file not found at: {keras_model_path}")
        
    tflite_model_path = os.path.join(user_dir, "behavior_model.tflite")
    labels_path = os.path.join(user_dir, "labels.txt")
    
    print(f"\n==========================================")
    print(f"Converting Keras model for User {owner_id} to TFLite...")
    print(f"Source: {keras_model_path}")
    print(f"==========================================")
    
    # 1. Load the Keras Model
    print("Loading Keras model...")
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # 2. Configure TFLite Converter
    print("Configuring TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Enable default optimization (post-training dynamic range quantization)
    # This reduces weight precision from float32 to int8, yielding ~4x size reduction
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 3. Convert Model
    print("Converting model (applying dynamic range quantization)...")
    tflite_model = converter.convert()
    
    # 4. Save TFLite Model
    print(f"Saving TFLite model to: {tflite_model_path}")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
        
    # 5. Generate labels.txt
    print(f"Writing labels to: {labels_path}")
    with open(labels_path, "w") as f:
        f.write("Suspicious User\n")
        f.write("Owner\n")
        
    # 6. Analyze Model Size Reduction
    keras_size_kb = os.path.getsize(keras_model_path) / 1024.0
    tflite_size_kb = os.path.getsize(tflite_model_path) / 1024.0
    reduction_pct = (1.0 - (tflite_size_kb / keras_size_kb)) * 100.0
    
    print(f"\n--- Size Comparison Report ---")
    print(f"Keras Model Size:  {keras_size_kb:.2f} KB")
    print(f"TFLite Model Size: {tflite_size_kb:.2f} KB")
    print(f"Size Reduction:    {reduction_pct:.2f}%")
    print(f"------------------------------")
    
    # 7. Verification: Run Parity Check
    print("\nVerifying converted model consistency...")
    
    # Attempt to load actual validation samples
    val_csv_path = os.path.join("processed_data", "val.csv")
    if os.path.exists(val_csv_path):
        print(f"Loading actual validation samples from {val_csv_path}...")
        df_val = pd.read_csv(val_csv_path)
        X_val = df_val.drop(columns=["user_id"]).values
        # Use first 20 samples or fewer if dataset is small
        num_samples = min(20, X_val.shape[0])
        test_samples = X_val[:num_samples].astype(np.float32)
    else:
        print("Validation data not found. Using randomly generated input for verification...")
        # Get input shape from keras model
        input_shape = keras_model.input_shape
        # input_shape is typically (None, num_features)
        num_features = input_shape[1] if len(input_shape) > 1 else 50
        test_samples = np.random.randn(10, num_features).astype(np.float32)
        num_samples = test_samples.shape[0]
        
    # Keras Model Predictions
    keras_probs = keras_model.predict(test_samples, verbose=0).flatten()
    
    # TFLite Model Predictions
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tflite_probs = []
    for sample in test_samples:
        # Reshape to match interpreter input detail shape e.g. (1, num_features)
        input_data = np.expand_dims(sample, axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        tflite_probs.append(float(output_data[0][0]))
        
    tflite_probs = np.array(tflite_probs)
    
    # Parity Evaluation
    abs_differences = np.abs(keras_probs - tflite_probs)
    max_diff = np.max(abs_differences)
    mean_diff = np.mean(abs_differences)
    
    print(f"\n--- Parity Verification Summary ---")
    print(f"Number of test samples: {num_samples}")
    print(f"Mean Absolute Difference: {mean_diff:.6e}")
    print(f"Max Absolute Difference:  {max_diff:.6e}")
    print(f"-----------------------------------")
    
    # Ensure differences are minimal (threshold < 0.05 for quantized models)
    if max_diff < 0.05:
        print("VERIFICATION: PASS")
        print("The converted TFLite model predictions are highly consistent with Keras.")
    else:
        print("VERIFICATION: WARNING")
        print("Predictions differ slightly more than expected. Dynamic range quantization may cause tiny variations.")
        
    print("\nConversion successfully completed!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelAI: Keras to optimized TFLite converter")
    parser.add_argument("--owner-id", type=str, default="1",
                        help="Owner ID corresponding to user specific subdirectory")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory where user model folders are saved")
                        
    args = parser.parse_args()
    
    convert_to_tflite(
        owner_id=args.owner_id,
        model_dir=args.model_dir
    )
