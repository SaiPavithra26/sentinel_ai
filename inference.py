import os
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from feature_engineering import (
    KeystrokeFeatureExtractor,
    TouchFeatureExtractor,
    MotionFeatureExtractor
)
from preprocess import clean_and_interpolate_timestamps

def load_first_session_raw(user_folder, session_duration_mins=5):
    """
    Loads raw user logs, cleans timestamps, and extracts features for the very first session window.
    This simulates real-time data collection of a single session window.
    """
    print(f"Loading raw logs for simulation from: {user_folder}")
    
    # Locate files
    files = os.listdir(user_folder)
    acc_file = [f for f in files if "HandPhone" in f and "Accelerometer" in f]
    gyro_file = [f for f in files if "HandPhone" in f and "Gyroscope" in f]
    touch_file = [f for f in files if "HandPhone" in f and "TouchEvent" in f]
    key_file = [f for f in files if "HandPhone" in f and "Keyboard" in f]
    
    acc_df = pd.read_csv(os.path.join(user_folder, acc_file[0])) if acc_file else None
    gyro_df = pd.read_csv(os.path.join(user_folder, gyro_file[0])) if gyro_file else None
    touch_df = pd.read_csv(os.path.join(user_folder, touch_file[0])) if touch_file else None
    key_df = pd.read_csv(os.path.join(user_folder, key_file[0])) if key_file else None
    
    # Preprocess timestamps and interpolate
    acc_df = clean_and_interpolate_timestamps(acc_df)
    gyro_df = clean_and_interpolate_timestamps(gyro_df)
    touch_df = clean_and_interpolate_timestamps(touch_df)
    key_df = clean_and_interpolate_timestamps(key_df)
    
    # Find overall timeline start
    parsed_timestamps = []
    for df in [acc_df, gyro_df, touch_df, key_df]:
        if df is not None and "time" in df.columns:
            df.dropna(subset=["time"], inplace=True)
            parsed_timestamps.extend(df["time"].tolist())
            
    if not parsed_timestamps:
        raise ValueError("No valid timestamps found in logs.")
        
    start_time = min(parsed_timestamps)
    end_time = start_time + pd.Timedelta(minutes=session_duration_mins)
    
    # Filter event data for this single session window
    sess_acc = acc_df[(acc_df["time"] >= start_time) & (acc_df["time"] < end_time)] if acc_df is not None else None
    sess_gyro = gyro_df[(gyro_df["time"] >= start_time) & (gyro_df["time"] < end_time)] if gyro_df is not None else None
    sess_touch = touch_df[(touch_df["time"] >= start_time) & (touch_df["time"] < end_time)] if touch_df is not None else None
    sess_key = key_df[(key_df["time"] >= start_time) & (key_df["time"] < end_time)] if key_df is not None else None
    
    # Extract features
    key_extractor = KeystrokeFeatureExtractor()
    touch_extractor = TouchFeatureExtractor()
    acc_extractor = MotionFeatureExtractor(prefix="acc")
    gyro_extractor = MotionFeatureExtractor(prefix="gyro")
    
    session_duration_sec = session_duration_mins * 60.0
    k_feats = key_extractor.extract(sess_key, duration_seconds=session_duration_sec)
    t_feats = touch_extractor.extract(sess_touch)
    a_feats = acc_extractor.extract(sess_acc)
    g_feats = gyro_extractor.extract(sess_gyro)
    
    combined_feats = {}
    combined_feats.update(k_feats)
    combined_feats.update(t_feats)
    combined_feats.update(a_feats)
    combined_feats.update(g_feats)
    
    return combined_feats

def preprocess_sample(raw_feats, params):
    """
    Applies the exact preprocessing pipeline (clipping, imputation, scaling, feature selection)
    to a raw feature dictionary using parameters loaded from JSON.
    """
    # 1. Align features in original training feature ordering
    feature_cols = params["feature_cols"]
    sample_vector = []
    for col in feature_cols:
        val = raw_feats.get(col, np.nan)
        sample_vector.append(val)
    
    sample_vector = np.array(sample_vector, dtype=float)
    
    # 2. Apply outlier clipping
    clipping_bounds = params["clipping_bounds"]
    for i, col in enumerate(feature_cols):
        if col in clipping_bounds:
            lower = clipping_bounds[col]["lower"]
            upper = clipping_bounds[col]["upper"]
            sample_vector[i] = np.clip(sample_vector[i], lower, upper)
            
    # 3. Apply median imputation
    imputer_medians = params["imputer_medians"]
    for i, col in enumerate(feature_cols):
        if np.isnan(sample_vector[i]):
            sample_vector[i] = imputer_medians[col]
            
    # 4. Apply MinMaxScaler
    minmax_min = params["minmax_min"]
    minmax_scale = params["minmax_scale"]
    for i, col in enumerate(feature_cols):
        sample_vector[i] = sample_vector[i] * minmax_scale[col] + minmax_min[col]
        
    # 5. Filter selected features in correct order
    selected_features = params["selected_features"]
    selected_vector = []
    
    for col in selected_features:
        idx = feature_cols.index(col)
        selected_vector.append(sample_vector[idx])
        
    selected_vector = np.array(selected_vector, dtype=float)
    
    # 6. Apply StandardScaler (on selected features ONLY)
    standard_mean = params["standard_mean"]
    standard_scale = params["standard_scale"]
    for i, col in enumerate(selected_features):
        selected_vector[i] = (selected_vector[i] - standard_mean[col]) / standard_scale[col]
        
    # Reshape for model input: (1, num_features)
    return np.expand_dims(selected_vector, axis=0)

def run_inference(user_folder, owner_id, model_dir):
    """
    Runs end-to-end inference simulation on Keras and TFLite.
    """
    # 1. Load Preprocessing Parameters
    json_path = os.path.join("models", "preprocess_params.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Preprocessing parameters not found at {json_path}. Run preprocess.py first.")
        
    with open(json_path, "r") as f:
        params = json.load(f)
        
    # 2. Extract first session features
    raw_feats = load_first_session_raw(user_folder)
    
    # 3. Preprocess sample
    X_processed = preprocess_sample(raw_feats, params).astype(np.float32)
    print(f"Processed feature vector shape: {X_processed.shape}")
    print("Sample feature values (first 5):", X_processed[0, :5])
    
    # 4. Resolve Model Subdirectory
    user_specific_dir = os.path.join(model_dir, f"user_{owner_id}")
    if os.path.exists(os.path.join(user_specific_dir, "behavior_model.keras")):
        model_dir = user_specific_dir
        
    keras_path = os.path.join(model_dir, "behavior_model.keras")
    tflite_path = os.path.join(model_dir, "behavior_model.tflite")
    user_config_path = os.path.join(model_dir, "user_config.json")
    
    if not os.path.exists(keras_path) or not os.path.exists(tflite_path):
        raise FileNotFoundError(f"Keras or TFLite model files not found under {model_dir}. Train and convert first.")
        
    # Load user specific config for EER threshold
    threshold = 0.5
    if os.path.exists(user_config_path):
        with open(user_config_path, "r") as f:
            user_cfg = json.load(f)
            threshold = user_cfg.get("threshold", 0.5)
            print(f"Loaded calibrated threshold for Owner {owner_id}: {threshold:.4f}")
            
    # 5. Keras Model Inference
    print("\n--- Running Keras Model Prediction ---")
    keras_model = tf.keras.models.load_model(keras_path)
    keras_prob = float(keras_model.predict(X_processed, verbose=0)[0][0])
    keras_pred = "Owner" if keras_prob >= threshold else "Suspicious User"
    print(f"Keras Probability of Owner: {keras_prob:.6f} -> Result: {keras_pred}")
    
    # 6. TFLite Model Inference
    print("\n--- Running TFLite Interpreter Prediction ---")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]["index"], X_processed)
    interpreter.invoke()
    
    tflite_results = interpreter.get_tensor(output_details[0]["index"])
    tflite_prob = float(tflite_results[0][0])
    tflite_pred = "Owner" if tflite_prob >= threshold else "Suspicious User"
    print(f"TFLite Probability of Owner: {tflite_prob:.6f} -> Result: {tflite_pred}")
    
    # 7. Check Consistency
    difference = abs(keras_prob - tflite_prob)
    print("\n==========================================")
    print(f"Probability Discrepancy: {difference:.6e}")
    if difference < 2e-3:
        print("VERIFICATION: PASS (Keras and TFLite match)")
    else:
        print("VERIFICATION: FAIL (Mismatch in predictions)")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelAI: Real-Time Inference Simulation")
    parser.add_argument("--user-dir", type=str, default=r"d:\Documents\sentinel_ai\dataset\BB-MAS_Dataset\1",
                        help="Path to the user data folder to simulate")
    parser.add_argument("--owner-id", type=str, default="1",
                        help="Legitimate Owner ID to test against")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Base directory containing models")
                        
    args = parser.parse_args()
    
    run_inference(
        user_folder=args.user_dir,
        owner_id=args.owner_id,
        model_dir=args.model_dir
    )
