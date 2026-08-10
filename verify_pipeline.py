import os
import pandas as pd
import numpy as np
import subprocess
import joblib

def run_step(cmd, step_name):
    print(f"\n=== Running: {step_name} ===")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print stdout or errors
    if result.returncode != 0:
        print("\n--- ERROR STDOUT ---")
        print(result.stdout)
        print("\n--- ERROR STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Step '{step_name}' failed with exit code {result.returncode}")
    else:
        print("Success! Output snippet:")
        # Show first 15 lines of stdout
        lines = result.stdout.splitlines()
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print("  ...")

def run_verification():
    # Step 1: Run preprocessing pipeline using cached features if available
    preprocess_cmd = [
        "python", "preprocess.py",
        "--dataset-dir", r"d:\Documents\sentinel_ai\dataset\BB-MAS_Dataset"
    ]
    run_step(preprocess_cmd, "Preprocessing Pipeline")
    
    # Step 2: Verify outputs from preprocessing
    print("\n=== Verifying Preprocessing Outputs ===")
    train_path = os.path.join("processed_data", "train.csv")
    val_path = os.path.join("processed_data", "val.csv")
    test_path = os.path.join("processed_data", "test.csv")
    
    assert os.path.exists(train_path), "train.csv is missing!"
    assert os.path.exists(val_path), "val.csv is missing!"
    assert os.path.exists(test_path), "test.csv is missing!"
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Train Shape: {df_train.shape}")
    print(f"Val Shape:   {df_val.shape}")
    print(f"Test Shape:  {df_test.shape}")
    
    assert not df_train.isna().any().any(), "Train set contains NaNs!"
    assert not df_val.isna().any().any(), "Val set contains NaNs!"
    assert not df_test.isna().any().any(), "Test set contains NaNs!"
    print("Verification: Zero NaNs in all outputs. (Imputation: PASS)")
    
    # Verify standard scaler and MinMaxScaler exist
    assert os.path.exists(os.path.join("models", "imputer.joblib")), "imputer is missing!"
    assert os.path.exists(os.path.join("models", "minmax_scaler.joblib")), "minmax scaler is missing!"
    assert os.path.exists(os.path.join("models", "standard_scaler.joblib")), "standard scaler is missing!"
    assert os.path.exists(os.path.join("models", "feature_selector.joblib")), "feature selector is missing!"
    assert os.path.exists(os.path.join("models", "preprocess_params.json")), "preprocess_params.json is missing!"
    print("Verification: All scaler files and preprocessing parameters JSON saved. (PASS)")
    
    # Step 3: Run Training for Owner ID 1 (Samsung S6 user)
    # Train for 15 epochs to ensure it finishes quickly but triggers early stopping/calibration
    train_cmd = [
        "python", "train.py",
        "--owner-id", "1",
        "--epochs", "15",
        "--batch-size", "16",
        "--data-dir", "processed_data",
        "--model-dir", "models"
    ]
    run_step(train_cmd, "MLP Model Training")
    
    # Verify training outputs
    model_path = os.path.join("models", "user_1", "behavior_model.keras")
    user_config_path = os.path.join("models", "user_1", "user_config.json")
    assert os.path.exists(model_path), "behavior_model.keras was not saved!"
    assert os.path.exists(user_config_path), "user_config.json calibration is missing!"
    print("Verification: Keras model and EER threshold calibration JSON saved. (PASS)")
    
    # Step 4: Convert Keras model to quantized TFLite model
    convert_cmd = [
        "python", "convert_tflite.py",
        "--owner-id", "1",
        "--model-dir", "models"
    ]
    run_step(convert_cmd, "TFLite Model Conversion")
    
    # Verify TFLite output
    tflite_path = os.path.join("models", "user_1", "behavior_model.tflite")
    assert os.path.exists(tflite_path), "TFLite behavior_model.tflite is missing!"
    print("Verification: Optimized TFLite model successfully compiled. (PASS)")
    
    # Step 5: Execute Inference parity check
    inference_cmd = [
        "python", "inference.py",
        "--user-dir", r"d:\Documents\sentinel_ai\dataset\BB-MAS_Dataset\1",
        "--owner-id", "1"
    ]
    run_step(inference_cmd, "Inference Simulation & Parity Check")
    
    print("\n==============================================")
    print("      ALL END-TO-END PIPELINE CHECKS PASSED!  ")
    print("==============================================")

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        print(f"\nVerification Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
