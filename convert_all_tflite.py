import os
import glob
import json
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODELS_DIR = "models"

KERAS_FILENAME = "behavior_model.keras"
TFLITE_FILENAME = "behavior_model.tflite"
LABELS_FILENAME = "labels.txt"


# ============================================================
# CONVERT ONE MODEL
# ============================================================

def convert_model(keras_path):
    user_dir = os.path.dirname(keras_path)
    user_name = os.path.basename(user_dir)
    tflite_path = os.path.join(user_dir, TFLITE_FILENAME)
    labels_path = os.path.join(user_dir, LABELS_FILENAME)

    print("\n" + "=" * 70)
    print(f"Converting: {user_name}")
    print(f"Keras:  {keras_path}")
    print(f"TFLite: {tflite_path}")
    print("=" * 70)

    try:
        # ----------------------------------------------------
        # 1. Load trained Keras model
        # ----------------------------------------------------
        print("Loading Keras model...")
        model = tf.keras.models.load_model(keras_path, compile=False)
        print("Keras model loaded successfully.")

        # ----------------------------------------------------
        # 2. Create TFLite converter
        # ----------------------------------------------------
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # ----------------------------------------------------
        # 3. Enable optimization (Dynamic range quantization)
        # ----------------------------------------------------
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # ----------------------------------------------------
        # 4. Convert
        # ----------------------------------------------------
        print("Converting to TensorFlow Lite...")
        tflite_model = converter.convert()

        # ----------------------------------------------------
        # 5. Save TFLite model
        # ----------------------------------------------------
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved ({len(tflite_model) / 1024:.2f} KB)")

        # ----------------------------------------------------
        # 6. Generate labels.txt
        # ----------------------------------------------------
        with open(labels_path, "w") as f:
            f.write("Suspicious User\n")
            f.write("Owner\n")
        print("labels.txt created.")

        # ----------------------------------------------------
        # 7. Verify TFLite model
        # ----------------------------------------------------
        print("Verifying TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) != 1:
            raise RuntimeError(f"Expected 1 input, found {len(input_details)}")
        if len(output_details) != 1:
            raise RuntimeError(f"Expected 1 output, found {len(output_details)}")

        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        output_shape = output_details[0]["shape"]
        output_dtype = output_details[0]["dtype"]

        print(f"Input shape : {input_shape}")
        print(f"Input dtype : {input_dtype}")
        print(f"Output shape: {output_shape}")
        print(f"Output dtype: {output_dtype}")

        # ----------------------------------------------------
        # 8. Run test inference
        # ----------------------------------------------------
        input_size = input_shape[1]
        test_input = np.zeros((1, input_size), dtype=input_dtype)
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        confidence = float(output.flatten()[0])

        print(f"Test output : {output}")
        print(f"Confidence  : {confidence:.6f}")

        # ----------------------------------------------------
        # 9. Check confidence range
        # ----------------------------------------------------
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"Invalid sigmoid output: {confidence}")

        print(f"RESULT: PASS - {user_name}")
        return True

    except Exception as e:
        print(f"RESULT: FAIL - {user_name}")
        print(f"Error: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n")
    print("=" * 70)
    print("       SENTINELAI - BULK TFLITE CONVERSION")
    print("=" * 70)

    # Find all Keras models
    pattern = os.path.join(MODELS_DIR, "user_*", KERAS_FILENAME)
    keras_models = sorted(
        glob.glob(pattern),
        key=lambda x: int(os.path.basename(os.path.dirname(x)).replace("user_", ""))
    )

    total_models = len(keras_models)
    print(f"\nKeras models found: {total_models}")

    if total_models == 0:
        print("ERROR: No Keras models found.")
        return

    # Convert every model
    successful = []
    failed = []

    for index, keras_path in enumerate(keras_models, start=1):
        user_name = os.path.basename(os.path.dirname(keras_path))
        print(f"\n[{index}/{total_models}] Processing {user_name}")
        result = convert_model(keras_path)
        if result:
            successful.append(user_name)
        else:
            failed.append(user_name)

    # Final verification
    print("\n")
    print("=" * 70)
    print("             FINAL TFLITE VERIFICATION")
    print("=" * 70)

    tflite_models = glob.glob(os.path.join(MODELS_DIR, "user_*", TFLITE_FILENAME))
    labels_files = glob.glob(os.path.join(MODELS_DIR, "user_*", LABELS_FILENAME))

    print(f"\nKeras models : {total_models}")
    print(f"TFLite models: {len(tflite_models)}")
    print(f"Labels files : {len(labels_files)}")

    # Find missing TFLite models
    missing = []
    for keras_path in keras_models:
        user_dir = os.path.dirname(keras_path)
        tflite_path = os.path.join(user_dir, TFLITE_FILENAME)
        if not os.path.exists(tflite_path):
            missing.append(os.path.basename(user_dir))

    print("\n" + "-" * 70)
    if len(missing) == 0:
        print("Missing TFLite models: 0")
    else:
        print(f"Missing TFLite models: {len(missing)}")
        for user in missing:
            print(f"  MISSING: {user}")

    # Final summary
    print("\n" + "=" * 70)
    print("                    SUMMARY")
    print("=" * 70)

    print(f"Total Keras models : {total_models}")
    print(f"Successful         : {len(successful)}")
    print(f"Failed             : {len(failed)}")
    print(f"TFLite models      : {len(tflite_models)}")
    print(f"labels.txt files   : {len(labels_files)}")

    if failed:
        print("\nFailed users:")
        for user in failed:
            print(f"  - {user}")

    if missing:
        print("\nMissing TFLite models:")
        for user in missing:
            print(f"  - {user}")

    if (
        len(successful) == total_models
        and len(tflite_models) == total_models
        and len(labels_files) == total_models
        and len(missing) == 0
    ):
        print("\n" + "=" * 70)
        print("       ALL 116 MODELS CONVERTED SUCCESSFULLY")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("       CONVERSION INCOMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
