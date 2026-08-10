import os
import glob
import numpy as np
import tensorflow as tf

MODEL_PATTERN = "models/user_*/behavior_model.tflite"

models = sorted(glob.glob(MODEL_PATTERN))

print("=" * 70)
print("        TFLITE MODEL VERIFICATION")
print("=" * 70)

print(f"Found models: {len(models)}")

if not models:
    print("ERROR: No TFLite models found.")
    exit(1)

passed = 0
failed = 0

for model_path in models:

    user = os.path.basename(os.path.dirname(model_path))

    print("\n" + "-" * 70)
    print(f"Testing: {user}")
    print(f"Path: {model_path}")

    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)

        # Allocate tensors
        interpreter.allocate_tensors()

        # Get input/output information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) != 1:
            raise ValueError(
                f"Expected 1 input tensor, found {len(input_details)}"
            )

        if len(output_details) != 1:
            raise ValueError(
                f"Expected 1 output tensor, found {len(output_details)}"
            )

        input_info = input_details[0]
        output_info = output_details[0]

        print("Input shape :", input_info["shape"])
        print("Input dtype :", input_info["dtype"])
        print("Output shape:", output_info["shape"])
        print("Output dtype:", output_info["dtype"])

        # Expected model input
        expected_features = 50

        input_shape = input_info["shape"]

        if input_shape[-1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, "
                f"but model expects {input_shape[-1]}"
            )

        # Create dummy input
        dummy_input = np.zeros(
            input_shape,
            dtype=input_info["dtype"]
        )

        # Run inference
        interpreter.set_tensor(
            input_info["index"],
            dummy_input
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_info["index"]
        )

        print("Output:", output)

        # Check output
        if output.size == 0:
            raise ValueError("Empty output")

        confidence = float(output.flatten()[0])

        print(f"Confidence: {confidence:.6f}")

        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"Confidence {confidence} outside [0,1]"
            )

        print("RESULT: PASS")

        passed += 1

    except Exception as e:

        print("RESULT: FAIL")
        print("ERROR:", e)

        failed += 1


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total models : {len(models)}")
print(f"Passed       : {passed}")
print(f"Failed       : {failed}")

if failed == 0:
    print("\nALL TFLITE MODELS PASSED.")
else:
    print(f"\nWARNING: {failed} models failed.")

print("=" * 70)