import pandas as pd
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/user_1/behavior_model.tflite"
TEST_PATH = "processed_data/test.csv"

# --------------------------------------------------
# Load test data
# --------------------------------------------------
df = pd.read_csv(TEST_PATH)
print("Test dataset shape:", df.shape)

# Remove user_id
X = df.drop(columns=["user_id"])
print("Feature shape:", X.shape)

if X.shape[1] != 50:
    raise ValueError(f"Expected 50 features, found {X.shape[1]}")

# Pick one real sample
sample = X.iloc[0].values.astype(np.float32)
print("\nReal feature vector:")
print(sample)

# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_info = input_details[0]
output_info = output_details[0]

# --------------------------------------------------
# Prepare input
# --------------------------------------------------
sample = sample.reshape(input_info["shape"])
interpreter.set_tensor(input_info["index"], sample)

# --------------------------------------------------
# Run inference
# --------------------------------------------------
interpreter.invoke()

output = interpreter.get_tensor(output_info["index"])
confidence = float(output.flatten()[0])

print("\n======================================")
print("REAL INFERENCE RESULT")
print("======================================")
print("Model:", MODEL_PATH)
print("Output:", output)
print(f"Confidence: {confidence:.6f}")

if 0 <= confidence <= 1:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")
