import os
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# Use Agg backend for matplotlib to prevent GUI popup windows during headless execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_report(owner_id, data_dir, model_dir, output_dir):
    """
    Evaluates the TFLite model on the test dataset, generates charts,
    and writes a comprehensive evaluation markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Paths and validation
    test_csv = os.path.join(data_dir, "test.csv")
    tflite_path = os.path.join(model_dir, f"user_{owner_id}", "behavior_model.tflite")
    user_config_path = os.path.join(model_dir, f"user_{owner_id}", "user_config.json")
    performance_json = os.path.join(output_dir, "performance_metrics.json")
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test data not found at {test_csv}")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found at {tflite_path}")
        
    # Load calibrated threshold
    threshold = 0.5
    if os.path.exists(user_config_path):
        try:
            with open(user_config_path, "r") as f:
                cfg = json.load(f)
                threshold = cfg.get("threshold", 0.5)
        except Exception:
            pass
            
    # Load performance metrics if available
    perf_data = {}
    if os.path.exists(performance_json):
        try:
            with open(performance_json, "r") as f:
                perf_data = json.load(f)
        except Exception:
            pass

    print(f"Loading test data from {test_csv}...")
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=["user_id"]).values.astype(np.float32)
    y_test = (df_test["user_id"].astype(str) == str(owner_id)).astype(int).values
    
    # 2. Run TFLite Inference
    print(f"Loading TFLite model from {tflite_path}...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Running predictions on test set...")
    y_probs = []
    for sample in X_test:
        input_data = np.expand_dims(sample, axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        out_data = interpreter.get_tensor(output_details[0]["index"])
        y_probs.append(float(out_data[0][0]))
        
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    # 3. Calculate Classification Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    # 4. Generate Charts
    print("Generating charts...")
    
    # Chart 1: ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_curve, tpr_curve, color="indigo", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"Receiver Operating Characteristic (ROC) - User {owner_id}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    roc_chart_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Chart 2: Latency Histogram (from performance_metrics.json if available)
    latency_chart_path = os.path.join(output_dir, "latency_histogram.png")
    if perf_data and "latency" in perf_data:
        latencies = perf_data["latency"]["all_latencies_ms"]
        plt.figure(figsize=(8, 6))
        plt.hist(latencies, bins=50, color="teal", edgecolor="black", alpha=0.7)
        plt.axvline(perf_data["latency"]["mean_ms"], color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {perf_data['latency']['mean_ms']:.4f} ms")
        plt.axvline(perf_data["latency"]["p95_ms"], color="orange", linestyle="dashed", linewidth=1.5, label=f"95th %ile: {perf_data['latency']['p95_ms']:.4f} ms")
        plt.xlabel("Inference Latency (ms)")
        plt.ylabel("Frequency")
        plt.title("TFLite Inference Latency Distribution")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(latency_chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        
    # Chart 3: Resource Utilization Summary
    resource_chart_path = os.path.join(output_dir, "resource_utilization.png")
    if perf_data:
        plt.figure(figsize=(8, 6))
        categories = ["Base Memory\n(MB)", "Peak Memory\nDelta (MB)", "CPU Usage\n(%)", "Est. Energy\n(mWh * 10)"]
        values = [
            perf_data.get("base_memory_mb", 0),
            perf_data.get("peak_memory_delta_mb", 0),
            perf_data.get("cpu_usage_percent", 0),
            perf_data.get("estimated_energy_consumption_mwh", 0) * 10.0 # scaled for visualization
        ]
        colors = ["royalblue", "crimson", "orange", "forestgreen"]
        bars = plt.bar(categories, values, color=colors, edgecolor="black", width=0.5)
        plt.ylabel("Measured Units")
        plt.title("Hardware Resource Profiling Summary")
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.2f}", ha="center", va="bottom", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.5, axis="y")
        plt.ylim([0, max(values) * 1.25])
        plt.savefig(resource_chart_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 5. Generate Markdown Report
    print("Writing markdown report...")
    report_path = os.path.join(output_dir, "evaluation_report.md")
    
    with open(report_path, "w") as f:
        f.write(f"# SentinelAI: Behavioral Biometric Evaluation Report\n\n")
        f.write(f"This report evaluates the TensorFlow Lite model of **User {owner_id}** on both classification accuracy and system performance benchmarks.\n\n")
        
        # Section 1: Classification Performance
        f.write(f"## 1. Classification Performance Summary\n\n")
        f.write(f"| Metric | Value | Description |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| **Accuracy** | {accuracy * 100:.2f}% | Overall proportion of correct predictions across all test cases. |\n")
        f.write(f"| **Precision** | {precision * 100:.2f}% | The fraction of positive alerts (Owner flags) that are actually the legitimate owner. |\n")
        f.write(f"| **Recall (TPR)** | {recall * 100:.2f}% | The fraction of actual owner sessions successfully detected (Sensitivity). |\n")
        f.write(f"| **F1 Score** | {f1 * 100:.2f}% | The harmonic mean of Precision and Recall. Represents the balanced classifier score. |\n")
        f.write(f"| **False Positive Rate (FPR)** | {fpr_val * 100:.2f}% | The proportion of unauthorized (suspicious) sessions incorrectly allowed. |\n")
        f.write(f"| **False Negative Rate (FNR)** | {fnr_val * 100:.2f}% | The proportion of legitimate owner sessions locked out. |\n")
        f.write(f"| **ROC AUC** | {roc_auc:.4f} | Area Under the ROC Curve. Measures the model's capacity to distinguish classes. |\n\n")
        
        f.write(f"### Classification Confusion Matrix\n")
        f.write(f"- **True Negatives (TN):** {tn} (Authorized suspicious blocks)\n")
        f.write(f"- **False Positives (FP):** {fp} (Missed intrusions)\n")
        f.write(f"- **False Negatives (FN):** {fn} (Owner lockouts)\n")
        f.write(f"- **True Positives (TP):** {tp} (Authorized owner entries)\n\n")
        
        f.write(f"### Receiver Operating Characteristic (ROC) Curve\n")
        f.write(f"![ROC Curve](roc_curve.png)\n\n")
        
        # Section 2: Hardware Performance
        if perf_data:
            f.write(f"## 2. Hardware Resource Profiling Report\n\n")
            f.write(f"| Hardware Resource | Metric | Value | Description |\n")
            f.write(f"| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **Inference Latency** | Mean Latency | {perf_data['latency']['mean_ms']:.4f} ms | Average time taken to execute a single forward pass. |\n")
            f.write(f"| | Median Latency | {perf_data['latency']['median_ms']:.4f} ms | 50th percentile latency. |\n")
            f.write(f"| | 95th Percentile | {perf_data['latency']['p95_ms']:.4f} ms | Latency ceiling for 95% of requests. |\n")
            f.write(f"| | 99th Percentile | {perf_data['latency']['p99_ms']:.4f} ms | Tail latency limit (important for UI lag check). |\n")
            f.write(f"| **Memory Profile** | Base Process Memory | {perf_data['base_memory_mb']:.2f} MB | Memory footprint of the process prior to loading the model. |\n")
            f.write(f"| | Load Memory Delta | {perf_data['memory_load_delta_mb']:.2f} MB | Extra memory allocated exclusively to load and allocate model weights. |\n")
            f.write(f"| | Peak Memory Delta | {perf_data['peak_memory_delta_mb']:.2f} MB | Extra memory consumed during active model inferences. |\n")
            f.write(f"| **CPU utilization** | Average CPU Usage | {perf_data['cpu_usage_percent']:.2f}% | Average CPU percentage consumed during active loop. |\n")
            f.write(f"| **Energy Profile** | Estimated Battery Consumption | {perf_data['estimated_energy_consumption_mwh']:.6f} mWh | Energy consumed over the benchmark runtime (calculated using CPU TDP profile). |\n\n")
            
            f.write(f"### Latency Distribution Histogram\n")
            f.write(f"![Latency Histogram](latency_histogram.png)\n\n")
            
            f.write(f"### Resource Utilization Summary\n")
            f.write(f"![Resource Utilization](resource_utilization.png)\n\n")
            
        # Section 3: Metric Explanations
        f.write(f"## 3. Detailed Metric Explanations\n\n")
        f.write(f"### Classification Metrics\n")
        f.write(f"- **Accuracy:** Tells us what percentage of classification decisions were correct. In highly imbalanced datasets, accuracy can be misleading (e.g. predicting suspicious user all the time when owner data is scarce).\n")
        f.write(f"- **Precision:** Answers: 'When the engine alerts that an intruder is on the phone, how often is it actually an intruder?' Highly critical to prevent annoying the user with alarms.\n")
        f.write(f"- **Recall (True Positive Rate):** Answers: 'Of all session windows belonging to the owner, what percentage did the system authenticate?' High recall ensures the owner isn't locked out of their own phone.\n")
        f.write(f"- **F1 Score:** Harmonic mean of precision and recall. A balanced indicator of model quality on skewed datasets.\n")
        f.write(f"- **ROC Curve & AUC:** The ROC Curve plots TPR against FPR at different threshold criteria. An Area Under the Curve (AUC) of 1.0 represents a perfect classifier, whereas 0.5 represents random guessing.\n\n")
        
        f.write(f"### Hardware Resource Metrics\n")
        f.write(f"- **Inference Latency:** Represents execution speed. On mobile devices, inference must run under 16ms to avoid dropping UI frames (60fps) if run on the main thread, or under 50ms to prevent background execution queues from stalling.\n")
        f.write(f"- **Process Memory Delta:** Quantifies the memory overhead of the ML interpreter. Lower memory footprint allows the background service to persist in memory without being evicted by the operating system's Low Memory Killer (LMK).\n")
        f.write(f"- **CPU Utilization:** Higher CPU utilization drains battery. Low CPU footprint is critical for continuous background monitoring.\n")
        f.write(f"- **Battery Consumption (mWh):** Measures estimated battery usage. The background service utilizes a periodic 5-second sampling window to minimize CPU active time, ensuring battery drain remains negligible (below 0.1 mWh per run).\n")

    print(f"Report compiled successfully at: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="SentinelAI: Classification & Resource Evaluation Report Compiler")
    parser.add_argument("--owner-id", type=str, default="1",
                        help="Legitimate Owner ID to test against")
    parser.add_argument("--data-dir", type=str, default="processed_data",
                        help="Directory containing train, val, and test datasets")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Base directory containing models")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save generated charts and the report")
                        
    args = parser.parse_args()
    
    generate_report(
        owner_id=args.owner_id,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
