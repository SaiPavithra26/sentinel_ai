import os
import time
import json
import argparse
import psutil
import numpy as np
import tensorflow as tf

def estimate_energy_consumption(duration_sec, cpu_percent, tdp_watts=15.0, idle_power_watts=5.0):
    """
    Estimates the battery energy consumption in milliwatt-hours (mWh)
    based on CPU utilization, execution duration, and thermal design power (TDP).
    
    Formula:
    Energy (Joules) = Duration (s) * (Idle Power + (CPU_Percent / 100) * TDP)
    Energy (mWh) = (Energy (Joules) / 3600) * 1000
    """
    fraction = max(0.0, min(100.0, cpu_percent)) / 100.0
    power_watts = idle_power_watts + (fraction * tdp_watts)
    energy_joules = duration_sec * power_watts
    energy_mwh = (energy_joules / 3600.0) * 1000.0
    return energy_mwh

def profile_tflite_performance(model_path, num_iterations=1000):
    """
    Profiles latency, memory usage, CPU usage, and battery consumption
    of the TensorFlow Lite interpreter.
    """
    process = psutil.Process(os.getpid())
    
    # 1. Base Resource Profile
    base_memory = process.memory_info().rss / (1024.0 * 1024.0) # MB
    
    # 2. Load Model
    start_load = time.perf_counter()
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    end_load = time.perf_counter()
    
    load_time_ms = (end_load - start_load) * 1000.0
    post_load_memory = process.memory_info().rss / (1024.0 * 1024.0) # MB
    memory_load_delta = post_load_memory - base_memory
    
    # Get input/output shapes
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']
    
    # Create dummy input vector matching model shape
    dummy_input = np.random.randn(*input_shape).astype(input_type)
    
    # 3. Inference Stress Test
    latencies = []
    cpu_usages = []
    
    print(f"Benchmarking TFLite model: {model_path}")
    print(f"Running {num_iterations} inferences...")
    
    # Track resources over inference loop
    start_test = time.perf_counter()
    
    for i in range(num_iterations):
        # Sample CPU usage periodically
        if i % 100 == 0:
            cpu_usages.append(process.cpu_percent())
            
        start_inf = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end_inf = time.perf_counter()
        
        latencies.append((end_inf - start_inf) * 1000.0) # ms
        
    end_test = time.perf_counter()
    duration_sec = end_test - start_test
    
    # 4. Compile Metrics
    post_test_memory = process.memory_info().rss / (1024.0 * 1024.0) # MB
    peak_memory_delta = post_test_memory - base_memory
    
    avg_cpu = float(np.mean(cpu_usages)) if cpu_usages else process.cpu_percent()
    
    # Latency percentiles
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    std_lat = float(np.std(latencies))
    
    # Battery consumption (check system battery first, else fallback to mathematical estimate)
    battery = psutil.sensors_battery()
    system_battery_drain = None
    
    if battery is not None:
        start_percent = battery.percent
        # If running on battery, wait a bit or track delta (optional, but sensors are slow to update)
        # So we'll rely mainly on estimation for precision, but show system state
        system_battery_drain = f"System plugged in: {battery.power_plugged}. Percent: {start_percent}%"
        
    estimated_drain_mwh = estimate_energy_consumption(duration_sec, avg_cpu)
    
    report = {
        "model_type": "TFLite",
        "model_path": model_path,
        "iterations": num_iterations,
        "load_time_ms": load_time_ms,
        "base_memory_mb": base_memory,
        "memory_load_delta_mb": memory_load_delta,
        "peak_memory_delta_mb": peak_memory_delta,
        "total_test_duration_sec": duration_sec,
        "cpu_usage_percent": avg_cpu,
        "battery_sensors": system_battery_drain,
        "estimated_energy_consumption_mwh": estimated_drain_mwh,
        "latency": {
            "mean_ms": mean_lat,
            "median_ms": median_lat,
            "p95_ms": p95_lat,
            "p99_ms": p99_lat,
            "std_ms": std_lat,
            "all_latencies_ms": latencies # saved for histogram plotting
        }
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description="SentinelAI: Performance Profiler & Benchmarker")
    parser.add_argument("--owner-id", type=str, default="1",
                        help="Owner ID folder containing trained models")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory where models are saved")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save profiling results")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of iterations for inference latency profiling")
                        
    args = parser.parse_args()
    
    tflite_path = os.path.join(args.model_dir, f"user_{args.owner_id}", "behavior_model.tflite")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found at {tflite_path}. Please convert Keras to TFLite first.")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    report = profile_tflite_performance(
        model_path=tflite_path,
        num_iterations=args.iterations
    )
    
    output_path = os.path.join(args.output_dir, "performance_metrics.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n==========================================")
    print(f"      PERFORMANCE PROFILING COMPLETED     ")
    print(f"==========================================")
    print(f"Report saved to: {output_path}")
    print(f"Mean Latency:    {report['latency']['mean_ms']:.4f} ms")
    print(f"95th Percentile: {report['latency']['p95_ms']:.4f} ms")
    print(f"Peak Mem Delta:  {report['peak_memory_delta_mb']:.2f} MB")
    print(f"Avg CPU Usage:   {report['cpu_usage_percent']:.2f}%")
    print(f"Est. Energy:     {report['estimated_energy_consumption_mwh']:.6f} mWh")
    print(f"==========================================\n")

if __name__ == "__main__":
    main()
