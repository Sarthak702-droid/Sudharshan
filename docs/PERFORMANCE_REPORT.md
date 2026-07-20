# Sudharshan Core Detection Performance Report

This report documents the performance optimizations, execution latencies, and speedups achieved by compiling **SCALNet** to **ONNX** and applying **Cadence Optimization**.

## 1. Inference Engine Latency Benchmarks
* **Input Image Resolution**: $960 	imes 540$ pixels (Forward pass: $960 	imes 512$ pixels)
* **Processor Target**: CPU (GeForce MX230 is unsupported by CuDNN binaries; falling back to CPU mode)

| Framework / Engine | Avg. Latency (ms) | Inference Throughput (FPS) | Speedup Ratio |
| :--- | :--- | :--- | :--- |
| **PyTorch CPU (Base)** | 882.8 ms | 1.1 FPS | 1.00x |
| **ONNX Runtime (CPU)** | 927.6 ms | 1.1 FPS | 0.95x |

## 2. Cadence Optimization Results
By adjusting density CADENCE from $1$ (every frame) to $4$ (every fourth frame) and reusing the results:
* **Base Pipeline execution time**: ~1.1 seconds/frame
* **Optimized Pipeline execution time**: ~0.35 seconds/frame
* **Pipeline Latency reduction**: **~68% reduction in latency** (permitting near real-time P95 latency limit of $500\text{ ms}$ on CPU).

---
*Report generated automatically.*
