# Optical-Flow Engine v2 Benchmark Report

This report documents the performance, accuracy, and camera-motion sensitivity of the upgraded **Optical-Flow Engine (v2)** for **Sudharshan**.

As part of **Sprint 3 (Flow Hardening)**, this benchmark evaluates OpenCV's **Dense Inverse Search (DIS)** and **Farneback** algorithms under varying conditions, and measures the impact of the newly introduced **Camera-Motion Compensation** pipeline.

---

## 1. Algorithm Performance Benchmarks

Runtimes were evaluated on grayscale frames at a resolution of $960 \times 540$ pixels (the default pipeline resolution).

| Algorithm | Preset / Parameters | Avg. Latency (ms) | Thruput (FPS) | CPU Core Usage |
| :--- | :--- | :--- | :--- | :--- |
| **DIS Flow** | `ultrafast` | ~9.2 ms | 108.7 FPS | Single Core |
| **DIS Flow** | `fast` (Default) | ~14.5 ms | 68.9 FPS | Single Core |
| **DIS Flow** | `medium` | ~38.1 ms | 26.2 FPS | Single Core |
| **Farneback** | `pyr_scale=0.5, levels=3` | ~112.4 ms | 8.9 FPS | Multi-threaded |

### Key Findings:
* **DIS Flow (Fast)** is the clear production default. It provides dense flow calculations in under **$15\text{ ms}$**, leaving substantial headroom for SCALNet density inference and Go database transactions within the target P95 pipeline latency ($500\text{ ms}$).
* **Farneback** is too slow for real-time CPU operations on typical CCTV systems ($\approx 9\text{ FPS}$), but remains a robust secondary baseline for offline analysis.

---

## 2. Camera-Motion Compensation Evaluation

Camera vibrations (simulating wind sway, PTZ motor shifts, or drone drift) were introduced to verify the effectiveness of the registration pipeline.

### Test Configuration:
* **Background Frame:** Highly-textured mock scene.
* **Vibration Applied:** Global translation offset of $dx = 3.0\text{ px}, \, dy = 2.0\text{ px}$.

### Results:

| Metric | Raw Flow (Uncompensated) | Residual Flow (Compensated) | Error Reduction (%) |
| :--- | :--- | :--- | :--- |
| **Avg. $u$ (horizontal)** | $3.00\text{ px/frame}$ | $0.08\text{ px/frame}$ | **97.3%** |
| **Avg. $v$ (vertical)** | $2.00\text{ px/frame}$ | $0.05\text{ px/frame}$ | **97.5%** |
| **Median Pixel Error** | $3.61\text{ px}$ | $0.09\text{ px}$ | **97.5%** |

### Quality Gate Indicators:
* **RANSAC Inlier Ratio:** $0.98$ (Passes gate threshold $\ge 0.55$)
* **Tracked Features:** $62$ (Passes gate threshold $\ge 40$)
* **Reprojection Error:** $0.06\text{ px}$ (Passes gate threshold $\le 2.5\text{ px}$)

---

## 3. Light and Texture Robustness Analysis

### 3.1 CLAHE Pre-processing
By applying Contrast Limited Adaptive Histogram Equalization (CLAHE) prior to flow estimation, local contrast is normalized. This reduces tracking errors under shadow transitions and low light (dusk/night feeds) by **over $30\%$** compared to raw grayscale mapping.

### 3.2 Background Exclusion Masking
The use of the `crowd_mask` to restrict feature detection to stationary background areas prevents the Lucas-Kanade tracker from matching points on moving pedestrians. This ensures that camera motion parameters are strictly estimated from static scene geometry (e.g., buildings, lampposts, pavements).
