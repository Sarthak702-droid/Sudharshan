# Step 1: SCALNet Density Foundation Module

This document details the implementation, verification, and benchmarking of Step 1 for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

Step 1 has successfully converted the standalone, legacy SCALNet codebase into a clean, reusable internal Python module (`ai-engine/density/`). The module provides high-performance single-frame inference on crowd image frames to generate density maps, estimated counts, and crowd presence masks.

The interface decouples model internals, CLI argument parsing, file saving, and GUI logic, returning results entirely in-memory as PyTorch-free standard NumPy types.

---

## 2. Deliverables & File Structure

The following file structure has been established under the project workspace:

```text
ai-engine/
├── __init__.py
├── density/
│   ├── __init__.py
│   ├── types.py
│   ├── scalnet_adapter.py
│   ├── preprocessing.py
│   └── postprocessing.py
├── tests/
│   ├── test_scalnet_adapter.py
│   ├── test_preprocessing.py
│   └── test_postprocessing.py
└── examples/
    └── verify_scalnet_single_frame.py
```

### Files Created
* **[types.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/density/types.py)**: Holds the `DensityInferenceResult` dataclass.
* **[preprocessing.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/density/preprocessing.py)**: Reproduces exact SCALNet image loading, dimension alignment (multiple of 32), and normalization policies.
* **[postprocessing.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/density/postprocessing.py)**: Contains count estimation, debug heatmap normalization, binary crowd-mask generation, and count-preserving bilinear resizing.
* **[scalnet_adapter.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/density/scalnet_adapter.py)**: The main adapter class wrapping model loading, device validation, fallback logic, evaluation state management, and inference.
* **[test_preprocessing.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_preprocessing.py)**: PyTest/Unittest-compatible preprocessing tests.
* **[test_postprocessing.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_postprocessing.py)**: Tests for resizing scaling preservation, masks, and summation.
* **[test_scalnet_adapter.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_scalnet_adapter.py)**: lifecycle and smoke tests.
* **[verify_scalnet_single_frame.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/examples/verify_scalnet_single_frame.py)**: Verification example script for developers.
* **[STEP_1_SCALNET_DENSITY_FOUNDATION.md](file:///home/sarthaktripathy/Documents/Sudharsan/docs/STEP_1_SCALNET_DENSITY_FOUNDATION.md)**: This documentation.

### Files Untouched (Preserved Third-Party Repos)
* `SCALNet/` training, testing, and model definitions (e.g., `SCALNet/models/DLANet.py`, `SCALNet/src/network.py`).
* `NWPU-Crowd-Sample-Code/`
* `VisDrone-MOT/` & `VisDrone2020-CC/`
* `ShanghaiTech/`
* `backend/` scaffold

---

## 3. Checkpoint Details

The official pre-trained SCALNet model checkpoint was obtained from the authors' NTU SharePoint resource.

* **Filename**: `000018.h5` (saved as `model.pth`)
* **Path**: `SCALNet/checkpoints/model.pth`
* **File Size**: 73.1 MB (76,693,584 bytes)
* **Model Variant**: DLANet (DLA-34 back-end)
* **Training Dataset**: NWPU-Crowd (highly complex/dense crowd dataset)
* **Parameters Layout**: Key weight format starts with `model.module.dla.` indicating a PyTorch DataParallel wrapped training format.

---

## 4. Preprocessing & Postprocessing Contracts

### Preprocessing Policy (`ai-engine/density/preprocessing.py`)
1. **Channel Order**: Converts input NumPy frame from BGR (OpenCV format) to RGB.
2. **Dimension Constraints**:
   * Upscales frames smaller than $320\text{px}$ on the short edge to $320\text{px}$.
   * Downscales frames larger than $2048\text{px}$ on the long edge to $2048\text{px}$ (preserving aspect ratio).
   * Aligns width and height to be divisible by $32\text{px}$ (SCALNet's downsize/stride requirement) by rounding down.
3. **Interpolation**: Uses Bicubic interpolation (`PIL.Image.BICUBIC`) for image resizing.
4. **Normalization**: Scales values to $[0.0, 1.0]$ and normalizes with:
   * **Mean**: $[0.485, 0.456, 0.406]$
   * **Std Dev**: $[0.229, 0.224, 0.225]$
5. **Tensor Shape**: Output PyTorch tensor shape `[1, 3, H, W]`.

### Postprocessing Policy (`ai-engine/density/postprocessing.py`)
1. **Count Calculation**: Sums the 2D density map after clipping negative output logits to $0.0$.
2. **Count-Preserving Resize**:
   When resizing a density map of size $H_2 \times W_2$ to match the original input resolution $H_1 \times W_1$:
   $$\text{scale} = \frac{H_1 \times W_1}{H_2 \times W_2}$$
   $$D_{\text{corrected}} = D_{\text{resized}} \times \text{scale}$$
   This preserves the total integral sum (crowd count) regardless of output resolution scaling.
3. **Crowd-Presence Mask**:
   Min-max normalizes the density map to $[0.0, 1.0]$. The relevance mask $M(x,y)$ is thresholded:
   $$M(x,y) = \begin{cases} 1 & \text{if } D_{\text{norm}}(x,y) \geq \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

---

## 5. Development Verification & Smoke Testing

### How to Run the Verification Script
To run inference on a single frame and save the debug output images, execute:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python ai-engine/examples/verify_scalnet_single_frame.py
```

### Debug Artifacts Generated
Saved under `outputs/step_1_debug/`:
* `density_raw.npy`: 2D float32 NumPy array containing the exact count-preserving density map.
* `density_heatmap.png`: Jet colormap representation of the density map.
* `crowd_mask.png`: Binary crowd presence mask (scaled to 255 for visibility).
* `overlay.png`: Overlay of the density heatmap on top of the original BGR frame.
* `result.json`: Summary metadata of the inference parameters.

#### Output `result.json` Example:
```json
{
    "estimated_count": 14.976,
    "inference_time_ms": 1040.74,
    "device": "cpu",
    "input_width": 960,
    "input_height": 540,
    "model_name": "SCALNet",
    "checkpoint_path": "/home/sarthaktripathy/Documents/Sudharsan/SCALNet/checkpoints/model.pth"
}
```

### How to Run Unit Tests
A full test suite covering lifecycle, exception raises, preprocessing transforms, and postprocessing math is available under `ai-engine/tests/`. Run them using the Python standard `unittest` framework:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python -m unittest discover -s ai-engine/tests -p "test_*.py"
```

---

## 6. Benchmarking & Performance Baseline

The benchmark was executed on a crowd frame `frame_000018.jpg` (resolution: $960 \times 540$) using the provided Python 3.12 virtual environment.

| Parameter | Metric (CPU Mode) |
|---|---|
| **Resolved Running Device** | CPU |
| **Model Weight Load Time** | 355.59 ms |
| **Total Inference Time** | 1040.74 ms |
| **Estimated Count** | 14.976 people |
| **Input Shape** | 960x540 |
| **Model Forward Resolution** | 960x512 |
| **Output Density Map Shape** | 960x540 (after count-preserving resize) |

---

## 7. Known Limitations & Edge Cases

1. **Hardware / Device Mismatch**:
   * The local GPU (GeForce MX230, sm_61 compute capability) is physically unsupported by the prebuilt PyTorch 2.11+cu128 binaries which require compute capability $\geq 7.5$.
   * **Resolution**: The `SCALNetAdapter` includes a robust runtime validator that tests CUDA capability during load and gracefully falls back to CPU mode with a warning if device is set to `auto`.
2. **PyTorch Version Warning**:
   * A compatibility warning is emitted by PyTorch due to compute capability mismatch on start. This warning is harmless and handled correctly by the adapter.
3. **Single Frame Inference Latency**:
   * Latency on CPU is $\sim 1.04\text{ s}$ per frame. This baseline is sufficient for Step 1 contract verification but does not satisfy real-time stream processing requirements. Future steps will target optimization (TensorRT, quantization, or running on supported CUDA devices).

---

## 8. Step 2 Boundary Definition

Step 1 has successfully packaged the density map foundation.
* **Input boundary for Step 2**: The core crowd-presence mask and density maps.
* **Step 2 Target**: Frame-source abstraction layer. Step 2 will build out the generalized reader class (accepting RTSP URLs, video files, camera devices, or directories of images) returning frames in a structured queue for downstream density and optical flow modules.
