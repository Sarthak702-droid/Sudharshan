# Step 3: Overlapping Grid Engine Module

This document details the implementation, verification, and benchmarking of Step 3 for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

Step 3 has successfully introduced the overlapping grid division engine (`ai-engine/grid/`). The module divides the monitored corridor region into rectangular grids of customizable size and overlap, validates mathematical guidelines, and applies polygon clipping for non-rectangular camera perspective zones.

The grid engine is fully typed, tested, and features a visual developer validation renderer to trace grid configurations.

---

## 2. Deliverables & File Structure

The following file structure has been established under the project workspace:

```text
ai-engine/
├── grid/
│   ├── __init__.py
│   ├── types.py
│   └── generator.py
└── tests/
    └── test_grid.py
```

### Files Created
* **[types.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/grid/types.py)**: Holds the `GridBox` dataclass (storing grid ID, column/row indices, pixel bounds `(x1, y1) -> (x2, y2)`, full grid area, effective clipped area, and boundary polygon).
* **[generator.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/grid/generator.py)**: The `GridGenerator` engine implementing grid coordinate generation, step-size overlap offsets, boundary-uncovered tail corrections, and OpenCV-based mask polygon intersection calculations.
* **[test_grid.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_grid.py)**: Tests validating overlap bounds exceptions (e.g. disallowing 33% and 50% overlaps), coordinate step calculations, tail coverage grid additions, and intersection polygon area mathematics.
* **[verify_grid.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/examples/verify_grid.py)**: Verification script generating 45 grids inside a simulated perspective street boundary corridor.

---

## 3. Mathematical Overlap & Boundary Policies

### Step Size Calculation
Let grid size be $G$ and overlap ratio $r$:
$$\text{step\_size} = G \times (1 - r)$$
$$\text{overlap\_size} = G - \text{step\_size}$$

* **Rule Validation**: As per user guidelines, overlap ratios exceeding $0.25$ or close to $\approx 33.3\%$ or $\approx 50\%$ are strictly rejected to avoid scaling errors or integer alignment conflicts.

### Uncovered Boundary Tail Correction
If a monitored corridor length $L$ leaves an uncovered segment at the end because $x_{\text{last}} + G > L$, a final column/row of grid windows is automatically added starting at:
$$x_{\text{start}} = L - G$$
This ensures $100\%$ spatial coverage across event borders.

### Corridor Clipping (Intersection Area)
To clip grids to non-rectangular route corridors, we use a binary mask where the polygon is filled with `cv2.fillPoly`. For each grid window bounding box:
$$\text{effective\_area} = \sum_{y=y_1}^{y_2} \sum_{x=x_1}^{x_2} \text{mask}(x,y)$$
Grids with $\text{effective\_area} < 1\%$ of the total full grid area ($G^2$) are automatically discarded.

---

## 4. Verification & Testing

### How to Run Grid Layout Verification
To divide a $960 \times 540$ frame into $120\text{px}$ grids with $20\%$ overlap, run:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python ai-engine/examples/verify_grid.py --grid_size 120 --overlap 0.20
```

* **Visual Output:** A canvas visualization is rendered at [outputs/step_3_debug/grids_overlay.png](file:///home/sarthaktripathy/Documents/Sudharsan/outputs/step_3_debug/grids_overlay.png).
  * Green boxes indicate fully active interior grids ($100\%$ area coverage).
  * Yellow/Cyan boxes represent boundary grids clipped to the perspective trapezoid corridor.

### How to Run Grid Tests
Execute unit tests for Step 1 (Adapter), Step 2 (Reader), and Step 3 (Grid Generator):
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python -m unittest discover -s ai-engine/tests -p "test_*.py"
```
* **Status:** `OK` (All 21 tests passing successfully).

---

## 5. Step 4 Boundary Definition

With the **Density Adapter (Step 1)**, **Frame Ingestion (Step 2)**, and **Grid Segmentation (Step 3)** complete, we are ready for **Step 4 (Optical-Flow Engine)**.
* **Input boundary for Step 4**: Real-time BGR frame sequences from the `FrameReader` and active `GridBox` coordinates.
* **Step 4 Target**: Implement the optical flow algorithm (such as dense DIS flow or sparse Lucas-Kanade) inside active grid windows to generate movement vectors $(v_x, v_y)$ for crowd tracking.
