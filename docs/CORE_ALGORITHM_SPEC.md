# Core Algorithm Specification & Telemetry Field Dictionary

This document serves as the official specification for the core mathematical formulas, processing logic, and telemetry outputs of **Sudharshan** (Live Crowd Flow Intelligence Engine).

This specification represents the primary deliverable for **Sprint 1 (Audit & Freeze)** to establish ground truth contract definitions and audit formulas prior to physical calibration.

---

## 1. Step 4: Optical-Flow Engine

The Optical-Flow Engine calculates dense motion vector displacements $F(x,y) = (u, v)$ between consecutive video frames $I_{t-1}$ and $I_t$.

### 1.1 Local Preprocessing (CLAHE)
To stabilize vector tracking in challenging visual conditions (dust, shadows, or twilight), frames are converted to grayscale and enhanced using **Contrast Limited Adaptive Histogram Equalization (CLAHE)** before calculation:
$$\hat{I}(x,y) = \text{CLAHE}\Big(\text{Grayscale}\big(I(x,y)\big)\Big)$$
* **clipLimit:** $2.0$
* **tileGridSize:** $8 \times 8$ pixels

### 1.2 Base Displacements
The default optical flow method is OpenCV's **Dense Inverse Search (DIS)**. This generates two 2D matrices:
* $f_x(x,y) = u$: Horizontal pixel displacement per frame.
* $f_y(x,y) = v$: Vertical pixel displacement per frame.

### 1.3 Grid-Wise Aggregation
For each overlapping grid cell $g$, we calculate the density-weighted mean displacement $(\bar{u}_g, \bar{v}_g)$ using the normalized local density map $D(x,y)$ as weights:
$$\bar{u}_g = \frac{\sum_{(x,y) \in g} u(x,y) \cdot D(x,y)}{\sum_{(x,y) \in g} D(x,y) + \epsilon}$$
$$\bar{v}_g = \frac{\sum_{(x,y) \in g} v(x,y) \cdot D(x,y)}{\sum_{(x,y) \in g} D(x,y) + \epsilon}$$
*(Where $\epsilon = 10^{-6}$ to prevent division-by-zero).*

---

## 2. Step 5: Density-Flow Spatial Fusion

The Fusion Aggregator integrates the output of the density estimator (SCALNet) and the optical flow engine to evaluate crowd density, speeds, conflicts, and overall congestion risk levels.

### 2.1 Perspective Scaling Audit (Disabled)
> [!WARNING]
> The legacy arbitrary $y$-position perspective scaling factor:
> $$\text{scale}_{\text{heuristic}} = 1.0 + 1.5 \cdot \left(1.0 - \frac{y_{\text{center}}}{H_{\text{frame}}}\right)$$
> has been **formally disabled** and is restricted from production runs. It is uncalibrated and does not correspond to a valid camera projection model. All spatial parameters default to unscaled image-space values ($\text{scale} = 1.0$) until proper camera homography matrix calibration is applied in Sprint 2.

### 2.2 Crowd Congestion Risk Formulation
The congestion score for each grid cell $g$ is a bounded metric ($[0.0, 100.0]$) computed via a weighted combination of five distinct indicators:
$$\text{Congestion Score}_g = 100.0 \cdot \text{clamp}\Big( 0.35 \cdot S_{\text{density}} + 0.20 \cdot S_{\text{slow}} + 0.20 \cdot S_{\text{stagnation}} + 0.15 \cdot S_{\text{conflict}} + 0.10 \cdot S_{\text{reverse}}, \, 0.0, \, 1.0 \Big)$$

#### Component Calculations:
1. **Density Score ($S_{\text{density}}$):**
   $$S_{\text{density}} = \text{clamp}\left( \frac{\text{Density}_g}{\text{Threshold}_{\text{critical\_density}}}, \, 0.0, \, 1.0 \right)$$
2. **Slow Speed Score ($S_{\text{slow}}$):**
   $$S_{\text{slow}} = 1.0 - \text{clamp}\left( \frac{\text{Speed}_g}{\text{Threshold}_{\text{normal\_speed}}}, \, 0.0, \, 1.0 \right)$$
3. **Stagnation Score ($S_{\text{stagnation}}$):** Represents the co-occurrence of high density and slow movement.
   $$S_{\text{stagnation}} = S_{\text{density}} \cdot S_{\text{slow}}$$
4. **Flow Conflict Score ($S_{\text{conflict}}$):** Combines grid circular variance (local turbulence) and neighbor motion mismatch.
   $$S_{\text{conflict}} = 0.70 \cdot \text{Variance}_{\text{circular}} + 0.30 \cdot \text{Conflict}_{\text{neighbors}}$$
5. **Reverse Flow Score ($S_{\text{reverse}}$):** Measures backward motion against the expected corridor vector $\vec{E}_g$.
   $$S_{\text{reverse}} = \max\left(0.0, \, -\vec{V}_g \cdot \vec{E}_g\right) \quad \text{where } \|\vec{V}_g\|_2 = 1$$

---

## 3. Telemetry Field Dictionary

Every telemetry record output by the **Sudharshan** pipeline is defined by the following contract. All safety-critical warnings and forecasts are explicitly designated as **Experimental (Unvalidated Heuristics)**.

| Field Name | Data Type | Physical Unit | Description / Formula | Status |
| :--- | :--- | :--- | :--- | :--- |
| `grid_id` | `string` | N/A | Unique identifier of the grid cell in format `G_Row_Col` (e.g., `G_02_04`). | **Validated** |
| `frame_id` | `integer` | N/A | Sequence frame index. | **Validated** |
| `timestamp_sec` | `float` | Seconds | Cumulative timestamp since sequence start. | **Validated** |
| `crowd_count` | `float` | people | Estimated number of persons in the grid cell. | **Validated (Image-space)** |
| `density` | `float` | people/pixel | Grid crowd count divided by the grid pixel area. | **Validated (Image-space)** |
| `mean_dx` | `float` | pixels/frame | Density-weighted average horizontal pixel velocity ($\bar{u}_g$). | **Validated** |
| `mean_dy` | `float` | pixels/frame | Density-weighted average vertical pixel velocity ($\bar{v}_g$). | **Validated** |
| `motion_magnitude` | `float` | pixels/frame | Magnitude of the average flow vector: $V = \sqrt{\bar{u}_g^2 + \bar{v}_g^2}$. | **Validated** |
| `direction` | `string` | N/A | Cardinal direction label (e.g., `EAST`, `NORTH-WEST`, `STATIONARY`). | **Validated** |
| `direction_deg` | `float` | Degrees | Angle of the average flow vector, mapped from $[0.0, 360.0)$. | **Validated** |
| `coherence` | `float` | N/A | Vector alignment parameter in range $[0.0, 1.0]$. Computed as $1 - S_{\text{conflict}}$. | **Validated** |
| `risk_score` | `float` | N/A | The aggregated congestion score in range $[0.0, 100.0]$. | **Validated Contract** |
| `risk_level` | `string` | N/A | Categorical danger level mapping: `GREEN` ($<40$), `YELLOW` ($<60$), `ORANGE` ($<80$), `RED` ($\ge 80$). | **Validated Contract** |
| `confidence` | `float` | N/A | Composite indicator of metric reliability in range $[0.0, 1.0]$. | **Validated Contract** |
| `turbulence_score` | `float` | N/A | Angular circular variance of the local flow vector field. | **Experimental** |
| `speed_surge_warning` | `boolean` | N/A | Flag indicating a sudden speed increase $\ge 2\times$ the historical moving average. | **Experimental (Unvalidated Heuristic)** |
| `stasis_warning` | `boolean` | N/A | Flag indicating sustained high density and stagnation ($\text{speed} < 0.3$) for $\ge 2.0$ seconds. | **Experimental (Unvalidated Heuristic)** |
| `turbulence_warning` | `boolean` | N/A | Flag indicating chaotic local flow ($S_{\text{conflict}} > 0.6$) under dense packing. | **Experimental (Unvalidated Heuristic)** |
| `predicted_count` | `float` | people | Least-squares linear extrapolation of the crowd count $15$ frames in the future. | **Experimental (Forecast)** |
| `predicted_congestion_score`| `float` | N/A | Least-squares linear extrapolation of the congestion score $15$ frames in the future. | **Experimental (Forecast)** |
| `predicted_risk_level` | `string` | N/A | Categorical danger level mapping computed on the predicted congestion score. | **Experimental (Forecast)** |
| `trend_slope` | `float` | val/frame | Least-squares linear regression slope ($m$) evaluated over the last $15$ frames. | **Experimental (Forecast)** |
