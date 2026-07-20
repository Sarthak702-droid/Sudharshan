# AI and Backend Algorithm Hardening

## Implemented safeguards

- Optical-flow vectors are rejected when non-finite, below the configured noise
  floor, above the physical plausibility ceiling, or associated with a scene cut.
- Frame reliability combines registration inliers, reprojection error, texture,
  usable-vector coverage, and camera-motion reliability.
- Grid vectors use density weighting and an iteratively reweighted Huber
  estimator initialized at the component median. Near-zero and invalid vectors
  do not contribute direction evidence.
- Crowd presence uses a smooth probability and temporal hysteresis. Crowd class
  changes require three consistent frames. Classes are `EMPTY`, `SPARSE`,
  `MODERATE`, `DENSE`, and `CRITICAL`.
- Calibrated grids use people per square metre. Speed is converted to metres per
  second from an explicit scale or a local homography area-Jacobian estimate.
- Risk and event telemetry remains visible on unreliable frames, but automatic
  alerts are suppressed unless the metric passes the confidence gate.
- Temporal output includes acceleration and spatial divergence.
- Backend ingestion rejects invalid ranges, NaN/Inf values, malformed timestamps,
  unknown fields, empty batches, and oversized batches. Retries are idempotent,
  and stale WebSocket clients cannot block ingestion.

## Configuration

The new thresholds are in `configs/grid_config.yaml` under `flow` and
`risk`. `meters_per_pixel` must remain `null` until a camera is calibrated.
Metric-grid mode can derive a local scale from its ground/image polygons.

## Verification

- Python unit tests cover scene cuts, outlier rejection, confidence gating,
  physical units, crowd-class hysteresis, and existing behavior.
- Go tests cover validation and retry idempotency against a temporary SQLite DB.
- A deterministic synthetic test with 20% extreme vector outliers reduced vector
  endpoint error from 4.0851 to 0.1848 pixels/frame (95.48%). This is a robustness
  test, not a substitute for held-out field accuracy evaluation.

## Accuracy boundary

Algorithmic hardening reduces known false-motion and duplicate-alert failure
modes. A real accuracy claim still requires camera-specific calibration and a
held-out labelled video set containing crowd direction, count, speed, counter-flow,
stagnation, dispersal, and scene-cut examples.
