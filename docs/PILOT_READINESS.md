# Sudharshan Core Detection Pilot Readiness Pack

This document details the security policies, deployment packages, and operator decision support configurations for launching the **Sudharshan** crowd-flow monitoring engine in a field pilot environment.

---

## 1. Security & Privacy Safeguards
* **Zero Biometric Capture**: Sudharshan does not perform facial recognition, iris scanning, gait signature mapping, or any form of unique identity tracking. Telemetry is purely grid-level count, speed, and direction.
* **On-Premise Boundaries**: Designed for local on-prem execution within standard CCTV networks. It does not require outbound internet access or cloud services.
* **Audio Scrubbing**: The ingest pipeline operates strictly on visual frames and discards audio channels entirely.

## 2. Telemetry Persistence & Databases
* **Local Database**: Persistent metrics are saved in a local SQLite database (`sudharshan.db`) managed by the Go backend.
* **Schema Auto-Migrations**: The Go backend dynamically verifies column schemas on start and migrates tables automatically to add advanced telemetry metrics (`turbulence_score`, etc.) without loss of historical records.
* **Aggregated Retention Policy**: Telmetry records are written at 1-5 second intervals instead of raw frame increments to prevent storage bottlenecks.

## 3. High-Availability & Stream Reconnects
* **Threaded Queue Buffering**: The ingest reader spawns background worker threads using a thread-safe 1-frame queue. This guarantees that downstream processing is always provided with the most recent frame and drops stale frames.
* **Automatic RTSP Reconnects**: If network latency or camera outages disrupt live RTSP feeds, the background ingest reader automatically retries connection initialization every 2.0 seconds until restored.

## 4. Operator Limits & Decision Support
* **Experimental Warning Flags**: All automated alarm events (Speed Surge, Stasis/Crush Hazards, and Flow Turbulence) are designated as **Experimental Decision Support Alerts**.
* **Decision Authority**: Telemetry is designed to augment operator awareness, not trigger automatic police dispatch or physical barrier blockades.
* **Hysteresis Buffers**: Transitions between risk states require persistence over time thresholds (e.g. 2.0 seconds) and include a 10% safety margin to suppress flashing alarm states on control monitors.

## 5. Startup & Deployment Package
To deploy the pilot locally:
1. Initialize backend persistence and WebSocket listeners:
   ```bash
   ./run_backend.sh
   ```
2. Start the pipeline processing:
   ```bash
   ./run_pipeline_sample.sh
   ```
   To run with ONNX optimization:
   ```bash
   ./run_pipeline_sample.sh --use_onnx
   ```
   To change predictor models:
   ```bash
   ./run_pipeline_sample.sh --predictor_type gru
   ```
