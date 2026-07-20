# Step 2: Common Frame-Source Abstraction Module

This document details the implementation, verification, and benchmarking of Step 2 for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

Step 2 has successfully introduced the common frame-source abstraction module (`ai-engine/ingest/`). It wraps raw frame capture sources (directories of sorted images, RTSP streaming feeds, local webcams, and video files) under a unified Python iterator interface.

For high-latency streams (like RTSP and camera hardware), it supports a real-time threaded background reader to drop stale frames and keep only the latest frame fresh, eliminating latency lag over time. For offline sources, it supports controlled frame-rate throttling (via `fps_limit`) and EOF looping.

---

## 2. Deliverables & File Structure

The following file structure has been established under the project workspace:

```text
ai-engine/
├── ingest/
│   ├── __init__.py
│   ├── types.py
│   └── reader.py
└── tests/
    └── test_reader.py
```

### Files Created
* **[types.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/ingest/types.py)**: Holds the `IngestedFrame` dataclass (storing frame pixel buffer, 0-indexed count, read timestamp, source string, width, and height).
* **[reader.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/ingest/reader.py)**: The `FrameReader` controller implementing resource capture (OpenCV `VideoCapture` or alphanumeric image directories), real-time thread queuing, fps-throttling, and context management.
* **[test_reader.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_reader.py)**: Test suite covering source type classification, image directory reading, loop boundary resets, and context manager lifecycles.
* **[verify_reader.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/examples/verify_reader.py)**: Verification script to ingest sample frame sequences and test performance benchmarks.

---

## 3. Preprocessing Ingestion Policies

### Image Directory Parsing
1. **Valid Extensions**: `.jpg`, `.jpeg`, `.png`, `.bmp` (case-insensitive).
2. **Natural Sorting**: Alphanumeric sorting logic (using `re.split` natural keys) to ensure frames like `frame_9.jpg` precede `frame_10.jpg` correctly.
3. **Looping**: Resets the internal index counter to $0$ on EOF when `loop=True`.

### Real-Time Threaded Mode
1. **Background Consumption**: Spawns a background thread consuming frames from `cv2.VideoCapture` at maximum frame rate.
2. **Buffer Management**: Utilizes a thread-safe `queue.Queue` of `maxsize=1`. If a new frame arrives before the downstream engine consumes the previous one, the old frame is discarded. This guarantees the client always obtains the latest captured state (0-lag accumulation).

---

## 4. Verification & Testing

### How to Run Ingest Verification
Ingest frames from the event's rendered directory at a simulated 10 FPS rate:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python ai-engine/examples/verify_reader.py --fps 10.0
```

### Ingestion Performance Output:
```text
Initializing FrameReader for source: /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/rendered_frames
Settings: fps_limit=10.0, loop=False
Detected Source Type: directory
Reader is running: True
------------------------------------------------------------
Frame 000 | Shape: 540x960x3 | Timestamp: 1784477624.1075 | Read Latency: 8.59 ms
Frame 001 | Shape: 540x960x3 | Timestamp: 1784477624.2176 | Read Latency: 109.72 ms
Frame 002 | Shape: 540x960x3 | Timestamp: 1784477624.3290 | Read Latency: 111.10 ms
Frame 003 | Shape: 540x960x3 | Timestamp: 1784477624.4392 | Read Latency: 110.17 ms
Frame 004 | Shape: 540x960x3 | Timestamp: 1784477624.5500 | Read Latency: 110.79 ms
------------------------------------------------------------
Successfully verified FrameReader.
Read 5 frames.
Average Frame Ingestion Latency: 90.07 ms
```

### How to Run Ingest Tests
Execute unit tests for both Step 1 (Density Adapter) and Step 2 (Ingest FrameReader):
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python -m unittest discover -s ai-engine/tests -p "test_*.py"
```
* **Status:** `OK` (All 18 tests passing successfully).

---

## 5. Step 3 Boundary Definition

With both the **Density Adapter (Step 1)** and the **Frame Reader (Step 2)** complete, we are ready for **Step 3 (Overlapping Grid Engine)**.
* **Input boundary for Step 3**: Real-time BGR frame sequences from the `FrameReader`.
* **Step 3 Target**: Build the grid segmenter module that divides frames into localized grids, configures relative overlaps, handles coordinates, and runs grid-level crowd estimation to feed the backend.
