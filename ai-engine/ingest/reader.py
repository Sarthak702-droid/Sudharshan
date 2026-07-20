import cv2
import numpy as np
import os
import re
import sys
import time
import threading
import queue
import warnings
from pathlib import Path
from typing import Union, Optional, List

from .types import IngestedFrame


class FrameReaderError(Exception):
    """Base exception for FrameReader errors."""
    pass


class SourceOpenError(FrameReaderError):
    """Raised when the frame source cannot be opened."""
    pass


class FrameReader:
    def __init__(
        self,
        source: Union[str, int],
        fps_limit: Optional[float] = None,
        realtime: Optional[bool] = None,
        loop: bool = False,
    ) -> None:
        """Initializes the FrameReader.

        Args:
            source: Path to video file, directory of images, RTSP URL, or local camera index (int).
            fps_limit: If set, limits the maximum frames yielded per second (for offline sources).
            realtime: If True, uses a background thread to read frames to prevent latency build-up
                      (recommended for RTSP streams and camera indexes). If None, auto-detects.
            loop: If True, loops back to the beginning upon reaching EOF (only for video files and directories).
        """
        self.source = source
        self.fps_limit = fps_limit
        self.loop = loop

        self.source_type = self._detect_source_type(source)

        # Auto-detect realtime mode
        if realtime is None:
            self.realtime = self.source_type in ("rtsp", "camera")
        else:
            self.realtime = realtime

        # State management variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.image_files: List[Path] = []
        self.image_index = 0
        self.frame_index = 0

        # Threading variables for real-time streaming
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)

        # Timing
        self._last_read_time = 0.0

    def _detect_source_type(self, source: Union[str, int]) -> str:
        """Helper to classify the source type."""
        if isinstance(source, int):
            return "camera"

        source_str = str(source).strip()

        # Check for network streams
        lower_str = source_str.lower()
        if (
            lower_str.startswith("rtsp://")
            or lower_str.startswith("rtmp://")
            or lower_str.startswith("http://")
            or lower_str.startswith("https://")
        ):
            return "rtsp"

        path = Path(source_str)
        if path.is_dir():
            return "directory"
        elif path.is_file():
            return "video"
        else:
            # Check if it looks like a camera index string
            if source_str.isdigit():
                self.source = int(source_str)
                return "camera"
            raise SourceOpenError(f"Unsupported or non-existent source: {source}")

    def _init_source(self) -> None:
        """Initializes the underlying capture driver."""
        if self.source_type in ("camera", "rtsp", "video"):
            source_val = self.source
            self.cap = cv2.VideoCapture(source_val)
            if not self.cap.isOpened():
                raise SourceOpenError(f"Failed to open OpenCV VideoCapture source: {self.source}")
        elif self.source_type == "directory":
            dir_path = Path(str(self.source))
            valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

            # Gather files and sort naturally
            files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

            def natural_keys(text):
                return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

            self.image_files = sorted(files, key=natural_keys)
            self.image_index = 0

            if not self.image_files:
                raise SourceOpenError(f"No valid image files found in directory: {self.source}")

    def start(self) -> None:
        """Starts the frame reading process."""
        if self._running:
            return

        self._init_source()
        self._running = True
        self.frame_index = 0
        self._last_read_time = 0.0

        if self.realtime:
            # Spawn a background thread to consume frames and keep queue fresh (1-frame buffer)
            self._thread = threading.Thread(target=self._background_reader, daemon=True)
            self._thread.start()

    def _background_reader(self) -> None:
        """Background thread target that consumes frames continuously."""
        consecutive_failures = 0
        while self._running:
            try:
                frame_data = self._read_next_raw()
                if frame_data is None:
                    if self.source_type in ("rtsp", "camera"):
                        consecutive_failures += 1
                        warnings.warn(f"[!] RTSP/Camera feed connection lost (fail count: {consecutive_failures}). Reconnecting in 2.0s...")
                        time.sleep(2.0)
                        if self.cap:
                            self.cap.release()
                        self.cap = cv2.VideoCapture(self.source)
                        continue
                    else:
                        # Video EOF reached
                        if self.loop:
                            self._reset_source()
                            continue
                        else:
                            break

                consecutive_failures = 0
                # Keep only the latest frame to prevent lag build-up
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put(frame_data)
            except Exception as e:
                warnings.warn(f"Background reader error: {e}")
                time.sleep(0.01)

    def _reset_source(self) -> None:
        """Resets stream back to first frame (for loop feature)."""
        if self.source_type in ("camera", "rtsp", "video") and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif self.source_type == "directory":
            self.image_index = 0

    def _read_next_raw(self) -> Optional[np.ndarray]:
        """Reads a raw frame from the active source."""
        if self.source_type in ("camera", "rtsp", "video") and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        elif self.source_type == "directory":
            if self.image_index >= len(self.image_files):
                return None
            img_path = self.image_files[self.image_index]
            frame = cv2.imread(str(img_path))
            self.image_index += 1
            return frame
        return None

    def read(self) -> Optional[IngestedFrame]:
        """Reads and returns the next IngestedFrame, applying fps_limits if set."""
        if not self._running:
            raise FrameReaderError("Reader is not running. Call start() first.")

        # Limit frame rate if requested
        if self.fps_limit and self._last_read_time > 0.0:
            target_interval = 1.0 / self.fps_limit
            elapsed = time.perf_counter() - self._last_read_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        raw_frame = None
        if self.realtime:
            # Fetch from background queue
            try:
                # If stream ended or is blocked, wait shortly
                raw_frame = self._frame_queue.get(timeout=2.0)
            except queue.Empty:
                raw_frame = None
        else:
            # Sync fetch
            raw_frame = self._read_next_raw()
            if raw_frame is None and self.loop:
                self._reset_source()
                raw_frame = self._read_next_raw()

        if raw_frame is None:
            return None

        h, w = raw_frame.shape[:2]
        ingested = IngestedFrame(
            frame=raw_frame,
            frame_index=self.frame_index,
            timestamp=time.time(),
            source=str(self.source),
            width=w,
            height=h,
        )

        self.frame_index += 1
        self._last_read_time = time.perf_counter()

        return ingested

    def stop(self) -> None:
        """Stops reading and releases all open resources."""
        self._running = False

        # Stop background thread
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Empty queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        # Release capture
        if self.cap:
            self.cap.release()
            self.cap = None

        self.image_files = []
        self.image_index = 0

    def __enter__(self) -> "FrameReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __iter__(self) -> "FrameReader":
        return self

    def __next__(self) -> IngestedFrame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
