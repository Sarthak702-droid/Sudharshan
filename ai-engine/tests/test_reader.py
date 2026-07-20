import unittest
import numpy as np
import cv2
import tempfile
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "ai-engine") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-engine"))

from ingest.types import IngestedFrame
from ingest.reader import FrameReader, SourceOpenError, FrameReaderError


class TestFrameReader(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for mock images
        self.test_dir = tempfile.mkdtemp()
        self.dir_path = Path(self.test_dir)

        # Write some dummy images (64x64 BGR)
        self.frame_paths = []
        for i in range(3):
            img_path = self.dir_path / f"frame_{i:03d}.jpg"
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[10:20, 10:20] = i * 50  # Add some distinct pixels
            cv2.imwrite(str(img_path), img)
            self.frame_paths.append(img_path)

    def tearDown(self):
        # Clean up temp directory
        shutil.rmtree(self.test_dir)

    def test_source_type_detection(self):
        # Check camera (int)
        reader_cam = FrameReader(source=0, realtime=False)
        self.assertEqual(reader_cam.source_type, "camera")

        # Check RTSP
        reader_rtsp = FrameReader(source="rtsp://localhost:8554/stream", realtime=False)
        self.assertEqual(reader_rtsp.source_type, "rtsp")

        # Check directory
        reader_dir = FrameReader(source=self.test_dir, realtime=False)
        self.assertEqual(reader_dir.source_type, "directory")

        # Check invalid source raises SourceOpenError
        with self.assertRaises(SourceOpenError):
            FrameReader(source="invalid_source_path_xyz.mp4", realtime=False)

    def test_directory_reading_sequential(self):
        # Initialize reader on image directory (realtime=False for deterministic sequential reads)
        reader = FrameReader(source=self.test_dir, realtime=False)

        # Must call start() before reading
        with self.assertRaises(FrameReaderError):
            reader.read()

        reader.start()
        self.assertEqual(len(reader.image_files), 3)

        # Read frames
        for i in range(3):
            ingested = reader.read()
            self.assertIsNotNone(ingested)
            self.assertIsInstance(ingested, IngestedFrame)
            self.assertEqual(ingested.frame_index, i)
            self.assertEqual(ingested.width, 64)
            self.assertEqual(ingested.height, 64)
            self.assertEqual(ingested.source, self.test_dir)
            self.assertEqual(ingested.frame.shape, (64, 64, 3))

        # 4th read should return None (EOF)
        self.assertIsNone(reader.read())
        reader.stop()

    def test_looping_behavior(self):
        # Loop=True should reset to index 0 on EOF
        reader = FrameReader(source=self.test_dir, realtime=False, loop=True)
        reader.start()

        # Read 5 frames from a directory of 3 images
        indices = []
        for _ in range(5):
            ingested = reader.read()
            self.assertIsNotNone(ingested)
            indices.append(ingested.frame_index)

        # frame_index increments monotonically, but the underlying images cycle
        self.assertEqual(indices, [0, 1, 2, 3, 4])
        self.assertEqual(reader.image_index, 2)  # Resets to 0, read 1st, read 2nd -> index is 2

        reader.stop()

    def test_context_manager(self):
        with FrameReader(source=self.test_dir, realtime=False) as reader:
            self.assertTrue(reader._running)
            ingested = reader.read()
            self.assertIsNotNone(ingested)
        self.assertFalse(reader._running)


if __name__ == "__main__":
    unittest.main()
