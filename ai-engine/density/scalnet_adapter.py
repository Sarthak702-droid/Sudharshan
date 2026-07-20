import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch

from .types import DensityInferenceResult
from .preprocessing import preprocess_image
from .postprocessing import (
    calculate_count,
    create_crowd_mask,
    resize_density_preserve_count,
)


class SCALNetCheckpointNotFound(FileNotFoundError):
    """Raised when the specified SCALNet checkpoint path does not exist."""
    pass


class SCALNetCheckpointIncompatible(ValueError):
    """Raised when the checkpoint weights cannot be loaded into the DLANet model structure."""
    pass


class SCALNetNotLoaded(RuntimeError):
    """Raised when attempting inference before the SCALNet model is successfully loaded."""
    pass


class InvalidFrameError(ValueError):
    """Raised when the input frame is invalid (e.g. wrong type or shape)."""
    pass


class SCALNetAdapter:
    def __init__(
        self,
        scalnet_root: Path,
        checkpoint_path: Path,
        device: str = "auto",
        mask_threshold: float = 0.01,
        use_onnx: bool = False,
    ) -> None:
        self.scalnet_root = Path(scalnet_root)
        self.checkpoint_path = Path(checkpoint_path)
        self._requested_device = device
        self.mask_threshold = mask_threshold
        self.use_onnx = use_onnx

        # Device selection logic
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                warnings.warn("CUDA is not available. Falling back to CPU for SCALNet inference.")
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._onnx_session = None
        self._is_loaded = False

    def load(self) -> None:
        """Loads the SCALNet architecture and pre-trained weights from the checkpoint or ONNX model."""
        # Check ONNX vs PyTorch checkpoint
        if self.use_onnx:
            onnx_path = self.checkpoint_path.with_suffix(".onnx")
            if not onnx_path.exists():
                # Fallback to model.onnx in checkpoints folder
                onnx_path = self.checkpoint_path.parent / "model.onnx"

            if not onnx_path.exists():
                raise SCALNetCheckpointNotFound(f"ONNX model file not found at: {onnx_path}")

            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                # Default thread count
                sess_options.intra_op_num_threads = 4
                self._onnx_session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                self._is_loaded = True
                return
            except Exception as e:
                raise RuntimeError(f"Failed to load ONNX Runtime session: {e}")

        # Normal PyTorch loading path
        if not self.checkpoint_path.exists():
            raise SCALNetCheckpointNotFound(f"Checkpoint file not found at: {self.checkpoint_path}")

        # Add SCALNet root directories to PYTHONPATH dynamically for relative imports to resolve
        scalnet_root_str = str(self.scalnet_root.resolve())
        src_dir = str((self.scalnet_root / "src").resolve())

        if scalnet_root_str not in sys.path:
            sys.path.insert(0, scalnet_root_str)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        try:
            from models.DLANet import DLANet
            from src.network import load_net
        except ImportError as e:
            raise ImportError(
                f"Failed to import SCALNet model definition. Check if 'scalnet_root' is correct. Details: {e}"
            )

        try:
            # Instantiate model
            model = DLANet()

            # Load weights using custom load_net (reads H5 weight files)
            load_net(str(self.checkpoint_path), model, prefix="model.module.")

            # Move model to target device and set to eval mode
            model.to(self.device)
            model.eval()

            # Test device compatibility with a dummy forward pass
            if self.device == "cuda":
                try:
                    dummy_input = torch.zeros(1, 3, 64, 64, device="cuda")
                    with torch.no_grad():
                        model(dummy_input)
                except Exception as cuda_err:
                    if self._requested_device == "auto":
                        warnings.warn(
                            f"CUDA is available but execution failed (e.g. GPU compute capability mismatch). "
                            f"Falling back to CPU. Details: {cuda_err}"
                        )
                        self.device = "cpu"
                        model.to(self.device)
                    else:
                        raise cuda_err

            self._model = model
            self._is_loaded = True
        except Exception as e:
            if isinstance(e, SCALNetCheckpointNotFound):
                raise e
            raise SCALNetCheckpointIncompatible(
                f"Failed to load checkpoint '{self.checkpoint_path}' into DLANet structure: {e}"
            )

    def infer(self, frame: np.ndarray) -> DensityInferenceResult:
        """Runs crowd density inference on a single frame in memory."""
        if not self.is_loaded():
            raise SCALNetNotLoaded("Model is not loaded. Call load() before infer().")

        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError("Input frame must be a NumPy ndarray.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InvalidFrameError("Input frame must have 3 dimensions and 3 color channels (Height, Width, 3).")

        t_start = time.perf_counter()

        # 1. Preprocess
        try:
            tensor, orig_w, orig_h, nwd, nht = preprocess_image(frame)
        except Exception as e:
            raise InvalidFrameError(f"Error during preprocessing: {e}")

        # 2. Run model forward (ONNX vs PyTorch)
        if self.use_onnx and self._onnx_session is not None:
            try:
                # Align shapes to 512x960 (which model expects after preprocessing alignment)
                # Ensure input shape is exactly 1x3x512x960 (static shape exported)
                input_np = tensor.cpu().numpy()
                if input_np.shape != (1, 3, 512, 960):
                    # Pad or resize input numpy to match exactly 1x3x512x960 static model shape
                    # Create blank aligned buffer
                    aligned_input = np.zeros((1, 3, 512, 960), dtype=np.float32)
                    ch, cw = min(512, input_np.shape[2]), min(960, input_np.shape[3])
                    aligned_input[0, :, :ch, :cw] = input_np[0, :, :ch, :cw]
                    input_np = aligned_input

                input_name = self._onnx_session.get_inputs()[0].name
                outputs = self._onnx_session.run(None, {input_name: input_np})
                # DLANet outputs a list/tuple: (hm, dm)
                dm_np_raw = outputs[1]
                dm_np = dm_np_raw.squeeze(0).squeeze(0)
            except Exception as e:
                raise RuntimeError(f"Error during ONNX Runtime inference: {e}")
        else:
            tensor = tensor.to(self.device)
            try:
                with torch.no_grad():
                    hm, dm = self._model(tensor)
            except Exception as e:
                raise RuntimeError(f"Error during forward pass inference: {e}")
            dm_np = dm.squeeze(0).squeeze(0).cpu().numpy()

        # 3. Postprocess
        dm_np = np.clip(dm_np, a_min=0.0, a_max=None)

        # Estimate the crowd count on the raw/downsampled output map
        count = calculate_count(dm_np)

        # Count-preserving resize back to original input dimensions
        density_map_resized = resize_density_preserve_count(dm_np, orig_h, orig_w)

        # Generate crowd-presence mask on resized density map
        mask = create_crowd_mask(density_map_resized, self.mask_threshold)

        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000.0

        return DensityInferenceResult(
            density_map=density_map_resized,
            estimated_count=count,
            crowd_mask=mask,
            inference_time_ms=inference_time_ms,
            device="onnx_cpu" if self.use_onnx else self.device,
            input_width=orig_w,
            input_height=orig_h,
            model_name="SCALNet_ONNX" if self.use_onnx else "SCALNet",
            checkpoint_path=str(self.checkpoint_path),
        )

    def is_loaded(self) -> bool:
        """Returns True if the model is currently loaded, False otherwise."""
        return self._is_loaded

    def unload(self) -> None:
        """Unloads the model from memory."""
        self._model = None
        self._onnx_session = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
