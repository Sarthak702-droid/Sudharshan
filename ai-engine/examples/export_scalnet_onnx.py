import os
import sys
import torch
from pathlib import Path

# Add SCALNet root directories to PYTHONPATH dynamically for relative imports to resolve
base_dir = Path(__file__).resolve().parent.parent.parent
scalnet_root = base_dir / "SCALNet"
scalnet_root_str = str(scalnet_root.resolve())
src_dir = str((scalnet_root / "src").resolve())

if scalnet_root_str not in sys.path:
    sys.path.insert(0, scalnet_root_str)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from models.DLANet import DLANet
    from src.network import load_net
except ImportError as e:
    print(f"Failed to import SCALNet model definition. Check if path is correct: {e}")
    sys.exit(1)

def export_onnx():
    checkpoint_path = scalnet_root / "checkpoints" / "model.pth"
    onnx_path = scalnet_root / "checkpoints" / "model.onnx"

    print(f"Loading DLANet from checkpoint: {checkpoint_path}...")
    model = DLANet()
    load_net(str(checkpoint_path), model, prefix="model.module.")
    model.eval()
    model.cpu()

    # Define dummy input matching default processing resolution (aspect ratio aligned width/height multiple of 32)
    # Default is 960x540. Aligned to multiple of 32 is 960x512.
    dummy_input = torch.zeros(1, 3, 512, 960, dtype=torch.float32)

    print(f"Exporting model to ONNX: {onnx_path}...")

    # Export to ONNX with static input shape
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    print("ONNX export completed successfully!")

if __name__ == "__main__":
    export_onnx()
