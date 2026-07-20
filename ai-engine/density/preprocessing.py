import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def preprocess_image(
    frame: np.ndarray, max_size: int = 2048, min_size: int = 320, downsize: int = 32
) -> tuple[torch.Tensor, int, int, int, int]:
    """Preprocesses a single input frame for SCALNet.

    Args:
        frame: Input image frame as a NumPy array (BGR format).
        max_size: Maximum dimension limit.
        min_size: Minimum dimension limit.
        downsize: Grid size multiple to align spatial dimensions.

    Returns:
        tensor: Preprocessed float32 PyTorch tensor of shape [1, 3, H_aligned, W_aligned].
        orig_w: Original width of the frame.
        orig_h: Original height of the frame.
        new_w: Resized aligned width.
        new_h: Resized aligned height.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("Input frame must be a NumPy array.")

    # Check frame validity
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Input frame must be a 3-channel image (Height, Width, Channels).")

    # Original dimensions
    orig_h, orig_w = frame.shape[:2]

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    wd, ht = orig_w, orig_h
    resize = False

    # Apply SCALNet size constraints
    if wd > max_size or ht > max_size:
        nwd = int(wd * 1.0 / max(wd, ht) * max_size)
        nht = int(ht * 1.0 / max(wd, ht) * max_size)
        resize = True
        wd, ht = nwd, nht

    if wd < min_size or ht < min_size:
        nwd = int(wd * 1.0 / min(wd, ht) * min_size)
        nht = int(ht * 1.0 / min(wd, ht) * min_size)
        resize = True
        wd, ht = nwd, nht

    # Align with downsize stride (32)
    nht = (ht // downsize) * downsize
    nwd = (wd // downsize) * downsize

    # Resize if needed
    if nht != orig_h or nwd != orig_w or resize:
        img = img.resize((nwd, nht), resample=Image.BICUBIC)

    # Normalize and convert to tensor
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])

    tensor = transform(img)  # Shape [3, H, W]
    tensor = tensor.unsqueeze(0)  # Shape [1, 3, H, W]

    return tensor, orig_w, orig_h, nwd, nht
