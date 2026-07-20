#!/usr/bin/env python3
"""Fine-tune SCALNet on the ShanghaiTech Part A and Part B datasets.

This is a Python 3 training path for the original SCALNet repository.  It reads
ShanghaiTech's ``GT_IMG_*.mat`` point annotations directly and writes HDF5
weights that can be loaded by the project's existing ``SCALNetAdapter``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


SCALNET_ROOT = Path(__file__).resolve().parent
SRC_DIR = SCALNET_ROOT / "src"
for module_path in (SCALNET_ROOT, SRC_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from models.DLANet import DLANet  # noqa: E402
from src.loc_loss import LocLoss  # noqa: E402
from src.network import load_net  # noqa: E402


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def annotation_points(mat_path: Path) -> np.ndarray:
    data = sio.loadmat(mat_path)
    try:
        points = data["image_info"][0, 0]["location"][0, 0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unsupported ShanghaiTech annotation: {mat_path}") from exc
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def discover_samples(dataset_root: Path, split: str, parts: tuple[str, ...]):
    samples: list[tuple[Path, Path, str]] = []
    for part in parts:
        subset = dataset_root / f"part_{part}_final" / f"{split}_data"
        image_dir = subset / "images"
        label_dir = subset / "ground_truth"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing ShanghaiTech subset: {subset}")
        for image_path in sorted(image_dir.glob("IMG_*.jpg"), key=natural_key):
            label_path = label_dir / f"GT_{image_path.stem}.mat"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing annotation for {image_path.name}: {label_path}")
            samples.append((image_path, label_path, part))
    return samples


class ShanghaiTechDataset(Dataset):
    def __init__(
        self,
        samples,
        crop_size: int,
        training: bool,
        sigma: float,
        seed: int,
    ) -> None:
        self.samples = list(samples)
        self.crop_size = crop_size
        self.training = training
        self.sigma = sigma
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, image: Image.Image, points: np.ndarray, index: int):
        width, height = image.size
        scale = max(self.crop_size / width, self.crop_size / height, 1.0)
        if scale > 1.0:
            width, height = math.ceil(width * scale), math.ceil(height * scale)
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            points = points * scale

        if self.training:
            # A per-sample RNG keeps worker behavior reproducible while changing
            # crops between epochs through the DataLoader's shuffled order.
            rng = random.Random(self.seed + index + random.randrange(1 << 20))
            left = rng.randint(0, width - self.crop_size)
            top = rng.randint(0, height - self.crop_size)
            flip = rng.random() < 0.5
        else:
            left = max((width - self.crop_size) // 2, 0)
            top = max((height - self.crop_size) // 2, 0)
            flip = False

        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        inside = (
            (points[:, 0] >= left) & (points[:, 0] < left + self.crop_size)
            & (points[:, 1] >= top) & (points[:, 1] < top + self.crop_size)
        )
        points = points[inside].copy()
        points[:, 0] -= left
        points[:, 1] -= top
        if flip:
            image = TF.hflip(image)
            points[:, 0] = self.crop_size - 1 - points[:, 0]
        return image, points

    def _targets(self, points: np.ndarray) -> torch.Tensor:
        size = self.crop_size
        impulses = np.zeros((size, size), dtype=np.float32)
        for x, y in points:
            ix = int(np.clip(round(float(x)), 0, size - 1))
            iy = int(np.clip(round(float(y)), 0, size - 1))
            impulses[iy, ix] += 1.0

        density = gaussian_filter(impulses, sigma=self.sigma, mode="constant")
        if density.sum() > 0:
            density *= float(len(points)) / float(density.sum())

        # SCALNet's localization branch expects isolated Gaussian peaks whose
        # centers equal one, while its density branch must integrate to count.
        heatmap = gaussian_filter((impulses > 0).astype(np.float32),
                                  sigma=self.sigma, mode="constant")
        peak = float(heatmap.max())
        if peak > 0:
            heatmap /= peak
        return torch.from_numpy(np.stack((heatmap, density)).astype(np.float32))

    def __getitem__(self, index: int):
        image_path, label_path, part = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        points = annotation_points(label_path)
        image, points = self._crop(image, points, index)
        tensor = TF.normalize(TF.to_tensor(image), IMAGE_MEAN, IMAGE_STD)
        return tensor, self._targets(points), float(len(points)), image_path.name, part


def save_adapter_checkpoint(path: Path, model: torch.nn.Module) -> None:
    """Write the key prefix expected by SCALNetAdapter/load_net."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in model.state_dict().items():
            handle.create_dataset(f"model.module.{key}", data=value.detach().cpu().numpy())


def evaluate(model, loader, device, max_batches: int | None = None) -> float:
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for batch_index, (images, _targets, counts, _names, _parts) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            _heatmaps, density_maps = model(images.to(device))
            predictions = density_maps.clamp_min(0).sum(dim=(1, 2, 3)).cpu()
            errors.extend(torch.abs(predictions - counts).tolist())
    return float(np.mean(errors)) if errors else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=SCALNET_ROOT.parent / "ShanghaiTech")
    parser.add_argument("--parts", nargs="+", choices=("A", "B"), default=("A", "B"))
    parser.add_argument("--checkpoint", type=Path,
                        default=SCALNET_ROOT / "checkpoints" / "model.pth")
    parser.add_argument("--output-dir", type=Path,
                        default=SCALNET_ROOT / "outputs" / "shanghaitech_scalnet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only SCALNet's localization and density heads")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Debug/smoke-test limit; omit for full training")
    parser.add_argument("--max-val-samples", type=int, default=128,
                        help="Validation crop limit per epoch; 0 uses the full test split")
    parser.add_argument("--seed", type=int, default=2468)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 32:
        raise ValueError("epochs/batch-size must be positive and crop-size must be at least 32")
    if args.crop_size % 32:
        raise ValueError("crop-size must be divisible by 32 for SCALNet")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    parts = tuple(args.parts)
    train_samples = discover_samples(args.dataset_root, "train", parts)
    val_samples = discover_samples(args.dataset_root, "test", parts)
    if args.max_train_samples is not None:
        train_samples = train_samples[:args.max_train_samples]
    if args.max_val_samples:
        val_samples = val_samples[:args.max_val_samples]

    train_data = ShanghaiTechDataset(train_samples, args.crop_size, True, args.sigma, args.seed)
    val_data = ShanghaiTechDataset(val_samples, args.crop_size, False, args.sigma, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_data)} (parts {', '.join(parts)})")
    print(f"Validation samples: {len(val_data)}")
    model = DLANet()
    load_net(str(args.checkpoint), model, prefix="model.module.")
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for head in (model.dla.hm, model.dla.dm):
            for parameter in head.parameters():
                parameter.requires_grad = True
    model.to(device)

    criterion = LocLoss().to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_mae = float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, targets, _counts, _names, _parts) in enumerate(train_loader, 1):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            heatmaps, density_maps = model(images)
            localization_loss, density_loss = criterion(heatmaps, density_maps, targets)
            loss = localization_loss + density_loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            if step == 1 or step % 25 == 0 or step == len(train_loader):
                print(f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                      f"loss={float(loss.detach()):.6f}", flush=True)

        train_loss = running_loss / max(len(train_loader), 1)
        val_mae = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_crop_mae": val_mae}
        history.append(record)
        save_adapter_checkpoint(args.output_dir / "latest.h5", model)
        if val_mae < best_mae:
            best_mae = val_mae
            save_adapter_checkpoint(args.output_dir / "best.h5", model)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch} train_loss={train_loss:.6f} validation_crop_mae={val_mae:.3f}")

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "parts": list(parts),
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "epochs": args.epochs,
        "device": str(device),
        "freeze_backbone": args.freeze_backbone,
        "best_validation_crop_mae": best_mae,
        "elapsed_seconds": time.time() - started,
        "best_checkpoint": str((args.output_dir / "best.h5").resolve()),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
