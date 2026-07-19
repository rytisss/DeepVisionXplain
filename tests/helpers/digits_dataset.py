"""Test dataset generation from the scikit-learn `digits` open dataset.

Creates the torchvision `ImageFolder` layout that `ClassificationDataModule`
consumes (`<split>/<class_name>/*.png`). The digits dataset (1797 real 8x8
handwritten-digit images, 10 classes) ships inside the scikit-learn package,
so tests use real, class-separable images while remaining fully offline and
deterministic.
"""

from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import load_digits


def generate_classification_dataset(
    root: Path,
    num_classes: int = 2,
    images_per_class: int = 6,
    image_size: int = 224,
    channels: int = 3,
) -> dict[str, str]:
    """Generates train/val/test ImageFolder splits from the digits dataset.

    Digit `k` becomes class `k`; each split takes a disjoint slice of that
    digit's samples, upscaled to the requested image size.

    Args:
        root (Path): Directory to create the dataset in.
        num_classes (int, optional): Number of class folders (max 10). Defaults to 2.
        images_per_class (int, optional): Images per class per split. Defaults to 6.
        image_size (int, optional): Square image size in pixels. Defaults to 224.
        channels (int, optional): 1 saves single-channel grayscale PNGs, 3 saves
        3-channel PNGs. Defaults to 3.

    Returns:
        dict[str, str]: Mapping of split name ('train'/'val'/'test') to its directory path.
    """
    if channels not in (1, 3):
        raise ValueError(f'channels must be 1 or 3, got {channels}')

    digits = load_digits()
    splits = {}
    for split_index, split in enumerate(('train', 'val', 'test')):
        for cls in range(num_classes):
            cls_dir = root / split / str(cls)
            cls_dir.mkdir(parents=True, exist_ok=True)
            class_images = digits.images[digits.target == cls]
            start = split_index * images_per_class
            for i, image in enumerate(class_images[start : start + images_per_class]):
                image = (image / 16.0 * 255.0).astype(np.uint8)  # digits pixels are 0..16
                image = cv2.resize(
                    image, (image_size, image_size), interpolation=cv2.INTER_NEAREST
                )
                if channels == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(cls_dir / f'img_{i:03d}.png'), image)
        splits[split] = str(root / split)
    return splits
