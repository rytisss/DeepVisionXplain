"""Preview training-time augmentations without running a full training session.

Loads the same Hydra experiment config `train.py` would use, instantiates the
*real* `data.train_transforms` pipeline (the exact objects the DataLoader will
call every epoch), and applies it to a handful of real tiles pulled straight
from the training data directory. Saves side-by-side comparison grids -
original tile next to individually-labeled augmentations (flip, rotation,
exposure) plus a few fully-random combined draws - to their own output
folder, so an augmentation config can be sanity-checked before committing to
a big run.

This script lives in scripts/, which is baked into the Docker image rather
than bind-mounted like configs/ - any further edits need a new image
build/push before they take effect inside the container.

Run inside the running container, e.g.:

    docker compose -f docker-compose.prod.yaml exec deepvisionxplain \
        python scripts/preview_augmentations.py \
        experiment=train_cnn_multi_seats_aug \
        data.train_data_dir=/app/data/<dataset>/train \
        --num-samples 6 --num-random 3

Any extra Hydra override not already present in the config schema needs a
leading '+' (e.g. `+foo=bar`), same as with train.py.

Output goes under <log_dir>/augmentation_previews/<experiment>_<timestamp>/,
i.e. inside the mounted logs folder, so the PNGs show up on the host too.
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import ListConfig
from PIL import Image
from torchvision.transforms import functional as TF

rootutils.setup_root(__file__, indicator=['pyproject.toml'], pythonpath=True)

from src.utils import RankedLogger  # noqa: E402

# This script calls the Hydra Compose API directly (not @hydra.main), so none of the
# logging setup from configs/hydra/default.yaml runs - configure a plain console
# handler ourselves, otherwise RankedLogger.info() calls print nothing.
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = RankedLogger(__name__, rank_zero_only=True)

IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp')


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Split our own preview flags from the Hydra overrides passed on the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--num-samples', type=int, default=6, help='Source tiles to preview (spread across classes).'
    )
    parser.add_argument(
        '--num-random',
        type=int,
        default=3,
        help='Fully-random augmented variants per tile, drawn from the real training Compose.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Where to save previews. Defaults to <log_dir>/augmentation_previews/<experiment>_<timestamp>.',
    )
    parser.add_argument('--seed', type=int, default=42)
    args, overrides = parser.parse_known_args()
    return args, overrides


def load_cfg(overrides: list[str]):
    """Compose the same train.yaml config train.py would, from the project's configs/ dir."""
    GlobalHydra.instance().clear()
    # This script lives in scripts/, one level below the project root, alongside configs/ -
    # resolve the sibling directory rather than assuming configs/ is next to this file.
    config_dir = str((Path(__file__).parent.parent / 'configs').resolve())
    with initialize_config_dir(version_base='1.3', config_dir=config_dir):
        cfg = compose(config_name='train.yaml', overrides=overrides)
    return cfg


def find_rotation_params(cfg, default_degrees: float = 10.0, default_fill=0) -> tuple[float, Any]:
    for t in cfg.data.train_transforms.transforms:
        if t._target_.endswith('RandomRotation'):
            degrees = t.degrees
            degrees = degrees[1] if isinstance(degrees, (list, tuple, ListConfig)) else degrees
            fill = t.get('fill', default_fill)
            fill = tuple(fill) if isinstance(fill, (list, tuple, ListConfig)) else fill
            return degrees, fill
    return default_degrees, default_fill


def find_color_jitter(cfg) -> tuple[float, float]:
    for t in cfg.data.train_transforms.transforms:
        if t._target_.endswith('ColorJitter'):
            return t.get('brightness', 0.0), t.get('contrast', 0.0)
    return 0.0, 0.0


def gather_samples(train_data_dir: str, num_samples: int, seed: int) -> list[tuple[str, Path]]:
    """Pick a handful of real tiles, spread across class folders."""
    train_data_dir = Path(train_data_dir)
    class_dirs = sorted(d for d in train_data_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f'No class subfolders found under {train_data_dir}')

    rng = random.Random(seed)
    per_class = max(1, num_samples // len(class_dirs))
    samples: list[tuple[str, Path]] = []
    for class_dir in class_dirs:
        images_dir = class_dir / 'images' if (class_dir / 'images').is_dir() else class_dir
        image_paths: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(images_dir.glob(ext))
        image_paths.sort()
        if not image_paths:
            continue
        chosen = rng.sample(image_paths, min(per_class, len(image_paths)))
        samples.extend((class_dir.name, p) for p in chosen)
    return samples[:num_samples] if num_samples else samples


def build_illustrative_variants(
    image: Image.Image,
    image_size: tuple[int, int],
    rotation_degrees: float,
    rotation_fill: Any,
    brightness: float,
    contrast: float,
) -> dict[str, Image.Image]:
    """Deterministic, individually-labeled variants so each augmentation's effect is
    obvious on its own, rather than only ever seeing random combined draws."""
    resized = TF.resize(image, list(image_size))
    variants: dict[str, Image.Image] = {'Original (resized)': resized}
    variants['Horizontal flip'] = TF.hflip(resized)
    variants['Vertical flip'] = TF.vflip(resized)
    variants[f'Rotate +{rotation_degrees:g} deg'] = TF.rotate(resized, rotation_degrees, fill=rotation_fill)
    variants[f'Rotate -{rotation_degrees:g} deg'] = TF.rotate(resized, -rotation_degrees, fill=rotation_fill)
    if brightness:
        variants[f'Brighter (+{brightness * 100:g}%)'] = TF.adjust_brightness(resized, 1 + brightness)
        variants[f'Darker (-{brightness * 100:g}%)'] = TF.adjust_brightness(resized, max(0.0, 1 - brightness))
    if contrast:
        variants[f'Contrast +{contrast * 100:g}%'] = TF.adjust_contrast(resized, 1 + contrast)
        variants[f'Contrast -{contrast * 100:g}%'] = TF.adjust_contrast(resized, max(0.0, 1 - contrast))
    return variants


def to_display_array(img) -> np.ndarray:
    if torch.is_tensor(img):
        return img.clamp(0, 1).permute(1, 2, 0).numpy()
    return np.asarray(img)


def main() -> None:
    args, overrides = parse_args()
    cfg = load_cfg(overrides)

    experiment_name = next(
        (o.split('=', 1)[1] for o in overrides if o.startswith('experiment=')), 'default'
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(cfg.paths.log_dir)
        / 'augmentation_previews'
        / f'{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rotation_degrees, rotation_fill = find_rotation_params(cfg)
    brightness, contrast = find_color_jitter(cfg)

    log.info(f'Instantiating datamodule <{cfg.data._target_}> to reuse its real train_transforms...')
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='fit')
    real_train_transform = datamodule.data_train.transform

    samples = gather_samples(cfg.data.train_data_dir, args.num_samples, args.seed)
    log.info(f'Previewing {len(samples)} tile(s) from {cfg.data.train_data_dir}')

    for idx, (class_name, image_path) in enumerate(samples):
        image = Image.open(image_path).convert('RGB')
        variants = build_illustrative_variants(
            image, tuple(cfg.data.image_size), rotation_degrees, rotation_fill, brightness, contrast
        )

        for r in range(args.num_random):
            seed_r = args.seed + idx * 100 + r
            torch.manual_seed(seed_r)
            variants[f'Random (seed={seed_r})'] = real_train_transform(image)

        n = len(variants)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        for ax, (title, img) in zip(axes, variants.items()):
            ax.imshow(to_display_array(img))
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        fig.suptitle(f'class={class_name}  |  {image_path.name}', fontsize=10)
        plt.tight_layout()

        out_path = output_dir / f'{idx:02d}_class{class_name}_{image_path.stem}.png'
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        log.info(f'Saved {out_path}')

    log.info(f'Done. {len(samples)} preview grid(s) written to {output_dir}')


if __name__ == '__main__':
    main()
