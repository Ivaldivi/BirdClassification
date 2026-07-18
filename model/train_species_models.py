"""Train per-family species classifiers for hierarchical prediction."""

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_training import gather_labels, train_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIES_DATA_DIR = PROJECT_ROOT / "data" / "species"
SPECIES_CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints" / "species"

MIN_SPECIES_COUNT = 2
MIN_IMAGES_PER_SPECIES = 10
MIN_TOTAL_IMAGES = 40


def family_slug(family_name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", family_name.strip().lower())
    return slug.strip("_") or "unknown"


def count_images(folder: Path) -> int:
    return len(list(folder.glob("*.jpg")))


def should_train_family(family_dir: Path) -> tuple[bool, str]:
    species_dirs = [entry for entry in family_dir.iterdir() if entry.is_dir()]
    if len(species_dirs) < MIN_SPECIES_COUNT:
        return False, f"only {len(species_dirs)} species folders"

    qualifying_species = [
        species_dir for species_dir in species_dirs if count_images(species_dir) >= MIN_IMAGES_PER_SPECIES
    ]
    total_images = sum(count_images(species_dir) for species_dir in species_dirs)

    if len(qualifying_species) < MIN_SPECIES_COUNT:
        return False, "not enough species with minimum image counts"
    if total_images < MIN_TOTAL_IMAGES:
        return False, f"only {total_images} total images"
    return True, f"{len(qualifying_species)} species, {total_images} images"


def train_species_models(
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> list[Path]:
    saved_paths = []
    SPECIES_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for family_dir in sorted(SPECIES_DATA_DIR.iterdir()):
        if not family_dir.is_dir():
            continue

        family_name = family_dir.name
        should_train, reason = should_train_family(family_dir)
        if not should_train:
            print(f"Skipping {family_name}: {reason}", flush=True)
            continue

        checkpoint_path = SPECIES_CHECKPOINT_DIR / f"{family_slug(family_name)}.pth"
        print(f"Training species model for {family_name} ({reason})", flush=True)
        train_model(
            data_dir=family_dir,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["parent_family"] = family_name
        torch.save(checkpoint, checkpoint_path)
        saved_paths.append(checkpoint_path)

    print(f"Trained {len(saved_paths)} species models", flush=True)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train species models for each bird family.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if not SPECIES_DATA_DIR.exists():
        raise FileNotFoundError(
            f"No species data found at {SPECIES_DATA_DIR}. "
            "Run the iNaturalist query/download pipeline first."
        )

    families = gather_labels(SPECIES_DATA_DIR)
    print(f"Found {len(families)} family folders under {SPECIES_DATA_DIR}")
    train_species_models(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)


if __name__ == "__main__":
    main()
