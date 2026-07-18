"""Load trained bird classifiers and run hierarchical family -> species predictions."""

import re
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
FAMILY_CHECKPOINT = CHECKPOINT_DIR / "family_classifier.pth"
SPECIES_CHECKPOINT_DIR = CHECKPOINT_DIR / "species"
LEGACY_CHECKPOINT = CHECKPOINT_DIR / "bird_classifier.pth"
CONFIDENCE_THRESHOLD = 0.75
SPECIES_CONFIDENCE_THRESHOLD = 0.60
MIN_FAMILY_CONFIDENCE_FOR_SPECIES = 0.50

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def family_slug(family_name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", family_name.strip().lower())
    return slug.strip("_") or "unknown"


def load_oriented_image(source) -> Image.Image:
    """Open an image and apply EXIF orientation so it displays and predicts upright."""
    image = Image.open(source)
    return ImageOps.exif_transpose(image)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def default_checkpoint() -> Path:
    if FAMILY_CHECKPOINT.exists():
        return FAMILY_CHECKPOINT
    return LEGACY_CHECKPOINT


def load_model(checkpoint_path: Path | str | None = None) -> tuple[nn.Module, list[str], torch.device]:
    checkpoint_path = Path(checkpoint_path or default_checkpoint())
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {checkpoint_path}. "
            "Run `python model/model_training.py --data-dir data/families` first."
        )

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names: list[str] = checkpoint["class_names"]
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, device


def predict(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return up to top_k (class_name, confidence) pairs sorted by confidence."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(top_k, len(class_names))
    confidences, indices = torch.topk(probabilities, top_k)
    return [(class_names[idx], conf.item()) for conf, idx in zip(confidences, indices)]


def species_checkpoint_for_family(family_name: str) -> Path | None:
    checkpoint_path = SPECIES_CHECKPOINT_DIR / f"{family_slug(family_name)}.pth"
    return checkpoint_path if checkpoint_path.exists() else None


def format_level_prediction(
    predictions: list[tuple[str, float]],
    label: str,
    threshold: float,
) -> dict:
    if not predictions:
        return {
            "label": label,
            "confident": False,
            "message": f"No {label.lower()} predictions available.",
            "options": [],
        }

    top_name, top_conf = predictions[0]
    if top_conf >= threshold:
        return {
            "label": label,
            "confident": True,
            "message": f"**{label}:** {top_name} ({top_conf * 100:.1f}% confident)",
            "top_prediction": top_name,
            "top_confidence": top_conf,
            "options": predictions,
        }

    return {
        "label": label,
        "confident": False,
        "message": (
            f"**{label}:** less than {threshold * 100:.0f}% confident in one answer. "
            f"Most likely options:"
        ),
        "top_prediction": top_name,
        "top_confidence": top_conf,
        "options": predictions,
    }


def predict_hierarchical(
    image: Image.Image,
    family_model: nn.Module,
    family_class_names: list[str],
    device: torch.device,
    family_threshold: float = CONFIDENCE_THRESHOLD,
    species_threshold: float = SPECIES_CONFIDENCE_THRESHOLD,
    min_family_confidence_for_species: float = MIN_FAMILY_CONFIDENCE_FOR_SPECIES,
) -> dict:
    family_predictions = predict(image, family_model, family_class_names, device, top_k=5)
    family_result = format_level_prediction(family_predictions, "Family", family_threshold)

    species_result = None
    top_family, top_family_conf = family_predictions[0]
    species_checkpoint = species_checkpoint_for_family(top_family)

    if species_checkpoint and top_family_conf >= min_family_confidence_for_species:
        species_model, species_class_names, _ = load_model(species_checkpoint)
        species_predictions = predict(image, species_model, species_class_names, device, top_k=5)
        species_result = format_level_prediction(species_predictions, "Species", species_threshold)
    elif not species_checkpoint:
        species_result = {
            "label": "Species",
            "confident": False,
            "message": f"No species model is available yet for **{top_family}**.",
            "options": [],
        }

    overall_confident = family_result["confident"] and (
        species_result is None or species_result.get("confident", False)
    )

    return {
        "confident": overall_confident,
        "family": family_result,
        "species": species_result,
    }


# Backward-compatible aliases
DEFAULT_CHECKPOINT = default_checkpoint()


def format_prediction(predictions: list[tuple[str, float]], threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    return format_level_prediction(predictions, "Bird group", threshold)
