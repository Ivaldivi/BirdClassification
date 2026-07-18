import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAMILIES_DIR = DATA_DIR / "families"
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
FAMILY_CHECKPOINT_PATH = CHECKPOINT_DIR / "family_classifier.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / "bird_classifier.pth"

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def gather_labels(data_directory: Path) -> list[str]:
    """Return folder names under the data directory (one folder per bird class)."""
    labels = sorted(
        entry.name for entry in data_directory.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    print(f"There are {len(labels)} labels")
    return labels


def create_data_loaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dataset = datasets.ImageFolder(root=str(data_dir), transform=TRAIN_TRANSFORM)
    eval_dataset = datasets.ImageFolder(root=str(data_dir), transform=EVAL_TRANSFORM)
    class_names = train_dataset.classes

    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    train_split, val_split, test_split = random_split(
        range(len(train_dataset)), [train_size, val_size, test_size], generator=generator
    )

    num_workers = 0
    train_loader = DataLoader(
        Subset(train_dataset, train_split.indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_split.indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_split.indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, class_names


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(
    data_dir: Path = DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> Path:
    device = get_device()
    print(f"Training on {device}", flush=True)

    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=data_dir, batch_size=batch_size
    )
    num_classes = len(class_names)
    print(
        f"Dataset: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / "
        f"{len(test_loader.dataset)} test images across {num_classes} classes",
        flush=True,
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "val_accuracy": val_acc,
                    "data_dir": str(data_dir),
                },
                checkpoint_path,
            )
            print(f"Saved new best model to {checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}", flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bird image classifier.")
    parser.add_argument("--data-dir", type=Path, default=FAMILIES_DIR, help="Image folder root.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=FAMILY_CHECKPOINT_PATH,
        help="Where to save the trained model.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}. "
            "Run the iNaturalist query/download pipeline first."
        )

    labels = gather_labels(args.data_dir)
    print(f"Classes: {labels[:5]} ... ({len(labels)} total)")
    train_model(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
