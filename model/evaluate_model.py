"""Analyze bird classifier performance on the held-out test set."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import DEFAULT_CHECKPOINT, get_device, load_model
from model_training import CHECKPOINT_PATH, create_data_loaders

REPORT_DIR = Path(__file__).resolve().parent / "reports"


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference on a dataloader and return labels, preds, probabilities, paths."""
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    image_paths = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
        y_prob.append(probabilities)

    dataset = loader.dataset
    for index in dataset.indices:
        image_paths.append(dataset.dataset.samples[index][0])

    return (
        np.array(y_true),
        np.array(y_pred),
        np.concatenate(y_prob, axis=0),
        image_paths,
    )


def summarize_confidence(y_true, y_pred, y_prob):
    correct_mask = y_true == y_pred
    top_confidences = y_prob.max(axis=1)

    return {
        "avg_confidence_correct": float(top_confidences[correct_mask].mean()) if correct_mask.any() else 0.0,
        "avg_confidence_incorrect": float(top_confidences[~correct_mask].mean()) if (~correct_mask).any() else 0.0,
        "pct_predictions_above_75pct_confidence": float((top_confidences >= 0.75).mean()),
        "accuracy_when_above_75pct_confidence": float(
            correct_mask[top_confidences >= 0.75].mean()
        ) if (top_confidences >= 0.75).any() else 0.0,
        "accuracy_when_below_75pct_confidence": float(
            correct_mask[top_confidences < 0.75].mean()
        ) if (top_confidences < 0.75).any() else 0.0,
    }


def per_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    class_rows = []
    for class_name in class_names:
        metrics = report[class_name]
        class_rows.append(
            {
                "class": class_name,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1-score"], 4),
                "support": int(metrics["support"]),
            }
        )

    class_rows.sort(key=lambda row: row["f1_score"])
    return class_rows, report


def worst_classes(class_rows, count=10):
    return class_rows[:count]


def best_classes(class_rows, count=10):
    return list(reversed(class_rows[-count:]))


def sample_misclassifications(y_true, y_pred, y_prob, image_paths, class_names, limit=15):
    mistakes = []
    for true_idx, pred_idx, probs, path in zip(y_true, y_pred, y_prob, image_paths):
        if true_idx == pred_idx:
            continue
        mistakes.append(
            {
                "image_path": path,
                "true_class": class_names[true_idx],
                "predicted_class": class_names[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
            }
        )

    mistakes.sort(key=lambda row: row["confidence"], reverse=True)
    return mistakes[:limit]


def build_report(checkpoint_path: Path, batch_size: int = 32) -> dict:
    device = get_device()
    model, class_names, _ = load_model(checkpoint_path)
    _, _, test_loader, _ = create_data_loaders(batch_size=batch_size)

    y_true, y_pred, y_prob, image_paths = collect_predictions(model, test_loader, device)

    class_rows, sklearn_report = per_class_metrics(y_true, y_pred, class_names)
    confidence_summary = summarize_confidence(y_true, y_pred, y_prob)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "num_test_images": int(len(y_true)),
        "num_classes": len(class_names),
        "overall": {
            "top1_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "top3_accuracy": round(float(top_k_accuracy_score(y_true, y_prob, k=3, labels=range(len(class_names)))), 4),
            "top5_accuracy": round(float(top_k_accuracy_score(y_true, y_prob, k=5, labels=range(len(class_names)))), 4),
            "macro_f1": round(float(sklearn_report["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(sklearn_report["weighted avg"]["f1-score"]), 4),
        },
        "confidence_summary": confidence_summary,
        "worst_classes_by_f1": worst_classes(class_rows),
        "best_classes_by_f1": best_classes(class_rows),
        "sample_misclassifications": sample_misclassifications(
            y_true, y_pred, y_prob, image_paths, class_names
        ),
        "per_class_metrics": class_rows,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return report


def print_report_summary(report: dict) -> None:
    overall = report["overall"]
    confidence = report["confidence_summary"]

    print("Bird Classifier Performance Report")
    print("=" * 40)
    print(f"Test images: {report['num_test_images']}")
    print(f"Classes: {report['num_classes']}")
    print()
    print("Overall accuracy")
    print(f"  Top-1: {overall['top1_accuracy'] * 100:.2f}%")
    print(f"  Top-3: {overall['top3_accuracy'] * 100:.2f}%")
    print(f"  Top-5: {overall['top5_accuracy'] * 100:.2f}%")
    print(f"  Macro F1: {overall['macro_f1']:.4f}")
    print(f"  Weighted F1: {overall['weighted_f1']:.4f}")
    print()
    print("Confidence behavior")
    print(f"  Avg confidence when correct: {confidence['avg_confidence_correct'] * 100:.2f}%")
    print(f"  Avg confidence when incorrect: {confidence['avg_confidence_incorrect'] * 100:.2f}%")
    print(f"  Predictions >= 75% confidence: {confidence['pct_predictions_above_75pct_confidence'] * 100:.2f}%")
    print(
        "  Accuracy when >= 75% confidence: "
        f"{confidence['accuracy_when_above_75pct_confidence'] * 100:.2f}%"
    )
    print(
        "  Accuracy when < 75% confidence: "
        f"{confidence['accuracy_when_below_75pct_confidence'] * 100:.2f}%"
    )
    print()
    print("Worst classes by F1")
    for row in report["worst_classes_by_f1"]:
        print(
            f"  {row['class']}: "
            f"F1={row['f1_score']:.3f}, recall={row['recall']:.3f}, support={row['support']}"
        )
    print()
    print("Best classes by F1")
    for row in report["best_classes_by_f1"]:
        print(
            f"  {row['class']}: "
            f"F1={row['f1_score']:.3f}, recall={row['recall']:.3f}, support={row['support']}"
        )
    print()
    print("Sample misclassifications")
    for row in report["sample_misclassifications"][:10]:
        print(
            f"  true={row['true_class']} | pred={row['predicted_class']} "
            f"({row['confidence'] * 100:.1f}%) | {row['image_path']}"
        )


def save_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"performance_report_{timestamp}.json"
    text_path = output_dir / f"performance_report_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    with text_path.open("w", encoding="utf-8") as file:
        overall = report["overall"]
        confidence = report["confidence_summary"]
        file.write("Bird Classifier Performance Report\n")
        file.write("=" * 40 + "\n")
        file.write(f"Generated: {report['generated_at']}\n")
        file.write(f"Checkpoint: {report['checkpoint']}\n")
        file.write(f"Test images: {report['num_test_images']}\n\n")
        file.write("Overall accuracy\n")
        file.write(f"Top-1: {overall['top1_accuracy'] * 100:.2f}%\n")
        file.write(f"Top-3: {overall['top3_accuracy'] * 100:.2f}%\n")
        file.write(f"Top-5: {overall['top5_accuracy'] * 100:.2f}%\n")
        file.write(f"Macro F1: {overall['macro_f1']:.4f}\n")
        file.write(f"Weighted F1: {overall['weighted_f1']:.4f}\n\n")
        file.write("Confidence behavior\n")
        for key, value in confidence.items():
            file.write(f"{key}: {value}\n")
        file.write("\nPer-class metrics\n")
        for row in report["per_class_metrics"]:
            file.write(
                f"{row['class']}: precision={row['precision']}, recall={row['recall']}, "
                f"f1={row['f1_score']}, support={row['support']}\n"
            )

    return json_path, text_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bird classifier performance.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR,
        help="Directory where JSON and text reports will be saved.",
    )
    args = parser.parse_args()

    report = build_report(args.checkpoint, batch_size=args.batch_size)
    print_report_summary(report)
    json_path, text_path = save_report(report, args.output_dir)
    print()
    print(f"Saved JSON report to {json_path}")
    print(f"Saved text report to {text_path}")


if __name__ == "__main__":
    main()
