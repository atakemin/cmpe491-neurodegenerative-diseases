import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MRI classification model")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory (ImageFolder format)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to .pth model weights"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu"
    )

    return parser.parse_args()


def build_model(device):
    model = models.resnet18(pretrained=False)

    # 1-channel input
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )

    model.to(device)
    model.eval()
    return model


def main():
    args = get_args()

    print("Using device:", args.device)

    # Validation / test transforms 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    print("Class mapping:", dataset.class_to_idx)
    print("Number of samples:", len(dataset))

    # Model
    model = build_model(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).int()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== Evaluation Results =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
