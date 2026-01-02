import argparse
import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default="./dataset",
                        help="Path to dataset root (with class folders)")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")

    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")

    parser.add_argument("--lr", type=float, default=8e-6,
                        help="Learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")

    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")

    return parser.parse_args()


# Main
def main():
    args = parse_args()
    set_seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs

    # Transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Datasets and Dataloaders
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.2 * total_size)

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size]
    )

    val_ds.dataset.transform = eval_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=args.num_workers)

    print("Class mapping:", full_dataset.class_to_idx)
    print(f"Train: {train_size}, Val: {val_size}")


    # Pos Weight
    targets = [label for _, label in full_dataset.samples]
    num_neg = targets.count(0)
    num_pos = targets.count(1)

    pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)

    print(f"Pos_weight: {pos_weight.item():.4f}")

    # Model
    model = models.resnet18(pretrained=True)

    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last 2 blocks + head
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    # Training loop
    best_val_loss = np.inf

    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # VALIDATION
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_mri_model.pth")
            print("Model improved and saved.")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
