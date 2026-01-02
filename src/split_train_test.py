import os
import shutil
import random
from pathlib import Path


def split_train_test(
    dataset_dir,
    test_ratio=0.15,
    seed=42,
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tiff")
):
    random.seed(seed)

    dataset_dir = Path(dataset_dir)

    class_dirs = [dataset_dir / "0", dataset_dir / "1"]
    for cdir in class_dirs:
        if not cdir.exists():
            raise FileNotFoundError(f"Missing class folder: {cdir}")

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    # Create train/test structure
    for split_dir in [train_dir, test_dir]:
        for cls in ["0", "1"]:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    # Move everything into train first
    for cls in ["0", "1"]:
        src_dir = dataset_dir / cls
        dst_dir = train_dir / cls

        for file in src_dir.iterdir():
            if file.is_file():
                shutil.move(str(file), str(dst_dir / file.name))

        # Remove empty original folder
        src_dir.rmdir()

    # Now split train -> test
    for cls in ["0", "1"]:
        cls_train_dir = train_dir / cls
        cls_test_dir = test_dir / cls

        images = [
            f for f in cls_train_dir.iterdir()
            if f.suffix.lower() in extensions
        ]

        n_test = int(len(images) * test_ratio)
        test_samples = random.sample(images, n_test)

        for img in test_samples:
            shutil.move(str(img), str(cls_test_dir / img.name))

        print(
            f"Class {cls}: "
            f"{len(images) - n_test} train, {n_test} test"
        )

    print("\n Train/test split completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset containing 0 and 1 folders"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction of data to move to test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    split_train_test(
        dataset_dir=args.dataset_dir,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
