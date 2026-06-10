"""Cache LDAE latents for all preprocessed NIfTI scans.

Encodes every .nii / .nii.gz file in the input folder through the frozen
LDAE AutoencoderKL and saves the latent mu as a .pt file in the output folder.

Output filename: <Image Data ID>.pt
Each .pt file contains a tensor of shape (3, 16, 20, 16) — the scaled latent mu.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import _load_nii
from vae import BrainVAE


class NiftiFolder(Dataset):
    """All .nii / .nii.gz files under data_dir, skipping already-cached ones."""

    def __init__(self, data_dir: Path, output_dir: Path):
        all_files = sorted(
            list(data_dir.glob("**/*.nii")) +
            list(data_dir.glob("**/*.nii.gz"))
        )
        # Skip files whose .pt already exists in output_dir
        self.files = [
            f for f in all_files
            if not (output_dir / f"{f.name.split('.')[0]}.pt").exists()
        ]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        try:
            x = _load_nii(path)          # (1, 128, 160, 128)
            valid = True
        except Exception as e:
            print(f"[Error] {path.name}: {e}")
            x = torch.zeros(1, 128, 160, 128)
            valid = False
        return x, str(path), valid


def collate(batch):
    xs, paths, valids = zip(*batch)
    return torch.stack(xs), list(paths), list(valids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache LDAE latents to disk")
    p.add_argument("--data_dir",           required=True,  help="Folder with preprocessed .nii files")
    p.add_argument("--output_dir",         required=True,  help="Where to save .pt latent files")
    p.add_argument("--pretrained_weights", required=True,  help="LDAE AutoencoderKL checkpoint")
    p.add_argument("--batch_size",         type=int, default=4)
    p.add_argument("--num_workers",        type=int, default=4)
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = NiftiFolder(Path(args.data_dir), output_dir)
    print(f"Found {len(ds)} scans to encode (skipping already cached)")

    if len(ds) == 0:
        print("Nothing to do.")
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=(args.device == "cuda"))

    vae = BrainVAE(pretrained_weights=args.pretrained_weights, freeze_encoder=False).to(device)
    vae.eval()

    total = len(ds)
    done = 0

    with torch.no_grad():
        for xs, paths, valids in loader:
            xs = xs.to(device)
            mu, _ = vae.encode(xs)
            mu = mu * vae.scale_factor   # (B, 3, 16, 20, 16)

            for i, (path, valid) in enumerate(zip(paths, valids)):
                if not valid:
                    done += 1
                    continue
                image_id = Path(path).name.split(".")[0]
                out_path = output_dir / f"{image_id}.pt"
                torch.save(mu[i].cpu(), out_path)
                done += 1

            print(f"  {done}/{total}", end="\r", flush=True)

    print(f"\nDone. Latents saved to {output_dir}")


if __name__ == "__main__":
    main()
