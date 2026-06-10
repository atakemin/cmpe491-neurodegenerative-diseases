"""Batch inference on the test split.

For every longitudinal pair in the test set, encodes x_prev with the frozen
LDAE AE (or loads a pre-cached latent), runs the diffusion UNet to predict
z_next, decodes with the LDAE decoder, and saves the predicted NIfTI next to
a CSV summarising all pairs.

Usage (NIfTI mode):
  python inference.py \\
      --data_dir /path/to/nii_files \\
      --csv_path /path/to/adni.csv \\
      --pretrained_weights /path/to/autoencoderkl.pth \\
      --diffusion_weights  ./runs/diffusion/diffusion_best.pt \\
      --output_dir         ./inference_results \\
      --seed 42

Usage (latent cache mode — faster, skips VAE encoding):
  python inference.py \\
      --data_dir       /path/to/nii_files \\
      --latent_dir     /path/to/latents \\
      --csv_path       /path/to/adni.csv \\
      --pretrained_weights /path/to/autoencoderkl.pth \\
      --diffusion_weights  ./runs/diffusion/diffusion_best.pt \\
      --output_dir         ./inference_results \\
      --cfg_scale 2.0 \\
      --max_pairs 50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import Stage2Dataset
from diffusion import DiffusionSchedule, UNet3D, ddpm_sample
from vae import BrainVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference on test split")

    p.add_argument("--data_dir",           required=True,  help="Folder containing .nii files")
    p.add_argument("--csv_path",           required=True,  help="ADNI metadata CSV")
    p.add_argument("--pretrained_weights", required=True,  help="LDAE AutoencoderKL checkpoint")
    p.add_argument("--diffusion_weights",  required=True,  help="Trained diffusion UNet weights (.pt)")
    p.add_argument("--output_dir",         default="./inference_results")
    p.add_argument("--seed",               type=int, default=42)

    # Latent cache mode: skip VAE encoding, load z_prev from .pt files
    p.add_argument("--latent_dir",         default=None,
                   help="Folder with pre-cached .pt latents (skips VAE encoding if provided)")

    # Diffusion sampling params
    p.add_argument("--n_samples",          type=int,   default=5,
                   help="DDPM samples to average per pair (more = smoother but slower)")
    p.add_argument("--cfg_scale",          type=float, default=1.0,
                   help="Classifier-free guidance scale (1.0 = no CFG, >1.0 = stronger conditioning)")
    p.add_argument("--diffusion_steps",    type=int,   default=1000)
    p.add_argument("--emb_dim",            type=int,   default=256)

    # How many pairs to infer (stops after N pairs, saves CSV after each one)
    p.add_argument("--max_pairs",          type=int,   default=None,
                   help="Stop after this many pairs (default: all test pairs)")

    p.add_argument("--batch_size",         type=int,   default=1)
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--split_train",        type=float, default=0.80)
    p.add_argument("--split_val",          type=float, default=0.10)
    p.add_argument("--split_test",         type=float, default=0.10)
    p.add_argument("--max_delta_years",    type=float, default=None)
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


@torch.no_grad()
def predict_pair(
    vae: BrainVAE,
    model: UNet3D,
    schedule: DiffusionSchedule,
    x_prev: torch.Tensor,
    cond: dict,
    n_samples: int,
    device: torch.device,
    latent_path: Path | None = None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Returns predicted scan as (C, H, W, D) cpu tensor.

    If latent_path is given, loads z_prev from disk instead of encoding x_prev.
    """
    if latent_path is not None:
        z_prev = torch.load(latent_path, map_location=device, weights_only=True).unsqueeze(0)
    else:
        z_prev = vae.encode(x_prev)[0] * vae.scale_factor

    if n_samples > 1:
        z_prev_rep = z_prev.expand(n_samples, -1, -1, -1, -1)
        cond_rep = {k: v.expand(n_samples, *v.shape[1:]) for k, v in cond.items()}
    else:
        z_prev_rep = z_prev
        cond_rep = cond

    z_pred = ddpm_sample(
        model, schedule, z_prev_rep, cond_rep,
        latent_shape=tuple(z_prev.shape[1:]), device=device,
        cfg_scale=cfg_scale,
    )

    if n_samples > 1:
        z_pred = z_pred.mean(dim=0, keepdim=True)

    x_hat = vae.decode(z_pred)
    return x_hat.squeeze(0).cpu()


def tensor_to_nii(arr: np.ndarray, ref_path: Path) -> nib.Nifti1Image:
    ref = nib.load(str(ref_path))
    if arr.ndim == 4:
        arr = arr[0]
    return nib.Nifti1Image(arr, ref.affine, ref.header)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ratios = (args.split_train, args.split_val, args.split_test)
    use_latents = args.latent_dir is not None
    latent_dir = Path(args.latent_dir) if use_latents else None

    if use_latents:
        print(f"Latent cache mode: z_prev loaded from {latent_dir}")
    if args.cfg_scale > 1.0:
        print(f"CFG scale: {args.cfg_scale}")

    test_ds = Stage2Dataset(
        args.data_dir, args.csv_path,
        split="test",
        split_ratios=split_ratios,
        seed=args.seed,
        max_delta_years=args.max_delta_years,
    )

    n_total = len(test_ds)
    if args.max_pairs is not None:
        n_total = min(n_total, args.max_pairs)

    print(f"Test pairs available: {len(test_ds)}  |  Will infer: {n_total}")

    if len(test_ds) == 0:
        raise RuntimeError("No test pairs found. Check --data_dir, --csv_path and split ratios.")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # VAE (always needed for decoding; encoding skipped in latent mode)
    vae = BrainVAE(pretrained_weights=args.pretrained_weights, freeze_encoder=False).to(device)
    vae.eval()

    model = UNet3D(in_channels=6, out_channels=3, emb_dim=args.emb_dim).to(device)
    state = torch.load(args.diffusion_weights, map_location="cpu")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    schedule = DiffusionSchedule(T=args.diffusion_steps)
    schedule.to(device)

    records = []
    csv_out = output_dir / "inference_results.csv"

    for idx, batch in enumerate(loader):
        if idx >= n_total:
            break

        x_prev = batch["x_prev"].to(device)

        age_prev = batch["age_prev"].item()
        age_next = batch["age_next"].item()
        delta_t  = batch["delta_t"].item()
        sex      = batch["sex"].item()
        group_id = batch["group_id"].item()

        cond = {
            "age_prev": batch["age_prev"].to(device),
            "age_next": batch["age_next"].to(device),
            "delta_t":  batch["delta_t"].to(device),
            "sex":      batch["sex"].to(device),
            "group_id": batch["group_id"].to(device),
        }

        pair = test_ds.pairs[idx]
        path_prev = pair["path_prev"]
        path_next = pair["path_next"]

        # Resolve cached latent path from image_id (stem of path_prev without extension)
        latent_path = None
        if use_latents:
            image_id = Path(path_prev).name.split(".")[0]
            candidate = latent_dir / f"{image_id}.pt"
            if candidate.exists():
                latent_path = candidate
            else:
                print(f"[{idx}] Latent not found ({candidate}), falling back to on-the-fly encoding")

        x_hat = predict_pair(
            vae, model, schedule, x_prev, cond,
            n_samples=args.n_samples,
            device=device,
            latent_path=latent_path,
            cfg_scale=args.cfg_scale,
        )

        out_name = f"{idx:04d}_pred_age{age_next:.1f}.nii.gz"
        out_path = output_dir / out_name
        nib.save(tensor_to_nii(x_hat.numpy(), path_prev), str(out_path))

        records.append({
            "idx":       idx,
            "path_prev": str(path_prev),
            "path_next": str(path_next),
            "path_pred": str(out_path),
            "age_prev":  age_prev,
            "age_next":  age_next,
            "delta_t":   delta_t,
            "sex":       sex,
            "group_id":  group_id,
        })

        pd.DataFrame(records).to_csv(csv_out, index=False)
        print(f"[{idx+1}/{n_total}]  age {age_prev:.1f} → {age_next:.1f}  saved {out_name}", flush=True)

    print(f"\nDone. {len(records)} predictions saved to {output_dir}")
    print(f"Summary CSV: {csv_out}")


if __name__ == "__main__":
    main()
