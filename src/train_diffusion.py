"""Train Stage 2: Conditional Latent DDPM.

Supports two data modes:
  --latent_dir   Pre-cached .pt latents (fast, recommended)
  --data_dir     Raw NIfTI files encoded on-the-fly via frozen VAE (slow)

New features vs previous version:
  CFG dropout  : --cfg_dropout 0.15  (randomly null conditioning during training)
  z0 rec loss  : --lambda_rec 0.1    (direct reconstruction loss on predicted latent)
  CFG scale    : set in inference.py  (guidance strength at sampling time)

Usage:
  python train_diffusion.py \\
      --latent_dir /archive2/adni/latents/ \\
      --csv_path   /archive2/LDAE/LDAE/ADSP-PHC__ADNI_T1_1.0_4_07_2026.csv \\
      --output_dir ./runs/diffusion_v3 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Stage2Dataset, Stage2LatentDataset
from diffusion import DiffusionSchedule, UNet3D, diffusion_loss
from vae import BrainVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage-2 Conditional Latent DDPM")

    # Data — one of these two is required
    p.add_argument("--latent_dir",         default=None,  help="Folder with cached .pt latents (fast)")
    p.add_argument("--data_dir",           default=None,  help="Folder with .nii files (on-the-fly encoding)")
    p.add_argument("--csv_path",           required=True, help="ADNI metadata CSV")
    p.add_argument("--pretrained_weights", default=None,  help="LDAE AE checkpoint (required if using --data_dir)")
    p.add_argument("--output_dir",         default="./runs/diffusion")
    p.add_argument("--seed",               type=int, default=42)

    p.add_argument("--medicalnet_weights", default=None)

    # Training
    p.add_argument("--epochs",             type=int,   default=300)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--val_every",          type=int,   default=5)
    p.add_argument("--save_every",         type=int,   default=10)
    p.add_argument("--diffusion_steps",    type=int,   default=1000)
    p.add_argument("--emb_dim",            type=int,   default=256)
    p.add_argument("--max_delta_years",    type=float, default=None)
    p.add_argument("--split_train",        type=float, default=0.80)
    p.add_argument("--split_val",          type=float, default=0.10)
    p.add_argument("--split_test",         type=float, default=0.10)
    p.add_argument("--amp",                action="store_true")
    p.add_argument("--grad_clip",          type=float, default=0.5)

    # New: CFG + reconstruction loss
    p.add_argument("--cfg_dropout",        type=float, default=0.15,
                   help="Probability of nulling conditioning during training")
    p.add_argument("--lambda_rec",         type=float, default=0.1,
                   help="Weight of z0 reconstruction loss")

    return p.parse_args()


def get_z(batch: dict, vae, device: torch.device, use_latents: bool):
    """Return (z_prev, z_next) on device, from either cached latents or on-the-fly encoding."""
    if use_latents:
        return batch["z_prev"].to(device), batch["z_next"].to(device)
    else:
        with torch.no_grad():
            mu_prev, _ = vae.encode(batch["x_prev"].to(device))
            mu_next, _ = vae.encode(batch["x_next"].to(device))
        return mu_prev * vae.scale_factor, mu_next * vae.scale_factor


def main() -> None:
    args = parse_args()

    if args.latent_dir is None and args.data_dir is None:
        raise ValueError("Provide either --latent_dir or --data_dir")
    if args.data_dir is not None and args.pretrained_weights is None:
        raise ValueError("--pretrained_weights required when using --data_dir")

    set_seed(args.seed)
    use_latents = args.latent_dir is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ratios = (args.split_train, args.split_val, args.split_test)

    if use_latents:
        train_ds = Stage2LatentDataset(args.latent_dir, args.csv_path, split="train",
                                       split_ratios=split_ratios, seed=args.seed,
                                       max_delta_years=args.max_delta_years)
        val_ds   = Stage2LatentDataset(args.latent_dir, args.csv_path, split="val",
                                       split_ratios=split_ratios, seed=args.seed,
                                       max_delta_years=args.max_delta_years)
    else:
        train_ds = Stage2Dataset(args.data_dir, args.csv_path, split="train",
                                 split_ratios=split_ratios, seed=args.seed,
                                 max_delta_years=args.max_delta_years)
        val_ds   = Stage2Dataset(args.data_dir, args.csv_path, split="val",
                                 split_ratios=split_ratios, seed=args.seed,
                                 max_delta_years=args.max_delta_years)

    print(f"Train pairs: {len(train_ds)}  |  Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # VAE only needed for on-the-fly encoding
    vae = None
    if not use_latents:
        vae = BrainVAE(pretrained_weights=args.pretrained_weights, freeze_encoder=False).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    model = UNet3D(in_channels=6, out_channels=3, emb_dim=args.emb_dim).to(device)
    if args.medicalnet_weights:
        model.load_medicalnet_encoder(args.medicalnet_weights)

    schedule = DiffusionSchedule(T=args.diffusion_steps)
    schedule.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_val_loss = float("inf")
    log_path = output_dir / "loss_log.csv"
    n_train = len(train_loader)
    log_interval = max(1, n_train // 10)

    with open(log_path, "w") as f:
        f.write("epoch,step,train_loss,eps_loss,rec_loss,val_loss\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_eps = train_rec = 0.0

        for step, batch in enumerate(train_loader, 1):
            z_prev, z_next = get_z(batch, vae, device, use_latents)
            cond = {
                "age_prev": batch["age_prev"].to(device),
                "age_next": batch["age_next"].to(device),
                "delta_t":  batch["delta_t"].to(device),
                "sex":      batch["sex"].to(device),
                "group_id": batch["group_id"].to(device),
            }

            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss, eps_l, rec_l = diffusion_loss(
                    model, schedule, z_next, z_prev, cond,
                    cfg_dropout=args.cfg_dropout,
                    lambda_rec=args.lambda_rec,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()

            train_loss += loss.item()
            train_eps  += eps_l.item()
            train_rec  += rec_l.item()

            if step % log_interval == 0 or step == n_train:
                with open(log_path, "a") as f:
                    f.write(f"{epoch},{step},{loss.item():.6f},"
                            f"{eps_l.item():.6f},{rec_l.item():.6f},\n")

        train_loss /= n_train
        train_eps  /= n_train
        train_rec  /= n_train

        # ---- validate ----
        if epoch % args.val_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    z_prev, z_next = get_z(batch, vae, device, use_latents)
                    cond = {
                        "age_prev": batch["age_prev"].to(device),
                        "age_next": batch["age_next"].to(device),
                        "delta_t":  batch["delta_t"].to(device),
                        "sex":      batch["sex"].to(device),
                        "group_id": batch["group_id"].to(device),
                    }
                    loss_v, _, _ = diffusion_loss(
                        model, schedule, z_next, z_prev, cond,
                        cfg_dropout=0.0, lambda_rec=args.lambda_rec,
                    )
                    val_loss += loss_v.item()
            val_loss /= len(val_loader)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{n_train},{train_loss:.6f},"
                        f"{train_eps:.6f},{train_rec:.6f},{val_loss:.6f}\n")
            print(f"Epoch {epoch:4d}  train={train_loss:.4f}  "
                  f"eps={train_eps:.4f}  rec={train_rec:.4f}  val={val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / "diffusion_best.pt")
        else:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{n_train},{train_loss:.6f},"
                        f"{train_eps:.6f},{train_rec:.6f},\n")
            print(f"Epoch {epoch:4d}  train={train_loss:.4f}  "
                  f"eps={train_eps:.4f}  rec={train_rec:.4f}", flush=True)

        scheduler.step()

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"diffusion_epoch{epoch:04d}.pt")

    torch.save(model.state_dict(), output_dir / "diffusion_final.pt")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
