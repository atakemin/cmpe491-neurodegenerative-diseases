"""Stage 2: Conditional Latent DDPM.

Operates entirely in the VAE latent space  (3 × 16 × 20 × 16).

Conditioning:
  - Spatial:  z_prev (3 ch) concatenated with noisy z_t (3 ch) → 6-ch UNet input.
  - Scalars:  age_prev, age_next, delta_t, sex   injected via AdaGN at every ResBlock.
  - Discrete: group_id (0=CN,1=MCI,2=AD)         embedded + added to scalar embedding.

UNet encoder arms can be initialised from MedicalNet ResNet-50 weights.

DDPM schedule: T=1000, linear β from 1e-4 to 2e-2.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def linear_beta_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class DiffusionSchedule:
    def __init__(self, T: int = 1000):
        betas = linear_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.T = T
        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register("posterior_variance",
                      betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def register(self, name: str, val: torch.Tensor) -> None:
        setattr(self, name, val)

    def to(self, device):
        for attr in ("betas", "alphas_cumprod", "sqrt_alphas_cumprod",
                     "sqrt_one_minus_alphas_cumprod", "posterior_variance"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z0)
        s_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        s_b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return s_a * z0 + s_b * noise


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, identical to DDPM / Stable Diffusion."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class AdaGN(nn.Module):
    """Adaptive Group Normalisation conditioned on a scalar embedding."""

    def __init__(self, num_channels: int, emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        # (B, C) -> (B, C, 1, 1, 1) to broadcast over spatial dims
        extra = x.ndim - 2
        scale = scale.view(scale.shape[0], scale.shape[1], *([1] * extra))
        shift = shift.view(shift.shape[0], shift.shape[1], *([1] * extra))
        return self.norm(x) * (1.0 + scale) + shift


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.adagn = AdaGN(out_ch, emb_dim)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.adagn(self.conv2(h), emb)
        return self.act(h + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.res = ResBlock3D(in_ch, out_ch, emb_dim)
        self.down = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, emb)
        return self.down(h), h   # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.res = ResBlock3D(in_ch + skip_ch, out_ch, emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, emb)


# ---------------------------------------------------------------------------
# Conditioning MLP
# ---------------------------------------------------------------------------

class ConditioningMLP(nn.Module):
    """Fuses timestep + scalar covariates + group_id into a single embedding.

    Embedding index n_groups is reserved as the null/dropped class for CFG.
    """

    def __init__(self, emb_dim: int = 256, n_groups: int = 3):
        super().__init__()
        self.null_idx = n_groups          # index used when conditioning is dropped
        # 4 scalar signals: age_prev, age_next, delta_t, sex
        self.scalar_proj = nn.Linear(4, emb_dim)
        self.group_emb = nn.Embedding(n_groups + 1, emb_dim)   # +1 for null class
        self.time_proj = nn.Linear(emb_dim, emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(
        self,
        t: torch.Tensor,
        age_prev: torch.Tensor,
        age_next: torch.Tensor,
        delta_t: torch.Tensor,
        sex: torch.Tensor,
        group_id: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_proj(timestep_embedding(t, self.time_proj.in_features))
        scalars = torch.stack([age_prev, age_next, delta_t, sex], dim=-1)
        s_emb = F.silu(self.scalar_proj(scalars))
        g_emb = self.group_emb(group_id)
        return self.fuse(torch.cat([t_emb, s_emb, g_emb], dim=-1))


# ---------------------------------------------------------------------------
# 3D UNet denoiser
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    Lightweight 3D UNet for denoising in latent space.

    Input channels: 6  (3 z_t + 3 z_prev)
    Base channels:  64 → 128 → 256 → 512 (bottleneck)
    Depth:          3 down / 3 up
    """

    BASE_CHANNELS = (64, 128, 256, 512)

    def __init__(self, in_channels: int = 6, out_channels: int = 3, emb_dim: int = 256,
                 n_groups: int = 3):
        super().__init__()
        C = self.BASE_CHANNELS
        self.emb_dim = emb_dim

        self.cond_mlp = ConditioningMLP(emb_dim, n_groups=n_groups)
        self.stem = nn.Conv3d(in_channels, C[0], 3, padding=1)

        self.down1 = DownBlock(C[0], C[1], emb_dim)
        self.down2 = DownBlock(C[1], C[2], emb_dim)
        self.down3 = DownBlock(C[2], C[3], emb_dim)

        self.mid = ResBlock3D(C[3], C[3], emb_dim)

        self.up3 = UpBlock(C[3], C[3], C[2], emb_dim)
        self.up2 = UpBlock(C[2], C[2], C[1], emb_dim)
        self.up1 = UpBlock(C[1], C[1], C[0], emb_dim)

        self.head = nn.Conv3d(C[0], out_channels, 1)

    @property
    def null_group_idx(self) -> int:
        return self.cond_mlp.null_idx

    def forward(
        self,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
        t: torch.Tensor,
        age_prev: torch.Tensor,
        age_next: torch.Tensor,
        delta_t: torch.Tensor,
        sex: torch.Tensor,
        group_id: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.cond_mlp(t, age_prev, age_next, delta_t, sex, group_id)

        x = torch.cat([z_t, z_prev], dim=1)
        x = self.stem(x)

        x, s1 = self.down1(x, emb)
        x, s2 = self.down2(x, emb)
        x, s3 = self.down3(x, emb)

        x = self.mid(x, emb)

        x = self.up3(x, s3, emb)
        x = self.up2(x, s2, emb)
        x = self.up1(x, s1, emb)

        return self.head(x)

    def load_medicalnet_encoder(self, weights_path: str | Path) -> None:
        """Attempt to initialise the three DownBlock conv weights from a MedicalNet ResNet-50 checkpoint."""
        ckpt = torch.load(str(weights_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # MedicalNet stores weights as "module.layer1…" etc.
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        mapping = {
            "layer1": self.down1.res.conv1,
            "layer2": self.down2.res.conv1,
            "layer3": self.down3.res.conv1,
        }
        loaded = 0
        for layer_name, target in mapping.items():
            key = f"{layer_name}.0.conv1.weight"
            if key in state:
                src = state[key]
                if src.shape == target.weight.shape:
                    target.weight.data.copy_(src)
                    loaded += 1
        print(f"[UNet3D] Loaded {loaded}/3 encoder layers from MedicalNet weights.")


# ---------------------------------------------------------------------------
# Diffusion training step
# ---------------------------------------------------------------------------

def _drop_cond(cond: Dict[str, torch.Tensor], null_group_idx: int,
               cfg_dropout: float) -> Dict[str, torch.Tensor]:
    """Randomly null out conditioning for CFG training."""
    B = cond["age_prev"].shape[0]
    mask = torch.rand(B, device=cond["age_prev"].device) < cfg_dropout  # True → drop
    result = {}
    for k, v in cond.items():
        if k == "group_id":
            result[k] = torch.where(mask, torch.full_like(v, null_group_idx), v)
        else:
            result[k] = v * (~mask).float()
    return result


def diffusion_loss(
    model: UNet3D,
    schedule: DiffusionSchedule,
    z_next: torch.Tensor,
    z_prev: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    cfg_dropout: float = 0.0,
    lambda_rec: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ε-prediction MSE + optional z0 reconstruction loss.
    Returns (total_loss, eps_loss, rec_loss).
    """
    B = z_next.shape[0]
    t = torch.randint(0, schedule.T, (B,), device=z_next.device)
    noise = torch.randn_like(z_next)
    z_t = schedule.q_sample(z_next, t, noise)

    # CFG: randomly drop conditioning during training
    null_idx = getattr(model, "null_group_idx",
                       getattr(getattr(model, "module", model), "null_group_idx", 3))
    cond_in = _drop_cond(cond, null_idx, cfg_dropout) if cfg_dropout > 0 else cond

    pred_noise = model(
        z_t, z_prev, t,
        cond_in["age_prev"], cond_in["age_next"], cond_in["delta_t"],
        cond_in["sex"], cond_in["group_id"],
    )

    eps_loss = F.mse_loss(pred_noise, noise)

    # z0 reconstruction: algebraically recover predicted clean latent
    if lambda_rec > 0:
        s_a = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1).to(z_t.device)
        s_b = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1).to(z_t.device)
        z0_pred = (z_t - s_b * pred_noise) / s_a.clamp(min=1e-8)
        rec_loss = F.l1_loss(z0_pred, z_next)
        total = eps_loss + lambda_rec * rec_loss
    else:
        rec_loss = torch.zeros(1, device=z_next.device)
        total = eps_loss

    return total, eps_loss, rec_loss


# ---------------------------------------------------------------------------
# Sampling (DDPM reverse process)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(
    model: UNet3D,
    schedule: DiffusionSchedule,
    z_prev: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    latent_shape: Tuple[int, ...] = (3, 16, 20, 16),
    device: torch.device | str = "cpu",
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    B = z_prev.shape[0]
    z = torch.randn(B, *latent_shape, device=device)
    schedule.to(device)

    # Build null conditioning for CFG
    null_idx = getattr(model, "null_group_idx",
                       getattr(getattr(model, "module", model), "null_group_idx", 3))
    null_cond = {
        "age_prev": torch.zeros_like(cond["age_prev"]),
        "age_next": torch.zeros_like(cond["age_next"]),
        "delta_t":  torch.zeros_like(cond["delta_t"]),
        "sex":      torch.zeros_like(cond["sex"]),
        "group_id": torch.full_like(cond["group_id"], null_idx),
    }

    for i in reversed(range(schedule.T)):
        t = torch.full((B,), i, device=device, dtype=torch.long)

        pred_cond = model(
            z, z_prev, t,
            cond["age_prev"], cond["age_next"], cond["delta_t"], cond["sex"], cond["group_id"],
        )

        if cfg_scale > 1.0:
            pred_null = model(
                z, z_prev, t,
                null_cond["age_prev"], null_cond["age_next"], null_cond["delta_t"],
                null_cond["sex"], null_cond["group_id"],
            )
            pred_noise = pred_null + cfg_scale * (pred_cond - pred_null)
        else:
            pred_noise = pred_cond

        beta_t  = schedule.betas[i]
        alpha_t = 1.0 - beta_t
        ac_t    = schedule.alphas_cumprod[i]

        z = (1.0 / alpha_t.sqrt()) * (z - (beta_t / (1.0 - ac_t).sqrt()) * pred_noise)
        if i > 0:
            z = z + schedule.posterior_variance[i].sqrt() * torch.randn_like(z)

    return z
