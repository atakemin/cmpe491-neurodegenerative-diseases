"""Stage 1: 3D VAE — thin finetune wrapper around LDAE's AutoencoderKL.

Loads the LDAE AutoencoderKL architecture from LDAE/src/ldae/ae_kl.py using the
aekl.yaml config and optionally initialises weights from a pretrained checkpoint.

Input:  (B, 1, 128, 160, 128)
Latent: (B, 3, 16, 20, 16)  — 8× downsampling, scale_factor = 0.8730
Output: (B, 1, 128, 160, 128)

Loss: β-VAE  =>  L1_recon  +  β * KL(q || N(0,I))
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

# LDAE path resolution — expects LDAE repo at ../LDAE/LDAE relative to this file
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LDAE_SRC = _REPO_ROOT / "LDAE" / "LDAE" / "src"
if str(_LDAE_SRC) not in sys.path:
    sys.path.insert(0, str(_LDAE_SRC))


SCALE_FACTOR = 0.8730   # LDAE's published latent scale factor


def _build_ae_kl() -> nn.Module:
    """Instantiate LDAE's AutoencoderKL using the canonical aekl.yaml config."""
    try:
        from generative.networks.nets import AutoencoderKL
    except ImportError as e:
        raise ImportError("monai-generative is required: pip install monai-generative") from e

    return AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 128, 128),
        latent_channels=3,
        num_res_blocks=2,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )


class BrainVAE(nn.Module):
    """Finetunable wrapper around LDAE's AutoencoderKL."""

    def __init__(self, pretrained_weights: str | Path | None = None, freeze_encoder: bool = True):
        super().__init__()
        self.ae = _build_ae_kl()
        self.scale_factor = SCALE_FACTOR

        if pretrained_weights is not None:
            self._load_weights(Path(pretrained_weights))

        if freeze_encoder:
            for name, p in self.ae.named_parameters():
                if "encoder" in name or ("quant_conv" in name and "post_quant_conv" not in name):
                    p.requires_grad = False
            print("[BrainVAE] Encoder frozen — only decoder will be trained.")

    def _load_weights(self, ckpt: Path) -> None:
        if not ckpt.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {ckpt}")
        state = torch.load(str(ckpt), map_location="cpu")
        # LDAE LitAutoencoderKL stores weights under 'state_dict' key
        if isinstance(state, dict) and "state_dict" in state:
            raw = state["state_dict"]
            # Strip "model." prefix that LightningModule adds
            state = {k.replace("model.", "", 1): v for k, v in raw.items()}
        missing, unexpected = self.ae.load_state_dict(state, strict=False)
        if missing:
            print(f"[BrainVAE] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[BrainVAE] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    # ------------------------------------------------------------------
    # Encode / decode API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var) of shape (B, 3, 16, 20, 16)."""
        with torch.no_grad():
            z_mu, z_sigma = self.ae.encode(x)
        log_var = 2.0 * torch.log(z_sigma.detach() + 1e-8)
        return z_mu.detach(), log_var

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_scaled = z / self.scale_factor
        if self.training:
            return cp.checkpoint(self.ae.decode_stage_2_outputs, z_scaled, use_reentrant=False)
        return self.ae.decode_stage_2_outputs(z_scaled)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (recon, mu, log_var)."""
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decode(z * self.scale_factor)
        return recon, mu, log_var


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """β-VAE loss.  Returns (total, recon_term, kl_term)."""
    recon_loss = F.l1_loss(recon, x, reduction="mean")
    # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))  averaged over batch
    kl = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl
