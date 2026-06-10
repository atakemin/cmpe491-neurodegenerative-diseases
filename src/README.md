# Longitudinal Brain MRI Synthesis via Conditional Latent Diffusion

A two-stage framework for predicting future brain MRI scans from longitudinal ADNI data. Stage 1 uses a pretrained LDAE AutoencoderKL to compress scans into a compact latent space; Stage 2 trains a conditional 3D DDPM in that latent space to predict how a patient's brain will look at a future timepoint.

> The VAE (Stage 1) uses the pretrained weights from [LDAE](https://github.com/GabrieleLozupone/LDAE/issues). Clone their repository and follow their instructions to obtain `autoencoderkl.pth`.

---

## File Overview

**`dataset.py`**
Loads ADNI NIfTI scans and metadata CSV, builds patient-level train/val/test splits, and provides longitudinal scan pair datasets for both raw NIfTI and pre-cached latent modes. Groups are mapped as CN/SMC→0, EMCI/MCI/LMCI→1, AD→2.

**`vae.py`**
Wraps the LDAE AutoencoderKL with a frozen encoder for latent caching and an active decoder for reconstructing predictions back to MRI space. Handles weight loading from the LDAE checkpoint format.

**`diffusion.py`**
Implements the DDPM noise schedule, 3D UNet denoiser with Adaptive Group Normalisation conditioning, classifier-free guidance, and the combined ε-prediction + latent reconstruction loss. This is the core model definition.

**`cache_latents.py`**
Encodes all NIfTI scans once through the frozen VAE and saves each as a `<image_id>.pt` latent file. Run this before training to avoid repeated on-the-fly encoding (≈20× faster training).

**`train_diffusion.py`**
Trains the conditional latent DDPM with cosine LR scheduling, AMP, gradient clipping, CFG dropout, and optional latent reconstruction loss. Logs train/val loss to CSV and saves the best checkpoint.

**`inference.py`**
Runs the trained diffusion model on the test split, saves each predicted scan as a NIfTI file, and writes a summary CSV. Supports pre-cached latents and classifier-free guidance scale at inference time.

**`evaluate.py`**
Computes PSNR, SSIM, MAE, NCC, and per-axis middle-slice PSNR/SSIM between predicted and ground-truth scans. Reads the CSV produced by `inference.py` and saves a per-sample metrics CSV with a printed summary.

---

## Setup

```bash
# Create and activate a Python 3.10 environment
conda create -n brainmri python=3.10 -y
conda activate brainmri

# Install dependencies
pip install -r requirements.txt
```

> **LDAE weights:** Clone the [LDAE repository](https://github.com/GabrieleLozupone/LDAE/issues) and follow their instructions to obtain `autoencoderkl.pth`. You only need the checkpoint file — the LDAE codebase does not need to be installed.

---

## Usage

### 1. Cache latents (run once before training)

```bash
python cache_latents.py \
    --data_dir   /data/adni/nii/ \
    --output_dir /data/adni/latents/ \
    --pretrained_weights /path/to/autoencoderkl.pth \
    --batch_size 4
```

### 2. Train the diffusion model

```bash
python train_diffusion.py \
    --latent_dir /data/adni/latents/ \
    --csv_path   /data/adni/adni_metadata.csv \
    --output_dir ./runs/exp1 \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --cfg_dropout 0.15 \
    --lambda_rec 0.1 \
    --amp \
    --seed 42
```

### 3. Run inference on the test split

```bash
python inference.py \
    --data_dir           /data/adni/nii/ \
    --latent_dir         /data/adni/latents/ \
    --csv_path           /data/adni/adni_metadata.csv \
    --pretrained_weights /path/to/autoencoderkl.pth \
    --diffusion_weights  ./runs/exp1/diffusion_best.pt \
    --output_dir         ./results/exp1 \
    --cfg_scale 2.0 \
    --n_samples 1 \
    --max_pairs 100 \
    --seed 42
```

### 4. Evaluate predictions

```bash
python evaluate.py \
    --results_dir ./results/exp1 \
    --output_csv  ./results/exp1/metrics.csv
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- `nibabel`, `pandas`, `numpy`, `scikit-image`
- [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels) (for AutoencoderKL)
- LDAE pretrained weights (`autoencoderkl.pth`)
