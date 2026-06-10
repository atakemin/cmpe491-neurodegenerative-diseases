"""Evaluation of predicted MRI scans against ground truth.

Metrics:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index, computed slice-wise along axial axis)
  - MAE   (Mean Absolute Error)
  - NCC   (Normalised Cross-Correlation)

Normalisation: both predicted and ground-truth volumes are mapped to [0, 1]
using the ground-truth's own p1/p99 percentile range.  This is the same
normalisation used during training and is the correct reference frame for
grayscale MRI — the predicted image is evaluated relative to the intensity
scale of the GT scan it is trying to match.

Usage:
  python evaluate.py \\
      --results_dir ./inference_results \\
      --output_csv  ./inference_results/metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(vol: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Clip to [p1, p99] then scale to [0, 1]."""
    if p99 <= p1:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip(vol, p1, p99)
    return ((vol - p1) / (p99 - p1)).astype(np.float32)


TARGET_SHAPE = (128, 160, 128)


def load_and_normalise(path: str | Path, p1: float | None = None, p99: float | None = None):
    """Load NIfTI, resize to TARGET_SHAPE if needed, normalise.
    Returns (normalised_volume, p1, p99).
    """
    arr = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if arr.shape != TARGET_SHAPE:
        arr = resize(arr, TARGET_SHAPE, order=1, mode="reflect",
                     anti_aliasing=True, preserve_range=True).astype(np.float32)
    if p1 is None or p99 is None:
        p1 = float(np.percentile(arr, 1))
        p99 = float(np.percentile(arr, 99))
    return normalise(arr, p1, p99), p1, p99


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim_3d(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Axial slice-wise SSIM averaged over all slices."""
    scores = []
    for i in range(gt.shape[2]):          # iterate over D (axial slices)
        sl_gt   = gt[:, :, i]
        sl_pred = pred[:, :, i]
        if sl_gt.max() - sl_gt.min() < 1e-6:
            continue                       # skip blank slices
        s = structural_similarity(
            sl_gt, sl_pred,
            data_range=data_range,
            win_size=7,
        )
        scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")


def slice_metrics(gt: np.ndarray, pred: np.ndarray,
                  data_range: float = 1.0) -> dict:
    """PSNR and SSIM on the middle slice of each axis.

    Volume shape is (H, W, D).
    Returns dict with keys: psnr_x, ssim_x, psnr_y, ssim_y, psnr_z, ssim_z
    """
    H, W, D = gt.shape
    slices = {
        "x": (gt[H // 2, :, :],  pred[H // 2, :, :]),   # coronal-ish
        "y": (gt[:, W // 2, :],  pred[:, W // 2, :]),    # sagittal-ish
        "z": (gt[:, :, D // 2],  pred[:, :, D // 2]),    # axial middle
    }
    result = {}
    for axis, (sl_gt, sl_pred) in slices.items():
        result[f"psnr_{axis}"] = psnr(sl_gt, sl_pred, data_range)
        if sl_gt.max() - sl_gt.min() < 1e-6:
            result[f"ssim_{axis}"] = float("nan")
        else:
            result[f"ssim_{axis}"] = float(structural_similarity(
                sl_gt, sl_pred, data_range=data_range, win_size=7,
            ))
    return result


def mae(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(gt - pred)))


def ncc(gt: np.ndarray, pred: np.ndarray) -> float:
    """Normalised Cross-Correlation in [-1, 1]."""
    gt_z   = gt   - gt.mean()
    pred_z = pred - pred.mean()
    denom  = np.sqrt((gt_z ** 2).sum() * (pred_z ** 2).sum())
    if denom < 1e-10:
        return float("nan")
    return float((gt_z * pred_z).sum() / denom)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predicted MRI scans")
    p.add_argument("--results_dir", required=True,
                   help="Folder produced by inference.py (contains inference_results.csv)")
    p.add_argument("--output_csv",  default=None,
                   help="Where to save per-sample metrics CSV (default: results_dir/metrics.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv  = Path(args.output_csv) if args.output_csv else results_dir / "metrics.csv"

    manifest = pd.read_csv(results_dir / "inference_results.csv")
    print(f"Evaluating {len(manifest)} pairs …\n")

    records = []

    for i, row in manifest.iterrows():
        path_gt   = Path(row["path_next"])
        path_pred = Path(row["path_pred"])

        if not path_gt.exists():
            print(f"[{i}] GT not found: {path_gt}")
            continue
        if not path_pred.exists():
            print(f"[{i}] Pred not found: {path_pred}")
            continue

        # Normalise both volumes using GT's percentile range
        gt_vol, p1, p99 = load_and_normalise(path_gt)
        pred_vol, _, _  = load_and_normalise(path_pred, p1=p1, p99=p99)

        m_psnr = psnr(gt_vol, pred_vol)
        m_ssim = ssim_3d(gt_vol, pred_vol)
        m_mae  = mae(gt_vol, pred_vol)
        m_ncc  = ncc(gt_vol, pred_vol)
        m_slices = slice_metrics(gt_vol, pred_vol)

        records.append({
            "idx":      i,
            "age_prev": row["age_prev"],
            "age_next": row["age_next"],
            "delta_t":  row["delta_t"],
            "group_id": row["group_id"],
            "psnr":     round(m_psnr, 4),
            "ssim":     round(m_ssim, 4),
            "mae":      round(m_mae,  6),
            "ncc":      round(m_ncc,  4),
            **{k: round(v, 4) for k, v in m_slices.items()},
        })

        print(
            f"[{i+1}/{len(manifest)}]  age {row['age_prev']:.1f}→{row['age_next']:.1f}  "
            f"PSNR={m_psnr:.2f}  SSIM={m_ssim:.4f}  MAE={m_mae:.5f}  NCC={m_ncc:.4f}\n"
            f"  mid-slice  "
            f"X: PSNR={m_slices['psnr_x']:.2f} SSIM={m_slices['ssim_x']:.4f}  "
            f"Y: PSNR={m_slices['psnr_y']:.2f} SSIM={m_slices['ssim_y']:.4f}  "
            f"Z: PSNR={m_slices['psnr_z']:.2f} SSIM={m_slices['ssim_z']:.4f}",
            flush=True,
        )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*65}")
    print(f"  Samples evaluated : {len(df)}")
    print(f"  PSNR  mean ± std  : {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
    print(f"  SSIM  mean ± std  : {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
    print(f"  MAE   mean ± std  : {df['mae'].mean():.5f} ± {df['mae'].std():.5f}")
    print(f"  NCC   mean ± std  : {df['ncc'].mean():.4f} ± {df['ncc'].std():.4f}")
    print(f"  --- Middle-slice metrics ---")
    for axis in ("x", "y", "z"):
        pk, sk = f"psnr_{axis}", f"ssim_{axis}"
        print(f"  {axis.upper()}-mid PSNR mean ± std : {df[pk].mean():.2f} ± {df[pk].std():.2f} dB")
        print(f"  {axis.upper()}-mid SSIM mean ± std : {df[sk].mean():.4f} ± {df[sk].std():.4f}")
    print(f"{'='*65}")
    print(f"\nPer-sample metrics saved to: {output_csv}")


if __name__ == "__main__":
    main()
