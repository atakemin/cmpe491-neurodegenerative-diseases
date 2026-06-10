"""ADNI dataset utilities for Stage 1 (single-scan VAE) and Stage 2 (longitudinal pairs for DDPM).

CSV is the ADNI metadata file; NIfTI files are named <Image Data ID>.nii (or .nii.gz).

Group mapping applied at load time:
  CN / SMC            → 0 (CN)
  EMCI / MCI / LMCI  → 1 (MCI)
  AD                  → 2 (AD)
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUP_MAP = {
    "CN": 0, "SMC": 0,
    "EMCI": 1, "MCI": 1, "LMCI": 1,
    "AD": 2,
}

TARGET_SHAPE = (128, 160, 128)   # H × W × D


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {s!r}")


def _load_nii(path: Path) -> torch.Tensor:
    """Load a NIfTI file and return a float32 tensor of shape (1, H, W, D)."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    # Resize to model's expected spatial shape if needed
    if arr.shape != TARGET_SHAPE:
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
        t = torch.nn.functional.interpolate(t, size=TARGET_SHAPE, mode="trilinear", align_corners=False)
        arr = t.squeeze().numpy()
    # Intensity clamp + normalise to [0, 1]
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    if p99 > p1:
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, D)


def _find_nii(data_dir: Path, image_id: str) -> Path:
    for suffix in (".nii", ".nii.gz"):
        p = data_dir / f"{image_id}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"NIfTI not found for Image Data ID {image_id!r} in {data_dir}")


# ---------------------------------------------------------------------------
# Split builder — patient-level
# ---------------------------------------------------------------------------

def build_patient_split(
    subject_ids: List[str],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} subject ID lists."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)
    ids = sorted(set(subject_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": ids[:n_train],
        "val":   ids[n_train: n_train + n_val],
        "test":  ids[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Stage 1 — single-scan dataset
# ---------------------------------------------------------------------------

class Stage1Dataset(Dataset):
    """One record per scan; used for VAE (reconstruction) training."""

    def __init__(
        self,
        data_dir: str | Path,
        csv_path: str | Path,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        data_dir = Path(data_dir)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Normalise column names (ADNI CSV uses "Image Data ID", "Subject", etc.)
        df = df.rename(columns={
            "Image Data ID": "image_id",
            "Subject":       "subject_id",
            "Age":           "age",
            "Sex":           "sex",
            "Group":         "group",
            "Acq Date":      "acq_date",
        })

        df["group_id"] = df["group"].map(GROUP_MAP)
        df = df.dropna(subset=["image_id", "subject_id", "age", "group_id"])

        subject_split = build_patient_split(
            df["subject_id"].tolist(), ratios=split_ratios, seed=seed
        )
        subjects_in_split = set(subject_split[split])
        df = df[df["subject_id"].isin(subjects_in_split)].reset_index(drop=True)

        # Keep only scans whose NIfTI file exists
        records = []
        for _, row in df.iterrows():
            try:
                nii_path = _find_nii(data_dir, str(row["image_id"]))
                records.append({
                    "path":      nii_path,
                    "subject_id": row["subject_id"],
                    "age":       float(row["age"]),
                    "sex":       0.0 if str(row["sex"]).upper().startswith("F") else 1.0,
                    "group_id":  int(row["group_id"]),
                })
            except FileNotFoundError:
                pass

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        x = _load_nii(r["path"])
        return {
            "x":        x,
            "age":      torch.tensor(r["age"], dtype=torch.float32),
            "sex":      torch.tensor(r["sex"], dtype=torch.float32),
            "group_id": torch.tensor(r["group_id"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Stage 2 — longitudinal pair dataset
# ---------------------------------------------------------------------------

class Stage2Dataset(Dataset):
    """One record per ordered (earlier, later) scan pair from the same subject."""

    def __init__(
        self,
        data_dir: str | Path,
        csv_path: str | Path,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        max_delta_years: Optional[float] = None,
    ):
        data_dir = Path(data_dir)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        print(f"[dataset] CSV columns: {df.columns.tolist()}")
        print(f"[dataset] CSV shape: {df.shape}")
        print(f"[dataset] Sample groups: {df['Group'].unique()[:10] if 'Group' in df.columns else 'NO Group COLUMN'}")

        df = df.rename(columns={
            "Image Data ID": "image_id",
            "Subject":       "subject_id",
            "Age":           "age",
            "Sex":           "sex",
            "Group":         "group",
            "Acq Date":      "acq_date",
        })

        df["group_id"] = df["group"].map(GROUP_MAP)
        before = len(df)
        df = df.dropna(subset=["image_id", "subject_id", "age", "acq_date", "group_id"])
        print(f"[dataset] Rows after dropna: {len(df)} (dropped {before - len(df)})")
        df["date_parsed"] = df["acq_date"].map(_parse_date)

        subject_split = build_patient_split(
            df["subject_id"].tolist(), ratios=split_ratios, seed=seed
        )
        subjects_in_split = set(subject_split[split])
        df = df[df["subject_id"].isin(subjects_in_split)].reset_index(drop=True)

        # Diagnose file discovery
        sample_ids = df["image_id"].iloc[:5].tolist()
        print(f"[dataset] Sample image IDs from CSV: {sample_ids}")
        for sid in sample_ids:
            for suffix in (".nii", ".nii.gz"):
                p = data_dir / f"{sid}{suffix}"
                print(f"[dataset]   {p}  exists={p.exists()}")

        pairs = []
        for subj, grp in df.groupby("subject_id"):
            grp = grp.sort_values("date_parsed").reset_index(drop=True)
            scans = []
            for _, row in grp.iterrows():
                try:
                    nii_path = _find_nii(data_dir, str(row["image_id"]))
                    scans.append({
                        "path":     nii_path,
                        "age":      float(row["age"]),
                        "sex":      0.0 if str(row["sex"]).upper().startswith("F") else 1.0,
                        "group_id": int(row["group_id"]),
                        "date":     row["date_parsed"],
                    })
                except FileNotFoundError:
                    pass

            for i in range(len(scans)):
                for j in range(i + 1, len(scans)):
                    s_i, s_j = scans[i], scans[j]
                    delta_days = (s_j["date"] - s_i["date"]).days
                    delta_years = delta_days / 365.25
                    if delta_years <= 0:
                        continue
                    if max_delta_years is not None and delta_years > max_delta_years:
                        continue
                    pairs.append({
                        "path_prev":  s_i["path"],
                        "path_next":  s_j["path"],
                        "age_prev":   s_i["age"],
                        "age_next":   s_j["age"],
                        "delta_t":    delta_years,
                        "sex":        s_i["sex"],
                        "group_id":   s_i["group_id"],
                    })

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        p = self.pairs[idx]
        x_prev = _load_nii(p["path_prev"])
        x_next = _load_nii(p["path_next"])
        return {
            "x_prev":   x_prev,
            "x_next":   x_next,
            "age_prev": torch.tensor(p["age_prev"], dtype=torch.float32),
            "age_next": torch.tensor(p["age_next"], dtype=torch.float32),
            "delta_t":  torch.tensor(p["delta_t"],  dtype=torch.float32),
            "sex":      torch.tensor(p["sex"],       dtype=torch.float32),
            "group_id": torch.tensor(p["group_id"],  dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Stage 2 — latent pair dataset (uses cached .pt files from cache_latents.py)
# ---------------------------------------------------------------------------

class Stage2LatentDataset(Dataset):
    """Same pairing logic as Stage2Dataset but loads pre-cached latent tensors
    instead of NIfTI files.  Requires cache_latents.py to have been run first.

    Each .pt file contains a (3, 16, 20, 16) latent tensor.
    """

    def __init__(
        self,
        latent_dir: str | Path,
        csv_path: str | Path,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        max_delta_years: Optional[float] = None,
    ):
        latent_dir = Path(latent_dir)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        df = df.rename(columns={
            "Image Data ID": "image_id",
            "Subject":       "subject_id",
            "Age":           "age",
            "Sex":           "sex",
            "Group":         "group",
            "Acq Date":      "acq_date",
        })

        df["group_id"] = df["group"].map(GROUP_MAP)
        df = df.dropna(subset=["image_id", "subject_id", "age", "acq_date", "group_id"])
        df["date_parsed"] = df["acq_date"].map(_parse_date)

        subject_split = build_patient_split(
            df["subject_id"].tolist(), ratios=split_ratios, seed=seed
        )
        subjects_in_split = set(subject_split[split])
        df = df[df["subject_id"].isin(subjects_in_split)].reset_index(drop=True)

        pairs = []
        for subj, grp in df.groupby("subject_id"):
            grp = grp.sort_values("date_parsed").reset_index(drop=True)
            scans = []
            for _, row in grp.iterrows():
                pt_path = latent_dir / f"{row['image_id']}.pt"
                if not pt_path.exists():
                    continue
                scans.append({
                    "pt_path":  pt_path,
                    "age":      float(row["age"]),
                    "sex":      0.0 if str(row["sex"]).upper().startswith("F") else 1.0,
                    "group_id": int(row["group_id"]),
                    "date":     row["date_parsed"],
                })

            for i in range(len(scans)):
                for j in range(i + 1, len(scans)):
                    s_i, s_j = scans[i], scans[j]
                    delta_days = (s_j["date"] - s_i["date"]).days
                    delta_years = delta_days / 365.25
                    if delta_years <= 0:
                        continue
                    if max_delta_years is not None and delta_years > max_delta_years:
                        continue
                    pairs.append({
                        "pt_prev":  s_i["pt_path"],
                        "pt_next":  s_j["pt_path"],
                        "age_prev": s_i["age"],
                        "age_next": s_j["age"],
                        "delta_t":  delta_years,
                        "sex":      s_i["sex"],
                        "group_id": s_i["group_id"],
                    })

        self.pairs = pairs
        print(f"[Stage2LatentDataset] {split}: {len(pairs)} pairs from {latent_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        p = self.pairs[idx]
        z_prev = torch.load(p["pt_prev"], map_location="cpu", weights_only=True)
        z_next = torch.load(p["pt_next"], map_location="cpu", weights_only=True)
        return {
            "z_prev":   z_prev,
            "z_next":   z_next,
            "age_prev": torch.tensor(p["age_prev"], dtype=torch.float32),
            "age_next": torch.tensor(p["age_next"], dtype=torch.float32),
            "delta_t":  torch.tensor(p["delta_t"],  dtype=torch.float32),
            "sex":      torch.tensor(p["sex"],       dtype=torch.float32),
            "group_id": torch.tensor(p["group_id"],  dtype=torch.long),
        }
