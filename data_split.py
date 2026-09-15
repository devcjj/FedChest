"""Partition local NIH, RSNA, and SIIM datasets into 3 distinct hospital nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_nih_hospital(root: Path) -> Path:
    """Prepare Hospital 1 (NIH Bounding Boxes & Multi-label Pathology)."""
    data_entry_path = root / "Data_Entry_2017.csv"
    bbox_path = root / "BBox_List_2017.csv"
    images_dir = root / "images"

    if not data_entry_path.is_file():
        raise FileNotFoundError(f"Missing NIH Data Entry CSV: {data_entry_path}")

    metadata = pd.read_csv(data_entry_path)
    available_images = {p.name for p in images_dir.iterdir() if p.is_file()} if images_dir.is_dir() else set()
    filtered = metadata[metadata["Image Index"].astype(str).isin(available_images)].copy()
    if filtered.empty:
        filtered = metadata.head(500).copy()

    # Merge bounding box annotations if available
    if bbox_path.is_file():
        bboxes = pd.read_csv(bbox_path)
        bboxes.columns = [c.strip() for c in bboxes.columns]
        # Bbox columns format: Image Index, Finding Label, Bbox [x,y,w,h],,,
        if "Bbox [x" in bboxes.columns or len(bboxes.columns) >= 6:
            bbox_renamed = bboxes.rename(columns={
                bboxes.columns[0]: "Image Index",
                bboxes.columns[1]: "BBox_Label",
                bboxes.columns[2]: "BBox_x",
                bboxes.columns[3]: "BBox_y",
                bboxes.columns[4]: "BBox_w",
                bboxes.columns[5]: "BBox_h",
            })
            filtered = filtered.merge(bbox_renamed[["Image Index", "BBox_Label", "BBox_x", "BBox_y", "BBox_w", "BBox_h"]], on="Image Index", how="left")

    filtered["Dataset_Type"] = "nih"
    out_dir = root / "data" / "hospital_nih"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "metadata.csv"
    filtered.to_csv(out_csv, index=False)
    print(f"[Hospital NIH] Metadata saved to {out_csv} ({len(filtered)} records)")
    return out_csv


def prepare_rsna_hospital(root: Path) -> Path:
    """Prepare Hospital 2 (RSNA Pneumonia Bounding Boxes)."""
    labels_path = root / "stage_2_train_labels.csv"
    images_dir = root / "stage_2_train_images"

    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing RSNA Train Labels CSV: {labels_path}")

    metadata = pd.read_csv(labels_path)
    available_dcm = {p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix == ".dcm"} if images_dir.is_dir() else set()
    filtered = metadata[metadata["patientId"].astype(str).isin(available_dcm)].copy()
    if filtered.empty:
        filtered = metadata.head(500).copy()

    filtered["Dataset_Type"] = "rsna"
    out_dir = root / "data" / "hospital_rsna"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "metadata.csv"
    filtered.to_csv(out_csv, index=False)
    print(f"[Hospital RSNA] Metadata saved to {out_csv} ({len(filtered)} records)")
    return out_csv


def prepare_siim_hospital(root: Path) -> Path:
    """Prepare Hospital 3 (SIIM Pneumothorax Masks)."""
    train_path = root / "siim-acr-pneumothorax-segmentation" / "stage_2_train.csv"
    images_dir = root / "siim-acr-pneumothorax-segmentation" / "stage_2_images"

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing SIIM Train CSV: {train_path}")

    metadata = pd.read_csv(train_path)
    # Ensure ImageId column exists
    if "ImageId" not in metadata.columns and len(metadata.columns) >= 2:
        metadata.rename(columns={metadata.columns[1]: "ImageId"}, inplace=True)

    available_dcm = {p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix == ".dcm"} if images_dir.is_dir() else set()
    filtered = metadata[metadata["ImageId"].astype(str).isin(available_dcm)].copy()
    if filtered.empty:
        filtered = metadata.head(500).copy()

    # Determine binary Pneumothorax presence
    filtered["HasPneumothorax"] = filtered["EncodedPixels"].apply(
        lambda rle: 0 if pd.isna(rle) or str(rle).strip() in ("-1", "") else 1
    )
    filtered["Dataset_Type"] = "siim"

    out_dir = root / "data" / "hospital_siim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "metadata.csv"
    filtered.to_csv(out_csv, index=False)
    print(f"[Hospital SIIM] Metadata saved to {out_csv} ({len(filtered)} records)")
    return out_csv


def split_metadata(root_dir: Path | None = None, seed: int = 42) -> tuple[Path, Path, Path]:
    """Prepare all 3 hospital datasets."""
    root = (root_dir or Path(__file__).resolve().parent).resolve()
    nih_path = prepare_nih_hospital(root)
    rsna_path = prepare_rsna_hospital(root)
    siim_path = prepare_siim_hospital(root)
    return nih_path, rsna_path, siim_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outputs = split_metadata(args.root_dir, args.seed)
    print(f"\nSuccessfully initialized 3 Hospital Nodes:\n - {outputs[0]}\n - {outputs[1]}\n - {outputs[2]}")


if __name__ == "__main__":
    main()