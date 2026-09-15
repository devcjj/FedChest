"""PyTorch datasets for NIH, RSNA, and SIIM hospital nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_NAMES = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
)
IMAGE_SIZE = 224


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> Tensor:
    """Load PNG or DICOM image as a normalized grayscale tensor."""
    if not image_path.is_file():
        # Fallback to dummy tensor if image file missing
        return transform(Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=128))

    if image_path.suffix.lower() == ".dcm":
        try:
            import pydicom
            dcm = pydicom.dcmread(image_path, force=True)
            arr = dcm.pixel_array.astype(float)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
            else:
                arr = np.zeros_like(arr)
            image = Image.fromarray(arr.astype("uint8")).convert("L")
        except Exception:
            image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=128)
    else:
        try:
            with Image.open(image_path) as img:
                image = img.convert("L")
        except Exception:
            image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=128)

    return transform(image)


class BaseHospitalDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Base class providing shared transform and tensor utility."""

    def __init__(self, node_dir: str | Path) -> None:
        self.node_dir = Path(node_dir).resolve()
        self.metadata_path = self.node_dir / "metadata.csv"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Missing node metadata: {self.metadata_path}")
        self.frame = pd.read_csv(self.metadata_path)
        if self.frame.empty:
            raise ValueError(f"Node metadata is empty: {self.metadata_path}")
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])


class NIHHospitalDataset(BaseHospitalDataset):
    """Load Hospital 1: NIH ChestX-ray14 image, tabular features, and 14 binary targets."""

    def __init__(self, node_dir: str | Path) -> None:
        super().__init__(node_dir)
        self.images_dir = self._find_images_dir()

    def _find_images_dir(self) -> Path:
        for parent in (self.node_dir, *self.node_dir.parents):
            candidate = parent / "images"
            if candidate.is_dir():
                return candidate
        return self.node_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row: Any = self.frame.iloc[index]
        image_path = self.images_dir / str(row.get("Image Index", ""))
        image_tensor = load_image_tensor(image_path, self.transform)

        age = min(max(float(row.get("Patient Age", 50.0) or 50.0), 0.0), 100.0) / 100.0
        gender = str(row.get("Patient Gender", "M")).strip().upper()
        view = str(row.get("View Position", "PA")).strip().upper()
        metadata_tensor = torch.tensor([age, float(gender == "M"), float(view == "AP")], dtype=torch.float32)

        labels_text = {label.strip() for label in str(row.get("Finding Labels", "")).split("|")}
        label_tensor = torch.tensor([float(label in labels_text) for label in LABEL_NAMES], dtype=torch.float32)
        return image_tensor, metadata_tensor, label_tensor


class RSNAHospitalDataset(BaseHospitalDataset):
    """Load Hospital 2: RSNA Pneumonia bounding boxes dataset."""

    def __init__(self, node_dir: str | Path) -> None:
        super().__init__(node_dir)
        self.images_dir = self._find_images_dir()

    def _find_images_dir(self) -> Path:
        for parent in (self.node_dir, *self.node_dir.parents):
            candidate = parent / "stage_2_train_images"
            if candidate.is_dir():
                return candidate
        return self.node_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row: Any = self.frame.iloc[index]
        patient_id = str(row.get("patientId", ""))
        image_path = self.images_dir / f"{patient_id}.dcm"
        image_tensor = load_image_tensor(image_path, self.transform)

        metadata_tensor = torch.tensor([0.5, 1.0, 0.0], dtype=torch.float32)

        # Target = Pneumonia (index 6 in LABEL_NAMES)
        target_val = float(row.get("Target", 0.0) or 0.0)
        labels = [0.0] * len(LABEL_NAMES)
        labels[6] = target_val  # Pneumonia
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        return image_tensor, metadata_tensor, label_tensor


class SIIMHospitalDataset(BaseHospitalDataset):
    """Load Hospital 3: SIIM Pneumothorax segmentation mask dataset."""

    def __init__(self, node_dir: str | Path) -> None:
        super().__init__(node_dir)
        self.images_dir = self._find_images_dir()

    def _find_images_dir(self) -> Path:
        for parent in (self.node_dir, *self.node_dir.parents):
            candidate = parent / "siim-acr-pneumothorax-segmentation" / "stage_2_images"
            if candidate.is_dir():
                return candidate
        return self.node_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row: Any = self.frame.iloc[index]
        image_id = str(row.get("ImageId", ""))
        image_path = self.images_dir / f"{image_id}.dcm"
        image_tensor = load_image_tensor(image_path, self.transform)

        metadata_tensor = torch.tensor([0.5, 1.0, 0.0], dtype=torch.float32)

        # Target = Pneumothorax (index 7 in LABEL_NAMES)
        has_ptx = float(row.get("HasPneumothorax", 0.0) or 0.0)
        labels = [0.0] * len(LABEL_NAMES)
        labels[7] = has_ptx  # Pneumothorax
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        return image_tensor, metadata_tensor, label_tensor


def get_hospital_dataset(node_dir: str | Path) -> BaseHospitalDataset:
    """Factory to instantiate the appropriate PyTorch Dataset based on node metadata."""
    node_path = Path(node_dir).resolve()
    metadata_csv = node_path / "metadata.csv"
    if not metadata_csv.is_file():
        # Default fallback
        return NIHHospitalDataset(node_path)

    df_sample = pd.read_csv(metadata_csv, nrows=5)
    dtype = str(df_sample.get("Dataset_Type", [""])[0]).lower() if "Dataset_Type" in df_sample.columns else ""

    if dtype == "rsna" or "patientId" in df_sample.columns:
        return RSNAHospitalDataset(node_path)
    elif dtype == "siim" or "ImageId" in df_sample.columns or "HasPneumothorax" in df_sample.columns:
        return SIIMHospitalDataset(node_path)
    else:
        return NIHHospitalDataset(node_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for name in ("hospital_nih", "hospital_rsna", "hospital_siim"):
        path = root / "data" / name
        if (path / "metadata.csv").is_file():
            ds = get_hospital_dataset(path)
            img, meta, lbl = ds[0]
            print(f"[{name}] type={type(ds).__name__}, samples={len(ds)}, img={tuple(img.shape)}, labels_sum={lbl.sum().item()}")