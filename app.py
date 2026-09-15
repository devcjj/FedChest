"""Clinician-facing inference API for the FedChest AI prototype."""

import base64
import io
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from model import IMAGE_SIZE, build_model
from report_engine import build_report

LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
)
PHI_TAGS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "InstitutionName", "InstitutionAddress", "ReferringPhysicianName", "PerformingPhysicianName",
    "OperatorsName", "AccessionNumber", "StudyID", "StudyDate", "StudyTime", "OtherPatientIDs",
    "OtherPatientNames", "DeviceSerialNumber", "StationName",
)

app = FastAPI(title="FedChest AI", version="0.1.0")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model().to(_DEVICE)
        checkpoint = Path(__file__).resolve().parent / "checkpoints" / "global_fedchest_model.pt"
        if checkpoint.is_file():
            payload = torch.load(checkpoint, map_location=_DEVICE, weights_only=False)
            state_dict = payload.get("model_state_dict", payload)
            _MODEL.load_state_dict(state_dict, strict=False)
        _MODEL.eval()
    return _MODEL


def decode_upload(filename: str, content: bytes) -> tuple[np.ndarray, int]:
    if filename.lower().endswith(".dcm"):
        try:
            import pydicom
        except ImportError as exc:
            raise HTTPException(status_code=501, detail="Install pydicom to process DICOM uploads") from exc
        dataset = pydicom.dcmread(io.BytesIO(content), force=True)
        stripped = 0
        for tag_name in PHI_TAGS:
            if hasattr(dataset, tag_name):
                delattr(dataset, tag_name)
                stripped += 1
        pixels = dataset.pixel_array.astype(np.float32)
        return pixels, stripped

    try:
        with Image.open(io.BytesIO(content)) as image:
            return np.asarray(image.convert("L"), dtype=np.float32), 0
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload must be a readable PNG, JPG, or DICOM file") from exc


def preprocess(image: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    image = np.nan_to_num(image, copy=False)
    low, high = float(image.min()), float(image.max())
    if high > low:
        image = (image - low) / (high - low) * 2048.0 - 1024.0
    image = Image.fromarray(image.astype(np.float32), mode="F").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    windowed = np.asarray(image, dtype=np.float32).clip(-1024.0, 1024.0)
    normalized = (windowed + 1024.0) / 2048.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    return tensor, normalized


def find_boxes(heatmap: np.ndarray, threshold: float = 0.6) -> list[dict[str, int]]:
    mask = heatmap >= threshold
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            queue = deque([(row, column)])
            visited[row, column] = True
            points = []
            while queue:
                current_row, current_column = queue.popleft()
                points.append((current_row, current_column))
                for next_row, next_column in (
                    (current_row - 1, current_column), (current_row + 1, current_column),
                    (current_row, current_column - 1), (current_row, current_column + 1),
                ):
                    if 0 <= next_row < height and 0 <= next_column < width:
                        if mask[next_row, next_column] and not visited[next_row, next_column]:
                            visited[next_row, next_column] = True
                            queue.append((next_row, next_column))
            if len(points) >= 4:
                rows, columns = zip(*points)
                boxes.append({"x": int(min(columns)), "y": int(min(rows)), "w": int(max(columns) - min(columns) + 1), "h": int(max(rows) - min(rows) + 1)})
    return boxes


def encode_heatmap(heatmap: np.ndarray) -> str:
    rgb = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((1.0 - np.abs(heatmap - 0.5) * 2) * 180, 0, 180).astype(np.uint8)
    rgb[..., 2] = np.clip((1.0 - heatmap) * 220, 0, 220).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def predict(image_tensor: torch.Tensor, metadata_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model = get_model()
    image_tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    try:
        outputs = model(image_tensor, metadata_tensor)
    except TypeError:
        outputs = model(image_tensor)
    logits = outputs["probabilities"] if isinstance(outputs, dict) else outputs
    selected = logits[:, : len(LABELS)]
    probabilities = selected if selected.max() <= 1.0 and selected.min() >= 0.0 else torch.sigmoid(selected)
    top_index = int(probabilities[0].argmax())
    selected[0, top_index].backward()
    saliency = image_tensor.grad.detach().abs().amax(dim=1)[0]
    saliency = saliency / saliency.amax().clamp_min(1e-8)
    return probabilities.detach().cpu().numpy()[0], saliency.cpu().numpy()


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).with_name("dashboard.html"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": str(_DEVICE)}


@app.post("/predict")
async def prediction(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise HTTPException(status_code=415, detail="Send JSON with image_base64, filename, age, sex, and view_position")
    payload = await request.json()
    try:
        content = base64.b64decode(payload["image_base64"], validate=True)
        filename = str(payload.get("filename", "image.png"))
        age = int(payload["age"])
        sex = str(payload["sex"]).upper()
        view_position = str(payload["view_position"]).upper()
    except (KeyError, TypeError, ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid input fields") from exc
    if not 0 <= age <= 120 or sex not in {"F", "M", "O"} or view_position not in {"AP", "PA"}:
        raise HTTPException(status_code=422, detail="Use age 0-120, sex F/M/O, and view_position AP/PA")

    started = time.perf_counter()
    try:
        raw_image, stripped = decode_upload(filename, content)
        tensor, _ = preprocess(raw_image)

        age_norm = min(max(float(age), 0.0), 100.0) / 100.0
        is_male = float(sex == "M")
        is_ap = float(view_position == "AP")
        metadata_tensor = torch.tensor([[age_norm, is_male, is_ap]], dtype=torch.float32, device=_DEVICE)

        probabilities, heatmap = predict(tensor, metadata_tensor)
        rounded = [round(float(value), 4) for value in probabilities]
        return JSONResponse({
            "probabilities": [{"label": label, "probability": probability} for label, probability in zip(LABELS, rounded)],
            "heatmap_base64": encode_heatmap(heatmap),
            "bounding_boxes": find_boxes(heatmap),
            "report": build_report(LABELS, probabilities, age, sex, view_position),
            "phi_tags_stripped": stripped,
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
            "disclaimer": "Decision support only; not a diagnosis.",
        })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"detail": f"Inference failed: {str(exc)}"})


import threading

_TRAINING_STATE = {
    "is_training": False,
    "mode": "idle",
    "node": "",
    "step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "message": "Ready"
}


@app.get("/api/nodes")
def get_nodes() -> JSONResponse:
    """Return available hospital nodes and sample counts."""
    root = Path(__file__).resolve().parent / "data"
    nodes = []
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and (child / "metadata.csv").is_file():
                try:
                    df = pd.read_csv(child / "metadata.csv")
                    nodes.append({
                        "id": child.name,
                        "name": child.name.replace("_", " ").title(),
                        "samples": len(df),
                        "dataset_type": str(df.get("Dataset_Type", ["custom"])[0]) if "Dataset_Type" in df.columns else "custom"
                    })
                except Exception:
                    pass
    return JSONResponse({"nodes": nodes, "training_state": _TRAINING_STATE})


@app.get("/api/train-status")
def get_train_status() -> JSONResponse:
    return JSONResponse(_TRAINING_STATE)


@app.post("/api/upload-training-sample")
async def upload_training_sample(request: Request) -> JSONResponse:
    """Upload a new scan with labels to a local hospital training node."""
    payload = await request.json()
    try:
        node_id = str(payload.get("node_id", "hospital_nih"))
        filename = str(payload.get("filename", "sample.png"))
        content = base64.b64decode(payload["image_base64"])
        selected_labels = payload.get("labels", [])  # list of label strings
        age = int(payload.get("age", 50))
        sex = str(payload.get("sex", "M")).upper()
        view_position = str(payload.get("view_position", "PA")).upper()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid payload") from exc

    root = Path(__file__).resolve().parent / "data" / node_id
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save image file
    save_path = images_dir / filename
    save_path.write_bytes(content)

    # Append row to metadata.csv
    metadata_csv = root / "metadata.csv"
    finding_labels = "|".join(selected_labels) if selected_labels else "No Finding"
    new_row = pd.DataFrame([{
        "Image Index": filename,
        "Finding Labels": finding_labels,
        "Patient Age": age,
        "Patient Gender": sex,
        "View Position": view_position,
        "Dataset_Type": "nih"
    }])

    if metadata_csv.is_file():
        df = pd.read_csv(metadata_csv)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(metadata_csv, index=False)

    return JSONResponse({"status": "success", "node": node_id, "samples_total": len(df), "saved_file": filename})


def _run_local_training_thread(node_id: str, epochs: int, batch_size: int) -> None:
    global _MODEL, _TRAINING_STATE
    _TRAINING_STATE["is_training"] = True
    _TRAINING_STATE["mode"] = "local"
    _TRAINING_STATE["node"] = node_id
    _TRAINING_STATE["step"] = 0
    _TRAINING_STATE["loss"] = 0.0
    _TRAINING_STATE["message"] = f"Training local node {node_id}..."

    try:
        from dataset import get_hospital_dataset
        from torch.utils.data import DataLoader
        from model import trainable_parameters

        root = Path(__file__).resolve().parent
        node_dir = root / "data" / node_id
        dataset = get_hospital_dataset(node_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = get_model()
        model.train()
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=1e-3)
        criterion = torch.nn.BCELoss()

        total_steps = len(loader) * epochs
        _TRAINING_STATE["total_steps"] = total_steps

        current_step = 0
        for epoch in range(epochs):
            for images, metadata, targets in loader:
                images, metadata, targets = images.to(_DEVICE), metadata.to(_DEVICE), targets.to(_DEVICE)
                optimizer.zero_grad()
                try:
                    outputs = model(images, metadata)
                except TypeError:
                    outputs = model(images)
                logits = outputs["probabilities"] if isinstance(outputs, dict) else outputs
                logits = logits[:, : targets.shape[1]]
                probs = logits if logits.max() <= 1.0 and logits.min() >= 0.0 else torch.sigmoid(logits)
                loss = criterion(probs, targets)
                loss.backward()
                optimizer.step()

                current_step += 1
                _TRAINING_STATE["step"] = current_step
                _TRAINING_STATE["loss"] = round(float(loss.item()), 4)
                _TRAINING_STATE["message"] = f"Epoch {epoch+1}/{epochs} | Step {current_step}/{total_steps} | Loss: {loss.item():.4f}"

        # Save Checkpoint
        ckpt = root / "checkpoints" / "global_fedchest_model.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "node": node_id}, ckpt)
        _MODEL = None  # Force reload on next predict
        _TRAINING_STATE["message"] = f"Local training complete! Checkpoint saved to {ckpt.name}."
    except Exception as exc:
        _TRAINING_STATE["message"] = f"Training failed: {str(exc)}"
    finally:
        _TRAINING_STATE["is_training"] = False


@app.post("/api/train-local")
def train_local_endpoint(request: Request) -> JSONResponse:
    global _TRAINING_STATE
    if _TRAINING_STATE["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    try:
        data = request.query_params
        node_id = data.get("node_id", "hospital_nih")
        epochs = int(data.get("epochs", 2))
        batch_size = int(data.get("batch_size", 4))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid params") from exc

    thread = threading.Thread(target=_run_local_training_thread, args=(node_id, epochs, batch_size), daemon=True)
    thread.start()
    return JSONResponse({"status": "started", "node": node_id, "epochs": epochs})


