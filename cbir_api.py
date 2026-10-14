"""Local API for the Medical CBIR System's React interface."""
from __future__ import annotations

import base64
import io
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from skimage import color, io as skio, transform
from skimage.util import img_as_float

from src.cbir_skimage import IndexeurCBIR, charger_images_dossier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CLASSES = ("benign", "malignant", "normal")
METHOD_LABELS = {
    "intensite": "Intensity histogram",
    "texture": "GLCM texture",
    "forme": "Hu shape moments",
}
DISTANCE_LABELS = {
    "euclidienne": "Euclidean distance",
    "cosinus": "Cosine distance",
}
app = FastAPI(title="Medical CBIR API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:5174", "http://localhost:5174"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _class_from_path(path: str) -> str:
    normalized = str(path).replace("\\", "/").lower()
    for class_name in CLASSES:
        if f"/{class_name}/" in f"/{normalized.strip('/')}/":
            return class_name
    return "unknown"


def _prepare_image(payload: bytes, filename: str) -> np.ndarray:
    if not payload:
        raise ValueError("The uploaded file is empty.")
    try:
        image = skio.imread(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"Could not decode {filename}: {exc}") from exc

    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)
    elif image.ndim != 2:
        raise ValueError(f"Expected a grayscale or RGB image, received shape {image.shape}.")

    image = img_as_float(image)
    if not np.isfinite(image).all():
        raise ValueError("The image contains non-finite pixel values.")
    resized = transform.resize(image, (128, 128), anti_aliasing=True, preserve_range=True)
    return np.clip(resized, 0.0, 1.0).astype(np.float64)


def _png_data_url(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.clip(array, 0, 1)
        pil_image = Image.fromarray((array * 255).round().astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        if np.issubdtype(array.dtype, np.floating):
            array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
        pil_image = Image.fromarray(array)
    else:
        raise ValueError("Unsupported image shape for preview.")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _dataset_counts() -> dict[str, int]:
    counts = {}
    base = DATA_DIR / "breast-ultrasound-images-dataset" / "Dataset_BUSI_with_GT"
    for class_name in CLASSES:
        class_dir = base / class_name
        if not class_dir.is_dir():
            counts[class_name] = 0
            continue
        counts[class_name] = sum(
            1 for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and "_mask" not in p.name.lower()
        )
    return counts


@lru_cache(maxsize=8)
def _get_indexer(methods: tuple[str, ...]) -> tuple[IndexeurCBIR, list[str], list[np.ndarray]]:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"BUSI dataset folder not found at {DATA_DIR}. See the README for the expected dataset layout."
        )
    images, names = charger_images_dossier(str(DATA_DIR), max_images=None, random_seed=42)
    if len(images) < 2:
        raise FileNotFoundError(
            "No usable BUSI images were found. Check that data/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT contains benign, malignant and normal folders."
        )
    if len(set(methods)) == 0:
        raise ValueError("Select at least one feature descriptor.")
    indexer = IndexeurCBIR()
    indexer.indexer(images, names, methodes=methods)
    return indexer, names, images


def _read_saved_summary() -> dict | None:
    path = REPORTS_DIR / "metrics_summary.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    p1_match = re.search(r"Mean Precision@1:\s*([0-9.]+)", text, re.I)
    p5_match = re.search(r"Mean Precision@5:\s*([0-9.]+)", text, re.I)
    return {
        "source": path.relative_to(ROOT).as_posix(),
        "precision_at_1": float(p1_match.group(1)) if p1_match else None,
        "precision_at_5": float(p5_match.group(1)) if p5_match else None,
        "text": text,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Medical CBIR API"}


@app.get("/api/status")
def status():
    counts = _dataset_counts()
    available = sum(counts.values()) >= 2
    return {
        "dataset_available": available,
        "dataset_root": "data/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT",
        "class_counts": counts,
        "total_images": sum(counts.values()),
        "classes": list(CLASSES),
        "saved_evaluation": _read_saved_summary(),
    }


@app.post("/api/search")
async def search_images(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    distance_metric: str = Form("euclidienne"),
    methods: list[str] = Form(["intensite", "texture", "forme"]),
):
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 10.")
    if distance_metric not in DISTANCE_LABELS:
        raise HTTPException(status_code=422, detail="Choose Euclidean or cosine distance.")
    selected = tuple(m for m in METHOD_LABELS if m in methods)
    if not selected:
        raise HTTPException(status_code=422, detail="Select at least one descriptor.")
    filename = file.filename or "query-image"
    if Path(filename).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, JPEG, WEBP or BMP ultrasound image.")
    payload = await file.read()
    try:
        query_image = _prepare_image(payload, filename)
        indexer, file_names, images = _get_indexer(selected)
        indices, distances = indexer.rechercher(
            query_image,
            top_k=int(top_k),
            metrique=distance_metric,
            methodes=selected,
        )
        items = []
        for rank, (idx, distance) in enumerate(zip(indices, distances), start=1):
            items.append({
                "rank": rank,
                "filename": Path(file_names[idx].replace("\\", "/")).name,
                "relative_path": str(file_names[idx]).replace("\\", "/"),
                "class": _class_from_path(file_names[idx]),
                "distance": round(float(distance), 6),
                "preview": _png_data_url(images[idx]),
            })
        return {
            "query": {
                "filename": filename,
                "preview": _png_data_url(query_image),
                "width": 128,
                "height": 128,
            },
            "top_k": int(top_k),
            "distance_metric": distance_metric,
            "distance_label": DISTANCE_LABELS[distance_metric],
            "methods": [{"id": m, "label": METHOD_LABELS[m]} for m in selected],
            "dataset_size": len(file_names),
            "results": items,
            "note": "Similarity ranking only. Retrieved BUSI class labels describe the dataset folders and must not be interpreted as a diagnosis for the query image.",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image retrieval failed: {exc}") from exc
