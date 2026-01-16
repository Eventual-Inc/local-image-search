"""Shared utilities for local image search."""

import sys
from pathlib import Path

# Setup path for clip module
_CORE_DIR = Path(__file__).parent.resolve()
_CLIP_DIR = _CORE_DIR / "clip"
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

import clip
import daft
from daft import DataType, Series
from PIL import Image
import numpy as np

# Paths relative to this file
MODEL_PATH = str(_CLIP_DIR / "mlx_model")
DB_PATH = str(_CORE_DIR / "embeddings.lance")

# Image extensions to search for
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

# Benchmark: ~280 images/second on M4 Max for batches of 225+
IMAGES_PER_SECOND = 280


@daft.cls
class EmbedImages:
    """Daft UDF to generate CLIP embeddings for images."""

    def __init__(self):
        self.model, _, self.img_processor = clip.load(MODEL_PATH)

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), 512))
    def __call__(self, paths: Series):
        """Takes a Series of image paths, returns a list of 512-dim embeddings."""
        images = [Image.open(p).convert("RGB") for p in paths.to_pylist()]
        pixel_values = self.img_processor(images)
        output = self.model(pixel_values=pixel_values)
        return [np.array(emb) for emb in output.image_embeds]


def load_model():
    """Load the CLIP model, tokenizer, and image processor."""
    return clip.load(MODEL_PATH)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_text(model, tokenizer, text: str) -> np.ndarray:
    """Embed a text query."""
    tokens = tokenizer([text])
    output = model(input_ids=tokens)
    return np.array(output.text_embeds[0])


def find_images(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all image files in a directory, excluding hidden directories."""
    images = []
    pattern = "**/*" if recursive else "*"

    for path in directory.glob(pattern):
        # Skip hidden directories (like .venv, .git)
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
