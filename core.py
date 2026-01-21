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
import pillow_heif
pillow_heif.register_heif_opener()  # Enable HEIC/HEIF support in PIL

# Paths relative to this file
MODEL_PATH = str(_CLIP_DIR / "mlx_model")
DB_PATH = str(_CORE_DIR / "embeddings.lance")

# Image extensions to search for
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg",  # JPEG
    ".png",           # PNG
    ".gif",           # GIF
    ".webp",          # WebP
    ".bmp",           # BMP
    ".tiff", ".tif",  # TIFF
    ".heic", ".heif", # iPhone photos
}

# Benchmark: ~280 images/second on M4 Max for batches of 225+
IMAGES_PER_SECOND = 280

# Default directories to exclude when scanning home directory
DEFAULT_EXCLUDE_DIRS = [
    "Library",
    ".Trash",
    ".cache",
    "Cache",
    "node_modules",
    ".git",
    ".venv",
    "venv",
]


@daft.cls
class EmbedImages:
    """Daft UDF to generate CLIP embeddings for images."""

    def __init__(self):
        self.model, _, self.img_processor = clip.load(MODEL_PATH)

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), 512))
    def __call__(self, paths: Series):
        """Takes a Series of image paths, returns a list of 512-dim embeddings."""
        path_list = paths.to_pylist()
        images = []
        failed = []
        for p in path_list:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
                failed.append(p)
                images.append(Image.new("RGB", (224, 224)))  # placeholder

        pixel_values = self.img_processor(images)
        output = self.model(pixel_values=pixel_values)
        embeddings = [np.array(emb) for emb in output.image_embeds]

        # Zero out embeddings for failed images
        for i, p in enumerate(path_list):
            if p in failed:
                embeddings[i] = np.zeros(512, dtype=np.float32)

        return embeddings


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


def find_images(directory: Path, recursive: bool = True, show_progress: bool = True, exclude_dirs: list[str] | None = None) -> list[Path]:
    """Find all image files in a directory using find command.

    Args:
        directory: Root directory to search
        recursive: Whether to search subdirectories
        show_progress: Whether to print progress
        exclude_dirs: List of directory names to exclude (e.g. ["Library", ".cache"])
    """
    import subprocess
    import sys

    # Build -name conditions for each extension
    name_args = []
    for ext in IMAGE_EXTENSIONS:
        if name_args:
            name_args.append("-o")
        name_args.extend(["-name", f"*{ext}"])

    # Build find command
    cmd = ["find", str(directory)]
    if not recursive:
        cmd.extend(["-maxdepth", "1"])

    # Build prune conditions for excluded directories
    prune_args = ["-name", ".*"]  # Always exclude hidden directories
    if exclude_dirs:
        for exclude in exclude_dirs:
            prune_args.extend(["-o", "-name", exclude])

    cmd.extend(["("] + prune_args + [")", "-prune", "-o"])
    cmd.extend(["-type", "f", "("] + name_args + [")", "-print"])

    # Stream output and show progress
    paths = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in process.stdout:
        path = line.strip()
        if path:
            paths.append(Path(path))
            if show_progress and len(paths) % 1000 == 0:
                print(f"\rFound: {len(paths):,} images...", end="", flush=True)
    process.wait()

    if show_progress and len(paths) >= 1000:
        print()  # newline after progress

    return sorted(paths)


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
