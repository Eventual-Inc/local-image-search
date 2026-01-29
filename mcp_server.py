#!/usr/bin/env python3
"""MCP server for local image search."""

import fcntl
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import daft
import numpy as np
from mcp.server.fastmcp import FastMCP

from core import load_model, embed_text, cosine_similarity, DB_PATH, MODEL_PATH, DEFAULT_EXCLUDE_DIRS
from embed import sync_embeddings

# File-based lock to prevent concurrent refreshes across processes
LOCK_FILE = Path(DB_PATH).parent / ".embedding_refresh.lock"


def log(msg: str):
    """Log to stderr (stdout is reserved for MCP protocol)."""
    print(msg, file=sys.stderr, flush=True)


# Create MCP server
mcp = FastMCP("local-image-search")

# Global state - loaded on startup
model = None
tokenizer = None
embeddings_df = None
image_dir = None
exclude_dirs = None  # Directories to exclude from scanning
model_loading = False  # True while model is being downloaded/loaded

# Embedding refresh state
REFRESH_INTERVAL = int(os.environ.get("REFRESH_INTERVAL", "60"))  # default 1 minute


def get_status_info() -> dict:
    """Get current service status."""
    if model_loading:
        return {
            "ready": False,
            "status": "downloading_model",
            "message": "Model is downloading (~600MB). Please wait 1-2 minutes."
        }
    if model is None:
        return {
            "ready": False,
            "status": "loading_model",
            "message": "Model is loading. Please wait a moment."
        }
    if embeddings_df is None or len(embeddings_df) == 0:
        return {
            "ready": False,
            "status": "syncing_embeddings",
            "message": "Initial embedding sync in progress. This may take a few minutes depending on the number of images."
        }
    return {
        "ready": True,
        "status": "ready",
        "total_images": len(embeddings_df)
    }


@mcp.tool()
def get_status() -> dict:
    """Check if the image search service is ready.

    Returns:
        Status dict with 'ready' boolean and 'message' or 'total_images'
    """
    return get_status_info()


@mcp.tool()
def search_images(query: str, limit: int = 5) -> list[dict]:
    """Search for images matching a text query.

    Args:
        query: Natural language description of the image to find
        limit: Maximum number of results to return (default: 5)

    Returns:
        List of matching images with paths and similarity scores
    """
    global model, tokenizer, embeddings_df

    # Check if service is ready
    status = get_status_info()
    if not status["ready"]:
        return [status]

    # Embed the query text
    query_embedding = embed_text(model, tokenizer, query)

    # Get all embeddings and paths
    data = embeddings_df.to_pydict()
    paths = data["path"]
    vectors = data["vector"]

    # Compute similarities
    scores = []
    for i, vec in enumerate(vectors):
        vec_array = np.array(vec, dtype=np.float32)
        # Skip zero vectors (failed images)
        if np.allclose(vec_array, 0):
            scores.append(-1.0)
        else:
            scores.append(cosine_similarity(query_embedding, vec_array))

    # Sort by score descending
    ranked = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)

    # Return top results
    results = [
        {"path": path, "score": round(score, 3)}
        for path, score in ranked[:limit]
        if score > 0  # exclude failed images
    ]

    return results


def ensure_model_exists():
    """Download and convert CLIP model if not present."""
    model_path = Path(MODEL_PATH)

    # Check if model exists (look for model.safetensors or model.safetensors.index.json)
    if (model_path / "model.safetensors").exists() or (model_path / "model.safetensors.index.json").exists():
        return True

    log("Model not found. Downloading and converting CLIP model (~600MB)...")
    log("This only needs to happen once.")

    # Run convert.py from the clip directory
    clip_dir = model_path.parent
    convert_script = clip_dir / "convert.py"

    if not convert_script.exists():
        log(f"Error: convert.py not found at {convert_script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(convert_script)],
            cwd=str(clip_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            log(f"Error downloading model: {result.stderr}")
            return False

        log("Model downloaded and converted successfully.")
        return True

    except Exception as e:
        log(f"Error downloading model: {e}")
        return False


def reload_embeddings():
    """Reload embeddings from Lance DB."""
    global embeddings_df

    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        log(f"Reloaded {len(embeddings_df)} embeddings")
    else:
        embeddings_df = None
        log("No embeddings found")


def embedding_refresh_loop():
    """Background loop to refresh embeddings periodically."""
    global image_dir, exclude_dirs

    while True:
        # Add random jitter (0-30 seconds) to prevent thundering herd
        jitter = random.uniform(0, 30)
        time.sleep(jitter)

        # Try to acquire file-based lock (non-blocking) to coordinate across processes
        lock_file = None
        try:
            LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
            lock_file = open(LOCK_FILE, "w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            log("Another process is refreshing embeddings, skipping this cycle")
            if lock_file:
                lock_file.close()
            time.sleep(REFRESH_INTERVAL)
            continue

        try:
            if image_dir and image_dir.exists():
                log(f"Starting embedding refresh for {image_dir}...")
                sync_embeddings(image_dir, log_fn=log, exclude_dirs=exclude_dirs)
                reload_embeddings()
            else:
                log(f"Image directory not set or doesn't exist: {image_dir}")
        except Exception as e:
            log(f"Embedding refresh failed: {e}")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

        time.sleep(REFRESH_INTERVAL)


def startup_task():
    """Background task to download model and load embeddings."""
    global model, tokenizer, embeddings_df, image_dir, model_loading

    model_loading = True

    # Ensure model exists (download if needed)
    if not ensure_model_exists():
        log("Failed to download model.")
        model_loading = False
        return

    log("Loading CLIP model...")
    model, tokenizer, _ = load_model()
    model_loading = False

    log("Loading embeddings...")
    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        log(f"Loaded {len(embeddings_df)} embeddings")
    else:
        log("No embeddings found.")

    # Start background embedding refresh thread
    if image_dir:
        refresh_thread = threading.Thread(target=embedding_refresh_loop, daemon=True)
        refresh_thread.start()
        log(f"Background embedding refresh started (every {REFRESH_INTERVAL}s)")


def main():
    """Main entry point."""
    global image_dir, exclude_dirs

    # Parse EXCLUDE_DIRS from environment (comma-separated)
    exclude_env = os.environ.get("EXCLUDE_DIRS", "").strip()
    custom_excludes = [d.strip() for d in exclude_env.split(",") if d.strip()] if exclude_env else None

    # Parse image directory from command line
    if len(sys.argv) > 1:
        # Custom root provided
        image_dir = Path(sys.argv[1]).expanduser().resolve()
        # Use custom excludes if provided, otherwise no excludes
        exclude_dirs = custom_excludes
        log(f"Image directory: {image_dir}")
        if exclude_dirs:
            log(f"Excluding: {', '.join(exclude_dirs)}")
    else:
        # No root provided - use home with defaults (unless custom excludes provided)
        image_dir = Path.home()
        if custom_excludes:
            # Custom excludes override defaults
            exclude_dirs = custom_excludes
            log(f"Image directory: {image_dir} (default)")
            log(f"Excluding: {', '.join(exclude_dirs)}")
        else:
            # Use default excludes
            exclude_dirs = DEFAULT_EXCLUDE_DIRS
            log(f"Image directory: {image_dir} (default)")
            log(f"Excluding (defaults): {', '.join(exclude_dirs)}")

    # Start model loading in background
    startup_thread = threading.Thread(target=startup_task, daemon=True)
    startup_thread.start()

    # Run the MCP server (starts immediately, responds with status while loading)
    mcp.run()


if __name__ == "__main__":
    main()
