#!/usr/bin/env python3
"""MCP server for local image search."""

import sys
import threading
import time
from pathlib import Path

import daft
import numpy as np
from mcp.server.fastmcp import FastMCP

from core import load_model, embed_text, cosine_similarity, DB_PATH
from embed import sync_embeddings


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

# Embedding refresh state
embedding_lock = threading.Lock()
REFRESH_INTERVAL = 300  # 5 minutes


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

    if embeddings_df is None:
        return [{"error": "No embeddings loaded. Run embed.py first."}]

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
    global image_dir

    while True:
        # Try to acquire lock (non-blocking)
        if not embedding_lock.acquire(blocking=False):
            log("Embedding refresh still in progress, skipping this cycle")
        else:
            try:
                if image_dir and image_dir.exists():
                    log(f"Starting embedding refresh for {image_dir}...")
                    sync_embeddings(image_dir, log_fn=log)
                    reload_embeddings()
                else:
                    log(f"Image directory not set or doesn't exist: {image_dir}")
            except Exception as e:
                log(f"Embedding refresh failed: {e}")
            finally:
                embedding_lock.release()

        time.sleep(REFRESH_INTERVAL)


def main():
    """Main entry point."""
    global model, tokenizer, embeddings_df, image_dir

    # Parse image directory from command line
    if len(sys.argv) > 1:
        image_dir = Path(sys.argv[1]).expanduser().resolve()
        log(f"Image directory: {image_dir}")
    else:
        log("Warning: No image directory specified. Background refresh disabled.")

    log("Loading CLIP model...")
    model, tokenizer, _ = load_model()

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
        log("Background embedding refresh started (every 5 min)")

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
